/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_ndhwc_multi_w.h
 * \brief
 */

#ifndef AVG_POOL3D_NDHWC_MULTI_W_H_
#define AVG_POOL3D_NDHWC_MULTI_W_H_

#include "kernel_operator.h"
#include "avg_pool3d_common.h"

namespace AvgPool3d {
template <typename T, int32_t QUEUE_DEPTH>
class KernelAvgPool3dMultiW {
public:
    __aicore__ inline KernelAvgPool3dMultiW() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTiling(const AvgPool3DTilingData* __restrict__ tiling);
    __aicore__ inline void CopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding);
    __aicore__ inline void CopyOut(int64_t offset, uint16_t blockCount, uint32_t blockLen);
    __aicore__ inline void DataCopyOutNonPad(
        LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint16_t blockCount, uint32_t validDataLen);
    __aicore__ inline void ReduceMeanMultiWindow(int64_t outputPointIdx, int64_t windowNum);
    __aicore__ inline void ReduceSumMultiWindow(
        const Index& startIndex, const Index& endIndex, LocalTensor<float>& sumBufLocal, int64_t outputPointIdx,
        int64_t nOffset, int64_t windowNum);
    __aicore__ inline void ReduceSumRow(
        const Index& startIndex, LocalTensor<float>& sumBufLocal, LocalTensor<T>& inputLocal,
        int64_t outputPointIdx, int64_t windowNum);
    __aicore__ inline void ReduceSumRowRepeat(
        const Index& startIndex, LocalTensor<float>& sumBufLocal, LocalTensor<T>& inputLocal, int64_t windowNum);


    int64_t windowWNum;
    PoolMem<T, QUEUE_DEPTH> poolMem;

    bool isSumWithRepeat;
    bool isSamePoolSize;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::InitTiling(const AvgPool3DTilingData* __restrict__ tiling) {
    poolMem.inputShape = PoolShape(tiling->inN, tiling->inC, tiling->inD, tiling->inH, tiling->inW);
    poolMem.outputShape = PoolShape(tiling->inN, tiling->inC, tiling->outD, tiling->outH, tiling->outW);

    poolMem.poolParam = PoolParameter(tiling->kD, tiling->kH, tiling->kW, tiling->dD, tiling->dH, tiling->dW,
                              tiling->pD, tiling->pH, tiling->pW, tiling->divisorOverride, tiling->countIncludePad);

    poolMem.indexBuf.SetComputeParameter(poolMem.outputShape, poolMem.inputShape, poolMem.poolParam);

    poolMem.numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);
    poolMem.inC = tiling->inC;
    poolMem.alignC = AlignUp(poolMem.inC, poolMem.numPerBlock);
    windowWNum = tiling->windowWNum;

    poolMem.outputPointNum = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
    poolMem.outputPointOffset = GetBlockIdx() < tiling->formerNum
        ? tiling->formerLength * GetBlockIdx()
        : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);
    poolMem.lastPointOffset = poolMem.outputPointNum + poolMem.outputPointOffset - 1;
    poolMem.atomicAddNum = poolMem.outputPointNum < tiling->atomicAddNum ? poolMem.outputPointNum : tiling->atomicAddNum;
    poolMem.validTailLen = poolMem.inC % poolMem.numPerBlock;
    poolMem.usedCoreNum = tiling->usedCoreNum;

    poolMem.indexBufLen = tiling->indexBufLen;

    uint32_t floatNumPerBlock = GetDataBlockSizeInBytes() / sizeof(float);
    uint32_t src1RepStride = poolMem.alignC / floatNumPerBlock * poolMem.poolParam.strideW;

    isSumWithRepeat = (poolMem.poolParam.padW == 0 && !tiling->ceilMode) && src1RepStride <= UINT8_MAX;
    isSamePoolSize =
        poolMem.poolParam.divisorOverride || ((poolMem.poolParam.countIncludePad || poolMem.poolParam.padW == 0) && !tiling->ceilMode);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::CopyIn(
    int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding) {
    AvgPool3d::CopyInTemplate(poolMem, offset, blockCount, blockLen, rightPadding);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::DataCopyOutNonPad(
    LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint16_t blockCount, uint32_t validDataLen) {
    int64_t curPointIdx = outputPointIdx;
    for (int i = 0; i < blockCount; i++, curPointIdx++) {
        PipeBarrier<PIPE_MTE3>();
        if ((validDataLen < poolMem.numPerBlock) && (curPointIdx >= poolMem.lastPointOffset - poolMem.atomicAddNum)) {
            uint64_t mask0 = (1ul << poolMem.numPerBlock) - (1ul << validDataLen);
            uint64_t mask[2] = {mask0, 0};
            Duplicate<T>(outputLocal[i * poolMem.alignC], 0, mask, 1, 1, 1);
            VToMTE3Sync();
            if (curPointIdx > poolMem.lastPointOffset - poolMem.atomicAddNum) {
                SetAtomicAdd<T>();
                DataCopy(poolMem.outputGlobal[curPointIdx * validDataLen], outputLocal[i * poolMem.alignC], poolMem.alignC);
                SetAtomicNone();
                AscendC::PipeBarrier<PIPE_MTE3>();
            } else {
                DataCopy(poolMem.outputGlobal[curPointIdx * validDataLen], outputLocal[i * poolMem.alignC], poolMem.alignC);
            }
        } else if (curPointIdx == poolMem.lastPointOffset) {
            DataCopy(poolMem.outputGlobal[curPointIdx * validDataLen], outputLocal[i * poolMem.alignC], poolMem.alignC - poolMem.numPerBlock);
            uint32_t mask = poolMem.numPerBlock * 2;
            uint64_t gatherOffset = blockCount * poolMem.alignC - mask;
            HandleTailMask(outputLocal, gatherOffset, poolMem, mask);
            DataCopy(poolMem.outputGlobal[(curPointIdx + 1) * validDataLen - poolMem.numPerBlock], outputLocal[gatherOffset], poolMem.numPerBlock);
        } else {
            DataCopy(poolMem.outputGlobal[curPointIdx * validDataLen], outputLocal[i * poolMem.alignC], poolMem.alignC);
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::CopyOut(
    int64_t offset, uint16_t blockCount, uint32_t blockLen) {
    LocalTensor<T> outputLocal = poolMem.outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
    if (blockLen == poolMem.alignC) {
        DataCopyParams copyParams{blockCount, static_cast<uint16_t>(blockLen / poolMem.numPerBlock), 0, 0};
        DataCopy(poolMem.outputGlobal[offset * blockLen], outputLocal, copyParams);
    } else {
        DataCopyOutNonPad(outputLocal, offset, blockCount, blockLen);
    }
#else
    DataCopyExtParams copyParams{blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
    DataCopyPad(poolMem.outputGlobal[offset * blockLen], outputLocal, copyParams);
#endif
    poolMem.outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceSumRow(
    const Index& startIndex, LocalTensor<float>& sumBufLocal, LocalTensor<T>& inputLocal,
    int64_t outputPointIdx, int64_t windowNum) {
    for (int64_t in = outputPointIdx, offset = 0; in < outputPointIdx + windowNum; ++in, offset += poolMem.alignC) {
        Index index;
        poolMem.indexBuf.GetWIndex(in, index);

        SToVSync();

        for (int64_t iw = index.W.start - startIndex.W.start; iw < index.W.end - startIndex.W.start; ++iw) {
            if constexpr (std::is_same_v<T, float>) {
                Add(sumBufLocal[offset], sumBufLocal[offset], inputLocal[iw * poolMem.alignC], poolMem.alignC);
            } else {
                Add(sumBufLocal[offset], sumBufLocal[offset],
                    inputLocal.template ReinterpretCast<float>()[iw * poolMem.alignC], poolMem.alignC);
            }
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceSumRowRepeat(
    const Index& startIndex, LocalTensor<float>& sumBufLocal, LocalTensor<T>& inputLocal, int64_t windowNum) {
    int64_t poolSize = startIndex.W.end - startIndex.W.start;

    uint32_t floatNumPerBlock = GetDataBlockSizeInBytes() / sizeof(float);
    int64_t loop = (poolMem.alignC + floatNumPerBlock * 8 - 1) / (floatNumPerBlock * 8);

    uint8_t repStride = poolMem.alignC / floatNumPerBlock;
    uint8_t src1RepStride = poolMem.alignC / floatNumPerBlock * poolMem.poolParam.strideW;

    for (int64_t i = 0; i < poolSize; ++i) {
        for (int64_t j = 0; j < loop; ++j) {
            int64_t mask = j < loop - 1 ? floatNumPerBlock * 8 : poolMem.alignC - (loop - 1) * floatNumPerBlock * 8;

            BinaryRepeatParams repeatParams;
            repeatParams.dstBlkStride = 1;
            repeatParams.src0BlkStride = 1;
            repeatParams.src1BlkStride = 1;
            repeatParams.dstRepStride = repStride;
            repeatParams.src0RepStride = repStride;
            repeatParams.src1RepStride = src1RepStride;

            int64_t offset = j * floatNumPerBlock * 8;
            int64_t src1Offset = i * poolMem.alignC + offset;

            if constexpr (std::is_same_v<T, float>) {
                Add(sumBufLocal[offset], sumBufLocal[offset], inputLocal[src1Offset], mask, windowNum, repeatParams);
            } else {
                Add(sumBufLocal[offset], sumBufLocal[offset],
                    inputLocal.template ReinterpretCast<float>()[src1Offset], mask, windowNum, repeatParams);
            }
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceSumMultiWindow(
    const Index& startIndex, const Index& endIndex, LocalTensor<float>& sumBufLocal,
    int64_t outputPointIdx, int64_t nOffset, int64_t windowNum) {
    int64_t dstart = startIndex.D.start;
    int64_t dend = startIndex.D.end;
    int64_t hstart = startIndex.H.start;
    int64_t hend = startIndex.H.end;
    int64_t wStartOffset = startIndex.W.start * poolMem.inC;

    uint16_t blockCount = static_cast<uint16_t>(endIndex.W.end - startIndex.W.start);
    uint8_t rightPadding = static_cast<uint8_t>(poolMem.alignC - poolMem.inC);

    for (int64_t id = dstart; id < dend; ++id) {
        int64_t dOffset = id * poolMem.inputShape.strideD * poolMem.inC;
        for (int64_t ih = hstart; ih < hend; ++ih) {
            int64_t hOffset = ih * poolMem.inputShape.strideH * poolMem.inC;

            CopyIn(nOffset * poolMem.inputShape.strideN + dOffset + hOffset + wStartOffset, blockCount, poolMem.inC, rightPadding);
            LocalTensor<T> inputLocal = poolMem.inputQueue.template DeQue<T>();

            if constexpr (!std::is_same_v<T, float>) {
                Cast(inputLocal.template ReinterpretCast<float>(), inputLocal[poolMem.inputBufLen], RoundMode::CAST_NONE,
                     poolMem.inputBufLen);
            }

            if (isSumWithRepeat) [[likely]] {
                ReduceSumRowRepeat(startIndex, sumBufLocal, inputLocal, windowNum);
            } else {
                ReduceSumRow(startIndex, sumBufLocal, inputLocal, outputPointIdx, windowNum);
            }

            poolMem.inputQueue.FreeTensor(inputLocal);
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceMeanMultiWindow(
    int64_t outputPointIdx, int64_t windowNum) {
    Index startIndex;
    poolMem.indexBuf.GetIndex(outputPointIdx, startIndex);
    Index endIndex;
    poolMem.indexBuf.GetIndex(outputPointIdx + windowNum - 1, endIndex);

    int64_t len = windowNum * poolMem.alignC;

    SToVSync();

    Duplicate(poolMem.sumBufLocal, 0.0f, len);

    ReduceSumMultiWindow(startIndex, endIndex, poolMem.sumBufLocal, outputPointIdx,
                         outputPointIdx / poolMem.outputShape.strideC, windowNum);

    if (isSamePoolSize) [[likely]] {
        int64_t poolSize = poolMem.poolParam.divisorOverride
                            ? poolMem.poolParam.divisorOverride
                            : startIndex.D.poolSize * startIndex.H.poolSize * startIndex.W.poolSize;
        float factor = 1.0f / static_cast<float>(poolSize);

        Muls(poolMem.sumBufLocal, poolMem.sumBufLocal, factor, windowWNum * poolMem.alignC);
    } else {
        for (int64_t i = outputPointIdx, offset = 0; i < outputPointIdx + windowNum; ++i, offset += poolMem.alignC) {
            Index index;
            poolMem.indexBuf.GetWIndex(i, index);
            int64_t poolSize = startIndex.D.poolSize * startIndex.H.poolSize * index.W.poolSize;
            float factor = 1.0f / static_cast<float>(poolSize);

            SToVSync();

            Muls(poolMem.sumBufLocal[offset], poolMem.sumBufLocal[offset], factor, poolMem.alignC);
        }
    }

    LocalTensor<T> outputLocal = poolMem.outputQueue.template AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
#if __CCE_AICORE__ < 220
        Adds(outputLocal, poolMem.sumBufLocal, 0.0f, len);
#else
        DataCopy(outputLocal, poolMem.sumBufLocal, len);
#endif
    } else if constexpr (std::is_same_v<T, half>) {
        Cast(outputLocal, poolMem.sumBufLocal, RoundMode::CAST_NONE, len);
    } else {
        Cast(outputLocal, poolMem.sumBufLocal, RoundMode::CAST_RINT, len);
    }
    poolMem.outputQueue.EnQue(outputLocal);

    CopyOut(outputPointIdx, static_cast<uint16_t>(windowNum), static_cast<uint32_t>(poolMem.inC));
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe) {
    InitTiling(tiling);

    poolMem.pipe = pipe;
    poolMem.inputGlobal.SetGlobalBuffer((__gm__ T*)x);
    poolMem.outputGlobal.SetGlobalBuffer((__gm__ T*)y);

    poolMem.inputBufLen = (windowWNum * poolMem.poolParam.strideW + poolMem.poolParam.kernelW) * poolMem.alignC;

    pipe->InitBuffer(poolMem.inputQueue, QUEUE_DEPTH, poolMem.inputBufLen * sizeof(float));
    pipe->InitBuffer(poolMem.outputQueue, QUEUE_DEPTH, windowWNum * poolMem.alignC * sizeof(T));

    InitCommonBuffers(poolMem, workspace);
    pipe->InitBuffer(poolMem.sumBuf, windowWNum * poolMem.alignC * sizeof(float));
    poolMem.sumBufLocal = poolMem.sumBuf.template Get<float>();

    poolMem.indexBuf.Init(pipe, poolMem.indexBufLen);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::Process() {
#if __CCE_AICORE__ < 220
    AvgPool3d::HandleAtomicAdd(poolMem);
#endif
    int64_t curWindowWNum = windowWNum;
    for (int64_t outputPointIdx = poolMem.outputPointOffset, count = 0;
        outputPointIdx < poolMem.outputPointOffset + poolMem.outputPointNum; outputPointIdx += curWindowWNum, count += curWindowWNum) {
        curWindowWNum = (count + windowWNum) < poolMem.outputPointNum ? windowWNum : poolMem.outputPointNum - count;

        int64_t newRowWindowWNum = (outputPointIdx + curWindowWNum) % poolMem.outputShape.W;
        curWindowWNum = newRowWindowWNum != 0 && newRowWindowWNum < curWindowWNum
                          ? curWindowWNum - newRowWindowWNum : curWindowWNum;

        ReduceMeanMultiWindow(outputPointIdx, curWindowWNum);
    }
}

} // namespace AvgPool3d

#endif // AVG_POOL3D_NDHWC_MULTI_W_H_
