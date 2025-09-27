/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTiling(const AvgPool3DTilingData* tiling);
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

    TPipe* pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inputQueue;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueue;

    TBuf<QuePosition::VECCALC> tmpPattern;
    TBuf<TPosition::VECCALC> sumBuf;
    LocalTensor<float> sumBufLocal;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;

    int64_t inC;
    int64_t alignC;
    int64_t outputPointNum;
    int64_t outputPointOffset;
    int64_t lastPointOffset;
    int64_t windowWNum;
    int64_t atomicAddNum;

    PoolShape inputShape;
    PoolShape outputShape;

    int64_t indexBufLen;
    IndexBuffer indexBuf;
    PoolParameter poolParam;

    uint32_t numPerBlock;
    uint32_t inputBufLen;
    int32_t validTailLen;

    bool isSumWithRepeat;
    bool isSamePoolSize;

    TQue<QuePosition::VECIN, QUEUE_DEPTH> syncWorkQueue;
    GlobalTensor<int32_t> syncTensorsGM;
    TBuf<TPosition::VECCALC> clearTensorBuff;
    uint32_t usedCoreNum;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::InitTiling(const AvgPool3DTilingData* tiling) {
    inputShape = PoolShape(tiling->inN, tiling->inC, tiling->inD, tiling->inH, tiling->inW);
    outputShape = PoolShape(tiling->inN, tiling->inC, tiling->outD, tiling->outH, tiling->outW);

    poolParam = PoolParameter(tiling->kD, tiling->kH, tiling->kW, tiling->dD, tiling->dH, tiling->dW,
                              tiling->pD, tiling->pH, tiling->pW, tiling->divisorOverride, tiling->countIncludePad);

    indexBuf.SetComputeParameter(outputShape, inputShape, poolParam);

    numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);
    inC = tiling->inC;
    alignC = AlignUp(inC, numPerBlock);
    windowWNum = tiling->windowWNum;

    outputPointNum = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
    outputPointOffset = GetBlockIdx() < tiling->formerNum
        ? tiling->formerLength * GetBlockIdx()
        : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);
    lastPointOffset = outputPointNum + outputPointOffset - 1;
    atomicAddNum = outputPointNum < tiling->atomicAddNum ? outputPointNum : tiling->atomicAddNum;
    validTailLen = inC % numPerBlock;
    usedCoreNum = tiling->usedCoreNum;

    indexBufLen = tiling->indexBufLen;

    uint32_t floatNumPerBlock = GetDataBlockSizeInBytes() / sizeof(float);
    uint32_t src1RepStride = alignC / floatNumPerBlock * poolParam.strideW;

    isSumWithRepeat = (poolParam.padW == 0 && !tiling->ceilMode) && src1RepStride <= UINT8_MAX;
    isSamePoolSize =
        poolParam.divisorOverride || ((poolParam.countIncludePad || poolParam.padW == 0) && !tiling->ceilMode);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::CopyIn(
    int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding) {
    LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
#if __CCE_AICORE__ < 220
    if constexpr (std::is_same_v<T, float>) {
        if (blockLen == alignC) {
            DataCopyParams copyParams{blockCount, static_cast<uint16_t>(blockLen / numPerBlock), 0, 0};
            DataCopy(inputLocal, inputGlobal[offset], copyParams);
        } else {
            for (int i = 0; i < blockCount; i++) {
                DataCopy(inputLocal[i * alignC], inputGlobal[offset + i * blockLen], alignC);
            }
        }
    } else {
        if (blockLen == alignC) {
            DataCopyParams copyParams{blockCount, static_cast<uint16_t>(blockLen / numPerBlock), 0, 0};
            DataCopy(inputLocal[inputBufLen], inputGlobal[offset], copyParams);
        } else {
            for (int i = 0; i < blockCount; i++) {
                DataCopy(inputLocal[inputBufLen + i * alignC], inputGlobal[offset + i * blockLen], alignC);
            }
        }
    }
#else
    DataCopyExtParams copyParams{blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, rightPadding, 0};
    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(inputLocal, inputGlobal[offset], copyParams, padParams);
    } else {
        DataCopyPad(inputLocal[inputBufLen], inputGlobal[offset], copyParams, padParams);
    }
#endif
    inputQueue.EnQue(inputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::DataCopyOutNonPad(
    LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint16_t blockCount, uint32_t validDataLen) {
    int64_t curPointIdx = outputPointIdx;
    for (int i = 0; i < blockCount; i++, curPointIdx++) {
        PipeBarrier<PIPE_MTE3>();
        if ((validDataLen < numPerBlock) && (curPointIdx >= lastPointOffset - atomicAddNum)) {
            uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
            uint64_t mask[2] = {mask0, 0};
            Duplicate<T>(outputLocal[i * alignC], 0, mask, 1, 1, 1);
            VToMTE3Sync();
            if (curPointIdx > lastPointOffset - atomicAddNum) {
                SetAtomicAdd<T>();
                DataCopy(outputGlobal[curPointIdx * validDataLen], outputLocal[i * alignC], alignC);
                SetAtomicNone();
                AscendC::PipeBarrier<PIPE_MTE3>();
            } else {
                DataCopy(outputGlobal[curPointIdx * validDataLen], outputLocal[i * alignC], alignC);
            }
        } else if (curPointIdx == lastPointOffset) {
            DataCopy(outputGlobal[curPointIdx * validDataLen], outputLocal[i * alignC], alignC - numPerBlock);
            int32_t lastLeftShift = validTailLen;
            uint32_t mask = numPerBlock * 2;
            uint64_t rsvdCnt = 0;
            uint64_t gatherOffset = blockCount * alignC - mask;
            MTE3ToVSync();
            if constexpr (std::is_same_v<T, float>) {
                LocalTensor<uint32_t> bufPattern = tmpPattern.Get<uint32_t>();
                int32_t preLeftShift = numPerBlock + lastLeftShift;

                bufPattern.SetValue(0, (1u << preLeftShift) - (1u << lastLeftShift));
                SToVSync();
                GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset], bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
            } else {
                LocalTensor<uint16_t> bufPattern = tmpPattern.Get<uint16_t>();
                int32_t preLeftShift = numPerBlock - lastLeftShift;

                bufPattern.SetValue(0, ((1u << preLeftShift) - 1u) << lastLeftShift);
                bufPattern.SetValue(1, (1u << lastLeftShift) - 1u);
                SToVSync();
                GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset], bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
            }
            VToMTE3Sync();
            DataCopy(outputGlobal[(curPointIdx + 1) * validDataLen - numPerBlock], outputLocal[gatherOffset], numPerBlock);
        } else {
            DataCopy(outputGlobal[curPointIdx * validDataLen], outputLocal[i * alignC], alignC);
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::CopyOut(
    int64_t offset, uint16_t blockCount, uint32_t blockLen) {
    LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
    if (blockLen == alignC) {
        DataCopyParams copyParams{blockCount, static_cast<uint16_t>(blockLen / numPerBlock), 0, 0};
        DataCopy(outputGlobal[offset * blockLen], outputLocal, copyParams);
    } else {
        DataCopyOutNonPad(outputLocal, offset, blockCount, blockLen);
    }
#else
    DataCopyExtParams copyParams{blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGlobal[offset * blockLen], outputLocal, copyParams);
#endif
    outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceSumRow(
    const Index& startIndex, LocalTensor<float>& sumBufLocal, LocalTensor<T>& inputLocal,
    int64_t outputPointIdx, int64_t windowNum) {
    for (int64_t in = outputPointIdx, offset = 0; in < outputPointIdx + windowNum; ++in, offset += alignC) {
        Index index;
        indexBuf.GetWIndex(in, index);

        SToVSync();

        for (int64_t iw = index.W.start - startIndex.W.start; iw < index.W.end - startIndex.W.start; ++iw) {
            if constexpr (std::is_same_v<T, float>) {
                Add(sumBufLocal[offset], sumBufLocal[offset], inputLocal[iw * alignC], alignC);
            } else {
                Add(sumBufLocal[offset], sumBufLocal[offset],
                    inputLocal.template ReinterpretCast<float>()[iw * alignC], alignC);
            }
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceSumRowRepeat(
    const Index& startIndex, LocalTensor<float>& sumBufLocal, LocalTensor<T>& inputLocal, int64_t windowNum) {
    int64_t poolSize = startIndex.W.end - startIndex.W.start;

    uint32_t floatNumPerBlock = GetDataBlockSizeInBytes() / sizeof(float);
    int64_t loop = (alignC + floatNumPerBlock * 8 - 1) / (floatNumPerBlock * 8);

    uint8_t repStride = alignC / floatNumPerBlock;
    uint8_t src1RepStride = alignC / floatNumPerBlock * poolParam.strideW;

    for (int64_t i = 0; i < poolSize; ++i) {
        for (int64_t j = 0; j < loop; ++j) {
            int64_t mask = j < loop - 1 ? floatNumPerBlock * 8 : alignC - (loop - 1) * floatNumPerBlock * 8;

            BinaryRepeatParams repeatParams;
            repeatParams.dstBlkStride = 1;
            repeatParams.src0BlkStride = 1;
            repeatParams.src1BlkStride = 1;
            repeatParams.dstRepStride = repStride;
            repeatParams.src0RepStride = repStride;
            repeatParams.src1RepStride = src1RepStride;

            int64_t offset = j * floatNumPerBlock * 8;
            int64_t src1Offset = i * alignC + offset;

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
    int64_t wStartOffset = startIndex.W.start * inC;

    uint16_t blockCount = static_cast<uint16_t>(endIndex.W.end - startIndex.W.start);
    uint8_t rightPadding = static_cast<uint8_t>(alignC - inC);

    for (int64_t id = dstart; id < dend; ++id) {
        int64_t dOffset = id * inputShape.strideD * inC;
        for (int64_t ih = hstart; ih < hend; ++ih) {
            int64_t hOffset = ih * inputShape.strideH * inC;

            CopyIn(nOffset * inputShape.strideN + dOffset + hOffset + wStartOffset, blockCount, inC, rightPadding);
            LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();

            if constexpr (!std::is_same_v<T, float>) {
                Cast(inputLocal.template ReinterpretCast<float>(), inputLocal[inputBufLen], RoundMode::CAST_NONE,
                     inputBufLen);
            }

            if (isSumWithRepeat) [[likely]] {
                ReduceSumRowRepeat(startIndex, sumBufLocal, inputLocal, windowNum);
            } else {
                ReduceSumRow(startIndex, sumBufLocal, inputLocal, outputPointIdx, windowNum);
            }

            inputQueue.FreeTensor(inputLocal);
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::ReduceMeanMultiWindow(
    int64_t outputPointIdx, int64_t windowNum) {
    Index startIndex;
    indexBuf.GetIndex(outputPointIdx, startIndex);
    Index endIndex;
    indexBuf.GetIndex(outputPointIdx + windowNum - 1, endIndex);

    int64_t len = windowNum * alignC;

    SToVSync();

    Duplicate(sumBufLocal, 0.0f, len);

    ReduceSumMultiWindow(startIndex, endIndex, sumBufLocal, outputPointIdx,
                         outputPointIdx / outputShape.strideC, windowNum);

    if (isSamePoolSize) [[likely]] {
        int64_t poolSize = poolParam.divisorOverride
                            ? poolParam.divisorOverride
                            : startIndex.D.poolSize * startIndex.H.poolSize * startIndex.W.poolSize;
        float factor = 1.0f / static_cast<float>(poolSize);

        Muls(sumBufLocal, sumBufLocal, factor, windowWNum * alignC);
    } else {
        for (int64_t i = outputPointIdx, offset = 0; i < outputPointIdx + windowNum; ++i, offset += alignC) {
            Index index;
            indexBuf.GetWIndex(i, index);
            int64_t poolSize = startIndex.D.poolSize * startIndex.H.poolSize * index.W.poolSize;
            float factor = 1.0f / static_cast<float>(poolSize);

            SToVSync();

            Muls(sumBufLocal[offset], sumBufLocal[offset], factor, alignC);
        }
    }

    LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
#if __CCE_AICORE__ < 220
        Adds(outputLocal, sumBufLocal, 0.0f, len);
#else
        DataCopy(outputLocal, sumBufLocal, len);
#endif
    } else if constexpr (std::is_same_v<T, half>) {
        Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, len);
    } else {
        Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, len);
    }
    outputQueue.EnQue(outputLocal);

    CopyOut(outputPointIdx, static_cast<uint16_t>(windowNum), static_cast<uint32_t>(inC));
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* tiling, TPipe* pipe) {
    InitTiling(tiling);

    inputGlobal.SetGlobalBuffer((__gm__ T*)x);
    outputGlobal.SetGlobalBuffer((__gm__ T*)y);

    inputBufLen = (windowWNum * poolParam.strideW + poolParam.kernelW) * alignC;

    pipe->InitBuffer(inputQueue, QUEUE_DEPTH, inputBufLen * sizeof(float));
    pipe->InitBuffer(outputQueue, QUEUE_DEPTH, windowWNum * alignC * sizeof(T));

#if __CCE_AICORE__ < 220
    if (atomicAddNum) {
        pipe->InitBuffer(tmpPattern, numPerBlock * sizeof(T));

        pipe->InitBuffer(syncWorkQueue, QUEUE_DEPTH, 8 * 32 * sizeof(int32_t));
        syncTensorsGM.SetGlobalBuffer((__gm__ int32_t *)workspace, usedCoreNum * 8 * 32);
        pipe->InitBuffer(clearTensorBuff, DEFAULT_CLEAR_UB_SIZE * sizeof(T));
    } else if (validTailLen != 0) {
        pipe->InitBuffer(tmpPattern, numPerBlock * sizeof(T));
    }
#endif
    pipe->InitBuffer(sumBuf, windowWNum * alignC * sizeof(float));
    sumBufLocal = sumBuf.Get<float>();

    indexBuf.Init(pipe, indexBufLen);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dMultiW<T, QUEUE_DEPTH>::Process() {
#if __CCE_AICORE__ < 220
    if (atomicAddNum) {
        LocalTensor<T> clearUb = clearTensorBuff.Get<T>();
        Duplicate(clearUb, (T)0, DEFAULT_CLEAR_UB_SIZE);

        VToMTE3Sync();
        int64_t curOutputPointIdx = lastPointOffset;
        for (int i = 0; i < atomicAddNum; i++, curOutputPointIdx--) {
            DataCopy<T>(outputGlobal[curOutputPointIdx * inC], clearUb, numPerBlock);
        }

        DataCopy(syncTensorsGM[0], clearUb.template ReinterpretCast<int32_t>(), usedCoreNum * 8 * 32);
        LocalTensor<int32_t> syncLocalTensor = syncWorkQueue.template AllocTensor<int32_t>();
        AscendC::SyncAll(syncTensorsGM, syncLocalTensor, int32_t(usedCoreNum));
        syncWorkQueue.FreeTensor(syncLocalTensor);
    }
#endif
    int64_t curWindowWNum = windowWNum;
    for (int64_t outputPointIdx = outputPointOffset, count = 0;
        outputPointIdx < outputPointOffset + outputPointNum; outputPointIdx += curWindowWNum, count += curWindowWNum) {
        curWindowWNum = (count + windowWNum) < outputPointNum ? windowWNum : outputPointNum - count;

        int64_t newRowWindowWNum = (outputPointIdx + curWindowWNum) % outputShape.W;
        curWindowWNum = newRowWindowWNum != 0 && newRowWindowWNum < curWindowWNum
                          ? curWindowWNum - newRowWindowWNum : curWindowWNum;

        ReduceMeanMultiWindow(outputPointIdx, curWindowWNum);
    }
}

} // namespace AvgPool3d

#endif // AVG_POOL3D_NDHWC_MULTI_W_H_
