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
 * \file avg_pool3_d_ndhwc_split_c.h
 * \brief
 */

#ifndef AVG_POOL3D_NDHWC_SPLIT_C_H_
#define AVG_POOL3D_NDHWC_SPLIT_C_H_

#include "kernel_operator.h"
#include "avg_pool3d_common.h"

namespace AvgPool3d {
template <typename T, int32_t QUEUE_DEPTH>
class KernelAvgPool3dSplitC {
public:
    __aicore__ inline KernelAvgPool3dSplitC() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTiling(const AvgPool3DTilingData* __restrict__ tiling);
    __aicore__ inline void CopyIn(int64_t offset, int64_t len);
    __aicore__ inline void CopyOut(int64_t offset, int64_t len);
    __aicore__ inline void DataCopyOutNonPad(LocalTensor<T>& outputLocal, int64_t offset, int64_t validDataLen);
    __aicore__ inline void ReduceMeanWindow(int64_t outputPointIdx);
    __aicore__ inline void ReduceSumWindow(
        const Index& index, LocalTensor<float>& sumBufLocal, int64_t nOffset, int64_t cOffset, int64_t len);

    int64_t tileC;
    int64_t cTailLength;
    int64_t cTailAlign;

    PoolMem<T, QUEUE_DEPTH> poolMem;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::InitTiling(const AvgPool3DTilingData* __restrict__ tiling) {
    poolMem.inputShape = PoolShape(tiling->inN, tiling->inC, tiling->inD, tiling->inH, tiling->inW);
    poolMem.outputShape = PoolShape(tiling->inN, tiling->inC, tiling->outD, tiling->outH, tiling->outW);

    poolMem.poolParam = PoolParameter(tiling->kD, tiling->kH, tiling->kW, tiling->dD, tiling->dH, tiling->dW,
                              tiling->pD, tiling->pH, tiling->pW, tiling->divisorOverride, tiling->countIncludePad);

    poolMem.indexBuf.SetComputeParameter(poolMem.outputShape, poolMem.inputShape, poolMem.poolParam);

    poolMem.numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);
    poolMem.inC = tiling->inC;
    tileC = tiling->tileC;
    poolMem.indexBufLen = tiling->indexBufLen;

    poolMem.outputPointNum = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
    poolMem.outputPointOffset = GetBlockIdx() < tiling->formerNum
        ? tiling->formerLength * GetBlockIdx()
        : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);
    poolMem.nextCoreAddrOffset = (poolMem.outputPointOffset + poolMem.outputPointNum) * poolMem.inC;
    poolMem.atomicAddNum = tiling->atomicAddNum;
    cTailLength = poolMem.inC % tileC;
    cTailAlign = AlignUp(cTailLength, poolMem.numPerBlock);
    poolMem.validTailLen = cTailLength % poolMem.numPerBlock;
    poolMem.usedCoreNum = tiling->usedCoreNum;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::CopyIn(int64_t offset, int64_t len) {
    LocalTensor<T> inputLocal = poolMem.inputQueue.template AllocTensor<T>();
#if __CCE_AICORE__ < 220
    if constexpr (std::is_same_v<T, float>) {
        if (len == tileC) {
            DataCopyParams copyParams{1, static_cast<uint16_t>(len / poolMem.numPerBlock), 0, 0};
            DataCopy(inputLocal, poolMem.inputGlobal[offset], copyParams);
        } else {
            DataCopy(inputLocal, poolMem.inputGlobal[offset], cTailAlign);
        }
    } else {
      if (len == tileC) {
          DataCopyParams copyParams{1, static_cast<uint16_t>(len / poolMem.numPerBlock), 0, 0};
          DataCopy(inputLocal[tileC], poolMem.inputGlobal[offset], copyParams);
      } else {
          DataCopy(inputLocal[tileC], poolMem.inputGlobal[offset], cTailAlign);
      }
    }
#else
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(inputLocal, poolMem.inputGlobal[offset], copyParams, padParams);
    } else {
        DataCopyPad(inputLocal[tileC], poolMem.inputGlobal[offset], copyParams, padParams);
    }
#endif
    poolMem.inputQueue.EnQue(inputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::DataCopyOutNonPad(
    LocalTensor<T>& outputLocal, int64_t offset, int64_t validDataLen) {
    if ((validDataLen < poolMem.numPerBlock) && (offset + validDataLen * poolMem.atomicAddNum >= poolMem.nextCoreAddrOffset)) {
        uint64_t mask0 = (1ul << poolMem.numPerBlock) - (1ul << validDataLen);
        uint64_t mask[2] = {mask0, 0};
        Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
        VToMTE3Sync();
        SetAtomicAdd<T>();
        DataCopy(poolMem.outputGlobal[offset], outputLocal, cTailAlign);
        SetAtomicNone();
        AscendC::PipeBarrier<PIPE_MTE3>();
    } else if ((poolMem.validTailLen != 0) && (offset + validDataLen == poolMem.nextCoreAddrOffset)) {
        DataCopy(poolMem.outputGlobal[offset], outputLocal, cTailAlign - poolMem.numPerBlock);
        int32_t lastLeftShift = poolMem.validTailLen;
        uint32_t mask = poolMem.numPerBlock * 2;
        uint64_t rsvdCnt = 0;
        uint64_t gatherOffset = cTailAlign - mask;
        MTE3ToVSync();
        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<uint32_t> bufPattern = poolMem.tmpPattern.template Get<uint32_t>();
            int32_t preLeftShift = poolMem.numPerBlock + lastLeftShift;

            bufPattern.SetValue(0, (1u << preLeftShift) - (1u << lastLeftShift));
            SToVSync();
            GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset], bufPattern, true, mask, {1, 1, 8, 8},
                       rsvdCnt);
        } else {
            LocalTensor<uint16_t> bufPattern = poolMem.tmpPattern.template Get<uint16_t>();
            int32_t preLeftShift = poolMem.numPerBlock - lastLeftShift;

            bufPattern.SetValue(0, ((1u << preLeftShift) - 1u) << lastLeftShift);
            bufPattern.SetValue(1, (1u << lastLeftShift) - 1u);
            SToVSync();
            GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset], bufPattern, true, mask, {1, 1, 8, 8},
                       rsvdCnt);
        }
        VToMTE3Sync();
        DataCopy(poolMem.outputGlobal[poolMem.nextCoreAddrOffset - poolMem.numPerBlock], outputLocal[gatherOffset], poolMem.numPerBlock);
    } else {
        DataCopy(poolMem.outputGlobal[offset], outputLocal, cTailAlign);
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::CopyOut(int64_t offset, int64_t len) {
    LocalTensor<T> outputLocal = poolMem.outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
    if (len == tileC) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(len / poolMem.numPerBlock), 0, 0};
        DataCopy(poolMem.outputGlobal[offset], outputLocal, copyParams);
    } else {
        DataCopyOutNonPad(outputLocal, offset, len);
    }
#else
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(poolMem.outputGlobal[offset], outputLocal, copyParams);
#endif
    poolMem.outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::ReduceSumWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t nOffset, int64_t cOffset, int64_t len) {
    int64_t dstart = index.D.start;
    int64_t dend = index.D.end;
    int64_t hstart = index.H.start;
    int64_t hend = index.H.end;
    int64_t wstart = index.W.start;
    int64_t wend = index.W.end;

    int64_t startOffset = nOffset * poolMem.inputShape.strideN + cOffset;
    for (int64_t id = dstart; id < dend; ++id) {
        int64_t dOffset = id * poolMem.inputShape.strideD;
        for (int64_t ih = hstart; ih < hend; ++ih) {
            int64_t hOffset = ih * poolMem.inputShape.strideH;
            for (int64_t iw = wstart; iw < wend; ++iw) {
                CopyIn(startOffset + (dOffset + hOffset + iw * poolMem.inputShape.strideW) * poolMem.inC, len);

                LocalTensor<T> inputLocal = poolMem.inputQueue.template DeQue<T>();
                if constexpr (std::is_same_v<T, float>) {
                    Add(sumBufLocal, sumBufLocal, inputLocal, len);
                } else {
                    Cast(inputLocal.template ReinterpretCast<float>(), inputLocal[tileC], RoundMode::CAST_NONE, len);
                    Add(sumBufLocal, sumBufLocal, inputLocal.template ReinterpretCast<float>(), len);
                }
                poolMem.inputQueue.FreeTensor(inputLocal);
            }
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::ReduceMeanWindow(int64_t outputPointIdx) {
    Index index;
    poolMem.indexBuf.GetIndex(outputPointIdx, index);

    int64_t poolSize = poolMem.poolParam.divisorOverride ?
                      poolMem.poolParam.divisorOverride : index.D.poolSize * index.H.poolSize * index.W.poolSize;
    float factor = 1.0f / static_cast<float>(poolSize);

    SToVSync();

    int64_t cLoop = (poolMem.inC + tileC - 1) / tileC;
    int64_t cOffset = 0;
    for (int64_t i = 0; i < cLoop; ++i) {
        int64_t count = i < cLoop - 1 ? tileC : poolMem.inC - (cLoop - 1) * tileC;

        Duplicate(poolMem.sumBufLocal, 0.0f, count);

        ReduceSumWindow(index, poolMem.sumBufLocal, outputPointIdx / poolMem.outputShape.strideC, cOffset, count);
        CastAndEnqueueOutput<T, QUEUE_DEPTH>(poolMem, count, factor);

        CopyOut(outputPointIdx * poolMem.inC + cOffset, count);

        cOffset += count;
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe) {
    InitTiling(tiling);

    poolMem.pipe = pipe;
    poolMem.inputGlobal.SetGlobalBuffer((__gm__ T*)x);
    poolMem.outputGlobal.SetGlobalBuffer((__gm__ T*)y);

    pipe->InitBuffer(poolMem.inputQueue, QUEUE_DEPTH, tileC * sizeof(float));
    pipe->InitBuffer(poolMem.outputQueue, QUEUE_DEPTH, tileC * sizeof(T));

    InitCommonBuffers(poolMem, workspace);

    pipe->InitBuffer(poolMem.sumBuf, tileC * sizeof(float));
    poolMem.sumBufLocal = poolMem.sumBuf.template Get<float>();

    poolMem.indexBuf.Init(pipe, poolMem.indexBufLen);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitC<T, QUEUE_DEPTH>::Process() {
#if __CCE_AICORE__ < 220
    int64_t curOutputPointIdx = poolMem.outputPointNum + poolMem.outputPointOffset - 1;
    AvgPool3d::HandleAtomicAddWithTail(poolMem, curOutputPointIdx, cTailLength);
#endif
    for (int64_t outputPointIdx = poolMem.outputPointOffset;
        outputPointIdx < poolMem.outputPointOffset + poolMem.outputPointNum; ++outputPointIdx) {
        ReduceMeanWindow(outputPointIdx);
    }
}

} // namespace AvgPool3d

#endif // AVG_POOL3D_NDHWC_SPLIT_C_H_
