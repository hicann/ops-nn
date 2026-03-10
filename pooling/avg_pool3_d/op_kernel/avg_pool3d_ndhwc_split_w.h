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
 * \file avg_pool3_d_ndhwc_split_w.h
 * \brief
 */

#ifndef AVG_POOL3D_NDHWC_SPLIT_W_H_
#define AVG_POOL3D_NDHWC_SPLIT_W_H_

#include "kernel_operator.h"
#include "avg_pool3d_common.h"

namespace AvgPool3d {
template <typename T, int32_t QUEUE_DEPTH>
class KernelAvgPool3dSplitW {
public:
    __aicore__ inline KernelAvgPool3dSplitW() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTiling(const AvgPool3DTilingData* __restrict__ tiling);
    __aicore__ inline void CopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding);
    __aicore__ inline void CopyOut(int64_t offset, int64_t len);
    __aicore__ inline void DataCopyOutNonPad(
        LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint32_t validDataLen);
    __aicore__ inline void ReduceMeanWindow(int64_t outputPointIdx);
    __aicore__ inline void ReduceSumWindow(const Index& index, LocalTensor<float>& sumBufLocal, int64_t nOffset);

    int64_t tileInput;
    PoolMem<T, QUEUE_DEPTH> poolMem;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::InitTiling(const AvgPool3DTilingData* __restrict__ tiling) {
    poolMem.inputShape = PoolShape(tiling->inN, tiling->inC, tiling->inD, tiling->inH, tiling->inW);
    poolMem.outputShape = PoolShape(tiling->inN, tiling->inC, tiling->outD, tiling->outH, tiling->outW);

    poolMem.poolParam = PoolParameter(tiling->kD, tiling->kH, tiling->kW, tiling->dD, tiling->dH, tiling->dW,
                              tiling->pD, tiling->pH, tiling->pW, tiling->divisorOverride, tiling->countIncludePad);

    poolMem.indexBuf.SetComputeParameter(poolMem.outputShape, poolMem.inputShape, poolMem.poolParam);

    poolMem.numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);
    poolMem.inC = tiling->inC;
    poolMem.alignC = AlignUp(poolMem.inC, poolMem.numPerBlock);
    tileInput = tiling->tileInput;

    poolMem.outputPointNum = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
    poolMem.outputPointOffset = GetBlockIdx() < tiling->formerNum
        ? tiling->formerLength * GetBlockIdx()
        : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);
    poolMem.lastPointOffset = poolMem.outputPointNum + poolMem.outputPointOffset - 1;
    poolMem.atomicAddNum = poolMem.outputPointNum < tiling->atomicAddNum ? poolMem.outputPointNum : tiling->atomicAddNum;
    poolMem.indexBufLen = tiling->indexBufLen;
    poolMem.validTailLen = poolMem.inC % poolMem.numPerBlock;
    poolMem.usedCoreNum = tiling->usedCoreNum;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::CopyIn(
    int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding) {
    AvgPool3d::CopyInTemplate(poolMem, offset, blockCount, blockLen, rightPadding);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::DataCopyOutNonPad(
    LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint32_t validDataLen) {
    if ((validDataLen < poolMem.numPerBlock) && (outputPointIdx >= poolMem.lastPointOffset - poolMem.atomicAddNum)) {
      uint64_t mask0 = (1ul << poolMem.numPerBlock) - (1ul << validDataLen);
      uint64_t mask[2] = {mask0, 0};
      Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
      VToMTE3Sync();
      if (outputPointIdx > poolMem.lastPointOffset - poolMem.atomicAddNum) {
          SetAtomicAdd<T>();
          DataCopy(poolMem.outputGlobal[outputPointIdx * validDataLen], outputLocal, poolMem.alignC);
          SetAtomicNone();
          AscendC::PipeBarrier<PIPE_MTE3>();
      } else {
          DataCopy(poolMem.outputGlobal[outputPointIdx * validDataLen], outputLocal, poolMem.alignC);
      }
    } else if (outputPointIdx == poolMem.lastPointOffset) {
        DataCopy(poolMem.outputGlobal[outputPointIdx * validDataLen], outputLocal, poolMem.alignC - poolMem.numPerBlock);
        uint32_t mask = poolMem.numPerBlock * 2;
        uint64_t gatherOffset = poolMem.alignC - mask;
        HandleTailMask(outputLocal, gatherOffset, poolMem, mask);
        DataCopy(poolMem.outputGlobal[(outputPointIdx + 1) * validDataLen - poolMem.numPerBlock], outputLocal[gatherOffset], poolMem.numPerBlock);
    } else {
        DataCopy(poolMem.outputGlobal[outputPointIdx * validDataLen], outputLocal, poolMem.alignC);
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::CopyOut(int64_t offset, int64_t len) {
    LocalTensor<T> outputLocal = poolMem.outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
    if (len == poolMem.alignC) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(len / poolMem.numPerBlock), 0, 0};
        DataCopy(poolMem.outputGlobal[offset * len], outputLocal, copyParams);
    } else {
        DataCopyOutNonPad(outputLocal, offset, len);
    }
#else
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(poolMem.outputGlobal[offset * len], outputLocal, copyParams);
#endif
    poolMem.outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::ReduceSumWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t nOffset) {
    int64_t dstart = index.D.start;
    int64_t dend = index.D.end;
    int64_t hstart = index.H.start;
    int64_t hend = index.H.end;
    int64_t wstart = index.W.start;
    int64_t wend = index.W.end;

    int64_t kW = (wend - wstart + tileInput - 1) / tileInput;
    uint8_t rightPadding = static_cast<uint8_t>(poolMem.alignC - poolMem.inC);

    for (int64_t id = dstart; id < dend; ++id) {
        int64_t dOffset = id * poolMem.inputShape.strideD * poolMem.inC;
        for (int64_t ih = hstart; ih < hend; ++ih) {
            int64_t hOffset = ih * poolMem.inputShape.strideH * poolMem.inC;
            for (int64_t j = 0, iw = wstart; j < kW; ++j) {
                int64_t tileNum = j < kW - 1 ? tileInput : wend - iw;

                CopyIn(nOffset * poolMem.inputShape.strideN + dOffset + hOffset + iw * poolMem.inC,
                      static_cast<uint16_t>(tileNum), static_cast<uint32_t>(poolMem.inC), rightPadding);
                LocalTensor<T> inputLocal = poolMem.inputQueue.template DeQue<T>();

                if constexpr (!std::is_same_v<T, float>) {
                    Cast(inputLocal.template ReinterpretCast<float>(), inputLocal[poolMem.inputBufLen],
                         RoundMode::CAST_NONE, poolMem.inputBufLen);
                }

                for (int64_t i = 0; i < tileNum; ++i) {
                    if constexpr (std::is_same_v<T, float>) {
                        Add(sumBufLocal, sumBufLocal, inputLocal[i * poolMem.alignC], poolMem.alignC);
                    } else {
                        Add(sumBufLocal, sumBufLocal, inputLocal.template ReinterpretCast<float>()[i * poolMem.alignC], poolMem.alignC);
                    }
                }

                iw += tileNum;

                poolMem.inputQueue.FreeTensor(inputLocal);
            }
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::ReduceMeanWindow(int64_t outputPointIdx) {
    Index index;
    poolMem.indexBuf.GetIndex(outputPointIdx, index);

    int64_t poolSize = poolMem.poolParam.divisorOverride ?
                       poolMem.poolParam.divisorOverride : index.D.poolSize * index.H.poolSize * index.W.poolSize;
    float factor = 1.0f / static_cast<float>(poolSize);

    SToVSync();

    Duplicate(poolMem.sumBufLocal, 0.0f, poolMem.alignC);

    ReduceSumWindow(index, poolMem.sumBufLocal, outputPointIdx / poolMem.outputShape.strideC);
    Muls(poolMem.sumBufLocal, poolMem.sumBufLocal, factor, poolMem.alignC);

    LocalTensor<T> outputLocal = poolMem.outputQueue.template AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
#if __CCE_AICORE__ < 220
        Adds(outputLocal, poolMem.sumBufLocal, 0.0f, poolMem.alignC);
#else
        DataCopy(outputLocal, poolMem.sumBufLocal, poolMem.alignC);
#endif
    } else if constexpr (std::is_same_v<T, half>) {
        Cast(outputLocal, poolMem.sumBufLocal, RoundMode::CAST_NONE, poolMem.alignC);
    } else {
        Cast(outputLocal, poolMem.sumBufLocal, RoundMode::CAST_RINT, poolMem.alignC);
    }
    poolMem.outputQueue.EnQue(outputLocal);

    CopyOut(outputPointIdx, poolMem.inC);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe) {
    InitTiling(tiling);

    poolMem.pipe = pipe;
    poolMem.inputGlobal.SetGlobalBuffer((__gm__ T*)x);
    poolMem.outputGlobal.SetGlobalBuffer((__gm__ T*)y);

    poolMem.inputBufLen = tileInput * poolMem.alignC;
    pipe->InitBuffer(poolMem.inputQueue, QUEUE_DEPTH, poolMem.inputBufLen * sizeof(float));
    pipe->InitBuffer(poolMem.outputQueue, QUEUE_DEPTH, poolMem.alignC * sizeof(T));

    pipe->InitBuffer(poolMem.sumBuf, poolMem.alignC * sizeof(float));
    poolMem.sumBufLocal = poolMem.sumBuf.template Get<float>();

    poolMem.indexBuf.Init(pipe, poolMem.indexBufLen);
    InitCommonBuffers(poolMem, workspace);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::Process() {
#if __CCE_AICORE__ < 220
    AvgPool3d::HandleAtomicAdd(poolMem);
#endif
    for (int64_t outputPointIdx = poolMem.outputPointOffset;
        outputPointIdx < poolMem.outputPointOffset + poolMem.outputPointNum; ++outputPointIdx) {
        ReduceMeanWindow(outputPointIdx);
    }
}

} // namespace AvgPool3d

#endif // AVG_POOL3D_NDHWC_SPLIT_W_H_
