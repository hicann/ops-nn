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
 * \file avg_pool3_d_ncdhw_reduce_d.h
 * \brief
 */

#ifndef AVG_POOL3D_NCDHW_REDUCE_D_H_
#define AVG_POOL3D_NCDHW_REDUCE_D_H_

#include "kernel_operator.h"
#include "avg_pool3d_common.h"

namespace AvgPool3d {
template <typename T, int32_t QUEUE_DEPTH>
class KernelAvgPool3dReduceD {
public:
    __aicore__ inline KernelAvgPool3dReduceD() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTiling(const AvgPool3DTilingData* __restrict__ tiling);
    __aicore__ inline void CopyIn(int64_t offset, int64_t len);
    __aicore__ inline void CopyOut(int64_t offset, int64_t len);
    __aicore__ inline void DataCopyOutNonPad(LocalTensor<T>& outputLocal, int64_t offset, int64_t validDataLen);
    __aicore__ inline void ReduceMeanDWindow(int64_t dIdx);
    __aicore__ inline void ReduceSumDWindow(
      const Index& index, LocalTensor<float>& sumBufLocal, int64_t startOffset, int64_t len);

    int64_t hwLength;
    int64_t tileHW;
    int64_t ncdBlockLength;
    int64_t ncdOffset;
    int64_t hwTailLength;
    int64_t hwTailAlign;

    PoolMem<T, QUEUE_DEPTH> poolMem;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::InitTiling(const AvgPool3DTilingData* __restrict__ tiling) {
    poolMem.inputShape = PoolShape(tiling->inN, tiling->inC, tiling->inD, tiling->inH, tiling->inW);
    poolMem.outputShape = PoolShape(tiling->inN, tiling->inC, tiling->outD, tiling->outH, tiling->outW);

    poolMem.poolParam = PoolParameter(tiling->kD, tiling->kH, tiling->kW, tiling->dD, tiling->dH, tiling->dW,
                              tiling->pD, tiling->pH, tiling->pW, tiling->divisorOverride, tiling->countIncludePad);

    poolMem.numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);

    hwLength = tiling->inH * tiling->inW;
    tileHW = tiling->tileHW;

    ncdBlockLength = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
    ncdOffset = GetBlockIdx() < tiling->formerNum
      ? tiling->formerLength * GetBlockIdx()
      : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);
    poolMem.nextCoreAddrOffset = (ncdOffset + ncdBlockLength) * hwLength;
    poolMem.atomicAddNum = tiling->atomicAddNum;
    hwTailLength = hwLength % tileHW;
    hwTailAlign = AlignUp(hwTailLength, poolMem.numPerBlock);
    poolMem.validTailLen = hwTailLength % poolMem.numPerBlock;
    poolMem.usedCoreNum = tiling->usedCoreNum;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::CopyIn(int64_t offset, int64_t len) {
    LocalTensor<T> inputLocal = poolMem.inputQueue.template AllocTensor<T>();
#if __CCE_AICORE__ < 220
    if constexpr (std::is_same_v<T, float>) {
        if (len == tileHW) {
            DataCopyParams copyParams{1, static_cast<uint16_t>(len / poolMem.numPerBlock), 0, 0};
            DataCopy(inputLocal, poolMem.inputGlobal[offset], copyParams);
        } else {
            DataCopy(inputLocal, poolMem.inputGlobal[offset], hwTailAlign);
        }
    } else {
        if (len == tileHW) {
            DataCopyParams copyParams{1, static_cast<uint16_t>(len / poolMem.numPerBlock), 0, 0};
            DataCopy(inputLocal[tileHW], poolMem.inputGlobal[offset], copyParams);
        } else {
            DataCopy(inputLocal[tileHW], poolMem.inputGlobal[offset], hwTailAlign);
        }
    }
#else
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(inputLocal, poolMem.inputGlobal[offset], copyParams, padParams);
    } else {
        DataCopyPad(inputLocal[tileHW], poolMem.inputGlobal[offset], copyParams, padParams);
    }
#endif
    poolMem.inputQueue.EnQue(inputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::DataCopyOutNonPad(
    LocalTensor<T>& outputLocal, int64_t offset, int64_t validDataLen) {
    if ((validDataLen < poolMem.numPerBlock) && (offset + validDataLen * poolMem.atomicAddNum >= poolMem.nextCoreAddrOffset)) {
        uint64_t mask0 = (1ul << poolMem.numPerBlock) - (1ul << validDataLen);
        uint64_t mask[2] = {mask0, 0};
        Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
        VToMTE3Sync();
        SetAtomicAdd<T>();
        DataCopy(poolMem.outputGlobal[offset], outputLocal, hwTailAlign);
        SetAtomicNone();
        AscendC::PipeBarrier<PIPE_MTE3>();
    } else if ((poolMem.validTailLen != 0) && (offset + validDataLen == poolMem.nextCoreAddrOffset)) {
        DataCopy(poolMem.outputGlobal[offset], outputLocal, hwTailAlign - poolMem.numPerBlock);
        uint32_t mask = poolMem.numPerBlock * 2;
        uint64_t gatherOffset = hwTailAlign - mask;
        HandleTailMask(outputLocal, gatherOffset, poolMem, mask);
        DataCopy(poolMem.outputGlobal[poolMem.nextCoreAddrOffset - poolMem.numPerBlock], outputLocal[gatherOffset], poolMem.numPerBlock);
    } else {
        DataCopy(poolMem.outputGlobal[offset], outputLocal, hwTailAlign);
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::CopyOut(int64_t offset, int64_t len) {
    LocalTensor<T> outputLocal = poolMem.outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
    if (len == tileHW) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(len / poolMem.numPerBlock), 0, 0};
        DataCopy(poolMem.outputGlobal[offset], outputLocal, copyParams);
    } else {
        DataCopyOutNonPad(outputLocal, offset, len);
    }
#else
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(T)), static_cast<uint32_t>(0), 
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(poolMem.outputGlobal[offset], outputLocal, copyParams);
#endif
    poolMem.outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::ReduceSumDWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t startOffset, int64_t len) {
    int64_t dstart = index.D.start;
    int64_t dend = index.D.end;

    for (int64_t id = dstart; id < dend; ++id) {
        int64_t dOffset = id * poolMem.inputShape.strideD;

        CopyIn(startOffset + dOffset, len);

        LocalTensor<T> inputLocal = poolMem.inputQueue.template DeQue<T>();
        if constexpr (std::is_same_v<T, float>) {
            Add(sumBufLocal, sumBufLocal, inputLocal, len);
        } else {
            Cast(inputLocal.template ReinterpretCast<float>(), inputLocal[tileHW], RoundMode::CAST_NONE, len);
            Add(sumBufLocal, sumBufLocal, inputLocal.template ReinterpretCast<float>(), len);
        }
        poolMem.inputQueue.FreeTensor(inputLocal);
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::ReduceMeanDWindow(int64_t dIdx) {
    Index index;

    uint64_t ncIdx = dIdx / poolMem.outputShape.D;
    uint64_t outputDIdx = dIdx % poolMem.outputShape.D;
    index.D.Compute(outputDIdx, poolMem.inputShape.D, poolMem.poolParam.kernelD, poolMem.poolParam.strideD, poolMem.poolParam.padD,
                    poolMem.poolParam.countIncludePad);

    int64_t poolSize = poolMem.poolParam.divisorOverride ? poolMem.poolParam.divisorOverride : index.D.poolSize;
    float factor = 1.0f / static_cast<float>(poolSize);

    SToVSync();

    int64_t hwLoop = (hwLength + tileHW - 1) / tileHW;
    int64_t hwOffset = 0;
    for (int64_t i = 0; i < hwLoop; ++i) {
        int64_t count = i < hwLoop - 1 ? tileHW : hwLength - (hwLoop - 1) * tileHW;

        Duplicate(poolMem.sumBufLocal, 0.0f, count);

        int64_t startOffset = ncIdx * poolMem.inputShape.strideC + hwOffset;

        ReduceSumDWindow(index, poolMem.sumBufLocal, startOffset, count);
        CastAndEnqueueOutput<T, QUEUE_DEPTH>(poolMem, count, factor);

        CopyOut(ncIdx * poolMem.outputShape.strideC + outputDIdx * hwLength + hwOffset, count);

        hwOffset += count;
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe) {
    InitTiling(tiling);

    poolMem.pipe = pipe;
    poolMem.inputGlobal.SetGlobalBuffer((__gm__ T*)x);
    poolMem.outputGlobal.SetGlobalBuffer((__gm__ T*)y);

    pipe->InitBuffer(poolMem.inputQueue, QUEUE_DEPTH, tileHW * sizeof(float));
    pipe->InitBuffer(poolMem.outputQueue, QUEUE_DEPTH, tileHW * sizeof(T));

    InitCommonBuffers(poolMem, workspace);

    pipe->InitBuffer(poolMem.sumBuf, tileHW * sizeof(float));
    poolMem.sumBufLocal = poolMem.sumBuf.template Get<float>();
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::Process() {
#if __CCE_AICORE__ < 220
    int64_t curOutputPointIdx = ncdBlockLength + ncdOffset - 1;
    AvgPool3d::HandleAtomicAddWithTail(poolMem, curOutputPointIdx, hwTailLength);
#endif

    for (int64_t dIdx = ncdOffset; dIdx < ncdOffset + ncdBlockLength; ++dIdx) {
        ReduceMeanDWindow(dIdx);
    }
}

} // namespace AvgPool3d

#endif // AVG_POOL3D_NCDHW_REDUCE_D_H_
