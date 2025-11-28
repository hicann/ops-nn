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

    TPipe* pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inputQueue;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueue;

    TBuf<QuePosition::VECCALC> tmpPattern;
    TBuf<TPosition::VECCALC> sumBuf;
    LocalTensor<float> sumBufLocal;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;

    int64_t hwLength;
    int64_t tileHW;
    int64_t ncdBlockLength;
    int64_t ncdOffset;
    int64_t nextCoreAddrOffset;
    int64_t atomicAddNum;
    int64_t hwTailLength;
    int64_t hwTailAlign;

    PoolShape inputShape;
    PoolShape outputShape;

    int64_t indexBufLen;
    IndexBuffer indexBuf;
    PoolParameter poolParam;

    uint32_t numPerBlock;
    int32_t validTailLen;

    TQue<QuePosition::VECIN, QUEUE_DEPTH> syncWorkQueue;
    GlobalTensor<int32_t> syncTensorsGM;
    TBuf<TPosition::VECCALC> clearTensorBuff;
    uint32_t usedCoreNum;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::InitTiling(const AvgPool3DTilingData* __restrict__ tiling) {
    inputShape = PoolShape(tiling->inN, tiling->inC, tiling->inD, tiling->inH, tiling->inW);
    outputShape = PoolShape(tiling->inN, tiling->inC, tiling->outD, tiling->outH, tiling->outW);

    poolParam = PoolParameter(tiling->kD, tiling->kH, tiling->kW, tiling->dD, tiling->dH, tiling->dW,
                              tiling->pD, tiling->pH, tiling->pW, tiling->divisorOverride, tiling->countIncludePad);

    numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);

    hwLength = tiling->inH * tiling->inW;
    tileHW = tiling->tileHW;

    ncdBlockLength = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
    ncdOffset = GetBlockIdx() < tiling->formerNum
      ? tiling->formerLength * GetBlockIdx()
      : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);
    nextCoreAddrOffset = (ncdOffset + ncdBlockLength) * hwLength;
    atomicAddNum = tiling->atomicAddNum;
    hwTailLength = hwLength % tileHW;
    hwTailAlign = AlignUp(hwTailLength, numPerBlock);
    validTailLen = hwTailLength % numPerBlock;
    usedCoreNum = tiling->usedCoreNum;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::CopyIn(int64_t offset, int64_t len) {
    LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
#if __CCE_AICORE__ < 220
    if constexpr (std::is_same_v<T, float>) {
        if (len == tileHW) {
            DataCopyParams copyParams{1, static_cast<uint16_t>(len / numPerBlock), 0, 0};
            DataCopy(inputLocal, inputGlobal[offset], copyParams);
        } else {
            DataCopy(inputLocal, inputGlobal[offset], hwTailAlign);
        }
    } else {
        if (len == tileHW) {
            DataCopyParams copyParams{1, static_cast<uint16_t>(len / numPerBlock), 0, 0};
            DataCopy(inputLocal[tileHW], inputGlobal[offset], copyParams);
        } else {
            DataCopy(inputLocal[tileHW], inputGlobal[offset], hwTailAlign);
        }
    }
#else
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(inputLocal, inputGlobal[offset], copyParams, padParams);
    } else {
        DataCopyPad(inputLocal[tileHW], inputGlobal[offset], copyParams, padParams);
    }
#endif
    inputQueue.EnQue(inputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::DataCopyOutNonPad(
    LocalTensor<T>& outputLocal, int64_t offset, int64_t validDataLen) {
    if ((validDataLen < numPerBlock) && (offset + validDataLen * atomicAddNum >= nextCoreAddrOffset)) {
        uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
        uint64_t mask[2] = {mask0, 0};
        Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
        VToMTE3Sync();
        SetAtomicAdd<T>();
        DataCopy(outputGlobal[offset], outputLocal, hwTailAlign);
        SetAtomicNone();
        AscendC::PipeBarrier<PIPE_MTE3>();
    } else if ((validTailLen != 0) && (offset + validDataLen == nextCoreAddrOffset)) {
        DataCopy(outputGlobal[offset], outputLocal, hwTailAlign - numPerBlock);
        int32_t lastLeftShift = validTailLen;
        uint32_t mask = numPerBlock * 2;
        uint64_t rsvdCnt = 0;
        uint64_t gatherOffset = hwTailAlign - mask;
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
        DataCopy(outputGlobal[nextCoreAddrOffset - numPerBlock], outputLocal[gatherOffset], numPerBlock);
    } else {
        DataCopy(outputGlobal[offset], outputLocal, hwTailAlign);
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::CopyOut(int64_t offset, int64_t len) {
    LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
    if (len == tileHW) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(len / numPerBlock), 0, 0};
        DataCopy(outputGlobal[offset], outputLocal, copyParams);
    } else {
        DataCopyOutNonPad(outputLocal, offset, len);
    }
#else
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGlobal[offset], outputLocal, copyParams);
#endif
    outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::ReduceSumDWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t startOffset, int64_t len) {
    int64_t dstart = index.D.start;
    int64_t dend = index.D.end;

    for (int64_t id = dstart; id < dend; ++id) {
        int64_t dOffset = id * inputShape.strideD;

        CopyIn(startOffset + dOffset, len);

        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        if constexpr (std::is_same_v<T, float>) {
            Add(sumBufLocal, sumBufLocal, inputLocal, len);
        } else {
            Cast(inputLocal.template ReinterpretCast<float>(), inputLocal[tileHW], RoundMode::CAST_NONE, len);
            Add(sumBufLocal, sumBufLocal, inputLocal.template ReinterpretCast<float>(), len);
        }
        inputQueue.FreeTensor(inputLocal);
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::ReduceMeanDWindow(int64_t dIdx) {
    Index index;

    uint64_t ncIdx = dIdx / outputShape.D;
    uint64_t outputDIdx = dIdx % outputShape.D;
    index.D.Compute(outputDIdx, inputShape.D, poolParam.kernelD, poolParam.strideD, poolParam.padD,
                    poolParam.countIncludePad);

    int64_t poolSize = poolParam.divisorOverride ? poolParam.divisorOverride : index.D.poolSize;
    float factor = 1.0f / static_cast<float>(poolSize);

    SToVSync();

    int64_t hwLoop = (hwLength + tileHW - 1) / tileHW;
    int64_t hwOffset = 0;
    for (int64_t i = 0; i < hwLoop; ++i) {
        int64_t count = i < hwLoop - 1 ? tileHW : hwLength - (hwLoop - 1) * tileHW;

        Duplicate(sumBufLocal, 0.0f, count);

        int64_t startOffset = ncIdx * inputShape.strideC + hwOffset;

        ReduceSumDWindow(index, sumBufLocal, startOffset, count);
        Muls(sumBufLocal, sumBufLocal, factor, count);

        LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
#if __CCE_AICORE__ < 220
            Adds(outputLocal, sumBufLocal, 0.0f, AlignUp(count, numPerBlock));
#else
            DataCopy(outputLocal, sumBufLocal, AlignUp(count, numPerBlock));
#endif
        } else if constexpr (std::is_same_v<T, half>) {
            Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, count);
        } else {
            Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, count);
        }
        outputQueue.EnQue(outputLocal);

        CopyOut(ncIdx * outputShape.strideC + outputDIdx * hwLength + hwOffset, count);

        hwOffset += count;
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe) {
    InitTiling(tiling);

    inputGlobal.SetGlobalBuffer((__gm__ T*)x);
    outputGlobal.SetGlobalBuffer((__gm__ T*)y);

    pipe->InitBuffer(inputQueue, QUEUE_DEPTH, tileHW * sizeof(float));
    pipe->InitBuffer(outputQueue, QUEUE_DEPTH, tileHW * sizeof(T));

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

    pipe->InitBuffer(sumBuf, tileHW * sizeof(float));
    sumBufLocal = sumBuf.Get<float>();
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dReduceD<T, QUEUE_DEPTH>::Process() {
#if __CCE_AICORE__ < 220
    if (atomicAddNum) {
        LocalTensor<T> clearUb = clearTensorBuff.Get<T>();
        Duplicate(clearUb, (T)0, DEFAULT_CLEAR_UB_SIZE);

        VToMTE3Sync();
        int64_t curOutputPointIdx = ncdBlockLength + ncdOffset - 1;
        for (int i = 0; i < atomicAddNum; i++, curOutputPointIdx--) {
            DataCopy<T>(outputGlobal[curOutputPointIdx * hwTailLength], clearUb, numPerBlock);
        }

        DataCopy(syncTensorsGM[0], clearUb.template ReinterpretCast<int32_t>(), usedCoreNum * 8 * 32);
        LocalTensor<int32_t> syncLocalTensor = syncWorkQueue.template AllocTensor<int32_t>();
        AscendC::SyncAll(syncTensorsGM, syncLocalTensor, int32_t(usedCoreNum));
        syncWorkQueue.FreeTensor(syncLocalTensor);
    }
#endif

    for (int64_t dIdx = ncdOffset; dIdx < ncdOffset + ncdBlockLength; ++dIdx) {
        ReduceMeanDWindow(dIdx);
    }
}

} // namespace AvgPool3d

#endif // AVG_POOL3D_NCDHW_REDUCE_D_H_
