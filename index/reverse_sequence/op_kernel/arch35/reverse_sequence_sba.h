/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reverse_sequence_sba.h
 * \brief
 */
#ifndef REVERSE_SEQUENCE_SBA_KERNEL_H_
#define REVERSE_SEQUENCE_SBA_KERNEL_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "reverse_sequence_struct.h"

namespace ReverseSequence {

using namespace AscendC;

static constexpr int64_t SPLIT_DIM_A1 = 4;

template <typename T, typename SeqType, typename CompType>
class ReverseSequenceSBA
{
public:
    __aicore__ inline ReverseSequenceSBA(TPipe *pipe, const ReverseSequenceA1SBATilingData *tilingData)
                        : pipe_(pipe), tilingData_(tilingData) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR seqLengths, GM_ADDR y);
    __aicore__ inline void Process();
private:
    template <int64_t SplitMode>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void ComputeSplitA1(int64_t srcOffset, int64_t inDimA);
    __aicore__ inline void ComputeSplitA(int64_t srcOffset, int64_t a1Offset, int64_t sStart, int64_t bStart, int64_t aStart, int64_t inDimA);
    __aicore__ inline void ComputeSplitS(int64_t srcOffset, int64_t a1Offset, int64_t sStart, int64_t inDimS);
    __aicore__ inline void ComputeSplitB(int64_t srcOffset, int64_t a1Offset, int64_t sStart, int64_t bStart, int64_t inDimB);
    __aicore__ inline void ReverseCompute(CompType curOffset, CompType xUbFactor);
    __aicore__ inline void CopyInMultiDim(int64_t offset, int64_t blockCount, int64_t blockLen);
    __aicore__ inline void CopyInSingleDim(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyOutSingleDim(int64_t globalOffset, LocalTensor<T> yLocal, int64_t blockLen);
    __aicore__ inline void GetCurrentSeqLength(int64_t bStart);
    TPipe *pipe_;
    const ReverseSequenceA1SBATilingData *tilingData_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<SeqType> seqGm_;
    int64_t ubBlockSize_ = platform::GetUbBlockSize();
    int64_t eleBlk_ = 1;
    int64_t bStart_ = -1;
    int64_t seqLen_ = 0;
    int64_t batchSize_ = 0;
    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 0;
};

template <typename T, typename SeqType, typename CompType>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void ReverseA1Compute(
    __local_mem__ T* xLocal, __gm__ SeqType* seqGm, __local_mem__ T* outLocal, CompType xUbFactor,
    CompType batchDim, CompType seqDim, CompType reverseSize, CompType m0, CompType m1, CompType m2,
    CompType m3, CompType shift0, CompType shift1, CompType shift2, CompType shift3);

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::Init(GM_ADDR x, GM_ADDR seqLengths, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    seqGm_.SetGlobalBuffer((__gm__ SeqType*)seqLengths);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_->InitBuffer(dataQueue_, DB_BUFFER, tilingData_->inUbSize * sizeof(T));
    if (tilingData_->splitMode == SPLIT_DIM_A1) {
        pipe_->InitBuffer(outQueue_, DB_BUFFER, tilingData_->inUbSize * sizeof(T));
    }
    eleBlk_ = ubBlockSize_ / tilingData_->dtypeSize;
    blockIdx_ = GetBlockIdx();
 	blockNum_ = GetBlockNum();
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::Process()
{
    if (blockIdx_ > blockNum_) {
 	    return;
 	}

    if (tilingData_->splitMode == SPLIT_DIM_A1) {
        BaseCompute<SPLIT_DIM_A1>();
    } else if (tilingData_->splitMode == SPLIT_DIM_S) {
        BaseCompute<SPLIT_DIM_S>();
    } else if (tilingData_->splitMode == SPLIT_DIM_B) {
        BaseCompute< SPLIT_DIM_B>();
    } else {
        BaseCompute<SPLIT_DIM_A>();
    }
}

template <typename T, typename SeqType, typename CompType>
template <int64_t SplitMode>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::BaseCompute()
{
    int64_t startIdx = 0;
    int64_t endIdx = 0;
    if (GetBlockIdx() < tilingData_->blockTail) {
        startIdx = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx = startIdx + tilingData_->blockFactor + 1;
    } else {
        startIdx = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx = startIdx + tilingData_->blockFactor;
    }
    // startIdx到 endIdx指的是 ub循环次数
    // inDimB指的是每次 ub循环  B轴的行数
    for (int64_t loopIdx = startIdx; loopIdx < endIdx; loopIdx++) {
        int64_t a1Idx = loopIdx / (tilingData_->sLoop * tilingData_->bLoop * tilingData_->aLoop);
        int64_t a1LoopNum = a1Idx * tilingData_->sLoop * tilingData_->bLoop * tilingData_->aLoop;
        int64_t sIdx = (loopIdx - a1LoopNum) / (tilingData_->bLoop * tilingData_->aLoop);
        int64_t bIdx = (loopIdx - a1LoopNum - sIdx * tilingData_->bLoop * tilingData_->aLoop) / tilingData_->aLoop;
        int64_t aIdx = loopIdx % tilingData_->aLoop;
        int64_t inDimA1 =
            a1Idx == tilingData_->a1Loop - 1 ? tilingData_->a1Dim - a1Idx * tilingData_->ubFactorA1 : tilingData_->ubFactorA1;
        int64_t inDimS =
            sIdx == tilingData_->sLoop - 1 ? tilingData_->sDim - sIdx * tilingData_->ubFactorS : tilingData_->ubFactorS;
        int64_t inDimB =
            bIdx == tilingData_->bLoop - 1 ? tilingData_->bDim - bIdx * tilingData_->ubFactorB : tilingData_->ubFactorB;
        int64_t inDimA =
            aIdx == tilingData_->aLoop - 1 ? tilingData_->aDim - aIdx * tilingData_->ubFactorA : tilingData_->ubFactorA;

        int64_t a1Start = a1Idx * tilingData_->ubFactorA1;
        int64_t sStart = sIdx * tilingData_->ubFactorS;
        int64_t bStart = bIdx * tilingData_->ubFactorB;
        int64_t aStart = aIdx * tilingData_->ubFactorA;
        int64_t a1Offset = a1Start * tilingData_->sDim * tilingData_->bDim * tilingData_->aDim;
        int64_t srcOffset = a1Offset + sStart * tilingData_->bDim * tilingData_->aDim + bStart * tilingData_->aDim + aStart;
        if constexpr (SplitMode == SPLIT_DIM_A1) {
            ComputeSplitA1(srcOffset, inDimA1);
        } else if constexpr (SplitMode == SPLIT_DIM_A) {
            ComputeSplitA(srcOffset, a1Offset, sStart, bStart, aStart, inDimA);
        } else if constexpr (SplitMode == SPLIT_DIM_B) {
            ComputeSplitB(srcOffset, a1Offset, sStart, bStart, inDimB);
        } else {
            ComputeSplitS(srcOffset, a1Offset, sStart, inDimS);
        }
    }
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::ComputeSplitA1(int64_t srcOffset, int64_t inDimA1)
{
    int64_t xUbFactor = inDimA1 * tilingData_->sDim * tilingData_->bDim * tilingData_->aDim;
    CopyInSingleDim(srcOffset, xUbFactor);
    ReverseCompute(srcOffset, xUbFactor);
    LocalTensor<T> yLocal = outQueue_.DeQue<T>();
    CopyOutSingleDim(srcOffset, yLocal, xUbFactor);
    outQueue_.FreeTensor(yLocal);
}

template <typename T, typename SeqType, typename CompType>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void ReverseA1Compute(
    __local_mem__ T* xLocal, __gm__ SeqType* seqGm, __local_mem__ T* outLocal, CompType xUbFactor,
    CompType batchDim, CompType seqDim, CompType reverseSize, CompType m0, CompType m1, CompType m2,
    CompType m3, CompType shift0, CompType shift1, CompType shift2, CompType shift3)
{
    for (CompType xOffset = Simt::GetThreadIdx(); xOffset < xUbFactor; xOffset += Simt::GetThreadNum()) {
        CompType batchPreAxis = Simt::UintDiv(xOffset, m0, shift0);
        CompType batchIdx = Simt::UintDiv(batchPreAxis, m1, shift1);
        CompType batchDimIdx = batchPreAxis - batchIdx * batchDim; // xOffSet / batchSize % batchDim

        CompType reverseNum = static_cast<CompType>(seqGm[batchDimIdx]);
        CompType seqPreAxis = Simt::UintDiv(xOffset, m2, shift2); // xOffSet / seqSize
        CompType seqIdx = Simt::UintDiv(seqPreAxis, m3, shift3);  // xOffSet / seqSize / seqDim
        CompType seqDimIdx = seqPreAxis - seqIdx * seqDim; // xOffSet / seqSize % seqDim

        CompType reverseSizeMod = xOffset - seqPreAxis * reverseSize; // xOffSet % reverseSize
        
        if (reverseNum > seqDim) {
            reverseNum = seqDim;
        }

        if (seqDimIdx < reverseNum) {
            CompType reverseOffset = seqIdx * seqDim * reverseSize + (reverseNum - 1 - seqDimIdx) * reverseSize + reverseSizeMod; // 
            outLocal[reverseOffset] = xLocal[xOffset];
        }

        if (seqDimIdx >= reverseNum) {
            outLocal[xOffset] = xLocal[xOffset];
        }
    }
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::ReverseCompute(CompType curOffset, CompType xUbFactor)
{
    auto vWaitMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(vWaitMTEEventID);
    WaitFlag<HardEvent::MTE2_V>(vWaitMTEEventID);
    LocalTensor<T> xLocal = dataQueue_.DeQue<T>();
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
    ReverseSequenceQuickDivParam<CompType> params;
    GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<CompType>(tilingData_->batchSize));
    GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<CompType>(tilingData_->bDim));
    GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<CompType>(tilingData_->reverseSize));
    GetUintDivMagicAndShift(params.m3, params.shift3, static_cast<CompType>(tilingData_->sDim));

    Simt::VF_CALL<ReverseA1Compute<T, SeqType, CompType>>(
        Simt::Dim3(USED_THREAD), (__local_mem__ T*)(xLocal.GetPhyAddr()), (__gm__ SeqType*)(seqGm_.GetPhyAddr()),
        (__local_mem__ T*)(outLocal.GetPhyAddr()), xUbFactor, tilingData_->bDim, tilingData_->sDim,
        tilingData_->reverseSize, params.m0, params.m1, params.m2, params.m3, params.shift0,
        params.shift1, params.shift2, params.shift3);

    outQueue_.EnQue(outLocal);
    dataQueue_.FreeTensor(xLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::CopyInMultiDim(int64_t offset, int64_t blockCount, int64_t blockLen)
{
    LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
    int64_t alignBlockLen = ops::Aligned(blockLen, eleBlk_);
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = blockCount;
    copyExtParams.blockLen = blockLen * tilingData_->dtypeSize;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = (alignBlockLen - blockLen) * tilingData_->dtypeSize / platform::GetUbBlockSize();
    DataCopyPad(xLocal, xGm_[offset], copyExtParams, padExtParams);
    dataQueue_.EnQue<T>(xLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::CopyInSingleDim(int64_t offset, int64_t blockLen)
{
    auto mte2WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(mte2WaitVEventID);
    WaitFlag<HardEvent::V_MTE2>(mte2WaitVEventID);
    LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = 1;
    copyExtParams.blockLen = blockLen * tilingData_->dtypeSize;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[offset], copyExtParams, padParams);
    dataQueue_.EnQue<T>(xLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::CopyOutSingleDim(int64_t globalOffset, LocalTensor<T> yLocal, int64_t blockLen)
{
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = 1;
    copyExtParams.blockLen = blockLen * tilingData_->dtypeSize;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(yGm_[globalOffset], yLocal, copyExtParams);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::GetCurrentSeqLength(int64_t bStart)
{
   if (bStart_ != bStart) {
        bStart_ = bStart;
        seqLen_ = static_cast<int64_t>(seqGm_.GetValue(bStart_));
    } else {
        return;
    }
    if (seqLen_ <= 1) {
        seqLen_ = 0;
    }
    if (seqLen_ > tilingData_->sDim) {
        seqLen_ = tilingData_->sDim;
    }
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::ComputeSplitA(
    int64_t srcOffset, int64_t a1Offset, int64_t sStart, int64_t bStart, int64_t aStart, int64_t inDimA)
{
    CopyInSingleDim(srcOffset, inDimA);
    GetCurrentSeqLength(bStart);
    int64_t outOffset = a1Offset + bStart * tilingData_->aDim + aStart;
    if (sStart < seqLen_) {
        outOffset += (seqLen_ - sStart - 1) * tilingData_->bDim * tilingData_->aDim;
    } else {
        outOffset = srcOffset;
    }
    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
    CopyOutSingleDim(outOffset, yLocal, inDimA);
    dataQueue_.FreeTensor(yLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::ComputeSplitB(
    int64_t srcOffset, int64_t a1Offset, int64_t sStart, int64_t bStart, int64_t inDimB)
{
    CopyInMultiDim(srcOffset, inDimB, tilingData_->aDim);
    int64_t alignBlockLen = ops::Aligned(tilingData_->aDim, eleBlk_);
    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
    int64_t sOffset = sStart * tilingData_->bDim * tilingData_->aDim;
    for (int64_t bIndex = 0; bIndex < inDimB; bIndex++) {
        GetCurrentSeqLength(bIndex + bStart);
        int64_t outOffset = a1Offset + (bIndex + bStart) * tilingData_->aDim;
        if (sStart < seqLen_) {
            outOffset += (seqLen_ - sStart - 1) * tilingData_->bDim * tilingData_->aDim;
        } else {
            outOffset += sOffset;
        }
        CopyOutSingleDim(outOffset, yLocal[bIndex * alignBlockLen], tilingData_->aDim);
    }
    dataQueue_.FreeTensor(yLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSBA<T, SeqType, CompType>::ComputeSplitS(int64_t srcOffset, int64_t a1Offset, int64_t sStart, int64_t inDimS)
{
    CopyInMultiDim(srcOffset, inDimS * tilingData_->bDim, tilingData_->aDim);
    int64_t alignBlockLen = ops::Aligned(tilingData_->aDim, eleBlk_);
    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
    for (int64_t sIndex = 0; sIndex < inDimS; sIndex++) {
        int64_t sIdx = sIndex + sStart;
        int64_t sOffset = sIndex * tilingData_->bDim;
        for (int64_t bIndex = 0; bIndex < tilingData_->bDim; bIndex++) {
            GetCurrentSeqLength(bIndex);
            int64_t outOffset = a1Offset + bIndex * tilingData_->aDim;
            if (sIdx < seqLen_) {
                outOffset += (seqLen_ - sIdx - 1) * tilingData_->bDim * tilingData_->aDim;
            } else {
                outOffset += sIdx * tilingData_->bDim * tilingData_->aDim;
            }
            CopyOutSingleDim(outOffset, yLocal[(sOffset + bIndex) * alignBlockLen], tilingData_->aDim);
        }
    }
    dataQueue_.FreeTensor(yLocal);
}
}

#endif