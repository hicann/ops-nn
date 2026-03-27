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
 * \file reverse_sequence_bs.h
 * \brief
 */
#ifndef REVERSE_SEQUENCE_BS_KERNEL_H_
#define REVERSE_SEQUENCE_BS_KERNEL_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "reverse_sequence_struct.h"

namespace ReverseSequence {
using namespace AscendC;

static constexpr int64_t DBUFFER = 2;
static constexpr int64_t SPLIT_MODE_A = 1;
static constexpr int64_t SPLIT_MODE_B = 2;
static constexpr int64_t SPLIT_MODE_S = 3;
static constexpr uint32_t USED_THREADS = 512;

template <typename T, typename SeqType, typename CompType>
class ReverseSequenceBS
{
public:
    __aicore__ inline ReverseSequenceBS(TPipe *pipe, const ReverseSequenceBSTilingData *tilingData)
                        : pipe_(pipe), tilingData_(tilingData) {};
     __aicore__ inline void Init(GM_ADDR x, GM_ADDR seqLengths, GM_ADDR y);
     __aicore__ inline void Process();
private:
    template <int64_t SplitMode>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void ComputeSplitS(int64_t srcOffset, int64_t aStart, int64_t bStart, int64_t sStart, int64_t dimSNum);
    __aicore__ inline void ComputeSplitB(int64_t srcOffset, int64_t bStart, int64_t dimNum);
    __aicore__ inline void ComputeSplitA(int64_t srcOffset, int64_t dimNum);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyOut(int64_t offset, int64_t blockLen);
    __aicore__ inline void GetCurrentSeqLength(int64_t bStart);
    __aicore__ inline void ReverseSCompute(int64_t seqLen, int64_t sStart, int64_t dimNum);
    __aicore__ inline void ReverseBSCompute(int64_t bStart, int64_t dimNum, int64_t dimSNum);
    __aicore__ inline void ReverseABSCompute(int64_t dimNum, int64_t dimBSNum, int64_t dimSNum);
    __aicore__ inline void SplitSSingleCopyOut(int64_t srcOffset, int64_t dimSNums);
    TPipe *pipe_;
    const ReverseSequenceBSTilingData *tilingData_;    
    TQue<QuePosition::VECIN, DBUFFER> inputQue_;
    // 输出ub
    TQue<QuePosition::VECOUT, DBUFFER> outQue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<SeqType> seqGm_;
    int64_t ubBlockSize_ = platform::GetUbBlockSize();
    int64_t eleBlk_ = 1;
    int64_t bStart_ = -1;
    int64_t seqLen_ = 0;
};

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::Init(GM_ADDR x, GM_ADDR seqLengths, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    seqGm_.SetGlobalBuffer((__gm__ SeqType*)seqLengths);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, DBUFFER, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(outQue_, DBUFFER, tilingData_->inUbSize * sizeof(T));
    eleBlk_ = ubBlockSize_ / tilingData_->dtypeSize;
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }

    if (tilingData_->splitMode == SPLIT_MODE_S) {
        BaseCompute<SPLIT_MODE_S>();
    } else if (tilingData_->splitMode == SPLIT_MODE_B) {
        BaseCompute<SPLIT_MODE_B>();
    } else {
        BaseCompute<SPLIT_MODE_A>();
    }
}

template <typename T, typename SeqType, typename CompType>
template <int64_t SplitMode>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::BaseCompute()
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
    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        int64_t aIdx = idx / (tilingData_->sLoop * tilingData_->bLoop);
        int64_t bIdx = (idx - aIdx * tilingData_->sLoop * tilingData_->bLoop) / tilingData_->sLoop;
        int64_t sIdx = idx % tilingData_->sLoop;
        int64_t inDimA =
            aIdx == tilingData_->aLoop - 1 ? tilingData_->aDim - aIdx * tilingData_->ubFactorA : tilingData_->ubFactorA;
        int64_t inDimB =
            bIdx == tilingData_->bLoop - 1 ? tilingData_->bDim - bIdx * tilingData_->ubFactorB : tilingData_->ubFactorB;
        int64_t inDimS =
            sIdx == tilingData_->sLoop - 1 ? tilingData_->sDim - sIdx * tilingData_->ubFactorS : tilingData_->ubFactorS;
        
        int64_t aStart = aIdx * tilingData_->ubFactorA;
        int64_t bStart = bIdx * tilingData_->ubFactorB;
        int64_t sStart = sIdx * tilingData_->ubFactorS;
        int64_t srcOffset = aStart * tilingData_->bDim * tilingData_->sDim + bStart * tilingData_->sDim + sStart;
        if constexpr (SplitMode == SPLIT_MODE_S) {
            ComputeSplitS(srcOffset, aStart, bStart, sStart, inDimS);
        } else if constexpr (SplitMode == SPLIT_MODE_B) {
            ComputeSplitB(srcOffset, bStart, inDimB * inDimS);
        } else {
            ComputeSplitA(srcOffset, inDimA * inDimB * inDimS);
        }
    }
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::CopyIn(int64_t offset, int64_t blockLen)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = 1;
    copyExtParams.blockLen = blockLen * tilingData_->dtypeSize;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[offset], copyExtParams, padParams);
    inputQue_.EnQue<T>(xLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::CopyOut(int64_t offset, int64_t blockLen)
{
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = 1;
    copyExtParams.blockLen = blockLen * tilingData_->dtypeSize;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    LocalTensor<T> yLocal = outQue_.DeQue<T>();
    DataCopyPad(yGm_[offset], yLocal, copyExtParams);
    outQue_.FreeTensor(yLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::SplitSSingleCopyOut(int64_t srcOffset, int64_t dimSNums)
{
    LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    __local_mem__ int8_t* xLocalAddr = (__local_mem__ int8_t*)xLocal.GetPhyAddr();
    __local_mem__ int8_t* yLocalAddr = (__local_mem__ int8_t*)yLocal.GetPhyAddr();
    uint16_t repeatNum = platform::GetVRegSize();
    uint32_t totalNum = dimSNums * tilingData_->dtypeSize;
    uint16_t loop = (totalNum + repeatNum - 1) / repeatNum;
    __VEC_SCOPE__
    {
        uint32_t maskNums = totalNum;
        MicroAPI::RegTensor<int8_t> v0;
        AscendC::MicroAPI::MaskReg preg;
        for (uint16_t i = 0; i < loop; i++) {
            preg = AscendC::MicroAPI::UpdateMask<int8_t>(maskNums);
            MicroAPI::DataCopy(v0, xLocalAddr + i * repeatNum);
            MicroAPI::DataCopy(yLocalAddr + i * repeatNum, v0, preg);
        }
    }
    inputQue_.FreeTensor(xLocal);
    outQue_.EnQue(yLocal);
    CopyOut(srcOffset, dimSNums);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::GetCurrentSeqLength(int64_t bStart)
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
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREADS) inline void ReverseSSimtCompute(
    __gm__ T* xGm, __gm__ SeqType* seqGm, __local_mem__ T* xLocal, __local_mem__ T* outLocal, CompType sStart, CompType seqLen, CompType dimNum)
{
    for (CompType i = Simt::GetThreadIdx(); i < dimNum; i += Simt::GetThreadNum()) {
        CompType seqIdx = i + sStart; // i % dimSNum

        if (seqIdx < seqLen) {
            CompType inLocalOffset = dimNum - i - 1;
            outLocal[i] = xLocal[inLocalOffset];
        }
    }
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::ReverseSCompute(int64_t seqLen, int64_t sStart, int64_t dimNum)
{
    LocalTensor<T> outLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();

    Simt::VF_CALL<ReverseSSimtCompute<T, SeqType, CompType>>(
        Simt::Dim3(USED_THREADS), (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ SeqType*)(seqGm_.GetPhyAddr()), (__local_mem__ T*)(xLocal.GetPhyAddr()),
        (__local_mem__ T*)(outLocal.GetPhyAddr()), static_cast<CompType>(sStart), static_cast<CompType>(seqLen), static_cast<CompType>(dimNum));
    
    outQue_.EnQue(outLocal);
    inputQue_.FreeTensor(xLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::ComputeSplitS(
    int64_t srcOffset, int64_t aStart, int64_t bStart, int64_t sStart, int64_t dimSNum)
{
    GetCurrentSeqLength(bStart);
    if (sStart < seqLen_) {
        int64_t outOffset = aStart * tilingData_->bDim * tilingData_->sDim + bStart * tilingData_->sDim;
        if (sStart + dimSNum < seqLen_) {
            CopyIn(srcOffset, dimSNum);
            ReverseSCompute(seqLen_, sStart, dimSNum);
            auto mte3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
            WaitFlag<HardEvent::V_MTE3>(mte3WaitVEventID);

            int64_t offset = outOffset + seqLen_ - sStart - dimSNum;
            CopyOut(offset, dimSNum);
        } else {
            int64_t reverseDims = seqLen_ - sStart;
            CopyIn(srcOffset, reverseDims);
            ReverseSCompute(seqLen_, sStart, reverseDims);
            auto mte3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
            WaitFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
            int64_t offset = outOffset + seqLen_ - sStart - reverseDims;
            CopyOut(offset, reverseDims);

            offset = srcOffset + reverseDims;
            int64_t remainNums = dimSNum - reverseDims;
            CopyIn(offset, remainNums);
            SplitSSingleCopyOut(offset, remainNums); 
        }
    } else {
        CopyIn(srcOffset, dimSNum);
        SplitSSingleCopyOut(srcOffset, dimSNum); //copyIn->copyOut
    }
}

template <typename T, typename SeqType, typename CompType>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREADS) inline void ReverseBSSimtCompute(__gm__ T* xGm, __gm__ SeqType* seqGm, __local_mem__ T* xLocal, 
    __local_mem__ T* outLocal, CompType bStart, CompType dimNum, CompType dimSNum, CompType m0, CompType shift0)
{
    for (CompType i = Simt::GetThreadIdx(); i < dimNum; i += Simt::GetThreadNum()) {
        CompType curBIdx = bStart;
        CompType bOffset = Simt::UintDiv(i, m0, shift0); // i / dimSNum
        curBIdx += bOffset;
        CompType seqLen = static_cast<CompType>(seqGm[curBIdx]);
        CompType seqIdx = i - bOffset * dimSNum; // i % dimSNum

        if (seqLen > dimSNum) {
            seqLen = dimSNum;
        }

        if (seqIdx < seqLen) {
            CompType inLocalOffset = bOffset * dimSNum + seqLen - seqIdx - 1;
            outLocal[i] = xLocal[inLocalOffset];
        }

        if (seqIdx >= seqLen) {
            outLocal[i] = xLocal[i];
        }
    }
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::ReverseBSCompute(int64_t bStart, int64_t dimNum, int64_t dimSNum)
{
    LocalTensor<T> outLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();

    CompType m0 = 0;
    CompType shift0 = 0;
    GetUintDivMagicAndShift(m0, shift0, static_cast<CompType>(dimSNum));

    Simt::VF_CALL<ReverseBSSimtCompute<T, SeqType, CompType>>(
        Simt::Dim3(USED_THREADS), (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ SeqType*)(seqGm_.GetPhyAddr()), (__local_mem__ T*)(xLocal.GetPhyAddr()),
        (__local_mem__ T*)(outLocal.GetPhyAddr()), static_cast<CompType>(bStart), static_cast<CompType>(dimNum), 
        static_cast<CompType>(dimSNum), m0, shift0);
    
    outQue_.EnQue(outLocal);
    inputQue_.FreeTensor(xLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::ComputeSplitB(
    int64_t srcOffset, int64_t bStart, int64_t dimNum)
{
    CopyIn(srcOffset, dimNum);
    ReverseBSCompute(bStart, dimNum, tilingData_->sDim);

    auto mte3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
    WaitFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
    CopyOut(srcOffset, dimNum);
}

template <typename T, typename SeqType, typename CompType>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREADS) inline void ReverseABSSimtCompute(
    __gm__ T* xGm, __gm__ SeqType* seqGm, __local_mem__ T* xLocal, __local_mem__ T* outLocal, CompType dimNum, CompType dimBSNum, 
    CompType dimSNum, CompType m0, CompType m1, CompType shift0, CompType shift1)
{
    for (CompType i = Simt::GetThreadIdx(); i < dimNum; i += Simt::GetThreadNum()) {
        CompType curAIdx = Simt::UintDiv(i, m0, shift0); // i / dimBSNum
        CompType curBS = i - curAIdx * dimBSNum;  // // i % dimBSNum
        CompType curBIdx = Simt::UintDiv(curBS, m1, shift1); // i % dimBSNum / dimSNum
        CompType seqLen = static_cast<CompType>(seqGm[curBIdx]);
        CompType seqIdx = curBS - curBIdx * dimSNum; // i % dimBSNum % dimSNum

        if (seqLen > dimSNum) {
            seqLen = dimSNum;
        }

        if (seqIdx < seqLen) {
            CompType inLocalOffset = curAIdx * dimBSNum + curBIdx * dimSNum + seqLen - seqIdx - 1;
            outLocal[i] = xLocal[inLocalOffset];
        }

        if (seqIdx >= seqLen) {
            outLocal[i] = xLocal[i];
        }
    }
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::ReverseABSCompute(int64_t dimNum, int64_t dimBSNum, int64_t dimSNum)
{
    LocalTensor<T> outLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();

    CompType m0 = 0;
    CompType m1 = 0;
    CompType shift0 = 0;
    CompType shift1 = 0;
    GetUintDivMagicAndShift(m0, shift0, static_cast<CompType>(dimBSNum));
    GetUintDivMagicAndShift(m1, shift1, static_cast<CompType>(dimSNum));

    Simt::VF_CALL<ReverseABSSimtCompute<T, SeqType, CompType>>(
        Simt::Dim3(USED_THREADS), (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ SeqType*)(seqGm_.GetPhyAddr()), (__local_mem__ T*)(xLocal.GetPhyAddr()),
        (__local_mem__ T*)(outLocal.GetPhyAddr()), static_cast<CompType>(dimNum), static_cast<CompType>(dimBSNum),
        static_cast<CompType>(dimSNum), m0,  m1, shift0, shift1);
    
    outQue_.EnQue(outLocal);
    inputQue_.FreeTensor(xLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceBS<T, SeqType, CompType>::ComputeSplitA(int64_t srcOffset, int64_t dimNum)
{
    CopyIn(srcOffset, dimNum);
    ReverseABSCompute(dimNum, tilingData_->bDim * tilingData_->sDim, tilingData_->sDim);

    auto mte3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
    WaitFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
    CopyOut(srcOffset, dimNum);
}
}

#endif