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
 * \file reverse_sequence_bas.h
 * \brief
 */
#ifndef REVERSE_SEQUENCE_BAS_KERNEL_H_
#define REVERSE_SEQUENCE_BAS_KERNEL_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "reverse_sequence_struct.h"

namespace ReverseSequence {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;
static constexpr int64_t SPLIT_S = 1;
static constexpr int64_t SPLIT_A = 2;
static constexpr int64_t SPLIT_B = 3;
static constexpr int64_t DOUBLE_BUFFER = 2;


template <typename T, typename SeqType>
class ReverseSequenceBAS
{
public:
    __aicore__ inline ReverseSequenceBAS(TPipe *pipe, const ReverseSequenceBASTilingData *tilingData)
                        : pipe_(pipe), tilingData_(tilingData) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR seqLengths, GM_ADDR y);
    __aicore__ inline void Process();

private:
    template <int64_t SplitMode>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void ComputeSplitA(int64_t srcOffset, int64_t bStart, int64_t dimANum);
    __aicore__ inline void ComputeSplitS(int64_t srcOffset, int64_t bStart, int64_t sStart, int64_t aStart, int64_t dimSNum);
    __aicore__ inline void ComputeSplitB(int64_t srcOffset, int64_t bStart, int64_t dimBNum);
    __aicore__ inline void GetCurrentSeqLength(int64_t bStart);
    __aicore__ inline void CopyInSingleDim(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyInSeqs(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyOutSingleDim(int64_t offset, int64_t blockLen);
    __aicore__ inline void ComputeSplitSReverse(int64_t srcOffset, int64_t dimSNum, int64_t outOffset);
    __aicore__ inline void ComputeSplitSCopy(int64_t srcOffset, int64_t copyDims);

    TPipe *pipe_;
    const ReverseSequenceBASTilingData *tilingData_;    
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inputQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> seqQue_;
    // 输出ub
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<SeqType> seqGm_;
    int64_t ubBlockSize_ = platform::GetUbBlockSize();
    int64_t eleBlk_ = 1;
    int64_t bStart_ = -1;
    int64_t seqLen_ = 0;
};

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::Init(GM_ADDR x, GM_ADDR seqLengths, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    seqGm_.SetGlobalBuffer((__gm__ SeqType*)seqLengths);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    if (tilingData_->splitMode == SPLIT_B) {
        pipe_->InitBuffer(inputQue_, DOUBLE_BUFFER, tilingData_->inUbSize * sizeof(T));
        pipe_->InitBuffer(seqQue_, DOUBLE_BUFFER, tilingData_->seqUbByte);
        pipe_->InitBuffer(outQue_, DOUBLE_BUFFER, tilingData_->inUbSize * sizeof(T));
    } else {
        pipe_->InitBuffer(inputQue_, DOUBLE_BUFFER, tilingData_->inUbSize * sizeof(T));
        pipe_->InitBuffer(outQue_, DOUBLE_BUFFER, tilingData_->inUbSize * sizeof(T));
    }
    
    eleBlk_ = ubBlockSize_ / tilingData_->dtypeSize;
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }

    if (tilingData_->splitMode == SPLIT_S) {
        BaseCompute<SPLIT_S>();
    } else if (tilingData_->splitMode == SPLIT_B) {
        BaseCompute<SPLIT_B>();
    } else {
        BaseCompute<SPLIT_A>();
    }
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::GetCurrentSeqLength(int64_t bStart)
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

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::CopyInSingleDim(int64_t offset, int64_t blockLen)
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
template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::CopyInSeqs(int64_t offset, int64_t blockLen)
{
    LocalTensor<SeqType> seqLocal = seqQue_.AllocTensor<SeqType>();
    DataCopyPadExtParams<SeqType> padParams{false, 0, 0, 0};
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = 1;
    copyExtParams.blockLen = blockLen * sizeof(SeqType);
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(seqLocal, seqGm_[offset], copyExtParams, padParams);
    seqQue_.EnQue<SeqType>(seqLocal);
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::CopyOutSingleDim(int64_t offset, int64_t blockLen)
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

template <typename T, typename SeqType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtSplitSReverse(__local_mem__ T* xLocal, __local_mem__ T* yLocal, uint32_t dimSNum)
{
    for (uint32_t i = Simt::GetThreadIdx(); i < dimSNum; i += Simt::GetThreadNum()) {
        yLocal[i] = xLocal[dimSNum - i - 1];
    }
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::ComputeSplitSReverse(
    int64_t srcOffset, int64_t dimSNum, int64_t outOffset)
{
    CopyInSingleDim(srcOffset, dimSNum);
    LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();

    Simt::VF_CALL<SimtSplitSReverse<T, SeqType>>(
        Simt::Dim3(THREAD_NUM), (__local_mem__ T*)(xLocal.GetPhyAddr()), (__local_mem__ T*)(yLocal.GetPhyAddr()), dimSNum);
    
    inputQue_.FreeTensor(xLocal);
    outQue_.EnQue(yLocal);
    CopyOutSingleDim(outOffset, dimSNum);
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::ComputeSplitSCopy(int64_t srcOffset, int64_t copyDims)
{
    CopyInSingleDim(srcOffset, copyDims);
    LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    __local_mem__ int8_t* xLocalAddr = (__local_mem__ int8_t*)xLocal.GetPhyAddr();
    __local_mem__ int8_t* yLocalAddr = (__local_mem__ int8_t*)yLocal.GetPhyAddr();
    uint16_t repeatNum = platform::GetVRegSize();
    uint32_t totalNum = copyDims * tilingData_->dtypeSize;
    uint16_t loop = (totalNum + repeatNum - 1) / repeatNum;
    __VEC_SCOPE__
    {
        uint32_t updateNum = totalNum;
        MicroAPI::RegTensor<int8_t> v0;
        AscendC::MicroAPI::MaskReg preg;
        for (uint16_t i = 0; i < loop; i++) {
            preg = AscendC::MicroAPI::UpdateMask<int8_t>(updateNum);
            MicroAPI::DataCopy(v0, xLocalAddr + i * repeatNum);
            MicroAPI::DataCopy(yLocalAddr + i * repeatNum, v0, preg);
        }
    }
    inputQue_.FreeTensor(xLocal);
    outQue_.EnQue(yLocal);
    CopyOutSingleDim(srcOffset, copyDims);
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::ComputeSplitS(
    int64_t srcOffset, int64_t bStart, int64_t sStart, int64_t aStart, int64_t dimSNum)
{
    GetCurrentSeqLength(bStart);
    int64_t outOffset = bStart * tilingData_->sDim * tilingData_->aDim + aStart * tilingData_->sDim;
    if (sStart + dimSNum <= seqLen_) { // <
        // 小于等于seq_len部分 simt gather
        outOffset += seqLen_ - dimSNum - sStart;
        ComputeSplitSReverse(srcOffset,  dimSNum, outOffset);
    } else if (sStart >= seqLen_) {
        // 直接搬入搬出
        ComputeSplitSCopy(srcOffset, dimSNum); // copy(out, intput)
    } else {
        int64_t reverseDims = seqLen_ - sStart;
        outOffset += seqLen_ - reverseDims - sStart;
        ComputeSplitSReverse(srcOffset,  reverseDims, outOffset);
        int64_t copyDims = dimSNum - reverseDims;
        int64_t offset = srcOffset + reverseDims;
        ComputeSplitSCopy(offset, copyDims);
    }
}

template <typename T, typename SeqType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtSplitAReverse(__local_mem__ T* xLocal, __local_mem__ T* yLocal, uint32_t totalNum, uint32_t seqLen_, uint32_t m0, uint32_t shift0, uint32_t sDim)
{
    for (uint32_t i = Simt::GetThreadIdx(); i < totalNum; i += Simt::GetThreadNum()) {
        uint32_t dimAIdx = Simt::UintDiv(i, m0, shift0);
        uint32_t curAOffset = dimAIdx * sDim;
        uint32_t dimSIdx = i - curAOffset;
        yLocal[i] = (dimSIdx < seqLen_) ? xLocal[curAOffset + seqLen_ - dimSIdx - 1] : xLocal[i];
    }
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::ComputeSplitA(
    int64_t srcOffset, int64_t bStart, int64_t dimANum)
{
    GetCurrentSeqLength(bStart);
    CopyInSingleDim(srcOffset, dimANum * tilingData_->sDim);
    LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    
    uint32_t m0 = 1;
    uint32_t shift0 = 1;
    GetUintDivMagicAndShift(m0, shift0, static_cast<uint32_t>(tilingData_->sDim));
    uint32_t totalNum = dimANum * tilingData_->sDim;
    Simt::VF_CALL<SimtSplitAReverse<T, SeqType>>(
        Simt::Dim3(THREAD_NUM), (__local_mem__ T*)(xLocal.GetPhyAddr()), (__local_mem__ T*)(yLocal.GetPhyAddr()), totalNum, seqLen_, m0, shift0, tilingData_->sDim);
    
    inputQue_.FreeTensor(xLocal);
    outQue_.EnQue(yLocal);
    CopyOutSingleDim(srcOffset, dimANum * tilingData_->sDim);
}

template <typename T, typename SeqType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtSplitBReverse(__local_mem__ T* xLocal, __local_mem__ SeqType* seqLocal, __local_mem__ T* yLocal, uint32_t asNum, uint32_t m0, uint32_t shift0, uint32_t sDim, uint32_t dimBNum)
{
    for (uint32_t dimBIdx = Simt::GetThreadIdx<1>(); dimBIdx < dimBNum; dimBIdx += Simt::GetThreadNum<1>()) {
        SeqType seqLen = seqLocal[dimBIdx];
        uint32_t curBOffset = dimBIdx * asNum;
        for (uint32_t i = Simt::GetThreadIdx<0>(); i < asNum; i += Simt::GetThreadNum<0>()) {
            uint32_t dimAIdx = Simt::UintDiv(i, m0, shift0);
            uint32_t curAOffset = dimAIdx * sDim;
            uint32_t dimSIdx = i - curAOffset;
            yLocal[curBOffset + i] = (dimSIdx < seqLen) ? xLocal[curBOffset + curAOffset + seqLen - dimSIdx - 1] : xLocal[curBOffset + i];
        }
    }
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::ComputeSplitB(int64_t srcOffset, int64_t bStart, int64_t dimBNum)
{
    CopyInSingleDim(srcOffset, dimBNum * tilingData_->sDim * tilingData_->aDim);
    CopyInSeqs(bStart, dimBNum);
    LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    LocalTensor<SeqType> seqLocal = seqQue_.DeQue<SeqType>();

    uint32_t m0 = 1;
    uint32_t shift0 = 1;
    GetUintDivMagicAndShift(m0, shift0, static_cast<uint32_t>(tilingData_->sDim));
    uint32_t threadNumX = static_cast<uint32_t>(tilingData_->threadNumX);
    uint32_t threadNumY = THREAD_NUM / threadNumX;
    uint32_t asNum = tilingData_->aDim * tilingData_->sDim;
    Simt::VF_CALL<SimtSplitBReverse<T, SeqType>>(
        Simt::Dim3({threadNumX, threadNumY}), (__local_mem__ T*)(xLocal.GetPhyAddr()), (__local_mem__ SeqType*)(seqLocal.GetPhyAddr()),
        (__local_mem__ T*)(yLocal.GetPhyAddr()), asNum, m0, shift0, tilingData_->sDim, dimBNum);
    
    inputQue_.FreeTensor(xLocal);
    seqQue_.FreeTensor(seqLocal);
    outQue_.EnQue(yLocal);
    CopyOutSingleDim(srcOffset, dimBNum * tilingData_->sDim * tilingData_->aDim);
}

template <typename T, typename SeqType>
template <int64_t SplitMode>
__aicore__ inline void ReverseSequenceBAS<T, SeqType>::BaseCompute()
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
        int64_t bIdx = idx / (tilingData_->sLoop * tilingData_->aLoop);
        int64_t sIdx = (idx - bIdx * tilingData_->sLoop * tilingData_->aLoop) / tilingData_->aLoop;
        int64_t aIdx = idx % tilingData_->aLoop;
        int64_t inDimB =
            bIdx == tilingData_->bLoop - 1 ? tilingData_->bDim - bIdx * tilingData_->ubFactorB : tilingData_->ubFactorB;
        int64_t inDimS =
            sIdx == tilingData_->sLoop - 1 ? tilingData_->sDim - sIdx * tilingData_->ubFactorS : tilingData_->ubFactorS;
        int64_t inDimA =
            aIdx == tilingData_->aLoop - 1 ? tilingData_->aDim - aIdx * tilingData_->ubFactorA : tilingData_->ubFactorA;
        
        int64_t bStart = bIdx * tilingData_->ubFactorB;
        int64_t sStart = sIdx * tilingData_->ubFactorS;
        int64_t aStart = aIdx * tilingData_->ubFactorA;
        int64_t srcOffset = bStart * tilingData_->sDim * tilingData_->aDim + aStart * tilingData_->sDim + sStart;
        if constexpr (SplitMode == SPLIT_A) {
            ComputeSplitA(srcOffset, bStart, inDimA);
        } else if constexpr (SplitMode == SPLIT_S) {
            ComputeSplitS(srcOffset, bStart, sStart, aStart, inDimS);
        } else {
            ComputeSplitB(srcOffset, bStart, inDimB);
        }
    }
}

}

#endif