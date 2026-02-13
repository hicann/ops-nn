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
 * \file reverse_sequence_bsa.h
 * \brief
 */
#ifndef REVERSE_SEQUENCE_BSA_KERNEL_H_
#define REVERSE_SEQUENCE_BSA_KERNEL_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "reverse_sequence_struct.h"

namespace ReverseSequence {

using namespace AscendC;

constexpr int32_t B64 = 8;
constexpr int32_t B8 = 1;
constexpr int32_t B16 = 2;
constexpr int32_t B32 = 4;
static constexpr int64_t DB_BUFFER = 2;
static constexpr int64_t SPLIT_DIM_A = 1;
static constexpr int64_t SPLIT_DIM_S = 2;
static constexpr int64_t SPLIT_DIM_B = 3;
static constexpr int64_t GATHER_DIM_S = 0;
static constexpr int64_t GATHER_DIM_B = 1;
static constexpr int64_t NOT_GATHER = 1001;

template <typename T>
struct GetGatherType {
    using type =
        typename std::conditional<std::is_same<T, int8_t>::value, int16_t,
                                  typename std::conditional<std::is_same<T, uint8_t>::value, uint16_t, T>::type>::type;
};

template <typename T>
struct IndexTypeGet {
    using type = typename std::conditional<sizeof(T) == B8 || sizeof(T) == B16, uint16_t, uint32_t>::type;
};

template <typename T>
struct VciTypeGet {
    using type = typename std::conditional<
        std::is_same<T, uint32_t>::value, int32_t,
        typename std::conditional<std::is_same<T, uint16_t>::value, int16_t,
                                  typename std::conditional<std::is_same<T, uint64_t>::value, int64_t, T>::type>::type>::type;
};

template <typename T, typename SeqType>
class ReverseSequenceBSA
{
public:
    __aicore__ inline ReverseSequenceBSA(TPipe *pipe, const ReverseSequenceBSATilingData *tilingData)
                        : pipe_(pipe), tilingData_(tilingData) {};
     __aicore__ inline void Init(GM_ADDR x, GM_ADDR seqLengths, GM_ADDR y);
     __aicore__ inline void Process();
private:
    template <typename U, int64_t GatherMode, int64_t SplitMode>
    __aicore__ inline void BaseCompute();
    template <typename U>
    __aicore__ inline void GenDimSGatherIndex(int64_t reverseLen, int64_t stride, LocalTensor<U>& indexLocal);
    template <typename U>
    __aicore__ inline void GenGatherDimBIndex(int64_t notReverseLen, int64_t seqLen, int64_t stride, LocalTensor<U>& indexLocal);
    __aicore__ inline void ComputeSplitA(int64_t srcOffset, int64_t bStart, int64_t sStart, int64_t aStart, int64_t dimANum);
    template <typename U, int64_t GatherMode>
    __aicore__ inline void ComputeSplitS(int64_t srcOffset, int64_t bStart, int64_t sStart, int64_t dimSNum);
    template <typename U>
    __aicore__ inline void ComputeSplitSGatherSmallA(int64_t srcOffset, int64_t bStart, int64_t sStart, int64_t dimSNum);
    __aicore__ inline void ComputeSplitSCopyA(int64_t srcOffset, int64_t copyDims);
    __aicore__ inline void ComputeSplitSReverseBigA(int64_t srcOffset, int64_t bStart, int64_t sStart, int64_t dimSNum);
    template <typename U, int64_t GatherMode>
    __aicore__ inline void ComputeSplitB(int64_t srcOffset, int64_t bStart, int64_t dimBNum);
    template <typename U>
    __aicore__ inline void ComputeSplitBGather(__local_mem__ T* srcAddr, __local_mem__ T* dstAddr, int32_t totalNum);
    __aicore__ inline void ComputeSplitBNotGather(__local_mem__ int8_t* srcAddr, __local_mem__ int8_t* dstAddr);
    __aicore__ inline void CopyInMultiDim(int64_t offset, int64_t blockCount, int64_t blockLen);
    __aicore__ inline void CopyOutMultiDim(int64_t offset, int64_t blockCount, int64_t blockLen);
    __aicore__ inline void CopyInSingleDim(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyOutSingleDim(int64_t offset, int64_t blockLen);
    __aicore__ inline void GetCurrentSeqLength(int64_t bStart);
    TPipe *pipe_;
    const ReverseSequenceBSATilingData *tilingData_;    
    TQue<QuePosition::VECIN, DB_BUFFER> inputQue_;
    // 输出ub
    TQue<QuePosition::VECOUT, DB_BUFFER> outQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<SeqType> seqGm_;
    int64_t ubBlockSize_ = platform::GetUbBlockSize();
    int64_t eleBlk_ = 1;
    int64_t bStart_ = -1;
    int64_t seqLen_ = 0;
};

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::Init(GM_ADDR x, GM_ADDR seqLengths, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    seqGm_.SetGlobalBuffer((__gm__ SeqType*)seqLengths);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    if (tilingData_->splitMode == SPLIT_DIM_A) {
        pipe_->InitBuffer(dataQueue_, DB_BUFFER, tilingData_->inUbSize * sizeof(T));
    } else {
        pipe_->InitBuffer(inputQue_, DB_BUFFER, tilingData_->inUbSize * sizeof(T));
        pipe_->InitBuffer(outQue_, DB_BUFFER, tilingData_->inUbSize * sizeof(T));
    }
    using indiceType = typename IndexTypeGet<T>::type;
    pipe_->InitBuffer(indexBuf_, tilingData_->gatherUbSize * sizeof(indiceType));
    eleBlk_ = ubBlockSize_ / tilingData_->dtypeSize;
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::Process()
{
    using indiceType = typename IndexTypeGet<T>::type;

    if (tilingData_->splitMode == SPLIT_DIM_S) {
        if (tilingData_->gatherMode != NOT_GATHER) {
            BaseCompute<indiceType, GATHER_DIM_S, SPLIT_DIM_S>();
        } else {
            BaseCompute<indiceType, NOT_GATHER, SPLIT_DIM_S>();
        }
    } else if (tilingData_->splitMode == SPLIT_DIM_B) {
         if (tilingData_->gatherMode != NOT_GATHER) {
            BaseCompute<indiceType, GATHER_DIM_B, SPLIT_DIM_B>();
        } else {
            BaseCompute<indiceType, NOT_GATHER, SPLIT_DIM_B>();
        }
    } else {
        BaseCompute<indiceType, NOT_GATHER, SPLIT_DIM_A>();
    }
}

template <typename T, typename SeqType>
template <typename U, int64_t GATHER_MODE, int64_t SplitMode>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::BaseCompute()
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
        int64_t srcOffset = bStart * tilingData_->sDim * tilingData_->aDim + sStart * tilingData_->aDim + aStart;
        if constexpr (SplitMode == SPLIT_DIM_A) {
            ComputeSplitA(srcOffset, bStart, sStart, aStart, inDimA);
        } else if constexpr (SplitMode == SPLIT_DIM_S) {
            ComputeSplitS<U, GATHER_MODE>(srcOffset, bStart, sStart, inDimS);
        } else {
            ComputeSplitB<U, GATHER_MODE>(srcOffset, bStart, inDimB);
        }
    }
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::CopyInMultiDim(int64_t offset, int64_t blockCount, int64_t blockLen)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
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
    inputQue_.EnQue<T>(xLocal);
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::CopyOutMultiDim(int64_t offset, int64_t blockCount, int64_t blockLen)
{
    DataCopyExtParams copyExtParams;
    int64_t alignBlockLen = ops::Aligned(blockLen, eleBlk_);
    copyExtParams.blockCount = blockCount;
    copyExtParams.blockLen = blockLen * tilingData_->dtypeSize;
    copyExtParams.srcStride = (alignBlockLen - blockLen) * tilingData_->dtypeSize / platform::GetUbBlockSize();
    copyExtParams.dstStride = 0;
    LocalTensor<T> yLocal = outQue_.DeQue<T>();
    DataCopyPad(yGm_[offset], yLocal, copyExtParams);
    outQue_.FreeTensor(yLocal);
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::CopyInSingleDim(int64_t offset, int64_t blockLen)
{
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

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::CopyOutSingleDim(int64_t offset, int64_t blockLen)
{
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = 1;
    copyExtParams.blockLen = blockLen * tilingData_->dtypeSize;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
    DataCopyPad(yGm_[offset], yLocal, copyExtParams);
    dataQueue_.FreeTensor(yLocal);
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::GetCurrentSeqLength(int64_t bStart)
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
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::ComputeSplitA(
    int64_t srcOffset, int64_t bStart, int64_t sStart, int64_t aStart, int64_t dimANum)
{
    CopyInSingleDim(srcOffset, dimANum);
    GetCurrentSeqLength(bStart);
    int64_t outOffset = bStart * tilingData_->sDim * tilingData_->aDim;
    if (sStart < seqLen_) {
        outOffset += (seqLen_ - sStart - 1) * tilingData_->aDim + aStart;
    } else {
        outOffset = srcOffset;
    }
    CopyOutSingleDim(outOffset, dimANum);
}

template <typename T, typename SeqType>
template <typename U, int64_t GatherMode>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::ComputeSplitS(
    int64_t srcOffset, int64_t bStart, int64_t sStart, int64_t dimSNum)
{
    GetCurrentSeqLength(bStart);
    if constexpr (GatherMode == GATHER_DIM_S) {
        // 小于seq_len部分 gather
        if (sStart + dimSNum < seqLen_) {
            ComputeSplitSGatherSmallA<U>(srcOffset, bStart, sStart,  dimSNum);
        } else if (sStart >= seqLen_) {
            //直接搬入搬出
            ComputeSplitSCopyA(srcOffset, dimSNum);
        } else {
            int64_t reverseDims = seqLen_ - sStart;
            ComputeSplitSGatherSmallA<U>(srcOffset, bStart, sStart,  reverseDims);
            int64_t copyDims = dimSNum - reverseDims;
            int64_t offset = srcOffset + reverseDims * tilingData_->aDim;
            ComputeSplitSCopyA(offset, copyDims);

        }
    } else {
        if (sStart + dimSNum < seqLen_) {
            ComputeSplitSReverseBigA(srcOffset, bStart, sStart,  dimSNum);
        } else if (sStart >= seqLen_) {
            //直接搬入搬出
            ComputeSplitSCopyA(srcOffset, dimSNum);
        } else {
            int64_t reverseDims = seqLen_ - sStart;
            ComputeSplitSReverseBigA(srcOffset, bStart, sStart,  reverseDims);
            int64_t copyDims = dimSNum - reverseDims;
            int64_t offset = srcOffset + reverseDims * tilingData_->aDim;
            ComputeSplitSCopyA(offset, copyDims);
        }
    }
}

template <typename T, typename SeqType>
template <typename U>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::GenDimSGatherIndex(
    int64_t reverseLen, int64_t stride, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    // (dimsNum - i / dimA - 1) * stride + i % dimA
    uint16_t repeatNum = platform::GetVRegSize() / sizeof(U);
    uint16_t loopNum = (reverseLen * stride + repeatNum - 1) / repeatNum;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd7;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, i * repeatNum);
            MicroAPI::Duplicate(v1, (U)stride, p0);
            MicroAPI::Duplicate(v2, (U)reverseLen, p0);
            MicroAPI::Duplicate(v3, (U)1, p0);

            MicroAPI::Div(vd1, v0, v1, p0);
            MicroAPI::Sub(vd2, v2, vd1, p0);
            MicroAPI::Sub(vd3, vd2, v3, p0); 
            MicroAPI::Mul(vd4, vd3, v1, p0); // (dimSNum - i/ dimA - 1) * stride

            MicroAPI::Mul(vd5, vd1, v1, p0);
            MicroAPI::Sub(vd6, v0, vd5, p0); // i % dimA
            MicroAPI::Add(vd7, vd4, vd6, p0);
            MicroAPI::DataCopy(dstAddr  + i * repeatNum, vd7, p0);
        }
    }
}

template <typename T, typename SeqType>
template <typename U>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::ComputeSplitSGatherSmallA(
    int64_t srcOffset, int64_t bStart, int64_t sStart, int64_t dimSNum)
{
    CopyInMultiDim(srcOffset, 1, dimSNum * tilingData_->aDim);
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    GenDimSGatherIndex<U>(dimSNum, tilingData_->aDim, indexLocal);
    LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* yLocalAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    uint32_t repeatNum = platform::GetVRegSize() / sizeof(U);
    uint16_t totalNum = dimSNum * tilingData_->aDim;
    uint16_t loop = (totalNum + repeatNum - 1) / repeatNum;
    __VEC_SCOPE__
    {
        using RegDstT = typename std::conditional<sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
        using gatherType = typename GetGatherType<T>::type;
        uint32_t updateNum = totalNum;
        uint32_t updateNumB8 = totalNum;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<gatherType> vd0;
        RegDstT vd1;
        MicroAPI::MaskReg pMask;
        MicroAPI::MaskReg pMaskB8;
        for (uint16_t i = 0; i < loop; i++) {
            pMask = MicroAPI::UpdateMask<U>(updateNum);
            MicroAPI::DataCopy(v0, indexAddr + i * repeatNum);
            if constexpr (sizeof(T) == 1) {
                MicroAPI::DataCopyGather(vd0, xLocalAddr, v0, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint8_t>&)vd1, vd0);
                MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(pMaskB8, pMask);
                MicroAPI::DataCopy(yLocalAddr + i * repeatNum, vd1, pMaskB8);
            } else {
                MicroAPI::DataCopyGather(vd1, xLocalAddr, v0, pMask);
                MicroAPI::DataCopy(yLocalAddr + i * repeatNum, vd1, pMask);
            }
        }
    }
    inputQue_.FreeTensor(xLocal);
    outQue_.EnQue(yLocal);
    int64_t outOffset = bStart * tilingData_->sDim * tilingData_->aDim + (seqLen_ - dimSNum - sStart) * tilingData_->aDim;
    CopyOutMultiDim(outOffset, 1, dimSNum * tilingData_->aDim);
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::ComputeSplitSCopyA(int64_t srcOffset, int64_t copyDims)
{
    CopyInMultiDim(srcOffset, 1, copyDims * tilingData_->aDim);
    LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    __local_mem__ int8_t* xLocalAddr = (__local_mem__ int8_t*)xLocal.GetPhyAddr();
    __local_mem__ int8_t* yLocalAddr = (__local_mem__ int8_t*)yLocal.GetPhyAddr();
    uint16_t repeatNum = platform::GetVRegSize();
    uint32_t totalNum = copyDims * tilingData_->aDim * tilingData_->dtypeSize;
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
    CopyOutMultiDim(srcOffset, 1, copyDims * tilingData_->aDim);
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::ComputeSplitSReverseBigA(
    int64_t srcOffset, int64_t bStart, int64_t sStart, int64_t dimSNum)
{
    CopyInMultiDim(srcOffset, dimSNum, tilingData_->aDim);
    LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    __local_mem__ int8_t* xLocalAddr = (__local_mem__ int8_t*)xLocal.GetPhyAddr();
    __local_mem__ int8_t* yLocalAddr = (__local_mem__ int8_t*)yLocal.GetPhyAddr();
    uint16_t repeatNum = platform::GetVRegSize();
    uint16_t dimASize = tilingData_->aDim * tilingData_->dtypeSize;
    uint16_t dimALoop = (dimASize + repeatNum - 1) / repeatNum;
    uint16_t dimSLoop = dimSNum;
    uint32_t dimSStride = ops::Aligned(static_cast<int32_t>(dimASize), static_cast<int32_t>(ubBlockSize_));
    uint32_t seqLen = dimSNum;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int8_t> inReg;
        AscendC::MicroAPI::MaskReg preg;
        for (uint16_t i = 0; i < dimSLoop; i++) {
            auto srcAddr = xLocalAddr + i * dimSStride;
            auto dstAddr = yLocalAddr + (seqLen - i - 1) * dimSStride;
            uint32_t updateNum = dimSStride;
            for (uint16_t j = 0; j < dimALoop; j++) {
                preg = AscendC::MicroAPI::UpdateMask<int8_t>(updateNum);
                auto curSrcAddr = srcAddr + j * repeatNum;
                auto curDstAddr = dstAddr + j * repeatNum;
                AscendC::MicroAPI::DataCopy(inReg, curSrcAddr);
                AscendC::MicroAPI::DataCopy(curDstAddr, inReg, preg);
            }
        }
    }
    inputQue_.FreeTensor(xLocal);
    outQue_.EnQue(yLocal);
    int64_t outOffset = bStart * tilingData_->sDim * tilingData_->aDim + (seqLen_ - dimSNum - sStart) * tilingData_->aDim;
    CopyOutMultiDim(outOffset, dimSNum, tilingData_->aDim);
}

template <typename T, typename SeqType>
template <typename U>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::GenGatherDimBIndex(
    int64_t notReverseLen, int64_t seqLen, int64_t stride, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    auto dstAddr1 = dstAddr + seqLen * stride;
    // (dimsNum - i / dimA - 1)
    uint32_t startNum = seqLen * stride;
    uint16_t repeatNum = platform::GetVRegSize() / sizeof(U);
    uint16_t loopNum = (startNum + repeatNum - 1) / repeatNum;
    uint16_t notReverseTotalNum = notReverseLen * stride;
    uint16_t notReverseLoopNum = notReverseTotalNum / repeatNum;
    uint16_t tailNum = notReverseTotalNum - notReverseLoopNum * repeatNum;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        uint32_t updateNum = startNum;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd7;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> vOne;
        for (uint16_t i = 0; i < loopNum; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(updateNum);;
            MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, i * repeatNum);
            MicroAPI::Duplicate(v1, (U)stride, p0);
            MicroAPI::Duplicate(v2, (U)seqLen, p0);
            MicroAPI::Duplicate(vOne, (U)1, p0);
            MicroAPI::Div(vd1, v0, v1, p0);
            MicroAPI::Sub(vd2, v2, vd1, p0);
            MicroAPI::Sub(vd3, vd2, vOne, p0);
            MicroAPI::Mul(vd4, vd3, v1, p0); // (dimSNum - i/ dimA - 1) * stride
            MicroAPI::Mul(vd5, vd1, v1, p0);
            // i % dimA
            MicroAPI::Sub(vd6, v0, vd5, p0);
            MicroAPI::Add(vd7, vd4, vd6, p0);
            MicroAPI::DataCopy(dstAddr  + i * repeatNum, vd7, p0);
        }

        MicroAPI::RegTensor<U> v5;
        MicroAPI::UnalignReg u0;
        for (uint16_t i = 0; i < notReverseLoopNum; i++) {
            MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v5, startNum + i * repeatNum);
            MicroAPI::DataCopyUnAlign(dstAddr1, v5, u0, repeatNum);
        }
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v5, startNum + notReverseLoopNum * repeatNum);
        MicroAPI::DataCopyUnAlign(dstAddr1, v5, u0, tailNum);
        MicroAPI::DataCopyUnAlignPost(dstAddr1, u0, 0);
    }
}

template <typename T, typename SeqType>
template <typename U, int64_t GatherMode>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::ComputeSplitB(int64_t srcOffset, int64_t bStart, int64_t dimBNum)
{
    if constexpr (GatherMode == GATHER_DIM_B) {
        CopyInMultiDim(srcOffset, dimBNum, tilingData_->aDim * tilingData_->sDim); // SA 对齐拷入
        LocalTensor<U> indexLocal = indexBuf_.Get<U>();
        LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
        LocalTensor<T> xLocal = inputQue_.DeQue<T>();
        __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
        __local_mem__ T* yLocalAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
        int32_t dimBStride = ops::Aligned(tilingData_->aDim * tilingData_->sDim, eleBlk_);
        int32_t totalNum = tilingData_->aDim * tilingData_->sDim;
        for (int64_t i = 0; i < dimBNum; i++) {
            auto srcAddr = xLocalAddr + i * dimBStride;
            auto dstAddr = yLocalAddr + i * dimBStride;
            GetCurrentSeqLength(bStart + i);
            int64_t notReverseLen = tilingData_->sDim - seqLen_;
            GenGatherDimBIndex<U>(notReverseLen, seqLen_, tilingData_->aDim, indexLocal);
            ComputeSplitBGather<U>(srcAddr, dstAddr, totalNum);
        }
        inputQue_.FreeTensor(xLocal);
        outQue_.EnQue(yLocal);
        CopyOutMultiDim(srcOffset, dimBNum, tilingData_->aDim * tilingData_->sDim);
    } else {
        CopyInMultiDim(srcOffset, dimBNum * tilingData_->sDim, tilingData_->aDim);  //A对齐拷入拷出
        LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
        LocalTensor<T> xLocal = inputQue_.DeQue<T>();
        __local_mem__ int8_t* xLocalAddr = (__local_mem__ int8_t*)xLocal.GetPhyAddr();
        __local_mem__ int8_t* yLocalAddr = (__local_mem__ int8_t*)yLocal.GetPhyAddr();
        int32_t dimBStride = tilingData_->sDim *ops::Aligned(tilingData_->aDim * tilingData_->dtypeSize, ubBlockSize_);
        for (int64_t i = 0; i < dimBNum; i++) {
            GetCurrentSeqLength(bStart + i);
            auto srcAddr = xLocalAddr + i * dimBStride;
            auto dstAddr = yLocalAddr + i * dimBStride;
            ComputeSplitBNotGather(srcAddr, dstAddr);
        }
        inputQue_.FreeTensor(xLocal);
        outQue_.EnQue(yLocal);
        CopyOutMultiDim(srcOffset, dimBNum * tilingData_->sDim, tilingData_->aDim);
    }
}

template <typename T, typename SeqType>
template <typename U>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::ComputeSplitBGather(
    __local_mem__ T* srcAddr, __local_mem__ T* dstAddr, int32_t totalNum)
{
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    uint32_t repeatNum = platform::GetVRegSize() / sizeof(U);
    uint32_t gatherNum = totalNum;
    uint16_t loop = (gatherNum + repeatNum - 1) / repeatNum;
    auto srcAddr1 = srcAddr;
    auto dstAddr1 = dstAddr;
    __VEC_SCOPE__
    {
        using RegDstT = typename std::conditional<sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>,
                                      MicroAPI::RegTensor<T>>::type;
        using gatherType = typename GetGatherType<T>::type;
        uint32_t updateNum = gatherNum;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<gatherType> vd0;
        RegDstT vd1;
        MicroAPI::MaskReg pMask;
        MicroAPI::MaskReg pMaskB8;
        for (uint16_t i = 0; i < loop; i++) {
            pMask = MicroAPI::UpdateMask<U>(updateNum);
            MicroAPI::DataCopy(v0, indexAddr + i * repeatNum);
            if constexpr (sizeof(T) == 1) {
                MicroAPI::DataCopyGather(vd0, srcAddr1, v0, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint8_t>&)vd1, vd0);
                MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(pMaskB8, pMask);
                MicroAPI::DataCopy(dstAddr1 + i * repeatNum, vd1, pMaskB8);
            } else {
                MicroAPI::DataCopyGather(vd1, srcAddr1, v0, pMask);
                MicroAPI::DataCopy(dstAddr1 + i * repeatNum, vd1, pMask);
            }
        }
    }
}

template <typename T, typename SeqType>
__aicore__ inline void ReverseSequenceBSA<T, SeqType>::ComputeSplitBNotGather(
    __local_mem__ int8_t* srcAddr, __local_mem__ int8_t* dstAddr)
{
    uint16_t repeatNum = platform::GetVRegSize();
    uint32_t dimASize = tilingData_->aDim * tilingData_->dtypeSize;
    uint32_t dimSStride = ops::Aligned(static_cast<int32_t>(dimASize), static_cast<int32_t>(ubBlockSize_));
    auto srcAddr2 = srcAddr + seqLen_ * dimSStride;
    auto dstAddr2 = dstAddr + seqLen_ * dimSStride;
    uint16_t dimALoop = (dimASize + repeatNum - 1) / repeatNum;
    uint16_t dimSLoop = seqLen_;
    uint32_t notRevrseTotalNum = (tilingData_->sDim - seqLen_) * dimSStride;
    uint16_t notRevrseLoop = (notRevrseTotalNum + repeatNum - 1) / repeatNum;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int8_t> inReg;
        AscendC::MicroAPI::MaskReg preg;
        for (uint16_t i = 0; i < dimSLoop; i++) {
            auto srcAddr1 = srcAddr + i * dimSStride;
            auto dstAddr1 = dstAddr + (dimSLoop - i - 1) * dimSStride;
            uint32_t updateNum = dimSStride;
            for (uint16_t j = 0; j < dimALoop; j++) {
                preg = AscendC::MicroAPI::UpdateMask<int8_t>(updateNum);
                auto curSrcAddr = srcAddr1 + j * repeatNum;
                auto curDstAddr = dstAddr1 + j * repeatNum;
                AscendC::MicroAPI::DataCopy(inReg, curSrcAddr);
                AscendC::MicroAPI::DataCopy(curDstAddr, inReg, preg);
            }
        }
        uint32_t notReverseNum = notRevrseTotalNum;
        MicroAPI::RegTensor<int8_t> v0;
        AscendC::MicroAPI::MaskReg preg1;
        for (uint16_t i = 0; i < notRevrseLoop; i++) {
            preg1 = AscendC::MicroAPI::UpdateMask<int8_t>(notReverseNum);
            MicroAPI::DataCopy(v0, srcAddr2 + i * repeatNum);
            MicroAPI::DataCopy(dstAddr2 + i * repeatNum, v0, preg1);
        }
    }
}
}

#endif