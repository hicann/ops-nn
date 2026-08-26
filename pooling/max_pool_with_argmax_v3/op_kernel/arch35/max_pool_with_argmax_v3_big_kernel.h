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
 * \file max_pool_with_argmax_v3_big_kernel.h
 * \brief
 */
#ifndef MAX_POOL_WITH_ARGMAX_V3_BIG_KERNEL_H_
#define MAX_POOL_WITH_ARGMAX_V3_BIG_KERNEL_H_

#include "max_pool_with_argmax_v3_base.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace MaxPoolWithArgMaxV3BigKernel {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000; // -inf 0xFF800000
constexpr int32_t OUT_BUFFER_LEN = 1024;
constexpr int32_t EIGHT = 8;
constexpr int32_t FOUR = 4;
constexpr int32_t TWO = 2;

constexpr Reg::CastTrait castTraitB322B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::CAST_RINT};

constexpr Reg::CastTrait castTraitB162B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};

constexpr Reg::CastTrait castTraitB322B64 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};

template <typename T, typename U>
__aicore__ inline void StoreOneElement(const __ubuf__ void* output, Reg::RegTensor<U>& src, Reg::MaskReg& preg,
                                       uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_FIRST_ELEMENT_B16>((__ubuf__ half*)(output) + offset, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B16>((__ubuf__ bfloat16_t*)(output) + offset,
                                                                            xBf16, preg);
    } else if constexpr (sizeof(T) == FOUR) {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)output) + offset,
                                                                       (Reg::RegTensor<float>&)src, preg);
    } else {
        Reg::UnalignRegForStore u0;
        auto dstAddr = (__ubuf__ T*)(output) + offset;
        Reg::StoreUnAlign(dstAddr, src, u0, 1);
        Reg::StoreUnAlignPost(dstAddr, u0, 0);
    }
}

template <typename T, typename U>
__aicore__ inline void LoadOneElement(const __ubuf__ void* input, Reg::RegTensor<U>& dst, Reg::MaskReg& preg,
                                      uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_BRC_B16>(xFp16, (__ubuf__ half*)(input) + offset);
        Reg::Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>(xBf16, (__ubuf__ bfloat16_t*)(input) + offset);
        Reg::Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else if constexpr (sizeof(T) == FOUR) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(dst, ((__ubuf__ T*)(input)) + offset);
    } else {
        Reg::UnalignRegForLoad u0;
        auto srcAddr = (__ubuf__ T*)(input) + offset;
        Reg::LoadUnAlignPre(u0, srcAddr);
        Reg::LoadUnAlign(dst, u0, srcAddr, 1);
    }
}

template <typename T>
__aicore__ inline void LoadOneTensor(const __ubuf__ void* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                     Reg::AddrReg& offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16, (__ubuf__ half*)(input), offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xBf16, (__ubuf__ bfloat16_t*)(input), offset);
        Reg::Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        Reg::LoadAlign(dst, (__ubuf__ float*)(input), offset);
    }
}

template <typename T, bool SPLITKW>
__aicore__ inline void CalcRealIndex(Reg::RegTensor<T>& resIndex, Reg::RegTensor<int32_t>& index, int64_t curKw,
                                     int64_t inputW, int64_t offset)
{
    Reg::MaskReg pregOneIndex = Reg::CreateMask<int32_t, Reg::MaskPattern::VL1>();

    Reg::RegTensor<T> indexCast;
    if constexpr (IsSameType<T, int64_t>::value) {
        Reg::Cast<int64_t, int32_t, castTraitB322B64>(indexCast, index, pregOneIndex);
    } else {
        Reg::Move(indexCast, index, pregOneIndex);
    }
    if constexpr (SPLITKW) {
        Reg::Adds(resIndex, indexCast, (T)offset, pregOneIndex);
    } else {
        Reg::RegTensor<T> wLen;
        Reg::RegTensor<T> v0;
        Reg::RegTensor<T> v1;
        Reg::Duplicate(wLen, (T)curKw, pregOneIndex);
        Reg::Div(v0, indexCast, wLen, pregOneIndex);
        Reg::Muls(resIndex, v0, inputW, pregOneIndex);
        Reg::Adds(resIndex, resIndex, (T)offset, pregOneIndex);
        Reg::Mul(wLen, wLen, v0, pregOneIndex);
        Reg::Sub(v0, indexCast, wLen, pregOneIndex);
        Reg::Add(resIndex, resIndex, v0, pregOneIndex);
    }
}

template <typename T>
__aicore__ inline void DuplicateNegInf(const __ubuf__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    Reg::RegTensor<T> v0;
    Reg::UnalignRegForStore u0;
    DuplicateNegInfReg<T>(v0);
    __ubuf__ T* addr = (__ubuf__ T*)dstAddr + offset;
    Reg::StoreUnAlign(addr, v0, u0, calNum);
    Reg::StoreUnAlignPost(addr, u0, 0);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename T>
__aicore__ inline void ReduceMaxWithIndex(Reg::RegTensor<T>& dst, Reg::RegTensor<int32_t>& dstIndex,
                                          Reg::RegTensor<T>& src, Reg::RegTensor<int32_t>& srcIndex,
                                          int32_t indexPadValue)
{
    // select first max value or last nan from one reg
    Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg notNanMaskReg;
    Reg::MaskReg nanMaskReg;
    Reg::RegTensor<T> vd1;
    Reg::RegTensor<T> vd2;
    Reg::RegTensor<int32_t> nanIndex;
    Reg::Duplicate(nanIndex, indexPadValue);
    Reg::Compare<T, CMPMODE::NE>(nanMaskReg, src, src, maskAll);                       // nan mask
    Reg::Not(notNanMaskReg, nanMaskReg, maskAll);                                      // not nan mask
    Reg::Select(nanIndex, srcIndex, nanIndex, nanMaskReg);                             // nan index
    Reg::Reduce<Reg::ReduceType::MAX>(nanIndex, nanIndex, maskAll);                    // max nan index
    Reg::Reduce<Reg::ReduceType::MAX>(vd1, src, notNanMaskReg);                        // max value
    Reg::Duplicate(vd2, vd1, maskAll);                                                 // max value
    Reg::Compare<T, CMPMODE::EQ>(notNanMaskReg, src, vd2, maskAll);                    // nan mask
    Reg::Reduce<Reg::ReduceType::MIN>(dstIndex, srcIndex, notNanMaskReg);              // not nan max index
    Reg::Compares<int32_t, CMPMODE::NE>(nanMaskReg, nanIndex, indexPadValue, maskAll); // nan
    Reg::Select(dstIndex, nanIndex, dstIndex, nanMaskReg);
    Reg::Duplicate(dstIndex, dstIndex, maskAll);
    Reg::Compare<int32_t, CMPMODE::EQ>(notNanMaskReg, dstIndex, srcIndex, maskAll);
    Reg::Reduce<Reg::ReduceType::MAX>(dst, src, notNanMaskReg); // max value
    // all value in the kernel is -inf
    Reg::Compares<int32_t, CMPMODE::EQ>(notNanMaskReg, dstIndex, indexPadValue, maskAll);
    Reg::Duplicate(nanIndex, static_cast<int32_t>(0));
    Reg::Select(dstIndex, nanIndex, dstIndex, notNanMaskReg);
}

template <typename T, typename U, typename TINDEX>
__aicore__ inline void MergeMaxRes(Reg::RegTensor<U>& res, Reg::RegTensor<TINDEX>& realResIndex,
                                   const __ubuf__ T* dstLocalAddr, const __ubuf__ TINDEX* indexLocalAddr,
                                   int32_t offset)
{
    // merge cur result with pre result
    Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg notNanMaskReg;
    Reg::MaskReg nanMaskReg;
    Reg::MaskReg pregOne = Reg::CreateMask<U, Reg::MaskPattern::VL1>();
    Reg::RegTensor<U> lastRes;
    Reg::RegTensor<TINDEX> lastResIndex;
    LoadOneElement<T, U>(dstLocalAddr, lastRes, pregOne, offset);
    Reg::Compare<U, CMPMODE::NE>(nanMaskReg, res, res, maskAll);        // cur nan
    Reg::Compare<U, CMPMODE::GT>(notNanMaskReg, res, lastRes, maskAll); // cur large > last
    Reg::Xor(notNanMaskReg, notNanMaskReg, nanMaskReg, maskAll);        // gt & nan
    Reg::Select(res, res, lastRes, notNanMaskReg);                      // nan index
    LoadOneElement<TINDEX, TINDEX>(indexLocalAddr, lastResIndex, pregOne, offset);
    Reg::Select(realResIndex, realResIndex, lastResIndex, notNanMaskReg); // nan index
    Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
}

template <typename T1, typename T2, typename TINDEX>
class MaxPoolWithArgmaxV3BigKernel {
public:
    __aicore__ inline MaxPoolWithArgmaxV3BigKernel(TPipe* pipe,
                                                   const MaxPoolWithArgmaxV3BigKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcKernelSize(int64_t curIdx, int64_t& curkH, int64_t& curkW, int64_t& curInOffset);
    template <bool SPLIT_KERNEL>
    __aicore__ inline void BaseCompute(int64_t beginIdx, int64_t endIdx, int64_t maxCount);
    __aicore__ inline void CopyInSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyInMultiRows(int64_t offset, int64_t blockLen, int64_t blockCount);
    __aicore__ inline void CopyMaxOut(int64_t curIdx);
    __aicore__ inline void NoSplitKernelProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset,
                                                int64_t maxCount);
    __aicore__ inline void SplitKernelProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset,
                                              int64_t maxCount);
    template <bool MERGE, bool SPLITKW>
    __aicore__ inline void ComputeSingle(int32_t localCurIdx, int64_t dataCount, int64_t offset, int64_t curKw);
    template <bool CLEAR>
    __aicore__ inline void InitOutLocal(int32_t localCurIdx);
    __aicore__ inline int64_t min(int64_t a, int64_t b) { return (a > b) ? b : a; }

    TPipe* pipe_;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    // 输出ub
    TBuf<> maxUBOutput_;
    TBuf<> indexUBOutput_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> maxGm_;
    GlobalTensor<TINDEX> indicesGm_;

    const MaxPoolWithArgmaxV3BigKernelTilingData* tilingData_;

    int64_t inHW_ = 1;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t beginIdx_ = 0;
    int64_t endIdx_ = 0;
};

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices)
{
    inHW_ = tilingData_->hInDim * tilingData_->wInDim;
    if (GetBlockIdx() < tilingData_->blockTail) {
        beginIdx_ = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx_ = beginIdx_ + tilingData_->blockFactor + 1;
    } else {
        beginIdx_ = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx_ = beginIdx_ + tilingData_->blockFactor;
    }
    // GM
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    maxGm_.SetGlobalBuffer((__gm__ T1*)y);
    indicesGm_.SetGlobalBuffer((__gm__ TINDEX*)indices);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->maxCount * sizeof(T1));
    pipe_->InitBuffer(maxUBOutput_, OUT_BUFFER_LEN * sizeof(T1));
    pipe_->InitBuffer(indexUBOutput_, OUT_BUFFER_LEN * sizeof(TINDEX));
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::Process()
{
    if (tilingData_->kH * tilingData_->kW <= tilingData_->maxCount) {
        BaseCompute<false>(beginIdx_, endIdx_, tilingData_->maxCount);
    } else {
        BaseCompute<true>(beginIdx_, endIdx_, tilingData_->maxCount);
    }
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::CalcKernelSize(int64_t curIdx, int64_t& curkH,
                                                                                    int64_t& curkW,
                                                                                    int64_t& curInOffset)
{
    if (tilingData_->isSigOut) {
        curInOffset = curIdx * inHW_;
        curOriginIndex_ = 0;
        curkH = min(tilingData_->kH - tilingData_->pH, tilingData_->hInDim);
        curkW = min(tilingData_->kW - tilingData_->pW, tilingData_->wInDim);
        return;
    }
    int64_t outHW = tilingData_->hOutDim * tilingData_->wOutDim;
    int64_t cur2D = curIdx % outHW;
    int64_t curNc = curIdx / outHW;
    int64_t curHo = cur2D / tilingData_->wOutDim;
    int64_t curWo = cur2D % tilingData_->wOutDim;

    curOriginH_ = tilingData_->sH * curHo - tilingData_->pH;
    if (curOriginH_ < 0) {
        curkH = min(tilingData_->kH + curOriginH_, tilingData_->hInDim);
        curOriginH_ = 0;
    } else {
        curkH = min(tilingData_->hInDim - curOriginH_, tilingData_->kH);
    }

    curOriginW_ = tilingData_->sW * curWo - tilingData_->pW;
    if (curOriginW_ < 0) {
        curkW = min(tilingData_->kW + curOriginW_, tilingData_->wInDim);
        curOriginW_ = 0;
    } else {
        curkW = min(tilingData_->wInDim - curOriginW_, tilingData_->kW);
    }

    curOriginIndex_ = curOriginH_ * tilingData_->wInDim + curOriginW_;
    curInOffset = curNc * inHW_ + curOriginIndex_;
}

template <typename T1, typename T2, typename TINDEX>
template <bool SPLIT_KERNEL>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::BaseCompute(int64_t beginIdx, int64_t endIdx,
                                                                                 int64_t maxCount)
{
    int64_t curkH = 1;
    int64_t curkW = 1;
    int64_t curInOffset = 0;
    // current blockdim range
    for (int64_t idx = beginIdx; idx < endIdx; idx++) {
        CalcKernelSize(idx, curkH, curkW, curInOffset);
        constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
        int32_t localCurIdx = (idx - beginIdx) % maxLocalLen;
        if constexpr (SPLIT_KERNEL) {
            InitOutLocal<true>(localCurIdx);
            SplitKernelProcess(localCurIdx, curkH, curkW, curInOffset, maxCount);
        } else {
            InitOutLocal<false>(localCurIdx);
            NoSplitKernelProcess(localCurIdx, curkH, curkW, curInOffset, maxCount);
        }
        CopyMaxOut(idx);
    }
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::CopyInSingleRow(int64_t offset, int64_t blockLen)
{
    LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();

    DataCopyPadExtParams<T1> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T1);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[offset], extParams, padExtParams);
    inputQue_.EnQue(xLocal);
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::CopyInMultiRows(int64_t offset, int64_t blockLen,
                                                                                     int64_t blockCount)
{
    LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();

    DataCopyPadExtParams<T1> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T1);
    extParams.srcStride = (tilingData_->wInDim - blockLen) * sizeof(T1);
    extParams.dstStride = 0;
    DataCopyPad<T1, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
    inputQue_.EnQue(xLocal);
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::CopyMaxOut(int64_t curIdx)
{
    constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
    int32_t localCurIdx = (curIdx - beginIdx_) % maxLocalLen;

    if (localCurIdx == maxLocalLen - 1 || curIdx == endIdx_ - 1) {
        LocalTensor<T1> maxOutLocal = maxUBOutput_.Get<T1>();
        LocalTensor<TINDEX> indexLocal = indexUBOutput_.Get<TINDEX>();

        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = (localCurIdx + 1) * sizeof(T1);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        DataCopyPad(maxGm_[curIdx - localCurIdx], maxOutLocal, extParams);
        extParams.blockLen = (localCurIdx + 1) * sizeof(TINDEX);
        DataCopyPad(indicesGm_[curIdx - localCurIdx], indexLocal, extParams);
    }
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::NoSplitKernelProcess(int32_t localCurIdx,
                                                                                          int64_t curkH, int64_t curkW,
                                                                                          int64_t curInOffset,
                                                                                          int64_t maxCount)
{
    CopyInMultiRows(curInOffset, curkW, curkH);
    ComputeSingle<false, false>(localCurIdx, curkW * curkH, curOriginIndex_, curkW);
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::SplitKernelProcess(int32_t localCurIdx,
                                                                                        int64_t curkH, int64_t curkW,
                                                                                        int64_t curInOffset,
                                                                                        int64_t maxCount)
{
    int64_t realIndex = 0;
    int64_t inputOffset = curInOffset;
    int64_t kernelOffset = curOriginIndex_;
    int64_t maxIndex = 0;

    if (curkW <= maxCount) {
        // 整行搬入
        int64_t hFactor = maxCount / curkW;
        int64_t hLoops = (curkH + hFactor - 1) / hFactor;
        int64_t hTail = curkH - (hLoops - 1) * hFactor;

        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
            CopyInMultiRows(inputOffset, curkW, curhFactor);
            ComputeSingle<true, false>(localCurIdx, curkW * curhFactor, kernelOffset, curkW);
            inputOffset += curhFactor * tilingData_->wInDim;
            kernelOffset += curhFactor * tilingData_->wInDim;
        }
    } else {
        // 单行很大，单行循环搬
        int64_t hLoops = curkH;
        int64_t wFactor = maxCount;
        int64_t wLoops = (curkW + wFactor - 1) / wFactor;
        int64_t wTail = curkW - (wLoops - 1) * wFactor;

        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            inputOffset = curInOffset + hLoop * tilingData_->wInDim;
            kernelOffset = curOriginIndex_ + hLoop * tilingData_->wInDim;
            for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
                int32_t curFactor = wLoop == wLoops - 1 ? wTail : wFactor;
                CopyInSingleRow(inputOffset, curFactor);
                ComputeSingle<true, true>(localCurIdx, curFactor, kernelOffset, curkW);
                inputOffset += curFactor;
                kernelOffset += curFactor;
            }
        }
    }
}

template <typename T1, typename T2, typename TINDEX>
template <bool CLEAR>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::InitOutLocal(int32_t localCurIdx)
{
    if (localCurIdx != 0) {
        return;
    }

    constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
    event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);

    if constexpr (!CLEAR) { // kerel 全载场景无需merge，因此无需初始化output
        return;
    }
    LocalTensor<T1> maxOutLocal = maxUBOutput_.Get<T1>();
    __ubuf__ T1* dstAddr = (__ubuf__ T1*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T1);
    uint16_t repeatTimes = CeilDivision(maxLocalLen, repeatElm);
    uint32_t num = maxLocalLen;
    __VEC_SCOPE__
    {
        Reg::RegTensor<T1> v0;
        DuplicateNegInfReg<T1>(v0);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            Reg::MaskReg p0 = Reg::UpdateMask<T1>(num);
            Reg::AddrReg offsetReg = Reg::CreateAddrReg<T1>(i, repeatElm);
            Reg::StoreAlign(dstAddr, v0, offsetReg, p0);
        }
    }
}

template <typename T1, typename T2, typename TINDEX>
template <bool MERGE, bool SPLITKW>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernel<T1, T2, TINDEX>::ComputeSingle(int32_t localCurIdx,
                                                                                   int64_t dataCount, int64_t offset,
                                                                                   int64_t curKw)
{
    LocalTensor<T1> maxOutLocal = maxUBOutput_.Get<T1>();
    LocalTensor<TINDEX> indexLocal = indexUBOutput_.Get<TINDEX>();
    LocalTensor<T1> xLocal = inputQue_.DeQue<T1>();
    union {
        T2 f;
        unsigned int i;
    } minValue;
    minValue.i = FLOAT32_NEG_INF;
    __ubuf__ T1* xLocalAddr = (__ubuf__ T1*)xLocal.GetPhyAddr();
    __ubuf__ T1* dstLocalAddr = (__ubuf__ T1*)maxOutLocal.GetPhyAddr();
    __ubuf__ TINDEX* indexLocalAddr = (__ubuf__ TINDEX*)indexLocal.GetPhyAddr();
    constexpr int32_t padIndex = -1;

    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T2);
    uint16_t repeatTimes = CeilDivision(dataCount, repeatElm);
    uint32_t num = repeatTimes * repeatElm; // 需要vreg_len对齐
    uint32_t padNum = num - dataCount;
    TINDEX inputW = tilingData_->wInDim;
    __VEC_SCOPE__
    {
        DuplicateNegInf<T1>(xLocalAddr, padNum, dataCount);
        Reg::RegTensor<T2> vd0;
        Reg::RegTensor<T2> vd1;
        Reg::RegTensor<T2> vd2;
        Reg::RegTensor<T2> vd3;
        Reg::RegTensor<T2> res;
        Reg::RegTensor<int32_t> resIndex;
        Reg::RegTensor<int32_t> index;
        Reg::MaskReg cmpMaskNanReg;
        Reg::MaskReg cmpMaskReg;
        Reg::MaskReg maskAll = Reg::CreateMask<T2, Reg::MaskPattern::ALL>();

        Reg::Duplicate(resIndex, padIndex);
        Reg::Duplicate(res, minValue.f);
        Reg::Arange(index, 0);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            Reg::MaskReg p0 = Reg::UpdateMask<T2>(num);
            Reg::AddrReg offset = Reg::CreateAddrReg<T1>(i, repeatElm);
            LoadOneTensor<T1>(xLocalAddr, vd0, p0, offset);
            Reg::Compare<T2, CMPMODE::NE>(cmpMaskNanReg, vd0, vd0, maskAll); // cur nan
            Reg::Compare<T2, CMPMODE::GT>(cmpMaskReg, vd0, res, maskAll);    // cur large > last
            Reg::Xor(cmpMaskReg, cmpMaskReg, cmpMaskNanReg, maskAll);        // gt & nan
            Reg::Select(res, vd0, res, cmpMaskReg);
            Reg::Select(resIndex, index, resIndex, cmpMaskReg);
            Reg::Adds(index, index, repeatElm, maskAll);
        }
        ReduceMaxWithIndex<T2>(res, index, res, resIndex, padIndex);
        Reg::MaskReg pregOne = Reg::CreateMask<T2, Reg::MaskPattern::VL1>();
        Reg::RegTensor<TINDEX> realResIndex;
        CalcRealIndex<TINDEX, SPLITKW>(realResIndex, index, curKw, inputW, offset);
        if constexpr (MERGE) {
            // merge cur result with last result
            MergeMaxRes<T1, T2, TINDEX>(res, realResIndex, dstLocalAddr, indexLocalAddr, localCurIdx);
        }
        StoreOneElement<TINDEX, TINDEX>(indexLocalAddr, realResIndex, pregOne, localCurIdx);
        StoreOneElement<T1, T2>(dstLocalAddr, res, pregOne, localCurIdx);
    }
    inputQue_.FreeTensor<T1>(xLocal);
}

} // namespace MaxPoolWithArgMaxV3BigKernel
#endif // MAX_POOL_WITH_ARGMAX_V3_BIG_KERNEL_H_
