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
 * \file max_pool3d_with_argmax_v2_big_kernel.h
 * \brief
 */
#ifndef MAX_POOL3D_WITH_ARGMAX_V2_BIG_KERNEL_REGBASE_H_
#define MAX_POOL3D_WITH_ARGMAX_V2_BIG_KERNEL_REGBASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "../inc/load_store_utils.h"
#include "max_pool3d_with_argmax_v2_tiling_struct.h"

namespace MaxPool3DWithArgmaxV2WithBigKernelRegbase {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000; // -inf 0xFF800000
constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
constexpr int32_t OUT_BUFFER_LEN = 1024;
constexpr int32_t EIGHT = 8;
constexpr int32_t FOUR = 4;
constexpr int32_t THREE = 3;
constexpr int32_t TWO = 2;
constexpr int32_t ONE = 1;
constexpr int32_t ZERO = 0;
constexpr int64_t BLOCK_DATA = 32;

constexpr Reg::CastTrait castB42B2 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                      RoundMode::CAST_RINT};

constexpr Reg::CastTrait castB22B4 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                      RoundMode::UNKNOWN};

constexpr Reg::CastTrait castB42B8 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                      RoundMode::UNKNOWN};

template <typename T, typename U>
__aicore__ inline void StoreOneNum(const __ubuf__ void* output, Reg::RegTensor<U>& src, Reg::MaskReg& preg,
                                   uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::StoreAlign<half, Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(((__ubuf__ half*)(output)) + offset,
                                                                      (Reg::RegTensor<half>&)src, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::Cast<bfloat16_t, float, castB42B2>(xBf16, src, preg);
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
__aicore__ inline void LoadOneNum(const __ubuf__ void* input, Reg::RegTensor<U>& dst, Reg::MaskReg& preg,
                                  uint32_t offset)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>(xBf16, (__ubuf__ bfloat16_t*)(input) + offset);
        Reg::Cast<float, bfloat16_t, castB22B4>(dst, xBf16, preg);
    } else if constexpr (IsSameType<T, half>::value) {
        Reg::LoadAlign<half, Reg::LoadDist::DIST_BRC_B16>(dst, (__ubuf__ half*)(input) + offset);
    } else if constexpr (sizeof(T) == FOUR) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(dst, ((__ubuf__ T*)(input)) + offset);
    } else {
        Reg::UnalignRegForLoad u0;
        auto srcAddr = (__ubuf__ T*)(input) + offset;
        Reg::LoadUnAlignPre(u0, srcAddr);
        Reg::LoadUnAlign(dst, u0, srcAddr, 1);
    }
}

template <typename T, typename U>
__aicore__ inline void LoadOneRegTensor(const __ubuf__ void* input, Reg::RegTensor<U>& dst, Reg::MaskReg& preg,
                                        int32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::LoadAlign<half>(dst, (__ubuf__ half*)(input) + offset);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xBf16, (__ubuf__ bfloat16_t*)(input) + offset);
        Reg::Cast<float, bfloat16_t, castB22B4>(dst, xBf16, preg);
    } else {
        Reg::LoadAlign(dst, (__ubuf__ float*)(input) + offset);
    }
}

template <typename T>
__aicore__ inline void SetAllNegInfReg(Reg::RegTensor<T>& negInfReg)
{
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr (std::is_same<T, half>::value) {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
    }
}

template <typename T>
__aicore__ inline void SetNegInfLocalMem(const __ubuf__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    Reg::RegTensor<T> v0;
    Reg::UnalignRegForStore u0;
    SetAllNegInfReg<T>(v0);
    __ubuf__ T* addr = (__ubuf__ T*)dstAddr + offset;
    Reg::StoreUnAlign(addr, v0, u0, calNum);
    Reg::StoreUnAlignPost(addr, u0, 0);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename T, typename MIDINDEX>
__aicore__ inline void GetMaxAndIndex(Reg::RegTensor<T>& dst, Reg::RegTensor<MIDINDEX>& dstIndex,
                                      Reg::RegTensor<T>& src, Reg::RegTensor<MIDINDEX>& srcIndex,
                                      MIDINDEX indexPadValue, Reg::MaskReg& maskAll, Reg::MaskReg& nanMaskReg,
                                      Reg::MaskReg& notNanMaskReg, Reg::RegTensor<T>& vd1)
{
    // select first max value or last nan from one reg
    Reg::RegTensor<T> vd2;
    Reg::RegTensor<MIDINDEX> nanIndex;
    Reg::Duplicate(nanIndex, indexPadValue);
    Reg::Compare<T, CMPMODE::NE>(nanMaskReg, src, src, maskAll);          // nan mask
    Reg::Not(notNanMaskReg, nanMaskReg, maskAll);                         // not nan mask
    Reg::Select(nanIndex, srcIndex, nanIndex, nanMaskReg);                // nan index
    Reg::Reduce<Reg::ReduceType::MAX>(nanIndex, nanIndex, maskAll);       // max nan index
    Reg::Reduce<Reg::ReduceType::MAX>(vd1, src, notNanMaskReg);           // max value
    Reg::Duplicate(vd2, vd1, maskAll);                                    // max value
    Reg::Compare<T, CMPMODE::EQ>(notNanMaskReg, src, vd2, maskAll);       // nan mask, all max value index
    Reg::Reduce<Reg::ReduceType::MIN>(dstIndex, srcIndex, notNanMaskReg); // not nan max index
    Reg::Compares<MIDINDEX, CMPMODE::NE>(nanMaskReg, nanIndex, indexPadValue, maskAll);
    Reg::Select(dstIndex, nanIndex, dstIndex, nanMaskReg);
    Reg::Duplicate(dstIndex, dstIndex, maskAll);
    Reg::Compare<MIDINDEX, CMPMODE::EQ>(notNanMaskReg, dstIndex, srcIndex, maskAll);
    Reg::Reduce<Reg::ReduceType::MAX>(dst, src, notNanMaskReg);
    // all value in the kernel is -inf
    Reg::Compares<MIDINDEX, CMPMODE::EQ>(notNanMaskReg, dstIndex, indexPadValue, maskAll);
    Reg::Duplicate(nanIndex, static_cast<MIDINDEX>(0));
    Reg::Select(dstIndex, nanIndex, dstIndex, notNanMaskReg);
}

template <typename T, typename U, typename TINDEX>
__aicore__ inline void MergeMaxAndIndex(Reg::RegTensor<U>& res, Reg::RegTensor<TINDEX>& realResIndex,
                                        const __ubuf__ T* dstLocalAddr, const __ubuf__ TINDEX* indexLocalAddr,
                                        int32_t offset, int32_t isPadValue, Reg::MaskReg& maskAll,
                                        Reg::MaskReg& nanMaskReg, Reg::MaskReg& notNanMaskReg, Reg::MaskReg& pregOne,
                                        Reg::RegTensor<U>& lastRes)
{
    // merge cur result with pre result
    Reg::RegTensor<TINDEX> lastResIndex;
    LoadOneNum<T, U>(dstLocalAddr, lastRes, pregOne, offset);
    Reg::Compare<U, CMPMODE::NE>(nanMaskReg, res, res, maskAll);        // cur nan
    Reg::Compare<U, CMPMODE::GT>(notNanMaskReg, res, lastRes, maskAll); // cur large > last
    Reg::Xor(notNanMaskReg, notNanMaskReg, nanMaskReg, maskAll);        // gt & nan
    Reg::Select(res, res, lastRes, notNanMaskReg);                      // nan index
    LoadOneNum<TINDEX, TINDEX>(indexLocalAddr, lastResIndex, pregOne, offset);
    Reg::Compares<TINDEX, CMPMODE::EQ>(nanMaskReg, lastResIndex, isPadValue, maskAll);
    Reg::Select(lastResIndex, realResIndex, lastResIndex, nanMaskReg);
    Reg::Select(realResIndex, realResIndex, lastResIndex, notNanMaskReg);
    Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
}

template <typename T1, typename TINDEX>
class MaxPool3DWithArgmaxV2BigKernelRegbase {
public:
    __aicore__ inline MaxPool3DWithArgmaxV2BigKernelRegbase(
        TPipe* pipe,
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcRealKernelSize(int64_t curIdx, int64_t& curkD, int64_t& curkH, int64_t& curkW,
                                              int64_t& curInOffset);
    // Multi-core-per-window (used only when tilingData_->multiCoreNum > 1)
    __aicore__ inline void MulCoreProcess();
    // Phase 1: compute this sub-core's local max/index over its W sub-range and publish to workspace.
    __aicore__ inline void MulCoreComputeLocal(int64_t idx, int64_t innerBlockIdx, int64_t multiCoreNum,
                                               int64_t cBlockIdx);
    // Phase 2: sub-core 0 reduces the multiCoreNum local results of one window and stores the final output.
    __aicore__ inline void MulCoreReduceAndStore(int64_t idx, int64_t multiCoreNum);
    __aicore__ inline bool IsNanScalar(T1 v);
    __aicore__ inline float ToCmpFloat(T1 v);
    template <bool SPLIT_KERNEL>
    __aicore__ inline void BaseCompute(int64_t beginIdx, int64_t endIdx, int64_t maxCount);
    template <typename T, typename MIDINDEX, bool SPLITKW, bool SPLITKHW, bool SPLITKDHW>
    __aicore__ inline void CalcRealIndex(Reg::RegTensor<T>& resIndex, Reg::RegTensor<MIDINDEX>& index, int64_t curkW,
                                         int64_t curkH, int64_t curkHW, int64_t inputH, int64_t inputW, int64_t offset);
    __aicore__ inline void CopyInLine(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyInPlane(int64_t offset, int64_t blockLen, int64_t blockCount);
    __aicore__ inline void CopyInCube(int64_t offset, int64_t blockLen, int64_t curkHWAlign, int64_t curkHW,
                                      int64_t blockCount, int64_t dFactor);
    __aicore__ inline void CopyOut(int64_t curIdx);
    __aicore__ inline void NoSplitKernelProcess(int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW,
                                                int64_t curInOffset, int64_t maxCount);
    __aicore__ inline void SplitKernelProcess(int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW,
                                              int64_t curInOffset, int64_t maxCount);
    template <bool MERGE, bool SPLITKW, bool SPLITKHW, bool SPLITKDHW>
    __aicore__ inline void ComputeSingleBlock(int32_t localCurIdx, int64_t dataCount, int64_t offset, int64_t curkW,
                                              int64_t curkH, int64_t curkHW);
    template <bool CLEAR>
    __aicore__ inline void InitOutBuffer(int32_t localCurIdx);
    __aicore__ inline int64_t min(int64_t a, int64_t b) { return (a > b) ? b : a; }

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<QuePosition::VECOUT> maxUBOutput_;
    TBuf<QuePosition::VECOUT> indexUBOutput_;
    TBuf<QuePosition::VECOUT> resMidBuf_;
    TBuf<QuePosition::VECOUT> indexMidBuf_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> maxGm_;
    GlobalTensor<TINDEX> indicesGm_;
    GlobalTensor<T1> maxValueWsGm_;  // per-sub-core local max value (multi-core-per-window)
    GlobalTensor<TINDEX> indexWsGm_; // per-sub-core local real linear index

    const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* tilingData_;

    uint32_t ubBlockSize = platform::GetUbBlockSize();
    int64_t ubAlignNum = ubBlockSize / sizeof(T1);
    int64_t kwAlign = ops::CeilAlign(tilingData_->kW, ubAlignNum);
    int64_t inDHW_ = 1;
    int64_t inHW_ = 1;
    int64_t outDHW_ = tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim;
    int64_t outHW_ = tilingData_->hOutDim * tilingData_->wOutDim;
    int64_t curOriginD_ = 0;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t beginIdx_ = 0;
    int64_t endIdx_ = 0;
};

template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices,
                                                                               GM_ADDR workspace)
{
    inHW_ = tilingData_->hInDim * tilingData_->wInDim;
    inDHW_ = tilingData_->dInDim * inHW_;
    if (tilingData_->multiCoreNum > 1) {
        beginIdx_ = 0;
        endIdx_ = 0;
    } else if (GetBlockIdx() < tilingData_->blockTail) {
        beginIdx_ = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx_ = beginIdx_ + tilingData_->blockFactor + 1;
    } else {
        beginIdx_ = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx_ = beginIdx_ + tilingData_->blockFactor;
    }
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    maxGm_.SetGlobalBuffer((__gm__ T1*)y);
    indicesGm_.SetGlobalBuffer((__gm__ TINDEX*)indices);

    if (tilingData_->multiCoreNum > 1) {
        // Workspace layout must match host GetWorkspaceSize: value region first (numSlots*sizeof(T1), 32B aligned),
        // then index region (numSlots*sizeof(TINDEX)). numSlots == coreNums * multiCoreNum == blockDim.
        int64_t numSlots = tilingData_->coreNums * tilingData_->multiCoreNum;
        int64_t valueBytes = ops::CeilAlign(numSlots * static_cast<int64_t>(sizeof(T1)), BLOCK_DATA);
        maxValueWsGm_.SetGlobalBuffer((__gm__ T1*)workspace);
        indexWsGm_.SetGlobalBuffer((__gm__ TINDEX*)(workspace + valueBytes));
    }

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->maxCount * sizeof(T1));
    pipe_->InitBuffer(maxUBOutput_, OUT_BUFFER_LEN * sizeof(T1));
    pipe_->InitBuffer(indexUBOutput_, OUT_BUFFER_LEN * sizeof(TINDEX));
    pipe_->InitBuffer(resMidBuf_, ubBlockSize);
    pipe_->InitBuffer(indexMidBuf_, ubBlockSize);
}

template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::Process()
{
    if (tilingData_->multiCoreNum > 1) {
        MulCoreProcess();
        return;
    }
    if (tilingData_->kD * tilingData_->kH * tilingData_->kW <= tilingData_->maxCount) {
        BaseCompute<false>(beginIdx_, endIdx_, tilingData_->maxCount);

    } else {
        BaseCompute<true>(beginIdx_, endIdx_, tilingData_->maxCount);
    }
}

template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::CalcRealKernelSize(
    int64_t curIdx, int64_t& curkD, int64_t& curkH, int64_t& curkW, int64_t& curInOffset)
{
    if (tilingData_->isSigOut) {
        curInOffset = curIdx * inDHW_;
        curOriginIndex_ = 0;
        curkD = min(tilingData_->kD - tilingData_->pD, tilingData_->dInDim);
        curkH = min(tilingData_->kH - tilingData_->pH, tilingData_->hInDim);
        curkW = min(tilingData_->kW - tilingData_->pW, tilingData_->wInDim);
        return;
    }
    int64_t curNc = curIdx / outDHW_;
    int64_t cur3D = curIdx - curNc * outDHW_;
    int64_t curDo = cur3D / outHW_;
    int64_t curHo = (cur3D - curDo * outHW_) / tilingData_->wOutDim;
    int64_t curWo = (cur3D - curDo * outHW_) - curHo * tilingData_->wOutDim;

    curOriginD_ = tilingData_->sD * curDo - tilingData_->pD;
    if (curOriginD_ < 0) {
        curkD = min(tilingData_->kD + curOriginD_, tilingData_->dInDim);
        curOriginD_ = 0;
    } else {
        curkD = min(tilingData_->dInDim - curOriginD_, tilingData_->kD);
    }
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
    curOriginIndex_ = curOriginD_ * inHW_ + curOriginH_ * tilingData_->wInDim + curOriginW_;
    curInOffset = curNc * inDHW_ + curOriginIndex_;
}

template <typename T1, typename TINDEX>
template <typename T, typename MIDINDEX, bool SPLITKW, bool SPLITKHW, bool SPLITKDHW>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::CalcRealIndex(Reg::RegTensor<T>& resIndex,
                                                                                        Reg::RegTensor<MIDINDEX>& index,
                                                                                        int64_t curkW, int64_t curkH,
                                                                                        int64_t curkHW, int64_t inputH,
                                                                                        int64_t inputW, int64_t offset)
{
    Reg::MaskReg pregOneIndex = Reg::CreateMask<MIDINDEX, Reg::MaskPattern::VL1>();
    Reg::MaskReg pregOneIndexB4 = Reg::CreateMask<int32_t, Reg::MaskPattern::VL1>();
    Reg::RegTensor<T> indexCast;
    Reg::RegTensor<int32_t> indexCast32;
    if constexpr (IsSameType<MIDINDEX, int16_t>::value && IsSameType<T, int32_t>::value) {
        Reg::Cast<int32_t, MIDINDEX, castB22B4>(indexCast, index, pregOneIndex);
    } else if constexpr (IsSameType<MIDINDEX, int16_t>::value && IsSameType<T, int64_t>::value) {
        Reg::Cast<int32_t, MIDINDEX, castB22B4>(indexCast32, index, pregOneIndex);
        Reg::Cast<int64_t, int32_t, castB42B8>(indexCast, indexCast32, pregOneIndexB4);

    } else if constexpr (IsSameType<MIDINDEX, int32_t>::value && IsSameType<T, int64_t>::value) {
        Reg::Cast<int64_t, MIDINDEX, castB42B8>(indexCast, index, pregOneIndex);
    } else {
        Reg::Move(indexCast, index, pregOneIndex);
    }
    if constexpr (SPLITKDHW) {
        Reg::RegTensor<T> tempHW;
        Reg::RegTensor<T> wLen;
        Reg::RegTensor<T> hReg;
        Reg::Duplicate(tempHW, (T)curkHW, pregOneIndex);
        Reg::Duplicate(wLen, (T)curkW, pregOneIndex);
        Reg::Duplicate(hReg, (T)inputH, pregOneIndex);
        // d = index / curkHW, remainder = index % curkHW
        Reg::RegTensor<T> d;
        Reg::Div(d, indexCast, tempHW, pregOneIndex);
        Reg::RegTensor<T> prod;
        Reg::Mul(prod, d, tempHW, pregOneIndex);
        Reg::RegTensor<T> remainder;
        Reg::Sub(remainder, indexCast, prod, pregOneIndex);
        // h = remainder / curkW
        Reg::RegTensor<T> h;
        Reg::Div(h, remainder, wLen, pregOneIndex);
        // independent Muls grouped together; remainder is already computed before prod is reused
        Reg::RegTensor<T> inputHW;
        Reg::RegTensor<T> hOffset;
        Reg::Muls(inputHW, hReg, inputW, pregOneIndex);
        Reg::Muls(prod, h, (T)curkW, pregOneIndex);
        Reg::Muls(hOffset, h, inputW, pregOneIndex);
        // w = remainder % curkW, dOffset = d * inputHW
        Reg::RegTensor<T> w;
        Reg::Sub(w, remainder, prod, pregOneIndex);
        Reg::RegTensor<T> dOffset;
        Reg::Mul(dOffset, d, inputHW, pregOneIndex);
        // resIndex = dOffset + offset + hOffset + w
        Reg::Adds(resIndex, dOffset, (T)offset, pregOneIndex);
        Reg::Add(resIndex, resIndex, hOffset, pregOneIndex);
        Reg::Add(resIndex, resIndex, w, pregOneIndex);
    }
    if constexpr (SPLITKHW) {
        Reg::RegTensor<T> wLen;
        Reg::RegTensor<T> v0;
        Reg::Duplicate(wLen, (T)curkW, pregOneIndex);
        Reg::Div(v0, indexCast, wLen, pregOneIndex);
        Reg::Muls(resIndex, v0, inputW, pregOneIndex);
        Reg::Adds(resIndex, resIndex, (T)offset, pregOneIndex);
        Reg::Mul(wLen, wLen, v0, pregOneIndex);
        Reg::Sub(v0, indexCast, wLen, pregOneIndex);
        Reg::Add(resIndex, resIndex, v0, pregOneIndex);
    }
    if constexpr (SPLITKW) {
        Reg::Adds(resIndex, indexCast, (T)offset, pregOneIndex);
    }
}

template <typename T1, typename TINDEX>
template <bool SPLIT_KERNEL>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::BaseCompute(int64_t beginIdx, int64_t endIdx,
                                                                                      int64_t maxCount)
{
    int64_t curkD = 1;
    int64_t curkH = 1;
    int64_t curkW = 1;
    int64_t curInOffset = 0;

    for (int64_t idx = beginIdx; idx < endIdx; idx++) {
        CalcRealKernelSize(idx, curkD, curkH, curkW, curInOffset);
        constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
        int32_t localCurIdx = (idx - beginIdx) % maxLocalLen;
        if constexpr (SPLIT_KERNEL) {
            InitOutBuffer<true>(localCurIdx);
            SplitKernelProcess(localCurIdx, curkD, curkH, curkW, curInOffset, maxCount);
        } else {
            InitOutBuffer<false>(localCurIdx);
            NoSplitKernelProcess(localCurIdx, curkD, curkH, curkW, curInOffset, maxCount);
        }
        CopyOut(idx);
    }
}

template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::CopyInLine(int64_t offset, int64_t blockLen)
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

template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::CopyInPlane(int64_t offset, int64_t blockLen,
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

template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::CopyInCube(int64_t offset, int64_t blockLen,
                                                                                     int64_t curkHWAlign,
                                                                                     int64_t curkHW, int64_t blockCount,
                                                                                     int64_t dFactor)
{
    LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();
    union {
        T1 f;
        uint32_t i;
    } minValue;
    if constexpr (IsSameType<T1, half>::value) {
        minValue.i = FLOAT16_NEG_INF;
    } else if constexpr (IsSameType<T1, float>::value) {
        minValue.i = FLOAT32_NEG_INF;
    } else {
        minValue.i = BFLOAT16_NEG_INF;
    }

    DataCopyPadExtParams<T1> padExtParams;
    padExtParams.isPad = true;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = curkHWAlign - curkHW;
    padExtParams.paddingValue = minValue.f;

    DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T1);
    extParams.srcStride = (tilingData_->wInDim - blockLen) * sizeof(T1);
    extParams.dstStride = 0;
    for (int64_t d = 0; d < dFactor; d++) {
        int64_t xLocalOffset = d * curkHWAlign;
        DataCopyPad<T1, PaddingMode::Compact>(xLocal[xLocalOffset], xGm_[offset + d * inHW_], extParams, padExtParams);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::CopyOut(int64_t curIdx)
{
    constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
    int32_t localCurIdx = (curIdx - beginIdx_) % maxLocalLen;

    if (localCurIdx == maxLocalLen - ONE || curIdx == endIdx_ - ONE) {
        LocalTensor<T1> maxOutLocal = maxUBOutput_.Get<T1>();
        LocalTensor<TINDEX> indexLocal = indexUBOutput_.Get<TINDEX>();

        DataCopyExtParams extParams;
        extParams.blockCount = ONE;
        extParams.blockLen = (localCurIdx + ONE) * sizeof(T1);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        DataCopyPad(maxGm_[curIdx - localCurIdx], maxOutLocal, extParams);
        extParams.blockLen = (localCurIdx + ONE) * sizeof(TINDEX);
        DataCopyPad(indicesGm_[curIdx - localCurIdx], indexLocal, extParams);
    }
}

template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::NoSplitKernelProcess(
    int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t maxCount)
{
    if (curkD * curkH * curkW == 0) {
        return;
    }
    int64_t inputOffset = curInOffset;
    int64_t kernelOffset = curOriginIndex_;
    int64_t curkHW = curkH * curkW;
    int64_t curkHWAlign = ops::CeilAlign(curkHW, ubAlignNum);

    CopyInCube(inputOffset, curkW, curkHWAlign, curkHW, curkH, curkD);
    ComputeSingleBlock<false, false, false, true>(localCurIdx, curkD * curkHWAlign, kernelOffset, curkW, curkH,
                                                  curkHWAlign);
}

template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::SplitKernelProcess(
    int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t maxCount)
{
    if (curkD * curkH * curkW == 0) {
        return;
    }
    int64_t realIndex = 0;
    int64_t inputOffset = curInOffset;
    int64_t kernelOffset = curOriginIndex_;
    int64_t maxIndex = 0;
    int64_t curkHW = curkH * curkW;
    int64_t curkHWAlign = ops::CeilAlign(curkHW, ubAlignNum);

    if (tilingData_->wInDim == curkW && tilingData_->hInDim == curkH) {
        int64_t curkDHW = curkHW * curkD;
        int64_t dhwLoops = (curkDHW + maxCount - 1) / maxCount;
        int32_t dhwFactor = maxCount;
        int32_t dhwTail = curkDHW % maxCount;
        if (dhwTail == 0) {
            dhwTail = dhwFactor;
        }
        for (int32_t dhwLoop = 0; dhwLoop < dhwLoops; dhwLoop++) {
            int32_t curFactor = dhwLoop == dhwLoops - 1 ? dhwTail : dhwFactor;
            CopyInLine(inputOffset, curFactor);
            ComputeSingleBlock<true, true, false, false>(localCurIdx, curFactor, kernelOffset, curkW, curkH, curkHW);
            inputOffset += curFactor;
            kernelOffset += curFactor;
        }
    } else if (curkH * curkW <= maxCount) {
        if (curkHWAlign <= maxCount) {
            int64_t dFactor = maxCount / curkHWAlign;
            int64_t dLoops = (curkD + dFactor - 1) / dFactor;
            int64_t dTail = curkD - (dLoops - 1) * dFactor;
            for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
                int64_t curdFactor = dLoop == dLoops - 1 ? dTail : dFactor;
                inputOffset = curInOffset + dLoop * dFactor * inHW_;
                kernelOffset = curOriginIndex_ + dLoop * dFactor * inHW_;
                CopyInCube(inputOffset, curkW, curkHWAlign, curkHW, curkH, curdFactor);
                ComputeSingleBlock<true, false, false, true>(localCurIdx, curdFactor * curkHWAlign, kernelOffset, curkW,
                                                             curkH, curkHWAlign);
            }
        } else {
            for (int64_t dLoop = 0; dLoop < curkD; dLoop++) {
                inputOffset = curInOffset + dLoop * inHW_;
                kernelOffset = curOriginIndex_ + dLoop * inHW_;
                CopyInPlane(inputOffset, curkW, curkH);
                ComputeSingleBlock<true, false, true, false>(localCurIdx, curkHW, kernelOffset, curkW, curkH, curkHW);
            }
        }
    } else if (curkW <= maxCount) {
        int64_t dLoops = curkD;
        int64_t hFactor = maxCount / curkW;
        int64_t hLoops = (curkH + hFactor - 1) / hFactor;
        int64_t hTail = curkH - (hLoops - 1) * hFactor;
        for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
            inputOffset = curInOffset + dLoop * inHW_;
            kernelOffset = curOriginIndex_ + dLoop * inHW_;
            for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
                int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
                CopyInPlane(inputOffset, curkW, curhFactor);
                ComputeSingleBlock<true, false, true, false>(localCurIdx, curkW * curhFactor, kernelOffset, curkW,
                                                             curkH, curkHW);
                inputOffset += hFactor * tilingData_->wInDim;
                kernelOffset += hFactor * tilingData_->wInDim;
            }
        }
    } else {
        int64_t dLoops = curkD;
        int64_t hLoops = curkH;
        int64_t wFactor = maxCount;
        int64_t wLoops = (curkW + wFactor - 1) / wFactor;
        int64_t wTail = curkW - (wLoops - 1) * wFactor;
        for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
            for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
                inputOffset = curInOffset + dLoop * inHW_ + hLoop * tilingData_->wInDim;
                kernelOffset = curOriginIndex_ + dLoop * inHW_ + hLoop * tilingData_->wInDim;
                for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
                    int32_t curFactor = wLoop == wLoops - 1 ? wTail : wFactor;
                    CopyInLine(inputOffset, curFactor);
                    ComputeSingleBlock<true, true, false, false>(localCurIdx, curFactor, kernelOffset, curkW, curkH,
                                                                 curkHW);
                    inputOffset += wFactor;
                    kernelOffset += wFactor;
                }
            }
        }
    }
}

template <typename T1, typename TINDEX>
template <bool CLEAR>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::InitOutBuffer(int32_t localCurIdx)
{
    if (localCurIdx != 0) {
        return;
    }
    constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
    event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    if constexpr (!CLEAR) {
        return;
    }
    LocalTensor<T1> maxOutLocal = maxUBOutput_.Get<T1>();
    __ubuf__ T1* dstAddr = (__ubuf__ T1*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T1);
    uint16_t repeatTimes = (maxLocalLen + static_cast<int64_t>(repeatElm) - 1) / static_cast<int64_t>(repeatElm);
    LocalTensor<TINDEX> indexLocal = indexUBOutput_.Get<TINDEX>();
    __ubuf__ TINDEX* indexAddr = (__ubuf__ TINDEX*)indexLocal.GetPhyAddr();
    constexpr uint32_t repeatIndexElm = platform::GetVRegSize() / sizeof(TINDEX);
    uint16_t repeatIndexTimes = (maxLocalLen + static_cast<int64_t>(repeatIndexElm) - 1) /
                                static_cast<int64_t>(repeatIndexElm);

    uint32_t maxNum = maxLocalLen;
    uint32_t indexNum = maxLocalLen;
    TINDEX defaultIndex = -1;
    __VEC_SCOPE__
    {
        Reg::RegTensor<T1> v0;
        SetAllNegInfReg<T1>(v0);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            Reg::MaskReg p0 = Reg::UpdateMask<T1>(maxNum);
            Reg::AddrReg offsetReg = Reg::CreateAddrReg<T1>(i, repeatElm);
            Reg::StoreAlign(dstAddr, v0, offsetReg, p0);
        }
        Reg::RegTensor<TINDEX> v1;
        Reg::Duplicate(v1, defaultIndex);
        for (uint16_t i = 0; i < repeatIndexTimes; i++) {
            Reg::MaskReg p0 = Reg::UpdateMask<TINDEX>(indexNum);
            Reg::AddrReg offsetReg = Reg::CreateAddrReg<TINDEX>(i, repeatIndexElm);
            Reg::StoreAlign(indexAddr, v1, offsetReg, p0);
        }
    }
}

template <typename T1, typename TINDEX>
template <bool MERGE, bool SPLITKW, bool SPLITKHW, bool SPLITKDHW>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::ComputeSingleBlock(
    int32_t localCurIdx, int64_t dataCount, int64_t offset, int64_t curkW, int64_t curkH, int64_t curkHW)
{
    using calculateType = std::conditional_t<std::is_same<T1, half>::value, half, float>;
    using calIndexType = std::conditional_t<std::is_same<T1, half>::value, int16_t, int32_t>;
    LocalTensor<T1> maxOutLocal = maxUBOutput_.Get<T1>();
    LocalTensor<TINDEX> indexLocal = indexUBOutput_.Get<TINDEX>();
    LocalTensor<T1> xLocal = inputQue_.DeQue<T1>();
    union {
        calculateType f;
        uint32_t i;
    } minValue;
    if constexpr (IsSameType<T1, half>::value) {
        minValue.i = FLOAT16_NEG_INF;
    } else {
        minValue.i = FLOAT32_NEG_INF;
    }
    __ubuf__ T1* xLocalAddr = (__ubuf__ T1*)xLocal.GetPhyAddr();
    __ubuf__ T1* dstLocalAddr = (__ubuf__ T1*)maxOutLocal.GetPhyAddr();
    __ubuf__ TINDEX* indexLocalAddr = (__ubuf__ TINDEX*)indexLocal.GetPhyAddr();
    constexpr calIndexType padIndex = -1;

    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(calculateType);
    uint16_t repeatTimes = (dataCount + repeatElm - 1) / repeatElm;
    uint32_t num = repeatTimes * repeatElm;
    uint32_t padNum = num - dataCount;
    TINDEX inputW = tilingData_->wInDim;
    TINDEX inputH = tilingData_->hInDim;
    __ubuf__ calculateType* resMidAddr = (__ubuf__ calculateType*)resMidBuf_.Get<calculateType>().GetPhyAddr();
    __ubuf__ calculateType* indexMidAddr = (__ubuf__ calculateType*)indexMidBuf_.Get<calculateType>().GetPhyAddr();
    __VEC_SCOPE__
    {
        SetNegInfLocalMem<T1>(xLocalAddr, padNum, dataCount);
        Reg::RegTensor<calculateType> val0;
        Reg::RegTensor<calculateType> val1;
        Reg::RegTensor<calculateType> res0;
        Reg::RegTensor<calculateType> res1;
        Reg::RegTensor<calIndexType> resIndex0;
        Reg::RegTensor<calIndexType> resIndex1;
        Reg::RegTensor<calIndexType> index0;
        Reg::RegTensor<calIndexType> index1;
        Reg::RegTensor<calIndexType> idxMin;
        Reg::RegTensor<calIndexType> idxMax;
        Reg::MaskReg nanReg0;
        Reg::MaskReg gtReg0;
        Reg::MaskReg nanReg1;
        Reg::MaskReg gtReg1;
        Reg::MaskReg maskAll = Reg::CreateMask<calculateType, Reg::MaskPattern::ALL>();

        Reg::Duplicate(resIndex0, padIndex);
        Reg::Duplicate(resIndex1, padIndex);
        Reg::Duplicate(res0, minValue.f);
        Reg::Duplicate(res1, minValue.f);
        Reg::Arange(index0, 0);
        Reg::Adds(index1, index0, repeatElm, maskAll); // stream 1 starts one VReg later

        uint16_t halfLoops = repeatTimes / 2;
        for (uint16_t i = 0; i < halfLoops; i++) {
            LoadOneRegTensor<T1, calculateType>(xLocalAddr, val0, maskAll, 2 * i * repeatElm);
            LoadOneRegTensor<T1, calculateType>(xLocalAddr, val1, maskAll, (2 * i + 1) * repeatElm);
            Reg::Compare<calculateType, CMPMODE::NE>(nanReg0, val0, val0, maskAll);
            Reg::Compare<calculateType, CMPMODE::NE>(nanReg1, val1, val1, maskAll);
            Reg::Compare<calculateType, CMPMODE::GT>(gtReg0, val0, res0, maskAll);
            Reg::Compare<calculateType, CMPMODE::GT>(gtReg1, val1, res1, maskAll);
            Reg::Xor(gtReg0, gtReg0, nanReg0, maskAll);
            Reg::Xor(gtReg1, gtReg1, nanReg1, maskAll);
            Reg::Select(res0, val0, res0, gtReg0);
            Reg::Select(res1, val1, res1, gtReg1);
            Reg::Select(resIndex0, index0, resIndex0, gtReg0);
            Reg::Select(resIndex1, index1, resIndex1, gtReg1);
            Reg::Adds(index0, index0, 2 * repeatElm, maskAll);
            Reg::Adds(index1, index1, 2 * repeatElm, maskAll);
        }
        // Odd tail: repeatTimes need not be even.
        uint16_t tailLoops = repeatTimes & 1;
        for (uint16_t t = 0; t < tailLoops; t++) {
            LoadOneRegTensor<T1, calculateType>(xLocalAddr, val0, maskAll, (repeatTimes - 1) * repeatElm);
            Reg::Compare<calculateType, CMPMODE::NE>(nanReg0, val0, val0, maskAll);
            Reg::Compare<calculateType, CMPMODE::GT>(gtReg0, val0, res0, maskAll);
            Reg::Xor(gtReg0, gtReg0, nanReg0, maskAll);
            Reg::Select(res0, val0, res0, gtReg0);
            Reg::Select(resIndex0, index0, resIndex0, gtReg0);
        }
        // Merge stream 1 into stream 0
        Reg::Compare<calIndexType, CMPMODE::GT>(gtReg0, resIndex1, resIndex0, maskAll); // i1 > i0
        Reg::Select(idxMin, resIndex0, resIndex1, gtReg0);
        Reg::Select(idxMax, resIndex1, resIndex0, gtReg0);
        Reg::Compare<calculateType, CMPMODE::NE>(nanReg0, res0, res0, maskAll); // res0 nan
        Reg::Compare<calculateType, CMPMODE::NE>(nanReg1, res1, res1, maskAll); // res1 nan
        Reg::Compare<calculateType, CMPMODE::GT>(gtReg1, res1, res0, maskAll);  // res1 > res0
        Reg::Compare<calculateType, CMPMODE::EQ>(gtReg0, res1, res0, maskAll);  // res1 == res0 (tie)
        Reg::Xor(gtReg1, gtReg1, nanReg1, maskAll);                             // choose res1 as winner
        Reg::And(nanReg1, nanReg0, nanReg1, maskAll);                           // both nan
        Reg::Select(resIndex0, resIndex1, resIndex0, gtReg1);                   // default: follow value winner
        Reg::Select(resIndex0, idxMin, resIndex0, gtReg0);                      // tie -> first index
        Reg::Select(resIndex0, idxMax, resIndex0, nanReg1);                     // both nan -> last index
        Reg::Select(res0, res1, res0, gtReg1);

        GetMaxAndIndex<calculateType, calIndexType>(res0, index0, res0, resIndex0, padIndex, maskAll, nanReg0, gtReg0,
                                                    val0);
        Reg::MaskReg pregOne = Reg::CreateMask<calculateType, Reg::MaskPattern::VL1>();
        StoreOneNum<calculateType, calculateType>(resMidAddr, res0, pregOne, 0);
        StoreOneNum<calculateType, calculateType>(indexMidAddr, (Reg::RegTensor<calculateType>&)index0, pregOne, 0);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }

    __VEC_SCOPE__
    {
        Reg::MaskReg pregOne = Reg::CreateMask<calculateType, Reg::MaskPattern::VL1>();
        Reg::RegTensor<calculateType> res;
        Reg::RegTensor<calIndexType> index;
        LoadOneNum<calculateType, calculateType>(resMidAddr, res, pregOne, 0);
        LoadOneNum<calculateType, calculateType>(indexMidAddr, (Reg::RegTensor<calculateType>&)index, pregOne, 0);

        Reg::RegTensor<TINDEX> realResIndex;
        CalcRealIndex<TINDEX, calIndexType, SPLITKW, SPLITKHW, SPLITKDHW>(realResIndex, index, curkW, curkH, curkHW,
                                                                          inputH, inputW, offset);
        if constexpr (MERGE) {
            Reg::MaskReg maskAll = Reg::CreateMask<calculateType, Reg::MaskPattern::ALL>();
            Reg::MaskReg cmpMaskNanReg;
            Reg::MaskReg cmpMaskReg;
            Reg::RegTensor<calculateType> vd0;
            MergeMaxAndIndex<T1, calculateType, TINDEX>(res, realResIndex, dstLocalAddr, indexLocalAddr, localCurIdx,
                                                        padIndex, maskAll, cmpMaskNanReg, cmpMaskReg, pregOne, vd0);
        }
        StoreOneNum<TINDEX, TINDEX>(indexLocalAddr, realResIndex, pregOne, localCurIdx);
        StoreOneNum<T1, calculateType>(dstLocalAddr, res, pregOne, localCurIdx);
    }
    inputQue_.FreeTensor<T1>(xLocal);
}

// Convert a stored value to float for scalar comparison in the cross-core merge.
template <typename T1, typename TINDEX>
__aicore__ inline float MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::ToCmpFloat(T1 v)
{
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        return ToFloat(v); // ToFloat supports bfloat16_t; half/float use direct conversion below.
    } else {
        return static_cast<float>(v);
    }
}

// Scalar NaN test: a value is NaN iff it does not equal itself.
template <typename T1, typename TINDEX>
__aicore__ inline bool MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::IsNanScalar(T1 v)
{
    float f = ToCmpFloat(v);
    return f != f;
}

// Multi-core-per-window
template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::MulCoreProcess()
{
    int64_t cBlockIdx = GetBlockIdx();
    int64_t multiCoreNum = tilingData_->multiCoreNum;
    if (cBlockIdx >= tilingData_->coreNums * multiCoreNum) {
        return;
    }
    int64_t idx = cBlockIdx / multiCoreNum;
    int64_t innerBlockIdx = cBlockIdx % multiCoreNum;

    MulCoreComputeLocal(idx, innerBlockIdx, multiCoreNum, cBlockIdx);

    SyncAll();

    // Phase 2: only sub-core 0 of each window reduces the multiCoreNum local results.
    if (innerBlockIdx != 0) {
        return;
    }
    MulCoreReduceAndStore(idx, multiCoreNum);
}

// Phase 1: clip the window to input bounds for this output index, compute this sub-core's local
// max/index over its W sub-range, and publish the result to workspace slot cBlockIdx.
template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::MulCoreComputeLocal(int64_t idx,
                                                                                              int64_t innerBlockIdx,
                                                                                              int64_t multiCoreNum,
                                                                                              int64_t cBlockIdx)
{
    int64_t curkD = 1;
    int64_t curkH = 1;
    int64_t curkW = 1;
    int64_t curInOffset = 0;
    CalcRealKernelSize(idx, curkD, curkH, curkW, curInOffset);

    int64_t wSplitSize = ops::CeilDiv(curkW, multiCoreNum);
    int64_t wStart = innerBlockIdx * wSplitSize;
    int64_t subKw = 0;
    if (wStart < curkW) {
        subKw = min(wSplitSize, curkW - wStart);
    }

    constexpr int32_t localCurIdx = 0;
    InitOutBuffer<true>(localCurIdx);
    if (subKw > 0) {
        int64_t subInOffset = curInOffset + wStart;
        curOriginIndex_ += wStart;
        SplitKernelProcess(localCurIdx, curkD, curkH, subKw, subInOffset, tilingData_->maxCount);
        curOriginIndex_ -= wStart;
    }

    // Publish local result (value + real linear index) to workspace slot cBlockIdx.
    LocalTensor<T1> maxOutLocal = maxUBOutput_.Get<T1>();
    LocalTensor<TINDEX> indexLocal = indexUBOutput_.Get<TINDEX>();
    event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    DataCopyExtParams vParams;
    vParams.blockCount = ONE;
    vParams.blockLen = sizeof(T1);
    vParams.srcStride = 0;
    vParams.dstStride = 0;
    DataCopyPad(maxValueWsGm_[cBlockIdx], maxOutLocal, vParams);
    DataCopyExtParams iParams;
    iParams.blockCount = ONE;
    iParams.blockLen = sizeof(TINDEX);
    iParams.srcStride = 0;
    iParams.dstStride = 0;
    DataCopyPad(indexWsGm_[cBlockIdx], indexLocal, iParams);
}

// Phase 2: sub-core 0 reduces the multiCoreNum per-sub-core local results of one window and stores
// the final max value and real linear index to the output GM.
template <typename T1, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, TINDEX>::MulCoreReduceAndStore(int64_t idx,
                                                                                                int64_t multiCoreNum)
{
    LocalTensor<T1> maxOutLocal = maxUBOutput_.Get<T1>();
    LocalTensor<TINDEX> indexLocal = indexUBOutput_.Get<TINDEX>();

    int64_t startSlot = idx * multiCoreNum;
    T1 bestVal = maxOutLocal.GetValue(0);
    TINDEX bestIdx = indexLocal.GetValue(0);
    bool bestIsNan = IsNanScalar(bestVal);
    float bestCmp = ToCmpFloat(bestVal);
    for (int64_t s = 1; s < multiCoreNum; s++) {
        T1 v = maxValueWsGm_.GetValue(startSlot + s);
        TINDEX ridx = indexWsGm_.GetValue(startSlot + s);
        bool vIsNan = IsNanScalar(v);
        if (bestIsNan) {
            // Already NaN: only a NaN with a larger real index replaces it (last-NaN semantics).
            if (vIsNan && ridx > bestIdx) {
                bestIdx = ridx;
            }
        } else if (vIsNan) {
            // NaN outranks any real value (NaN treated as max).
            bestVal = v;
            bestIdx = ridx;
            bestIsNan = true;
            bestCmp = ToCmpFloat(v);
        } else {
            float vCmp = ToCmpFloat(v);
            // first-max: strictly greater wins; on tie, the smaller real linear index wins.
            if (vCmp > bestCmp || (vCmp == bestCmp && ridx < bestIdx)) {
                bestVal = v;
                bestIdx = ridx;
                bestCmp = vCmp;
            }
        }
    }

    event_t eventIdMTE3toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMTE3toS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMTE3toS);
    maxOutLocal.SetValue(0, bestVal);
    indexLocal.SetValue(0, bestIdx);
    event_t eventIdStoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdStoMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdStoMTE3);
    DataCopyExtParams outParams;
    outParams.blockCount = ONE;
    outParams.blockLen = sizeof(T1);
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(maxGm_[idx], maxOutLocal, outParams);
    outParams.blockLen = sizeof(TINDEX);
    DataCopyPad(indicesGm_[idx], indexLocal, outParams);
}

} // namespace MaxPool3DWithArgmaxV2WithBigKernelRegbase
#endif // MAX_POOL_WITH_ARGMAX_V2_BIG_KERNEL_H_
