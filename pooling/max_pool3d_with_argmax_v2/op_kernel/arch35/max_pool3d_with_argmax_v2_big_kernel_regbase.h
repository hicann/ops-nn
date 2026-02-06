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

namespace MaxPool3DWithArgmaxV2WithBigKernelRegbase
{
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;  // -inf 0xFF800000
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

constexpr MicroAPI::CastTrait castB42B2 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr MicroAPI::CastTrait castB22B4 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr MicroAPI::CastTrait castB42B8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename T, typename U>
__aicore__ inline void StoreOneNum(const __local_mem__ void* output, MicroAPI::RegTensor<U>& src,
                                       MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        MicroAPI::Cast<half, float, castB42B2>(xFp16, src, preg);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>((__local_mem__ half*)(output) + offset,
                                                                              xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::Cast<bfloat16_t, float, castB42B2>(xBf16, src, preg);
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(
            (__local_mem__ bfloat16_t*)(output) + offset, xBf16, preg);
    } else if constexpr (sizeof(T) == FOUR) {
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)output) + offset,
                                                                               (MicroAPI::RegTensor<float>&)src, preg);
    } else {
        MicroAPI::UnalignReg u0;
        auto dstAddr = (__local_mem__ T*)(output) + offset;
        MicroAPI::DataCopyUnAlign(dstAddr, src, u0, 1);
        MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
    }
}

template <typename T, typename U>
__aicore__ inline void LoadOneNum(const __local_mem__ void* input, MicroAPI::RegTensor<U>& dst,
                                      MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(xBf16,
                                                                         (__local_mem__ bfloat16_t*)(input) + offset);
        MicroAPI::Cast<float, bfloat16_t, castB22B4>(dst, xBf16, preg);
    } else if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(xFp16, (__local_mem__ half*)(input) + offset);
        MicroAPI::Cast<float, half, castB22B4>(dst, xFp16, preg);
    } else if constexpr (sizeof(T) == FOUR) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(dst, ((__local_mem__ T*)(input)) + offset);
    } else {
        MicroAPI::UnalignReg u0;
        auto srcAddr = (__local_mem__ T*)(input) + offset;
        MicroAPI::DataCopyUnAlignPre(u0, srcAddr);
        MicroAPI::DataCopyUnAlign(dst, u0, srcAddr, 1);
    }
}

template <typename T>
__aicore__ inline void LoadOneRegTensor(const __local_mem__ void* input, MicroAPI::RegTensor<float>& dst,
                                     MicroAPI::MaskReg& preg, MicroAPI::AddrReg& offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, (__local_mem__ half*)(input), offset);
        Cast<float, half, castB22B4>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xBf16, (__local_mem__ bfloat16_t*)(input),
                                                                            offset);
        MicroAPI::Cast<float, bfloat16_t, castB22B4>(dst, xBf16, preg);
    } else {
        MicroAPI::DataCopy(dst, (__local_mem__ float*)(input), offset);
    }
}

template <typename T>
__aicore__ inline void SetAllNegInfReg(MicroAPI::RegTensor<T>& negInfReg)
{
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr(std::is_same<T, float>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr(std::is_same<T, half>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        MicroAPI::Duplicate((MicroAPI::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
    }
}

template <typename T>
__aicore__ inline void SetNegInfLocalMem(const __local_mem__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    MicroAPI::RegTensor<T> v0;
    MicroAPI::UnalignReg u0;
    SetAllNegInfReg<T>(v0);
    __local_mem__ T* addr = (__local_mem__ T*)dstAddr + offset;
    MicroAPI::DataCopyUnAlign(addr, v0, u0, calNum);
    MicroAPI::DataCopyUnAlignPost(addr, u0, 0);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename T>
__aicore__ inline void GetMaxAndIndex(MicroAPI::RegTensor<T>& dst, MicroAPI::RegTensor<int32_t>& dstIndex,
                                          MicroAPI::RegTensor<T>& src, MicroAPI::RegTensor<int32_t>& srcIndex,
                                          int32_t indexPadValue)
{
    // select first max value or last nan from one reg
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg notNanMaskReg;
    MicroAPI::MaskReg nanMaskReg;
    MicroAPI::RegTensor<T> vd1;
    MicroAPI::RegTensor<T> vd2;
    MicroAPI::RegTensor<int32_t> nanIndex;
    MicroAPI::Duplicate(nanIndex, indexPadValue);
    MicroAPI::Compare<T, CMPMODE::NE>(nanMaskReg, src, src, maskAll);                             // nan mask
    MicroAPI::MaskNot(notNanMaskReg, nanMaskReg, maskAll);                                        // not nan mask
    MicroAPI::Select(nanIndex, srcIndex, nanIndex, nanMaskReg);                                   // nan index
    MicroAPI::ReduceMax(nanIndex, nanIndex, maskAll);                                             // max nan index wsunmoon
    MicroAPI::ReduceMax(vd1, src, notNanMaskReg);                                                 // max value
    MicroAPI::Duplicate(vd2, vd1, maskAll);                                                       // max value
    MicroAPI::Compare<T, CMPMODE::EQ>(notNanMaskReg, src, vd2, maskAll);                          // nan mask, all max value index
    MicroAPI::ReduceMin(dstIndex, srcIndex, notNanMaskReg);                                       // not nan max index
    MicroAPI::CompareScalar<int32_t, CMPMODE::NE>(nanMaskReg, nanIndex, indexPadValue, maskAll);
    MicroAPI::Select(dstIndex, nanIndex, dstIndex, nanMaskReg);
    MicroAPI::Duplicate(dstIndex, dstIndex, maskAll);
    MicroAPI::Compare<int32_t, CMPMODE::EQ>(notNanMaskReg, dstIndex, srcIndex, maskAll);
    MicroAPI::ReduceMax(dst, src, notNanMaskReg);
    // all value in the kernel is -inf
    MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(notNanMaskReg, dstIndex, indexPadValue, maskAll);
    MicroAPI::Duplicate(nanIndex, static_cast<int32_t>(0));
    MicroAPI::Select(dstIndex, nanIndex, dstIndex, notNanMaskReg);
}

template <typename T, typename U, typename TINDEX>
__aicore__ inline void MergeMaxAndIndex(MicroAPI::RegTensor<U>& res, MicroAPI::RegTensor<TINDEX>& realResIndex,
                                   const __local_mem__ T* dstLocalAddr, const __local_mem__ TINDEX* indexLocalAddr,
                                   int32_t offset, int32_t isPadValue)
{
    // merge cur result with pre result
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg notNanMaskReg;
    MicroAPI::MaskReg nanMaskReg;
    MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<U> lastRes;
    MicroAPI::RegTensor<TINDEX> lastResIndex;
    LoadOneNum<T, U>(dstLocalAddr, lastRes, pregOne, offset);
    MicroAPI::Compare<U, CMPMODE::NE>(nanMaskReg, res, res, maskAll);         // cur nan
    MicroAPI::Compare<U, CMPMODE::GT>(notNanMaskReg, res, lastRes, maskAll);  // cur large > last
    MicroAPI::MaskXor(notNanMaskReg, notNanMaskReg, nanMaskReg, maskAll);     // gt & nan
    MicroAPI::Select(res, res, lastRes, notNanMaskReg);                       // nan index
    LoadOneNum<TINDEX, TINDEX>(indexLocalAddr, lastResIndex, pregOne, offset);
    MicroAPI::CompareScalar<TINDEX, CMPMODE::EQ>(nanMaskReg, lastResIndex, isPadValue, maskAll);
    MicroAPI::Select(lastResIndex, realResIndex, lastResIndex, nanMaskReg);
    MicroAPI::Select(realResIndex, realResIndex, lastResIndex, notNanMaskReg); 
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

template <typename T1, typename T2, typename TINDEX>
class MaxPool3DWithArgmaxV2BigKernelRegbase
{
public:
    __aicore__ inline MaxPool3DWithArgmaxV2BigKernelRegbase(TPipe* pipe,
                                                   const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcRealKernelSize(int64_t curIdx, int64_t& curkD, int64_t& curkH, int64_t& curkW, int64_t& curInOffset);
    template <bool SPLIT_KERNEL>
    __aicore__ inline void BaseCompute(int64_t beginIdx, int64_t endIdx, int64_t maxCount);
    template <typename T, bool SPLITKW, bool SPLITKHW, bool SPLITKDHW>
    __aicore__ inline void CalcRealIndex(MicroAPI::RegTensor<T>& resIndex, MicroAPI::RegTensor<int32_t>& index,
                                     int64_t curkW, int64_t curkH, int64_t inputH, int64_t inputW, int64_t offset);
    __aicore__ inline void CopyInCube(int64_t offset, int64_t blockLen, int64_t blockCount, int64_t loopCount);
    __aicore__ inline void CopyOut(int64_t curIdx);
    __aicore__ inline void NoSplitKernelProcess(int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW, int64_t curInOffset,
                                                int64_t maxCount);
    __aicore__ inline void SplitKernelProcess(int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW, int64_t curInOffset,
                                              int64_t maxCount);
    template <bool MERGE, bool SPLITKW, bool SPLITKHW, bool SPLITKDHW>
    __aicore__ inline void ComputeSingleBlock(int32_t localCurIdx, int64_t dataCount, int64_t offset, int64_t curkW, int64_t curkH);
    template <bool CLEAR>
    __aicore__ inline void InitOutBuffer(int32_t localCurIdx);
    __aicore__ inline int64_t min(int64_t a, int64_t b)
    {
        return (a > b) ? b : a;
    }

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<QuePosition::VECOUT> maxUBOutput_;
    TBuf<QuePosition::VECOUT> indexUBOutput_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> maxGm_;
    GlobalTensor<TINDEX> indicesGm_;

    const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* tilingData_;

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

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices)
{
    inDHW_ = tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim;
    inHW_ = tilingData_->hInDim * tilingData_->wInDim;
    if (GetBlockIdx() < tilingData_->blockTail) {
        beginIdx_ = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx_ = beginIdx_ + tilingData_->blockFactor + 1;
    } else {
        beginIdx_ = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx_ = beginIdx_ + tilingData_->blockFactor;
    }
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    maxGm_.SetGlobalBuffer((__gm__ T1*)y);
    indicesGm_.SetGlobalBuffer((__gm__ TINDEX*)indices);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->maxCount * sizeof(T1));
    pipe_->InitBuffer(maxUBOutput_, OUT_BUFFER_LEN * sizeof(T1));
    pipe_->InitBuffer(indexUBOutput_, OUT_BUFFER_LEN * sizeof(TINDEX));
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::Process()
{
    if (tilingData_->kD * tilingData_->kH * tilingData_->kW <= tilingData_->maxCount) {
        BaseCompute<false>(beginIdx_, endIdx_, tilingData_->maxCount);
        
    } else {
        BaseCompute<true>(beginIdx_, endIdx_, tilingData_->maxCount);
    }
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::CalcRealKernelSize(int64_t curIdx, int64_t& curkD, int64_t& curkH,
                                                                                    int64_t& curkW,
                                                                                    int64_t& curInOffset)
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

template <typename T1, typename T2, typename TINDEX>
template <typename T, bool SPLITKW, bool SPLITKHW, bool SPLITKDHW>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::CalcRealIndex(MicroAPI::RegTensor<T>& resIndex, MicroAPI::RegTensor<int32_t>& index,
                                     int64_t curkW, int64_t curkH, int64_t inputH, int64_t inputW, int64_t offset)
{
    MicroAPI::MaskReg pregOneIndex = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<T> indexCast;
    if constexpr (IsSameType<T, int64_t>::value) {
        MicroAPI::Cast<int64_t, int32_t, castB42B8>(indexCast, index, pregOneIndex);
    } else {
        MicroAPI::Copy(indexCast, index, pregOneIndex);
    }
    if constexpr (SPLITKDHW) {
        MicroAPI::RegTensor<T> hLen;
        MicroAPI::RegTensor<T> tempHW;
        MicroAPI::Duplicate(hLen, (T)curkH, pregOneIndex);
        MicroAPI::Muls(tempHW, hLen, curkW, pregOneIndex);
        MicroAPI::RegTensor<T> d;
        MicroAPI::Div(d, indexCast, tempHW, pregOneIndex);
        MicroAPI::RegTensor<T> prod;
        MicroAPI::Mul(prod, d, tempHW, pregOneIndex);
        MicroAPI::RegTensor<T> remainder;
        MicroAPI::Sub(remainder, indexCast, prod, pregOneIndex);
        MicroAPI::RegTensor<T> h;
        MicroAPI::RegTensor<T> wLen;
        MicroAPI::Duplicate(wLen, (T)curkW, pregOneIndex);
        MicroAPI::Div(h, remainder, wLen, pregOneIndex);
        MicroAPI::Muls(prod, h, (T)curkW, pregOneIndex);
        MicroAPI::RegTensor<T> w;
        MicroAPI::Sub(w, remainder, prod, pregOneIndex);
        MicroAPI::RegTensor<T> inputHW;
        MicroAPI::RegTensor<T> hReg;
        MicroAPI::Duplicate(hReg, (T)inputH, pregOneIndex);
        MicroAPI::Muls(inputHW, hReg, inputW, pregOneIndex);
        MicroAPI::RegTensor<T> dOffset;
        MicroAPI::Mul(dOffset, d, inputHW, pregOneIndex);
        MicroAPI::RegTensor<T> hOffset;
        MicroAPI::Muls(hOffset, h, inputW, pregOneIndex);
        MicroAPI::Adds(resIndex, dOffset, (T)offset, pregOneIndex);
        MicroAPI::Add(resIndex, resIndex, hOffset, pregOneIndex);
        MicroAPI::Add(resIndex, resIndex, w, pregOneIndex);
    }
    if constexpr (SPLITKHW) {
        MicroAPI::RegTensor<T> wLen;
        MicroAPI::RegTensor<T> v0;
        MicroAPI::RegTensor<T> v1;
        MicroAPI::Duplicate(wLen, (T)curkW, pregOneIndex);
        MicroAPI::Div(v0, indexCast, wLen, pregOneIndex);  
        MicroAPI::Muls(resIndex, v0, inputW, pregOneIndex);
        MicroAPI::Adds(resIndex, resIndex, (T)offset, pregOneIndex);
        MicroAPI::Mul(wLen, wLen, v0, pregOneIndex);
        MicroAPI::Sub(v0, indexCast, wLen, pregOneIndex);
        MicroAPI::Add(resIndex, resIndex, v0, pregOneIndex);
    }
    if constexpr (SPLITKW) {
        MicroAPI::Adds(resIndex, indexCast, (T)offset, pregOneIndex);
    }
}

template <typename T1, typename T2, typename TINDEX>
template <bool SPLIT_KERNEL>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::BaseCompute(int64_t beginIdx, int64_t endIdx,
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

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::CopyInCube(int64_t offset, int64_t blockLen,
                                                                          int64_t blockCount, int64_t loopCount)
{
    LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();
    MultiCopyLoopInfo<THREE> loopInfo;
    loopInfo.loopSize[ZERO] = blockLen;
    loopInfo.loopSize[ONE] = blockCount;
    loopInfo.loopSize[TWO] = loopCount;

    loopInfo.loopSrcStride[ZERO] = ONE;
    loopInfo.loopSrcStride[ONE] = tilingData_->wInDim;
    loopInfo.loopSrcStride[TWO] = tilingData_->hInDim * tilingData_->wInDim;

    loopInfo.loopDstStride[ZERO] = ONE;
    loopInfo.loopDstStride[ONE] = blockLen;
    loopInfo.loopDstStride[TWO] = blockLen * blockCount;

    static constexpr MultiCopyConfig config = {false};
    MultiCopyParams<T1, THREE> paramsMain = {loopInfo};
    DataCopy<T1, THREE, config>(xLocal, xGm_[offset], paramsMain);
    inputQue_.EnQue(xLocal);
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::CopyOut(int64_t curIdx)
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

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::NoSplitKernelProcess(int32_t localCurIdx,
                                                                                          int64_t curkD, int64_t curkH, int64_t curkW,
                                                                                          int64_t curInOffset,
                                                                                          int64_t maxCount)
{
    if (curkD * curkH * curkW == 0) {
        return;
    }
    CopyInCube(curInOffset, curkW, curkH, curkD);
    ComputeSingleBlock<false, false, false, true>(localCurIdx, curkD * curkH * curkW, curOriginIndex_, curkW, curkH);
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::SplitKernelProcess(int32_t localCurIdx, int64_t curkD,
                                                                                        int64_t curkH, int64_t curkW,
                                                                                        int64_t curInOffset,
                                                                                        int64_t maxCount)
{
    if (curkD * curkH * curkW == 0) {
        return;
    }
    int64_t realIndex = 0;
    int64_t inputOffset = curInOffset;
    int64_t kernelOffset = curOriginIndex_;
    int64_t maxIndex = 0;

    if (curkH * curkW <= maxCount) {
        int64_t dFactor = maxCount / (curkH * curkW);
        int64_t dLoops = (curkD + dFactor - 1) / dFactor;
        int64_t dTail = curkD - (dLoops - 1) * dFactor;
        for(int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
            int64_t curdFactor = dLoop == dLoops - 1 ? dTail : dFactor;
            inputOffset = curInOffset + dLoop * dFactor * tilingData_->hInDim * tilingData_->wInDim;
            kernelOffset = curOriginIndex_ + dLoop * dFactor * tilingData_->hInDim * tilingData_->wInDim;
            CopyInCube(inputOffset, curkW, curkH, curdFactor);
            ComputeSingleBlock<true, false, false, true>(localCurIdx, curdFactor * curkH * curkW, kernelOffset, curkW, curkH);           
        }
    } else if(curkW <= maxCount) {
        int64_t dLoops = curkD;
        int64_t hFactor = maxCount / curkW;
        int64_t hLoops = (curkH + hFactor - 1) / hFactor;
        int64_t hTail = curkH  - (hLoops - 1) * hFactor;
        for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
            inputOffset = curInOffset + dLoop * tilingData_->hInDim * tilingData_->wInDim;
            kernelOffset = curOriginIndex_ + dLoop * tilingData_->hInDim * tilingData_->wInDim;
            for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
                int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
                CopyInCube(inputOffset, curkW, curhFactor, 1);
                ComputeSingleBlock<true, false, true, false>(localCurIdx, curkW * curhFactor, kernelOffset, curkW, curkH);
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
            for(int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
                inputOffset = curInOffset + dLoop * tilingData_->hInDim * tilingData_->wInDim + hLoop * tilingData_->wInDim;
                kernelOffset = curOriginIndex_ + dLoop * tilingData_->hInDim * tilingData_->wInDim + hLoop * tilingData_->wInDim;
                for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
                    int32_t curFactor = wLoop == wLoops - 1 ? wTail : wFactor;
                    CopyInCube(inputOffset, curFactor, 1, 1);
                    ComputeSingleBlock<true, true, false, false>(localCurIdx, curFactor, kernelOffset, curkW, curkH);
                    inputOffset += wFactor;
                    kernelOffset += wFactor;
                }
            }
        }
    }
}

template <typename T1, typename T2, typename TINDEX>
template <bool CLEAR>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::InitOutBuffer(int32_t localCurIdx)
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
    __local_mem__ T1* dstAddr = (__local_mem__ T1*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T1);
    uint16_t repeatTimes = (maxLocalLen + static_cast<int64_t>(repeatElm) - 1) / static_cast<int64_t>(repeatElm);
    LocalTensor<TINDEX> indexLocal = indexUBOutput_.Get<TINDEX>();
    __local_mem__ TINDEX* indexAddr = (__local_mem__ TINDEX*)indexLocal.GetPhyAddr();
    constexpr uint32_t repeatIndexElm = platform::GetVRegSize() / sizeof(TINDEX);
    uint16_t repeatIndexTimes = (maxLocalLen + static_cast<int64_t>(repeatIndexElm) - 1) / static_cast<int64_t>(repeatIndexElm);

    uint32_t maxNum = maxLocalLen;
    uint32_t indexNum = maxLocalLen;
    TINDEX defaultIndex = -1;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T1> v0;
        SetAllNegInfReg<T1>(v0);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<T1>(maxNum);
            MicroAPI::AddrReg offsetReg = MicroAPI::CreateAddrReg<T1>(i, repeatElm);
            MicroAPI::DataCopy(dstAddr, v0, offsetReg, p0);
        }
        MicroAPI::RegTensor<TINDEX> v1;
        MicroAPI::Duplicate(v1, defaultIndex);
        for (uint16_t i = 0; i < repeatIndexTimes; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<TINDEX>(indexNum);
            MicroAPI::AddrReg offsetReg = MicroAPI::CreateAddrReg<TINDEX>(i, repeatIndexElm);
            MicroAPI::DataCopy(indexAddr, v1, offsetReg, p0);
        }
    }
}

template <typename T1, typename T2, typename TINDEX>
template <bool MERGE, bool SPLITKW, bool SPLITKHW, bool SPLITKDHW>
__aicore__ inline void MaxPool3DWithArgmaxV2BigKernelRegbase<T1, T2, TINDEX>::ComputeSingleBlock(int32_t localCurIdx,
                                                                                   int64_t dataCount, int64_t offset,
                                                                                   int64_t curkW, int64_t curkH)
{
    LocalTensor<T1> maxOutLocal = maxUBOutput_.Get<T1>();
    LocalTensor<TINDEX> indexLocal = indexUBOutput_.Get<TINDEX>();
    LocalTensor<T1> xLocal = inputQue_.DeQue<T1>();
    union {
        T2 f;
        unsigned int i;
    } minValue;
    minValue.i = FLOAT32_NEG_INF;
    __local_mem__ T1* xLocalAddr = (__local_mem__ T1*)xLocal.GetPhyAddr();
    __local_mem__ T1* dstLocalAddr = (__local_mem__ T1*)maxOutLocal.GetPhyAddr();
    __local_mem__ TINDEX* indexLocalAddr = (__local_mem__ TINDEX*)indexLocal.GetPhyAddr();
    constexpr int32_t padIndex = -1;

    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T2);
    uint16_t repeatTimes = (dataCount + repeatElm - 1) / repeatElm;
    uint32_t num = repeatTimes * repeatElm;
    uint32_t padNum = num - dataCount;
    TINDEX inputW = tilingData_->wInDim;
    TINDEX inputH = tilingData_->hInDim;
    __VEC_SCOPE__
    {
        SetNegInfLocalMem<T1>(xLocalAddr, padNum, dataCount);
        
        MicroAPI::RegTensor<T2> vd0;
        MicroAPI::RegTensor<T2> res;
        MicroAPI::RegTensor<int32_t> resIndex;
        MicroAPI::RegTensor<int32_t> index;
        MicroAPI::MaskReg cmpMaskNanReg;
        MicroAPI::MaskReg cmpMaskReg;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(resIndex, padIndex);
        MicroAPI::Duplicate(res, minValue.f);
        MicroAPI::Arange(index, 0);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<T2>(num);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T1>(i, repeatElm);
            LoadOneRegTensor<T1>(xLocalAddr, vd0, p0, offset);
            MicroAPI::Compare<T2, CMPMODE::NE>(cmpMaskNanReg, vd0, vd0, maskAll);
            MicroAPI::Compare<T2, CMPMODE::GT>(cmpMaskReg, vd0, res, maskAll); 
            MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg, cmpMaskNanReg, maskAll);
            MicroAPI::Select(res, vd0, res, cmpMaskReg);
            MicroAPI::Select(resIndex, index, resIndex, cmpMaskReg);
            MicroAPI::Adds(index, index, repeatElm, maskAll);
        }
        GetMaxAndIndex<T2>(res, index, res, resIndex, padIndex);
        MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::VL1>();    
        MicroAPI::RegTensor<TINDEX> realResIndex;
        CalcRealIndex<TINDEX, SPLITKW, SPLITKHW, SPLITKDHW>(realResIndex, index, curkW, curkH, inputH, inputW, offset);
        if constexpr (MERGE) {
            MergeMaxAndIndex<T1, T2, TINDEX>(res, realResIndex, dstLocalAddr, indexLocalAddr, localCurIdx, padIndex);
        }
        StoreOneNum<TINDEX, TINDEX>(indexLocalAddr, realResIndex, pregOne, localCurIdx);
        StoreOneNum<T1, T2>(dstLocalAddr, res, pregOne, localCurIdx);
    }
    inputQue_.FreeTensor<T1>(xLocal);
}

}  // namespace MaxPool3DWithArgmaxV2BigKernelRegbase
#endif  // MAX_POOL_WITH_ARGMAX_V2_BIG_KERNEL_H_