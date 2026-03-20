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
 * \file avg_pool_common.h
 * \brief
 */
#ifndef AVG_POOL_COMMON_H_
#define AVG_POOL_COMMON_H_


#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"


namespace AvgPool
{
using namespace AscendC;

constexpr int32_t INDEX_SIZE = 256;
constexpr int32_t B64 = 8;
constexpr int32_t B8 = 1;
constexpr int32_t B16 = 2;
constexpr int32_t B32 = 4;

constexpr int32_t BUFFER_NUM = 2;

constexpr int32_t GATHER_SINGLE_ROW = 0;
constexpr int32_t GATHER_MULTI_ROW = 1;
constexpr int32_t GATHER_MULTI_BATCH = 2;
constexpr int32_t GATHER_SINGLE_KERNEL = 3;
constexpr int32_t NOT_GATHER = 1001;

constexpr int32_t SCATTER_SINGLE_ROW = 0;
constexpr int32_t SCATTER_MULTI_ROW = 1;
constexpr int32_t COPY_SINGLE_ROW = 2;

constexpr int32_t SPLIT_COLS = 1;
constexpr int32_t SPLIT_ROWS = 2;
constexpr int32_t SPLIT_BATCHS = 3;
constexpr uint16_t INT64_MAXREGNUM = 8;

constexpr int32_t ONE = 1;
constexpr int32_t TWO = 2;
constexpr int32_t THREE = 3;
constexpr int32_t FOUR = 4;
constexpr int32_t FIVE = 5;
constexpr int32_t SIX = 6;
constexpr int32_t SEVEN = 7;
constexpr int32_t EIGHT = 8;
constexpr int32_t NINE = 9;
constexpr int32_t TEN = 10;
constexpr int32_t ELEVEN = 11;
constexpr int32_t TWELVE = 12;
constexpr int32_t THIRTEEN = 13;
constexpr int32_t FOURTEEN = 14;
constexpr int32_t FIFTEEN = 15;
constexpr int32_t SIXTEEN = 16;

struct CalcDivisorParam {
    int64_t kH = 0;
    int64_t kW = 0;
    int64_t sH = 0;
    int64_t sW = 0;
    int64_t topPad = 0;
    int64_t bottomPad = 0;
    int64_t leftPad = 0;
    int64_t rightPad = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t hIn = 0;
    int64_t wIn = 0;
};

template <typename T>
struct GetComputeType {
    using type = typename std::conditional<std::is_same<T, bool>::value, int8_t, T>::type;
};

template <typename T>
struct GetGatherType {
    using type =
        typename std::conditional<std::is_same<T, int8_t>::value, int16_t,
                                  typename std::conditional<std::is_same<T, uint8_t>::value, uint16_t, T>::type>::type;
};

template <typename T>
struct VciTypeGet {
    using type = typename std::conditional<
        std::is_same<T, uint32_t>::value, int32_t,
        typename std::conditional<std::is_same<T, uint16_t>::value, int16_t,
                                  typename std::conditional<std::is_same<T, uint64_t>::value, int64_t, T>::type>::type>::type;
};

template <typename T>
struct IndexTypeGet {
    using type = typename std::conditional<sizeof(T) == B8 || sizeof(T) == B16, uint16_t, uint32_t>::type;
};

constexpr MicroAPI::CastTrait castTraitB16ToB32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr AscendC::MicroAPI::CastTrait castTraitB32ToB16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait CAST_INT32_TO_FP32 = {
  AscendC::MicroAPI::RegLayout::UNKNOWN,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT
};

constexpr AscendC::MicroAPI::CastTrait CAST_INT64_TO_FP32 = {
  AscendC::MicroAPI::RegLayout::UNKNOWN,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT
};

constexpr MicroAPI::CastTrait castTraitT2Fp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr MicroAPI::CastTrait castTraitFp322T = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

struct PoolParamsForDim {
    int64_t in = 0;
    int64_t o = 0;
    int64_t k = 0;
    int64_t s = 0;
    int64_t pl = 0;
    int64_t pr = 0;
};

__aicore__ inline void CalcKernelSizeCore(const PoolParamsForDim& paramsInfo, int64_t& curk, int64_t& curkWithPad,
    int64_t& curOrigin)
{
    curOrigin = paramsInfo.s * paramsInfo.o - paramsInfo.pl;  // left
    int64_t leftInvaild = 0;
    if (curOrigin < 0) {
        leftInvaild = -curOrigin;  // 0 左侧有几个无效k
    }
    // min(in - origin - leftinvaild, k)
    curk = min(paramsInfo.in - curOrigin - leftInvaild, paramsInfo.k - leftInvaild);
    // min (in + pr - origin, k)
    curkWithPad = min(paramsInfo.in + paramsInfo.pr - curOrigin, paramsInfo.k);
    curOrigin += leftInvaild;  // 矫正到curOrigin +轴位置
}

template <typename T>
__aicore__ inline void CustomDuplicate(__local_mem__ T* dstAddr, uint32_t calNum, uint16_t loop)
{
    uint32_t sreg = calNum;
    MicroAPI::RegTensor<T> v0;
    MicroAPI::Duplicate(v0, (T)0);
    constexpr uint16_t repeatElm = Ops::Base::GetVRegSize() / sizeof(T);
    for (uint16_t i = 0; i < loop; i++) {
        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, repeatElm);
        MicroAPI::DataCopy(dstAddr, v0, offset, preg);
    }
}

template <typename T>
__aicore__ inline void CustomCopy(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                  uint32_t srcBatchStride, uint32_t srcRowStride, uint32_t dstBatchStride,
                                  uint32_t dstRowStride, uint32_t dstRowOffset, uint32_t dstColOffset, uint16_t batch,
                                  uint16_t rows, uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm)
{
    MicroAPI::RegTensor<T> v0;
    MicroAPI::UnalignReg u0;

    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t j = 0; j < rows; j++) {
            __local_mem__ T* curSrcAddr = (__local_mem__ T*)srcAddr + i * srcBatchStride + j * srcRowStride;
            __local_mem__ T* curDstAddr =
                (__local_mem__ T*)dstAddr + i * dstBatchStride + (j + dstRowOffset) * dstRowStride + dstColOffset;
            for (uint16_t k = 0; k < loopCols; k++) {
                MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, repeatElm);
            }
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
            MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, tailCols);
            MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterSingleRow(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                                    uint16_t srcBatchStride, uint16_t srcRowStride,
                                                    uint16_t dstBatchStride, uint16_t dstRowStride,
                                                    uint16_t dstRowOffset, uint16_t dstColOffset, uint16_t batch,
                                                    uint16_t rows, uint16_t loopCols, uint16_t cols, uint16_t repeatElm)
{
    MicroAPI::RegTensor<T> v0;
    MicroAPI::RegTensor<U> sIndex;
    MicroAPI::MaskReg preg;
    using regType = typename VciTypeGet<U>::type;
    MicroAPI::Arange((MicroAPI::RegTensor<regType>&)sIndex, 0);
    auto dstAddr1 = (__local_mem__ T*)dstAddr + dstRowOffset * dstRowStride + dstColOffset;
    for (uint16_t i = 0; i < batch; i++) {
        auto dstAddr2 = dstAddr1 + i * dstBatchStride;
        auto srcAddr1 = (__local_mem__ T*)srcAddr + i * srcBatchStride;
        uint32_t sreg = cols;
        for (uint16_t j = 0; j < loopCols; j++) {
            auto curDstAddr = dstAddr2 + j * repeatElm;
            auto curSrcAddr = srcAddr1 + j * repeatElm;
            preg = MicroAPI::UpdateMask<U>(sreg);
            for (uint16_t k = 0; k < rows; k++) {
                MicroAPI::DataCopy(v0, curSrcAddr + k * srcRowStride);
                MicroAPI::DataCopyScatter(curDstAddr + k * dstRowStride, v0, sIndex, preg);
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterMultiRows(const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr,
                                                    MicroAPI::RegTensor<U> index, uint32_t srcBatchStride,
                                                    uint32_t srcRowStride, uint32_t dstBatchStride,
                                                    uint32_t dstRowStride, uint32_t dstOffset, uint16_t batch,
                                                    uint16_t loopRows, uint32_t repeatElm, uint32_t tailElm)
{
    MicroAPI::RegTensor<T> vd1;
    MicroAPI::RegTensor<U> v1, v2, v3, v4;
    MicroAPI::RegTensor<U> gIndex;
    uint32_t sreg = repeatElm;
    uint32_t tailSreg = tailElm;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg preg;
    MicroAPI::MaskReg tailPreg;
    preg = MicroAPI::UpdateMask<U>(sreg);
    tailPreg = MicroAPI::UpdateMask<U>(tailSreg);
    using regType = typename VciTypeGet<U>::type;
    MicroAPI::RegTensor<U> vd0;
    MicroAPI::Arange((MicroAPI::RegTensor<regType>&)gIndex, 0);
    __local_mem__ T* curDstAddr = (__local_mem__ T*)dstAddr + dstOffset; 
    for (uint16_t i = 0; i < batch; i++) {
        MicroAPI::Adds(v1, index, i * dstBatchStride, maskAll);
        MicroAPI::Adds(v3, gIndex, i * srcBatchStride, maskAll);
        for (uint16_t j = 0; j < loopRows; j++) {
            MicroAPI::Adds(v2, v1, j * dstRowStride, preg);
            MicroAPI::Adds(v4, v3, j * srcRowStride, preg);
            MicroAPI::DataCopyGather(vd1, srcAddr, v4, preg);
            MicroAPI::DataCopyScatter(curDstAddr, vd1, v2, preg);
        }
        MicroAPI::Adds(v2, v1, loopRows * dstRowStride, tailPreg);
        MicroAPI::Adds(v4, v3, loopRows * srcRowStride, tailPreg);
        MicroAPI::DataCopyGather(vd1, srcAddr, v4, tailPreg);
        MicroAPI::DataCopyScatter(curDstAddr, vd1, v2, tailPreg);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPoolB32Impl(RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kH,
                                   uint16_t kW, U rowStrideInub, float32_t divisor, MicroAPI::MaskReg& pMask, uint16_t channels = 1)
{
    RegDstT vd1;
    MicroAPI::RegTensor<U> v0;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::RegTensor<float32_t> divisorReg;

    MicroAPI::Duplicate(res, (T)0);
    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        MicroAPI::Adds(v0, index, hIdx * rowStrideInub, pMask);
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            MicroAPI::Adds(v1, v0, wIdx * channels, pMask);
            MicroAPI::DataCopyGather(vd1, srcAddr, v1, pMask);
            MicroAPI::Add(res, vd1, res, pMask);
        }
    }
    if constexpr (!NO_DIV) {
        MicroAPI::Duplicate(divisorReg, divisor);
        MicroAPI::Div(res, res, divisorReg, pMask);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPoolB16Impl(RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kH,
                                   uint16_t kW, U rowStrideInub, float32_t divisor, MicroAPI::MaskReg& pMask, uint16_t channels = 1)
{
    MicroAPI::RegTensor<T> vd1;
    MicroAPI::RegTensor<T> zero;
    MicroAPI::RegTensor<U> v0;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::RegTensor<float32_t> tmpRes1;
    MicroAPI::RegTensor<float32_t> tmpRes2;
    MicroAPI::RegTensor<float32_t> left;
    MicroAPI::RegTensor<float32_t> right;
    MicroAPI::RegTensor<float32_t> divisorReg;
    MicroAPI::RegTensor<T> tmpLeft;
    MicroAPI::RegTensor<T> tmpRight;
    MicroAPI::Duplicate(tmpRes1, (float32_t)0);
    MicroAPI::Duplicate(tmpRes2, (float32_t)0);
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate((MicroAPI::RegTensor<float16_t>&)zero, (float16_t)0);
    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        MicroAPI::Adds(v0, index, hIdx * rowStrideInub, pMask);
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            MicroAPI::Adds(v1, v0, wIdx * channels, pMask);
            MicroAPI::DataCopyGather(vd1, srcAddr, v1, pMask);
            MicroAPI::Interleave(tmpLeft, tmpRight,  vd1, zero);
            MicroAPI::Cast<float32_t, T, castTraitB16ToB32>(left, tmpLeft, maskAll);
            MicroAPI::Cast<float32_t, T, castTraitB16ToB32>(right, tmpRight, maskAll);
            MicroAPI::Add(tmpRes1, tmpRes1, left, maskAll);
            MicroAPI::Add(tmpRes2, tmpRes2, right, maskAll);
        }
    }
    if constexpr (NO_DIV) {
        MicroAPI::Copy((MicroAPI::RegTensor<float32_t>&)res.reg[0], tmpRes1);
        MicroAPI::Copy((MicroAPI::RegTensor<float32_t>&)res.reg[1], tmpRes2);
    } else {
        MicroAPI::Duplicate(divisorReg, divisor);
        MicroAPI::Div(tmpRes1, tmpRes1, divisorReg, maskAll);
        MicroAPI::Div(tmpRes2, tmpRes2, divisorReg, maskAll);
        MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(tmpLeft, tmpRes1, maskAll);
        MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(tmpRight, tmpRes2, maskAll);
        MicroAPI::DeInterleave(res, zero, tmpLeft, tmpRight);
    }
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannelB32(__local_mem__ M* dstLocalAddr, __local_mem__ M* srcLocalAddr,
                                            uint16_t kH, uint16_t kW, U rowStrideInUb, 
                                            uint16_t alignChannels, uint16_t repeatElms, float32_t divisor)
{
    MicroAPI::RegTensor<M> res;
    MicroAPI::RegTensor<M> vd0;
    MicroAPI::RegTensor<M> divRegs;
    uint32_t num = repeatElms;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::UnalignReg u0;
    __local_mem__ M* curSrcAddr = srcLocalAddr;

    MicroAPI::Duplicate(res, (float32_t)0);

    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            auto aReg = MicroAPI::CreateAddrReg<U>(hIdx, rowStrideInUb, wIdx, alignChannels);
            MicroAPI::DataCopy(vd0, curSrcAddr, aReg);
            MicroAPI::Add(res, vd0, res, p0);
        }
    }
    MicroAPI::Duplicate(divRegs, divisor);
    MicroAPI::Div(res, res, divRegs, p0);
    MicroAPI::DataCopy(dstLocalAddr, res, p0);
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannelB16(__local_mem__ M* dstLocalAddr, __local_mem__ M* srcLocalAddr,
                                            uint16_t kH, uint16_t kW, U rowStrideInUb, 
                                            uint16_t alignChannels, uint16_t repeatElms, float32_t divisor)
{
    MicroAPI::RegTensor<M> res;
    MicroAPI::RegTensor<M> vd0;
    MicroAPI::RegTensor<M> zero;
    MicroAPI::RegTensor<float32_t> tmpRes1;
    MicroAPI::RegTensor<float32_t> tmpRes2;
    MicroAPI::RegTensor<float32_t> left;
    MicroAPI::RegTensor<float32_t> right;
    MicroAPI::RegTensor<float32_t> divisorReg;
    MicroAPI::RegTensor<M> tmpLeft;
    MicroAPI::RegTensor<M> tmpRight;

    uint32_t num = repeatElms;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::UnalignReg u0;
    __local_mem__ M* curSrcAddr = srcLocalAddr;
    MicroAPI::MaskReg defaultMask = MicroAPI::CreateMask<M, MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate((MicroAPI::RegTensor<float16_t>&)zero, (float16_t)0);
    MicroAPI::Duplicate(res, (float32_t)0);
    
    MicroAPI::Duplicate(tmpRes1, (float32_t)0);
    MicroAPI::Duplicate(tmpRes2, (float32_t)0);

    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            auto aReg = MicroAPI::CreateAddrReg<U>(hIdx, rowStrideInUb, wIdx, alignChannels);
            MicroAPI::DataCopy(vd0, curSrcAddr, aReg);
            MicroAPI::Interleave(tmpLeft, tmpRight, vd0, zero);
            MicroAPI::Cast<float32_t, M, castTraitB16ToB32>(left, tmpLeft, defaultMask);
            MicroAPI::Cast<float32_t, M, castTraitB16ToB32>(right, tmpRight, defaultMask);
            MicroAPI::Add(tmpRes1, tmpRes1, left, defaultMask);
            MicroAPI::Add(tmpRes2, tmpRes2, right, defaultMask);
        }
    }
    
    MicroAPI::Duplicate(divisorReg, divisor);
    MicroAPI::Div(tmpRes1, tmpRes1, divisorReg, defaultMask);
    MicroAPI::Div(tmpRes2, tmpRes2, divisorReg, defaultMask);
    MicroAPI::Cast<M, float32_t, castTraitB32ToB16>(tmpLeft, tmpRes1, defaultMask);
    MicroAPI::Cast<M, float32_t, castTraitB32ToB16>(tmpRight, tmpRes2, defaultMask);
    MicroAPI::DeInterleave(res, zero, tmpLeft, tmpRight);
    MicroAPI::DataCopy(dstLocalAddr, res, p0);
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannel(__local_mem__ M* dstLocalAddr, __local_mem__ M* srcLocalAddr,
                                           uint16_t kH, uint16_t kW, U rowStrideInUb, 
                                           uint16_t alignChannels, uint16_t repeatElms, float32_t divisor)
{
    if constexpr (sizeof(M) == TWO) {
        AvgPoolSingleChannelB16<M, U>(dstLocalAddr, srcLocalAddr, kH, kW, rowStrideInUb, 
                                     alignChannels, repeatElms, divisor);
    } else {
        AvgPoolSingleChannelB32<M, U>(dstLocalAddr, srcLocalAddr, kH, kW, rowStrideInUb, 
                                     alignChannels, repeatElms, divisor);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPoolImpl(RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kH,
                                   uint16_t kW, U rowStrideInub, float32_t divisor, MicroAPI::MaskReg& pMask, uint16_t channels = 1) {
    if constexpr(sizeof(T) == TWO) {
        AvgPoolB16Impl<T, U, NO_DIV>(res, srcAddr, index, kH, kW, rowStrideInub, divisor, pMask, channels);
    } else {
        AvgPoolB32Impl<T, U, NO_DIV>(res, srcAddr, index, kH, kW, rowStrideInub, divisor, pMask, channels);
    }
}

template <typename T, typename U, typename Z, bool NO_DIV=false>
__aicore__ inline void AvgPoolSplitW(__local_mem__ Z* dstLocalAddr, __local_mem__ T* srcAddr,
                                     MicroAPI::RegTensor<U>& index, uint16_t kH, uint16_t kW, uint16_t loopH,
                                     uint16_t loopW, U oneLoopStrideH, U oneLoopStrideW, U rowStrideInub,
                                     uint16_t oneLoopElements, uint16_t tailLoopElements, U halfLoopOut0,
                                     U halfLoopOut1, U tailHalfLoopOut0, U tailHalfLoopOut1, float32_t divisor, uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<sizeof(T)==B16 && std::is_same<Z, float32_t>::value, MicroAPI::RegTensor<Z, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
    RegDstT res;
    MicroAPI::RegTensor<U> v0;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::RegTensor<U> v2;
    MicroAPI::RegTensor<U> v3;
    MicroAPI::RegTensor<U> v4;
    MicroAPI::UnalignReg u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailNum);
    __local_mem__ Z* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopH; i++) {
        MicroAPI::Adds(v0, index, i * oneLoopStrideH, p0);
        for (uint16_t j = 0; j < loopW; j++) {
            MicroAPI::Adds(v2, v0, j * oneLoopStrideW, p0);
            AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v2, kH, kW, rowStrideInub, divisor, p0, channels);
            if constexpr (sizeof(T)==B16  && std::is_same<Z, float32_t>::value) {
                MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[0], u0, halfLoopOut0);
                MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[1], u0, halfLoopOut1);
            } else {
                MicroAPI::DataCopyUnAlign(dstAddr, res, u0, oneLoopElements);
            }
        }
        MicroAPI::Adds(v2, v0, loopW * oneLoopStrideW, pTail);
        AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v2, kH, kW, rowStrideInub, divisor, pTail, channels);
        if constexpr(sizeof(T)==B16  && std::is_same<Z, float32_t>::value) {
            MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[0], u0, tailHalfLoopOut0);
            MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[1], u0, tailHalfLoopOut1);
        } else {
            MicroAPI::DataCopyUnAlign(dstAddr, res, u0, tailLoopElements);
        }
    }
    MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
}

template <typename T, typename U, typename Z, bool NO_DIV=false>
__aicore__ inline void AvgPoolSplitH(__local_mem__ Z* dstLocalAddr, __local_mem__ T* srcAddr,
                                     MicroAPI::RegTensor<U>& index, uint16_t kH, uint16_t kW, uint16_t loopN,
                                     uint16_t loopH, U oneChannelElements, U rowStrideInub, U oneLoopStride,
                                     uint16_t oneLoopElements, uint16_t tailLoopElements,  U halfLoopOut0,
                                     U halfLoopOut1, U tailHalfLoopOut0, U tailHalfLoopOut1, float32_t divisor, uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<sizeof(T)==B16 && std::is_same<Z, float32_t>::value, MicroAPI::RegTensor<Z, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
    RegDstT res;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::RegTensor<U> v2;
    MicroAPI::RegTensor<U> v3;
    MicroAPI::RegTensor<U> v4;
    MicroAPI::UnalignReg u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailNum);
    __local_mem__ Z* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopN; i++) {
        MicroAPI::Adds(v1, index, i * oneChannelElements, p0);
        for (uint16_t j = 0; j < loopH; j++) {
            MicroAPI::Adds(v2, v1, j * oneLoopStride, p0);
            AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v2, kH, kW, rowStrideInub, divisor, p0, channels);
            if constexpr (sizeof(T)==B16  && std::is_same<Z, float32_t>::value) {
                MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[0], u0, halfLoopOut0);
                MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[1], u0, halfLoopOut1);
            } else {
                MicroAPI::DataCopyUnAlign(dstAddr, res, u0, oneLoopElements);
            }
        }
        MicroAPI::Adds(v2, v1, loopH * oneLoopStride, pTail);
        AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v2, kH, kW, rowStrideInub, divisor, pTail, channels);
        if constexpr(sizeof(T)==B16  && std::is_same<Z, float32_t>::value) {
            MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[0], u0, tailHalfLoopOut0);
            MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[1], u0, tailHalfLoopOut1);
        } else {
            MicroAPI::DataCopyUnAlign(dstAddr, res, u0, tailLoopElements);
        }
    }
    MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
}

template <typename T, typename U, typename Z, bool NO_DIV=false>
__aicore__ inline void AvgPoolSplitBatch(__local_mem__ Z* dstLocalAddr, __local_mem__ T* srcAddr,
                                         MicroAPI::RegTensor<U>& index, uint16_t kH, uint16_t kW, uint16_t loopN,
                                         U rowStrideInub, U oneLoopStride, uint16_t oneLoopElements,
                                         uint16_t tailLoopElements,  U halfLoopOut0, U halfLoopOut1,
                                         U tailHalfLoopOut0, U tailHalfLoopOut1, float32_t divisor, uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<sizeof(T)==B16 && std::is_same<Z, float32_t>::value, MicroAPI::RegTensor<Z, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
    RegDstT res;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::UnalignReg u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailNum);
    __local_mem__ Z* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopN; i++) {
        MicroAPI::Adds(v1, index, i * oneLoopStride, p0);
        AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v1, kH, kW, rowStrideInub, divisor, p0, channels);
        if constexpr (sizeof(T)==B16  && std::is_same<Z, float32_t>::value) {
            MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[0], u0, halfLoopOut0);
            MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[1], u0, halfLoopOut1);
        } else {
            MicroAPI::DataCopyUnAlign(dstAddr, res, u0, oneLoopElements);
        }
    }
    MicroAPI::Adds(v1, index, loopN * oneLoopStride, pTail);
    AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v1, kH, kW, rowStrideInub, divisor, pTail, channels);
    if constexpr(sizeof(T)==B16  && std::is_same<Z, float32_t>::value) {
        MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[0], u0, tailHalfLoopOut0);
        MicroAPI::DataCopyUnAlign(dstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[1], u0, tailHalfLoopOut1);
    } else {
        MicroAPI::DataCopyUnAlign(dstAddr, res, u0, tailLoopElements);
    }
    MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
}


template <typename U, bool SingleRow>
__aicore__ inline void GenScatterIndex(uint32_t wIn, uint32_t wInDst, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;

        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        if constexpr (SingleRow) {
            MicroAPI::DataCopy(dstAddr, v0, p0);
        } else {
            MicroAPI::Duplicate(v1, (U)wIn, p0);
            MicroAPI::Duplicate(v2, (U)wInDst, p0);

            MicroAPI::Div(vd1, v0, v1, p0);
            MicroAPI::Mul(vd2, vd1, v2, p0);
            MicroAPI::Mul(vd3, vd1, v1, p0);
            MicroAPI::Sub(vd4, v0, vd3, p0);
            MicroAPI::Add(vd5, vd2, vd4, p0);
            MicroAPI::DataCopy(dstAddr, vd5, p0);
        }
    }
}

template <typename U, bool SingleRow>
__aicore__ inline void NHWCGenScatterIndex(uint32_t wIn, uint32_t wInDstElms, uint32_t channels, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd7;
        MicroAPI::RegTensor<U> vd8;
        MicroAPI::RegTensor<U> vd9;
        MicroAPI::RegTensor<U> vd10;

        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        if constexpr (SingleRow) {
            MicroAPI::DataCopy(dstAddr, v0, p0);
        } else {
            MicroAPI::Duplicate(v1, (U)wIn, p0);
            MicroAPI::Duplicate(v2, (U)wInDstElms, p0);
            MicroAPI::Duplicate(v3, (U)channels, p0);

            MicroAPI::Div(vd1, v0, v3, p0);     // i / channels
            MicroAPI::Div(vd2, vd1, v1, p0);     // i / channels / win
            MicroAPI::Mul(vd3, vd2, v2, p0);    // i / channels / win * winDst

            MicroAPI::Mul(vd4, vd2, v1, p0);     // i / channels / win * win
            MicroAPI::Sub(vd5, vd1, vd4, p0);    // i / channels mod win
            MicroAPI::Mul(vd6, vd5, v3, p0);    // ( i / channels mod win) * channels
            MicroAPI::Add(vd7, vd3, vd6, p0);   // i / channels / win * winDst + i / channels mod win * channels

            MicroAPI::Mul(vd8, vd1, v3, p0);
            MicroAPI::Sub(vd9, v0, vd8, p0);    // i mod channels

            MicroAPI::Add(vd10, vd9, vd7, p0);  // (i / channels / win * winDst + i / channels mod win) * channels + i mod channels
            MicroAPI::DataCopy(dstAddr, vd10, p0);
        }
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexSingleRow(uint32_t wStride, uint32_t channels, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    // i * wStride
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<regType> tmp;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;

        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wStride, p0);
        MicroAPI::Duplicate(v2, (U)channels, p0); // channels
        MicroAPI::Div(vd0, v0, v2, p0);   // i / channels
        MicroAPI::Mul(vd1, vd0, v2, p0);
        MicroAPI::Sub(vd5, v0, vd1, p0);   // i % channel
        MicroAPI::Mul(vd2, vd0, v1, p0);  // (i / channel * wstride)
        MicroAPI::Mul(vd3, vd2, v2, p0);  // (i / channel * wstride * channels)
        MicroAPI::Add(vd4, vd3, vd5, p0);   // (i / channel * wstride * channels) + i % channel
        MicroAPI::DataCopy(dstAddr, vd4, p0);
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexMultiRow(uint32_t wFactorOut, uint32_t wInElms, uint32_t hStride, uint32_t wStride,
                                              uint32_t channels, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    // i / wFactorOut * wIn * hStride + i % wFactorOut * wStride
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;
        MicroAPI::RegTensor<U> v4;
        MicroAPI::RegTensor<U> v5;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd7;
        MicroAPI::RegTensor<U> vd8;
        MicroAPI::RegTensor<U> vd9;
        MicroAPI::RegTensor<U> vd10;
        MicroAPI::RegTensor<U> vd11;
        MicroAPI::RegTensor<U> vd12;
        MicroAPI::RegTensor<U> vd13;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wFactorOut, p0);
        MicroAPI::Duplicate(v2, (U)wInElms, p0);
        MicroAPI::Duplicate(v3, (U)hStride, p0);
        MicroAPI::Duplicate(v4, (U)wStride, p0);
        MicroAPI::Duplicate(v5, (U)channels, p0);

        MicroAPI::Div(vd1, v0, v5, p0);    // i / channels
        MicroAPI::Div(vd2, vd1, v1, p0);    // i / channels / wFactorOut
        MicroAPI::Mul(vd3, vd2, v2, p0);   // (i  / channels / wFactorOut * wIn)
        MicroAPI::Mul(vd4, vd3, v3, p0);   // (i / channels / wFactorOut * wIn * hStride

        MicroAPI::Mul(vd5, vd2, v1, p0);   // (i / channels / wFactorOut * wFactorOut)
        MicroAPI::Sub(vd6, vd1, vd5, p0);   // (i  / channels) % wFactor
        MicroAPI::Mul(vd7, vd6, v4, p0);   // (i  / channels) % wFactorOut * wStride
        MicroAPI::Mul(vd8, vd7, v5, p0);     // ( i  / channels) % wFactorOut * wStride) * channels

        MicroAPI::Add(vd9, vd8, vd4, p0);  // (i  / channels) / wFactorOut * wIn * hStride + (i  / channels) % wFactorOut * wStride* channels)
        MicroAPI::Mul(vd11, vd1, v5, p0);    // i / channels * channels
        MicroAPI::Sub(vd12, v0, vd11, p0);  // i mod channel
        MicroAPI::Add(vd13, vd9, vd12, p0);
        MicroAPI::DataCopy(dstAddr, vd13, p0);
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexMultiBatch(uint32_t hFactorOut, uint32_t wFactorOut, uint32_t hIn, uint32_t wInElms,
                                                uint32_t hStride, uint32_t wStride, uint32_t channels, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    U batchElemtsIn = hIn * wInElms;
    U batchElemtsOut = hFactorOut * wFactorOut * channels;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;
        MicroAPI::RegTensor<U> v4;
        MicroAPI::RegTensor<U> v5;
        MicroAPI::RegTensor<U> v6;
        MicroAPI::RegTensor<U> v7;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd8;
        MicroAPI::RegTensor<U> vd12;
        MicroAPI::RegTensor<U> vd14;
        MicroAPI::RegTensor<U> vd17;
        MicroAPI::RegTensor<U> vd18;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wFactorOut, p0);
        MicroAPI::Duplicate(v2, (U)wInElms, p0);
        MicroAPI::Duplicate(v3, (U)hStride, p0);
        MicroAPI::Duplicate(v4, (U)wStride, p0);
        MicroAPI::Duplicate(v5, (U)channels, p0);
        MicroAPI::Duplicate(v6, (U)batchElemtsIn, p0);
        MicroAPI::Duplicate(v7, (U)batchElemtsOut, p0);

        MicroAPI::Div(vd1, v0, v7, p0);   // i / (rows * cols * channels)
        MicroAPI::Mul(vd2, vd1, v6, p0);  // i / (rows * cols * channels) * batchElemtsIn       n

        MicroAPI::Mul(vd4, vd1, v7, p0);  // (i / (rows * cols * channels) * (rows * cols * channels) 
        MicroAPI::Sub(vd4, v0, vd4, p0);  // i % (rows * cols *channels) 

        MicroAPI::Div(vd5, vd4, v5, p0);     // hwoffset / channels
        MicroAPI::Div(vd6, vd5, v1, p0);     // hwoffset / channels / wfout
        MicroAPI::Mul(vd8, vd6, v2, p0);    // hwoffset / channels / wfout * win
        MicroAPI::Mul(vd8, vd8, v3, p0);    // hwoffset / channels / wfout * hstride  h

        MicroAPI::Mul(vd12, vd6, v1, p0);    // hwoffset / channels / wfout * wfout
        MicroAPI::Sub(vd12, vd5, vd12, p0);  // hwoffset / channels % wfout 
        MicroAPI::Mul(vd12, vd12, v4, p0);    // hwoffset / channels % wfout * wstride
        MicroAPI::Mul(vd12, vd12, v5, p0);    // (hwoffset / channels % wfout * wstride) * channels

        MicroAPI::Add(vd14, vd12, vd8, p0);    // hwoffset / channels / wfout * hstride + hwoffset / channels % wfout * wstride
        MicroAPI::Add(vd14, vd14, vd2, p0);    // (hwoffset / channels / wfout * hstride + hwoffset / channels / wfout * wstride) * channels + i / (rows * cols * channels) * batchElemtsIn

        MicroAPI::Div(vd17, v0, v5, p0);   // i / channels
        MicroAPI::Mul(vd17, vd17, v5, p0);   // i / channels * channels
        MicroAPI::Sub(vd17, v0, vd17, p0);   // i % channels

        MicroAPI::Add(vd18, vd14, vd17, p0);
        MicroAPI::DataCopy(dstAddr, vd18, p0);
    }
}

__aicore__ inline void FastDivImpl(MicroAPI::RegTensor<uint32_t>& res, MicroAPI::RegTensor<uint32_t>& src, MicroAPI::RegTensor<uint32_t>& magic, int16_t shift, MicroAPI::MaskReg &mask) 
{
    MicroAPI::RegTensor<uint32_t> tmp;
    MicroAPI::Mull(tmp, res, src, magic, mask);                                                      
    MicroAPI::Add(tmp, src, res, mask);                                                                
    MicroAPI::ShiftRights(res, tmp, shift, mask);   
}

template <typename T, bool INCLUDE_PAD, typename RegT>
__aicore__ inline void CalcWindowSize(MicroAPI::RegTensor<float>& res, RegT& src, T kD, T sD, T negFrontPad, T dIn, T dInAndBackendPad, MicroAPI::MaskReg &mask) 
{
    RegT tmp1, tmp2;
    MicroAPI::Muls(tmp1, src, sD, mask);   // (didx * sd)
    MicroAPI::Adds(tmp2, tmp1, negFrontPad, mask);   // (didx * sd - fPad)
    MicroAPI::Adds(tmp1, tmp2, kD, mask);   // dstart + kD
    if constexpr (INCLUDE_PAD) {
        MicroAPI::Mins(tmp1, tmp1, dInAndBackendPad, mask);     
    } else {
        MicroAPI::Maxs(tmp2, tmp2, 0, mask);
        MicroAPI::Mins(tmp1, tmp1, dIn, mask);
    }
    MicroAPI::Sub(tmp1, tmp1, tmp2, mask);
    if constexpr (std::is_same<T, int32_t>::value) {
        MicroAPI::Cast<float, T, CAST_INT32_TO_FP32>(res, tmp1, mask);
    } else {
        MicroAPI::Cast<float, T, CAST_INT64_TO_FP32>(res, tmp1, mask);
    }    
}

template <bool countIncludePad, bool PAD_MULTI_BATCH>
__aicore__ inline void ComputeDivisorImplB32(__local_mem__ float* divAddr, const CalcDivisorParam &param, int32_t start, int32_t total)   {
    __local_mem__ float*  dstAddr = divAddr;
    int32_t oneRegLength = Ops::Base::GetVRegSize() / sizeof(float32_t);
    int32_t oneBatchOut = param.outH * param.outW;
    int32_t totalNum = total;
    uint16_t loopNum = Ops::Base::CeilDiv(totalNum, oneRegLength);
    int32_t kH = param.kH;
    int32_t kW = param.kW;
    int32_t sH = param.sH;
    int32_t sW = param.sW;

    int32_t negTopPad = -1 * param.topPad;
    int32_t hInAndBottomPad = param.hIn + param.bottomPad;
    int32_t negLeftPad = -1 * param.leftPad;
    int32_t wInAndRightPad = param.wIn + param.rightPad;
    int32_t hIn = param.hIn;
    int32_t wIn = param.wIn;
    uint32_t m0, m1;
    uint32_t shift0, shift1;

    GetUintDivMagicAndShift<uint32_t>(m0, shift0, param.outW);
    GetUintDivMagicAndShift<uint32_t>(m1, shift1, oneBatchOut);
    int32_t outW = param.outW;
    int32_t outH = param.outH;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> v0;
        MicroAPI::RegTensor<int32_t> v1;
        MicroAPI::RegTensor<int32_t> v2;
        MicroAPI::RegTensor<int32_t> v3;
        MicroAPI::RegTensor<uint32_t> magic0;
        MicroAPI::RegTensor<uint32_t> magic1;
        MicroAPI::RegTensor<int32_t> vd0;
        MicroAPI::RegTensor<int32_t> vd1;
        MicroAPI::RegTensor<int32_t> vd2;
        MicroAPI::RegTensor<int32_t> vd3;

        MicroAPI::RegTensor<float32_t> res;
        MicroAPI::RegTensor<float32_t> hWindow;
        MicroAPI::RegTensor<float32_t> wWindow;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(v1, outW, p0);
        MicroAPI::Duplicate(v2, outH, p0);

        MicroAPI::Duplicate(magic0, m0, p0);
        MicroAPI::Duplicate(magic1, m1, p0);
     
        uint32_t sreg = totalNum;
        for (uint16_t i = 0; i < loopNum; i++)
        {
            MicroAPI::Arange(v0, i * oneRegLength + start);
            MicroAPI::AddrReg resOffset = MicroAPI::CreateAddrReg<float32_t>(i, oneRegLength);
            MicroAPI::MaskReg pWrite = MicroAPI::UpdateMask<float32_t>(sreg);
            if constexpr (PAD_MULTI_BATCH) {
                MicroAPI::Duplicate(v3, oneBatchOut, p0);
                FastDivImpl((MicroAPI::RegTensor<uint32_t> &)vd1, (MicroAPI::RegTensor<uint32_t> &)v0, magic1, shift1, p0);
                MicroAPI::Mul(vd2, vd1, v3, p0);
                MicroAPI::Sub(v0, v0, vd2, p0);
            }
            FastDivImpl((MicroAPI::RegTensor<uint32_t> &)vd1, (MicroAPI::RegTensor<uint32_t> &)v0, magic0, shift0, p0); // (i / outhw) -> hidx
            MicroAPI::Mul(vd2, vd1, v1, p0);
            MicroAPI::Sub(vd3, v0, vd2, p0);

            CalcWindowSize<int32_t, countIncludePad>(hWindow, vd1, kH, sH, negTopPad, hIn, hInAndBottomPad, p0);
            CalcWindowSize<int32_t, countIncludePad>(wWindow, vd3, kW, sW, negLeftPad, wIn, wInAndRightPad, p0);
            MicroAPI::Mul(res, hWindow, wWindow, p0);
            MicroAPI::DataCopy(dstAddr, res, resOffset, pWrite);             
        }
    }  
}


template <bool countIncludePad, bool PAD_MULTI_BATCH>
__aicore__ inline void ComputeDivisorImplB64(__local_mem__ float* divAddr, const CalcDivisorParam &param, int32_t start, int32_t total)   {

    __local_mem__ float*  dstAddr = divAddr;
    int64_t oneRegLength = Ops::Base::GetVRegSize() / sizeof(float32_t);
    int32_t oneBatchOut = param.outH * param.outW;
    int32_t outPlane = param.outW;
    int64_t totalNum = total;
    uint16_t loopNum = Ops::Base::CeilDiv(totalNum, oneRegLength);
    int64_t kH = param.kH;
    int64_t kW = param.kW;
    int64_t sH = param.sH;
    int64_t sW = param.sW;

    int64_t negTopPad = -1 * param.topPad;
    int64_t hInAndBottomPad = param.hIn + param.bottomPad;
    int64_t negLeftPad = -1 * param.leftPad;
    int64_t wInAndRightPad = param.wIn + param.rightPad;
    int64_t hIn = param.hIn;
    int64_t wIn = param.wIn;
    int64_t outW = param.outW;
    int64_t outH = param.outH;
    __VEC_SCOPE__
    {
        using RegDstT = typename MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo>;
        RegDstT v0;
        RegDstT v1;
        RegDstT v2;
        RegDstT v3;
        RegDstT v4;

        RegDstT vd0;
        RegDstT vd1;
        RegDstT vd2;
        RegDstT vd3;

        MicroAPI::RegTensor<float32_t> dWindow;
        MicroAPI::RegTensor<float32_t> hWindow;
        MicroAPI::RegTensor<float32_t> wWindow;
        MicroAPI::RegTensor<float32_t> res;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(v1, outW, p0);
        MicroAPI::Duplicate(v2, outH, p0);
    
        uint32_t sreg = oneBatchOut;
        for (uint16_t i = 0; i < loopNum; i++)
        {
            MicroAPI::Arange(v0, i * oneRegLength + start);
            MicroAPI::AddrReg resOffset = MicroAPI::CreateAddrReg<float32_t>(i, oneRegLength);
            MicroAPI::MaskReg pWrite = MicroAPI::UpdateMask<float32_t>(sreg);
            MicroAPI::Div(vd1, v0, v1, p0);    // i / outw 
            CalcWindowSize<int64_t, countIncludePad>(hWindow, vd1, kH, sH, negTopPad, hIn, hInAndBottomPad, p0);
            MicroAPI::DataCopy(dstAddr, hWindow, resOffset, pWrite);             
        }
        uint32_t sreg1 = oneBatchOut;
        for (uint16_t i = 0; i < loopNum; i++)
        {
            MicroAPI::Arange(v0, i * oneRegLength + start);
            MicroAPI::AddrReg resOffset = MicroAPI::CreateAddrReg<float32_t>(i, oneRegLength);
            MicroAPI::MaskReg pWrite = MicroAPI::UpdateMask<float32_t>(sreg1);
            MicroAPI::Div(vd3, v0, v1, p0);    // i / outw 
            MicroAPI::Mul(vd3, vd3, v1, p0);   // (i / outw * outw)
            MicroAPI::Sub(vd3, v0, vd3, p0);   // i % outhw  
     
            CalcWindowSize<int64_t, countIncludePad>(wWindow, vd3, kW, sW, negLeftPad, wIn, wInAndRightPad, p0);
            MicroAPI::DataCopy(res, dstAddr, resOffset);
            MicroAPI::Mul(res, res, wWindow, p0);
            MicroAPI::DataCopy(dstAddr, res, resOffset, pWrite);
        }
    } 

    if  (PAD_MULTI_BATCH && (oneBatchOut < total)) {
        uint32_t diff = (total - oneBatchOut);
        uint16_t loopNum = diff / oneBatchOut;
        auto startAddr = dstAddr;
        auto writeAddr = dstAddr + oneBatchOut;
        if (oneBatchOut < oneRegLength) {
            uint32_t repeatElm = oneBatchOut;
            __VEC_SCOPE__
            {
                auto curDstAddr = writeAddr;
                MicroAPI::UnalignReg u0;
                MicroAPI::RegTensor<float32_t> v0;
                MicroAPI::DataCopy(v0, startAddr);
                for (uint16_t k = 0; k < loopNum; k++) {
                    MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);  
            }
        } else {
            uint32_t repeatElm = oneRegLength;
            uint16_t loopInner = oneBatchOut / oneRegLength;
            uint16_t tailInner = oneBatchOut - loopInner * oneRegLength;
            if (tailInner == 0) {
                loopInner -= 1;
                tailInner = oneRegLength;
            }
            __VEC_SCOPE__
            {
                auto curDstAddr = writeAddr;
                MicroAPI::UnalignReg u0, u1;
                MicroAPI::RegTensor<float32_t> v0;
                MicroAPI::DataCopy(v0, startAddr);
                for (uint16_t i = 0; i < loopNum; i++) {
                    auto curSrcAddr = startAddr;
                    MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
                    for (uint16_t k = 0; k < loopInner; k++) {
                        MicroAPI::DataCopyUnAlign(v0, u0, curSrcAddr, repeatElm);
                        MicroAPI::DataCopyUnAlign(curDstAddr, v0, u1, repeatElm);
                    }
                    MicroAPI::DataCopyUnAlign(v0, u0, curSrcAddr, tailInner);
                    MicroAPI::DataCopyUnAlign(curDstAddr, v0, u1, tailInner);       
                } 
                MicroAPI::DataCopyUnAlignPost(curDstAddr, u1, 0);             
            }            
        }
    }
}

template <typename T>
__aicore__ inline void AvgPoolDivNormChannel(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, __local_mem__ float32_t* divAddr, uint32_t num, uint32_t channel=1)
{
    uint32_t oneRegChannel = Ops::Base::GetVRegSize() / sizeof(float32_t) / channel;
    uint16_t oneRegNum = oneRegChannel * channel;

    uint16_t loopNum = num  / oneRegChannel;
    uint16_t tailNum = (num - loopNum * oneRegChannel) * channel;
    if (tailNum == 0) {
        loopNum -= 1;
        tailNum = oneRegNum;
    }
    __VEC_SCOPE__
    {   
        MicroAPI::RegTensor<float32_t> src;
        MicroAPI::RegTensor<float32_t> div;
        MicroAPI::RegTensor<float32_t> tmp;
        MicroAPI::RegTensor<T> res;
        MicroAPI::RegTensor<uint32_t> index;
        MicroAPI::UnalignReg u0, u1;
        auto curDstAddr = dstAddr;
        auto curSrcAddr = srcAddr;
        uint32_t mainSreg = oneRegNum;
        uint32_t tailSreg = tailNum;
        MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float32_t>(mainSreg);
        MicroAPI::MaskReg pMaskTail = MicroAPI::UpdateMask<float32_t>(tailSreg);
        MicroAPI::Arange((MicroAPI::RegTensor<int32_t>&)index, 0);
        MicroAPI::RegTensor<uint32_t> channelDiv;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(channelDiv, channel, p0);
        MicroAPI::Div(index, index, channelDiv, p0); 
        MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
        for (uint16_t i = 0; i < loopNum; i++) {
            MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, oneRegNum); 
            MicroAPI::DataCopyGather(div, divAddr + i * oneRegChannel, index, pMask);  
            if constexpr(std::is_same<T, float32_t>::value) {
                MicroAPI::Div(res, src, div, pMask); 
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, oneRegNum); 
            } else {
                MicroAPI::Div(tmp, src, div, pMask); 
                MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, oneRegNum); 
            }
        }
        MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, tailNum); 
        MicroAPI::DataCopyGather(div, divAddr + loopNum * oneRegChannel, index, pMaskTail); 
        if constexpr(std::is_same<T, float32_t>::value) {
            MicroAPI::Div(res, src, div, pMask); 
            MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailNum); 
        } else {
            MicroAPI::Div(tmp, src, div, pMask); 
            MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
            MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
            MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailNum); 
        }
        MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);   
    } 
}

template <typename T, bool CHANNEL_BROADACAST=false>
__aicore__ inline void AvgPoolDivNorm(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, __local_mem__ float32_t* divAddr, uint32_t num, uint32_t channel=1)
{
    if constexpr (CHANNEL_BROADACAST) {
       return AvgPoolDivNormChannel(dstAddr, srcAddr, divAddr, num, channel);
    }
    uint16_t oneRegNum = Ops::Base::GetVRegSize() / sizeof(float32_t);
    uint16_t loopNum = (num + oneRegNum - 1) / oneRegNum;
    __VEC_SCOPE__
    {   
        MicroAPI::RegTensor<float32_t> src;
        MicroAPI::RegTensor<float32_t> div;
        MicroAPI::RegTensor<float32_t> tmp;
        MicroAPI::RegTensor<T> res;

        MicroAPI::UnalignReg u0;
        MicroAPI::DataCopyUnAlignPre(u0, divAddr);
        uint32_t sreg = num;
        for (uint16_t i = 0; i < loopNum; i++) {
            MicroAPI::AddrReg srcOffset = MicroAPI::CreateAddrReg<float32_t>(i, oneRegNum);
            MicroAPI::AddrReg dstOffset = MicroAPI::CreateAddrReg<T>(i, oneRegNum);
            MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float32_t>(sreg);

            MicroAPI::DataCopy(src, srcAddr, srcOffset);  
            MicroAPI::DataCopyUnAlign(div, u0, divAddr, oneRegNum);

            if constexpr(std::is_same<T, float32_t>::value) {
                MicroAPI::Div(res, src, div, pMask); 
                MicroAPI::DataCopy(dstAddr, res, dstOffset, pMask);
            } else {
                MicroAPI::Div(tmp, src, div, pMask); 
                MicroAPI::MaskReg newMask;
                MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(newMask, pMask);
                MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
                MicroAPI::DataCopy(dstAddr, res, dstOffset, newMask);
            }
        }
    } 
}

template <typename T, bool CHANNEL_BROADACAST=false>
__aicore__ inline void AvgPoolDivBatchV1(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, __local_mem__ float32_t* divAddr, uint32_t batchNum, uint32_t batchElement, uint32_t channel=1)
{
    uint32_t oneRegChannel = Ops::Base::GetVRegSize() / sizeof(float32_t) / channel;
    uint16_t oneRegNum = oneRegChannel * channel;
    uint16_t loopNum = batchElement / oneRegChannel;
    uint16_t tailNum = (batchElement - loopNum * oneRegChannel) * channel;
    uint16_t loopBatch = batchNum;
    __VEC_SCOPE__
    {   
        MicroAPI::RegTensor<float32_t> src;
        MicroAPI::RegTensor<float32_t> div;
        MicroAPI::RegTensor<float32_t> tmp;
        MicroAPI::RegTensor<T> res;
        MicroAPI::RegTensor<uint32_t> index;
        MicroAPI::UnalignReg u0, u1;
        auto curSrcAddr = srcAddr;
        auto curDstAddr = dstAddr;

        uint32_t mainSreg = oneRegNum;
        uint32_t tailSreg = tailNum;
        MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float32_t>(mainSreg);
        MicroAPI::MaskReg pMaskTail = MicroAPI::UpdateMask<float32_t>(tailSreg);        
        MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
        if constexpr(CHANNEL_BROADACAST) {
            MicroAPI::Arange((MicroAPI::RegTensor<int32_t>&)index, 0);
            MicroAPI::RegTensor<uint32_t> channelDiv;
            MicroAPI::MaskReg p0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(channelDiv, channel, p0);
            MicroAPI::Div(index, index, channelDiv, p0); 
        }
        for (uint16_t i = 0; i < loopBatch; i++)
        {
            uint32_t sreg = batchElement;
            for (uint16_t j = 0; j < loopNum; j++) {
                MicroAPI::AddrReg divOffset = MicroAPI::CreateAddrReg<float32_t>(j, oneRegNum);
                MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, oneRegNum); 
                if constexpr(CHANNEL_BROADACAST) { 
                   MicroAPI::DataCopyGather(div, divAddr + j * oneRegChannel, index, pMask); 
                } else {
                    MicroAPI::DataCopy(div, divAddr, divOffset); 
                } 
                if constexpr(std::is_same<T, float32_t>::value) {
                    MicroAPI::Div(res, src, div, pMask); 
                    MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, oneRegNum);  
                } else {
                    MicroAPI::Div(tmp, src, div, pMask); 
                    MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                    MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
                    MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, oneRegNum);  
                }
            }
            MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, tailNum); 
            if constexpr(CHANNEL_BROADACAST) { 
                MicroAPI::DataCopyGather(div, divAddr + loopNum * oneRegChannel, index, pMaskTail); 
            } else {
                MicroAPI::DataCopy(div, divAddr + loopNum * oneRegNum);  
            }        
            if constexpr(std::is_same<T, float32_t>::value) {
                MicroAPI::Div(res, src, div, pMask); 
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailNum);  
            } else {
                MicroAPI::Div(tmp, src, div, pMask); 
                MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailNum);  
            }
        }
        MicroAPI::DataCopyUnAlignPost(curDstAddr, u1, 0);
    } 
}

template <typename T, bool CHANNEL_BROADACAST=false>
__aicore__ inline void AvgPoolDivBatchV2(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, __local_mem__ float32_t* divAddr, uint32_t batchNum, uint32_t batchElement, uint32_t channel=1)
{
    constexpr uint16_t oneRegNum = Ops::Base::GetVRegSize() / sizeof(float32_t);
    uint16_t onceRepeatBatch = oneRegNum / (batchElement * channel);
    uint16_t loopNum = batchNum / onceRepeatBatch;
    uint16_t onceRepeatNum = onceRepeatBatch * batchElement * channel;
    uint16_t tailRepeatNum = (batchNum - loopNum * onceRepeatBatch) * batchElement * channel;
    __VEC_SCOPE__
    {   
        MicroAPI::RegTensor<float32_t> src;
        MicroAPI::RegTensor<float32_t> div;
        MicroAPI::RegTensor<float32_t> tmp;
        MicroAPI::RegTensor<T> res;
        MicroAPI::RegTensor<uint32_t> index;
        MicroAPI::UnalignReg u0, u1;
        auto curSrcAddr = srcAddr;
        auto curDstAddr = dstAddr;
        uint32_t mainSreg = onceRepeatNum;
        MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float32_t>(mainSreg);
        if constexpr(CHANNEL_BROADACAST) {
            MicroAPI::Arange((MicroAPI::RegTensor<int32_t>&)index, 0);
            MicroAPI::RegTensor<uint32_t> channelDiv;
            MicroAPI::MaskReg p0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(channelDiv, channel, p0);
            MicroAPI::Div(index, index, channelDiv, p0); 
        }
        MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
        if constexpr(CHANNEL_BROADACAST) { 
            MicroAPI::DataCopyGather(div, divAddr, index, pMask); 
        } else {
            MicroAPI::DataCopy(div, divAddr); 
        } 
        for (uint16_t i = 0; i < loopNum; i++) { 
            MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, onceRepeatNum); 
            if constexpr(std::is_same<T, float32_t>::value) {
                MicroAPI::Div(res, src, div, pMask); 
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, onceRepeatNum);  
            } else {
                MicroAPI::Div(tmp, src, div, pMask); 
                MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, onceRepeatNum);  
            }
        }
        MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, tailRepeatNum); 
        if constexpr(std::is_same<T, float32_t>::value) {
            MicroAPI::Div(res, src, div, pMask); 
            MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailRepeatNum);  
        } else {
            MicroAPI::Div(tmp, src, div, pMask); 
            MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
            MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
            MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailRepeatNum);  
        }    
        MicroAPI::DataCopyUnAlignPost(curDstAddr, u1, 0);    
    } 
}

template <typename T, bool CHANNEL_BROADACAST=false>
__aicore__ inline void AvgPoolDivMultiBatch(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, __local_mem__ float32_t* divAddr, uint32_t batchNum, uint32_t batchElement, uint32_t channel=1) {
    uint32_t oneVL = Ops::Base::GetVRegSize() / sizeof(float32_t);
    if (batchElement * channel > oneVL) {
        AvgPoolDivBatchV1<T, CHANNEL_BROADACAST>(dstAddr, srcAddr, divAddr, batchNum, batchElement, channel);
    } else {
        AvgPoolDivBatchV2<T, CHANNEL_BROADACAST>(dstAddr, srcAddr, divAddr, batchNum, batchElement, channel);
    }
}

__aicore__ inline void ComputeDivisorCommon(int64_t computeMode, __local_mem__ float* dstAddr, const CalcDivisorParam &param, int64_t start, int64_t num)   {
    switch (computeMode) {
        case 0:
            ComputeDivisorImplB32<false, false>(dstAddr, param, start, num);
            break;
        case 1:
            ComputeDivisorImplB32<false, true>(dstAddr, param, start, num);
            break;
        case 2:
            ComputeDivisorImplB32<true, false>(dstAddr, param, start, num);
            break;
        case 3:
            ComputeDivisorImplB32<true, true>(dstAddr, param, start, num);
            break;
        case 4:
            ComputeDivisorImplB64<false, false>(dstAddr, param, start, num);
            break;
        case 5:
            ComputeDivisorImplB64<false, true>(dstAddr, param, start, num);
            break;
        case 6:
            ComputeDivisorImplB64<true, false>(dstAddr, param, start, num);
            break;
        case 7:
            ComputeDivisorImplB64<true, true>(dstAddr, param, start, num);
            break;
    }
}

template <typename T>
__aicore__ inline void DuplicateReg(MicroAPI::RegTensor<T>& reg, MicroAPI::MaskReg mask)
{
    MicroAPI::Duplicate(reg, 0, mask);
}

template <typename T>
__aicore__ inline void DuplicateValue(const __local_mem__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    uint32_t num = calNum;
    MicroAPI::RegTensor<T> v0;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<T>(num);
    MicroAPI::UnalignReg u0;
    DuplicateReg<T>(v0, p0);
    __local_mem__ T* addr = (__local_mem__ T*)dstAddr + offset;
    MicroAPI::DataCopyUnAlign(addr, v0, u0, calNum);
    MicroAPI::DataCopyUnAlignPost(addr, u0, 0);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename T>
__aicore__ inline void MergeAvgParaRes(MicroAPI::RegTensor<T>& res, __local_mem__ T* dstLocalAddr, int32_t num)
{
    // merge cur result with pre result
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<T> lastRes;
    AscendC::MicroAPI::UnalignReg u0;
    auto curSrcAddr = dstLocalAddr;
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    AscendC::MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
    AscendC::MicroAPI::DataCopyUnAlign(lastRes, u0, curSrcAddr, num);
    MicroAPI::Add(res, res, lastRes, pregAll);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

template <typename T, typename RegDstT>
__aicore__ inline void LoadOneElement(const __local_mem__ void* input, RegDstT& dst, uint32_t offset)
{
    MicroAPI::UnalignReg u0;
    auto srcAddr = (__local_mem__ T*)(input) + offset;
    MicroAPI::DataCopyUnAlignPre(u0, srcAddr);
    MicroAPI::DataCopyUnAlign(dst, u0, srcAddr, 1);
}

template <typename T, typename RegDstT>
__aicore__ inline void MergeSumRes(RegDstT& res, const __local_mem__ T* dstLocalAddr, int32_t offset)
{
    // merge cur result with pre result
    MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();
    RegDstT lastRes;
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    LoadOneElement<T>(dstLocalAddr, lastRes, offset);
    MicroAPI::Add(res, res, lastRes, pregOne);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

template <bool MaskMergeMode, typename T, typename U>
__aicore__ inline void SumWithGather(MicroAPI::RegTensor<T>& res, __local_mem__ T* srcAddr,
    MicroAPI::RegTensor<U>& index, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> vd1;
    MicroAPI::DataCopyGather(vd1, srcAddr, index, mask);
    if constexpr (MaskMergeMode) {
        MicroAPI::RegTensor<T> tmp;
        MicroAPI::Add(tmp, vd1, res, mask);
        MicroAPI::Copy<T, MicroAPI::MaskMergeMode::MERGING>(res, tmp, mask);
    } else {
        MicroAPI::Add(res, vd1, res, mask);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexMultiBatch(uint32_t hFactorOut, uint32_t wFactorOut, uint32_t batchElemtsIn, uint32_t wIn,
                                                uint32_t hStride, uint32_t wStride, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    U batchElemtsOut = hFactorOut * wFactorOut;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;
        MicroAPI::RegTensor<U> v4;
        MicroAPI::RegTensor<U> v5;
        MicroAPI::RegTensor<U> v6;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd7;
        MicroAPI::RegTensor<U> vd8;
        MicroAPI::RegTensor<U> vd9;
        MicroAPI::RegTensor<U> vd10;
        MicroAPI::RegTensor<U> vd11;
        MicroAPI::RegTensor<U> vd12;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wFactorOut, p0);
        MicroAPI::Duplicate(v2, (U)wIn, p0);
        MicroAPI::Duplicate(v3, (U)hStride, p0);
        MicroAPI::Duplicate(v4, (U)wStride, p0);
        MicroAPI::Duplicate(v5, (U)batchElemtsIn, p0);
        MicroAPI::Duplicate(v6, (U)batchElemtsOut, p0);

        MicroAPI::Div(vd1, v0, v6, p0);   // i / (rows * cols)
        MicroAPI::Mul(vd2, vd1, v5, p0);  // i / (rows * cols) * batchElemtsIn
        MicroAPI::Mul(vd3, vd1, v6, p0);  // (i / wFactorOut * wIn * hStride)
        MicroAPI::Sub(vd4, v0, vd3, p0);  // i % (rows * cols)

        MicroAPI::Div(vd5, vd4, v1, p0);     // hwoffset / cols
        MicroAPI::Mul(vd6, vd5, v2, p0);     // hwoffset / cols * wIn
        MicroAPI::Mul(vd7, vd6, v3, p0);     // hwoffset / cols * wIn * hStride
        MicroAPI::Mul(vd8, vd5, v1, p0);     // hwoffset / cols * cols
        MicroAPI::Sub(vd9, vd4, vd8, p0);    // hwoffset % cols
        MicroAPI::Mul(vd10, vd9, v4, p0);    // hwoffset % cols * wStride
        MicroAPI::Add(vd11, vd7, vd10, p0);  // hwoffset / cols * wIn * hStride + hwoffset % cols * wStride
        MicroAPI::Add(vd12, vd2, vd11, p0);
        MicroAPI::DataCopy(dstAddr, vd12, p0);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexMultiRow(uint32_t wFactorOut, uint32_t wIn, uint32_t hStride, uint32_t wStride,
                                              LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    // i / wFactorOut * wIn * hStride + i % wFactorOut * wStride
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;
        MicroAPI::RegTensor<U> v4;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd7;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wFactorOut, p0);
        MicroAPI::Duplicate(v2, (U)wIn, p0);
        MicroAPI::Duplicate(v3, (U)hStride, p0);
        MicroAPI::Duplicate(v4, (U)wStride, p0);

        MicroAPI::Div(vd1, v0, v1, p0);    // i / wFactorOut
        MicroAPI::Mul(vd2, vd1, v2, p0);   // (i / wFactorOut * wIn)
        MicroAPI::Mul(vd3, vd2, v3, p0);   // (i / wFactorOut * wIn * hStride)
        MicroAPI::Mul(vd4, vd1, v1, p0);   // (i / wFactorOut * wFactorOut)
        MicroAPI::Sub(vd5, v0, vd4, p0);   // i % wFactor
        MicroAPI::Mul(vd6, vd5, v4, p0);   // i % wFactorOut * wStride
        MicroAPI::Add(vd7, vd3, vd6, p0);  // (i / wFactorOut * wIn * hStride + i % wFactorOut * wStride)
        MicroAPI::DataCopy(dstAddr, vd7, p0);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexSingleRow(uint32_t wStride, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    // i * wStride
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wStride, p0);
        MicroAPI::Mul(vd0, v0, v1, p0);  // (i / wFactorOut * wIn)
        MicroAPI::DataCopy(dstAddr, vd0, p0);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexSingleKernel(uint32_t wIn, uint32_t kW, uint32_t kH,
                                              LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    uint16_t repeatNum = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t loopNum = (kW * kH + repeatNum - 1) / repeatNum;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++)
        {
            MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, i * repeatNum);
            MicroAPI::Duplicate(v1, (U)kW, p0);
            MicroAPI::Duplicate(v2, (U)wIn, p0);

            MicroAPI::Div(vd1, v0, v1, p0);
            MicroAPI::Mul(vd2, vd1, v2, p0);
            MicroAPI::Mul(vd3, vd1, v1, p0);
            MicroAPI::Sub(vd4, v0, vd3, p0);
            MicroAPI::Add(vd5, vd2, vd4, p0);
            MicroAPI::DataCopy(dstAddr  + i * repeatNum, vd5, p0);
        }
    }
}

template <uint16_t REG_NUM, uint16_t IDX, typename U>
__aicore__ inline void LoadIndex(__local_mem__ U* indexAddr, MicroAPI::RegTensor<U>& index)
{
    constexpr uint32_t repeatNum = Ops::Base::GetVRegSize() / sizeof(U);
    if constexpr (REG_NUM > IDX) {
        MicroAPI::DataCopy(index, indexAddr + IDX * repeatNum);
    }
}

template <typename T, typename Z>
__aicore__ inline void DivCompute(MicroAPI::RegTensor<T>& res, MicroAPI::RegTensor<Z>& sum, float32_t divisor)
{
    MicroAPI::RegTensor<Z> divisorReg;
    uint32_t scalar = 1;
    if constexpr (sizeof(T) == TWO) {
        // B16类型
        MicroAPI::RegTensor<Z> divRes;
        MicroAPI::Duplicate(divisorReg, divisor);
        MicroAPI::MaskReg divMask = MicroAPI::UpdateMask<Z>(scalar);
        MicroAPI::Div(divRes, sum, divisorReg, divMask);

        // 将RegTensor<Z>(即RegTensor<float32_t>类型)转为RegTensor<B16>类型。
        scalar = 1;
        MicroAPI::MaskReg castMask = MicroAPI::UpdateMask<T>(scalar);
        MicroAPI::Cast<T, Z, castTraitB32ToB16>(res, divRes, castMask);

    } else {
        // B32类型, 此处即float32类型
        MicroAPI::Duplicate(divisorReg, divisor);
        MicroAPI::MaskReg divMask = MicroAPI::UpdateMask<Z>(scalar);
        MicroAPI::Div(res, sum, divisorReg, divMask);
    }
}

template <typename T, typename U, typename Z>
__aicore__ inline void ReduceSumWithGatherOne(MicroAPI::RegTensor<Z>& res, __local_mem__ T* srcAddr,
    MicroAPI::RegTensor<U>& index, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> vd1;
    if constexpr (sizeof(T) == TWO) {
        // B16类型需转换为更高精度类型，防止溢出和精度丢失
        MicroAPI::DataCopyGather(vd1, srcAddr, index, mask);

        // B16类型转为float32, 此处Z为float32类型
        MicroAPI::RegTensor<Z> low;
        MicroAPI::RegTensor<Z> left;
        MicroAPI::RegTensor<Z> right;
        MicroAPI::RegTensor<T> tmpLeft;
        MicroAPI::RegTensor<T> tmpRight;
        MicroAPI::RegTensor<T> zero;
        MicroAPI::Duplicate(zero, (T)0);
        MicroAPI::Interleave(tmpLeft, tmpRight,  vd1, zero);
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<Z, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Cast<Z, T, castTraitB16ToB32>(left, tmpLeft, maskAll);
        MicroAPI::Cast<Z, T, castTraitB16ToB32>(right, tmpRight, maskAll);

        MicroAPI::Add(low, left, right, maskAll);
        MicroAPI::ReduceSum(res, low, maskAll);
    } else {
        // B32类型
        MicroAPI::DataCopyGather(vd1, srcAddr, index, mask);
        
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::ReduceSum(res, vd1, maskAll);
    }
}

template <typename T, typename U, typename Z>
__aicore__ inline void ReduceSumWithGather(MicroAPI::RegTensor<Z>& res, __local_mem__ T* srcAddr,
    MicroAPI::RegTensor<U>& index, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> vd1;
    if constexpr (sizeof(T) == TWO) {
        // B16类型需转换为更高精度类型，防止溢出和精度丢失
        MicroAPI::DataCopyGather(vd1, srcAddr, index, mask);

        // B16类型转为float32, 此处Z为float32类型
        MicroAPI::RegTensor<Z> low;
        MicroAPI::RegTensor<Z> left;
        MicroAPI::RegTensor<Z> right;
        MicroAPI::RegTensor<T> tmpLeft;
        MicroAPI::RegTensor<T> tmpRight;
        MicroAPI::RegTensor<T> zero;
        MicroAPI::Duplicate(zero, (T)0);
        MicroAPI::Interleave(tmpLeft, tmpRight,  vd1, zero);
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<Z, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Cast<Z, T, castTraitB16ToB32>(left, tmpLeft, maskAll);
        MicroAPI::Cast<Z, T, castTraitB16ToB32>(right, tmpRight, maskAll);

        MicroAPI::Add(low, left, right, maskAll);
        MicroAPI::Add(res, low, res, maskAll);
    } else {
        // B32类型
        MicroAPI::DataCopyGather(vd1, srcAddr, index, mask);

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Add(res, res, vd1, maskAll);
    }
}

template <uint16_t REG_NUM, uint16_t IDX, typename U, typename T, typename Z>
__aicore__ inline void ComputeReduceSumWithGather(MicroAPI::RegTensor<Z>&res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index,
    MicroAPI::MaskReg& mask)
{
    if constexpr (REG_NUM > IDX) {
        ReduceSumWithGather<T, U, Z>(res, srcAddr, index, mask);
    }
}

template <typename T, typename U, uint16_t REG_NUM>
__aicore__ inline void AvgPoolSingleKernelCommon(__local_mem__ T* dstLocalAddr, __local_mem__ T* xLocalAddr,
                                     __local_mem__ U* indexAddr, uint16_t loopN,
                                     uint16_t loopH, uint16_t loopW, U oneChannelElements, U oneLoopStrideH,
                                     U oneLoopStrideW,
                                     uint16_t tailLoopElements, float32_t divisor)
{
    if constexpr (sizeof(T) == sizeof(int64_t) && REG_NUM > INT64_MAXREGNUM) {
        return;
    }

    using Z = typename std::conditional<sizeof(T) == B16, float32_t, T>::type;
    __VEC_SCOPE__
        {
            MicroAPI::RegTensor<U> index[SIXTEEN];
            MicroAPI::UnalignReg u0;
            uint32_t tailNum = tailLoopElements;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailNum);

            MicroAPI::DataCopy(index[0], indexAddr);
            LoadIndex<REG_NUM, ONE>(indexAddr, index[ONE]);
            LoadIndex<REG_NUM, TWO>(indexAddr, index[TWO]);
            LoadIndex<REG_NUM, THREE>(indexAddr, index[THREE]);
            LoadIndex<REG_NUM, FOUR>(indexAddr, index[FOUR]);
            LoadIndex<REG_NUM, FIVE>(indexAddr, index[FIVE]);
            LoadIndex<REG_NUM, SIX>(indexAddr, index[SIX]);
            LoadIndex<REG_NUM, SEVEN>(indexAddr, index[SEVEN]);
            LoadIndex<REG_NUM, EIGHT>(indexAddr, index[EIGHT]);
            LoadIndex<REG_NUM, NINE>(indexAddr, index[NINE]);
            LoadIndex<REG_NUM, TEN>(indexAddr, index[TEN]);
            LoadIndex<REG_NUM, ELEVEN>(indexAddr, index[ELEVEN]);
            LoadIndex<REG_NUM, TWELVE>(indexAddr, index[TWELVE]);
            LoadIndex<REG_NUM, THIRTEEN>(indexAddr, index[THIRTEEN]);
            LoadIndex<REG_NUM, FOURTEEN>(indexAddr, index[FOURTEEN]);
            LoadIndex<REG_NUM, FIFTEEN>(indexAddr, index[FIFTEEN]);
            __local_mem__ T* dstAddr = dstLocalAddr;
            for (uint16_t i = 0; i < loopN; i++) {
                __local_mem__ T* srcAddr = xLocalAddr + i * oneChannelElements;
                for (uint16_t j = 0; j < loopH; j++) {
                    __local_mem__ T* srcAddrH = srcAddr + j * oneLoopStrideH;
                    for (uint16_t k = 0; k < loopW; k++) {
                        __local_mem__ T* srcAddrW = srcAddrH + k * oneLoopStrideW;
                        MicroAPI::RegTensor<T> res;
                        MicroAPI::RegTensor<Z> reduceSumRes;
                        MicroAPI::RegTensor<Z> sum;
                        MicroAPI::Duplicate(sum, (Z)0);

                        if constexpr (REG_NUM == 1) {
                            ReduceSumWithGatherOne<T, U, Z>(sum, srcAddrW, index[0], pTail);
                        } else {
                            ReduceSumWithGather<T, U, Z>(sum, srcAddrW, index[0], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, TWO>(sum, srcAddrW, index[ONE], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, THREE>(sum, srcAddrW, index[TWO], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, FOUR>(sum, srcAddrW, index[THREE], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, FIVE>(sum, srcAddrW, index[FOUR], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, SIX>(sum, srcAddrW, index[FIVE], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, SEVEN>(sum, srcAddrW, index[SIX], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, EIGHT>(sum, srcAddrW, index[SEVEN], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, NINE>(sum, srcAddrW, index[EIGHT], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, TEN>(sum, srcAddrW, index[NINE], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, ELEVEN>(sum, srcAddrW, index[TEN], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, TWELVE>(sum, srcAddrW, index[ELEVEN], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, THIRTEEN>(sum, srcAddrW, index[TWELVE], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, FOURTEEN>(sum, srcAddrW, index[THIRTEEN], maskAll);
                            ComputeReduceSumWithGather<REG_NUM, FIFTEEN>(sum, srcAddrW, index[FOURTEEN], maskAll);
                            ReduceSumWithGather<T, U, Z>(sum, srcAddrW, index[REG_NUM-1], pTail);
                            MicroAPI::ReduceSum(sum, sum, maskAll);
                        }
                        DivCompute(res, sum, divisor);

                        uint32_t elementCount = 1;
                        MicroAPI::DataCopyUnAlign(dstAddr, res, u0, elementCount);
                    }
                }
            }
            MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
        }
}

template <typename T, typename U>
__aicore__ inline void AvgPoolSingleKernelDefault(__local_mem__ T* dstLocalAddr, __local_mem__ T* xLocalAddr,
                                     __local_mem__ U* indexAddr, uint16_t loopN,
                                     uint16_t loopH, uint16_t loopW, U oneChannelElements, U oneLoopStrideH,
                                     U oneLoopStrideW,
                                     float32_t divisor, uint16_t regNum, uint16_t kernelSize)
{
    using Z = typename std::conditional<sizeof(T) == B16, float32_t, T>::type;
    __VEC_SCOPE__
        {
            MicroAPI::RegTensor<U> index;
            MicroAPI::UnalignReg u0;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
            
            __local_mem__ T* dstAddr = dstLocalAddr;
            for (uint16_t i = 0; i < loopN; i++) {
                __local_mem__ T* srcAddr = xLocalAddr + i * oneChannelElements;
                for (uint16_t j = 0; j < loopH; j++) {
                    __local_mem__ T* srcAddrH = srcAddr + j * oneLoopStrideH;
                    for (uint16_t k = 0; k < loopW; k++) {
                        __local_mem__ T* srcAddrW = srcAddrH + k * oneLoopStrideW;
                        MicroAPI::RegTensor<T> res;
                        MicroAPI::RegTensor<Z> sum;
                        MicroAPI::Duplicate(sum, (Z)0);

                        uint32_t tailNum = kernelSize;
                        for (uint16_t m = 0; m < regNum; m++) {
                            constexpr uint32_t repeatNum = Ops::Base::GetVRegSize() / sizeof(U);
                            MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailNum);
                            MicroAPI::DataCopy(index, indexAddr + m * repeatNum);
                            ReduceSumWithGather<T, U, Z>(sum, srcAddrW, index, pTail);
                        }
                        MicroAPI::ReduceSum(sum, sum, maskAll);
                        DivCompute(res, sum, divisor);

                        uint32_t elementCount = 1;
                        MicroAPI::DataCopyUnAlign(dstAddr, res, u0, elementCount);
                    }
                }
            }
            MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
        }
}

} // AvgPool

#endif  // AVG_POOL_COMMON_H_