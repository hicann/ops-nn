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
#include "pool_utils/arch35/index/pool_2d_gather_scatter_index.h"
#include "pool_utils/arch35/compute/pool_fast_div.h"
#include "pool_utils/pool_type_traits.h"
#include "pool_utils/arch35/data_move/pool_reg_element_data_move.h"

namespace AvgPool {
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
    using type = typename std::conditional<
        std::is_same<T, int8_t>::value, int16_t,
        typename std::conditional<std::is_same<T, uint8_t>::value, uint16_t, T>::type>::type;
};

template <typename T>
struct IndexTypeGet {
    using type = typename std::conditional<sizeof(T) == B8 || sizeof(T) == B16, uint16_t, uint32_t>::type;
};

constexpr Reg::CastTrait castTraitB16ToB32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait castTraitB32ToB16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr Reg::DivSpecificMode divHighPrecisionMode = {Reg::MaskMergeMode::ZEROING, true,
                                                       DivAlgo::PRECISION_0ULP_FTZ_TRUE};

constexpr AscendC::Reg::CastTrait CAST_INT32_TO_FP32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait CAST_INT64_TO_FP32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};

constexpr Reg::CastTrait castTraitT2Fp32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::UNKNOWN};

constexpr Reg::CastTrait castTraitFp322T = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::CAST_RINT};

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
    curOrigin = paramsInfo.s * paramsInfo.o - paramsInfo.pl; // left
    int64_t leftInvaild = 0;
    if (curOrigin < 0) {
        leftInvaild = -curOrigin; // 0 左侧有几个无效k
    }
    // min(in - origin - leftinvaild, k)
    curk = min(paramsInfo.in - curOrigin - leftInvaild, paramsInfo.k - leftInvaild);
    // min (in + pr - origin, k)
    curkWithPad = min(paramsInfo.in + paramsInfo.pr - curOrigin, paramsInfo.k);
    curOrigin += leftInvaild; // 矫正到curOrigin +轴位置
}

template <typename T>
__aicore__ inline void CustomDuplicate(__ubuf__ T* dstAddr, uint32_t calNum, uint16_t loop)
{
    uint32_t sreg = calNum;
    Reg::RegTensor<T> v0;
    Reg::Duplicate(v0, (T)0);
    constexpr uint16_t repeatElm = Ops::Base::GetVRegSize() / sizeof(T);
    for (uint16_t i = 0; i < loop; i++) {
        Reg::MaskReg preg = Reg::UpdateMask<T>(sreg);
        Reg::AddrReg offset = Reg::CreateAddrReg<T>(i, repeatElm);
        Reg::StoreAlign(dstAddr, v0, offset, preg);
    }
}

template <typename T>
__aicore__ inline void CustomCopy(const __ubuf__ T* dstAddr, const __ubuf__ T* srcAddr, uint32_t srcBatchStride,
                                  uint32_t srcRowStride, uint32_t dstBatchStride, uint32_t dstRowStride,
                                  uint32_t dstRowOffset, uint32_t dstColOffset, uint16_t batch, uint16_t rows,
                                  uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm)
{
    Reg::RegTensor<T> v0;
    Reg::UnalignRegForStore u0;

    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t j = 0; j < rows; j++) {
            __ubuf__ T* curSrcAddr = (__ubuf__ T*)srcAddr + i * srcBatchStride + j * srcRowStride;
            __ubuf__ T* curDstAddr = (__ubuf__ T*)dstAddr + i * dstBatchStride + (j + dstRowOffset) * dstRowStride +
                                     dstColOffset;
            for (uint16_t k = 0; k < loopCols; k++) {
                Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                Reg::StoreUnAlign(curDstAddr, v0, u0, repeatElm);
            }
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
            Reg::StoreUnAlign(curDstAddr, v0, u0, tailCols);
            Reg::StoreUnAlignPost(curDstAddr, u0, 0);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterSingleRow(const __ubuf__ T* dstAddr, const __ubuf__ T* srcAddr,
                                                    uint16_t srcBatchStride, uint16_t srcRowStride,
                                                    uint16_t dstBatchStride, uint16_t dstRowStride,
                                                    uint16_t dstRowOffset, uint16_t dstColOffset, uint16_t batch,
                                                    uint16_t rows, uint16_t loopCols, uint16_t cols, uint16_t repeatElm)
{
    Reg::RegTensor<T> v0;
    Reg::RegTensor<U> sIndex;
    Reg::MaskReg preg;
    using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
    Reg::Arange((Reg::RegTensor<regType>&)sIndex, 0);
    auto dstAddr1 = (__ubuf__ T*)dstAddr + dstRowOffset * dstRowStride + dstColOffset;
    for (uint16_t i = 0; i < batch; i++) {
        auto dstAddr2 = dstAddr1 + i * dstBatchStride;
        auto srcAddr1 = (__ubuf__ T*)srcAddr + i * srcBatchStride;
        uint32_t sreg = cols;
        for (uint16_t j = 0; j < loopCols; j++) {
            auto curDstAddr = dstAddr2 + j * repeatElm;
            auto curSrcAddr = srcAddr1 + j * repeatElm;
            preg = Reg::UpdateMask<U>(sreg);
            for (uint16_t k = 0; k < rows; k++) {
                Reg::LoadAlign(v0, curSrcAddr + k * srcRowStride);
                Reg::Scatter(curDstAddr + k * dstRowStride, v0, sIndex, preg);
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterMultiRows(const __ubuf__ T* dstAddr, const __ubuf__ T* srcAddr,
                                                    Reg::RegTensor<U> index, uint32_t srcBatchStride,
                                                    uint32_t srcRowStride, uint32_t dstBatchStride,
                                                    uint32_t dstRowStride, uint32_t dstOffset, uint16_t batch,
                                                    uint16_t loopRows, uint32_t repeatElm, uint32_t tailElm)
{
    Reg::RegTensor<T> vd1;
    Reg::RegTensor<U> v1, v2, v3, v4;
    Reg::RegTensor<U> gIndex;
    uint32_t sreg = repeatElm;
    uint32_t tailSreg = tailElm;
    Reg::MaskReg maskAll = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
    Reg::MaskReg preg;
    Reg::MaskReg tailPreg;
    preg = Reg::UpdateMask<U>(sreg);
    tailPreg = Reg::UpdateMask<U>(tailSreg);
    using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
    Reg::RegTensor<U> vd0;
    Reg::Arange((Reg::RegTensor<regType>&)gIndex, 0);
    __ubuf__ T* curDstAddr = (__ubuf__ T*)dstAddr + dstOffset;
    for (uint16_t i = 0; i < batch; i++) {
        Reg::Adds(v1, index, i * dstBatchStride, maskAll);
        Reg::Adds(v3, gIndex, i * srcBatchStride, maskAll);
        for (uint16_t j = 0; j < loopRows; j++) {
            Reg::Adds(v2, v1, j * dstRowStride, preg);
            Reg::Adds(v4, v3, j * srcRowStride, preg);
            Reg::Gather(vd1, srcAddr, v4, preg);
            Reg::Scatter(curDstAddr, vd1, v2, preg);
        }
        Reg::Adds(v2, v1, loopRows * dstRowStride, tailPreg);
        Reg::Adds(v4, v3, loopRows * srcRowStride, tailPreg);
        Reg::Gather(vd1, srcAddr, v4, tailPreg);
        Reg::Scatter(curDstAddr, vd1, v2, tailPreg);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPoolB32Impl(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index, uint16_t kH,
                                      uint16_t kW, U rowStrideInub, float32_t divisor, Reg::MaskReg& pMask,
                                      uint16_t channels = 1)
{
    RegDstT vd1;
    Reg::RegTensor<U> v0;
    Reg::RegTensor<U> v1;
    Reg::RegTensor<float32_t> divisorReg;

    Reg::Duplicate(res, (T)0);
    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        Reg::Adds(v0, index, hIdx * rowStrideInub, pMask);
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            Reg::Adds(v1, v0, wIdx * channels, pMask);
            Reg::Gather(vd1, srcAddr, v1, pMask);
            Reg::Add(res, vd1, res, pMask);
        }
    }
    if constexpr (!NO_DIV) {
        Reg::Duplicate(divisorReg, divisor);
        Reg::Div<float32_t, &divHighPrecisionMode>(res, res, divisorReg, pMask);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPoolB16Impl(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index, uint16_t kH,
                                      uint16_t kW, U rowStrideInub, float32_t divisor, Reg::MaskReg& pMask,
                                      uint16_t channels = 1)
{
    Reg::RegTensor<T> vd1;
    Reg::RegTensor<T> zero;
    Reg::RegTensor<U> v0;
    Reg::RegTensor<U> v1;
    Reg::RegTensor<float32_t> tmpRes1;
    Reg::RegTensor<float32_t> tmpRes2;
    Reg::RegTensor<float32_t> left;
    Reg::RegTensor<float32_t> right;
    Reg::RegTensor<float32_t> divisorReg;
    Reg::RegTensor<T> tmpLeft;
    Reg::RegTensor<T> tmpRight;
    Reg::Duplicate(tmpRes1, (float32_t)0);
    Reg::Duplicate(tmpRes2, (float32_t)0);
    Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Duplicate((Reg::RegTensor<float16_t>&)zero, (float16_t)0);
    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        Reg::Adds(v0, index, hIdx * rowStrideInub, pMask);
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            Reg::Adds(v1, v0, wIdx * channels, pMask);
            Reg::Gather(vd1, srcAddr, v1, pMask);
            Reg::Interleave(tmpLeft, tmpRight, vd1, zero);
            Reg::Cast<float32_t, T, castTraitB16ToB32>(left, tmpLeft, maskAll);
            Reg::Cast<float32_t, T, castTraitB16ToB32>(right, tmpRight, maskAll);
            Reg::Add(tmpRes1, tmpRes1, left, maskAll);
            Reg::Add(tmpRes2, tmpRes2, right, maskAll);
        }
    }
    if constexpr (NO_DIV) {
        Reg::Move((Reg::RegTensor<float32_t>&)res.reg[0], tmpRes1);
        Reg::Move((Reg::RegTensor<float32_t>&)res.reg[1], tmpRes2);
    } else {
        Reg::Duplicate(divisorReg, divisor);
        Reg::Div(tmpRes1, tmpRes1, divisorReg, maskAll);
        Reg::Div(tmpRes2, tmpRes2, divisorReg, maskAll);
        Reg::Cast<T, float32_t, castTraitB32ToB16>(tmpLeft, tmpRes1, maskAll);
        Reg::Cast<T, float32_t, castTraitB32ToB16>(tmpRight, tmpRes2, maskAll);
        Reg::DeInterleave(res, zero, tmpLeft, tmpRight);
    }
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannelB32(__ubuf__ M* dstLocalAddr, __ubuf__ M* srcLocalAddr, uint16_t kH,
                                               uint16_t kW, U rowStrideInUb, uint16_t alignChannels,
                                               uint16_t repeatElms, float32_t divisor)
{
    Reg::RegTensor<M> res;
    Reg::RegTensor<M> vd0;
    Reg::RegTensor<M> divRegs;
    uint32_t num = repeatElms;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::UnalignRegForLoad u0;
    __ubuf__ M* curSrcAddr = srcLocalAddr;

    Reg::Duplicate(res, (float32_t)0);

    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            auto aReg = Reg::CreateAddrReg<U>(hIdx, rowStrideInUb, wIdx, alignChannels);
            Reg::LoadAlign(vd0, curSrcAddr, aReg);
            Reg::Add(res, vd0, res, p0);
        }
    }
    Reg::Duplicate(divRegs, divisor);
    Reg::Div<M, &divHighPrecisionMode>(res, res, divRegs, p0);
    Reg::StoreAlign(dstLocalAddr, res, p0);
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannelB16(__ubuf__ M* dstLocalAddr, __ubuf__ M* srcLocalAddr, uint16_t kH,
                                               uint16_t kW, U rowStrideInUb, uint16_t alignChannels,
                                               uint16_t repeatElms, float32_t divisor)
{
    Reg::RegTensor<M> res;
    Reg::RegTensor<M> vd0;
    Reg::RegTensor<M> zero;
    Reg::RegTensor<float32_t> tmpRes1;
    Reg::RegTensor<float32_t> tmpRes2;
    Reg::RegTensor<float32_t> left;
    Reg::RegTensor<float32_t> right;
    Reg::RegTensor<float32_t> divisorReg;
    Reg::RegTensor<M> tmpLeft;
    Reg::RegTensor<M> tmpRight;

    uint32_t num = repeatElms;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::UnalignRegForLoad u0;
    __ubuf__ M* curSrcAddr = srcLocalAddr;
    Reg::MaskReg defaultMask = Reg::CreateMask<M, Reg::MaskPattern::ALL>();

    Reg::Duplicate((Reg::RegTensor<float16_t>&)zero, (float16_t)0);
    Reg::Duplicate(res, (float32_t)0);

    Reg::Duplicate(tmpRes1, (float32_t)0);
    Reg::Duplicate(tmpRes2, (float32_t)0);

    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            auto aReg = Reg::CreateAddrReg<U>(hIdx, rowStrideInUb, wIdx, alignChannels);
            Reg::LoadAlign(vd0, curSrcAddr, aReg);
            Reg::Interleave(tmpLeft, tmpRight, vd0, zero);
            Reg::Cast<float32_t, M, castTraitB16ToB32>(left, tmpLeft, defaultMask);
            Reg::Cast<float32_t, M, castTraitB16ToB32>(right, tmpRight, defaultMask);
            Reg::Add(tmpRes1, tmpRes1, left, defaultMask);
            Reg::Add(tmpRes2, tmpRes2, right, defaultMask);
        }
    }

    Reg::Duplicate(divisorReg, divisor);
    Reg::Div(tmpRes1, tmpRes1, divisorReg, defaultMask);
    Reg::Div(tmpRes2, tmpRes2, divisorReg, defaultMask);
    Reg::Cast<M, float32_t, castTraitB32ToB16>(tmpLeft, tmpRes1, defaultMask);
    Reg::Cast<M, float32_t, castTraitB32ToB16>(tmpRight, tmpRes2, defaultMask);
    Reg::DeInterleave(res, zero, tmpLeft, tmpRight);
    Reg::StoreAlign(dstLocalAddr, res, p0);
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannel(__ubuf__ M* dstLocalAddr, __ubuf__ M* srcLocalAddr, uint16_t kH,
                                            uint16_t kW, U rowStrideInUb, uint16_t alignChannels, uint16_t repeatElms,
                                            float32_t divisor)
{
    if constexpr (sizeof(M) == TWO) {
        AvgPoolSingleChannelB16<M, U>(dstLocalAddr, srcLocalAddr, kH, kW, rowStrideInUb, alignChannels, repeatElms,
                                      divisor);
    } else {
        AvgPoolSingleChannelB32<M, U>(dstLocalAddr, srcLocalAddr, kH, kW, rowStrideInUb, alignChannels, repeatElms,
                                      divisor);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPoolImpl(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index, uint16_t kH,
                                   uint16_t kW, U rowStrideInub, float32_t divisor, Reg::MaskReg& pMask,
                                   uint16_t channels = 1)
{
    if constexpr (sizeof(T) == TWO) {
        AvgPoolB16Impl<T, U, NO_DIV>(res, srcAddr, index, kH, kW, rowStrideInub, divisor, pMask, channels);
    } else {
        AvgPoolB32Impl<T, U, NO_DIV>(res, srcAddr, index, kH, kW, rowStrideInub, divisor, pMask, channels);
    }
}

template <typename T, typename U, typename Z, bool NO_DIV = false>
__aicore__ inline void AvgPoolSplitW(__ubuf__ Z* dstLocalAddr, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                     uint16_t kH, uint16_t kW, uint16_t loopH, uint16_t loopW, U oneLoopStrideH,
                                     U oneLoopStrideW, U rowStrideInub, uint16_t oneLoopElements,
                                     uint16_t tailLoopElements, U halfLoopOut0, U halfLoopOut1, U tailHalfLoopOut0,
                                     U tailHalfLoopOut1, float32_t divisor, uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<sizeof(T) == B16 && std::is_same<Z, float32_t>::value,
                                              Reg::RegTensor<Z, Reg::RegTraitNumTwo>, Reg::RegTensor<T>>::type;
    RegDstT res;
    Reg::RegTensor<U> v0;
    Reg::RegTensor<U> v1;
    Reg::RegTensor<U> v2;
    Reg::RegTensor<U> v3;
    Reg::RegTensor<U> v4;
    Reg::UnalignRegForStore u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::MaskReg pTail = Reg::UpdateMask<U>(tailNum);
    __ubuf__ Z* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopH; i++) {
        Reg::Adds(v0, index, i * oneLoopStrideH, p0);
        for (uint16_t j = 0; j < loopW; j++) {
            Reg::Adds(v2, v0, j * oneLoopStrideW, p0);
            AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v2, kH, kW, rowStrideInub, divisor, p0, channels);
            if constexpr (sizeof(T) == B16 && std::is_same<Z, float32_t>::value) {
                Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[0], u0, halfLoopOut0);
                Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[1], u0, halfLoopOut1);
            } else {
                Reg::StoreUnAlign(dstAddr, res, u0, oneLoopElements);
            }
        }
        Reg::Adds(v2, v0, loopW * oneLoopStrideW, pTail);
        AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v2, kH, kW, rowStrideInub, divisor, pTail, channels);
        if constexpr (sizeof(T) == B16 && std::is_same<Z, float32_t>::value) {
            Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[0], u0, tailHalfLoopOut0);
            Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[1], u0, tailHalfLoopOut1);
        } else {
            Reg::StoreUnAlign(dstAddr, res, u0, tailLoopElements);
        }
    }
    Reg::StoreUnAlignPost(dstAddr, u0, 0);
}

template <typename T, typename U, typename Z, bool NO_DIV = false>
__aicore__ inline void AvgPoolSplitH(__ubuf__ Z* dstLocalAddr, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                     uint16_t kH, uint16_t kW, uint16_t loopN, uint16_t loopH, U oneChannelElements,
                                     U rowStrideInub, U oneLoopStride, uint16_t oneLoopElements,
                                     uint16_t tailLoopElements, U halfLoopOut0, U halfLoopOut1, U tailHalfLoopOut0,
                                     U tailHalfLoopOut1, float32_t divisor, uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<sizeof(T) == B16 && std::is_same<Z, float32_t>::value,
                                              Reg::RegTensor<Z, Reg::RegTraitNumTwo>, Reg::RegTensor<T>>::type;
    RegDstT res;
    Reg::RegTensor<U> v1;
    Reg::RegTensor<U> v2;
    Reg::RegTensor<U> v3;
    Reg::RegTensor<U> v4;
    Reg::UnalignRegForStore u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::MaskReg pTail = Reg::UpdateMask<U>(tailNum);
    __ubuf__ Z* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopN; i++) {
        Reg::Adds(v1, index, i * oneChannelElements, p0);
        for (uint16_t j = 0; j < loopH; j++) {
            Reg::Adds(v2, v1, j * oneLoopStride, p0);
            AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v2, kH, kW, rowStrideInub, divisor, p0, channels);
            if constexpr (sizeof(T) == B16 && std::is_same<Z, float32_t>::value) {
                Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[0], u0, halfLoopOut0);
                Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[1], u0, halfLoopOut1);
            } else {
                Reg::StoreUnAlign(dstAddr, res, u0, oneLoopElements);
            }
        }
        Reg::Adds(v2, v1, loopH * oneLoopStride, pTail);
        AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v2, kH, kW, rowStrideInub, divisor, pTail, channels);
        if constexpr (sizeof(T) == B16 && std::is_same<Z, float32_t>::value) {
            Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[0], u0, tailHalfLoopOut0);
            Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[1], u0, tailHalfLoopOut1);
        } else {
            Reg::StoreUnAlign(dstAddr, res, u0, tailLoopElements);
        }
    }
    Reg::StoreUnAlignPost(dstAddr, u0, 0);
}

template <typename T, typename U, typename Z, bool NO_DIV = false>
__aicore__ inline void AvgPoolSplitBatch(__ubuf__ Z* dstLocalAddr, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                         uint16_t kH, uint16_t kW, uint16_t loopN, U rowStrideInub, U oneLoopStride,
                                         uint16_t oneLoopElements, uint16_t tailLoopElements, U halfLoopOut0,
                                         U halfLoopOut1, U tailHalfLoopOut0, U tailHalfLoopOut1, float32_t divisor,
                                         uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<sizeof(T) == B16 && std::is_same<Z, float32_t>::value,
                                              Reg::RegTensor<Z, Reg::RegTraitNumTwo>, Reg::RegTensor<T>>::type;
    RegDstT res;
    Reg::RegTensor<U> v1;
    Reg::UnalignRegForStore u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::MaskReg pTail = Reg::UpdateMask<U>(tailNum);
    __ubuf__ Z* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopN; i++) {
        Reg::Adds(v1, index, i * oneLoopStride, p0);
        AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v1, kH, kW, rowStrideInub, divisor, p0, channels);
        if constexpr (sizeof(T) == B16 && std::is_same<Z, float32_t>::value) {
            Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[0], u0, halfLoopOut0);
            Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[1], u0, halfLoopOut1);
        } else {
            Reg::StoreUnAlign(dstAddr, res, u0, oneLoopElements);
        }
    }
    Reg::Adds(v1, index, loopN * oneLoopStride, pTail);
    AvgPoolImpl<T, U, NO_DIV>(res, srcAddr, v1, kH, kW, rowStrideInub, divisor, pTail, channels);
    if constexpr (sizeof(T) == B16 && std::is_same<Z, float32_t>::value) {
        Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[0], u0, tailHalfLoopOut0);
        Reg::StoreUnAlign(dstAddr, (Reg::RegTensor<float32_t>&)res.reg[1], u0, tailHalfLoopOut1);
    } else {
        Reg::StoreUnAlign(dstAddr, res, u0, tailLoopElements);
    }
    Reg::StoreUnAlignPost(dstAddr, u0, 0);
}

template <typename T, bool INCLUDE_PAD, typename RegT>
__aicore__ inline void CalcWindowSize(Reg::RegTensor<float>& res, RegT& src, T kD, T sD, T negFrontPad, T dIn,
                                      T dInAndBackendPad, Reg::MaskReg& mask)
{
    RegT tmp1, tmp2;
    Reg::Muls(tmp1, src, sD, mask);           // (didx * sd)
    Reg::Adds(tmp2, tmp1, negFrontPad, mask); // (didx * sd - fPad)
    Reg::Adds(tmp1, tmp2, kD, mask);          // dstart + kD
    if constexpr (INCLUDE_PAD) {
        Reg::Mins(tmp1, tmp1, dInAndBackendPad, mask);
    } else {
        Reg::Maxs(tmp2, tmp2, 0, mask);
        Reg::Mins(tmp1, tmp1, dIn, mask);
    }
    Reg::Sub(tmp1, tmp1, tmp2, mask);
    if constexpr (std::is_same<T, int32_t>::value) {
        Reg::Cast<float, T, CAST_INT32_TO_FP32>(res, tmp1, mask);
    } else {
        Reg::Cast<float, T, CAST_INT64_TO_FP32>(res, tmp1, mask);
    }
}

template <bool countIncludePad, bool PAD_MULTI_BATCH>
__aicore__ inline void ComputeDivisorImplB32(__ubuf__ float* divAddr, const CalcDivisorParam& param, int32_t start,
                                             int32_t total)
{
    __ubuf__ float* dstAddr = divAddr;
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
        Reg::RegTensor<int32_t> v0;
        Reg::RegTensor<int32_t> v1;
        Reg::RegTensor<int32_t> v2;
        Reg::RegTensor<int32_t> v3;
        Reg::RegTensor<uint32_t> magic0;
        Reg::RegTensor<uint32_t> magic1;
        Reg::RegTensor<int32_t> vd0;
        Reg::RegTensor<int32_t> vd1;
        Reg::RegTensor<int32_t> vd2;
        Reg::RegTensor<int32_t> vd3;

        Reg::RegTensor<float32_t> res;
        Reg::RegTensor<float32_t> hWindow;
        Reg::RegTensor<float32_t> wWindow;
        Reg::MaskReg p0 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();

        Reg::Duplicate(v1, outW, p0);
        Reg::Duplicate(v2, outH, p0);

        Reg::Duplicate(magic0, m0, p0);
        Reg::Duplicate(magic1, m1, p0);

        uint32_t sreg = totalNum;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange(v0, i * oneRegLength + start);
            Reg::AddrReg resOffset = Reg::CreateAddrReg<float32_t>(i, oneRegLength);
            Reg::MaskReg pWrite = Reg::UpdateMask<float32_t>(sreg);
            if constexpr (PAD_MULTI_BATCH) {
                Reg::Duplicate(v3, oneBatchOut, p0);
                PoolUtils::Compute::FastDivImpl((Reg::RegTensor<uint32_t>&)vd1, (Reg::RegTensor<uint32_t>&)v0, magic1,
                                                shift1, p0);
                Reg::Mul(vd2, vd1, v3, p0);
                Reg::Sub(v0, v0, vd2, p0);
            }
            PoolUtils::Compute::FastDivImpl((Reg::RegTensor<uint32_t>&)vd1, (Reg::RegTensor<uint32_t>&)v0, magic0,
                                            shift0, p0); // (i / outhw) -> hidx
            Reg::Mul(vd2, vd1, v1, p0);
            Reg::Sub(vd3, v0, vd2, p0);

            CalcWindowSize<int32_t, countIncludePad>(hWindow, vd1, kH, sH, negTopPad, hIn, hInAndBottomPad, p0);
            CalcWindowSize<int32_t, countIncludePad>(wWindow, vd3, kW, sW, negLeftPad, wIn, wInAndRightPad, p0);
            Reg::Mul(res, hWindow, wWindow, p0);
            Reg::StoreAlign(dstAddr, res, resOffset, pWrite);
        }
    }
}

template <bool countIncludePad, bool PAD_MULTI_BATCH>
__aicore__ inline void ComputeDivisorImplB64(__ubuf__ float* divAddr, const CalcDivisorParam& param, int32_t start,
                                             int32_t total)
{
    __ubuf__ float* dstAddr = divAddr;
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
        using RegDstT = typename Reg::RegTensor<int64_t, Reg::RegTraitNumTwo>;
        RegDstT v0;
        RegDstT v1;
        RegDstT v2;
        RegDstT v3;
        RegDstT v4;

        RegDstT vd0;
        RegDstT vd1;
        RegDstT vd2;
        RegDstT vd3;

        Reg::RegTensor<float32_t> dWindow;
        Reg::RegTensor<float32_t> hWindow;
        Reg::RegTensor<float32_t> wWindow;
        Reg::RegTensor<float32_t> res;
        Reg::MaskReg p0 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();

        Reg::Duplicate(v1, outW, p0);
        Reg::Duplicate(v2, outH, p0);

        uint32_t sreg = oneBatchOut;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange(v0, i * oneRegLength + start);
            Reg::AddrReg resOffset = Reg::CreateAddrReg<float32_t>(i, oneRegLength);
            Reg::MaskReg pWrite = Reg::UpdateMask<float32_t>(sreg);
            Reg::Div(vd1, v0, v1, p0); // i / outw
            CalcWindowSize<int64_t, countIncludePad>(hWindow, vd1, kH, sH, negTopPad, hIn, hInAndBottomPad, p0);
            Reg::StoreAlign(dstAddr, hWindow, resOffset, pWrite);
        }
        uint32_t sreg1 = oneBatchOut;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange(v0, i * oneRegLength + start);
            Reg::AddrReg resOffset = Reg::CreateAddrReg<float32_t>(i, oneRegLength);
            Reg::MaskReg pWrite = Reg::UpdateMask<float32_t>(sreg1);
            Reg::Div(vd3, v0, v1, p0);  // i / outw
            Reg::Mul(vd3, vd3, v1, p0); // (i / outw * outw)
            Reg::Sub(vd3, v0, vd3, p0); // i % outhw

            CalcWindowSize<int64_t, countIncludePad>(wWindow, vd3, kW, sW, negLeftPad, wIn, wInAndRightPad, p0);
            Reg::LoadAlign(res, dstAddr, resOffset);
            Reg::Mul(res, res, wWindow, p0);
            Reg::StoreAlign(dstAddr, res, resOffset, pWrite);
        }
    }

    if (PAD_MULTI_BATCH && (oneBatchOut < total)) {
        uint32_t diff = (total - oneBatchOut);
        uint16_t loopNum = diff / oneBatchOut;
        auto startAddr = dstAddr;
        auto writeAddr = dstAddr + oneBatchOut;
        if (oneBatchOut < oneRegLength) {
            uint32_t repeatElm = oneBatchOut;
            __VEC_SCOPE__
            {
                auto curDstAddr = writeAddr;
                Reg::UnalignRegForStore u0;
                Reg::RegTensor<float32_t> v0;
                Reg::LoadAlign(v0, startAddr);
                for (uint16_t k = 0; k < loopNum; k++) {
                    Reg::StoreUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                Reg::StoreUnAlignPost(curDstAddr, u0, 0);
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
                Reg::UnalignRegForLoad u0;
                Reg::UnalignRegForStore u1;
                Reg::RegTensor<float32_t> v0;
                Reg::LoadAlign(v0, startAddr);
                for (uint16_t i = 0; i < loopNum; i++) {
                    auto curSrcAddr = startAddr;
                    Reg::LoadUnAlignPre(u0, curSrcAddr);
                    for (uint16_t k = 0; k < loopInner; k++) {
                        Reg::LoadUnAlign(v0, u0, curSrcAddr, repeatElm);
                        Reg::StoreUnAlign(curDstAddr, v0, u1, repeatElm);
                    }
                    Reg::LoadUnAlign(v0, u0, curSrcAddr, tailInner);
                    Reg::StoreUnAlign(curDstAddr, v0, u1, tailInner);
                }
                Reg::StoreUnAlignPost(curDstAddr, u1, 0);
            }
        }
    }
}

template <typename T>
__aicore__ inline void AvgPoolDivNormChannel(__ubuf__ T* dstAddr, __ubuf__ float32_t* srcAddr,
                                             __ubuf__ float32_t* divAddr, uint32_t num, uint32_t channel = 1)
{
    uint32_t oneRegChannel = Ops::Base::GetVRegSize() / sizeof(float32_t) / channel;
    uint16_t oneRegNum = oneRegChannel * channel;

    uint16_t loopNum = num / oneRegChannel;
    uint16_t tailNum = (num - loopNum * oneRegChannel) * channel;
    if (tailNum == 0) {
        loopNum -= 1;
        tailNum = oneRegNum;
    }
    __VEC_SCOPE__
    {
        Reg::RegTensor<float32_t> src;
        Reg::RegTensor<float32_t> div;
        Reg::RegTensor<float32_t> tmp;
        Reg::RegTensor<T> res;
        Reg::RegTensor<uint32_t> index;
        Reg::UnalignRegForLoad u0;
        Reg::UnalignRegForStore u1;
        auto curDstAddr = dstAddr;
        auto curSrcAddr = srcAddr;
        uint32_t mainSreg = oneRegNum;
        uint32_t tailSreg = tailNum;
        Reg::MaskReg pMask = Reg::UpdateMask<float32_t>(mainSreg);
        Reg::MaskReg pMaskTail = Reg::UpdateMask<float32_t>(tailSreg);
        Reg::Arange((Reg::RegTensor<int32_t>&)index, 0);
        Reg::RegTensor<uint32_t> channelDiv;
        Reg::MaskReg p0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Duplicate(channelDiv, channel, p0);
        Reg::Div(index, index, channelDiv, p0);
        Reg::LoadUnAlignPre(u0, curSrcAddr);
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::LoadUnAlign(src, u0, curSrcAddr, oneRegNum);
            Reg::Gather(div, divAddr + i * oneRegChannel, index, pMask);
            if constexpr (std::is_same<T, float32_t>::value) {
                Reg::Div<float32_t, &divHighPrecisionMode>(res, src, div, pMask);
                Reg::StoreUnAlign(curDstAddr, res, u1, oneRegNum);
            } else {
                Reg::Div(tmp, src, div, pMask);
                Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
                Reg::StoreUnAlign(curDstAddr, res, u1, oneRegNum);
            }
        }
        Reg::LoadUnAlign(src, u0, curSrcAddr, tailNum);
        Reg::Gather(div, divAddr + loopNum * oneRegChannel, index, pMaskTail);
        if constexpr (std::is_same<T, float32_t>::value) {
            Reg::Div<float32_t, &divHighPrecisionMode>(res, src, div, pMaskTail);
            Reg::StoreUnAlign(curDstAddr, res, u1, tailNum);
        } else {
            Reg::Div(tmp, src, div, pMaskTail);
            Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMaskTail);
            Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
            Reg::StoreUnAlign(curDstAddr, res, u1, tailNum);
        }
        Reg::StoreUnAlignPost(curDstAddr, u1, 0);
    }
}

template <typename T, bool CHANNEL_BROADACAST = false>
__aicore__ inline void AvgPoolDivNorm(__ubuf__ T* dstAddr, __ubuf__ float32_t* srcAddr, __ubuf__ float32_t* divAddr,
                                      uint32_t num, uint32_t channel = 1)
{
    if constexpr (CHANNEL_BROADACAST) {
        return AvgPoolDivNormChannel(dstAddr, srcAddr, divAddr, num, channel);
    }
    uint16_t oneRegNum = Ops::Base::GetVRegSize() / sizeof(float32_t);
    uint16_t loopNum = (num + oneRegNum - 1) / oneRegNum;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float32_t> src;
        Reg::RegTensor<float32_t> div;
        Reg::RegTensor<float32_t> tmp;
        Reg::RegTensor<T> res;

        Reg::UnalignRegForLoad u0;
        Reg::LoadUnAlignPre(u0, divAddr);
        uint32_t sreg = num;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::AddrReg srcOffset = Reg::CreateAddrReg<float32_t>(i, oneRegNum);
            Reg::AddrReg dstOffset = Reg::CreateAddrReg<T>(i, oneRegNum);
            Reg::MaskReg pMask = Reg::UpdateMask<float32_t>(sreg);

            Reg::LoadAlign(src, srcAddr, srcOffset);
            Reg::LoadUnAlign(div, u0, divAddr, oneRegNum);

            if constexpr (std::is_same<T, float32_t>::value) {
                Reg::Div<float32_t, &divHighPrecisionMode>(res, src, div, pMask);
                Reg::StoreAlign(dstAddr, res, dstOffset, pMask);
            } else {
                Reg::Div(tmp, src, div, pMask);
                Reg::MaskReg newMask;
                Reg::Pack<Reg::HighLowPart::LOWEST>(newMask, pMask);
                Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
                Reg::StoreAlign(dstAddr, res, dstOffset, newMask);
            }
        }
    }
}

template <typename T, bool CHANNEL_BROADACAST = false>
__aicore__ inline void AvgPoolDivBatchV1(__ubuf__ T* dstAddr, __ubuf__ float32_t* srcAddr, __ubuf__ float32_t* divAddr,
                                         uint32_t batchNum, uint32_t batchElement, uint32_t channel = 1)
{
    uint32_t oneRegChannel = Ops::Base::GetVRegSize() / sizeof(float32_t) / channel;
    uint16_t oneRegNum = oneRegChannel * channel;
    uint16_t loopNum = batchElement / oneRegChannel;
    uint16_t tailNum = (batchElement - loopNum * oneRegChannel) * channel;
    uint16_t loopBatch = batchNum;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float32_t> src;
        Reg::RegTensor<float32_t> div;
        Reg::RegTensor<float32_t> tmp;
        Reg::RegTensor<T> res;
        Reg::RegTensor<uint32_t> index;
        Reg::UnalignRegForLoad u0;
        Reg::UnalignRegForStore u1;
        auto curSrcAddr = srcAddr;
        auto curDstAddr = dstAddr;

        uint32_t mainSreg = oneRegNum;
        uint32_t tailSreg = tailNum;
        Reg::MaskReg pMask = Reg::UpdateMask<float32_t>(mainSreg);
        Reg::MaskReg pMaskTail = Reg::UpdateMask<float32_t>(tailSreg);
        Reg::LoadUnAlignPre(u0, curSrcAddr);
        if constexpr (CHANNEL_BROADACAST) {
            Reg::Arange((Reg::RegTensor<int32_t>&)index, 0);
            Reg::RegTensor<uint32_t> channelDiv;
            Reg::MaskReg p0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
            Reg::Duplicate(channelDiv, channel, p0);
            Reg::Div(index, index, channelDiv, p0);
        }
        for (uint16_t i = 0; i < loopBatch; i++) {
            uint32_t sreg = batchElement;
            for (uint16_t j = 0; j < loopNum; j++) {
                Reg::AddrReg divOffset = Reg::CreateAddrReg<float32_t>(j, oneRegNum);
                Reg::LoadUnAlign(src, u0, curSrcAddr, oneRegNum);
                if constexpr (CHANNEL_BROADACAST) {
                    Reg::Gather(div, divAddr + j * oneRegChannel, index, pMask);
                } else {
                    Reg::LoadAlign(div, divAddr, divOffset);
                }
                if constexpr (std::is_same<T, float32_t>::value) {
                    Reg::Div<float32_t, &divHighPrecisionMode>(res, src, div, pMask);
                    Reg::StoreUnAlign(curDstAddr, res, u1, oneRegNum);
                } else {
                    Reg::Div(tmp, src, div, pMask);
                    Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                    Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
                    Reg::StoreUnAlign(curDstAddr, res, u1, oneRegNum);
                }
            }
            Reg::LoadUnAlign(src, u0, curSrcAddr, tailNum);
            if constexpr (CHANNEL_BROADACAST) {
                Reg::Gather(div, divAddr + loopNum * oneRegChannel, index, pMaskTail);
            } else {
                Reg::LoadAlign(div, divAddr + loopNum * oneRegNum);
            }
            if constexpr (std::is_same<T, float32_t>::value) {
                Reg::Div<float32_t, &divHighPrecisionMode>(res, src, div, pMaskTail);
                Reg::StoreUnAlign(curDstAddr, res, u1, tailNum);
            } else {
                Reg::Div(tmp, src, div, pMaskTail);
                Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMaskTail);
                Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
                Reg::StoreUnAlign(curDstAddr, res, u1, tailNum);
            }
        }
        Reg::StoreUnAlignPost(curDstAddr, u1, 0);
    }
}

template <typename T, bool CHANNEL_BROADACAST = false>
__aicore__ inline void AvgPoolDivBatchV2(__ubuf__ T* dstAddr, __ubuf__ float32_t* srcAddr, __ubuf__ float32_t* divAddr,
                                         uint32_t batchNum, uint32_t batchElement, uint32_t channel = 1)
{
    constexpr uint16_t oneRegNum = Ops::Base::GetVRegSize() / sizeof(float32_t);
    uint16_t onceRepeatBatch = oneRegNum / (batchElement * channel);
    uint16_t loopNum = batchNum / onceRepeatBatch;
    uint16_t onceRepeatNum = onceRepeatBatch * batchElement * channel;
    uint16_t tailRepeatNum = (batchNum - loopNum * onceRepeatBatch) * batchElement * channel;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float32_t> src;
        Reg::RegTensor<float32_t> div;
        Reg::RegTensor<float32_t> tmp;
        Reg::RegTensor<T> res;
        Reg::RegTensor<uint32_t> index;
        Reg::UnalignRegForLoad u0;
        Reg::UnalignRegForStore u1;
        auto curSrcAddr = srcAddr;
        auto curDstAddr = dstAddr;
        uint32_t mainSreg = onceRepeatNum;
        uint32_t tailSreg = tailRepeatNum;
        Reg::MaskReg pMask = Reg::UpdateMask<float32_t>(mainSreg);
        Reg::MaskReg pTailMask = Reg::UpdateMask<float32_t>(tailSreg);
        if constexpr (CHANNEL_BROADACAST) {
            Reg::Arange((Reg::RegTensor<int32_t>&)index, 0);
            Reg::RegTensor<uint32_t> channelDiv;
            Reg::MaskReg p0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
            Reg::Duplicate(channelDiv, channel, p0);
            Reg::Div(index, index, channelDiv, p0);
        }
        Reg::LoadUnAlignPre(u0, curSrcAddr);
        if constexpr (CHANNEL_BROADACAST) {
            Reg::Gather(div, divAddr, index, pMask);
        } else {
            Reg::LoadAlign(div, divAddr);
        }
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::LoadUnAlign(src, u0, curSrcAddr, onceRepeatNum);
            if constexpr (std::is_same<T, float32_t>::value) {
                Reg::Div<float32_t, &divHighPrecisionMode>(res, src, div, pMask);
                Reg::StoreUnAlign(curDstAddr, res, u1, onceRepeatNum);
            } else {
                Reg::Div(tmp, src, div, pMask);
                Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
                Reg::StoreUnAlign(curDstAddr, res, u1, onceRepeatNum);
            }
        }

        Reg::LoadUnAlign(src, u0, curSrcAddr, tailRepeatNum);
        if constexpr (std::is_same<T, float32_t>::value) {
            Reg::Div<float32_t, &divHighPrecisionMode>(res, src, div, pTailMask);
            Reg::StoreUnAlign(curDstAddr, res, u1, tailRepeatNum);
        } else {
            Reg::Div(tmp, src, div, pTailMask);
            Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pTailMask);
            Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
            Reg::StoreUnAlign(curDstAddr, res, u1, tailRepeatNum);
        }
        Reg::StoreUnAlignPost(curDstAddr, u1, 0);
    }
}

template <typename T, bool CHANNEL_BROADACAST = false>
__aicore__ inline void AvgPoolDivMultiBatch(__ubuf__ T* dstAddr, __ubuf__ float32_t* srcAddr,
                                            __ubuf__ float32_t* divAddr, uint32_t batchNum, uint32_t batchElement,
                                            uint32_t channel = 1)
{
    uint32_t oneVL = Ops::Base::GetVRegSize() / sizeof(float32_t);
    if (batchElement * channel > oneVL) {
        AvgPoolDivBatchV1<T, CHANNEL_BROADACAST>(dstAddr, srcAddr, divAddr, batchNum, batchElement, channel);
    } else {
        AvgPoolDivBatchV2<T, CHANNEL_BROADACAST>(dstAddr, srcAddr, divAddr, batchNum, batchElement, channel);
    }
}

__aicore__ inline void ComputeDivisorCommon(int64_t computeMode, __ubuf__ float* dstAddr, const CalcDivisorParam& param,
                                            int64_t start, int64_t num)
{
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
__aicore__ inline void DuplicateReg(Reg::RegTensor<T>& reg, Reg::MaskReg mask)
{
    Reg::Duplicate(reg, 0, mask);
}

template <typename T>
__aicore__ inline void DuplicateValue(const __ubuf__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    uint32_t num = calNum;
    Reg::RegTensor<T> v0;
    Reg::MaskReg p0 = Reg::UpdateMask<T>(num);
    Reg::UnalignRegForStore u0;
    DuplicateReg<T>(v0, p0);
    __ubuf__ T* addr = (__ubuf__ T*)dstAddr + offset;
    Reg::StoreUnAlign(addr, v0, u0, calNum);
    Reg::StoreUnAlignPost(addr, u0, 0);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename T>
__aicore__ inline void MergeAvgParaRes(Reg::RegTensor<T>& res, __ubuf__ T* dstLocalAddr, int32_t num)
{
    // merge cur result with pre result
    Reg::MaskReg pregAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::RegTensor<T> lastRes;
    AscendC::Reg::UnalignRegForLoad u0;
    auto curSrcAddr = dstLocalAddr;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadUnAlignPre(u0, curSrcAddr);
    AscendC::Reg::LoadUnAlign(lastRes, u0, curSrcAddr, num);
    Reg::Add(res, res, lastRes, pregAll);
    Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
}

template <typename T, typename RegDstT>
__aicore__ inline void LoadOneElement(const __ubuf__ void* input, RegDstT& dst, uint32_t offset)
{
    Reg::UnalignRegForLoad u0;
    auto srcAddr = (__ubuf__ T*)(input) + offset;
    Reg::LoadUnAlignPre(u0, srcAddr);
    Reg::LoadUnAlign(dst, u0, srcAddr, 1);
}

template <typename T, typename RegDstT>
__aicore__ inline void MergeSumRes(RegDstT& res, const __ubuf__ T* dstLocalAddr, int32_t offset)
{
    // merge cur result with pre result
    Reg::MaskReg pregOne = Reg::CreateMask<T, Reg::MaskPattern::VL1>();
    RegDstT lastRes;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    LoadOneElement<T>(dstLocalAddr, lastRes, offset);
    Reg::Add(res, res, lastRes, pregOne);
    Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
}

template <bool MaskMergeMode, typename T, typename U>
__aicore__ inline void SumWithGather(Reg::RegTensor<T>& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                     Reg::MaskReg& mask)
{
    Reg::RegTensor<T> vd1;
    Reg::Gather(vd1, srcAddr, index, mask);
    if constexpr (MaskMergeMode) {
        Reg::RegTensor<T> tmp;
        Reg::Add(tmp, vd1, res, mask);
        Reg::Move<T, Reg::MaskMergeMode::MERGING>(res, tmp, mask);
    } else {
        Reg::Add(res, vd1, res, mask);
    }
}

template <typename T, typename Z>
__aicore__ inline void DivCompute(Reg::RegTensor<T>& res, Reg::RegTensor<Z>& sum, float32_t divisor)
{
    Reg::RegTensor<Z> divisorReg;
    uint32_t scalar = 1;
    if constexpr (sizeof(T) == TWO) {
        // B16类型
        Reg::RegTensor<Z> divRes;
        Reg::Duplicate(divisorReg, divisor);
        Reg::MaskReg divMask = Reg::UpdateMask<Z>(scalar);
        Reg::Div(divRes, sum, divisorReg, divMask);

        // 将RegTensor<Z>(即RegTensor<float32_t>类型)转为RegTensor<B16>类型。
        scalar = 1;
        Reg::MaskReg castMask = Reg::UpdateMask<T>(scalar);
        Reg::Cast<T, Z, castTraitB32ToB16>(res, divRes, castMask);

    } else {
        // B32类型, 此处即float32类型
        Reg::Duplicate(divisorReg, divisor);
        Reg::MaskReg divMask = Reg::UpdateMask<Z>(scalar);
        Reg::Div<Z, &divHighPrecisionMode>(res, sum, divisorReg, divMask);
    }
}

template <typename T, typename U, typename Z>
__aicore__ inline void ReduceSumWithGatherOne(Reg::RegTensor<Z>& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                              Reg::MaskReg& mask)
{
    Reg::RegTensor<T> vd1;
    if constexpr (sizeof(T) == TWO) {
        // B16类型需转换为更高精度类型，防止溢出和精度丢失
        Reg::Gather(vd1, srcAddr, index, mask);

        // B16类型转为float32, 此处Z为float32类型
        Reg::RegTensor<Z> low;
        Reg::RegTensor<Z> left;
        Reg::RegTensor<Z> right;
        Reg::RegTensor<T> tmpLeft;
        Reg::RegTensor<T> tmpRight;
        Reg::RegTensor<T> zero;
        Reg::Duplicate(zero, (T)0);
        Reg::Interleave(tmpLeft, tmpRight, vd1, zero);
        Reg::MaskReg maskAll = Reg::CreateMask<Z, Reg::MaskPattern::ALL>();
        Reg::Cast<Z, T, castTraitB16ToB32>(left, tmpLeft, maskAll);
        Reg::Cast<Z, T, castTraitB16ToB32>(right, tmpRight, maskAll);

        Reg::Add(low, left, right, maskAll);
        Reg::Reduce<Reg::ReduceType::SUM>(res, low, maskAll);
    } else {
        // B32类型
        Reg::Gather(vd1, srcAddr, index, mask);

        Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
        Reg::Reduce<Reg::ReduceType::SUM>(res, vd1, maskAll);
    }
}

template <typename T, typename U, typename Z>
__aicore__ inline void ReduceSumWithGather(Reg::RegTensor<Z>& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                           Reg::MaskReg& mask)
{
    Reg::RegTensor<T> vd1;
    if constexpr (sizeof(T) == TWO) {
        // B16类型需转换为更高精度类型，防止溢出和精度丢失
        Reg::Gather(vd1, srcAddr, index, mask);

        // B16类型转为float32, 此处Z为float32类型
        Reg::RegTensor<Z> low;
        Reg::RegTensor<Z> left;
        Reg::RegTensor<Z> right;
        Reg::RegTensor<T> tmpLeft;
        Reg::RegTensor<T> tmpRight;
        Reg::RegTensor<T> zero;
        Reg::Duplicate(zero, (T)0);
        Reg::Interleave(tmpLeft, tmpRight, vd1, zero);
        Reg::MaskReg maskAll = Reg::CreateMask<Z, Reg::MaskPattern::ALL>();
        Reg::Cast<Z, T, castTraitB16ToB32>(left, tmpLeft, maskAll);
        Reg::Cast<Z, T, castTraitB16ToB32>(right, tmpRight, maskAll);

        Reg::Add(low, left, right, maskAll);
        Reg::Add(res, low, res, maskAll);
    } else {
        // B32类型
        Reg::Gather(vd1, srcAddr, index, mask);

        Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
        Reg::Add(res, res, vd1, maskAll);
    }
}

template <uint16_t REG_NUM, uint16_t IDX, typename U, typename T, typename Z>
__aicore__ inline void ComputeReduceSumWithGather(Reg::RegTensor<Z>& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                                  Reg::MaskReg& mask)
{
    if constexpr (REG_NUM > IDX) {
        ReduceSumWithGather<T, U, Z>(res, srcAddr, index, mask);
    }
}

template <typename T, typename U, uint16_t REG_NUM>
__aicore__ inline void AvgPoolSingleKernelCommon(__ubuf__ T* dstLocalAddr, __ubuf__ T* xLocalAddr,
                                                 __ubuf__ U* indexAddr, uint16_t loopN, uint16_t loopH, uint16_t loopW,
                                                 U oneChannelElements, U oneLoopStrideH, U oneLoopStrideW,
                                                 uint16_t tailLoopElements, float32_t divisor)
{
    if constexpr (sizeof(T) == sizeof(int64_t) && REG_NUM > INT64_MAXREGNUM) {
        return;
    }

    using Z = typename std::conditional<sizeof(T) == B16, float32_t, T>::type;
    __VEC_SCOPE__
    {
        Reg::RegTensor<U> index[SIXTEEN];
        Reg::UnalignRegForStore u0;
        uint32_t tailNum = tailLoopElements;
        Reg::MaskReg maskAll = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::MaskReg pTail = Reg::UpdateMask<U>(tailNum);

        Reg::LoadAlign(index[0], indexAddr);
        PoolUtils::DataMove::LoadIndex<REG_NUM, ONE>(indexAddr, index[ONE]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, TWO>(indexAddr, index[TWO]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, THREE>(indexAddr, index[THREE]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, FOUR>(indexAddr, index[FOUR]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, FIVE>(indexAddr, index[FIVE]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, SIX>(indexAddr, index[SIX]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, SEVEN>(indexAddr, index[SEVEN]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, EIGHT>(indexAddr, index[EIGHT]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, NINE>(indexAddr, index[NINE]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, TEN>(indexAddr, index[TEN]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, ELEVEN>(indexAddr, index[ELEVEN]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, TWELVE>(indexAddr, index[TWELVE]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, THIRTEEN>(indexAddr, index[THIRTEEN]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, FOURTEEN>(indexAddr, index[FOURTEEN]);
        PoolUtils::DataMove::LoadIndex<REG_NUM, FIFTEEN>(indexAddr, index[FIFTEEN]);
        __ubuf__ T* dstAddr = dstLocalAddr;
        for (uint16_t i = 0; i < loopN; i++) {
            __ubuf__ T* srcAddr = xLocalAddr + i * oneChannelElements;
            for (uint16_t j = 0; j < loopH; j++) {
                __ubuf__ T* srcAddrH = srcAddr + j * oneLoopStrideH;
                for (uint16_t k = 0; k < loopW; k++) {
                    __ubuf__ T* srcAddrW = srcAddrH + k * oneLoopStrideW;
                    Reg::RegTensor<T> res;
                    Reg::RegTensor<Z> reduceSumRes;
                    Reg::RegTensor<Z> sum;
                    Reg::Duplicate(sum, (Z)0);

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
                        ReduceSumWithGather<T, U, Z>(sum, srcAddrW, index[REG_NUM - 1], pTail);
                        Reg::Reduce<Reg::ReduceType::SUM>(sum, sum, maskAll);
                    }
                    DivCompute(res, sum, divisor);

                    uint32_t elementCount = 1;
                    Reg::StoreUnAlign(dstAddr, res, u0, elementCount);
                }
            }
        }
        Reg::StoreUnAlignPost(dstAddr, u0, 0);
    }
}

template <typename T, typename U>
__aicore__ inline void AvgPoolSingleKernelDefault(__ubuf__ T* dstLocalAddr, __ubuf__ T* xLocalAddr,
                                                  __ubuf__ U* indexAddr, uint16_t loopN, uint16_t loopH, uint16_t loopW,
                                                  U oneChannelElements, U oneLoopStrideH, U oneLoopStrideW,
                                                  float32_t divisor, uint16_t regNum, uint16_t kernelSize)
{
    using Z = typename std::conditional<sizeof(T) == B16, float32_t, T>::type;
    __VEC_SCOPE__
    {
        Reg::RegTensor<U> index;
        Reg::UnalignRegForStore u0;
        Reg::MaskReg maskAll = Reg::CreateMask<U, Reg::MaskPattern::ALL>();

        __ubuf__ T* dstAddr = dstLocalAddr;
        for (uint16_t i = 0; i < loopN; i++) {
            __ubuf__ T* srcAddr = xLocalAddr + i * oneChannelElements;
            for (uint16_t j = 0; j < loopH; j++) {
                __ubuf__ T* srcAddrH = srcAddr + j * oneLoopStrideH;
                for (uint16_t k = 0; k < loopW; k++) {
                    __ubuf__ T* srcAddrW = srcAddrH + k * oneLoopStrideW;
                    Reg::RegTensor<T> res;
                    Reg::RegTensor<Z> sum;
                    Reg::Duplicate(sum, (Z)0);

                    uint32_t tailNum = kernelSize;
                    for (uint16_t m = 0; m < regNum; m++) {
                        constexpr uint32_t repeatNum = Ops::Base::GetVRegSize() / sizeof(U);
                        Reg::MaskReg pTail = Reg::UpdateMask<U>(tailNum);
                        Reg::LoadAlign(index, indexAddr + m * repeatNum);
                        ReduceSumWithGather<T, U, Z>(sum, srcAddrW, index, pTail);
                    }
                    Reg::Reduce<Reg::ReduceType::SUM>(sum, sum, maskAll);
                    DivCompute(res, sum, divisor);

                    uint32_t elementCount = 1;
                    Reg::StoreUnAlign(dstAddr, res, u0, elementCount);
                }
            }
        }
        Reg::StoreUnAlignPost(dstAddr, u0, 0);
    }
}

} // namespace AvgPool

#endif // AVG_POOL_COMMON_H_
