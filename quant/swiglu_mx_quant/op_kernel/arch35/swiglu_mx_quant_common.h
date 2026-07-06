/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swiglu_mx_quant_common.h
 * \brief Common definitions and shared regbase impl for SwiGLU + MX quantization
 */

#ifndef SWIGLU_MX_QUANT_COMMON_H
#define SWIGLU_MX_QUANT_COMMON_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace SwigluMxQuant {
// ==================== Type Traits ====================
template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {};

// ==================== Constants ====================
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint32_t NAN_CUSTOMIZATION_FP32 = 0x7f810000;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t EXP_MASK_FP16 = 0x7c00;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t THRESHOLD_E2M1 = 0x0100;
constexpr uint16_t THRESHOLD_E1M2 = 0x0080;
constexpr uint16_t NEW_MANTISSA = 0x0008;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint32_t SPECIAL_EXP_THRESHOLD_FP32 = 0x00400000;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint32_t FP4_E2M1_FP32_MAX_EXP = 0x01000000;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t NEG_ONE = -1;
constexpr float FOUR = 4.0;
constexpr float ONE_FOURTH = 0.25;
constexpr int32_t NEG_ZERO = 0x80000000;
constexpr uint16_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925;
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925;
constexpr uint16_t INVALID_FLOAT16 = 0x7c00;
constexpr uint32_t ZERO_FOR_ALL = 0x00000000;
constexpr uint32_t EXP_254 = 0x000000fe;
constexpr uint32_t HALF_FOR_MAN = 0x00400000;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t QUANT_ONCE_NUM = 256;
constexpr int64_t X_ONCE_NUM = 512;
constexpr int64_t QUANT_ONCE_NUM_FP4 = 128;
constexpr int64_t SCALE_ONCE_NUM = 8;
constexpr int64_t CONST_64 = 64;
constexpr int64_t CONST_32 = 32;
constexpr int64_t CONST_2 = 2;
constexpr int64_t CONST_4 = 4;
// ascend950 hardware constants (VReg=256B, UB block=32B)
constexpr uint32_t VF_LEN_T = Ops::Base::GetVRegSize() / sizeof(half);     // 128
constexpr uint32_t VF_LEN_FP32 = Ops::Base::GetVRegSize() / sizeof(float); // 64
constexpr uint32_t ONE_BLOCK_UB = Ops::Base::GetUbBlockSize();
constexpr uint32_t ONE_BLOCK_NUM = ONE_BLOCK_UB / sizeof(half); // 16
// ==================== Cast Traits ====================
static constexpr AscendC::MicroAPI::CastTrait CAST_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_ONE = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_HALF_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_80 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_81 = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_82 = {
    AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_83 = {
    AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

using namespace AscendC;

// ===================================================================
// Shared helpers for axis=-2 quantization
// Used by: last_not_last, not_last_not_last
// ===================================================================

template <typename T>
__aicore__ inline void PadZeroM(__ubuf__ T* swigluOutAddr, uint32_t num)
{
    uint16_t times = CeilDivision(num, VF_LEN_T);
    uint32_t size = num;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> zeroReg;
        AscendC::MicroAPI::MaskReg mask;
        MicroAPI::Duplicate(zeroReg, 0);
        for (uint16_t i = 0; i < times; i++) {
            mask = AscendC::MicroAPI::UpdateMask<T>(size);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, VF_LEN_T);
            AscendC::MicroAPI::DataCopy(swigluOutAddr, zeroReg, offset, mask);
        }
    }
}

template <typename U, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeFP4FromHalf(MicroAPI::RegTensor<float>& Reg, MicroAPI::MaskReg& pregAll32)
{
    MicroAPI::MaskReg zeroMask, specialMask, negInfMask;
    MicroAPI::RegTensor<int32_t> negZero, maxExpFP32, exp0FP32, exp1FP32;
    MicroAPI::Duplicate(negZero, NEG_ZERO);
    MicroAPI::Compare<int32_t, CMPMODE::EQ>(negInfMask, (MicroAPI::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<U, fp4x2_e1m2_t>::value) {
        MicroAPI::Muls(Reg, Reg, FOUR, pregAll32);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        MicroAPI::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        MicroAPI::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        MicroAPI::And(exp0FP32, (MicroAPI::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        MicroAPI::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        MicroAPI::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        MicroAPI::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        MicroAPI::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        MicroAPI::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::Mul(Reg, Reg, (MicroAPI::RegTensor<float>&)exp1FP32, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        MicroAPI::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        MicroAPI::Mul(Reg, Reg, (MicroAPI::RegTensor<float>&)exp0FP32, pregAll32);
    }
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    MicroAPI::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    MicroAPI::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    MicroAPI::Select<int32_t>((MicroAPI::RegTensor<int32_t>&)Reg, negZero, (MicroAPI::RegTensor<int32_t>&)Reg,
                              zeroMask);
}

template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeScaleOcpNotLastAxis(uint16_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr,
                                                  __ubuf__ uint16_t* mxScaleReciprocalAddr, uint16_t dtypeYMaxExp)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> x0, x1;
        MicroAPI::RegTensor<bfloat16_t> x0BF16, x1BF16;
        MicroAPI::RegTensor<uint16_t> x0ExpBF16, x1ExpBF16, x0ExpFP16, x1ExpFP16;
        MicroAPI::RegTensor<uint16_t> expMaskBF16, expMax1Dim2, expMax2Dim2, yMaxExp;
        MicroAPI::RegTensor<uint16_t> nanE8M0, biasE8M0, zero, nanBF16, specialExp;
        MicroAPI::RegTensor<uint16_t> mxScale2ZeroB16, mxScale2OneB16;
        MicroAPI::RegTensor<uint8_t> mxScale2ZeroB8, mxScale2OneB8;
        MicroAPI::RegTensor<uint16_t> reversedShareExp2Zero, reversedShareExp2One;
        MicroAPI::RegTensor<uint16_t> expMaskFP16;
        MicroAPI::MaskReg infMask, zeroMask, invalidDataMask;
        MicroAPI::MaskReg infNanDataMask0, infNanDataMask1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        MicroAPI::Duplicate(expMaskFP16, EXP_MASK_FP16);
        MicroAPI::Duplicate(expMax1Dim2, static_cast<uint16_t>(0));
        MicroAPI::Duplicate(expMax2Dim2, static_cast<uint16_t>(0));
        MicroAPI::Duplicate(yMaxExp, dtypeYMaxExp);
        MicroAPI::Duplicate(nanE8M0, MAX_EXP_FOR_FP8);
        MicroAPI::Duplicate(biasE8M0, BF16_EXP_BIAS);
        MicroAPI::Duplicate(zero, static_cast<uint16_t>(0));
        MicroAPI::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        MicroAPI::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < blockCount; i++) {
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, 256);
            if constexpr (IsSameType<T, half>::value) {
                MicroAPI::And(x0ExpFP16, (MicroAPI::RegTensor<uint16_t>&)x0, expMaskFP16, maskAll);
                MicroAPI::And(x1ExpFP16, (MicroAPI::RegTensor<uint16_t>&)x1, expMaskFP16, maskAll);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask0, x0ExpFP16, expMaskFP16, maskAll);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask1, x1ExpFP16, expMaskFP16, maskAll);
                MicroAPI::Cast<bfloat16_t, T, CAST_HALF_TO_BF16>(x0BF16, x0, maskAll);
                MicroAPI::Cast<bfloat16_t, T, CAST_HALF_TO_BF16>(x1BF16, x1, maskAll);
                MicroAPI::And(x0ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x0BF16, expMaskBF16, maskAll);
                MicroAPI::And(x1ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x1BF16, expMaskBF16, maskAll);
                MicroAPI::Select<uint16_t>(x0ExpBF16, x0ExpBF16, expMaskBF16, infNanDataMask0);
                MicroAPI::Select<uint16_t>(x1ExpBF16, x1ExpBF16, expMaskBF16, infNanDataMask1);
            } else {
                MicroAPI::And(x0ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x0, expMaskBF16, maskAll);
                MicroAPI::And(x1ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x1, expMaskBF16, maskAll);
            }
            MicroAPI::Max(expMax1Dim2, expMax1Dim2, x0ExpBF16, maskAll);
            MicroAPI::Max(expMax2Dim2, expMax2Dim2, x1ExpBF16, maskAll);
        }

        // Even half
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMax1Dim2, expMaskBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax1Dim2, zero, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax1Dim2, yMaxExp, maskAll);
        MicroAPI::Select<uint16_t>(expMax1Dim2, yMaxExp, expMax1Dim2, invalidDataMask);
        MicroAPI::Sub(expMax1Dim2, expMax1Dim2, yMaxExp, maskAll);
        MicroAPI::ShiftRights(mxScale2ZeroB16, expMax1Dim2, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, nanE8M0, infMask);
        MicroAPI::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, zero, zeroMask);
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2ZeroB8, mxScale2ZeroB16);
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax1Dim2, biasE8M0, maskAll);
        MicroAPI::Sub(reversedShareExp2Zero, biasE8M0, expMax1Dim2, maskAll);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, zero, zeroMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, specialExp, reversedShareExp2Zero, invalidDataMask);

        // Odd half
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMax2Dim2, expMaskBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax2Dim2, zero, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax2Dim2, yMaxExp, maskAll);
        MicroAPI::Select<uint16_t>(expMax2Dim2, yMaxExp, expMax2Dim2, invalidDataMask);
        MicroAPI::Sub(expMax2Dim2, expMax2Dim2, yMaxExp, maskAll);
        MicroAPI::ShiftRights(mxScale2OneB16, expMax2Dim2, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, nanE8M0, infMask);
        MicroAPI::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, zero, zeroMask);
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2OneB8, mxScale2OneB16);
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax2Dim2, biasE8M0, maskAll);
        MicroAPI::Sub(reversedShareExp2One, biasE8M0, expMax2Dim2, maskAll);
        MicroAPI::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, zero, zeroMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2One, specialExp, reversedShareExp2One, invalidDataMask);

        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(mxScaleAddr, mxScale2ZeroB8, mxScale2OneB8,
                                                                        maskB8);
        MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(mxScaleReciprocalAddr, reversedShareExp2Zero,
                                                                          reversedShareExp2One, maskAll);
    }
}

template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeScaleCuBLASNotLastAxis(uint16_t blockCount, __ubuf__ T* xAddr,
                                                     __ubuf__ uint8_t* mxScaleAddr,
                                                     __ubuf__ uint16_t* mxScaleReciprocalAddr, uint32_t invDtypeMax)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> x0, x1;
        MicroAPI::RegTensor<uint16_t> absMax0, absMax1;
        MicroAPI::RegTensor<uint16_t> absMax1Dim2, absMax2Dim2;
        MicroAPI::RegTensor<uint32_t> maxFP32_0, maxFP32_1;
        MicroAPI::RegTensor<uint32_t> expFP32_0, expFP32_1;
        MicroAPI::RegTensor<uint32_t> manFP32_0, manFP32_1;
        MicroAPI::RegTensor<uint16_t> scale1B16_0, scale1B16_1;
        MicroAPI::RegTensor<uint16_t> scale1BF16;
        MicroAPI::RegTensor<uint8_t> mxScale1B8;
        MicroAPI::RegTensor<uint8_t> mxScale2B8;

        MicroAPI::RegTensor<uint16_t> absMask;
        MicroAPI::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        MicroAPI::RegTensor<uint32_t> invMax, manMaskFP32;
        MicroAPI::Duplicate(invMax, invDtypeMax);
        MicroAPI::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        MicroAPI::RegTensor<uint16_t> biasE8M0, nanBF16, specialExp, maxEleBF16;
        MicroAPI::Duplicate(biasE8M0, BF16_EXP_BIAS);
        MicroAPI::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        MicroAPI::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);
        MicroAPI::Duplicate(maxEleBF16, MAX_EXP_FOR_BF16);

        MicroAPI::Duplicate(absMax1Dim2, 0);
        MicroAPI::Duplicate(absMax2Dim2, 0);

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskFP32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg p0, p1, p0Odd, p1Odd, infMask, invalidDataMask;

        for (uint16_t i = 0; i < blockCount; i++) {
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, 256);
            MicroAPI::And(absMax0, (MicroAPI::RegTensor<uint16_t>&)x0, absMask, maskAll);
            MicroAPI::And(absMax1, (MicroAPI::RegTensor<uint16_t>&)x1, absMask, maskAll);
            MicroAPI::Max(absMax1Dim2, absMax1Dim2, absMax0, maskAll);
            MicroAPI::Max(absMax2Dim2, absMax2Dim2, absMax1, maskAll);
        }

        // process absMax1Dim2
        MicroAPI::Cast<float, T, CAST_ZERO>((MicroAPI::RegTensor<float>&)maxFP32_0,
                                            (MicroAPI::RegTensor<T>&)absMax1Dim2, maskAll);
        MicroAPI::Mul((MicroAPI::RegTensor<float>&)maxFP32_0, (MicroAPI::RegTensor<float>&)maxFP32_0,
                      (MicroAPI::RegTensor<float>&)invMax, maskFP32);
        MicroAPI::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32_0, ZERO_FOR_ALL, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32_0, EXP_254, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32_0, ZERO_FOR_ALL, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, expFP32_0, ZERO_FOR_ALL, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32_0, SPECIAL_EXP_THRESHOLD_FP32, p1);
        MicroAPI::MaskOr(p0, p0, p1, maskFP32);
        MicroAPI::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
        MicroAPI::Select(manFP32_0, maxFP32_0, expFP32_0, p0);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_0, manFP32_0);
        MicroAPI::Cast<float, T, CAST_ONE>((MicroAPI::RegTensor<float>&)maxFP32_1, (MicroAPI::RegTensor<T>&)absMax1Dim2,
                                           maskAll);
        MicroAPI::Mul((MicroAPI::RegTensor<float>&)maxFP32_1, (MicroAPI::RegTensor<float>&)maxFP32_1,
                      (MicroAPI::RegTensor<float>&)invMax, maskFP32);
        MicroAPI::ShiftRights(expFP32_1, maxFP32_1, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_1, maxFP32_1, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0Odd, expFP32_1, ZERO_FOR_ALL, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0Odd, expFP32_1, EXP_254, p0Odd);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0Odd, manFP32_1, ZERO_FOR_ALL, p0Odd);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1Odd, expFP32_1, ZERO_FOR_ALL, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1Odd, manFP32_1, SPECIAL_EXP_THRESHOLD_FP32, p1Odd);
        MicroAPI::MaskOr(p0Odd, p0Odd, p1Odd, maskFP32);
        MicroAPI::Adds(maxFP32_1, expFP32_1, 1, maskFP32);
        MicroAPI::Select(manFP32_1, maxFP32_1, expFP32_1, p0Odd);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_1, manFP32_1);
        MicroAPI::Interleave(scale1B16_0, scale1B16_1, scale1B16_0, scale1B16_1);
        MicroAPI::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, scale1B16_0);
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
        MicroAPI::Sub(absMax0, biasE8M0, scale1BF16, maskAll);
        MicroAPI::Select<uint16_t>(absMax0, absMax0, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(absMax0, specialExp, absMax0, invalidDataMask);

        // process absMax2Dim2
        MicroAPI::Cast<float, T, CAST_ZERO>((MicroAPI::RegTensor<float>&)maxFP32_0,
                                            (MicroAPI::RegTensor<T>&)absMax2Dim2, maskAll);
        MicroAPI::Mul((MicroAPI::RegTensor<float>&)maxFP32_0, (MicroAPI::RegTensor<float>&)maxFP32_0,
                      (MicroAPI::RegTensor<float>&)invMax, maskFP32);
        MicroAPI::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32_0, ZERO_FOR_ALL, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32_0, EXP_254, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32_0, ZERO_FOR_ALL, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, expFP32_0, ZERO_FOR_ALL, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32_0, SPECIAL_EXP_THRESHOLD_FP32, p1);
        MicroAPI::MaskOr(p0, p0, p1, maskFP32);
        MicroAPI::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
        MicroAPI::Select(manFP32_0, maxFP32_0, expFP32_0, p0);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_0, manFP32_0);
        MicroAPI::Cast<float, T, CAST_ONE>((MicroAPI::RegTensor<float>&)maxFP32_1, (MicroAPI::RegTensor<T>&)absMax2Dim2,
                                           maskAll);
        MicroAPI::Mul((MicroAPI::RegTensor<float>&)maxFP32_1, (MicroAPI::RegTensor<float>&)maxFP32_1,
                      (MicroAPI::RegTensor<float>&)invMax, maskFP32);
        MicroAPI::ShiftRights(expFP32_1, maxFP32_1, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_1, maxFP32_1, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0Odd, expFP32_1, ZERO_FOR_ALL, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0Odd, expFP32_1, EXP_254, p0Odd);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0Odd, manFP32_1, ZERO_FOR_ALL, p0Odd);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1Odd, expFP32_1, ZERO_FOR_ALL, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1Odd, manFP32_1, SPECIAL_EXP_THRESHOLD_FP32, p1Odd);
        MicroAPI::MaskOr(p0Odd, p0Odd, p1Odd, maskFP32);
        MicroAPI::Adds(maxFP32_1, expFP32_1, 1, maskFP32);
        MicroAPI::Select(manFP32_1, maxFP32_1, expFP32_1, p0Odd);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_1, manFP32_1);
        MicroAPI::Interleave(scale1B16_0, scale1B16_1, scale1B16_0, scale1B16_1);
        MicroAPI::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2B8, scale1B16_0);
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
        MicroAPI::Sub(absMax1, biasE8M0, scale1BF16, maskAll);
        MicroAPI::Select<uint16_t>(absMax1, absMax1, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(absMax1, specialExp, absMax1, invalidDataMask);

        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(mxScaleAddr, mxScale1B8, mxScale2B8, maskB8);
        MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(mxScaleReciprocalAddr, absMax0, absMax1,
                                                                          maskAll);
    }
}

template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeCastToFP8NotLastAxis(int64_t dataLen, uint16_t blockCount, __ubuf__ T* xAddr,
                                                   __ubuf__ uint16_t* mxScaleReciprocalAddr, __ubuf__ uint8_t* yAddr)
{
    static constexpr MicroAPI::CastTrait castTraitFp32toY = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> x;
        MicroAPI::RegTensor<float> x0FP32, x1FP32;
        MicroAPI::RegTensor<uint16_t> reversedShareExp;
        MicroAPI::RegTensor<float> reversedShareExp0FP32, reversedShareExp1FP32;
        MicroAPI::RegTensor<U> yZeroFP8, yOneFP8;
        MicroAPI::MaskReg pregAll8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::DataCopy<uint16_t, MicroAPI::LoadDist::DIST_NORM>(reversedShareExp, mxScaleReciprocalAddr);
        MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(reversedShareExp0FP32,
                                                     (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
        MicroAPI::Cast<float, bfloat16_t, CAST_ONE>(reversedShareExp1FP32,
                                                    (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);

        for (uint16_t j = 0; j < blockCount; j++) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(x, xAddr + j * dataLen);
            MicroAPI::Cast<float, T, CAST_ZERO>(x0FP32, x, pregAll16);
            MicroAPI::Cast<float, T, CAST_ONE>(x1FP32, x, pregAll16);
            MicroAPI::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
            MicroAPI::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);
            MicroAPI::Cast<U, float, castTraitFp32toY>(yZeroFP8, (MicroAPI::RegTensor<float>&)x0FP32, pregAll32);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t>&)yZeroFP8,
                                                                              (MicroAPI::RegTensor<uint32_t>&)yZeroFP8);
            MicroAPI::Cast<U, float, castTraitFp32toY>(yOneFP8, (MicroAPI::RegTensor<float>&)x1FP32, pregAll32);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t>&)yOneFP8,
                                                                              (MicroAPI::RegTensor<uint32_t>&)yOneFP8);
            MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t>&)yZeroFP8, (MicroAPI::RegTensor<uint16_t>&)yOneFP8,
                                 (MicroAPI::RegTensor<uint16_t>&)yZeroFP8, (MicroAPI::RegTensor<uint16_t>&)yOneFP8);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)yZeroFP8,
                                                                             (MicroAPI::RegTensor<uint16_t>&)yZeroFP8);
            DataCopy(yAddr + (j * dataLen), (MicroAPI::RegTensor<uint8_t>&)yZeroFP8, pregAll8);
        }
    }
}

template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeYBF16ToFP4NotLastAxis(int64_t dataLen, uint16_t blockCount, __ubuf__ T* xAddr,
                                                    __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr)
{
    static constexpr MicroAPI::CastTrait castTraitBF16toFp4 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg dataMaskB8 = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::MaskReg dataMaskB16 = MicroAPI::CreateMask<half>();
        MicroAPI::RegTensor<T> x0, x1;
        MicroAPI::RegTensor<uint16_t> reversedShareExp0, reversedShareExp1;
        MicroAPI::RegTensor<T> dim0x0, dim0x1;
        MicroAPI::RegTensor<U> dim0x0FP4, dim0x1FP4;
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
            reversedShareExp0, reversedShareExp1, mxScale2ReciprocalAddr, 256);
        for (uint16_t i = 0; i < blockCount; i++) {
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, 256);
            MicroAPI::Mul(dim0x0, x0, (MicroAPI::RegTensor<T>&)reversedShareExp0, dataMaskB16);
            MicroAPI::Mul(dim0x1, x1, (MicroAPI::RegTensor<T>&)reversedShareExp1, dataMaskB16);
            MicroAPI::Interleave(dim0x0, dim0x1, dim0x0, dim0x1);
            MicroAPI::Cast<U, T, castTraitBF16toFp4>(dim0x0FP4, dim0x0, dataMaskB16);
            MicroAPI::Cast<U, T, castTraitBF16toFp4>(dim0x1FP4, dim0x1, dataMaskB16);
            MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y2Addr + (i * dataLen / 2), (MicroAPI::RegTensor<uint8_t>&)dim0x0FP4, dataMaskB8);
            MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y2Addr + 64 + (i * dataLen / 2), (MicroAPI::RegTensor<uint8_t>&)dim0x1FP4, dataMaskB8);
        }
    }
}

template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeYFP16ToFP4NotLastAxis(int64_t dataLen, uint16_t blockCount, __ubuf__ T* xAddr,
                                                    __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr)
{
    static constexpr MicroAPI::CastTrait castTraitFp32toBF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitBF16toFp4 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, roundMode};
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg dataMaskB8 = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::MaskReg dataMaskB16 = MicroAPI::CreateMask<half>();
        MicroAPI::MaskReg dataMaskB32 = MicroAPI::CreateMask<float>();
        MicroAPI::RegTensor<T> x0, x1;
        MicroAPI::RegTensor<float> x0ZeroFP32, x0OneFP32, x1ZeroFP32, x1OneFP32;
        MicroAPI::RegTensor<float> reversedShareExp0ZeroFP32, reversedShareExp0OneFP32;
        MicroAPI::RegTensor<float> reversedShareExp1ZeroFP32, reversedShareExp1OneFP32;
        MicroAPI::RegTensor<float> dim0x0ZeroFP32, dim0x0OneFP32, dim0x1ZeroFP32, dim0x1OneFP32;
        MicroAPI::RegTensor<bfloat16_t> dim0x0ZeroBF16, dim0x0OneBF16, dim0x1ZeroBF16, dim0x1OneBF16;
        MicroAPI::RegTensor<uint16_t> reversedShareExp0, reversedShareExp1;
        MicroAPI::RegTensor<U> dim0x0FP4, dim0x1FP4;
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
            reversedShareExp0, reversedShareExp1, mxScale2ReciprocalAddr, 256);
        MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(reversedShareExp0ZeroFP32,
                                                     (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp0, dataMaskB16);
        MicroAPI::Cast<float, bfloat16_t, CAST_ONE>(reversedShareExp0OneFP32,
                                                    (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp0, dataMaskB16);
        MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(reversedShareExp1ZeroFP32,
                                                     (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp1, dataMaskB16);
        MicroAPI::Cast<float, bfloat16_t, CAST_ONE>(reversedShareExp1OneFP32,
                                                    (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp1, dataMaskB16);
        for (uint16_t i = 0; i < blockCount; i++) {
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, 256);
            MicroAPI::Cast<float, T, CAST_ZERO>(x0ZeroFP32, x0, dataMaskB16);
            MicroAPI::Cast<float, T, CAST_ONE>(x0OneFP32, x0, dataMaskB16);
            MicroAPI::Cast<float, T, CAST_ZERO>(x1ZeroFP32, x1, dataMaskB16);
            MicroAPI::Cast<float, T, CAST_ONE>(x1OneFP32, x1, dataMaskB16);
            MicroAPI::Mul(dim0x0ZeroFP32, reversedShareExp0ZeroFP32, x0ZeroFP32, dataMaskB32);
            MicroAPI::Mul(dim0x0OneFP32, reversedShareExp0OneFP32, x0OneFP32, dataMaskB32);
            ComputeFP4FromHalf<U, roundMode>(dim0x0ZeroFP32, dataMaskB32);
            ComputeFP4FromHalf<U, roundMode>(dim0x0OneFP32, dataMaskB32);
            MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim0x0ZeroBF16, dim0x0ZeroFP32, dataMaskB32);
            MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim0x0OneBF16, dim0x0OneFP32, dataMaskB32);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)dim0x0ZeroBF16, (MicroAPI::RegTensor<uint32_t>&)dim0x0ZeroBF16);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)dim0x0OneBF16, (MicroAPI::RegTensor<uint32_t>&)dim0x0OneBF16);
            MicroAPI::Interleave(dim0x0ZeroBF16, dim0x0OneBF16, dim0x0ZeroBF16, dim0x0OneBF16);
            MicroAPI::Mul(dim0x1ZeroFP32, reversedShareExp1ZeroFP32, x1ZeroFP32, dataMaskB32);
            MicroAPI::Mul(dim0x1OneFP32, reversedShareExp1OneFP32, x1OneFP32, dataMaskB32);
            ComputeFP4FromHalf<U, roundMode>(dim0x1ZeroFP32, dataMaskB32);
            ComputeFP4FromHalf<U, roundMode>(dim0x1OneFP32, dataMaskB32);
            MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim0x1ZeroBF16, dim0x1ZeroFP32, dataMaskB32);
            MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim0x1OneBF16, dim0x1OneFP32, dataMaskB32);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)dim0x1ZeroBF16, (MicroAPI::RegTensor<uint32_t>&)dim0x1ZeroBF16);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)dim0x1OneBF16, (MicroAPI::RegTensor<uint32_t>&)dim0x1OneBF16);
            MicroAPI::Interleave(dim0x1ZeroBF16, dim0x1OneBF16, dim0x1ZeroBF16, dim0x1OneBF16);
            MicroAPI::Interleave(dim0x0ZeroBF16, dim0x1ZeroBF16, dim0x0ZeroBF16, dim0x1ZeroBF16);
            MicroAPI::Cast<U, bfloat16_t, castTraitBF16toFp4>(dim0x0FP4, dim0x0ZeroBF16, dataMaskB16);
            MicroAPI::Cast<U, bfloat16_t, castTraitBF16toFp4>(dim0x1FP4, dim0x1ZeroBF16, dataMaskB16);
            MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y2Addr + (i * dataLen / 2), (MicroAPI::RegTensor<uint8_t>&)dim0x0FP4, dataMaskB8);
            MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y2Addr + 64 + (i * dataLen / 2), (MicroAPI::RegTensor<uint8_t>&)dim0x1FP4, dataMaskB8);
        }
    }
}

template <typename T_IDX>
__aicore__ inline void ReduceAllVf(LocalTensor<T_IDX>& reduceSumUb, LocalTensor<T_IDX>& groupIndexUb,
                                   int64_t groupIndexNum)
{
    uint32_t vfTidx = Ops::Base::GetVRegSize() / sizeof(T_IDX); // VReg=256B, adapt to T_IDX
    uint16_t times = groupIndexNum / vfTidx;
    uint32_t tailNum = groupIndexNum % vfTidx;
    uint16_t tailTimes = tailNum != 0 ? 1 : 0;
    auto dstAddr = (__ubuf__ T_IDX*)reduceSumUb.GetPhyAddr();
    auto srcAddr = (__ubuf__ T_IDX*)groupIndexUb.GetPhyAddr();
    auto srcAddr1 = (__ubuf__ T_IDX*)groupIndexUb[times * vfTidx].GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T_IDX> addReg, reduceSumReg, reduceSumTReg, srcReg;
        AscendC::MicroAPI::Duplicate(addReg, 0);
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T_IDX, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < times; i++) {
            AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<T_IDX>(i, vfTidx);
            AscendC::MicroAPI::DataCopy(srcReg, srcAddr, srcIdxOffset);
            AscendC::MicroAPI::Add(addReg, addReg, srcReg, mask);
        }
        AscendC::MicroAPI::ReduceSum(reduceSumReg, addReg, mask);
        for (uint16_t j = 0; j < tailTimes; j++) {
            AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<T_IDX>(tailNum);
            AscendC::MicroAPI::DataCopy(srcReg, srcAddr1);
            AscendC::MicroAPI::ReduceSum(reduceSumTReg, srcReg, maskT);
            AscendC::MicroAPI::Add(reduceSumReg, reduceSumTReg, reduceSumReg, maskT);
        }
        AscendC::MicroAPI::MaskReg maskOne = AscendC::MicroAPI::CreateMask<T_IDX, MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::DataCopy(dstAddr, reduceSumReg, maskOne);
    }
}

template <typename T>
__aicore__ inline void ComputeVfMaxExpVfLast(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, int64_t dim0OnceSize,
                                             int64_t alignDim1Size)
{
    uint32_t totalCountInUB = dim0OnceSize * alignDim1Size;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    uint16_t invalidFp16 = INVALID_FLOAT16;
    uint16_t maxExpbf16 = MAX_EXP_FOR_BF16;
    int64_t onceNum = QUANT_ONCE_NUM;
    int64_t scaleNum = SCALE_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect0, vdExpSelect1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMaskBF16, maxExpbf16);
        AscendC::MicroAPI::RegTensor<uint16_t> invalidmaskfp16;
        AscendC::MicroAPI::Duplicate(invalidmaskfp16, invalidFp16);
        AscendC::MicroAPI::MaskReg scaleMask1, scaleMask2, invalidDataMask0, invalidDataMask1;
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceNum);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::And(vdExpSelect0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, invalidmaskfp16,
                                       scaleMask1);
                AscendC::MicroAPI::And(vdExpSelect1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, invalidmaskfp16,
                                       scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, vdExpSelect0, invalidmaskfp16,
                                                                  scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, vdExpSelect1, invalidmaskfp16,
                                                                  scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, CAST_HALF_TO_BF16>(vdExp0BF16, vdExp0, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, CAST_HALF_TO_BF16>(vdExp1BF16, vdExp1, scaleMask1);
                AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16,
                                       scaleMask1);
                AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16,
                                       scaleMask1);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16,
                                       scaleMask1);
                AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16,
                                       scaleMask1);
            }
            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, scaleNum);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

template <typename T>
__aicore__ inline void ComputeVfMaxExpVfBLASLast(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr,
                                                 int64_t dim0OnceSize, int64_t alignDim1Size)
{
    uint32_t totalCountInUB = dim0OnceSize * alignDim1Size;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    int64_t onceNum = QUANT_ONCE_NUM;
    int64_t scaleNum = SCALE_ONCE_NUM;
    uint16_t absMaskFor16Bit = ABS_MASK_FOR_16BIT;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> absMask16Bit;
        AscendC::MicroAPI::Duplicate(absMask16Bit, absMaskFor16Bit);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg scaleMask1, scaleMask2;
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceNum);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, absMask16Bit, scaleMask1);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, absMask16Bit, scaleMask1);
            AscendC::MicroAPI::Max(vdMaxExp, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, scaleNum);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

template <typename T>
__aicore__ inline void ComputeScaleLast(uint16_t fEmax, __ubuf__ uint16_t* maxExpAddr,
                                        __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                        int64_t dim0OnceSize, int64_t alignDim1Size)
{
    uint32_t totalScaleInUB = dim0OnceSize * (alignDim1Size / CONST_32);
    uint16_t loopNumScale = CeilDivision(totalScaleInUB, QUANT_ONCE_NUM_FP4);
    uint16_t maxExpBf16 = MAX_EXP_FOR_BF16;
    int64_t onceNum = QUANT_ONCE_NUM_FP4;
    int64_t onceNumMxScale = CONST_64;
    uint16_t bf16ExpBias = BF16_EXP_BIAS;
    uint16_t maxExpFp8 = MAX_EXP_FOR_FP8;
    uint16_t nanCustomZation = NAN_CUSTOMIZATION;
    uint16_t specailExpThreshold = SPECIAL_EXP_THRESHOLD;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMask, maxExpBf16);
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, sharedExp, scaleValue, scaleBias, halfScale;
        AscendC::MicroAPI::Duplicate(maxExpValue, fEmax);
        AscendC::MicroAPI::Duplicate(scaleBias, bf16ExpBias);
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor, zeroRegTensor, nanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, maxExpFp8);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, nanCustomZation);
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, specailExpThreshold);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, onceNum);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                                     onceNumMxScale, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, onceNum, preMaskScale);
        }
    }
}

template <typename T, typename U, AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeDataF4Last(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                         __ubuf__ int8_t* outLocalAddr, int64_t dim0OnceSize, int64_t dim1AlignSize)
{
    uint32_t totalCountInUB = dim0OnceSize * dim1AlignSize;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    int64_t onceYNum = OUT_ELE_NUM_ONE_BLK;
    static constexpr AscendC::MicroAPI::CastTrait castTrait = {AscendC::MicroAPI::RegLayout::ZERO,
                                                               AscendC::MicroAPI::SatMode::UNKNOWN,
                                                               AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
        AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toBF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                MicroAPI::MaskMergeMode::ZEROING, roundMode};
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP4, vdExp1FP4;
        MicroAPI::RegTensor<float> halfScaleForMulFP32;
        MicroAPI::RegTensor<float> vdExp0ZeroFP32, vdExp0OneFP32, vdExp1ZeroFP32, vdExp1OneFP32;
        MicroAPI::RegTensor<bfloat16_t> vdExp0ZeroBF16, vdExp0OneBF16, vdExp1ZeroBF16, vdExp1OneBF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
        MicroAPI::MaskReg dataMaskB16 = MicroAPI::CreateMask<half>();
        MicroAPI::MaskReg dataMaskB32 = MicroAPI::CreateMask<float>();
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                      onceXNum);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce);
            if constexpr (IsSame<T, half>::value && roundMode == RoundMode::CAST_FLOOR) {
                AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16, vdExp1BF16;
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                AscendC::MicroAPI::Mul(vdExp0BF16, vdExp0BF16,
                                       (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1BF16, vdExp1BF16,
                                       (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);
            } else if constexpr (IsSame<T, half>::value && roundMode != RoundMode::CAST_FLOOR) {
                MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(
                    halfScaleForMulFP32, (MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);
                MicroAPI::Cast<float, T, CAST_ZERO>(vdExp0ZeroFP32, vdExp0, dataMaskB16);
                MicroAPI::Cast<float, T, CAST_ONE>(vdExp0OneFP32, vdExp0, dataMaskB16);
                MicroAPI::Mul(vdExp0ZeroFP32, vdExp0ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                MicroAPI::Mul(vdExp0OneFP32, vdExp0OneFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<U, roundMode>(vdExp0ZeroFP32, dataMaskB32);
                ComputeFP4FromHalf<U, roundMode>(vdExp0OneFP32, dataMaskB32);
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp0ZeroBF16, vdExp0ZeroFP32,
                                                                                dataMaskB32);
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp0OneBF16, vdExp0OneFP32,
                                                                                dataMaskB32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)vdExp0ZeroBF16, (MicroAPI::RegTensor<uint32_t>&)vdExp0ZeroBF16);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)vdExp0OneBF16, (MicroAPI::RegTensor<uint32_t>&)vdExp0OneBF16);
                MicroAPI::Interleave(vdExp0ZeroBF16, vdExp0OneBF16, vdExp0ZeroBF16, vdExp0OneBF16);
                MicroAPI::Cast<float, T, CAST_ZERO>(vdExp1ZeroFP32, vdExp1, dataMaskB16);
                MicroAPI::Cast<float, T, CAST_ONE>(vdExp1OneFP32, vdExp1, dataMaskB16);
                MicroAPI::Mul(vdExp1ZeroFP32, vdExp1ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                MicroAPI::Mul(vdExp1OneFP32, vdExp1OneFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<U, roundMode>(vdExp1ZeroFP32, dataMaskB32);
                ComputeFP4FromHalf<U, roundMode>(vdExp1OneFP32, dataMaskB32);
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp1ZeroBF16, vdExp1ZeroFP32,
                                                                                dataMaskB32);
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp1OneBF16, vdExp1OneFP32,
                                                                                dataMaskB32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)vdExp1ZeroBF16, (MicroAPI::RegTensor<uint32_t>&)vdExp1ZeroBF16);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)vdExp1OneBF16, (MicroAPI::RegTensor<uint32_t>&)vdExp1OneBF16);
                MicroAPI::Interleave(vdExp1ZeroBF16, vdExp1OneBF16, vdExp1ZeroBF16, vdExp1OneBF16);
                AscendC::MicroAPI::Interleave(vdExp0ZeroBF16, vdExp1ZeroBF16, vdExp0ZeroBF16, vdExp1ZeroBF16);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0ZeroBF16, dataMask1);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1ZeroBF16, dataMask1);
            } else {
                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            }
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP4, onceYNum, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP4, onceYNum, dataMask1);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void ComputeDataF8Last(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                         __ubuf__ int8_t* outLocalAddr, int64_t dim0OnceSize, int64_t dim1AlignSize)
{
    uint32_t totalCountInUB = dim0OnceSize * dim1AlignSize;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero, vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP8Zero, vdExp0FP8One;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP8Zero, vdExp1FP8One;
        AscendC::MicroAPI::MaskReg
            maskAll = AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg
            maskAllB8 = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                      onceXNum);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
                AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(
                    floatScaleForMul, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
                AscendC::MicroAPI::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, maskAll);
            } else {
                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
            }
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_80>(vdExp0FP8Zero, vdExp0FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_82>(vdExp0FP8One, vdExp0FP32One, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_81>(vdExp1FP8Zero, vdExp1FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_83>(vdExp1FP8One, vdExp1FP32One, maskAll);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8One, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8Zero, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8One, maskAllB8);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, onceXNum, maskAllB8);
        }
    }
}

template <typename T>
__aicore__ inline void ComputeScaleBLASLast(uint32_t dtypeMax, __ubuf__ uint16_t* maxExpAddr,
                                            __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                            int64_t dim0OnceSize, int64_t alignDim1Size)
{
    uint32_t totalScaleInUB = dim0OnceSize * (alignDim1Size / CONST_32);
    uint16_t loopNumScale = CeilDivision(totalScaleInUB, CONST_64);
    uint32_t manMaskFloat = MAN_MASK_FLOAT;
    uint32_t maxExpForFp32 = MAX_EXP_FOR_FP32;
    uint32_t fp32ExpBiasCublas = FP32_EXP_BIAS_CUBLAS;
    uint32_t nanCustomiZation = NAN_CUSTOMIZATION_PACK;
    uint32_t maxExpForFp8InFp32 = MAX_EXP_FOR_FP8_IN_FP32;
    uint32_t zeroForAll = ZERO_FOR_ALL;
    uint32_t Exp254 = EXP_254;
    uint32_t halfForMan = HALF_FOR_MAN;
    int64_t onceNum = CONST_64;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> max16, expOut;
        AscendC::MicroAPI::RegTensor<uint32_t> max32, exp32, man32, normalExp32, expAddOne32, extractExp;
        AscendC::MicroAPI::RegTensor<uint32_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> recExpOut;
        AscendC::MicroAPI::RegTensor<uint32_t> invMax, manMaskFP32, expMask, zeroRegTensor32, scaleBias;
        AscendC::MicroAPI::RegTensor<uint32_t> nanRegTensor, fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(invMax, dtypeMax);
        AscendC::MicroAPI::Duplicate(manMaskFP32, manMaskFloat);
        AscendC::MicroAPI::Duplicate(expMask, maxExpForFp32);
        AscendC::MicroAPI::Duplicate(zeroRegTensor32, 0);
        AscendC::MicroAPI::Duplicate(scaleBias, fp32ExpBiasCublas);
        AscendC::MicroAPI::Duplicate(nanRegTensor, nanCustomiZation);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, maxExpForFp8InFp32);
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, p0, p1, p2;
        AscendC::MicroAPI::MaskReg maskHalf = AscendC::MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::VL64>();
        AscendC::MicroAPI::MaskReg preMaskScale = AscendC::MicroAPI::CreateMask<uint32_t>();
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<uint16_t>();
        for (uint16_t i = 0; i < loopNumScale; i++) {
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(max16, maxExpAddr, onceNum);
            AscendC::MicroAPI::Cast<float, T, CAST_ZERO>((AscendC::MicroAPI::RegTensor<float>&)max32,
                                                         (AscendC::MicroAPI::RegTensor<T>&)max16, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);
            AscendC::MicroAPI::Mul((AscendC::MicroAPI::RegTensor<float>&)max32,
                                   (AscendC::MicroAPI::RegTensor<float>&)max32,
                                   (AscendC::MicroAPI::RegTensor<float>&)invMax, preMaskScale);
            AscendC::MicroAPI::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, preMaskScale);
            AscendC::MicroAPI::And(man32, max32, manMaskFP32, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, zeroForAll, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, Exp254, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, zeroForAll, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p0, p0, p1, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p0, p0, p2, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32, zeroForAll, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, halfForMan, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p1, p1, p2, preMaskScale);
            AscendC::MicroAPI::MaskOr(p0, p0, p1, preMaskScale);
            AscendC::MicroAPI::Adds(expAddOne32, exp32, 1, preMaskScale);
            AscendC::MicroAPI::Select(extractExp, expAddOne32, exp32, p0);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(expOut, extractExp);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, expOut, CONST_32,
                                                                                     maskHalf);
            AscendC::MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, extractExp, preMaskScale);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(recExpOut, halfScale);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, recExpOut, onceNum, mask);
        }
    }
}

template <typename T>
__aicore__ inline void ComputeSwigluV1Second(int64_t calcCol, int64_t calcRow, __ubuf__ T* actAddr,
                                             __ubuf__ T* gateAddr, __ubuf__ T* swigluAddr)
{
    int32_t alignDim1In = ((calcCol + ONE_BLOCK_NUM - 1) / ONE_BLOCK_NUM) * ONE_BLOCK_NUM;
    uint16_t dim0VfTimes = calcRow;
    uint16_t dim1VfTimes = CeilDivision(calcCol, 64);
    int32_t ubOutRow = QUANT_ONCE_NUM;
    float negScalarOne = -1.0f;
    float scalarOne = 1.0f;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregX1, vregX2;
        MicroAPI::RegTensor<float> vregX1F, vregX2F, negReg, expReg, addsReg, sigmoidReg, outFReg;
        MicroAPI::RegTensor<T> outTReg;
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t row = 0; row < dim0VfTimes; row++) {
            for (uint16_t col = 0; col < dim1VfTimes; col++) {
                MicroAPI::AddrReg srcIdx = MicroAPI::CreateAddrReg<T>(row, alignDim1In, col, 64);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, actAddr, srcIdx);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, gateAddr, srcIdx);
                MicroAPI::Cast<float, T, CAST_ZERO>(vregX1F, vregX1, mask);
                MicroAPI::Cast<float, T, CAST_ZERO>(vregX2F, vregX2, mask);
                MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask);
                MicroAPI::Exp(expReg, negReg, mask);
                MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
                MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask);
                MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask);
                MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                MicroAPI::AddrReg outOffset = MicroAPI::CreateAddrReg<T>(row, ubOutRow, col, 64);
                DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(swigluAddr, outTReg, outOffset, mask);
            }
        }
    }
}
// ===================================================================
// Shared SwiGLU compute helpers (standard mode=0)
// ===================================================================
template <typename T>
__aicore__ inline void ComputeVfSwigluV1(__local_mem__ T* x1UbAddr, __local_mem__ T* x2UbAddr,
                                         __local_mem__ T* swigluUbAddr, int64_t dim0OnceSize, int64_t dim1OnceSize,
                                         int64_t dim1AlignSize)
{
    // 在计算swiglu时需把pad 0做了
    uint16_t dim0VfTimes = dim0OnceSize;
    uint16_t dim1VfTimes = dim1OnceSize / VF_LEN_FP32;
    uint32_t dim1Tail = dim1OnceSize % VF_LEN_FP32;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    uint32_t alignDim1In = ((dim1OnceSize + ONE_BLOCK_NUM - 1) / ONE_BLOCK_NUM) * ONE_BLOCK_NUM;
    uint32_t alignDim1Out = dim1AlignSize;
    auto x1UbAddr1 = x1UbAddr;
    auto x2UbAddr1 = x2UbAddr;
    auto swigluUbAddr1 = swigluUbAddr;
    auto swigluUbAddr2 = swigluUbAddr;
    T numZero = 0;
    if (dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        uint32_t padNum = alignDim1Out - dim1VfTimes * VF_LEN_FP32;
        if (padNum <= VF_LEN_FP32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = VF_LEN_FP32;
            mask3Num = padNum - VF_LEN_FP32;
        }
        int32_t offsetAlgin = dim1VfTimes * VF_LEN_FP32;
        x1UbAddr1 = x1UbAddr + offsetAlgin;
        x2UbAddr1 = x2UbAddr + offsetAlgin;
        swigluUbAddr1 = swigluUbAddr + offsetAlgin;
        swigluUbAddr2 = swigluUbAddr + offsetAlgin + dim1TailTimes * VF_LEN_FP32;
    }
    float scalarOne = 1.0f;
    float negScalarOne = -1.0f;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vregX1;
        AscendC::MicroAPI::RegTensor<T> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<T> outTReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<T>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<T>(
                    dim0vfLoopIdx, alignDim1In, dim1vfLoopIdx, 64);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddr,
                                                                                             srcIdxOffset);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddr,
                                                                                             srcIdxOffset);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask);
                AscendC::MicroAPI::Exp(expReg, negReg, mask);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask);

                AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out,
                                                                                           dim1vfLoopIdx, 64);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluUbAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In);
            AscendC::MicroAPI::AddrReg outOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out);
            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddr1,
                                                                                             srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddr1,
                                                                                             srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);

                AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluUbAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                Duplicate<T>(vregX1, numZero);
                DataCopy<T>(swigluUbAddr2, vregX1, outOffset1, mask3);
            }
        }
    }
}
} // namespace SwigluMxQuant
#endif // SWIGLU_MX_QUANT_COMMON_H
