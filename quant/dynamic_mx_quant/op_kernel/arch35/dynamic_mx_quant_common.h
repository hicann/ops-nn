/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_common.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_COMMON_H
#define DYNAMIC_MX_QUANT_COMMON_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "../inc/platform.h"
#include "dynamic_mx_quant_tilingdata.h"

namespace DynamicMxQuant {

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using TrueType = IntegralConstant<bool, true>;
using FalseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public FalseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public TrueType {};

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIM2 = 2;
constexpr int64_t DIGIT_ZERO = 0;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FP4_SCALE_FACTOR = 4;
constexpr int64_t DIGIT_EIGHT = 8;
constexpr int64_t DIGIT_SIXTY_THREE = 63;
constexpr float DIGIT_ZERO_FLOAT = 0.0;
constexpr float DIGIT_SIX_FLOAT = 6.0;
constexpr float DIGIT_SEVEN_FLOAT = 7.0;
constexpr int64_t MODE_ZERO = 0;
constexpr int64_t MODE_ONE = 1;
constexpr int64_t MODE_TWO = 2;
constexpr int64_t MODE_THREE = 3;

constexpr uint32_t VF_LEN_16 = platform::GetVRegSize() / sizeof(uint16_t);
constexpr uint32_t VF_LEN_16_DOUBLE = VF_LEN_16 * 2;
constexpr uint32_t VF_LEN_32 = platform::GetVRegSize() / sizeof(uint32_t);
constexpr uint32_t VF_LEN_32_DOUBLE = VF_LEN_32 * 2;
constexpr int64_t UB_BLOCK_SIZE_ = platform::GetUbBlockSize() == 0 ? 32 : platform::GetUbBlockSize();
constexpr int64_t FLOAT_PER_UB_BLOCK = UB_BLOCK_SIZE_ / sizeof(float);
constexpr uint16_t ELEMENT_AFTER_REDUCE_ = platform::GetVRegSize() / UB_BLOCK_SIZE_;

constexpr uint16_t BF16_ADD_VALUE_MAN1 = 0x003f;
constexpr uint16_t BF16_ADD_VALUE_MAN2 = 0x001f;
constexpr int64_t FP4_OUT_ELE_PER_BLK = 64;
constexpr int64_t FP8_OUT_ELE_PER_BLK = 32;
constexpr uint16_t BF16_NAN_CUSTOM = 0x7f81;
constexpr uint32_t FP32_NAN_CUSTOM = 0x7f810000;
constexpr uint16_t BF16_MAX_EXP = 0x7f80;
constexpr uint32_t FP32_MX_MAX_EXP = 0x7f800000;
constexpr uint16_t FP8_DEFAULT_MAX_EXP = 0x00ff;
constexpr uint32_t FP8_MAX_EXP_IN_FP32 = 0x000000ff;
constexpr uint16_t FP4_E2M1_SPECIAL_VALUE = 0x00ff;
constexpr uint16_t FP4_E1M2_SPECIAL_VALUE = 0x007f;
constexpr uint16_t FP4_E2M1_THRESHOLD = 0x0100;
constexpr uint16_t FP4_E1M2_THRESHOLD = 0x0080;
constexpr uint16_t FP4_NEW_MANTISSA = 0x0008;
constexpr uint16_t BF16_SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint32_t FP32_SPECIAL_EXP_THRESHOLD = 0x00400000;
constexpr int16_t BF16_SHR_NUM = 7;
constexpr int16_t FP32_SHR_NUM = 23;
constexpr int16_t FP32_PACK_SHR_NUM = 16;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint32_t FP4_E2M1_FP32_MX_MAX_EXP = 0x01000000;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint32_t FP32_MX_EXP_BIAS = 0x7f000000;
constexpr int64_t MODE_ROUND = 0;
constexpr int64_t MODE_FLOOR = 1;
constexpr int64_t MODE_RINT = 4;
constexpr uint16_t FP4_E1M2_BF16_MAX_EXP = 0x0000;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr int32_t FP32_BIAS_VALUE = 127;
constexpr int32_t FP32_BIAS_NEG_VALUE = -127;
constexpr int32_t FP32_NEG_ONE = -1;
constexpr float FP4_SCALE_FACTOR = 4.0;
constexpr float FP4_INV_SCALE_FACTOR = 0.25;
constexpr int32_t FP32_NEG_ZERO_BITS = 0x80000000;
constexpr int32_t SCALE_BUFFER_SIZE = 16 * 1024;
constexpr int32_t MAX_MTE_BLOCK_COUNT = 4095;
constexpr uint16_t FP32_NAN_PACK = 0x00007f81;
constexpr uint16_t BF16_ABS_MASK = 0x7fff;
constexpr uint32_t FP32_ABS_MASK = 0x7fffffff;
constexpr uint32_t FP32_SUB_NUM_FOR_SCALE = 0x000000e1;
constexpr uint16_t BF16_SUB_NUM_FOR_SCALE = 0x00e1;
constexpr uint32_t FP32_MX_MAN_MASK = 0x007fffff;
constexpr uint32_t FP32_MX_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t FP8_E5M2_MAX_FLOAT_BITS = 0x37924925; // 1/57344的float32表示 57334是E5M2所能表示的最大值
constexpr uint32_t FP8_E4M3_MAX_FLOAT_BITS = 0x3b124925; // 1/448的float32表示 448是E4M3所能表示的最大值
constexpr uint32_t FP32_NUMBER_ZERO = 0x00000000;
constexpr uint32_t FP32_NUMBER_254 = 0x000000fe;
constexpr uint32_t FP32_NUMBER_HALF = 0x00400000;
constexpr float FP8_E4M3_INV_MAX = 0.002232142857;
constexpr float FP8_E5M2_INV_MAX = 0.000017438616;
constexpr uint16_t BF16_MAX = 0x7fff;
constexpr uint16_t BF16_POS_INF = 0x7f80;
constexpr uint16_t BF16_NEG_INF = 0xff80;
constexpr uint16_t FP16_INF = 0x7c00;
constexpr uint16_t FP16_MAX_EXP = 0x7c00;
constexpr uint16_t FP16_INVALID = 0x7c00;
constexpr uint16_t FP16_NEG_INF = 0xfc00;
constexpr float INV_DTYPE_MAX = 0;
constexpr uint32_t FP32_LAST_16_MAN_MASK = 0x0000ffff;
constexpr uint32_t FP32_MAN_ODD_MASK = 0x00010000;

template <typename T>
__aicore__ inline constexpr int16_t GetShrNum()
{
    if constexpr (IsSame<T, bfloat16_t>::value) {
        return BF16_SHR_NUM;
    } else {
        return FP32_SHR_NUM;
    }
}

template <typename T>
__aicore__ inline constexpr T GetMaxExp()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return BF16_MAX_EXP;
    } else {
        return FP32_MX_MAX_EXP;
    }
}

template <typename T>
__aicore__ inline constexpr T GetAbsForX()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return BF16_ABS_MASK;
    } else {
        return FP32_ABS_MASK;
    }
}

template <typename T>
__aicore__ inline constexpr T GetSubNumForScale()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return FP32_SUB_NUM_FOR_SCALE;
    } else {
        return BF16_SUB_NUM_FOR_SCALE;
    }
}

template <typename T>
__aicore__ inline constexpr T GetFp4MaxExp()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return FP4_E2M1_BF16_MAX_EXP;
    } else {
        return FP4_E2M1_FP32_MX_MAX_EXP;
    }
}

template <typename T>
__aicore__ inline constexpr T GetFp8MaxExp()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return FP8_DEFAULT_MAX_EXP;
    } else {
        return FP8_MAX_EXP_IN_FP32;
    }
}

template <typename T>
__aicore__ inline constexpr T GetMaxBias()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return BF16_EXP_BIAS;
    } else {
        return FP32_MX_EXP_BIAS;
    }
}

template <typename T>
__aicore__ inline constexpr T GetNanValue()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return BF16_NAN_CUSTOM;
    } else {
        return FP32_NAN_CUSTOM;
    }
}

template <typename T>
__aicore__ inline constexpr T GetSpecialExp()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return BF16_SPECIAL_EXP_THRESHOLD;
    } else {
        return FP32_SPECIAL_EXP_THRESHOLD;
    }
}

template <AscendC::RoundMode roundMode, typename outType, typename inType>
__aicore__ inline void CalcElement(
    AscendC::Reg::RegTensor<inType>& in, AscendC::Reg::RegTensor<int32_t>& maxEle, AscendC::Reg::MaskReg mask)
{
    AscendC::Reg::RegTensor<float> y1;
    AscendC::Reg::MaskReg negValueMask;
    AscendC::Reg::MaskReg zeroMask;
    AscendC::Reg::MaskReg negZeroMask;
    AscendC::Reg::MaskReg zeroNegMask;
    AscendC::Reg::RegTensor<int32_t> negZero;
    AscendC::Reg::Duplicate(negZero, FP32_NEG_ZERO_BITS);
    AscendC::Reg::CompareScalar<int32_t, AscendC::CMPMODE::EQ>(
        zeroNegMask, (AscendC::Reg::RegTensor<int32_t>&)in, FP32_NEG_ZERO_BITS, mask);
    if constexpr (IsSame<outType, fp4x2_e2m1_t>::value) {
        AscendC::Reg::RegTensor<int32_t> exp1;
        AscendC::Reg::RegTensor<int32_t> exp2;
        AscendC::Reg::And(exp1, (AscendC::Reg::RegTensor<int32_t>&)in, maxEle, mask);
        AscendC::Reg::ShiftRights(exp1, exp1, FP32_SHR_NUM, mask);
        AscendC::Reg::Adds(exp1, exp1, FP32_BIAS_NEG_VALUE, mask);
        AscendC::Reg::Maxs(exp1, exp1, 0, mask);
        AscendC::Reg::Adds(exp1, exp1, FP32_NEG_ONE, mask);
        AscendC::Reg::Muls(exp2, exp1, FP32_NEG_ONE, mask);
        AscendC::Reg::Adds(exp2, exp2, FP32_BIAS_VALUE, mask);
        AscendC::Reg::ShiftLefts(exp2, exp2, FP32_SHR_NUM, mask);

        AscendC::Reg::Mul(y1, in, (AscendC::Reg::RegTensor<float>&)exp2, mask);
        AscendC::Reg::Adds(exp1, exp1, FP32_BIAS_VALUE, mask);
        AscendC::Reg::ShiftLefts(exp1, exp1, FP32_SHR_NUM, mask);
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(negValueMask, y1, 0, mask);
        AscendC::Reg::Truncate<float, roundMode>(y1, y1, mask);
        AscendC::Reg::Mul(in, y1, (AscendC::Reg::RegTensor<float>&)exp1, mask);
    } else {
        AscendC::Reg::Muls(y1, in, FP4_SCALE_FACTOR, mask);
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(negValueMask, y1, 0, mask);
        AscendC::Reg::Truncate<float, roundMode>(y1, y1, mask);
        AscendC::Reg::Muls(in, y1, FP4_INV_SCALE_FACTOR, mask);
    }
    AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::EQ>(zeroMask, in, 0, mask);
    AscendC::Reg::MaskAnd(negZeroMask, zeroMask, negValueMask, mask);
    AscendC::Reg::MaskOr(zeroMask, negZeroMask, zeroNegMask, mask);
    AscendC::Reg::Copy((AscendC::Reg::RegTensor<int32_t>&)in, negZero, zeroMask);
}

template <AscendC::RoundMode roundMode, typename outType, typename inType, typename calcTypeInt>
__aicore__ inline void CalcElement(
    AscendC::Reg::RegTensor<inType>& in, AscendC::Reg::RegTensor<calcTypeInt>& scaleReprocal,
    AscendC::Reg::RegTensor<calcTypeInt>& maxEle, AscendC::Reg::RegTensor<uint8_t>& out, AscendC::Reg::MaskReg mask)
{
    static constexpr AscendC::Reg::CastTrait castTrait = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr AscendC::Reg::CastTrait castTraitFp32ToBf16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING, roundMode};
    AscendC::Reg::RegTensor<bfloat16_t> valueRegTensor;
    AscendC::Reg::RegTensor<outType> y;
    AscendC::Reg::RegTensor<uint16_t> yRegTensor;
    AscendC::Reg::Mul(in, in, (AscendC::Reg::RegTensor<inType>&)scaleReprocal, mask);
    if constexpr (IsSame<inType, float>::value) {
        CalcElement<roundMode, outType, inType>(in, (AscendC::Reg::RegTensor<int32_t>&)maxEle, mask);
        AscendC::Reg::Cast<bfloat16_t, inType, castTraitFp32ToBf16>(valueRegTensor, in, mask);
        AscendC::Reg::Pack(
            (AscendC::Reg::RegTensor<uint16_t>&)valueRegTensor, (AscendC::Reg::RegTensor<uint32_t>&)valueRegTensor);
        AscendC::Reg::Cast<outType, bfloat16_t, castTrait>(y, valueRegTensor, mask);
    } else {
        AscendC::Reg::Cast<outType, inType, castTrait>(y, in, mask);
    }

    AscendC::Reg::Pack(yRegTensor, (AscendC::Reg::RegTensor<uint32_t>&)y);
    AscendC::Reg::Pack(out, yRegTensor);
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_COMMON_H
