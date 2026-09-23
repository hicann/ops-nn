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
 * \file swiglu_group_quant_axis1_common.h
 * \brief Numerical contract shared by single-axis MX V2 and dual-axis route 1.
 */

#ifndef OPS_NN_SWIGLU_GROUP_QUANT_AXIS1_COMMON_H
#define OPS_NN_SWIGLU_GROUP_QUANT_AXIS1_COMMON_H

#include "kernel_operator.h"

namespace SwigluGroupQuantAxis1 {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t FP32_EXPONENT_MASK = 0x7f800000U;
constexpr uint32_t FP32_MANTISSA_MASK = 0x007fffffU;
constexpr uint32_t FP32_HALF_MANTISSA = 0x00400000U;
constexpr uint32_t E8M0_EXPONENT_BIAS_SUM = 254U;
constexpr uint32_t E8M0_NAN = 255U;

constexpr CastTrait CAST_B16_TO_B32_LAYOUT0 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                               RoundMode::UNKNOWN};
constexpr CastTrait CAST_B16_TO_B32_LAYOUT1 = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                               RoundMode::UNKNOWN};
constexpr CastTrait CAST_B32_TO_B16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait CAST_F32_TO_FP8 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait CAST_F32_TO_FP8_LAYOUT1 = {RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                               RoundMode::CAST_RINT};
constexpr CastTrait CAST_F32_TO_FP8_LAYOUT2 = {RegLayout::TWO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                               RoundMode::CAST_RINT};
constexpr CastTrait CAST_F32_TO_FP8_LAYOUT3 = {RegLayout::THREE, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                               RoundMode::CAST_RINT};

template <typename T>
__simd_callee__ inline void LoadInput(RegTensor<T>& packed, RegTensor<float>& layout0, RegTensor<float>& layout1,
                                      __ubuf__ T* input, MaskReg& mask)
{
    LoadAlign(packed, input);
    Cast<float, T, CAST_B16_TO_B32_LAYOUT0>(layout0, packed, mask);
    Cast<float, T, CAST_B16_TO_B32_LAYOUT1>(layout1, packed, mask);
}

template <typename T>
__simd_callee__ inline void RoundToInput(RegTensor<float>& value, MaskReg& mask)
{
    RegTensor<T> rounded;
    Cast<T, float, CAST_B32_TO_B16>(rounded, value, mask);
    Cast<float, T, CAST_B16_TO_B32_LAYOUT0>(value, rounded, mask);
}

template <typename T>
__simd_callee__ inline void StoreInput(__ubuf__ T* output, RegTensor<float>& value, MaskReg& mask, uint32_t offset)
{
    RegTensor<T> packed;
    Cast<T, float, CAST_B32_TO_B16>(packed, value, mask);
    StoreAlign<T, StoreDist::DIST_PACK_B32>(output + offset, packed, mask);
}

template <typename T, bool hasClamp, bool hasAttrs = true>
__simd_callee__ inline void ActivationCoreWithScratch(RegTensor<float>& output, RegTensor<float>& a,
                                                      RegTensor<float>& b, RegTensor<float>& neg, RegTensor<float>& exp,
                                                      RegTensor<float>& denominator, RegTensor<float>& sigmoid,
                                                      RegTensor<float>& gate, MaskReg& mask, float clampLimit,
                                                      float alpha, float bias)
{
    if constexpr (hasClamp) {
        Mins(a, a, clampLimit, mask);
        Mins(b, b, clampLimit, mask);
        Maxs(b, b, -clampLimit, mask);
    }
    if constexpr (hasAttrs) {
        Muls(neg, a, -alpha, mask);
    } else {
        Muls(neg, a, -1.0F, mask);
    }
    Exp(exp, neg, mask);
    Adds(denominator, exp, 1.0F, mask);
    Div(sigmoid, a, denominator, mask);
    if constexpr (hasAttrs) {
        Adds(gate, b, bias, mask);
        Mul(output, sigmoid, gate, mask);
    } else {
        Mul(output, sigmoid, b, mask);
    }
}

template <typename T, bool hasClamp, bool hasAttrs = true>
__simd_callee__ inline void ActivationCore(RegTensor<float>& output, RegTensor<float>& a, RegTensor<float>& b,
                                           RegTensor<float>& tmp, MaskReg& mask, float clampLimit, float alpha,
                                           float bias)
{
    ActivationCoreWithScratch<T, hasClamp, hasAttrs>(output, a, b, output, output, output, output, tmp, mask,
                                                     clampLimit, alpha, bias);
}

template <typename T, bool hasClamp, bool hasAttrs = true>
__simd_callee__ inline void ActivationWithAttrs(RegTensor<float>& output, RegTensor<float>& a, RegTensor<float>& b,
                                                RegTensor<float>& tmp, MaskReg& mask, float clampLimit, float alpha,
                                                float bias)
{
    ActivationCore<T, hasClamp, hasAttrs>(output, a, b, tmp, mask, clampLimit, alpha, bias);
    RoundToInput<T>(output, mask);
}

template <typename T, bool hasWeight>
__simd_callee__ inline void ApplyWeight(RegTensor<float>& value, RegTensor<float>& weight, MaskReg& mask)
{
    if constexpr (hasWeight) {
        Mul(value, value, weight, mask);
        RoundToInput<T>(value, mask);
    }
}

template <typename TW>
__simd_callee__ inline void LoadWeight(RegTensor<float>& weight, RegTensor<TW>& packed, __ubuf__ TW* input,
                                       MaskReg& mask)
{
    if constexpr (IsSameType<TW, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(weight, input);
    } else {
        LoadAlign<TW, LoadDist::DIST_BRC_B16>(packed, input);
        Cast<float, TW, CAST_B16_TO_B32_LAYOUT0>(weight, packed, mask);
    }
}

__simd_callee__ inline void ReduceMxAmax(RegTensor<float>& amax, RegTensor<float>& value0, RegTensor<float>& value1,
                                         MaskReg& allMask, MaskReg& scaleMask)
{
    RegTensor<float> max0;
    RegTensor<float> max1;
    RegTensor<float> maxLayout0;
    RegTensor<float> maxLayout1;
    Abs(value0, value0, allMask);
    Abs(value1, value1, allMask);
    Max(max0, value0, value1, allMask);
    ReduceDataBlock<ReduceType::MAX>(max1, max0, allMask);
    DeInterleave(maxLayout0, maxLayout1, max1, max1);
    Max(amax, maxLayout0, maxLayout1, scaleMask);
}

template <bool inverseB16 = false>
__simd_callee__ inline void EncodeMxScale(RegTensor<uint32_t>& encoded, RegTensor<uint32_t>& inverse,
                                          RegTensor<float>& amax, RegTensor<uint32_t>& coefficient, MaskReg& mask)
{
    RegTensor<float> scaled;
    RegTensor<uint32_t> exponent;
    RegTensor<uint32_t> mantissa;
    RegTensor<uint32_t> plusOne;
    RegTensor<uint32_t> zero;
    RegTensor<uint32_t> mantissaMask;
    RegTensor<uint32_t> bias254;
    RegTensor<uint32_t> nanScale;
    RegTensor<uint32_t> nanInverse;
    MaskReg normal;
    MaskReg subnormal;
    MaskReg roundUp;
    MaskReg isZero;
    MaskReg nonFinite;
    Duplicate(zero, static_cast<uint32_t>(0), mask);
    Duplicate(mantissaMask, FP32_MANTISSA_MASK, mask);
    Duplicate(bias254, E8M0_EXPONENT_BIAS_SUM, mask);
    Duplicate(nanScale, E8M0_NAN, mask);
    if constexpr (inverseB16) {
        Duplicate(nanInverse, static_cast<uint32_t>(0x00007f81U), mask);
    } else {
        Duplicate(nanInverse, static_cast<uint32_t>(0x7f810000U), mask);
    }
    Compare<uint32_t, CMPMODE::EQ>(isZero, (RegTensor<uint32_t>&)amax, zero, mask);
    Compares<uint32_t, CMPMODE::GE>(nonFinite, (RegTensor<uint32_t>&)amax, FP32_EXPONENT_MASK, mask);
    Mul(scaled, amax, (RegTensor<float>&)coefficient, mask);
    ShiftRights(exponent, (RegTensor<uint32_t>&)scaled, static_cast<int16_t>(23), mask);
    And(mantissa, (RegTensor<uint32_t>&)scaled, mantissaMask, mask);
    Compares<uint32_t, CMPMODE::GT>(normal, exponent, static_cast<uint32_t>(0), mask);
    Compares<uint32_t, CMPMODE::LT>(normal, exponent, E8M0_EXPONENT_BIAS_SUM, normal);
    Compares<uint32_t, CMPMODE::GT>(normal, mantissa, static_cast<uint32_t>(0), normal);
    Compares<uint32_t, CMPMODE::EQ>(subnormal, exponent, static_cast<uint32_t>(0), mask);
    Compares<uint32_t, CMPMODE::GT>(subnormal, mantissa, FP32_HALF_MANTISSA, subnormal);
    Or(roundUp, normal, subnormal, mask);
    Adds(plusOne, exponent, 1, mask);
    Select(encoded, plusOne, exponent, roundUp);
    Select(encoded, nanScale, encoded, nonFinite);
    Select(encoded, zero, encoded, isZero);
    Sub(inverse, bias254, encoded, mask);
    if constexpr (inverseB16) {
        ShiftLefts(inverse, inverse, static_cast<int16_t>(7), mask);
    } else {
        ShiftLefts(inverse, inverse, static_cast<int16_t>(23), mask);
    }
    Select(inverse, nanInverse, inverse, nonFinite);
    Select(inverse, zero, inverse, isZero);
}

template <typename T, uint32_t layout = 0>
__simd_callee__ inline void CastQuantized(RegTensor<T>& quantized, RegTensor<float>& value, MaskReg& mask)
{
    if constexpr (IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value) {
        if constexpr (layout == 0) {
            Cast<T, float, CAST_F32_TO_FP8>(quantized, value, mask);
        } else if constexpr (layout == 1) {
            Cast<T, float, CAST_F32_TO_FP8_LAYOUT1>(quantized, value, mask);
        } else if constexpr (layout == 2) {
            Cast<T, float, CAST_F32_TO_FP8_LAYOUT2>(quantized, value, mask);
        } else {
            Cast<T, float, CAST_F32_TO_FP8_LAYOUT3>(quantized, value, mask);
        }
    }
}

template <typename T>
__simd_callee__ inline void QuantizeAndStore(RegTensor<float>& value, RegTensor<float>& inverse, __ubuf__ T* output,
                                             MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value) {
        RegTensor<T> quantized;
        Mul(value, value, inverse, mask);
        CastQuantized<T>(quantized, value, mask);
        StoreAlign<T, StoreDist::DIST_PACK4_B32>(output + offset, quantized, mask);
    }
}
} // namespace SwigluGroupQuantAxis1

#endif // OPS_NN_SWIGLU_GROUP_QUANT_AXIS1_COMMON_H
