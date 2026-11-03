/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NORM_L2_NORMALIZE_SCALING_H_
#define OPS_NORM_L2_NORMALIZE_SCALING_H_

#include "kernel_operator.h"

namespace NsL2Normalize {
// Normal denominators are positive. Negative values encode a rescaled norm.
// For s >= 2^124 use x*2^-96; its norm is >= 2^-34.
// For s < 2^-120 use x*2^96 and store its norm times 2^-72:
// even FLT_TRUE_MIN produces a normal encoded norm (>= 2^-125),
// and the encoded norm is < 2^-36. The disjoint ranges need no extra buffer.
constexpr float SCALE_SUM_HIGH = 0x1p124f;
constexpr float SCALE_SUM_LOW = 0x1p-120f;
constexpr float SCALE_DOWN = 0x1p-96f;
constexpr float SCALE_UP = 0x1p96f;
constexpr float SCALE_LOW_ENCODE = 0x1p-72f;
constexpr float SCALE_LOW_NUMERATOR = 0x1p24f;
constexpr float SCALE_NORM_SPLIT = 0x1p-35f;
constexpr float SCALE_HIGH_MARKER = -1.0f;
constexpr float SCALE_LOW_MARKER = -2.0f;
constexpr uint32_t SCALE_LANES = 64;
constexpr AscendC::Reg::DivSpecificMode SCALE_DIV_MODE{AscendC::Reg::MaskMergeMode::ZEROING, false,
                                                       AscendC::DivAlgo::PRECISION_1ULP_FTZ_FALSE};
// Retained finite norms in the first pass have sums in [2^-120, 2^124),
// below the fast-inverse sqrt limit 2^126. Keep their original 0-ULP mode.
// High/low lanes become retry markers; rescaled passes retain the full-range
// 1-ULP FTZ_FALSE mode independently of the ordinary-input precision choice.
constexpr AscendC::Reg::SqrtSpecificMode NORMAL_SQRT_MODE_0ULP{AscendC::Reg::MaskMergeMode::ZEROING, true,
                                                               AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
constexpr AscendC::Reg::SqrtSpecificMode SCALE_SQRT_MODE{AscendC::Reg::MaskMergeMode::ZEROING, false,
                                                         AscendC::SqrtAlgo::PRECISION_1ULP_FTZ_FALSE};

__simd_vf__ inline void PrepareScaledNormVf(__ubuf__ float* sums, __ubuf__ float* denom, uint32_t count, float eps,
                                            uint16_t repeats)
{
    using namespace AscendC;
    Reg::RegTensor<float> sum, norm, highMarker, lowMarker;
    Reg::MaskReg high, low;
    Reg::Duplicate(highMarker, SCALE_HIGH_MARKER);
    Reg::Duplicate(lowMarker, SCALE_LOW_MARKER);
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeats; ++i) {
        const int32_t off = static_cast<int32_t>(i) * SCALE_LANES;
        auto mask = Reg::UpdateMask<float>(remaining);
        Reg::LoadAlign(sum, sums + off);
        Reg::Maxs(sum, sum, eps, mask);
        Reg::Compares<float, CMPMODE::GE>(high, sum, SCALE_SUM_HIGH, mask);
        Reg::Compares<float, CMPMODE::LT>(low, sum, SCALE_SUM_LOW, mask);
        Reg::Sqrt<float, &NORMAL_SQRT_MODE_0ULP>(norm, sum, mask);
        Reg::Select(norm, highMarker, norm, high);
        Reg::Select(norm, lowMarker, norm, low);
        Reg::StoreAlign(denom + off, norm, mask);
    }
}

// Return bit 0 for overflow-risk slices and bit 1 for underflow-risk slices.
// All-mask accumulation retains flags from preceding full register blocks.
__simd_vf__ inline void ScalingFlagsVf(__ubuf__ float* denom, __ubuf__ float* scratch, uint32_t count, uint16_t repeats)
{
    using namespace AscendC;
    Reg::RegTensor<float> values, zero, one, high, low, highAcc, lowAcc, highRed, lowRed, result;
    Reg::MaskReg isHigh, isLow;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zero, 0.0f);
    Reg::Duplicate(one, 1.0f);
    Reg::Duplicate(highAcc, 0.0f);
    Reg::Duplicate(lowAcc, 0.0f);
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeats; ++i) {
        auto mask = Reg::UpdateMask<float>(remaining);
        Reg::LoadAlign(values, denom + static_cast<int32_t>(i) * SCALE_LANES);
        Reg::Compares<float, CMPMODE::EQ>(isHigh, values, SCALE_HIGH_MARKER, mask);
        Reg::Compares<float, CMPMODE::EQ>(isLow, values, SCALE_LOW_MARKER, mask);
        Reg::Select(high, one, zero, isHigh);
        Reg::Select(low, one, zero, isLow);
        Reg::Max(highAcc, highAcc, high, all);
        Reg::Max(lowAcc, lowAcc, low, all);
    }
    Reg::Reduce<Reg::ReduceType::MAX, float>(highRed, highAcc, all);
    Reg::Reduce<Reg::ReduceType::MAX, float>(lowRed, lowAcc, all);
    Reg::Muls(lowRed, lowRed, 2.0f, all);
    Reg::Add(result, highRed, lowRed, all);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scratch, result, all);
}

__simd_vf__ inline void ScaledPostElewiseVf(__ubuf__ float* sums, __ubuf__ float* denom, uint32_t count, float eps,
                                            float marker, float encode, uint16_t repeats)
{
    using namespace AscendC;
    Reg::RegTensor<float> sum, previous, normalized;
    Reg::MaskReg update;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeats; ++i) {
        const int32_t off = static_cast<int32_t>(i) * SCALE_LANES;
        auto mask = Reg::UpdateMask<float>(remaining);
        Reg::LoadAlign(sum, sums + off);
        Reg::LoadAlign(previous, denom + off);
        Reg::Maxs(sum, sum, eps, mask);
        Reg::Sqrt<float, &SCALE_SQRT_MODE>(normalized, sum, mask);
        Reg::Muls(normalized, normalized, -encode, mask);
        Reg::Compares<float, CMPMODE::EQ>(update, previous, marker, mask);
        Reg::Select(normalized, normalized, previous, update);
        Reg::StoreAlign(denom + off, normalized, mask);
    }
}
} // namespace NsL2Normalize
#endif
