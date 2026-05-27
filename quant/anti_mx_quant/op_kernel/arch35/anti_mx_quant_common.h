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
 * \file anti_mx_quant_common.h
 * \brief Common definitions and helper functions for AntiMxQuant kernel.
 */

#ifndef ANTI_MX_QUANT_COMMON_H
#define ANTI_MX_QUANT_COMMON_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "../inc/platform.h"
#include "anti_mx_quant_tilingdata.h"

namespace AntiMxQuant {

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

template <typename T>
__aicore__ inline constexpr bool IsFp8Type()
{
    return IsSame<T, fp8_e4m3fn_t>::value || IsSame<T, fp8_e5m2_t>::value;
}

template <typename T>
__aicore__ inline constexpr bool IsFp4Type()
{
    return IsSame<T, fp4x2_e2m1_t>::value || IsSame<T, fp4x2_e1m2_t>::value;
}

template <typename T>
__aicore__ inline constexpr bool IsFp16Type()
{
    return IsSame<T, half>::value;
}

template <typename T>
__aicore__ inline constexpr bool IsBf16Type()
{
    return IsSame<T, bfloat16_t>::value;
}

template <typename T>
__aicore__ inline constexpr bool IsFp32Type()
{
    return IsSame<T, float>::value;
}

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_ZERO = 0;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_EIGHT = 8;
constexpr int64_t DIGIT_SIXTEEN = 16;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t SPLIT_N = 512;
constexpr int16_t SHR_NUM_FOR_FP4 = 4;

constexpr uint32_t vfLen8 = platform::GetVRegSize() / sizeof(uint8_t);     // 256
constexpr uint32_t vfLen16 = platform::GetVRegSize() / sizeof(uint16_t);   // 128
constexpr uint32_t vfLen32 = platform::GetVRegSize() / sizeof(uint32_t);   // 64
constexpr uint32_t vfLen16Double = vfLen16 * DIGIT_TWO;                    // 256
constexpr uint32_t vfLen8Double = vfLen8 * DIGIT_TWO;                      // 512
constexpr int64_t UBBlockSize_ = platform::GetUbBlockSize();
constexpr uint16_t elementAfterReduce_ = platform::GetVRegSize() / UBBlockSize_; // 8

// Cast traits for FP8_E8M0 to BF16 (2 layouts: ZERO=even indices, ONE=odd indices)
static constexpr AscendC::Reg::CastTrait castTraitFp8E8M0ToBf16_0 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait castTraitFp8E8M0ToBf16_1 = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

// Cast traits for BF16 to FP32 (2 layouts for 128 BF16 -> 2 FP32 registers)
static constexpr AscendC::Reg::CastTrait castTraitBf16ToFp32_0 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait castTraitBf16ToFp32_1 = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

// Cast trait for FP32 to BF16 (single layout, used by FP4 path BF16->FP16)
static constexpr AscendC::Reg::CastTrait castTraitFp32ToBf16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

// Cast traits for FP32 to BF16 (2 layouts for multi-layout Cast + Add merge)
static constexpr AscendC::Reg::CastTrait castTraitFp32ToBf16_0 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait castTraitFp32ToBf16_1 = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

// Cast trait for FP32 to FP16 (single layout, used by FP4 path BF16->FP16)
static constexpr AscendC::Reg::CastTrait castTraitBf16ToFp16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

// Cast traits for FP32 to FP16 (2 layouts for multi-layout Cast + Add merge)
static constexpr AscendC::Reg::CastTrait castTraitFp32ToFp16_0 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait castTraitFp32ToFp16_1 = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

// Cast traits for FP8 to FP32 (4 layouts for 256 FP8 -> 4 FP32 registers)
static constexpr AscendC::Reg::CastTrait castTraitFp8ToFp32_0 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait castTraitFp8ToFp32_1 = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait castTraitFp8ToFp32_2 = {
    AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait castTraitFp8ToFp32_3 = {
    AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

// Cast trait for FP4 to BF16 (only UNKNOWN layout, need interleave + shift for all 4 values)
static constexpr AscendC::Reg::CastTrait castTraitFp4ToBf16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

} // namespace AntiMxQuant

#endif // ANTI_MX_QUANT_COMMON_H
