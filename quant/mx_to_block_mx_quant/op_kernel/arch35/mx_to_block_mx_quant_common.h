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
 * \file mx_to_block_mx_quant_common.h
 * \brief Common definitions and helper functions for MxToBlockMxQuant kernel.
 */

#ifndef MX_TO_BLOCK_MX_QUANT_COMMON_H
#define MX_TO_BLOCK_MX_QUANT_COMMON_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "../inc/platform.h"
#include "mx_to_block_mx_quant_tilingdata.h"

namespace MxToBlockMxQuantNs {
using namespace AscendC;

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
__aicore__ inline constexpr bool IsFp4Type()
{
    return IsSame<T, fp4x2_e2m1_t>::value || IsSame<T, fp4x2_e1m2_t>::value;
}

template <typename T>
__aicore__ inline constexpr bool IsFp8Type()
{
    return IsSame<T, fp8_e4m3fn_t>::value || IsSame<T, fp8_e5m2_t>::value;
}

constexpr uint64_t TPL_ROW_ALIGNED = 0;
constexpr uint64_t TPL_ROW_NOT_ALIGNED = 1;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_SIXTEEN = 16;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t SPLIT_M = 64;
constexpr int64_t SPLIT_N = 512;
constexpr int64_t MXSCALE_NUM_ALIGN32 = 32;
constexpr int64_t SCALE_TMP_SIZE = 256;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;

constexpr uint32_t vfLen8 = platform::GetVRegSize() / sizeof(uint8_t);   // 256
constexpr uint32_t vfLen16 = platform::GetVRegSize() / sizeof(uint16_t); // 128
constexpr uint32_t vfLen32 = platform::GetVRegSize() / sizeof(uint32_t); // 64
constexpr int64_t UBBlockSize_ = platform::GetUbBlockSize();

constexpr uint32_t DIGIT_EIGHT = 8;
constexpr uint16_t scaleLoopStep = static_cast<uint16_t>(UBBlockSize_);
constexpr uint16_t scale1OutLoopNum = 4;

static constexpr Reg::CastTrait castTraitFp8E8M0ToBf16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

static constexpr Reg::CastTrait castTraitFp4ToBf16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

static constexpr Reg::CastTrait castTraitUint8ToUint16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

static constexpr Reg::CastTrait castTraitBf16ToFp32_0 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr Reg::CastTrait castTraitBf16ToFp32_1 = {Reg::RegLayout::ONE, Reg::SatMode::SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

static constexpr Reg::CastTrait castTrait32to80 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING,
                                                   RoundMode::CAST_RINT};
static constexpr Reg::CastTrait castTrait32to81 = {Reg::RegLayout::ONE, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING,
                                                   RoundMode::CAST_RINT};
static constexpr Reg::CastTrait castTrait32to82 = {Reg::RegLayout::TWO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING,
                                                   RoundMode::CAST_RINT};
static constexpr Reg::CastTrait castTrait32to83 = {Reg::RegLayout::THREE, Reg::SatMode::SAT,
                                                   Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

} // namespace MxToBlockMxQuantNs

#endif // MX_TO_BLOCK_MX_QUANT_COMMON_H
