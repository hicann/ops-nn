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
 * \brief Common definitions and utilities for Swiglu + MX quantization
 */

#ifndef SWIGLU_MX_QUANT_COMMON_H
#define SWIGLU_MX_QUANT_COMMON_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/math_util.h"

namespace SwigluMxQuant {
template <typename Tp, Tp v> struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename> struct IsSame : public falseType {};
template <typename Tp> struct IsSame<Tp, Tp> : public trueType {};
// Constants
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint32_t NAN_CUSTOMIZATION_FP32 = 0x7f810000;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
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
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
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
constexpr uint32_t FP8_E5M2_MAX = 0x37924925; // 1/57344的float32表示 57334是E5M2所能表示的最大值
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925; // 1/448的float32表示 448是E4M3所能表示的最大值
constexpr uint16_t INVALID_FLOAT16 = 0x7c00;
constexpr uint32_t ZERO_FOR_ALL = 0x00000000;
constexpr uint32_t EXP_254 = 0x000000fe;
constexpr uint32_t HALF_FOR_MAN = 0x00400000;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t QUANT_ONCE_NUM = 256;
constexpr int64_t X_ONCE_NUM = 512; // -1轴量化基本块是256, x基本块大小就是512
constexpr int64_t QUANT_ONCE_NUM_FP4 = 128; // 256个fp4的大小
constexpr int64_t SCALE_ONCE_NUM = 8; // 256个数对应的scale是8
constexpr int64_t CONST_64 = 64;
constexpr int64_t CONST_32 = 32; // 每32个数计算一个scale
constexpr int64_t CONST_2 = 2; // swiglu的activate_dim必须对2整除
constexpr int64_t CONST_4 = 4; // 输出是fp4时, x的输入最后一维必须对4整除

static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_FP16_TO_FP32 = { AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };
constexpr static AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16_BF16 = { AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_FP16_TO_FP32_ONE = { AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };
static constexpr AscendC::MicroAPI::CastTrait CAST_HALF_TO_BF16 = { AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC };
static constexpr AscendC::MicroAPI::CastTrait CAST_ZERO = { AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };
static constexpr AscendC::MicroAPI::CastTrait CAST_ONE = { AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };

static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_80 = { AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_81 = { AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_82 = { AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_83 = { AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
} // namespace SwigluMxQuant
#endif // SWIGLU_MX_QUANT_COMMON_H
