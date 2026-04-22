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
 * \file grouped_dynamic_mx_quant_common.h
 * \brief
 */

#ifndef GRROUPED_DYNAMIC_MX_QUANT_COMMON_H
#define GRROUPED_DYNAMIC_MX_QUANT_COMMON_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "grouped_dynamic_mx_quant_tilingdata.h"

namespace GroupedDynamicMxQuant {
template<typename Tp, Tp v>
struct IntegralConstant {
  static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {
};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {
};

constexpr int64_t DB_BUFFER = 2;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;

constexpr uint16_t ABS_FOR_UINT16 = 0x7fff;
constexpr uint32_t MAN_FOR_FP32 = 0x007fffff;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
constexpr uint32_t NUMBER_HALF = 0x00400000;
constexpr float FP8_E4M3_INV_MAX = 0.002232142857;        // 1/448
constexpr float FP8_E5M2_INV_MAX = 0.000017438616;        // 1/57344
}
#endif // GRROUPED_DYNAMIC_MX_QUANT_COMMON_H