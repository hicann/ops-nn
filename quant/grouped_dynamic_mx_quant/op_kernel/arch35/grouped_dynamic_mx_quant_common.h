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
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t BLOCK_SIZE = 32;

constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;

} // namespace GroupedDynamicMxQuant
#endif // GRROUPED_DYNAMIC_MX_QUANT_COMMON_H
