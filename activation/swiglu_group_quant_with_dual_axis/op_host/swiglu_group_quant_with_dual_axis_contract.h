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
 * \file swiglu_group_quant_with_dual_axis_contract.h
 * \brief Input, output and attribute index contract shared by the host and kernel sides.
 */

#ifndef OPS_NN_SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_CONTRACT_H
#define OPS_NN_SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_CONTRACT_H

#include <cstddef>
#include <cstdint>

namespace ops::swiglu_group_quant_with_dual_axis {
enum InputIndex : size_t { X = 0, WEIGHT = 1, GROUP_INDEX = 2 };
enum OutputIndex : size_t { Y1 = 0, MX_SCALE1 = 1, Y2 = 2, MX_SCALE2 = 3, Y_ORIGIN = 4 };
enum AttrIndex : size_t { DST_TYPE = 0, QUANT_MODE = 1, CLAMP_LIMIT = 2, OUTPUT_ORIGIN = 3, ALPHA = 4, BIAS = 5 };

constexpr int64_t kDstTypeE5M2 = 35;
constexpr int64_t kDstTypeE4M3Fn = 36;
constexpr int64_t kDualAxisMode = 1;
constexpr int64_t kMxBlockSize = 32;
constexpr int64_t kScalePair = 2;
constexpr int64_t kScalePairElements = 64;
constexpr int64_t kUnknownDim = -1;
} // namespace ops::swiglu_group_quant_with_dual_axis

#endif
