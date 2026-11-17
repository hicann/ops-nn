/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>

namespace ChunkLossGradOptions {
constexpr uint32_t BUILD_MASK = 931;
constexpr uint32_t BF16_VT128 = 1;
constexpr uint32_t REUSE_METADATA = 2;
constexpr uint32_t OUTPUT_DOUBLE_BUFFER = 4;
constexpr uint32_t SEPARATE_L_P = 8;
constexpr uint32_t HOIST_SUM_RECIPROCAL = 32;
constexpr uint32_t P_VIA_GM = 64;
constexpr uint32_t REUSE_LOGITS_WEIGHT = 128;
constexpr uint32_t LOGITS_GROUP_TWO = 256;
constexpr uint32_t REUSE_LOGITS_L0A = 512;
constexpr uint32_t MODE_SHIFT = 8;
static_assert(!(BUILD_MASK & HOIST_SUM_RECIPROCAL) || (BUILD_MASK & REUSE_METADATA),
              "Reciprocal hoisting requires metadata reuse");
static_assert(!(BUILD_MASK & P_VIA_GM) || !(BUILD_MASK & (SEPARATE_L_P | OUTPUT_DOUBLE_BUFFER)),
              "P GM comparison uses its own full-task ping-pong buffers");
static_assert(!(BUILD_MASK & LOGITS_GROUP_TWO) || (BUILD_MASK & REUSE_LOGITS_WEIGHT),
              "Two-fragment groups require logits weight reuse");
static_assert(!(BUILD_MASK & REUSE_LOGITS_WEIGHT) || !(BUILD_MASK & P_VIA_GM),
              "Logits weight reuse is validated with the P UB route");
static_assert(!(BUILD_MASK & REUSE_LOGITS_L0A) || (BUILD_MASK & REUSE_LOGITS_WEIGHT),
              "L0A reuse requires grouped logits weight reuse");
} // namespace ChunkLossGradOptions
