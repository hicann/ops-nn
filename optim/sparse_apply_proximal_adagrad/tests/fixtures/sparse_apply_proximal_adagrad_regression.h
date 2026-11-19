/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_APPLY_PROXIMAL_ADAGRAD_REGRESSION_H_
#define SPARSE_APPLY_PROXIMAL_ADAGRAD_REGRESSION_H_

#include <array>
#include <cstdint>

namespace sparse_apply_proximal_adagrad_regression {
constexpr int64_t ROWS = 2;
constexpr int64_t COLS = 2;
constexpr int32_t INDEX = 1;
constexpr float LR = 1.0F;
constexpr float ZERO = 0.0F;
using StateBits = std::array<uint32_t, ROWS * COLS>;
using Gradient = std::array<float, COLS>;

struct Fixture {
    const char* name;
    StateBits accum;
    Gradient grad;
    StateBits expectedVar;
    StateBits expectedAccum;
};

constexpr StateBits VAR_BITS = {0x40800000U, 0x40a00000U, 0x41000000U, 0xc0800000U}; // [4, 5; 8, -4]
constexpr StateBits SUBNORMAL_ACCUM = {0x3f800000U, 0x40000000U, 0x00000001U, 0x00000001U};
constexpr Fixture ZERO_GRADIENT = {
    "ZeroGradientPreservesSubnormalAccum", SUBNORMAL_ACCUM, {0.0F, 0.0F}, VAR_BITS, SUBNORMAL_ACCUM};

// Row 1: accum 3 + 1 = 4, so lr / sqrt(accum) = 1/2. Row 0 is not indexed.
constexpr Fixture NONZERO_GRADIENT = {
    "NonzeroGradientUpdatesBothStates",
    {0x3f800000U, 0x40000000U, 0x40400000U, 0x40400000U}, // [1, 2; 3, 3]
    {1.0F, -1.0F},
    {0x40800000U, 0x40a00000U, 0x40f00000U, 0xc0600000U}, // [4, 5; 7.5, -3.5]
    {0x3f800000U, 0x40000000U, 0x40800000U, 0x40800000U}, // [1, 2; 4, 4]
};
} // namespace sparse_apply_proximal_adagrad_regression

#endif // SPARSE_APPLY_PROXIMAL_ADAGRAD_REGRESSION_H_
