/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file quant_batch_matmul_v4_tiling_info.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_TILING_INFO_H
#define QUANT_BATCH_MATMUL_V4_TILING_INFO_H

#include <cstddef>
#include <cstdint>

namespace optiling {

// input IDX
constexpr size_t X1_IDX = 0UL;
constexpr size_t X2_IDX = 1UL;
constexpr size_t BIAS_IDX = 2UL;
constexpr size_t X1_SCALE_IDX = 3UL;
constexpr size_t X2_SCALE_IDX = 4UL;
constexpr size_t X2_OFFSET_IDX = 7UL;
constexpr size_t Y_SCALE_IDX = 5UL;
constexpr size_t Y_OFFSET_IDX = 8UL;
// output IDX
constexpr size_t Y_OUTPUT_IDX = 0UL;
// attr IDX
constexpr size_t TRANSPOSE_X1_IDX = 2UL;
constexpr size_t TRANSPOSE_X2_IDX = 3UL;
constexpr size_t GROUP_SIZE_IDX = 4UL;

// antiquant type
enum class QuantBatchMatmulV4QuantType : std::uint8_t {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4
};

// kernel template type
enum class KernelTemplateType : std::uint8_t {
    MSD_BASIS = 1,
    PERBLOCK_BASIS = 2,
    PERGROUP_BASIS = 3
};
} // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V4_TILING_INFO_H