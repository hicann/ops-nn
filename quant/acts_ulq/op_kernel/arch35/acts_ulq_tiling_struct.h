/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef ACTS_ULQ_TILING_STRUCT_H_
#define ACTS_ULQ_TILING_STRUCT_H_

#include <cstdint>

constexpr int64_t kMaxInputSlots = 3;
constexpr int64_t kMaxOutputSlots = 4;
constexpr int64_t kPhysNodes = 6;
constexpr int64_t kCastBufs = 1;
constexpr int64_t kTotalBufsFp16 = kPhysNodes + kCastBufs;

struct SplitResult {
    int64_t axis;
    int64_t a_i;
    int64_t a_o;
    int64_t a_i_tail;
};

struct MultiCoreResult {
    int64_t num_cores;
    int64_t total_tiles;
    int64_t tiles_main;
    int64_t cores_tail;
};

template <int64_t kRank>
struct ActsUlqTilingData {
    SplitResult split;
    MultiCoreResult multicore;
    int64_t rank;
    int64_t per_buf_bytes;
    int64_t per_buf_elems;
    int64_t max_bro_shape[kRank];

    int64_t num_inputs;
    int64_t num_outputs;
    int64_t input_shapes[kMaxInputSlots][kRank];
    int64_t input_strides[kMaxInputSlots][kRank];
    int64_t output_shapes[kMaxOutputSlots][kRank];
    int64_t output_strides[kMaxOutputSlots][kRank];

    int64_t fixed_min;
    int64_t num_bits;
};

#endif
