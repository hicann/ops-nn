/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// dequantize TilingData — templated by rank
// Location: operators/dequantize/op_kernel/arch35/dequantize_tiling_struct.h
#ifndef DEQUANTIZE_TILING_STRUCT_H_
#define DEQUANTIZE_TILING_STRUCT_H_

#include <cstdint>

constexpr int64_t kMaxInputSlots = 3;  // x + min_range + max_range (all tensor inputs via NDDMA)
constexpr int64_t kMaxOutputSlots = 1; // y
constexpr int64_t kPhysNodes = 5;      // B0=x + B1=min_range + B2=max_range + B3=y + B4=temp

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
struct DequantizeTilingData {
    SplitResult split;
    MultiCoreResult multicore;
    int64_t rank;
    int64_t per_buf_bytes;
    int64_t max_bro_shape[kRank];
    int64_t num_inputs;
    int64_t num_outputs;
    int64_t input_shapes[kMaxInputSlots][kRank];
    int64_t input_strides[kMaxInputSlots][kRank];
    int64_t output_shapes[kMaxOutputSlots][kRank];
    int64_t output_strides[kMaxOutputSlots][kRank];
    float bias;      // int8: 128.0, uint8: 0.0, int32: 2147483648.0; SCALED: 0.0
    float inv_range; // MIN_COMBINED/MIN_FIRST: 1/(2^bits-1); SCALED: 1/max_fixed
};

#endif // DEQUANTIZE_TILING_STRUCT_H_
