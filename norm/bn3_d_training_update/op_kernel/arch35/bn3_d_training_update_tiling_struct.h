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
 * \file bn3_d_training_update_tiling_struct.h
 * \brief
 */
#ifndef BN3_D_TRAINING_UPDATE_TILING_STRUCT_H
#define BN3_D_TRAINING_UPDATE_TILING_STRUCT_H

#include <cstdint>

// ---- §5.2 design constants (shared by all RANK instantiations) ----
constexpr int64_t kMaxInputSlots = 7;  // x, sum, square_sum, scale, offset, mean, variance
constexpr int64_t kMaxOutputSlots = 5; // y, mean, variance, batch_mean, batch_variance
constexpr int64_t kPhysNodes = 5;      // physical live TBuf slots P (see §5.3 conclusion table)

// ---- §5.3 UB-split result (source: FindSplitAxisBn3d) ----
struct SplitResult {
    int64_t axis;     // UB split axis (channel-last internal view, innermost-out numbering)
    int64_t a_i;      // inner-axis tile size (element count, along ub_split_axis)
    int64_t a_o;      // outer-axis tile count
    int64_t a_i_tail; // tail-block size (element count)
};

// ---- §5.3 multi-core result (source: MultiCoreSplitBn3d) ----
struct MultiCoreResult {
    int64_t num_cores;   // cores participating in the computation
    int64_t total_tiles; // total tile count (x along the main split axis)
    int64_t tiles_main;  // main-tile count per core
    int64_t cores_tail;  // cores that handle one extra tile
};

// ---- §5.2 TilingData template struct ----
template <int64_t kRank>
struct BN3DTrainingUpdateTilingData {
    SplitResult split;         // UB split result (source: FindSplitAxis)
    MultiCoreResult multicore; // multi-core split result (source: MultiCoreSplit)
    int64_t rank;              // effective rank (4 or 5)
    int64_t per_buf_bytes;     // per-buffer bytes = (UB / P) & ~31

    int64_t max_bro_shape[kRank]; // channel-last internal view ([N,D?,H,W,C], pad with 1)

    int64_t num_inputs;                             // = 7
    int64_t num_outputs;                            // = 5
    int64_t input_shapes[kMaxInputSlots][kRank];    // padded shapes (stats always (1,..,1,C))
    int64_t input_strides[kMaxInputSlots][kRank];   // GM strides (broadcast axis = 0)
    int64_t output_shapes[kMaxOutputSlots][kRank];  // padded shapes (y = x.shape; stats (1,..,1,C))
    int64_t output_strides[kMaxOutputSlots][kRank]; // GM strides

    // ---- operator parameters (host-precomputed; kernel does not recompute div/branch) ----
    int32_t channel_axis;   // {NCHW:1, NCDHW:1, NHWC:3, NDHWC:4}; used by kernel NDDMA stride
    int32_t C;              // channel count = sum.shape[0]
    int64_t num;            // reduce-domain element count = x.size / C
    int32_t dtype_id;       // {f32:0, f16:1, bf16:2}; kernel selects Process path
    float num_rec;          // 1.0f / num
    float factor;           // attr factor
    float one_minus_factor; // 1.0f - factor
    float epsilon;          // attr epsilon
    float bessel_scaler;    // num==1 ? 0.0f : num / (num - 1)
};

// ---- Concrete (non-template) aliases for the two supported ranks ----
//   Used by the kernel-binary build chain (REGISTER_TILING_DEFAULT needs a plain
//   type name; the template instantiation BN3DTrainingUpdateTilingData<N> cannot
//   serve as a section identifier). Task 24 (kernel impl) consumes these.
using BN3DTrainingUpdateTilingDataRank4 = BN3DTrainingUpdateTilingData<4>;
using BN3DTrainingUpdateTilingDataRank5 = BN3DTrainingUpdateTilingData<5>;

// ---- Compile-time guard: both instantiations must be referenceable at host compile time ----
//   Forces the compiler to fully parse & instantiate the template for RANK=4 and RANK=5
//   wherever this header is included by a host translation unit.
static_assert(sizeof(BN3DTrainingUpdateTilingDataRank4) > 0, "BN3DTrainingUpdateTilingData<4> must be instantiable");
static_assert(sizeof(BN3DTrainingUpdateTilingDataRank5) > 0, "BN3DTrainingUpdateTilingData<5> must be instantiable");

#endif
