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
 * \file bn3_d_training_update_tiling_common.h
 * \brief
 */
#ifndef BN3_D_TRAINING_UPDATE_TILING_COMMON_H
#define BN3_D_TRAINING_UPDATE_TILING_COMMON_H

#include <cstdint>
#include <vector>

#include "graph/types.h" // ge::Format / ge::DataType

namespace optiling {

// ---- §5.3 common design constants (single source of truth) ----
//   Mirrored verbatim from DESIGN §5.9 (UB capacity verification):
//     ub_per_core  = 256 × 1024 = 262144 B  (Ascend 950 per-core UB)
//     kPhysNodes   = 5                         (segment-B dominant physical TBuf slots)
//     per_buf_bytes = (ub_per_core / kPhysNodes) & ~31 = (262144 / 5) & ~31 = 52416 B
constexpr int64_t kBn3dUbPerCore = 262144;                                    // bytes (256 KiB)
constexpr int64_t kBn3dPhysNodes = 5;                                         // kPhysNodes (segment-B peak)
constexpr int64_t kBn3dPerBufBytes = (kBn3dUbPerCore / kBn3dPhysNodes) & ~31; // = 52416

// ---- §5.3 UB-split result (mirrors kernel-side SplitResult; POD-only) ----
struct Bn3dSplitResult {
    int64_t axis;     // UB split axis (channel-last internal view, innermost-out numbering)
    int64_t a_i;      // inner-axis tile size (element count, along ub_split_axis)
    int64_t a_o;      // outer-axis tile count
    int64_t a_i_tail; // tail-block size (element count)
};

// ---- §5.3 multi-core result (mirrors kernel-side MultiCoreResult; POD-only) ----
struct Bn3dMultiCoreResult {
    int64_t num_cores;   // cores participating in the computation
    int64_t total_tiles; // total tile count (x along the main split axis)
    int64_t tiles_main;  // main-tile count per core
    int64_t cores_tail;  // cores that handle one extra tile
};

// ============================================================================
// §3.1.2 format → channel_axis resolution
//   OpDef declares only explicit formats (NCHW/NCDHW/NHWC/NDHWC — no ND).
//   Channel axis is determined directly from the format table.
//   Any unexpected format returns -1 (error).
// ============================================================================
int32_t FormatToChannelAxis(ge::Format fmt, int64_t rank_x, const std::vector<int64_t>& x_shape, int64_t C);

// ============================================================================
// §5.1 rank(x) → tilingKey mapping
//   rank_x == 4 → 0   (NCHW / NHWC,  RANK=4)
//   rank_x == 5 → 1   (NCDHW / NDHWC, RANK=5)
//   otherwise  → -1   (shape_mismatch — InferShape should already have rejected)
//   Stub: returns INT32_MIN sentinel so the UT 0-pass baseline is honest
//   (avoids coincidentally matching either 0, 1, or -1).
// ============================================================================
int32_t ChooseTilingKeyBn3d(int64_t rank_x);

// ============================================================================
// §5.3 PadAndSqueezeBn3d
//   Normalises the 7 inputs + 5 outputs into the channel-last internal view
//   max_bro_shape = [N, D?, H, W, C] (rank = rank_x, last dim = C).
//
//   - x (input 0) is rearranged by moving channel_axis to the last position;
//     the other axes keep their original order.
//   - the 6 (C,) statistics inputs (sum/square_sum/scale/offset/mean/variance)
//     become [1,1,...,1,C] (rank-1 leading 1s + trailing C = broadcast tensors).
//   - y (output 0) follows x; the 4 (C,) statistics outputs follow the
//     statistics broadcast pattern.
//
//   Stub: leaves outputs EMPTY (caller-visible: out vectors unchanged) and
//   returns false. The UT compares each element and so fails on every case
//   (honest 0-pass baseline).
// ============================================================================
bool PadAndSqueezeBn3d(const std::vector<std::vector<int64_t>>& input_shapes,  // 7 inputs (raw)
                       const std::vector<std::vector<int64_t>>& output_shapes, // 5 outputs (raw)
                       int64_t channel_axis,                                   // {NCHW:1,NCDHW:1,NHWC:3,NDHWC:4}
                       std::vector<int64_t>& max_bro_shape,                    // out: channel-last view
                       std::vector<std::vector<int64_t>>& normal_input_shapes,
                       std::vector<std::vector<int64_t>>& normal_output_shapes);

// ============================================================================
// §5.3 CheckBroadcastShapeBn3d
//   After normalisation, verifies every tensor agrees on the C dimension
//   (last axis) and that x.shape == y.shape. Returns true on success.
//   Stub: returns false unconditionally.
// ============================================================================
bool CheckBroadcastShapeBn3d(const std::vector<std::vector<int64_t>>& normal_input_shapes,
                             const std::vector<std::vector<int64_t>>& normal_output_shapes, int64_t rank);

// ============================================================================
// §5.3 PrecomputeInputStrides
//   For a normalised (channel-last) shape, compute the GM stride for each axis.
//   Broadcast axes (size == 1, except the trailing C axis) get stride 0; the
//   trailing C axis always gets stride 1 (it is the only non-broadcast dim for
//   the (C,) statistics). For x (which has no broadcast axis), strides are the
//   standard row-major strides.
//
//   Implementation note (DESIGN §5.3 line 714):
//     "(C,) 统计量 input_strides 前 rank-1 轴 stride=0, 末轴 stride=1"
//
//   Stub: leaves strides EMPTY.
// ============================================================================
void PrecomputeInputStrides(const std::vector<int64_t>& normal_shape, // channel-last view of one tensor
                            int64_t rank,
                            std::vector<int64_t>& strides); // out: rank-sized stride vector

// ============================================================================
// §5.3 PrecomputeInputStridesByOrigin
//   Like PrecomputeInputStrides but for a tensor stored DENSELY in its ORIGINAL
//   storage order (e.g. x in NCHW or NHWC). Computes GM strides in the
//   channel-last internal coordinate system, accounting for the channel_axis
//   position in the original layout.
//
//   For x (input 0) and y (output 0), this returns the correct strides for the
//   kernel to read/write GM via the channel-last view while the underlying GM
//   data is laid out in the caller's original format (e.g. NCHW dense).
//
//   For the (C,) statistics tensors, use PrecomputeInputStrides (broadcast
//   pattern [1,..,1,C]).
// ============================================================================
void PrecomputeInputStridesByOrigin(const std::vector<int64_t>& original_shape, // e.g. [N,C,H,W] for NCHW x
                                    int64_t channel_axis,                       // index of C in original_shape
                                    int64_t rank,
                                    std::vector<int64_t>& strides); // out: rank-sized in channel-last view

// ============================================================================
// §5.3 FindSplitAxisBn3d
//   Innermost-out scan over max_bro_shape to pick the UB single-split axis.
//   per_buf_bytes = (ub_per_core / phys_nodes) & ~31
//   per_buf_elems = per_buf_bytes / dtype_size
//   Walk k = rank-1 → 0; inner *= shape[k] accumulates the inner-axis product.
//   On the first k where shape[k]*inner > per_buf_elems:
//       a_i       = per_buf_elems / inner
//       a_o       = ceildiv(shape[k], a_i)
//       a_i_tail  = shape[k] - (a_o - 1) * a_i  (== shape[k] % a_i, or a_i if 0)
//       axis      = k
//   If never triggered: axis=0, a_i=shape[0], a_o=1, a_i_tail=shape[0].
//
//   Stub: leaves `out` UNTOUCHED (caller memset-zeroes it first).
// ============================================================================
void FindSplitAxisBn3d(const std::vector<int64_t>& max_bro_shape,
                       int64_t dtype_size,  // sizeof(T): {f16:2, f32:4, bf16:2}
                       int64_t ub_per_core, // = kBn3dUbPerCore
                       int64_t phys_nodes,  // = kBn3dPhysNodes
                       Bn3dSplitResult& out);

// ============================================================================
// §5.3 MultiCoreSplitBn3d
//   outer_prod = ∏_{j<ub_split.axis} max_bro_shape[j]
//   total_tiles = outer_prod * ub_split.a_o
//   num_cores   = min(total_tiles, max_cores); if total_tiles==0 → 0
//   tiles_main  = total_tiles / num_cores   (when num_cores > 0)
//   cores_tail  = total_tiles % num_cores   (when num_cores > 0)
//
//   Stub: leaves `out` UNTOUCHED.
// ============================================================================
void MultiCoreSplitBn3d(const std::vector<int64_t>& max_bro_shape, const Bn3dSplitResult& ub_split, int64_t max_cores,
                        Bn3dMultiCoreResult& out);

} // namespace optiling

#endif
