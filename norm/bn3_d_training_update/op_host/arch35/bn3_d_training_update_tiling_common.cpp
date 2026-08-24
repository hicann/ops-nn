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
 * \file bn3_d_training_update_tiling_common.cpp
 * \brief
 */
#include "bn3_d_training_update_tiling_common.h"

namespace optiling {

// ============================================================================
// §3.1.2 format → channel_axis resolution
//   Explicit formats: NCHW→1, NCDHW→1, NHWC→3, NDHWC→4.
//   ND fallback: use shape heuristic (same as golden _channel_axis):
//     unique size==C axis; C==1→first size-1; multi-hit→default format axis.
// ============================================================================
int32_t FormatToChannelAxis(ge::Format fmt, int64_t rank_x, const std::vector<int64_t>& x_shape, int64_t C)
{
    switch (fmt) {
        case ge::FORMAT_NCHW:
            return 1;
        case ge::FORMAT_NHWC:
            return 3;
        case ge::FORMAT_NCDHW:
            return 1;
        case ge::FORMAT_NDHWC:
            return 4;
        default:
            return -1;
    }
}

// ============================================================================
// §5.1 rank(x) → tilingKey mapping
//   rank_x == 4 → 0   (NCHW / NHWC,  RANK=4)
//   rank_x == 5 → 1   (NCDHW / NDHWC, RANK=5)
//   otherwise  → -1   (shape_mismatch — InferShape should already have rejected)
// ============================================================================
int32_t ChooseTilingKeyBn3d(int64_t rank_x)
{
    switch (rank_x) {
        case 4:
            return 0;
        case 5:
            return 1;
        default:
            return -1;
    }
}

// ============================================================================
// §5.3 PadAndSqueezeBn3d
//   Normalises the 7 inputs + 5 outputs into the channel-last internal view
//   max_bro_shape = [N, D?, H, W, C] (rank = rank_x, last dim = C).
// ============================================================================
bool PadAndSqueezeBn3d(const std::vector<std::vector<int64_t>>& input_shapes,
                       const std::vector<std::vector<int64_t>>& output_shapes, int64_t channel_axis,
                       std::vector<int64_t>& max_bro_shape, std::vector<std::vector<int64_t>>& normal_input_shapes,
                       std::vector<std::vector<int64_t>>& normal_output_shapes)
{
    const int64_t num_inputs = static_cast<int64_t>(input_shapes.size());
    const int64_t num_outputs = static_cast<int64_t>(output_shapes.size());
    const int64_t rank_x = static_cast<int64_t>(input_shapes[0].size());

    // 1. x's channel-last view: move channel_axis to last, others in original order.
    std::vector<int64_t> x_cl;
    x_cl.reserve(static_cast<size_t>(rank_x));
    for (int64_t d = 0; d < rank_x; ++d) {
        if (d != channel_axis)
            x_cl.push_back(input_shapes[0][static_cast<size_t>(d)]);
    }
    x_cl.push_back(input_shapes[0][static_cast<size_t>(channel_axis)]);

    // 2. coordinate system = x_cl
    max_bro_shape = x_cl;
    const int64_t rank = rank_x;

    // 3. normalise inputs: x same; 6 (C,) stats → [1,..,1,C]
    normal_input_shapes.assign(static_cast<size_t>(num_inputs), std::vector<int64_t>(static_cast<size_t>(rank), 1));
    normal_input_shapes[0] = x_cl;
    for (int64_t i = 1; i < num_inputs; ++i) {
        for (int64_t d = 0; d < rank - 1; ++d) {
            normal_input_shapes[static_cast<size_t>(i)][static_cast<size_t>(d)] = 1;
        }
        normal_input_shapes[static_cast<size_t>(i)]
                           [static_cast<size_t>(rank - 1)] = input_shapes[static_cast<size_t>(i)][0];
    }

    // 4. normalise outputs: y same as x_cl; 4 (C,) outputs → [1,..,1,C]
    normal_output_shapes.assign(static_cast<size_t>(num_outputs), std::vector<int64_t>(static_cast<size_t>(rank), 1));
    normal_output_shapes[0] = x_cl;
    for (int64_t i = 1; i < num_outputs; ++i) {
        for (int64_t d = 0; d < rank - 1; ++d) {
            normal_output_shapes[static_cast<size_t>(i)][static_cast<size_t>(d)] = 1;
        }
        normal_output_shapes[static_cast<size_t>(i)]
                            [static_cast<size_t>(rank - 1)] = output_shapes[static_cast<size_t>(i)][0];
    }
    return true;
}

// ============================================================================
// §5.3 CheckBroadcastShapeBn3d
//   After normalisation, verifies every tensor agrees on the C dimension
//   (last axis). Returns true on success.
// ============================================================================
bool CheckBroadcastShapeBn3d(const std::vector<std::vector<int64_t>>& normal_input_shapes,
                             const std::vector<std::vector<int64_t>>& normal_output_shapes, int64_t rank)
{
    const int64_t c_dim = rank - 1;
    int64_t ref_c = -1;
    for (size_t i = 0; i < normal_input_shapes.size(); ++i) {
        const int64_t c = normal_input_shapes[i][static_cast<size_t>(c_dim)];
        if (c != 1) {
            if (ref_c == -1) {
                ref_c = c;
            } else if (c != ref_c) {
                return false;
            }
        }
    }
    // C==1 degenerate (single-channel, SZ-6): every input has channel count 1
    // and is skipped by the ref_c scan — treat ref_c as 1 so outputs validate.
    // (Fixes the 561002 tiling rejection that blocked all C==1 shapes.)
    if (ref_c == -1) {
        ref_c = 1;
    }
    for (size_t i = 0; i < normal_output_shapes.size(); ++i) {
        const int64_t c = normal_output_shapes[i][static_cast<size_t>(c_dim)];
        if (c != ref_c) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// §5.3 PrecomputeInputStrides
//   For a normalised (channel-last) shape, compute the GM stride for each axis.
//     - last axis (C): always stride 1
//     - any non-last axis with size == 1: stride 0 (broadcast axis)
//     - any non-last axis with size > 1: row-major stride = product of inner axes
// ============================================================================
void PrecomputeInputStrides(const std::vector<int64_t>& normal_shape, int64_t rank, std::vector<int64_t>& strides)
{
    strides.assign(static_cast<size_t>(rank), 0);
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; --k) {
        const int64_t sz = normal_shape[static_cast<size_t>(k)];
        if (k == rank - 1) {
            strides[static_cast<size_t>(k)] = 1;
        } else if (sz == 1) {
            strides[static_cast<size_t>(k)] = 0;
        } else {
            strides[static_cast<size_t>(k)] = inner;
        }
        inner *= sz;
    }
}

// ============================================================================
// §5.3 PrecomputeInputStridesByOrigin
//   For a tensor stored DENSELY in original storage order (e.g. NCHW), compute
//   the GM stride of each axis in the CHANNEL-LAST internal coordinate system.
//
//   Algorithm:
//     1) Compute original-axis dense strides: orig_stride[d] = product of
//        original_shape[d+1..rank-1] (row-major).
//     2) Build channel-last-axis → original-axis map:
//          cl_view axis 0..rank-2 → original axes [0..channel_axis-1,
//                                          channel_axis+1..rank-1] (skip channel).
//          cl_view axis rank-1     → original axis channel_axis.
//     3) strides[cl_view_axis] = orig_stride[mapped_orig_axis].
// ============================================================================
void PrecomputeInputStridesByOrigin(const std::vector<int64_t>& original_shape, int64_t channel_axis, int64_t rank,
                                    std::vector<int64_t>& strides)
{
    strides.assign(static_cast<size_t>(rank), 0);
    if (rank <= 0 || channel_axis < 0 || channel_axis >= rank)
        return;

    // 1) Original-axis dense strides (row-major).
    std::vector<int64_t> origStride(static_cast<size_t>(rank), 0);
    int64_t acc = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
        origStride[static_cast<size_t>(d)] = acc;
        acc *= original_shape[static_cast<size_t>(d)];
    }

    // 2) Channel-last axis → original axis map.
    //    cl axis 0..rank-2 correspond to original axes (skip channel_axis),
    //    cl axis rank-1 corresponds to original channel_axis.
    std::vector<int64_t> clToOrig(static_cast<size_t>(rank), 0);
    int64_t origIdx = 0;
    for (int64_t d = 0; d < rank - 1; ++d) {
        if (origIdx == channel_axis)
            ++origIdx; // skip channel
        clToOrig[static_cast<size_t>(d)] = origIdx;
        ++origIdx;
    }
    clToOrig[static_cast<size_t>(rank - 1)] = channel_axis;

    // 3) Channel-last view strides = mapped original strides.
    for (int64_t d = 0; d < rank; ++d) {
        strides[static_cast<size_t>(d)] = origStride[static_cast<size_t>(clToOrig[static_cast<size_t>(d)])];
    }
}

// ============================================================================
// §5.3 FindSplitAxisBn3d
//   Innermost-out scan over max_bro_shape to pick the UB single-split axis.
//   per_buf_bytes = (ub_per_core / phys_nodes) & ~31
//   per_buf_elems = per_buf_bytes / dtype_size
// ============================================================================
void FindSplitAxisBn3d(const std::vector<int64_t>& max_bro_shape, int64_t dtype_size, int64_t ub_per_core,
                       int64_t phys_nodes, Bn3dSplitResult& out)
{
    const int64_t per_buf_bytes = (ub_per_core / phys_nodes) & ~31;
    const int64_t per_buf_elems = per_buf_bytes / dtype_size;
    const int64_t rank = static_cast<int64_t>(max_bro_shape.size());

    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; --k) {
        const int64_t sz = max_bro_shape[static_cast<size_t>(k)];
        if (sz * inner > per_buf_elems) {
            out.axis = k;
            out.a_i = per_buf_elems / inner;
            out.a_o = (sz + out.a_i - 1) / out.a_i;
            const int64_t rem = sz % out.a_i;
            out.a_i_tail = (rem == 0) ? out.a_i : rem;
            return;
        }
        if (k == 0) {
            out.axis = 0;
            out.a_i = max_bro_shape[0];
            out.a_o = 1;
            out.a_i_tail = max_bro_shape[0];
            return;
        }
        inner *= sz;
    }
}

// ============================================================================
// §5.3 MultiCoreSplitBn3d
//   outer_prod = ∏_{j<ub_split.axis} max_bro_shape[j]
//   total_tiles = outer_prod * ub_split.a_o
//   num_cores   = min(total_tiles, max_cores); if total_tiles==0 → 0
//   tiles_main  = total_tiles / num_cores   (when num_cores > 0)
//   cores_tail  = total_tiles % num_cores   (when num_cores > 0)
// ============================================================================
void MultiCoreSplitBn3d(const std::vector<int64_t>& max_bro_shape, const Bn3dSplitResult& ub_split, int64_t max_cores,
                        Bn3dMultiCoreResult& out)
{
    const int64_t k = ub_split.axis;
    int64_t outer_prod = 1;
    for (int64_t j = 0; j < k; ++j) {
        outer_prod *= max_bro_shape[static_cast<size_t>(j)];
    }
    out.total_tiles = outer_prod * ub_split.a_o;

    out.num_cores = (out.total_tiles < max_cores) ? out.total_tiles : max_cores;
    if (out.num_cores > 0) {
        out.tiles_main = out.total_tiles / out.num_cores;
        out.cores_tail = out.total_tiles % out.num_cores;
    } else {
        out.tiles_main = 0;
        out.cores_tail = 0;
    }
}

} // namespace optiling
