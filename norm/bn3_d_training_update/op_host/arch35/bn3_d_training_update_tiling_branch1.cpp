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
 * \file bn3_d_training_update_tiling_branch1.cpp
 * \brief
 */
#include "bn3_d_training_update_tiling_branch1.h"

#include <cstring>

namespace optiling {

namespace {

// ---- Array-width guard (kRank = 5) ----
//   max_bro_shape / input_shapes[i] / output_shapes[i] / input_strides[i] /
//   output_strides[i] are all RANK-wide; this impl is only invoked for RANK=5.
constexpr int64_t kBranchRank = 5;
constexpr int64_t kNumInputs = kMaxInputSlots;   // 7
constexpr int64_t kNumOutputs = kMaxOutputSlots; // 5

// ---------------------------------------------------------------------------
// CopyVectorToCArray<5>(src, dst) — lift a rank-5 vector into a fixed C array.
//   Leftover slots (if src has fewer than 5 elements) stay at the caller's
//   pre-set value (caller memset-zeroed `out`).
// ---------------------------------------------------------------------------
void CopyVectorToRankArray5(const std::vector<int64_t>& src, int64_t (&dst)[5])
{
    const int64_t n = static_cast<int64_t>(src.size());
    for (int64_t j = 0; j < n && j < kBranchRank; ++j) {
        dst[j] = src[static_cast<size_t>(j)];
    }
}

} // anonymous namespace

// ============================================================================
// §1 + §2 ComputeBranch1Tiling — real §2 formulas (Task 26)
//   Walks the §1/§2 step list. Uses the SAME-source common formulas as the
//   Task 20 host glue so that UT oracle (hand-coded) and impl (common funcs)
//   cross-validate the same math.
//   NOTE: BN3DTrainingUpdateTilingDataRank5 lives at GLOBAL scope.
// ============================================================================
void ComputeBranch1Tiling(const Bn3dBranch1Inputs& in, ::BN3DTrainingUpdateTilingDataRank5& out)
{
    // -----------------------------------------------------------------------
    // 1. PadAndSqueezeBn3d (§5.3) → channel-last view + normalised shapes.
    //    input_shapes[0] (x, 5D) is rearranged into [N,D,H,W,C]; the 6 (C,)
    //    statistics inputs become [1,1,1,1,C]; outputs follow the same pattern.
    // -----------------------------------------------------------------------
    std::vector<int64_t> max_bro_shape;
    std::vector<std::vector<int64_t>> normal_input_shapes;
    std::vector<std::vector<int64_t>> normal_output_shapes;
    (void)PadAndSqueezeBn3d(in.input_shapes, in.output_shapes, static_cast<int64_t>(in.channel_axis), max_bro_shape,
                            normal_input_shapes, normal_output_shapes);

    // Defensive: PadAndSqueezeBn3d may not size the vectors if shape data is
    // malformed; caller has already memset-zeroed `out`, so the §1 scalar
    // fields below are still safe.
    if (max_bro_shape.size() != static_cast<size_t>(kBranchRank)) {
        return;
    }

    // -----------------------------------------------------------------------
    // 2. max_bro_shape[5] + input_shapes[7][5] / output_shapes[5][5].
    // -----------------------------------------------------------------------
    CopyVectorToRankArray5(max_bro_shape, out.max_bro_shape);

    for (int64_t i = 0; i < kNumInputs; ++i) {
        if (static_cast<size_t>(i) < normal_input_shapes.size() &&
            normal_input_shapes[static_cast<size_t>(i)].size() == static_cast<size_t>(kBranchRank)) {
            CopyVectorToRankArray5(normal_input_shapes[static_cast<size_t>(i)],
                                   out.input_shapes[static_cast<size_t>(i)]);
        }
    }
    for (int64_t i = 0; i < kNumOutputs; ++i) {
        if (static_cast<size_t>(i) < normal_output_shapes.size() &&
            normal_output_shapes[static_cast<size_t>(i)].size() == static_cast<size_t>(kBranchRank)) {
            CopyVectorToRankArray5(normal_output_shapes[static_cast<size_t>(i)],
                                   out.output_shapes[static_cast<size_t>(i)]);
        }
    }

    // -----------------------------------------------------------------------
    // 3. input_strides[7][5] / output_strides[5][5] (§5.3 — the 5-D NDDMA
    //    loopSrcStride/loopDstStride configuration, DESIGN-BRANCH-1 §1)
    //    - x (input 0) / y (output 0): stored DENSELY in the caller's ORIGINAL
    //      storage format (e.g. NCDHW dense [N,C,D,H,W]). Use
    //      PrecomputeInputStridesByOrigin to map the channel-last internal
    //      coordinate view onto the original-layout dense GM buffer. This is
    //      correct for BOTH NCDHW (channel_axis=1) and NDHWC (channel_axis=4).
    //    - 6 (C,) statistics inputs / 4 (C,) statistics outputs: original
    //      storage is already [C] (rank-1), normalised to [1,1,1,1,C]
    //      broadcast. Use PrecomputeInputStrides (broadcast axis stride=0,
    //      C stride=1).
    // -----------------------------------------------------------------------
    std::vector<int64_t> strides;
    // x (input 0): original dense layout strides in channel-last coords.
    if (in.input_shapes[0].size() == static_cast<size_t>(kBranchRank)) {
        PrecomputeInputStridesByOrigin(in.input_shapes[0], static_cast<int64_t>(in.channel_axis), kBranchRank, strides);
        CopyVectorToRankArray5(strides, out.input_strides[0]);
    }
    // 6 (C,) stat inputs: broadcast [1,1,1,1,C] strides.
    for (int64_t i = 1; i < kNumInputs; ++i) {
        if (static_cast<size_t>(i) < normal_input_shapes.size() &&
            normal_input_shapes[static_cast<size_t>(i)].size() == static_cast<size_t>(kBranchRank)) {
            PrecomputeInputStrides(normal_input_shapes[static_cast<size_t>(i)], kBranchRank, strides);
            CopyVectorToRankArray5(strides, out.input_strides[static_cast<size_t>(i)]);
        }
    }
    // y (output 0): original dense layout strides in channel-last coords.
    if (in.output_shapes[0].size() == static_cast<size_t>(kBranchRank)) {
        PrecomputeInputStridesByOrigin(in.output_shapes[0], static_cast<int64_t>(in.channel_axis), kBranchRank,
                                       strides);
        CopyVectorToRankArray5(strides, out.output_strides[0]);
    }
    // 4 (C,) stat outputs: broadcast [1,1,1,1,C] strides.
    for (int64_t i = 1; i < kNumOutputs; ++i) {
        if (static_cast<size_t>(i) < normal_output_shapes.size() &&
            normal_output_shapes[static_cast<size_t>(i)].size() == static_cast<size_t>(kBranchRank)) {
            PrecomputeInputStrides(normal_output_shapes[static_cast<size_t>(i)], kBranchRank, strides);
            CopyVectorToRankArray5(strides, out.output_strides[static_cast<size_t>(i)]);
        }
    }

    // -----------------------------------------------------------------------
    // 4. FindSplitAxisBn3d (§5.3 / DESIGN-BRANCH-1 §2) → out.split.{axis,
    //    a_i, a_o, a_i_tail}. Innermost-out scan k=4(C)→0(N) over the
    //    channel-last [N,D,H,W,C] view.
    //    NO scatter 32B-alignment adjustment: DESIGN-BRANCH-1 §2 handles
    //    non-aligned tails via DataCopyPad + runtime NDDMA loopSize fill
    //    (user shapes are NOT required to align to 32B).
    // -----------------------------------------------------------------------
    Bn3dSplitResult ub_split{};
    FindSplitAxisBn3d(max_bro_shape, in.elem_bytes, kBn3dUbPerCore, kBn3dPhysNodes, ub_split);
    out.split.axis = ub_split.axis;
    out.split.a_i = ub_split.a_i;
    out.split.a_o = ub_split.a_o;
    out.split.a_i_tail = ub_split.a_i_tail;

    // -----------------------------------------------------------------------
    // 5. MultiCoreSplitBn3d (§5.3 / §2) → out.multicore.{...}.
    // -----------------------------------------------------------------------
    Bn3dMultiCoreResult multi{};
    MultiCoreSplitBn3d(max_bro_shape, ub_split, in.max_cores, multi);
    out.multicore.num_cores = multi.num_cores;
    out.multicore.total_tiles = multi.total_tiles;
    out.multicore.tiles_main = multi.tiles_main;
    out.multicore.cores_tail = multi.cores_tail;

    // -----------------------------------------------------------------------
    // 6. Scalar §1 fields.
    //    rank = 5 (this is branch 1).
    //    per_buf_bytes = (ub_per_core / kPhysNodes) & ~31 = 52416 (DESIGN §2).
    //    channel_axis / C / num / dtype_id flow straight from the host inputs.
    // -----------------------------------------------------------------------
    out.rank = kBranchRank;
    out.per_buf_bytes = kBn3dPerBufBytes;
    out.channel_axis = in.channel_axis;
    out.C = in.C;
    out.num = in.num;
    out.dtype_id = in.dtype_id;
    out.num_inputs = kNumInputs;
    out.num_outputs = kNumOutputs;
}

} // namespace optiling
