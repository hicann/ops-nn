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
 * \file bn3_d_training_update_tiling_branch1.h
 * \brief
 */
#ifndef BN3_D_TRAINING_UPDATE_TILING_BRANCH1_H
#define BN3_D_TRAINING_UPDATE_TILING_BRANCH1_H

#include <cstdint>
#include <vector>

#include "norm/bn3_d_training_update/op_kernel/arch35/bn3_d_training_update_tiling_struct.h" // BN3DTrainingUpdateTilingData<5>
#include "bn3_d_training_update_tiling_common.h" // §5.3 common formulas + constants

namespace optiling {

// ============================================================================
// Bn3dBranch1Inputs — POD bundle of branch-1 (RANK=5) host-side inputs that
//   fully determine the §1 / §2 outputs. Derived from gert::TilingContext in
//   TilingFuncBN3DTrainingUpdate (Task 20), then handed to ComputeBranch1Tiling.
//
//   Fields:
//     input_shapes   — 7 raw input shapes (x + 6 stats); each is rank-5 for x,
//                       rank-1 (C,) for the 6 statistics.
//     output_shapes  — 5 raw output shapes (y + 4 stats); y follows x,
//                       the 4 stats outputs are (C,).
//     channel_axis   — {NCDHW:1, NDHWC:4} (DESIGN-BRANCH-1 §0 table for rank5).
//     C              — channel count = sum.shape[0] (== x.shape[channel_axis]).
//     num            — reduce-domain element count = N·D·H·W = x.size / C.
//     dtype_id       — {f32:0, f16:1, bf16:2} (DESIGN §5.2 table).
//     elem_bytes     — sizeof(T): {f32:4, f16:2, bf16:2}; feeds FindSplitAxis.
//     max_cores      — platform AIV count (e.g. 56 on ascend950).
// ============================================================================
struct Bn3dBranch1Inputs {
    std::vector<std::vector<int64_t>> input_shapes;  // 7 inputs (raw)
    std::vector<std::vector<int64_t>> output_shapes; // 5 outputs (raw)
    int32_t channel_axis = -1;                       // {NCDHW:1, NDHWC:4}
    int32_t C = 0;                                   // = sum.shape[0]
    int64_t num = 0;                                 // = N·D·H·W
    int32_t dtype_id = -1;                           // {f32:0, f16:1, bf16:2}
    int64_t elem_bytes = 0;                          // sizeof(T)
    int64_t max_cores = 1;                           // platform AIV count
};

// ============================================================================
// §1 + §2 ComputeBranch1Tiling
//   Fills the RANK=5 BN3DTrainingUpdateTilingData POD from the host-side
//   inputs. The body walks the §2 formulas:
//     1. PadAndSqueezeBn3d → max_bro_shape[5] (channel-last) + normalised shapes
//     2. PrecomputeInputStrides for x and each (C,) stat (broadcast axis=0)
//     3. FindSplitAxisBn3d → split.{axis, a_i, a_o, a_i_tail}
//     4. MultiCoreSplitBn3d → multicore.{num_cores, total_tiles, tiles_main, cores_tail}
//     5. Scalar host-budget fields: channel_axis / C / num / dtype_id / rank=5 /
//        per_buf_bytes=(ub/P)&~31 = 52416 / num_rec=1/num / bessel_scaler /
//        factor / one_minus_factor / epsilon.
//
//   NOTE: BN3DTrainingUpdateTilingDataRank5 lives at GLOBAL scope (the kernel
//   tiling struct header has no `namespace optiling`); the function itself is
//   in `optiling::` so the TilingFunc glue and UT can both call it.
//
//   STAGE EXPECTATION (Task 25 — UT written BEFORE impl):
//     The body is a STUB that leaves `out` UNTOUCHED (caller memset-zeroes it
//     first). Honest 0-pass baseline: every case fails because the oracle's
//     non-zero expected fields do not match the all-zero actual.
//
//     Task 26 will replace this stub with the real §2 formulas.
// ============================================================================
void ComputeBranch1Tiling(const Bn3dBranch1Inputs& in, ::BN3DTrainingUpdateTilingDataRank5& out);

} // namespace optiling

#endif
