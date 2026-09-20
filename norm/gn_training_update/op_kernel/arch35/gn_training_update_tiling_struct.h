/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// gn_training_update_package/op_kernel/arch35/gn_training_update_tiling_struct.h
// =============================================================================
//
// ROLE: Tiling data structures shared between host-side tiling and device-side
//   kernel for the GnTrainingUpdate operator on arch35 (Ascend 950).
//   Field-for-field transcription of design/TilingData.md §1 (the two branches
//   RANK_4 / RANK_8 share the same templated struct, differing only in the
//   RANK template parameter; no branch-private fields).
//
// CONTENTS (design/TilingData.md §1):
//   - MAX_INPUT_SLOTS  = 7  (x/sum/square_sum/scale/offset/mean/variance;
//                            mean/variance are official-IR reserved slots,
//                            placeholder only, never consumed by CopyIn)
//   - MAX_OUTPUT_SLOTS = 3  (y/batch_mean/batch_variance)
//   - PHYS_NODES       = 3  (physical live-buffer count P = TBuf slot count;
//                            derivation: HostTiling.md §3.3.2 P trace)
//   - SplitResult          (UB split axis parameters, from FindSplitAxis)
//   - MultiCoreResult      (multi-core distribution, from MultiCoreSplit)
//   - GnTrainingUpdateTilingData<RANK> (main tiling data container)
//
// UNITS (design/TilingData.md §2/§3): every shape/stride/split field is in
//   ELEMENT counts (maxBroShape coordinate system); perBufBytes is in BYTES
//   (= (ubSize / PHYS_NODES) & ~31, 32B-aligned); multicore.numCores is a
//   core count. dtype does NOT enter TilingData (static dispatch via
//   tilingKey / DTYPE_X macro).
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : GnTrainingUpdate   — struct name prefix
//   snake_case   : gn_training_update  — filename
//   UPPER_SNAKE  : GN_TRAINING_UPDATE  — (header guard via #pragma once)
// =============================================================================

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Slot counts (design/TilingData.md §1)
// ---------------------------------------------------------------------------
constexpr int64_t MAX_INPUT_SLOTS = 7;  // x/sum/square_sum/scale/offset/mean/variance
constexpr int64_t MAX_OUTPUT_SLOTS = 3; // y/batch_mean/batch_variance
constexpr int64_t PHYS_NODES = 3;       // P = TBuf slot count (HostTiling.md §3.3.2)

// ---------------------------------------------------------------------------
// SplitResult — UB split parameters computed by FindSplitAxis
//   axis     — UB split axis (dimension index in the maxBroShape coordinate system)
//   aI       — inner tile size (elements, <= perBufElems)
//   aO       — tile count along the split axis
//   aITail   — last-tile size (elements; == aI when evenly divided)
// ---------------------------------------------------------------------------
struct SplitResult {
    int64_t axis;
    int64_t aI;
    int64_t aO;
    int64_t aITail;
};

// ---------------------------------------------------------------------------
// MultiCoreResult — multi-core distribution computed by MultiCoreSplit
//   numCores   — participating cores (<= max available)
//   totalTiles — total tile count (0 for N=0 empty batch: kernel zero-loop
//                short-circuit)
//   tilesMain  — base tiles per core
//   coresTail  — number of tail cores processing tilesMain+1 tiles
// ---------------------------------------------------------------------------
struct MultiCoreResult {
    int64_t numCores;
    int64_t totalTiles;
    int64_t tilesMain;
    int64_t coresTail;
};

// ---------------------------------------------------------------------------
// GnTrainingUpdateTilingData<RANK> — complete tiling data for the kernel
//
// Instantiated at RANK=4 (effective rank <= 4: every legal input of this op)
// and RANK=8 (5-8, defensive fallback). Identical layout; only the array
// dims scale with RANK.
//
// All shape/stride arrays are front-padded to RANK dims (padding dim:
// shape=1, stride=0); broadcast dims carry stride 0 (NDDMA along-the-way
// expansion); mean/variance rows (slots 5/6) are placeholder rows filled
// with 0 and never consumed by the kernel.
// ---------------------------------------------------------------------------
template <int64_t RANK>
struct GnTrainingUpdateTilingData {
    SplitResult split;                             // UB split result (source: FindSplitAxis)
    MultiCoreResult multicore;                     // multi-core split result (source: MultiCoreSplit)
    int64_t rank;                                  // actual effective rank (after group-view fold +
                                                   //   PadAndSqueeze + MergeAxes; NCHW<=3, NHWC<=4)
    int64_t perBufBytes;                           // per-buffer bytes = (ubSize/PHYS_NODES) & ~31
    int64_t maxBroShape[RANK];                     // broadcast upper-bound shape (unified coordinate
                                                   //   system: NCHW folds to [N,G,M], NHWC to
                                                   //   [N,H*W,G,C/G], M=(C/G)*H*W)
    int64_t numInputs;                             // actual input slot count (=7)
    int64_t numOutputs;                            // actual output tensor count (=3)
    int64_t hasAffine;                             // 1 = scale present (affine path, golden:
                                                   //   affine iff scale present);
                                                   // 0 = absent (pure normalization path)
    int64_t hasOffset;                             // 1 = scale AND offset present (offset consumed);
                                                   // 0 = offset absent -> identity 0 (golden: y=ŷ·scale);
                                                   //   ignored when hasAffine == 0 (scale absent ->
                                                   //   offset ignored per golden)
    int64_t inputShapes[MAX_INPUT_SLOTS][RANK];    // per-input folded+padded shape
    int64_t inputStrides[MAX_INPUT_SLOTS][RANK];   // per-input GM stride (broadcast
                                                   //   axis = 0; mean/variance rows not consumed)
    int64_t outputShapes[MAX_OUTPUT_SLOTS][RANK];  // per-output folded+padded shape
    int64_t outputStrides[MAX_OUTPUT_SLOTS][RANK]; // per-output GM stride
    float epsilon;                                 // attr epsilon (>0), sqrt(var+epsilon) guard
    float invM;                                    // 1.0f/M, M=(C/G)*H*W elements per group;
                                                   //   host precomputes the reciprocal, kernel uses
                                                   //   Mul instead of Div: mean=sum*invM,
                                                   //   var=square_sum*invM-mean^2
};
