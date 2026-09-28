/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file gemm_syrk_basic.h
 * \brief GemmSyrk Blaze kernel: C = alpha * (A @ A^T) + beta * C, in-place on C.
 *
 * Single-fetch syrk assembly (Blaze BlockMmadSyrk + GemmUniversal syrk
 * specialization, MIX 1 AIC : 2 AIV):
 *   - TRANS == false: a is the row-major (..., m, k) ND tensor. Each A
 *     row-block is fetched from GM exactly once per (row-block pair,
 *     k-chunk) with a single nd2nz CopyGM2L1. The NZ arrangement of X(m, k)
 *     is byte-identical to the ZN arrangement of X^T(k, m), so one L1 image
 *     feeds both cube inputs: an NZ view sources L0A (CopyL12L0A) and a ZN
 *     view sources L0B (CopyL12L0B). On the diagonal (i == j) one fetch
 *     literally serves both L0 buffers.
 *   - TRANS == true: a is stored transposed as (..., k, m). The kernel binds
 *     a DNExt (m, k) GM view over that storage, so the same single fetch per
 *     block pair routes to the dn2nz copy and produces the identical dual-view
 *     L1 image (NZ(a^T block) == ZN(a block)); the L0A/L0B feeds, the single
 *     Mmad chain and the dual fixpipe are unchanged. C = alpha * (a^T @ a) + beta * C.
 *   - The kernel iterates the upper triangle only; each processed slot
 *     computes the tile pair {(i, j), (j, i)} from the two shared fetches,
 *     halving the total GM->L1 traffic versus a generic matmul composition.
 *   - The epilogue reads beta * C from C's GM address (cInGm) and writes
 *     alpha * acc + beta * C back to the output address (cGm). The aclnn
 *     layer binds cInGm and cGm to the same device buffer (in-place). Every
 *     staged GM->UB read completes before the UB->GM overwrite, and all
 *     tiles are disjoint, so the aliasing is safe.
 *   - The complete symmetric matrix (upper and lower triangle) is written.
 *
 * Tiling contract (enforced by the standalone GemmSyrkTiling): one symmetric
 * square block baseBlock = baseM = baseN = mL1 = nL1 (the (j, i) mirror tile
 * swaps the M/N extents of the pair, so one size satisfies the epilogue's
 * single-N-chunk rule and row clamps for both tiles) with baseBlock^2 * 4B <=
 * L0C_SIZE / 4 (UB hosts both fp32 accumulator images fixpiped from the
 * single L0C slot plus both AIVs' staging), no per-core tail splitting, and
 * L1 double buffering with kL1 sized so one stage's A+B images fit an L1 half.
 */

#pragma once

#include "blaze/epilogue/block/block_epilogue_fmm_with_scale_add.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_syrk.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_syrk.h"
#include "blaze/gemm/kernel/kernel_matmul_syrk.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "gemm_syrk_tiling_data.h"

namespace GemmSyrkAdvanced {

// Fixed L1 double buffering: the host tiling sizes kL1 for two stages whose
// A+B images (2 * baseBlock * kL1 * dtype) each fit an L1 half.
constexpr uint8_t SYRK_L1_STAGES = 2U;

template <typename ElementType, bool TRANS>
__aicore__ inline void GemmSyrkBlazeKernel(GM_ADDR aGm, GM_ADDR cInGm, GM_ADDR cGm, GM_ADDR workspaceGm,
                                           const GemmSyrkTilingData& tilingData)
{
    using AccType = float;
    // TRANS: the DNExt (m, k) view over the transposed (k, m) storage; the
    // GM->L1 fetch auto-routes to dn2nz and yields the same dual-view image.
    using LayoutA = AscendC::Std::conditional_t<TRANS, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulSyrk;
    // Compact upper-triangle scheduler: stride polling over triangle-only
    // slots keeps every core busy instead of locking onto column densities.
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerSyrkTriangular<ProblemShape>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, LayoutA, ElementType, LayoutA, AccType,
                                                    LayoutC, ElementType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFmmWithScaleAdd<DispatchPolicy, ElementType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelParams = typename MatmulKernel::Params;

    // x3/output reuse the UB space after the FP32 accumulator, so accumulator
    // UB ping-pong and L0C double buffering stay disabled: each slot's single
    // L0C accumulator is fixpiped out twice (nz2nd + nz2dn) to the two AIVs.
    // The syrk tiling contract collapses the block geometry into ONE size
    // (baseM == baseN == mL1 == nL1 == baseBlock), consumed as a single
    // `block` param by the block / scheduler layers.
    const uint32_t block = tilingData.baseBlock;

    KernelParams params = {{tilingData.m, tilingData.n, tilingData.k, tilingData.batch},
                           // The block layer reads B from the single aGmAddr source (the ZN L1
                           // view of the A fetch). The two AIV sub-blocks split work by
                           // accumulator layout -- sub 0 runs the (i, j) epilogue on the ND
                           // fixpipe image, sub 1 the (j, i) epilogue on the DN-transposed
                           // (nz2dn) image, reading the genuine beta * C[j, i] rows.
                           {aGm, workspaceGm, tilingData.k, block, tilingData.kL1, tilingData.baseK, SYRK_L1_STAGES},
                           // x3 == output == C: in-place read of beta * C and overwrite with the result.
                           {cInGm, cGm, tilingData.alpha, tilingData.beta},
                           // Scheduler params: the single symmetric block; no tail splitting (a
                           // single tail block per axis carries the remainder).
                           {block}};

    MatmulKernel kernel;
    kernel(params);
}

} // namespace GemmSyrkAdvanced
