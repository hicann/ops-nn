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
 * BlockCtx: per-block partition result carrier (no loop variables), plus the
 * BlockScheduler that fills it. Mode-agnostic: the M-mode / HW-mode block
 * partition itself lives in the spatial schemes (spatial/m_scheme.h,
 * spatial/hw_scheme.h); the scheduler only appends the mode-agnostic derived
 * quantities (K geometry, per-core N1 count). Carries the FULL ops-nn feature
 * set: group axis, inner batch, HW-mode Wo split.
 *
 * Members are intentionally not zero-initialised: the scheme's Schedule assigns
 * every member that downstream code reads, which avoids a long chain of scalar
 * stores at construction. Transient values that only live inside the scheduler
 * are plain locals and are not carried in the struct.
 */
#ifndef CONV2D_SMALL_KERNEL_BLOCK_CTX_H
#define CONV2D_SMALL_KERNEL_BLOCK_CTX_H

#include "conv2d_small_kernel_common.h"

namespace small_conv {

// The tiling-key InnerBatch SEL reuses the legacy ConvInnerBatch definitions
// (CONV_INNER_BATCH_SINGLE / CONV_INNER_BATCH_KERNEL_1X1_MULTI /
// CONV_INNER_BATCH_MULTI): the folded modes are decided at the dispatch/pipeline
// level, so the block context itself carries no innerbatch template parameter.
template <typename Traits, bool IsNHWCin_, bool IsNHWCout_, ConvFormat WeightFmt_>
struct BlockCtx {
    using FmapT = typename Traits::FmapType;
    using WeightT = typename Traits::WeightType;
    using BiasT = typename Traits::BiasType;
    using OutputT = typename Traits::OutputType;
    using Output1T = typename Traits::Output1Type;
    using L0cT = typename Traits::L0cT;

    static constexpr uint32_t GK0 = Traits::GK0;
    static constexpr uint32_t GK0Fmap = Traits::GK0Fmap;
    static constexpr uint32_t L0C_ELEMS = Traits::L0C_ELEMS;
    static constexpr uint32_t AL0_BUF_ELEMS = Traits::AL0_BUF_ELEMS;
    static constexpr uint32_t BL0_BUF_ELEMS = Traits::BL0_BUF_ELEMS;
    static constexpr bool isNHWCin = IsNHWCin_;
    static constexpr bool isNHWCout = IsNHWCout_;
    static constexpr ConvFormat weightFmt = WeightFmt_;

    bool active;

    // Block partition indices (group / N index kept for weight/bias addressing)
    uint32_t groupIdx;
    uint32_t nIdx;

    // N direction
    // Mutable: the scheme seeds it from group iteration 0; the unified group
    // loop in the processors updates it per group iteration (mirrors the legacy
    // CalcActualCoForGroupIter side effect on actualCo_).
    mutable uint32_t actualCo; // valid N for this core (Fixpipe nSize)
    uint32_t nIdxStart;        // N start offset in cout
    uint32_t n1PerCore;        // CeilDiv(actualCo, GN0)

    // K direction
    uint32_t cinAligned;
    uint32_t kTotalFmap;
    uint32_t k1Total;

    // M direction (M-mode: flattened M; HW-mode: Ho rows)
    uint32_t mIdxStart;
    uint32_t actualM;
    uint32_t hoL0;

    // HW-mode columns
    uint32_t hoIdxStart;
    uint32_t woIdxStart;
    uint32_t actualHo;
    uint32_t actualWo;
    uint32_t singleCoreWo;
    uint32_t woL0;

    // fmap / batch
    uint32_t batchStart;
    uint32_t singleCoreBatch;
    mutable uint32_t curBatchIdx; // current batch being processed (updated per inner-batch loop)

    // Inner-batch fold group: curBatchIdx is the current group's first map;
    // curIb is the group's real map count — tiling.innerBatch, or the TAIL
    // count when singleCoreBatch % innerBatch != 0 (so curIb != tiling.innerBatch
    // for an uneven last group). Other fold params derive from the tiling data
    // at the use sites.
    mutable uint32_t curIb;

    // A16W8: fmap-side k0 aligned cin. The L1 fmap NZ matrix k extent aligns to
    // the FMAP k0 (C0/sizeof(FmapT)), not the weight k0. Equal for fp16*fp16 /
    // int8*int8, different when they differ.
    uint32_t cinAlignedFmap;

    // Group-axis iteration (M-mode; HW-mode keeps the trivial single step)
    uint32_t singleGroupIter;
    uint32_t groupBlockStride;
    uint32_t groupCinStep;
    uint32_t groupCoutStep;
    bool isGroupDimTail;
    uint32_t enlargeTail;
    uint32_t perGroupCo; // effective co per group/fweight (adjusted for enlarge tail)
};

// Mode-agnostic tail of the block partition: K geometry and per-core N1 count,
// derived from the scheme-assigned cinAligned / actualCo.
template <typename BlockCtx, typename TilingData, typename Scheme>
struct BlockScheduler {
    Scheme scheme_;

    __aicore__ inline BlockCtx operator()(const TilingData& tiling) const
    {
        BlockCtx blockCtx = scheme_.Schedule(tiling);
        if (!blockCtx.active) {
            return blockCtx;
        }
        blockCtx.cinAligned = static_cast<uint32_t>(AlignB(tiling.singleCoreCi, BlockCtx::GK0));
        blockCtx.kTotalFmap = static_cast<uint32_t>(AlignB(tiling.singleCoreCi, BlockCtx::GK0Fmap)) * tiling.kh *
                              tiling.kw;
        blockCtx.cinAlignedFmap = static_cast<uint32_t>(AlignB(tiling.singleCoreCi, BlockCtx::GK0Fmap));
        blockCtx.n1PerCore = static_cast<uint32_t>(CeilDiv(blockCtx.actualCo, GN0));
        blockCtx.k1Total = static_cast<uint32_t>(CeilDiv(blockCtx.cinAligned * tiling.kh * tiling.kw, BlockCtx::GK0));
        // curIb starts at the first group's size; the group loop updates it
        // (tail-group aware).
        blockCtx.curIb = (blockCtx.singleCoreBatch > 0 && tiling.innerBatch > 1) ?
                             static_cast<uint32_t>(tiling.innerBatch) :
                             1;
        return blockCtx;
    }
};

} // namespace small_conv

#endif // CONV2D_SMALL_KERNEL_BLOCK_CTX_H
