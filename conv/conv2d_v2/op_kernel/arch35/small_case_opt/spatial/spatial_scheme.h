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
 * Spatial scheme concept: the M-mode / HW-mode iteration modules.
 *
 * A spatial scheme owns EVERY mode-specific decision of the conv2d small
 * kernels, so the L1-strategy processors (full_load / cin_split / fm_partload)
 * are mode-agnostic and compose with either scheme:
 *
 *   Pipeline (entry) -> Scheme (M-mode | HW-mode) x L1 strategy
 *                     -> Processor (orchestration) -> TileOps (leaf operations).
 *
 * Scheme interface (implemented by MScheme and HwScheme):
 *   Schedule(tiling)              -- block partition + group params + M/HW ranges
 *   GroupIters / GroupAt          -- group-axis iteration plan (HW: one trivial step)
 *   ResolveBlockWindow            -- fmap GM window covered by this core
 *   MaxChunkHoSpan                -- Ho rows spanned by the widest L0 chunk
 *   ChunkFirst / ChunkNext        -- spatial chunk iteration (fills SpatialChunk)
 *   PeekNextWindow / FirstChunkWindow -- prefetch peeks (next chunk / batch chunk 0)
 *   LoadFmapWindow                -- GM->L1 window copy (mode-specific Nd2Nz/Dn2Nz)
 *   OutBaseOff / BlockBasePosM    -- output GM base offset / block-level Load3D POSM
 */
#ifndef CONV2D_SMALL_KERNEL_SPATIAL_SCHEME_H
#define CONV2D_SMALL_KERNEL_SPATIAL_SCHEME_H

#include "../conv2d_small_kernel_common.h"

namespace small_conv {

// fmap load window of one block (M-mode: full-width; HW-mode: H/W window).
// orgWin is the L1-padded width: tiling.win for M-mode, the W-window width for
// HW-mode.
struct FmapRange {
    uint32_t curHi;
    uint32_t padTop;
    uint32_t padBottom;
    uint32_t hiStart;
    uint32_t curWi;
    uint32_t padLeft;
    uint32_t padRight;
    uint32_t wiStart;
    uint32_t orgWin;
};

// Group-axis iteration step: the N range and GM channel offsets of one
// group/fweight iteration. HW-mode always yields a single trivially-valid plan.
struct GroupPlan {
    bool valid;
    uint32_t actualCo;     // valid N of this iteration (Fixpipe nSize)
    uint32_t groupCoutOff; // output-channel offset in cout
    uint32_t groupChanOff; // input-channel offset in cin
};

// Fmap window of a chunk used by the prefetch peeks (padTop / pads are not
// needed: only the GM load window matters for a preload).
struct ChunkPeek {
    uint32_t curHi;
    uint32_t hiLoadOff;
    uint32_t curWi;
    uint32_t wiLoadOff;
};

// Per-chunk spatial context produced by the scheme's chunk iteration. Carries
// the fmap window, the L0 M-tile geometry, the Load3D start column for both L1
// window granularities (chunk window: cin-split / fm-partload; block window:
// full-load) and the fixpipe/output geometry.
struct SpatialChunk {
    // fmap GM window of the chunk
    uint32_t curHi;
    uint32_t padTop;
    uint32_t hiLoadOff;
    uint32_t curWi;   // M-mode: always tiling.win (full width)
    uint32_t padLeft; // HW-mode: chunk-window pads; M-mode: tiling pads
    uint32_t padRight;
    uint32_t wiLoadOff;

    // L0 M-tile geometry
    uint32_t curM;
    uint32_t curMAlign;

    // Load3D POSM: position of the chunk's first output column relative to the
    // chunk's own L1 window (cin-split / fm-partload stream chunk windows)
    uint32_t chunkWinPosM;
    // ... and relative to the whole block window (full-load keeps the block
    // window resident in L1 and iterates chunks over it)
    uint32_t blockWinPosM;
    // Load3D padList left/right (M-mode: tiling pads; HW-mode: chunk pads)
    uint32_t l3dPadLeft;
    uint32_t l3dPadRight;

    // fixpipe / output geometry
    uint32_t outOff;
    uint32_t fpMSize;
    uint32_t fpDnNum;
    uint32_t fpDstDnStride;
};

// Shared Hi window of an output-row span [hoStart, hoEnd]: real GM rows to
// load plus the top pad seen by Load3D. Mode-agnostic; the schemes only differ
// in how they map the chunk offset onto this span.
template <typename TilingData>
__aicore__ inline void CalcHiWindow(const TilingData& tiling, uint32_t hoStart, uint32_t hoEnd, uint32_t& curHi,
                                    uint32_t& padTop, uint32_t& hiLoadOff)
{
    uint32_t hiStart = hoStart * tiling.strideH;
    uint32_t hiEnd = hoEnd * tiling.strideH + tiling.dilationH * (tiling.kh - 1);
    if (hiStart < tiling.padTop) {
        padTop = tiling.padTop - hiStart;
        hiLoadOff = 0;
    } else {
        padTop = 0;
        hiLoadOff = hiStart - tiling.padTop;
    }
    uint32_t needHi = (hoEnd - hoStart) * tiling.strideH + tiling.dilationH * (tiling.kh - 1) - padTop + 1;
    uint32_t maxGmRows = static_cast<uint32_t>(tiling.hin) - hiLoadOff;
    curHi = (needHi > maxGmRows) ? maxGmRows : needHi;
}

// Shared Wi window of an output-column span [woStart, woStart + curWo - 1]
// (HW-mode only; M-mode always loads the full width).
template <typename TilingData>
__aicore__ inline void CalcWiWindow(const TilingData& tiling, uint32_t woStart, uint32_t curWo, uint32_t& curWi,
                                    uint32_t& padLeft, uint32_t& padRight, uint32_t& wiLoadOff)
{
    uint32_t woEnd = woStart + curWo - 1;
    uint32_t wiStart = woStart * tiling.strideW;
    uint32_t wiEnd = woEnd * tiling.strideW + tiling.dilationW * (tiling.kw - 1);
    curWi = wiEnd - wiStart + 1;
    padLeft = 0;
    padRight = 0;
    wiLoadOff = wiStart;
    if (wiStart < tiling.padLeft) {
        padLeft = tiling.padLeft - wiStart;
        curWi -= padLeft;
        wiLoadOff = 0;
    } else {
        wiLoadOff = wiStart - tiling.padLeft;
    }
    if (wiEnd >= static_cast<uint32_t>(tiling.win) + tiling.padLeft) {
        padRight = wiEnd - (static_cast<uint32_t>(tiling.win) + tiling.padLeft) + 1;
        curWi -= padRight;
    }
}

// Shared batch-range setup of the block partition.
template <typename BlockCtx, typename TilingData>
__aicore__ inline bool ScheduleBatchRange(BlockCtx& blockCtx, const TilingData& tiling, uint32_t blockBatchIdx)
{
    uint32_t configured = static_cast<uint32_t>(tiling.singleCoreBatch == 0 ? 1u : tiling.singleCoreBatch);
    blockCtx.batchStart = blockBatchIdx * configured;
    if (blockCtx.batchStart >= tiling.batch) {
        return false;
    }
    uint32_t remaining = static_cast<uint32_t>(tiling.batch) - blockCtx.batchStart;
    blockCtx.singleCoreBatch = (remaining < configured) ? remaining : configured;
    blockCtx.curBatchIdx = blockCtx.batchStart;
    return true;
}

// Compile-time scheme selection used by the pipeline entries.
template <bool Cond, typename TrueT, typename FalseT>
struct SchemeSelect {
    using Type = FalseT;
};

template <typename TrueT, typename FalseT>
struct SchemeSelect<true, TrueT, FalseT> {
    using Type = TrueT;
};

} // namespace small_conv

#endif // CONV2D_SMALL_KERNEL_SPATIAL_SCHEME_H
