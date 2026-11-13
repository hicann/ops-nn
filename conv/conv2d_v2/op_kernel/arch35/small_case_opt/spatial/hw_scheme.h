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
 * HwScheme: the HW-mode spatial iteration module.
 *
 * The block output range is partitioned across cores along H and W separately
 * (blockDim = batchDim * nDim * hoDim * woDim); each core iterates its Ho range
 * in hoL0-sized chunks and its Wo range in woL0-sized chunks, and every chunk
 * loads only its own H/W window from GM. The group axis is trivial (groups==1
 * is guaranteed by the tiling), so GroupIters is a single valid step. Composes
 * with every L1 strategy.
 */
#ifndef CONV2D_SMALL_KERNEL_HW_SCHEME_H
#define CONV2D_SMALL_KERNEL_HW_SCHEME_H

#include "spatial_scheme.h"
#include "../conv2d_small_kernel_block_ctx.h"

namespace small_conv {

template <typename BlockCtx, typename TilingData>
struct HwScheme {
    using FmapT = typename BlockCtx::FmapT;
    static constexpr bool isHw = true;

    // ---- block partition (batch / N / Ho / Wo), trivial group params, H/W ranges
    __aicore__ inline BlockCtx Schedule(const TilingData& tiling) const
    {
        BlockCtx blockCtx;
        blockCtx.active = false;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t totalBlocks = tiling.batchDim * tiling.nDim * tiling.hoDim * tiling.woDim;
        if (blockIdx >= totalBlocks)
            return blockCtx;
        uint32_t nxhxw = tiling.nDim * tiling.hoDim * tiling.woDim;
        uint32_t blockBatchIdx = blockIdx / nxhxw;
        uint32_t rem = blockIdx % nxhxw;
        uint32_t blockNIdx = rem / (tiling.hoDim * tiling.woDim);
        rem = rem % (tiling.hoDim * tiling.woDim);
        uint32_t blockHoIdx = rem / tiling.woDim;
        uint32_t blockWoIdx = rem % tiling.woDim;

        if (!ScheduleBatchRange(blockCtx, tiling, blockBatchIdx))
            return blockCtx;
        blockCtx.groupIdx = 0;
        blockCtx.nIdx = blockNIdx;

        // HW-mode: groups guaranteed == 1 by caller; no group-axis iteration.
        blockCtx.singleGroupIter = 1;
        blockCtx.groupBlockStride = 0;
        blockCtx.groupCinStep = tiling.cin;
        blockCtx.groupCoutStep = tiling.cout;
        blockCtx.isGroupDimTail = false;
        blockCtx.enlargeTail = 0;
        blockCtx.perGroupCo = tiling.cout;

        // N-tail: last N-partition may have fewer real channels than singleCoreCo.
        uint32_t coutStart = blockCtx.nIdx * tiling.singleCoreCo;
        if (coutStart >= tiling.cout) {
            blockCtx.actualCo = 0;
        } else if (coutStart + tiling.singleCoreCo > tiling.cout) {
            blockCtx.actualCo = tiling.cout - coutStart;
        } else {
            blockCtx.actualCo = tiling.singleCoreCo;
        }
        blockCtx.nIdxStart = coutStart;

        blockCtx.hoIdxStart = blockHoIdx * static_cast<uint32_t>(tiling.singleCoreHo);
        if (blockCtx.hoIdxStart >= tiling.hout) {
            return blockCtx;
        }
        blockCtx.actualHo = static_cast<uint32_t>(tiling.singleCoreHo);
        if (blockCtx.hoIdxStart + blockCtx.actualHo > tiling.hout)
            blockCtx.actualHo = tiling.hout - blockCtx.hoIdxStart;

        blockCtx.singleCoreWo = static_cast<uint32_t>(tiling.singleCoreWo);
        blockCtx.woIdxStart = blockWoIdx * blockCtx.singleCoreWo;
        if (blockCtx.woIdxStart >= tiling.wout) {
            return blockCtx;
        }
        blockCtx.actualWo = blockCtx.singleCoreWo;
        if (blockCtx.woIdxStart + blockCtx.actualWo > tiling.wout)
            blockCtx.actualWo = tiling.wout - blockCtx.woIdxStart;

        blockCtx.actualM = blockCtx.actualHo * blockCtx.actualWo;
        blockCtx.hoL0 = tiling.hoL0;
        blockCtx.woL0 = tiling.woL0;
        blockCtx.active = true;
        return blockCtx;
    }

    // ---- group-axis iteration: one trivially-valid step
    __aicore__ inline uint32_t GroupIters(const BlockCtx& blockCtx) const
    {
        (void)blockCtx;
        return 1;
    }

    __aicore__ inline GroupPlan GroupAt(const BlockCtx& blockCtx, const TilingData& tiling, uint32_t groupIter) const
    {
        (void)tiling;
        (void)groupIter;
        GroupPlan groupPlan;
        groupPlan.valid = true;
        groupPlan.actualCo = blockCtx.actualCo;
        groupPlan.groupCoutOff = 0;
        groupPlan.groupChanOff = 0;
        return groupPlan;
    }

    // ---- block window: this core's Ho rows x Wo columns
    __aicore__ inline FmapRange ResolveBlockWindow(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        FmapRange window;
        uint32_t hiStart = blockCtx.hoIdxStart * tiling.strideH;
        uint32_t hiEnd = hiStart + (blockCtx.actualHo - 1) * tiling.strideH + tiling.dilationH * (tiling.kh - 1);
        bool topPad = (hiStart < tiling.padTop);
        bool botPad = (hiEnd >= static_cast<uint32_t>(tiling.hin) + tiling.padTop);
        window.padTop = topPad ? (tiling.padTop - hiStart) : 0;
        window.padBottom = botPad ? (hiEnd - (static_cast<uint32_t>(tiling.hin) + tiling.padTop) + 1) : 0;
        window.hiStart = topPad ? 0 : (hiStart - tiling.padTop);
        window.curHi = hiEnd - hiStart + 1 - window.padTop - window.padBottom;

        uint32_t wiStart = blockCtx.woIdxStart * tiling.strideW;
        uint32_t wiEnd = wiStart + (blockCtx.actualWo - 1) * tiling.strideW + tiling.dilationW * (tiling.kw - 1);
        bool leftPad = (wiStart < tiling.padLeft);
        bool rightPad = (wiEnd >= static_cast<uint32_t>(tiling.win) + tiling.padLeft);
        window.padLeft = leftPad ? (tiling.padLeft - wiStart) : 0;
        window.padRight = rightPad ? (wiEnd - (static_cast<uint32_t>(tiling.win) + tiling.padLeft) + 1) : 0;
        window.wiStart = leftPad ? 0 : (wiStart - tiling.padLeft);
        window.curWi = wiEnd - wiStart + 1 - window.padLeft - window.padRight;
        window.orgWin = window.curWi;
        return window;
    }

    // Ho rows spanned by the widest hoL0-row chunk (L1 height sizing).
    __aicore__ inline uint32_t MaxChunkHoSpan(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        (void)tiling;
        return blockCtx.hoL0 - 1; // hoL0 is a count of Ho rows in HW-mode.
    }

    // ---- chunk iteration over the core's Ho x Wo range (W advances first)
    struct ChunkIter {
        uint32_t hoOff;
        uint32_t woOff;
    };

    __aicore__ inline bool ChunkFirst(const BlockCtx& blockCtx, const TilingData& tiling, ChunkIter& it,
                                      SpatialChunk& chunk) const
    {
        it.hoOff = 0;
        it.woOff = 0;
        return FillChunk(blockCtx, tiling, 0, 0, chunk);
    }

    __aicore__ inline bool ChunkNext(const BlockCtx& blockCtx, const TilingData& tiling, ChunkIter& it,
                                     SpatialChunk& chunk) const
    {
        it.woOff += blockCtx.woL0;
        if (it.woOff >= blockCtx.actualWo) {
            it.woOff = 0;
            it.hoOff += blockCtx.hoL0;
            if (it.hoOff >= blockCtx.actualHo)
                return false;
        }
        return FillChunk(blockCtx, tiling, it.hoOff, it.woOff, chunk);
    }

    __aicore__ inline bool FillChunk(const BlockCtx& blockCtx, const TilingData& tiling, uint32_t hoOff, uint32_t woOff,
                                     SpatialChunk& chunk) const
    {
        uint32_t curHo = blockCtx.hoL0;
        if (hoOff + curHo > blockCtx.actualHo)
            curHo = blockCtx.actualHo - hoOff;
        uint32_t curWo = blockCtx.woL0;
        if (woOff + curWo > blockCtx.actualWo)
            curWo = blockCtx.actualWo - woOff;
        chunk.curM = curHo * curWo;
        chunk.curMAlign = static_cast<uint32_t>(AlignB(chunk.curM, GM0));

        uint32_t hoStart = blockCtx.hoIdxStart + hoOff;
        uint32_t hoEnd = hoStart + curHo - 1;
        CalcHiWindow(tiling, hoStart, hoEnd, chunk.curHi, chunk.padTop, chunk.hiLoadOff);
        CalcWiWindow(tiling, blockCtx.woIdxStart + woOff, curWo, chunk.curWi, chunk.padLeft, chunk.padRight,
                     chunk.wiLoadOff);

        // HW-mode chunks load only their own window, so Load3D starts at FMatrix
        // column 0 relative to the chunk window; relative to the block window
        // (kept fully resident by full-load) the chunk starts at hoOff/woOff.
        chunk.chunkWinPosM = 0;
        chunk.blockWinPosM = hoOff * static_cast<uint32_t>(tiling.wout) + woOff;
        chunk.l3dPadLeft = chunk.padLeft;
        chunk.l3dPadRight = chunk.padRight;

        // Row-split fixpipe when this core covers only part of the output width.
        bool needRowSplit = (blockCtx.actualWo < static_cast<uint32_t>(tiling.wout));
        chunk.outOff = (blockCtx.hoIdxStart + hoOff) * static_cast<uint32_t>(tiling.wout) + blockCtx.woIdxStart + woOff;
        chunk.fpMSize = needRowSplit ? curWo : chunk.curM;
        chunk.fpDnNum = needRowSplit ? curHo : 1;
        chunk.fpDstDnStride = needRowSplit ? static_cast<uint32_t>(tiling.wout) :
                                             static_cast<uint32_t>(tiling.hout * tiling.wout);
        return true;
    }

    // Window of the next chunk in this batch (W advances first, wrapping into
    // H; prefetch peek, no advance).
    __aicore__ inline bool PeekNextWindow(const BlockCtx& blockCtx, const TilingData& tiling, const ChunkIter& it,
                                          ChunkPeek& window) const
    {
        uint32_t nWoOff = it.woOff + blockCtx.woL0;
        uint32_t nHoOff = it.hoOff;
        if (nWoOff >= blockCtx.actualWo) {
            nWoOff = 0;
            nHoOff = it.hoOff + blockCtx.hoL0;
        }
        if (nHoOff >= blockCtx.actualHo)
            return false;
        uint32_t nCurHo = blockCtx.hoL0;
        if (nHoOff + nCurHo > blockCtx.actualHo)
            nCurHo = blockCtx.actualHo - nHoOff;
        uint32_t nCurWo = blockCtx.woL0;
        if (nWoOff + nCurWo > blockCtx.actualWo)
            nCurWo = blockCtx.actualWo - nWoOff;
        uint32_t padTop;
        CalcHiWindow(tiling, blockCtx.hoIdxStart + nHoOff, blockCtx.hoIdxStart + nHoOff + nCurHo - 1, window.curHi,
                     padTop, window.hiLoadOff);
        uint32_t padLeft;
        uint32_t padRight;
        CalcWiWindow(tiling, blockCtx.woIdxStart + nWoOff, nCurWo, window.curWi, padLeft, padRight, window.wiLoadOff);
        return true;
    }

    // Window of the batch's first chunk (next-batch prefetch peek).
    __aicore__ inline void FirstChunkWindow(const BlockCtx& blockCtx, const TilingData& tiling, ChunkPeek& window) const
    {
        uint32_t nCurHo = blockCtx.hoL0;
        if (nCurHo > blockCtx.actualHo)
            nCurHo = blockCtx.actualHo;
        uint32_t nCurWo = blockCtx.woL0;
        if (nCurWo > blockCtx.actualWo)
            nCurWo = blockCtx.actualWo;
        uint32_t padTop;
        CalcHiWindow(tiling, blockCtx.hoIdxStart, blockCtx.hoIdxStart + nCurHo - 1, window.curHi, padTop,
                     window.hiLoadOff);
        uint32_t padLeft;
        uint32_t padRight;
        CalcWiWindow(tiling, blockCtx.woIdxStart, nCurWo, window.curWi, padLeft, padRight, window.wiLoadOff);
    }

    // ---- GM -> L1 copy of one H/W window (row-gathered when narrower than win)
    __aicore__ inline void LoadFmapWindow(const TilingData& tiling, const GlobalTensor<FmapT>& fmapGm,
                                          LocalTensor<FmapT>& al1, uint32_t curHi, uint32_t hiLoadOff, uint32_t curWi,
                                          uint32_t wiLoadOff, uint32_t cinOff, uint32_t curCin) const
    {
        if constexpr (BlockCtx::isNHWCin) {
            Nd2NzParams copyParams;
            if (curWi == static_cast<uint32_t>(tiling.win)) {
                // Full-width: single Nd
                copyParams.ndNum = 1;
                copyParams.nValue = curHi * curWi;
                copyParams.srcNdMatrixStride = 0;
                copyParams.dstNzMatrixStride = 0;
            } else {
                // W-window: one Nd per loaded H row
                copyParams.ndNum = curHi;
                copyParams.nValue = curWi;
                copyParams.srcNdMatrixStride = static_cast<uint32_t>(tiling.win * tiling.cin);
                // dst row stride follows the fmap c0 granularity, not the weight GK0
                // (mixed dtype: GK0=32 vs GK0Fmap=16 doubles the dst row stride).
                // See CASE-2026-010.
                copyParams.dstNzMatrixStride = curWi * BlockCtx::GK0Fmap;
            }
            copyParams.dValue = curCin;
            copyParams.srcDValue = static_cast<uint32_t>(tiling.cin);
            copyParams.dstNzC0Stride = curHi * curWi;
            copyParams.dstNzNStride = 1;
            uint64_t gmOff = static_cast<uint64_t>(hiLoadOff) * tiling.win * tiling.cin +
                             static_cast<uint64_t>(wiLoadOff) * tiling.cin + cinOff;
            DataCopy(al1, fmapGm[gmOff], copyParams);
        } else {
            Dn2NzParams copyParams;
            if (curWi == static_cast<uint32_t>(tiling.win)) {
                // Full-width rows: single contiguous Dn.
                copyParams.dnNum = 1;
                copyParams.nValue = curHi * curWi;
                copyParams.srcDnMatrixStride = 0;
                copyParams.dstNzMatrixStride = 0;
            } else {
                // W-window: one Dn per loaded H row, stride win in GM.
                copyParams.dnNum = curHi;
                copyParams.nValue = curWi;
                copyParams.srcDnMatrixStride = static_cast<uint32_t>(tiling.win);
                // dst row stride follows the fmap c0 granularity (16 fp16 slots
                // per position, what LoadData<A2,A1,FmapT>/Load3D consume), not the
                // weight GK0. With int8 weights GK0=32 != GK0Fmap=16, which doubled
                // the dst row stride and made L1 hold consecutive rows (effective
                // dilationH=1). See CASE-2026-010.
                copyParams.dstNzMatrixStride = curWi * BlockCtx::GK0Fmap;
            }
            copyParams.dValue = curCin;
            copyParams.srcDValue = static_cast<uint32_t>(tiling.hin * tiling.win);
            copyParams.dstNzC0Stride = curHi * curWi;
            copyParams.dstNzNStride = 1;
            uint64_t gmOff = static_cast<uint64_t>(cinOff) * tiling.hin * tiling.win +
                             static_cast<uint64_t>(hiLoadOff) * tiling.win + wiLoadOff;
            DataCopy(al1, fmapGm[gmOff], copyParams);
        }
    }

    // ---- output GM base offset (within the batch / channel slab)
    __aicore__ inline uint32_t OutBaseOff(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        return blockCtx.hoIdxStart * static_cast<uint32_t>(tiling.wout) + blockCtx.woIdxStart;
    }

    // ---- block-level Load3D POSM (full-load keeps the block window in L1)
    __aicore__ inline uint32_t BlockBasePosM(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        (void)tiling;
        return blockCtx.woIdxStart;
    }
};

} // namespace small_conv

#endif // CONV2D_SMALL_KERNEL_HW_SCHEME_H
