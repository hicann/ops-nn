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
 * MScheme: the M-mode spatial iteration module.
 *
 * The block output range hout*wout is flattened into one M axis and partitioned
 * across cores (blockDim = batchDim * nDim * hoDim(M) * groupDim); each core
 * iterates its M range in hoL0-sized chunks that always span the full input
 * width. The group axis is iterated inside the core (singleGroupIter), so
 * groups>1 / groupOpt enlarge are supported. Composes with every L1 strategy.
 */
#ifndef CONV2D_SMALL_KERNEL_M_SCHEME_H
#define CONV2D_SMALL_KERNEL_M_SCHEME_H

#include "spatial_scheme.h"
#include "../conv2d_small_kernel_block_ctx.h"

namespace small_conv {

// Group-axis iteration params (mirrors ops-nn ComputeGroupIterParams).
template <typename BlockCtx, typename TilingData>
struct ComputeGroupIterParamsTool {
    __aicore__ inline void operator()(BlockCtx& blockCtx, const TilingData& tiling) const
    {
        uint32_t groupCin = tiling.cin / tiling.groups;
        uint32_t groupCout = tiling.cout / tiling.groups;
        uint32_t wholeDim = tiling.groupOpt > 0 ? tiling.groupOpt : tiling.groups > 1 ? tiling.groups : 1;

        uint32_t maxDimPerCore = CeilDiv(wholeDim, tiling.groupDim);
        uint32_t realDim = CeilDiv(wholeDim, maxDimPerCore);
        blockCtx.groupBlockStride = maxDimPerCore;
        blockCtx.isGroupDimTail = (blockCtx.groupIdx + 1 == realDim);

        if (blockCtx.groupIdx >= realDim) {
            blockCtx.singleGroupIter = 0;
        } else {
            blockCtx.singleGroupIter = blockCtx.isGroupDimTail ? wholeDim - (realDim - 1) * maxDimPerCore :
                                                                 maxDimPerCore;
        }

        blockCtx.enlargeTail = 0;
        blockCtx.perGroupCo = groupCout;
        if (tiling.groupOpt > 0) {
            blockCtx.groupCinStep = tiling.cinOpt;
            blockCtx.groupCoutStep = tiling.coutOpt;
            blockCtx.perGroupCo = tiling.coutOpt;
            blockCtx.enlargeTail = tiling.groups % tiling.enlarge;
            if (blockCtx.isGroupDimTail && blockCtx.enlargeTail != 0) {
                blockCtx.perGroupCo = blockCtx.enlargeTail * groupCout;
            }
        } else if (tiling.groups > 1) {
            blockCtx.groupCinStep = groupCin;
            blockCtx.groupCoutStep = groupCout;
        } else {
            blockCtx.groupCinStep = tiling.cin;
            blockCtx.groupCoutStep = tiling.cout;
        }
    }
};

// Actual valid N for one group/fweight iteration (mirrors CalcActualCoForGroupIter).
template <typename BlockCtx, typename TilingData>
struct CalcActualCoForGroupIterTool {
    __aicore__ inline uint32_t operator()(const BlockCtx& blockCtx, const TilingData& tiling, uint32_t groupIter) const
    {
        uint32_t curPerGroupCo = blockCtx.perGroupCo;
        if (tiling.groupOpt > 0 &&
            !(blockCtx.isGroupDimTail && blockCtx.enlargeTail != 0 && groupIter + 1 == blockCtx.singleGroupIter)) {
            curPerGroupCo = tiling.coutOpt;
        }
        uint32_t coutStart = blockCtx.nIdx * tiling.singleCoreCo;
        if (coutStart >= curPerGroupCo) {
            return INVALID_GROUP_ITER;
        }
        return (coutStart + tiling.singleCoreCo > curPerGroupCo) ? curPerGroupCo - coutStart : tiling.singleCoreCo;
    }
};

template <typename BlockCtx, typename TilingData>
struct MScheme {
    using FmapT = typename BlockCtx::FmapT;
    static constexpr bool isHw = false;

    // ---- block partition (batch / group / N / M), group params, M range
    __aicore__ inline BlockCtx Schedule(const TilingData& tiling) const
    {
        BlockCtx blockCtx;
        blockCtx.active = false;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t totalBlocks = tiling.batchDim * tiling.nDim * tiling.hoDim * tiling.groupDim;
        if (blockIdx >= totalBlocks)
            return blockCtx;
        uint32_t nxm = tiling.nDim * tiling.hoDim;
        uint32_t blockBatchIdx = blockIdx / (tiling.groupDim * nxm);
        uint32_t rem = blockIdx % (tiling.groupDim * nxm);
        uint32_t blockGroupIdx = rem / nxm;
        rem = rem % nxm;
        uint32_t blockNIdx = rem / tiling.hoDim;
        uint32_t blockMIdx = rem % tiling.hoDim;

        if (!ScheduleBatchRange(blockCtx, tiling, blockBatchIdx))
            return blockCtx;
        blockCtx.groupIdx = blockGroupIdx;
        blockCtx.nIdx = blockNIdx;

        ComputeGroupIterParamsTool<BlockCtx, TilingData> groupParams;
        groupParams(blockCtx, tiling);
        if (blockCtx.singleGroupIter == 0)
            return blockCtx;
        CalcActualCoForGroupIterTool<BlockCtx, TilingData> actualCo;
        blockCtx.actualCo = actualCo(blockCtx, tiling, 0);
        if (blockCtx.actualCo == INVALID_GROUP_ITER) {
            blockCtx.actualCo = 0;
        }
        blockCtx.nIdxStart = blockCtx.nIdx * tiling.singleCoreCo;

        uint32_t totalM = static_cast<uint32_t>(tiling.hout * tiling.wout);
        blockCtx.mIdxStart = blockMIdx * static_cast<uint32_t>(tiling.singleCoreHo);
        if (blockCtx.mIdxStart >= totalM) {
            return blockCtx;
        }
        blockCtx.actualM = static_cast<uint32_t>(tiling.singleCoreHo);
        if (blockCtx.mIdxStart + blockCtx.actualM > totalM)
            blockCtx.actualM = totalM - blockCtx.mIdxStart;
        blockCtx.hoL0 = tiling.hoL0;
        blockCtx.active = true;
        return blockCtx;
    }

    // ---- group-axis iteration
    __aicore__ inline uint32_t GroupIters(const BlockCtx& blockCtx) const { return blockCtx.singleGroupIter; }

    __aicore__ inline GroupPlan GroupAt(const BlockCtx& blockCtx, const TilingData& tiling, uint32_t groupIter) const
    {
        GroupPlan groupPlan;
        CalcActualCoForGroupIterTool<BlockCtx, TilingData> actualCo;
        uint32_t co = actualCo(blockCtx, tiling, groupIter);
        groupPlan.valid = (co != INVALID_GROUP_ITER);
        groupPlan.actualCo = groupPlan.valid ? co : 0;
        groupPlan.groupCoutOff = (blockCtx.groupIdx * blockCtx.groupBlockStride + groupIter) * blockCtx.groupCoutStep;
        groupPlan.groupChanOff = (blockCtx.groupIdx * blockCtx.groupBlockStride + groupIter) * blockCtx.groupCinStep;
        return groupPlan;
    }

    // ---- block window: full input width, H rows of the core's M range
    __aicore__ inline FmapRange ResolveBlockWindow(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        FmapRange window;
        uint32_t orgWo = static_cast<uint32_t>(tiling.wout);
        uint32_t hoStart = blockCtx.mIdxStart / orgWo;
        uint32_t hoEnd = (blockCtx.mIdxStart + blockCtx.actualM - 1) / orgWo;
        uint32_t hiStart = hoStart * tiling.strideH;
        uint32_t hiEnd = hoEnd * tiling.strideH + tiling.dilationH * (tiling.kh - 1);
        bool topPad = (hiStart < tiling.padTop);
        bool botPad = (hiEnd >= static_cast<uint32_t>(tiling.hin) + tiling.padTop);
        window.padTop = topPad ? (tiling.padTop - hiStart) : 0;
        window.padBottom = botPad ? (hiEnd - (static_cast<uint32_t>(tiling.hin) + tiling.padTop) + 1) : 0;
        window.hiStart = topPad ? 0 : (hiStart - tiling.padTop);
        window.curHi = hiEnd - hiStart + 1 - window.padTop - window.padBottom;

        window.orgWin = static_cast<uint32_t>(tiling.win);
        window.curWi = window.orgWin;
        window.padLeft = tiling.padLeft;
        window.padRight = tiling.padRight;
        window.wiStart = 0;
        return window;
    }

    // Ho rows spanned by the widest hoL0-sized M chunk (L1 height sizing).
    __aicore__ inline uint32_t MaxChunkHoSpan(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        return (blockCtx.hoL0 + tiling.wout - 2) / tiling.wout;
    }

    // ---- chunk iteration over the core's flattened M range
    struct ChunkIter {
        uint32_t mOff;
    };

    __aicore__ inline bool ChunkFirst(const BlockCtx& blockCtx, const TilingData& tiling, ChunkIter& it,
                                      SpatialChunk& chunk) const
    {
        it.mOff = 0;
        return FillChunk(blockCtx, tiling, 0, chunk);
    }

    __aicore__ inline bool ChunkNext(const BlockCtx& blockCtx, const TilingData& tiling, ChunkIter& it,
                                     SpatialChunk& chunk) const
    {
        it.mOff += blockCtx.hoL0;
        if (it.mOff >= blockCtx.actualM)
            return false;
        return FillChunk(blockCtx, tiling, it.mOff, chunk);
    }

    __aicore__ inline bool FillChunk(const BlockCtx& blockCtx, const TilingData& tiling, uint32_t mOff,
                                     SpatialChunk& chunk) const
    {
        uint32_t curM = blockCtx.hoL0;
        if (mOff + curM > blockCtx.actualM)
            curM = blockCtx.actualM - mOff;
        uint32_t hoStart = (blockCtx.mIdxStart + mOff) / static_cast<uint32_t>(tiling.wout);
        uint32_t hoEnd = (blockCtx.mIdxStart + mOff + curM - 1) / static_cast<uint32_t>(tiling.wout);
        CalcHiWindow(tiling, hoStart, hoEnd, chunk.curHi, chunk.padTop, chunk.hiLoadOff);

        // M-mode windows always cover the full input width; the tiling pads sit
        // at the FMatrix edges and are applied by Load3D's padList.
        chunk.curWi = static_cast<uint32_t>(tiling.win);
        chunk.padLeft = tiling.padLeft;
        chunk.padRight = tiling.padRight;
        chunk.wiLoadOff = 0;

        chunk.curM = curM;
        chunk.curMAlign = static_cast<uint32_t>(AlignB(curM, GM0));
        chunk.chunkWinPosM = (blockCtx.mIdxStart + mOff) % static_cast<uint32_t>(tiling.wout);
        chunk.blockWinPosM = mOff + (blockCtx.mIdxStart % static_cast<uint32_t>(tiling.wout));
        chunk.l3dPadLeft = tiling.padLeft;
        chunk.l3dPadRight = tiling.padRight;

        chunk.outOff = blockCtx.mIdxStart + mOff;
        chunk.fpMSize = curM;
        chunk.fpDnNum = 1;
        chunk.fpDstDnStride = static_cast<uint32_t>(tiling.hout * tiling.wout);
        return true;
    }

    // Window of the next chunk in this batch (prefetch peek, no advance).
    __aicore__ inline bool PeekNextWindow(const BlockCtx& blockCtx, const TilingData& tiling, const ChunkIter& it,
                                          ChunkPeek& window) const
    {
        uint32_t nextOff = it.mOff + blockCtx.hoL0;
        if (nextOff >= blockCtx.actualM)
            return false;
        uint32_t curM = blockCtx.hoL0;
        if (nextOff + curM > blockCtx.actualM)
            curM = blockCtx.actualM - nextOff;
        uint32_t hoStart = (blockCtx.mIdxStart + nextOff) / static_cast<uint32_t>(tiling.wout);
        uint32_t hoEnd = (blockCtx.mIdxStart + nextOff + curM - 1) / static_cast<uint32_t>(tiling.wout);
        uint32_t padTop;
        CalcHiWindow(tiling, hoStart, hoEnd, window.curHi, padTop, window.hiLoadOff);
        window.curWi = static_cast<uint32_t>(tiling.win);
        window.wiLoadOff = 0;
        return true;
    }

    // Window of the batch's first chunk (next-batch prefetch peek).
    __aicore__ inline void FirstChunkWindow(const BlockCtx& blockCtx, const TilingData& tiling, ChunkPeek& window) const
    {
        uint32_t curM = blockCtx.hoL0;
        if (curM > blockCtx.actualM)
            curM = blockCtx.actualM;
        uint32_t hoStart = blockCtx.mIdxStart / static_cast<uint32_t>(tiling.wout);
        uint32_t hoEnd = (blockCtx.mIdxStart + curM - 1) / static_cast<uint32_t>(tiling.wout);
        uint32_t padTop;
        CalcHiWindow(tiling, hoStart, hoEnd, window.curHi, padTop, window.hiLoadOff);
        window.curWi = static_cast<uint32_t>(tiling.win);
        window.wiLoadOff = 0;
    }

    // ---- GM -> L1 copy of one full-width window (single Nd/Dn per cin block)
    __aicore__ inline void LoadFmapWindow(const TilingData& tiling, const GlobalTensor<FmapT>& fmapGm,
                                          LocalTensor<FmapT>& al1, uint32_t curHi, uint32_t hiLoadOff, uint32_t curWi,
                                          uint32_t wiLoadOff, uint32_t cinOff, uint32_t curCin) const
    {
        (void)wiLoadOff; // M-mode is always full width: the offset is 0.
        if constexpr (BlockCtx::isNHWCin) {
            Nd2NzParams copyParams;
            copyParams.ndNum = 1;
            copyParams.nValue = curHi * curWi;
            copyParams.dValue = curCin;
            copyParams.srcDValue = static_cast<uint32_t>(tiling.cin);
            copyParams.srcNdMatrixStride = 0;
            copyParams.dstNzC0Stride = curHi * curWi;
            copyParams.dstNzNStride = 1;
            copyParams.dstNzMatrixStride = 0;
            uint64_t gmOff = static_cast<uint64_t>(hiLoadOff) * curWi * tiling.cin + cinOff;
            DataCopy(al1, fmapGm[gmOff], copyParams);
        } else {
            Dn2NzParams copyParams;
            copyParams.dnNum = 1;
            copyParams.nValue = curHi * curWi;
            copyParams.dValue = curCin;
            copyParams.srcDnMatrixStride = 0;
            copyParams.srcDValue = static_cast<uint32_t>(tiling.hin * tiling.win);
            copyParams.dstNzC0Stride = curHi * curWi;
            copyParams.dstNzNStride = 1;
            copyParams.dstNzMatrixStride = 0;
            uint64_t gmOff = static_cast<uint64_t>(cinOff) * tiling.hin * tiling.win +
                             static_cast<uint64_t>(hiLoadOff) * curWi;
            DataCopy(al1, fmapGm[gmOff], copyParams);
        }
    }

    // ---- output GM base offset (within the batch / channel slab)
    __aicore__ inline uint32_t OutBaseOff(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        (void)tiling;
        return blockCtx.mIdxStart;
    }

    // ---- block-level Load3D POSM (full-load keeps the block window in L1)
    __aicore__ inline uint32_t BlockBasePosM(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        return blockCtx.mIdxStart % static_cast<uint32_t>(tiling.wout);
    }
};

} // namespace small_conv

#endif // CONV2D_SMALL_KERNEL_M_SCHEME_H
