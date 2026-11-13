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
 * CinSplitProcessor: cin-direction L1 split. fmap streams through a persistent
 * two-buffer L1 ping-pong, weight is loaded per cin-block, and all cin blocks
 * accumulate into one L0C tile per chunk. Keeps the FULL ops-nn feature set
 * (groups, NHWC in/out, inner-batch, N-split, dual-output).
 *
 * Mode-agnostic: the M-mode / HW-mode spatial iteration (partition, group
 * loop, chunk loop, prefetch targets, fixpipe geometry) is delegated to the
 * Scheme module, so this processor composes with either scheme.
 */
#ifndef CIN_SPLIT_PROCESSOR_H
#define CIN_SPLIT_PROCESSOR_H

#include "../conv2d_small_kernel_common.h"
#include "../conv2d_small_kernel_block_ctx.h"
#include "../conv2d_small_kernel_tile_ops.h"

namespace small_conv {

template <typename BlockCtx, typename TilingData>
struct CinState {
    using FmapT = typename BlockCtx::FmapT;
    using WeightT = typename BlockCtx::WeightT;
    GlobalTensor<FmapT> fmapGm;
    GlobalTensor<WeightT> filterGm;
    L1Layout l1;
    uint32_t cinL1;
    uint32_t cinL1Blocks;
    uint32_t n1Total;
    uint32_t n1PerCore;
    uint32_t al1BufBytes;
    uint32_t orgWin;
    uint32_t l1Pingpong;
    uint32_t l0pp;
    // A1 double-buffering switch. Cin-split derives al1BufBytes from chunk
    // geometry and always keeps it on; fm-partload (tiling-sized buffers)
    // mirrors the tiling's pBufferFlag bit3 (AL1_DB) into it.
    bool al1Db;
};

// Prefetch target issued at the last cin block of a chunk:
//   1) next N tile (same window), 2) next spatial chunk, 3) next batch's chunk 0.
struct NextUnitInfo {
    bool hasNextN;
    bool hasNextUnit;
    bool nextIsBatch;
    uint32_t nextBatchIdx;
    uint32_t nextHi;
    uint32_t nextHiLoadOff;
    uint32_t nextWi;
    uint32_t nextWiLoadOff;
};

template <typename BlockCtx, typename TilingData, typename Scheme>
struct CinSplitProcessor {
    using FmapT = typename BlockCtx::FmapT;
    using WeightT = typename BlockCtx::WeightT;
    using BiasT = typename BlockCtx::BiasT;
    using OutputT = typename BlockCtx::OutputT;
    using Output1T = typename BlockCtx::Output1T;
    using L0cT = typename BlockCtx::L0cT;
    static constexpr uint32_t GK0 = BlockCtx::GK0;
    static constexpr uint32_t GK0Fmap = BlockCtx::GK0Fmap;
    using State = CinState<BlockCtx, TilingData>;

    Scheme scheme_;
    SetupGmTool<BlockCtx, TilingData> setupGm_;
    LoadBiasScaleL1Tool<BlockCtx, TilingData> loadBiasScale_;
    LoadBiasToBTTool<BlockCtx, TilingData> loadBiasToBt_;
    LoadAL0Tool<BlockCtx> loadAl0_;
    LoadBL0Tool<BlockCtx> loadBl0_;
    InitMmadParamsTool<BlockCtx, TilingData> initMmadParams_;
    CopyOutTool<BlockCtx, TilingData> copyOut_;

    // --------------------------------------------------------------- Init
    __aicore__ inline void InitState(const BlockCtx& blockCtx, const TilingData& tiling, State& state) const
    {
        // Transients stay local: they are not needed after the layout is built.
        uint32_t coutAligned = static_cast<uint32_t>(AlignB(tiling.cout, GN0));
        uint32_t k1Total = blockCtx.k1Total;
        state.n1Total = coutAligned / GN0;
        state.n1PerCore = blockCtx.n1PerCore;
        state.l1Pingpong = 0;
        state.l0pp = 0;
        // Self-splitting user of the chunk loop: the per-buffer size comes from
        // the chunk geometry, so the double buffer is always kept.
        state.al1Db = true;

        state.cinL1 = blockCtx.cinAligned;
        if (tiling.kernelHxkernelW == 1) {
            if (blockCtx.cinAligned >= CIN_L1_SPLIT_QUARTER * MIN_GK0_MULTIPLIER_PER_SPLIT_1X1 * GK0) {
                state.cinL1 = static_cast<uint32_t>(AlignB(blockCtx.cinAligned / CIN_L1_SPLIT_QUARTER, GK0));
            } else if (blockCtx.cinAligned >= CIN_L1_SPLIT_HALF * MIN_GK0_MULTIPLIER_PER_SPLIT_1X1 * GK0) {
                state.cinL1 = static_cast<uint32_t>(AlignB(blockCtx.cinAligned / CIN_L1_SPLIT_HALF, GK0));
            }
        } else {
            if (blockCtx.cinAligned >= CIN_L1_SPLIT_QUARTER * GK0) {
                state.cinL1 = static_cast<uint32_t>(AlignB(blockCtx.cinAligned / CIN_L1_SPLIT_QUARTER, GK0));
            } else if (blockCtx.cinAligned >= CIN_L1_SPLIT_HALF * GK0) {
                state.cinL1 = static_cast<uint32_t>(AlignB(blockCtx.cinAligned / CIN_L1_SPLIT_HALF, GK0));
            }
        }
        state.cinL1Blocks = static_cast<uint32_t>(CeilDiv(AlignB(tiling.singleCoreCi, GK0Fmap), state.cinL1));

        // Mode-specific L1 width (M-mode: full width; HW-mode: block W window)
        // and widest-chunk Ho span come from the scheme.
        FmapRange r = scheme_.ResolveBlockWindow(blockCtx, tiling);
        state.orgWin = r.orgWin;
        uint32_t maxHoRelEnd = scheme_.MaxChunkHoSpan(blockCtx, tiling);
        uint32_t maxHiL1 = maxHoRelEnd * tiling.strideH + tiling.dilationH * (tiling.kh - 1) + tiling.padTop + 1;
        uint32_t hinFull = static_cast<uint32_t>(tiling.hin) + tiling.padTop + tiling.padBottom;
        if (maxHiL1 > hinFull)
            maxHiL1 = hinFull;

        uint32_t al1ElemPerBuf = maxHiL1 * state.orgWin * state.cinL1;
        state.al1BufBytes = al1ElemPerBuf * sizeof(FmapT);

        ComputeL1LayoutTool<BlockCtx, TilingData> l1Layout;
        state.l1 = l1Layout(tiling, 2, al1ElemPerBuf, k1Total * state.n1PerCore * GN0 * GK0);
    }

    // Largest factor of (cinL1*kernelHxW/GK0) not exceeding kL0/GK0, in K0 units.
    __aicore__ inline uint32_t CalcKL0(const TilingData& tiling, const State& state) const
    {
        uint32_t kL1OverK0 = state.cinL1 * tiling.kernelHxkernelW / GK0;
        uint32_t maxFactor = kL1OverK0 / 2;
        uint32_t maxAllowedFactor = tiling.kL0 / GK0;
        uint32_t upperBound = (maxFactor < maxAllowedFactor) ? maxFactor : maxAllowedFactor;
        for (uint32_t i = upperBound; i > 0; --i) {
            if (kL1OverK0 % i == 0)
                return i * GK0;
        }
        return GK0;
    }

    __aicore__ inline void LoadFmapL1Chunk(const TilingData& tiling, State& state, uint32_t bufIdx, uint32_t curHi,
                                           uint32_t hiLoadOff, uint32_t curWi, uint32_t wiLoadOff, uint32_t cinOff,
                                           uint32_t curCin) const
    {
        uint32_t elemCount = curHi * curWi * curCin;
        LocalTensor<FmapT> al1(TPosition::A1, bufIdx * state.al1BufBytes, elemCount);
        scheme_.LoadFmapWindow(tiling, state.fmapGm, al1, curHi, hiLoadOff, curWi, wiLoadOff, cinOff, curCin);
    }

    __aicore__ inline void LoadWeightL1Block(const BlockCtx& blockCtx, const TilingData& tiling, State& state,
                                             uint32_t kOff, uint32_t curK) const
    {
        uint32_t k1Block = static_cast<uint32_t>(CeilDiv(curK, GK0));
        uint32_t k1Start = kOff / GK0;
        uint32_t perK1 = state.n1PerCore * GN0 * GK0;
        uint32_t elemCount = k1Block * perK1;
        uint32_t l1ByteOff = state.l1.bl1OffBytes + k1Start * perK1 * sizeof(WeightT);
        LocalTensor<WeightT> bl1(TPosition::B1, l1ByteOff, elemCount);
        if (tiling.nDim == 1) {
            uint32_t gmOff = k1Start * state.n1Total * GN0 * GK0;
            DataCopy(bl1, state.filterGm[gmOff], elemCount);
        } else {
            uint32_t n1Start = blockCtx.nIdx * tiling.singleCoreCo / GN0;
            uint32_t tileBytes = GN0 * GK0 * sizeof(WeightT);
            uint32_t srcGmOff = k1Start * state.n1Total * GN0 * GK0 + n1Start * GN0 * GK0;
            uint16_t blkLen = static_cast<uint16_t>((state.n1PerCore * tileBytes) / 32);
            uint16_t srcGap = static_cast<uint16_t>(((state.n1Total - state.n1PerCore) * tileBytes) / 32);
            DataCopyParams cp(static_cast<uint16_t>(k1Block), blkLen, srcGap, 0);
            DataCopy(bl1, state.filterGm[srcGmOff], cp);
        }
    }

    __aicore__ inline Load3DCtx SetupLoad3DForChunk(const BlockCtx& blockCtx, const TilingData& tiling,
                                                    const SpatialChunk& c, uint32_t curCin) const
    {
        (void)blockCtx;
        uint8_t padList[PAD_LIST_NUM] = {static_cast<uint8_t>(c.l3dPadLeft), static_cast<uint8_t>(c.l3dPadRight),
                                         static_cast<uint8_t>(c.padTop), static_cast<uint8_t>(PAD_BOTTOM_VALUE)};
        Load3DSetFMatrixCal(c.curHi, c.curWi, padList);
        Load3DSetPaddingCal(tiling.offsetx);

        Load3DCtx ctx;
        ctx.xmTmp = ((static_cast<uint64_t>(c.curMAlign) & MASK_16) << MSTEP_OFFSET) |
                    ((static_cast<uint64_t>(c.chunkWinPosM) & MASK_16) << POSM_OFFSET);
        ctx.xtBase = ((static_cast<uint64_t>(tiling.strideW) & MASK_6) << 0) |
                     ((static_cast<uint64_t>(tiling.kw) & MASK_8) << KERNELW_OFFSET) |
                     ((static_cast<uint64_t>(tiling.kh) & MASK_8) << KERNELH_OFFSET) |
                     ((static_cast<uint64_t>(tiling.strideH) & MASK_6) << STRIDEH_OFFSET) |
                     ((static_cast<uint64_t>(tiling.kw) & NINTH_BIT_MASK) << KERNELW_HIGHEST_BIT_OFFSET) |
                     ((static_cast<uint64_t>(tiling.kh) & NINTH_BIT_MASK) << KERNELH_HIGHEST_BIT_OFFSET) |
                     ((static_cast<uint64_t>(tiling.dilationW) & MASK_8) << DILATIONW_OFFSET) |
                     ((static_cast<uint64_t>(tiling.dilationH) & MASK_8) << DILATIONH_OFFSET) |
                     ((static_cast<uint64_t>(curCin) & MASK_16) << CIN_OFFSET);
        SetLoadDataRepeatWithStride(LoadDataRepeatParamWithStride(0, 1, 0, static_cast<uint16_t>(c.curMAlign / GM0)));
        return ctx;
    }

    __aicore__ inline void RunKL0Loop(const BlockCtx& blockCtx, const TilingData& tiling, State& state,
                                      LocalTensor<FmapT>& al1, LocalTensor<WeightT>& bl1, LocalTensor<L0cT>& cl0,
                                      MmadParams& mmadParams, uint32_t kOff, uint32_t curKL1Fmap, uint32_t kl1,
                                      uint32_t kL0, uint32_t kL0Iters, const Load3DCtx& ctx, event_t relEv,
                                      uint32_t nOff, uint32_t curN, bool& needLoadBias) const
    {
        const bool isNSplit = tiling.nL0 < tiling.nBL1;
        uint32_t kl0Start = kOff / kL0;
        uint32_t kl0End = static_cast<uint32_t>(CeilDiv(kOff + curKL1Fmap, kL0));
        if (kl0End > kL0Iters)
            kl0End = kL0Iters;
        for (uint32_t kl0 = kl0Start; kl0 < kl0End; kl0++) {
            uint32_t kOffInner = kl0 * kL0;
            uint32_t curKInner = kL0;
            uint32_t curKInnerKAL0 = kL0;
            if (kOffInner + curKInner > kOff + curKL1Fmap)
                curKInner = kOff + curKL1Fmap - kOffInner;
            if (kl0 == kL0Iters - 1) {
                curKInnerKAL0 = blockCtx.kTotalFmap % kL0;
                curKInnerKAL0 = (curKInnerKAL0 == 0) ? kL0 : curKInnerKAL0;
                curKInner = static_cast<uint32_t>(AlignB(curKInnerKAL0, GK0));
            }
            uint32_t lbuf = state.l0pp & 1;
            event_t lev = static_cast<event_t>(lbuf);
            LocalTensor<FmapT> al0(TPosition::A2, lbuf * AL0_BUF_BYTES, BlockCtx::AL0_BUF_ELEMS);
            LocalTensor<WeightT> bl0(TPosition::B2, lbuf * BL0_BUF_BYTES, BlockCtx::BL0_BUF_ELEMS);

            WaitFlag<HardEvent::M_MTE1>(lev);
            loadAl0_(al1, al0, kOffInner - kOff, curKInnerKAL0, ctx);
            loadBl0_(blockCtx, bl0, bl1, kOffInner, curKInner, nOff, curN);
            if (tiling.hasBias) {
                if (needLoadBias) {
                    WaitFlag<HardEvent::MTE2_MTE1>(EVT_BIAS_DONE);
                }
                if (kl1 == 0 && kl0 == kl0Start && (needLoadBias || isNSplit)) {
                    loadBiasToBt_(tiling, state.l1, nOff, curN);
                    if (needLoadBias && !isNSplit) {
                        SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
                    }
                }
            }
            needLoadBias = false;
            SetFlag<HardEvent::MTE1_M>(lev);
            WaitFlag<HardEvent::MTE1_M>(lev);
            if (kl0 == kl0End - 1) {
                SetFlag<HardEvent::MTE1_MTE2>(relEv);
            }

            bool isLast = (kl1 == state.cinL1Blocks - 1) && (kl0 == kl0End - 1);
            mmadParams.unitFlag = isLast ? UNIT_FLAG_ENABLE_WITH_FLIP : UNIT_FLAG_ENABLE_ONLY;
            mmadParams.k = curKInnerKAL0;
            Mmad(cl0, al0, bl0, mmadParams);
            mmadParams.cmatrixInitVal = false;
            mmadParams.cmatrixSource = false;
            SetFlag<HardEvent::M_MTE1>(lev);
            state.l0pp++;
        }
    }

    // One A1 chunk prefetch (next cin block or boundary block 0): waits for the
    // target buffer, streams the fmap chunk (optionally the matching weight L1
    // block), and notifies MTE1.
    __aicore__ inline void PrefetchAl1Chunk(const BlockCtx& blockCtx, const TilingData& tiling, State& state,
                                            uint32_t bufIdx, uint32_t cinOff, uint32_t curCinOri, uint32_t hi,
                                            uint32_t hiLoadOff, uint32_t wi, uint32_t wiLoadOff, bool withWeight,
                                            uint32_t kernelHxW) const
    {
        event_t ev = (bufIdx == 0) ? EVT_FMAP_BUF0 : EVT_FMAP_BUF1;
        WaitFlag<HardEvent::MTE1_MTE2>(ev);
        LoadFmapL1Chunk(tiling, state, bufIdx, hi, hiLoadOff, wi, wiLoadOff, cinOff, curCinOri);
        if (withWeight) {
            LoadWeightL1Block(blockCtx, tiling, state, cinOff * kernelHxW,
                              static_cast<uint32_t>(AlignB(curCinOri, GK0Fmap)) * kernelHxW);
        }
        SetFlag<HardEvent::MTE2_MTE1>(ev);
    }

    // Runs the cin-block ping-pong for one output chunk / one N tile.
    __aicore__ inline void RunChunkCinBlocks(const BlockCtx& blockCtx, const TilingData& tiling, State& state,
                                             LocalTensor<L0cT>& cl0, const SpatialChunk& chunk, uint32_t kL0,
                                             uint32_t kL0Iters, bool loadWeight, bool firstCinPrefetched,
                                             const NextUnitInfo& nextUnit, GM_ADDR x, uint32_t curGroupChanOff,
                                             uint32_t nOff, uint32_t curN, bool& needLoadBias) const
    {
        uint32_t kernelHxW = tiling.kh * tiling.kw;
        uint32_t curMAlign = chunk.curMAlign;
        MmadParams mmadParams;
        initMmadParams_(tiling, mmadParams, curMAlign, static_cast<uint32_t>(AlignB(curN, GN0)));
        LocalTensor<WeightT> bl1Full(TPosition::B1, state.l1.bl1OffBytes, state.l1.bl1ElemCount);

        if (!firstCinPrefetched) {
            uint32_t b0 = state.al1Db ? (state.l1Pingpong & 1) : 0;
            uint32_t cin0 = (state.cinL1 > tiling.singleCoreCi) ? tiling.singleCoreCi : state.cinL1;
            PrefetchAl1Chunk(blockCtx, tiling, state, b0, 0, cin0, chunk.curHi, chunk.hiLoadOff, chunk.curWi,
                             chunk.wiLoadOff, loadWeight, kernelHxW);
        }

        for (uint32_t kl1 = 0; kl1 < state.cinL1Blocks; kl1++) {
            uint32_t cinOff = kl1 * state.cinL1;
            uint32_t curCinOri = state.cinL1;
            if (cinOff + curCinOri > tiling.singleCoreCi)
                curCinOri = tiling.singleCoreCi - cinOff;
            uint32_t curCin = static_cast<uint32_t>(AlignB(curCinOri, GK0Fmap));
            uint32_t curBuf = state.al1Db ? (state.l1Pingpong & 1) : 0;
            event_t curEv = (curBuf == 0) ? EVT_FMAP_BUF0 : EVT_FMAP_BUF1;

            // Prefetch target for this iteration: the next cin block of the chunk,
            // else (last block) the next N tile's block 0 or the next unit's block 0.
            bool pfNextBlock = (kl1 + 1 < state.cinL1Blocks);
            bool pfBoundaryN = !pfNextBlock && nextUnit.hasNextN;
            bool pfBoundaryUnit = !pfNextBlock && !nextUnit.hasNextN && nextUnit.hasNextUnit;
            uint32_t pfCinOff = pfNextBlock ? (kl1 + 1) * state.cinL1 : 0;
            uint32_t pfCurCinOri = (state.cinL1 > tiling.singleCoreCi) ? tiling.singleCoreCi : state.cinL1;
            if (pfNextBlock) {
                pfCurCinOri = state.cinL1;
                if (pfCinOff + pfCurCinOri > tiling.singleCoreCi)
                    pfCurCinOri = tiling.singleCoreCi - pfCinOff;
            }
            uint32_t pfHi = chunk.curHi;
            uint32_t pfHiLoadOff = chunk.hiLoadOff;
            uint32_t pfWi = chunk.curWi;
            uint32_t pfWiLoadOff = chunk.wiLoadOff;
            if (pfBoundaryUnit) {
                pfHi = nextUnit.nextHi;
                pfHiLoadOff = nextUnit.nextHiLoadOff;
                pfWi = nextUnit.nextWi;
                pfWiLoadOff = nextUnit.nextWiLoadOff;
            }
            bool pfValid = pfNextBlock || pfBoundaryN || pfBoundaryUnit;
            bool pfBatchGm = pfBoundaryUnit && nextUnit.nextIsBatch;

            if (state.al1Db && pfValid) {
                // Double buffer: the next chunk streams into the alternate buffer
                // while MTE1/MMAD consume the current one (overlapped).
                uint32_t nBuf = (state.l1Pingpong + 1) & 1;
                if (pfBatchGm) {
                    setupFmapGmBatch(tiling, x, nextUnit.nextBatchIdx, curGroupChanOff, state.fmapGm);
                }
                PrefetchAl1Chunk(blockCtx, tiling, state, nBuf, pfCinOff, pfCurCinOri, pfHi, pfHiLoadOff, pfWi,
                                 pfWiLoadOff, pfNextBlock && loadWeight, kernelHxW);
                if (pfBatchGm) {
                    setupFmapGmBatch(tiling, x, blockCtx.curBatchIdx, curGroupChanOff, state.fmapGm);
                }
            }

            WaitFlag<HardEvent::MTE2_MTE1>(curEv);
            Load3DCtx ctx = SetupLoad3DForChunk(blockCtx, tiling, chunk, curCin);
            uint32_t al1ElemCount = chunk.curHi * chunk.curWi * curCin;
            LocalTensor<FmapT> al1(TPosition::A1, curBuf * state.al1BufBytes, al1ElemCount);
            RunKL0Loop(blockCtx, tiling, state, al1, bl1Full, cl0, mmadParams, cinOff * kernelHxW, curCin * kernelHxW,
                       kl1, kL0, kL0Iters, ctx, curEv, nOff, curN, needLoadBias);
            if (!state.al1Db && pfValid) {
                // Single A1 buffer (tiling pBufferFlag bit3 off): the preload must
                // wait until this L0 loop releases the buffer, so it is issued
                // after RunKL0Loop, serialized into buffer 0.
                if (pfBatchGm) {
                    setupFmapGmBatch(tiling, x, nextUnit.nextBatchIdx, curGroupChanOff, state.fmapGm);
                }
                PrefetchAl1Chunk(blockCtx, tiling, state, 0, pfCinOff, pfCurCinOri, pfHi, pfHiLoadOff, pfWi,
                                 pfWiLoadOff, pfNextBlock && loadWeight, kernelHxW);
                if (pfBatchGm) {
                    setupFmapGmBatch(tiling, x, blockCtx.curBatchIdx, curGroupChanOff, state.fmapGm);
                }
            }
            state.l1Pingpong++;
        }
    }

    // ---------------------------------------------------------------- entry
    __aicore__ inline void operator()(const BlockCtx& blockCtx, const TilingData& tiling, GM_ADDR x, GM_ADDR filter,
                                      GM_ADDR bias, GM_ADDR y, const ExtendParams* ep) const
    {
        if (!blockCtx.active || blockCtx.actualCo == 0 || blockCtx.singleCoreBatch == 0)
            return;
        State state;
        InitState(blockCtx, tiling, state);
        uint32_t kL0 = CalcKL0(tiling, state);
        uint32_t kL0Iters = static_cast<uint32_t>(CeilDiv(blockCtx.kTotalFmap, kL0));

        GlobalTensor<BiasT> biasGm;
        GlobalTensor<OutputT> outputGm;
        GlobalTensor<Output1T> output1Gm;
        GlobalTensor<uint64_t> scale0Gm, scale1Gm;
        GlobalTensor<float> reluWeight0Gm, reluWeight1Gm;

        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
        SetFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF0);
        SetFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF1);
        GM_ADDR y1 = (tiling.dualOutput != 0) ? ep->y1 : y;

        Process(blockCtx, tiling, state, biasGm, outputGm, output1Gm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm,
                kL0, kL0Iters, y, y1, x, filter, bias, ep);

        WaitFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF1);
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
    }

    // Unified group-axis / inner-batch / spatial-chunk / N-tile loop. The group
    // protocol is identical for both schemes (HW-mode degenerates to the single
    // trivially-valid plan), so the event handshake stays balanced for every
    // bias / N-split combination:
    //   entry arm -> per-group wait(hasBias) -> in-tile re-arm(!nSplit) or
    //   group-end re-arm(nSplit) -> exit drain.
    __aicore__ inline void Process(const BlockCtx& blockCtx, const TilingData& tiling, State& state,
                                   GlobalTensor<BiasT>& biasGm, GlobalTensor<OutputT>& outputGm,
                                   GlobalTensor<Output1T>& output1Gm, GlobalTensor<uint64_t>& scale0Gm,
                                   GlobalTensor<uint64_t>& scale1Gm, GlobalTensor<float>& reluWeight0Gm,
                                   GlobalTensor<float>& reluWeight1Gm, uint32_t kL0, uint32_t kL0Iters, GM_ADDR y,
                                   GM_ADDR y1, GM_ADDR x, GM_ADDR filter, GM_ADDR bias, const ExtendParams* ep) const
    {
        bool needLoadBias = true;
        SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
        for (uint32_t groupIter = 0; groupIter < scheme_.GroupIters(blockCtx); groupIter++) {
            GroupPlan groupPlan = scheme_.GroupAt(blockCtx, tiling, groupIter);
            if (!groupPlan.valid) {
                continue;
            }
            // Per-iteration actual co (legacy CalcActualCoForGroupIter side
            // effect): downstream N loop bounds / bias loads / fixpipe nSize.
            blockCtx.actualCo = groupPlan.actualCo;
            setupGm_(blockCtx, tiling, x, filter, bias, y, ep, groupIter, groupPlan.groupCoutOff,
                     groupPlan.groupChanOff, scheme_.OutBaseOff(blockCtx, tiling), state.fmapGm, state.filterGm, biasGm,
                     outputGm, output1Gm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm);
            // Per-group N stride for LoadWeightL1Block: each fweight only packs
            // groupCoutStep channels, not the whole cout (legacy n1Total_ update).
            state.n1Total = static_cast<uint32_t>(AlignB(blockCtx.groupCoutStep, GN0)) / GN0;

            // Bias-L1 handoff wait: only when a bias reload actually happens,
            // otherwise the event would unbalance and deadlock for !hasBias.
            if (tiling.hasBias) {
                WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
            }
            loadBiasScale_(blockCtx, tiling, biasGm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm, state.l1);
            // The Fixpipe reads vector scale/relu from L1 and does not sync on
            // MTE2 by itself: gate every later Fixpipe on the bias/scale copy.
            SetFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));

            for (uint32_t batchIdx = 0; batchIdx < blockCtx.singleCoreBatch; batchIdx++) {
                blockCtx.curBatchIdx = blockCtx.batchStart + batchIdx;
                setupFmapGmBatch(tiling, x, blockCtx.curBatchIdx, groupPlan.groupChanOff, state.fmapGm);
                ProcessOneBatch(blockCtx, tiling, state, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm, kL0,
                                kL0Iters, y, y1, x, groupPlan.groupCoutOff, groupPlan.groupChanOff, batchIdx, true,
                                needLoadBias);
            }
            if (tiling.hasBias && tiling.nL0 < tiling.nBL1) {
                SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
            }
            needLoadBias = true;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
    }

    __aicore__ inline void setupFmapGmBatch(const TilingData& tiling, GM_ADDR x, uint32_t curBatch,
                                            uint32_t groupChanOff, GlobalTensor<FmapT>& fmapGm) const
    {
        uint64_t groupFmapOff;
        if constexpr (BlockCtx::isNHWCin) {
            groupFmapOff = groupChanOff;
        } else {
            groupFmapOff = static_cast<uint64_t>(groupChanOff) * tiling.hin * tiling.win;
        }
        uint64_t batchFmapOff = static_cast<uint64_t>(curBatch) * tiling.cin * tiling.hin * tiling.win;
        fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ FmapT*>(x) + batchFmapOff + groupFmapOff,
                               tiling.cin * tiling.hin * tiling.win);
    }

    // One inner batch: scheme-driven spatial chunk loop x N-tile loop. Shared
    // by the cin-split processor and the fm-partload processor, which differ
    // only in weightWithFirstChunk (cin-split streams the weight with the first
    // chunk's cin blocks; fm-partload weight-loads once up front).
    __aicore__ inline void ProcessOneBatch(const BlockCtx& blockCtx, const TilingData& tiling, State& state,
                                           GlobalTensor<uint64_t>& scale0Gm, GlobalTensor<uint64_t>& scale1Gm,
                                           GlobalTensor<float>& reluWeight0Gm, GlobalTensor<float>& reluWeight1Gm,
                                           uint32_t kL0, uint32_t kL0Iters, GM_ADDR y, GM_ADDR y1, GM_ADDR x,
                                           uint32_t groupCoutOff, uint32_t groupChanOff, uint32_t batchIdx,
                                           bool weightWithFirstChunk, bool& needLoadBias) const
    {
        bool hasNextBatch = (batchIdx + 1 < blockCtx.singleCoreBatch);
        uint32_t nextBatchIdx = blockCtx.batchStart + batchIdx + 1;
        typename Scheme::ChunkIter chunkIter;
        SpatialChunk chunk;
        bool firstCinPrefetched = false;
        bool firstChunk = true;
        for (bool more = scheme_.ChunkFirst(blockCtx, tiling, chunkIter, chunk); more;
             more = scheme_.ChunkNext(blockCtx, tiling, chunkIter, chunk)) {
            ChunkPeek peek;
            bool hasNextChunk = scheme_.PeekNextWindow(blockCtx, tiling, chunkIter, peek);
            for (uint32_t nOff = 0; nOff < blockCtx.actualCo; nOff += tiling.nL0) {
                uint32_t curN = tiling.nL0;
                if (nOff + curN > blockCtx.actualCo)
                    curN = blockCtx.actualCo - nOff;
                LocalTensor<L0cT> cl0(TPosition::CO1, 0, BlockCtx::L0C_ELEMS);
                // Chunk-0 weight preload: only the first unit of the whole core
                // (cin-split streams the weight with the first chunk's cin blocks).
                bool loadWeight = weightWithFirstChunk && firstChunk && (batchIdx == 0) && (nOff == 0);
                bool chunkPrefetched = firstCinPrefetched || !firstChunk || (nOff > 0) || (batchIdx > 0);
                NextUnitInfo nextUnit;
                nextUnit.hasNextN = (nOff + curN < blockCtx.actualCo);
                nextUnit.hasNextUnit = hasNextChunk || hasNextBatch;
                nextUnit.nextIsBatch = !hasNextChunk && hasNextBatch;
                nextUnit.nextBatchIdx = nextBatchIdx;
                if (hasNextChunk) {
                    nextUnit.nextHi = peek.curHi;
                    nextUnit.nextHiLoadOff = peek.hiLoadOff;
                    nextUnit.nextWi = peek.curWi;
                    nextUnit.nextWiLoadOff = peek.wiLoadOff;
                } else if (hasNextBatch) {
                    ChunkPeek first;
                    scheme_.FirstChunkWindow(blockCtx, tiling, first);
                    nextUnit.nextHi = first.curHi;
                    nextUnit.nextHiLoadOff = first.hiLoadOff;
                    nextUnit.nextWi = first.curWi;
                    nextUnit.nextWiLoadOff = first.wiLoadOff;
                }
                RunChunkCinBlocks(blockCtx, tiling, state, cl0, chunk, kL0, kL0Iters, loadWeight, chunkPrefetched,
                                  nextUnit, x, groupChanOff, nOff, curN, needLoadBias);
                firstChunk = false;
                firstCinPrefetched = hasNextChunk;

                copyOut_.template DoCopyOut<OutputT, 0>(blockCtx, tiling, y, cl0, state.l1, chunk.outOff, chunk.fpMSize,
                                                        chunk.curMAlign, curN, chunk.fpDnNum, chunk.fpDstDnStride, nOff,
                                                        groupCoutOff, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm);
                if (tiling.dualOutput) {
                    copyOut_.template DoCopyOut<Output1T, 1>(blockCtx, tiling, y1, cl0, state.l1, chunk.outOff,
                                                             chunk.fpMSize, chunk.curMAlign, curN, chunk.fpDnNum,
                                                             chunk.fpDstDnStride, nOff, groupCoutOff, scale0Gm,
                                                             scale1Gm, reluWeight0Gm, reluWeight1Gm);
                }
            }
        }
    }
};

} // namespace small_conv

#endif // CIN_SPLIT_PROCESSOR_H
