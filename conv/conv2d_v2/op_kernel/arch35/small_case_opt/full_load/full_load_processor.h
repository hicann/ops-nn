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
 * FullLoadProcessor: fmap + weight fully resident in L1, scheme-driven chunk
 * loop with L0A/L0B ping-pong and a K-loop per chunk. Keeps the FULL ops-nn
 * feature set: group-axis loop (M-mode), inner-batch double buffering, N-split,
 * dual-output.
 *
 * Mode-agnostic: the M-mode / HW-mode spatial iteration (partition, group
 * loop, chunk loop, fixpipe geometry, block-window Load3D) is delegated to the
 * Scheme module, so this processor composes with either scheme.
 */
#ifndef FULL_LOAD_PROCESSOR_H
#define FULL_LOAD_PROCESSOR_H

#include "../conv2d_small_kernel_common.h"
#include "../conv2d_small_kernel_block_ctx.h"
#include "../conv2d_small_kernel_tile_ops.h"

namespace small_conv {

template <typename BlockCtx, typename TilingData, typename Scheme>
struct FullLoadProcessor {
    using FmapT = typename BlockCtx::FmapT;
    using WeightT = typename BlockCtx::WeightT;
    using BiasT = typename BlockCtx::BiasT;
    using OutputT = typename BlockCtx::OutputT;
    using Output1T = typename BlockCtx::Output1T;
    using L0cT = typename BlockCtx::L0cT;
    static constexpr uint32_t GK0 = BlockCtx::GK0;

    Scheme scheme_;
    ComputeL1LayoutTool<BlockCtx, TilingData> l1Layout_;
    SetupGmTool<BlockCtx, TilingData> setupGm_;
    LoadWeightL1Tool<BlockCtx, TilingData> loadWeight_;
    LoadBiasScaleL1Tool<BlockCtx, TilingData> loadBiasScale_;
    LoadBiasToBTTool<BlockCtx, TilingData> loadBiasToBt_;
    SetupLoad3DBaseTool<BlockCtx, TilingData> setupLoad3d_;
    LoadAL0Tool<BlockCtx> loadAl0_;
    LoadBL0Tool<BlockCtx> loadBl0_;
    InitMmadParamsTool<BlockCtx, TilingData> initMmadParams_;
    CopyOutTool<BlockCtx, TilingData> copyOut_;

    __aicore__ inline void operator()(const BlockCtx& blockCtx, const TilingData& tiling, GM_ADDR x, GM_ADDR filter,
                                      GM_ADDR bias, GM_ADDR y, const ExtendParams* ep) const
    {
        uint32_t kL0MaxIter = static_cast<uint32_t>(CeilDiv(blockCtx.kTotalFmap, tiling.kL0));

        FmapRange blockWindow = scheme_.ResolveBlockWindow(blockCtx, tiling);
        uint32_t al1ElemPerBuf = blockWindow.curHi * blockWindow.orgWin * blockCtx.cinAligned;
        uint32_t bl1ElemCount = blockCtx.k1Total * blockCtx.n1PerCore * GN0 * GK0;

        // enableBatchDoubleBuffer: only when the complete two-buffer layout
        // (fmap x2 + weight + bias/scale/relu tails) fits in L1.
        L1Layout l1Two = l1Layout_(tiling, 2, al1ElemPerBuf, bl1ElemCount);
        bool enableBatchDoubleBuffer = (blockCtx.singleCoreBatch > 1) && (l1Two.endBytes <= TOTAL_L1_SIZE);
        L1Layout l1Layout = enableBatchDoubleBuffer ? l1Two : l1Layout_(tiling, 1, al1ElemPerBuf, bl1ElemCount);

        LocalTensor<FmapT> al1(TPosition::A1, 0, l1Layout.al1ElemCount);
        LocalTensor<WeightT> bl1(TPosition::B1, l1Layout.bl1OffBytes, l1Layout.bl1ElemCount);

        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
        uint32_t l0pp = 0;

        Process(blockCtx, tiling, blockWindow, l1Layout, x, filter, bias, y, ep, enableBatchDoubleBuffer, al1, bl1,
                l0pp, kL0MaxIter);

        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
    }

private:
    // Unified group-axis / inner-batch / spatial-chunk / N-tile loop with the
    // full-load L1-residency protocol. EVT_GROUP_L1_DONE serializes the L1
    // buffer reuse across group iterations; the bias handshake degenerates to
    // a single balanced arm/wait for the HW scheme's trivial one-step plan.
    __aicore__ inline void Process(const BlockCtx& blockCtx, const TilingData& tiling, const FmapRange& blockWindow,
                                   const L1Layout& l1Layout, GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                   const ExtendParams* ep, bool enableBatchDoubleBuffer, LocalTensor<FmapT>& al1,
                                   LocalTensor<WeightT>& bl1, uint32_t& l0pp, uint32_t kL0MaxIter) const
    {
        GM_ADDR y1 = (tiling.dualOutput != 0) ? ep->y1 : y;
        bool needLoadBias = true;
        SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
        SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_L1_DONE);
        for (uint32_t groupIter = 0; groupIter < scheme_.GroupIters(blockCtx); groupIter++) {
            GroupPlan groupPlan = scheme_.GroupAt(blockCtx, tiling, groupIter);
            if (!groupPlan.valid) {
                continue;
            }
            // Per-iteration actual co (legacy CalcActualCoForGroupIter side
            // effect): all downstream consumers (N loop bounds, bias/scale
            // loads, fixpipe nSize) must see this iteration's value.
            blockCtx.actualCo = groupPlan.actualCo;

            WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_L1_DONE);
            GlobalTensor<FmapT> fmapGm;
            GlobalTensor<WeightT> filterGm;
            GlobalTensor<BiasT> biasGm;
            GlobalTensor<OutputT> outputGm;
            GlobalTensor<Output1T> output1Gm;
            GlobalTensor<uint64_t> scale0Gm, scale1Gm;
            GlobalTensor<float> reluWeight0Gm, reluWeight1Gm;
            setupGm_(blockCtx, tiling, x, filter, bias, y, ep, groupIter, groupPlan.groupCoutOff,
                     groupPlan.groupChanOff, scheme_.OutBaseOff(blockCtx, tiling), fmapGm, filterGm, biasGm, outputGm,
                     output1Gm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm);

            // Bias-L1 handoff wait: only when a bias reload actually happens.
            // The event is armed once at entry and re-armed by the BT load /
            // the iteration end; an unconditional wait would unbalance it for
            // !hasBias and deadlock the M pipe.
            if (tiling.hasBias) {
                WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
            }
            loadBiasScale_(blockCtx, tiling, biasGm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm, l1Layout);
            SetFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));

            // Block-level Load3D context: the whole block window stays in L1
            // and the chunk loop selects its window via POSM.
            Load3DCtx ctx = setupLoad3d_(blockCtx, tiling, blockWindow, scheme_.BlockBasePosM(blockCtx, tiling));
            if (blockCtx.singleCoreBatch <= 1) {
                scheme_.LoadFmapWindow(tiling, fmapGm, al1, blockWindow.curHi, blockWindow.hiStart, blockWindow.curWi,
                                       blockWindow.wiStart, 0, blockCtx.groupCinStep);
                loadWeight_(blockCtx, tiling, filterGm, bl1);
                SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);
                WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);
                ProcessChunks(blockCtx, tiling, l1Layout, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm, kL0MaxIter,
                              y, y1, groupPlan.groupCoutOff, needLoadBias, al1, bl1, ctx, l0pp);
            } else if (!enableBatchDoubleBuffer) {
                loadWeight_(blockCtx, tiling, filterGm, bl1);
                SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);
                WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);
                for (uint32_t batchIdx = 0; batchIdx < blockCtx.singleCoreBatch; batchIdx++) {
                    blockCtx.curBatchIdx = blockCtx.batchStart + batchIdx;
                    GlobalTensor<FmapT> fmapGmB;
                    setupFmapGmBatch(tiling, x, blockCtx.batchStart + batchIdx, groupPlan.groupChanOff, fmapGmB);
                    scheme_.LoadFmapWindow(tiling, fmapGmB, al1, blockWindow.curHi, blockWindow.hiStart,
                                           blockWindow.curWi, blockWindow.wiStart, 0, blockCtx.groupCinStep);
                    SetFlag<HardEvent::MTE2_MTE1>(EVT_BATCH_BUF0);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVT_BATCH_BUF0);
                    ProcessChunks(blockCtx, tiling, l1Layout, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm,
                                  kL0MaxIter, y, y1, groupPlan.groupCoutOff, needLoadBias, al1, bl1, ctx, l0pp);
                    if (batchIdx + 1 < blockCtx.singleCoreBatch) {
                        SetFlag<HardEvent::MTE1_MTE2>(EVT_BATCH_BUF0);
                        WaitFlag<HardEvent::MTE1_MTE2>(EVT_BATCH_BUF0);
                    }
                }
            } else {
                loadWeight_(blockCtx, tiling, filterGm, bl1);
                SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);
                WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);
                uint32_t al1ElemPerBuf = l1Layout.al1ElemCount / 2;
                {
                    GlobalTensor<FmapT> fmapGm0;
                    setupFmapGmBatch(tiling, x, blockCtx.batchStart, groupPlan.groupChanOff, fmapGm0);
                    LocalTensor<FmapT> al1Buf0(TPosition::A1, 0, al1ElemPerBuf);
                    scheme_.LoadFmapWindow(tiling, fmapGm0, al1Buf0, blockWindow.curHi, blockWindow.hiStart,
                                           blockWindow.curWi, blockWindow.wiStart, 0, blockCtx.groupCinStep);
                }
                SetFlag<HardEvent::MTE2_MTE1>(EVT_BATCH_BUF0);
                for (uint32_t batchIdx = 0; batchIdx < blockCtx.singleCoreBatch; batchIdx++) {
                    blockCtx.curBatchIdx = blockCtx.batchStart + batchIdx;
                    uint32_t curBuf = batchIdx & 1;
                    uint32_t nextBuf = 1 - curBuf;
                    if (batchIdx + 1 < blockCtx.singleCoreBatch) {
                        GlobalTensor<FmapT> fmapGmN;
                        setupFmapGmBatch(tiling, x, blockCtx.batchStart + batchIdx + 1, groupPlan.groupChanOff,
                                         fmapGmN);
                        if (batchIdx > 0) {
                            WaitFlag<HardEvent::MTE1_MTE2>(nextBuf ? EVT_BATCH_BUF1 : EVT_BATCH_BUF0);
                        }
                        LocalTensor<FmapT> al1Next(TPosition::A1, nextBuf * l1Layout.al1BufBytes, al1ElemPerBuf);
                        scheme_.LoadFmapWindow(tiling, fmapGmN, al1Next, blockWindow.curHi, blockWindow.hiStart,
                                               blockWindow.curWi, blockWindow.wiStart, 0, blockCtx.groupCinStep);
                        SetFlag<HardEvent::MTE2_MTE1>(nextBuf ? EVT_BATCH_BUF1 : EVT_BATCH_BUF0);
                    }
                    WaitFlag<HardEvent::MTE2_MTE1>(curBuf ? EVT_BATCH_BUF1 : EVT_BATCH_BUF0);
                    LocalTensor<FmapT> al1Cur(TPosition::A1, curBuf * l1Layout.al1BufBytes, al1ElemPerBuf);
                    ProcessChunks(blockCtx, tiling, l1Layout, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm,
                                  kL0MaxIter, y, y1, groupPlan.groupCoutOff, needLoadBias, al1Cur, bl1, ctx, l0pp);
                    if (batchIdx + 2 < blockCtx.singleCoreBatch) {
                        SetFlag<HardEvent::MTE1_MTE2>(curBuf ? EVT_BATCH_BUF1 : EVT_BATCH_BUF0);
                    }
                }
            }
            if (tiling.hasBias && tiling.nL0 < tiling.nBL1) {
                SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
            }
            needLoadBias = true;
            SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_L1_DONE);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_L1_DONE);
    }

    // K-loop: accumulates one output chunk over all K0 blocks.
    __aicore__ inline void MmadAccumulateTile(const BlockCtx& blockCtx, const TilingData& tiling,
                                              LocalTensor<FmapT>& al1, LocalTensor<WeightT>& bl1,
                                              LocalTensor<L0cT>& cl0, uint32_t curMAlign, uint32_t mmadN,
                                              uint32_t kL0MaxIter, const Load3DCtx& ctx, const L1Layout& l1Layout,
                                              bool& needLoadBias, uint32_t nOff, uint32_t curN, uint32_t& l0pp) const
    {
        MmadParams mmadParams;
        initMmadParams_(tiling, mmadParams, curMAlign, mmadN);
        const bool isNSplit = tiling.nL0 < tiling.nBL1;
        for (uint32_t kl0 = 0; kl0 < kL0MaxIter; kl0++) {
            uint32_t kOff = kl0 * tiling.kL0;
            uint32_t curK = tiling.kL0;
            uint32_t curKAL0 = tiling.kL0;
            if (kl0 == kL0MaxIter - 1) {
                curKAL0 = blockCtx.kTotalFmap % tiling.kL0;
                curKAL0 = (curKAL0 == 0) ? tiling.kL0 : curKAL0;
                curK = static_cast<uint32_t>(AlignB(curKAL0, GK0));
            }
            uint32_t buf = l0pp & 1;
            event_t ev = static_cast<event_t>(buf);
            LocalTensor<FmapT> al0(TPosition::A2, buf * AL0_BUF_BYTES, BlockCtx::AL0_BUF_ELEMS);
            LocalTensor<WeightT> bl0(TPosition::B2, buf * BL0_BUF_BYTES, BlockCtx::BL0_BUF_ELEMS);

            WaitFlag<HardEvent::M_MTE1>(ev);
            loadAl0_(al1, al0, kOff, curKAL0, ctx);
            loadBl0_(blockCtx, bl0, bl1, kOff, curK, nOff, mmadN);
            if (tiling.hasBias) {
                if (needLoadBias) {
                    WaitFlag<HardEvent::MTE2_MTE1>(EVT_BIAS_DONE);
                }
                if (kl0 == 0 && (needLoadBias || isNSplit)) {
                    loadBiasToBt_(tiling, l1Layout, nOff, curN);
                    if (needLoadBias && !isNSplit) {
                        SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
                    }
                }
            }
            needLoadBias = false;
            SetFlag<HardEvent::MTE1_M>(ev);
            WaitFlag<HardEvent::MTE1_M>(ev);

            mmadParams.k = curKAL0;
            mmadParams.unitFlag = (kl0 == kL0MaxIter - 1) ? UNIT_FLAG_ENABLE_WITH_FLIP : UNIT_FLAG_ENABLE_ONLY;
            Mmad(cl0, al0, bl0, mmadParams);
            mmadParams.cmatrixInitVal = false;
            mmadParams.cmatrixSource = false;
            SetFlag<HardEvent::M_MTE1>(ev);
            l0pp++;
        }
    }

    // Scheme-driven chunk loop over the resident block window x N-tile loop.
    __aicore__ inline void ProcessChunks(const BlockCtx& blockCtx, const TilingData& tiling, const L1Layout& l1Layout,
                                         GlobalTensor<uint64_t>& scale0Gm, GlobalTensor<uint64_t>& scale1Gm,
                                         GlobalTensor<float>& reluWeight0Gm, GlobalTensor<float>& reluWeight1Gm,
                                         uint32_t kL0MaxIter, GM_ADDR y, GM_ADDR y1, uint32_t groupCoutOff,
                                         bool& needLoadBias, LocalTensor<FmapT>& al1, LocalTensor<WeightT>& bl1,
                                         Load3DCtx& ctx, uint32_t& l0pp) const
    {
        typename Scheme::ChunkIter chunkIter;
        SpatialChunk chunk;
        for (bool more = scheme_.ChunkFirst(blockCtx, tiling, chunkIter, chunk); more;
             more = scheme_.ChunkNext(blockCtx, tiling, chunkIter, chunk)) {
            // The block window stays resident in L1; the chunk selects its
            // output range via the block-window-relative POSM.
            ctx.xmTmp = ((static_cast<uint64_t>(chunk.curMAlign) & MASK_16) << MSTEP_OFFSET) |
                        ((static_cast<uint64_t>(chunk.blockWinPosM) & MASK_16) << POSM_OFFSET);
            SetLoadDataRepeatWithStride(
                LoadDataRepeatParamWithStride(0, 1, 0, static_cast<uint16_t>(chunk.curMAlign / GM0)));

            if (needLoadBias) {
                SetFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
            }
            for (uint32_t nOff = 0; nOff < blockCtx.actualCo; nOff += tiling.nL0) {
                uint32_t curN = tiling.nL0;
                if (nOff + curN > blockCtx.actualCo)
                    curN = blockCtx.actualCo - nOff;
                uint32_t curMmadN = static_cast<uint32_t>(AlignB(curN, GN0));
                LocalTensor<L0cT> cl0(TPosition::CO1, 0, BlockCtx::L0C_ELEMS);
                MmadAccumulateTile(blockCtx, tiling, al1, bl1, cl0, chunk.curMAlign, curMmadN, kL0MaxIter, ctx,
                                   l1Layout, needLoadBias, nOff, curN, l0pp);
                copyOut_.template DoCopyOut<OutputT, 0>(blockCtx, tiling, y, cl0, l1Layout, chunk.outOff, chunk.fpMSize,
                                                        chunk.curMAlign, curN, chunk.fpDnNum, chunk.fpDstDnStride, nOff,
                                                        groupCoutOff, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm);
                if (tiling.dualOutput) {
                    copyOut_.template DoCopyOut<Output1T, 1>(blockCtx, tiling, y1, cl0, l1Layout, chunk.outOff,
                                                             chunk.fpMSize, chunk.curMAlign, curN, chunk.fpDnNum,
                                                             chunk.fpDstDnStride, nOff, groupCoutOff, scale0Gm,
                                                             scale1Gm, reluWeight0Gm, reluWeight1Gm);
                }
            }
        }
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
};

} // namespace small_conv

#endif // FULL_LOAD_PROCESSOR_H
