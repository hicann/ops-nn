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
 * FmPartloadProcessor: fmap chunked over Cin through the shared two-buffer L1
 * ping-pong, weight one-shot fullload into L1. Reuses CinSplitProcessor's
 * unified group/batch/chunk loop with weightWithFirstChunk=false (composition,
 * no duplicated loop code).
 *
 * Mode-agnostic: the M-mode / HW-mode spatial iteration is delegated to the
 * Scheme module, so this processor composes with either scheme.
 */
#ifndef FM_PARTLOAD_PROCESSOR_H
#define FM_PARTLOAD_PROCESSOR_H

#include "../cin_split/cin_split_processor.h"

namespace small_conv {

template <typename BlockCtx, typename TilingData, typename Scheme>
struct FmPartloadProcessor {
    using FmapT = typename BlockCtx::FmapT;
    using WeightT = typename BlockCtx::WeightT;
    using BiasT = typename BlockCtx::BiasT;
    using OutputT = typename BlockCtx::OutputT;
    using Output1T = typename BlockCtx::Output1T;
    using L0cT = typename BlockCtx::L0cT;
    using State = CinState<BlockCtx, TilingData>;

    Scheme scheme_;
    CinSplitProcessor<BlockCtx, TilingData, Scheme> base_;
    SetupGmTool<BlockCtx, TilingData> setupGm_;
    LoadWeightL1Tool<BlockCtx, TilingData> loadWeight_;
    LoadBiasScaleL1Tool<BlockCtx, TilingData> loadBiasScale_;

    // FM-partload uses the host-decided kAL1 to derive cinL1 (one fmap L1 buffer
    // sized by aL1SpaceSize), unlike the cin-split heuristic.
    __aicore__ inline void InitStateFm(const BlockCtx& blockCtx, const TilingData& tiling, State& state) const
    {
        constexpr uint32_t GK0 = BlockCtx::GK0;
        constexpr uint32_t GK0Fmap = BlockCtx::GK0Fmap;
        uint32_t coutAligned = static_cast<uint32_t>(AlignB(tiling.cout, GN0));
        uint32_t k1Total = blockCtx.k1Total;
        state.n1Total = coutAligned / GN0;
        state.n1PerCore = blockCtx.n1PerCore;
        state.l1Pingpong = 0;
        state.l0pp = 0;
        // The per-buffer size comes from the tiling (aL1SpaceSize), so the A1
        // buffer count must mirror the tiling's pBufferFlag bit3 (AL1_DB): a
        // single buffer with serialized cin-block preloads when the bit is off,
        // with weight/bias/scale placed right after it.
        state.al1Db = (tiling.pBufferFlag & AL1_DB_FLAG) != 0;

        state.cinL1 = static_cast<uint32_t>(AlignB(tiling.kAL1 / tiling.kernelHxkernelW, GK0));
        state.cinL1Blocks = static_cast<uint32_t>(CeilDiv(AlignB(tiling.singleCoreCi, GK0Fmap), state.cinL1));

        // Mode-specific L1 width (M-mode: full width; HW-mode: block W window).
        FmapRange r = scheme_.ResolveBlockWindow(blockCtx, tiling);
        state.orgWin = r.orgWin;

        state.al1BufBytes = tiling.aL1SpaceSize;
        uint32_t al1ElemPerBuf = tiling.aL1SpaceSize / sizeof(FmapT);

        ComputeL1LayoutTool<BlockCtx, TilingData> l1Layout;
        state.l1 = l1Layout(tiling, state.al1Db ? 2 : 1, al1ElemPerBuf, k1Total * state.n1PerCore * GN0 * GK0);
    }

    __aicore__ inline void operator()(const BlockCtx& blockCtx, const TilingData& tiling, GM_ADDR x, GM_ADDR filter,
                                      GM_ADDR bias, GM_ADDR y, const ExtendParams* ep) const
    {
        if (!blockCtx.active || blockCtx.actualCo == 0 || blockCtx.singleCoreBatch == 0)
            return;
        State state;
        InitStateFm(blockCtx, tiling, state);

        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
        SetFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF0);
        SetFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF1);

        Process(blockCtx, tiling, state, x, filter, bias, y, ep);

        WaitFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF1);
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
    }

private:
    // Unified group-axis / inner-batch / spatial-chunk loop reusing the
    // cin-split chunk machinery with the one-shot weight load per group. The
    // scheme's group plan degenerates to a single step in HW-mode.
    __aicore__ inline void Process(const BlockCtx& blockCtx, const TilingData& tiling, State& state, GM_ADDR x,
                                   GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* ep) const
    {
        GM_ADDR y1 = (tiling.dualOutput != 0) ? ep->y1 : y;
        bool needLoadBias = true;
        SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
        for (uint32_t groupIter = 0; groupIter < scheme_.GroupIters(blockCtx); groupIter++) {
            GroupPlan groupPlan = scheme_.GroupAt(blockCtx, tiling, groupIter);
            if (!groupPlan.valid) {
                continue;
            }
            blockCtx.actualCo = groupPlan.actualCo;
            GlobalTensor<BiasT> biasGm;
            GlobalTensor<OutputT> outputGm;
            GlobalTensor<Output1T> output1Gm;
            GlobalTensor<uint64_t> scale0Gm, scale1Gm;
            GlobalTensor<float> reluWeight0Gm, reluWeight1Gm;
            setupGm_(blockCtx, tiling, x, filter, bias, y, ep, groupIter, groupPlan.groupCoutOff,
                     groupPlan.groupChanOff, scheme_.OutBaseOff(blockCtx, tiling), state.fmapGm, state.filterGm, biasGm,
                     outputGm, output1Gm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm);
            // Per-group N stride for the shared LoadWeightL1Block addressing.
            state.n1Total = static_cast<uint32_t>(AlignB(blockCtx.groupCoutStep, GN0)) / GN0;

            if (tiling.hasBias) {
                WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
            }
            loadBiasScale_(blockCtx, tiling, biasGm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm, state.l1);
            LocalTensor<WeightT> bl1Full(TPosition::B1, state.l1.bl1OffBytes, state.l1.bl1ElemCount);
            loadWeight_(blockCtx, tiling, state.filterGm, bl1Full);

            SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);
            WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);
            // Gate every later Fixpipe on the bias/scale/relu L1 copy. Legacy
            // behaviour armed this whenever the (single) bias load ran, regardless
            // of hasBias — vector quant/relu must be covered too.
            SetFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));

            uint32_t kL0 = static_cast<uint32_t>(tiling.kL0);
            uint32_t kL0Iters = static_cast<uint32_t>(CeilDiv(blockCtx.kTotalFmap, kL0));
            for (uint32_t batchIdx = 0; batchIdx < blockCtx.singleCoreBatch; batchIdx++) {
                blockCtx.curBatchIdx = blockCtx.batchStart + batchIdx;
                base_.setupFmapGmBatch(tiling, x, blockCtx.curBatchIdx, groupPlan.groupChanOff, state.fmapGm);
                base_.ProcessOneBatch(blockCtx, tiling, state, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm, kL0,
                                      kL0Iters, y, y1, x, groupPlan.groupCoutOff, groupPlan.groupChanOff, batchIdx,
                                      false, needLoadBias);
            }
            if (tiling.hasBias && tiling.nL0 < tiling.nBL1) {
                SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
            }
            needLoadBias = true;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
    }
};

} // namespace small_conv

#endif // FM_PARTLOAD_PROCESSOR_H
