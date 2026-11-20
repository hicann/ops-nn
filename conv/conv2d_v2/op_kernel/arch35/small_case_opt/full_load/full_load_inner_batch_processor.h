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
 * InnerBatchProcessor: batch-folded variant of the full-load base class (the
 * pipeline selects it only when the InnerBatch SEL != SINGLE). It INHERITS
 * FullLoadProcessor and reuses the per-instruction level (MmadAccumulateTile /
 * ProcessChunks with the fold-mode branches, tools, setupFmapGmBatch); only
 * the flow level differs: the core's singleCoreBatch maps are folded into
 * groups of innerBatch maps (legacy batchIter semantics) — one GM->L1 DMA per
 * group, one compute pass, one fixpipe — and when the two-group AL1 layout
 * fits L1 the next group's fmap is prefetched into the alternate buffer
 * (group double buffering).
 */
#ifndef FULL_LOAD_INNER_BATCH_PROCESSOR_H
#define FULL_LOAD_INNER_BATCH_PROCESSOR_H

#include "../conv2d_small_kernel_tile_ops.h"
#include "../conv2d_inner_batch_tile_ops.h"
#include "full_load_processor.h"

namespace small_conv {

template <typename BlockCtx, typename TilingData, typename Scheme, int8_t InnerBatchMode>
struct InnerBatchProcessor : FullLoadProcessor<BlockCtx, TilingData, Scheme, InnerBatchMode,
                                               CopyOutInnerBatchTool<BlockCtx, TilingData, InnerBatchMode>> {
    using Base = FullLoadProcessor<BlockCtx, TilingData, Scheme, InnerBatchMode,
                                   CopyOutInnerBatchTool<BlockCtx, TilingData, InnerBatchMode>>;
    using FmapT = typename BlockCtx::FmapT;
    using WeightT = typename BlockCtx::WeightT;
    using BiasT = typename BlockCtx::BiasT;
    using L0cT = typename BlockCtx::L0cT;

    // Fold-specific tools; copyOut_ shadows the base member so the inherited
    // ProcessChunks dispatches to the folded fixpipe.
    LoadAL1InnerBatchTool<BlockCtx, TilingData, InnerBatchMode> loadFmapIb_;

    __aicore__ inline void operator()(const BlockCtx& blockCtx, const TilingData& tiling, GM_ADDR x, GM_ADDR filter,
                                      GM_ADDR bias, GM_ADDR y, const ExtendParams* ep) const
    {
        uint32_t kTotalCur = blockCtx.kTotalFmap;

        FmapRange blockWindow = this->scheme_.ResolveBlockWindow(blockCtx, tiling);
        // One folded group of innerBatch maps; the group count and tail size
        // derive straight from the tiling data.
        const uint32_t loopBatch = CeilDiv(blockCtx.singleCoreBatch, static_cast<uint32_t>(tiling.innerBatch));
        uint32_t al1ElemPerBuf = static_cast<uint32_t>(tiling.innerBatch) * blockWindow.curHi * blockWindow.orgWin *
                                 blockCtx.cinAlignedFmap;
        uint32_t bl1ElemCount = blockCtx.k1Total * blockCtx.n1PerCore * GN0 * Base::GK0;

        // Group double buffering: prefetch the next group's fmap into the
        // alternate AL1 buffer while the current group computes; only when the
        // complete two-buffer layout fits in L1 — otherwise groups reload
        // serially.
        L1Layout l1Two = this->l1Layout_(tiling, 2, al1ElemPerBuf, bl1ElemCount);
        bool enableGroupDoubleBuffer = (loopBatch > 1) && (l1Two.endBytes <= TOTAL_L1_SIZE);
        L1Layout l1Layout = enableGroupDoubleBuffer ? l1Two : this->l1Layout_(tiling, 1, al1ElemPerBuf, bl1ElemCount);

        LocalTensor<FmapT> al1(TPosition::A1, 0, l1Layout.al1ElemCount);
        LocalTensor<WeightT> bl1(TPosition::B1, l1Layout.bl1OffBytes, l1Layout.bl1ElemCount);

        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
        uint32_t l0pp = 0;

        Process(blockCtx, tiling, blockWindow, l1Layout, x, filter, bias, y, ep, enableGroupDoubleBuffer, al1, bl1,
                l0pp, kTotalCur);

        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
    }

private:
    __aicore__ inline void Process(const BlockCtx& blockCtx, const TilingData& tiling, const FmapRange& blockWindow,
                                   const L1Layout& l1Layout, GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                   const ExtendParams* ep, bool enableGroupDoubleBuffer, LocalTensor<FmapT>& al1,
                                   LocalTensor<WeightT>& bl1, uint32_t& l0pp, uint32_t kTotalCur) const
    {
        GM_ADDR y1 = (tiling.dualOutput != 0) ? ep->y1 : y;
        bool needLoadBias = true;
        SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
        SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_L1_DONE);
        // innerBatch and groups are mutually exclusive (host/SEL/fullload all
        // guarantee it), so this conv-group loop runs a single iteration.
        for (uint32_t groupIter = 0; groupIter < this->scheme_.GroupIters(blockCtx); groupIter++) {
            GroupPlan groupPlan = this->scheme_.GroupAt(blockCtx, tiling, groupIter);
            if (!groupPlan.valid) {
                continue;
            }
            blockCtx.actualCo = groupPlan.actualCo;

            WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_L1_DONE);
            GlobalTensor<FmapT> fmapGm;
            GlobalTensor<WeightT> filterGm;
            GlobalTensor<BiasT> biasGm;
            GlobalTensor<typename Base::OutputT> outputGm;
            GlobalTensor<typename Base::Output1T> output1Gm;
            GlobalTensor<uint64_t> scale0Gm, scale1Gm;
            GlobalTensor<float> reluWeight0Gm, reluWeight1Gm;
            this->setupGm_(blockCtx, tiling, x, filter, bias, y, ep, groupIter, groupPlan.groupCoutOff,
                           groupPlan.groupChanOff, this->scheme_.OutBaseOff(blockCtx, tiling), fmapGm, filterGm, biasGm,
                           outputGm, output1Gm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm);

            this->loadWeight_(blockCtx, tiling, filterGm, bl1);
            SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);
            WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);

            // Arm the M<->FIX handshake so the first tile's Mmad may write L0C.
            SetFlag<HardEvent::FIX_M>(EVT_CL0_DONE);

            if (enableGroupDoubleBuffer) {
                ProcessGroupsDoubleBuffer(blockCtx, tiling, blockWindow, l1Layout, x, y, y1, groupPlan.groupCoutOff,
                                          groupPlan.groupChanOff, biasGm, scale0Gm, scale1Gm, reluWeight0Gm,
                                          reluWeight1Gm, needLoadBias, al1, bl1, l0pp, kTotalCur);
            } else {
                ProcessGroupsSerial(blockCtx, tiling, blockWindow, l1Layout, x, y, y1, groupPlan.groupCoutOff,
                                    groupPlan.groupChanOff, biasGm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm,
                                    needLoadBias, al1, bl1, l0pp, kTotalCur);
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

    // Serial group reload: load -> fence -> compute -> fence, one AL1 buffer.
    __aicore__ inline void ProcessGroupsSerial(const BlockCtx& blockCtx, const TilingData& tiling,
                                               const FmapRange& blockWindow, const L1Layout& l1Layout, GM_ADDR x,
                                               GM_ADDR y, GM_ADDR y1, uint32_t groupCoutOff, uint32_t groupChanOff,
                                               GlobalTensor<BiasT>& biasGm, GlobalTensor<uint64_t>& scale0Gm,
                                               GlobalTensor<uint64_t>& scale1Gm, GlobalTensor<float>& reluWeight0Gm,
                                               GlobalTensor<float>& reluWeight1Gm, bool& needLoadBias,
                                               LocalTensor<FmapT>& al1, LocalTensor<WeightT>& bl1, uint32_t& l0pp,
                                               uint32_t kTotalCur) const
    {
        const uint32_t innerBatch = static_cast<uint32_t>(tiling.innerBatch);
        const uint32_t loopBatch = CeilDiv(blockCtx.singleCoreBatch, innerBatch);
        const uint32_t tail = blockCtx.singleCoreBatch - (loopBatch - 1) * innerBatch;
        for (uint32_t batchGroupIdx = 0; batchGroupIdx < loopBatch; batchGroupIdx++) {
            blockCtx.curIb = (batchGroupIdx + 1 == loopBatch) ? tail : innerBatch;
            blockCtx.curBatchIdx = blockCtx.batchStart + batchGroupIdx * innerBatch;
            Load3DCtx ctxG = this->setupLoad3d_(blockCtx, tiling, blockWindow,
                                                this->scheme_.BlockBasePosM(blockCtx, tiling));
            GlobalTensor<FmapT> fmapGmG;
            this->setupFmapGmBatch(tiling, x, blockCtx.curBatchIdx, groupChanOff, fmapGmG);
            loadFmapIb_(blockCtx, tiling, fmapGmG, al1, blockWindow.curHi, blockWindow.hiStart, blockWindow.curWi);
            if (batchGroupIdx == 0) {
                // Per-channel bias/scale/relu: load once, queued behind the
                // first fmap DMA (innerbatch and groups are mutually exclusive).
                if (tiling.hasBias) {
                    WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
                }
                this->loadBiasScale_(blockCtx, tiling, biasGm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm,
                                     l1Layout);
                SetFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
            }
            SetFlag<HardEvent::MTE2_MTE1>(EVT_BATCH_BUF0);
            WaitFlag<HardEvent::MTE2_MTE1>(EVT_BATCH_BUF0);
            this->ProcessChunks(blockCtx, tiling, l1Layout, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm, kTotalCur,
                                y, y1, groupCoutOff, needLoadBias, al1, bl1, ctxG, l0pp);
            if (batchGroupIdx + 1 < loopBatch) {
                SetFlag<HardEvent::MTE1_MTE2>(EVT_BATCH_BUF0);
                WaitFlag<HardEvent::MTE1_MTE2>(EVT_BATCH_BUF0);
            }
        }
    }

    // Group double buffering: prefetch group g+1's fmap into the alternate
    // buffer while group g computes; the prefetch's curIb/GM base ride on a
    // local BlockCtx copy so the live blockCtx keeps group g's values for the
    // fixpipe's GM addressing.
    __aicore__ inline void ProcessGroupsDoubleBuffer(
        const BlockCtx& blockCtx, const TilingData& tiling, const FmapRange& blockWindow, const L1Layout& l1Layout,
        GM_ADDR x, GM_ADDR y, GM_ADDR y1, uint32_t groupCoutOff, uint32_t groupChanOff, GlobalTensor<BiasT>& biasGm,
        GlobalTensor<uint64_t>& scale0Gm, GlobalTensor<uint64_t>& scale1Gm, GlobalTensor<float>& reluWeight0Gm,
        GlobalTensor<float>& reluWeight1Gm, bool& needLoadBias, LocalTensor<FmapT>& al1, LocalTensor<WeightT>& bl1,
        uint32_t& l0pp, uint32_t kTotalCur) const
    {
        const uint32_t innerBatch = static_cast<uint32_t>(tiling.innerBatch);
        const uint32_t loopBatch = CeilDiv(blockCtx.singleCoreBatch, innerBatch);
        const uint32_t tail = blockCtx.singleCoreBatch - (loopBatch - 1) * innerBatch;
        uint32_t al1ElemPerBuf = l1Layout.al1ElemCount / 2;
        uint32_t lastGroup = loopBatch - 1;
        // Preload group 0 into buffer 0.
        BlockCtx bcPre = blockCtx;
        bcPre.curIb = (lastGroup == 0) ? tail : innerBatch;
        GlobalTensor<FmapT> fmapGm0;
        this->setupFmapGmBatch(tiling, x, blockCtx.batchStart, groupChanOff, fmapGm0);
        LocalTensor<FmapT> al1Buf0(TPosition::A1, 0, al1ElemPerBuf);
        loadFmapIb_(bcPre, tiling, fmapGm0, al1Buf0, blockWindow.curHi, blockWindow.hiStart, blockWindow.curWi);
        // Per-channel bias/scale/relu: load once, behind the group-0 preload.
        if (tiling.hasBias) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
        }
        this->loadBiasScale_(blockCtx, tiling, biasGm, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm, l1Layout);
        SetFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
        SetFlag<HardEvent::MTE2_MTE1>(EVT_BATCH_BUF0);
        for (uint32_t batchGroupIdx = 0; batchGroupIdx < loopBatch; batchGroupIdx++) {
            blockCtx.curIb = (batchGroupIdx == lastGroup) ? tail : innerBatch;
            blockCtx.curBatchIdx = blockCtx.batchStart + batchGroupIdx * innerBatch;
            uint32_t curBuf = batchGroupIdx & 1;
            uint32_t nextBuf = 1 - curBuf;
            if (batchGroupIdx < lastGroup) {
                // Prefetch the next group into the alternate buffer.
                BlockCtx bcNext = blockCtx;
                bcNext.curIb = (batchGroupIdx + 1 == lastGroup) ? tail : innerBatch;
                GlobalTensor<FmapT> fmapGmN;
                this->setupFmapGmBatch(tiling, x, blockCtx.batchStart + (batchGroupIdx + 1) * innerBatch, groupChanOff,
                                       fmapGmN);
                if (batchGroupIdx > 0) {
                    WaitFlag<HardEvent::MTE1_MTE2>(nextBuf ? EVT_BATCH_BUF1 : EVT_BATCH_BUF0);
                }
                LocalTensor<FmapT> al1Next(TPosition::A1, nextBuf * l1Layout.al1BufBytes, al1ElemPerBuf);
                loadFmapIb_(bcNext, tiling, fmapGmN, al1Next, blockWindow.curHi, blockWindow.hiStart,
                            blockWindow.curWi);
                SetFlag<HardEvent::MTE2_MTE1>(nextBuf ? EVT_BATCH_BUF1 : EVT_BATCH_BUF0);
            }
            WaitFlag<HardEvent::MTE2_MTE1>(curBuf ? EVT_BATCH_BUF1 : EVT_BATCH_BUF0);
            Load3DCtx ctxG = this->setupLoad3d_(blockCtx, tiling, blockWindow,
                                                this->scheme_.BlockBasePosM(blockCtx, tiling));
            LocalTensor<FmapT> al1Cur(TPosition::A1, curBuf * l1Layout.al1BufBytes, al1ElemPerBuf);
            this->ProcessChunks(blockCtx, tiling, l1Layout, scale0Gm, scale1Gm, reluWeight0Gm, reluWeight1Gm, kTotalCur,
                                y, y1, groupCoutOff, needLoadBias, al1Cur, bl1, ctxG, l0pp);
            if (batchGroupIdx + 2 < loopBatch) {
                SetFlag<HardEvent::MTE1_MTE2>(curBuf ? EVT_BATCH_BUF1 : EVT_BATCH_BUF0);
            }
        }
    }
};

} // namespace small_conv

#endif // FULL_LOAD_INNER_BATCH_PROCESSOR_H
