/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file full_mn_kernel_processor.h
 * \brief FullMNKernelProcessor: kernel-layer K-loop with L1 segment pre-load ping-pong.
 */

#ifndef FULL_MN_KERNEL_PROCESSOR_H
#define FULL_MN_KERNEL_PROCESSOR_H

#include "full_mn_kernel_common.h"
#include "full_mn_kernel_block_ctx.h"
#include "full_mn_kernel_tile_ops.h"
#include "../conv2d_v2_weight_5hd_impl.h"

namespace full_mn {

template <typename BlockCtx, typename TilingData, typename OneProc>
struct BatchProcessor {
    __aicore__ inline void operator()(BlockCtx& bc, const TilingData& tiling, GM_ADDR x, GM_ADDR filter, GM_ADDR bias,
                                      GM_ADDR y, GM_ADDR workspace) const
    {
        SetHF32Mode(tiling.hf32Enable);
        SetHF32TransMode(tiling.hf32TransMode);
        uint32_t singleCoreBatch = tiling.singleCoreBatch;
        uint32_t baseBatchIdx = bc.batchIdx;
        uint32_t totalBatch = tiling.batch;
        uint32_t batchLoop = (baseBatchIdx + singleCoreBatch) > totalBatch ? (totalBatch - baseBatchIdx) :
                                                                             singleCoreBatch;
        OneProc oneProc_;
        for (uint32_t i = 0; i < batchLoop; i++) {
            bc.batchIdx = baseBatchIdx + i;
            oneProc_(bc, tiling, x, filter, bias, y, workspace);
        }
    }
};

template <typename BlockCtx, typename TilingData>
struct FullMNKernelProcessor {
    using FmapT = typename BlockCtx::FmapType;
    using WeightT = typename BlockCtx::WeightType;
    using OutputT = typename BlockCtx::OutputType;
    using BiasT = typename BlockCtx::BiasType;
    using L0cT = typename BlockCtx::L0cType;

    LoadAL1SegTool<BlockCtx, TilingData> loadAl1Seg_;
    LoadBL1Seg5HDTool<BlockCtx, TilingData> loadBl1Seg5HD_;
    LoadBiasL1Tool<BlockCtx, TilingData> loadBiasL1_;
    LoadBiasToBTTool<BlockCtx, TilingData> loadBiasToBt_;
    SetupLoad3DTool<BlockCtx, TilingData> setupLoad3d_;
    LoadAL0Tool<BlockCtx> loadAl0_;
    LoadBL0Tool<BlockCtx> loadBl0_;
    MmadTool<BlockCtx> mmad_;
    CopyOutTool<BlockCtx, TilingData> copyOut_;

    mutable uint32_t loadedChunkCnt_ = 0;

    __aicore__ inline void operator()(const BlockCtx& bc, const TilingData& tiling, GM_ADDR x, GM_ADDR filter,
                                      GM_ADDR bias, GM_ADDR y, GM_ADDR workspace) const
    {
        uint32_t orgWo = static_cast<uint32_t>(tiling.wout);
        uint32_t hoStart = bc.mIdxStart / orgWo;
        uint32_t hoEnd = (bc.mIdxStart + bc.actualM - 1) / orgWo;
        uint32_t hiStart = hoStart * tiling.strideH;
        uint32_t hiEnd = hoEnd * tiling.strideH + tiling.dilationH * (tiling.kh - 1);
        uint32_t hiTotal = hiEnd - hiStart + 1;

        int32_t padTopL1 = 0;
        int32_t padBottomL1 = 0;
        uint32_t curHiLoad = hiTotal;
        uint32_t hiLoadStart = hiStart;

        if (hiStart < tiling.padTop) {
            padTopL1 = static_cast<int32_t>(tiling.padTop - hiStart);
            curHiLoad -= static_cast<uint32_t>(padTopL1);
            hiLoadStart = 0;
        } else {
            hiLoadStart = hiStart - tiling.padTop;
        }

        uint32_t hinVal = static_cast<uint32_t>(tiling.hin);
        if (hiEnd >= hinVal + tiling.padTop) {
            padBottomL1 = static_cast<int32_t>(hiEnd - (hinVal + tiling.padTop) + 1);
            curHiLoad -= static_cast<uint32_t>(padBottomL1);
        }

        // ===== Stage 1: GM address setup =====
        uint64_t batchFmapOff = static_cast<uint64_t>(bc.batchIdx) * tiling.cin * tiling.hin * tiling.win;
        GlobalTensor<FmapT> fmapGm;
        fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ FmapT*>(x) + batchFmapOff +
                               static_cast<uint64_t>(hiLoadStart) * tiling.win);
        GlobalTensor<WeightT> filterGm;
        GlobalTensor<WeightT> weight5HDGm;
        filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(filter));
        weight5HDGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(workspace));

        uint64_t hwOut = static_cast<uint64_t>(tiling.hout) * tiling.wout;
        uint64_t batchOutOff = static_cast<uint64_t>(bc.batchIdx) * tiling.cout * hwOut;
        GlobalTensor<OutputT> outputGm;
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(y) + batchOutOff +
                                 static_cast<uint64_t>(bc.nIdxStart) * hwOut + bc.mIdxStart);

        GlobalTensor<BiasT> biasGm;
        if (tiling.hasBias) {
            biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(bias) + static_cast<uint64_t>(bc.nIdxStart));
        }

        // ===== Stage 2: Load K segment 0 to AL1/BL1 Buf[0] =====
        // ===== L1 buffer layout calculation =====
        uint32_t khxkw = tiling.kh * tiling.kw;
        uint32_t al1ElemCount = tiling.kAL1 / khxkw * curHiLoad * tiling.win;
        uint32_t al1BufBytes = AlignB(al1ElemCount * sizeof(FmapT), ADDR_ALIGN_SIZE);
        // ===== L1 tensor creation (ping-pong: Buf[0] and Buf[1]) =====
        LocalTensor<FmapT> al1Buf0(TPosition::A1, 0, al1ElemCount);
        LocalTensor<FmapT> al1Buf1(TPosition::A1, al1BufBytes, al1ElemCount);

        uint32_t bl1Offset = AlignB(DOUBLE_BUFFER_NUM * al1BufBytes, ADDR_ALIGN_SIZE);
        uint32_t bl1ElemCount = tiling.kBL1 * tiling.nBL1;
        uint32_t bl1BufBytes = bl1ElemCount * sizeof(WeightT);
        LocalTensor<WeightT> bl1Buf0(TPosition::B1, bl1Offset, bl1ElemCount);
        LocalTensor<WeightT> bl1Buf1(TPosition::B1, bl1Offset + bl1BufBytes, bl1ElemCount);

        // batch 边界屏障：上一 batch 的 MTE1 段内 L1 读（loadAl0_/loadBl0_）全部完成后，
        // 本 batch 的 MTE2 写（覆写同一 al1/bl1 缓冲）才允许执行——防跨 batch L1 读写竞争。
        // 首 batch 由该 Set 立即放行（MTE1 队列为空）；每 batch 末尾配套 Set 维持配对。
        SetFlag<HardEvent::MTE1_MTE2>(EVT_BATCH_BARRIER);
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_BATCH_BARRIER);

        loadAl1Seg_(bc, tiling, curHiLoad, hiLoadStart, fmapGm, 0, al1Buf0);
        const uint32_t kBL1Segs0 = Bl1SegCount(tiling);
        const bool isFstVisit = loadedChunkCnt_ == 0U;
        if (isFstVisit) {
            CrossCoreWaitFlag<Conv2dFunc::WEIGHT_5HD_SYNC_MODE_2, PIPE_MTE2>(Conv2dFunc::WEIGHT_5HD_SYNC_ID_BASE);
            loadedChunkCnt_ = 1U;
        }
        loadBl1Seg5HD_(bc, tiling, weight5HDGm, 0, bl1Buf0);
        if (isFstVisit && kBL1Segs0 > Conv2dFunc::WEIGHT_5HD_EVENT_ID_NUM) {
            CrossCoreSetFlag<Conv2dFunc::WEIGHT_5HD_SYNC_MODE_2, PIPE_MTE2>(Conv2dFunc::WEIGHT_5HD_SYNC_ID_BASE);
        }

        uint32_t biasL1Offset = AlignB(bl1Offset + DOUBLE_BUFFER_NUM * bl1BufBytes, ADDR_ALIGN_SIZE);
        if (tiling.hasBias) {
            loadBiasL1_(bc, tiling, biasGm, biasL1Offset);
        }

        // ===== L0 tensor creation (ping-pong) =====
        uint32_t al0BufElems = AL0_BUF_BYTES / sizeof(FmapT);
        uint32_t bl0BufElems = BL0_BUF_BYTES / sizeof(WeightT);
        LocalTensor<FmapT> al0Buf0(TPosition::A2, 0, al0BufElems);
        LocalTensor<FmapT> al0Buf1(TPosition::A2, AL0_BUF_BYTES, al0BufElems);
        LocalTensor<WeightT> bl0Buf0(TPosition::B2, 0, bl0BufElems);
        LocalTensor<WeightT> bl0Buf1(TPosition::B2, BL0_BUF_BYTES, bl0BufElems);
        LocalTensor<L0cT> cl0(TPosition::CO1, 0, L0C_ELEMS);

        SetFlag<HardEvent::MTE2_MTE1>(EVT_AL1_BASE);
        SetFlag<HardEvent::MTE2_MTE1>(EVT_BL1_BASE);
        WaitFlag<HardEvent::MTE2_MTE1>(EVT_AL1_BASE);
        WaitFlag<HardEvent::MTE2_MTE1>(EVT_BL1_BASE);

        if (tiling.hasBias) {
            loadBiasToBt_(bc, tiling, biasL1Offset);
        }
        // ===== Stage 3: Setup Load3D invariants =====
        uint64_t load3dXtBase = 0;
        uint64_t load3dXmTmp = 0;
        setupLoad3d_(bc, tiling, curHiLoad, padTopL1, load3dXtBase, load3dXmTmp);

        // ===== Stage 4: K-loop =====
        uint32_t cinAligned = static_cast<uint32_t>(AlignB(tiling.singleCoreCi, BlockCtx::GK0));
        uint32_t kTotal = cinAligned * tiling.kh * tiling.kw;
        uint32_t currentML0Align = static_cast<uint32_t>(AlignB(bc.actualM, M0));
        KLoop(bc, tiling, kTotal, currentML0Align, curHiLoad, hiLoadStart, fmapGm, filterGm, weight5HDGm, outputGm, cl0,
              al1Buf0, al1Buf1, bl1Buf0, bl1Buf1, al0Buf0, al0Buf1, bl0Buf0, bl0Buf1, load3dXtBase, load3dXmTmp);

        // ===== Stage 5: Fixpipe + CopyOut =====
        copyOut_(bc, tiling, outputGm, cl0);
    }

private:
    static __aicore__ inline uint32_t Bl1SegCount(const TilingData& tiling)
    {
        uint32_t cinAligned = static_cast<uint32_t>(AlignB(tiling.singleCoreCi, BlockCtx::GK0));
        return static_cast<uint32_t>(
            CeilDiv(static_cast<uint64_t>(cinAligned) * tiling.kh * tiling.kw, static_cast<uint64_t>(tiling.kBL1)));
    }

    __aicore__ inline void KLoop(const BlockCtx& bc, const TilingData& tiling, uint32_t kTotal,
                                 uint32_t currentML0Align, uint32_t curHiLoad, uint32_t hiLoadStart,
                                 const GlobalTensor<FmapT>& fmapGm, const GlobalTensor<WeightT>& filterGm,
                                 const GlobalTensor<WeightT>& weight5HDGm, GlobalTensor<OutputT>& outputGm,
                                 LocalTensor<L0cT>& cl0, LocalTensor<FmapT>& al1Buf0, LocalTensor<FmapT>& al1Buf1,
                                 LocalTensor<WeightT>& bl1Buf0, LocalTensor<WeightT>& bl1Buf1,
                                 LocalTensor<FmapT>& al0Buf0, LocalTensor<FmapT>& al0Buf1,
                                 LocalTensor<WeightT>& bl0Buf0, LocalTensor<WeightT>& bl0Buf1, uint64_t load3dXtBase,
                                 uint64_t load3dXmTmp) const
    {
        uint32_t kL0 = tiling.kL0;
        uint32_t ddr2l0LoopK = static_cast<uint32_t>(CeilDiv(kTotal, kL0));      // k方向L0迭代总数
        uint32_t multiKAL1 = CeilDiv(tiling.kAL1, kL0);                          // 每段 L1 -> L0A 载多少次
        uint32_t multiKBL1 = CeilDiv(tiling.kBL1, kL0);                          // 每段 L1 -> L0B 载多少次
        uint32_t kAL1Segs = static_cast<uint32_t>(CeilDiv(kTotal, tiling.kAL1)); // AL1总段数
        uint32_t kBL1Segs = static_cast<uint32_t>(CeilDiv(kTotal, tiling.kBL1)); // BL1总段数

        uint32_t k0FmapTail = static_cast<uint32_t>(C0_SIZE) / sizeof(FmapT);
        uint32_t alignCinATailInCore = static_cast<uint32_t>(AlignB(tiling.cinATailInCore, k0FmapTail));

        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));

        MmadParams mp;
        mp.m = currentML0Align;
        mp.n = static_cast<uint32_t>(AlignB(bc.actualCo, N0));
        mp.cmatrixInitVal = !(tiling.hasBias);
        mp.cmatrixSource = static_cast<bool>(tiling.hasBias);

        uint32_t kIter = 0;
        uint32_t kInAL1 = 0;
        uint32_t kInBL1 = 0;
        uint32_t kAL1Iter = 1;
        uint32_t kBL1Iter = 1;
        uint8_t al1PingPongFlag = 0;
        uint8_t bl1PingPongFlag = 0;

        uint32_t load2d_src_stride = tiling.nBL1 / N0;

        while (kIter < ddr2l0LoopK) {
            uint32_t currentKL0 = (kIter == ddr2l0LoopK - 1) ? (kTotal - kIter * kL0) : kL0;
            uint8_t al0PingPongFlag = kIter & 1;

            if (kInAL1 == 0 && kAL1Iter < kAL1Segs) {
                LocalTensor<FmapT>& al1Next = (al1PingPongFlag == 0) ? al1Buf1 : al1Buf0;
                SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(EVT_AL1_BASE + (al1PingPongFlag ^ 1)));
                WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(EVT_AL1_BASE + (al1PingPongFlag ^ 1)));
                loadAl1Seg_(bc, tiling, curHiLoad, hiLoadStart, fmapGm, kAL1Iter, al1Next);
                SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_AL1_BASE + (al1PingPongFlag ^ 1)));
            }
            if (kInBL1 == 0 && kBL1Iter < kBL1Segs) {
                LocalTensor<WeightT>& bl1Next = (bl1PingPongFlag == 0) ? bl1Buf1 : bl1Buf0;
                SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(EVT_BL1_BASE + (bl1PingPongFlag ^ 1)));
                WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(EVT_BL1_BASE + (bl1PingPongFlag ^ 1)));
                const bool isFstVisit = kBL1Iter >= loadedChunkCnt_;
                if (isFstVisit) {
                    CrossCoreWaitFlag<Conv2dFunc::WEIGHT_5HD_SYNC_MODE_2, PIPE_MTE2>(
                        Conv2dFunc::WEIGHT_5HD_SYNC_ID_BASE + (kBL1Iter & 1));
                    loadedChunkCnt_++;
                }
                loadBl1Seg5HD_(bc, tiling, weight5HDGm, kBL1Iter, bl1Next);
                if (isFstVisit && kBL1Iter + Conv2dFunc::WEIGHT_5HD_EVENT_ID_NUM < kBL1Segs) {
                    CrossCoreSetFlag<Conv2dFunc::WEIGHT_5HD_SYNC_MODE_2, PIPE_MTE2>(
                        Conv2dFunc::WEIGHT_5HD_SYNC_ID_BASE + (kBL1Iter & 1));
                }
                SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_BL1_BASE + (bl1PingPongFlag ^ 1)));
            }

            WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(al0PingPongFlag));

            uint32_t posK = kInAL1 * kL0;
            uint32_t channelSize = (kAL1Iter == kAL1Segs) ? alignCinATailInCore : tiling.cinAInCore;

            LocalTensor<FmapT>& al1 = (al1PingPongFlag == 0) ? al1Buf0 : al1Buf1;
            LocalTensor<FmapT>& al0 = (al0PingPongFlag == 0) ? al0Buf0 : al0Buf1;
            LocalTensor<WeightT>& bl1 = (bl1PingPongFlag == 0) ? bl1Buf0 : bl1Buf1;
            LocalTensor<WeightT>& bl0 = (al0PingPongFlag == 0) ? bl0Buf0 : bl0Buf1;

            loadAl0_(al1, al0, posK, currentKL0, channelSize, load3dXtBase, load3dXmTmp);
            uint32_t KStartPosition = kInBL1 * (kL0 / BlockCtx::GK0);
            loadBl0_(bc, bl1, bl0, KStartPosition, currentKL0, load2d_src_stride);

            SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(al0PingPongFlag));
            WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(al0PingPongFlag));

            mp.k = currentKL0;
            mp.unitFlag = (kIter == ddr2l0LoopK - 1) ? UNIT_FLAG_ENABLE_WITH_FLIP : UNIT_FLAG_ENABLE_ONLY;
            mmad_(cl0, al0, bl0, mp);
            SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(al0PingPongFlag));

            mp.cmatrixInitVal = false;
            mp.cmatrixSource = false;

            kIter++;
            kInAL1++;
            kInBL1++;
            if (kInAL1 == multiKAL1) {
                if (kAL1Iter < kAL1Segs) {
                    WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_AL1_BASE + (al1PingPongFlag ^ 1)));
                }
                kAL1Iter++;
                kInAL1 = 0;
                al1PingPongFlag ^= 1;
            }
            if (kInBL1 == multiKBL1) {
                if (kBL1Iter < kBL1Segs) {
                    WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_BL1_BASE + (bl1PingPongFlag ^ 1)));
                }
                kBL1Iter++;
                kInBL1 = 0;
                bl1PingPongFlag ^= 1;
            }
        }

        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
    }
};

} // namespace full_mn

#endif // FULL_MN_KERNEL_PROCESSOR_H
