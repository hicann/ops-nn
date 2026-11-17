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
 * ChunkLoss arch22 CubeOp: Pass1 Z GEMM + Pass2 dH GEMM.
 * Pure AscendC (no Catlass). Schedule mirrors Catlass cube_op:
 * full-K HandZ/HandH, reuse blocks, SoftSync unitFlag CrossCore.
 */
#ifndef CHUNK_LOSS_CUBE_OP_ARCH22_H_
#define CHUNK_LOSS_CUBE_OP_ARCH22_H_

#include "chunk_loss_common_arch22.h"
#include "chunk_loss_hand_mmad.h"

namespace chunkloss {

template <class Element, uint32_t TileM = kTileMDefault, bool kPreloadZ = true, bool kFullLoadH = true,
          bool kCallbackZ = false, bool kBatchZ = false, bool kWsNz = kWsNzDefault>
class CubeOp {
public:
    static constexpr uint32_t kTileM = TileM;
    static constexpr bool kUsePreloadZ = kPreloadZ;
    static constexpr bool kUseFullLoadH = kFullLoadH;
    // case19 Large: Catlass uses classic Preload (Wait at tile start), not Callback.
    static constexpr bool kUseCallbackZ = true;
    static constexpr bool kUseBatchZ = false;
    static constexpr bool kUseWsNz = kWsNz;
    static_assert(!kWsNz, "arch22 hand cube currently ships ND workspace (kWsNzDefault=false)");

    using HandZ = HandMmadZ<Element, TileM, kTileN, 256, 64, true>;
    using HandH = HandMmadH<Element, TileM, kTileH, 256, 64, true>;

    CHUNK_LOSS_DEVICE void Process(Params const& params)
    {
        CubeBuf cubeBuf;
        cubeBuf.Init();

        uint32_t nSplit = params.nSplit < 1u ? 1u : params.nSplit;
        uint32_t mTiles = (params.m + kTileM - 1) / kTileM;
        if (mTiles == 0) {
            mTiles = 1;
        }
        uint32_t nTiles = (params.v + kTileN - 1) / kTileN;
        uint32_t hTiles = (params.h + kTileH - 1) / kTileH;

        AscendC::GlobalTensor<Element> gmX;
        AscendC::GlobalTensor<Element> gmW;
        AscendC::GlobalTensor<Element> gmLogit;
        AscendC::GlobalTensor<Element> gmDz;
        AscendC::GlobalTensor<float> gmDhAcc;
        constexpr uint32_t kElemBytes = static_cast<uint32_t>(sizeof(Element));
        gmX.SetGlobalBuffer((__gm__ Element*)params.hidden);
        gmW.SetGlobalBuffer((__gm__ Element*)params.weight);
        gmLogit.SetGlobalBuffer((__gm__ Element*)params.workspace);
        gmDz.SetGlobalBuffer(
            (__gm__ Element*)(params.workspace + DzBytesOffset(params.aicNum, params.h, params.v, kElemBytes, kTileM,
                                                               mTiles, nSplit, params.rematZ, params.logitAsZ)));
        gmDhAcc.SetGlobalBuffer((__gm__ float*)(params.workspace + FloatBaseBytes(params.aicNum, kTileM, kElemBytes)) +
                                DhAccOffsetFloats(params.aicNum, kTileM));

        uint32_t core = AscendC::GetBlockIdx();
        uint32_t mtBegin = 0;
        uint32_t mtStep = 1;
        uint32_t ni = 0;
        uint32_t ntStep = 1;
        if (nSplit > 1u) {
            if (core >= mTiles * nSplit) {
                return;
            }
            mtBegin = core % mTiles;
            mtStep = mTiles;
            ni = core / mTiles;
            ntStep = nSplit;
        } else {
            mtBegin = core;
            mtStep = params.aicNum;
            ni = 0;
            ntStep = 1;
        }

        for (uint32_t mt = mtBegin; mt < mTiles; mt += mtStep) {
            uint32_t m0 = mt * kTileM;
            uint32_t actualM = m0 + kTileM <= params.m ? kTileM : params.m - m0;
            uint64_t offX = static_cast<uint64_t>(m0) * params.h;
            uint64_t dhBase = DhAccBaseFloats(nSplit > 1u ? mt : core, kTileM, params.h, nSplit, ni);

            {
                HandZ blockZ(cubeBuf);
                uint32_t tileIter = 0;
                for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                    uint32_t stage = tileIter & (kStages - 1u);
                    uint32_t n0 = nt * kTileN;
                    uint32_t actualN = n0 + kTileN <= params.v ? kTileN : params.v - n0;
                    uint64_t offW = static_cast<uint64_t>(n0) * params.h;
                    uint64_t slot = (static_cast<uint64_t>(core) * kStages + stage) * kTileM * kTileN;
                    bool hasNext = (nt + ntStep < nTiles);
                    uint32_t n0Next = hasNext ? (nt + ntStep) * kTileN : n0;
                    uint32_t actualNNext = hasNext ? (n0Next + kTileN <= params.v ? kTileN : params.v - n0Next) :
                                                     actualN;
                    uint64_t offWNext = static_cast<uint64_t>(n0Next) * params.h;
                    // Callback: Wait AIV after Mmad / before Fixpipe. K2 prefetch overlaps Wait.
                    bool waitAiv = (tileIter >= kStages);
                    blockZ.Compute(gmX[offX], gmW[offW], gmLogit[slot], actualM, actualN, params.h, params.h, params.h,
                                   gmX[offX], gmW[offWNext], actualM, actualNNext, params.h, tileIter == 0u, hasNext,
                                   waitAiv, stage, /*setAicDone=*/true);
                    ++tileIter;
                }
                {
                    uint32_t pending = tileIter < kStages ? tileIter : kStages;
                    for (uint32_t i = 0; i < pending; ++i) {
                        uint32_t stage = (tileIter - pending + i) & (kStages - 1u);
                        AscendC::CrossCoreWaitFlag(FlagAivFree(stage));
                    }
                }
            }

            uint64_t dzCore = static_cast<uint64_t>(core) * kStages * kTileM * kTileN;
            if (params.rematZ != 0u) {
                uint32_t tileIter = 0;
                for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                    uint32_t n0 = nt * kTileN;
                    uint32_t actualN = n0 + kTileN <= params.v ? kTileN : params.v - n0;
                    uint32_t bank = tileIter & 1u;
                    uint64_t slot = (static_cast<uint64_t>(core) * kStages + bank) * kTileM * kTileN;
                    uint64_t dzSlot = dzCore + static_cast<uint64_t>(bank) * kTileM * kTileN;
                    uint64_t offW = static_cast<uint64_t>(n0) * params.h;
                    {
                        HandZ blockZ(cubeBuf);
                        blockZ.Compute(gmX[offX], gmW[offW], gmLogit[slot], actualM, actualN, params.h, params.h,
                                       params.h, gmX[offX], gmW[offW], actualM, actualN, params.h, true, false);
                        SetRemat(bank);
                    }
                    HandH blockH(cubeBuf);
                    {
                        uint64_t offWh0 = static_cast<uint64_t>(n0) * params.h;
                        uint32_t actualH0 = kTileH <= params.h ? kTileH : params.h;
                        blockH.PrefetchB(gmW[offWh0], actualH0, actualN, params.h);
                    }
                    WaitAivFreeP2(bank);
                    if (tileIter > 0u) {
                        AscendC::SetAtomicAdd<float>();
                    }
                    for (uint32_t ht = 0; ht < hTiles; ++ht) {
                        uint32_t h0 = ht * kTileH;
                        uint32_t actualH = h0 + kTileH <= params.h ? kTileH : params.h - h0;
                        uint64_t offWh = static_cast<uint64_t>(n0) * params.h + h0;
                        bool needLoadL1 = (!kUseFullLoadH) || (ht == 0);
                        bool hasNextH = (ht + 1u < hTiles);
                        uint32_t h0Next = hasNextH ? (ht + 1u) * kTileH : h0;
                        uint32_t actualHNext = hasNextH ? (h0Next + kTileH <= params.h ? kTileH : params.h - h0Next) :
                                                          actualH;
                        uint64_t offWhNext = static_cast<uint64_t>(n0) * params.h + h0Next;
                        blockH.Compute(gmDz[dzSlot], gmW[offWh], gmDhAcc[dhBase + h0], actualM, actualH, actualN,
                                       kTileN, params.h, params.h, needLoadL1, gmW[offWhNext], actualHNext, actualN,
                                       hasNextH);
                    }
                    if (tileIter > 0u) {
                        AscendC::SetAtomicNone();
                    }
                    SetTempReady();
                    ++tileIter;
                }
            } else {
                uint32_t tileIter = 0;
                bool rematPosted = false;
                HandH blockH(cubeBuf);
                for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                    uint32_t n0 = nt * kTileN;
                    uint32_t actualN = n0 + kTileN <= params.v ? kTileN : params.v - n0;
                    uint32_t bank = tileIter & (kStages - 1u);
                    uint64_t dzSlot = dzCore + static_cast<uint64_t>(bank) * kTileM * kTileN;
                    // Prior iter may have SetRemat(next) early so BuildDz overlaps HandH.
                    if (!rematPosted) {
                        SetRemat(bank);
                    }
                    rematPosted = false;
                    // Prefetch W before Wait so MTE2 overlaps AIV BuildDz (skip if prior N left B prefetched).
                    if (!blockH.HasPrefetchedB()) {
                        uint64_t offWh0 = static_cast<uint64_t>(n0) * params.h;
                        uint32_t actualH0 = kTileH <= params.h ? kTileH : params.h;
                        blockH.PrefetchB(gmW[offWh0], actualH0, actualN, params.h);
                    }
                    WaitAivFreeP2(bank);
                    if (tileIter > 0u) {
                        AscendC::SetAtomicAdd<float>();
                    }
                    bool hasNextN = (nt + ntStep < nTiles);
                    uint32_t n0Next = hasNextN ? (nt + ntStep) * kTileN : n0;
                    uint32_t actualNNext = hasNextN ? (n0Next + kTileN <= params.v ? kTileN : params.v - n0Next) :
                                                      actualN;
                    // BuildDz(next) ∥ HandH(cur): arm next Remat + Temp only when another N remains.
                    // Last tile keeps Temp after HandH so AIV cannot cast dhAcc while AIC AtomicAdds.
                    if (hasNextN) {
                        SetRemat((bank + 1u) & (kStages - 1u));
                        rematPosted = true;
                        SetTempReady();
                    }
                    for (uint32_t ht = 0; ht < hTiles; ++ht) {
                        uint32_t h0 = ht * kTileH;
                        uint32_t actualH = h0 + kTileH <= params.h ? kTileH : params.h - h0;
                        uint64_t offWh = static_cast<uint64_t>(n0) * params.h + h0;
                        bool needLoadL1 = (!kUseFullLoadH) || (ht == 0);
                        bool hasNextH = (ht + 1u < hTiles);
                        bool prefetchNext = hasNextH || hasNextN;
                        uint32_t h0Next = hasNextH ? (ht + 1u) * kTileH : 0u;
                        uint32_t actualHNext = hasNextH ? (h0Next + kTileH <= params.h ? kTileH : params.h - h0Next) :
                                                          (kTileH <= params.h ? kTileH : params.h);
                        uint32_t nForNext = hasNextH ? actualN : actualNNext;
                        uint64_t offWhNext = hasNextH ? (static_cast<uint64_t>(n0) * params.h + h0Next) :
                                                        (static_cast<uint64_t>(n0Next) * params.h);
                        blockH.Compute(gmDz[dzSlot], gmW[offWh], gmDhAcc[dhBase + h0], actualM, actualH, actualN,
                                       kTileN, params.h, params.h, needLoadL1, gmW[offWhNext], actualHNext, nForNext,
                                       prefetchNext);
                    }
                    if (tileIter > 0u) {
                        AscendC::SetAtomicNone();
                    }
                    if (!hasNextN) {
                        SetTempReady();
                    }
                    ++tileIter;
                }
            }
            (void)kCallbackZ;
            (void)kBatchZ;
            (void)kUsePreloadZ;
        }
    }
};

} // namespace chunkloss

#endif
