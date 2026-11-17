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
 * \file chunk_loss_kernel_arch35.h
 * \brief ChunkLoss arch35 设备侧 Kernel 主实现
 *
 * Pass1：Cube 计算 Z=X W^T，Vector online LSE；Pass2：BuildDz 与 dH=dZ W（Cube 或 Vector float）。
 */
#ifndef CHUNKLOSS_KERNEL_ARCH35_H_
#define CHUNKLOSS_KERNEL_ARCH35_H_

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstdint>
#include "kernel_operator.h"
#include "chunk_loss_common_arch35.h"
#include "cube_op.h"
#include "vector_api/copy_z_dz.h"
#include "vector_api/vf_online_lse.h"
#include "vector_api/vf_build_dz.h"
#include "vector_api/vf_accum_dh.h"

namespace chunkloss {

template <class Element, uint32_t TileM = kTileMDefault, bool kL0cToUb = false>
class Kernel {
public:
    static constexpr uint32_t kHalfM = TileM / 2;
    static constexpr bool kUseL0cToUb = kL0cToUb;
    static constexpr uint32_t kUbLogitStageBytes = kHalfM * kTileN * static_cast<uint32_t>(sizeof(float));
    static constexpr uint32_t kUbLogitBytes = kStages * kUbLogitStageBytes;
    static constexpr bool kDzNz = false;
    // Pass2 H 分块；LoadDhTile 行跨度与 kTileH 一致。
    static constexpr uint32_t kTileHPass2 = kTileH;

    CHUNKLOSS_DEVICE Kernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CHUNKLOSS_DEVICE void operator()(Params const& params);

    template <>
    CHUNKLOSS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        AscendC::TPipe pipe;
        CubeOp<Element, TileM, kL0cToUb> cube;
        cube.Init(&pipe);

        uint32_t nSplit = params.nSplit < 1u ? 1u : params.nSplit;
        uint32_t deterministic = params.deterministic ? 1u : 0u;
        uint32_t mTiles = (params.m + TileM - 1) / TileM;
        if (mTiles == 0) {
            mTiles = 1;
        }
        uint32_t nTiles = (params.v + kTileN - 1) / kTileN;
        uint32_t hTiles = (params.h + kTileHPass2 - 1) / kTileHPass2;

        AscendC::GlobalTensor<Element> gmX, gmW, gmDz;
        AscendC::GlobalTensor<float> gmLogit, gmDhAcc, gmTempDh;
        constexpr uint32_t kElemBytes = static_cast<uint32_t>(sizeof(Element));
        gmX.SetGlobalBuffer((__gm__ Element*)params.hidden);
        gmW.SetGlobalBuffer((__gm__ Element*)params.weight);
        gmLogit.SetGlobalBuffer((__gm__ float*)params.workspace);
        gmDz.SetGlobalBuffer(
            (__gm__ Element*)(params.workspace + DzBytesOffset(params.aicNum, params.h, params.v, kElemBytes, TileM,
                                                               mTiles, nSplit, deterministic)));
        gmDhAcc.SetGlobalBuffer((__gm__ float*)(params.workspace + FloatBaseBytes(params.aicNum, kElemBytes, TileM)) +
                                DhAccOffsetFloats(params.aicNum));
        gmTempDh.SetGlobalBuffer((__gm__ float*)(params.workspace + FloatBaseBytes(params.aicNum, kElemBytes, TileM)) +
                                 TempDhOffsetFloats(params.aicNum, params.h, TileM, mTiles, nSplit));

        // L0C→UB: VECCALC on same TPipe as SoftSync Resource.ubBuf (offset 0).
        // Separate ubPipe breaks AIC/AIV shared UB → CrossCore hang.
        AscendC::TBuf<AscendC::TPosition::VECCALC> ubLogitBuf;
        AscendC::LocalTensor<float> ubLogit0, ubLogit1;
        if constexpr (kL0cToUb) {
            pipe.InitBuffer(ubLogitBuf, kUbSize);
            ubLogit0 = ubLogitBuf.GetWithOffset<float>(kHalfM * kTileN, 0);
            ubLogit1 = ubLogitBuf.GetWithOffset<float>(kHalfM * kTileN, kUbLogitStageBytes);
        }

        uint32_t core = AscendC::GetBlockIdx();
        uint32_t mtBegin = 0, mtStep = 1, ni = 0, ntStep = 1;
        if (nSplit > 1u) {
            if (core >= mTiles * nSplit) {
                cube.Fini();
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
            uint32_t m0 = mt * TileM;
            uint32_t actualM = m0 + TileM <= params.m ? TileM : params.m - m0;
            // Private per-core dhAcc: nSplit siblings never share writers (det-safe).
            uint64_t dhBase = static_cast<uint64_t>(core) * TileM * params.h;
            AscendC::GlobalTensor<Element> gmXRow = gmX[static_cast<uint64_t>(m0) * params.h];

            uint32_t tileIter = 0;
            for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                VTail vts[2];
                uint32_t nVt = CollectVTails(nt, params.v, vts);
                for (uint32_t vi = 0; vi < nVt; ++vi) {
                    uint32_t stage = tileIter & 1u;
                    if (tileIter >= kStages) {
                        uint16_t freeFlag = stage == 0 ? kFlagAivFree0 : kFlagAivFree1;
                        if constexpr (kL0cToUb) {
                            AscendC::CrossCoreWaitFlag<kSyncModeL0cUb, PIPE_FIX>(freeFlag);
                            AscendC::CrossCoreWaitFlag<kSyncModeL0cUb, PIPE_FIX>(freeFlag + kFlagIdMax);
                        } else {
                            AscendC::CrossCoreWaitFlag(freeFlag);
                        }
                    }
                    VTail vt = vts[vi];
                    uint64_t slot = (static_cast<uint64_t>(core) * kStages + stage) * TileM * kTileN;
                    bool hasNext = (vi + 1u < nVt) || (nt + ntStep < nTiles);
                    VTail vtNext = vt;
                    if (vi + 1u < nVt) {
                        vtNext = vts[vi + 1u];
                    } else if (nt + ntStep < nTiles) {
                        VTail nxt[2];
                        CollectVTails(nt + ntStep, params.v, nxt);
                        vtNext = nxt[0];
                    }
                    if constexpr (kL0cToUb) {
                        cube.SetUbDst(stage == 0 ? ubLogit0 : ubLogit1, kTileN);
                    }
                    cube.MatmulZW(gmXRow, gmW, gmLogit[slot], actualM, vt.nCube, params.h, vt.n0Cube, vtNext.n0Cube,
                                  vtNext.nCube, /*isFirst=*/(tileIter == 0), hasNext);
                    if constexpr (kL0cToUb) {
                        AscendC::CrossCoreSetFlag<kSyncModeL0cUb, PIPE_FIX>(kFlagAicDone);
                        AscendC::CrossCoreSetFlag<kSyncModeL0cUb, PIPE_FIX>(kFlagAicDone + kFlagIdMax);
                    } else {
                        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(kFlagAicDone);
                    }
                    ++tileIter;
                }
            }
            uint32_t pending = tileIter < kStages ? tileIter : kStages;
            for (uint32_t i = 0; i < pending; ++i) {
                uint32_t stage = (tileIter - pending + i) & 1u;
                uint16_t freeFlag = stage == 0 ? kFlagAivFree0 : kFlagAivFree1;
                if constexpr (kL0cToUb) {
                    AscendC::CrossCoreWaitFlag<kSyncModeL0cUb, PIPE_FIX>(freeFlag);
                    AscendC::CrossCoreWaitFlag<kSyncModeL0cUb, PIPE_FIX>(freeFlag + kFlagIdMax);
                } else {
                    AscendC::CrossCoreWaitFlag(freeFlag);
                }
            }
            if constexpr (kL0cToUb) {
                cube.ResetForPass2();
            }
            // Vector float dH 时跳过 Cube Pass2（无需 AIC↔AIV Pass2 握手）。
            const bool useVecDh = chunkloss::UseVecDh(params.m, params.v, nSplit);
            if (!useVecDh) {
                uint64_t dzCore = static_cast<uint64_t>(core) * kStages * TileM * kTileN;
                uint64_t tempBase = static_cast<uint64_t>(core) * TileM * params.h;
                tileIter = 0;
                for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                    VTail vts[2];
                    uint32_t nVt = CollectVTails(nt, params.v, vts);
                    for (uint32_t vi = 0; vi < nVt; ++vi) {
                        VTail vt = vts[vi];
                        uint32_t bank = tileIter & 1u;
                        uint64_t dzSlot = dzCore + static_cast<uint64_t>(bank) * TileM * kTileN;
                        AscendC::CrossCoreWaitFlag(bank == 0 ? kFlagAivFree0 : kFlagAivFree1);
                        // rem-V/short-M: TempDh RMW; otherwise AtomicAdd(+FIX when deterministic).
                        // nSplit: private stripes → sibling reduce (no cross-core AtomicAdd).
                        bool needAccum = (tileIter > 0u);
                        bool remVTemp = false;
                        if (!kL0cToUb && nSplit <= 1u) {
                            uint32_t nTilesV = (params.v + kTileN - 1u) / kTileN;
                            uint32_t m128 = (params.m + kTileMDefault - 1u) / kTileMDefault;
                            if (m128 == 0u) {
                                m128 = 1u;
                            }
                            constexpr uint32_t kAicFull = 28u;
                            if (nTilesV > 1u && m128 < kAicFull && ((params.v % kTileN) != 0u || m128 <= 1u)) {
                                remVTemp = true;
                            }
                        }
                        // Prefer AtomicAdd for rem-V (better match vs float golden than TempDh RMW).
                        bool useTempDh = false;
                        (void)remVTemp;
                        bool useAtomic = needAccum && !useTempDh;
                        bool writeTemp = needAccum && useTempDh;
                        if (useAtomic) {
                            AscendC::SetAtomicAdd<float>();
                        }
                        for (uint32_t ht = 0; ht < hTiles; ++ht) {
                            uint32_t h0 = ht * kTileHPass2;
                            uint32_t actualH = h0 + kTileHPass2 <= params.h ? kTileHPass2 : params.h - h0;
                            const bool loadA = (ht == 0u);
                            const bool hasNextH = (ht + 1u < hTiles);
                            uint32_t h0Next = 0u;
                            uint32_t actualHNext = 0u;
                            if (hasNextH) {
                                h0Next = (ht + 1u) * kTileHPass2;
                                actualHNext = h0Next + kTileHPass2 <= params.h ? kTileHPass2 : params.h - h0Next;
                            }
                            if (writeTemp) {
                                cube.MatmulDH(gmDz[dzSlot], gmW, gmTempDh[tempBase + h0], actualM, vt.nCube, actualH,
                                              h0, vt.n0Cube, params.h, loadA, hasNextH, actualHNext, h0Next);
                            } else {
                                cube.MatmulDH(gmDz[dzSlot], gmW, gmDhAcc[dhBase + h0], actualM, vt.nCube, actualH, h0,
                                              vt.n0Cube, params.h, loadA, hasNextH, actualHNext, h0Next);
                            }
                        }
                        cube.ReleaseDhA();
                        if (useAtomic) {
                            AscendC::SetAtomicNone();
                        }
                        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(bank == 0 ? kFlagTempReady : kFlagAddDone);
                        ++tileIter;
                    }
                }
            }
        }
        cube.Fini();
    }

    template <>
    CHUNKLOSS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        uint32_t sub = AscendC::GetSubBlockIdx();
        uint32_t core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        // Always carve CE workspace from Resource.ubBuf so L0C→UB Fixpipe offsets match AIC.
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;
        pipe.InitBuffer(ubBuf, kUbSize);
        // Pass2 RMW/cast reuses logits/dzUb; size to max(N, H tile).
        constexpr uint32_t kTileWide = kTileHPass2 > kTileN ? kTileHPass2 : kTileN;
        constexpr uint32_t kLogitRegion = kUseL0cToUb ? kUbLogitBytes :
                                                        (kHalfM * kTileWide * static_cast<uint32_t>(sizeof(float)));
        constexpr uint32_t kOffMax = kLogitRegion;
        constexpr uint32_t kMaxBytes = (kHalfM * kTileN / kDataPerRepeat) * 2u * sizeof(float);
        constexpr uint32_t kOffSum = kOffMax + kMaxBytes;
        constexpr uint32_t kOffBrcb = kOffSum + kMaxBytes;
        constexpr uint32_t kBrcbBytes = kHalfM * 8u * sizeof(float);
        constexpr uint32_t kOffPreMax = kOffBrcb + kBrcbBytes;
        constexpr uint32_t kOffPreSum = kOffPreMax + kHalfM * sizeof(float);
        constexpr uint32_t kOffKeep = kOffPreSum + kHalfM * sizeof(float);
        constexpr uint32_t kOffScratch = kOffKeep + kHalfM * sizeof(float);
        constexpr uint32_t kOffDz = kOffScratch + 32u * sizeof(float);
        constexpr uint32_t kDzBytes = kHalfM * kTileWide * static_cast<uint32_t>(sizeof(Element));
        constexpr uint32_t kOffDhTmp = kOffDz + kDzBytes;
        // AtomicAdd path skips dhTmp; TempDh RMW needs a float H-tile.
        constexpr uint32_t kDhTmpBytes = kUseL0cToUb ? 0u :
                                                       (kHalfM * kTileHPass2 * static_cast<uint32_t>(sizeof(float)));
        static_assert(kOffDhTmp + kDhTmpBytes <= kUbSize, "AIV ub map OOB");

        AscendC::LocalTensor<float> logits = ubBuf.GetWithOffset<float>(kHalfM * kTileWide, 0);
        auto maxTensor = ubBuf.GetWithOffset<float>(kMaxBytes / sizeof(float), kOffMax);
        auto sumTensor = ubBuf.GetWithOffset<float>(kMaxBytes / sizeof(float), kOffSum);
        auto brcbMax = ubBuf.GetWithOffset<float>(kBrcbBytes / sizeof(float), kOffBrcb);
        auto preMax = ubBuf.GetWithOffset<float>(kHalfM, kOffPreMax);
        auto preSum = ubBuf.GetWithOffset<float>(kHalfM, kOffPreSum);
        auto keepMax = ubBuf.GetWithOffset<float>(kHalfM, kOffKeep);
        auto scratch = ubBuf.GetWithOffset<float>(32, kOffScratch);
        auto dzUb = ubBuf.GetWithOffset<Element>(kHalfM * kTileWide, kOffDz);
        auto dhTmp = kUseL0cToUb ? ubBuf.GetWithOffset<float>(1, 0) :
                                   ubBuf.GetWithOffset<float>(kHalfM * kTileHPass2, kOffDhTmp);

        AscendC::GlobalTensor<float> gmLogit;
        AscendC::GlobalTensor<float> gmZ;
        AscendC::GlobalTensor<Element> gmDz;
        AscendC::GlobalTensor<float> gmDhAcc;
        AscendC::GlobalTensor<float> gmState;
        AscendC::GlobalTensor<int64_t> gmLabel;
        AscendC::GlobalTensor<float> gmPartial;
        AscendC::GlobalTensor<float> gmLoss;
        AscendC::GlobalTensor<float> gmMaxOut;
        AscendC::GlobalTensor<float> gmSumOut;
        AscendC::GlobalTensor<Element> gmDH;
        AscendC::GlobalTensor<Element> gmW;
        constexpr uint32_t kElemBytes = static_cast<uint32_t>(sizeof(Element));
        uint32_t nSplit = params.nSplit < 1u ? 1u : params.nSplit;
        uint32_t deterministic = params.deterministic ? 1u : 0u;
        uint32_t mTiles = (params.m + TileM - 1) / TileM;
        if (mTiles == 0) {
            mTiles = 1;
        }
        uint64_t floatBase = FloatBaseBytes(params.aicNum, kElemBytes, TileM);
        gmLogit.SetGlobalBuffer((__gm__ float*)params.workspace);
        gmZ.SetGlobalBuffer((__gm__ float*)(params.workspace + ZFullBytesOffset(params.aicNum, params.h, kElemBytes,
                                                                                TileM, mTiles, nSplit, deterministic)));
        gmDz.SetGlobalBuffer(
            (__gm__ Element*)(params.workspace + DzBytesOffset(params.aicNum, params.h, params.v, kElemBytes, TileM,
                                                               mTiles, nSplit, deterministic)));
        gmDhAcc.SetGlobalBuffer((__gm__ float*)(params.workspace + floatBase) + DhAccOffsetFloats(params.aicNum));
        AscendC::GlobalTensor<float> gmTempDh;
        gmTempDh.SetGlobalBuffer((__gm__ float*)(params.workspace + floatBase) +
                                 TempDhOffsetFloats(params.aicNum, params.h, TileM, mTiles, nSplit));
        gmLabel.SetGlobalBuffer((__gm__ int64_t*)params.labels);
        gmPartial.SetGlobalBuffer((__gm__ float*)(params.workspace + floatBase) +
                                  PartialOffsetFloats(params.aicNum, params.h, TileM, mTiles, nSplit));
        gmState.SetGlobalBuffer((__gm__ float*)(params.workspace + floatBase) +
                                StatePartialOffsetFloats(params.aicNum, params.h, TileM, mTiles, nSplit));
        gmLoss.SetGlobalBuffer((__gm__ float*)params.loss);
        gmMaxOut.SetGlobalBuffer((__gm__ float*)params.maxOut);
        gmSumOut.SetGlobalBuffer((__gm__ float*)params.sumOut);
        gmDH.SetGlobalBuffer((__gm__ Element*)params.dH);
        gmW.SetGlobalBuffer((__gm__ Element*)params.weight);

        uint32_t nTiles = (params.v + kTileN - 1) / kTileN;
        uint32_t hTiles = (params.h + kTileHPass2 - 1) / kTileHPass2;
        const bool useVecDh = chunkloss::UseVecDh(params.m, params.v, nSplit);
        float invAlpha = params.alpha; // scale factor: loss/dZ *= alpha
        // Vec: unscaled dZ (+one-hot). Cube: dZ=S*(softmax-oh) with S=min(V,65504) so -S fits half.
        float dzScale = 1.0f;
        if (!useVecDh) {
            constexpr float kFp16Max = 65504.0f;
            float vScale = static_cast<float>(params.v);
            dzScale = (vScale > kFp16Max) ? kFp16Max : ((vScale < 1.0f) ? 1.0f : vScale);
        }
        float dhScale = invAlpha / dzScale;
        float acc = 0.0f;

        uint32_t mtBegin = 0;
        uint32_t mtStep = 1;
        uint32_t ni = 0;
        uint32_t ntStep = 1;
        bool active = true;
        if (nSplit > 1u) {
            if (core >= mTiles * nSplit) {
                active = false;
            } else {
                mtBegin = core % mTiles;
                mtStep = mTiles;
                ni = core / mTiles;
                ntStep = nSplit;
            }
        } else {
            mtBegin = core;
            mtStep = params.aicNum;
            ni = 0;
            ntStep = 1;
        }

        if (active) {
            for (uint32_t mt = mtBegin; mt < mTiles; mt += mtStep) {
                uint32_t m0 = mt * TileM;
                uint32_t actualM = m0 + TileM <= params.m ? TileM : params.m - m0;
                // L0C→UB SPLIT_M delivers CeilDiv(M,2) rows per AIV (HW); GM path keeps floor half.
                uint32_t half = kUseL0cToUb ? ((actualM + 1u) / 2u) : (actualM / 2u);
                uint32_t rowBegin = sub == 0 ? 0 : half;
                uint32_t rows = sub == 0 ? half : (actualM > half ? actualM - half : 0u);
                uint32_t rowsEven = rows & ~1u;

                bool valid[kHalfM];
                bool got[kHalfM];
                float tgt[kHalfM];
                int64_t lab[kHalfM];
                float oddM = kPadLogit;
                float oddS = 0.0f;
                for (uint32_t i = 0; i < kHalfM; ++i) {
                    valid[i] = false;
                    got[i] = false;
                    tgt[i] = 0.0f;
                    lab[i] = params.ignoreIndex;
                }
                if (rows > 0) {
                    AscendC::Duplicate(preMax.template ReinterpretCast<int32_t>(), kNegInfBits, rows);
                    AscendC::Duplicate(preSum, 0.0f, rows);
                    for (uint32_t i = 0; i < rows; ++i) {
                        lab[i] = gmLabel.GetValue(m0 + rowBegin + i);
                        valid[i] = lab[i] != params.ignoreIndex && lab[i] >= 0 &&
                                   lab[i] < static_cast<int64_t>(params.v);
                    }
                }

                uint64_t zBase = (nSplit > 1u) ? static_cast<uint64_t>(mt) * TileM * params.v :
                                                 static_cast<uint64_t>(core) * TileM * params.v;
                uint32_t tileIter = 0;
                for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                    VTail vts[2];
                    uint32_t nVt = CollectVTails(nt, params.v, vts);
                    for (uint32_t vi = 0; vi < nVt; ++vi) {
                        uint32_t stage = tileIter & 1u;
                        if constexpr (kUseL0cToUb) {
                            // Mode 4 wait on PIPE_V: Fixpipe→UB visible before CE (matmul_full_dequant).
                            AscendC::CrossCoreWaitFlag<kSyncModeL0cUb, PIPE_V>(kFlagAicDone);
                        } else {
                            AscendC::CrossCoreWaitFlag(kFlagAicDone);
                        }
                        VTail vt = vts[vi];
                        uint64_t slot = (static_cast<uint64_t>(core) * kStages + stage) * TileM * kTileN;
                        if (rows > 0) {
                            if (tileIter > 0u) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                            }
                            if constexpr (kUseL0cToUb) {
                                // CE on Fixpipe float at same byte offsets AIC SetUbDst uses.
                                logits = ubBuf.GetWithOffset<float>(kHalfM * kTileN, stage * kUbLogitStageBytes);
                                if (vt.nCube < kTileN) {
                                    for (uint32_t r = 0; r < rows; ++r) {
                                        AscendC::Duplicate(logits[r * kTileN + vt.nCube], kPadLogit, kTileN - vt.nCube);
                                    }
                                    AscendC::PipeBarrier<PIPE_V>();
                                }
                            } else {
                                chunkloss::CopyLogits(logits, gmLogit[slot + static_cast<uint64_t>(rowBegin) * kTileN],
                                                      rows, vt.nCube);
                            }
                            chunkloss::MaskPadLeftLogits(logits, rows, vt.padLeft);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                            if (vt.padLeft > 0u) {
                                // Stub: pack one aligned row into scratch, then MTE3 pad-copy (src 32B-aligned).
                                AscendC::GlobalTensor<float>
                                    zDst = gmZ[zBase + static_cast<uint64_t>(rowBegin) * params.v + vt.n0];
                                for (uint32_t r = 0; r < rows; ++r) {
                                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                                    for (uint32_t c = 0; c < vt.actualN; ++c) {
                                        scratch.SetValue(c, logits.GetValue(r * kTileN + vt.padLeft + c));
                                    }
                                    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
                                    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
                                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                                    AscendC::DataCopyPad(
                                        zDst[static_cast<uint64_t>(r) * params.v], scratch,
                                        {1, static_cast<uint32_t>(vt.actualN * sizeof(float)), 0, 0, 0});
                                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                                }
                            } else {
                                chunkloss::StoreDzTile(gmZ, logits, zBase, rowBegin, rows, vt.n0, vt.actualN, params.v,
                                                       0);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                            int64_t nBegin = static_cast<int64_t>(vt.n0);
                            int64_t nCube0 = static_cast<int64_t>(vt.n0Cube);
                            for (uint32_t i = 0; i < rows; ++i) {
                                if (!valid[i] || got[i]) {
                                    continue;
                                }
                                int64_t y = lab[i];
                                if (y >= nBegin && y < nBegin + static_cast<int64_t>(vt.actualN)) {
                                    tgt[i] = logits.GetValue(i * kTileN + static_cast<uint32_t>(y - nCube0));
                                    got[i] = true;
                                }
                            }
                            if ((rows & 1u) != 0) {
                                chunkloss::UpdateOddRowAt(logits, rows - 1u, vt.nCube, oddM, oddS, scratch);
                            }
                            if (rowsEven > 0) {
                                chunkloss::OnlineMax(logits, maxTensor, preMax, rowsEven);
                                AscendC::Adds(keepMax, maxTensor, 0.0f, rowsEven);
                                chunkloss::OnlineSum(logits, sumTensor, brcbMax, preMax, preSum, maxTensor, rowsEven);
                                AscendC::Adds(preMax, keepMax, 0.0f, rowsEven);
                                AscendC::Adds(preSum, sumTensor, 0.0f, rowsEven);
                            }
                        }
                        uint16_t freeFlag = stage == 0 ? kFlagAivFree0 : kFlagAivFree1;
                        if constexpr (kUseL0cToUb) {
                            AscendC::CrossCoreSetFlag<kSyncModeL0cUb, PIPE_MTE2>(freeFlag);
                        } else {
                            AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(freeFlag);
                        }
                        ++tileIter;
                    } // vi
                } // nt
                if (rows > 0 && tileIter > 0u) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                }

                if (rows > 0) {
                    if ((rows & 1u) != 0) {
                        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                        preMax.SetValue(rows - 1u, oddM);
                        preSum.SetValue(rows - 1u, oddS);
                    }
                }

                if (nSplit > 1u) {
                    if (rows > 0) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::DataCopyPad(gmState[static_cast<uint64_t>(core) * TileM + rowBegin], preMax,
                                             {1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0});
                        AscendC::DataCopyPad(gmState[static_cast<uint64_t>(params.aicNum) * TileM +
                                                     static_cast<uint64_t>(core) * TileM + rowBegin],
                                             preSum, {1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0});
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    AscendC::SyncAll();
                    if (rows > 0) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
                        for (uint32_t i = 0; i < rows; ++i) {
                            float newMax = kPadLogit;
                            for (uint32_t c = 0; c < nSplit; ++c) {
                                uint32_t sib = mt + c * mTiles;
                                float pm = gmState.GetValue(static_cast<uint64_t>(sib) * TileM + rowBegin + i);
                                if (pm > newMax) {
                                    newMax = pm;
                                }
                            }
                            float newSum = 0.0f;
                            for (uint32_t c = 0; c < nSplit; ++c) {
                                uint32_t sib = mt + c * mTiles;
                                float pm = gmState.GetValue(static_cast<uint64_t>(sib) * TileM + rowBegin + i);
                                float ps = gmState.GetValue(static_cast<uint64_t>(params.aicNum) * TileM +
                                                            static_cast<uint64_t>(sib) * TileM + rowBegin + i);
                                scratch.SetValue(0, pm - newMax);
                                for (uint32_t t = 1; t < 8; ++t) {
                                    scratch.SetValue(t, -80.0f);
                                }
                                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
                                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
                                AscendC::Exp(scratch, scratch, 8);
                                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
                                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
                                newSum += ps * scratch.GetValue(0);
                            }
                            preMax.SetValue(i, newMax);
                            preSum.SetValue(i, newSum);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
                        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
                    }
                }

                if (rows > 0) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    if (nSplit == 1u || ni == 0u) {
                        AscendC::DataCopyPad(gmMaxOut[m0 + rowBegin], preMax,
                                             {1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0});
                        AscendC::DataCopyPad(gmSumOut[m0 + rowBegin], preSum,
                                             {1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0});
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                    for (uint32_t base = 0; base < rows; base += 8) {
                        uint32_t n = rows - base > 8 ? 8 : rows - base;
                        for (uint32_t i = 0; i < 8; ++i) {
                            scratch.SetValue(i, i < n ? preSum.GetValue(base + i) : 1.0f);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
                        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
                        AscendC::Ln(keepMax, scratch, 8);
                        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                        for (uint32_t i = 0; i < n; ++i) {
                            if (!valid[base + i] || !got[base + i]) {
                                continue;
                            }
                            acc += preMax.GetValue(base + i) + keepMax.GetValue(i) - tgt[base + i];
                        }
                    }
                }

                uint64_t dhBase = static_cast<uint64_t>(core) * TileM * params.h;
                if (nSplit > 1u) {
                    // Barrier only: each core writes a private stripe (no shared zero).
                    AscendC::SyncAll();
                }
                uint64_t dzCore = static_cast<uint64_t>(core) * kStages * TileM * kTileN;
                uint64_t tempBase = static_cast<uint64_t>(core) * TileM * params.h;
                uint32_t localTiles = 0;
                for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                    VTail tmp[2];
                    localTiles += CollectVTails(nt, params.v, tmp);
                }
                if (useVecDh) {
                    // Vector float dH：清零 stripe 后在 UB 累加 dZ@W。
                    if (rows > 0) {
                        for (uint32_t ht = 0; ht < hTiles; ++ht) {
                            uint32_t h0 = ht * kTileHPass2;
                            uint32_t actualH = h0 + kTileHPass2 <= params.h ? kTileHPass2 : params.h - h0;
                            chunkloss::ZeroDhTile(gmDhAcc, dhTmp, dhBase, rowBegin, rows, h0, actualH, params.h);
                        }
                        tileIter = 0;
                        for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                            VTail vts[2];
                            uint32_t nVt = CollectVTails(nt, params.v, vts);
                            for (uint32_t vi = 0; vi < nVt; ++vi) {
                                VTail vt = vts[vi];
                                chunkloss::LoadZAsLogits(logits, gmZ, zBase, rowBegin, rows, vt.n0Cube, vt.nCube,
                                                         params.v);
                                chunkloss::MaskPadLeftLogits(logits, rows, vt.padLeft);
                                // BuildDz leaves float dZ in logits; dzUb is half cast (scratch for W load).
                                chunkloss::BuildDz<Element>(logits, dzUb, preMax, preSum, brcbMax, keepMax, valid, lab,
                                                            rows, vt.nCube, vt.n0Cube, dzScale, vt.padLeft, true);
                                for (uint32_t ht = 0; ht < hTiles; ++ht) {
                                    uint32_t h0 = ht * kTileHPass2;
                                    uint32_t actualH = h0 + kTileHPass2 <= params.h ? kTileHPass2 : params.h - h0;
                                    chunkloss::LoadDhTile(dhTmp, gmDhAcc, dhBase, rowBegin, rows, h0, actualH,
                                                          params.h);
                                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                                    chunkloss::VecAccumDhTile<Element>(dhTmp, logits, dzUb, maxTensor, gmW, rows,
                                                                       vt.n0Cube, vt.nCube, h0, actualH, params.h);
                                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                                    chunkloss::StoreDhTile(gmDhAcc, dhTmp, dhBase, rowBegin, rows, h0, actualH,
                                                           params.h);
                                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                                }
                                ++tileIter;
                            }
                        }
                    }
                } else {
                    tileIter = 0;
                    for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                        VTail vts[2];
                        uint32_t nVt = CollectVTails(nt, params.v, vts);
                        for (uint32_t vi = 0; vi < nVt; ++vi) {
                            VTail vt = vts[vi];
                            uint32_t bank = tileIter & 1u;
                            uint64_t dzSlot = dzCore + static_cast<uint64_t>(bank) * TileM * kTileN;
                            bool useTempDh = false;
                            if (!useTempDh && tileIter >= kTempBanks) {
                                AscendC::CrossCoreWaitFlag(bank == 0 ? kFlagTempReady : kFlagAddDone);
                            }
                            if (rows > 0) {
                                chunkloss::LoadZAsLogits(logits, gmZ, zBase, rowBegin, rows, vt.n0Cube, vt.nCube,
                                                         params.v);
                                chunkloss::MaskPadLeftLogits(logits, rows, vt.padLeft);
                                // Cube: S*(softmax-oh) in half; S capped so -S is finite.
                                chunkloss::BuildDz<Element>(logits, dzUb, preMax, preSum, brcbMax, keepMax, valid, lab,
                                                            rows, vt.nCube, vt.n0Cube, dzScale, vt.padLeft, true);
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                                uint32_t dzCount = ((rows * kTileN * sizeof(Element) + 31u) / 32u) * 32u /
                                                   sizeof(Element);
                                AscendC::DataCopy(gmDz[dzSlot + static_cast<uint64_t>(rowBegin) * kTileN], dzUb,
                                                  dzCount);
                                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                            }
                            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(bank == 0 ? kFlagAivFree0 : kFlagAivFree1);
                            if (useTempDh) {
                                AscendC::CrossCoreWaitFlag(bank == 0 ? kFlagTempReady : kFlagAddDone);
                                if (tileIter > 0u && rows > 0) {
                                    for (uint32_t ht = 0; ht < hTiles; ++ht) {
                                        uint32_t h0 = ht * kTileHPass2;
                                        uint32_t actualH = h0 + kTileHPass2 <= params.h ? kTileHPass2 : params.h - h0;
                                        chunkloss::LoadDhTile(logits, gmDhAcc, dhBase, rowBegin, rows, h0, actualH,
                                                              params.h);
                                        chunkloss::LoadDhTile(dhTmp, gmTempDh, tempBase, rowBegin, rows, h0, actualH,
                                                              params.h);
                                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                                        AscendC::Add(logits, logits, dhTmp, rows * kTileHPass2);
                                        AscendC::PipeBarrier<PIPE_V>();
                                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                                        chunkloss::StoreDhTile(gmDhAcc, logits, dhBase, rowBegin, rows, h0, actualH,
                                                               params.h);
                                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                                    }
                                }
                            }
                            ++tileIter;
                        } // vi
                    } // nt
                    // Drain AtomicAdd completion flags when not on TempDh path.
                    bool useTempDhDrain = false;
                    if (!useTempDhDrain) {
                        uint32_t pending = localTiles < kTempBanks ? localTiles : kTempBanks;
                        for (uint32_t i = 0; i < pending; ++i) {
                            uint32_t bank = (localTiles - pending + i) & 1u;
                            AscendC::CrossCoreWaitFlag(bank == 0 ? kFlagTempReady : kFlagAddDone);
                        }
                    }
                }

                if (rows > 0 && (nSplit == 1u || ni == 0u)) {
                    if (nSplit > 1u) {
                        AscendC::SyncAll();
                    }
                    for (uint32_t ht = 0; ht < hTiles; ++ht) {
                        uint32_t h0 = ht * kTileHPass2;
                        uint32_t actualH = h0 + kTileHPass2 <= params.h ? kTileHPass2 : params.h - h0;
                        chunkloss::LoadDhTile(logits, gmDhAcc, dhBase, rowBegin, rows, h0, actualH, params.h);
                        // Ordered sibling reduce (ni==0): private stripes → one writer.
                        if (nSplit > 1u) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                            for (uint32_t c = 1u; c < nSplit; ++c) {
                                uint32_t sib = mt + c * mTiles;
                                uint64_t sibBase = static_cast<uint64_t>(sib) * TileM * params.h;
                                chunkloss::LoadDhTile(dhTmp, gmDhAcc, sibBase, rowBegin, rows, h0, actualH, params.h);
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                                AscendC::Add(logits, logits, dhTmp, rows * kTileHPass2);
                                AscendC::PipeBarrier<PIPE_V>();
                            }
                        } else {
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                        }
                        // Undo dZ scale and apply *alpha on float dhAcc before fp16 store.
                        AscendC::Muls(logits, logits, dhScale, rows * kTileHPass2);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Cast(dzUb, logits, AscendC::RoundMode::CAST_ROUND, rows * kTileHPass2);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                        chunkloss::StoreDhElem(gmDH, dzUb, static_cast<uint64_t>(m0 + rowBegin) * params.h + h0, rows,
                                               actualH, params.h);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                    }
                } else if (nSplit > 1u) {
                    AscendC::SyncAll();
                }
            }
        } else if (nSplit > 1u) {
            AscendC::SyncAll();
            AscendC::SyncAll();
            AscendC::SyncAll();
        }

        AscendC::Duplicate(scratch, 0.0f, kPartialStride);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        scratch.SetValue(0, acc);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopy(gmPartial[(core * kStages + sub) * kPartialStride], scratch, kPartialStride);
        AscendC::SyncAll();
        if (core == 0 && sub == 0) {
            uint32_t nPartial = params.aicNum * kStages;
            AscendC::DataCopy(logits, gmPartial, nPartial * kPartialStride);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
            float total = 0.0f;
            for (uint32_t i = 0; i < nPartial; ++i) {
                total += logits.GetValue(i * kPartialStride);
            }
            scratch.SetValue(0, total * params.alpha);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(gmLoss, scratch, {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
        }
    }
};

template <class Element, uint32_t TileM = kTileMDefault, bool kL0cToUb = false>
__global__ __aicore__ void Launch(Params params)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Kernel<Element, TileM, kL0cToUb> op;
    op(params);
}

} // namespace chunkloss

#endif
