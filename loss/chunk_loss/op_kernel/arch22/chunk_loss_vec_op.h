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
 * ChunkLoss arch22 VecOp: Pass1 online LSE/loss + Pass2 BuildDz + dH cast (AscendC Vector).
 */
#ifndef CHUNK_LOSS_VEC_OP_ARCH22_H_
#define CHUNK_LOSS_VEC_OP_ARCH22_H_

#include "chunk_loss_common_arch22.h"

namespace chunkloss {

template <class Element, uint32_t TileM = kTileMDefault, bool kPreloadZ = true, bool kFullLoadH = true,
          bool kCallbackZ = false, bool kBatchZ = false, bool kWsNz = kWsNzDefault>
class VecOp {
public:
    static constexpr uint32_t kTileM = TileM;
    static constexpr uint32_t kHalfM = TileM / 2u;
    static constexpr bool kUsePreloadZ = kPreloadZ;
    static constexpr bool kUseFullLoadH = kFullLoadH;
    static constexpr bool kUseCallbackZ = true;
    static constexpr bool kUseBatchZ = false;
    static constexpr bool kUseWsNz = kWsNz;
    // FAG-style NZ helpers: elems per 32B block / C0 fractal.
    static constexpr uint32_t kElePerBlk = 32u / static_cast<uint32_t>(sizeof(Element));
    static constexpr uint32_t kVecRepeat = 8u;

    // NZ GM row pitch is C0 (fractal), ND pitch is TileN.
    CHUNK_LOSS_DEVICE
    static uint64_t WsRowOff(uint32_t rowBegin)
    {
        if constexpr (kUseWsNz) {
            return static_cast<uint64_t>(rowBegin) * kC0Size;
        }
        return static_cast<uint64_t>(rowBegin) * kTileN;
    }

    CHUNK_LOSS_DEVICE
    void Process(Params const& params)
    {
        uint32_t sub = AscendC::GetSubBlockIdx();
        uint32_t core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t nSplit = params.nSplit < 1u ? 1u : params.nSplit;
        uint32_t mTiles = (params.m + kTileM - 1) / kTileM;
        if (mTiles == 0) {
            mTiles = 1;
        }
        uint32_t nTiles = (params.v + kTileN - 1) / kTileN;
        uint32_t hTiles = (params.h + kTileH - 1) / kTileH;
        float invAlpha = params.alpha; // scale factor: loss/dZ *= alpha
        float acc = 0.0f;

        // Single UB pool (r023): logits | reduce(max/sum) | brcb | preMax/preSum/keep | scratch | dz
        constexpr uint32_t kReduceFloats = (kHalfM * kTileN / kDataPerRepeat) * 2u;
        constexpr uint32_t kOffMax = kHalfM * kTileN;
        constexpr uint32_t kOffSum = kOffMax + kReduceFloats;
        constexpr uint32_t kOffBrcb = kOffSum + kReduceFloats;
        constexpr uint32_t kOffPreMax = kOffBrcb + kHalfM * 8u;
        constexpr uint32_t kOffPreSum = kOffPreMax + kHalfM;
        constexpr uint32_t kOffKeep = kOffPreSum + kHalfM;
        constexpr uint32_t kOffScratch = kOffKeep + kHalfM;
        constexpr uint32_t kOffDzBytes = (kOffScratch + 32u) * static_cast<uint32_t>(sizeof(float));
        constexpr uint32_t kUbBytes = kOffDzBytes + kHalfM * kTileN * static_cast<uint32_t>(sizeof(Element));

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> bufUb;
        pipe.InitBuffer(bufUb, kUbBytes);

        auto logits = bufUb.GetWithOffset<float>(kHalfM * kTileN, 0);
        auto maxTensor = bufUb.GetWithOffset<float>(kReduceFloats, kOffMax * sizeof(float));
        auto sumTensor = bufUb.GetWithOffset<float>(kReduceFloats, kOffSum * sizeof(float));
        auto brcbMax = bufUb.GetWithOffset<float>(kHalfM * 8u, kOffBrcb * sizeof(float));
        auto preMax = bufUb.GetWithOffset<float>(kHalfM, kOffPreMax * sizeof(float));
        auto preSum = bufUb.GetWithOffset<float>(kHalfM, kOffPreSum * sizeof(float));
        auto keepMax = bufUb.GetWithOffset<float>(kHalfM, kOffKeep * sizeof(float));
        auto scratch = bufUb.GetWithOffset<float>(32u, kOffScratch * sizeof(float));
        auto dzUb = bufUb.GetWithOffset<Element>(kHalfM * kTileN, kOffDzBytes);

        AscendC::GlobalTensor<Element> gmLogit;
        AscendC::GlobalTensor<Element> gmZ;
        AscendC::GlobalTensor<Element> gmDz;
        AscendC::GlobalTensor<float> gmDhAcc;
        AscendC::GlobalTensor<int64_t> gmLabel;
        AscendC::GlobalTensor<float> gmPartial;
        AscendC::GlobalTensor<float> gmState;
        AscendC::GlobalTensor<float> gmLoss;
        AscendC::GlobalTensor<float> gmMaxOut;
        AscendC::GlobalTensor<float> gmSumOut;
        AscendC::GlobalTensor<Element> gmDH;
        constexpr uint32_t kElemBytes = static_cast<uint32_t>(sizeof(Element));
        uint64_t floatBase = FloatBaseBytes(params.aicNum, kTileM, kElemBytes);
        gmLogit.SetGlobalBuffer((__gm__ Element*)params.workspace);
        gmZ.SetGlobalBuffer((__gm__ Element*)(params.workspace + ZFullBytesOffset(params.aicNum, params.h, kTileM,
                                                                                  kElemBytes, mTiles, nSplit)));
        gmDz.SetGlobalBuffer(
            (__gm__ Element*)(params.workspace + DzBytesOffset(params.aicNum, params.h, params.v, kElemBytes, kTileM,
                                                               mTiles, nSplit, params.rematZ, params.logitAsZ)));
        gmDhAcc.SetGlobalBuffer((__gm__ float*)(params.workspace + floatBase) +
                                DhAccOffsetFloats(params.aicNum, kTileM));
        gmLabel.SetGlobalBuffer((__gm__ int64_t*)params.labels);
        gmPartial.SetGlobalBuffer((__gm__ float*)(params.workspace + floatBase) +
                                  PartialOffsetFloats(params.aicNum, params.h, kTileM, mTiles, nSplit));
        gmState.SetGlobalBuffer((__gm__ float*)(params.workspace + floatBase) +
                                StatePartialOffsetFloats(params.aicNum, params.h, kTileM, mTiles, nSplit));
        gmLoss.SetGlobalBuffer((__gm__ float*)params.loss);
        gmMaxOut.SetGlobalBuffer((__gm__ float*)params.maxOut);
        gmSumOut.SetGlobalBuffer((__gm__ float*)params.sumOut);
        gmDH.SetGlobalBuffer((__gm__ Element*)params.dH);

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
                uint32_t m0 = mt * kTileM;
                uint32_t actualM = m0 + kTileM <= params.m ? kTileM : params.m - m0;
                // Ceil-split so M=1 lands on AIV0 (floor left AIV0 empty and flaked dH on ShortV).
                uint32_t half = (actualM + 1u) / 2u;
                uint32_t rowBegin = sub == 0 ? 0 : half;
                uint32_t rows = sub == 0 ? half : (actualM - half);
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
                    // GM labels: DataCopyPad → UB, then once-per-row Local GetValue (no GM GetValue).
                    AscendC::LocalTensor<int64_t> labUb = maxTensor.template ReinterpretCast<int64_t>();
                    AscendC::DataCopyPadExtParams<int64_t> labPad{false, 0, 0, static_cast<int64_t>(0)};
                    AscendC::DataCopyPad(labUb, gmLabel[m0 + rowBegin],
                                         {1, static_cast<uint32_t>(rows * sizeof(int64_t)), 0, 0, 0}, labPad);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
                    for (uint32_t i = 0; i < rows; ++i) {
                        lab[i] = labUb.GetValue(i);
                        valid[i] = lab[i] != params.ignoreIndex && lab[i] >= 0 &&
                                   lab[i] < static_cast<int64_t>(params.v);
                    }
                }

                uint32_t vPad = ZFullVPad(params.v);
                uint64_t zBase = (nSplit > 1u) ? static_cast<uint64_t>(mt) * kTileM * vPad :
                                                 static_cast<uint64_t>(core) * kTileM * vPad;
                uint32_t tileIter = 0;
                for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                    uint32_t stage = tileIter & (kStages - 1u);
                    // Latency batchZ: one Wait covers even+odd pair (AIC Sets only on odd/last).
                    if constexpr (kUseBatchZ) {
                        if ((tileIter & 1u) == 0u) {
                            AscendC::CrossCoreWaitFlag(kFlagAicDone);
                        }
                    } else {
                        AscendC::CrossCoreWaitFlag(kFlagAicDone);
                    }
                    uint32_t n0 = nt * kTileN;
                    uint32_t actualN = n0 + kTileN <= params.v ? kTileN : params.v - n0;
                    uint64_t slot = (static_cast<uint64_t>(core) * kStages + stage) * kTileM * kTileN;
                    if (rows > 0) {
                        // Checkpoint: prior StoreDz uses dzUb; drain before CopyLogits reuses it.
                        bool storeZ = (params.rematZ == 0u && params.logitAsZ == 0u);
                        if (storeZ && tileIter > 0u) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                        }
                        // Element → AivFree → StoreZ(MTE2_MTE3) → Cast → tgt from float logits.
                        // Read tgt after Cast: scalar Element→float (esp. bf16) is rejected by bisheng.
                        // StoreZ before Cast so MTE3 ∥ Cast (both only read dzUb).
                        // Element → AivFree → StoreZ ∥ Cast. NZ: CopyLogits does NZ2ND+Cast in one go.
                        if constexpr (kUseWsNz) {
                            CopyLogits(logits, dzUb, gmLogit[slot + WsRowOff(rowBegin)], rows, actualN);
                            // Always free AIC after logit load (incl. logitAsZ SingleN) so Pass2 Remat
                            // cannot race past unread Z on M=1 / ShortV.
                            AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(FlagAivFree(stage));
                            if (storeZ) {
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                                StoreDzTile(gmZ, dzUb, zBase, rowBegin, rows, n0, actualN, params.v);
                                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                            }
                        } else {
                            CopyLogitsElement(dzUb, gmLogit[slot + static_cast<uint64_t>(rowBegin) * kTileN], rows,
                                              actualN);
                            AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(FlagAivFree(stage));
                            if (storeZ) {
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                                StoreDzTile(gmZ, dzUb, zBase, rowBegin, rows, n0, actualN, params.v);
                                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                            }
                            CastLogitsFromElement(logits, dzUb, rows, actualN);
                        }
                        if (params.skipCe == 0u) {
                            // Only V_S+GetValue when this N-window holds a still-missing label
                            // (~1/nTiles tiles). Otherwise Cast→OnlineMax with no scalar bubble.
                            int64_t nBegin = static_cast<int64_t>(n0);
                            bool needTgt = false;
                            for (uint32_t i = 0; i < rows; ++i) {
                                if (!valid[i] || got[i]) {
                                    continue;
                                }
                                int64_t y = lab[i];
                                if (y >= nBegin && y < nBegin + static_cast<int64_t>(actualN)) {
                                    needTgt = true;
                                    break;
                                }
                            }
                            if (needTgt) {
                                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
                                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
                                for (uint32_t i = 0; i < rows; ++i) {
                                    if (!valid[i] || got[i]) {
                                        continue;
                                    }
                                    int64_t y = lab[i];
                                    if (y >= nBegin && y < nBegin + static_cast<int64_t>(actualN)) {
                                        tgt[i] = logits.GetValue(i * kTileN + static_cast<uint32_t>(y - nBegin));
                                        got[i] = true;
                                    }
                                }
                            }
                            if ((rows & 1u) != 0) {
                                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
                                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
                                UpdateOddRowAt(logits, rows - 1u, actualN, oddM, oddS, scratch, brcbMax);
                            }
                            if (rowsEven > 0) {
                                OnlineMax(logits, maxTensor, preMax, rowsEven);
                                AscendC::Adds(keepMax, maxTensor, 0.0f, rowsEven);
                                OnlineSum(logits, sumTensor, brcbMax, preMax, preSum, maxTensor, rowsEven);
                                AscendC::Adds(preMax, keepMax, 0.0f, rowsEven);
                                AscendC::Adds(preSum, sumTensor, 0.0f, rowsEven);
                            }
                        }
                    } else {
                        AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(FlagAivFree(stage));
                    }
                    ++tileIter;
                }
                if (params.rematZ == 0u && params.logitAsZ == 0u && rows > 0 && tileIter > 0u) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                }

                if (params.skipCe == 0u && rows > 0) {
                    if ((rows & 1u) != 0) {
                        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                        preMax.SetValue(rows - 1u, oddM);
                        preSum.SetValue(rows - 1u, oddS);
                    }
                }

                uint64_t dhBase = DhAccBaseFloats(nSplit > 1u ? mt : core, kTileM, params.h, nSplit, ni);
                // nSplit: publish state + zero each sibling's private dhAcc, then ONE SyncAll.
                // nSplit==1 (logitAsZ/ZFull Accu): zero dhAcc so AIC can always-AtomicAdd.
                if (nSplit > 1u) {
                    if (params.skipCe == 0u && rows > 0) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::DataCopyPad(gmState[static_cast<uint64_t>(core) * kTileM + rowBegin], preMax,
                                             {1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0});
                        AscendC::DataCopyPad(gmState[static_cast<uint64_t>(params.aicNum) * kTileM +
                                                     static_cast<uint64_t>(core) * kTileM + rowBegin],
                                             preSum, {1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0});
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    if (rows > 0) {
                        AscendC::Duplicate(logits, 0.0f, rows * kTileH);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        for (uint32_t ht = 0; ht < hTiles; ++ht) {
                            uint32_t h0 = ht * kTileH;
                            uint32_t actualH = h0 + kTileH <= params.h ? kTileH : params.h - h0;
                            StoreDhTile(gmDhAcc, logits, dhBase, rowBegin, rows, h0, actualH, params.h);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                    AscendC::SyncAll();
                } else if (params.rematZ == 0u && rows > 0) {
                    // Single dhAcc stripe (AtomicAdd accumulates N-tiles); zero once.
                    AscendC::Duplicate(logits, 0.0f, rows * kTileH);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    for (uint32_t ht = 0; ht < hTiles; ++ht) {
                        uint32_t h0 = ht * kTileH;
                        uint32_t actualH = h0 + kTileH <= params.h ? kTileH : params.h - h0;
                        StoreDhTile(gmDhAcc, logits, dhBase, rowBegin, rows, h0, actualH, params.h);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                if (params.skipCe == 0u && nSplit > 1u) {
                    if (rows > 0) {
                        // logits UB: [nSplit][kHalfM] max | [nSplit][kHalfM] sum (nSplit≤4, kHalfM≤64).
                        AscendC::LocalTensor<float> mBuf = logits;
                        AscendC::LocalTensor<float> sBuf = logits[nSplit * kHalfM];
                        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
                        for (uint32_t c = 0; c < nSplit; ++c) {
                            uint32_t sib = mt + c * mTiles;
                            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
                            AscendC::DataCopyPad(mBuf[c * kHalfM],
                                                 gmState[static_cast<uint64_t>(sib) * kTileM + rowBegin], cp, pad);
                            AscendC::DataCopyPad(sBuf[c * kHalfM],
                                                 gmState[static_cast<uint64_t>(params.aicNum) * kTileM +
                                                         static_cast<uint64_t>(sib) * kTileM + rowBegin],
                                                 cp, pad);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                        // Fold siblings with vector Max/Sub/Exp/Mul/Add (nSplit=2..4; no per-row Get/Set).
                        AscendC::LocalTensor<float> t0 = keepMax;
                        AscendC::LocalTensor<float> t1 = maxTensor;
                        AscendC::Adds(preMax, mBuf, 0.0f, rows);
                        AscendC::Adds(preSum, sBuf, 0.0f, rows);
                        AscendC::PipeBarrier<PIPE_V>();
                        for (uint32_t c = 1; c < nSplit; ++c) {
                            AscendC::Max(t1, preMax, mBuf[c * kHalfM], rows);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Sub(t0, preMax, t1, rows);
                            AscendC::Sub(preMax, mBuf[c * kHalfM], t1, rows);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Exp(t0, t0, rows);
                            AscendC::Exp(preMax, preMax, rows);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Mul(t0, preSum, t0, rows);
                            AscendC::Mul(preMax, sBuf[c * kHalfM], preMax, rows);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Add(preSum, t0, preMax, rows);
                            AscendC::Adds(preMax, t1, 0.0f, rows);
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                    }
                }

                if (params.skipCe == 0u && rows > 0) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    // Only one N-sibling writes max/sum outputs (P3: skipMaxSum skips GM dump).
                    if (params.skipMaxSum == 0u && (nSplit == 1u || ni == 0u)) {
                        AscendC::DataCopyPad(gmMaxOut[m0 + rowBegin], preMax,
                                             {1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0});
                        AscendC::DataCopyPad(gmSumOut[m0 + rowBegin], preSum,
                                             {1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0});
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    // Vector Ln + masked nll reduce (avoid per-row GetValue into acc).
                    AscendC::Ln(keepMax, preSum, rows);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(maxTensor, preMax, keepMax, rows);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate(sumTensor, 0.0f, rows);
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                    for (uint32_t i = 0; i < rows; ++i) {
                        if (valid[i] && got[i]) {
                            sumTensor.SetValue(i, tgt[i]);
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
                    AscendC::Sub(maxTensor, maxTensor, sumTensor, rows);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                    for (uint32_t i = 0; i < rows; ++i) {
                        if (!valid[i] || !got[i]) {
                            maxTensor.SetValue(i, 0.0f);
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
                    AscendC::WholeReduceSum(scratch, maxTensor, static_cast<uint64_t>(rows), 1, 1, 1, 8);
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
                    acc += scratch.GetValue(0);
                }

                // dhAcc already zeroed + SyncAll'd with nSplit merge barrier above.
                uint64_t dzCore = static_cast<uint64_t>(core) * kStages * kTileM * kTileN;
                uint32_t localTiles = 0;
                for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                    ++localTiles;
                }
                tileIter = 0;
                if (params.rematZ != 0u) {
                    // RematZ: Wait AIC remat into logit banks; no ZFull LoadZ.
                    for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                        uint32_t n0 = nt * kTileN;
                        uint32_t actualN = n0 + kTileN <= params.v ? kTileN : params.v - n0;
                        uint32_t bank = tileIter & 1u;
                        uint64_t dzSlot = dzCore + static_cast<uint64_t>(bank) * kTileM * kTileN;
                        uint64_t slot = (static_cast<uint64_t>(core) * kStages + bank) * kTileM * kTileN;
                        AscendC::CrossCoreWaitFlag(FlagRemat(bank));
                        if (rows > 0) {
                            CopyLogits(logits, dzUb, gmLogit[slot + WsRowOff(rowBegin)], rows, actualN);
                            BuildDz(logits, dzUb, preMax, preSum, brcbMax, keepMax, valid, lab, rows, actualN, n0,
                                    invAlpha);
                            StoreDz(gmDz, dzUb, logits, dzSlot + WsRowOff(rowBegin), rows, actualN);
                        }
                        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FlagAivFreeP2(bank));
                        // v30: drain Temp per tile (batched Wait overflowed FFTS when nTiles>16).
                        AscendC::CrossCoreWaitFlag(FlagTemp());
                        ++tileIter;
                    }
                } else if (params.logitAsZ != 0u) {
                    // Ping-pong: Wait AIC FlagRemat → BuildDz → Set FlagAivFreeP2 (same as RematZ).
                    for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                        uint32_t n0 = nt * kTileN;
                        uint32_t actualN = n0 + kTileN <= params.v ? kTileN : params.v - n0;
                        uint32_t bank = tileIter & (kStages - 1u);
                        uint64_t dzSlot = dzCore + static_cast<uint64_t>(bank) * kTileM * kTileN;
                        uint64_t slot = (static_cast<uint64_t>(core) * kStages + bank) * kTileM * kTileN;
                        AscendC::CrossCoreWaitFlag(FlagRemat(bank));
                        if (rows > 0) {
                            // Always reload logits: Pass1 CE/NLL clobbers UB (BuildDzFromExp was stale).
                            CopyLogits(logits, dzUb, gmLogit[slot + WsRowOff(rowBegin)], rows, actualN);
                            BuildDz(logits, dzUb, preMax, preSum, brcbMax, keepMax, valid, lab, rows, actualN, n0,
                                    invAlpha);
                            StoreDz(gmDz, dzUb, logits, dzSlot + WsRowOff(rowBegin), rows, actualN);
                        }
                        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FlagAivFreeP2(bank));
                        AscendC::CrossCoreWaitFlag(FlagTemp());
                        ++tileIter;
                    }
                } else {
                    // ZFull: same Remat-style ping-pong (no remat GEMM). ZFull is ND; dZ out NZ when kWsNz.
                    for (uint32_t nt = ni; nt < nTiles; nt += ntStep) {
                        uint32_t n0 = nt * kTileN;
                        uint32_t actualN = n0 + kTileN <= params.v ? kTileN : params.v - n0;
                        uint32_t bank = tileIter & (kStages - 1u);
                        uint64_t dzSlot = dzCore + static_cast<uint64_t>(bank) * kTileM * kTileN;
                        uint64_t slot = (static_cast<uint64_t>(core) * kStages + bank) * kTileM * kTileN;
                        (void)slot;
                        AscendC::CrossCoreWaitFlag(FlagRemat(bank));
                        if (rows > 0) {
                            LoadZAsLogits(logits, dzUb, gmZ, zBase, rowBegin, rows, n0, actualN, params.v);
                            // skipCe: reuse Element Z as fake dZ (same HBM traffic as BuildDz store).
                            if (params.skipCe == 0u) {
                                BuildDz(logits, dzUb, preMax, preSum, brcbMax, keepMax, valid, lab, rows, actualN, n0,
                                        invAlpha);
                            } else {
                                // skipCe path still stores Element Z as dZ — need V_MTE3 after any prior V.
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                            }
                            StoreDz(gmDz, dzUb, logits, dzSlot + WsRowOff(rowBegin), rows, actualN);
                        }
                        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FlagAivFreeP2(bank));
                        AscendC::CrossCoreWaitFlag(FlagTemp());
                        ++tileIter;
                    }
                }
                // FlagTemp drained per N-tile above (v30); avoid FFTS credit overflow.
                {
                    (void)localTiles;
                    // N-tile dH already AtomicAdd'd into stripe0 (no bank merge).
                    // One SyncAll for all cores (incl. idle); leader merges then casts — no extra barrier.
                    if (nSplit > 1u) {
                        AscendC::SyncAll();
                    }
                    if (rows > 0 && (nSplit == 1u || ni == 0u)) {
                        uint64_t dhDst = DhAccBaseFloats(nSplit > 1u ? mt : core, kTileM, params.h, nSplit, 0);
                        if (nSplit > 1u) {
                            // Leader si=0: merge sibling partials in ascending si (FP32 Add).
                            // Cap chunk so 2*rows*chunk fits in logits UB (odd TileM halves).
                            uint32_t kMergeH = (kHalfM * kTileN / 2u) / rows;
                            if (kMergeH == 0u) {
                                kMergeH = 1u;
                            }
                            if (kMergeH > kTileH / 2u) {
                                kMergeH = kTileH / 2u;
                            }
                            for (uint32_t si = 1u; si < nSplit; ++si) {
                                uint64_t dhSrc = DhAccBaseFloats(mt, kTileM, params.h, nSplit, si);
                                for (uint32_t ht = 0; ht < hTiles; ++ht) {
                                    uint32_t h0 = ht * kTileH;
                                    uint32_t actualH = h0 + kTileH <= params.h ? kTileH : params.h - h0;
                                    for (uint32_t hoff = 0; hoff < actualH; hoff += kMergeH) {
                                        uint32_t chunk = (hoff + kMergeH <= actualH) ? kMergeH : (actualH - hoff);
                                        uint32_t cols = rows * chunk;
                                        AscendC::LocalTensor<float> ubDst = logits;
                                        AscendC::LocalTensor<float> ubSrc = logits[cols];
                                        LoadDhTile(ubDst, gmDhAcc, dhDst, rowBegin, rows, h0 + hoff, chunk, params.h);
                                        LoadDhTile(ubSrc, gmDhAcc, dhSrc, rowBegin, rows, h0 + hoff, chunk, params.h);
                                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                                        AscendC::Add(ubDst, ubDst, ubSrc, cols);
                                        AscendC::PipeBarrier<PIPE_V>();
                                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                                        StoreDhTile(gmDhAcc, ubDst, dhDst, rowBegin, rows, h0 + hoff, chunk, params.h);
                                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                                    }
                                }
                            }
                        }
                        for (uint32_t ht = 0; ht < hTiles; ++ht) {
                            uint32_t h0 = ht * kTileH;
                            uint32_t actualH = h0 + kTileH <= params.h ? kTileH : params.h - h0;
                            LoadDhTile(logits, gmDhAcc, dhDst, rowBegin, rows, h0, actualH, params.h);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                            AscendC::Cast(dzUb, logits, AscendC::RoundMode::CAST_RINT, rows * kTileH);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                            StoreDhElem(gmDH, dzUb, static_cast<uint64_t>(m0 + rowBegin) * params.h + h0, rows, actualH,
                                        params.h);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                        }
                    }
                }
            }
        } else if (nSplit > 1u) {
            // Idle: match active — merge+clear folded into one SyncAll, then store barrier.
            AscendC::SyncAll(); // after Pass1 publish+clear
            AscendC::SyncAll(); // before dH store
        }

        AscendC::Duplicate(scratch, 0.0f, kPartialStride);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        // Deterministic loss: private partial + SyncAll + core0 fixed-index reduce (no AtomicAdd).
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

private:
    CHUNK_LOSS_DEVICE
    static void StoreDzTile(AscendC::GlobalTensor<Element> gmZ, AscendC::LocalTensor<Element>& ub, uint64_t zBase,
                            uint32_t rowBegin, uint32_t rows, uint32_t n0, uint32_t actualN, uint32_t /*v*/)
    {
        // Tile-major: [nTile][TileM][TileN] within stripe (r029 P2).
        uint32_t nt = n0 / kTileN;
        uint64_t tileBase = zBase + (static_cast<uint64_t>(nt) * kTileM + rowBegin) * kTileN;
        AscendC::GlobalTensor<Element> dst = gmZ[tileBase];
        if (actualN == kTileN) {
            AscendC::DataCopy(dst, ub, rows * kTileN);
            return;
        }
        bool srcAligned = ((kTileN - actualN) * sizeof(Element)) % 32u == 0;
        if (srcAligned) {
            // UB→GM: srcStride=UB gap (32B blocks), dstStride=GM gap (bytes) within TileN-pitched rows.
            uint32_t srcGapBlocks = ((kTileN - actualN) * sizeof(Element)) / 32u;
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows),
                                              static_cast<uint32_t>(actualN * sizeof(Element)), srcGapBlocks,
                                              static_cast<uint32_t>((kTileN - actualN) * sizeof(Element)), 0};
            AscendC::DataCopyPad(dst, ub, params);
            return;
        }
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopyPad(dst[static_cast<uint64_t>(r) * kTileN], ub[r * kTileN],
                                 {1, static_cast<uint32_t>(actualN * sizeof(Element)), 0, 0, 0});
        }
    }

    CHUNK_LOSS_DEVICE
    static void LoadZAsLogits(AscendC::LocalTensor<float>& logits, AscendC::LocalTensor<Element>& dzUb,
                              AscendC::GlobalTensor<Element> gmZ, uint64_t zBase, uint32_t rowBegin, uint32_t rows,
                              uint32_t n0, uint32_t actualN, uint32_t /*v*/)
    {
        // Prior BuildDz Exp mutates logits on PIPE_V; drain before MTE2 refill.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        uint32_t nt = n0 / kTileN;
        uint64_t tileBase = zBase + (static_cast<uint64_t>(nt) * kTileM + rowBegin) * kTileN;
        AscendC::GlobalTensor<Element> src = gmZ[tileBase];
        if (actualN == kTileN) {
            AscendC::DataCopy(dzUb, src, rows * kTileN);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(logits, dzUb, AscendC::RoundMode::CAST_NONE, rows * kTileN);
            return;
        }
        const Element kZero = ElementZero<Element>();
        AscendC::Duplicate(dzUb, kZero, rows * kTileN);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        bool dstAligned = ((kTileN - actualN) * sizeof(Element)) % 32u == 0;
        if (dstAligned) {
            // GM→UB: srcStride=GM gap (bytes) across TileN-pitched rows, dstStride=UB gap (32B).
            uint32_t dstGapBlocks = ((kTileN - actualN) * sizeof(Element)) / 32u;
            AscendC::DataCopyExtParams params{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(actualN * sizeof(Element)),
                static_cast<uint32_t>((kTileN - actualN) * sizeof(Element)), dstGapBlocks, 0};
            AscendC::DataCopyPadExtParams<Element> pad{false, 0, 0, kZero};
            AscendC::DataCopyPad(dzUb, src, params, pad);
        } else {
            for (uint32_t r = 0; r < rows; ++r) {
                AscendC::DataCopyPad(dzUb[r * kTileN], src[static_cast<uint64_t>(r) * kTileN],
                                     {1, static_cast<uint32_t>(actualN * sizeof(Element)), 0, 0, 0},
                                     {false, 0, 0, kZero});
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(logits, dzUb, AscendC::RoundMode::CAST_NONE, rows * kTileN);
        AscendC::PipeBarrier<PIPE_V>();
        // BuildDz Exp spans TileN — pad cols must be kPadLogit, not 0.
        FillPadLogit(logits, rows, actualN);
    }

    CHUNK_LOSS_DEVICE
    static void LoadDhTile(AscendC::LocalTensor<float>& ub, AscendC::GlobalTensor<float> gmDhAcc, uint64_t dhBase,
                           uint32_t rowBegin, uint32_t rows, uint32_t h0, uint32_t actualH, uint32_t h)
    {
        AscendC::GlobalTensor<float> src = gmDhAcc[dhBase + static_cast<uint64_t>(rowBegin) * h + h0];
        if (actualH == kTileH && h == kTileH) {
            AscendC::DataCopy(ub, src, rows * kTileH);
            return;
        }
        uint32_t dstGapBlocks = ((kTileH - actualH) * sizeof(float)) / 32u;
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(actualH * sizeof(float)),
                                          static_cast<uint32_t>((h - actualH) * sizeof(float)), dstGapBlocks, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(ub, src, params, pad);
    }

    CHUNK_LOSS_DEVICE
    static void StoreDhTile(AscendC::GlobalTensor<float> gmDhAcc, AscendC::LocalTensor<float>& ub, uint64_t dhBase,
                            uint32_t rowBegin, uint32_t rows, uint32_t h0, uint32_t actualH, uint32_t h)
    {
        AscendC::GlobalTensor<float> dst = gmDhAcc[dhBase + static_cast<uint64_t>(rowBegin) * h + h0];
        if (actualH == kTileH && h == kTileH) {
            AscendC::DataCopy(dst, ub, rows * kTileH);
            return;
        }
        uint32_t srcGapBlocks = ((kTileH - actualH) * sizeof(float)) / 32u;
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(actualH * sizeof(float)),
                                          srcGapBlocks, static_cast<uint32_t>((h - actualH) * sizeof(float)), 0};
        AscendC::DataCopyPad(dst, ub, params);
    }

    CHUNK_LOSS_DEVICE
    static void StoreDhElem(AscendC::GlobalTensor<Element> gmDH, AscendC::LocalTensor<Element>& ub, uint64_t gmOffset,
                            uint32_t rows, uint32_t actualH, uint32_t h)
    {
        AscendC::GlobalTensor<Element> dst = gmDH[gmOffset];
        if (actualH == kTileH && h == kTileH) {
            AscendC::DataCopy(dst, ub, rows * kTileH);
            return;
        }
        bool srcAligned = ((kTileH - actualH) * sizeof(Element)) % 32u == 0;
        if (srcAligned) {
            uint32_t srcGapBlocks = ((kTileH - actualH) * sizeof(Element)) / 32u;
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows),
                                              static_cast<uint32_t>(actualH * sizeof(Element)), srcGapBlocks,
                                              static_cast<uint32_t>((h - actualH) * sizeof(Element)), 0};
            AscendC::DataCopyPad(dst, ub, params);
            return;
        }
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopyPad(dst[static_cast<uint64_t>(r) * h], ub[r * kTileH],
                                 {1, static_cast<uint32_t>(actualH * sizeof(Element)), 0, 0, 0});
        }
    }

    // FAG-style: MM NZ GM → UB (fractal rows). dstStride=1 inserts 1-block gaps for NZ2ND.
    // UB capacity must be ≥ rows*s2 + (s2/C0-1)*kElePerBlk — use logits Element view, not dzUb.
    CHUNK_LOSS_DEVICE
    static void NzCopyIn(AscendC::LocalTensor<Element>& dst, AscendC::GlobalTensor<Element> src, uint32_t s1,
                         uint32_t s2Align, uint32_t s1InnerAlign)
    {
        AscendC::DataCopyParams p;
        p.blockCount = static_cast<uint16_t>(s2Align / kC0Size);
        p.blockLen = static_cast<uint16_t>(s1 * kC0Size / kElePerBlk);
        p.srcStride = static_cast<uint16_t>(s1InnerAlign * kC0Size / kElePerBlk - p.blockLen);
        p.dstStride = 1;
        AscendC::DataCopy(dst, src, p);
    }

    // FAG-style: UB NZ (with gaps) → GM NZ (tight). srcStride=1 skips UB gaps.
    CHUNK_LOSS_DEVICE
    static void NzCopyOut(AscendC::GlobalTensor<Element> dst, AscendC::LocalTensor<Element>& src, uint32_t s1,
                          uint32_t s2Align, uint32_t s1InnerAlign)
    {
        AscendC::DataCopyParams p;
        p.blockCount = static_cast<uint16_t>(s2Align / kC0Size);
        p.blockLen = static_cast<uint16_t>(s1 * kC0Size / kElePerBlk);
        p.srcStride = 1;
        p.dstStride = static_cast<uint16_t>(s1InnerAlign * kC0Size / kElePerBlk - p.blockLen);
        AscendC::DataCopy(dst, src, p);
    }

    // NZ UB → ND UB (FAG NZ2ND). src=tmp(NZ), dst=out(ND).
    CHUNK_LOSS_DEVICE
    static void Nz2Nd(AscendC::LocalTensor<Element>& dstNd, AscendC::LocalTensor<Element>& srcNz, uint32_t s1,
                      uint32_t s2Align)
    {
        AscendC::CopyRepeatParams rp;
        rp.srcStride = static_cast<uint16_t>(s1 * kC0Size / kElePerBlk + 1);
        rp.dstStride = static_cast<uint16_t>(kC0Size / kElePerBlk);
        rp.srcRepeatSize = static_cast<uint16_t>(kC0Size / kElePerBlk);
        rp.dstRepeatSize = static_cast<uint16_t>(s2Align / kElePerBlk);
        uint16_t c0Repeat = static_cast<uint16_t>(kC0Size / kElePerBlk);
        uint16_t c1Repeat = static_cast<uint16_t>(s2Align / kC0Size / kVecRepeat);
        uint16_t c1Remain = static_cast<uint16_t>(s2Align / kC0Size % kVecRepeat);
        for (uint16_t i = 0; i < c0Repeat; ++i) {
            for (uint16_t j = 0; j < c1Repeat; ++j) {
                AscendC::Copy(dstNd[i * kElePerBlk + j * kVecRepeat * kC0Size],
                              srcNz[i * kElePerBlk + j * kVecRepeat * (s1 * kC0Size + kElePerBlk)],
                              static_cast<uint64_t>(kVecRepeat * kElePerBlk), static_cast<uint16_t>(s1), rp);
            }
            if (c1Remain > 0) {
                AscendC::Copy(dstNd[i * kElePerBlk + c1Repeat * kVecRepeat * kC0Size],
                              srcNz[i * kElePerBlk + c1Repeat * kVecRepeat * (s1 * kC0Size + kElePerBlk)],
                              static_cast<uint64_t>(kVecRepeat * c1Remain), static_cast<uint16_t>(s1), rp);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ND UB → NZ UB (FAG ND2NZ). dst=nz, src=nd.
    CHUNK_LOSS_DEVICE
    static void Nd2Nz(AscendC::LocalTensor<Element>& dstNz, AscendC::LocalTensor<Element>& srcNd, uint32_t s1,
                      uint32_t s2Align)
    {
        AscendC::CopyRepeatParams rp;
        rp.dstStride = static_cast<uint16_t>(s1 * kC0Size / kElePerBlk + 1);
        rp.srcStride = static_cast<uint16_t>(kC0Size / kElePerBlk);
        rp.dstRepeatSize = static_cast<uint16_t>(kC0Size / kElePerBlk);
        rp.srcRepeatSize = static_cast<uint16_t>(s2Align / kElePerBlk);
        uint16_t c1Repeat = static_cast<uint16_t>(s2Align / kC0Size / kVecRepeat);
        uint16_t c1Remain = static_cast<uint16_t>(s2Align / kC0Size % kVecRepeat);
        // Element is 2B (fp16/bf16); reuse half view for Copy width like FAG.
        auto dstH = dstNz.template ReinterpretCast<half>();
        auto srcH = srcNd.template ReinterpretCast<half>();
        for (uint16_t j = 0; j < c1Repeat; ++j) {
            AscendC::Copy(dstH[j * 8 * (s1 + 1) * kC0Size], srcH[j * 128],
                          static_cast<uint64_t>(kVecRepeat * kElePerBlk), static_cast<uint16_t>(s1), rp);
        }
        if (c1Remain > 0) {
            AscendC::Copy(dstH[c1Repeat * 8 * (s1 + 1) * kC0Size], srcH[c1Repeat * 128],
                          static_cast<uint64_t>(kElePerBlk * c1Remain), static_cast<uint16_t>(s1), rp);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    CHUNK_LOSS_DEVICE
    static uint32_t NzAlignN(uint32_t actualN) { return (actualN + kC0Size - 1u) / kC0Size * kC0Size; }

    // GM → Element UB only (no Cast). ND staging path (kWsNz=false).
    CHUNK_LOSS_DEVICE
    static void CopyLogitsElementNd(AscendC::LocalTensor<Element>& dzUb, AscendC::GlobalTensor<Element> gmRows,
                                    uint32_t rows, uint32_t actualN)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        if (actualN == kTileN) {
            AscendC::DataCopy(dzUb, gmRows, rows * kTileN);
            return;
        }
        const Element kZero = ElementZero<Element>();
        AscendC::Duplicate(dzUb, kZero, rows * kTileN);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        bool dstAligned = ((kTileN - actualN) * sizeof(Element)) % 32u == 0;
        if (dstAligned) {
            uint32_t dstGapBlocks = ((kTileN - actualN) * sizeof(Element)) / 32u;
            AscendC::DataCopyExtParams params{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(actualN * sizeof(Element)),
                static_cast<uint32_t>((kTileN - actualN) * sizeof(Element)), dstGapBlocks, 0};
            AscendC::DataCopyPadExtParams<Element> pad{false, 0, 0, kZero};
            AscendC::DataCopyPad(dzUb, gmRows, params, pad);
        } else {
            for (uint32_t r = 0; r < rows; ++r) {
                AscendC::DataCopyPad(dzUb[r * kTileN], gmRows[r * kTileN],
                                     {1, static_cast<uint32_t>(actualN * sizeof(Element)), 0, 0, 0},
                                     {false, 0, 0, kZero});
            }
        }
    }

    CHUNK_LOSS_DEVICE
    static void CopyLogitsElement(AscendC::LocalTensor<Element>& dzUb, AscendC::GlobalTensor<Element> gmRows,
                                  uint32_t rows, uint32_t actualN)
    {
        // ND-only helper (Pass1 CE may call without float tmp). NZ uses CopyLogits.
        CopyLogitsElementNd(dzUb, gmRows, rows, actualN);
    }

    CHUNK_LOSS_DEVICE
    static void CastLogitsFromElement(AscendC::LocalTensor<float>& logits, AscendC::LocalTensor<Element>& dzUb,
                                      uint32_t rows, uint32_t actualN)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (actualN == kTileN) {
            AscendC::Cast(logits, dzUb, AscendC::RoundMode::CAST_NONE, rows * kTileN);
            return;
        }
        AscendC::Duplicate(logits, kPadLogit, rows * kTileN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(logits, dzUb, AscendC::RoundMode::CAST_NONE, rows * kTileN);
        AscendC::PipeBarrier<PIPE_V>();
        FillPadLogit(logits, rows, actualN);
    }

    CHUNK_LOSS_DEVICE
    static void CopyLogits(AscendC::LocalTensor<float>& logits, AscendC::LocalTensor<Element>& dzUb,
                           AscendC::GlobalTensor<Element> gmRows, uint32_t rows, uint32_t actualN)
    {
        if constexpr (kUseWsNz) {
            // FAG: NZCopyIn → logits(tmp, fits gaps) → NZ2ND → dzUb(ND) → Cast float.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            uint32_t s2 = (actualN >= kTileN) ? kTileN : NzAlignN(actualN);
            auto tmpNz = logits.template ReinterpretCast<Element>();
            NzCopyIn(tmpNz, gmRows, rows, s2, kTileM);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            Nz2Nd(dzUb, tmpNz, rows, s2);
            // Do not Duplicate Element pad at actualN (V=300 → actualN=44 misaligned → 507015).
            // Cast full tile then FillPadLogit (float path handles actualN%8).
            AscendC::Cast(logits, dzUb, AscendC::RoundMode::CAST_NONE, rows * kTileN);
            if (actualN < kTileN) {
                AscendC::PipeBarrier<PIPE_V>();
                FillPadLogit(logits, rows, actualN);
            }
            return;
        }
        CopyLogitsElementNd(dzUb, gmRows, rows, actualN);
        CastLogitsFromElement(logits, dzUb, rows, actualN);
    }

    // Pad cols with kPadLogit. Prefer Duplicate on 32B-aligned float offsets.
    // Misaligned actualN: ≤7 scalar fills to next align8, then Duplicate the rest (avoid full-row SetValue).
    CHUNK_LOSS_DEVICE
    static void FillPadLogit(AscendC::LocalTensor<float>& logits, uint32_t rows, uint32_t actualN)
    {
        if (actualN >= kTileN) {
            return;
        }
        uint32_t align0 = (actualN + 7u) & ~7u;
        if (align0 > kTileN) {
            align0 = kTileN;
        }
        if (align0 > actualN) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t c = actualN; c < align0; ++c) {
                    logits.SetValue(r * kTileN + c, kPadLogit);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        }
        if (align0 < kTileN) {
            uint32_t pad = kTileN - align0;
            for (uint32_t r = 0; r < rows; ++r) {
                AscendC::Duplicate(logits[r * kTileN + align0], kPadLogit, pad);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Zero pad cols (BuildDz). Same hybrid align rule as FillPadLogit.
    // Caller must have drained PIPE_V→S before entry when using the scalar head path.
    CHUNK_LOSS_DEVICE
    static void FillPadZero(AscendC::LocalTensor<float>& logits, uint32_t rows, uint32_t actualN, bool const* valid)
    {
        if (actualN >= kTileN) {
            return;
        }
        uint32_t align0 = (actualN + 7u) & ~7u;
        if (align0 > kTileN) {
            align0 = kTileN;
        }
        if (align0 > actualN) {
            for (uint32_t r = 0; r < rows; ++r) {
                if (!valid[r]) {
                    continue;
                }
                for (uint32_t c = actualN; c < align0; ++c) {
                    logits.SetValue(r * kTileN + c, 0.0f);
                }
            }
        }
        if (align0 < kTileN) {
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
            uint32_t pad = kTileN - align0;
            for (uint32_t r = 0; r < rows; ++r) {
                if (!valid[r]) {
                    continue;
                }
                AscendC::Duplicate(logits[r * kTileN + align0], 0.0f, pad);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Scalar online max/sum over [0, actualN). Must not mutate logits (Pass2 reloads Z).
    // Vector WholeReduce* in-place Exp would destroy the odd row → keep GetValue+Exp(scratch).
    CHUNK_LOSS_DEVICE
    static void UpdateOddRowAt(AscendC::LocalTensor<float>& logits, uint32_t row, uint32_t actualN, float& oddM,
                               float& oddS, AscendC::LocalTensor<float>& scratch, AscendC::LocalTensor<float>& brcbMax)
    {
        (void)brcbMax;
        float tileMax = kPadLogit;
        for (uint32_t c = 0; c < actualN; ++c) {
            float z = logits.GetValue(row * kTileN + c);
            if (z > tileMax) {
                tileMax = z;
            }
        }
        float newMax = oddM > tileMax ? oddM : tileMax;
        float tileSum = 0.0f;
        for (uint32_t c = 0; c < actualN; c += 8) {
            uint32_t nElt = actualN - c > 8 ? 8 : actualN - c;
            for (uint32_t i = 0; i < 8; ++i) {
                float z = i < nElt ? logits.GetValue(row * kTileN + c + i) : newMax - 80.0f;
                scratch.SetValue(i, z - newMax);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
            AscendC::Exp(scratch, scratch, 8);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
            for (uint32_t i = 0; i < nElt; ++i) {
                tileSum += scratch.GetValue(i);
            }
        }
        float scale = 0.0f;
        if (oddS != 0.0f) {
            scratch.SetValue(0, oddM - newMax);
            for (uint32_t i = 1; i < 8; ++i) {
                scratch.SetValue(i, -80.0f);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
            AscendC::Exp(scratch, scratch, 8);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
            scale = scratch.GetValue(0);
        }
        oddS = oddS * scale + tileSum;
        oddM = newMax;
    }

    // logits already hold exp(z-m) from OnlineSum (SingleN even-row full tile).
    // brcbMax/keepMax: same Brcb-scale scratch as BuildDz (caller passes reduce temps).
    CHUNK_LOSS_DEVICE
    static void BuildDzFromExp(AscendC::LocalTensor<float>& logits, AscendC::LocalTensor<Element>& dzUb,
                               AscendC::LocalTensor<float>& preSum, AscendC::LocalTensor<float>& brcbMax,
                               AscendC::LocalTensor<float>& keepMax, bool const* valid, int64_t const* lab,
                               uint32_t rows, uint32_t actualN, uint32_t n0, float invAlpha)
    {
        PackDzRowScale(keepMax, preSum, valid, rows, invAlpha);
        uint8_t repeatSub = static_cast<uint8_t>(rows);
        AscendC::BinaryRepeatParams rp{1, 1, 0, 32, 32, 1};
        AscendC::Brcb(brcbMax, keepMax, static_cast<uint8_t>((rows + 7) / 8), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(logits, logits, brcbMax, static_cast<uint64_t>(kDataPerRepeat), repeatSub, rp);
        AscendC::Mul(logits[kDataPerRepeat], logits[kDataPerRepeat], brcbMax, static_cast<uint64_t>(kDataPerRepeat),
                     repeatSub, rp);
        AscendC::Mul(logits[kDataPerRepeat * 2], logits[kDataPerRepeat * 2], brcbMax,
                     static_cast<uint64_t>(kDataPerRepeat), repeatSub, rp);
        AscendC::Mul(logits[kDataPerRepeat * 3], logits[kDataPerRepeat * 3], brcbMax,
                     static_cast<uint64_t>(kDataPerRepeat), repeatSub, rp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        // Sparse onehot — ≤1 col/row; Adds(count=1) needs 32B align → keep Get/SetValue.
        for (uint32_t r = 0; r < rows; ++r) {
            if (!valid[r]) {
                continue;
            }
            int64_t y = lab[r];
            int64_t nBegin = static_cast<int64_t>(n0);
            if (y >= nBegin && y < nBegin + static_cast<int64_t>(actualN)) {
                uint32_t col = static_cast<uint32_t>(y - nBegin);
                float v = logits.GetValue(r * kTileN + col);
                logits.SetValue(r * kTileN + col, v - invAlpha);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
        AscendC::Cast(dzUb, logits, AscendC::RoundMode::CAST_RINT, rows * kTileN);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }

    // keepMax[r] = valid[r] ? invAlpha / preSum[r] : 0  via vector Div + sparse clear.
    CHUNK_LOSS_DEVICE
    static void PackDzRowScale(AscendC::LocalTensor<float>& keepMax, AscendC::LocalTensor<float>& preSum,
                               bool const* valid, uint32_t rows, float invAlpha)
    {
        AscendC::Duplicate(keepMax, invAlpha, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(keepMax, keepMax, preSum, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        for (uint32_t r = 0; r < rows; ++r) {
            if (!valid[r]) {
                keepMax.SetValue(r, 0.0f);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
    }

    CHUNK_LOSS_DEVICE
    static void BuildDz(AscendC::LocalTensor<float>& logits, AscendC::LocalTensor<Element>& dzUb,
                        AscendC::LocalTensor<float>& preMax, AscendC::LocalTensor<float>& preSum,
                        AscendC::LocalTensor<float>& brcbMax, AscendC::LocalTensor<float>& keepMax, bool const* valid,
                        int64_t const* lab, uint32_t rows, uint32_t actualN, uint32_t n0, float invAlpha)
    {
        // dZ = alpha * ((exp(z - m) / s) - onehot)  on valid rows; else 0.
        AscendC::Brcb(brcbMax, preMax, static_cast<uint8_t>((rows + 7) / 8), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        uint8_t repeatSub = static_cast<uint8_t>(rows);
        AscendC::BinaryRepeatParams rp{1, 1, 0, 32, 32, 1};
        AscendC::Sub(logits, logits, brcbMax, static_cast<uint64_t>(kDataPerRepeat), repeatSub, rp);
        AscendC::Sub(logits[kDataPerRepeat], logits[kDataPerRepeat], brcbMax, static_cast<uint64_t>(kDataPerRepeat),
                     repeatSub, rp);
        AscendC::Sub(logits[kDataPerRepeat * 2], logits[kDataPerRepeat * 2], brcbMax,
                     static_cast<uint64_t>(kDataPerRepeat), repeatSub, rp);
        AscendC::Sub(logits[kDataPerRepeat * 3], logits[kDataPerRepeat * 3], brcbMax,
                     static_cast<uint64_t>(kDataPerRepeat), repeatSub, rp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(logits, logits, rows * kTileN);
        AscendC::PipeBarrier<PIPE_V>();
        PackDzRowScale(keepMax, preSum, valid, rows, invAlpha);
        AscendC::Brcb(brcbMax, keepMax, static_cast<uint8_t>((rows + 7) / 8), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(logits, logits, brcbMax, static_cast<uint64_t>(kDataPerRepeat), repeatSub, rp);
        AscendC::Mul(logits[kDataPerRepeat], logits[kDataPerRepeat], brcbMax, static_cast<uint64_t>(kDataPerRepeat),
                     repeatSub, rp);
        AscendC::Mul(logits[kDataPerRepeat * 2], logits[kDataPerRepeat * 2], brcbMax,
                     static_cast<uint64_t>(kDataPerRepeat), repeatSub, rp);
        AscendC::Mul(logits[kDataPerRepeat * 3], logits[kDataPerRepeat * 3], brcbMax,
                     static_cast<uint64_t>(kDataPerRepeat), repeatSub, rp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        // Pad zero: hybrid ≤7 SetValue then Duplicate (misaligned full Duplicate → 507015).
        FillPadZero(logits, rows, actualN, valid);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        // Sparse onehot — ≤1 col/row; Adds(count=1) needs 32B align → keep Get/SetValue.
        for (uint32_t r = 0; r < rows; ++r) {
            if (!valid[r]) {
                continue;
            }
            int64_t y = lab[r];
            if (y >= static_cast<int64_t>(n0) && y < static_cast<int64_t>(n0 + actualN)) {
                uint32_t col = static_cast<uint32_t>(y - static_cast<int64_t>(n0));
                logits.SetValue(r * kTileN + col, logits.GetValue(r * kTileN + col) - invAlpha);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Cast(dzUb, logits, AscendC::RoundMode::CAST_RINT, rows * kTileN);
        // Hand off to MTE3 store: caller WaitFlag(V_MTE3) then DataCopy (cast∥prep overlap).
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }

    CHUNK_LOSS_DEVICE
    static void StoreDz(AscendC::GlobalTensor<Element>& gmDz, AscendC::LocalTensor<Element>& dzUb,
                        AscendC::LocalTensor<float>& logitsTmp, uint64_t dzOff, uint32_t rows, uint32_t actualN)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        if constexpr (kUseWsNz) {
            // ND in dzUb → ND2NZ into logits tmp (fits gaps) → NzCopyOut strips gaps to GM.
            uint32_t s2 = (actualN >= kTileN) ? kTileN : NzAlignN(actualN);
            auto tmpNz = logitsTmp.template ReinterpretCast<Element>();
            Nd2Nz(tmpNz, dzUb, rows, s2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            NzCopyOut(gmDz[dzOff], tmpNz, rows, s2, kTileM);
        } else {
            (void)logitsTmp;
            (void)actualN;
            uint32_t dzCount = ((rows * kTileN * sizeof(Element) + 31u) / 32u) * 32u / sizeof(Element);
            AscendC::DataCopy(gmDz[dzOff], dzUb, dzCount);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    CHUNK_LOSS_DEVICE
    static void OnlineMax(AscendC::LocalTensor<float>& logits, AscendC::LocalTensor<float>& maxTensor,
                          AscendC::LocalTensor<float>& preMax, uint32_t rows)
    {
        int32_t repeatTimes = static_cast<int32_t>(rows * kTileN / kDataPerRepeat);
        if (repeatTimes > kMaxReduceRepeat) {
            int32_t first = (repeatTimes / 2 + 7) / 8 * 8;
            int32_t second = repeatTimes - first;
            AscendC::WholeReduceMax(maxTensor, logits, static_cast<uint64_t>(kDataPerRepeat), first, 1, 1, 8);
            AscendC::WholeReduceMax(maxTensor[first * 2], logits[first * kDataPerRepeat],
                                    static_cast<uint64_t>(kDataPerRepeat), second, 1, 1, 8);
        } else {
            AscendC::WholeReduceMax(maxTensor, logits, static_cast<uint64_t>(kDataPerRepeat), repeatTimes, 1, 1, 8);
        }
        AscendC::PipeBarrier<PIPE_V>();
        uint8_t dupRepeat = static_cast<uint8_t>((rows * 8 + kDataPerRepeat - 1) / kDataPerRepeat);
        uint64_t dupMask[2] = {0b1010101010101010101010101010101010101010101010101010101010101010u, 0};
        AscendC::Duplicate(maxTensor.template ReinterpretCast<int32_t>(), kNegInfBits, dupMask, dupRepeat, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax(maxTensor, maxTensor, dupRepeat, static_cast<uint64_t>(kDataPerRepeat), 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(maxTensor, maxTensor, preMax, rows);
    }

    CHUNK_LOSS_DEVICE
    static void OnlineSum(AscendC::LocalTensor<float>& logits, AscendC::LocalTensor<float>& sumTensor,
                          AscendC::LocalTensor<float>& brcbMax, AscendC::LocalTensor<float>& preMax,
                          AscendC::LocalTensor<float>& preSum, AscendC::LocalTensor<float>& maxTensor, uint32_t rows)
    {
        uint8_t brcbRepeat = static_cast<uint8_t>((rows + 7) / 8);
        AscendC::Brcb(brcbMax, maxTensor, brcbRepeat, {1, 8});
        AscendC::Sub(preMax, preMax, maxTensor, rows);
        // Zero sum early: overlaps Brcb/Sub(preMax); drops a later PipeBarrier.
        AscendC::Duplicate(sumTensor, 0.0f, rows * 8);
        AscendC::PipeBarrier<PIPE_V>();
        uint8_t repeatSub = static_cast<uint8_t>(rows);
        AscendC::BinaryRepeatParams repeatParams{1, 1, 0, 32, 32, 1};
        AscendC::Sub(logits, logits, brcbMax, static_cast<uint64_t>(kDataPerRepeat), repeatSub, repeatParams);
        AscendC::Sub(logits[kDataPerRepeat], logits[kDataPerRepeat], brcbMax, static_cast<uint64_t>(kDataPerRepeat),
                     repeatSub, repeatParams);
        AscendC::Sub(logits[kDataPerRepeat * 2], logits[kDataPerRepeat * 2], brcbMax,
                     static_cast<uint64_t>(kDataPerRepeat), repeatSub, repeatParams);
        AscendC::Sub(logits[kDataPerRepeat * 3], logits[kDataPerRepeat * 3], brcbMax,
                     static_cast<uint64_t>(kDataPerRepeat), repeatSub, repeatParams);
        // Exp(preMax) ∥ Exp(logits) — independent dst; Mul(preSum) only needs preMax.
        AscendC::Exp(preMax, preMax, rows);
        AscendC::Exp(logits, logits, rows * kTileN);
        AscendC::Mul(preSum, preSum, preMax, rows);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t halfRepeat = static_cast<int32_t>(rows * 2);
        AscendC::WholeReduceSum(sumTensor, logits, static_cast<uint64_t>(kDataPerRepeat), halfRepeat, 2, 1, 8);
        AscendC::WholeReduceSum(sumTensor[halfRepeat * 2], logits[halfRepeat * kDataPerRepeat],
                                static_cast<uint64_t>(kDataPerRepeat), halfRepeat, 2, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(sumTensor, sumTensor, static_cast<uint64_t>(8), static_cast<int32_t>(rows), 1, 1, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(sumTensor, preSum, sumTensor, rows);
    }
};

} // namespace chunkloss

#endif
