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
 * ChunkLoss arch22 common: constants, workspace, scenario, Params, CV flags.
 */
#ifndef CHUNK_LOSS_COMMON_ARCH22_H_
#define CHUNK_LOSS_COMMON_ARCH22_H_

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

// Device/host macros (AscendC; no Catlass)
#ifndef CHUNK_LOSS_DEVICE
#define CHUNK_LOSS_DEVICE __forceinline__ __aicore__
#endif
#ifdef __CCE__
#ifndef CHUNK_LOSS_HOST_DEVICE
#define CHUNK_LOSS_HOST_DEVICE __forceinline__[host, aicore]
#endif
#else
#ifndef CHUNK_LOSS_HOST_DEVICE
#define CHUNK_LOSS_HOST_DEVICE inline
#endif
#endif
#ifndef CHUNK_LOSS_GLOBAL
#define CHUNK_LOSS_GLOBAL __global__ __aicore__
#endif

namespace chunkloss {
constexpr uint32_t kTileMDefault = 128;
constexpr uint32_t kTileN = 256;
constexpr uint32_t kTileH = 256;
constexpr uint32_t kStages = 2; // Pass1 logit GM + Pass2 dZ GM double-buffer depth
// CV workspace NZ (FAG-style): Cube Fixpipe CFG_NZ → logit/dZ GM; Vec NZCopyIn+NZ2ND / ND2NZ.
// Public tensors remain FORMAT_ND; only CV workspace (logit/dZ ping-pong) uses NZ.
// ND workspace for hand cube (Accu Actuals match catlass baseline; NZ Fixpipe left for later).
constexpr bool kWsNzDefault = false;
constexpr uint32_t kC0Size = 16; // fractal C0 for fp16/bf16
// MARKER_FLAG_P2_v26
// r030: kStages=4 A/B — #10 0.087→0.099 (logitAsZ) / 0.099 (ZFull) ROLLBACK.
// round_005 (catlass FullLoadA): Pass2 dH keeps dZ resident in L1 across H tiles (needLoadL1).
// round_008: AIC Fixpipe → dHAcc; same-core multi-nt uses single-writer AtomicAdd (deterministic).
// Deterministic redesign: forbid multi-writer Atomic (shared nSplit stripe / atomicLoss).
// round_016: adaptive TileM (32/64/128) when M underfills AIC; keep 128 for target shape.
// round_023: drop dead TempDh WS; pack AIV UB into one TBuf (logits|reduce|state|dz).
// round_024: Pass1 staging fp16 (cut float logit GM MTE; online matches Z checkpoint).
// round_025: Fixpipe→Z direct — ROLLBACK (strided Fixpipe slower; #20 +3%).
// FFTS AIC↔AIV base flag ids must be <=7 (catlass FFTS_MAX_FLAG).
// ids 8/9/10 are CrossCoreBarrier — must NOT reuse as mode-2 flags.
//
// Pass2 flag map (Pass1 leftovers live on 5/6/7 — never reuse those here):
//   0     Temp          AIC→AIV  dH tile done (one Set per N-tile)
//   1/2   FlagAivFreeP2 AIV→AIC  dZ ready (per bank; BOTH AIVs Set)
//   3/4   FlagRemat     AIC→AIV  start BuildDz (per bank)
//   5     AicDone       Pass1 only
//   6/7   AivFree       Pass1 only
//
// v29b: FlagRemat ≠ FlagAivFreeP2 (shared 3/4: AIV0 FreeP2 woke AIV1 Remat early).
// Set/Wait ×1 (×2 deadlocks). Always reload logits (BuildDzFromExp was stale after CE).
// v30: Wait FlagTemp per N-tile (not after loop) — FFTS credit overflow hung nTiles>16 (id168).
//      Mid nSplit>1 still disabled in SelectScenario (bench); ACLNN entry keeps nSplit=1.
constexpr uint16_t kFlagTempReady = 0;
constexpr uint16_t kFlagAivFreeP2_0 = 1;
constexpr uint16_t kFlagAivFreeP2_1 = 2;
constexpr uint16_t kFlagRemat0 = 3;
constexpr uint16_t kFlagRemat1 = 4;
constexpr uint16_t kFlagAicDone = 5;
constexpr uint16_t kFlagAivFree0 = 6;
constexpr uint16_t kFlagAivFree1 = 7;

CHUNK_LOSS_DEVICE uint16_t FlagAivFree(uint32_t stage) { return stage == 0u ? kFlagAivFree0 : kFlagAivFree1; }
CHUNK_LOSS_DEVICE uint16_t FlagAivFreeP2(uint32_t bank) { return bank == 0u ? kFlagAivFreeP2_0 : kFlagAivFreeP2_1; }
CHUNK_LOSS_DEVICE uint16_t FlagRemat(uint32_t bank) { return bank == 0u ? kFlagRemat0 : kFlagRemat1; }
CHUNK_LOSS_DEVICE uint16_t FlagTemp() { return kFlagTempReady; }
CHUNK_LOSS_DEVICE void SetRemat(uint32_t bank) { AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FlagRemat(bank)); }
CHUNK_LOSS_DEVICE void SetTempReady() { AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FlagTemp()); }
CHUNK_LOSS_DEVICE void WaitAivFreeP2(uint32_t bank) { AscendC::CrossCoreWaitFlag(FlagAivFreeP2(bank)); }
constexpr int32_t kNegInfBits = static_cast<int32_t>(0xFF800000);
constexpr float kPadLogit = -1.0e30f;
constexpr uint32_t kDataPerRepeat = 64;
constexpr int32_t kMaxReduceRepeat = 255;
constexpr uint32_t kPartialStride = 8;

// Pass1 ping-pong staging in Element (fp16/bf16).
CHUNK_LOSS_HOST_DEVICE uint64_t LogitElements(uint32_t aicNum, uint32_t tileM = kTileMDefault)
{
    return static_cast<uint64_t>(aicNum) * kStages * tileM * kTileN;
}

CHUNK_LOSS_HOST_DEVICE uint64_t PartialFloats(uint32_t aicNum)
{
    return static_cast<uint64_t>(aicNum) * kStages * kPartialStride;
}

CHUNK_LOSS_HOST_DEVICE uint64_t DhAccFloats(uint32_t aicNum, uint32_t h, uint32_t tileM = kTileMDefault)
{
    return static_cast<uint64_t>(aicNum) * tileM * h;
}

// r029 P2: Z checkpoint is tile-major [stripe][nTile][TileM][TileN] (FLOMS mmOut style),
// not row-major [stripe][TileM][V]. Same bytes when V%TileN==0; denser Pass2 LoadZ.
CHUNK_LOSS_HOST_DEVICE uint32_t ZFullVPad(uint32_t v) { return ((v + kTileN - 1u) / kTileN) * kTileN; }

CHUNK_LOSS_HOST_DEVICE uint64_t ZFullElements(uint32_t aicNum, uint32_t v, uint32_t tileM = kTileMDefault,
                                              uint32_t mTiles = 0, uint32_t nSplit = 1, uint32_t rematZ = 0,
                                              uint32_t logitAsZ = 0)
{
    // rematZ: Pass2 recomputes Z; logitAsZ (SingleN): Pass2 rereads Pass1 logit GM — no ZFull.
    if (rematZ != 0u || logitAsZ != 0u) {
        return 0;
    }
    uint32_t stripes = (nSplit > 1u && mTiles > 0u) ? mTiles : aicNum;
    return static_cast<uint64_t>(stripes) * tileM * ZFullVPad(v);
}

// Deterministic: nSplit siblings each own a private stripe (no shared AtomicAdd).
// ×kStages: logitAsZ/ZFull Pass2 writes per-N-tile banks (no AtomicAdd under FFTS).
CHUNK_LOSS_HOST_DEVICE uint64_t DhAccStripes(uint32_t aicNum, uint32_t mTiles = 0, uint32_t nSplit = 1)
{
    uint64_t base = (nSplit > 1u && mTiles > 0u) ? (static_cast<uint64_t>(mTiles) * static_cast<uint64_t>(nSplit)) :
                                                   aicNum;
    return base * kStages;
}

CHUNK_LOSS_HOST_DEVICE uint64_t DhAccBaseFloats(uint32_t mtOrCore, uint32_t tileM, uint32_t h, uint32_t nSplit = 1,
                                                uint32_t ni = 0)
{
    uint64_t stripe = (nSplit > 1u) ? (static_cast<uint64_t>(mtOrCore) * nSplit + ni) : mtOrCore;
    return stripe * kStages * static_cast<uint64_t>(tileM) * h;
}

CHUNK_LOSS_HOST_DEVICE uint64_t StatePartialFloats(uint32_t aicNum, uint32_t tileM, uint32_t nSplit = 1)
{
    // Per-core max+sum partials for online logsumexp merge across N-siblings.
    if (nSplit <= 1u) {
        return 0;
    }
    return static_cast<uint64_t>(aicNum) * tileM * 2u;
}

CHUNK_LOSS_HOST_DEVICE uint64_t FloatWorkspaceFloats(uint32_t aicNum, uint32_t h, uint32_t tileM = kTileMDefault,
                                                     uint32_t mTiles = 0, uint32_t nSplit = 1)
{
    uint32_t dhStripes = static_cast<uint32_t>(DhAccStripes(aicNum, mTiles, nSplit));
    return static_cast<uint64_t>(dhStripes) * tileM * h + PartialFloats(aicNum) +
           StatePartialFloats(aicNum, tileM, nSplit);
}

CHUNK_LOSS_HOST_DEVICE uint64_t WorkspaceBytes(uint32_t aicNum, uint32_t h, uint32_t v, uint32_t elementBytes,
                                               uint32_t tileM = kTileMDefault, uint32_t m = 0, uint32_t nSplit = 1,
                                               uint32_t rematZ = 0, uint32_t logitAsZ = 0)
{
    uint32_t mTiles = 0;
    if (m > 0u && tileM > 0u) {
        mTiles = (m + tileM - 1u) / tileM;
        if (mTiles == 0) {
            mTiles = 1;
        }
    }
    uint64_t logit = LogitElements(aicNum, tileM) * elementBytes;
    uint64_t floats = FloatWorkspaceFloats(aicNum, h, tileM, mTiles, nSplit) * sizeof(float);
    uint64_t zFull = ZFullElements(aicNum, v, tileM, mTiles, nSplit, rematZ, logitAsZ) * elementBytes;
    uint64_t dZ = static_cast<uint64_t>(aicNum) * kStages * tileM * kTileN * elementBytes;
    return ((logit + floats + zFull + dZ) + 511u) & ~static_cast<uint64_t>(511);
}

// Byte offset of float region (dhAcc | partial | state).
CHUNK_LOSS_HOST_DEVICE uint64_t FloatBaseBytes(uint32_t aicNum, uint32_t tileM, uint32_t elementBytes)
{
    return LogitElements(aicNum, tileM) * elementBytes;
}

CHUNK_LOSS_HOST_DEVICE uint64_t DhAccOffsetFloats(uint32_t /*aicNum*/, uint32_t /*tileM*/ = kTileMDefault)
{
    return 0; // relative to float base
}

CHUNK_LOSS_HOST_DEVICE uint64_t PartialOffsetFloats(uint32_t aicNum, uint32_t h, uint32_t tileM = kTileMDefault,
                                                    uint32_t mTiles = 0, uint32_t nSplit = 1)
{
    uint32_t dhStripes = static_cast<uint32_t>(DhAccStripes(aicNum, mTiles, nSplit));
    return static_cast<uint64_t>(dhStripes) * tileM * h;
}

CHUNK_LOSS_HOST_DEVICE uint64_t StatePartialOffsetFloats(uint32_t aicNum, uint32_t h, uint32_t tileM,
                                                         uint32_t mTiles = 0, uint32_t nSplit = 1)
{
    return PartialOffsetFloats(aicNum, h, tileM, mTiles, nSplit) + PartialFloats(aicNum);
}

CHUNK_LOSS_HOST_DEVICE uint64_t ZFullBytesOffset(uint32_t aicNum, uint32_t h, uint32_t tileM, uint32_t elementBytes,
                                                 uint32_t mTiles = 0, uint32_t nSplit = 1)
{
    return FloatBaseBytes(aicNum, tileM, elementBytes) +
           FloatWorkspaceFloats(aicNum, h, tileM, mTiles, nSplit) * sizeof(float);
}

CHUNK_LOSS_HOST_DEVICE uint64_t DzBytesOffset(uint32_t aicNum, uint32_t h, uint32_t v, uint32_t elementBytes,
                                              uint32_t tileM = kTileMDefault, uint32_t mTiles = 0, uint32_t nSplit = 1,
                                              uint32_t rematZ = 0, uint32_t logitAsZ = 0)
{
    return ZFullBytesOffset(aicNum, h, tileM, elementBytes, mTiles, nSplit) +
           ZFullElements(aicNum, v, tileM, mTiles, nSplit, rematZ, logitAsZ) * elementBytes;
}

// Host: fill AIC when M is small.
// - Prefer maxPer==1 (one M-tile/core) to avoid CrossCore amplification on long V.
// - If still underfilled and nTiles is modest, allow maxPer==2 (helps mid-V like #14).
// - Large M: minimize maxPer, prefer TileM=128 (protect target #20).
CHUNK_LOSS_HOST_DEVICE uint32_t SelectTileM(uint32_t m, uint32_t aicNum, uint32_t v = 0)
{
    if (aicNum == 0) {
        aicNum = 1;
    }
    const uint32_t cands[3] = {32u, 64u, 128u};
    uint32_t nTiles = v == 0 ? 0u : (v + kTileN - 1u) / kTileN;
    uint32_t best = kTileMDefault;
    uint32_t bestUsed = 0;
    bool foundUnit = false;
    for (uint32_t i = 0; i < 3; ++i) {
        uint32_t tm = cands[i];
        uint32_t mTiles = (m + tm - 1u) / tm;
        if (mTiles == 0) {
            mTiles = 1;
        }
        uint32_t used = mTiles < aicNum ? mTiles : aicNum;
        uint32_t maxPer = (mTiles + aicNum - 1u) / aicNum;
        if (maxPer != 1u) {
            continue;
        }
        if (!foundUnit || used > bestUsed || (used == bestUsed && tm > best)) {
            best = tm;
            bestUsed = used;
            foundUnit = true;
        }
    }
    // Modest V: allow one extra M-tile/core to fill remaining AICs (#14 TileM=32).
    // A/B r027: removing this (force TileM=64 or N-split 112) slowed #14 0.44→0.56/0.47.
    constexpr uint32_t kAllowDoubleNTiles = 32u;
    if (foundUnit && bestUsed < aicNum && nTiles > 0u && nTiles <= kAllowDoubleNTiles) {
        for (uint32_t i = 0; i < 3; ++i) {
            uint32_t tm = cands[i];
            uint32_t mTiles = (m + tm - 1u) / tm;
            if (mTiles == 0) {
                mTiles = 1;
            }
            uint32_t used = mTiles < aicNum ? mTiles : aicNum;
            uint32_t maxPer = (mTiles + aicNum - 1u) / aicNum;
            if (maxPer > 2u) {
                continue;
            }
            if (used > bestUsed || (used == bestUsed && tm > best)) {
                best = tm;
                bestUsed = used;
            }
        }
        return best;
    }
    if (foundUnit) {
        return best;
    }
    uint32_t bestMaxPer = 0xffffffffu;
    best = kTileMDefault;
    bestUsed = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        uint32_t tm = cands[i];
        uint32_t mTiles = (m + tm - 1u) / tm;
        if (mTiles == 0) {
            mTiles = 1;
        }
        uint32_t used = mTiles < aicNum ? mTiles : aicNum;
        uint32_t maxPer = (mTiles + aicNum - 1u) / aicNum;
        if (used > bestUsed || (used == bestUsed && maxPer < bestMaxPer) ||
            (used == bestUsed && maxPer == bestMaxPer && tm > best)) {
            best = tm;
            bestUsed = used;
            bestMaxPer = maxPer;
        }
    }
    return best;
}

// Scenario templates (r017/r028/r030):
//   Small    — tiny shapes still Pingpong
//   ShortV   — nTiles∈[2,4]: Preload(+Callback opt); was misrouted Small (#04/#10)
//   SingleN  — nTiles==1: large TileM, Preload; cut V-pingpong tax (#12)
//   Latency  — mid-V CrossCore (#14): Callback; RematZ deferred
//   Mid      — long-V underfill (#15/#18): Preload + N-split + Z checkpoint
//   Large    — M saturates AIC (#16/#17/#20): TileM=128 + checkpoint
enum class Scenario : uint32_t {
    Small = 0,
    Mid = 1,
    Large = 2,
    Latency = 3,
    ShortV = 4,
    SingleN = 5,
    LargeLite = 6, // mid-V Large (#19); never #20
};

struct ScenarioCfg {
    Scenario scenario = Scenario::Large;
    uint32_t tileM = kTileMDefault;
    uint32_t nSplit = 1; // >1: N-siblings per M-tile (ShortM/LongV Mid)
    uint32_t rematZ = 0; // 1: Pass2 rematerializes Z; skip ZFull HBM
    bool preloadZ = true;
    bool fullLoadH = true;
    bool callbackZ = false; // Latency-only: PreloadAsyncWithCallback (r029 isolate)
    bool batchZ = false;    // Latency-only: coalesce AicDone every 2 N-tiles (P1 Step3)
    bool logitAsZ = false;  // SingleN: Pass2 reads logit GM; skip ZFull (r030)
};

CHUNK_LOSS_HOST_DEVICE ScenarioCfg SelectScenario(uint32_t m, uint32_t h, uint32_t v, uint32_t aicNum)
{
    ScenarioCfg cfg;
    if (aicNum == 0) {
        aicNum = 1;
    }
    uint32_t nTiles = (v + kTileN - 1u) / kTileN;
    uint32_t hTiles = (h + kTileH - 1u) / kTileH;
    uint32_t mTiles128 = (m + kTileMDefault - 1u) / kTileMDefault;
    if (mTiles128 == 0) {
        mTiles128 = 1;
    }
    cfg.tileM = SelectTileM(m, aicNum, v);
    cfg.nSplit = 1;
    cfg.rematZ = 0;
    cfg.callbackZ = false;
    cfg.batchZ = false;
    cfg.logitAsZ = false;
    cfg.fullLoadH = (hTiles > 1u);

    // Large M: protect #20. mid-V → LargeLite (#19) for Element dH / AtomicAdd loss.
    constexpr uint32_t kLatencyMaxNTiles = 32u;
    constexpr uint32_t kLargeLiteMaxV = 4096u;
    if (mTiles128 >= aicNum) {
        const bool lite = (v <= kLargeLiteMaxV);
        cfg.scenario = lite ? Scenario::LargeLite : Scenario::Large;
        cfg.tileM = kTileMDefault;
        cfg.nSplit = 1;
        // RematZ FlagRemat path KEEP (FORCE_REMAT); default OFF — #19 0.187→0.308 (+Z GEMM) ROLLBACK.
        cfg.rematZ = 0;
        // Large/#20: Fixpipe Callback AicDone — CE overlaps next Z (r029 348→~322; Nov14 ~341).
        // LargeLite mid-V kept classic Preload (callback optional later).
        cfg.callbackZ = !lite;
        cfg.batchZ = false;
        cfg.preloadZ = true;
        cfg.fullLoadH = true;
        return cfg;
    }

    uint64_t gemmElems = static_cast<uint64_t>(m) * static_cast<uint64_t>(h) * static_cast<uint64_t>(v);
    constexpr uint64_t kMidElems = 256ull * 1024ull * 1024ull;
    if (nTiles >= 16u || gemmElems >= kMidElems) {
        cfg.preloadZ = true;
        cfg.fullLoadH = true;
        cfg.nSplit = 1;
        uint32_t mTilesAdapt = (m + cfg.tileM - 1u) / cfg.tileM;
        if (mTilesAdapt == 0) {
            mTilesAdapt = 1;
        }
        uint32_t usedAdapt = mTilesAdapt < aicNum ? mTilesAdapt : aicNum;
        // Long-V Mid (#15/#18): Z checkpoint; nSplit disabled (v30 Accu hang).
        // Strict `>` so #13 (V=8192 → nTiles==32) falls through to Latency/ShortM.
        constexpr uint32_t kNSplitMinNTiles = 32u;
        if ((mTiles128 == 1u || usedAdapt < aicNum) && nTiles > kNSplitMinNTiles) {
            cfg.scenario = Scenario::Mid;
            cfg.nSplit = 1; // v30: nSplit>1 Accu hang (Pass2 SyncAll/Temp)
            cfg.tileM = kTileMDefault;
            cfg.rematZ = 0;
            cfg.preloadZ = true;
            cfg.fullLoadH = true;
            return cfg;
        }
        // Mid-V underfill (#14) or ShortM long-V (#13): Latency RematZ.
        if (nTiles <= kLatencyMaxNTiles) {
            // #13 ShortM: v30 force nSplit=1 (nSplit=2..4 hung Accu id97/168/…).
            if (mTiles128 == 1u) {
                cfg.scenario = Scenario::Mid;
                cfg.rematZ = 0;
                cfg.tileM = kTileMDefault;
                cfg.nSplit = 1;
                // Preload KEEP (flat～微利). Latency+Callback A/B: #13 ~0.072 flat → ROLLBACK.
                cfg.preloadZ = true;
                cfg.fullLoadH = true;
                return cfg;
            }
            // #14: skipCe≈full → CE税~2%，瓶颈是 TileM=32 CrossCore vs baseline chunk GEMM.
            // v30: skip Mid N-split (Accu hang); fall through to Latency.
            if (false && mTiles128 >= 4u && mTiles128 < aicNum && nTiles >= 8u) {
                uint32_t bestTm = 0;
                uint32_t bestNs = 0;
                uint32_t bestUsed = 0;
                const uint32_t tmCands[3] = {112u, 128u, 64u};
                for (uint32_t ci = 0; ci < 3; ++ci) {
                    uint32_t tm = tmCands[ci];
                    uint32_t mt = (m + tm - 1u) / tm;
                    if (mt == 0) {
                        mt = 1;
                    }
                    if (mt >= aicNum) {
                        continue;
                    }
                    uint32_t ns = aicNum / mt;
                    if (ns < 2u || nTiles < ns) {
                        continue;
                    }
                    uint32_t used = mt * ns;
                    if (used > bestUsed || (used == bestUsed && tm > bestTm)) {
                        bestUsed = used;
                        bestTm = tm;
                        bestNs = ns;
                    }
                }
                if (bestNs >= 2u) {
                    cfg.scenario = Scenario::Mid;
                    cfg.tileM = bestTm;
                    cfg.nSplit = bestNs;
                    // RematZ 0.208 / preload0 0.144 / Callback flat → KEEP preload Mid checkpoint.
                    cfg.rematZ = 0;
                    cfg.preloadZ = true;
                    cfg.fullLoadH = true;
                    cfg.callbackZ = false;
                    cfg.batchZ = false;
                    return cfg;
                }
            }
            // Latency Callback fallback (r029).
            cfg.scenario = Scenario::Latency;
            cfg.rematZ = 0;
            cfg.preloadZ = true;
            cfg.callbackZ = true;
            cfg.batchZ = false;
            cfg.nSplit = 1;
            return cfg;
        }
        cfg.scenario = Scenario::Mid;
        return cfg;
    }

    // r030: short-V losers (#04/#10/#12) must not stay on Small Pingpong (msprof mac~2–3%).
    // SingleN only when enough M-stripes (TileM=128); tiny M stays Small (#03 regress otherwise).
    if (nTiles == 1u && mTiles128 >= 4u) {
        // #12: TileM=128 + logitAsZ fused CE→dZ; Pingpong Z.
        cfg.scenario = Scenario::SingleN;
        cfg.tileM = kTileMDefault;
        cfg.nSplit = 1;
        cfg.preloadZ = false;
        cfg.fullLoadH = (hTiles > 1u);
        cfg.callbackZ = false;
        cfg.logitAsZ = true;
        return cfg;
    }
    if (nTiles <= 4u && mTiles128 < aicNum) {
        // ShortV (#04/#10): Pingpong+Element dH. Callback/BatchZ A/B: #10 0.082→0.098 ROLLBACK.
        cfg.scenario = Scenario::ShortV;
        cfg.nSplit = 1;
        // Preload A/B (#10): median 0.034→0.037 flat/worse → KEEP pingpong.
        cfg.preloadZ = false;
        cfg.fullLoadH = (hTiles > 1u);
        cfg.callbackZ = false;
        cfg.batchZ = false;
        // logitAsZ when nTiles≤kStages (#04=2). #10 nTiles=4 needs kStages=4 — A/B regress ROLLBACK.
        cfg.logitAsZ = (nTiles <= kStages);
        // TileM=128/64 A/B (#10): 抬 skipCe 或墙钟变差 → KEEP SelectTileM(=32).
        return cfg;
    }

    // Small: Pingpong; FullLoadA only when H multi-tile helps.
    cfg.scenario = Scenario::Small;
    cfg.preloadZ = false;
    cfg.fullLoadH = (hTiles > 1u);
    return cfg;
}

struct Params {
    GM_ADDR hidden;
    GM_ADDR weight;
    GM_ADDR labels;
    GM_ADDR loss;
    GM_ADDR maxOut;
    GM_ADDR sumOut;
    GM_ADDR dH;
    GM_ADDR workspace;
    uint32_t m;
    uint32_t h;
    uint32_t v;
    uint32_t aicNum;
    uint32_t nSplit;
    uint32_t rematZ;     // 1: Latency template — rematerialize Z in Pass2, no ZFull (deferred)
    uint32_t skipMaxSum; // 1: P3 — skip GM dump of max/sum (loss/dH only)
    uint32_t logitAsZ;   // 1: SingleN — Pass2 CopyLogits from staging; no ZFull
    uint32_t skipCe;     // 1: diagnostic — Z+StoreDz+dH only (no Online CE / BuildDz math)
    float alpha;
    int64_t ignoreIndex;
};

template <class Element>
CHUNK_LOSS_DEVICE Element ElementZero()
{
    return Element{};
}

} // namespace chunkloss

#endif
