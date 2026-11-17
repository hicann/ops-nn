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
 * \file chunk_loss_common_arch35.h
 * \brief ChunkLoss arch35 公共常量、workspace 布局与场景选择
 *
 * 定义 tile 尺寸、CrossCore flag、workspace 偏移，以及 SelectTileM / SelectScenario。
 */
#ifndef CHUNKLOSS_ARCH35_COMMON_H_
#define CHUNKLOSS_ARCH35_COMMON_H_

#include <cstdint>
#include "kernel_operator.h"

namespace chunkloss {

#ifndef CHUNKLOSS_DEVICE
#define CHUNKLOSS_DEVICE __aicore__ inline
#endif

#ifndef CHUNKLOSS_SET_FLAG
#define CHUNKLOSS_SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define CHUNKLOSS_WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#endif

// Tile sizes for Ascend950 MIX_AIC_1_2.
constexpr uint32_t kTileMDefault = 128;
constexpr uint32_t kTileN = 256;
constexpr uint32_t kCubeAlign = 16;
// Pass2 H tile：双 L0C bank，Fixpipe 可与下一次 Mmad 重叠。
constexpr uint32_t kTileH = 256;
constexpr uint32_t kL0KPass2 = 32;
constexpr uint32_t kTileHLarge = kTileH;
constexpr uint32_t kL0KPass2Large = kL0KPass2;
constexpr uint32_t kL1K = 256;
constexpr uint32_t kL0K = 64;
constexpr uint32_t kStages = 2;
constexpr uint32_t kTempBanks = 2;

/**
 * Multi-N V-tail: floor rem to 16-align for Cube; residual <16 uses a second
 * backed-up aligned window with padLeft masked in Vec so LSE/dH do not double-count.
 */
struct VTail {
    uint32_t n0{0};      // logical ownership start for this sub-window
    uint32_t actualN{0}; // logical ownership length
    uint32_t n0Cube{0};  // Cube GM window start
    uint32_t nCube{0};   // Cube window length (always 16-aligned here)
    uint32_t padLeft{0}; // n0 - n0Cube (overlap to mask)
};

CHUNKLOSS_DEVICE VTail MakeVTail(uint32_t nt, uint32_t v)
{
    VTail t;
    t.n0 = nt * kTileN;
    t.actualN = (t.n0 + kTileN <= v) ? kTileN : (v - t.n0);
    t.n0Cube = t.n0;
    t.nCube = t.actualN;
    t.padLeft = 0;
    if (t.n0 > 0u && t.actualN > 0u && (t.actualN % kCubeAlign) != 0u) {
        // Floor rem to 16-align; stub handled by MakeVTailStub in the N-loop.
        t.nCube = (t.actualN / kCubeAlign) * kCubeAlign;
        t.actualN = t.nCube;
        t.n0Cube = t.n0;
        t.padLeft = 0;
    }
    return t;
}

/** Final <16 stub after floored rem (backed-up 16-aligned Cube window). */
CHUNKLOSS_DEVICE bool MakeVTailStub(uint32_t nt, uint32_t v, VTail& stub)
{
    uint32_t n0 = nt * kTileN;
    uint32_t rem = (n0 + kTileN <= v) ? kTileN : (v - n0);
    if (!(n0 > 0u && rem > 0u && (rem % kCubeAlign) != 0u)) {
        return false;
    }
    uint32_t floored = (rem / kCubeAlign) * kCubeAlign;
    uint32_t stubN = rem - floored;
    if (stubN == 0u) {
        return false;
    }
    stub.n0 = n0 + floored;
    stub.actualN = stubN;
    stub.nCube = kCubeAlign;
    stub.padLeft = stub.nCube - stub.actualN;
    stub.n0Cube = stub.n0 - stub.padLeft;
    return stub.n0Cube + stub.nCube == v;
}

CHUNKLOSS_DEVICE void MaskPadLeftLogits(AscendC::LocalTensor<float>& logits, uint32_t rows, uint32_t padLeft)
{
    if (padLeft == 0u) {
        return;
    }
    // Duplicate requires 32B-friendly lengths; padLeft is often 4.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < padLeft; ++c) {
            logits.SetValue(r * kTileN + c, -1.0e30f);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
}

/** Expand one host N-tile into 1–2 Cube/Vec windows (floor rem + optional stub). */
CHUNKLOSS_DEVICE uint32_t CollectVTails(uint32_t nt, uint32_t v, VTail out[2])
{
    out[0] = MakeVTail(nt, v);
    uint32_t n = 1u;
    VTail stub;
    if (MakeVTailStub(nt, v, stub)) {
        out[n++] = stub;
    }
    return n;
}

// CrossCore AIC↔AIV (mode 2). L0C→UB uses mode 4 with flagId+kFlagIdMax.
constexpr uint16_t kFlagAicDone = 5;
constexpr uint16_t kFlagAivFree0 = 6;
constexpr uint16_t kFlagAivFree1 = 7;
constexpr uint16_t kFlagTempReady = 8;
constexpr uint16_t kFlagAddDone = 9;
constexpr uint16_t kFlagIdMax = 16;
constexpr uint16_t kSyncModeL0cUb = 4;

constexpr int32_t kNegInfBits = static_cast<int32_t>(0xFF800000);
constexpr float kPadLogit = -1.0e30f;
constexpr uint32_t kDataPerRepeat = 64;
constexpr int32_t kMaxReduceRepeat = 255;
constexpr uint32_t kPartialStride = 8;

// Ascend950 / DAV_3510 on-chip sizes.
constexpr int32_t kL1Size = 512 * 1024;
constexpr int32_t kL0AbSize = 64 * 1024;
constexpr int32_t kL0CSize = 256 * 1024;
constexpr int32_t kUbSize = 248 * 1024;

CHUNKLOSS_DEVICE uint16_t FlagAivFree(uint32_t bank) { return bank == 0u ? kFlagAivFree0 : kFlagAivFree1; }
CHUNKLOSS_DEVICE uint16_t FlagTemp(uint32_t bank) { return bank == 0u ? kFlagTempReady : kFlagAddDone; }

CHUNKLOSS_DEVICE uint32_t RoundUp16(uint32_t x) { return (x + 15u) & ~15u; }

CHUNKLOSS_DEVICE uint64_t LogitElements(uint32_t aicNum, uint32_t tileM = kTileMDefault)
{
    return static_cast<uint64_t>(aicNum) * kStages * tileM * kTileN;
}

CHUNKLOSS_DEVICE uint64_t PartialFloats(uint32_t aicNum)
{
    return static_cast<uint64_t>(aicNum) * kStages * kPartialStride;
}

// Always per-core private stripes so nSplit workers never cross-AtomicAdd (det-safe).
CHUNKLOSS_DEVICE uint64_t DhAccStripes(uint32_t aicNum, uint32_t /*mTiles*/ = 0, uint32_t /*nSplit*/ = 1)
{
    return aicNum;
}

// ZFull is still shared per m-tile under nSplit (Pass1 logit checkpoint).
CHUNKLOSS_DEVICE uint64_t ZFullStripes(uint32_t aicNum, uint32_t mTiles = 0, uint32_t nSplit = 1)
{
    return (nSplit > 1u && mTiles > 0u) ? mTiles : aicNum;
}

CHUNKLOSS_DEVICE uint64_t StatePartialFloats(uint32_t aicNum, uint32_t tileM, uint32_t nSplit = 1)
{
    if (nSplit <= 1u) {
        return 0;
    }
    return static_cast<uint64_t>(aicNum) * tileM * 2u;
}

CHUNKLOSS_DEVICE uint64_t ZFullElements(uint32_t aicNum, uint32_t v, uint32_t tileM = kTileMDefault,
                                        uint32_t mTiles = 0, uint32_t nSplit = 1)
{
    uint32_t stripes = static_cast<uint32_t>(ZFullStripes(aicNum, mTiles, nSplit));
    return static_cast<uint64_t>(stripes) * tileM * v;
}

// TempDh RMW buffer when deterministic path needs private dH accumulation.
CHUNKLOSS_DEVICE uint64_t TempDhFloats(uint32_t aicNum, uint32_t h, uint32_t tileM = kTileMDefault,
                                       uint32_t deterministic = 0)
{
    return deterministic ? static_cast<uint64_t>(aicNum) * tileM * h : 0;
}

CHUNKLOSS_DEVICE uint64_t FloatWorkspaceFloats(uint32_t aicNum, uint32_t h, uint32_t tileM = kTileMDefault,
                                               uint32_t mTiles = 0, uint32_t nSplit = 1, uint32_t deterministic = 0)
{
    uint32_t dhStripes = static_cast<uint32_t>(DhAccStripes(aicNum, mTiles, nSplit));
    return static_cast<uint64_t>(dhStripes) * tileM * h + PartialFloats(aicNum) +
           StatePartialFloats(aicNum, tileM, nSplit) + TempDhFloats(aicNum, h, tileM, deterministic);
}

// Pass1 ping-pong logits stay fp32 (Cube Fixpipe NoQuant) so online LSE matches fp32 golden.
CHUNKLOSS_DEVICE uint64_t LogitBytes(uint32_t aicNum, uint32_t tileM = kTileMDefault)
{
    return LogitElements(aicNum, tileM) * sizeof(float);
}

CHUNKLOSS_DEVICE uint64_t FloatBaseBytes(uint32_t aicNum, uint32_t /*elementBytes*/, uint32_t tileM = kTileMDefault)
{
    return LogitBytes(aicNum, tileM);
}

CHUNKLOSS_DEVICE uint64_t WorkspaceBytes(uint32_t aicNum, uint32_t h, uint32_t v, uint32_t elementBytes,
                                         uint32_t tileM = kTileMDefault, uint32_t m = 0, uint32_t nSplit = 1,
                                         uint32_t deterministic = 0)
{
    uint32_t mTiles = 0;
    if (m > 0u && tileM > 0u) {
        mTiles = (m + tileM - 1u) / tileM;
        if (mTiles == 0) {
            mTiles = 1;
        }
    }
    uint64_t logit = LogitBytes(aicNum, tileM);
    uint64_t floats = FloatWorkspaceFloats(aicNum, h, tileM, mTiles, nSplit, deterministic) * sizeof(float);
    // ZFull fp32 so Pass2 BuildDz remat matches Pass1 float LSE (no Element round-trip).
    uint64_t zFull = ZFullElements(aicNum, v, tileM, mTiles, nSplit) * sizeof(float);
    uint64_t dZ = static_cast<uint64_t>(aicNum) * kStages * tileM * kTileN * elementBytes;
    return ((logit + floats + zFull + dZ) + 511u) & ~static_cast<uint64_t>(511);
}

CHUNKLOSS_DEVICE uint64_t DhAccOffsetFloats(uint32_t /*aicNum*/)
{
    return 0; // relative to float base
}

CHUNKLOSS_DEVICE uint64_t PartialOffsetFloats(uint32_t aicNum, uint32_t h, uint32_t tileM = kTileMDefault,
                                              uint32_t mTiles = 0, uint32_t nSplit = 1)
{
    uint32_t dhStripes = static_cast<uint32_t>(DhAccStripes(aicNum, mTiles, nSplit));
    return static_cast<uint64_t>(dhStripes) * tileM * h;
}

CHUNKLOSS_DEVICE uint64_t StatePartialOffsetFloats(uint32_t aicNum, uint32_t h, uint32_t tileM, uint32_t mTiles = 0,
                                                   uint32_t nSplit = 1)
{
    return PartialOffsetFloats(aicNum, h, tileM, mTiles, nSplit) + PartialFloats(aicNum);
}

CHUNKLOSS_DEVICE uint64_t TempDhOffsetFloats(uint32_t aicNum, uint32_t h, uint32_t tileM = kTileMDefault,
                                             uint32_t mTiles = 0, uint32_t nSplit = 1)
{
    return StatePartialOffsetFloats(aicNum, h, tileM, mTiles, nSplit) + StatePartialFloats(aicNum, tileM, nSplit);
}

CHUNKLOSS_DEVICE uint64_t ZFullBytesOffset(uint32_t aicNum, uint32_t h, uint32_t elementBytes,
                                           uint32_t tileM = kTileMDefault, uint32_t mTiles = 0, uint32_t nSplit = 1,
                                           uint32_t deterministic = 0)
{
    return FloatBaseBytes(aicNum, elementBytes, tileM) +
           FloatWorkspaceFloats(aicNum, h, tileM, mTiles, nSplit, deterministic) * sizeof(float);
}

CHUNKLOSS_DEVICE uint64_t DzBytesOffset(uint32_t aicNum, uint32_t h, uint32_t v, uint32_t elementBytes,
                                        uint32_t tileM = kTileMDefault, uint32_t mTiles = 0, uint32_t nSplit = 1,
                                        uint32_t deterministic = 0)
{
    return ZFullBytesOffset(aicNum, h, elementBytes, tileM, mTiles, nSplit, deterministic) +
           ZFullElements(aicNum, v, tileM, mTiles, nSplit) * sizeof(float);
}

// 按 M/V 与可用 AIC 数选择 TileM（32/64/128）。
CHUNKLOSS_DEVICE uint32_t SelectTileM(uint32_t m, uint32_t aicNum, uint32_t v = 0)
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
        if (!foundUnit || used > bestUsed || (used == bestUsed && tm < best)) {
            best = tm;
            bestUsed = used;
            foundUnit = true;
        }
    }
    // Modest-V: allow denser TileM only when core utilization gain is material.
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
            if (used > bestUsed) {
                uint32_t gain = used - bestUsed;
                if (gain * 3u >= aicNum) {
                    best = tm;
                    bestUsed = used;
                }
            } else if (used == bestUsed && tm < best) {
                best = tm;
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

struct ScenarioCfg {
    uint32_t tileM = kTileMDefault;
    uint32_t nSplit = 1;
    bool l0cToUb = false; // Pass1 L0C→UB when eligible (nSplit==1, enough M tiles)
    // 0: nondeterministic AtomicAdd; 1: deterministic AtomicAdd(+FIX); 2: TempDh RMW
    uint32_t deterministic = 0;
};

CHUNKLOSS_DEVICE ScenarioCfg SelectScenario(uint32_t m, uint32_t /*h*/, uint32_t v, uint32_t aicNum,
                                            uint32_t deterministic = 0)
{
    ScenarioCfg cfg;
    if (aicNum == 0) {
        aicNum = 1;
    }
    uint32_t nTiles = (v + kTileN - 1u) / kTileN;
    uint32_t mTiles128 = (m + kTileMDefault - 1u) / kTileMDefault;
    if (mTiles128 == 0) {
        mTiles128 = 1;
    }
    cfg.tileM = SelectTileM(m, aicNum, v);
    cfg.nSplit = 1;
    cfg.l0cToUb = false;
    cfg.deterministic = deterministic ? 1u : 0u;
    const bool blockNSplit = false;
    if (mTiles128 >= aicNum) {
        cfg.tileM = kTileMDefault;
        // L0C→UB only when M tiles are enough to fill AIC (tiny M stays on GM path).
        constexpr uint32_t kL0cMinMTiles = 28u;
        cfg.l0cToUb = (mTiles128 >= kL0cMinMTiles);
    } else {
        constexpr uint32_t kNSplitMinNTiles = 32u;
        constexpr uint32_t kLatencyMaxNTiles = 32u;
        constexpr uint32_t kShortMMaxSplit = 8u;
        if (nTiles > kNSplitMinNTiles) {
            uint32_t tm = kTileMDefault;
            uint32_t mt = (m + tm - 1u) / tm;
            if (mt == 0) {
                mt = 1;
            }
            if (mt < aicNum && !blockNSplit) {
                uint32_t ns = aicNum / mt;
                if (ns >= 2u && nTiles >= ns) {
                    cfg.tileM = tm;
                    cfg.nSplit = ns;
                    return cfg; // nSplit>1 uses GM logit path
                }
            }
        } else if (mTiles128 == 1u && nTiles <= kLatencyMaxNTiles) {
            uint32_t ns = aicNum;
            if (ns > kShortMMaxSplit) {
                ns = kShortMMaxSplit;
            }
            if (ns >= 2u && nTiles >= ns && !blockNSplit) {
                cfg.tileM = kTileMDefault;
                cfg.nSplit = ns;
                return cfg;
            }
        } else {
            if (mTiles128 >= (aicNum * 2u / 3u) && nTiles <= kLatencyMaxNTiles) {
                cfg.tileM = 112u;
            }
        }
    }
    // Small M remainder with unitFlag Fixpipe→UB is unsafe; keep GM path.
    {
        uint32_t rem = m % cfg.tileM;
        if (cfg.l0cToUb && m > cfg.tileM && rem != 0u && rem < 16u) {
            cfg.l0cToUb = false;
        }
    }
    if (blockNSplit) {
        cfg.nSplit = 1;
    }
    // Prefer TileM with rem==0 or rem>=16 (unitFlag-safe remainder).
    {
        const uint32_t cands[4] = {32u, 64u, 112u, 128u};
        uint32_t rem = m % cfg.tileM;
        if (rem != 0u && rem < 16u) {
            for (uint32_t i = 0; i < 4; ++i) {
                uint32_t tm = cands[i];
                uint32_t r = m % tm;
                if (r == 0u || r >= 16u) {
                    cfg.tileM = tm;
                    break;
                }
            }
        }
    }
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
    uint32_t deterministic; // 0/1 AtomicAdd(+FIX); 2 TempDh RMW
    float alpha;
    int64_t ignoreIndex;
};

} // namespace chunkloss

#endif
