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
 * \file chunk_loss_tiling_arch35.cpp
 * \brief ChunkLoss arch35 Host tiling 实现
 *
 * 根据 shape/确定性开关选择 tileM、nSplit、workspace，并写入 TilingData。
 */
#include "chunk_loss_tiling.h"

#include <cmath>
#include <limits>

#include "acl/acl_rt.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t kTileMDefault = 128;
constexpr uint32_t kTileN = 256;
constexpr uint32_t kStages = 2;
constexpr uint32_t kPartialStride = 8;

// Host mirror of arch35 SelectTileM (keep in sync with kernel).
uint32_t SelectTileM(uint32_t m, uint32_t aicNum, uint32_t v)
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
        // Prefer smaller TileM when core usage ties.
        if (!foundUnit || used > bestUsed || (used == bestUsed && tm < best)) {
            best = tm;
            bestUsed = used;
            foundUnit = true;
        }
    }
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
    bool l0cToUb = false;
    uint32_t deterministic = 0;
};

// Host mirror of arch35 SelectScenario.
ScenarioCfg SelectScenario(uint32_t m, uint32_t /*h*/, uint32_t v, uint32_t aicNum, uint32_t deterministic = 0)
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
                    return cfg;
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
    {
        uint32_t rem = m % cfg.tileM;
        if (cfg.l0cToUb && m > cfg.tileM && rem != 0u && rem < 16u) {
            cfg.l0cToUb = false;
        }
    }
    if (blockNSplit) {
        cfg.nSplit = 1;
    }
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

uint64_t UserWorkspace(uint32_t aicNum, uint32_t h, uint32_t v, uint32_t elementBytes, uint32_t deterministic,
                       uint32_t tileM, uint32_t m, uint32_t nSplit)
{
    // DhAcc: always per-core private stripes. ZFull: shared per m-tile under nSplit.
    uint32_t mTiles = 0;
    if (m > 0u && tileM > 0u) {
        mTiles = (m + tileM - 1u) / tileM;
        if (mTiles == 0) {
            mTiles = 1;
        }
    }
    uint32_t zStripes = (nSplit > 1u && mTiles > 0u) ? mTiles : aicNum;
    // Pass1 logit ping-pong is fp32 (matches Cube L0C accumulate / online LSE).
    uint64_t logit = static_cast<uint64_t>(aicNum) * kStages * tileM * kTileN * sizeof(float);
    uint64_t dhAcc = static_cast<uint64_t>(aicNum) * tileM * h * sizeof(float);
    uint64_t partial = static_cast<uint64_t>(aicNum) * kStages * kPartialStride * sizeof(float);
    uint64_t statePartial = (nSplit > 1u) ? static_cast<uint64_t>(aicNum) * tileM * 2u * sizeof(float) : 0;
    // TempDh RMW buffer when deterministic!=0 (aic * tileM * h).
    uint64_t tempDh = deterministic ? static_cast<uint64_t>(aicNum) * tileM * h * sizeof(float) : 0;
    // ZFull fp32: Pass2 remat must match Pass1 float LSE / golden bmm(out_dtype=fp32).
    uint64_t zFull = static_cast<uint64_t>(zStripes) * tileM * v * sizeof(float);
    uint64_t dZ = static_cast<uint64_t>(aicNum) * kStages * tileM * kTileN * elementBytes;
    return ((logit + dhAcc + partial + statePartial + tempDh + zFull + dZ) + 511u) & ~static_cast<uint64_t>(511);
}
} // namespace

static ge::graphStatus TilingChunkLoss(gert::TilingContext* context)
{
    auto hidden = context->GetInputShape(0);
    auto weight = context->GetInputShape(1);
    auto hiddenDesc = context->GetInputDesc(0);
    auto weightDesc = context->GetInputDesc(1);
    auto labelDesc = context->GetInputDesc(2);
    auto lossDesc = context->GetOutputDesc(0);
    auto maxDesc = context->GetOutputDesc(1);
    auto sumDesc = context->GetOutputDesc(2);
    auto dhDesc = context->GetOutputDesc(3);
    if (hidden == nullptr || weight == nullptr || hiddenDesc == nullptr || weightDesc == nullptr ||
        labelDesc == nullptr || lossDesc == nullptr || maxDesc == nullptr || sumDesc == nullptr || dhDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto hDtype = hiddenDesc->GetDataType();
    auto wDtype = weightDesc->GetDataType();
    if ((hDtype != ge::DT_FLOAT16 && hDtype != ge::DT_BF16) || hDtype != wDtype) {
        OP_LOGE(context->GetNodeName(), "hidden and weight must both be float16 or bfloat16");
        return ge::GRAPH_FAILED;
    }
    if (labelDesc->GetDataType() != ge::DT_INT64 || lossDesc->GetDataType() != ge::DT_FLOAT ||
        maxDesc->GetDataType() != ge::DT_FLOAT || sumDesc->GetDataType() != ge::DT_FLOAT ||
        dhDesc->GetDataType() != hDtype) {
        OP_LOGE(context->GetNodeName(), "dtype mismatch on labels/loss/max/sum/dH");
        return ge::GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float* alpha = attrs->GetAttrPointer<float>(0);
    const int64_t* ignore = attrs->GetAttrPointer<int64_t>(1);
    if (alpha == nullptr || ignore == nullptr || !std::isfinite(*alpha) || *alpha <= 0.0f) {
        OP_LOGE(context->GetNodeName(), "alpha must be finite and > 0");
        return ge::GRAPH_FAILED;
    }

    uint32_t m = static_cast<uint32_t>(hidden->GetOriginShape().GetDim(0));
    uint32_t h = static_cast<uint32_t>(hidden->GetOriginShape().GetDim(1));
    uint32_t v = static_cast<uint32_t>(weight->GetOriginShape().GetDim(0));
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aic = static_cast<uint32_t>(platform.GetCoreNumAic());
    if (aic == 0) {
        aic = 1;
    }
    // Cap AIC to hardware Mix block count on 950PR (idle blocks hurt CrossCore).
    constexpr uint32_t kAicCap950 = 28u;
    if (aic > kAicCap950) {
        aic = kAicCap950;
    }
    // GetDeterministic→mode1 AtomicAdd+FIX (nSplit kept). rem-V/short-M force TempDh below.
    uint32_t deterministic = context->GetDeterministic() != 0 ? 1u : 0u;
    const uint32_t aicFull = aic;
    ScenarioCfg scen = SelectScenario(m, h, v, aic, deterministic);
    uint32_t tileM = scen.tileM;
    uint32_t nSplit = scen.nSplit < 1u ? 1u : scen.nSplit;
    deterministic = scen.deterministic;
    uint32_t mTiles = (m + tileM - 1u) / tileM;
    if (mTiles == 0) {
        mTiles = 1;
    }
    uint32_t mTiles128 = (m + kTileMDefault - 1u) / kTileMDefault;
    if (mTiles128 == 0) {
        mTiles128 = 1;
    }
    uint32_t nTilesV = (v + kTileN - 1u) / kTileN;
    // rem-V / 短 M 场景：强制 TempDh + nSplit=1，保证 dH 累加正确。
    constexpr uint32_t kShortMNsplitMinN = 8u; // 与 SelectScenario 中 kShortMMaxSplit 一致
    const bool remVTempKeep = (nSplit <= 1u && nTilesV > 1u && mTiles128 < aicFull &&
                               ((v % kTileN) != 0u || mTiles128 <= 1u));
    const bool shortMForceTempDh = (nSplit > 1u && mTiles128 <= 1u && nTilesV >= kShortMNsplitMinN);
    if (remVTempKeep || shortMForceTempDh) {
        deterministic = 1u;
        nSplit = 1u;
        scen.l0cToUb = false;
    }
    // Vector float dH 路径需要 nSplit=1 与 GM Pass2。
    constexpr uint64_t kCubeOnlyMV = 2000000000ull;
    if (static_cast<uint64_t>(m) * static_cast<uint64_t>(v) < kCubeOnlyMV) {
        nSplit = 1u;
        scen.l0cToUb = false;
    }
    uint32_t want = (nSplit > 1u) ? (mTiles * nSplit) : mTiles;
    if (aic > want) {
        aic = want;
    }
    if (aic == 0) {
        aic = 1;
    }

    ChunkLossTilingData data;
    data.set_m(m);
    data.set_h(h);
    data.set_v(v);
    data.set_aicNum(aic);
    data.set_nSplit(nSplit);
    data.set_deterministic(deterministic);
    data.set_tileM(tileM);
    data.set_alpha(*alpha);
    data.set_ignoreIndex(*ignore);
    {
        void* syncAddr = nullptr;
        if (aclrtGetHardwareSyncAddr(&syncAddr) == ACL_SUCCESS && syncAddr != nullptr) {
            data.set_syncBaseAddr(reinterpret_cast<uint64_t>(syncAddr));
        } else {
            data.set_syncBaseAddr(0ull);
        }
    }
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    context->SetBlockDim(aic);
    // Per-dtype .o (fp16/bf16) each only registers tilingKey 0; dtype is selected by binary.
    // Setting key=1 for bf16 makes aclrtBinaryGetFunctionByEntry miss the kernel.
    context->SetTilingKey(0);

    auto compile = reinterpret_cast<const ChunkLossCompileInfo*>(context->GetCompileInfo());
    int64_t sys = compile == nullptr ? 16 * 1024 * 1024 : compile->sysWorkspaceSize;
    uint32_t elemBytes = 2u;
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(sys) +
                   static_cast<size_t>(UserWorkspace(aic, h, v, elemBytes, deterministic, tileM, m, nSplit));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseChunkLoss(gert::TilingParseContext* context)
{
    auto compile = context->GetCompiledInfo<ChunkLossCompileInfo>();
    if (compile == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compile->aicNum = static_cast<uint32_t>(platform.GetCoreNumAic());
    compile->sysWorkspaceSize = static_cast<int64_t>(platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkLoss).Tiling(TilingChunkLoss).TilingParse<ChunkLossCompileInfo>(TilingParseChunkLoss);
} // namespace optiling
