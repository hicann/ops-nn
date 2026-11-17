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
 * \file chunk_loss_tiling_arch22.cpp
 * \brief ChunkLoss arch22 Host tiling 实现
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
constexpr uint32_t kTileM = 128;
constexpr uint32_t kTileN = 256;
constexpr uint32_t kTileH = 256;
constexpr uint32_t kStages = 2;
constexpr uint32_t kPartialStride = 8;

uint64_t UserWorkspace(uint32_t aicNum, uint32_t h, uint32_t v, uint32_t elementBytes)
{
    // r023/r024: TempDh retired; Pass1 logit staging is Element.
    // Match kernel: ShortV uses logitAsZ (M>1) or rematZ (M=1) — both skip ZFull.
    uint32_t nTiles = (v + kTileN - 1u) / kTileN;
    uint32_t vPad = (nTiles > 0u && nTiles <= kStages) ? 0u : (((v + kTileN - 1u) / kTileN) * kTileN);
    uint64_t logit = static_cast<uint64_t>(aicNum) * kStages * kTileM * kTileN * elementBytes;
    // ×kStages: Pass2 per-N-tile dhAcc banks (kernel merges at cast; avoids AtomicAdd FFTS race).
    uint64_t dhAcc = static_cast<uint64_t>(aicNum) * kStages * kTileM * h;
    uint64_t partial = static_cast<uint64_t>(aicNum) * kStages * kPartialStride;
    uint64_t zFull = static_cast<uint64_t>(aicNum) * kTileM * vPad * elementBytes;
    uint64_t dZ = static_cast<uint64_t>(aicNum) * kStages * kTileM * kTileN * elementBytes;
    return ((logit + (dhAcc + partial) * sizeof(float) + zFull + dZ) + 511u) & ~static_cast<uint64_t>(511);
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
    uint32_t mTiles = (m + kTileM - 1) / kTileM;
    if (aic > mTiles) {
        aic = mTiles;
    }

    ChunkLossTilingData data;
    data.set_m(m);
    data.set_h(h);
    data.set_v(v);
    data.set_aicNum(aic);
    data.set_nSplit(1u);
    data.set_deterministic(0u);
    data.set_tileM(kTileM);
    data.set_alpha(*alpha);
    data.set_ignoreIndex(*ignore);
    {
        void* syncAddr = nullptr;
        // SoftSync Launch uses this addr; ACLNN FFTS alone breaks AIV→AIC Wait (empty dZ0).
        if (aclrtGetHardwareSyncAddr(&syncAddr) == ACL_SUCCESS && syncAddr != nullptr) {
            data.set_syncBaseAddr(reinterpret_cast<uint64_t>(syncAddr));
        } else {
            data.set_syncBaseAddr(0ull);
        }
    }
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    context->SetBlockDim(aic);
    // Per-dtype opc binaries already select the kernel; keep key 0 so
    // BinaryGetFunctionByEntry matches the single entry in each .o.
    context->SetTilingKey(0);

    auto compile = reinterpret_cast<const ChunkLossCompileInfo*>(context->GetCompileInfo());
    int64_t sys = compile == nullptr ? 16 * 1024 * 1024 : compile->sysWorkspaceSize;
    uint32_t elemBytes = hDtype == ge::DT_BF16 ? 2u : 2u;
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(sys) + static_cast<size_t>(UserWorkspace(aic, h, v, elemBytes));
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
