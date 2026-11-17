/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include <algorithm>
#include <limits>
#include <cmath>
namespace optiling {
struct ChunkLossGradCompileInfo {};
// No compile-time JSON attributes: all resources come from the runtime platform.
static ge::graphStatus ParseChunkLossGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
inline int64_t GetChunkLossGradIgnoreIndex(gert::TilingContext* context)
{
    const auto* attrs = context->GetAttrs();
    const auto* value = attrs ? attrs->GetInt(1) : nullptr;
    return value ? *value : -100;
}
inline float GetChunkLossGradAlpha(gert::TilingContext* context)
{
    const auto* attrs = context->GetAttrs();
    const auto* value = attrs ? attrs->GetFloat(0) : nullptr;
    return value ? *value : 0.0f;
}
// Each MIX group owns complete [Vt,H] output rows and reduces N locally.
template <class TilingData, uint32_t TOKEN_TILE, uint32_t VOCAB_TILE>
ge::graphStatus BuildChunkLossGradTiling(gert::TilingContext* context)
{
    if (!context || !context->GetPlatformInfo() || !context->GetInputShape(0) || !context->GetInputShape(1) ||
        !context->GetInputDesc(0) || !context->GetInputDesc(1) || !context->GetRawTilingData()) {
        return ge::GRAPH_FAILED;
    }
    const float alpha = GetChunkLossGradAlpha(context);
    if (!std::isfinite(alpha) || alpha <= 0.0f) {
        return ge::GRAPH_FAILED;
    }
    const auto& hiddenShape = context->GetInputShape(0)->GetStorageShape();
    const auto& weightShape = context->GetInputShape(1)->GetStorageShape();
    if (hiddenShape.GetDimNum() != 2 || weightShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    int64_t numTokens = hiddenShape[0], hiddenDim = hiddenShape[1], vocabSize = weightShape[0];
    if (numTokens <= 0 || hiddenDim <= 0 || vocabSize <= 0 || weightShape[1] != hiddenDim || numTokens > INT32_MAX ||
        hiddenDim > INT32_MAX || vocabSize > INT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetInputDesc(0)->GetDataType();
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) || context->GetInputDesc(1)->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }
    for (int i = 2; i < 6; ++i) {
        auto shape = context->GetInputShape(i);
        auto desc = context->GetInputDesc(i);
        if (!shape || !desc || shape->GetStorageShape().GetDimNum() != 1 ||
            shape->GetStorageShape()[0] != (i == 5 ? 1 : numTokens) ||
            desc->GetDataType() != (i == 2 ? ge::DT_INT64 : ge::DT_FLOAT)) {
            return ge::GRAPH_FAILED;
        }
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t numCoreGroups = std::min<uint64_t>(platform.GetCoreNumAic(), (vocabSize + VOCAB_TILE - 1) / VOCAB_TILE);
    if (!numCoreGroups || platform.GetCoreNumAiv() < numCoreGroups * 2) {
        return ge::GRAPH_FAILED;
    }
    TilingData tilingData;
    tilingData.set_alpha(alpha);
    tilingData.set_reserved(0);
    tilingData.set_ignoreIndex(GetChunkLossGradIgnoreIndex(context));
    tilingData.set_numTokens(numTokens);
    tilingData.set_hiddenDim(hiddenDim);
    tilingData.set_vocabSize(vocabSize);
    tilingData.set_coreNum(numCoreGroups);
    tilingData.set_tokenTile(TOKEN_TILE);
    tilingData.set_vocabTile(VOCAB_TILE);
    tilingData.set_vectorChunk(4096);
    auto align512 = [](uint64_t value) { return (value + 511) / 512 * 512; };
    const uint64_t logitsBytes = align512(uint64_t(TOKEN_TILE) * VOCAB_TILE * 4);
    const uint64_t logitsGradBytes = align512(uint64_t(TOKEN_TILE) * VOCAB_TILE * 2);
    const uint64_t partialBytes = align512(uint64_t(VOCAB_TILE) * hiddenDim * 4);
    const uint64_t perCore = logitsBytes + logitsGradBytes + 2 * partialBytes;
    const uint64_t systemWorkspaceBytes = platform.GetLibApiWorkSpaceSize();
    if (perCore > (std::numeric_limits<uint64_t>::max() - systemWorkspaceBytes) / numCoreGroups) {
        return ge::GRAPH_FAILED;
    }
    tilingData.set_logitsBytes(logitsBytes);
    tilingData.set_logitsGradBytes(logitsGradBytes);
    tilingData.set_partialBytes(partialBytes);
    tilingData.set_coreWorkspaceBytes(perCore);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    // Two float VECIN queues and one half/bf16 VECOUT queue.
    if (ubBytes < 4096 * (4 + 4 + 2) + 8192) {
        return ge::GRAPH_FAILED;
    }
    auto matmulDtype = dtype == ge::DT_BF16 ? matmul_tiling::DataType::DT_BFLOAT16 :
                                              matmul_tiling::DataType::DT_FLOAT16;
    auto makeMatmul = [&](TCubeTiling& cubeTiling, bool transA, bool transB, int rows, int columns, int reductionSize) {
        matmul_tiling::MultiCoreMatmulTiling matmulTiling(platform);
        matmulTiling.SetDim(1);
        matmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, transA);
        matmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, transB);
        matmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                              matmul_tiling::DataType::DT_FLOAT);
        matmulTiling.SetBias(false);
        matmulTiling.SetShape(rows, columns, reductionSize);
        matmulTiling.SetOrgShape(rows, columns, reductionSize);
        matmulTiling.SetSingleShape(rows, columns, reductionSize);
        return matmulTiling.GetTiling(cubeTiling) >= 0;
    };
    if (!makeMatmul(tilingData.logitsTiling, false, true, TOKEN_TILE, VOCAB_TILE, hiddenDim) ||
        !makeMatmul(tilingData.weightTiling, true, false, VOCAB_TILE, hiddenDim, TOKEN_TILE)) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(numCoreGroups);
    // DTYPE_HIDDEN is selected by the fp16/bf16 binary; both use the default kernel key.
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = perCore * numCoreGroups + systemWorkspaceBytes;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
