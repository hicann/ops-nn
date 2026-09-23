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
 * \file turbo_quant_tiling.cpp
 * \brief
 */
#include "turbo_quant_tiling.h"

#include <limits>
#include <string>

#include "graph/utils/type_utils.h"
#include "util/shape_util.h"

namespace optiling {
constexpr size_t TQ_LATENT_DIM_NUM = 2;
constexpr size_t TQ_DIM_TOKEN = 0;
constexpr size_t TQ_DIM_HEAD = 1;

static ge::graphStatus TilingFuncForTurboQuant(gert::TilingContext* context)
{
    OP_LOGD(context, "Begin to do TilingForTurboQuant");
    const gert::StorageShape* latentShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, latentShape);
    const gert::StorageShape* centroidsShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, centroidsShape);
    const auto* latentDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, latentDesc);
    const auto* centroidsDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, centroidsDesc);

    if (latentDesc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "latent",
                                  ge::TypeUtils::DataTypeToSerialString(latentDesc->GetDataType()).c_str(), "DT_FLOAT");
        return ge::GRAPH_FAILED;
    }
    if (centroidsDesc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "centroids",
                                  ge::TypeUtils::DataTypeToSerialString(centroidsDesc->GetDataType()).c_str(),
                                  "DT_FLOAT");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& latent = latentShape->GetStorageShape();
    if (latent.GetDimNum() != TQ_LATENT_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "latent", std::to_string(latent.GetDimNum()).c_str(),
                                     std::to_string(TQ_LATENT_DIM_NUM).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t numTokens = latent.GetDim(TQ_DIM_TOKEN);
    int64_t headDim = latent.GetDim(TQ_DIM_HEAD);
    if (numTokens < 0) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "numTokens", std::to_string(numTokens).c_str(),
                                  "non-negative");
        return ge::GRAPH_FAILED;
    }
    if (headDim != TQ_SUPPORTED_HEAD_DIM) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "headDim", std::to_string(headDim).c_str(),
                                  std::to_string(TQ_SUPPORTED_HEAD_DIM).c_str());
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& centroids = centroidsShape->GetStorageShape();
    if (centroids.GetDimNum() != 1 || centroids.GetDim(0) != TQ_N_CENT) {
        OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "centroids", Ops::Base::ToString(centroids).c_str(),
                                  ("[" + std::to_string(TQ_N_CENT) + "]").c_str());
        return ge::GRAPH_FAILED;
    }

    if (numTokens > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "numTokens", std::to_string(numTokens).c_str(),
                                  std::to_string(std::numeric_limits<uint32_t>::max()).c_str());
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum < 1) {
        coreNum = 1;
    }

    // A batch of tokens is fully resident in UB, so the split across cores is purely over tokens.
    uint64_t tokens = static_cast<uint64_t>(numTokens < 1 ? 1 : numTokens);
    uint64_t tokensPerCore = (tokens + static_cast<uint64_t>(coreNum) - 1) / coreNum;
    uint64_t blockDim = (tokens + tokensPerCore - 1) / tokensPerCore;

    int64_t packedBytes = headDim / TQ_ELEMENTS_PER_BYTE;
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                              "failed to query the UB size");
        return ge::GRAPH_FAILED;
    }
    int64_t budget = static_cast<int64_t>(ubSize) - TqFixedBytes(headDim) - TQ_UB_RESERVE;
    int64_t bytesPerToken = TqBytesPerToken(headDim, packedBytes);
    int64_t maxByUb = (budget > 0 && bytesPerToken > 0) ? budget / bytesPerToken : 1;

    // Batching only pays off up to what a core actually owns; below that it would idle cores instead.
    int64_t tokensPerBatch = static_cast<int64_t>(tokensPerCore);
    if (tokensPerBatch > maxByUb) {
        tokensPerBatch = maxByUb;
    }
    if (tokensPerBatch > TQ_MAX_TOKENS_PER_BATCH) {
        tokensPerBatch = TQ_MAX_TOKENS_PER_BATCH;
    }
    if (tokensPerBatch < 1) {
        tokensPerBatch = 1;
    }

    TurboQuantTilingData tilingData;
    tilingData.set_numTokens(static_cast<uint32_t>(numTokens));
    tilingData.set_tokensPerCore(static_cast<uint32_t>(tokensPerCore));
    tilingData.set_headDim(static_cast<uint32_t>(headDim));
    tilingData.set_packedBytes(static_cast<uint32_t>(packedBytes));
    tilingData.set_tokensPerBatch(static_cast<uint32_t>(tokensPerBatch));

    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "tilingData", std::to_string(tilingData.GetDataSize()).c_str(),
            ("exceeds the tiling capacity " + std::to_string(context->GetRawTilingData()->GetCapacity())).c_str());
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_LOGD(context, "End to do TilingForTurboQuant");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForTurboQuant(gert::TilingParseContext* /* context */) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(TurboQuant)
    .Tiling(TilingFuncForTurboQuant)
    .TilingParse<TurboQuantCompileInfo>(TilingPrepareForTurboQuant);
} // namespace optiling
