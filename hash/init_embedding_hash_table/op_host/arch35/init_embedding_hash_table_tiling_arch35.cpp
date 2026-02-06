/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file init_embedding_hash_table_tiling_arch35.cpp
 * \brief InitEmbeddingHashTable tiling impl
 */
#include <vector>

#include "init_embedding_hash_table_tiling_arch35.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"

namespace optiling {
using namespace Ops::Base;

namespace {
constexpr uint32_t MAX_THREAD = 512;
constexpr uint64_t ADDR_ALING_FACTOR = 8;

constexpr uint32_t INPUT_TABLE_HANDLE_IDX = 0;
constexpr uint32_t OPTIONAL_INPUT_SAMPLED_VALUES_IDX = 1;
constexpr uint32_t REQUIRED_ATTR_BUCKET_SIZE_IDX = 0;
constexpr uint32_t REQUIRED_ATTR_EMBEDDING_DIM_IDX = 1;
constexpr uint32_t ATTR_INITIALIZER_MODE_IDX = 2;
constexpr uint32_t ATTR_CONSTANT_VALUE_IDX = 3;

constexpr uint32_t RANDOM_MODE = 0;
constexpr uint32_t CONSTANT_MODE = 1;

constexpr uint32_t BUCKET_HANDLE_SIZE = 1;
constexpr uint32_t BUCKET_COUNTER_SIZE = 1;
constexpr uint32_t BUCKET_FLAG_SIZE = 1;

constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16777216; // 16 * 1024 * 1024;
}  // namespace

ge::graphStatus Tiling4InitEmbeddingHashTable(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4InitEmbeddingHashTable is begin");
    const InitEmbeddingHashTableCompileInfo *compileInfo =
        reinterpret_cast<const InitEmbeddingHashTableCompileInfo *>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    InitEmbeddingHashTableTilingData tiling;

    auto *sampledValuesTensor = context->GetOptionalInputTensor(OPTIONAL_INPUT_SAMPLED_VALUES_IDX);
    auto sampledValuesDescPtr = context->GetOptionalInputDesc(OPTIONAL_INPUT_SAMPLED_VALUES_IDX);
    if (sampledValuesTensor != nullptr) {
        OP_CHECK_IF((sampledValuesDescPtr != nullptr) && (sampledValuesDescPtr->GetDataType() != ge::DT_FLOAT),
            OP_LOGE(context->GetNodeName(), "Current sampled_values only support float32."),
            return ge::GRAPH_FAILED);
    }

    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto *bucketSize = attrs->GetAttrPointer<int64_t>(REQUIRED_ATTR_BUCKET_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, bucketSize);
    OP_CHECK_IF(*bucketSize < 0, OP_LOGE(context->GetNodeName(), "InitEmbeddingHashTable bucketSize must >= 0."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InitEmbeddingHashTable bucketSize : %ld", static_cast<int64_t>(*bucketSize));

    auto *embeddingDim = attrs->GetAttrPointer<int64_t>(REQUIRED_ATTR_EMBEDDING_DIM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDim);
    OP_CHECK_IF(*embeddingDim < 0, OP_LOGE(context->GetNodeName(), "InitEmbeddingHashTable embeddingDim must >= 0."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InitEmbeddingHashTable embeddingDim : %ld", static_cast<int64_t>(*embeddingDim));

    auto *initializerMode = attrs->GetAttrPointer<char>(ATTR_INITIALIZER_MODE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, initializerMode);
    int64_t mode{0};
    float constValue{0};
    auto *constantValue = attrs->GetAttrPointer<float>(ATTR_CONSTANT_VALUE_IDX);
    if (strcmp(initializerMode, "random") == 0) {
        mode = RANDOM_MODE;
        auto sampledValuesShapePtr = context->GetInputShape(OPTIONAL_INPUT_SAMPLED_VALUES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, sampledValuesShapePtr);
        auto sampledValuesShape = sampledValuesShapePtr->GetStorageShape();
        int64_t sampledValuesShapeSize = sampledValuesShape.GetShapeSize();
        OP_CHECK_IF(static_cast<int64_t>(*embeddingDim) * static_cast<int64_t>(*bucketSize) != sampledValuesShapeSize,
            OP_LOGE(context->GetNodeName(),
                "InitEmbeddingHashTable in randdom mode, sampledValues shape should = bucketSize x embeddingDim."),
            return ge::GRAPH_FAILED);
    } else if (strcmp(initializerMode, "constant") == 0) {
        OP_CHECK_NULL_WITH_CONTEXT(context, constantValue);
        mode = CONSTANT_MODE;
        constValue = static_cast<float>(*constantValue);
    } else {
        OP_LOGE(context->GetNodeName(), "Invalid initializer mode, only support random and constant!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "InitEmbeddingHashTable mode : %ld", static_cast<int64_t>(mode));
    OP_LOGD(context->GetNodeName(), "InitEmbeddingHashTable constantValue : %f", static_cast<float>(*constantValue));

    int64_t bucketValueSize = CeilDiv((uint64_t)(*embeddingDim * sizeof(float)), ADDR_ALING_FACTOR);
    int64_t bucketLength = BUCKET_HANDLE_SIZE + BUCKET_COUNTER_SIZE + BUCKET_FLAG_SIZE + bucketValueSize;
    OP_LOGD(context->GetNodeName(), "InitEmbeddingHashTable bucketLength : %ld", static_cast<int64_t>(bucketLength));

    int64_t threadDim = std::min(static_cast<uint32_t>(compileInfo->maxThread), MAX_THREAD);
    int64_t numBlocks = CeilDiv(static_cast<int64_t>(*bucketSize), threadDim);
    numBlocks = std::min(numBlocks, compileInfo->coreNum);
    context->SetBlockDim(numBlocks);

    context->SetTilingKey(0);
    tiling.set_bucketSize(static_cast<int64_t>(*bucketSize));
    tiling.set_embeddingDim(static_cast<int64_t>(*embeddingDim));
    tiling.set_initializerMode(static_cast<int64_t>(mode));
    tiling.set_constantValue(static_cast<float>(constValue));
    tiling.set_bucketLength(static_cast<int64_t>(bucketLength));
    tiling.set_useThreadDim(static_cast<int64_t>(threadDim));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ASCENDC_TOOLS_WORKSPACE;
    OP_LOGD(context->GetNodeName(), "Tiling4InitEmbeddingHashTable is end");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4InitEmbeddingHashTable(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<InitEmbeddingHashTableCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "The core num is invaild."), return ge::GRAPH_FAILED);
    compileInfo->maxThread = GetSimtMaxThreadNum(context);
    OP_CHECK_IF((compileInfo->maxThread <= 0), OP_LOGE(context->GetNodeName(), "The Thread num is invaild."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InitEmbeddingHashTable)
    .Tiling(Tiling4InitEmbeddingHashTable)
    .TilingParse<InitEmbeddingHashTableCompileInfo>(TilingPrepare4InitEmbeddingHashTable);
}  // namespace optiling