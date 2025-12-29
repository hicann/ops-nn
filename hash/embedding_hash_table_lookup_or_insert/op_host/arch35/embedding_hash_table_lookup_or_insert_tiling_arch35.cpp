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
 * \file embedding_hash_table_lookup_or_insert_tiling_arch35.cpp
 * \brief embedding_hash_table_lookup_or_insert_tiling
 */
#include <vector>
#include "embedding_hash_table_lookup_or_insert_tiling_arch35.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "tiling/tiling_api.h"

namespace optiling {
using namespace Ops::Base;

constexpr uint32_t MAX_THREAD = 1024;

constexpr uint32_t INPUT_BUCKET_SIZE_IDX = 0;
constexpr uint32_t INPUT_EMBEDDINGDIM_IDX = 1;
constexpr uint32_t INPUT_FILTER_MODE_IDX = 2;
constexpr uint32_t INPUT_FILTER_FREQ_IDX = 3;
constexpr uint32_t INPUT_DEFAULT_KEY_OR_VALUE_IDX = 4;
constexpr uint32_t INPUT_DEFAULT_KEY_IDX = 5;
constexpr uint32_t INPUT_DEFAULT_VALUE_IDX = 6;
constexpr uint32_t INPUT_FILTER_KEY_FLAG_IDX = 7;
constexpr uint32_t INPUT_filter_KEY_IDX = 8;
constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16777216; // 16 * 1024 * 1024;

struct LookupOrInsertCompileInfo {
    uint32_t maxThread;
    uint32_t coreNum;
};

ge::graphStatus Tiling4LookupOrInsert(gert::TilingContext *context)
{
    const LookupOrInsertCompileInfo* compile_info = reinterpret_cast<const LookupOrInsertCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    LookupOtInsertTilingData tiling;
    // get the number of keys
    auto const inShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto const inShapeVal = inShape->GetStorageShape();
    int64_t keyNum = inShapeVal.GetShapeSize();
    tiling.set_keyNum(keyNum);

    auto keyDtypePtr = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDtypePtr);
    ge::DataType keyType = keyDtypePtr->GetDataType();
    if (keyType != ge::DT_INT64) {
        OP_LOGE(context->GetNodeName(), "Invalid datatype of input key, only support int64");
        return ge::GRAPH_FAILED;
    }
    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto *size = attrs->GetAttrPointer<int64_t>(INPUT_BUCKET_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, size);
    auto *embeddingDim = attrs->GetAttrPointer<int64_t>(INPUT_EMBEDDINGDIM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDim);
    auto *filterMode = attrs->GetAttrPointer<char>(INPUT_FILTER_MODE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterMode);
    auto *filterFreq = attrs->GetAttrPointer<int64_t>(INPUT_FILTER_FREQ_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterFreq);
    auto *defaultKeyOrValue = attrs->GetAttrPointer<bool>(INPUT_DEFAULT_KEY_OR_VALUE_IDX);
    auto *defaultKey = attrs->GetAttrPointer<int64_t>(INPUT_DEFAULT_KEY_IDX);
    auto *defaultValue = attrs->GetAttrPointer<float>(INPUT_DEFAULT_VALUE_IDX);
    auto *filterKeyFlag = attrs->GetAttrPointer<bool>(INPUT_FILTER_KEY_FLAG_IDX);
    auto *filterKey = attrs->GetAttrPointer<int64_t>(INPUT_filter_KEY_IDX);
    if (strcmp(filterMode, "counter") == 0) {
        OP_LOGI(context->GetNodeName(), "Filter mode takes effect");
        tiling.set_filterMode(1);
    } else {
        tiling.set_filterMode(0);
    }
    tiling.set_size(static_cast<int64_t>(*size));
    tiling.set_embeddingDim(static_cast<int64_t>(*embeddingDim));
    tiling.set_filterFreq(static_cast<int64_t>(*filterFreq));
    tiling.set_defaultKeyOrValue(defaultKeyOrValue == nullptr ? 0 : static_cast<uint32_t>(*defaultKeyOrValue));
    tiling.set_defaultKey(defaultKey == nullptr ? 0 : static_cast<int64_t>(*defaultKey));
    tiling.set_defaultValue(defaultValue == nullptr ? 0.0f : static_cast<float>(*defaultValue));
    tiling.set_filterKeyFlag(filterKeyFlag == nullptr ? 0 : static_cast<uint32_t>(*filterKeyFlag));
    tiling.set_filterKey(filterKey == nullptr ? -1 : static_cast<int64_t>(*filterKey));
    context->SetTilingKey(0);
    int64_t usedThread = std::min(compile_info->maxThread, MAX_THREAD);
    tiling.set_usedThread(usedThread);
    context->SetBlockDim(std::min(static_cast<uint32_t>(CeilDiv(keyNum, usedThread)), compile_info->coreNum));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4LookupOrInsert(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<LookupOrInsertCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->maxThread = GetSimtMaxThreadNum(context);
    OP_CHECK_IF((compileInfo->maxThread <= 0), OP_LOGE(context->GetNodeName(), "Failed to get thread num."),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EmbeddingHashTableLookupOrInsert)
    .Tiling(Tiling4LookupOrInsert)
    .TilingParse<LookupOrInsertCompileInfo>(TilingPrepare4LookupOrInsert);
} // namespace optiling