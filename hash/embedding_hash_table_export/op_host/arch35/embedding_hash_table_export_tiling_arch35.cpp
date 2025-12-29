/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file embedding_hash_table_export_tiling_arch35.cpp
 * \brief
 */
#include <vector>
#include "embedding_hash_table_export_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/platform_util.h"

namespace optiling {
using namespace Ops::Base;

constexpr int64_t MAX_CORE_NUM = 64;
constexpr int64_t MAX_THREAD_NUM = 256;
constexpr int64_t BUFFER_NUM = 6;

constexpr int64_t OUTPUT_VALUES_IDX = 3;
constexpr int64_t INPUT_IDX_0 = 0;
constexpr int64_t INPUT_IDX_1 = 1;
constexpr int64_t INPUT_IDX_2 = 2;
constexpr int64_t INPUT_IDX_3 = 3;

constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;

// values support datatype
static const std::unordered_map<ge::DataType, int64_t> SUPPORT_VALUES_DATA_TYPE{
    { ge::DataType::DT_FLOAT, 4 },
};

ge::graphStatus TilingForEmbeddingHashTableExport(gert::TilingContext *context)
{
    const auto* compileInfo = reinterpret_cast<const EmbeddingHashTableExportCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    int64_t maxThreadNum = std::min(compileInfo->maxThreadNum, MAX_THREAD_NUM);
    int64_t maxCoreNum = std::min(compileInfo->coreNumAiv, MAX_CORE_NUM);

    auto const tableHandles = context->GetInputShape(INPUT_IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableHandles);
    int64_t tableHandlesShapeVal = tableHandles->GetStorageShape().GetShapeSize();
    auto const tableSizes = context->GetInputShape(INPUT_IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableSizes);
    int64_t tableSizesShapeVal = tableSizes->GetStorageShape().GetShapeSize();
    auto const embeddingDims = context->GetInputShape(INPUT_IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDims);
    int64_t embeddingDimsShapeVal = embeddingDims->GetStorageShape().GetShapeSize();
    auto const bucketSizes = context->GetInputShape(INPUT_IDX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context, bucketSizes);
    int64_t bucketSizesShapeVal = bucketSizes->GetStorageShape().GetShapeSize();
    if (!(tableHandlesShapeVal == tableSizesShapeVal && tableSizesShapeVal == embeddingDimsShapeVal &&
        embeddingDimsShapeVal == bucketSizesShapeVal)) {
        OP_LOGE(context->GetNodeName(), "Input shapeSize is not same. Please check.");
        return ge::GRAPH_FAILED;
    }

    EmbeddingHashTableExportTilingData tiling;
    tiling.set_maxThreadNum(maxThreadNum);
    tiling.set_maxCoreNum(maxCoreNum);
    tiling.set_tableNum(tableHandlesShapeVal);
    tiling.set_bitWidth(BUFFER_NUM);

    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *exportMode = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, exportMode);
    if (strncmp(exportMode, "new", sizeof("new") / sizeof(char)) == 0) {
        tiling.set_exportMode(1);
    } else {
        tiling.set_exportMode(0);
    }

    const uint8_t *filteredExportFlag = attrs->GetAttrPointer<uint8_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, filteredExportFlag);
    tiling.set_filteredExportFlag(*filteredExportFlag);

    context->SetTilingKey(tiling.get_bitWidth());
    context->SetBlockDim(maxCoreNum);
    context->SetLocalMemorySize(sizeof(int64_t) * maxThreadNum * BUFFER_NUM);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);

    int64_t coreSyncWorkspace = maxCoreNum * 8;
    workspace[0] = ASCENDC_TOOLS_WORKSPACE + coreSyncWorkspace * tableHandlesShapeVal;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForEmbeddingHashTableExport(gert::TilingParseContext *context)
{
    // 获取platformInfo
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // 获取compileInfo
    auto compileInfo = context->GetCompiledInfo<EmbeddingHashTableExportCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    // 设置maxThread和coreNum
    compileInfo->maxThreadNum = GetSimtMaxThreadNum(context);
    OP_CHECK_IF((compileInfo->maxThreadNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get thread num."),
                 return ge::GRAPH_FAILED);
    compileInfo->coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNumAiv <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                 return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(EmbeddingHashTableExport)
    .Tiling(TilingForEmbeddingHashTableExport)
    .TilingParse<EmbeddingHashTableExportCompileInfo>(TilingPrepareForEmbeddingHashTableExport);
} // namespace optiling