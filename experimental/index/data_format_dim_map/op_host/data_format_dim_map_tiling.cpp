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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "register/op_def_registry.h"
#include <cstring>
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "exe_graph/runtime/tensor.h"
#include "../op_kernel/data_format_dim_map_tiling_data.h"
#include "../op_kernel/data_format_dim_map_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetInputInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    totalIdx = EnsureNotScalar(inputShapePtr->GetStorageShape()).GetShapeSize();
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BuildExpandedTable(gert::TilingContext* context,
    int32_t expandedTable[10], int32_t& formatLen)
{
    const char* srcFormat = context->GetAttrs()->GetStr(0);
    const char* dstFormat = context->GetAttrs()->GetStr(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcFormat);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstFormat);

    formatLen = static_cast<int32_t>(strlen(srcFormat));
    OP_CHECK_IF(formatLen < 1 || formatLen > 5,
        OP_LOGE(context, "Invalid formatLen: %d", formatLen), return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int32_t>(strlen(dstFormat)) != formatLen,
        OP_LOGE(context, "src_format and dst_format length mismatch"), return ge::GRAPH_FAILED);

    for (int32_t i = 0; i < formatLen; i++) {
        bool found = false;
        for (int32_t j = 0; j < formatLen; j++) {
            if (dstFormat[j] == srcFormat[i]) {
                expandedTable[i] = j;
                found = true;
                break;
            }
        }
        OP_CHECK_IF(!found,
            OP_LOGE(context, "srcFormat[%d]='%c' not found in dstFormat", i, srcFormat[i]),
            return ge::GRAPH_FAILED);
    }
    for (int32_t i = 0; i < formatLen; i++) {
        expandedTable[formatLen + i] = expandedTable[i];
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DataFormatDimMapTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(GetInputInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetInputInfo error"), return ge::GRAPH_FAILED);

    int32_t expandedTable[10] = {0};
    int32_t formatLen = 0;
    OP_CHECK_IF(BuildExpandedTable(context, expandedTable, formatLen) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "BuildExpandedTable error"), return ge::GRAPH_FAILED);

    int64_t typeSize = (dataType == ge::DT_INT64) ? 8 : 4;
    int64_t ubBlockSize = GetUbBlockSize(context);

    DataFormatDimMapTilingData* tiling = context->GetTilingData<DataFormatDimMapTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(DataFormatDimMapTilingData), 0,
        sizeof(DataFormatDimMapTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t bytesPerElement = 4 * typeSize + 4 * 4 + 1;
    tiling->ubFactor = FloorAlign(static_cast<int64_t>(ubSize) / bytesPerElement, ubBlockSize);
    tiling->formatLen = formatLen;
    for (int i = 0; i < 2 * formatLen && i < 10; i++) {
        tiling->expandedTable[i] = expandedTable[i];
    }

    context->SetBlockDim(CeilDiv(totalIdx, tiling->blockFactor));
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDataFormatDimMap([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct DataFormatDimMapCompileInfo {};

IMPL_OP_OPTILING(DataFormatDimMap)
    .Tiling(DataFormatDimMapTilingFunc)
    .TilingParse<DataFormatDimMapCompileInfo>(TilingParseForDataFormatDimMap);

} // namespace optiling
