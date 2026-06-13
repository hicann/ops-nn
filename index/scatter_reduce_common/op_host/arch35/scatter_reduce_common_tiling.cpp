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
 * \file scatter_reduce_common_tiling.cpp
 * \brief Shared tiling for non-nd scatter reduce. Splits index entries across cores; per-index slices
 *        stay on one core (cross-core writes to the same var row are handled by the atomic kernel).
 */
#include "scatter_reduce_common_tiling.h"
#include "../../op_kernel/arch35/scatter_reduce_common_struct.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace optiling {
constexpr size_t VAR_IDX = 0;
constexpr size_t INDICES_IDX = 1;
constexpr size_t UPDATES_IDX = 2;
constexpr uint64_t SYS_WORKSPACE = 16UL * 1024UL * 1024UL;

ge::graphStatus TilingPrepareForScatterReduce(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ScatterReduceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

// coreNum 来源：kernel 路径用解析好的 compileInfo；aclnn runtime 不带 compileInfo，回退直接读 platform。
static ge::graphStatus ResolveCoreNum(gert::TilingContext* context, uint64_t& coreNum)
{
    coreNum = 1;
    auto compileInfo = reinterpret_cast<const ScatterReduceCompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr) {
        coreNum = (compileInfo->coreNum == 0) ? 1 : compileInfo->coreNum;
        return ge::GRAPH_SUCCESS;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterReduceCommonTiling(gert::TilingContext* context)
{
    auto varShapePtr = context->GetInputShape(VAR_IDX);
    auto indicesShapePtr = context->GetInputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShapePtr);
    auto& varShape = varShapePtr->GetStorageShape();
    auto& indicesShape = indicesShapePtr->GetStorageShape();

    uint64_t varFirstDim = (varShape.GetDimNum() == 0) ? 1 : varShape.GetDim(0);
    uint64_t varTotal = varShape.GetShapeSize();
    uint64_t sliceSize = (varFirstDim == 0) ? 0 : varTotal / varFirstDim;
    uint64_t indicesNum = indicesShape.GetShapeSize();

    uint64_t coreNum = 1;
    if (ResolveCoreNum(context, coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // split index entries across cores; each core handles whole slices, read directly from GM in the VF
    uint64_t blockNum = (indicesNum == 0) ? 1 : (indicesNum < coreNum ? indicesNum : coreNum);
    uint64_t perCoreIndices = (blockNum == 0) ? 0 : (indicesNum + blockNum - 1) / blockNum;
    blockNum = (perCoreIndices == 0) ? 1 : (indicesNum + perCoreIndices - 1) / perCoreIndices;
    uint64_t tailCoreIndices = indicesNum - (blockNum - 1) * perCoreIndices;

    auto* td = context->GetTilingData<ScatterReduceCommon::ScatterReduceSimtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    td->blockNum = blockNum;
    td->blockTilingSize = perCoreIndices * sliceSize;
    td->tailBlockTilingSize = tailCoreIndices * sliceSize;
    td->sliceSize = sliceSize;
    td->varFirstDim = varFirstDim;

    context->SetBlockDim(blockNum);
    context->SetTilingKey(0);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = SYS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
