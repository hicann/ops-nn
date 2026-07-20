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
 * \file masked_scatter_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/masked_scatter_tiling_data.h"
#include "../op_kernel/masked_scatter_tiling_key.h"

namespace optiling {
using namespace Ops::NN::OpTiling;
static constexpr const char* OP_NAME = "MaskedScatter";
static constexpr size_t WORKSPACE_SIZE = 0;

struct MaskedScatterCompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* tilingContext, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = tilingContext->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(OP_NAME, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取shape信息
static ge::graphStatus GetShapeInfo(gert::TilingContext* tilingContext, int64_t& numElemX, int64_t& numElemUpdates)
{
    auto inputX = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputX);
    numElemX = inputX->GetStorageShape().GetShapeSize();

    auto inputUpdates = tilingContext->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputUpdates);
    numElemUpdates = inputUpdates->GetStorageShape().GetShapeSize();

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* tilingContext)
{
    size_t* workspaceSizes = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, workspaceSizes);
    workspaceSizes[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* tilingContext)
{
    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_UINT8, ge::DT_INT8,
                                                   ge::DT_INT16, ge::DT_INT32,   ge::DT_BOOL,  ge::DT_BF16};
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(OP_NAME, "invalid dtype");
        return ge::GRAPH_FAILED;
    }

    // mask dtype校验
    auto maskDesc = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, maskDesc);
    if (maskDesc->GetDataType() != ge::DT_BOOL) {
        OP_LOGE(OP_NAME, "mask dtype must be bool");
        return ge::GRAPH_FAILED;
    }

    // x, updates, y dtype一致性校验
    auto updatesDesc = tilingContext->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, updatesDesc);
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    if (dataType != updatesDesc->GetDataType() || dataType != outputDesc->GetDataType()) {
        OP_LOGE(OP_NAME, "x, updates and y must have the same dtype");
        return ge::GRAPH_FAILED;
    }

    // shape校验
    auto xShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xShape);
    auto maskShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, maskShape);
    auto yShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yShape);
    if (xShape->GetStorageShape() != maskShape->GetStorageShape() ||
        xShape->GetStorageShape() != yShape->GetStorageShape()) {
        OP_LOGE(OP_NAME, "x, mask and y must have the same shape");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus MaskedScatterTilingFunc(gert::TilingContext* tilingContext)
{
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingContext);
    // 1. platform
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(tilingContext, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(OP_NAME, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);

    // 2. shapes & dtype
    int64_t numElemX = 0;
    int64_t numElemUpdates = 0;
    OP_CHECK_IF(GetShapeInfo(tilingContext, numElemX, numElemUpdates) != ge::GRAPH_SUCCESS,
                OP_LOGE(OP_NAME, "GetShapeInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAttrsInfo(tilingContext) != ge::GRAPH_SUCCESS, OP_LOGE(OP_NAME, "GetShapeAttrsInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetWorkspaceSize(tilingContext) != ge::GRAPH_SUCCESS, OP_LOGE(OP_NAME, "SetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 3. write tiling data
    MaskedScatterTilingData* tiling = tilingContext->GetTilingData<MaskedScatterTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(MaskedScatterTilingData), 0, sizeof(MaskedScatterTilingData)) != EOK,
                OP_LOGE(OP_NAME, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->numElemX = numElemX;
    tiling->numElemMask = numElemX;
    tiling->numElemUpdates = numElemUpdates;
    tiling->tilingCoreNum = coreNum;

    tilingContext->SetBlockDim(static_cast<uint32_t>(coreNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMaskedScatter([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaskedScatter)
    .Tiling(MaskedScatterTilingFunc)
    .TilingParse<MaskedScatterCompileInfo>(TilingParseForMaskedScatter);
} // namespace optiling
