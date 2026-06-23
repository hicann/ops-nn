/**
 * Copyright (c) 2026 Huawei Technologies
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant_tiling.cpp
 * \brief Tiling implementation for SwiGLU Group Quant operator
 */

#include "register/op_def_registry.h"
#include "swiglu_group_quant_tiling_utils.h"

namespace optiling {

constexpr uint32_t BATCH_MODE = 1;

inline static ge::graphStatus SetTilingDataForSwigluGroupQuant(gert::TilingContext *context,
    SwigluGroupQuantTilingData &tilingData)
{
    OP_LOGD(context, "SetTilingDataForSwigluGroupQuant start.");
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGD(context, "SetTilingDataForSwigluGroupQuant end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetCompileInfo(gert::TilingContext *context, SwigluGroupQuantCompileInfo &compileInfo)
{
    OP_LOGD(context, "GetCompileInfo start.");
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatform);

    if (totalCoreNum == 0 || ubSize <= 0) {
        OP_LOGD(context, "GetCompileInfo Failed, coreNum:%u, ubSize:%u.", totalCoreNum, ubSize);
        return ge::GRAPH_FAILED;
    }
    compileInfo.totalCore = totalCoreNum;
    compileInfo.ubSize = ubSize;
    OP_LOGD(context, "GetCompileInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetTillingData(gert::TilingContext *context, SwigluGroupQuantCompileInfo &compileInfo,
    SwigluGroupQuantTilingParam &tilingParam, SwigluGroupQuantTilingData &tilingData)
{
    OP_LOGD(context, "GetTillingData start.");

    // Variables to hold parsed info
    uint32_t totalTokens = 0;
    uint32_t dim2H = 0;
    uint32_t dimH = 0;
    bool hasWeight = false;
    bool isGroup = false;
    uint32_t groupNum = 0;
    std::vector<int64_t> groupTokens;
    bool hasClamp = false;
    bool outputOrigin = false;
    float clampLimit = CLAMP_LIMIT_DEFAULT;
    float dstTypeMaxFinite = DST_TYPE_MAX_FINITE_DEFAULT;

    // Check all parameters
    if (CheckSwigluGroupQuantOpParams(context, totalTokens, dim2H, dimH, hasWeight, isGroup, groupNum,
                      groupTokens, hasClamp, outputOrigin, clampLimit, dstTypeMaxFinite) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Calculate tile tokens
    CalculateTileTokens(compileInfo.ubSize, dimH, hasWeight, outputOrigin, tilingParam);

    // Calculate core distribution
    CalculateCoreDistribution(compileInfo.totalCore, totalTokens, isGroup, outputOrigin,
        groupNum, groupTokens, tilingParam);

    // Set tiling data
    SetSwigluGroupQuantTilingData(tilingParam, totalTokens, dim2H, dimH, hasWeight, isGroup, hasClamp,
        outputOrigin, clampLimit, dstTypeMaxFinite, tilingData);

    OP_LOGD(context, "GetTillingData end. usedCoreNum=%u, tileTokens=%u",
        tilingParam.usedCoreNum, tilingParam.tileTokens);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4SwigluGroupQuant(gert::TilingContext *context)
{
    OP_LOGD(context, "Tiling4SwigluGroupQuant start.");
    context->SetScheduleMode(BATCH_MODE);

    SwigluGroupQuantCompileInfo compileInfo;
    SwigluGroupQuantTilingParam tilingParam;
    SwigluGroupQuantTilingData tilingData;

    if (GetCompileInfo(context, compileInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (GetTillingData(context, compileInfo, tilingParam, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetTilingDataForSwigluGroupQuant(context, tilingData);

    context->SetBlockDim(tilingParam.usedCoreNum);

    // Set workspace: sysWorkspaceSize for library API + our data for coreMax
    size_t *workspaces = context->GetWorkspaceSizes(1);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaces[0] = sysWorkspaceSize + (tilingParam.usedCoreNum + SWIGLU_GROUP_QUANT_ONE) * SIZE_OF_FLOAT + BLOCK_SIZE;

    OP_LOGD(context, "Tiling4SwigluGroupQuant end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4SwigluGroupQuant(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4SwigluGroupQuant start and end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwigluGroupQuant)
    .Tiling(Tiling4SwigluGroupQuant)
    .TilingParse<SwigluGroupQuantCompileInfo>(TilingPrepare4SwigluGroupQuant);

} // namespace optiling
