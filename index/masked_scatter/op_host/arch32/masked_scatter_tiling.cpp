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
#include "masked_scatter_tiling.h"
#include "../../op_kernel/masked_scatter_tiling_data.h"
#include "../../op_kernel/masked_scatter_tiling_key.h"

namespace optiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t MASK_INDEX = 1;
constexpr uint32_t UPDATES_INDEX = 2;
constexpr uint32_t BYTES_MAX_UPDATE = 1024;
constexpr uint32_t MASK_DIV = 8;
constexpr uint32_t MASK_TILE_NUM = 64;

static std::map<ge::DataType, uint32_t> SUPPORT_DTYPE = {
    {ge::DataType::DT_FLOAT, 4}, {ge::DataType::DT_FLOAT16, 2},
    {ge::DataType::DT_UINT8, 1}, {ge::DataType::DT_INT8, 1},
    {ge::DataType::DT_INT16, 2}, {ge::DataType::DT_INT32, 4},
    {ge::DataType::DT_INT64, 8}, {ge::DataType::DT_BF16, 2},
    {ge::DataType::DT_DOUBLE, 8}, {ge::DataType::DT_BOOL, 1}
};

inline static bool IsSupportDtype(const std::set<ge::DataType> &supportDtype, const ge::DataType dtype)
{
    return (supportDtype.count(dtype) != 0);
}

class MaskedScatterV1Tiling {
public:
    MaskedScatterV1Tiling() = default;
    ~MaskedScatterV1Tiling() = default;

    ge::graphStatus RunKernelTiling(gert::TilingContext* context);

private:
    ge::graphStatus GetCompileInfo(const gert::TilingContext* context);
    ge::graphStatus CheckOpInputDtype(const gert::TilingContext* context);
    ge::graphStatus CheckOpInputShape(const gert::TilingContext* context);
    ge::graphStatus SetOpTilingData(gert::TilingContext* context);
    uint32_t ComputeupdateLineLength(gert::TilingContext* context);
    void TilingDataPrint(gert::TilingContext* context);
    MaskedScatterV1CompileInfo compileInfo;
    void SetTilingKeyMode(gert::TilingContext* context);

private:
    uint64_t ubSize = 0;
    uint32_t usedCoreNum = 0;
    uint32_t totalCoreNum = 0;
    uint32_t totalMaskLength = 1;
    uint32_t loopNum = 0;
    uint32_t remainNum = 0;
    uint32_t totalUpdatesNum = 0;
    uint32_t maskTileLength = 0;
    uint32_t updatesLineNum = 0;
    uint32_t updateLineLength = 0;
    uint32_t updatesNum = 0;
    bool isUpdateLengthExceed = false;
};

void MaskedScatterV1Tiling::SetTilingKeyMode(gert::TilingContext* context)
{
    uint64_t tilingKey = 0;
    if (!isUpdateLengthExceed) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    } else {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    }
}

ge::graphStatus MaskedScatterV1Tiling::GetCompileInfo(const gert::TilingContext* context)
{
    OP_LOGD(context, "GetCompileInfo start.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    ubSize = ubSizePlatform;
    totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (ubSize <= 0 || totalCoreNum <= 0) {
        OP_LOGE(context->GetNodeName(), "GetCompileInfo failed!");
        return ge::GRAPH_FAILED;
    }
    compileInfo.sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context, "GetCompileInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedScatterV1Tiling::CheckOpInputShape(const gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(X_INDEX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    auto maskShape = context->GetInputShape(MASK_INDEX)->GetStorageShape();
    size_t maskDimNum = maskShape.GetDimNum();
    auto updatesShape = context->GetInputShape(UPDATES_INDEX)->GetStorageShape();
    size_t updatesDimNum = updatesShape.GetDimNum();
    
    uint32_t updatesElement = 1;
    for (size_t i = 0; i < updatesDimNum; i++) {
        updatesElement = updatesElement * updatesShape.GetDim(i);
    }

    if (xDimNum != maskDimNum) {
        OP_LOGE(context->GetNodeName(), "Invalid shape!");
        return ge::GRAPH_FAILED;
    }

    if (maskShape == xShape) {
        for (size_t i = 0; i < maskDimNum; i++) {
            totalMaskLength = totalMaskLength * maskShape.GetDim(i);
        }
        updatesLineNum = 1;
        totalUpdatesNum = totalMaskLength;
        updatesNum = updatesElement;
        OP_LOGD(context, "CheckOpInputShape end.");
        return ge::GRAPH_SUCCESS;
    }

    if (maskShape.GetDim(maskDimNum - 1) != 1 || xShape.GetDim(xDimNum - 1) != updatesShape.GetDim(updatesDimNum - 1)) {
        OP_LOGE(context->GetNodeName(), "Invalid shape!");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < maskDimNum - 1; i++) {
        if (maskShape.GetDim(i) != xShape.GetDim(i)) {
            OP_LOGE(context->GetNodeName(), "Invalid shape!");
            return ge::GRAPH_FAILED;
        }
        totalMaskLength = totalMaskLength * maskShape.GetDim(i);
    }
    updatesLineNum = xShape.GetDim(xDimNum - 1);
    if (updatesLineNum == 0) {
        OP_LOGE(context->GetNodeName(), "Invalid shape!");
        return ge::GRAPH_FAILED;
    }
    totalUpdatesNum = totalMaskLength * updatesLineNum;
    updatesNum = updatesElement / updatesLineNum;
    OP_LOGD(context, "CheckOpInputShape end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedScatterV1Tiling::CheckOpInputDtype(const gert::TilingContext* context)
{
    auto xInputDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInputDesc);
    auto xDtype = xInputDesc->GetDataType();
    if (SUPPORT_DTYPE.find(xDtype) == SUPPORT_DTYPE.end()) {
        OP_LOGE(context->GetNodeName(), "Invalid xDtype!");
        return ge::GRAPH_FAILED;
    }

    auto maskInputDesc = context->GetInputDesc(MASK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskInputDesc);
    auto maskDtype = maskInputDesc->GetDataType();
    if (maskDtype != ge::DataType::DT_BOOL) {
        OP_LOGE(context->GetNodeName(), "Invalid maskDtype!");
        return ge::GRAPH_FAILED;
    }

    auto updatesInputDesc = context->GetInputDesc(UPDATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesInputDesc);
    auto updatesDtype = updatesInputDesc->GetDataType();
    if (updatesDtype != xDtype) {
        OP_LOGE(context->GetNodeName(), "Invalid updatesDtype!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t MaskedScatterV1Tiling::ComputeupdateLineLength(gert::TilingContext* context)
{
    auto updatesInputDesc = context->GetInputDesc(UPDATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesInputDesc);
    auto updatesDtype = updatesInputDesc->GetDataType();
    return updatesLineNum * SUPPORT_DTYPE[updatesDtype];
}

void MaskedScatterV1Tiling::TilingDataPrint(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "loopNum: %u.", loopNum);
    OP_LOGD(context->GetNodeName(), "remainNum: %u.", remainNum);
    OP_LOGD(context->GetNodeName(), "isUpdateLengthExceed: %d.", isUpdateLengthExceed);
    OP_LOGD(context->GetNodeName(), "updatesLineNum: %u.", updatesLineNum);
    OP_LOGD(context->GetNodeName(), "totalUpdatesNum: %u.", totalUpdatesNum);
    OP_LOGD(context->GetNodeName(), "maskTileLength: %u.", maskTileLength);
    OP_LOGD(context->GetNodeName(), "updatesNum: %u.", updatesNum);
}

ge::graphStatus MaskedScatterV1Tiling::SetOpTilingData(gert::TilingContext* context)
{
    usedCoreNum = totalMaskLength < totalCoreNum ? totalMaskLength : totalCoreNum;
    loopNum = totalMaskLength / usedCoreNum;
    remainNum = totalMaskLength % usedCoreNum;
    updateLineLength = ComputeupdateLineLength(context);
    if (updateLineLength <= BYTES_MAX_UPDATE) {
        isUpdateLengthExceed = false;
        maskTileLength = MASK_TILE_NUM;
    } else {
        isUpdateLengthExceed = true;
        maskTileLength = totalMaskLength;
    }
    TilingDataPrint(context);
    context->SetBlockDim(usedCoreNum);
    SetTilingKeyMode(context);
    MaskedScatterV1TilingData* tilingData = context->GetTilingData<MaskedScatterV1TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    OP_CHECK_IF(
        memset_s(tilingData, sizeof(MaskedScatterV1TilingData), 0, sizeof(MaskedScatterV1TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tilingData->loopNum = loopNum;
    tilingData->remainNum = remainNum;
    tilingData->updatesLineNum = updatesLineNum;
    tilingData->totalUpdatesNum = totalUpdatesNum;
    tilingData->maskTileLength = maskTileLength;
    tilingData->updatesNum = updatesNum;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = compileInfo.sysWorkspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedScatterV1Tiling::RunKernelTiling(gert::TilingContext* context)
{
    if (GetCompileInfo(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "RunKernelTiling GetCompileInfo failed.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        (CheckOpInputShape(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "input shape check failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CheckOpInputDtype(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "dtype is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (SetOpTilingData(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "set optiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus MaskedScatterV1TilingFunc(gert::TilingContext* context)
{
    MaskedScatterV1Tiling tiling;
    return tiling.RunKernelTiling(context);
}

static ge::graphStatus TilingParseForMaskedScatterV1(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Enter TilingParseForMaskedScatterV1.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaskedScatter)
.Tiling(MaskedScatterV1TilingFunc)
.TilingParse<MaskedScatterV1CompileInfo>(TilingParseForMaskedScatterV1);
} // namespace optiling
