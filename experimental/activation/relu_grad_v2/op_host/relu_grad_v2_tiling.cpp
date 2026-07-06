/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <set>

#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/relu_grad_v2_tiling_data.h"
#include "../op_kernel/relu_grad_v2_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;
constexpr int64_t COMPARE_ALIGN_BYTES = 256;
constexpr int64_t SCALAR_FP32_BUFFER_COEFFICIENT = 25;
constexpr int64_t CAST_SCALAR_FP16_BF16_BUFFER_COEFFICIENT = 24;

struct ReluGradV2CompileInfo {};

struct ReluGradV2TilingInfo {
    int64_t formerNum = 0;
    int64_t formerLength = 0;
    int64_t tailLength = 0;
    int64_t tileLength = 1;
    uint32_t blockDim = 1;
};

static const gert::Shape SCALAR_SHAPE = {1};

static const gert::Shape& EnsureNotScalar(const gert::Shape& shape)
{
    if (shape.GetDimNum() == 0) {
        return SCALAR_SHAPE;
    }
    return shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalLength, ge::DataType& dataType)
{
    auto* gradientsShape = context->GetInputShape(0);
    auto* featuresShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradientsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);

    const gert::Shape& gradientsStorageShape = EnsureNotScalar(gradientsShape->GetStorageShape());
    const gert::Shape& featuresStorageShape = EnsureNotScalar(featuresShape->GetStorageShape());
    OP_CHECK_IF(gradientsStorageShape != featuresStorageShape, OP_LOGE(context, "shape mismatch"),
                return ge::GRAPH_FAILED);
    totalLength = gradientsStorageShape.GetShapeSize();

    auto* gradientsDesc = context->GetInputDesc(0);
    auto* featuresDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradientsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresDesc);
    dataType = gradientsDesc->GetDataType();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
                                                   ge::DT_UINT8, ge::DT_INT32,   ge::DT_INT64};
    OP_CHECK_IF(supportedDtype.count(dataType) == 0, OP_LOGE(context, "invalid dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(featuresDesc->GetDataType() != dataType, OP_LOGE(context, "dtype mismatch"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static void SetEmptyTiling(ReluGradV2TilingData* tiling)
{
    tiling->formerNum = 0;
    tiling->formerLength = 0;
    tiling->tailLength = 0;
    tiling->tileLength = 1;
}

static bool IsFloatFamily(ge::DataType dataType)
{
    return dataType == ge::DT_FLOAT || dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16;
}

static int64_t GetBufferCoefficient(ge::DataType dataType, int64_t dtypeSize)
{
    if (dataType == ge::DT_FLOAT) {
        return SCALAR_FP32_BUFFER_COEFFICIENT;
    }
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        return CAST_SCALAR_FP16_BF16_BUFFER_COEFFICIENT;
    }
    return dtypeSize * 6;
}

static int64_t GetAlignElements(ge::DataType dataType, int64_t dtypeSize)
{
    if (dtypeSize == 0) {
        return 1;
    }
    int64_t alignElements = 32 / dtypeSize;
    if (alignElements <= 0) {
        alignElements = 1;
    }
    if (IsFloatFamily(dataType)) {
        int64_t compareAlignElements = COMPARE_ALIGN_BYTES / static_cast<int64_t>(sizeof(float));
        alignElements = std::max<int64_t>(alignElements, compareAlignElements);
    }
    return alignElements;
}

static ge::graphStatus BuildTilingData(int64_t totalLength, ge::DataType dataType, uint64_t ubSize, int64_t coreNum,
                                       ReluGradV2TilingInfo& info)
{
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    if (typeLength == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t dtypeSize = static_cast<int64_t>(typeLength);

    if (dtypeSize == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t cacheLineElements = std::max<int64_t>(1, CACHE_LINE_BYTE_LENGTH / dtypeSize);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t totalLengthCore = (totalLength + coreNum - 1) / coreNum;
    int64_t totalLengthCoreAlign = ((totalLengthCore + cacheLineElements - 1) / cacheLineElements) * cacheLineElements;
    int64_t usedCoreNum = std::max<int64_t>(1, (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign);

    info.formerNum = usedCoreNum - 1;
    info.formerLength = totalLengthCoreAlign;
    info.tailLength = totalLength - info.formerNum * info.formerLength;
    info.blockDim = static_cast<uint32_t>(usedCoreNum);

    int64_t bufferCoefficient = GetBufferCoefficient(dataType, dtypeSize);
    if (bufferCoefficient == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t maxTileElements = static_cast<int64_t>(ubSize) / bufferCoefficient;
    int64_t alignElements = GetAlignElements(dataType, dtypeSize);
    if (alignElements == 0) {
        return ge::GRAPH_FAILED;
    }
    info.tileLength = (maxTileElements / alignElements) * alignElements;
    if (info.tileLength <= 0) {
        info.tileLength = alignElements;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(gert::TilingContext* context, const ReluGradV2TilingInfo& info,
                                     ReluGradV2TilingData* tiling)
{
    tiling->formerNum = info.formerNum;
    tiling->formerLength = info.formerLength;
    tiling->tailLength = info.tailLength;
    tiling->tileLength = info.tileLength;
    context->SetBlockDim(info.blockDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluGradV2TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalLength = 0;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalLength, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    auto* tiling = context->GetTilingData<ReluGradV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ReluGradV2TilingData), 0, sizeof(ReluGradV2TilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    if (totalLength <= 0) {
        SetEmptyTiling(tiling);
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, dTypeX);
        return ge::GRAPH_SUCCESS;
    }

    ReluGradV2TilingInfo info;
    OP_CHECK_IF(BuildTilingData(totalLength, dataType, ubSize, coreNum, info) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "BuildTilingData error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetTilingData(context, info, tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetTilingData error"),
                return ge::GRAPH_FAILED);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReluGradV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReluGradV2).Tiling(ReluGradV2TilingFunc).TilingParse<ReluGradV2CompileInfo>(TilingParseForReluGradV2);
} // namespace optiling
