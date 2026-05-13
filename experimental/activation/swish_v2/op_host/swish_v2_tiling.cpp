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
#include "util/math_util.h"
#include "../op_kernel/swish_v2_tiling_data.h"
#include "../op_kernel/swish_v2_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;
constexpr int64_t MIN_BYTES_PER_CORE = 16 * 1024;
constexpr int64_t BUFFER_COEFFICIENT = 16;
constexpr size_t SCALE_ATTR_INDEX = 0;

struct SwishV2CompileInfo {};

static const gert::Shape SCALAR_SHAPE = {1};

static const gert::Shape &EnsureNotScalar(const gert::Shape &shape)
{
    if (shape.GetDimNum() == 0) {
        return SCALAR_SHAPE;
    }
    return shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, int64_t &coreNum)
{
    auto *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext *context, int64_t &totalLength, ge::DataType &dataType, float &scale)
{
    auto *inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    totalLength = EnsureNotScalar(inputX->GetStorageShape()).GetShapeSize();

    auto *inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(supportedDtype.count(dataType) == 0, OP_LOGE(context, "invalid dtype"), return ge::GRAPH_FAILED);

    scale = 1.0f;
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float *scaleAttr = attrs->GetAttrPointer<float>(SCALE_ATTR_INDEX);
    if (scaleAttr != nullptr) {
        scale = *scaleAttr;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static void SetTemplateParam(gert::TilingContext *context, ge::DataType dataType)
{
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
}

static void FillDefaultTiling(SwishV2TilingData *tiling, ge::DataType dataType, gert::TilingContext *context)
{
    tiling->formerNum = 0;
    tiling->formerLength = 0;
    tiling->tailLength = 0;
    tiling->tileLength = 1;
    context->SetBlockDim(1);
    SetTemplateParam(context, dataType);
}

static ge::graphStatus FillNormalTiling(
    gert::TilingContext *context, SwishV2TilingData *tiling, int64_t totalLength, ge::DataType dataType, uint64_t ubSize,
    int64_t coreNum)
{
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "typeLength is 0"), return ge::GRAPH_FAILED);

    int64_t dtypeSize = static_cast<int64_t>(typeLength);
    int64_t totalBytes = totalLength * dtypeSize;
    int64_t targetCoreNum = (totalBytes + MIN_BYTES_PER_CORE - 1) / MIN_BYTES_PER_CORE;
    targetCoreNum = std::max<int64_t>(1, targetCoreNum);
    targetCoreNum = std::min<int64_t>(targetCoreNum, coreNum);

    int64_t cacheLineElements = std::max<int64_t>(1, CACHE_LINE_BYTE_LENGTH / dtypeSize);
    int64_t totalLengthCore = (totalLength + targetCoreNum - 1) / targetCoreNum;
    int64_t totalLengthCoreAlign =
        ((totalLengthCore + cacheLineElements - 1) / cacheLineElements) * cacheLineElements;

    int64_t usedCoreNum = (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign;
    usedCoreNum = std::max<int64_t>(1, usedCoreNum);

    tiling->formerNum = usedCoreNum - 1;
    tiling->formerLength = totalLengthCoreAlign;
    tiling->tailLength = totalLength - tiling->formerNum * tiling->formerLength;

    int64_t maxTileElements = static_cast<int64_t>(ubSize) / BUFFER_COEFFICIENT;
    int64_t alignElements = std::max<int64_t>(1, 32 / dtypeSize);
    tiling->tileLength = (maxTileElements / alignElements) * alignElements;
    if (tiling->tileLength <= 0) {
        tiling->tileLength = alignElements;
    }

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    SetTemplateParam(context, dataType);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SwishV2TilingFunc(gert::TilingContext *context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalLength = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    float scale = 1.0f;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalLength, dataType, scale) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    auto *tiling = context->GetTilingData<SwishV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SwishV2TilingData), 0, sizeof(SwishV2TilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->scale = scale;

    if (totalLength <= 0) {
        FillDefaultTiling(tiling, dataType, context);
        return ge::GRAPH_SUCCESS;
    }

    return FillNormalTiling(context, tiling, totalLength, dataType, ubSize, coreNum);
}

static ge::graphStatus TilingParseForSwishV2([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwishV2).Tiling(SwishV2TilingFunc).TilingParse<SwishV2CompileInfo>(TilingParseForSwishV2);
}  // namespace optiling
