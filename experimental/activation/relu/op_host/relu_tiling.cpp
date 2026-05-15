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
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;
constexpr int64_t MIN_BYTES_PER_CORE = 16 * 1024;

struct ReluCompileInfo {};

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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext *context, int64_t &totalLength, ge::DataType &dataType)
{
    auto *inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    totalLength = EnsureNotScalar(inputX->GetStorageShape()).GetShapeSize();

    auto *inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT32, ge::DT_INT64};
    OP_CHECK_IF(supportedDtype.count(dataType) == 0, OP_LOGE(context, "invalid dtype"), return ge::GRAPH_FAILED);

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

static void FillDefaultTiling(ReluTilingData *tiling, uint32_t dTypeX, gert::TilingContext *context)
{
    tiling->formerNum = 0;
    tiling->formerLength = 0;
    tiling->tailLength = 0;
    tiling->tileLength = 1;
    context->SetBlockDim(1);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
}

static void FillNormalTiling(ReluTilingData *tiling, int64_t totalLength, uint64_t ubSize, int64_t coreNum,
                             ge::DataType dataType, uint32_t dTypeX, gert::TilingContext *context)
{
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "typeLength is 0"), return);

    int64_t dtypeSize = static_cast<int64_t>(typeLength);
    int64_t totalBytes = totalLength * dtypeSize;
    int64_t targetCoreNum = 1;
    if (dataType == ge::DT_INT64) {
        targetCoreNum = coreNum;
    } else {
        targetCoreNum = (totalBytes + MIN_BYTES_PER_CORE - 1) / MIN_BYTES_PER_CORE;
        targetCoreNum = std::max<int64_t>(1, targetCoreNum);
        targetCoreNum = std::min<int64_t>(targetCoreNum, coreNum);
    }

    int64_t cacheLineElements = std::max<int64_t>(1, CACHE_LINE_BYTE_LENGTH / dtypeSize);
    int64_t totalLengthCore = (totalLength + targetCoreNum - 1) / targetCoreNum;
    int64_t totalLengthCoreAlign =
        ((totalLengthCore + cacheLineElements - 1) / cacheLineElements) * cacheLineElements;

    int64_t usedCoreNum = (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign;
    usedCoreNum = std::max<int64_t>(1, usedCoreNum);
    int64_t formerNum = usedCoreNum - 1;
    int64_t formerLength = totalLengthCoreAlign;
    int64_t tailLength = totalLength - formerNum * formerLength;

    int64_t bufferCoefficient = 8;
    if (dataType == ge::DT_FLOAT16) {
        bufferCoefficient = 4;
    } else if (dataType == ge::DT_BF16) {
        bufferCoefficient = 8;
    } else if (dataType == ge::DT_INT8) {
        bufferCoefficient = 4;
    } else if (dataType == ge::DT_INT64) {
        bufferCoefficient = 32;
    }

    int64_t maxTileElements = static_cast<int64_t>(ubSize) / bufferCoefficient;
    int64_t alignElements = std::max<int64_t>(1, 32 / dtypeSize);
    int64_t tileLength = (maxTileElements / alignElements) * alignElements;
    if (tileLength <= 0) {
        tileLength = alignElements;
    }

    tiling->formerNum = formerNum;
    tiling->formerLength = formerLength;
    tiling->tailLength = tailLength;
    tiling->tileLength = tileLength;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext *context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalLength = 0;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalLength, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    auto *tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ReluTilingData), 0, sizeof(ReluTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    if (totalLength <= 0) {
        FillDefaultTiling(tiling, dTypeX, context);
        return ge::GRAPH_SUCCESS;
    }

    FillNormalTiling(tiling, totalLength, ubSize, coreNum, dataType, dTypeX, context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);
}  // namespace optiling
