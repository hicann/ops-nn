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
#include "../op_kernel/mish_grad_v2_tiling_data.h"
#include "../op_kernel/mish_grad_v2_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;
constexpr int64_t DATA_COPY_ALIGN_BYTES = 32;
constexpr int64_t BUFFER_COEFFICIENT = 36;

struct MishGradV2CompileInfo {};

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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext *context, int64_t &totalLength,
                                         ge::DataType &dataType, int64_t &haveTanhx)
{
    auto *gradShape = context->GetInputShape(0);
    auto *xShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape &gradStorageShape = EnsureNotScalar(gradShape->GetStorageShape());
    const gert::Shape &xStorageShape = EnsureNotScalar(xShape->GetStorageShape());
    OP_CHECK_IF(gradStorageShape != xStorageShape, OP_LOGE(context, "shape mismatch"), return ge::GRAPH_FAILED);

    auto *gradDesc = context->GetInputDesc(0);
    auto *xDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    dataType = gradDesc->GetDataType();
    OP_CHECK_IF(xDesc->GetDataType() != dataType, OP_LOGE(context, "dtype mismatch"), return ge::GRAPH_FAILED);

    if (context->GetInputShape(2) != nullptr) {
        auto *tanhxShape = context->GetInputShape(2);
        auto *tanhxDesc = context->GetInputDesc(2);
        OP_CHECK_NULL_WITH_CONTEXT(context, tanhxShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, tanhxDesc);
        const gert::Shape &tanhxStorageShape = EnsureNotScalar(tanhxShape->GetStorageShape());
        OP_CHECK_IF(gradStorageShape != tanhxStorageShape, OP_LOGE(context, "tanhx shape mismatch"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tanhxDesc->GetDataType() != dataType, OP_LOGE(context, "tanhx dtype mismatch"),
                    return ge::GRAPH_FAILED);
        haveTanhx = 1;
    } else {
        haveTanhx = 0;
    }

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(supportedDtype.count(dataType) == 0, OP_LOGE(context, "invalid dtype"), return ge::GRAPH_FAILED);

    totalLength = gradStorageShape.GetShapeSize();
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

static ge::graphStatus FillTilingData(MishGradV2TilingData *tiling, int64_t totalLength, ge::DataType dataType,
                                      uint64_t ubSize, int64_t coreNum, int64_t haveTanhx, uint32_t &blockDim)
{
    int64_t safeCoreNum = (coreNum <= 0) ? 1 : coreNum;
    if (totalLength <= 0) {
        tiling->formerNum = 0;
        tiling->formerLength = 0;
        tiling->tailLength = 0;
        tiling->tileLength = 1;
        tiling->haveTanhx = haveTanhx;
        blockDim = 1;
        return ge::GRAPH_SUCCESS;
    }

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    if (typeLength == 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t dtypeSize = static_cast<int64_t>(typeLength);
    int64_t cacheLineElements = std::max<int64_t>(1, CACHE_LINE_BYTE_LENGTH / dtypeSize);
    int64_t totalLengthCore = (totalLength + safeCoreNum - 1) / safeCoreNum;
    int64_t totalLengthCoreAlign =
        ((totalLengthCore + cacheLineElements - 1) / cacheLineElements) * cacheLineElements;
    int64_t usedCoreNum = (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign;
    usedCoreNum = std::max<int64_t>(1, usedCoreNum);
    int64_t formerNum = usedCoreNum - 1;
    int64_t formerLength = totalLengthCoreAlign;
    int64_t tailLength = totalLength - formerNum * formerLength;

    int64_t alignElements = std::max<int64_t>(1, DATA_COPY_ALIGN_BYTES / dtypeSize);
    int64_t maxTileElements = static_cast<int64_t>(ubSize) / BUFFER_COEFFICIENT;
    int64_t tileLength = (maxTileElements / alignElements) * alignElements;
    if (tileLength <= 0) {
        tileLength = alignElements;
    }

    tiling->formerNum = formerNum;
    tiling->formerLength = formerLength;
    tiling->tailLength = tailLength;
    tiling->tileLength = tileLength;
    tiling->haveTanhx = haveTanhx;
    blockDim = static_cast<uint32_t>(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MishGradV2TilingFunc(gert::TilingContext *context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalLength = 0;
    ge::DataType dataType;
    int64_t haveTanhx = 0;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalLength, dataType, haveTanhx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    auto *tiling = context->GetTilingData<MishGradV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(MishGradV2TilingData), 0, sizeof(MishGradV2TilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint32_t blockDim = 1;
    OP_CHECK_IF(FillTilingData(tiling, totalLength, dataType, ubSize, coreNum, haveTanhx, blockDim) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "FillTilingData error"), return ge::GRAPH_FAILED);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    context->SetBlockDim(blockDim);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMishGradV2([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MishGradV2)
    .Tiling(MishGradV2TilingFunc)
    .TilingParse<MishGradV2CompileInfo>(TilingParseForMishGradV2);
}  // namespace optiling
