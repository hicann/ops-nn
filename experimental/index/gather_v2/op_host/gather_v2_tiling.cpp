/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../op_kernel/gather_v2_tiling_data.h"
#include "../op_kernel/gather_v2_tiling_key.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace optiling {
namespace {
constexpr uint32_t RESERVED_UB_SIZE = 8 * 1024;
constexpr uint32_t MIN_TILE_COUNT = 1;
constexpr uint32_t BLOCK_BYTES = 32;

struct GatherV2CompileInfo {
    int32_t coreNum = 0;
    uint64_t ubSize = 0;
};

struct GatherV2TilingParams {
    uint64_t totalCount = 0;
    uint64_t innerSize = 0;
    uint64_t indicesCount = 0;
    uint64_t perCore = 0;
    uint64_t lastCore = 0;
    uint32_t tileCount = 0;
    uint32_t xBytes = 0;
    uint32_t indexBytes = 0;
};

uint64_t ShapeSize(const gert::Shape& shape)
{
    uint64_t size = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        size *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return size;
}

uint64_t RangeShapeSize(const gert::Shape& shape, int64_t begin, int64_t end)
{
    uint64_t size = 1;
    for (int64_t i = begin; i < end; ++i) {
        size *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return size;
}

void SetGatherV2TilingData(GatherV2TilingData* tiling, const GatherV2TilingParams& params, uint64_t axisDim)
{
    tiling->set_totalCount(params.totalCount);
    tiling->set_outerSize(1);
    tiling->set_axisDim(axisDim);
    tiling->set_innerSize(params.innerSize);
    tiling->set_indicesCount(params.indicesCount);
    tiling->set_perCoreCount(params.perCore);
    tiling->set_lastCoreCount(params.lastCore);
    tiling->set_tileCount(params.tileCount);
    tiling->set_xTypeBytes(params.xBytes);
    tiling->set_indexTypeBytes(params.indexBytes);
    tiling->set_axis(0);
    tiling->set_batchDims(0);
    tiling->set_negativeIndexSupport(0);
}

ge::graphStatus TilingPrepare4GatherV2(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<GatherV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(
        compileInfo->coreNum <= 0 || compileInfo->ubSize <= RESERVED_UB_SIZE,
        OP_LOGE(context, "invalid platform resource."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4GatherV2(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const GatherV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    const auto* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const auto* indicesShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    const auto* xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const auto* indicesDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);

    const gert::Shape& xStorageShape = xShape->GetStorageShape();
    const gert::Shape& indicesStorageShape = indicesShape->GetStorageShape();
    const int64_t xDimNum = xStorageShape.GetDimNum();
    OP_CHECK_IF(xDimNum <= 0, OP_LOGE(context, "x rank must be greater than 0."), return ge::GRAPH_FAILED);

    auto* tiling = context->GetTilingData<GatherV2TilingData>();
    GatherV2TilingParams params;
    OP_CHECK_IF(
        !ge::TypeUtils::GetDataTypeLength(xDesc->GetDataType(), params.xBytes),
        OP_LOGE(context, "failed to get x dtype length."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !ge::TypeUtils::GetDataTypeLength(indicesDesc->GetDataType(), params.indexBytes),
        OP_LOGE(context, "failed to get indices dtype length."), return ge::GRAPH_FAILED);
    params.indicesCount = ShapeSize(indicesStorageShape);
    // axis is a scalar input tensor, not an attribute. This skeleton records a
    // conservative axis-0 tiling layout; production tiling should add const-axis
    // extraction and select the real layout before enabling computation.
    constexpr int64_t defaultAxis = 0;
    params.innerSize = RangeShapeSize(xStorageShape, defaultAxis + 1, xDimNum);
    params.totalCount = params.indicesCount * params.innerSize;
    const uint32_t tileBytes = params.xBytes + params.indexBytes;
    params.tileCount = static_cast<uint32_t>((compileInfo->ubSize - RESERVED_UB_SIZE) / tileBytes);
    params.tileCount = params.tileCount / BLOCK_BYTES * BLOCK_BYTES;
    if (params.tileCount < MIN_TILE_COUNT) {
        params.tileCount = MIN_TILE_COUNT;
    }

    uint32_t blockDim = static_cast<uint32_t>(
        params.totalCount < static_cast<uint64_t>(compileInfo->coreNum) ? params.totalCount : compileInfo->coreNum);
    if (blockDim == 0) {
        blockDim = 1;
    }
    params.perCore = (params.totalCount + blockDim - 1) / blockDim;
    params.lastCore =
        params.totalCount > params.perCore * (blockDim - 1) ? params.totalCount - params.perCore * (blockDim - 1) : 0;

    SetGatherV2TilingData(tiling, params, static_cast<uint64_t>(xStorageShape.GetDim(defaultAxis)));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(GATHER_V2_TILING_KEY_BASIC);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(GatherV2).Tiling(Tiling4GatherV2).TilingParse<GatherV2CompileInfo>(TilingPrepare4GatherV2);
} // namespace optiling
