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
#include <vector>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/index_tiling_data.h"

namespace {
constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_INDICES = 3;
constexpr size_t MAX_RANK = 8;

std::vector<uint64_t> ShapeToVector(const gert::Shape& shape)
{
    std::vector<uint64_t> dims(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        dims[i] = static_cast<uint64_t>(shape.GetDim(i));
    }
    return dims;
}

std::vector<uint64_t> MakeStride(const std::vector<uint64_t>& shape)
{
    std::vector<uint64_t> stride(shape.size(), 1);
    uint64_t running = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        stride[static_cast<size_t>(i)] = running;
        running *= shape[static_cast<size_t>(i)];
    }
    return stride;
}

uint64_t ShapeSize(const std::vector<uint64_t>& shape)
{
    uint64_t total = 1;
    for (uint64_t dim : shape) {
        total *= dim;
    }
    return total;
}

ge::graphStatus BroadcastIndexShape(gert::TilingContext* context, const std::vector<uint32_t>& indexedDims,
    std::vector<uint64_t>& broadcastShape, std::vector<uint64_t>& indexInputStrides)
{
    size_t maxRank = 0;
    std::vector<std::vector<uint64_t>> shapes;
    for (size_t i = 0; i < indexedDims.size(); ++i) {
        const gert::StorageShape* curShape = context->GetDynamicInputShape(INPUT_INDICES, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, curShape);
        shapes.push_back(ShapeToVector(curShape->GetStorageShape()));
        maxRank = std::max(maxRank, shapes.back().size());
    }
    if (maxRank > MAX_RANK) {
        OP_LOGE(context->GetNodeName(), "indices rank must be <= %zu.", MAX_RANK);
        return ge::GRAPH_FAILED;
    }
    broadcastShape.assign(maxRank, 1);
    for (const auto& shape : shapes) {
        for (size_t i = 0; i < maxRank; ++i) {
            int64_t srcIdx = static_cast<int64_t>(shape.size()) - static_cast<int64_t>(maxRank - i);
            uint64_t dim = srcIdx < 0 ? 1 : shape[static_cast<size_t>(srcIdx)];
            if (dim == 0) {
                OP_LOGE(context->GetNodeName(), "indices shape dimension must be positive.");
                return ge::GRAPH_FAILED;
            }
            if (broadcastShape[i] != 1 && dim != 1 && broadcastShape[i] != dim) {
                OP_LOGE(context->GetNodeName(), "indices shapes can not broadcast.");
                return ge::GRAPH_FAILED;
            }
            broadcastShape[i] = std::max(broadcastShape[i], dim);
        }
    }
    indexInputStrides.assign(INDEX_MAX_INDEX_STRIDE_SIZE, 0);
    for (size_t ordinal = 0; ordinal < shapes.size(); ++ordinal) {
        const std::vector<uint64_t> curStride = MakeStride(shapes[ordinal]);
        for (size_t i = 0; i < maxRank; ++i) {
            int64_t srcIdx = static_cast<int64_t>(shapes[ordinal].size()) - static_cast<int64_t>(maxRank - i);
            if (srcIdx >= 0 && shapes[ordinal][static_cast<size_t>(srcIdx)] != 1) {
                indexInputStrides[ordinal * INDEX_MAX_RANK + i] = curStride[static_cast<size_t>(srcIdx)];
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

void FillArray(uint64_t* dst, const std::vector<uint64_t>& src)
{
    for (size_t i = 0; i < MAX_RANK; ++i) {
        dst[i] = i < src.size() ? src[i] : 1;
    }
}

void FillArray(uint32_t* dst, const std::vector<uint32_t>& src)
{
    for (size_t i = 0; i < MAX_RANK; ++i) {
        dst[i] = i < src.size() ? src[i] : 0;
    }
}

std::vector<uint32_t> GetIndexedDims(gert::TilingContext* context)
{
    std::vector<uint32_t> indexedDims;
    for (size_t i = 0; i < MAX_RANK; ++i) {
        if (context->GetDynamicInputShape(INPUT_INDICES, i) == nullptr) {
            break;
        }
        indexedDims.push_back(static_cast<uint32_t>(i));
    }
    return indexedDims;
}

ge::graphStatus GetAivCoreNum(gert::TilingContext* context, uint32_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (aivCoreNum <= 0) {
        OP_LOGE(context->GetNodeName(), "aiv core num must be positive.");
        return ge::GRAPH_FAILED;
    }
    coreNum = static_cast<uint32_t>(aivCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MakeOutputShape(gert::TilingContext* context, const std::vector<uint32_t>& indexedDims,
    const std::vector<uint64_t>& xShape, std::vector<uint64_t>& indexShape, std::vector<uint64_t>& indexInputStrides,
    std::vector<uint64_t>& yShape)
{
    if (BroadcastIndexShape(context, indexedDims, indexShape, indexInputStrides) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    yShape = indexShape;
    for (size_t i = indexedDims.size(); i < xShape.size(); ++i) {
        yShape.push_back(xShape[i]);
    }
    if (yShape.empty() || yShape.size() > MAX_RANK) {
        OP_LOGE(context->GetNodeName(), "y rank must be in [1, %zu].", MAX_RANK);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FillIndexTilingData(optiling::IndexTilingData* tiling, const std::vector<uint32_t>& indexedDims,
    const std::vector<uint64_t>& xShape, const std::vector<uint64_t>& yShape,
    const std::vector<uint64_t>& indexShape, const std::vector<uint64_t>& indexInputStrides)
{
    tiling->xRank = static_cast<uint32_t>(xShape.size());
    tiling->yRank = static_cast<uint32_t>(yShape.size());
    tiling->indexRank = static_cast<uint32_t>(indexShape.size());
    tiling->indexedDimNum = static_cast<uint32_t>(indexedDims.size());
    tiling->totalNum = ShapeSize(yShape);

    uint32_t indexedDimArray[MAX_RANK] = {0};
    uint64_t xShapeArray[MAX_RANK] = {0};
    uint64_t xStrideArray[MAX_RANK] = {0};
    uint64_t yShapeArray[MAX_RANK] = {0};
    uint64_t yStrideArray[MAX_RANK] = {0};
    uint64_t indexShapeArray[MAX_RANK] = {0};
    uint64_t indexStrideArray[MAX_RANK] = {0};
    uint64_t indexInputStrideArray[INDEX_MAX_INDEX_STRIDE_SIZE] = {0};
    FillArray(indexedDimArray, indexedDims);
    FillArray(xShapeArray, xShape);
    FillArray(xStrideArray, MakeStride(xShape));
    FillArray(yShapeArray, yShape);
    FillArray(yStrideArray, MakeStride(yShape));
    FillArray(indexShapeArray, indexShape);
    FillArray(indexStrideArray, MakeStride(indexShape));
    for (size_t i = 0; i < indexInputStrides.size() && i < INDEX_MAX_INDEX_STRIDE_SIZE; ++i) {
        indexInputStrideArray[i] = indexInputStrides[i];
    }
    std::copy(indexedDimArray, indexedDimArray + INDEX_MAX_RANK, tiling->indexedDims);
    std::copy(xShapeArray, xShapeArray + INDEX_MAX_RANK, tiling->xShape);
    std::copy(xStrideArray, xStrideArray + INDEX_MAX_RANK, tiling->xStride);
    std::copy(yShapeArray, yShapeArray + INDEX_MAX_RANK, tiling->yShape);
    std::copy(yStrideArray, yStrideArray + INDEX_MAX_RANK, tiling->yStride);
    std::copy(indexShapeArray, indexShapeArray + INDEX_MAX_RANK, tiling->indexShape);
    std::copy(indexStrideArray, indexStrideArray + INDEX_MAX_RANK, tiling->indexStride);
    std::copy(indexInputStrideArray, indexInputStrideArray + INDEX_MAX_INDEX_STRIDE_SIZE, tiling->indexInputStrides);
}
}  // namespace

namespace optiling {
struct IndexCompileInfo {};

static ge::graphStatus Tiling4Index(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    IndexTilingData* tiling = context->GetTilingData<IndexTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const gert::StorageShape* xShapePtr = context->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xStorageShape = xShapePtr->GetStorageShape();
    if (xStorageShape.GetDimNum() == 0 || xStorageShape.GetDimNum() > MAX_RANK) {
        OP_LOGE(context->GetNodeName(), "x rank must be in [1, %zu].", MAX_RANK);
        return ge::GRAPH_FAILED;
    }
    std::vector<uint32_t> indexedDims = GetIndexedDims(context);
    if (indexedDims.empty() || indexedDims.size() > static_cast<size_t>(xStorageShape.GetDimNum())) {
        OP_LOGE(context->GetNodeName(), "indexed dimension count is invalid.");
        return ge::GRAPH_FAILED;
    }

    std::vector<uint64_t> xShape = ShapeToVector(xStorageShape);
    std::vector<uint64_t> indexShape;
    std::vector<uint64_t> indexInputStrides;
    std::vector<uint64_t> yShape;
    if (MakeOutputShape(context, indexedDims, xShape, indexShape, indexInputStrides, yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    FillIndexTilingData(tiling, indexedDims, xShape, yShape, indexShape, indexInputStrides);

    uint32_t aivCoreNum = 0;
    if (GetAivCoreNum(context, aivCoreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum, tiling->totalNum));
    tiling->usedCoreNum = usedCoreNum;
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(0);
    context->GetRawTilingData()->SetDataSize(sizeof(IndexTilingData));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForIndex([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexAiCore)
    .Tiling(Tiling4Index)
    .TilingInputsDataDependency({1})
    .TilingParse<IndexCompileInfo>(TilingParseForIndex);
}  // namespace optiling
