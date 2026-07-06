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
#include <sstream>
#include <vector>
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "util/shape_util.h"

namespace {
constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_INDEXED_SIZES = 1;
constexpr size_t INPUT_INDICES = 3;
constexpr size_t OUTPUT_Y = 0;
constexpr size_t MAX_RANK = 8;

std::vector<int64_t> ShapeToVector(const gert::Shape& shape)
{
    std::vector<int64_t> dims(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        dims[i] = shape.GetDim(i);
    }
    return dims;
}

bool IsIndexedContinuous(const std::vector<int64_t>& indexedSizes)
{
    int64_t first = -1;
    int64_t last = -1;
    for (size_t i = 0; i < indexedSizes.size(); ++i) {
        if (indexedSizes[i] == 1) {
            if (first < 0) {
                first = static_cast<int64_t>(i);
            }
            last = static_cast<int64_t>(i);
        }
    }
    if (first < 0) {
        return true;
    }
    for (int64_t i = first; i <= last; ++i) {
        if (indexedSizes[static_cast<size_t>(i)] != 1) {
            return false;
        }
    }
    return true;
}

ge::graphStatus BroadcastShape(const char* opName, const std::vector<std::vector<int64_t>>& shapes,
                               std::vector<int64_t>& outShape)
{
    size_t maxRank = 0;
    for (const auto& shape : shapes) {
        maxRank = std::max(maxRank, shape.size());
    }
    outShape.assign(maxRank, 1);
    for (const auto& shape : shapes) {
        for (size_t i = 0; i < maxRank; ++i) {
            int64_t srcIdx = static_cast<int64_t>(shape.size()) - static_cast<int64_t>(maxRank - i);
            int64_t dim = srcIdx < 0 ? 1 : shape[static_cast<size_t>(srcIdx)];
            if (dim <= 0) {
                OP_LOGE(opName, "indices shape dimension must be positive, but got %ld.", dim);
                return ge::GRAPH_FAILED;
            }
            if (outShape[i] != 1 && dim != 1 && outShape[i] != dim) {
                OP_LOGE(opName, "indices shapes can not be broadcast.");
                return ge::GRAPH_FAILED;
            }
            outShape[i] = std::max(outShape[i], dim);
        }
    }
    return ge::GRAPH_SUCCESS;
}

std::vector<int64_t> MakeOutputShape(const std::vector<int64_t>& xShape, const std::vector<int64_t>& indexedSizes,
                                     const std::vector<int64_t>& indexShape)
{
    std::vector<int64_t> yShape;
    if (!IsIndexedContinuous(indexedSizes)) {
        yShape.insert(yShape.end(), indexShape.begin(), indexShape.end());
        for (size_t i = 0; i < indexedSizes.size(); ++i) {
            if (indexedSizes[i] != 1) {
                yShape.push_back(xShape[i]);
            }
        }
        return yShape;
    }

    bool inserted = false;
    for (size_t i = 0; i < indexedSizes.size(); ++i) {
        if (indexedSizes[i] == 1) {
            if (!inserted) {
                yShape.insert(yShape.end(), indexShape.begin(), indexShape.end());
                inserted = true;
            }
            continue;
        }
        yShape.push_back(xShape[i]);
    }
    return yShape;
}
} // namespace

namespace ops {
static ge::graphStatus InferShape4Index(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const gert::Tensor* xTensor = context->GetInputTensor(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    const gert::Tensor* sizesTensor = context->GetInputTensor(INPUT_INDEXED_SIZES);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizesTensor);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const gert::Shape& xStorageShape = xTensor->GetStorageShape();
    if (xStorageShape.GetDimNum() > MAX_RANK) {
        OP_LOGE(context->GetNodeName(), "x rank must be <= %zu.", MAX_RANK);
        return ge::GRAPH_FAILED;
    }
    const int64_t* indexedSizes = sizesTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, indexedSizes);
    const int64_t indexedSizesNum = sizesTensor->GetShapeSize();
    if (indexedSizesNum != static_cast<int64_t>(xStorageShape.GetDimNum())) {
        OP_LOGE(context->GetNodeName(), "indexed_sizes length must equal x rank.");
        return ge::GRAPH_FAILED;
    }

    std::vector<int64_t> xShape = ShapeToVector(xStorageShape);
    std::vector<int64_t> indexedSizesVec(static_cast<size_t>(indexedSizesNum));
    std::vector<std::vector<int64_t>> indexShapes;
    for (int64_t i = 0; i < indexedSizesNum; ++i) {
        indexedSizesVec[static_cast<size_t>(i)] = indexedSizes[i];
        if (indexedSizes[i] == 1) {
            const gert::Tensor* indexTensor = context->GetInputTensor(INPUT_INDICES + indexShapes.size());
            OP_CHECK_NULL_WITH_CONTEXT(context, indexTensor);
            indexShapes.push_back(ShapeToVector(indexTensor->GetStorageShape()));
        } else if (indexedSizes[i] != 0) {
            OP_LOGE(context->GetNodeName(), "indexed_sizes only supports 0 or 1.");
            return ge::GRAPH_FAILED;
        }
    }
    if (indexShapes.empty()) {
        OP_LOGE(context->GetNodeName(), "indexed_sizes must mark at least one indexed dimension.");
        return ge::GRAPH_FAILED;
    }

    std::vector<int64_t> broadcastIndexShape;
    if (BroadcastShape(context->GetNodeName(), indexShapes, broadcastIndexShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    std::vector<int64_t> outputShape = MakeOutputShape(xShape, indexedSizesVec, broadcastIndexShape);
    yShape->SetDimNum(outputShape.size());
    for (size_t i = 0; i < outputShape.size(); ++i) {
        yShape->SetDim(i, outputShape[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Index(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(OUTPUT_Y, context->GetInputDataType(INPUT_X));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexAiCore)
    .InferDataType(InferDataType4Index)
    .InferShape(InferShape4Index)
    .InputsDataDependency({INPUT_INDEXED_SIZES});
} // namespace ops
