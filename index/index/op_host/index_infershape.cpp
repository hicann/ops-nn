/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index.cc
 * \brief
 */
#include <vector>
#include <algorithm>
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
using namespace std;
namespace {
constexpr size_t IDX_X = 0;
constexpr size_t IDX_SIZES = 1;
constexpr size_t IDX_STRIDES = 2;
constexpr size_t IDX_INDICES_START = 3;
constexpr size_t IDX_Y = 0;
} // namespace

namespace ops {

graphStatus CanBroadcast(const string name, const vector<vector<int64_t>>& shapes)
{
    size_t max_dims = 0;
    for (const auto& shape : shapes) {
        max_dims = max(max_dims, shape.size());
    }

    for (size_t i = 0; i < max_dims; ++i) {
        int64_t current_dim = 1;
        for (const auto& shape : shapes) {
            // 计算当前维度在原始形状中的索引（从后往前）
            int64_t dim_index = static_cast<int64_t>(shape.size()) - static_cast<int64_t>(i) - 1;
            if (dim_index >= 0) {
                int64_t dim = shape[dim_index];
                if (dim <= 0) {
                    OP_LOGE(name, "Shape dimensions must be positive integers.");
                    return GRAPH_FAILED;
                }
                // 检查维度兼容性：要么相等，要么其中一个为1
                if (current_dim != 1 && dim != 1 && current_dim != dim) {
                    OP_LOGE(name, "Shapes cannot be broadcast, incompatible dimensions.");
                    return GRAPH_FAILED;
                }
                current_dim = max(current_dim, dim);
            }
        }
    }
    return GRAPH_SUCCESS;
}

vector<int64_t> ComputeBroadcastShape(const vector<vector<int64_t>>& shapes)
{
    vector<int64_t> broadcast_shape;
    size_t max_dims = 0;
    for (const auto& shape : shapes) {
        max_dims = max(max_dims, shape.size());
    }

    for (size_t i = 0; i < max_dims; ++i) {
        int64_t current_dim = 1;
        for (const auto& shape : shapes) {
            // 计算当前维度在原始形状中的索引（从后往前）
            int64_t dim_index = static_cast<int64_t>(shape.size()) - static_cast<int64_t>(i) - 1;
            if (dim_index >= 0) {
                int64_t dim = shape[dim_index];
                current_dim = max(current_dim, dim);
            }
        }

        broadcast_shape.push_back(current_dim);
    }
    // 反转得到正确的维度顺序（因为我们是从后往前处理的）
    reverse(broadcast_shape.begin(), broadcast_shape.end());
    return broadcast_shape;
}

// 将单个形状广播到目标形状
vector<int64_t> BroadcastSingleShape(const vector<int64_t>& original_shape, const vector<int64_t>& broadcast_shape)
{
    size_t prepend = broadcast_shape.size() - original_shape.size();

    vector<int64_t> result;
    result.insert(result.end(), prepend, 1);
    result.insert(result.end(), original_shape.begin(), original_shape.end());

    for (size_t i = 0; i < broadcast_shape.size(); ++i) {
        result[i] = broadcast_shape[i];
    }

    return result;
}

vector<vector<int64_t>> BroadcastAllShapes(const vector<vector<int64_t>>& shapes)
{
    if (shapes.empty()) {
        return {{}};
    }
    vector<int64_t> target_shape = ComputeBroadcastShape(shapes);
    vector<vector<int64_t>> broadcasted_shapes;
    for (const auto& shape : shapes) {
        broadcasted_shapes.push_back(BroadcastSingleShape(shape, target_shape));
    }
    return broadcasted_shapes;
}

bool AreIndicesContiguous(vector<int64_t> indexed_sizes)
{
    if (indexed_sizes.size() == 0ULL) {
        return true;
    }

    int64_t first = -1L;
    int64_t last = -1L;
    for (size_t i = 0; i < indexed_sizes.size(); ++i) {
        if (indexed_sizes[i] == 1) {
            if (first == -1LL) {
                first = static_cast<int64_t>(i);
            }
            last = static_cast<int64_t>(i);
        }
    }

    if (first == -1LL) {
        return true;
    }

    for (int64_t i = first; i <= last; ++i) {
        if (indexed_sizes[i] != 1) {
            return false;
        }
    }
    return true;
}

// 计算输出形状
vector<int64_t> ComputeOutputShape(
    const vector<int64_t>& xShape, vector<int64_t> indexed_sizes, const vector<int64_t>& indicesShape)
{
    size_t dimCount = indexed_sizes.size();
    vector<int64_t> outputShape;

    // 情况1：索引维度不连续
    if ((!AreIndicesContiguous(indexed_sizes))) {
        outputShape.insert(outputShape.end(), indicesShape.begin(), indicesShape.end());

        for (size_t i = 0; i < dimCount; ++i) {
            if (indexed_sizes[i] != 1) {
                outputShape.push_back(xShape[i]);
            }
        }
        return outputShape;
    }
    // 情况2：所有索引维度连续
    bool replaced = false;
    for (size_t i = 0; i < dimCount; ++i) {
        if (indexed_sizes[i] == 1 && !replaced) {
            outputShape.insert(outputShape.end(), indicesShape.begin(), indicesShape.end());
            replaced = true;
            while (i + 1UL < dimCount && indexed_sizes[i + 1UL] == 1) {
                ++i;
            }
        } else {
            outputShape.push_back(xShape[i]);
        }
    }
    return outputShape;
}

string VectorToString(const vector<int64_t>& vec)
{
    stringstream ss;
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << vec[i];
        if (i != vec.size() - 1) {
            ss << " ";
        }
    }
    return ss.str();
}

// 计算广播后的目标形状
static ge::graphStatus InferShape4Index(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "index infershape is begin");
    auto xTensor = context->GetInputTensor(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    auto xStorageShape = xTensor->GetStorageShape();
    size_t xDimNum = xStorageShape.GetDimNum();
    vector<int64_t> xShape(xDimNum);
    for (size_t i = 0; i < xDimNum; i++) {
        xShape[i] = xStorageShape.GetDim(i);
    }
    OP_LOGD(context->GetNodeName(), "input xShape = %s", VectorToString(xShape).c_str());

    auto sizes_tensor = context->GetInputTensor(IDX_SIZES);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizes_tensor);
    const int64_t* indexed_sizes = sizes_tensor->GetData<int64_t>();
    vector<int64_t> indexed_sizes_vec(xDimNum, 0);
    int64_t indexed_sizes_num = sizes_tensor->GetShapeSize();
    int indicesIdx = IDX_INDICES_START;
    vector<vector<int64_t>> indicesShapeList;
    for (int64_t i = 0; i < indexed_sizes_num; i++) {
        if (indexed_sizes[i] == 1) {
            indexed_sizes_vec[i] = 1;
            auto indicesTensor = context->GetInputTensor(indicesIdx);
            indicesIdx++;
            OP_CHECK_NULL_WITH_CONTEXT(context, indicesTensor);
            auto indicesStorageShape = indicesTensor->GetStorageShape();
            size_t indicesDimNum = indicesStorageShape.GetDimNum();
            vector<int64_t> indicesShape(indicesDimNum);
            for (size_t j = 0; j < indicesDimNum; j++) {
                indicesShape[j] = indicesStorageShape.GetDim(j);
            }
            OP_LOGD(context->GetNodeName(), "input indicesShape = %s", VectorToString(indicesShape).c_str());
            indicesShapeList.push_back(indicesShape);
        }
    }
    if (CanBroadcast(context->GetNodeName(), indicesShapeList) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }
    auto broadShapeList = BroadcastAllShapes(indicesShapeList);
    OP_LOGD(context->GetNodeName(), "calculate broadShape = %s", VectorToString(broadShapeList[0]).c_str());
    auto outputShape = ComputeOutputShape(xShape, indexed_sizes_vec, broadShapeList[0]);
    OP_LOGD(context->GetNodeName(), "calculate outputShape = %s", VectorToString(outputShape).c_str());

    auto yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(static_cast<size_t>(outputShape.size()));
    for (size_t i = 0; i < outputShape.size(); i++) {
        yShape->SetDim(i, outputShape[i]);
    }
    OP_LOGD(context->GetNodeName(), "output yShape = %s", Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Index(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "index inferdataType is begin");
    auto input_value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_value_dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Index)
    .InferDataType(InferDataType4Index)
    .InferShape(InferShape4Index)
    .InputsDataDependency({IDX_SIZES});
} // namespace ops