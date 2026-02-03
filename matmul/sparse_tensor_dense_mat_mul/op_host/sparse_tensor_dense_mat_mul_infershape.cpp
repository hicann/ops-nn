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
 * \file sparse_tensor_dense_mat_mul_infershape.cpp
 * \brief
 */

#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "graph/utils/type_utils.h"

using namespace ge;
namespace ops {

static constexpr size_t SPARSETENSORDENSEMATMUL_IDX_X1_INDICES = 0;
static constexpr size_t SPARSETENSORDENSEMATMUL_IDX_X1_VALUES = 1;
static constexpr size_t SPARSETENSORDENSEMATMUL_IDX_X1_SHAPE = 2;
static constexpr size_t SPARSETENSORDENSEMATMUL_IDX_X2 = 3;
static constexpr size_t SPARSETENSORDENSEMATMUL_IDX_OUT_Y = 0;
static constexpr size_t SPARSETENSORDENSEMATMUL_IDX_ATTR_ADJOINT_A = 0;
static constexpr size_t SPARSETENSORDENSEMATMUL_IDX_ATTR_ADJOINT_B = 1;
static constexpr size_t SPARSETENSORDENSEMATMUL_NUM_ZERO = 0;
static constexpr size_t SPARSETENSORDENSEMATMUL_NUM_ONE = 1;
static constexpr size_t SPARSETENSORDENSEMATMUL_NUM_TWO = 2;
static constexpr int64_t SPARSETENSORDENSEMATMUL_UNKNOWNDIM = -1;

static bool IsUnknownRankShape(const std::vector<int64_t>& shape_vec)
{
    return (shape_vec==ge::UNKNOWN_RANK);
}

static bool IsUnknownShape(const std::vector<int64_t>& shape_vec)
{
 	auto found = find(shape_vec.begin(), shape_vec.end(), ge::UNKNOWN_DIM);
 	return found != shape_vec.end();
}

static bool IsUnknown(const gert::Shape* shape) {
    std::vector<int64_t> shape_vec;
    for(size_t i = 0; i<shape->GetDimNum(); i++){
        shape_vec.emplace_back(shape->GetDim(i));
    }
 	return (IsUnknownRankShape(shape_vec) || IsUnknownShape(shape_vec));
}

static bool CheckUnknownShape(const gert::Shape* x1IndicesShape, const gert::Shape* x1ValuesShape,
    const gert::Shape* x1ShapeShape, const gert::Shape* x2Shape)
{
    return (IsUnknown(x1IndicesShape) || IsUnknown(x1ValuesShape) || IsUnknown(x1ShapeShape) || IsUnknown(x2Shape));
}

static ge::graphStatus ValidShapeCheck(
    gert::InferShapeContext* context, const gert::Shape* x1IndicesShape, const gert::Shape* x1ValuesShape,
    const gert::Shape* x1ShapeShape, const gert::Shape* x2Shape)
{
    OP_CHECK_IF(
        x2Shape->GetDimNum() != SPARSETENSORDENSEMATMUL_NUM_TWO,
        OP_LOGE(context->GetNodeName(), "Tensor x2 is not a matrix."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        x1ShapeShape->GetDimNum() != SPARSETENSORDENSEMATMUL_NUM_ONE,
        OP_LOGE(context->GetNodeName(), "Tensor x1_shape is not a vector."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        x1ShapeShape->GetDim(0) != SPARSETENSORDENSEMATMUL_NUM_TWO,
        OP_LOGE(
            context->GetNodeName(), "Tensor x1_shape must have 2 elements."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        x1ValuesShape->GetDimNum() != SPARSETENSORDENSEMATMUL_NUM_ONE,
        OP_LOGE(context->GetNodeName(), "Tensor x1_values is not a vector."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        x1IndicesShape->GetDimNum() != SPARSETENSORDENSEMATMUL_NUM_TWO,
        OP_LOGE(context->GetNodeName(), "Tensor x1_indices is not a matrix."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        x1ValuesShape->GetDim(0) != x1IndicesShape->GetDim(0),
        OP_LOGE(
            context->GetNodeName(),
            "Number of rows of x1_indices does not match number of entries in x1_values."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        x1IndicesShape->GetDim(1) != SPARSETENSORDENSEMATMUL_NUM_TWO,
        OP_LOGE(
            context->GetNodeName(),
            "Number of columns of x1_indices does not match number of entries in x1_shape"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetX1ShapeValue(
    gert::InferShapeContext* context, const gert::Tensor* x1ShapeTensor, std::vector<int64_t>& x1ShapeValue)
{
    auto x1ShapeDtype = x1ShapeTensor->GetDataType();
    OP_CHECK_IF(
        x1ShapeDtype != ge::DT_INT64,
        OP_LOGE(context->GetNodeName(), "Tensor x1_shape only support int64"),
        return ge::GRAPH_FAILED);
    const int64_t* constDataPtr = x1ShapeTensor->GetData<int64_t>();
    if (constDataPtr) {
        for (size_t i = 0; i < SPARSETENSORDENSEMATMUL_NUM_TWO; i++) {
            x1ShapeValue[i] = *(constDataPtr + i);
        }
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

static void SetUnknownOutputShape(gert::InferShapeContext* context, gert::Shape* yShape)
{
    yShape->SetDimNum(SPARSETENSORDENSEMATMUL_NUM_TWO);
    yShape->SetDim(SPARSETENSORDENSEMATMUL_NUM_ZERO, SPARSETENSORDENSEMATMUL_UNKNOWNDIM);
    yShape->SetDim(SPARSETENSORDENSEMATMUL_NUM_ONE, SPARSETENSORDENSEMATMUL_UNKNOWNDIM);
    OP_LOGD(context->GetNodeName(), "InferShape4SparseTensorDenseMatMul end with unknown out shape.");
}

static void SetOutputShape(gert::InferShapeContext* context, gert::Shape* yShape, const int64_t &outerLeft,
                           const int64_t &outerRight)
{
    yShape->SetDimNum(SPARSETENSORDENSEMATMUL_NUM_TWO);
    yShape->SetDim(SPARSETENSORDENSEMATMUL_NUM_ZERO, outerLeft);
    yShape->SetDim(SPARSETENSORDENSEMATMUL_NUM_ONE, outerRight);
    OP_LOGD(context->GetNodeName(), "after infershape, the y shape is (%ld, %ld).", outerLeft, outerRight);
    OP_LOGD(context->GetNodeName(), "InferShape4SparseTensorDenseMatMul end success.");
}

static ge::graphStatus InferShape4SparseTensorDenseMatMul(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4SparseTensorDenseMatMul enter");
    auto x1IndicesShape = context->GetInputShape(SPARSETENSORDENSEMATMUL_IDX_X1_INDICES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1IndicesShape);
    auto x1ValuesShape = context->GetInputShape(SPARSETENSORDENSEMATMUL_IDX_X1_VALUES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1ValuesShape);
    auto x1ShapeShape = context->GetInputShape(SPARSETENSORDENSEMATMUL_IDX_X1_SHAPE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1ShapeShape);
    auto x2Shape = context->GetInputShape(SPARSETENSORDENSEMATMUL_IDX_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* adjointPointA = attrs->GetAttrPointer<bool>(SPARSETENSORDENSEMATMUL_IDX_ATTR_ADJOINT_A);
    OPS_CHECK_NULL_WITH_CONTEXT(context, adjointPointA);
    const bool* adjointPointB = attrs->GetAttrPointer<bool>(SPARSETENSORDENSEMATMUL_IDX_ATTR_ADJOINT_B);
    OPS_CHECK_NULL_WITH_CONTEXT(context, adjointPointB);
    auto yShape = context->GetOutputShape(SPARSETENSORDENSEMATMUL_IDX_OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    if (CheckUnknownShape(x1IndicesShape, x1ValuesShape, x1ShapeShape, x2Shape)) {
        SetUnknownOutputShape(context, yShape);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        ValidShapeCheck(context, x1IndicesShape, x1ValuesShape, x1ShapeShape, x2Shape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "ValidShapeCheck failed."),
        return ge::GRAPH_FAILED);
    auto x1ShapeTensor = context->GetInputTensor(SPARSETENSORDENSEMATMUL_IDX_X1_SHAPE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1ShapeTensor);
    std::vector<int64_t> x1ShapeValue = {0, 0};
    if (GetX1ShapeValue(context, x1ShapeTensor, x1ShapeValue) != ge::GRAPH_SUCCESS) {
        SetUnknownOutputShape(context, yShape);
        return ge::GRAPH_SUCCESS;
    }
    int64_t x2ShapeDims[SPARSETENSORDENSEMATMUL_NUM_TWO] = {x2Shape->GetDim(0), x2Shape->GetDim(1)};
    const int64_t outerLeft = (*adjointPointA) ? x1ShapeValue[1] : x1ShapeValue[0];
    const int64_t innerLeft = (*adjointPointA) ? x1ShapeValue[0] : x1ShapeValue[1];
    const int64_t innerRight = (*adjointPointB) ? x2ShapeDims[1] : x2ShapeDims[0];
    const int64_t outerRight = (*adjointPointB) ? x2ShapeDims[0] : x2ShapeDims[1];
    OP_CHECK_IF(
        innerRight != innerLeft,
        OP_LOGE(
            context->GetNodeName(),
            "Cannot multiply x1 and x2 because inner dimension does not match: %ld vs %ld. Did you forget a transpose?"
            "Dimension of x1:(%ld, %ld), Dimensions of x2: (%ld, %ld).", innerLeft, innerRight, x1ShapeValue[0],
            x1ShapeValue[1], x2ShapeDims[0], x2ShapeDims[1]),
        return ge::GRAPH_FAILED);
    SetOutputShape(context, yShape, outerLeft, outerRight);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4SparseTensorDenseMatMul(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4SparseTensorDenseMatMul enter");
    auto x1ValueDtype = context->GetInputDataType(SPARSETENSORDENSEMATMUL_IDX_X1_VALUES);
    context->SetOutputDataType(SPARSETENSORDENSEMATMUL_IDX_OUT_Y, x1ValueDtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4SparseTensorDenseMatMul end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseTensorDenseMatMul)
    .InputsDataDependency({SPARSETENSORDENSEMATMUL_IDX_X1_SHAPE})
    .InferShape(InferShape4SparseTensorDenseMatMul)
    .InferDataType(InferDataType4SparseTensorDenseMatMul);
} // namespace ops