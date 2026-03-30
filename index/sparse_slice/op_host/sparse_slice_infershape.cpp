/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_slice_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
constexpr size_t kInputIndex0 = 0U;
constexpr size_t kInputIndex1 = 1U;
constexpr size_t kInputIndex2 = 2U;
constexpr size_t kInputIndex3 = 3U;
constexpr size_t kInputIndex4 = 4U;
constexpr size_t kInputIndex5 = 5U;

constexpr size_t kOutputIndex0 = 0U;
constexpr size_t kOutputIndex1 = 1U;
constexpr size_t kOutputIndex2 = 2U;
constexpr size_t kOutputIndex3 = 3U;

constexpr int64_t kRank0 = 0U;
constexpr int64_t kRank1 = 1U;
constexpr int64_t kRank2 = 2U;
constexpr int64_t kRank3 = 3U;

constexpr int64_t kNum0 = 0U;
constexpr int64_t kNum1 = 1U;
constexpr int64_t kNum2 = 2U;

constexpr int64_t OUTPUT_INDICES_SHAPE_RANK = 2;
constexpr int64_t OUTPUT_VALUES_SHAPE_RANK = 1;
constexpr int64_t SHAPE_IDX = 2;
constexpr int64_t START_IDX = 3;
constexpr int64_t SIZE_IDX = 4;

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    OP_CHECK_IF(
        output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
        return ge::GRAPH_FAILED);
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRangeForSparseSlice(gert::InferShapeRangeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeRangeForSparseSlice");
    auto indices_range = context->GetInputShapeRange(kInputIndex0);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices_range);
    auto shape_range = context->GetInputShapeRange(kInputIndex2);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_range);
    auto y_indices_range = context->GetOutputShapeRange(kOutputIndex0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_indices_range);
    auto y_values_range = context->GetOutputShapeRange(kOutputIndex1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_values_range);
    auto y_shape_range = context->GetOutputShapeRange(kOutputIndex2);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape_range);

    OP_CHECK_NULL_WITH_CONTEXT(context, indices_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, y_indices_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, y_indices_range->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, y_values_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, y_values_range->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape_range->GetMin());

    int64_t point_num = indices_range->GetMax()->GetDim(kInputIndex0);
    int64_t point_dim = indices_range->GetMax()->GetDim(kInputIndex1);
    y_indices_range->GetMax()->SetDimNum(OUTPUT_INDICES_SHAPE_RANK);
    y_indices_range->GetMax()->SetDim(kOutputIndex0, point_num);
    y_indices_range->GetMax()->SetDim(kOutputIndex1, point_dim);
    y_indices_range->GetMin()->SetDimNum(OUTPUT_INDICES_SHAPE_RANK);
    y_indices_range->GetMin()->SetDim(kOutputIndex0, 0);
    y_indices_range->GetMin()->SetDim(kOutputIndex1, point_dim);

    y_values_range->GetMax()->SetDimNum(OUTPUT_VALUES_SHAPE_RANK);
    y_values_range->GetMax()->SetDim(kOutputIndex0, point_num);
    y_values_range->GetMin()->SetDimNum(OUTPUT_VALUES_SHAPE_RANK);
    y_values_range->GetMin()->SetDim(kOutputIndex0, 0);

    y_shape_range->GetMax()->SetDimNum(OUTPUT_VALUES_SHAPE_RANK);
    y_shape_range->GetMax()->SetDim(kOutputIndex0, point_dim);
    y_shape_range->GetMin()->SetDimNum(OUTPUT_VALUES_SHAPE_RANK);
    y_shape_range->GetMin()->SetDim(kOutputIndex0, point_dim);
    OP_LOGD(
        context->GetNodeName(), "Get y_indices_range MAX %s.",
        Ops::Base::ToString(*(y_indices_range->GetMax())).c_str());
    OP_LOGD(
        context->GetNodeName(), "Get y_indices_range MIN %s.",
        Ops::Base::ToString(*(y_indices_range->GetMin())).c_str());
    OP_LOGD(
        context->GetNodeName(), "Get y_values_range MAX %s.", Ops::Base::ToString(*(y_values_range->GetMax())).c_str());
    OP_LOGD(
        context->GetNodeName(), "Get y_values_range MIN %s.", Ops::Base::ToString(*(y_values_range->GetMin())).c_str());
    OP_LOGD(
        context->GetNodeName(), "Get y_shape_range MAX %s.", Ops::Base::ToString(*(y_shape_range->GetMax())).c_str());
    OP_LOGD(
        context->GetNodeName(), "Get y_shape_range MIN %s.", Ops::Base::ToString(*(y_shape_range->GetMin())).c_str());
    OP_LOGI(context->GetNodeName(), "InferShapeRangeForSparseSlice run success");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForSparseSlice(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForSparseSlice");
    const gert::Shape* shape_shape = context->GetInputShape(kInputIndex2);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_shape);
    gert::Shape* y_indices_shape = context->GetOutputShape(kOutputIndex0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_indices_shape);
    y_indices_shape->SetDimNum(kNum2);
    y_indices_shape->SetDim(kOutputIndex0, UNKNOWN_DIM);
    y_indices_shape->SetDim(kOutputIndex1, (shape_shape->GetDim(kInputIndex0)));
    gert::Shape* y_values_shape = context->GetOutputShape(kOutputIndex1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_values_shape);
    if (SetAllUnknownDim(kRank1, y_values_shape) != GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "set all unknown dim failed.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* y_shape_shape = context->GetOutputShape(kOutputIndex2);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape_shape);
    *y_shape_shape = *shape_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForSparseSlice(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferDtypeForSparseSlice");
    if (context->SetOutputDataType(kOutputIndex0, DT_INT64) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "set output[y_indices] data type failed");
        return ge::GRAPH_FAILED;
    }
    DataType values_type = context->GetInputDataType(kInputIndex1);
    if (context->SetOutputDataType(kInputIndex1, values_type) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "set output[y_values] data type failed");
        return ge::GRAPH_FAILED;
    }
    if (context->SetOutputDataType(kInputIndex2, DT_INT64) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "set output[y_shape] data type failed");
        return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SparseSlice)
    .InferShape(InferShapeForSparseSlice)
    .InferShapeRange(InferShapeRangeForSparseSlice)
    .InferDataType(InferDtypeForSparseSlice)
    .InputsDataDependency({SHAPE_IDX, START_IDX, SIZE_IDX});
} // namespace ops