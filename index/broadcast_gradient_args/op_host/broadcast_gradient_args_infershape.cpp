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
 * \file broadcast_gradient_args_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "op_api/op_util.h"

using namespace ge;

namespace ops {
static constexpr size_t BROADCASTGRADIENTARGS_IN_IDX_X1 = 0;
static constexpr size_t BROADCASTGRADIENTARGS_IN_IDX_X2 = 1;
static constexpr size_t BROADCASTGRADIENTARGS_OUT_IDX_Y1 = 0;
static constexpr size_t BROADCASTGRADIENTARGS_OUT_IDX_Y2 = 1;

template <typename T>
static bool BroadcastGradientArgsTensorCheck(
    const gert::Tensor* small_tensor, const gert::Tensor* big_tensor, size_t small_len, size_t big_len,
    bool& shape_equal_flag)
{
    const T* small_array = small_tensor->GetData<T>();
    const T* big_array = big_tensor->GetData<T>();
    for (size_t i = 0; i < small_len; i++) {
        shape_equal_flag = shape_equal_flag && (small_array[small_len - i - 1] == big_array[big_len - i - 1]);
        if ((small_array[small_len - i - 1] != big_array[big_len - i - 1]) && (small_array[small_len - i - 1] != 1) &&
            (big_array[big_len - i - 1] != 1)) {
            return false;
        }
    }
    return true;
}

template <typename T>
static void ComputeOutputDims(
    const gert::Tensor* x1_tensor, const gert::Tensor* x2_tensor, size_t x1_len, size_t x2_len, size_t max_len,
    size_t& y1_dim, size_t& y2_dim)
{
    const T* x1_array = x1_tensor->GetData<T>();
    const T* x2_array = x2_tensor->GetData<T>();
    for (size_t i = 0; i < max_len; i++) {
        if (i >= x1_len) {
            y1_dim++;
        } else {
            if (x1_array[i] == 1) {
                y1_dim++;
            }
        }
        if (i >= x2_len) {
            y2_dim++;
        } else {
            if (x2_array[i] == 1) {
                y2_dim++;
            }
        }
    }
    return;
}

template <typename T>
static ge::graphStatus InferShape4BroadcastGradientArgsImpl(
    const gert::InferShapeContext* context, const gert::Tensor* x1_tensor, const gert::Tensor* x2_tensor,
    const gert::Shape* x1_shape, const gert::Shape* x2_shape, gert::Shape* y1_shape, gert::Shape* y2_shape)
{
    // 设置输出shape为-1
    y1_shape->SetDimNum(1);
    y2_shape->SetDimNum(1);
    y1_shape->SetDim(0, UNKNOWN_DIM);
    y2_shape->SetDim(0, UNKNOWN_DIM);

    if (!Ops::Nn::IsConstTensor(x1_tensor) || !Ops::Nn::IsConstTensor(x2_tensor)) {
        return ge::GRAPH_SUCCESS;
    }

    size_t x1_shape_len = x1_shape->GetDim(0);
    size_t x2_shape_len = x2_shape->GetDim(0);
    size_t max_shape_len = std::max(x1_shape_len, x2_shape_len);

    if (max_shape_len == 0) {
        y1_shape->SetDim(0, 0);
        y2_shape->SetDim(0, 0);
        return ge::GRAPH_SUCCESS;
    }

    bool shape_equal_flag = true;
    if (x1_shape_len < max_shape_len) {
        if (!BroadcastGradientArgsTensorCheck<T>(x1_tensor, x2_tensor, x1_shape_len, x2_shape_len, shape_equal_flag)) {
            OP_LOGE(context->GetNodeName(), "Inputs x1 and x2 do not satisfy broadcasting rules");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (!BroadcastGradientArgsTensorCheck<T>(x2_tensor, x1_tensor, x2_shape_len, x1_shape_len, shape_equal_flag)) {
            OP_LOGE(context->GetNodeName(), "Inputs x1 and x2 do not satisfy broadcasting rules");
            return ge::GRAPH_FAILED;
        }
    }

    if ((x1_shape_len == x2_shape_len) && shape_equal_flag) {
        y1_shape->SetDim(0, 0);
        y2_shape->SetDim(0, 0);
        return ge::GRAPH_SUCCESS;
    }

    size_t y1_dim = 0;
    size_t y2_dim = 0;
    ComputeOutputDims<T>(x1_tensor, x2_tensor, x1_shape_len, x2_shape_len, max_shape_len, y1_dim, y2_dim);

    y1_shape->SetDim(0, y1_dim);
    y2_shape->SetDim(0, y2_dim);

    return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4BroadcastGradientArgs(gert::InferShapeContext* context)
{
    // 获取输入输出的tensor和shape
    const gert::Tensor* x1_tensor = context->GetInputTensor(BROADCASTGRADIENTARGS_IN_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_tensor);
    const gert::Tensor* x2_tensor = context->GetInputTensor(BROADCASTGRADIENTARGS_IN_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2_tensor);
    // 获取输入输出的shape
    const gert::Shape* x1_shape = context->GetInputShape(BROADCASTGRADIENTARGS_IN_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    const gert::Shape* x2_shape = context->GetInputShape(BROADCASTGRADIENTARGS_IN_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    gert::Shape* y1_shape = context->GetOutputShape(BROADCASTGRADIENTARGS_OUT_IDX_Y1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1_shape);
    gert::Shape* y2_shape = context->GetOutputShape(BROADCASTGRADIENTARGS_OUT_IDX_Y2);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2_shape);
    ge::DataType x1_dtype = x1_tensor->GetDataType();
    ge::DataType x2_dtype = x2_tensor->GetDataType();
    if (x1_dtype != x2_dtype) {
        OP_LOGE(context->GetNodeName(), "x1 and x2 must have same data type!");
        return ge::GRAPH_FAILED;
    }
    // 设置输出shape为-1
    y1_shape->SetDimNum(1);
    y2_shape->SetDimNum(1);
    y1_shape->SetDim(0, UNKNOWN_DIM);
    y2_shape->SetDim(0, UNKNOWN_DIM);

    switch (x1_dtype) {
        case ge::DT_INT32: {
            return InferShape4BroadcastGradientArgsImpl<int32_t>(
                context, x1_tensor, x2_tensor, x1_shape, x2_shape, y1_shape, y2_shape);
        }
        case ge::DT_INT64: {
            return InferShape4BroadcastGradientArgsImpl<int64_t>(
                context, x1_tensor, x2_tensor, x1_shape, x2_shape, y1_shape, y2_shape);
        }
        default: {
            OP_LOGE(context->GetNodeName(), "x1 and x2 only support INT_32 or INT_64!");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeRange4BroadcastGradientArgs(gert::InferShapeRangeContext* context)
{
    OP_LOGI(context->GetNodeName(), "InferShapeRange4BroadcastGradientArgs running begin");
    auto x1_range = context->GetInputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_range);
    auto x2_range = context->GetInputShapeRange(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2_range);

    auto y1_range = context->GetOutputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1_range);
    auto y2_range = context->GetOutputShapeRange(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2_range);

    OP_CHECK_NULL_WITH_CONTEXT(context, x1_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, x2_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, y1_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, y2_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, y1_range->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, y2_range->GetMin());

    int64_t x1Max = x1_range->GetMax()->GetDim(0);
    int64_t x2Max = x2_range->GetMax()->GetDim(0);
    int64_t outMax = x1Max > x2Max ? x1Max : x2Max;

    y1_range->GetMax()->SetDimNum(1);
    y1_range->GetMax()->SetDim(0, outMax);
    y2_range->GetMax()->SetDimNum(1);
    y2_range->GetMax()->SetDim(0, outMax);

    y1_range->GetMin()->SetDimNum(1);
    y1_range->GetMin()->SetDim(0, 0);
    y2_range->GetMin()->SetDimNum(1);
    y2_range->GetMin()->SetDim(0, 0);

    OP_LOGD(context->GetNodeName(), "Get y1 shape MAX %s.", Ops::Base::ToString(*y1_range->GetMax()).c_str());
    OP_LOGD(context->GetNodeName(), "Get y2 shape MAX %s.", Ops::Base::ToString(*y2_range->GetMax()).c_str());
    OP_LOGD(context->GetNodeName(), "Get y1 shape MIN %s.", Ops::Base::ToString(*y1_range->GetMin()).c_str());
    OP_LOGD(context->GetNodeName(), "Get y2 shape MIN %s.", Ops::Base::ToString(*y2_range->GetMin()).c_str());
    OP_LOGI(context->GetNodeName(), "InferShapeRange4BroadcastGradientArgs run success");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4BroadcastGradientArgs(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto y1_dtype = context->GetInputDataType(BROADCASTGRADIENTARGS_IN_IDX_X1);
    auto y2_dtype = context->GetInputDataType(BROADCASTGRADIENTARGS_IN_IDX_X2);
    context->SetOutputDataType(BROADCASTGRADIENTARGS_OUT_IDX_Y1, y1_dtype);
    context->SetOutputDataType(BROADCASTGRADIENTARGS_OUT_IDX_Y2, y2_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BroadcastGradientArgs)
    .InferShape(InferShape4BroadcastGradientArgs)
    .InferDataType(InferDataType4BroadcastGradientArgs)
    .InferShapeRange(InferShapeRange4BroadcastGradientArgs)
    .OutputShapeDependOnCompute({BROADCASTGRADIENTARGS_OUT_IDX_Y1, BROADCASTGRADIENTARGS_OUT_IDX_Y2});
} // namespace ops