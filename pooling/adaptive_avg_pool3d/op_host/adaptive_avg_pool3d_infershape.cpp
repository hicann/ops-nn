/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
using namespace ge;

namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;
constexpr size_t OUTPUT_SIZE_INDEX = 0;

constexpr size_t X_DIMS = 5;
constexpr size_t OUTPUT_SIZE_DIMS = 3;
} // namespace

namespace ops {
static ge::graphStatus InferShape4AdaptiveAvgPool3d(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    auto attr_ptr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_ptr);
    auto output_size_ptr = attr_ptr->GetAttrPointer<gert::ContinuousVector>(OUTPUT_SIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_size_ptr);
    auto output_size = static_cast<const int64_t*>(output_size_ptr->GetData());

    if (Ops::Base::IsUnknownRank(*x_shape)) {
        Ops::Base::SetUnknownRank(*y_shape);
        return ge::GRAPH_SUCCESS;
    }

    size_t input_dim_num = x_shape->GetDimNum();
    OP_CHECK_IF(
        input_dim_num != X_DIMS,
        OP_LOGE(context, "The dims of x not equal 5."), return GRAPH_FAILED);
    y_shape->SetDimNum(input_dim_num);

    size_t output_size_len = output_size_ptr->GetSize();
    OP_CHECK_IF(
        output_size_len != OUTPUT_SIZE_DIMS,
        OP_LOGE(context, "The size of output_size not equal 3."),
        return GRAPH_FAILED);

    y_shape->SetDim(0, x_shape->GetDim(0));
    for (size_t i = 0; i < output_size_len; ++i) {
        y_shape->SetDim(i + 1, output_size[i]);
    }
    y_shape->SetDim(output_size_len + 1, x_shape->GetDim(output_size_len + 1));

    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4AdaptiveAvgPool3d(gert::InferDataTypeContext* context)
{
    auto x_dtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(Y_INDEX, x_dtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaptiveAvgPool3d)
    .InferShape(InferShape4AdaptiveAvgPool3d)
    .InferDataType(InferDtype4AdaptiveAvgPool3d);

} // namespace ops