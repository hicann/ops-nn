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
 * \file layer_norm_v3_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/runtime2_util.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
constexpr size_t input_x_index = 0;
constexpr size_t output_y_index = 0;
constexpr size_t output_mean_index = 1;
constexpr size_t output_rstd_index = 2;
constexpr size_t attr_begin_norm_axis_index = 0;

static graphStatus InferShape4LayerNormV3(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4LayerNormV3.");

    const gert::Shape* x_shape = context->GetInputShape(input_x_index);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(output_y_index);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* mean_shape = context->GetOutputShape(output_mean_index);
    OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);
    gert::Shape* rstd_shape = context->GetOutputShape(output_rstd_index);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* begin_norm_axis_ptr = attrs->GetAttrPointer<int64_t>(attr_begin_norm_axis_index);
    OP_CHECK_NULL_WITH_CONTEXT(context, begin_norm_axis_ptr);

    OP_CHECK_IF(
        !Ops::Nn::IsDimValid(x_shape->GetDimNum(), *begin_norm_axis_ptr),
        OP_LOGE(
            context->GetNodeName(), "%s",
            Ops::Nn::GenInvalidDimMsg("begin_norm_axis", x_shape->GetDimNum(), *begin_norm_axis_ptr).c_str()),
        return GRAPH_FAILED);

    int64_t begin_norm_axis_val = *begin_norm_axis_ptr < 0 ?
                                      *begin_norm_axis_ptr + static_cast<int64_t>(x_shape->GetDimNum()) :
                                      *begin_norm_axis_ptr;

    *y_shape = *x_shape;
    mean_shape->SetDimNum(x_shape->GetDimNum());
    rstd_shape->SetDimNum(x_shape->GetDimNum());

    for (size_t i = 0; i < x_shape->GetDimNum(); ++i) {
        if (static_cast<int64_t>(i) >= begin_norm_axis_val) {
            mean_shape->SetDim(i, 1);
            rstd_shape->SetDim(i, 1);
        } else {
            mean_shape->SetDim(i, x_shape->GetDim(i));
            rstd_shape->SetDim(i, x_shape->GetDim(i));
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4LayerNorm.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LayerNormV3).InferShape(InferShape4LayerNormV3);
} // namespace ops