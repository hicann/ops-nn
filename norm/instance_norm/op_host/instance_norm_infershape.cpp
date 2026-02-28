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
 * \file instance_norm_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {

static ge::graphStatus InferShape4InstanceNorm(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4InstanceNorm");

    // get input shapes
    const gert::Shape* x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto x_desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_desc);
    auto x_format = x_desc->GetOriginFormat();

    // get output shapes
    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* mean_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);
    gert::Shape* var_shape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_shape);

    *y_shape = *x_shape;
    size_t x_dim_num = x_shape->GetDimNum();
    mean_shape->SetDimNum(x_dim_num);
    var_shape->SetDimNum(x_dim_num);

    for (size_t i = 0; i < x_dim_num; i++) {
        if (i == 0) {
            mean_shape->SetDim(i, x_shape->GetDim(i));
            var_shape->SetDim(i, x_shape->GetDim(i));
        } else if (i == 1 && (x_format == ge::FORMAT_NCHW || x_format == ge::FORMAT_NCDHW || x_format == ge::FORMAT_ND)) {
            mean_shape->SetDim(i, x_shape->GetDim(i));
            var_shape->SetDim(i, x_shape->GetDim(i));
        } else if (i == (x_dim_num - 1) && (x_format == ge::FORMAT_NHWC || x_format == ge::FORMAT_NDHWC)) {
            mean_shape->SetDim(i, x_shape->GetDim(i));
            var_shape->SetDim(i, x_shape->GetDim(i));
        } else {
            mean_shape->SetDim(i, 1);
            var_shape->SetDim(i, 1);
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4InstanceNorm");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4InstanceNorm(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4InstanceNorm");
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    context->SetOutputDataType(2, context->GetInputDataType(1));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4InstanceNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InstanceNorm).InferShape(InferShape4InstanceNorm).InferDataType(InferDataType4InstanceNorm);
} // namespace ops
