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
 * \file rms_norm_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4RmsNorm(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4RmsNorm");

    // get input shapes
    const gert::Shape* x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    const gert::Shape* gamma_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    // get output shapes
    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    *y_shape = *x_shape;

    size_t x_dim_num = x_shape->GetDimNum();
    size_t gamma_dim_num = gamma_shape->GetDimNum();

    gert::Shape* rstd_shape = context->GetOutputShape(1);
    rstd_shape->SetDimNum(x_dim_num);
    for (size_t i = 0; i < x_dim_num; i++) {
        if (i < x_dim_num - gamma_dim_num) {
            rstd_shape->SetDim(i, x_shape->GetDim(i));
        } else {
            rstd_shape->SetDim(i, 1);
        }
    }

    OP_LOGD(context, "End to do InferShape4RmsNorm");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4RmsNorm(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4RmsNorm");
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, DT_FLOAT);
    OP_LOGD(context, "End to do InferDataType4RmsNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNorm).InferShape(InferShape4RmsNorm).InferDataType(InferDataType4RmsNorm);
} // namespace ops
