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
 * \file deep_norm_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

static constexpr int INPUT_X_INDEX = 0;
static constexpr int INPUT_GAMMA_INDEX = 3;
static constexpr int OUTPUT_MEAN_INDEX = 0;
static constexpr int OUTPUT_RSTD_INDEX = 1;
static constexpr int OUTPUT_Y_INDEX = 2;
using namespace ge;
namespace ops {

static ge::graphStatus InferShape4DeepNorm(gert::InferShapeContext* context)
{
    // infershape
    const gert::Shape* x_shape = context->GetInputShape(INPUT_X_INDEX);
    const gert::Shape* gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);

    size_t x_dim_num = x_shape->GetDimNum();
    size_t gamma_dim_num = gamma_shape->GetDimNum();
    OP_CHECK_IF(
        x_dim_num <= gamma_dim_num, OP_LOGE(context, "x dim num should not be smaller than gamma dim num."),
        return GRAPH_FAILED);

    gert::Shape* mean_shape = context->GetOutputShape(OUTPUT_MEAN_INDEX);
    gert::Shape* rstd_shape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    gert::Shape* y_shape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    mean_shape->SetDimNum(x_dim_num);
    rstd_shape->SetDimNum(x_dim_num);

    for (size_t i = 0; i < x_dim_num; i++) {
        if (i < x_dim_num - gamma_dim_num) {
            rstd_shape->SetDim(i, x_shape->GetDim(i));
            mean_shape->SetDim(i, x_shape->GetDim(i));
        } else {
            rstd_shape->SetDim(i, 1);
            mean_shape->SetDim(i, 1);
        }
    }

    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType4DeepNorm(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4DeepNorm");
    context->SetOutputDataType(OUTPUT_MEAN_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_RSTD_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    OP_LOGD(context, "End to do InferDataType4DeepNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DeepNorm).InferShape(InferShape4DeepNorm).InferDataType(InferDataType4DeepNorm);
} // namespace ops
