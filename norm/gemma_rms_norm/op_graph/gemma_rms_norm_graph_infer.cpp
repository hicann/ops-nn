/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * \file gemma_rms_norm_graph_infer.cpp
 * \brief gemma_rms_norm operater graph infer resource
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

static constexpr int INPUT_NODE_NUM = 2;
static constexpr int OUTPUT_NODE_NUM = 2;
static constexpr int X_IDX = 0;
static constexpr int GAMMA_IDX = 1;
static constexpr int Y_IDX = 0;
static constexpr int RSTD_IDX = 1;

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4GemmaRmsNorm(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("InferShape4GemmaRmsNorm", "Context is nullptr, check failed.");
        return GRAPH_FAILED;
    }
    if (context->GetComputeNodeInputNum() != INPUT_NODE_NUM ||
        context->GetComputeNodeOutputNum() != OUTPUT_NODE_NUM) {
        return GRAPH_FAILED;
    }

    const gert::Shape* x_shape = context->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    const gert::Shape* gamma_shape = context->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    gert::Shape* y_shape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* rstd_shape = context->GetOutputShape(RSTD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);

    *y_shape = *x_shape;

    if (Ops::Base::IsUnknownRank(*x_shape)) {
        *rstd_shape = *x_shape;
        OP_LOGI(context, "End to do InferShape4GemmaRmsNorm, x shape is [-2].");
        return GRAPH_SUCCESS;
    }
    if (Ops::Base::IsUnknownRank(*gamma_shape)) {
        *rstd_shape = *gamma_shape;
        OP_LOGI(context, "End to do InferShape4GemmaRmsNorm, gamma shape is [-2].");
        return GRAPH_SUCCESS;
    }
    if (x_shape->GetDimNum() < gamma_shape->GetDimNum()) {
        OP_LOGE(context, "Dim num of x should be no less than dim num of gamma.");
        return GRAPH_FAILED;
    }

    auto shape(*x_shape);
    size_t x_dim_num = x_shape->GetDimNum();
    size_t gamma_dim_num = gamma_shape->GetDimNum();
    for (size_t i = 0; i < x_dim_num; i++) {
        if (i < x_dim_num - gamma_dim_num) {
            shape.SetDim(i, x_shape->GetDim(i));
        } else {
            shape.SetDim(i, 1);
        }
    }
    *rstd_shape = shape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GemmaRmsNorm(gert::InferDataTypeContext* context)
{
    const auto input_data_type_0 = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_data_type_0);

    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GemmaRmsNorm)
    .InferShape(InferShape4GemmaRmsNorm)
    .InferDataType(InferDataType4GemmaRmsNorm);

} // namespace ops