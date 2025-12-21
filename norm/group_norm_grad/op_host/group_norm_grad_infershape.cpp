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
 * \file group_norm_grad_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t GROUPNORMGRAD_IDX_IN_DY = 0;
static constexpr size_t GROUPNORMGRAD_IDX_IN_GAMMA = 4;
static constexpr size_t GROUPNORMGRAD_IDX_OUT_DX = 0;
static constexpr size_t GROUPNORMGRAD_IDX_OUT_DGAMMA = 1;
static constexpr size_t GROUPNORMGRAD_IDX_OUT_DBETA = 2;

static ge::graphStatus GroupNormGradInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do GroupNormGradInferShape");

    // get input shapes
    const gert::Shape* dy_shape = context->GetInputShape(GROUPNORMGRAD_IDX_IN_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context, dy_shape);
    const gert::Shape* gamma_shape = context->GetInputShape(GROUPNORMGRAD_IDX_IN_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    // get output shapes
    gert::Shape* dx_shape = context->GetOutputShape(GROUPNORMGRAD_IDX_OUT_DX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dx_shape);
    gert::Shape* dgamma_shape = context->GetOutputShape(GROUPNORMGRAD_IDX_OUT_DGAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, dgamma_shape);
    gert::Shape* dbeta_shape = context->GetOutputShape(GROUPNORMGRAD_IDX_OUT_DBETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, dbeta_shape);

    *dx_shape = *dy_shape;
    *dgamma_shape = *gamma_shape;
    *dbeta_shape = *gamma_shape;

    OP_LOGD(context->GetNodeName(), "End to do GroupNormGradInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus GroupNormGradInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "GroupNormGradInferDtype enter");
    // Get input tout
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto inputDtype = context->GetInputDataType(GROUPNORMGRAD_IDX_IN_DY);
    context->SetOutputDataType(GROUPNORMGRAD_IDX_OUT_DX, inputDtype);
    context->SetOutputDataType(GROUPNORMGRAD_IDX_OUT_DGAMMA, inputDtype);
    context->SetOutputDataType(GROUPNORMGRAD_IDX_OUT_DBETA, inputDtype);

    OP_LOGD(context->GetNodeName(), "GroupNormGradInferDtype end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormGrad).InferShape(GroupNormGradInferShape).InferDataType(GroupNormGradInferDtype);
} // namespace ops
