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
 * \file instance_norm_grad_infershape.cpp
 * \brief InferShape/InferDataType for InstanceNormGrad. Mirrors A2 InstanceNormGradInferShape
 *        (nn_norm_ops.cc): pure SetShape/SetDataType, no validation, always GRAPH_SUCCESS.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t INSTANCENORMGRAD_IDX_IN_X = 1;
static constexpr size_t INSTANCENORMGRAD_IDX_IN_GAMMA = 4;
static constexpr size_t INSTANCENORMGRAD_IDX_OUT_PDX = 0;
static constexpr size_t INSTANCENORMGRAD_IDX_OUT_PDGAMMA = 1;
static constexpr size_t INSTANCENORMGRAD_IDX_OUT_PDBETA = 2;

static ge::graphStatus InstanceNormGradInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InstanceNormGradInferShape");

    const gert::Shape* x_shape = context->GetInputShape(INSTANCENORMGRAD_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    const gert::Shape* gamma_shape = context->GetInputShape(INSTANCENORMGRAD_IDX_IN_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);

    gert::Shape* pdx_shape = context->GetOutputShape(INSTANCENORMGRAD_IDX_OUT_PDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pdx_shape);
    gert::Shape* pdgamma_shape = context->GetOutputShape(INSTANCENORMGRAD_IDX_OUT_PDGAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, pdgamma_shape);
    gert::Shape* pdbeta_shape = context->GetOutputShape(INSTANCENORMGRAD_IDX_OUT_PDBETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, pdbeta_shape);

    *pdx_shape = *x_shape;
    *pdgamma_shape = *gamma_shape;
    *pdbeta_shape = *gamma_shape;

    OP_LOGD(context->GetNodeName(), "End to do InstanceNormGradInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InstanceNormGradInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InstanceNormGradInferDtype enter");
    auto inputDtype = context->GetInputDataType(0); // dy
    context->SetOutputDataType(INSTANCENORMGRAD_IDX_OUT_PDX, inputDtype);
    context->SetOutputDataType(INSTANCENORMGRAD_IDX_OUT_PDGAMMA, inputDtype);
    context->SetOutputDataType(INSTANCENORMGRAD_IDX_OUT_PDBETA, inputDtype);
    OP_LOGD(context->GetNodeName(), "InstanceNormGradInferDtype end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InstanceNormGrad).InferShape(InstanceNormGradInferShape).InferDataType(InstanceNormGradInferDtype);
} // namespace ops
