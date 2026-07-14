/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layer_normalization_grad_infershape.cpp
 * \brief LayerNormalizationGrad 算子的 shape 推理和数据类型推理实现
 *
 * 输出: dx shape = dy shape, dgamma/dbeta shape = gamma shape
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_2 = 2;

static ge::graphStatus InferShapeLayerNormalizationGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeLayerNormalizationGrad");

    const gert::Shape* dyShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShape);

    const gert::Shape* gammaShape = context->GetInputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);

    gert::Shape* dxShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);

    gert::Shape* dgammaShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dgammaShape);

    gert::Shape* dbetaShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, dbetaShape);

    *dxShape = *dyShape;
    *dgammaShape = *gammaShape;
    *dbetaShape = *gammaShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeLayerNormalizationGrad");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLayerNormalizationGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeLayerNormalizationGrad");

    const auto inputDataType = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, inputDataType);
    context->SetOutputDataType(1, inputDataType);
    context->SetOutputDataType(2, inputDataType);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeLayerNormalizationGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LayerNormalizationGrad)
    .InferShape(InferShapeLayerNormalizationGrad)
    .InferDataType(InferDataTypeLayerNormalizationGrad);
} // namespace ops
