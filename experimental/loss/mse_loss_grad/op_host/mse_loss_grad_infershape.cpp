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
 * \file mse_loss_grad_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeMseLossGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMseLossGrad");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    *yShape = *xShape; 

    OP_LOGD(context->GetNodeName(), "End to do InferShapeMseLossGrad");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4MseLossGrad(gert::InferDataTypeContext* context) {
    OP_LOGD(context->GetNodeName(), "InferDataType4MseLossGrad enter");
    auto input_x_dtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, input_x_dtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4MseLossGrad end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MseLossGrad).InferShape(InferShapeMseLossGrad).InferDataType(InferDataType4MseLossGrad);
} // namespace ops