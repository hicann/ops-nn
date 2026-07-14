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
 * \file huber_loss_grad_infershape.cpp
 * \brief HuberLossGrad算子的shape推理和数据类型推理实现
 *
 * 本文件提供输出张量shape和数据类型推理逻辑。
 * 输出shape和数据类型与输入predictions相同。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeHuberLossGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeHuberLossGrad");

    const gert::Shape* predictionsShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictionsShape);

    gert::Shape* gradOutputShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradOutputShape);

    *gradOutputShape = *predictionsShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeHuberLossGrad");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeHuberLossGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeHuberLossGrad");

    ge::DataType inputDataType = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, inputDataType);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeHuberLossGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HuberLossGrad).InferShape(InferShapeHuberLossGrad).InferDataType(InferDataTypeHuberLossGrad);
} // namespace ops
