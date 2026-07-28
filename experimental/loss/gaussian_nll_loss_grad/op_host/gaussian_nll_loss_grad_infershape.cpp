/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeGaussianNllLossGrad(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GaussianNllLossGrad", "context is null"), return ge::GRAPH_FAILED);
    const auto* inputShape = context->GetInputShape(1);
    const auto* varShape = context->GetInputShape(3);
    auto* gradInputShape = context->GetOutputShape(0);
    auto* gradVarShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradInputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradVarShape);
    *gradInputShape = *inputShape;
    *gradVarShape = *varShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGaussianNllLossGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GaussianNllLossGrad", "context is null"), return ge::GRAPH_FAILED);
    const ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GaussianNllLossGrad)
    .InferShape(InferShapeGaussianNllLossGrad)
    .InferDataType(InferDataTypeGaussianNllLossGrad);
} // namespace ops
