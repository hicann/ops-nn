/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape4SigmoidCrossEntropyWithLogitsGradV2(gert::InferShapeContext* context)
{
    const auto* predictShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    auto* gradientShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradientShape);
    *gradientShape = *predictShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4SigmoidCrossEntropyWithLogitsGradV2(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}

IMPL_OP_INFERSHAPE(SigmoidCrossEntropyWithLogitsGradV2)
    .InferShape(InferShape4SigmoidCrossEntropyWithLogitsGradV2)
    .InferDataType(InferDataType4SigmoidCrossEntropyWithLogitsGradV2);
} // namespace ops
