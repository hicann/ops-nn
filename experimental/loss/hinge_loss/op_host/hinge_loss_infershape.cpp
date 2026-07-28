/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static ge::graphStatus InferShapeHingeLoss(gert::InferShapeContext* context)
{
    const gert::Shape* predictShape = context->GetInputShape(0);
    const gert::Shape* targetShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
    OP_CHECK_IF(predictShape->GetDimNum() != targetShape->GetDimNum(),
                OP_LOGE(context, "predict and target rank mismatch"), return ge::GRAPH_FAILED);
    for (size_t i = 0; i < predictShape->GetDimNum(); ++i) {
        OP_CHECK_IF(predictShape->GetDim(i) != targetShape->GetDim(i),
                    OP_LOGE(context, "predict and target shape mismatch at dim %zu", i), return ge::GRAPH_FAILED);
    }
    gert::Shape* lossShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);
    *lossShape = *predictShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeHingeLoss(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HingeLoss).InferShape(InferShapeHingeLoss).InferDataType(InferDataTypeHingeLoss);
} // namespace ops
