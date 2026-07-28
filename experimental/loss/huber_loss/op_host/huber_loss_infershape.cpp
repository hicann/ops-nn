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
static bool SameShape(const gert::Shape* left, const gert::Shape* right)
{
    if (left->GetDimNum() != right->GetDimNum())
        return false;
    for (size_t i = 0; i < left->GetDimNum(); ++i)
        if (left->GetDim(i) != right->GetDim(i))
            return false;
    return true;
}
static ge::graphStatus InferShapeHuberLoss(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const auto* predictions = context->GetInputShape(0);
    const auto* targets = context->GetInputShape(1);
    auto* loss = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictions);
    OP_CHECK_NULL_WITH_CONTEXT(context, targets);
    OP_CHECK_NULL_WITH_CONTEXT(context, loss);
    OP_CHECK_IF(!SameShape(predictions, targets), OP_LOGE(context, "HuberLoss requires equal input shapes"),
                return ge::GRAPH_FAILED);
    *loss = *predictions;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(HuberLoss).InferShape(InferShapeHuberLoss);
} // namespace ops
