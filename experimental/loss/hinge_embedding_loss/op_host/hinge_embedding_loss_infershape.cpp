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
#include <cstring>

namespace ops {
static bool SameShape(const gert::Shape* lhs, const gert::Shape* rhs)
{
    if (lhs->GetDimNum() != rhs->GetDimNum())
        return false;
    for (size_t i = 0; i < lhs->GetDimNum(); ++i)
        if (lhs->GetDim(i) != rhs->GetDim(i))
            return false;
    return true;
}
static ge::graphStatus InferShapeHingeEmbeddingLoss(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("HingeEmbeddingLoss", "context is null"), return ge::GRAPH_FAILED);
    const auto* input0 = context->GetInputShape(0);
    const auto* input1 = context->GetInputShape(1);
    auto* output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input1);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    OP_CHECK_IF(!SameShape(input0, input1), OP_LOGE(context, "input shapes must match"), return ge::GRAPH_FAILED);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* reduction = attrs->GetAttrPointer<char>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, reduction);
    if (std::strcmp(reduction, "none") == 0) {
        *output = *input0;
    } else if (std::strcmp(reduction, "sum") == 0 || std::strcmp(reduction, "mean") == 0) {
        *output = gert::Shape({1});
    } else {
        OP_LOGE(context, "reduction must be none, sum, or mean");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(HingeEmbeddingLoss).InferShape(InferShapeHingeEmbeddingLoss);
} // namespace ops
