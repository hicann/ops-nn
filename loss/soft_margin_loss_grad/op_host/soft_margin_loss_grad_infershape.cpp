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
 * \file soft_margin_loss_grad_infershape.cpp
 * \brief SoftMarginLossGrad 形状/类型推导：out = numpy broadcast(self, target, grad_output)
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static int64_t BroadcastDim(int64_t a, int64_t b) {
    if (a == 1) return b;
    if (b == 1) return a;
    return a; // a==b 时取任一；冲突由 tiling CheckBroadcastShape 拦截
}

static ge::graphStatus InferShape4SoftMarginLossGrad(gert::InferShapeContext* context)
{
    const gert::Shape* s0 = context->GetInputShape(0);
    const gert::Shape* s1 = context->GetInputShape(1);
    const gert::Shape* s2 = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, s0);
    OP_CHECK_NULL_WITH_CONTEXT(context, s1);
    OP_CHECK_NULL_WITH_CONTEXT(context, s2);
    gert::Shape* out = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out);

    size_t r = s0->GetDimNum();
    r = std::max(r, s1->GetDimNum());
    r = std::max(r, s2->GetDimNum());
    out->SetDimNum(r);
    auto dimAt = [](const gert::Shape* s, size_t i, size_t r) -> int64_t {
        size_t sr = s->GetDimNum();
        return (i + sr < r) ? 1 : s->GetDim(i + sr - r);
    };
    for (size_t i = 0; i < r; i++) {
        int64_t d = BroadcastDim(BroadcastDim(dimAt(s0, i, r), dimAt(s1, i, r)), dimAt(s2, i, r));
        out->SetDim(i, d);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4SoftMarginLossGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftMarginLossGrad)
    .InferShape(InferShape4SoftMarginLossGrad)
    .InferDataType(InferDtype4SoftMarginLossGrad);

} // namespace ops
