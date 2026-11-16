/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4FusedMulApplyMomentum(gert::InferShapeContext* context)
{
    const gert::Shape* varShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    const gert::Shape* accumShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, accumShape);
    gert::Shape* outVarShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outVarShape);
    gert::Shape* outAccumShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outAccumShape);

    // 与rt1.0（TwoInOneOutDynamicInferNoBroadcast）一致：两输出均取var的shape；
    // var为未知rank时输出取accum的shape；var维为-1且accum对应维已知时用accum的值解析
    gert::Shape resolvedShape = *varShape;
    if (Ops::Base::IsUnknownRank(*varShape)) {
        resolvedShape = *accumShape;
    } else if (!Ops::Base::IsUnknownRank(*accumShape) && varShape->GetDimNum() == accumShape->GetDimNum()) {
        for (size_t i = 0; i < resolvedShape.GetDimNum(); i++) {
            if (resolvedShape.GetDim(i) == -1 && accumShape->GetDim(i) != -1) {
                resolvedShape.SetDim(i, accumShape->GetDim(i));
            }
        }
    }

    *outVarShape = resolvedShape;
    *outAccumShape = resolvedShape;
    OP_LOGI(context->GetNodeName(), "[InferShape] output shape=%s", Ops::Base::ToString(resolvedShape).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedMulApplyKerasMomentum).InferShape(InferShape4FusedMulApplyMomentum);

} // namespace ops
