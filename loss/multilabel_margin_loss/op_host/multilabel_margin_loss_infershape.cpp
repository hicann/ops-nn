/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#include <cstring>
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeMultilabelMarginLoss(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* is_target_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, is_target_shape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // reduction is a String attr ("none"/"mean"/"sum"); consistent with tiling and the aclnn launcher.
    const char* reductionStr = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionStr);

    // reduction == none: per-sample loss, output 1D (N) for a 2D (N, C) input,
    // scalar for a 1D (C) input. reduction == mean/sum: scalar.
    if (strcmp(reductionStr, "none") == 0 && x1_shape->GetDimNum() >= 2) {
        y_shape->SetDimNum(1);
        y_shape->SetDim(0, x1_shape->GetDim(0));
    } else {
        y_shape->SetDimNum(0);
    }

    *is_target_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MultilabelMarginLoss).InferShape(InferShapeMultilabelMarginLoss);
} // namespace ops
