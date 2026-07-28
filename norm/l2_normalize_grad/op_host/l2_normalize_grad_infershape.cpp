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
 * \file l2_normalize_grad_infershape.cpp
 * \brief L2NormalizeGrad infershape: dx has the shape and dtype of x (== y == dy).
 */

#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_X_IDX = 0;
static constexpr size_t OUTPUT_DX_IDX = 0;

static ge::graphStatus InferShape4L2NormalizeGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4L2NormalizeGrad.");
    const gert::Shape* x_shape = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* dx_shape = context->GetOutputShape(OUTPUT_DX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dx_shape);
    *dx_shape = *x_shape;
    OP_LOGD(context, "End to do InferShape4L2NormalizeGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4L2NormalizeGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4L2NormalizeGrad");
    context->SetOutputDataType(OUTPUT_DX_IDX, context->GetInputDataType(INPUT_X_IDX));
    OP_LOGD(context, "End to do InferDataType4L2NormalizeGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(L2NormalizeGrad).InferShape(InferShape4L2NormalizeGrad).InferDataType(InferDataType4L2NormalizeGrad);
} // namespace ops
