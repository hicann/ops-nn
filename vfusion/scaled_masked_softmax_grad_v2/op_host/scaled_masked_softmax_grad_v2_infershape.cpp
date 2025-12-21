/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scaled_masked_softmax_grad_v2_infershape.cpp
 * \brief
 */
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_Y_GRAD_IDX = 0;
static constexpr size_t INPUT_Y_IDX = 1;
static constexpr size_t INPUT_MASK_IDX = 2;
static constexpr size_t OUTPUT_X_GRAD_IDX = 0;

static ge::graphStatus ScaledMaskedSoftmaxGradV2InferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do ScaledMaskedSoftmaxGradV2InferShape");

    // get input shapes
    const gert::Shape* yGradShape = context->GetInputShape(INPUT_Y_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yGradShape);
    const gert::Shape* yShape = context->GetInputShape(INPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Shape* maskShape = context->GetInputShape(INPUT_MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape);
    // get output shapes
    gert::Shape* xGradShape = context->GetOutputShape(OUTPUT_X_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xGradShape);

    *xGradShape = *yGradShape;

    OP_LOGD(context, "End to do ScaledMaskedSoftmaxGradV2InferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus ScaledMaskedSoftmaxGradV2InferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "ScaledMaskedSoftmaxGradV2InferDtype enter");
    // Get input tout
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto inputDtype = context->GetInputDataType(INPUT_Y_GRAD_IDX);
    context->SetOutputDataType(OUTPUT_X_GRAD_IDX, inputDtype);

    OP_LOGD(context, "ScaledMaskedSoftmaxGradV2InferDtype end");

    return GRAPH_SUCCESS;
}

} // namespace ops

namespace ops {
IMPL_OP_INFERSHAPE(ScaledMaskedSoftmaxGradV2)
    .InferShape(ScaledMaskedSoftmaxGradV2InferShape)
    .InferDataType(ScaledMaskedSoftmaxGradV2InferDtype);
}
