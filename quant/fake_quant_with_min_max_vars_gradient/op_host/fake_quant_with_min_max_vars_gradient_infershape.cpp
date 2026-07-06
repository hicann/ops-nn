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
 * \file fake_quant_with_min_max_vars_gradient_infershape.cpp
 * \brief Shape/dtype inference for fake_quant_with_min_max_vars_gradient
 *
 * Validates:
 *  - gradients and x have identical shape
 *  - min and max have shape (1,)
 *  - backprops_wrt_x shape = x shape
 *  - backprop_wrt_min/max shape = (1,)
 *  - All outputs are float32
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_IDX_GRAD = 0;
static constexpr size_t INPUT_IDX_X = 1;
static constexpr size_t INPUT_IDX_MIN = 2;
static constexpr size_t INPUT_IDX_MAX = 3;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr size_t OUTPUT_IDX_MIN = 1;
static constexpr size_t OUTPUT_IDX_MAX = 2;

static graphStatus InferShapeForFakeQuantWithMinMaxVarsGradient(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForFakeQuantWithMinMaxVarsGradient");

    const gert::Shape* gradShape = context->GetInputShape(INPUT_IDX_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* minShape = context->GetInputShape(INPUT_IDX_MIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, minShape);
    const gert::Shape* maxShape = context->GetInputShape(INPUT_IDX_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxShape);

    // Validate gradients and x have identical shape
    OP_CHECK_IF(gradShape->GetDimNum() != xShape->GetDimNum(),
                OP_LOGE(context, "gradients and x must have identical shape."), return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
        OP_CHECK_IF(gradShape->GetDim(i) != xShape->GetDim(i),
                    OP_LOGE(context, "gradients and x must have identical shape."), return ge::GRAPH_FAILED);
    }

    // Validate min and max have shape (1,)
    OP_CHECK_IF(minShape->GetDimNum() != 1 || minShape->GetDim(0) != 1, OP_LOGE(context, "min must have shape (1,)."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxShape->GetDimNum() != 1 || maxShape->GetDim(0) != 1, OP_LOGE(context, "max must have shape (1,)."),
                return ge::GRAPH_FAILED);

    // Set output shapes
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xShape; // backprops_wrt_x shape = x shape

    gert::Shape* minOutShape = context->GetOutputShape(OUTPUT_IDX_MIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, minOutShape);
    minOutShape->SetDimNum(1);
    minOutShape->SetDim(0, 1);

    gert::Shape* maxOutShape = context->GetOutputShape(OUTPUT_IDX_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxOutShape);
    maxOutShape->SetDimNum(1);
    maxOutShape->SetDim(0, 1);

    OP_LOGD(context, "End to do InferShapeForFakeQuantWithMinMaxVarsGradient");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForFakeQuantWithMinMaxVarsGradient(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeForFakeQuantWithMinMaxVarsGradient");
    context->SetOutputDataType(OUTPUT_IDX_Y, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_IDX_MIN, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_IDX_MAX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantWithMinMaxVarsGradient)
    .InferShape(InferShapeForFakeQuantWithMinMaxVarsGradient)
    .InferDataType(InferDataTypeForFakeQuantWithMinMaxVarsGradient);
} // namespace ops
