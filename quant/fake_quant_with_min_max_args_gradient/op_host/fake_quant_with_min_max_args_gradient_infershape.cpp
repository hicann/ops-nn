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
 * \file fake_quant_with_min_max_args_gradient_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_IDX_GRAD = 0;
static constexpr size_t INPUT_IDX_X = 1;
static constexpr size_t OUTPUT_IDX_Y = 0;

static graphStatus InferShapeForFakeQuantWithMinMaxArgsGradient(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForFakeQuantWithMinMaxArgsGradient");
    const gert::Shape* gradShape = context->GetInputShape(INPUT_IDX_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    OP_CHECK_IF(gradShape->GetDimNum() != xShape->GetDimNum(),
                OP_LOGE(context, "gradients and x must have identical shape."),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
        OP_CHECK_IF(gradShape->GetDim(i) != xShape->GetDim(i),
                    OP_LOGE(context, "gradients and x must have identical shape."),
                    return ge::GRAPH_FAILED);
    }

    // y has same shape as x.
    *yShape = *xShape;

    OP_LOGD(context, "End to do InferShapeForFakeQuantWithMinMaxArgsGradient");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForFakeQuantWithMinMaxArgsGradient(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeForFakeQuantWithMinMaxArgsGradient");
    context->SetOutputDataType(OUTPUT_IDX_Y, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantWithMinMaxArgsGradient)
    .InferShape(InferShapeForFakeQuantWithMinMaxArgsGradient)
    .InferDataType(InferDataTypeForFakeQuantWithMinMaxArgsGradient);
} // namespace ops
