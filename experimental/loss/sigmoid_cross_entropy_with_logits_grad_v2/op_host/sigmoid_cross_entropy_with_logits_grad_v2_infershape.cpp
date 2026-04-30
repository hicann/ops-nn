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

using namespace ge;

namespace ops {
static constexpr int64_t IDX_PREDICT = 0;
static constexpr int64_t IDX_GRADIENT = 0;

static ge::graphStatus InferShapeSigmoidCrossEntropyWithLogitsGradV2(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSigmoidCrossEntropyWithLogitsGradV2");

    const gert::Shape* predict_shape = context->GetInputShape(IDX_PREDICT);
    OP_CHECK_NULL_WITH_CONTEXT(context, predict_shape);

    gert::Shape* gradient_shape = context->GetOutputShape(IDX_GRADIENT);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradient_shape);

    *gradient_shape = *predict_shape;
    OP_LOGD(context->GetNodeName(), "End to do InferShapeSigmoidCrossEntropyWithLogitsGradV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SigmoidCrossEntropyWithLogitsGradV2).InferShape(InferShapeSigmoidCrossEntropyWithLogitsGradV2);
} // namespace ops
