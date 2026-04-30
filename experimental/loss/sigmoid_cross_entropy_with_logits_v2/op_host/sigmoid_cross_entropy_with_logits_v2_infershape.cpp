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
 * \file sigmoid_cross_entropy_with_logits_v2_infershape.cpp
 * \brief infer shape
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_PREDICT = 0;
static constexpr int64_t IDX_LOSS = 0;

static ge::graphStatus InferShapeSigmoidCELoss(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSigmoidCE");

    const gert::Shape* predictShape = context->GetInputShape(IDX_PREDICT);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);

    gert::Shape* lossShape = context->GetOutputShape(IDX_LOSS);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);

    *lossShape = *predictShape;
    OP_LOGD(context->GetNodeName(), "End to do InferShapeSigmoidCE");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSigmoidCELoss(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(IDX_LOSS, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SigmoidCrossEntropyWithLogitsV2)
    .InferShape(InferShapeSigmoidCELoss)
    .InferDataType(InferDataTypeSigmoidCELoss);
} // namespace ops