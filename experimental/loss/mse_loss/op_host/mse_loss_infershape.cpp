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
 * \file mse_loss_infershape.cpp
 * \brief MseLoss 算子形状推导实现
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"
#include <cstring>

using namespace ge;

namespace ops {

static const gert::Shape g_scalar_shape = {1};

static bool IsSameShape(const gert::Shape* lhs, const gert::Shape* rhs)
{
    if (lhs->GetDimNum() != rhs->GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs->GetDimNum(); ++i) {
        if (lhs->GetDim(i) != rhs->GetDim(i)) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus InferShapeMseLoss(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MseLoss", "context is nullptr"), return ge::GRAPH_FAILED);
    const gert::Shape* predictShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    const gert::Shape* labelShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelShape);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_IF(!IsSameShape(predictShape, labelShape),
                OP_LOGE(context, "MseLoss requires predict and label to have the same shape"), return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* reduction = attrs->GetAttrPointer<char>(0);
    OP_CHECK_IF(reduction == nullptr, OP_LOGE(context, "failed to get reduction attribute"), return ge::GRAPH_FAILED);

    if (std::strcmp(reduction, "none") == 0) {
        *outputShape = *predictShape;
    } else if (std::strcmp(reduction, "sum") == 0 || std::strcmp(reduction, "mean") == 0) {
        *outputShape = g_scalar_shape;
    } else {
        OP_LOGE(context, "MseLoss invalid reduction %s", reduction);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MseLoss).InferShape(InferShapeMseLoss);

} // namespace ops
