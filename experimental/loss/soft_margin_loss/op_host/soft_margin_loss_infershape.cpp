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

/**
 * \file soft_margin_loss_infershape.cpp
 * \brief SoftMarginLoss shape inference
 *
 * - reduction=0 (none): output shape = input shape
 * - reduction=1 (mean) or 2 (sum): output shape = scalar (dimNum=0)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4SoftMarginLoss(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Get reduction attribute
    int64_t reduction = 1;  // default: mean
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* reductionPtr = attrs->GetAttrPointer<int64_t>(0);
        if (reductionPtr != nullptr) {
            reduction = *reductionPtr;
        }
    }

    if (reduction == 0) {
        // none: output shape = input shape
        *outputShape = *inputShape;
    } else {
        // mean or sum: output is scalar (0 dimensions)
        *outputShape = gert::Shape();
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftMarginLoss).InferShape(InferShape4SoftMarginLoss);

} // namespace ops
