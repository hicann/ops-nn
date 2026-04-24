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
/*!
 * \file apply_adadelta_infershape.cpp
 * \brief ApplyAdadelta shape inference
 *
 * All inputs share the same shape. Outputs = input shape (inplace).
 * 3 outputs: varOut, accumOut, accumUpdateOut
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4ApplyAdadelta(gert::InferShapeContext* context)
{
    // Input(0) = var
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 3 outputs: varOut(0), accumOut(1), accumUpdateOut(2)
    for (size_t i = 0; i < 3; i++) {
        gert::Shape* outputShape = context->GetOutputShape(i);
        if (outputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *outputShape = *inputShape;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdadelta).InferShape(InferShape4ApplyAdadelta);

} // namespace ops
