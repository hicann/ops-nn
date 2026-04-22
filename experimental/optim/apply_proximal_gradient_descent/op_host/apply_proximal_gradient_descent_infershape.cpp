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
 * \file apply_proximal_gradient_descent_infershape.cpp
 * \brief ApplyProximalGradientDescent 算子形状推导
 *
 * 逻辑：varOut.shape = var.shape, varOut.dtype = var.dtype
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4ApplyProximalGradientDescent(gert::InferShapeContext* context)
{
    const gert::Shape* varShape = context->GetInputShape(0);
    if (varShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *varShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyProximalGradientDescent).InferShape(InferShape4ApplyProximalGradientDescent);

} // namespace ops
