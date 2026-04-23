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
 * \file apply_proximal_adagrad_infershape.cpp
 * \brief ApplyProximalAdagrad shape / dtype inference.
 *
 * - var_out.shape   = var.shape
 * - accum_out.shape = accum.shape
 * - both outputs are float32
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4ApplyProximalAdagrad(gert::InferShapeContext* context)
{
    // Input 0 = var, output 0 = var_out (inplace)
    const gert::Shape* varShape = context->GetInputShape(0);
    if (varShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // Input 1 = accum, output 1 = accum_out (inplace)
    const gert::Shape* accumShape = context->GetInputShape(1);
    if (accumShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* varOutShape = context->GetOutputShape(0);
    if (varOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* accumOutShape = context->GetOutputShape(1);
    if (accumOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *varOutShape = *varShape;
    *accumOutShape = *accumShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ApplyProximalAdagrad(gert::InferDataTypeContext* context)
{
    // Outputs dtype follow var/accum (both must be float32 per spec).
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyProximalAdagrad)
    .InferShape(InferShape4ApplyProximalAdagrad)
    .InferDataType(InferDataType4ApplyProximalAdagrad);

} // namespace ops
