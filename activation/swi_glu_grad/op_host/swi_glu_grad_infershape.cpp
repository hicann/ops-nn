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
 * \file swi_glu_grad_infer.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForSwiGluGrad(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1_shape);

    gert::Shape *output_shape_1 = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, output_shape_1);

    *output_shape_1 = *x1_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwiGluGrad(gert::InferDataTypeContext *context) {
    const ge::DataType dtype = context->GetInputDataType(0);
    ge::graphStatus ret = context->SetOutputDataType(0, dtype);
    return ret;
}

IMPL_OP_INFERSHAPE(SwiGluGrad).InferShape(InferShapeForSwiGluGrad).InferDataType(InferDataTypeForSwiGluGrad);
}  // namespace ops
