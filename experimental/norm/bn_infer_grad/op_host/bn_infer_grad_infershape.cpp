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
 * \file bn_infer_grad_infershape.cpp
 * \brief BnInferGrad 算子形状推导实现
 *
 * 输出 x_backprop 的 shape 与输入 grads 完全相同。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4BnInferGrad(gert::InferShapeContext* context)
{
    // 获取输入 grads 的 shape
    const gert::Shape* gradsShape = context->GetInputShape(0);
    if (gradsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输出 x_backprop 的 shape 指针
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 输出与输入 shape 完全相同
    *outputShape = *gradsShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4BnInferGrad(gert::InferDataTypeContext* context)
{
    // x_backprop dtype = grads dtype
    const ge::DataType inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BnInferGrad)
    .InferShape(InferShape4BnInferGrad)
    .InferDataType(InferDataType4BnInferGrad);

} // namespace ops
