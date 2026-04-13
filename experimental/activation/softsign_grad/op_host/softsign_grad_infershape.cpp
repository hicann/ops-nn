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
 * \file softsign_grad_infershape.cpp
 * \brief SoftsignGrad 算子形状推导实现
 *
 * Elementwise 算子：output shape = gradients shape = features shape
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

// Elementwise 算子的形状推导
// 逻辑：输出 shape 与第一个输入 shape (gradients) 相同
static ge::graphStatus InferShape4SoftsignGrad(gert::InferShapeContext* context)
{
    const gert::Shape* gradients_shape = context->GetInputShape(0);
    if (gradients_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 输出 shape = 输入 shape
    *output_shape = *gradients_shape;

    return ge::GRAPH_SUCCESS;
}

// DataType 推导：输出与输入类型一致
static ge::graphStatus InferDataType4SoftsignGrad(gert::InferDataTypeContext* context)
{
    auto input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftsignGrad)
    .InferShape(InferShape4SoftsignGrad)
    .InferDataType(InferDataType4SoftsignGrad);

} // namespace ops
