/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static ge::graphStatus InferShape4EmbeddingDenseGrad(gert::InferShapeContext* context)
{
    auto grad_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_shape);

    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    int64_t input_shape_len = grad_shape->GetDimNum();
    if (input_shape_len <= 0) {
        OP_LOGE(context->GetNodeName(), "grad shape dim num must greater than 0, got %ld", input_shape_len);
        return ge::GRAPH_FAILED;
    }
    out_shape->SetDimNum(2);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto num_weights_ptr = attrs->GetAttrPointer<int64_t>(0);
    if (num_weights_ptr == nullptr) {
        OP_LOGE(context->GetNodeName(), "numWeights attr is null");
        return ge::GRAPH_FAILED;
    }
    auto num_weights = *num_weights_ptr;
    out_shape->SetDim(0, num_weights);
    out_shape->SetDim(1, grad_shape->GetDim(input_shape_len - 1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4EmbeddingDenseGrad(gert::InferDataTypeContext* context)
{
    auto grad_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, grad_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingDenseGrad)
    .InferShape(InferShape4EmbeddingDenseGrad)
    .InferDataType(InferDataType4EmbeddingDenseGrad);
} // namespace ops
