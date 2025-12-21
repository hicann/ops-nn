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
 * \file embedding_dense_grad_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace {
constexpr size_t EMBEDDING_DENSE_GRAD_IN_GRAD = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_IN_INDICES = 1;
constexpr size_t EMBEDDING_DENSE_GRAD_IN_POSIDX = 2;
constexpr size_t EMBEDDING_DENSE_GRAD_OUT_Y = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX = 1;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ = 2;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForEmbeddingDenseGradV2(gert::InferShapeContext* context)
{
    auto grad_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_shape);
    auto indices_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices_shape);
    auto pos_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_POSIDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pos_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto num_weights = attrs->GetAttrPointer<int64_t>(EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, num_weights);
    auto padding_idx = attrs->GetAttrPointer<int64_t>(EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, padding_idx);
    auto scale_grad_by_freq = attrs->GetAttrPointer<bool>(EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale_grad_by_freq);

    int64_t output_shape_len = 2;
    int64_t input_shape_len = grad_shape->GetDimNum();
    out_shape->SetDimNum(output_shape_len);
    out_shape->SetDim(0, *num_weights);
    out_shape->SetDim(1, grad_shape->GetDim(input_shape_len - 1));

    return GRAPH_SUCCESS;
}

static graphStatus InferDtypeForEmbeddingDenseGradV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDtypeForEmbeddingDenseGrad rt2.0");

    auto grad_dtype = context->GetInputDataType(EMBEDDING_DENSE_GRAD_IN_GRAD);

    context->SetOutputDataType(EMBEDDING_DENSE_GRAD_OUT_Y, grad_dtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDtypeForEmbeddingDenseGrad");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingDenseGradV2)
    .InferShape(InferShapeForEmbeddingDenseGradV2)
    .InferDataType(InferDtypeForEmbeddingDenseGradV2);
} // namespace ops