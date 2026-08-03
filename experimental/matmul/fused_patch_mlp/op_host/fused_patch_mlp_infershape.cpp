/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {

constexpr size_t IDX_X = 0;
constexpr size_t IDX_BIASES = 2;
constexpr size_t IDX_Y = 0;
constexpr size_t ATTR_NUM_LAYERS = 0;

static ge::graphStatus InferShape4FusedPatchMlp(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    const gert::Shape* biasesShape = context->GetInputShape(IDX_BIASES);
    gert::Shape* yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numLayersPtr = attrs->GetInt(ATTR_NUM_LAYERS);
    OP_CHECK_NULL_WITH_CONTEXT(context, numLayersPtr);
    const int64_t numLayers = *numLayersPtr;
    OP_CHECK_IF(numLayers <= 0, OP_LOGE(context->GetNodeName(), "num_layers must be positive."),
                return ge::GRAPH_FAILED);

    const size_t dimNum = xShape->GetDimNum();
    OP_CHECK_IF(dimNum < 2, OP_LOGE(context->GetNodeName(), "x must have at least two dimensions."),
                return ge::GRAPH_FAILED);
    const int64_t biasesLen = biasesShape->GetShapeSize();
    OP_CHECK_IF(biasesLen <= 0 || biasesLen % numLayers != 0,
                OP_LOGE(context->GetNodeName(), "bias length must be positive and divisible by num_layers."),
                return ge::GRAPH_FAILED);

    yShape->SetDimNum(dimNum);
    for (size_t i = 0; i + 1 < dimNum; ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }
    yShape->SetDim(dimNum - 1, biasesLen / numLayers);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FusedPatchMlp(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(IDX_Y, context->GetInputDataType(IDX_X));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedPatchMlp).InferShape(InferShape4FusedPatchMlp).InferDataType(InferDataType4FusedPatchMlp);

} // namespace ops
