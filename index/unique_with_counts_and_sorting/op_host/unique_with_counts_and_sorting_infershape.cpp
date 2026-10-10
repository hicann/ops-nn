/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr size_t INPUT_INDEX = 0;
constexpr size_t VALUES_OUTPUT = 0;
constexpr size_t INDICES_OUTPUT = 1;
constexpr size_t COUNTS_OUTPUT = 2;
constexpr size_t RETURN_INVERSE_ATTR = 0;
constexpr size_t RETURN_COUNTS_ATTR = 1;
constexpr size_t VECTOR_RANK = 1;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t MIN_UNIQUE_COUNT = 1;

void SetVectorShape(gert::Shape& shape, int64_t length)
{
    shape.SetDimNum(VECTOR_RANK);
    shape.SetDim(0, length);
}

ge::graphStatus InferShape4UniqueWithCountsAndSorting(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const auto* input = context->GetInputShape(INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    if (input->GetDimNum() == 0) {
        OP_LOGE(context->GetNodeName(), "Input data must be at least 1D.");
        return ge::GRAPH_FAILED;
    }
    auto* values = context->GetOutputShape(VALUES_OUTPUT);
    auto* indices = context->GetOutputShape(INDICES_OUTPUT);
    auto* counts = context->GetOutputShape(COUNTS_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, values);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices);
    OP_CHECK_NULL_WITH_CONTEXT(context, counts);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* returnInverse = attrs->GetAttrPointer<bool>(RETURN_INVERSE_ATTR);
    const auto* returnCounts = attrs->GetAttrPointer<bool>(RETURN_COUNTS_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnInverse);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnCounts);
    const bool withInverse = *returnInverse;
    const bool withCounts = *returnCounts;
    SetVectorShape(*values, UNKNOWN_DIM);
    // Preserve canndev's runtime output convention, including the counts-only branch.
    if (withInverse || withCounts) {
        *indices = *input;
        if (withCounts) {
            SetVectorShape(*counts, UNKNOWN_DIM);
        } else {
            counts->SetDimNum(0);
        }
    } else {
        indices->SetDimNum(VECTOR_RANK);
        counts->SetDimNum(VECTOR_RANK);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype4UniqueWithCountsAndSorting(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    // Preserve canndev's legacy inference: the AICPU kernel writes int64_t
    // indices/counts even when the prototype carries an out_idx attribute.
    context->SetOutputDataType(VALUES_OUTPUT, context->GetInputDataType(INPUT_INDEX));
    context->SetOutputDataType(INDICES_OUTPUT, ge::DT_INT64);
    context->SetOutputDataType(COUNTS_OUTPUT, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeRange4UniqueWithCountsAndSorting(gert::InferShapeRangeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const auto* input = context->GetInputShapeRange(INPUT_INDEX);
    auto* values = context->GetOutputShapeRange(VALUES_OUTPUT);
    auto* indices = context->GetOutputShapeRange(INDICES_OUTPUT);
    auto* counts = context->GetOutputShapeRange(COUNTS_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    OP_CHECK_NULL_WITH_CONTEXT(context, values);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices);
    OP_CHECK_NULL_WITH_CONTEXT(context, counts);
    OP_CHECK_NULL_WITH_CONTEXT(context, input->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, input->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, values->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, values->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, indices->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, indices->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, counts->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, counts->GetMax());
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* returnInverse = attrs->GetAttrPointer<bool>(RETURN_INVERSE_ATTR);
    const auto* returnCounts = attrs->GetAttrPointer<bool>(RETURN_COUNTS_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnInverse);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnCounts);
    const bool withInverse = *returnInverse;
    const bool withCounts = *returnCounts;
    // Inference is shared by all SoCs: retain canndev's range convention, including boundary shapes.
    const int64_t maximum = input->GetMax()->GetShapeSize();
    const int64_t minimum = MIN_UNIQUE_COUNT;
    SetVectorShape(*values->GetMin(), minimum);
    SetVectorShape(*values->GetMax(), maximum);
    if (withInverse || withCounts) {
        *indices->GetMin() = *input->GetMin();
        *indices->GetMax() = *input->GetMax();
    } else {
        SetVectorShape(*indices->GetMin(), 0);
        SetVectorShape(*indices->GetMax(), 0);
    }
    SetVectorShape(*counts->GetMin(), withCounts ? minimum : 0);
    SetVectorShape(*counts->GetMax(), withCounts ? maximum : 0);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(UniqueWithCountsAndSorting)
    .InferShape(InferShape4UniqueWithCountsAndSorting)
    .InferDataType(InferDtype4UniqueWithCountsAndSorting)
    .InferShapeRange(InferShapeRange4UniqueWithCountsAndSorting);
} // namespace ops
