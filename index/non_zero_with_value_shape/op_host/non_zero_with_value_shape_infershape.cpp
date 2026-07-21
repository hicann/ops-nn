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
namespace {
constexpr size_t kValueInputIndex = 0U;
constexpr size_t kIndexInputIndex = 1U;
constexpr size_t kOutValueIndex = 0U;
constexpr size_t kOutIndexIndex = 1U;
constexpr size_t kDynamicOutputRank = 1U;

int64_t GetRangeMaxDim(const gert::Shape* shape)
{
    if (shape == nullptr || shape->GetDimNum() == 0U) {
        return ge::UNKNOWN_DIM;
    }
    return shape->GetDim(0);
}
} // namespace

static ge::graphStatus InferShapeForNonZeroWithValueShape(gert::InferShapeContext* context)
{
    auto outValueShape = context->GetOutputShape(kOutValueIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, outValueShape);
    auto outIndexShape = context->GetOutputShape(kOutIndexIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, outIndexShape);

    outValueShape->SetDimNum(kDynamicOutputRank);
    outValueShape->SetDim(0, ge::UNKNOWN_DIM);
    outIndexShape->SetDimNum(kDynamicOutputRank);
    outIndexShape->SetDim(0, ge::UNKNOWN_DIM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRangeForNonZeroWithValueShape(gert::InferShapeRangeContext* context)
{
    auto valueRange = context->GetInputShapeRange(kValueInputIndex);
    auto indexRange = context->GetInputShapeRange(kIndexInputIndex);
    auto outValueRange = context->GetOutputShapeRange(kOutValueIndex);
    auto outIndexRange = context->GetOutputShapeRange(kOutIndexIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, indexRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, outValueRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, outIndexRange);

    outValueRange->GetMin()->SetDimNum(kDynamicOutputRank);
    outValueRange->GetMin()->SetDim(0, 1);
    outValueRange->GetMax()->SetDimNum(kDynamicOutputRank);
    outValueRange->GetMax()->SetDim(0, GetRangeMaxDim(valueRange->GetMax()));

    outIndexRange->GetMin()->SetDimNum(kDynamicOutputRank);
    outIndexRange->GetMin()->SetDim(0, 1);
    outIndexRange->GetMax()->SetDimNum(kDynamicOutputRank);
    outIndexRange->GetMax()->SetDim(0, GetRangeMaxDim(indexRange->GetMax()));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForNonZeroWithValueShape(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kOutValueIndex, context->GetInputDataType(kValueInputIndex));
    context->SetOutputDataType(kOutIndexIndex, context->GetInputDataType(kIndexInputIndex));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NonZeroWithValueShape)
    .InferShape(InferShapeForNonZeroWithValueShape)
    .InferShapeRange(InferShapeRangeForNonZeroWithValueShape)
    .InferDataType(InferDataTypeForNonZeroWithValueShape);
} // namespace ops
