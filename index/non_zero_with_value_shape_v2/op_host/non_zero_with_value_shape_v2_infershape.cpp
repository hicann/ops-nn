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
constexpr size_t kCountInputIndex = 2U;
constexpr size_t kV2OutValueIndex = 0U;
constexpr size_t kV2OutIndexIndex = 1U;
constexpr size_t kV2DynamicOutputRank = 1U;

int64_t GetV2RangeMaxDim(const gert::Shape* shape)
{
    if (shape == nullptr || shape->GetDimNum() == 0U) {
        return ge::UNKNOWN_DIM;
    }
    return shape->GetDim(0);
}
} // namespace

static ge::graphStatus InferShapeForNonZeroWithValueShapeV2(gert::InferShapeContext* context)
{
    auto v2ValueShape = context->GetOutputShape(kV2OutValueIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, v2ValueShape);
    auto v2IndexShape = context->GetOutputShape(kV2OutIndexIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, v2IndexShape);

    v2ValueShape->SetDimNum(kV2DynamicOutputRank);
    v2ValueShape->SetDim(0, ge::UNKNOWN_DIM);
    v2IndexShape->SetDimNum(kV2DynamicOutputRank);
    v2IndexShape->SetDim(0, ge::UNKNOWN_DIM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRangeForNonZeroWithValueShapeV2(gert::InferShapeRangeContext* context)
{
    auto v2ValueRange = context->GetInputShapeRange(kValueInputIndex);
    auto v2IndexRange = context->GetInputShapeRange(kIndexInputIndex);
    auto v2OutValueRange = context->GetOutputShapeRange(kV2OutValueIndex);
    auto v2OutIndexRange = context->GetOutputShapeRange(kV2OutIndexIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, v2ValueRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, v2IndexRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, v2OutValueRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, v2OutIndexRange);

    v2OutValueRange->GetMin()->SetDimNum(kV2DynamicOutputRank);
    v2OutValueRange->GetMin()->SetDim(0, 1);
    v2OutValueRange->GetMax()->SetDimNum(kV2DynamicOutputRank);
    v2OutValueRange->GetMax()->SetDim(0, GetV2RangeMaxDim(v2ValueRange->GetMax()));

    v2OutIndexRange->GetMin()->SetDimNum(kV2DynamicOutputRank);
    v2OutIndexRange->GetMin()->SetDim(0, 1);
    v2OutIndexRange->GetMax()->SetDimNum(kV2DynamicOutputRank);
    v2OutIndexRange->GetMax()->SetDim(0, GetV2RangeMaxDim(v2IndexRange->GetMax()));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForNonZeroWithValueShapeV2(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context->GetInputDataType(kIndexInputIndex) != ge::DT_INT32,
                OP_LOGE(context->GetNodeName(), "Dtype of index must be int32."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputDataType(kCountInputIndex) != ge::DT_INT32,
                OP_LOGE(context->GetNodeName(), "Dtype of count must be int32."), return ge::GRAPH_FAILED);
    context->SetOutputDataType(kV2OutValueIndex, context->GetInputDataType(kValueInputIndex));
    context->SetOutputDataType(kV2OutIndexIndex, context->GetInputDataType(kIndexInputIndex));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NonZeroWithValueShapeV2)
    .InferShape(InferShapeForNonZeroWithValueShapeV2)
    .InferShapeRange(InferShapeRangeForNonZeroWithValueShapeV2)
    .InferDataType(InferDataTypeForNonZeroWithValueShapeV2);
} // namespace ops
