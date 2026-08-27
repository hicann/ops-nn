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
#include "op_common/op_host/util/shape_util.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kUIndex = 0;
constexpr size_t kMInIndex = 1;
constexpr size_t kMOutIndex = 0;
constexpr size_t kSumURIndex = 1;
constexpr size_t kSumUCIndex = 2;
constexpr size_t kSumURCIndex = 3;
constexpr size_t kGlobalShapeIndex = 6;
constexpr size_t kRank = 2;
} // namespace

static bool IsScalarShape(const gert::Shape& shape)
{
    return shape.GetDimNum() == 0 ||
           (shape.GetDimNum() == 1 && (shape.GetDim(0) == 1 || shape.GetDim(0) == UNKNOWN_DIM));
}

static ge::graphStatus CheckScalarInputs(gert::InferShapeContext* context)
{
    for (size_t i = 2; i < 6; ++i) {
        const gert::Shape* scalarShape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, scalarShape);
        if (!IsScalarShape(*scalarShape)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "scalar inputs", "non-scalar shape",
                                                  "scalar inputs must be scalar or one-element tensors");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGlobalShape(gert::InferShapeContext* context)
{
    const gert::Shape* globalShape = context->GetOptionalInputShape(kGlobalShapeIndex);
    if (globalShape != nullptr && (globalShape->GetDimNum() != 1 || globalShape->GetDim(0) != 2)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "global_shape", "rank or length",
                                              "global_shape must be a 1D tensor with 2 elements");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ApplyCamePart3(gert::InferShapeContext* context)
{
    const gert::Shape* uShape = context->GetInputShape(kUIndex);
    const gert::Shape* mInShape = context->GetInputShape(kMInIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, uShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, mInShape);
    if (Ops::Base::IsUnknownRank(*uShape) || Ops::Base::IsUnknownRank(*mInShape)) {
        for (size_t i = 0; i < 4; ++i) {
            Ops::Base::SetUnknownRank(*context->GetOutputShape(i));
        }
        return ge::GRAPH_SUCCESS;
    }
    if (uShape->GetDimNum() != kRank || mInShape->GetDimNum() != kRank || uShape->GetDim(0) != mInShape->GetDim(0) ||
        uShape->GetDim(1) != mInShape->GetDim(1)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "u/m", "rank or dimensions",
                                              "u and m must be rank-2 tensors with equal shapes");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < kRank; ++i) {
        if (uShape->GetDim(i) == 0 || mInShape->GetDim(i) == 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "u/m", "zero dimension",
                                                  "u and m dimensions must be greater than zero");
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckScalarInputs(context) != ge::GRAPH_SUCCESS || CheckGlobalShape(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    *context->GetOutputShape(kMOutIndex) = *mInShape;
    auto* sumUR = context->GetOutputShape(kSumURIndex);
    auto* sumUC = context->GetOutputShape(kSumUCIndex);
    auto* sumURC = context->GetOutputShape(kSumURCIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumUR);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumUC);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumURC);
    sumUR->SetDimNum(1);
    sumUR->SetDim(0, uShape->GetDim(0));
    sumUC->SetDimNum(1);
    sumUC->SetDim(0, uShape->GetDim(1));
    sumURC->SetDimNum(1);
    sumURC->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ApplyCamePart3(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kMOutIndex, context->GetInputDataType(kMInIndex));
    context->SetOutputDataType(kSumURIndex, ge::DT_FLOAT);
    context->SetOutputDataType(kSumUCIndex, ge::DT_FLOAT);
    context->SetOutputDataType(kSumURCIndex, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyCamePart3).InferShape(InferShape4ApplyCamePart3).InferDataType(InferDataType4ApplyCamePart3);
} // namespace ops
