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

static ge::graphStatus CheckScalarInputs(const gert::InferShapeContext* context)
{
    // Validate known scalar shapes before dynamic shape handling.
    // Unknown dimensions are checked by runtime tiling; known shapes must contain one element.
    for (size_t index = 2; index <= 5; ++index) {
        const gert::Shape* shape = context->GetInputShape(index);
        if (shape == nullptr || Ops::Base::IsUnknownRank(*shape) || shape->GetDimNum() == 0) {
            continue;
        }
        bool hasKnownInvalidDim = false;
        for (size_t i = 0; i < shape->GetDimNum(); ++i) {
            const int64_t dim = shape->GetDim(i);
            if (dim >= 0 && dim != 1) {
                hasKnownInvalidDim = true;
                break;
            }
        }
        if (hasKnownInvalidDim) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "scalar input",
                                                  Ops::Base::ToString(*shape).c_str(),
                                                  "eps/beta1/clip_threshold/sum_square_u must contain exactly one "
                                                  "element");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGlobalShape(const gert::InferShapeContext* context)
{
    const gert::Shape* globalShape = context->GetOptionalInputShape(kGlobalShapeIndex);
    if (globalShape == nullptr || Ops::Base::IsUnknownRank(*globalShape)) {
        return ge::GRAPH_SUCCESS;
    }
    if (globalShape->GetDimNum() == 1 && globalShape->GetDim(0) < 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (globalShape->GetDimNum() != 1 || globalShape->GetDim(0) != 2) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "global_shape", "rank or length",
                                              "global_shape must be a 1D tensor with 2 elements");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ResolveMatrixShape(const gert::InferShapeContext* context, const gert::Shape& uShape,
                                          const gert::Shape& mInShape, gert::Shape& resolvedShape)
{
    if (uShape.GetDimNum() != kRank || mInShape.GetDimNum() != kRank) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "u/m", "rank", "u and m must be rank-2 tensors");
        return ge::GRAPH_FAILED;
    }

    resolvedShape.SetDimNum(kRank);
    for (size_t i = 0; i < kRank; ++i) {
        const int64_t uDim = uShape.GetDim(i);
        const int64_t mDim = mInShape.GetDim(i);
        int64_t resolvedDim = -1;
        if (uDim == -1) {
            resolvedDim = mDim;
        } else if (mDim == -1) {
            resolvedDim = uDim;
        } else if (uDim == mDim) {
            resolvedDim = uDim;
        } else {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "u/m", "dimensions",
                                                  "known dimensions of u and m must be equal");
            return ge::GRAPH_FAILED;
        }

        if (resolvedDim == 0 || resolvedDim < -1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "u/m", "dimension value",
                                                  "u and m dimensions must be positive or unknown (-1)");
            return ge::GRAPH_FAILED;
        }
        resolvedShape.SetDim(i, resolvedDim);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ApplyCamePart3(gert::InferShapeContext* context)
{
    const gert::Shape* uShape = context->GetInputShape(kUIndex);
    const gert::Shape* mInShape = context->GetInputShape(kMInIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, uShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, mInShape);
    if (CheckGlobalShape(context) != ge::GRAPH_SUCCESS || CheckScalarInputs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (Ops::Base::IsUnknownRank(*uShape) || Ops::Base::IsUnknownRank(*mInShape)) {
        for (size_t i = 0; i < 4; ++i) {
            auto* outputShape = context->GetOutputShape(i);
            OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
            Ops::Base::SetUnknownRank(*outputShape);
        }
        return ge::GRAPH_SUCCESS;
    }
    gert::Shape resolvedShape;
    if (ResolveMatrixShape(context, *uShape, *mInShape, resolvedShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto* mOut = context->GetOutputShape(kMOutIndex);
    auto* sumUR = context->GetOutputShape(kSumURIndex);
    auto* sumUC = context->GetOutputShape(kSumUCIndex);
    auto* sumURC = context->GetOutputShape(kSumURCIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, mOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumUR);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumUC);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumURC);
    *mOut = resolvedShape;
    sumUR->SetDimNum(1);
    sumUR->SetDim(0, resolvedShape.GetDim(0));
    sumUC->SetDimNum(1);
    sumUC->SetDim(0, resolvedShape.GetDim(1));
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
