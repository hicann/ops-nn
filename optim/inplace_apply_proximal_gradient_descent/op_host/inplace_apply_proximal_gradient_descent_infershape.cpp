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

namespace ops {
namespace {
constexpr size_t kInputCount = 5;
constexpr size_t kMaxTensorRank = 16;

bool AreInputFormatsValid(const gert::InferShapeContext* context)
{
    for (size_t i = 0; i < kInputCount; ++i) {
        const gert::CompileTimeTensorDesc* desc = context->GetInputDesc(i);
        if (desc == nullptr || desc->GetOriginFormat() != ge::FORMAT_ND || desc->GetStorageFormat() != ge::FORMAT_ND) {
            return false;
        }
    }
    return true;
}

bool IsUnknownRank(const gert::Shape* shape)
{
    return shape->GetDimNum() == 1 && shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

bool IsRankValid(const gert::Shape* shape) { return IsUnknownRank(shape) || shape->GetDimNum() <= kMaxTensorRank; }

bool IsScalarShapeValid(const gert::Shape* shape)
{
    if (IsUnknownRank(shape) || shape->GetDimNum() == 0) {
        return true;
    }
    return shape->GetDimNum() == 1 && (shape->GetDim(0) == 1 || shape->GetDim(0) == ge::UNKNOWN_DIM);
}

bool AreShapesCompatible(const gert::Shape* lhs, const gert::Shape* rhs)
{
    if (IsUnknownRank(lhs) || IsUnknownRank(rhs)) {
        return true;
    }
    const size_t rank = lhs->GetDimNum();
    if (rank != rhs->GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        const int64_t lhsDim = lhs->GetDim(i);
        const int64_t rhsDim = rhs->GetDim(i);
        if (lhsDim != rhsDim && lhsDim != ge::UNKNOWN_DIM && rhsDim != ge::UNKNOWN_DIM) {
            return false;
        }
    }
    return true;
}
} // namespace

static ge::graphStatus InferShapeForInplaceApplyProximalGradientDescent(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* varShape = context->GetInputShape(0);
    const gert::Shape* alphaShape = context->GetInputShape(1);
    const gert::Shape* l1Shape = context->GetInputShape(2);
    const gert::Shape* l2Shape = context->GetInputShape(3);
    const gert::Shape* deltaShape = context->GetInputShape(4);
    gert::Shape* outShape = context->GetOutputShape(0);
    if (varShape == nullptr || alphaShape == nullptr || l1Shape == nullptr || l2Shape == nullptr ||
        deltaShape == nullptr || outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (!AreInputFormatsValid(context) || !IsRankValid(varShape) || !IsRankValid(deltaShape) ||
        !AreShapesCompatible(varShape, deltaShape) || !IsScalarShapeValid(alphaShape) || !IsScalarShapeValid(l1Shape) ||
        !IsScalarShapeValid(l2Shape)) {
        return ge::GRAPH_FAILED;
    }
    const size_t rank = varShape->GetDimNum();
    outShape->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        outShape->SetDim(i, varShape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InplaceApplyProximalGradientDescent).InferShape(InferShapeForInplaceApplyProximalGradientDescent);
} // namespace ops
