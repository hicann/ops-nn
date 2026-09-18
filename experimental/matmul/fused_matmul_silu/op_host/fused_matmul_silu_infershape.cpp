/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>

#include "register/op_impl_registry.h"

namespace {

constexpr int64_t kBlockK = 64;
constexpr int64_t kMaxSupportedK = 4096;

bool IsKnownDim(int64_t dim) { return dim >= 0; }

bool IsValidKnownDim(int64_t dim) { return dim == -1 || dim > 0; }

bool HasValidDimensions(int64_t m, int64_t n, int64_t k, int64_t weightK, int64_t biasN)
{
    return IsValidKnownDim(m) && IsValidKnownDim(n) && IsValidKnownDim(k) && IsValidKnownDim(weightK) &&
           IsValidKnownDim(biasN);
}

bool HasCompatibleKnownDimensions(int64_t n, int64_t k, int64_t weightK, int64_t biasN)
{
    return (!IsKnownDim(k) || !IsKnownDim(weightK) || weightK == k) &&
           (!IsKnownDim(n) || !IsKnownDim(biasN) || biasN == n);
}

bool HasSupportedKnownK(int64_t k) { return !IsKnownDim(k) || (k % kBlockK == 0 && k <= kMaxSupportedK); }

ge::graphStatus InferShapeForFusedMatmulSilu(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* biasShape = context->GetInputShape(2);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || biasShape->GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t m = xShape->GetDim(0);
    const int64_t k = xShape->GetDim(1);
    const int64_t n = weightShape->GetDim(0);
    const int64_t weightK = weightShape->GetDim(1);
    const int64_t biasN = biasShape->GetDim(0);
    if (!HasValidDimensions(m, n, k, weightK, biasN) || !HasCompatibleKnownDimensions(n, k, weightK, biasN) ||
        !HasSupportedKnownK(k)) {
        return ge::GRAPH_FAILED;
    }

    yShape->SetDimNum(2);
    yShape->SetDim(0, m);
    yShape->SetDim(1, n);
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_INFERSHAPE(FusedMatmulSilu).InferShape(InferShapeForFusedMatmulSilu);
