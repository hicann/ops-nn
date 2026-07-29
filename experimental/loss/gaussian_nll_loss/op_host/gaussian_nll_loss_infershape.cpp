/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"
#include <cmath>
#include <cstring>

namespace ops {
namespace {
bool DimsCompatible(int64_t lhs, int64_t rhs) { return lhs < 0 || rhs < 0 || lhs == rhs; }

bool IsTargetShapeSupported(const gert::Shape& input, const gert::Shape& target)
{
    if (input.GetDimNum() != target.GetDimNum()) {
        return false;
    }
    size_t broadcastDimensionCount = 0;
    for (size_t i = 0; i < input.GetDimNum(); ++i) {
        if (DimsCompatible(input.GetDim(i), target.GetDim(i))) {
            continue;
        }
        if (target.GetDim(i) != 1 || ++broadcastDimensionCount > 1) {
            return false;
        }
    }
    return true;
}

bool IsVarShapeSupported(const gert::Shape& input, const gert::Shape& var)
{
    if (var.IsScalar()) {
        return true;
    }
    const size_t inputRank = input.GetDimNum();
    const size_t varRank = var.GetDimNum();
    if (varRank == inputRank) {
        bool same = true;
        for (size_t i = 0; i < inputRank; ++i) {
            same = same && DimsCompatible(input.GetDim(i), var.GetDim(i));
        }
        if (same) {
            return true;
        }
        if (inputRank == 0 || var.GetDim(inputRank - 1) != 1) {
            return false;
        }
        for (size_t i = 0; i + 1 < inputRank; ++i) {
            if (!DimsCompatible(input.GetDim(i), var.GetDim(i))) {
                return false;
            }
        }
        return true;
    }
    if (inputRank > 0 && varRank + 1 == inputRank) {
        for (size_t i = 0; i < varRank; ++i) {
            if (!DimsCompatible(input.GetDim(i), var.GetDim(i))) {
                return false;
            }
        }
        return true;
    }
    return false;
}
} // namespace

static ge::graphStatus InferShapeGaussianNllLoss(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GaussianNllLoss", "context is null"), return ge::GRAPH_FAILED);
    const auto* input = context->GetInputShape(0);
    const auto* target = context->GetInputShape(1);
    const auto* var = context->GetInputShape(2);
    auto* output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    OP_CHECK_NULL_WITH_CONTEXT(context, target);
    OP_CHECK_NULL_WITH_CONTEXT(context, var);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    OP_CHECK_IF(!IsTargetShapeSupported(*input, *target), OP_LOGE(context, "unsupported target shape"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsVarShapeSupported(*input, *var), OP_LOGE(context, "unsupported var shape"), return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* eps = attrs->GetAttrPointer<float>(1);
    const char* reduction = attrs->GetAttrPointer<char>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, eps);
    OP_CHECK_NULL_WITH_CONTEXT(context, reduction);
    OP_CHECK_IF(!std::isfinite(*eps) || *eps <= 0.0f, OP_LOGE(context, "eps must be finite and greater than zero"),
                return ge::GRAPH_FAILED);
    if (std::strcmp(reduction, "none") == 0) {
        *output = *input;
    } else if (std::strcmp(reduction, "sum") == 0 || std::strcmp(reduction, "mean") == 0) {
        *output = gert::Shape({1});
    } else {
        OP_LOGE(context, "reduction must be none, sum, or mean");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(GaussianNllLoss).InferShape(InferShapeGaussianNllLoss);
} // namespace ops
