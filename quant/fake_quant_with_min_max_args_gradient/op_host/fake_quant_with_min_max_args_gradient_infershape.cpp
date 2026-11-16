/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fake_quant_with_min_max_args_gradient_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

#include <cmath>

using namespace ge;

namespace ops {
static constexpr size_t INPUT_IDX_GRAD = 0;
static constexpr size_t INPUT_IDX_X = 1;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr size_t ATTR_MIN_INDEX = 0;
static constexpr size_t ATTR_MAX_INDEX = 1;
static constexpr size_t ATTR_NUM_BITS_INDEX = 2;
// spec.yaml rank_range [0,8]：Host 门禁独立拒绝超秩输入（kMaxDimNum 可表达 rank 9+）
static constexpr size_t MAX_RANK = 8;

// 属性语义校验（与 tiling 门禁同规则，满足两门禁独立拦截契约）：min/max 有限、min<max、num_bits∈[2,16]
static graphStatus ValidateAttrsForFakeQuantWithMinMaxArgsGradient(gert::InferShapeContext* context)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* attrMin = attrs->GetAttrPointer<float>(ATTR_MIN_INDEX);
    const float* attrMax = attrs->GetAttrPointer<float>(ATTR_MAX_INDEX);
    const int64_t* attrNumBits = attrs->GetAttrPointer<int64_t>(ATTR_NUM_BITS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrMin);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrMax);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrNumBits);

    OP_CHECK_IF(
        (std::isnan(*attrMin) || std::isnan(*attrMax) || std::isinf(*attrMin) || std::isinf(*attrMax)),
        OP_LOGE(context->GetNodeName(), "FakeQuantWithMinMaxArgsGradient requires finite min/max, got min=%f max=%f.",
                *attrMin, *attrMax),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((*attrMin >= *attrMax),
                OP_LOGE(context->GetNodeName(),
                        "FakeQuantWithMinMaxArgsGradient requires min < max, got min=%f max=%f.", *attrMin, *attrMax),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((*attrNumBits < 2 || *attrNumBits > 16),
                OP_LOGE(context->GetNodeName(), "FakeQuantWithMinMaxArgsGradient num_bits must be in [2,16], got %ld.",
                        *attrNumBits),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferShapeForFakeQuantWithMinMaxArgsGradient(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForFakeQuantWithMinMaxArgsGradient");
    const gert::Shape* gradShape = context->GetInputShape(INPUT_IDX_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 未知rank（-2）时无法确定不一致，跳过校验，避免动态场景误拦
    const bool gradUnknownRank = Ops::Base::IsUnknownRank(*gradShape);
    const bool xUnknownRank = Ops::Base::IsUnknownRank(*xShape);
    OP_CHECK_IF(!gradUnknownRank && !xUnknownRank && gradShape->GetDimNum() != xShape->GetDimNum(),
                OP_LOGE(context, "gradients and x must have identical shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xShape->GetDimNum() > MAX_RANK,
        OP_LOGE(context, "FakeQuantWithMinMaxArgsGradient supports rank in [0,8], got %zu.", xShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    // x为未知rank时跳过了rank相等校验，gradients的超秩须在此独立拦截（Host门禁）
    OP_CHECK_IF(
        !gradUnknownRank && gradShape->GetDimNum() > MAX_RANK,
        OP_LOGE(context, "FakeQuantWithMinMaxArgsGradient supports rank in [0,8], got %zu.", gradShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    if (!gradUnknownRank && !xUnknownRank) {
        for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
            // 未知维（-1）视为通配，仅双侧已知且不等时才判定不一致
            OP_CHECK_IF(gradShape->GetDim(i) != xShape->GetDim(i) && gradShape->GetDim(i) != ge::UNKNOWN_DIM &&
                            xShape->GetDim(i) != ge::UNKNOWN_DIM,
                        OP_LOGE(context, "gradients and x must have identical shape."), return ge::GRAPH_FAILED);
        }
    }
    OP_CHECK_IF((ValidateAttrsForFakeQuantWithMinMaxArgsGradient(context) != ge::GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "attribute validation failed."), return ge::GRAPH_FAILED);

    // y has same shape as x.
    *yShape = *xShape;

    OP_LOGI(context->GetNodeName(), "FakeQuantWithMinMaxArgsGradient output shape: %s.",
            Ops::Base::ToString(*yShape).c_str());
    OP_LOGD(context, "End to do InferShapeForFakeQuantWithMinMaxArgsGradient");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForFakeQuantWithMinMaxArgsGradient(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeForFakeQuantWithMinMaxArgsGradient");
    // 两门禁独立拦截契约：Host 门禁独立拒绝非 float32 输入，而非仅由 tiling 兜底
    OP_CHECK_IF((context->GetInputDataType(INPUT_IDX_GRAD) != ge::DT_FLOAT ||
                 context->GetInputDataType(INPUT_IDX_X) != ge::DT_FLOAT),
                OP_LOGE(context->GetNodeName(),
                        "FakeQuantWithMinMaxArgsGradient only supports float32 inputs, got gradients=%d x=%d.",
                        static_cast<int>(context->GetInputDataType(INPUT_IDX_GRAD)),
                        static_cast<int>(context->GetInputDataType(INPUT_IDX_X))),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_IDX_Y, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantWithMinMaxArgsGradient)
    .InferShape(InferShapeForFakeQuantWithMinMaxArgsGradient)
    .InferDataType(InferDataTypeForFakeQuantWithMinMaxArgsGradient);
} // namespace ops
