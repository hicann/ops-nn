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
#include "util/math_util.h"

namespace {
constexpr size_t INPUT_GRAD_Y = 0;
constexpr size_t INPUT_X = 1;
constexpr size_t INPUT_WEIGHT = 2;
constexpr size_t INPUT_Y_ORIGIN = 3;
constexpr size_t INPUT_GROUP_INDEX = 4;
constexpr size_t OUTPUT_Y1 = 0;
constexpr size_t OUTPUT_SCALE1 = 1;
constexpr size_t OUTPUT_Y2 = 2;
constexpr size_t OUTPUT_SCALE2 = 3;
constexpr size_t OUTPUT_GRAD_WEIGHT = 4;
constexpr size_t ATTR_DST_TYPE = 4;
constexpr int64_t FP8_E5M2 = 35;
constexpr int64_t FP8_E4M3FN = 36;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t SCALE_PAIR = 2;
constexpr int64_t SCALE_PACK = 64;
constexpr int64_t MIN_RANK = 2;
} // namespace

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto x = context->GetInputShape(INPUT_X);
    auto y1 = context->GetOutputShape(OUTPUT_Y1);
    auto scale1 = context->GetOutputShape(OUTPUT_SCALE1);
    auto y2 = context->GetOutputShape(OUTPUT_Y2);
    auto scale2 = context->GetOutputShape(OUTPUT_SCALE2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale2);
    const int64_t rank = x->GetDimNum();
    // InferShape needs at least two axes to derive the MX output dimensions.
    // Data-shape constraints are checked once by the tiling function.
    if (rank < MIN_RANK) {
        return ge::GRAPH_FAILED;
    }

    auto weight = context->GetOptionalInputShape(INPUT_WEIGHT);
    auto gradWeight = context->GetOutputShape(OUTPUT_GRAD_WEIGHT);
    if (weight != nullptr) {
        OP_CHECK_NULL_WITH_CONTEXT(context, gradWeight);
        *gradWeight = *weight;
    }

    *y1 = *x;
    *y2 = *x;
    scale1->SetDimNum(rank + 1);
    for (int64_t i = 0; i < rank - 1; ++i) {
        scale1->SetDim(i, x->GetDim(i));
    }
    const int64_t width = x->GetDim(rank - 1);
    scale1->SetDim(rank - 1, width > 0 ? Ops::Base::CeilDiv(width, SCALE_PACK) : UNKNOWN_DIM);
    scale1->SetDim(rank, SCALE_PAIR);

    auto groupIndex = context->GetOptionalInputShape(INPUT_GROUP_INDEX);
    int64_t groupCount = UNKNOWN_DIM;
    if (groupIndex != nullptr && groupIndex->GetDimNum() == 1) {
        const int64_t dim = groupIndex->GetDim(0);
        groupCount = dim > 0 ? dim : UNKNOWN_DIM;
    }
    if (groupIndex != nullptr) {
        int64_t rows = 1;
        for (int64_t i = 0; i < rank - 1; ++i) {
            if (x->GetDim(i) <= 0) {
                rows = UNKNOWN_DIM;
                break;
            }
            rows *= x->GetDim(i);
        }
        // Group场景，scale2固定为三维
        scale2->SetDimNum(3);
        const int64_t scale2Rows = rows == UNKNOWN_DIM || groupCount == UNKNOWN_DIM ? UNKNOWN_DIM :
                                                                                      rows / SCALE_PACK + groupCount;
        scale2->SetDim(0, scale2Rows);
        scale2->SetDim(1, width);
        scale2->SetDim(2, SCALE_PAIR);
    } else {
        scale2->SetDimNum(rank + 1);
        for (int64_t i = 0; i < rank - 2; ++i) {
            scale2->SetDim(i, x->GetDim(i));
        }
        const int64_t dimM = x->GetDim(rank - 2);
        scale2->SetDim(rank - 2, dimM > 0 ? Ops::Base::CeilDiv(dimM, SCALE_PACK) : UNKNOWN_DIM);
        scale2->SetDim(rank - 1, width);
        scale2->SetDim(rank, SCALE_PAIR);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto xType = context->GetInputDataType(INPUT_X);
    if ((xType != ge::DT_FLOAT16 && xType != ge::DT_BF16) || context->GetInputDataType(INPUT_GRAD_Y) != xType) {
        return ge::GRAPH_FAILED;
    }
    auto yOrigin = context->GetOptionalInputDesc(INPUT_Y_ORIGIN);
    if (yOrigin != nullptr && yOrigin->GetDataType() != xType) {
        return ge::GRAPH_FAILED;
    }
    auto weight = context->GetOptionalInputDesc(INPUT_WEIGHT);
    if (weight != nullptr) {
        const auto weightType = weight->GetDataType();
        if (weightType != ge::DT_FLOAT16 && weightType != ge::DT_BF16 && weightType != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(OUTPUT_GRAD_WEIGHT, weightType);
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto dstType = attrs->GetAttrPointer<int64_t>(ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstType);
    if (*dstType != FP8_E5M2 && *dstType != FP8_E4M3FN) {
        return ge::GRAPH_FAILED;
    }
    const auto fp8Type = *dstType == FP8_E5M2 ? ge::DT_FLOAT8_E5M2 : ge::DT_FLOAT8_E4M3FN;
    context->SetOutputDataType(OUTPUT_Y1, fp8Type);
    context->SetOutputDataType(OUTPUT_SCALE1, ge::DT_FLOAT8_E8M0);
    context->SetOutputDataType(OUTPUT_Y2, fp8Type);
    context->SetOutputDataType(OUTPUT_SCALE2, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluBackwardGroupQuantWithDualAxis).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ops
