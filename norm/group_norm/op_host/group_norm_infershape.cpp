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
 * \file group_norm_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_X = 0;
static constexpr size_t OUTPUT_Y = 0;
static constexpr size_t OUTPUT_MEAN = 1;
static constexpr size_t OUTPUT_VARIANCE = 2;
static constexpr size_t ATTR_NUM_GROUPS = 0;
static constexpr int64_t UNKNOWN_RANK = -2LL;
static constexpr int64_t UNKNOWN_DIM = -1LL;

static inline bool IsUnknownRank(const gert::Shape* shape)
{
    return shape->GetDimNum() == 1 && shape->GetDim(0) == UNKNOWN_RANK;
}

static ge::graphStatus GroupNormInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_IF(xShape->GetDimNum() < 2 && !IsUnknownRank(xShape),
                OP_LOGE(context->GetNodeName(), "The rank of x must be at least 2, got %zu", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y);
    gert::Shape* meanShape = context->GetOutputShape(OUTPUT_MEAN);
    gert::Shape* varianceShape = context->GetOutputShape(OUTPUT_VARIANCE);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, varianceShape);
    // 输出y的shape与输入x保持一致。
    *yShape = *xShape;
    meanShape->SetDimNum(0);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(ATTR_NUM_GROUPS);
    OP_CHECK_NULL_WITH_CONTEXT(context, numGroups);

    // 统计输出的shape为(N, num_groups)。
    if (IsUnknownRank(xShape)) {
        meanShape->AppendDim(UNKNOWN_RANK);
    } else {
        meanShape->AppendDim(xShape->GetDim(0) == UNKNOWN_DIM ? UNKNOWN_DIM : xShape->GetDim(0));
        meanShape->AppendDim(*numGroups);
    }
    *varianceShape = *meanShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GroupNormInferDataType(gert::InferDataTypeContext* context)
{
    // 三个输出的数据类型均继承输入x。
    auto inputDtype = context->GetInputDataType(INPUT_X);
    context->SetOutputDataType(OUTPUT_Y, inputDtype);
    context->SetOutputDataType(OUTPUT_MEAN, inputDtype);
    context->SetOutputDataType(OUTPUT_VARIANCE, inputDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNorm).InferShape(GroupNormInferShape).InferDataType(GroupNormInferDataType);
} // namespace ops
