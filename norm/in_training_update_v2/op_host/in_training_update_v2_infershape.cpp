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
 * \file in_training_update_v2_infershape.cpp
 * \brief Shape and dtype inference for INTrainingUpdateV2.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_SUM = 1;
constexpr size_t OUTPUT_Y = 0;
constexpr size_t OUTPUT_BATCH_MEAN = 1;
constexpr size_t OUTPUT_BATCH_VARIANCE = 2;
constexpr size_t PUBLIC_RANK = 4;

ge::graphStatus InferShapeForINTrainingUpdateV2(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("INTrainingUpdateV2", "infer shape context is null");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetRequiredInputShape(INPUT_X);
    const gert::Shape* sumShape = context->GetRequiredInputShape(INPUT_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumShape);
    OP_CHECK_IF(
        xShape->GetDimNum() != PUBLIC_RANK,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xShape->GetDimNum()).c_str(), "4"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sumShape->GetDimNum() != PUBLIC_RANK,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "sum", std::to_string(sumShape->GetDimNum()).c_str(), "4"),
        return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y);
    gert::Shape* batchMeanShape = context->GetOutputShape(OUTPUT_BATCH_MEAN);
    gert::Shape* batchVarianceShape = context->GetOutputShape(OUTPUT_BATCH_VARIANCE);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, batchMeanShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, batchVarianceShape);
    *yShape = *xShape;
    *batchMeanShape = *sumShape;
    *batchVarianceShape = *sumShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForINTrainingUpdateV2(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("INTrainingUpdateV2", "infer data type context is null");
        return ge::GRAPH_FAILED;
    }
    if (context->SetOutputDataType(OUTPUT_Y, context->GetRequiredInputDataType(INPUT_X)) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(OUTPUT_BATCH_MEAN, ge::DT_FLOAT) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(OUTPUT_BATCH_VARIANCE, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output dtype", "setter returned failure",
                                              "all inferred output dtypes must be writable");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(INTrainingUpdateV2)
    .InferShape(InferShapeForINTrainingUpdateV2)
    .InferDataType(InferDataTypeForINTrainingUpdateV2);
} // namespace ops
