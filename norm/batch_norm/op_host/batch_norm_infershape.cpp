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
 * \file batch_norm_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/runtime2_util.h"

using namespace ge;
namespace ops {
static constexpr int64_t X_INPUT_IDX = 0;
static constexpr int64_t SCALE_INPUT_IDX = 1;
static constexpr int64_t MEAN_INPUT_IDX = 3;
static constexpr int64_t VAR_INPUT_IDX = 4;
static constexpr int64_t Y_OUTPUT_IDX = 0;
static constexpr int64_t BATCH_MEAN_OUTPUT_IDX = 1;
static constexpr int64_t BATCH_VAR_OUTPUT_IDX = 2;
static constexpr int64_t RESERVE_SPACE_1_OUTPUT_IDX = 3;
static constexpr int64_t RESERVE_SPACE_2_OUTPUT_IDX = 4;
static constexpr int64_t RESERVE_SPACE_3_OUTPUT_IDX = 5;

static ge::graphStatus BatchNormInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* scaleShape = context->GetInputShape(SCALE_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    gert::Shape* yShape = context->GetOutputShape(Y_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* batchMeanShape = context->GetOutputShape(BATCH_MEAN_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, batchMeanShape);
    gert::Shape* batchVarianceShape = context->GetOutputShape(BATCH_VAR_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, batchVarianceShape);
    gert::Shape* reserveSpace1Shape = context->GetOutputShape(RESERVE_SPACE_1_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reserveSpace1Shape);
    gert::Shape* reserveSpace2Shape = context->GetOutputShape(RESERVE_SPACE_2_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reserveSpace2Shape);
    gert::Shape* reserveSpace3Shape = context->GetOutputShape(RESERVE_SPACE_3_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reserveSpace3Shape);

    *yShape = *xShape;
    *batchMeanShape = *scaleShape;
    *batchVarianceShape = *scaleShape;
    *reserveSpace1Shape = *scaleShape;
    *reserveSpace2Shape = *scaleShape;

    *reserveSpace3Shape = gert::Shape({1});

    return GRAPH_SUCCESS;
}

static ge::graphStatus BatchNormInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType xDtype = context->GetInputDataType(X_INPUT_IDX);
    context->SetOutputDataType(Y_OUTPUT_IDX, xDtype);
    context->SetOutputDataType(BATCH_MEAN_OUTPUT_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(BATCH_VAR_OUTPUT_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(RESERVE_SPACE_1_OUTPUT_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(RESERVE_SPACE_2_OUTPUT_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(RESERVE_SPACE_3_OUTPUT_IDX, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchNorm).InferShape(BatchNormInferShape).InferDataType(BatchNormInferDataType);
} // namespace ops