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
 * \file batch_norm_ext2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/runtime2_util_nn.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr int64_t X_INPUT_IDX = 0;
static constexpr int64_t SCALE_INPUT_IDX = 1;
static constexpr int64_t Y_OUTPUT_IDX = 0;
static constexpr int64_t MEAN_OUTPUT_IDX = 1;
static constexpr int64_t VARIANCE_OUTPUT_IDX = 2;
static constexpr int64_t RESERVE_SPACE_1_OUTPUT_IDX = 3;
static constexpr int64_t RESERVE_SPACE_2_OUTPUT_IDX = 4;

static ge::graphStatus BatchNormExt2InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* scaleShape = context->GetInputShape(SCALE_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    gert::Shape* yShape = context->GetOutputShape(Y_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* meanShape = context->GetOutputShape(MEAN_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    gert::Shape* varianceShape = context->GetOutputShape(VARIANCE_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varianceShape);
    gert::Shape* reserveSpace1Shape = context->GetOutputShape(RESERVE_SPACE_1_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reserveSpace1Shape);
    gert::Shape* reserveSpace2Shape = context->GetOutputShape(RESERVE_SPACE_2_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reserveSpace2Shape);

    // 动态 rank(-2):输出 shape 随输入传递未知 rank,避免把 -2 标记值推成非法形状
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*meanShape);
        Ops::Base::SetUnknownRank(*varianceShape);
        Ops::Base::SetUnknownRank(*reserveSpace1Shape);
        Ops::Base::SetUnknownRank(*reserveSpace2Shape);
        return GRAPH_SUCCESS;
    }
    if (Ops::Base::IsUnknownRank(*scaleShape)) {
        Ops::Base::SetUnknownRank(*meanShape);
        Ops::Base::SetUnknownRank(*varianceShape);
        Ops::Base::SetUnknownRank(*reserveSpace1Shape);
        Ops::Base::SetUnknownRank(*reserveSpace2Shape);
    }

    *yShape = *xShape;
    *meanShape = *scaleShape;
    *varianceShape = *scaleShape;
    *reserveSpace1Shape = *scaleShape;
    *reserveSpace2Shape = *scaleShape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus BatchNormExt2InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType xDtype = context->GetInputDataType(X_INPUT_IDX);
    context->SetOutputDataType(Y_OUTPUT_IDX, xDtype);
    context->SetOutputDataType(MEAN_OUTPUT_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(VARIANCE_OUTPUT_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(RESERVE_SPACE_1_OUTPUT_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(RESERVE_SPACE_2_OUTPUT_IDX, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchNormExt2).InferShape(BatchNormExt2InferShape).InferDataType(BatchNormExt2InferDataType);
} // namespace ops
