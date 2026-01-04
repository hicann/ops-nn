/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nll_loss_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
const int64_t INPUT_IDX_X = 0;
const int64_t INPUT_IDX_TARGET = 1;
const int64_t INPUT_SIZE_TWO = 2;
const int64_t INPUT_SIZE_FOUR = 4;
const int64_t NUMBER_TWO = 2;
const int64_t NUMBER_THREE = 3;
const int64_t IDX_ZERO = 0;
const int64_t IDX_ONE = 1;
static constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;

static inline bool IsUnknownRank(const gert::Shape* check_shape)
{
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

static inline ge::graphStatus SetUnknownRank(gert::Shape* output_shape)
{
    OP_CHECK_IF(
        output_shape == nullptr, OP_LOGD("SetUnknownRank", "the output_shape is nullptr, return unsuccess"),
        return ge::GRAPH_FAILED);
    output_shape->SetDimNum(0);
    output_shape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);

    OP_LOGD("SetUnknownRank", "set unknown rank = -2, output = %s", ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForNLLLoss(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "infershape is begin");
    auto x_shape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto target_shape = context->GetInputShape(INPUT_IDX_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context, target_shape);

    auto out_y_shape = context->GetOutputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_y_shape);
    auto out_tatal_weight_shape = context->GetOutputShape(INPUT_IDX_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_tatal_weight_shape);
    out_tatal_weight_shape->SetDimNum(0);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* attr_reduction = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_reduction);

    if (!IsUnknownRank(x_shape)) {
        OP_CHECK_IF(
            (x_shape->GetDimNum() != 1) && (x_shape->GetDimNum() != INPUT_SIZE_TWO) &&
                (x_shape->GetDimNum() != INPUT_SIZE_FOUR),
            OP_LOGE(context->GetNodeName(), "x_shape->GetDimNum() only support 1, 2 and 4"), return GRAPH_FAILED);
    }

    if (strcmp(attr_reduction, "none") == 0) {
        if (IsUnknownRank(x_shape)) {
            OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
            SetUnknownRank(out_y_shape);
            return ge::GRAPH_SUCCESS;
        } else if (x_shape->GetDimNum() == INPUT_SIZE_TWO) {
            // x is [N, C], y is [N]
            out_y_shape->SetDimNum(1);
            out_y_shape->SetDim(0, x_shape->GetDim(0));

            return GRAPH_SUCCESS;
        } else if (x_shape->GetDimNum() == INPUT_SIZE_FOUR) {
            // x is [N, C, H, W], y is [N, H, W]
            out_y_shape->SetDimNum(NUMBER_THREE);
            out_y_shape->SetDim(0, x_shape->GetDim(0));
            out_y_shape->SetDim(1, x_shape->GetDim(NUMBER_TWO));
            out_y_shape->SetDim(NUMBER_TWO, x_shape->GetDim(NUMBER_THREE));
            return GRAPH_SUCCESS;
        }
    }

    // reduction "mean" or "sum" or 1D, output y is scalar
    out_y_shape->SetDimNum(0);
    OP_LOGD(context->GetNodeName(), "infershape is success");
    return GRAPH_SUCCESS;
}

graphStatus InferDtypeForNLLLoss(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "inferdtype is begin");
    auto x_dtype = context->GetInputDataType(IDX_ZERO);
    OP_CHECK_IF(
        (x_dtype != ge::DT_FLOAT) && (x_dtype != ge::DT_FLOAT16) && (x_dtype != ge::DT_BF16),
        OP_LOGE(context->GetNodeName(), "x_dtype only support float, float16 and bfloat16"), return GRAPH_FAILED);

    auto target_dtype = context->GetInputDataType(IDX_ONE);
    OP_CHECK_IF(
        (target_dtype != ge::DT_INT32) && (target_dtype != ge::DT_INT64) && (target_dtype != ge::DT_UINT8),
        OP_LOGE(context->GetNodeName(), "target_dtype only support int32, int64 and uint8"), return GRAPH_FAILED);

    context->SetOutputDataType(IDX_ZERO, x_dtype);

    OP_LOGI(context->GetNodeName(), "inferdtype is success");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NLLLoss).InferShape(InferShapeForNLLLoss).InferDataType(InferDtypeForNLLLoss);
} // namespace ops
