/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mse_loss_grad_v2.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
using namespace ge;
namespace {
constexpr uint32_t INPUT_PREDICT_IDX = 0;
constexpr uint32_t INPUT_LABEL_IDX = 1;
constexpr uint32_t INPUT_DOUT_IDX = 2;
constexpr uint32_t OUTPUT_LOSS_GRAD_IDX = 0;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForMseLossGrad(gert::InferShapeContext* context)
{
    // input shape
    OP_LOGD(context->GetNodeName(), "MseLossGrad Begin.");
    const gert::Shape* predictShape = context->GetInputShape(INPUT_PREDICT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    const gert::Shape* labelShape = context->GetInputShape(INPUT_LABEL_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelShape);
    const gert::Shape* doutShape = context->GetInputShape(INPUT_DOUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, doutShape);

    // output shape
    gert::Shape* lossGradShape = context->GetOutputShape(OUTPUT_LOSS_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossGradShape);
    if (Ops::Base::IsUnknownRank(*predictShape)) {
        OP_LOGD(context->GetNodeName(), "Input shape is -2, set output shape to (-2,)");
        lossGradShape->SetDim(DIM_0, UNKNOWN_RANK_DIM_VALUE_);
        return ge::GRAPH_FAILED;
    }
    *lossGradShape = *predictShape;
    OP_LOGD(context->GetNodeName(), "InferShapeForMseLossGrad End.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMseLossGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType predictDtype = context->GetInputDataType(INPUT_PREDICT_IDX);
    context->SetOutputDataType(OUTPUT_LOSS_GRAD_IDX, predictDtype);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MseLossGradV2).InferShape(InferShapeForMseLossGrad).InferDataType(InferDataTypeForMseLossGrad);
} // namespace ops