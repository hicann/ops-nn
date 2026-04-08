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
 * \file sigmoid_cross_entropy_with_logits_infershape.cpp
 * \brief SigmoidCrossEntropyWithLogits 算子形状推导实现
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/infershape_elewise_util.h"

using namespace ge;
static constexpr size_t LOSS_INDEX = 0;
static constexpr size_t PREDICT_INDEX = 0;

namespace ops {
static ge::graphStatus InferShapeForSigmoidCrossEntropyWithLogits(gert::InferShapeContext* context)
{
    OP_LOGD(context, "InferShapeForSigmoidCrossEntropyWithLogits begin.");
    ge::graphStatus ret = ge::GRAPH_FAILED;
    ret = Ops::Base::InferShape4Elewise(context);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED,
                OP_LOGE(context, "InferShapeForSigmoidCrossEntropyWithLogits failed."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "InferShapeForSigmoidCrossEntropyWithLogits end");
    return ret;
}

static graphStatus InferDataTypeForSigmoidCrossEntropyWithLogits(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(LOSS_INDEX, context->GetInputDataType(PREDICT_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SigmoidCrossEntropyWithLogits)
    .InferShape(InferShapeForSigmoidCrossEntropyWithLogits)
    .InferDataType(InferDataTypeForSigmoidCrossEntropyWithLogits);
} // namespace ops
