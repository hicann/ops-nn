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
 * \file in_training_update_grad_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
constexpr size_t INPUT_VARIANCE_IDX = 2;
constexpr size_t OUTPUT_RES_GAMMA_IDX = 0;
constexpr size_t OUTPUT_RES_BETA_IDX = 1;

// Both outputs copy the shape of input #2 (variance): res_gamma.shape = res_beta.shape = variance.shape.
static ge::graphStatus InferShape4INTrainingUpdateGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4INTrainingUpdateGrad");

    const gert::Shape* variance_shape = context->GetInputShape(INPUT_VARIANCE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, variance_shape);
    gert::Shape* res_gamma_shape = context->GetOutputShape(OUTPUT_RES_GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, res_gamma_shape);
    gert::Shape* res_beta_shape = context->GetOutputShape(OUTPUT_RES_BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, res_beta_shape);

    *res_gamma_shape = *variance_shape;
    *res_beta_shape = *variance_shape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4INTrainingUpdateGrad");
    return GRAPH_SUCCESS;
}

// Both outputs are always float32.
static graphStatus InferDataType4INTrainingUpdateGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4INTrainingUpdateGrad");
    context->SetOutputDataType(OUTPUT_RES_GAMMA_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_RES_BETA_IDX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4INTrainingUpdateGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(INTrainingUpdateGrad)
    .InferShape(InferShape4INTrainingUpdateGrad)
    .InferDataType(InferDataType4INTrainingUpdateGrad);
} // namespace ops
