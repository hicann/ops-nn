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
 * \file apply_fused_ema_adam_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
static constexpr size_t INPUT_GRAD_INDEX = 0;
static constexpr size_t INPUT_VAR_INDEX = 1;
static constexpr size_t INPUT_M_INDEX = 2;
static constexpr size_t INPUT_V_INDEX = 3;
static constexpr size_t INPUT_S_INDEX = 4;
static constexpr size_t INPUT_STEP_INDEX = 5;
static constexpr size_t OUTPUT_VAR_INDEX = 0;
static constexpr size_t OUTPUT_M_INDEX = 1;
static constexpr size_t OUTPUT_V_INDEX = 2;
static constexpr size_t OUTPUT_S_INDEX = 3;

namespace ops {
static ge::graphStatus InferShapeForApplyFusedEmaAdam(gert::InferShapeContext* context)
{
    // input shape
    OP_LOGD(context, "Begin to do InferShapeForApplyFusedEmaAdam.");
    const gert::Shape* gradShape = context->GetInputShape(INPUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* varShape = context->GetInputShape(INPUT_VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    const gert::Shape* mShape = context->GetInputShape(INPUT_M_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, mShape);
    const gert::Shape* vShape = context->GetInputShape(INPUT_V_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);
    const gert::Shape* sShape = context->GetInputShape(INPUT_S_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sShape);
    const gert::Shape* stepShape = context->GetInputShape(INPUT_STEP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, stepShape);

    // output shape
    gert::Shape* varRefShape = context->GetOutputShape(OUTPUT_VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varRefShape);
    gert::Shape* mRefShape = context->GetOutputShape(OUTPUT_M_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, mRefShape);
    gert::Shape* vRefShape = context->GetOutputShape(OUTPUT_V_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, vRefShape);
    gert::Shape* sRefShape = context->GetOutputShape(OUTPUT_S_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sRefShape);

    *varRefShape = *varShape;
    *mRefShape = *mShape;
    *vRefShape = *vShape;
    *sRefShape = *sShape;

    OP_LOGD(context, "End to do InferShapeForApplyFusedEmaAdam.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForApplyFusedEmaAdam(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_VAR_INDEX, context->GetInputDataType(INPUT_VAR_INDEX));
    context->SetOutputDataType(OUTPUT_M_INDEX, context->GetInputDataType(INPUT_M_INDEX));
    context->SetOutputDataType(OUTPUT_V_INDEX, context->GetInputDataType(INPUT_V_INDEX));
    context->SetOutputDataType(OUTPUT_S_INDEX, context->GetInputDataType(INPUT_S_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyFusedEmaAdam)
    .InferShape(InferShapeForApplyFusedEmaAdam)
    .InferDataType(InferDataTypeForApplyFusedEmaAdam);
} // namespace ops