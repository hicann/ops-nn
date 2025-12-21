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
 * \file apply_ftrl_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
static constexpr size_t VAR_INDEX = 0;
static constexpr size_t ACCUM_INDEX = 1;
static constexpr size_t LINEAR_INDEX = 2;
static constexpr size_t GRAD_INDEX = 3;
static constexpr size_t LR_INDEX = 4;
static constexpr size_t L1_INDEX = 5;
static constexpr size_t L2_INDEX = 6;
static constexpr size_t LR_POWER_INDEX = 7;

namespace ops {
static ge::graphStatus InferShapeForApplyFtrl(gert::InferShapeContext* context)
{
    // input shape
    OP_LOGD(context, "Begin to do InferShapeForApplyFtrl.");
    const gert::Shape* varShape = context->GetInputShape(VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    const gert::Shape* accumShape = context->GetInputShape(ACCUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, accumShape);
    const gert::Shape* linearShape = context->GetInputShape(LINEAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, linearShape);
    const gert::Shape* gradShape = context->GetInputShape(GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* lrShape = context->GetInputShape(LR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lrShape);
    const gert::Shape* l1Shape = context->GetInputShape(L1_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, l1Shape);
    const gert::Shape* l2Shape = context->GetInputShape(L2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, l2Shape);
    const gert::Shape* lrPowerShape = context->GetInputShape(LR_POWER_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lrPowerShape);

    // output shape
    gert::Shape* varRefShape = context->GetOutputShape(VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varRefShape);

    *varRefShape = *varShape;

    OP_LOGD(context, "End to do InferShapeForApplyFtrl.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForApplyFtrl(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(VAR_INDEX, context->GetInputDataType(VAR_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyFtrl)
    .InferShape(InferShapeForApplyFtrl)
    .InferDataType(InferDataTypeForApplyFtrl);
} // namespace ops