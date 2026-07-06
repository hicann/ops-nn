/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hinge_loss_grad_infershape.cpp
 * \brief HingeLossGrad算子的shape推理和数据类型推理实现
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_PREDICT = 0;
static constexpr int64_t IDX_TARGET = 1;
static constexpr int64_t IDX_GRAD_OUTPUT = 2;
static constexpr int64_t IDX_GRAD_INPUT = 0;

static ge::graphStatus CheckSameShape(
    gert::InferShapeContext* context, const gert::Shape* lhs, const gert::Shape* rhs, const char* lhsName, const char* rhsName)
{
    OP_CHECK_IF(
        lhs->GetDimNum() != rhs->GetDimNum(),
        OP_LOGE(context, "%s and %s rank mismatch: %zu vs %zu", lhsName, rhsName, lhs->GetDimNum(), rhs->GetDimNum()),
        return ge::GRAPH_FAILED);

    for (size_t i = 0; i < lhs->GetDimNum(); ++i) {
        OP_CHECK_IF(
            lhs->GetDim(i) != rhs->GetDim(i),
            OP_LOGE(context, "%s and %s shape mismatch at dim %zu: %ld vs %ld", lhsName, rhsName, i,
                lhs->GetDim(i), rhs->GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeHingeLossGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeHingeLossGrad");

    const gert::Shape* predictShape = context->GetInputShape(IDX_PREDICT);
    const gert::Shape* targetShape = context->GetInputShape(IDX_TARGET);
    const gert::Shape* gradOutputShape = context->GetInputShape(IDX_GRAD_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradOutputShape);

    OP_CHECK_IF(
        CheckSameShape(context, predictShape, targetShape, "predict", "target") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "predict and target shape check failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckSameShape(context, predictShape, gradOutputShape, "predict", "grad_output") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "predict and grad_output shape check failed"),
        return ge::GRAPH_FAILED);

    gert::Shape* gradInputShape = context->GetOutputShape(IDX_GRAD_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradInputShape);
    *gradInputShape = *predictShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeHingeLossGrad");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeHingeLossGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeHingeLossGrad");

    ge::DataType dtype = context->GetInputDataType(IDX_PREDICT);
    context->SetOutputDataType(IDX_GRAD_INPUT, dtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeHingeLossGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HingeLossGrad).InferShape(InferShapeHingeLossGrad).InferDataType(InferDataTypeHingeLossGrad);
} // namespace ops
