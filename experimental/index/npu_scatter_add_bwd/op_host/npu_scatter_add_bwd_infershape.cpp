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
 * \file npu_scatter_add_bwd_infershape.cpp
 * \brief NpuScatterAddBwd 算子 shape/dtype 推导：x_grad 与 x、s_grad 与 s 形状及 dtype 一致
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t NPUSCATTERADDBWD_INPUT_X_INDEX = 1;
static constexpr size_t NPUSCATTERADDBWD_INPUT_S_INDEX = 2;
static constexpr size_t NPUSCATTERADDBWD_OUTPUT_X_GRAD_INDEX = 0;
static constexpr size_t NPUSCATTERADDBWD_OUTPUT_S_GRAD_INDEX = 1;

static graphStatus InferDataType4NpuScatterAddBwd(gert::InferDataTypeContext* context)
{
    auto xDtype = context->GetInputDataType(NPUSCATTERADDBWD_INPUT_X_INDEX);
    auto sDtype = context->GetInputDataType(NPUSCATTERADDBWD_INPUT_S_INDEX);
    context->SetOutputDataType(NPUSCATTERADDBWD_OUTPUT_X_GRAD_INDEX, xDtype);
    context->SetOutputDataType(NPUSCATTERADDBWD_OUTPUT_S_GRAD_INDEX, sDtype);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4NpuScatterAddBwd(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(NPUSCATTERADDBWD_INPUT_X_INDEX);
    const gert::Shape* sShape = context->GetInputShape(NPUSCATTERADDBWD_INPUT_S_INDEX);
    gert::Shape* xGradShape = context->GetOutputShape(NPUSCATTERADDBWD_OUTPUT_X_GRAD_INDEX);
    gert::Shape* sGradShape = context->GetOutputShape(NPUSCATTERADDBWD_OUTPUT_S_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xGradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sGradShape);
    *xGradShape = *xShape;
    *sGradShape = *sShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NpuScatterAddBwd)
    .InferShape(InferShape4NpuScatterAddBwd)
    .InferDataType(InferDataType4NpuScatterAddBwd);
} // namespace ops
