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
 * \file lamb_apply_optimizer_assign_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
constexpr size_t GRAD_IDX = 0;
constexpr size_t INPUTV_IDX = 1;
constexpr size_t INPUTM_IDX = 2;
constexpr size_t INPUT3_IDX = 3;
constexpr size_t OUTPUT0_IDX = 0;
constexpr size_t OUTPUTV_IDX = 1;
constexpr size_t OUTPUTM_IDX = 2;

static ge::graphStatus InferShape4LambApplyOptimizerAssign(gert::InferShapeContext* context)
{
    auto grad_shape = context->GetInputShape(GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_shape);
    auto inputv_shape = context->GetInputShape(INPUTV_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputv_shape);
    auto inputm_shape = context->GetInputShape(INPUTM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputm_shape);
    auto input3_shape = context->GetInputShape(INPUT3_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input3_shape);
    auto output0_shape = context->GetOutputShape(OUTPUT0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output0_shape);
    auto outputv_shape = context->GetOutputShape(OUTPUTV_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputv_shape);
    auto outputm_shape = context->GetOutputShape(OUTPUTM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputm_shape);

    // grad、inputv、inputm 三者形状必须完全相同：inputv/inputm 是原地更新的动量输出，
    // 输出形状由它们决定；grad 绑定在广播 DAG 的 In0 位，底层 Ops::Base 广播模板
    // (DoDimensionCollapse) 不支持对 In0 做广播——实测 grad 为标量、或任一维为 1 时
    // 均在 tiling 阶段被拒(“dim num is not same”/“dim index is not same with out”)。
    // 故此处直接按等形拒绝，避免放行后到 tiling 才抛 E90003。
    // 仅 input3 参与广播(右对齐，维度数可少于 inputv)。
    // 此处的判定与 tiling 的 CheckInplaceShapeConstraint 保持一致，避免两个 host
    // 阶段对同一组合给出不同结论。
    OP_CHECK_IF(!(*inputv_shape == *inputm_shape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "inputv and inputm",
                    (ToString(*inputv_shape) + " and " + ToString(*inputm_shape)).c_str(),
                    "inputv and inputm are in-place updated moments and must have the same shape"),
                return ge::GRAPH_FAILED);

    gert::Shape broadcast_shape;
    OP_CHECK_IF(!(*grad_shape == *inputv_shape),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "grad", ToString(*grad_shape).c_str(),
                    "grad does not support broadcast and must have exactly the same shape as inputv/inputm"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!BroadcastShape(input3_shape, inputv_shape, &broadcast_shape) || !(broadcast_shape == *inputv_shape),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "input3", ToString(*input3_shape).c_str(),
                    "input3 must be broadcastable into the in-place moment shape inputv/inputm"),
                return ge::GRAPH_FAILED);

    *output0_shape = *inputv_shape;
    *outputv_shape = *inputv_shape;
    *outputm_shape = *inputm_shape;

    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType4LambApplyOptimizerAssign(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "InferDataType4LambApplyOptimizerAssign enter");
    // output0 与 grad 同类型
    context->SetOutputDataType(0, context->GetInputDataType(0));
    // inputv 为同名 ref 输出
    context->SetOutputDataType(1, context->GetInputDataType(1));
    // inputm 为同名 ref 输出
    context->SetOutputDataType(2, context->GetInputDataType(2));
    OP_LOGD(context->GetNodeName(), "InferDataType4LambApplyOptimizerAssign end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LambApplyOptimizerAssign)
    .InferShape(InferShape4LambApplyOptimizerAssign)
    .InferDataType(InferDataType4LambApplyOptimizerAssign);
} // namespace ops
