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
 * \file apply_adam_with_amsgrad_v2_infershape.cpp
 * \brief ApplyAdamWithAmsgradV2 算子 InferShape 实现
 *
 * 输出 shape 与对应输入一致（out.shape = in.shape）：
 *   var_out.shape == var.shape, m_out == m, v_out == v, vhat_out == vhat
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

// 输入索引：var=0, m=1, v=2, vhat=3, beta1_power=4, beta2_power=5, lr=6,
//           beta1=7, beta2=8, epsilon=9, grad=10
// 输出索引：var_out=0, m_out=1, v_out=2, vhat_out=3
static ge::graphStatus InferShape4ApplyAdamWithAmsgradV2(gert::InferShapeContext* context)
{
    const gert::Shape* varShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    const gert::Shape* mShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mShape);
    const gert::Shape* vShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);
    const gert::Shape* vhatShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, vhatShape);

    gert::Shape* varOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varOutShape);
    gert::Shape* mOutShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mOutShape);
    gert::Shape* vOutShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, vOutShape);
    gert::Shape* vhatOutShape = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, vhatOutShape);

    *varOutShape = *varShape;
    *mOutShape = *mShape;
    *vOutShape = *vShape;
    *vhatOutShape = *vhatShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdamWithAmsgradV2).InferShape(InferShape4ApplyAdamWithAmsgradV2);

} // namespace ops
