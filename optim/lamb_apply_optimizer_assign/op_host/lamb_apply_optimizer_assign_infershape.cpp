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
    auto output0_shape = context->GetOutputShape(OUTPUT0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output0_shape);
    auto outputv_shape = context->GetOutputShape(OUTPUTV_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputv_shape);
    auto outputm_shape = context->GetOutputShape(OUTPUTM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputm_shape);

    OP_CHECK_IF(!BroadcastShape(grad_shape, inputv_shape, output0_shape),
                OP_LOGE(context->GetNodeName(), "shape %s and %s cannot broadcast!", ToString(*grad_shape).c_str(),
                        ToString(*inputv_shape).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!BroadcastShape(grad_shape, inputv_shape, outputv_shape),
                OP_LOGE(context->GetNodeName(), "shape %s and %s cannot broadcast!", ToString(*grad_shape).c_str(),
                        ToString(*inputv_shape).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!BroadcastShape(grad_shape, inputm_shape, outputm_shape),
                OP_LOGE(context->GetNodeName(), "shape %s and %s cannot broadcast!", ToString(*grad_shape).c_str(),
                        ToString(*inputm_shape).c_str()),
                return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(LambApplyOptimizerAssign).InferShape(InferShape4LambApplyOptimizerAssign);
} // namespace ops
