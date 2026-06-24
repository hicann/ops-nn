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
 * \file adam_apply_one_with_decay_assign_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
using namespace Ops::Base;
using namespace ge;
namespace ops {
constexpr size_t INPUT0_IDX = 0;
constexpr size_t INPUT_NUM = 11;
constexpr size_t OUTPUT_NUM = 3;

static ge::graphStatus CheckInputShapeEqual(gert::InferShapeContext* context, const gert::Shape* firstShape)
{
    for (size_t idx = 1; idx < INPUT_NUM; ++idx) {
        auto inputShape = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
        OP_CHECK_IF(
            ToString(*inputShape) != ToString(*firstShape),
            OP_LOGE(
                context->GetNodeName(), "shape %s and %s must be equal for AdamApplyOneWithDecayAssign!",
                ToString(*firstShape).c_str(), ToString(*inputShape).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4AdamApplyOneWithDecayAssign(gert::InferShapeContext* context)
{
    auto input0_shape = context->GetInputShape(INPUT0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0_shape);
    OP_CHECK_IF(
        CheckInputShapeEqual(context, input0_shape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "input shapes must be equal for AdamApplyOneWithDecayAssign"),
        return ge::GRAPH_FAILED);

    for (size_t idx = 0; idx < OUTPUT_NUM; ++idx) {
        auto outputShape = context->GetOutputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
        *outputShape = *input0_shape;
    }

    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AdamApplyOneWithDecayAssign).InferShape(InferShape4AdamApplyOneWithDecayAssign);
} // namespace ops
