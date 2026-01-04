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
 * \file adam_apply_one_with_decay_assign_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
constexpr size_t INPUT0_IDX = 0;
constexpr size_t INPUT1_IDX = 1;
constexpr size_t INPUT2_IDX = 2;
constexpr size_t INPUT3_IDX = 3;
constexpr size_t OUTPUT0_IDX = 0;
constexpr size_t OUTPUT1_IDX = 1;
constexpr size_t OUTPUT2_IDX = 2;

static ge::graphStatus InferShape4AdamApplyOneWithDecayAssign(gert::InferShapeContext* context)
{
    auto input0_shape = context->GetInputShape(INPUT0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0_shape);
    auto input1_shape = context->GetInputShape(INPUT1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input1_shape);
    auto input2_shape = context->GetInputShape(INPUT2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input2_shape);
    auto input3_shape = context->GetInputShape(INPUT3_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input3_shape);
    auto output0_shape = context->GetOutputShape(OUTPUT0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output0_shape);
    auto output1_shape = context->GetOutputShape(OUTPUT1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output1_shape);
    auto output2_shape = context->GetOutputShape(OUTPUT2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output2_shape);

    OP_CHECK_IF(
        !BroadcastShape(input0_shape, input1_shape, output0_shape),
        OP_LOGE(
            context->GetNodeName(), "shape %s and %s cannot broadcast!", ToString(*input0_shape).c_str(),
            ToString(*input1_shape).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !BroadcastShape(input0_shape, input2_shape, output1_shape),
        OP_LOGE(
            context->GetNodeName(), "shape %s and %s cannot broadcast!", ToString(*input0_shape).c_str(),
            ToString(*input2_shape).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !BroadcastShape(input2_shape, input3_shape, output2_shape),
        OP_LOGE(
            context->GetNodeName(), "shape %s and %s cannot broadcast!", ToString(*input2_shape).c_str(),
            ToString(*input3_shape).c_str()),
        return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdamApplyOneWithDecayAssign).InferShape(InferShape4AdamApplyOneWithDecayAssign);
} // namespace ops
