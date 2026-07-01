/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr int64_t INPUT_X = 0;
static constexpr int64_t INPUT_INDICES = 1;
static constexpr int64_t OUTPUT_Y = 0;

static ge::graphStatus InferShape4GatherV2(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* indicesShape = context->GetInputShape(INPUT_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    (void)indicesShape;
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // axis is a scalar input tensor. Dynamic axis value is unavailable in InferShape,
    // so the architecture skeleton keeps output shape equal to x until const-axis
    // inference is added.
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GatherV2).InferShape(InferShape4GatherV2);
} // namespace ops
