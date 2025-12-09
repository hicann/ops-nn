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
 * \file advance_step_infershape.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;
static constexpr int64_t IDX_3 = 3;
static constexpr int64_t IDX_4 = 4;
static constexpr int64_t IDX_5 = 5;

static constexpr size_t SIZE_2 = 2;

static ge::graphStatus InferShape4AdvanceStep(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4AdvanceStep");

    // get input shapes
    auto x1Shape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);

    // get output shapes
    auto y1Shape = context->GetOutputShape(IDX_0);
    auto y2Shape = context->GetOutputShape(IDX_1);
    auto y3Shape = context->GetOutputShape(IDX_2);
    auto y4Shape = context->GetOutputShape(IDX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y3Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y4Shape);

    *y1Shape = *x1Shape;
    *y2Shape = *x1Shape;
    *y3Shape = *x1Shape;
    *y4Shape = *x1Shape;

    OP_LOGD(context, "End to do InferShape4AdvanceStep");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AdvanceStep(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4AdvanceStep");

    auto input_dtype = context->GetInputDataType(IDX_0);

    context->SetOutputDataType(IDX_0, input_dtype);
    context->SetOutputDataType(IDX_1, input_dtype);
    context->SetOutputDataType(IDX_2, input_dtype);
    context->SetOutputDataType(IDX_3, input_dtype);

    OP_LOGD(context, "End to do InferDataType4AdvanceStep");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdvanceStep).InferShape(InferShape4AdvanceStep).InferDataType(InferDataType4AdvanceStep);
} // namespace ops