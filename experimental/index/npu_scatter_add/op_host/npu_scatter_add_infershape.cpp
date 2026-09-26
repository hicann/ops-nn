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
 * \file npu_scatter_add_infershape.cpp
 * \brief NpuScatterAdd 算子 shape/dtype 推导：输出与输入 y 一致（inplace 语义）
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t NPUSCATTERADD_INPUT_Y_INDEX = 1;
static constexpr size_t NPUSCATTERADD_OUTPUT_Y_INDEX = 0;

static graphStatus InferDataType4NpuScatterAdd(gert::InferDataTypeContext* context)
{
    auto yDtype = context->GetInputDataType(NPUSCATTERADD_INPUT_Y_INDEX);
    context->SetOutputDataType(NPUSCATTERADD_OUTPUT_Y_INDEX, yDtype);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4NpuScatterAdd(gert::InferShapeContext* context)
{
    const gert::Shape* yShape = context->GetInputShape(NPUSCATTERADD_INPUT_Y_INDEX);
    gert::Shape* yOutShape = context->GetOutputShape(NPUSCATTERADD_OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOutShape);
    *yOutShape = *yShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NpuScatterAdd).InferShape(InferShape4NpuScatterAdd).InferDataType(InferDataType4NpuScatterAdd);
} // namespace ops
