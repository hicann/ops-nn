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
 * \file scatter_min_infershape.cpp
 * \brief ScatterMin infershape: output var has the same shape/dtype as input var (in-place).
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static constexpr size_t SC_IN_VAR_IDX = 0;
static constexpr size_t SC_OUT_VAR_IDX = 0;

static ge::graphStatus InferShape4ScatterMin(gert::InferShapeContext* context)
{
    const gert::Shape* varShape = context->GetInputShape(SC_IN_VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    gert::Shape* outputShape = context->GetOutputShape(SC_OUT_VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    *outputShape = *varShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ScatterMin(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(SC_OUT_VAR_IDX, context->GetInputDataType(SC_IN_VAR_IDX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterMin).InferShape(InferShape4ScatterMin).InferDataType(InferDataType4ScatterMin);
} // namespace ops
