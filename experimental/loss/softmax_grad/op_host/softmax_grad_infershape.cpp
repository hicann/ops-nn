/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softmax_grad_infershape.cpp
 * \brief SoftmaxGrad算子的shape推理和数据类型推理实现
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeSoftmaxGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSoftmaxGrad");

    const gert::Shape* yShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* dxShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);

    *dxShape = *yShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeSoftmaxGrad");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSoftmaxGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeSoftmaxGrad");

    ge::DataType dtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, dtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeSoftmaxGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftmaxGrad).InferShape(InferShapeSoftmaxGrad).InferDataType(InferDataTypeSoftmaxGrad);
} // namespace ops
