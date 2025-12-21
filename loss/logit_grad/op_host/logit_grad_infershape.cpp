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
 * \file logit_grad_infershape.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShape4LogitGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4LogitGrad");

    // get input shapes
    auto xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    auto dxShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);

    size_t xDimNum = xShape->GetDimNum();
    dxShape->SetDimNum(xDimNum);

    *dxShape = *xShape;

    OP_LOGD(context, "End to do InferShape4LogitGrad");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4LogitGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4LogitGrad");

    auto input_dtype = context->GetInputDataType(IDX_0);

    context->SetOutputDataType(IDX_0, input_dtype);

    OP_LOGD(context, "End to do InferDataType4LogitGrad");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LogitGrad).InferShape(InferShape4LogitGrad).InferDataType(InferDataType4LogitGrad);
} // namespace ops