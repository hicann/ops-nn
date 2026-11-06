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
 * \file p_relu_grad_reduce_infershape.cpp
 * \brief PReluGradReduce InferShape/InferDataType (参考 canndev prelu_grad_reduce.cc)
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

constexpr size_t GRADS_INDEX = 0;
constexpr size_t WEIGHTS_INDEX = 2;
constexpr size_t DA_INDEX = 0;

static ge::graphStatus InferShape4PReluGradReduce(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4PReluGradReduce begin");
    auto weightsShape = context->GetInputShape(WEIGHTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShape);

    auto daShape = context->GetOutputShape(DA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, daShape);
    *daShape = *weightsShape;
    OP_LOGD(context->GetNodeName(), "InferShape4PReluGradReduce end");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4PReluGradReduce(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4PReluGradReduce begin");
    context->SetOutputDataType(DA_INDEX, context->GetInputDataType(GRADS_INDEX));
    OP_LOGD(context->GetNodeName(), "InferDataType4PReluGradReduce end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PReluGradReduce).InferShape(InferShape4PReluGradReduce).InferDataType(InferDataType4PReluGradReduce);
} // namespace ops
