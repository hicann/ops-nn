/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform/platform_info.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include <list>

using namespace ge;
namespace ops {

constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;

static ge::graphStatus InferShape4SoftmaxGradExt(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4SoftmaxGradExt begin");

    auto x1_shape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);

    auto y_shape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    *y_shape = *x1_shape;

    return ge::GRAPH_SUCCESS;

}

static ge::graphStatus InferDataType4SoftmaxGradExt(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4SoftmaxGradExt begin");
    auto x1_dtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(Y_INDEX, x1_dtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4SoftmaxGradExt end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftmaxGradExt).InferShape(InferShape4SoftmaxGradExt).InferDataType(InferDataType4SoftmaxGradExt);
} // namespace ops