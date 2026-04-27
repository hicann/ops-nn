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

constexpr size_t FEATURES_INDEX = 1;
constexpr size_t WEIGHTS_INDEX = 2;
constexpr size_t DX_INDEX = 0;
constexpr size_t DA_INDEX = 1;

static ge::graphStatus InferShape4PReluGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4PReluGrad begin");
    auto features_shape = context->GetInputShape(FEATURES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, features_shape);
    auto weights_shape = context->GetInputShape(WEIGHTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weights_shape);

    auto dx_shape = context->GetOutputShape(DX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dx_shape);
    *dx_shape = *features_shape;

    auto da_shape = context->GetOutputShape(DA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, da_shape);
    *da_shape = *weights_shape;
    return ge::GRAPH_SUCCESS;

}

static ge::graphStatus InferDataType4PReluGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4PReluGrad begin");
    auto freatures_dtype = context->GetInputDataType(FEATURES_INDEX);
    auto weights_dtype = context->GetInputDataType(WEIGHTS_INDEX);
    context->SetOutputDataType(DX_INDEX, freatures_dtype);
    context->SetOutputDataType(DA_INDEX, weights_dtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4PReluGrad end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PReluGrad).InferShape(InferShape4PReluGrad).InferDataType(InferDataType4PReluGrad);
} // namespace ops