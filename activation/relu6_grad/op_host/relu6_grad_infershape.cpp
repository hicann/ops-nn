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
 * \file relu6_grad_infershape.cpp
 * \brief
 */
#include "op_host/infershape_broadcast_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4Relu6Grad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4Relu6Grad in ops-nn");
    return Ops::Base::InferShape4Broadcast(context);
}

static ge::graphStatus InferDataType4Relu6Grad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "InferDataType4Relu6Grad enter");
    // backprops 与 gradients 同类型
    context->SetOutputDataType(0, context->GetInputDataType(0));
    OP_LOGD(context->GetNodeName(), "InferDataType4Relu6Grad end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Relu6Grad).InferShape(InferShape4Relu6Grad).InferDataType(InferDataType4Relu6Grad);

} // namespace ops
