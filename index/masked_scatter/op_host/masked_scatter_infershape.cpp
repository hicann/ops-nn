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
 * \file masked_scatter_infershape.cpp
 * \brief
 */

#include "op_host/infershape_elewise_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static ge::graphStatus InferShape4MaskedScatter(gert::InferShapeContext *context)
{
    return Ops::Base::InferShape4Elewise(context);
}

static ge::graphStatus InferDataType4MaskedScatter(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4SameAsInput enter");
    auto input_x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_x_dtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4SameAsInput end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MskedScatter).InferShape(InferShape4MaskedScatter).InferDataType(InferDataType4MaskedScatter);
} // namespace ops
