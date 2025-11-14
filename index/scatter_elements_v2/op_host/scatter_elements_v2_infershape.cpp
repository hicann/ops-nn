/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static graphStatus InferDataType4ScatterElementsV2(gert::InferDataTypeContext* context)
{
    auto var_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, var_dtype);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ScatterElementsV2(gert::InferShapeContext* context)
{
    const gert::Shape* var_in_shape = context->GetInputShape(0);
    gert::Shape* var_out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_in_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_out_shape);

    *var_out_shape = *var_in_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterElementsV2)
    .InferShape(InferShape4ScatterElementsV2)
    .InferDataType(InferDataType4ScatterElementsV2);
} // namespace ops
