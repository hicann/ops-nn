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
 * \file dynamic_quant_update_scatter_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4DynamicQuantUpdateScatter(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do Infershape of InferShape4DynamicQuantUpdateScatter.");
    const gert::Shape* varShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    const gert::Shape* varScaleShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, varScaleShape);

    gert::Shape* varOutputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varOutputShape);
    gert::Shape* varScaleOutputShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, varScaleOutputShape);

    if (Ops::Base::IsUnknownRank(*varShape)) {
        OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
        Ops::Base::SetUnknownRank(*varOutputShape);
    } else {
        *varOutputShape = *varShape;
    }

    if (Ops::Base::IsUnknownRank(*varScaleShape)) {
        OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
        Ops::Base::SetUnknownRank(*varScaleOutputShape);
    } else {
        *varScaleOutputShape = *varScaleShape;
    }

    OP_LOGD(context->GetNodeName(), "varOutputShape = %s.", Ops::Base::ToString(*varOutputShape).c_str());
    OP_LOGD(context->GetNodeName(), "varScaleOutputShape = %s.", Ops::Base::ToString(*varScaleOutputShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShape4DynamicQuantUpdateScatter.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DynamicQuantUpdateScatterInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    auto input_var_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_var_dtype);
    context->SetOutputDataType(1, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicQuantUpdateScatter)
    .InferShape(InferShape4DynamicQuantUpdateScatter)
    .InferDataType(DynamicQuantUpdateScatterInferDataType);
} // namespace ops