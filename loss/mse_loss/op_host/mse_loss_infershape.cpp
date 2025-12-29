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
 * \file mse_loss_infershape.cpp
 * \brief
 */

#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeTwoInOneOutWithReduction(gert::InferShapeContext* context)
{
    auto input_x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_x_shape);
    auto input_y_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_y_shape);

    OP_CHECK_IF(
        *input_x_shape != *input_y_shape,
        OP_LOGE(
            context->GetNodeName(), "input_x shape %s must be same as input_y shape %s",
            Ops::Base::ToString(*input_x_shape).c_str(), Ops::Base::ToString(*input_y_shape).c_str()),
        return GRAPH_FAILED);

    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char* reduction = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, reduction);
    if (strcmp(reduction, "none") == 0) {
        auto in_shape = context->GetInputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
        *out_shape = *in_shape;
    } else {
        // if reduction == "mean" or reduction == "sum" , output a scalar
        out_shape->SetDimNum(0);
    }

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MseLoss).InferShape(InferShapeTwoInOneOutWithReduction);

} // namespace ops