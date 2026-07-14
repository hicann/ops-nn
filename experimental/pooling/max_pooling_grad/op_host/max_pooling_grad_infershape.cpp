/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
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
 * \file max_pooling_grad_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4MaxPoolingGrad(gert::InferShapeContext* context)
{
    // 非重叠窗口反向：dy / x / y 三者形状必须一致
    const gert::Shape* dyShape = context->GetInputShape(0); // dy
    const gert::Shape* xShape = context->GetInputShape(1);  // x
    const gert::Shape* yShape = context->GetInputShape(2);  // y
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_IF(*xShape != *dyShape || *yShape != *dyShape,
                OP_LOGE(context->GetNodeName(), "x/y shapes must equal dy shape"), return ge::GRAPH_FAILED);

    gert::Shape* dxShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);
    *dxShape = *dyShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolingGrad).InferShape(InferShape4MaxPoolingGrad).InferOutDataTypeSameWithFirstInput();

} // namespace ops
