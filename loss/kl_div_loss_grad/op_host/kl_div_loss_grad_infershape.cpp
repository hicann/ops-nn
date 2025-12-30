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
 * \file elewise.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops
{

ge::graphStatus CopyShapeInput2OutputWithForKlDivLossGrad(gert::InferShapeContext* context, int64_t input_idx, int64_t output_idx) {
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(output_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4InIdx1AndOutIdx0(gert::InferShapeContext* context)
{
    constexpr size_t input_index = 1;
    constexpr size_t output_index = 0;
    return CopyShapeInput2OutputWithForKlDivLossGrad(context, input_index, output_index);
}

graphStatus InferDtypeForKlDivLossGrad(gert::InferDataTypeContext* context)
{
    auto xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KlDivLossGrad).InferShape(InferShape4InIdx1AndOutIdx0)
                                 .InferDataType(InferDtypeForKlDivLossGrad);
}
