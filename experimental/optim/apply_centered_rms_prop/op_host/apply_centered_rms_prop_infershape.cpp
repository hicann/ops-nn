/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file apply_centered_rms_prop_infershape.cpp
 * \brief ApplyCenteredRMSProp shape / dtype inference.
 *
 * - var_out.shape = var.shape
 * - mg_out.shape  = mg.shape
 * - ms_out.shape  = ms.shape
 * - mom_out.shape = mom.shape
 * - outputs dtype follow inputs (float16 or float32)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4ApplyCenteredRMSProp(gert::InferShapeContext* context)
{
    // Input 0 = var -> Output 0 = var_out
    const gert::Shape* varShape = context->GetInputShape(0);
    if (varShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // Input 1 = mg -> Output 1 = mg_out
    const gert::Shape* mgShape = context->GetInputShape(1);
    if (mgShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // Input 2 = ms -> Output 2 = ms_out
    const gert::Shape* msShape = context->GetInputShape(2);
    if (msShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // Input 3 = mom -> Output 3 = mom_out
    const gert::Shape* momShape = context->GetInputShape(3);
    if (momShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* varOutShape = context->GetOutputShape(0);
    gert::Shape* mgOutShape  = context->GetOutputShape(1);
    gert::Shape* msOutShape  = context->GetOutputShape(2);
    gert::Shape* momOutShape = context->GetOutputShape(3);
    if (varOutShape == nullptr || mgOutShape == nullptr ||
        msOutShape == nullptr || momOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *varOutShape = *varShape;
    *mgOutShape  = *mgShape;
    *msOutShape  = *msShape;
    *momOutShape = *momShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ApplyCenteredRMSProp(gert::InferDataTypeContext* context)
{
    // Outputs dtype follow their corresponding Ref input.
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    context->SetOutputDataType(2, context->GetInputDataType(2));
    context->SetOutputDataType(3, context->GetInputDataType(3));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyCenteredRMSProp)
    .InferShape(InferShape4ApplyCenteredRMSProp)
    .InferDataType(InferDataType4ApplyCenteredRMSProp);

} // namespace ops
