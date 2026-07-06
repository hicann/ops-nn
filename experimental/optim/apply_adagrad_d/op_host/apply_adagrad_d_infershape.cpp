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
 * \file apply_adagrad_d_infershape.cpp
 * \brief InferShape for ApplyAdagradD
 *   output[0] (var)   = input[0] (var) shape
 *   output[1] (accum) = input[1] (accum) shape
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace gert;

namespace ops {

static ge::graphStatus InferShapeApplyAdagradD(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShapeApplyAdagradD");

    constexpr size_t INPUT_VAR_IDX = 0;
    constexpr size_t INPUT_ACCUM_IDX = 1;
    constexpr size_t OUTPUT_VAR_IDX = 0;
    constexpr size_t OUTPUT_ACCUM_IDX = 1;

    // var output shape = var input shape
    const gert::Shape* varInShape = context->GetInputShape(INPUT_VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varInShape);
    gert::Shape* varOutShape = context->GetOutputShape(OUTPUT_VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varOutShape);
    *varOutShape = *varInShape;

    // accum output shape = accum input shape
    const gert::Shape* accumInShape = context->GetInputShape(INPUT_ACCUM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, accumInShape);
    gert::Shape* accumOutShape = context->GetOutputShape(OUTPUT_ACCUM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, accumOutShape);
    *accumOutShape = *accumInShape;

    OP_LOGD(context->GetNodeName(), "End InferShapeApplyAdagradD");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdagradD).InferShape(InferShapeApplyAdagradD);

} // namespace ops
