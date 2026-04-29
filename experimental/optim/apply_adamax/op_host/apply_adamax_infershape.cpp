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
/*!
 * \file apply_adamax_infershape.cpp
 * \brief ApplyAdamax shape & dtype inference
 *
 * Outputs (varOut, mOut, vOut) shape/dtype = corresponding inputs (var, m, v).
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;

namespace ops {

// Input indices (consistent with apply_adamax_def.cpp)
static constexpr size_t IDX_VAR = 0;
static constexpr size_t IDX_M = 1;
static constexpr size_t IDX_V = 2;

// Output indices
static constexpr size_t OUT_VAR = 0;
static constexpr size_t OUT_M = 1;
static constexpr size_t OUT_V = 2;

static ge::graphStatus InferShape4ApplyAdamax(gert::InferShapeContext* context)
{
    // varOut shape = var shape
    const gert::Shape* varShape = context->GetInputShape(IDX_VAR);
    if (varShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* varOutShape = context->GetOutputShape(OUT_VAR);
    if (varOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *varOutShape = *varShape;

    // mOut shape = m shape
    const gert::Shape* mShape = context->GetInputShape(IDX_M);
    if (mShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* mOutShape = context->GetOutputShape(OUT_M);
    if (mOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *mOutShape = *mShape;

    // vOut shape = v shape
    const gert::Shape* vShape = context->GetInputShape(IDX_V);
    if (vShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* vOutShape = context->GetOutputShape(OUT_V);
    if (vOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *vOutShape = *vShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ApplyAdamax(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUT_VAR, context->GetInputDataType(IDX_VAR));
    context->SetOutputDataType(OUT_M, context->GetInputDataType(IDX_M));
    context->SetOutputDataType(OUT_V, context->GetInputDataType(IDX_V));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdamax).InferShape(InferShape4ApplyAdamax);
IMPL_OP(ApplyAdamax).InferDataType(InferDataType4ApplyAdamax);

} // namespace ops
