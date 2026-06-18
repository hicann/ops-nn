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
 * \file apply_adam_d_infershape.cpp
 * \brief ApplyAdamD shape & dtype inference (ascend910b)
 *
 * Outputs (var, m, v) shape/dtype = corresponding inputs (var, m, v).
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;

namespace ops {
// Input indices (consistent with apply_adam_d_def.cpp)
static constexpr size_t IDX_VAR = 0;
static constexpr size_t IDX_M = 1;
static constexpr size_t IDX_V = 2;

// Output indices
static constexpr size_t OUT_VAR = 0;
static constexpr size_t OUT_M = 1;
static constexpr size_t OUT_V = 2;

static ge::graphStatus CopyShapeInput2Output(gert::InferShapeContext* context, size_t inIdx, size_t outIdx)
{
    const gert::Shape* inShape = context->GetInputShape(inIdx);
    if (inShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(outIdx);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ApplyAdamD(gert::InferShapeContext* context)
{
    if (CopyShapeInput2Output(context, IDX_VAR, OUT_VAR) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CopyShapeInput2Output(context, IDX_M, OUT_M) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CopyShapeInput2Output(context, IDX_V, OUT_V);
}

static ge::graphStatus InferDataType4ApplyAdamD(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUT_VAR, context->GetInputDataType(IDX_VAR));
    context->SetOutputDataType(OUT_M, context->GetInputDataType(IDX_M));
    context->SetOutputDataType(OUT_V, context->GetInputDataType(IDX_V));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdamD).InferShape(InferShape4ApplyAdamD);
IMPL_OP(ApplyAdamD).InferDataType(InferDataType4ApplyAdamD);

}  // namespace ops
