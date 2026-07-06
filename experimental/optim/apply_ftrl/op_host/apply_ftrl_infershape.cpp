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
 * \file apply_ftrl_infershape.cpp
 * \brief ApplyFtrl shape / dtype inference (ascend910b).
 *
 * - var_out.shape = var.shape (input 0); output dtype follows var.
 * - MED-3 shape consistency: accum/linear/grad must match var exactly (DimNum +
 *   per-dim size); a mismatch fails graph compilation (spec shape_mismatch).
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;

namespace ops {

// var / accum / linear / grad input indices.
static constexpr size_t kIdxVar = 0;
static constexpr size_t kIdxAccum = 1;
static constexpr size_t kIdxLinear = 2;
static constexpr size_t kIdxGrad = 3;

static bool ShapeEqual(const gert::Shape* a, const gert::Shape* b)
{
    if (a == nullptr || b == nullptr) {
        return false;
    }
    if (a->GetDimNum() != b->GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a->GetDimNum(); ++i) {
        if (a->GetDim(i) != b->GetDim(i)) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus InferShape4ApplyFtrl(gert::InferShapeContext* context)
{
    const gert::Shape* varShape = context->GetInputShape(kIdxVar);
    const gert::Shape* accumShape = context->GetInputShape(kIdxAccum);
    const gert::Shape* linearShape = context->GetInputShape(kIdxLinear);
    const gert::Shape* gradShape = context->GetInputShape(kIdxGrad);
    if (varShape == nullptr || accumShape == nullptr || linearShape == nullptr || gradShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // MED-3: the four element-wise tensors must share the exact same shape.
    if (!ShapeEqual(varShape, accumShape) || !ShapeEqual(varShape, linearShape) || !ShapeEqual(varShape, gradShape)) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* varOutShape = context->GetOutputShape(0);
    if (varOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *varOutShape = *varShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ApplyFtrl(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(kIdxVar));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyFtrl).InferShape(InferShape4ApplyFtrl).InferDataType(InferDataType4ApplyFtrl);

} // namespace ops
