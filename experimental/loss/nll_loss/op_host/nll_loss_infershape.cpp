/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_X_IDX = 0;
static constexpr size_t INPUT_TARGET_IDX = 1;
static constexpr size_t OUTPUT_Y_IDX = 0;
static constexpr size_t OUTPUT_TW_IDX = 1;
static constexpr size_t ATTR_REDUCTION_IDX = 0;

static ge::graphStatus InferShapeNllLoss(gert::InferShapeContext* context)
{
    const gert::Shape* targetShape = context->GetInputShape(INPUT_TARGET_IDX);
    if (targetShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    gert::Shape* twShape = context->GetOutputShape(OUTPUT_TW_IDX);
    if (yShape == nullptr || twShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const char* reduction = "mean";
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const char* r = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
        if (r != nullptr) {
            reduction = r;
        }
    }

    if (strcmp(reduction, "none") == 0) {
        *yShape = *targetShape;
    } else {
        yShape->SetDimNum(1);
        yShape->SetDim(0, 1);
    }
    twShape->SetDimNum(1);
    twShape->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeNllLoss(gert::InferDataTypeContext* context)
{
    ge::DataType xDtype = context->GetInputDataType(INPUT_X_IDX);
    context->SetOutputDataType(OUTPUT_Y_IDX, xDtype);
    context->SetOutputDataType(OUTPUT_TW_IDX, xDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NllLoss).InferShape(InferShapeNllLoss).InferDataType(InferDataTypeNllLoss);
} // namespace ops
