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
 * \file modulate_grad_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_GRAD_OUTPUT_IDX = 0;
static constexpr size_t INPUT_SCALE_IDX = 2;
static constexpr size_t INPUT_SHIFT_IDX = 3;
static constexpr size_t OUTPUT_GRAD_INPUT_IDX = 0;
static constexpr size_t OUTPUT_GRAD_SCALE_IDX = 1;
static constexpr size_t OUTPUT_GRAD_SHIFT_IDX = 2;

// grad_scale/grad_shift are optional outputs. Verified on real GE (single-op graph compile): GE does
// NOT compact optional outputs — an absent optional output is kept as an empty placeholder at its
// fixed IR index, so the IR index is also the output instance index. We therefore index outputs by
// their fixed IR index and only write an optional output when its matching optional input exists.
static ge::graphStatus ModulateGradInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do ModulateGradInferShape");

    // grad_input shares the shape of grad_output
    const gert::Shape* gradOutputShape = context->GetInputShape(INPUT_GRAD_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradOutputShape);
    gert::Shape* gradInputShape = context->GetOutputShape(OUTPUT_GRAD_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradInputShape);
    *gradInputShape = *gradOutputShape;

    // grad_scale is optional and shares the shape of the optional scale input
    const gert::Shape* scaleShape = context->GetOptionalInputShape(INPUT_SCALE_IDX);
    gert::Shape* gradScaleShape = context->GetOutputShape(OUTPUT_GRAD_SCALE_IDX);
    if (scaleShape != nullptr && gradScaleShape != nullptr) {
        *gradScaleShape = *scaleShape;
    }

    // grad_shift is optional and shares the shape of the optional shift input
    const gert::Shape* shiftShape = context->GetOptionalInputShape(INPUT_SHIFT_IDX);
    gert::Shape* gradShiftShape = context->GetOutputShape(OUTPUT_GRAD_SHIFT_IDX);
    if (shiftShape != nullptr && gradShiftShape != nullptr) {
        *gradShiftShape = *shiftShape;
    }

    OP_LOGD(context, "End to do ModulateGradInferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ModulateGradInferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "ModulateGradInferDataType enter");

    context->SetOutputDataType(OUTPUT_GRAD_INPUT_IDX, context->GetInputDataType(INPUT_GRAD_OUTPUT_IDX));

    // grad_scale/grad_shift are optional; only set them when the matching input is instantiated
    const ge::DataType scaleDtype = context->GetOptionalInputDataType(INPUT_SCALE_IDX);
    if (scaleDtype != ge::DT_UNDEFINED) {
        context->SetOutputDataType(OUTPUT_GRAD_SCALE_IDX, scaleDtype);
    }
    const ge::DataType shiftDtype = context->GetOptionalInputDataType(INPUT_SHIFT_IDX);
    if (shiftDtype != ge::DT_UNDEFINED) {
        context->SetOutputDataType(OUTPUT_GRAD_SHIFT_IDX, shiftDtype);
    }

    OP_LOGD(context, "ModulateGradInferDataType end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ModulateGrad).InferShape(ModulateGradInferShape).InferDataType(ModulateGradInferDataType);
} // namespace ops
