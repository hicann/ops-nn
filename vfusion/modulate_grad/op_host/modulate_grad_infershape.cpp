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
#include "util/shape_util.h"

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
    const gert::Shape* inputShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const bool gradOutputUnknownRank = Ops::Base::IsUnknownRank(*gradOutputShape);
    const bool inputUnknownRank = Ops::Base::IsUnknownRank(*inputShape);
    if ((!gradOutputUnknownRank && gradOutputShape->GetDimNum() != 3) ||
        (!inputUnknownRank && inputShape->GetDimNum() != 3)) {
        OP_LOGE(context, "ModulateGrad requires grad_output and input rank 3, got %zu and %zu.",
                gradOutputShape->GetDimNum(), inputShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    // -1 is the documented dynamic-shape marker; any other negative dim is invalid.
    for (size_t i = 0; i < 3 && !gradOutputUnknownRank; ++i) {
        if (gradOutputShape->GetDim(i) < -1) {
            OP_LOGE(context, "ModulateGrad requires dims >= -1, got grad_output dim %ld, input dim %ld.",
                    gradOutputShape->GetDim(i), inputUnknownRank ? ge::UNKNOWN_DIM_NUM : inputShape->GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t i = 0; i < 3 && !inputUnknownRank; ++i) {
        if (inputShape->GetDim(i) < -1) {
            OP_LOGE(context, "ModulateGrad requires input dims >= -1, got dim %ld.", inputShape->GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
    // Dynamic dims (-1) are resolved at tiling time; compare concrete dims only.
    if (!gradOutputUnknownRank && !inputUnknownRank) {
        for (size_t i = 0; i < 3; ++i) {
            const int64_t goDim = gradOutputShape->GetDim(i);
            const int64_t inDim = inputShape->GetDim(i);
            if (goDim >= 0 && inDim >= 0 && goDim != inDim) {
                OP_LOGE(context, "ModulateGrad requires grad_output shape equal to input shape, dim %zu: %ld vs %ld.",
                        i, goDim, inDim);
                return ge::GRAPH_FAILED;
            }
        }
    }
    const gert::Shape* referenceShape = !inputUnknownRank ? inputShape :
                                                            (!gradOutputUnknownRank ? gradOutputShape : nullptr);
    const int64_t B = referenceShape == nullptr ? ge::UNKNOWN_DIM : referenceShape->GetDim(0);
    const int64_t D = referenceShape == nullptr ? ge::UNKNOWN_DIM : referenceShape->GetDim(2);
    const bool inputEmpty = referenceShape != nullptr && (B == 0 || referenceShape->GetDim(1) == 0 || D == 0);
    auto checkOptional = [&](const gert::Shape* opt, const char* name) -> bool {
        if (opt == nullptr) {
            return true;
        }
        if (Ops::Base::IsUnknownRank(*opt)) {
            return true;
        }
        // An absent optional input is kept as a rank-0/zero-size empty placeholder at its
        // fixed IR index; every zero-size optional is treated as absent and skipped.
        // Match the tiling-side rule (GetShapeSize() == 0): placeholders like (0, 1)/(1, 0)
        // are also empty, not present tensors.
        const bool optEmpty = opt->GetDimNum() == 0 || opt->GetShapeSize() == 0;
        if (optEmpty) {
            return true;
        }
        if (inputEmpty) {
            OP_LOGE(context, "ModulateGrad requires %s empty or absent when grad_output/input is empty.", name);
            return false;
        }
        if (opt->GetDimNum() != 2) {
            OP_LOGE(context, "ModulateGrad requires %s rank 2, got %zu.", name, opt->GetDimNum());
            return false;
        }
        const int64_t dim0 = opt->GetDim(0);
        const int64_t dim1 = opt->GetDim(1);
        if (dim0 < -1 || dim1 < -1) {
            OP_LOGE(context, "ModulateGrad requires %s dims >= -1, got [%ld, %ld].", name, dim0, dim1);
            return false;
        }
        if (referenceShape != nullptr && ((dim0 >= 0 && B >= 0 && dim0 != B) || (dim1 >= 0 && D >= 0 && dim1 != D))) {
            OP_LOGE(context, "ModulateGrad requires %s shape [%ld, %ld], got [%ld, %ld].", name, B, D, dim0, dim1);
            return false;
        }
        return true;
    };
    const gert::Shape* scaleShapeRaw = context->GetOptionalInputShape(INPUT_SCALE_IDX);
    const gert::Shape* shiftShapeRaw = context->GetOptionalInputShape(INPUT_SHIFT_IDX);
    if (!checkOptional(scaleShapeRaw, "scale") || !checkOptional(shiftShapeRaw, "shift")) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* gradInputShape = context->GetOutputShape(OUTPUT_GRAD_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradInputShape);
    *gradInputShape = *gradOutputShape;

    // grad_scale is optional and shares the shape of the optional scale input
    gert::Shape* gradScaleShape = context->GetOutputShape(OUTPUT_GRAD_SCALE_IDX);
    if (scaleShapeRaw != nullptr && gradScaleShape != nullptr) {
        *gradScaleShape = *scaleShapeRaw;
    }

    // grad_shift is optional and shares the shape of the optional shift input
    gert::Shape* gradShiftShape = context->GetOutputShape(OUTPUT_GRAD_SHIFT_IDX);
    if (shiftShapeRaw != nullptr && gradShiftShape != nullptr) {
        *gradShiftShape = *shiftShapeRaw;
    }

    OP_LOGD(context, "End to do ModulateGradInferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ModulateGradInferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "ModulateGradInferDataType enter");

    const ge::DataType gradOutputDtype = context->GetInputDataType(INPUT_GRAD_OUTPUT_IDX);
    const ge::DataType inputDtype = context->GetInputDataType(1);
    if (gradOutputDtype != ge::DT_FLOAT16 && gradOutputDtype != ge::DT_BF16 && gradOutputDtype != ge::DT_FLOAT) {
        OP_LOGE(context, "ModulateGrad unsupported grad_output dtype %d.", static_cast<int32_t>(gradOutputDtype));
        return ge::GRAPH_FAILED;
    }
    if (inputDtype != gradOutputDtype) {
        OP_LOGE(context, "ModulateGrad requires input dtype %d equal to grad_output dtype %d.",
                static_cast<int32_t>(inputDtype), static_cast<int32_t>(gradOutputDtype));
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(OUTPUT_GRAD_INPUT_IDX, gradOutputDtype);

    // grad_scale/grad_shift are optional; only set them when the matching input is instantiated
    const ge::DataType scaleDtype = context->GetOptionalInputDataType(INPUT_SCALE_IDX);
    if (scaleDtype != ge::DT_UNDEFINED) {
        if (scaleDtype != gradOutputDtype) {
            OP_LOGE(context, "ModulateGrad requires scale dtype %d equal to grad_output dtype %d.",
                    static_cast<int32_t>(scaleDtype), static_cast<int32_t>(gradOutputDtype));
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(OUTPUT_GRAD_SCALE_IDX, scaleDtype);
    }
    const ge::DataType shiftDtype = context->GetOptionalInputDataType(INPUT_SHIFT_IDX);
    if (shiftDtype != ge::DT_UNDEFINED) {
        if (shiftDtype != gradOutputDtype) {
            OP_LOGE(context, "ModulateGrad requires shift dtype %d equal to grad_output dtype %d.",
                    static_cast<int32_t>(shiftDtype), static_cast<int32_t>(gradOutputDtype));
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(OUTPUT_GRAD_SHIFT_IDX, shiftDtype);
    }

    OP_LOGD(context, "ModulateGradInferDataType end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ModulateGrad).InferShape(ModulateGradInferShape).InferDataType(ModulateGradInferDataType);
} // namespace ops
