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
 * @file quant_max_infershape.cpp
 * @brief QuantMax shape and dtype inference
 *
 * Shape inference:
 * - Output y: same shape as input x
 * - Output amax: shape [1]
 *
 * Dtype inference:
 * - Output y: determined by dst_type attribute (34=HIFLOAT8, 35=FLOAT8_E5M2, 36=FLOAT8_E4M3FN)
 * - Output amax: same dtype as input x
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/ge_error_codes.h"

using namespace ge;

namespace ops {

// Input/Output indices
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_SCALE_INDEX = 1;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_AMAX_INDEX = 1;

// Attribute indices
constexpr size_t ATTR_ROUND_MODE_INDEX = 0;
constexpr size_t ATTR_DST_TYPE_INDEX = 1;

// dst_type enum values
constexpr int64_t DST_TYPE_HIFLOAT8 = 34;
constexpr int64_t DST_TYPE_FLOAT8_E5M2 = 35;
constexpr int64_t DST_TYPE_FLOAT8_E4M3FN = 36;

// Supported input dtype list
static const std::initializer_list<ge::DataType> SUPPORTED_X_DTYPE_LIST = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

/**
 * @brief Get output y dtype based on dst_type attribute
 * @param dst_type The dst_type attribute value
 * @return Corresponding ge::DataType for output y
 */
static ge::DataType GetOutputYDtype(int64_t dst_type)
{
    switch (dst_type) {
        case DST_TYPE_HIFLOAT8:
            return ge::DT_HIFLOAT8;
        case DST_TYPE_FLOAT8_E5M2:
            return ge::DT_FLOAT8_E5M2;
        case DST_TYPE_FLOAT8_E4M3FN:
            return ge::DT_FLOAT8_E4M3FN;
        default:
            return ge::DT_UNDEFINED;
    }
}

/**
 * @brief Infer output shapes for QuantMax
 * @param context InferShapeContext pointer
 * @return ge::GRAPH_SUCCESS on success, ge::GRAPH_FAILED on failure
 */
static ge::graphStatus InferShape4QuantMax(gert::InferShapeContext* context)
{
    // Get input x shape
    const gert::Shape* inputXShape = context->GetInputShape(INPUT_X_INDEX);
    if (inputXShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Get input x shape
    const gert::Shape* scaleShape = context->GetInputShape(INPUT_SCALE_INDEX);
    if (scaleShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Output y shape = input x shape
    gert::Shape* outputYShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    if (outputYShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputYShape = *inputXShape;

    // Output amax shape = [1]
    gert::Shape* outputAmaxShape = context->GetOutputShape(OUTPUT_AMAX_INDEX);
    if (outputAmaxShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outputAmaxShape->SetDimNum(1);
    outputAmaxShape->SetDim(0, 1);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief Infer output data types for QuantMax
 * @param context InferDataTypeContext pointer
 * @return ge::GRAPH_SUCCESS on success, ge::GRAPH_FAILED on failure
 */
static ge::graphStatus InferDataType4QuantMax(gert::InferDataTypeContext* context)
{
    // Get input x dtype
    const ge::DataType inputXDtype = context->GetInputDataType(INPUT_X_INDEX);
    if (inputXDtype == ge::DT_UNDEFINED) {
        return ge::GRAPH_FAILED;
    }

    // Validate input x dtype: must be FLOAT, FLOAT16 or BFLOAT16
    bool isValidXDtype = false;
    for (const auto& dtype : SUPPORTED_X_DTYPE_LIST) {
        if (inputXDtype == dtype) {
            isValidXDtype = true;
            break;
        }
    }
    if (!isValidXDtype) {
        return ge::GRAPH_FAILED;
    }

    // Get input scale dtype, must be FLOAT
    const ge::DataType inputScaleDtype = context->GetInputDataType(INPUT_SCALE_INDEX);
    if (inputScaleDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    // Get dst_type attribute
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_DST_TYPE_INDEX);
    if (dstTypePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t dstType = *dstTypePtr;

    // Validate dst_type: must be 34, 35 or 36
    if (dstType != DST_TYPE_HIFLOAT8 && dstType != DST_TYPE_FLOAT8_E5M2 && dstType != DST_TYPE_FLOAT8_E4M3FN) {
        return ge::GRAPH_FAILED;
    }

    // Infer output y dtype from dst_type
    ge::DataType outputYDtype = GetOutputYDtype(dstType);
    if (outputYDtype == ge::DT_UNDEFINED) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_Y_INDEX, outputYDtype);

    // Output amax dtype = input x dtype
    context->SetOutputDataType(OUTPUT_AMAX_INDEX, inputXDtype);

    return ge::GRAPH_SUCCESS;
}

// Register shape and dtype inference functions
IMPL_OP_INFERSHAPE(QuantMax).InferShape(InferShape4QuantMax).InferDataType(InferDataType4QuantMax);

} // namespace ops
