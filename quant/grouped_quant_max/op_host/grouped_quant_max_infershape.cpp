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
 * @file grouped_quant_max_infershape.cpp
 * @brief GroupedQuantMax shape and dtype inference
 *
 * Shape inference:
 * - Output y: same shape as input x
 * - Output amax: shape [num_groups]
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

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_SCALE_INDEX = 1;
constexpr size_t INPUT_GROUP_LIST_INDEX = 2;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_AMAX_INDEX = 1;
constexpr size_t ATTR_ROUND_MODE_INDEX = 0;
constexpr size_t ATTR_DST_TYPE_INDEX = 1;
constexpr int64_t DST_TYPE_HIFLOAT8 = 34;
constexpr int64_t DST_TYPE_FLOAT8_E5M2 = 35;
constexpr int64_t DST_TYPE_FLOAT8_E4M3FN = 36;

static const std::initializer_list<ge::DataType> SUPPORTED_X_DTYPE_LIST = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

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

static ge::graphStatus InferShape4GroupedQuantMax(gert::InferShapeContext* context)
{
    const gert::Shape* inputXShape = context->GetInputShape(INPUT_X_INDEX);
    if (inputXShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* scaleShape = context->GetInputShape(INPUT_SCALE_INDEX);
    if (scaleShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* groupListShape = context->GetInputShape(INPUT_GROUP_LIST_INDEX);
    if (groupListShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 检查y、amax的shape是否正确
    gert::Shape* outputYShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    if (outputYShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputYShape = *inputXShape;

    gert::Shape* outputAmaxShape = context->GetOutputShape(OUTPUT_AMAX_INDEX);
    auto num_groups = groupListShape->GetDim(0);
    if (outputAmaxShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outputAmaxShape->SetDimNum(1);
    outputAmaxShape->SetDim(0, num_groups);

    return ge::GRAPH_SUCCESS;
}

// 检查输入和输出参数的类型是否在可支持范围内，输出y的参数是否和dstType相对应
static ge::graphStatus InferDataType4GroupedQuantMax(gert::InferDataTypeContext* context)
{
    const ge::DataType inputXDtype = context->GetInputDataType(INPUT_X_INDEX);
    if (inputXDtype == ge::DT_UNDEFINED) {
        return ge::GRAPH_FAILED;
    }

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

    const ge::DataType inputScaleDtype = context->GetInputDataType(INPUT_SCALE_INDEX);
    if (inputScaleDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType inputGroupListDtype = context->GetInputDataType(INPUT_GROUP_LIST_INDEX);
    if (inputGroupListDtype != ge::DT_INT64) {
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_DST_TYPE_INDEX);
    if (dstTypePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t dstType = *dstTypePtr;

    if (dstType != DST_TYPE_HIFLOAT8 && dstType != DST_TYPE_FLOAT8_E5M2 && dstType != DST_TYPE_FLOAT8_E4M3FN) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType outputYDtype = GetOutputYDtype(dstType);
    if (outputYDtype == ge::DT_UNDEFINED) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_Y_INDEX, outputYDtype);
    context->SetOutputDataType(OUTPUT_AMAX_INDEX, inputXDtype);

    return ge::GRAPH_SUCCESS;
}

// Register shape and dtype inference functions
IMPL_OP_INFERSHAPE(GroupedQuantMax).InferShape(InferShape4GroupedQuantMax).InferDataType(InferDataType4GroupedQuantMax);

} // namespace ops
