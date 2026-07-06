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
 * \file swiglu_group_quant_infershape.cpp
 * \brief Shape and dtype inference for SwigluGroupQuant.
 */

#include <algorithm>
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
namespace {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t INPUT_IDX_GROUP_INDEX = 2;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_Y_SCALE = 1;
constexpr size_t OUTPUT_IDX_Y_ORIGIN = 2;
constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr size_t ATTR_INDEX_QUANT_MODE = 1;
constexpr int64_t UNKNOWN_DIM_VALUE = -1;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t PER_MX_FP16 = 32;
constexpr int64_t BLOCK_QUANT_MODE = 0;
constexpr int64_t MX_QUANT_MODE = 1;
constexpr int64_t STATIC_HIFP8_QUANT_MODE = 2;
constexpr int64_t DYNAMIC_HIFP8_QUANT_MODE = 3;
constexpr int64_t MX_SCALE_ALIGN_FACTOR = 2;

static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_HIFLOAT8};
} // namespace

graphStatus InferShape4SwigluGroupQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4SwigluGroupQuant.");
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* yScaleShape = context->GetOutputShape(OUTPUT_IDX_Y_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, yScaleShape);
    gert::Shape* yOriginShape = context->GetOutputShape(OUTPUT_IDX_Y_ORIGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOriginShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*yScaleShape);
        Ops::Base::SetUnknownRank(*yOriginShape);
        return ge::GRAPH_SUCCESS;
    }

    int64_t xRank = static_cast<int64_t>(xShape->GetDimNum());
    OP_CHECK_IF(xRank < 1,
                OP_LOGE(context->GetNodeName(), "The rank of x should be greater than 0, but is %ld.", xRank),
                return ge::GRAPH_FAILED);
    int64_t splitDim = xRank - 1;

    if (xShape->GetDim(splitDim) == UNKNOWN_DIM_VALUE) {
        *yShape = *xShape;
        *yOriginShape = *xShape;
        yScaleShape->SetDimNum(0);
        for (int64_t i = 0; i < xRank - 1; ++i) {
            yScaleShape->AppendDim(xShape->GetDim(i));
        }
        yScaleShape->AppendDim(UNKNOWN_DIM_VALUE);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(xShape->GetDim(splitDim) < 0 || xShape->GetDim(splitDim) % NUM_TWO != 0,
                OP_LOGE(context->GetNodeName(),
                        "The last dimension of x should be non-negative and divisible by 2, but got %ld.",
                        xShape->GetDim(splitDim)),
                return ge::GRAPH_FAILED);

    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);

    const int64_t* quantModeAttr = attrsPtr->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    bool isMxQuant = quantModeAttr != nullptr && *quantModeAttr == MX_QUANT_MODE;
    bool isDynamicHif8Quant = quantModeAttr != nullptr && *quantModeAttr == DYNAMIC_HIFP8_QUANT_MODE;
    bool isStaticHif8Quant = quantModeAttr != nullptr && *quantModeAttr == STATIC_HIFP8_QUANT_MODE;

    const int64_t* dstTypePtr = attrsPtr->GetAttrPointer<int64_t>(ATTR_INDEX_DST_TYPE);
    ge::DataType dstType = (isDynamicHif8Quant || isStaticHif8Quant) ?
                               ge::DT_HIFLOAT8 :
                               (dstTypePtr == nullptr ? ge::DT_FLOAT8_E4M3FN : static_cast<ge::DataType>(*dstTypePtr));
    bool isMxFp4Quant = dstType == ge::DT_FLOAT4_E1M2 || dstType == ge::DT_FLOAT4_E2M1;

    *yShape = *xShape;
    int64_t swigluLastDim = xShape->GetDim(splitDim) / NUM_TWO;
    if (isMxFp4Quant) {
        yShape->SetDim(splitDim, (swigluLastDim + NUM_TWO - 1) / NUM_TWO);
    } else {
        yShape->SetDim(splitDim, swigluLastDim);
    }

    *yOriginShape = *xShape;
    yOriginShape->SetDim(splitDim, swigluLastDim);

    yScaleShape->SetDimNum(0);
    if (isStaticHif8Quant) {
        yScaleShape->AppendDim(0);
    } else if (isDynamicHif8Quant) {
        const gert::Shape* groupIndexShape = context->GetOptionalInputShape(INPUT_IDX_GROUP_INDEX);
        yScaleShape->AppendDim(groupIndexShape == nullptr ? 1 : groupIndexShape->GetDim(0));
    } else {
        for (int64_t i = 0; i < xRank - 1; ++i) {
            yScaleShape->AppendDim(xShape->GetDim(i));
        }
    }
    if (isMxQuant) {
        int64_t tailDim = (swigluLastDim + PER_MX_FP16 - 1) / PER_MX_FP16;
        tailDim = (tailDim + MX_SCALE_ALIGN_FACTOR - 1) / MX_SCALE_ALIGN_FACTOR;
        yScaleShape->AppendDim(tailDim);
        yScaleShape->AppendDim(MX_SCALE_ALIGN_FACTOR);
    } else if (!isDynamicHif8Quant && !isStaticHif8Quant) {
        int64_t tailDim = (swigluLastDim + PER_BLOCK_FP16 - 1) / PER_BLOCK_FP16;
        yScaleShape->AppendDim(tailDim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4SwigluGroupQuant.");
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4SwigluGroupQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDtype4SwigluGroupQuant.");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);

    const int64_t* quantModeAttr = attrsPtr->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    bool isMxQuant = quantModeAttr != nullptr && *quantModeAttr == MX_QUANT_MODE;
    bool isDynamicHif8Quant = quantModeAttr != nullptr && *quantModeAttr == DYNAMIC_HIFP8_QUANT_MODE;
    bool isStaticHif8Quant = quantModeAttr != nullptr && *quantModeAttr == STATIC_HIFP8_QUANT_MODE;

    const int64_t* dstTypePtr = attrsPtr->GetAttrPointer<int64_t>(ATTR_INDEX_DST_TYPE);
    ge::DataType dstType = (isDynamicHif8Quant || isStaticHif8Quant) ?
                               ge::DT_HIFLOAT8 :
                               (dstTypePtr == nullptr ? ge::DT_FLOAT8_E4M3FN : static_cast<ge::DataType>(*dstTypePtr));
    OP_CHECK_IF(
        std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), dstType) == Y_SUPPORT_DTYPE_SET.end(),
        OP_LOGE(context->GetNodeName(),
                "dst_type is illegal, only supports FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2 or HIFLOAT8, "
                "but got %d(%s).",
                static_cast<int32_t>(dstType), ge::TypeUtils::DataTypeToAscendString(dstType).GetString()),
        return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_IDX_Y, dstType);

    context->SetOutputDataType(OUTPUT_IDX_Y_SCALE,
                               isMxQuant && !isDynamicHif8Quant ? ge::DT_FLOAT8_E8M0 : ge::DT_FLOAT);

    auto xDtype = context->GetInputDataType(INPUT_IDX_X);
    context->SetOutputDataType(OUTPUT_IDX_Y_ORIGIN, xDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDtype4SwigluGroupQuant.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluGroupQuant).InferShape(InferShape4SwigluGroupQuant).InferDataType(InferDtype4SwigluGroupQuant);
} // namespace ops