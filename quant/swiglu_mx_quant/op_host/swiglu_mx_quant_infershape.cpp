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
 * \file swiglu_mx_quant_infershape.cpp
 * \brief Shape inference for SwiGLU + MX quantization operator
 */

#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;

namespace ops {
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr int64_t UNKNOWN_RANK_DIM = -2;
constexpr size_t INDEX_INPUT_X = 0;
constexpr size_t INDEX_INPUT_GROUP_INDEX = 1;
constexpr size_t INDEX_OUTPUT_Y = 0;
constexpr size_t INDEX_OUTPUT_MXSCALE = 1;

// 属性索引（按 README 中的属性顺序）
constexpr size_t INDEX_ATTR_ACTIVATE_DIM = 0;
constexpr size_t INDEX_ATTR_AXIS = 7;
constexpr size_t INDEX_ATTR_DST_TYPE = 8;

constexpr int64_t SPLIT_NUM = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t ALIGN_NUM = 2;
constexpr size_t MAX_DIM_NUM = 7;

static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

graphStatus ComputeInferShape(gert::InferShapeContext* context, const gert::Shape* xShape, gert::Shape* yShape,
                              gert::Shape* mxscaleShape)
{
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);

    const int64_t* activateDim = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_ACTIVATE_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, activateDim);
    const int64_t* axis = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    // Normalize activate_dim
    int64_t xRank = xShape->GetDimNum();
    OP_CHECK_IF(xRank <= 1,
        OP_LOGE(context->GetNodeName(),
        "the rank of x must be greater than 1, but is %ld", xRank),
        return ge::GRAPH_FAILED);
    int64_t activateDimNorm = (*activateDim >= 0) ? static_cast<int64_t>(*activateDim)
                                                  : static_cast<int64_t>(*activateDim + xRank);
    // Normalize axis (for y shape)
    int64_t axisNorm = (*axis >= 0) ? static_cast<int64_t>(*axis)
                                   : static_cast<int64_t>(*axis + xRank);
    OP_CHECK_IF(activateDimNorm >= xRank || axisNorm >= xRank,
        OP_LOGE(context->GetNodeName(),
        "activateDim and axis must < xRank, but xRank is %ld, activateDim is %ld, axis is %ld", xRank,
        activateDimNorm, axisNorm),
        return ge::GRAPH_FAILED);
    // Validate activate_dim dimension is divisible by 2
    if (xShape->GetDim(activateDimNorm) != UNKNOWN_DIM_VALUE_ &&
        xShape->GetDim(activateDimNorm) % SPLIT_NUM != 0) {
        OP_LOGE(context->GetNodeName(), "The dimension [%zu] at activate_dim must be divisible by 2, but got [%ld].",
                activateDimNorm, xShape->GetDim(activateDimNorm));
        return ge::GRAPH_FAILED;
    }

    // Step 1: Compute y shape (SwiGLU output)
    // y 的 shape 与 x 的 shape 维度一致，在 activate_dim 维度上是 x 的一半
    *yShape = *xShape;
    if (xShape->GetDim(activateDimNorm) != UNKNOWN_DIM_VALUE_) {
        yShape->SetDim(activateDimNorm, xShape->GetDim(activateDimNorm) / SPLIT_NUM);
    }

    // Step 2: Compute mxscale shape
    // 获取可选输入 group_index
    const gert::Shape* groupIndexShape = context->GetOptionalInputShape(INDEX_INPUT_GROUP_INDEX);
    bool hasGroup = (groupIndexShape != nullptr);
    OP_CHECK_IF(
        hasGroup && groupIndexShape->GetDimNum() != 1,
        OP_LOGE(context->GetNodeName(), "the rank of group_index must be 1, but is %d", groupIndexShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    int64_t groupIndexNum = 0;
    if (hasGroup) {
        if (groupIndexShape->GetDim(0) != UNKNOWN_DIM_VALUE_) {
            groupIndexNum = groupIndexShape->GetDim(0);
        }
    }
    // 校验：当前仅支持 axis 为尾轴
    OP_CHECK_IF(axisNorm != xRank - 1,
        OP_LOGE(context->GetNodeName(), "Only axis=-1 (tail axis) is supported currently, but got axis=%ld (normalized=%ld).",
                *axis, axisNorm),
        return ge::GRAPH_FAILED);

    // 计算 mxscale 的倒数第二维
    int64_t yAxisSize = 0;
    if (yShape->GetDim(axisNorm) == UNKNOWN_DIM_VALUE_) {
        yAxisSize = UNKNOWN_DIM_VALUE_;
    } else {
        int64_t yDim = yShape->GetDim(axisNorm);
        if (hasGroup && axisNorm == yShape->GetDimNum() - 2) {
            // floor(yDim / BLOCK_SIZE) + groupIndexNum
            if (groupIndexNum == UNKNOWN_DIM_VALUE_ || groupIndexNum == UNKNOWN_RANK_DIM) {
                yAxisSize = UNKNOWN_DIM_VALUE_;
            } else {
                yAxisSize = yDim / (ALIGN_NUM * BLOCK_SIZE) + groupIndexNum;
            }
        } else {
            // ceil(yDim / BLOCK_SIZE)
            yAxisSize = Ops::Base::CeilDiv(yDim, ALIGN_NUM * BLOCK_SIZE);
        }
    }

    *mxscaleShape = *yShape;
    mxscaleShape->SetDim(axisNorm, yAxisSize);
    mxscaleShape->AppendDim(ALIGN_NUM);
    OP_LOGI(context->GetNodeName(), "End to do InferShapeForSwigluMxQuant");
    return ge::GRAPH_SUCCESS;
}

graphStatus InferShapeForSwigluMxQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForSwigluMxQuant");
    const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* mxscaleShape = context->GetOutputShape(INDEX_OUTPUT_MXSCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxscaleShape);

    OP_CHECK_IF(
        xShape->GetDimNum() < 1 || xShape->GetDimNum() > MAX_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "Input x rank[%lu] should be in [1, 7].", xShape->GetDimNum()),
        return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, mxscale shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*mxscaleShape);
        return ge::GRAPH_SUCCESS;
    }
    return ComputeInferShape(context, xShape, yShape, mxscaleShape);
}

ge::graphStatus InferDataTypeForSwigluMxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForSwigluMxQuant");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    OP_CHECK_IF(
        std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), outDtype) == Y_SUPPORT_DTYPE_SET.end(),
        OP_LOGE(
            context->GetNodeName(),
            "dst_type is illegal, only supports 40(FLOAT4_E2M1), 41(FLOAT4_E1M2), "
            "36(FLOAT8_E4M3FN) or 35(FLOAT8_E5M2). but got %d(%s) please check.",
            *dstDtype, ge::TypeUtils::DataTypeToAscendString(outDtype).GetString()),
        return ge::GRAPH_FAILED);
    context->SetOutputDataType(INDEX_OUTPUT_Y, outDtype);
    context->SetOutputDataType(INDEX_OUTPUT_MXSCALE, ge::DT_FLOAT8_E8M0);
    OP_LOGI(context->GetNodeName(), "End to do InferDataTypeForSwigluMxQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluMxQuant)
    .InferShape(InferShapeForSwigluMxQuant)
    .InferDataType(InferDataTypeForSwigluMxQuant);
} // namespace ops