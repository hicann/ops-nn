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
 * \file rotate_quant_infershape.cpp
 * \brief RotateQuant算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "error_util.h"
#include "log/log.h"
#include "util/math_util.h"

namespace ops {

constexpr int64_t X_INDEX = 0;
constexpr int64_t ROT_INDEX = 1;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t SCALE_INDEX = 1;
constexpr int32_t DIMENSION_2 = 2;

constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr uint32_t OUTPUT_NUM_ROTATE_QUANT = 2;
constexpr uint32_t INPUT_NUM_ROTATE_QUANT = 2;
const std::initializer_list<ge::DataType> OUT_TYPE_LIST = {ge::DT_INT8, ge::DT_INT4};

static ge::graphStatus CheckComputeNodeNum(gert::InferShapeContext* context)
{
    // 检查输入数量
    if (context->GetComputeNodeInputNum() != INPUT_NUM_ROTATE_QUANT) {
        OP_LOGE(
            context, "Input num must be %u, but got %zu", INPUT_NUM_ROTATE_QUANT, context->GetComputeNodeInputNum());
        return ge::GRAPH_FAILED;
    }

    // 检查输出数量
    if (context->GetComputeNodeOutputNum() != OUTPUT_NUM_ROTATE_QUANT) {
        OP_LOGE(
            context, "Output num must be %u, but got %zu", OUTPUT_NUM_ROTATE_QUANT, context->GetComputeNodeOutputNum());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeCheck(gert::InferShapeContext* context)
{
    if (context == nullptr || CheckComputeNodeNum(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < OUTPUT_NUM_ROTATE_QUANT; ++i) {
        if (context->GetOutputShape(i) == nullptr) {
            OP_LOGE(context, "Output shape %u is nullptr", i);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    const gert::Shape* rotShape = context->GetInputShape(ROT_INDEX);

    if (xShape == nullptr || rotShape == nullptr) {
        OP_LOGE(context, "Input shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    size_t xDimNum = xShape->GetDimNum();
    if (xDimNum != DIMENSION_2) {
        OP_LOGE(context, "x shape dimension must be 2, but got %zu", xDimNum);
        return ge::GRAPH_FAILED;
    }

    size_t rotDimNum = rotShape->GetDimNum();
    if (rotDimNum != DIMENSION_2) {
        OP_LOGE(context, "rot shape dimension must be 2, but got %zu", rotDimNum);
        return ge::GRAPH_FAILED;
    }

    int64_t xN = xShape->GetDim(1);
    int64_t rotK = rotShape->GetDim(0);
    int64_t rotN = rotShape->GetDim(1);
    if (rotK != rotN) {
        OP_LOGE(context, "rot must be a square matrix, but got shape [%ld, %ld]", rotK, rotN);
        return ge::GRAPH_FAILED;
    }

    if (rotK == 0) {
        OP_LOGE(context, "rot K dimension cannot be zero");
        return ge::GRAPH_FAILED;
    }
    if (xN % rotK != 0) {
        OP_LOGE(context, "x N dimension (%ld) must be divisible by rot K dimension (%ld)", xN, rotK);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RotateQuantInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape RotateQuant");
    if (InferShapeCheck(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputShape(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    gert::Shape* scaleShape = context->GetOutputShape(SCALE_INDEX);

    yShape->SetDimNum(DIMENSION_2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, xShape->GetDim(1));
    scaleShape->SetDimNum(1);
    scaleShape->SetDim(0, xShape->GetDim(0));
    OP_LOGD(context->GetNodeName(), "End to do InferShape RotateQuant");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RotateQuantInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "RotateQuantInferDataType begin");

    ge::DataType yDtype = ge::DT_INT8;

    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* pDstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_DST_TYPE);
        if (pDstDtype != nullptr) {
            int32_t dstDtype = *pDstDtype;
            yDtype = static_cast<ge::DataType>(dstDtype);

            OP_CHECK_IF(
                std::find(OUT_TYPE_LIST.begin(), OUT_TYPE_LIST.end(), yDtype) == OUT_TYPE_LIST.end(),
                OP_LOGE(context, "attr dst_type only support int4, int8"), return ge::GRAPH_FAILED);
        }
    }

    context->SetOutputDataType(Y_INDEX, yDtype);
    context->SetOutputDataType(SCALE_INDEX, ge::DT_FLOAT);

    OP_LOGD(context, "RotateQuantInferDataType end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotateQuant).InferShape(RotateQuantInferShape).InferDataType(RotateQuantInferDataType);

} // namespace ops
