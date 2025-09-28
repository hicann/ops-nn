/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gelu_quant_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static const size_t DST_TYPE_ATTR_INDEX = 2;
static const int32_t DTYPE_INT8 = 2;
static const int32_t DTYPE_HIFLOAT8 = 34;
static const int32_t DTYPE_FLOAT8_E5M2 = 35;
static const int32_t DTYPE_FLOAT8_E4M3 = 36;

static ge::graphStatus InferShapeCheck(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    if ((context->GetInputShape(0) == nullptr)) {
        return GRAPH_FAILED;
    }

    if ((context->GetOutputShape(0) == nullptr)) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus GeluQuantInferShape(gert::InferShapeContext* context)
{
    if (InferShapeCheck(context) == GRAPH_FAILED) {
        OP_LOGE("GeluQuant", "[GeluQuant] InferShapeCheck failed.");
        return GRAPH_FAILED;
    }

    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    gert::Shape* outScaleShape = context->GetOutputShape(1);
    *yShape = *xShape;
    outScaleShape->SetDimNum(xShape->GetDimNum() - 1);
    for (size_t i = 0; i < xShape->GetDimNum() - 1; ++i) {
        outScaleShape->SetDim(i, xShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus GeluQuantInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("GeluQuant", "[GeluQuant] GeluQuantInferDataType got context is nullptr.");
        return GRAPH_FAILED;
    }
    OP_LOGD(context, "GeluQuantInferDataType begin");
    ge::DataType inferDstType = ge::DT_INT8;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* pDstDtype = attrs->GetAttrPointer<int32_t>(DST_TYPE_ATTR_INDEX);
        if (pDstDtype != nullptr) {
            int32_t dstDtype = *pDstDtype;
            OP_CHECK_IF(
                dstDtype != DTYPE_INT8 && dstDtype != DTYPE_HIFLOAT8 && dstDtype != DTYPE_FLOAT8_E5M2 &&
                    dstDtype != DTYPE_FLOAT8_E4M3,
                OP_LOGE(
                    "GeluQuant",
                    "attr dst_type only support 2(int8), 34(hifloat8), 35(float8_e5m2), 36(float8_e4m3fn)"),
                return ge::GRAPH_FAILED);
            inferDstType = static_cast<ge::DataType>(dstDtype);
        }
    }

    context->SetOutputDataType(0, inferDstType);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(GeluQuant).InferShape(GeluQuantInferShape).InferDataType(GeluQuantInferDataType);
} // namespace ops
