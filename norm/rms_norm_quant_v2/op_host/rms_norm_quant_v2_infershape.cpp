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
 * \file rms_norm_quant_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

static constexpr int INPUT_X_IDX = 0;
static constexpr int INPUT_SCALE2_IDX = 3;
static constexpr int OUTPUT_Y1_IDX = 0;
static constexpr int OUTPUT_Y2_IDX = 1;
static constexpr int ATTR_INDEX_OF_DST_TYPE = 2;

using namespace ge;

namespace ops {
static const std::initializer_list<ge::DataType> OUT_TYPE_LIST = {
    DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN};
static ge::graphStatus InferShape4RmsNormQuantV2(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4RmsNormQuantV2");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* y1Shape = context->GetOutputShape(OUTPUT_Y1_IDX);
    gert::Shape* y2Shape = context->GetOutputShape(OUTPUT_Y2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    *y1Shape = *xShape;
    *y2Shape = *xShape;

    OP_LOGD(context, "End to do InferShape4RmsNormQuantV2");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4RmsNormQuantV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4RmsNormQuantV2");
    ge::DataType yDtype = ge::DT_INT8;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* pDstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
        if (pDstDtype != nullptr) {
            int32_t dstDtype = *pDstDtype;
            yDtype = static_cast<ge::DataType>(dstDtype);
            OP_CHECK_IF(
                std::find(OUT_TYPE_LIST.begin(), OUT_TYPE_LIST.end(), yDtype) == OUT_TYPE_LIST.end(),
                OP_LOGE(context,
                    "attr dst_type only support int8, int4, hifloat8, float8_e5m2, float8_e4m3fn"),
                return ge::GRAPH_FAILED);
        }
    }
    context->SetOutputDataType(OUTPUT_Y1_IDX, yDtype);
    context->SetOutputDataType(OUTPUT_Y2_IDX, yDtype);
    OP_LOGD(context, "End to do InferDataType4RmsNormQuantV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormQuantV2).InferShape(InferShape4RmsNormQuantV2).InferDataType(InferDataType4RmsNormQuantV2);
} // namespace ops
