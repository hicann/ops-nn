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
 * \file rms_norm_quant_v3_infershape.cpp
 * \brief
 */
#include <algorithm>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

static constexpr int INPUT_X_IDX = 0;
static constexpr int INPUT_GAMMA_IDX = 1;
static constexpr int OUTPUT_Y1_IDX = 0;
static constexpr int OUTPUT_Y2_IDX = 1;
static constexpr int OUTPUT_RSTD_IDX = 2;
static constexpr int ATTR_INDEX_OF_DST_TYPE = 2;
static constexpr int ATTR_INDEX_OF_OUTPUT_RSTD = 3;

using namespace ge;
using namespace Ops::Base;

namespace ops {
static const std::initializer_list<ge::DataType> OUT_TYPE_LIST = {DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                                                                  DT_FLOAT8_E4M3FN};

static ge::graphStatus InferShape4RmsNormQuantV3(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4RmsNormQuantV3");

    const gert::Shape* xShape = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* gammaShape = context->GetInputShape(INPUT_GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);

    gert::Shape* y1Shape = context->GetOutputShape(OUTPUT_Y1_IDX);
    gert::Shape* y2Shape = context->GetOutputShape(OUTPUT_Y2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);

    *y1Shape = *xShape;
    *y2Shape = *xShape;

    // rstd shape: A dims preserved, R dims set to 1
    auto* attrs = context->GetAttrs();
    const bool* outputRstdPtr = (attrs != nullptr) ? attrs->GetAttrPointer<bool>(ATTR_INDEX_OF_OUTPUT_RSTD) : nullptr;
    bool rstdEnable = (outputRstdPtr != nullptr) ? *outputRstdPtr : false;

    gert::Shape* rstdShape = context->GetOutputShape(OUTPUT_RSTD_IDX);
    if (rstdEnable) {
        OP_CHECK_NULL_WITH_CONTEXT(context, rstdShape);
    }

    if (IsUnknownRank(*xShape) || IsUnknownRank(*gammaShape)) {
        SetUnknownRank(*y1Shape);
        SetUnknownRank(*y2Shape);
        if (rstdEnable) {
            SetUnknownRank(*rstdShape);
        }
        OP_LOGI(context, "End InferShape with unknown rank.");
        return GRAPH_SUCCESS;
    }

    size_t xDimNum = xShape->GetDimNum();
    size_t gammaDimNum = gammaShape->GetDimNum();
    if (rstdEnable) {
        rstdShape->SetDimNum(xDimNum);
        for (size_t i = 0; i < xDimNum; i++) {
            if (i < xDimNum - gammaDimNum) {
                rstdShape->SetDim(i, xShape->GetDim(i));
            } else {
                rstdShape->SetDim(i, 1);
            }
        }
    }

    OP_LOGD(context, "End to do InferShape4RmsNormQuantV3");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4RmsNormQuantV3(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4RmsNormQuantV3");
    ge::DataType yDtype = ge::DT_INT8;
    bool rstdEnable = false;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* pDstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
        if (pDstDtype != nullptr) {
            int32_t dstDtype = *pDstDtype;
            yDtype = static_cast<ge::DataType>(dstDtype);
            OP_CHECK_IF(std::find(OUT_TYPE_LIST.begin(), OUT_TYPE_LIST.end(), yDtype) == OUT_TYPE_LIST.end(),
                        OP_LOGE(context, "attr dst_type only support int8, int4, hifloat8, float8_e5m2, float8_e4m3fn"),
                        return ge::GRAPH_FAILED);
        }
        const bool* outputRstdPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_OF_OUTPUT_RSTD);
        if (outputRstdPtr != nullptr) {
            rstdEnable = *outputRstdPtr;
        }
    }
    context->SetOutputDataType(OUTPUT_Y1_IDX, yDtype);
    context->SetOutputDataType(OUTPUT_Y2_IDX, yDtype);
    if (rstdEnable) {
        context->SetOutputDataType(OUTPUT_RSTD_IDX, ge::DT_FLOAT);
    }
    OP_LOGD(context, "End to do InferDataType4RmsNormQuantV3");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormQuantV3).InferShape(InferShape4RmsNormQuantV3).InferDataType(InferDataType4RmsNormQuantV3);
} // namespace ops
