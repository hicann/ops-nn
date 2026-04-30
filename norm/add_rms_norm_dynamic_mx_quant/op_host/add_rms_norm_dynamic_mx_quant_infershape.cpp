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
 * \file add_rms_norm_dynamic_mx_quant_infershape.cpp
 * \brief
 */
#include <algorithm>
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

static constexpr int X1_IDX = 0;
static constexpr int GAMMA_IDX = 2;
static constexpr int BETA_IDX = 3;

static constexpr int Y_IDX = 0;
static constexpr int X_IDX = 1;
static constexpr int MXSCALE_IDX = 2;
static constexpr int RSTD_IDX = 3;
static constexpr int ATTR_OUTPUT_RSTD_IDX = 4;

static constexpr int ATTR_INDEX_OF_DST_TYPE = 3;
static constexpr int64_t MX_BLOCK_SIZE = 32;
static constexpr int64_t ALIGN_NUM = 2;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;

using namespace ge;
using namespace Ops::Base;

namespace ops {

static const std::initializer_list<ge::DataType> OUT_TYPE_LIST = {
    DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2};

static ge::graphStatus InferShape4AddRmsNormDynamicMxQuant(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4AddRmsNormDynamicMxQuant");

    const gert::Shape* x1Shape = context->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* gammaShape = context->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);

    // Output shapes
    gert::Shape* yShape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* mxscaleShape = context->GetOutputShape(MXSCALE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxscaleShape);
    gert::Shape* xShape = context->GetOutputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // Attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* output_rstd = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_RSTD_IDX);
    bool rstd_enable = false;
    if (output_rstd != nullptr) {
        rstd_enable = *output_rstd;
    }

    // y and x output shapes = x1 shape
    *yShape = *x1Shape;
    *xShape = *x1Shape;

    // Handle unknown rank
    if (IsUnknownRank(*x1Shape) || IsUnknownRank(*gammaShape)) {
        SetUnknownRank(*mxscaleShape);
        // rstd (optional)
        if (rstd_enable) {
            gert::Shape* rstdShape = context->GetOutputShape(RSTD_IDX);
            OP_CHECK_NULL_WITH_CONTEXT(context, rstdShape);
            SetUnknownRank(*rstdShape);
        }
        OP_LOGI(context, "End InferShape with unknown rank.");
        return GRAPH_SUCCESS;
    }

    size_t x1DimNum = x1Shape->GetDimNum();
    size_t gammaDimNum = gammaShape->GetDimNum();

    // mxscale shape: same batch dims as x, last dim = CeilDiv(CeilDiv(R, 32), 2)
    // where R is the last gammaDimNum dims of x
    size_t dim = x1Shape->GetDimNum() - 1;
    int64_t dimSize = 0;
    if (x1Shape->GetDim(dim) == UNKNOWN_DIM_VALUE_) {
        dimSize = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize = (x1Shape->GetDim(dim) + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
        dimSize = (dimSize + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    *mxscaleShape = *x1Shape;
    mxscaleShape->SetDim(dim, dimSize);
    mxscaleShape->AppendDim(ALIGN_NUM);

    // rstd shape (optional): same dims as x1, trailing gamma dims set to 1
    if (rstd_enable) {
        gert::Shape* rstdShape = context->GetOutputShape(RSTD_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, rstdShape);
        rstdShape->SetDimNum(x1DimNum);
        for (size_t i = 0; i < x1DimNum; i++) {
            if (i < x1DimNum - gammaDimNum) {
                rstdShape->SetDim(i, x1Shape->GetDim(i));
            } else {
                rstdShape->SetDim(i, 1);
            }
        }
    }

    OP_LOGI(context, "End InferShape4AddRmsNormDynamicMxQuant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddRmsNormDynamicMxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4AddRmsNormDynamicMxQuant");
    ge::DataType yDtype = ge::DT_FLOAT4_E2M1;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* pDstDtype = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_OF_DST_TYPE);
        if (pDstDtype != nullptr) {
            yDtype = static_cast<ge::DataType>(*pDstDtype);
            OP_CHECK_IF(
                std::find(OUT_TYPE_LIST.begin(), OUT_TYPE_LIST.end(), yDtype) == OUT_TYPE_LIST.end(),
                OP_LOGE(context,
                        "attr dst_type only support 35(float8_e5m2), 36(float8_e4m3fn), 40(float4_e2m1), 41(float4_e1m2)"),
                return ge::GRAPH_FAILED);
        }
        const bool* output_rstd = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_RSTD_IDX);
        bool rstd_enable = false;
        if (output_rstd != nullptr) {
            rstd_enable = *output_rstd;
        }
        if (rstd_enable) {
            context->SetOutputDataType(RSTD_IDX, ge::DT_FLOAT);
        }
    }
    context->SetOutputDataType(Y_IDX, yDtype);
    context->SetOutputDataType(X_IDX, context->GetInputDataType(X1_IDX));
    context->SetOutputDataType(MXSCALE_IDX, ge::DT_FLOAT8_E8M0);

    OP_LOGD(context, "End InferDataType4AddRmsNormDynamicMxQuant");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormDynamicMxQuant)
    .InferShape(InferShape4AddRmsNormDynamicMxQuant)
    .InferDataType(InferDataType4AddRmsNormDynamicMxQuant);

} // namespace ops
