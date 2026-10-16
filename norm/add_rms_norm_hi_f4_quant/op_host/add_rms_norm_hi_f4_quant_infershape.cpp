/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file add_rms_norm_hi_f4_quant_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

static constexpr int X1_IDX = 0;
static constexpr int GAMMA_IDX = 2;
static constexpr int BETA_IDX = 3;

static constexpr int Y_IDX = 0;
static constexpr int X_IDX = 1;
static constexpr int SCALE_IDX = 2;
static constexpr int RSTD_IDX = 3;

static constexpr int ATTR_OUTPUT_RSTD_IDX = 1;
static constexpr int64_t MX_BLOCK_SIZE = 64;
static constexpr int64_t ALIGN_NUM = 1;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;

using namespace ge;
using namespace Ops::Base;

namespace ops {

static ge::graphStatus InferShape4AddRmsNormHiF4Quant(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4AddRmsNormHiF4Quant");

    const gert::Shape* x1Shape = context->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* gammaShape = context->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);

    gert::Shape* yShape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* scaleShape = context->GetOutputShape(SCALE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    gert::Shape* xShape = context->GetOutputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* output_rstd = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_RSTD_IDX);
    bool rstd_enable = false;
    if (output_rstd != nullptr) {
        rstd_enable = *output_rstd;
    }

    *yShape = *x1Shape;
    *xShape = *x1Shape;

    if (IsUnknownRank(*x1Shape) || IsUnknownRank(*gammaShape)) {
        SetUnknownRank(*scaleShape);
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

    size_t dim = x1Shape->GetDimNum() - 1;
    int64_t dimSize = 0;
    if (x1Shape->GetDim(dim) == UNKNOWN_DIM_VALUE_) {
        dimSize = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize = (x1Shape->GetDim(dim) + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
        dimSize = (dimSize + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    *scaleShape = *x1Shape;
    scaleShape->SetDim(dim, dimSize);
    scaleShape->AppendDim(ALIGN_NUM);

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

    OP_LOGI(context, "End InferShape4AddRmsNormHiF4Quant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddRmsNormHiF4Quant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4AddRmsNormHiF4Quant");

    context->SetOutputDataType(Y_IDX, Ops::NN::DT_HIFLOAT4);
    context->SetOutputDataType(X_IDX, context->GetInputDataType(X1_IDX));
    context->SetOutputDataType(SCALE_IDX, Ops::NN::DT_HIFLOAT4_SCALE);

    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* output_rstd = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_RSTD_IDX);
        bool rstd_enable = false;
        if (output_rstd != nullptr) {
            rstd_enable = *output_rstd;
        }
        if (rstd_enable) {
            context->SetOutputDataType(RSTD_IDX, Ops::NN::DT_FLOAT);
        }
    }

    OP_LOGD(context, "End InferDataType4AddRmsNormHiF4Quant");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormHiF4Quant)
    .InferShape(InferShape4AddRmsNormHiF4Quant)
    .InferDataType(InferDataType4AddRmsNormHiF4Quant);

} // namespace ops
