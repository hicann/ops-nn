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
 * \file layer_norm_quant_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t INPUT_GAMMA_IDX = 1;
constexpr size_t INPUT_BETA_IDX = 2;
constexpr size_t INPUT_SCALE_IDX = 3;
constexpr size_t INPUT_OFFSET_IDX = 4;
constexpr size_t OUTPUT_RESULT_IDX = 0;

static graphStatus InferShape4LayerNormQuant(gert::InferShapeContext* context) {
    OP_LOGD(context, "Begin to do LayerNormQuantInferShape");

    const gert::Shape* gammaShape = context->GetInputShape(INPUT_GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    const gert::Shape* betaShape = context->GetInputShape(INPUT_BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShape);
    const gert::Shape* scaleShape = context->GetInputShape(INPUT_SCALE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    const gert::Shape* offsetShape = context->GetInputShape(INPUT_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetShape);

    if (gammaShape->GetDimNum() != betaShape->GetDimNum()) {
        OP_LOGE(context, "intput_gamma_shape are not equal to intput_beta_shape");
        return GRAPH_FAILED;
    }
    for (size_t i = 0; i < gammaShape->GetDimNum(); i++) {
        if (gammaShape->GetDim(i) != betaShape->GetDim(i)) {
            OP_LOGE(context, "intput_gamma_shape dim are not equal to intput_beta_shape dim.");
            return GRAPH_FAILED;
        }
    }

    if (scaleShape->GetShapeSize() != 1) {
        OP_LOGE(context, "InputScale shape is not [1]");
        return GRAPH_FAILED;
    }
    if (offsetShape->GetShapeSize() != 1) {
        OP_LOGE(context, "InputOffset shape is not [1]");
        return GRAPH_FAILED;
    }

    const gert::Shape* xShape = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* resultShape = context->GetOutputShape(OUTPUT_RESULT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, resultShape);

    *resultShape = *xShape;
    OP_LOGD(context, "End to do InferShape4LayerNormQuant.");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4LayerNormQuant(gert::InferDataTypeContext* context) {
    OP_LOGD(context, "Begin to do InferDataType4LayerNormQuant");
    context->SetOutputDataType(OUTPUT_RESULT_IDX, DT_INT8);
    OP_LOGD(context, "End to do InferDataType4LayerNormQuant");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LayerNormQuant)
    .InferShape(InferShape4LayerNormQuant)
    .InferDataType(InferDataType4LayerNormQuant);
} // namespace ops
