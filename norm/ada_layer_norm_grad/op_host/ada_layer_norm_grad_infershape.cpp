/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {

static const int64_t INPUT_DY_INDEX = 0;
static const int64_t INPUT_X_INDEX = 1;
static const int64_t INPUT_RSTD_INDEX = 2;
static const int64_t INPUT_MEAN_INDEX = 3;
static const int64_t INPUT_SCALE_INDEX = 4;
static const int64_t INPUT_GAMMA_INDEX = 5;
static const int64_t INPUT_BETA_INDEX = 6;
static constexpr int OUTPUT_PD_X_INDEX = 0;
static constexpr int OUTPUT_PD_SCALE_INDEX = 1;
static constexpr int OUTPUT_PD_SHIFT_INDEX = 2;
static constexpr int OUTPUT_PD_GAMMA_INDEX = 3;
static constexpr int OUTPUT_PD_BETA_INDEX = 4;

static ge::graphStatus InferShapeAdaLayerNormGrad(gert::InferShapeContext* context) {
    if (context == nullptr) {
        OP_LOGE("AdaLayerNormGrad", "InferShapeContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* dy_shape = context->GetInputShape(INPUT_DY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dy_shape);

    const gert::Shape* x_shape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    const gert::Shape* rstd_shape = context->GetInputShape(INPUT_RSTD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);

    const gert::Shape* mean_shape = context->GetInputShape(INPUT_MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);

    const gert::Shape* scale_shape = context->GetInputShape(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale_shape);

    const gert::Shape* gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);

    const gert::Shape* beta_shape = context->GetInputShape(INPUT_BETA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, beta_shape);

    gert::Shape* output_pd_x_shape = context->GetOutputShape(OUTPUT_PD_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_pd_x_shape);

    gert::Shape* output_pd_scale_shape = context->GetOutputShape(OUTPUT_PD_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_pd_scale_shape);

    gert::Shape* output_pd_shift_shape = context->GetOutputShape(OUTPUT_PD_SHIFT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_pd_shift_shape);

    gert::Shape* output_pd_gamma_shape = context->GetOutputShape(OUTPUT_PD_GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_pd_gamma_shape);

    gert::Shape* output_pd_beta_shape = context->GetOutputShape(OUTPUT_PD_BETA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_pd_beta_shape);

    if (Ops::Base::IsUnknownRank(*dy_shape)) {
        Ops::Base::SetUnknownRank(*output_pd_x_shape);
    } else {
        *output_pd_x_shape = *dy_shape;
    }

    if (Ops::Base::IsUnknownRank(*scale_shape)) {
        Ops::Base::SetUnknownRank(*output_pd_scale_shape);
        Ops::Base::SetUnknownRank(*output_pd_shift_shape);
    } else {
        *output_pd_scale_shape = *scale_shape;
        *output_pd_shift_shape = *scale_shape;
    }

    if (Ops::Base::IsUnknownRank(*gamma_shape)) {
        Ops::Base::SetUnknownRank(*output_pd_gamma_shape);
        Ops::Base::SetUnknownRank(*output_pd_beta_shape);
    } else {
        *output_pd_gamma_shape = *gamma_shape;
        *output_pd_beta_shape = *gamma_shape;
    }

    OP_LOGD(context, "InferShapeAdaLayerNormGrad finished.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAdaLayerNormGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeAdaLayerNormGrad");
    context->SetOutputDataType(OUTPUT_PD_X_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    context->SetOutputDataType(OUTPUT_PD_SCALE_INDEX, context->GetInputDataType(INPUT_SCALE_INDEX));
    context->SetOutputDataType(OUTPUT_PD_SHIFT_INDEX, context->GetInputDataType(INPUT_SCALE_INDEX));
    context->SetOutputDataType(OUTPUT_PD_GAMMA_INDEX, context->GetInputDataType(INPUT_GAMMA_INDEX));
    context->SetOutputDataType(OUTPUT_PD_BETA_INDEX, context->GetInputDataType(INPUT_BETA_INDEX));
    OP_LOGD(context, "End to do InferDataTypeAdaLayerNormGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaLayerNormGrad).InferShape(InferShapeAdaLayerNormGrad).InferDataType(InferDataTypeAdaLayerNormGrad);
} // namespace ops