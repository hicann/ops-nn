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
 * \file layer_norm_grad_v3_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {

static const int64_t INPUT_DY_INDEX = 0;
static const int64_t INPUT_X_INDEX = 1;
static const int64_t INPUT_RSTD_INDEX = 2;
static const int64_t INPUT_GAMMA_INDEX = 4;
static constexpr int OUTPUT_PD_X_INDEX = 0;
static constexpr int OUTOUT_PD_GAMMA_INDEX = 1;
static constexpr int OUTOUT_PD_BETA_INDEX = 2;

static ge::graphStatus InferShapeLayerNormGradV3(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("LayerNormGradV3", "InferShapeContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Begin to do InferShapeLayerNormGradV3.");

    const gert::Shape* dy_shape = context->GetInputShape(INPUT_DY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dy_shape);

    const gert::Shape* x_shape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    const gert::Shape* rstd_shape = context->GetInputShape(INPUT_RSTD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);

    const gert::Shape* gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);

    gert::Shape* output_pd_x_shape = context->GetOutputShape(OUTPUT_PD_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_pd_x_shape);

    gert::Shape* output_pd_gamma_shape = context->GetOutputShape(OUTOUT_PD_GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_pd_gamma_shape);

    gert::Shape* output_pd_beta_shape = context->GetOutputShape(OUTOUT_PD_BETA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_pd_beta_shape);

    if (Ops::Base::IsUnknownRank(*dy_shape)) {
        Ops::Base::SetUnknownRank(*output_pd_x_shape);
    } else {
        *output_pd_x_shape = *dy_shape;
    }

    if (Ops::Base::IsUnknownRank(*gamma_shape)) {
        Ops::Base::SetUnknownRank(*output_pd_gamma_shape);
        Ops::Base::SetUnknownRank(*output_pd_beta_shape);
    } else {
        *output_pd_gamma_shape = *gamma_shape;
        *output_pd_beta_shape = *gamma_shape;
    }
    OP_LOGD(context, "End to do InferShapeLayerNormGradV3.");
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeLayerNormGradV3(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeLayerNormGradV3");
    context->SetOutputDataType(OUTPUT_PD_X_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    context->SetOutputDataType(OUTOUT_PD_GAMMA_INDEX, context->GetInputDataType(INPUT_GAMMA_INDEX));
    context->SetOutputDataType(OUTOUT_PD_BETA_INDEX, context->GetInputDataType(INPUT_GAMMA_INDEX));
    OP_LOGD(context, "End to do InferDataType4GridSampler3DGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LayerNormGradV3).InferShape(InferShapeLayerNormGradV3).InferDataType(InferDataTypeLayerNormGradV3);
} // namespace ops
