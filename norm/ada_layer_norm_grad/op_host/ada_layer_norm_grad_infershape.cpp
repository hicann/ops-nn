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
 * \file ada_layer_norm_grad_infershape.cpp
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
static const int64_t INPUT_MEAN_INDEX = 3;
static const int64_t INPUT_SCALE_INDEX = 4;
static const int64_t INPUT_GAMMA_INDEX = 5;
static const int64_t INPUT_BETA_INDEX = 6;
static constexpr int OUTPUT_PD_X_INDEX = 0;
static constexpr int OUTPUT_PD_SCALE_INDEX = 1;
static constexpr int OUTPUT_PD_SHIFT_INDEX = 2;
static constexpr int OUTPUT_PD_GAMMA_INDEX = 3;
static constexpr int OUTPUT_PD_BETA_INDEX = 4;


static bool CheckAllNotNull(const std::vector<const gert::Shape*>& shapes) {
    for (auto shape : shapes) {
        if (shape == nullptr) {
            return false;
        }
    }
    return true;
}

static void SetOutputShapeFromInput(const gert::Shape* input_shape,
                                   gert::Shape* output_shape) {
    if (Ops::Base::IsUnknownRank(*input_shape)) {
        Ops::Base::SetUnknownRank(*output_shape);
    } else {
        *output_shape = *input_shape;
    }
}

static void SetOutputShapeFromInput(const gert::Shape* input_shape,
                                   gert::Shape* output_shape1,
                                   gert::Shape* output_shape2) {
    if (Ops::Base::IsUnknownRank(*input_shape)) {
        Ops::Base::SetUnknownRank(*output_shape1);
        Ops::Base::SetUnknownRank(*output_shape2);
    } else {
        *output_shape1 = *input_shape;
        *output_shape2 = *input_shape;
    }
}

static ge::graphStatus InferShapeAdaLayerNormGrad(gert::InferShapeContext* context) {
    if (context == nullptr) {
        OP_LOGE("AdaLayerNormGrad", "InferShapeContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Begin to do InferShapeAdaLayerNormGrad.");

    // 获取输入形状
    const gert::Shape* dy_shape = context->GetInputShape(INPUT_DY_INDEX);
    const gert::Shape* x_shape = context->GetInputShape(INPUT_X_INDEX);
    const gert::Shape* rstd_shape = context->GetInputShape(INPUT_RSTD_INDEX);
    const gert::Shape* mean_shape = context->GetInputShape(INPUT_MEAN_INDEX);
    const gert::Shape* scale_shape = context->GetInputShape(INPUT_SCALE_INDEX);
    const gert::Shape* gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::Shape* beta_shape = context->GetInputShape(INPUT_BETA_INDEX);
    
    // 空指针检查
    if (!CheckAllNotNull({dy_shape, x_shape, rstd_shape, mean_shape, scale_shape, gamma_shape, beta_shape})) {
        return ge::GRAPH_FAILED;
    }

    // 获取输出形状
    gert::Shape* output_pd_x_shape = context->GetOutputShape(OUTPUT_PD_X_INDEX);
    gert::Shape* output_pd_scale_shape = context->GetOutputShape(OUTPUT_PD_SCALE_INDEX);
    gert::Shape* output_pd_shift_shape = context->GetOutputShape(OUTPUT_PD_SHIFT_INDEX);
    gert::Shape* output_pd_gamma_shape = context->GetOutputShape(OUTPUT_PD_GAMMA_INDEX);
    gert::Shape* output_pd_beta_shape = context->GetOutputShape(OUTPUT_PD_BETA_INDEX);
    
    if (!CheckAllNotNull({output_pd_x_shape, output_pd_scale_shape, output_pd_shift_shape, 
                                   output_pd_gamma_shape, output_pd_beta_shape})) {
        return ge::GRAPH_FAILED;
    }

    // 设置输出形状
    // pd_x: 与 dy 相同
    SetOutputShapeFromInput(dy_shape, output_pd_x_shape);
    // pd_scale: 与 scale 相同
    SetOutputShapeFromInput(scale_shape, output_pd_scale_shape);
    // pd_shift: 与 scale 相同（shift的梯度形状与scale相同）
    SetOutputShapeFromInput(scale_shape, output_pd_shift_shape);
    // pd_gamma/pd_beta: 与 gamma 相同
    SetOutputShapeFromInput(gamma_shape, output_pd_gamma_shape);
    SetOutputShapeFromInput(gamma_shape, output_pd_beta_shape);

    OP_LOGD(context, "End to do InferShapeAdaLayerNormGrad.");
    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus InferDataTypeAdaLayerNormGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeAdaLayerNormGrad");
    context->SetOutputDataType(OUTPUT_PD_X_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    context->SetOutputDataType(OUTPUT_PD_SCALE_INDEX, context->GetInputDataType(INPUT_SCALE_INDEX));
    // pd_shift 的 dtype 与 scale 相同
    context->SetOutputDataType(OUTPUT_PD_SHIFT_INDEX, context->GetInputDataType(INPUT_SCALE_INDEX));
    context->SetOutputDataType(OUTPUT_PD_GAMMA_INDEX, context->GetInputDataType(INPUT_GAMMA_INDEX));
    context->SetOutputDataType(OUTPUT_PD_BETA_INDEX, context->GetInputDataType(INPUT_BETA_INDEX));
    OP_LOGD(context, "End to do InferDataTypeAdaLayerNormGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaLayerNormGrad).InferShape(InferShapeAdaLayerNormGrad).InferDataType(InferDataTypeAdaLayerNormGrad);
} // namespace ops