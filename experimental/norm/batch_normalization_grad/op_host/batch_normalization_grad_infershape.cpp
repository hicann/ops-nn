/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_normalization_grad_infershape.cpp
 * \brief BatchNormalizationGrad算子的shape推理和数据类型推理实现
 *
 * 本文件提供输出张量shape和数据类型推理逻辑，并对输入做合法性校验：
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_GRAD_OUTPUT = 0;
static constexpr int64_t IDX_INPUT       = 1;
static constexpr int64_t IDX_WEIGHT      = 2;
static constexpr int64_t IDX_BIAS        = 3;
static constexpr int64_t IDX_SAVE_MEAN   = 4;
static constexpr int64_t IDX_SAVE_INVSTD = 5;
static constexpr int64_t IDX_OUT_GRAD_INPUT   = 0;
static constexpr int64_t IDX_OUT_GRAD_WEIGHT  = 1;
static constexpr int64_t IDX_OUT_GRAD_BIAS    = 2;

static ge::graphStatus InferShapeBatchNormalizationGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeBatchNormalizationGrad");

    // 1. 获取全部 6 个输入 shape
    const gert::Shape* gradOutputShape = context->GetInputShape(IDX_GRAD_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradOutputShape);

    const gert::Shape* inputShape = context->GetInputShape(IDX_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const gert::Shape* weightShape = context->GetInputShape(IDX_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);

    const gert::Shape* biasShape = context->GetInputShape(IDX_BIAS);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasShape);

    const gert::Shape* saveMeanShape = context->GetInputShape(IDX_SAVE_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, saveMeanShape);

    const gert::Shape* saveInvstdShape = context->GetInputShape(IDX_SAVE_INVSTD);
    OP_CHECK_NULL_WITH_CONTEXT(context, saveInvstdShape);

    // 2. 校验 grad_output 与 input shape 一致
    if (*gradOutputShape != *inputShape) {
        OP_LOGE(context, "grad_output and input must have the same shape");
        return GRAPH_FAILED;
    }

    // 3. 校验 weight 是一维且长度 == input 的 channel 维度
    if (weightShape->GetDimNum() != 1) {
        OP_LOGE(context, "weight must be 1D, got %zu", weightShape->GetDimNum());
        return GRAPH_FAILED;
    }
    int64_t numChannels = inputShape->GetDim(1);
    if (weightShape->GetDim(0) != numChannels) {
        OP_LOGE(context, "weight size(%ld) != input channel dim(%ld)",
                weightShape->GetDim(0), numChannels);
        return GRAPH_FAILED;
    }

    // 4. 校验 bias / save_mean / save_invstd 与 weight shape 一致
    if (*biasShape != *weightShape) {
        OP_LOGE(context, "bias must have the same shape as weight");
        return GRAPH_FAILED;
    }
    if (*saveMeanShape != *weightShape) {
        OP_LOGE(context, "save_mean must have the same shape as weight");
        return GRAPH_FAILED;
    }
    if (*saveInvstdShape != *weightShape) {
        OP_LOGE(context, "save_invstd must have the same shape as weight");
        return GRAPH_FAILED;
    }

    // 5. 设置输出 shape
    gert::Shape* gradInputShape = context->GetOutputShape(IDX_OUT_GRAD_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradInputShape);
    *gradInputShape = *inputShape;

    gert::Shape* gradWeightShape = context->GetOutputShape(IDX_OUT_GRAD_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradWeightShape);
    *gradWeightShape = *weightShape;

    gert::Shape* gradBiasShape = context->GetOutputShape(IDX_OUT_GRAD_BIAS);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradBiasShape);
    *gradBiasShape = *weightShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeBatchNormalizationGrad");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBatchNormalizationGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeBatchNormalizationGrad");

    // 获取 grad_output（首个输入）的数据类型，传播到所有输出
    ge::DataType inputDataType = context->GetInputDataType(IDX_GRAD_OUTPUT);
    context->SetOutputDataType(IDX_OUT_GRAD_INPUT, inputDataType);
    context->SetOutputDataType(IDX_OUT_GRAD_WEIGHT, inputDataType);
    context->SetOutputDataType(IDX_OUT_GRAD_BIAS, inputDataType);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeBatchNormalizationGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchNormalizationGrad).InferShape(InferShapeBatchNormalizationGrad).InferDataType(InferDataTypeBatchNormalizationGrad);
} // namespace ops
