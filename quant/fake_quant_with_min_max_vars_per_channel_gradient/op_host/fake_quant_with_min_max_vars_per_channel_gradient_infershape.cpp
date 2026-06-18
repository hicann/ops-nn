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
 * \file fake_quant_with_min_max_vars_per_channel_gradient_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_IDX_GRADIENTS = 0;
static constexpr size_t INPUT_IDX_X = 1;
static constexpr size_t INPUT_IDX_MIN = 2;
static constexpr size_t INPUT_IDX_MAX = 3;
static constexpr size_t OUTPUT_IDX_BPX = 0;
static constexpr size_t OUTPUT_IDX_BPMIN = 1;
static constexpr size_t OUTPUT_IDX_BPMAX = 2;

static ge::graphStatus FakeQuantWithMinMaxVarsPerChannelGradientInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do FakeQuantWithMinMaxVarsPerChannelGradientInferShape");

    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* minShape = context->GetInputShape(INPUT_IDX_MIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, minShape);
    const gert::Shape* maxShape = context->GetInputShape(INPUT_IDX_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxShape);

    gert::Shape* bpxShape = context->GetOutputShape(OUTPUT_IDX_BPX);
    OP_CHECK_NULL_WITH_CONTEXT(context, bpxShape);
    gert::Shape* bpMinShape = context->GetOutputShape(OUTPUT_IDX_BPMIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, bpMinShape);
    gert::Shape* bpMaxShape = context->GetOutputShape(OUTPUT_IDX_BPMAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, bpMaxShape);

    *bpxShape = *xShape;
    *bpMinShape = *minShape;
    *bpMaxShape = *maxShape;

    OP_LOGD(context, "End to do FakeQuantWithMinMaxVarsPerChannelGradientInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus FakeQuantWithMinMaxVarsPerChannelGradientInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "FakeQuantWithMinMaxVarsPerChannelGradientInferDtype enter");
    context->SetOutputDataType(OUTPUT_IDX_BPX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_IDX_BPMIN, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_IDX_BPMAX, ge::DT_FLOAT);
    OP_LOGD(context, "FakeQuantWithMinMaxVarsPerChannelGradientInferDtype end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantWithMinMaxVarsPerChannelGradient)
    .InferShape(FakeQuantWithMinMaxVarsPerChannelGradientInferShape)
    .InferDataType(FakeQuantWithMinMaxVarsPerChannelGradientInferDtype);
} // namespace ops
