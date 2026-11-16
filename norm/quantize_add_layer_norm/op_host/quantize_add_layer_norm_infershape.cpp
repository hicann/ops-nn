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
 * \file quantize_add_layer_norm.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

static constexpr int X1_IDX = 0;
static constexpr int X2_IDX = 1;
static constexpr int GAMMA_IDX = 2;
static constexpr int BETA_IDX = 3;
static constexpr int BIAS_IDX = 4;
static constexpr int SCALES_IDX = 5;
static constexpr int ZERO_POINTS_IDX = 6;

static constexpr int Y_IDX = 0;
static constexpr int X_IDX = 1;

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4QuantizeAddLayerNorm(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4QuantizeAddLayerNorm");

    // get input shapes
    const gert::Shape* x1Shape = context->GetInputShape(X1_IDX);
    const gert::Shape* x2Shape = context->GetInputShape(X2_IDX);
    OP_CHECK_IF(
        x1Shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x1",
                                                 "the input shape of required input x1 is missing in the infer-shape "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        x2Shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x2",
                                                 "the input shape of required input x2 is missing in the infer-shape "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    const gert::Shape* gammaShape = context->GetInputShape(GAMMA_IDX);
    OP_CHECK_IF(gammaShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "gamma",
                    "the input shape of required input gamma is missing in the infer-shape context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(Y_IDX);
    gert::Shape* xShape = context->GetOutputShape(X_IDX);
    OP_CHECK_IF(
        yShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "y",
                                                 "the output shape of required output y is missing in the infer-shape "
                                                 "context; check that the graph node provides all required outputs"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                 "the output shape of required output x is missing in the infer-shape "
                                                 "context; check that the graph node provides all required outputs"),
        return ge::GRAPH_FAILED);
    *yShape = *x1Shape;
    *xShape = *x1Shape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4QuantizeAddLayerNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantizeAddLayerNorm).InferShape(InferShape4QuantizeAddLayerNorm);
} // namespace ops
