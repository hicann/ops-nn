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
 * \file apply_adam_w_quant_infershape.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
static constexpr int64_t IDX_0 = 0; // INPUT_VAR
static constexpr int64_t IDX_1 = 1; // INPUT_GRAD
static constexpr int64_t IDX_2 = 2; // INPUT_STATE_M
static constexpr int64_t IDX_3 = 3; // INPUT_STATE_V
static constexpr int64_t IDX_4 = 4; // INPUT_QMAP_M
static constexpr int64_t IDX_5 = 5; // INPUT_QMAP_V
static constexpr int64_t IDX_6 = 6; // INPUT_ABSMAX_M
static constexpr int64_t IDX_7 = 7; // INPUT_ABSMAX_V
static constexpr int64_t OUT_IDX_0 = 0; // OUTPUT_VAR
static constexpr int64_t OUT_IDX_1 = 1; // OUTPUT_STATE_M
static constexpr int64_t OUT_IDX_2 = 2; // OUTPUT_STATE_V
static constexpr int64_t OUT_IDX_3 = 3; // OUTPUT_ABSMAX_M
static constexpr int64_t OUT_IDX_4 = 4; // OUTPUT_ABSMAX_V
static ge::graphStatus InferShape4ApplyAdamWQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4ApplyAdamWQuant");

    // get input shapes
    auto varShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    auto gradShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    auto stateMShape = context->GetInputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateMShape);
    auto stateVShape = context->GetInputShape(IDX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateVShape);
    auto qmapMShape = context->GetInputShape(IDX_4);
    OP_CHECK_NULL_WITH_CONTEXT(context, qmapMShape);
    auto qmapVShape = context->GetInputShape(IDX_5);
    OP_CHECK_NULL_WITH_CONTEXT(context, qmapVShape);
    auto absMShape = context->GetInputShape(IDX_6);
    OP_CHECK_NULL_WITH_CONTEXT(context, absMShape);
    auto absVShape = context->GetInputShape(IDX_7);
    OP_CHECK_NULL_WITH_CONTEXT(context, absVShape);

    // get output shapes
    auto y1Shape = context->GetOutputShape(OUT_IDX_0);
    auto y2Shape = context->GetOutputShape(OUT_IDX_1);
    auto y3Shape = context->GetOutputShape(OUT_IDX_2);
    auto y4Shape = context->GetOutputShape(OUT_IDX_3);
    auto y5Shape = context->GetOutputShape(OUT_IDX_4);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y3Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y4Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y5Shape);

    *y1Shape = *varShape;
    *y2Shape = *stateMShape;
    *y3Shape = *stateVShape;
    *y4Shape = *absMShape;
    *y5Shape = *absVShape;

    OP_LOGD(context, "End to do InferShape4ApplyAdamWQuant");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ApplyAdamWQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4ApplyAdamWQuant");

    auto inputVarDtype = context->GetInputDataType(IDX_0);
    auto inputStateMDtype = context->GetInputDataType(IDX_2);
    auto inputStateVDtype = context->GetInputDataType(IDX_3);
    auto inputAbsMDtype = context->GetInputDataType(IDX_6);
    auto inputAbsVDtype = context->GetInputDataType(IDX_7);

    context->SetOutputDataType(OUT_IDX_0, inputVarDtype);
    context->SetOutputDataType(OUT_IDX_1, inputStateMDtype);
    context->SetOutputDataType(OUT_IDX_2, inputStateVDtype);
    context->SetOutputDataType(OUT_IDX_3, inputAbsMDtype);
    context->SetOutputDataType(OUT_IDX_4, inputAbsVDtype);

    OP_LOGD(context, "End to do InferDataType4ApplyAdamWQuant");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdamWQuant).InferShape(InferShape4ApplyAdamWQuant).InferDataType(InferDataType4ApplyAdamWQuant);
} // namespace ops