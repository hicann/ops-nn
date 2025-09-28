/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file cross_entropy_loss_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
constexpr uint32_t INPUT_DATA_IDX = 0;
constexpr uint32_t INPUT_TARGET_IDX = 1;
constexpr uint32_t INPUT_WEIGHT_IDX = 2;
constexpr uint32_t OUTPUT_LOSS_IDX = 0;
constexpr uint32_t OUTPUT_LOGPROB_IDX = 1;
constexpr uint32_t ATTR_REDUCTION_IDX = 0;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t LOSS_SHAPE = 1;

namespace ops {
static ge::graphStatus InferShapeForCrossEntropyLoss(gert::InferShapeContext* context)
{
    // input shape
    OP_LOGD(context, "InferShapeForCrossEntropyLoss Begin.");
    const gert::Shape* inputShape = context->GetInputShape(INPUT_DATA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape* targetShape = context->GetInputShape(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);

    // output shape
    gert::Shape* lossShape = context->GetOutputShape(OUTPUT_LOSS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);
    gert::Shape* logprobShape = context->GetOutputShape(OUTPUT_LOGPROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, logprobShape);

    if (Ops::Base::IsUnknownRank(*inputShape)) { // -2
        Ops::Base::SetUnknownRank(*lossShape);
        Ops::Base::SetUnknownRank(*logprobShape);
    } else {
        OP_CHECK_IF(
            inputShape->GetDimNum() != DIM_NUM_2,
            OP_LOGE(context, "Input dim must be 2."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            targetShape->GetDimNum() != DIM_NUM_1,
            OP_LOGE(context, "target dim must be 1."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            inputShape->GetDim(DIM_0) != UNKNOWN_DIM && targetShape->GetDim(DIM_0) != UNKNOWN_DIM &&
                inputShape->GetDim(DIM_0) != targetShape->GetDim(DIM_0),
            OP_LOGE(context, "Input dim 0 should be equal to target dim 0."),
            return ge::GRAPH_FAILED);

        const gert::Shape* weightShape = context->GetOptionalInputShape(INPUT_WEIGHT_IDX); // optional input
        if (weightShape != nullptr) {
            OP_CHECK_IF(
                weightShape->GetDimNum() != DIM_NUM_1,
                OP_LOGE(context, "weight dim must be 1."),
                return ge::GRAPH_FAILED);

            OP_CHECK_IF(
                inputShape->GetDim(DIM_1) != UNKNOWN_DIM && weightShape->GetDim(DIM_0) != UNKNOWN_DIM &&
                    inputShape->GetDim(DIM_1) != weightShape->GetDim(DIM_0),
                OP_LOGE(
                    context, "Input dim 1 should be equal to weight dim 0."),
                return ge::GRAPH_FAILED);
        }
        const gert::RuntimeAttrs* attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const char* reduction = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
        if (reduction != nullptr && strcmp(reduction, "none") == 0) {
            lossShape->SetDimNum(DIM_NUM_1);
            lossShape->SetDim(DIM_0, inputShape->GetDim(DIM_0));
        } else {
            lossShape->SetDimNum(DIM_NUM_1);
            lossShape->SetDim(DIM_0, LOSS_SHAPE);
        }
        *logprobShape = *inputShape;
    }
    OP_LOGD(context, "InferShapeForCrossEntropyLoss End.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForCrossEntropyLoss(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDataTypeForCrossEntropyLoss Begin.");
    context->SetOutputDataType(OUTPUT_LOSS_IDX, context->GetInputDataType(INPUT_DATA_IDX));
    context->SetOutputDataType(OUTPUT_LOGPROB_IDX, context->GetInputDataType(INPUT_DATA_IDX));
    OP_LOGD(context, "InferDataTypeForCrossEntropyLoss End.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CrossEntropyLoss)
    .InferShape(InferShapeForCrossEntropyLoss)
    .InferDataType(InferDataTypeForCrossEntropyLoss);
} // namespace ops
