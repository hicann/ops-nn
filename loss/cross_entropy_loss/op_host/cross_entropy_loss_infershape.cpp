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
        OP_CHECK_IF(inputShape->GetDimNum() != DIM_NUM_2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        context->GetNodeName(), "input", std::to_string(inputShape->GetDimNum()).c_str(),
                        std::to_string(2).c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            targetShape->GetDimNum() != DIM_NUM_1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                        context->GetNodeName(), "target", std::to_string(targetShape->GetDimNum()).c_str(),
                        std::to_string(1).c_str()),
            return ge::GRAPH_FAILED);

        if(inputShape->GetDim(DIM_0) != UNKNOWN_DIM && targetShape->GetDim(DIM_0) != UNKNOWN_DIM && inputShape->GetDim(DIM_0) != targetShape->GetDim(DIM_0)){
            std::string shapeMsg = Ops::Base::ToString(*inputShape) + " and " + Ops::Base::ToString(*targetShape);
            std::string errMsg = "The dim 0 of input and target should be the same";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "input and target", shapeMsg.c_str(), errMsg.c_str());
            return ge::GRAPH_FAILED;
        }

        const gert::Shape* weightShape = context->GetOptionalInputShape(INPUT_WEIGHT_IDX); // optional input
        if (weightShape != nullptr) {
            OP_CHECK_IF(
                weightShape->GetDimNum() != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                        context->GetNodeName(), "weight", std::to_string(weightShape->GetDimNum()).c_str(),
                        std::to_string(DIM_NUM_1).c_str()),
                return ge::GRAPH_FAILED);

            if (inputShape->GetDim(DIM_1) != UNKNOWN_DIM && weightShape->GetDim(DIM_0) != UNKNOWN_DIM &&
                inputShape->GetDim(DIM_1) != weightShape->GetDim(DIM_0)) {
                std::string shapeMsg = Ops::Base::ToString(*inputShape) + " and " + Ops::Base::ToString(*targetShape);
                std::string errMsg = "The dim 1 of input and the dim 0 of weight should be the same";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "input and weight", shapeMsg.c_str(), errMsg.c_str());
                return ge::GRAPH_FAILED;
            }
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
