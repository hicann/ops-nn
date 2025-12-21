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
 * \file sparse_softmax_cross_entropy_with_logits_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
constexpr uint64_t INPUT_FEATURES_IDX = 0;
constexpr uint64_t INPUT_LABLES_IDX = 1;
constexpr uint64_t OUTPUT_LOSS_IDX = 0;
constexpr uint64_t OUTPUT_BACKPROP_IDX = 1;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;

namespace ops {
static graphStatus InferShape4SparseSoftmaxCrossEntropyWithLogits(gert::InferShapeContext* context) {
    OP_LOGD(context, "Begin to do InferShape4SparseSoftmaxCrossEntropyWithLogits.");
    const gert::Shape* featuresShape = context->GetInputShape(INPUT_FEATURES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);
    const gert::Shape* labelsShape = context->GetInputShape(INPUT_LABLES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelsShape);

    gert::Shape* lossShape = context->GetOutputShape(OUTPUT_LOSS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);
    gert::Shape* backpropShape = context->GetOutputShape(OUTPUT_BACKPROP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);

    if (!Ops::Base::IsUnknownRank(*featuresShape) && !Ops::Base::IsUnknownRank(*labelsShape)) {
        OP_CHECK_IF(
            labelsShape->GetDimNum() != (featuresShape->GetDimNum() - 1),
            OP_LOGE(context, "Labels.dimNum must equal featuresShape.dimNum - 1."),
            return ge::GRAPH_FAILED);

        for (uint32_t i = 0; i < featuresShape->GetDimNum() - 1; i++) {
            if (featuresShape->GetDim(i) != labelsShape->GetDim(i)) {
                OP_LOGE(context, "Labels.shape must equal features.shape except for last dim");
                return ge::GRAPH_FAILED;
            }
        }
        OP_CHECK_IF(
            featuresShape->GetDim(featuresShape->GetDimNum() - 1) <= 0,
            OP_LOGE(context, "Must have at least one class"),
            return ge::GRAPH_FAILED);
    }
    *lossShape = *labelsShape;
    *backpropShape = *featuresShape;
    OP_LOGD(context, "InferShapeForSparseSoftmaxCrossEntropyWithLogits End.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForSparseSoftmaxCrossEntropyWithLogits(gert::InferDataTypeContext *context) {
  OP_LOGD(context, "InferDataTypeForSparseSoftmaxCrossEntropyWithLogits Begin.");
  context->SetOutputDataType(OUTPUT_LOSS_IDX, context->GetInputDataType(INPUT_FEATURES_IDX));
  context->SetOutputDataType(OUTPUT_BACKPROP_IDX, context->GetInputDataType(INPUT_FEATURES_IDX));
  OP_LOGD(context, "InferDataTypeForSparseSoftmaxCrossEntropyWithLogits End.");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseSoftmaxCrossEntropyWithLogits)
  .InferShape(InferShape4SparseSoftmaxCrossEntropyWithLogits)
  .InferDataType(InferDataTypeForSparseSoftmaxCrossEntropyWithLogits);
}