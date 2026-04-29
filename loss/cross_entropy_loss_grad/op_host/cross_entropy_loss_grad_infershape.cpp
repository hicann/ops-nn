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
 * \file cross_entropy_loss_grad.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
constexpr uint64_t INPUT_Y_GRAD_IDX = 0;
constexpr uint64_t INPUT_LOG_PROB_IDX = 1;
constexpr uint64_t INPUT_TARGET_IDX = 2;
constexpr uint64_t INPUT_WEIGHT_IDX = 3;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint64_t OUTPUT_X_GRAD_IDX = 0;

static graphStatus InferShape4CrossEntropyLossGrad(gert::InferShapeContext* context) {
  OP_LOGD(context, "Begin to do InferShape4CrossEntropyLossGrad.");
  const gert::Shape* yGradShape = context->GetInputShape(INPUT_Y_GRAD_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, yGradShape);
  const gert::Shape* logProbShape = context->GetInputShape(INPUT_LOG_PROB_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, logProbShape);
  const gert::Shape* targetShape = context->GetInputShape(INPUT_TARGET_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
  const gert::Shape* weightShape = context->GetOptionalInputShape(INPUT_WEIGHT_IDX);

  gert::Shape* xGradShape = context->GetOutputShape(OUTPUT_X_GRAD_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, xGradShape);

  if (Ops::Base::IsUnknownRank(*logProbShape)) { // [-2]输入
    OP_LOGD(context, "Input shape is -2, set output shape to (-2)");
    Ops::Base::SetUnknownRank(*xGradShape);
    return ge::GRAPH_SUCCESS;
  } else {
      OP_CHECK_IF(
          logProbShape->GetDimNum() != DIM_NUM_2,
          OP_LOGE_FOR_INVALID_SHAPEDIM(
              context->GetNodeName(), "log_prob", std::to_string(logProbShape->GetDimNum()).c_str(),
              std::to_string(DIM_NUM_2).c_str()),
          return ge::GRAPH_FAILED);

      OP_CHECK_IF(
          targetShape->GetDimNum() != DIM_NUM_1,
          OP_LOGE_FOR_INVALID_SHAPEDIM(
              context->GetNodeName(), "target", std::to_string(targetShape->GetDimNum()).c_str(),
              std::to_string(DIM_NUM_1).c_str()),
          return ge::GRAPH_FAILED);

      OP_CHECK_IF(
          logProbShape->GetDim(DIM_0) != UNKNOWN_DIM && targetShape->GetDim(DIM_0) != UNKNOWN_DIM &&
              logProbShape->GetDim(DIM_0) != targetShape->GetDim(DIM_0),
          OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
              context->GetNodeName(), "log_prob and target",
              (Ops::Base::ToString(*logProbShape) + " and " + Ops::Base::ToString(*targetShape)).c_str(),
              "The dim 0 of log_prob and target must be the same"),
          return ge::GRAPH_FAILED);
  }

  if (weightShape != nullptr) {
    OP_LOGD(context, "InferShape4CrossEntropyLossGrad: weightShape is not null");
    OP_CHECK_IF(
        weightShape->GetDimNum() != DIM_NUM_1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context->GetNodeName(), "weight", std::to_string(weightShape->GetDimNum()).c_str(),
            std::to_string(DIM_NUM_1).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        logProbShape->GetDim(DIM_1) != UNKNOWN_DIM && weightShape->GetDim(DIM_0) != UNKNOWN_DIM &&
            logProbShape->GetDim(DIM_1) != weightShape->GetDim(DIM_0),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "log_prob and weight",
            (Ops::Base::ToString(*logProbShape) + " and " + Ops::Base::ToString(*weightShape)).c_str(),
            "The dim 1 of log_prob and the dim 0 of target must be the same"),
        return ge::GRAPH_FAILED);
  } else {
    OP_LOGD(context, "InferShape4CrossEntropyLossGrad: weightShape is null.");
  }

  *xGradShape = *logProbShape;
  OP_LOGD(context, "End to do InferShape4CrossEntropyLossGrad.");
  return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForCrossEntropyLossGrad(gert::InferDataTypeContext *context) {
  OP_LOGD(context, "InferDataTypeForCrossEntropyLossGrad Begin.");
  context->SetOutputDataType(OUTPUT_X_GRAD_IDX, context->GetInputDataType(INPUT_LOG_PROB_IDX));
  OP_LOGD(context, "InferDataTypeForCrossEntropyLossGrad End.");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CrossEntropyLossGrad)
  .InferShape(InferShape4CrossEntropyLossGrad)
  .InferDataType(InferDataTypeForCrossEntropyLossGrad);
}  // namespace ops
