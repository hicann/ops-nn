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
 * \file softmax_cross_entropy_with_logits.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
struct ScewlInputsDims{
  int64_t features_dim_0 = 0;
  int64_t features_dim_1 = 0;
  int64_t features_dim_2 = 0;
  int64_t features_dim_3 = 0;
  int64_t labels_dim_0 = 0;
  int64_t labels_dim_1 = 0;
  int64_t labels_dim_2 = 0;
  int64_t labels_dim_3 = 0;
};

static bool GetInputsDimsForScewl(const size_t& input_features_dim_num, const size_t& input_labels_dim_num,
                                  const gert::Shape* input_features_shape, const gert::Shape* input_labels_shape,
                                  ScewlInputsDims& inputs_dims) {
  if (input_features_dim_num == 2 && input_labels_dim_num == 2) {  // dim_num of ND is 2
    inputs_dims.features_dim_0 = input_features_shape->GetDim(0);
    inputs_dims.features_dim_1 = input_features_shape->GetDim(1);
    inputs_dims.labels_dim_0 = input_labels_shape->GetDim(0);
    inputs_dims.labels_dim_1 = input_labels_shape->GetDim(1);
  } else if (input_features_dim_num == 2 && input_labels_dim_num == 1) {  // dim_num of ND is 2
    inputs_dims.features_dim_0 = input_features_shape->GetDim(0);
    inputs_dims.features_dim_1 = input_features_shape->GetDim(1);
    inputs_dims.labels_dim_0 = 1;
    inputs_dims.labels_dim_1 = input_labels_shape->GetDim(0);
  } else if (input_features_dim_num == 1 && input_labels_dim_num == 2) {  // dim_num of ND is 2
    inputs_dims.features_dim_0 = 1;
    inputs_dims.features_dim_1 = input_features_shape->GetDim(0);
    inputs_dims.labels_dim_0 = input_labels_shape->GetDim(0);
    inputs_dims.labels_dim_1 = input_labels_shape->GetDim(1);
  } else if (input_features_dim_num == 4 && input_labels_dim_num == 4) {  // 4 dims inputs
    inputs_dims.features_dim_0 = input_features_shape->GetDim(0);
    inputs_dims.features_dim_1 = input_features_shape->GetDim(1);
    inputs_dims.features_dim_2 = input_features_shape->GetDim(2);  // 2 is dim_index
    inputs_dims.features_dim_3 = input_features_shape->GetDim(3);  // 3 is dim_index
    inputs_dims.labels_dim_0 = input_labels_shape->GetDim(0);
    inputs_dims.labels_dim_1 = input_labels_shape->GetDim(1);
    inputs_dims.labels_dim_2 = input_labels_shape->GetDim(2);  // 2 is dim_index
    inputs_dims.labels_dim_3 = input_labels_shape->GetDim(3);  // 3 is dim_index
  } else {
    OP_LOGE("SoftmaxCrossEntropyWithLogits", "[TBE Compiler] Get invalid shape");
    return false;
  }
  return true;
}

static bool ScewlCheckInputsDims(const size_t& input_features_dim_num, const size_t& input_labels_dim_num,
                                 ScewlInputsDims& inputs_dims, gert::InferShapeContext *context) {
  if (input_features_dim_num == 4 && input_labels_dim_num == 4) {  // 4 dims inputs
    OP_LOGI(context->GetNodeName(),
            "after complete, input_features_dim is [%ld, %ld, %ld, %ld]",
            inputs_dims.features_dim_0, inputs_dims.features_dim_1,
            inputs_dims.features_dim_2, inputs_dims.features_dim_3);
    OP_LOGI(context->GetNodeName(),
            "after complete, input_labels_dim is [%ld, %ld, %ld, %ld]",
            inputs_dims.labels_dim_0, inputs_dims.labels_dim_1, inputs_dims.labels_dim_2, inputs_dims.labels_dim_3);

    if (inputs_dims.features_dim_2 != inputs_dims.labels_dim_2 &&
            inputs_dims.features_dim_2 != 1 && inputs_dims.labels_dim_2 != 1) {
      OP_LOGE(context->GetNodeName(), "[TBE Compiler] not supported shape for dim2");
      return false;
    }
    if (inputs_dims.features_dim_3 != inputs_dims.labels_dim_3 &&
            inputs_dims.features_dim_3 != 1 && inputs_dims.labels_dim_3 != 1) {
      OP_LOGE(context->GetNodeName(), "[TBE Compiler] not supported shape for dim3");
      return false;
    }
  } else {
    OP_LOGI(context->GetNodeName(),
            "after complete, input_features_dim is [%ld, %ld]", inputs_dims.features_dim_0, inputs_dims.features_dim_1);
    OP_LOGI(context->GetNodeName(),
            "after complete, input_labels_dim is [%ld, %ld]", inputs_dims.labels_dim_0, inputs_dims.labels_dim_1);
  }

  if (inputs_dims.features_dim_0 != inputs_dims.labels_dim_0 &&
          inputs_dims.features_dim_0 != 1 && inputs_dims.labels_dim_0 != 1) {
    OP_LOGE(context->GetNodeName(), "[TBE Compiler] not supported shape for dim0");
    return false;
  }
  if (inputs_dims.features_dim_1 != inputs_dims.labels_dim_1 &&
          inputs_dims.features_dim_1 != 1 && inputs_dims.labels_dim_1 != 1) {
    OP_LOGE(context->GetNodeName(), "[TBE Compiler] not supported shape for dim1");
    return false;
  }
  return true;
}

static graphStatus InferShapeForSoftmaxCrossEntropyWithLogits(gert::InferShapeContext *context) {
  OP_LOGD(context->GetNodeName(), "Begin to do SoftmaxCrossEntropyWithLogitsInferShape");
  const gert::Shape *input_features_shape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, input_features_shape);
  size_t input_features_dim_num = input_features_shape->GetDimNum();
  const gert::Shape *input_labels_shape = context->GetInputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, input_labels_shape);
  size_t input_labels_dim_num = input_labels_shape->GetDimNum();
  OP_LOGD(context->GetNodeName(),
          "input_features_dim_num is %zu, input_labels_dim_num is %zu", input_features_dim_num, input_labels_dim_num);

  ScewlInputsDims inputs_dims;
  if (!GetInputsDimsForScewl(input_features_dim_num, input_labels_dim_num,
                             input_features_shape, input_labels_shape, inputs_dims)) {
    return GRAPH_FAILED;
  }

  if (!ScewlCheckInputsDims(input_features_dim_num, input_labels_dim_num, inputs_dims, context)) {
    return GRAPH_FAILED;
  }

  int64_t dim_0 = inputs_dims.features_dim_0 >= inputs_dims.labels_dim_0 ?
                      inputs_dims.features_dim_0 : inputs_dims.labels_dim_0;
  int64_t dim_1 = inputs_dims.features_dim_1 >= inputs_dims.labels_dim_1 ?
                      inputs_dims.features_dim_1 : inputs_dims.labels_dim_1;

  gert::Shape *output_loss_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, output_loss_shape);
  output_loss_shape->SetDimNum(0);
  output_loss_shape->AppendDim(dim_0);
  if (input_features_dim_num == 4 && input_labels_dim_num == 4) {  // 4 dims inputs
    output_loss_shape->AppendDim(1);
  }

  gert::Shape *output_backprop_shape = context->GetOutputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, output_backprop_shape);
  output_backprop_shape->SetDimNum(0);
  output_backprop_shape->AppendDim(dim_0);
  output_backprop_shape->AppendDim(dim_1);

  if (input_features_dim_num == 4 && input_labels_dim_num == 4) {  // 4 dims inputs
    int64_t dim_2 = inputs_dims.features_dim_2 >= inputs_dims.labels_dim_2 ?
                        inputs_dims.features_dim_2 : inputs_dims.labels_dim_2;
    int64_t dim_3 = inputs_dims.features_dim_3 >= inputs_dims.labels_dim_3 ?
                        inputs_dims.features_dim_3 : inputs_dims.labels_dim_3;

    output_loss_shape->AppendDim(dim_2);
    output_backprop_shape->AppendDim(dim_2);
    output_loss_shape->AppendDim(dim_3);
    output_backprop_shape->AppendDim(dim_3);
  }

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftmaxCrossEntropyWithLogits).InferShape(InferShapeForSoftmaxCrossEntropyWithLogits);
}  // namespace ops