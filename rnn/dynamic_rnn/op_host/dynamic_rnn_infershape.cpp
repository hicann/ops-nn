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
 * \file dynamic_rnn_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace ge;
namespace ops {
constexpr int X_SHAPE_SIZE_LIMIT = 3;
constexpr int CONSTANT_FOUR = 4;
constexpr int CONSTANT_TWO = 2;
constexpr int RNN_OUTPUT_INDEX_C = 2;
constexpr int RNN_OUTPUT_INDEX_I = 3;
constexpr int RNN_OUTPUT_INDEX_J = 4;
constexpr int RNN_OUTPUT_INDEX_F = 5;
constexpr int RNN_OUTPUT_INDEX_O = 6;
constexpr int RNN_OUTPUT_INDEX_TANHC = 7;
constexpr int64_t UNKNOWN_DIM_VALUE = -1;

static ge::graphStatus InferShape4DynamicRNN(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "InferShape4DynamicRNN start");
  auto x_shape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  auto w_shape = context->GetInputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, w_shape);

  auto y_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  auto outputh_shape = context->GetOutputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, outputh_shape);
  auto outputc_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_C);
  OP_CHECK_NULL_WITH_CONTEXT(context, outputc_shape);
  auto i_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_I);
  OP_CHECK_NULL_WITH_CONTEXT(context, i_shape);
  auto j_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_J);
  OP_CHECK_NULL_WITH_CONTEXT(context, j_shape);
  auto f_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_F);
  OP_CHECK_NULL_WITH_CONTEXT(context, f_shape);
  auto o_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_O);
  OP_CHECK_NULL_WITH_CONTEXT(context, o_shape);
  auto tanhc_shape = context->GetOutputShape(RNN_OUTPUT_INDEX_TANHC);
  OP_CHECK_NULL_WITH_CONTEXT(context, tanhc_shape);

  if (x_shape->GetDimNum() != X_SHAPE_SIZE_LIMIT) {
    OP_LOGE(context->GetNodeName(),
                                        "The input x shape dim is not 3, please check!");
    return ge::GRAPH_FAILED;
  }

  int64_t num_step = x_shape->GetDim(0);
  int64_t batch_size = x_shape->GetDim(1);
  int64_t hidden_size = 0;
  if (w_shape->GetDim(1) == UNKNOWN_DIM_VALUE) {
    hidden_size = UNKNOWN_DIM_VALUE;
  } else {
    hidden_size = w_shape->GetDim(1) / CONSTANT_FOUR;
  }

  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const char* direction = attrs->GetAttrPointer<char>(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, direction);

  if (strcmp(direction, "BIDIRECTIONAL") == 0) {
    *y_shape = {num_step, batch_size, CONSTANT_TWO * hidden_size};
    *outputh_shape = {num_step, batch_size, CONSTANT_TWO * hidden_size};
    *outputc_shape = {num_step, batch_size, CONSTANT_TWO * hidden_size};
    *i_shape = {CONSTANT_TWO * num_step, batch_size, hidden_size};
    *j_shape = {CONSTANT_TWO * num_step, batch_size, hidden_size};
    *f_shape = {CONSTANT_TWO * num_step, batch_size, hidden_size};
    *o_shape = {CONSTANT_TWO * num_step, batch_size, hidden_size};
    *tanhc_shape = {CONSTANT_TWO * num_step, batch_size, hidden_size};
  } else {
    *y_shape = {num_step, batch_size, hidden_size};
    *outputh_shape = {num_step, batch_size, hidden_size};
    *outputc_shape = {num_step, batch_size, hidden_size};
    *i_shape = {num_step, batch_size, hidden_size};
    *j_shape = {num_step, batch_size, hidden_size};
    *f_shape = {num_step, batch_size, hidden_size};
    *o_shape = {num_step, batch_size, hidden_size};
    *tanhc_shape = {num_step, batch_size, hidden_size};
  }
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DynamicRNN(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "InferDataType4DynamicRNN start");
  auto input_x_dtype = context->GetInputDataType(0);
  auto input_b_dtype = context->GetInputDataType(CONSTANT_TWO);

  OP_CHECK_IF(context->SetOutputDataType(0, input_b_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType y Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(1, input_x_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType output_h Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_C, input_b_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType output_c Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_I, input_b_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType i Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_J, input_b_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType j Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_F, input_b_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType f Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_O, input_b_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType o Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_TANHC, input_b_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType tanhc Fail"), return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "InferDataType4DynamicRNN end");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicRNN).InferShape(InferShape4DynamicRNN).InferDataType(InferDataType4DynamicRNN);
}  // namespace ops

