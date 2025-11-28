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
 * \file dynamic_rnnv2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
constexpr int X_SHAPE_SIZE_LIMIT = 3;
constexpr int WEIGHT_INPUT_SIZE_LIMIT = 2;
constexpr int RNN_OUTPUT_INDEX_TANHC = 7;
constexpr int RNN_OUTPUT_INDEX_O = 6;
constexpr int RNN_OUTPUT_INDEX_F = 5;
constexpr int RNN_OUTPUT_INDEX_J = 4;
constexpr int RNN_OUTPUT_INDEX_I = 3;
constexpr int RNN_OUTPUT_INDEX_C = 2;


static ge::graphStatus InferDataType4DynamicRNNV2(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "InferDataType4DynamicRNNV2 start");
  auto input_x_dtype = context->GetInputDataType(0);
  auto output_y_dtype = input_x_dtype;

  auto bias_dtype = context->GetOptionalInputDataType(3);
  auto initc_dtype = context->GetOptionalInputDataType(6);
  if (bias_dtype != ge::DT_UNDEFINED) {
    output_y_dtype = bias_dtype;
  } else if (initc_dtype != ge::DT_UNDEFINED) {
    output_y_dtype = initc_dtype;
  }

  OP_CHECK_IF(context->SetOutputDataType(0, output_y_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType y Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(1, input_x_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType output_h Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_C, output_y_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType output_c Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_I, output_y_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType i Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_J, output_y_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType j Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_F, output_y_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType f Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_O, output_y_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType o Fail"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_TANHC, output_y_dtype) != ge::GRAPH_SUCCESS,
           OP_LOGE(context->GetNodeName(), "SetOutputDataType tanhc Fail"), return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "InferDataType4DynamicRNNV2 end");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4DynamicRNNV2(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "InferShape4DynamicRNNV2 start");
  auto x_shape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  OP_CHECK_IF(x_shape->GetDimNum() != X_SHAPE_SIZE_LIMIT,
           OP_LOGE(context->GetNodeName(),
                                               "The dimension count of x should be 3, please check!"),
           return ge::GRAPH_FAILED);

  auto weight_hidden_shape = context->GetInputShape(2);
  OP_CHECK_NULL_WITH_CONTEXT(context, weight_hidden_shape);

  OP_CHECK_IF(weight_hidden_shape->GetDimNum() != WEIGHT_INPUT_SIZE_LIMIT,
           OP_LOGE(
               context->GetNodeName(), "The dimension count of weight hidden should be equal to 2, please check!"),
           return ge::GRAPH_FAILED);

  auto y_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  auto outputh_shape = context->GetOutputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, outputh_shape);
  auto outputc_shape = context->GetOutputShape(2);
  OP_CHECK_NULL_WITH_CONTEXT(context, outputc_shape);
  auto i_shape = context->GetOutputShape(3);
  OP_CHECK_NULL_WITH_CONTEXT(context, i_shape);
  auto j_shape = context->GetOutputShape(4);
  OP_CHECK_NULL_WITH_CONTEXT(context, j_shape);
  auto f_shape = context->GetOutputShape(5);
  OP_CHECK_NULL_WITH_CONTEXT(context, f_shape);
  auto o_shape = context->GetOutputShape(6);
  OP_CHECK_NULL_WITH_CONTEXT(context, o_shape);
  auto tanhc_shape = context->GetOutputShape(7);
  OP_CHECK_NULL_WITH_CONTEXT(context, tanhc_shape);

  int64_t num_step = x_shape->GetDim(0);
  int64_t batch_size = x_shape->GetDim(1);
  int64_t hidden_size = weight_hidden_shape->GetDim(0);

  *y_shape = {num_step, batch_size, hidden_size};
  *outputh_shape = {1, batch_size, hidden_size};
  *outputc_shape = {1, batch_size, hidden_size};
  *i_shape = {num_step, batch_size, hidden_size};
  *j_shape = {num_step, batch_size, hidden_size};
  *f_shape = {num_step, batch_size, hidden_size};
  *o_shape = {num_step, batch_size, hidden_size};
  *tanhc_shape = {num_step, batch_size, hidden_size};

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicRNNV2).InferShape(InferShape4DynamicRNNV2).InferDataType(InferDataType4DynamicRNNV2);

}  // namespace ops

