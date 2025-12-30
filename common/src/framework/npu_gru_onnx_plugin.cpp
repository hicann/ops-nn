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
 * \file npu_gru_plugin.cc
 * \brief onnx plugin for npu custom operator npu_gru
 */
#include "onnx_common.h"
#include "op_nn_proto_extend.h"

namespace {
const int OUTPUT_SIZE = 6;
}
namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNpuGru(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed!");
    return FAILED;
  }
  op_dest.SetAttr("gate_order", "rzh");
  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("original_type", "npu::1::NPUGru");
  int input_size = node->input_size();

  op_dest.DynamicInputRegister("x", input_size);
  op_dest.DynamicOutputRegister("y", OUTPUT_SIZE);

  return SUCCESS;
}

static Status ParseOpToGraphNpuGru(const ge::Operator& op, ge::Graph& graph) {
  std::string ori_name, gate_order;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }
  if (op.GetAttr("gate_order", gate_order) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get gate_order from op failed.");
    return FAILED;
  }
  auto data_x = ge::op::Data((ori_name + "_x").c_str()).set_attr_index(0);
  auto data_hx = ge::op::Data((ori_name + "_hx").c_str()).set_attr_index(1);
  auto data_weight_input = ge::op::Data((ori_name + "_weight_input").c_str()).set_attr_index(2);
  auto data_weight_hidden = ge::op::Data((ori_name + "_weight_hidden").c_str()).set_attr_index(3);
  auto data_bias_input = ge::op::Data((ori_name + "_bias_input").c_str()).set_attr_index(4);
  auto data_bias_hidden = ge::op::Data((ori_name + "_bias_hidden").c_str()).set_attr_index(5);
  auto data_seq_length = ge::op::Data((ori_name + "_seq_length").c_str()).set_attr_index(6);
  // data_seq_length is not used in DynamicGRUV2, connect it with identity to drop it 
  auto identity = ge::op::Identity((ori_name + "_identity").c_str()).set_input_x(data_seq_length);
  auto dynamic_gru_v2 = ge::op::DynamicGRUV2((ori_name + "_DynamicGRUV2").c_str())
                            .set_input_x(data_x)
                            .set_input_weight_input(data_weight_input)
                            .set_input_weight_hidden(data_weight_hidden)
                            .set_input_bias_input(data_bias_input)
                            .set_input_bias_hidden(data_bias_hidden)
                            .set_input_init_h(data_hx)
                            .set_attr_gate_order(gate_order);

  std::vector<ge::Operator> inputs{data_x, data_hx, data_weight_input, data_weight_hidden,
                                   data_bias_input, data_bias_hidden, data_seq_length};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexes;
  for (size_t i = 0; i < OUTPUT_SIZE; i++) {
    output_indexes.emplace_back(dynamic_gru_v2, std::vector<size_t>{i});
  }
  graph.SetInputs(inputs).SetOutputs(output_indexes);
  return SUCCESS;
}

// register npu_gru op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUGru"),
                   ge::AscendString("ai.onnx::11::NPUGru"),
                   ge::AscendString("ai.onnx::12::NPUGru"),
                   ge::AscendString("ai.onnx::13::NPUGru"),
                   ge::AscendString("ai.onnx::14::NPUGru"),
                   ge::AscendString("ai.onnx::15::NPUGru"),
                   ge::AscendString("ai.onnx::16::NPUGru"),
                   ge::AscendString("ai.onnx::17::NPUGru"),
                   ge::AscendString("ai.onnx::18::NPUGru")})
    .ParseParamsFn(ParseParamsNpuGru)
    .ParseOpToGraphFn(ParseOpToGraphNpuGru)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
