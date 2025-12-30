/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "op_nn_proto_extend.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNpuDropoutWithAddSoftmax(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int input_size = node->input_size();
  int output_size = node->output_size();
  op_dest.DynamicInputRegister("x", input_size);
  op_dest.DynamicOutputRegister("y", output_size);

  int required_attr_num = 0;
  float alpha = 0;
  int dim = -1;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "alpha" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      alpha = attr.f();
      required_attr_num++;
    } else if (attr.name() == "dim" && attr.type() == ge::onnx::AttributeProto::INT) {
      dim = attr.i();
    }
  }
  if (required_attr_num != 1) {
    OP_LOGE(GetOpName(op_dest).c_str(), "attr alpha is required.");
    return FAILED;
  }

  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("alpha", alpha);
  op_dest.SetAttr("dim", dim);
  op_dest.SetAttr("original_type", "npu::1::NPUDropoutWithAddSoftmax");
  return SUCCESS;
}

static Status ParseOpToGraphNpuDropoutWithAddSoftmax(const ge::Operator& op, ge::Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
  auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

  float alpha = 0;
  if (op.GetAttr("alpha", alpha) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get alpha from op failed");
    return FAILED;
  }
  int dim = -1;
  if (op.GetAttr("dim", dim) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get dim from op failed");
    return FAILED;
  }

  auto identity_data0_op = ge::op::Identity((ori_name + "_Identity_data0").c_str()).set_input_x(data0);
  auto identity_data1_op = ge::op::Identity((ori_name + "_Identity_data1").c_str()).set_input_x(data1);

  // output[0] mask dtype shoule be uint8.
  auto oneslike_op = ge::op::OnesLike((ori_name + "_OnesLike").c_str()).set_input_x(identity_data0_op);
  auto oneslike_cast_op = ge::op::Cast((ori_name + "_Cast_oneslike").c_str()).set_input_x(oneslike_op)
                                                                   .set_attr_dst_type(4);

  // opType[AxpyWithSoftmaxAndDropOutDoMask] only supports fp16.
  auto data0_cast_op = ge::op::Cast((ori_name + "_Cast_data0").c_str()).set_input_x(identity_data0_op)
                                                             .set_attr_dst_type(1);
  auto data1_cast_op = ge::op::Cast((ori_name + "_Cast_data1").c_str()).set_input_x(identity_data1_op)
                                                             .set_attr_dst_type(1);
  
  // retain = 1 is applied for inferencing.
  float retain = 1;
  auto AxpyWithSoftmaxAndDropOutDoMask = 
    ge::op::AxpyWithSoftmaxAndDropOutDoMask((ori_name + "_AxpyWithSoftmaxAndDropOutDoMask").c_str())
      .set_input_x1(data1_cast_op)
      .set_input_x2(data0_cast_op)
      .set_input_mask(oneslike_cast_op)
      .set_attr_alpha(alpha)
      .set_attr_input_keep_prob(retain)
      .set_attr_axis({dim});

  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.emplace_back(oneslike_cast_op, std::vector<std::size_t>{0});
  outputs.emplace_back(AxpyWithSoftmaxAndDropOutDoMask, std::vector<std::size_t>{0});
  outputs.emplace_back(AxpyWithSoftmaxAndDropOutDoMask, std::vector<std::size_t>{1});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register npu_dropout_with_add_softmax op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUDropoutWithAddSoftmax"), 
                 ge::AscendString("ai.onnx::11::NPUDropoutWithAddSoftmax"),
                 ge::AscendString("ai.onnx::12::NPUDropoutWithAddSoftmax"),
                 ge::AscendString("ai.onnx::13::NPUDropoutWithAddSoftmax"),
                 ge::AscendString("ai.onnx::14::NPUDropoutWithAddSoftmax"),
                 ge::AscendString("ai.onnx::15::NPUDropoutWithAddSoftmax"),
                 ge::AscendString("ai.onnx::16::NPUDropoutWithAddSoftmax"),
                 ge::AscendString("ai.onnx::17::NPUDropoutWithAddSoftmax"),
                 ge::AscendString("ai.onnx::18::NPUDropoutWithAddSoftmax")})
  .ParseParamsFn(ParseParamsNpuDropoutWithAddSoftmax)
  .ParseOpToGraphFn(ParseOpToGraphNpuDropoutWithAddSoftmax)
  .ImplyType(ImplyType::TVM);
} // namespace domi
