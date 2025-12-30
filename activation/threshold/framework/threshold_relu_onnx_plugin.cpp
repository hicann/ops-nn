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
#include "activation/threshold/op_graph/threshold_relu_proto.h"
using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;
static Status ParseParamsThresholdedRelu(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float alpha = 1.0;
  for (auto attr : node->attribute()) {
    if (attr.name() == "alpha") {
      alpha = attr.f();
    }
  }

  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("alpha", alpha);
  op_dest.SetAttr("original_type", "ai.onnx::11::ThresholdedRelu");
  
  op_dest.DynamicInputRegister("x", 1);
  op_dest.DynamicOutputRegister("y", 1);

  return SUCCESS;
}

static Status ParseOpToThresholdedRelu(const ge::Operator &op, Graph &graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  float alpha = 1.0f;
  op.GetAttr("alpha", alpha);
  auto data = op::Data((ori_name + "_data1").c_str()).set_attr_index(0);
  auto identity_op = op::Identity((ori_name + "_identity").c_str()).set_input_x(data);
  auto threshold_op = op::Threshold((ori_name + "_threshold").c_str()).set_input_x(identity_op).set_attr_threshold(alpha);
  auto mul_op = op::Mul((ori_name + "_mul").c_str()).set_input_x1(identity_op).set_input_x2(threshold_op);
  std::vector<ge::Operator> inputs{data};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(mul_op, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::10::ThresholdedRelu"),
                 ge::AscendString("ai.onnx::11::ThresholdedRelu"),
                 ge::AscendString("ai.onnx::12::ThresholdedRelu"),
                 ge::AscendString("ai.onnx::13::ThresholdedRelu"),
                 ge::AscendString("ai.onnx::14::ThresholdedRelu"),
                 ge::AscendString("ai.onnx::15::ThresholdedRelu"),
                 ge::AscendString("ai.onnx::16::ThresholdedRelu"),
                 ge::AscendString("ai.onnx::17::ThresholdedRelu"),
                 ge::AscendString("ai.onnx::18::ThresholdedRelu")})
  .ParseParamsFn(ParseParamsThresholdedRelu)
  .ParseOpToGraphFn(ParseOpToThresholdedRelu)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
