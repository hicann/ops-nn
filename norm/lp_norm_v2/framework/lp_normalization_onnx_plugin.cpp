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
 * \file lp_normalization_plugin.cc
 * \brief
 */
#include "onnx_common.h"
#include "norm/lp_norm_v2/op_graph/lp_norm_v2_proto.h"


using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
static Status parse_params_lp_normalization(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  // 1.add dynamic input and out
  op_dest.DynamicInputRegister("x", 1);
  op_dest.DynamicOutputRegister("y", 1);
  // 2.set original_type
  op_dest.SetAttr("original_type", "ai.onnx::1::LpNormalization");
  // 3.set attrs
  int axis = -1;
  int p = 2;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = static_cast<float>(attr.i());
    } else if (attr.name() == "p" && attr.type() == ge::onnx::AttributeProto::INT) {
      p = static_cast<int>(attr.i());
    }
  }
  op_dest.SetAttr("p", p);
  op_dest.SetAttr("axis", axis);
  op_dest.SetAttr("name", node->name());
  return SUCCESS;
}

static Status parse_op_to_graph_lp_normalization(const Operator& op, Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  auto data0_op = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
  auto identity_op = op::Identity((ori_name + "_Identity").c_str()).set_input_x(data0_op);
  auto lp_norm_op = op::LpNormV2((ori_name + "_LpNormV2").c_str()).set_input_x(identity_op);
  auto div_op = op::Div((ori_name + "_Div").c_str()).set_input_x1(identity_op).set_input_x2(lp_norm_op);

  int p_num = 2;
  int axis = -1;
  if (op.GetAttr("p", p_num) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get attr p from op failed");
    return FAILED;
  }
  if (op.GetAttr("axis", axis) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get attr axis from op failed");
    return FAILED;
  }
  lp_norm_op.set_attr_p(static_cast<float>(p_num));
  lp_norm_op.set_attr_axes({axis});
  lp_norm_op.set_attr_keepdim(true);
  lp_norm_op.set_attr_epsilon(0);

  std::vector<Operator> inputs{data0_op};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(div_op, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

// register LpNormalization op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::1::LpNormalization"),
                 ge::AscendString("ai.onnx::8::LpNormalization"),
                 ge::AscendString("ai.onnx::9::LpNormalization"),
                 ge::AscendString("ai.onnx::10::LpNormalization"),
                 ge::AscendString("ai.onnx::11::LpNormalization"),
                 ge::AscendString("ai.onnx::12::LpNormalization"),
                 ge::AscendString("ai.onnx::13::LpNormalization"),
                 ge::AscendString("ai.onnx::14::LpNormalization"),
                 ge::AscendString("ai.onnx::15::LpNormalization"),
                 ge::AscendString("ai.onnx::16::LpNormalization"),
                 ge::AscendString("ai.onnx::17::LpNormalization"),
                 ge::AscendString("ai.onnx::18::LpNormalization")})
  .ParseParamsFn(parse_params_lp_normalization)
  .ParseOpToGraphFn(parse_op_to_graph_lp_normalization)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
