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

using namespace ge;
namespace domi {
static const uint32_t INPUT_NUM = 2;
static const uint32_t OUTPUT_NUM = 1;

using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsInt8Transpose(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("Int8Transpose", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  op_dest.DynamicInputRegister("x", INPUT_NUM);
  op_dest.DynamicOutputRegister("y", OUTPUT_NUM);
  op_dest.SetAttr("original_type", "ai.onnx::11::Int8Transpose");

  std::vector<int64_t> axes = {};
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        axes.push_back(attr.ints(i));
      }
    }
  }

  int num = axes.size();
  std::vector<int64_t> dims = {num};
  ge::Tensor tensor = Vec2Tensor(axes, dims, ge::DT_INT64);
  op_dest.SetAttr("axes", tensor);
  op_dest.SetAttr("name", node->name());
  return SUCCESS;
}

static Status ParseOpToGraphInt8Transpose(const ge::Operator& op, Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);

  ge::Tensor perm;
  if (op.GetAttr("axes", perm) != SUCCESS) {
    OP_LOGE("Int8Transpose", "get perm from op failed");
    return FAILED;
  }

  auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(perm);
  auto int8transpose = op::Transpose((ori_name + "_Transpose").c_str()).set_input_x(data0).set_input_perm(data1);

  std::vector<ge::Operator> inputs{data0, data1};
  std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
  outputs.emplace_back(int8transpose, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register int8transpose op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Int8Transpose"),
                   ge::AscendString("ai.onnx::9::Int8Transpose"),
                   ge::AscendString("ai.onnx::10::Int8Transpose"),
                   ge::AscendString("ai.onnx::11::Int8Transpose"),
                   ge::AscendString("ai.onnx::12::Int8Transpose"),
                   ge::AscendString("ai.onnx::13::Int8Transpose"),
                   ge::AscendString("ai.onnx::14::Int8Transpose"),
                   ge::AscendString("ai.onnx::15::Int8Transpose"),
                   ge::AscendString("ai.onnx::16::Int8Transpose")})
    .ParseParamsFn(ParseParamsInt8Transpose)
    .ParseOpToGraphFn(ParseOpToGraphInt8Transpose)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
