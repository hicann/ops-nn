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
 * \file int8_dequantize_plugin.cc
 * \brief
 */
#include "onnx_common.h"
#include "op_nn_proto_extend.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

static Status ParseParamsInt8Dequantize(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE("ParseParamsInt8Dequantize", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float scale = 0.f;
  float offset = 0.f;
  for (auto& attr : node->attribute()) {
    if (attr.name().find("scale") != std::string::npos) {
      scale = attr.f();
    }
    if (attr.name().find("zero_point") != std::string::npos) {
      offset = static_cast<float>(attr.i());
    }
  }
  
  op_dest.DynamicInputRegister("x", 1);
  op_dest.DynamicOutputRegister("y", 1);
  op_dest.SetAttr("original_type", "ai.onnx::11::Int8Dequantize");
  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("offset", offset);
  op_dest.SetAttr("name", node->name());
  return SUCCESS;
}

static Status  ParseOpToGraphInt8Dequantize(const ge::Operator& op, Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  float scale = 0.f;
  float offset = 0.f;
  ge::Operator::OpListInt maxpool_ksize = {1, 1, 1, 1};
  ge::Operator::OpListInt maxpool_strides = {1, 1, 1, 1};
  op.GetAttr("scale", scale);
  op.GetAttr("offset", offset);
  auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(0);
  auto ascend_op = op::AscendAntiQuant((ori_name + "_Int8DeqAscendAntiquant").c_str())
                       .set_input_x(data1)
                       .set_attr_scale(scale)
                       .set_attr_offset(offset)
                       .set_attr_dtype(DT_FLOAT16);
  auto maxpool = op::MaxPool((ori_name + "_Int8DeqAscendMaxpool").c_str())
                     .set_input_x(ascend_op)
                     .set_attr_ksize(maxpool_ksize)
                     .set_attr_strides(maxpool_strides)
                     .set_attr_padding("VALID")
                     .set_attr_data_format("NCHW");
  std::vector<ge::Operator> inputs = {data1};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;

  output_indexs.emplace_back(maxpool, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Int8Dequantize"),
                   ge::AscendString("ai.onnx::9::Int8Dequantize"),
                   ge::AscendString("ai.onnx::10::Int8Dequantize"),
                   ge::AscendString("ai.onnx::11::Int8Dequantize"),
                   ge::AscendString("ai.onnx::12::Int8Dequantize"),
                   ge::AscendString("ai.onnx::13::Int8Dequantize"),
                   ge::AscendString("ai.onnx::14::Int8Dequantize"),
                   ge::AscendString("ai.onnx::15::Int8Dequantize"),
                   ge::AscendString("ai.onnx::16::Int8Dequantize")})
    .ParseParamsFn(ParseParamsInt8Dequantize)
    .ParseOpToGraphFn(ParseOpToGraphInt8Dequantize)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
