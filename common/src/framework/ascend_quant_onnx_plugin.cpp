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

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static void SetAttrFun(const NodeProto* node, ge::Operator& op_dest) {
  float scale = 0.0;
  float offset = 0.0;
  bool sqrtmode = false;
  int dtype = ge::DT_INT8;
  std::string round_mode = "Round";
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale = attr.f();
    } else if (attr.name() == "offset" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      offset = attr.f();
    } else if (attr.name() == "round_mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      round_mode = attr.s();
    } else if (attr.name() == "sqrt_mode" && attr.i() != 0) {
      sqrtmode = true;
    } else if (attr.name() == "dst_type") {
      std::string str_dst_type = attr.s();
      int pos = str_dst_type.find_last_not_of(' ');
      if (pos != 0) {
        str_dst_type.erase(pos + 1);
      }
      if (str_dst_type == "INT16") {
        dtype = ge::DT_INT16;
      } else if (str_dst_type == "INT4") {
        dtype = ge::DT_INT4;
      }
    }
  }
  op_dest.SetAttr("dst_type", dtype);
  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("offset", offset);
  op_dest.SetAttr("sqrt_mode", sqrtmode);
  op_dest.SetAttr("round_mode", round_mode);
}

static Status ParseParamsAscendQuant(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  SetAttrFun(node, op_dest);

  for (size_t i = 0; i < op_dest.GetInputsSize(); i++){
    auto tensor = op_dest.GetInputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_x = op_dest.UpdateInputDesc(i, tensor);
    if (ret_x != ge::GRAPH_SUCCESS){
      OP_LOGE("Quant", "update input format failed.");
      return FAILED;
    }
  }
  for (size_t i = 0; i < op_dest.GetOutputsSize(); i++){
    auto tensor = op_dest.GetOutputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_y = op_dest.UpdateOutputDesc(i, tensor);
    if (ret_y != ge::GRAPH_SUCCESS){
      OP_LOGE("Quant", "update output format failed.");
      return FAILED;
    }
  }

  return SUCCESS;
}

// register AscendQuant op info to GE
REGISTER_CUSTOM_OP("AscendQuant")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::AscendQuant"),
                   ge::AscendString("ai.onnx::9::AscendQuant"),
                   ge::AscendString("ai.onnx::10::AscendQuant"),
                   ge::AscendString("ai.onnx::11::AscendQuant"),
                   ge::AscendString("ai.onnx::12::AscendQuant"),
                   ge::AscendString("ai.onnx::13::AscendQuant"),
                   ge::AscendString("ai.onnx::14::AscendQuant"),
                   ge::AscendString("ai.onnx::15::AscendQuant"),
                   ge::AscendString("ai.onnx::16::AscendQuant"),
                   ge::AscendString("ai.onnx::17::AscendQuant"),
                   ge::AscendString("ai.onnx::18::AscendQuant"),
                   ge::AscendString("ai.onnx::19::AscendQuant"),
                   ge::AscendString("ai.onnx::20::AscendQuant"),
                   ge::AscendString("ai.onnx::21::AscendQuant"),
                   ge::AscendString("ai.onnx::22::AscendQuant")})
    .ParseParamsFn(ParseParamsAscendQuant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
