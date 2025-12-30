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

static Status ParseParamsAscendDequant(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int dtype = 0;  // DT_FLOAT
  bool sqrt_mode = false;
  bool relu_flag = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
      dtype = attr.i();
    } else if (attr.name() == "sqrt_mode" && attr.i() != 0) {
      sqrt_mode = true;
    } else if (attr.name() == "relu_flag" && attr.i() != 0) {
      relu_flag = true;
    }
  }

  op_dest.SetAttr("dtype", dtype);
  op_dest.SetAttr("sqrt_mode", sqrt_mode);
  op_dest.SetAttr("relu_flag", relu_flag);

  for (size_t i = 0; i < op_dest.GetInputsSize(); i++){
    auto tensor = op_dest.GetInputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_x = op_dest.UpdateInputDesc(i, tensor);
    if (ret_x != ge::GRAPH_SUCCESS){
      OP_LOGE("Dequant", "update input format failed.");
      return FAILED;
    }
  }
  for (size_t i = 0; i < op_dest.GetOutputsSize(); i++){
    auto tensor = op_dest.GetOutputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_y = op_dest.UpdateOutputDesc(i, tensor);
    if (ret_y != ge::GRAPH_SUCCESS){
      OP_LOGE("Dequant", "update output format failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

// register AscendDequant op info to GE
REGISTER_CUSTOM_OP("AscendDequant")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::AscendDequant"),
                   ge::AscendString("ai.onnx::9::AscendDequant"),
                   ge::AscendString("ai.onnx::10::AscendDequant"),
                   ge::AscendString("ai.onnx::11::AscendDequant"),
                   ge::AscendString("ai.onnx::12::AscendDequant"),
                   ge::AscendString("ai.onnx::13::AscendDequant"),
                   ge::AscendString("ai.onnx::14::AscendDequant"),
                   ge::AscendString("ai.onnx::15::AscendDequant"),
                   ge::AscendString("ai.onnx::16::AscendDequant"),
                   ge::AscendString("ai.onnx::17::AscendDequant"),
                   ge::AscendString("ai.onnx::18::AscendDequant"),
                   ge::AscendString("ai.onnx::19::AscendDequant"),
                   ge::AscendString("ai.onnx::20::AscendDequant"),
                   ge::AscendString("ai.onnx::21::AscendDequant"),
                   ge::AscendString("ai.onnx::22::AscendDequant")})
    .ParseParamsFn(ParseParamsAscendDequant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
