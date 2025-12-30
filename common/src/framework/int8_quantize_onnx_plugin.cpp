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

static Status ParseParamsInt8Quantize(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("Int8Quantize", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float scale = 0.0;
  float offset = 0.0;
  bool sqrt_mode = false;
  std::string round_mode = "Round";
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "Y_scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale = attr.f();
    } else if (attr.name() == "Y_zero_point" && attr.type() == ge::onnx::AttributeProto::INT) {
      offset = static_cast<float>(attr.i());
    }
  }

  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("offset", offset);
  op_dest.SetAttr("sqrt_mode", sqrt_mode);
  op_dest.SetAttr("round_mode", round_mode);

  for (size_t i = 0; i < op_dest.GetInputsSize(); i++) {
    auto tensor = op_dest.GetInputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_x = op_dest.UpdateInputDesc(i, tensor);
    if (ret_x != ge::GRAPH_SUCCESS) {
      OP_LOGE("Int8Quantize", "update input format failed.");
      return FAILED;
    }
  }
  for (size_t i = 0; i < op_dest.GetOutputsSize(); i++) {
    auto tensor = op_dest.GetOutputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_y = op_dest.UpdateOutputDesc(i, tensor);
    if (ret_y != ge::GRAPH_SUCCESS) {
      OP_LOGE("Int8Quantize", "update output format failed.");
      return FAILED;
    }
  }

  return SUCCESS;
}

// register AscendQuant op info to GE
REGISTER_CUSTOM_OP("AscendQuant")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Int8Quantize"), 
                   ge::AscendString("ai.onnx::9::Int8Quantize"), 
                   ge::AscendString("ai.onnx::10::Int8Quantize"),    
                   ge::AscendString("ai.onnx::11::Int8Quantize"), 
                   ge::AscendString("ai.onnx::12::Int8Quantize"), 
                   ge::AscendString("ai.onnx::13::Int8Quantize"),   
                   ge::AscendString("ai.onnx::14::Int8Quantize"), 
                   ge::AscendString("ai.onnx::15::Int8Quantize"), 
                   ge::AscendString("ai.onnx::16::Int8Quantize")})
    .ParseParamsFn(ParseParamsInt8Quantize)
    .ImplyType(ImplyType::TVM);
}  // namespace domi