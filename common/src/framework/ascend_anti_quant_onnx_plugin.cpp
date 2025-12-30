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

static Status ParseParamsAscendAntiQuant(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float scale = 0.0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale = attr.f();
      break;
    }
  }

  float offset = 0.0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "offset" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      offset = attr.f()*(-1);
      break;
    }
  }

  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("offset", offset);

  int dtype = 0;  // DT_FLOAT
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
      dtype = attr.i();
      break;
    }
  }

  bool sqrt_mode = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "sqrt_mode" && attr.i() != 0) {
      sqrt_mode = true;
      break;
    }
  }

  op_dest.SetAttr("dtype", dtype);
  op_dest.SetAttr("sqrt_mode", sqrt_mode);

  return SUCCESS;
}

// register AscendAntiQuant op info to GE
REGISTER_CUSTOM_OP("AscendAntiQuant")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::9::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::10::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::11::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::12::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::13::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::14::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::15::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::16::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::17::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::18::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::19::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::20::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::21::AscendAntiQuant"),
                   ge::AscendString("ai.onnx::22::AscendAntiQuant")})
    .ParseParamsFn(ParseParamsAscendAntiQuant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi