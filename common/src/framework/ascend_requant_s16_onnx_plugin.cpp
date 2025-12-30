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

static Status ParseParamsAscendRequantS16(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool dual_output = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "dual_output" && attr.i() != 0) {
      dual_output = true;
      break;
    }
  }

  bool relu_flag = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "relu_flag" && attr.i() != 0) {
      relu_flag = true;
      break;
    }
  }

  op_dest.SetAttr("dual_output", dual_output);
  op_dest.SetAttr("relu_flag", relu_flag);

  return SUCCESS;
}

// register AscendRequantS16 op info to GE
REGISTER_CUSTOM_OP("AscendRequantS16")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::AscendRequantS16"),
                   ge::AscendString("ai.onnx::9::AscendRequantS16"),
                   ge::AscendString("ai.onnx::10::AscendRequantS16"),
                   ge::AscendString("ai.onnx::11::AscendRequantS16"),
                   ge::AscendString("ai.onnx::12::AscendRequantS16"),
                   ge::AscendString("ai.onnx::13::AscendRequantS16"),
                   ge::AscendString("ai.onnx::14::AscendRequantS16"),
                   ge::AscendString("ai.onnx::15::AscendRequantS16"),
                   ge::AscendString("ai.onnx::16::AscendRequantS16"),
                   ge::AscendString("ai.onnx::17::AscendRequantS16"),
                   ge::AscendString("ai.onnx::18::AscendRequantS16"),
                   ge::AscendString("ai.onnx::19::AscendRequantS16"),
                   ge::AscendString("ai.onnx::20::AscendRequantS16"),
                   ge::AscendString("ai.onnx::21::AscendRequantS16"),
                   ge::AscendString("ai.onnx::22::AscendRequantS16")})
    .ParseParamsFn(ParseParamsAscendRequantS16)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
