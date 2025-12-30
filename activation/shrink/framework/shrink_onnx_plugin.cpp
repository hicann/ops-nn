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
static const float BIAS_DEFAULT = 0.0;
static const float LAMBD_DEFAULT = 0.5;

static Status ParseParamsShrink(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float bias = BIAS_DEFAULT;
  float lambd = LAMBD_DEFAULT;
  
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "bias" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
        bias = attr.f();
    } 
    if (attr.name() == "lambd" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
        lambd = attr.f();
    }
  }
  op_dest.SetAttr("bias", bias);
  op_dest.SetAttr("lambd", lambd);
  
  return SUCCESS;
}

// register Shrink op info to GE
REGISTER_CUSTOM_OP("Shrink")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::9::Shrink"),
                   ge::AscendString("ai.onnx::10::Shrink"),
                   ge::AscendString("ai.onnx::11::Shrink"),
                   ge::AscendString("ai.onnx::12::Shrink"),
                   ge::AscendString("ai.onnx::13::Shrink"),
                   ge::AscendString("ai.onnx::14::Shrink"),
                   ge::AscendString("ai.onnx::15::Shrink"),
                   ge::AscendString("ai.onnx::16::Shrink"),
                   ge::AscendString("ai.onnx::17::Shrink"),
                   ge::AscendString("ai.onnx::18::Shrink")})
    .ParseParamsFn(ParseParamsShrink)
    .ImplyType(ImplyType::TVM);
}  // namespace domi