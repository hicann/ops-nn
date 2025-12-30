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
static Status ParseParamsElu(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float alpha_value = 1.0;
  op_dest.SetAttr("alpha", alpha_value);
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "alpha" &&
        attr.type() == ge::onnx::AttributeProto::FLOAT) {
      alpha_value = attr.f();
      op_dest.SetAttr("alpha", alpha_value);
    }
  }
  return SUCCESS;
}
// register Elu op info to GE
REGISTER_CUSTOM_OP("Elu")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::Elu"),
                 ge::AscendString("ai.onnx::9::Elu"),
                 ge::AscendString("ai.onnx::10::Elu"),
                 ge::AscendString("ai.onnx::11::Elu"),
                 ge::AscendString("ai.onnx::12::Elu"),
                 ge::AscendString("ai.onnx::13::Elu"),
                 ge::AscendString("ai.onnx::14::Elu"),
                 ge::AscendString("ai.onnx::15::Elu"),
                 ge::AscendString("ai.onnx::16::Elu"),
                 ge::AscendString("ai.onnx::17::Elu"),
                 ge::AscendString("ai.onnx::18::Elu")})
  .ParseParamsFn(ParseParamsElu)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
