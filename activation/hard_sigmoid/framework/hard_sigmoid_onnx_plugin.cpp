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
 * \file hard_sigmoid_plugin.cc
 * \brief
 */

#include "onnx_common.h"

namespace domi {

static Status parse_params_hard_sigmoid(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float alpha = 0.2;
  float beta = 0.5;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "alpha" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      alpha = static_cast<float>(attr.f());
    } else if (attr.name() == "beta" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      beta = static_cast<float>(attr.f());
    }
  }
  op_dest.SetAttr("alpha", alpha);
  op_dest.SetAttr("beta", beta);
  return SUCCESS;
}

// register HardSigmoid op info to GE
REGISTER_CUSTOM_OP("HardSigmoid")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::1::HardSigmoid"),
                 ge::AscendString("ai.onnx::6::HardSigmoid"),
                 ge::AscendString("ai.onnx::8::HardSigmoid"),
                 ge::AscendString("ai.onnx::9::HardSigmoid"),
                 ge::AscendString("ai.onnx::10::HardSigmoid"),
                 ge::AscendString("ai.onnx::11::HardSigmoid"),
                 ge::AscendString("ai.onnx::12::HardSigmoid"),
                 ge::AscendString("ai.onnx::13::HardSigmoid"),
                 ge::AscendString("ai.onnx::14::HardSigmoid"),
                 ge::AscendString("ai.onnx::15::HardSigmoid"),
                 ge::AscendString("ai.onnx::16::HardSigmoid"),
                 ge::AscendString("ai.onnx::17::HardSigmoid"),
                 ge::AscendString("ai.onnx::18::HardSigmoid")})
  .ParseParamsFn(parse_params_hard_sigmoid)
  .ImplyType(ImplyType::TVM);
}  // namespace domi