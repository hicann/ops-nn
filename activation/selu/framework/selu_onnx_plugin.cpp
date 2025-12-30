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
 * \file selu_plugin.cpp
 * \brief
 */

#include <cmath>
#include "onnx_common.h"

namespace domi {
static const float ALPHA_DEFAULT = 1.67326324235;
static const float GAMMA_DEFAULT = 1.05070098736;
static const float DIFF_ABS = 0.00001;
static Status ParseParamsSelu(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float alpha = ALPHA_DEFAULT;
  float gamma = GAMMA_DEFAULT;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "alpha" && attr.type() == ge::onnx::AttributeProto::FLOAT) alpha = attr.f();
    if (attr.name() == "gamma" && attr.type() == ge::onnx::AttributeProto::FLOAT) gamma = attr.f();
  }
  if(fabs(alpha - ALPHA_DEFAULT) > DIFF_ABS || fabs(gamma - GAMMA_DEFAULT) > DIFF_ABS) {
    OP_LOGE(GetOpName(op_dest).c_str(), "the attr of alpha or gamma is not default. transform failed.");
    return FAILED;
  }

  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Selu")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Selu"),
                   ge::AscendString("ai.onnx::9::Selu"),
                   ge::AscendString("ai.onnx::10::Selu"),
                   ge::AscendString("ai.onnx::11::Selu"),
                   ge::AscendString("ai.onnx::12::Selu"),
                   ge::AscendString("ai.onnx::13::Selu"),
                   ge::AscendString("ai.onnx::14::Selu"),
                   ge::AscendString("ai.onnx::15::Selu"),
                   ge::AscendString("ai.onnx::16::Selu"),
                   ge::AscendString("ai.onnx::17::Selu"),
                   ge::AscendString("ai.onnx::18::Selu")})
    .ParseParamsFn(ParseParamsSelu)
    .ImplyType(ImplyType::TVM);
}  // namespace domi