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
using OpDesc = std::shared_ptr<ge::OpDesc>;
static Status ParseParamsLogSoftmax(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(),
            "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int axis = -1;
  for (auto attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis = attr.i();
    }
  }
  std::vector<int> axes(1, axis);
  op_dest.SetAttr("axes", axes);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LogSoftmaxV2")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::LogSoftmax"),
                 ge::AscendString("ai.onnx::9::LogSoftmax"),
                 ge::AscendString("ai.onnx::10::LogSoftmax"),
                 ge::AscendString("ai.onnx::11::LogSoftmax"),
                 ge::AscendString("ai.onnx::12::LogSoftmax"),
                 ge::AscendString("ai.onnx::13::LogSoftmax"),
                 ge::AscendString("ai.onnx::14::LogSoftmax"),
                 ge::AscendString("ai.onnx::15::LogSoftmax"),
                 ge::AscendString("ai.onnx::16::LogSoftmax"),
                 ge::AscendString("ai.onnx::17::LogSoftmax"),
                 ge::AscendString("ai.onnx::18::LogSoftmax")})
  .ParseParamsFn(ParseParamsLogSoftmax)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
