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
static Status  ParseParamsSoftmax(const Message *op_src, ge::Operator &op_dest, std::vector<int> &v_axis) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      v_axis.push_back(attr.i());
    }
  }

  return SUCCESS;
}

static Status  ParseParamsSoftmaxV11(const Message *op_src, ge::Operator &op_dest) {
  std::vector<int> v_axis;
  auto ret = ParseParamsSoftmax(op_src, op_dest, v_axis);
  if (ret != SUCCESS) {
    OP_LOGE(GetOpName(op_dest).c_str(), "acquire attr from NodeProto failed.");
    return FAILED;
  }
  if (v_axis.empty()) {
    v_axis.push_back(1);
  }
  op_dest.SetAttr("axes", v_axis);
  op_dest.SetAttr("need_fusion", 1);
  return SUCCESS;
}

static Status  ParseParamsSoftmaxV13(const Message *op_src, ge::Operator &op_dest) {
  std::vector<int> v_axis;
  auto ret = ParseParamsSoftmax(op_src, op_dest, v_axis);
  if (ret != SUCCESS) {
    OP_LOGE(GetOpName(op_dest).c_str(), "acquire attr from NodeProto failed.");
    return FAILED;
  }
  if (v_axis.empty()) {
    // default value in version 13.
    v_axis.push_back(-1);
  }
  op_dest.SetAttr("axes", v_axis);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("SoftmaxV2")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::Softmax"),
                 ge::AscendString("ai.onnx::9::Softmax"),
                 ge::AscendString("ai.onnx::10::Softmax"),
                 ge::AscendString("ai.onnx::11::Softmax"),
                 ge::AscendString("ai.onnx::12::Softmax")})
  .ParseParamsFn(ParseParamsSoftmaxV11)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("SoftmaxV2")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::13::Softmax"),
                 ge::AscendString("ai.onnx::14::Softmax"),
                 ge::AscendString("ai.onnx::15::Softmax"),
                 ge::AscendString("ai.onnx::16::Softmax"),
                 ge::AscendString("ai.onnx::17::Softmax"),
                 ge::AscendString("ai.onnx::18::Softmax")})
  .ParseParamsFn(ParseParamsSoftmaxV13)
  .ImplyType(ImplyType::TVM);
}  // namespace domi