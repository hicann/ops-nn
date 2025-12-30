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
 * \file softmax_cross_entropy_loss.cc
 * \brief
 */
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsSoftmaxCrossEntropyLoss(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  for (const auto &attr : node->attribute()) {
    if (attr.name() == "ignore_index" && attr.type() == ge::onnx::AttributeProto::INT) {
      int64_t ignore_index = attr.i();
      op_dest.SetAttr("ignore_index", ignore_index);
    } 
    if (attr.name() == "reduction" && attr.type() == ge::onnx::AttributeProto::STRING) {
      std::string reduction = attr.s();
      op_dest.SetAttr("reduction", reduction);
    }
  }

  return SUCCESS;
}
// register SoftmaxCrossEntropyLoss op info to GE
REGISTER_CUSTOM_OP("SoftmaxCrossEntropyLoss")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::12::SoftmaxCrossEntropyLoss"),
                 ge::AscendString("ai.onnx::13::SoftmaxCrossEntropyLoss"),
                 ge::AscendString("ai.onnx::14::SoftmaxCrossEntropyLoss"),
                 ge::AscendString("ai.onnx::15::SoftmaxCrossEntropyLoss"),
                 ge::AscendString("ai.onnx::16::SoftmaxCrossEntropyLoss"),
                 ge::AscendString("ai.onnx::17::SoftmaxCrossEntropyLoss"),
                 ge::AscendString("ai.onnx::18::SoftmaxCrossEntropyLoss")})
  .ParseParamsFn(ParseParamsSoftmaxCrossEntropyLoss)
  .ImplyType(ImplyType::TVM);
}  // namespace domi