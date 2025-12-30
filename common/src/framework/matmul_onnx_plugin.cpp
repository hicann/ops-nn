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
static Status ParseParamsMatMul(const Message* op_src, ge::Operator& op_dest)
{
  ge::AscendString op_name;
  if (op_dest.GetName(op_name) != ge::GRAPH_SUCCESS) {
    OP_LOGE("", "failed to get op_name");
    return FAILED;
  }
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(op_name.GetString(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
// add the attr to support the custom matmul transpose fusion
  bool trans_a = false;
  bool trans_b = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "transA" && attr.i() != 0) {
      trans_a = true;
    }
    if (attr.name() == "transB" && attr.i() != 0) {
      trans_b = true;
    }
  }

  op_dest.SetAttr("adj_x1", trans_a);
  op_dest.SetAttr("adj_x2", trans_b);
  if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
    OP_LOGE(op_name.GetString(), "failed to change format.");
    return FAILED;
  }
  return SUCCESS;
}

// register MatMul op info to GE
REGISTER_CUSTOM_OP("BatchMatMulV2")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::MatMul"),
                   ge::AscendString("ai.onnx::9::MatMul"),
                   ge::AscendString("ai.onnx::10::MatMul"),
                   ge::AscendString("ai.onnx::11::MatMul"),
                   ge::AscendString("ai.onnx::12::MatMul"),
                   ge::AscendString("ai.onnx::13::MatMul"),
                   ge::AscendString("ai.onnx::14::MatMul"),
                   ge::AscendString("ai.onnx::15::MatMul"),
                   ge::AscendString("ai.onnx::16::MatMul"),
                   ge::AscendString("ai.onnx::17::MatMul"),
                   ge::AscendString("ai.onnx::18::MatMul")})
    .ParseParamsFn(ParseParamsMatMul)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
