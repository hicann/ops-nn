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
static Status  ParseParamsBatchMatMul(const Message *opSrc, ge::Operator &opDst) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(opSrc);
  if (node == nullptr) {
    OP_LOGE(GetOpName(opDst).c_str(), "Dynamic cast opSrc to NodeProto failed.");
    return FAILED;
  }
  // onnx doesn't have transpose attr
  opDst.SetAttr("adj_x1", false);
  opDst.SetAttr("adj_x2", false);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("BatchMatMul")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::BatchMatMul"),
                 ge::AscendString("ai.onnx::9::BatchMatMul"),
                 ge::AscendString("ai.onnx::10::BatchMatMul"),
                 ge::AscendString("ai.onnx::11::BatchMatMul"),
                 ge::AscendString("ai.onnx::12::BatchMatMul"),
                 ge::AscendString("ai.onnx::13::BatchMatMul"),
                 ge::AscendString("ai.onnx::14::BatchMatMul"),
                 ge::AscendString("ai.onnx::15::BatchMatMul"),
                 ge::AscendString("ai.onnx::16::BatchMatMul")})
  .ParseParamsFn(ParseParamsBatchMatMul)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
