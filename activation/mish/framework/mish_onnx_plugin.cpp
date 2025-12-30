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
static Status ParseParamsMish(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  return SUCCESS;
}
// register Mish op info to GE
REGISTER_CUSTOM_OP("Mish")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::18::Mish"),
                 ge::AscendString("npu::1::NPUMish"),
                 ge::AscendString("ai.onnx::11::NPUMish"),
                 ge::AscendString("ai.onnx::12::NPUMish"),
                 ge::AscendString("ai.onnx::13::NPUMish"),
                 ge::AscendString("ai.onnx::14::NPUMish"),
                 ge::AscendString("ai.onnx::15::NPUMish"),
                 ge::AscendString("ai.onnx::16::NPUMish"),
                 ge::AscendString("ai.onnx::17::NPUMish"),
                 ge::AscendString("ai.onnx::18::NPUMish")})
  .ParseParamsFn(ParseParamsMish)
  .ImplyType(ImplyType::TVM);
}  // namespace domi