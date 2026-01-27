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
 * \file npu_fast_gelu.cc
 * \brief
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status  ParseParamsNpuFastGelu(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("FastGelu")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUFastGelu"),
                 ge::AscendString("ai.onnx::11::NPUFastGelu"),
                 ge::AscendString("ai.onnx::12::NPUFastGelu"),
                 ge::AscendString("ai.onnx::13::NPUFastGelu"),
                 ge::AscendString("ai.onnx::14::NPUFastGelu"),
                 ge::AscendString("ai.onnx::15::NPUFastGelu"),
                 ge::AscendString("ai.onnx::16::NPUFastGelu"),
                 ge::AscendString("ai.onnx::17::NPUFastGelu"),
                 ge::AscendString("ai.onnx::18::NPUFastGelu")})
  .ParseParamsFn(ParseParamsNpuFastGelu)
  .ImplyType(ImplyType::TVM);
} // domi