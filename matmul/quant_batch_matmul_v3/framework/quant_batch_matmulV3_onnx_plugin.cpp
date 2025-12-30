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
 * \file ops\built-in\framework\onnx_plugin\custom\quant_batch_matmulV3_plugin.cc
 * \brief
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsQuantBatchMatMulV3(const Message *opSrc, ge::Operator &opDst) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(opSrc);
  if (node == nullptr) {
    OP_LOGE(GetOpName(opDst).c_str(), "Dynamic cast opSrc to NodeProto failed.");
    return FAILED;
  }

  int a_dtype = 1;
  bool trans_x1 = false;
  bool trans_x2 = false;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
      a_dtype = attr.i();
      continue;
    }
    if (attr.name() == "transpose_x1" && attr.type() == ge::onnx::AttributeProto::INT) {
      if (attr.i() == 1) {
        trans_x1 = true;
      }
      continue;
    }
    if (attr.name() == "transpose_x2" && attr.type() == ge::onnx::AttributeProto::INT) {
      if (attr.i() == 1) {
        trans_x2 = true;
      }
      continue;
    }
  }
  opDst.SetAttr("dtype", a_dtype);
  opDst.SetAttr("transpose_x1", trans_x1);
  opDst.SetAttr("transpose_x2", trans_x2);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("QuantBatchMatmulV3")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::QuantBatchMatMul"),
                 ge::AscendString("ai.onnx::9::QuantBatchMatMul"),
                 ge::AscendString("ai.onnx::10::QuantBatchMatMul"),
                 ge::AscendString("ai.onnx::11::QuantBatchMatMul"),
                 ge::AscendString("ai.onnx::12::QuantBatchMatMul"),
                 ge::AscendString("ai.onnx::13::QuantBatchMatMul"),
                 ge::AscendString("ai.onnx::14::QuantBatchMatMul"),
                 ge::AscendString("ai.onnx::15::QuantBatchMatMul"),
                 ge::AscendString("ai.onnx::16::QuantBatchMatMul")})
  .ParseParamsFn(ParseParamsQuantBatchMatMulV3)
  .ImplyType(ImplyType::TVM);
}  // namespace domi