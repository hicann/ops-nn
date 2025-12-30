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

static const int dtype_int8 = 2;
static const int dtype_int32 = 3;

static Status ParseParamsNpuQuantize(const Message *opSrc, ge::Operator &opDest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(opSrc);
  if (node == nullptr) {
    OP_LOGE(GetOpName(opDest).c_str(), "Dynamic cast opSrc to NodeProto failed.");
    return FAILED;
  }

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      int axis = attr.i();
      opDest.SetAttr("axis", axis);
    } else if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
      int dtype = attr.i();
      if (dtype == dtype_int8) {
        opDest.SetAttr("dtype", "torch.qint8");
      } else if (dtype == dtype_int32) {
        opDest.SetAttr("dtype", "torch.qint32");
      } else {
        opDest.SetAttr("dtype", "torch.quint8");        
      }
    }
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Quantize")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUQuantize"),
                 ge::AscendString("ai.onnx::11::NPUQuantize"),
                 ge::AscendString("ai.onnx::12::NPUQuantize"),
                 ge::AscendString("ai.onnx::13::NPUQuantize"),
                 ge::AscendString("ai.onnx::14::NPUQuantize"),
                 ge::AscendString("ai.onnx::15::NPUQuantize"),
                 ge::AscendString("ai.onnx::16::NPUQuantize"),
                 ge::AscendString("ai.onnx::17::NPUQuantize"),
                 ge::AscendString("ai.onnx::18::NPUQuantize")})
  .ParseParamsFn(ParseParamsNpuQuantize)
  .ImplyType(ImplyType::TVM);
} // domi