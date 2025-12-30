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
static const int32_t DTYPE_INT8 = 2;

static Status ParseParamsNpuGroupQuant(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  ge::DataType attrDstType = ge::DT_INT8;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "dst_type") {
      int32_t iDstType = attr.i();
      if (iDstType != DTYPE_INT8) {
        OP_LOGE(GetOpName(op_dest).c_str(), "attr dst_type only support 2(int8)");
        return FAILED;
      }
      attrDstType = static_cast<ge::DataType>(iDstType);
    }
  }
  op_dest.SetAttr("dst_type", attrDstType);

  return SUCCESS;
}

// register GroupQuant op info to GE
REGISTER_CUSTOM_OP("GroupQuant")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::8::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::9::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::10::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::11::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::12::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::13::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::14::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::15::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::16::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::17::NPUGroupQuant"),
                   ge::AscendString("ai.onnx::18::NPUGroupQuant")})
    .ParseParamsFn(ParseParamsNpuGroupQuant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi