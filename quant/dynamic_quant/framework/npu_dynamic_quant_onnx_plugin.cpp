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

using namespace ge;

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status  ParseParamsNpuDynamicQuant(const Message* op_src, ge::Operator& op_dest) {
    const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
    if (node == nullptr) {
      OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
      return FAILED;
    }
    return SUCCESS;
}

// register npu_dynamic_quant op info to GE
REGISTER_CUSTOM_OP("DynamicQuant")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUDynamicQuant"),
                 ge::AscendString("ai.onnx::11::NPUDynamicQuant"),
                 ge::AscendString("ai.onnx::12::NPUDynamicQuant"),
                 ge::AscendString("ai.onnx::13::NPUDynamicQuant"),
                 ge::AscendString("ai.onnx::14::NPUDynamicQuant"),
                 ge::AscendString("ai.onnx::15::NPUDynamicQuant"),
                 ge::AscendString("ai.onnx::16::NPUDynamicQuant"),
                 ge::AscendString("ai.onnx::17::NPUDynamicQuant"),
                 ge::AscendString("ai.onnx::18::NPUDynamicQuant"),
                })
  .ParseParamsFn(ParseParamsNpuDynamicQuant)
  .ImplyType(ImplyType::TVM);

// register npu_dynamic_quant op info to GE
REGISTER_CUSTOM_OP("DynamicQuantV2")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUDynamicQuantV2"),
                 ge::AscendString("ai.onnx::11::NPUDynamicQuantV2"),
                 ge::AscendString("ai.onnx::12::NPUDynamicQuantV2"),
                 ge::AscendString("ai.onnx::13::NPUDynamicQuantV2"),
                 ge::AscendString("ai.onnx::14::NPUDynamicQuantV2"),
                 ge::AscendString("ai.onnx::15::NPUDynamicQuantV2"),
                 ge::AscendString("ai.onnx::16::NPUDynamicQuantV2"),
                 ge::AscendString("ai.onnx::17::NPUDynamicQuantV2"),
                 ge::AscendString("ai.onnx::18::NPUDynamicQuantV2"),
                })
  .ParseParamsFn(ParseParamsNpuDynamicQuant)
  .ImplyType(ImplyType::TVM);
} // namespace domi