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

// Required attr judge
static const int REQ_ATTR_NUM = 1;

static Status  ParseParamsNPUScatter(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  // The initialization of required attribute count
  int req_attr_count = 0;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "dim" && attr.type() == ge::onnx::AttributeProto::INT) {
      int dim = attr.i();
      op_dest.SetAttr("dimension", dim);
      ++req_attr_count;
    }
  }

  // Node must have required attribute dim
  if (req_attr_count != REQ_ATTR_NUM) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr dim.");
    return FAILED;
  }

  return SUCCESS;
}

// register Scatter op info to GE
REGISTER_CUSTOM_OP("ArgMaxGrad")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUScatter"), 
                   ge::AscendString("ai.onnx::11::NPUScatter"), 
                   ge::AscendString("ai.onnx::12::NPUScatter"),
                   ge::AscendString("ai.onnx::13::NPUScatter"), 
                   ge::AscendString("ai.onnx::14::NPUScatter"), 
                   ge::AscendString("ai.onnx::15::NPUScatter"),
                   ge::AscendString("ai.onnx::16::NPUScatter"), 
                   ge::AscendString("ai.onnx::17::NPUScatter"),
                   ge::AscendString("ai.onnx::18::NPUScatter")})
    .ParseParamsFn(ParseParamsNPUScatter)
    .ImplyType(ImplyType::TVM);
}  // namespace domi