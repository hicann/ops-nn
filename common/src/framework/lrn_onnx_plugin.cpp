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
static const uint32_t DIVID_SIZE = 2;
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsLRN(const Message *op_src, ge::Operator &op_dst) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dst).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float alpha = 0.0001;
  float beta = 0.75;
  float bias = 1.0;
  int32_t size = -1;
  int32_t depth_radius = -1;
  bool bfind_size = false;

  for (auto attr : node->attribute()) {
    if (attr.name() == "alpha") {
      alpha = attr.f();
    } else if (attr.name() == "beta") {
      beta = attr.f();
    } else if (attr.name() == "bias") {
      bias = attr.f();
    } else if (attr.name() == "size") {
      bfind_size = true;
      size = attr.i();
    }
  }

  if (!bfind_size) {
    OP_LOGE(GetOpName(op_dst).c_str(), "Message do not have attr size");
    return FAILED;
  }

  if (size == 0) {
    OP_LOGE(GetOpName(op_dst).c_str(), "Attr size must be odd num");
    return FAILED;
  }
  alpha = alpha / size;
  depth_radius = (size - 1) / DIVID_SIZE;
  op_dst.SetAttr("depth_radius", depth_radius);
  op_dst.SetAttr("bias", bias);
  op_dst.SetAttr("alpha", alpha);
  op_dst.SetAttr("beta", beta);
  op_dst.SetAttr("norm_region", "ACROSS_CHANNELS");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LRN")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::LRN"),
                 ge::AscendString("ai.onnx::9::LRN"),
                 ge::AscendString("ai.onnx::10::LRN"),
                 ge::AscendString("ai.onnx::11::LRN"),
                 ge::AscendString("ai.onnx::12::LRN"),
                 ge::AscendString("ai.onnx::13::LRN"),
                 ge::AscendString("ai.onnx::14::LRN"),
                 ge::AscendString("ai.onnx::15::LRN"),
                 ge::AscendString("ai.onnx::16::LRN"),
                 ge::AscendString("ai.onnx::17::LRN"),
                 ge::AscendString("ai.onnx::18::LRN")})
  .ParseParamsFn(ParseParamsLRN)
  .ImplyType(ImplyType::TVM);
}  //  namespace domi
