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
// Required attr judge
static const int REQ_ATTR_NUM = 1;
static Status ParseParamsNpuAnchorResponseFlags(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  std::vector<int> v_featmap_size = {};
  std::vector<int> v_strides = {};
  int num_base_anchors = 0;
  // The initialization of required attribute count
  int req_attr_num = 0;
  for (auto attr : node->attribute()) {
    if (attr.name() == "featmap_sizes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        v_featmap_size.push_back(attr.ints(i));
      }
    } else if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        v_strides.push_back(attr.ints(i));
      }
    } else if (attr.name() == "num_base_anchors" && attr.type() == ge::onnx::AttributeProto::INT) {
      num_base_anchors = attr.i();
      op_dest.SetAttr("num_base_anchors", num_base_anchors);
      ++req_attr_num;
    }
  }

  if (v_featmap_size.empty() || v_strides.empty() || req_attr_num != REQ_ATTR_NUM) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr featmap_size, strides, num_base_anchors");
    return FAILED;
  }

  op_dest.SetAttr("featmap_size", v_featmap_size);
  op_dest.SetAttr("strides", v_strides);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("AnchorResponseFlags")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::11::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::12::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::13::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::14::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::15::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::16::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::17::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::18::NPUAnchorResponseFlags")})
    .ParseParamsFn(ParseParamsNpuAnchorResponseFlags)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
