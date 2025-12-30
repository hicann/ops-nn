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

static Status ParseParamsBoundingBoxEncode(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("BoundingBoxEncode", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<float> means;
  std::vector<float> stds;
  std::map<string, float> attr_map = {{"means0", 0.0f}, {"means1", 0.0f}, {"means2", 0.0f}, {"means3", 0.0f},
                                      {"stds0", 1.0f}, {"stds1", 1.0f}, {"stds2", 1.0f}, {"stds3", 1.0f}};

  for (const auto& attr : node->attribute()) {
    if (attr_map.find(attr.name()) != attr_map.end() && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      attr_map[attr.name()] = attr.f();
    }
  }
  means.push_back(attr_map["means0"]);
  means.push_back(attr_map["means1"]);
  means.push_back(attr_map["means2"]);
  means.push_back(attr_map["means3"]);
  stds.push_back(attr_map["stds0"]);
  stds.push_back(attr_map["stds1"]);
  stds.push_back(attr_map["stds2"]);
  stds.push_back(attr_map["stds3"]);

  op_dest.SetAttr("means", means);
  op_dest.SetAttr("stds", stds);
  return SUCCESS;
}

// register Yolo op info to GE
REGISTER_CUSTOM_OP("BoundingBoxEncode")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::11::NPUBoundingBoxEncode"),
                   ge::AscendString("ai.onnx::12::NPUBoundingBoxEncode"),
                   ge::AscendString("ai.onnx::13::NPUBoundingBoxEncode"),
                   ge::AscendString("ai.onnx::14::NPUBoundingBoxEncode"),
                   ge::AscendString("ai.onnx::15::NPUBoundingBoxEncode"),
                   ge::AscendString("ai.onnx::16::NPUBoundingBoxEncode"),
                   ge::AscendString("ai.onnx::17::NPUBoundingBoxEncode"),
                   ge::AscendString("ai.onnx::18::NPUBoundingBoxEncode"),
                   ge::AscendString("npu::1::NPUBoundingBoxEncode")})
    .ParseParamsFn(ParseParamsBoundingBoxEncode)
    .ImplyType(ImplyType::TVM);
}  // namespace domi