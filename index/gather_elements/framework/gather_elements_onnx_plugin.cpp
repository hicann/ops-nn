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
 * \file gather_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {

static Status ParseParamsGatherElements(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int dim_value = 0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      dim_value = attr.i();
    }
  }
  op_dest.SetAttr("dim", dim_value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("GatherElements")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::GatherElements"),
                   ge::AscendString("ai.onnx::9::GatherElements"),
                   ge::AscendString("ai.onnx::10::GatherElements"),
                   ge::AscendString("ai.onnx::11::GatherElements"),
                   ge::AscendString("ai.onnx::12::GatherElements"),
                   ge::AscendString("ai.onnx::13::GatherElements"),
                   ge::AscendString("ai.onnx::14::GatherElements"),
                   ge::AscendString("ai.onnx::15::GatherElements"),
                   ge::AscendString("ai.onnx::16::GatherElements"),
                   ge::AscendString("ai.onnx::17::GatherElements"),
                   ge::AscendString("ai.onnx::18::GatherElements")})
    .ParseParamsFn(ParseParamsGatherElements)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
