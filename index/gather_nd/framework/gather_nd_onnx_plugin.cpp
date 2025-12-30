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

static Status ParseParamsGatherNd(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int batch_dims = 0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "batch_dims") {
      if (attr.i() != 0) {
        OP_LOGE(GetOpName(op_dest).c_str(), "only support batch_dims=0");
	      return FAILED;
      }
      batch_dims = attr.i();
    }
  }
  op_dest.SetAttr("batch_dims", batch_dims);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("GatherNd")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::GatherND"),
                   ge::AscendString("ai.onnx::9::GatherND"),
                   ge::AscendString("ai.onnx::10::GatherND"),
                   ge::AscendString("ai.onnx::11::GatherND"),
                   ge::AscendString("ai.onnx::12::GatherND"),
                   ge::AscendString("ai.onnx::13::GatherND"),
                   ge::AscendString("ai.onnx::14::GatherND"),
                   ge::AscendString("ai.onnx::15::GatherND"),
                   ge::AscendString("ai.onnx::16::GatherND"),
                   ge::AscendString("ai.onnx::17::GatherND"),
                   ge::AscendString("ai.onnx::18::GatherND")})
    .ParseParamsFn(ParseParamsGatherNd)
    .ImplyType(ImplyType::TVM);
}  // namespace domi