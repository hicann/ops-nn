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
static Status  ParseParamsIf(const Message *op_src, ge::Operator &op_dest) {
  const ge::onnx::NodeProto *node = dynamic_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int in_size = node->input_size() - 1;
  if (in_size < 0) {
    OP_LOGE(GetOpName(op_dest).c_str(), "If input num is less than 0.");
    return FAILED;
  }
  int out_size = node->output_size();
  (void)op_dest.DynamicInputRegister("input", in_size);
  (void)op_dest.DynamicOutputRegister("output", out_size);
  return SUCCESS;
}
static Status  ParseSubgraphPostFnIf(const ge::AscendString& subgraph_name, const ge::Graph& graph) {
  AutoMappingSubgraphIOIndexFunc auto_mapping_subgraph_index_func =
    FrameworkRegistry::Instance().GetAutoMappingSubgraphIOIndexFunc(ONNX);
  if (auto_mapping_subgraph_index_func == nullptr) {
    OP_LOGE("If", "auto mapping subgraph func is nullptr!");
    return FAILED;
  }
  return auto_mapping_subgraph_index_func(graph,
      [&](int data_index, int &parent_index) -> Status {
        parent_index = data_index + 1;
        return SUCCESS;
      },
      [&](int output_index, int &parent_index) -> Status {
        parent_index = output_index;
        return SUCCESS;
      });
}

// register if op info to GE
REGISTER_CUSTOM_OP("If")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::If"),
                 ge::AscendString("ai.onnx::9::If"),
                 ge::AscendString("ai.onnx::10::If"),
                 ge::AscendString("ai.onnx::11::If"),
                 ge::AscendString("ai.onnx::12::If"),
                 ge::AscendString("ai.onnx::13::If"),
                 ge::AscendString("ai.onnx::14::If"),
                 ge::AscendString("ai.onnx::15::If"),
                 ge::AscendString("ai.onnx::16::If"),
                 ge::AscendString("ai.onnx::17::If"),
                 ge::AscendString("ai.onnx::18::If")})
  .ParseParamsFn(ParseParamsIf)
  .ParseSubgraphPostFn(ParseSubgraphPostFnIf)
  .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
