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

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status  ParseParamsInt8GivenIntTensorFill(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE("ParseParamsInt8GivenIntTensorFill", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int32_t> vals;
  std::vector<int64_t> dims;
  for (auto &attr : node->attribute()) {
    if (attr.name() == "values") {
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        vals.push_back(attr.ints(i));
      }
    } else if(attr.name() == "shape") {
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        dims.push_back(attr.ints(i));
      }
    }
  }
  
  if (vals.empty()) {
    OP_LOGE("ParseParamsInt8GivenIntTensorFill", "Must have attr values");
    return FAILED;
  }

  if (dims.empty()) {
    dims.push_back((int64_t)vals.size());
  }

  ge::Tensor value_tensor = Vec2Tensor(vals, dims, ge::DT_INT32);
  op_dest.SetAttr("value", value_tensor);
  return SUCCESS;
}


REGISTER_CUSTOM_OP("Constant")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Int8GivenIntTensorFill"),
                   ge::AscendString("ai.onnx::9::Int8GivenIntTensorFill"),
                   ge::AscendString("ai.onnx::10::Int8GivenIntTensorFill"),
                   ge::AscendString("ai.onnx::11::Int8GivenIntTensorFill"),
                   ge::AscendString("ai.onnx::12::Int8GivenIntTensorFill"),
                   ge::AscendString("ai.onnx::13::Int8GivenIntTensorFill"),
                   ge::AscendString("ai.onnx::14::Int8GivenIntTensorFill"),
                   ge::AscendString("ai.onnx::15::Int8GivenIntTensorFill"),
                   ge::AscendString("ai.onnx::16::Int8GivenIntTensorFill")})
    .ParseParamsFn(ParseParamsInt8GivenIntTensorFill)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
