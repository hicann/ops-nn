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
#include "nlohmann/json.hpp"

using namespace ge;
using json = nlohmann::json;

namespace domi {

static Status  ParseParamsAntiQuant(const ge::Operator& op_src, ge::Operator& op_dest) {
    AscendString attrs_string;
    int dst_dtype = ge::DT_FLOAT16;
    constexpr bool sqrt_mode = false;

    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
      json attrs = json::parse(attrs_string.GetString());
      for (json& attr : attrs["attribute"]) {
          if (attr["name"] == "dst_dtype") {
              dst_dtype = attr["i"];
          }
      }
    }

    op_dest.SetAttr("dst_type", dst_dtype);
    op_dest.SetAttr("sqrt_mode", sqrt_mode);

    ge::TensorDesc tensor_desc = op_dest.GetOutputDesc(0);
    tensor_desc.SetDataType(static_cast<ge::DataType>(dst_dtype));
    auto ret_y = op_dest.UpdateOutputDesc("y", tensor_desc);
    if (ret_y != ge::GRAPH_SUCCESS) {
      ge::AscendString name = "";
      (void)tensor_desc.GetName(name);
      OP_LOGE(name.GetString(), "change output format failed.");
      return FAILED;
    }

    return SUCCESS;
}

// register npu_anti_quant op info to GE
REGISTER_CUSTOM_OP("AscendAntiQuantV2")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUAntiQuant"),
                 ge::AscendString("ai.onnx::11::NPUAntiQuant"),
                 ge::AscendString("ai.onnx::12::NPUAntiQuant"),
                 ge::AscendString("ai.onnx::13::NPUAntiQuant"),
                 ge::AscendString("ai.onnx::14::NPUAntiQuant"),
                 ge::AscendString("ai.onnx::15::NPUAntiQuant"),
                 ge::AscendString("ai.onnx::16::NPUAntiQuant"),
                 ge::AscendString("ai.onnx::17::NPUAntiQuant"),
                 ge::AscendString("ai.onnx::18::NPUAntiQuant"),
                })
  .ParseParamsByOperatorFn(ParseParamsAntiQuant)
  .ImplyType(ImplyType::TVM);
} // namespace domi
