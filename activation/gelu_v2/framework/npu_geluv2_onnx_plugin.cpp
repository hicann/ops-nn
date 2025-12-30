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
using NodeProto = ge::onnx::NodeProto;

Status ParseParamsNpuGeluV2(const ge::Operator& op_src, ge::Operator& op_dest) {
    AscendString attrsString;
    std::string approximateMode = "none";

    if (op_src.GetAttr("attribute", attrsString) == ge::GRAPH_SUCCESS) {
        json attrs = json::parse(attrsString.GetString());
        for (json& attr : attrs["attribute"]) {
            if (attr["name"] == "approximate") {
                approximateMode = attr["s"];
            }
        }
    }
  
    op_dest.SetAttr("approximate", approximateMode);
    return SUCCESS;
}


// register npu_gelu_v2 op info to GE
REGISTER_CUSTOM_OP("GeluV2")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUGeluV2")})
  .ParseParamsByOperatorFn(ParseParamsNpuGeluV2)
  .ImplyType(ImplyType::TVM);
} // domi