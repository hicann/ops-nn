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
static Status ParseOnnxParamsGroupNormRelu(const ge::Operator& op_src, ge::Operator& op_dest)
{
    AscendString attrs_string;
    int num_groups = 0;
    float eps = 0;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        json attrs = json::parse(attrs_string.GetString());
        for (json& attr : attrs["attribute"]) {
            if (attr["name"] == "eps") {
                std::string eps_str = attr["f"];
                eps = atof(eps_str.c_str());
            } else if (attr["name"] == "num_groups") {
                num_groups = attr["i"];
            }
        }
    }

    op_dest.SetAttr("num_groups", num_groups);
    op_dest.SetAttr("eps", eps);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("GroupNormRelu")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::GroupNormRelu"), 
                   ge::AscendString("ai.onnx::9::GroupNormRelu"),
                   ge::AscendString("ai.onnx::10::GroupNormRelu"), 
                   ge::AscendString("ai.onnx::11::GroupNormRelu"),
                   ge::AscendString("ai.onnx::12::GroupNormRelu"), 
                   ge::AscendString("ai.onnx::13::GroupNormRelu"),
                   ge::AscendString("ai.onnx::14::GroupNormRelu"), 
                   ge::AscendString("ai.onnx::15::GroupNormRelu"),
                   ge::AscendString("ai.onnx::16::GroupNormRelu")})
    .ParseParamsByOperatorFn(ParseOnnxParamsGroupNormRelu)
    .ImplyType(ImplyType::TVM);
} // namespace domi