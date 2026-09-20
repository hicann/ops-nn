/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "plugin_util.h"
#include "register/register.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace domi {
static Status ParseGroupNormReluAttr(json& attr, int& num_groups, float& eps)
{
    if (attr["name"] == "eps") {
        if (attr.contains("f")) {
            std::string eps_str = attr["f"];
            if (!StrToFloat(eps_str, eps)) {
                OP_LOGE("group_normal_relu", "invalid eps value: %s", eps_str.c_str());
                return FAILED;
            }
        } else {
            // GE 序列化 float 属性时会省略值为 0 的 "f" 字段，此时 eps 实际值为 0
            eps = 0.0f;
        }
    } else if (attr["name"] == "num_groups") {
        if (attr.contains("i")) {
            num_groups = attr["i"].get<int>();
        } else {
            // GE 序列化 int 属性时会省略值为 0 的 "i" 字段，此时 num_groups 实际值为 0
            num_groups = 0;
        }
    }
    return SUCCESS;
}

static Status ParseOnnxParamsGroupNormRelu(const ge::Operator& op_src, ge::Operator& op_dest)
{
    ge::AscendString attrs_string;
    int num_groups = 0;
    float eps = 0;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());

            for (json& attr : attrs["attribute"]) {
                if (ParseGroupNormReluAttr(attr, num_groups, eps) != SUCCESS) {
                    return FAILED;
                }
            }
        }

        op_dest.SetAttr("num_groups", num_groups);
        op_dest.SetAttr("eps", eps);
    } catch (const json::parse_error& e) {
        OP_LOGE("group_normal_relu", "JSON parse error: %s", e.what());
        return FAILED;
    } catch (const json::exception& e) {
        OP_LOGE("group_normal_relu", "JSON processing error: %s", e.what());
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("GroupNormRelu")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::GroupNormRelu"), ge::AscendString("ai.onnx::9::GroupNormRelu"),
                   ge::AscendString("ai.onnx::10::GroupNormRelu"), ge::AscendString("ai.onnx::11::GroupNormRelu"),
                   ge::AscendString("ai.onnx::12::GroupNormRelu"), ge::AscendString("ai.onnx::13::GroupNormRelu"),
                   ge::AscendString("ai.onnx::14::GroupNormRelu"), ge::AscendString("ai.onnx::15::GroupNormRelu"),
                   ge::AscendString("ai.onnx::16::GroupNormRelu")})
    .ParseParamsByOperatorFn(ParseOnnxParamsGroupNormRelu)
    .ImplyType(ImplyType::TVM);
} // namespace domi
