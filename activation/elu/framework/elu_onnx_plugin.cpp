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

namespace domi {
using json = nlohmann::json;
static Status ParseParamsElu(const ge::Operator& op_src, ge::Operator& op_dest)
{
    float alpha_value = 1.0;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    if (attr.value("name", "") == "alpha" && attr.contains("f")) {
                        std::string alpha_str = attr["f"];
                        if (!StrToFloat(alpha_str, alpha_value)) {
                            OP_LOGE(GetOpName(op_dest).c_str(), "invalid alpha value: %s", alpha_str.c_str());
                            return FAILED;
                        }
                    }
                }
            }
        } catch (const nlohmann::json::exception& e) {
            OP_LOGE(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
            return FAILED;
        } catch (...) {
            OP_LOGE(GetOpName(op_dest).c_str(), "get unknown exception, please check compile info json.");
            return FAILED;
        }
    }
    op_dest.SetAttr("alpha", alpha_value);
    return SUCCESS;
}
// register Elu op info to GE
REGISTER_CUSTOM_OP("Elu")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Elu"), ge::AscendString("ai.onnx::9::Elu"),
                   ge::AscendString("ai.onnx::10::Elu"), ge::AscendString("ai.onnx::11::Elu"),
                   ge::AscendString("ai.onnx::12::Elu"), ge::AscendString("ai.onnx::13::Elu"),
                   ge::AscendString("ai.onnx::14::Elu"), ge::AscendString("ai.onnx::15::Elu"),
                   ge::AscendString("ai.onnx::16::Elu"), ge::AscendString("ai.onnx::17::Elu"),
                   ge::AscendString("ai.onnx::18::Elu")})
    .ParseParamsByOperatorFn(ParseParamsElu)
    .ImplyType(ImplyType::TVM);
} // namespace domi
