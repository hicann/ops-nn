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

static Status ParseParamsAdaCast(const ge::Operator& op_src, ge::Operator& op_dst)
{
    int pixel_value = 65535;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    if (attr.value("name", "") == "pixel" && attr.contains("i")) {
                        pixel_value = attr["i"].get<int>();
                    }
                }
            }
        } catch (const nlohmann::json::exception& e) {
            OP_LOGE(GetOpName(op_dst).c_str(), "JSON parse error: %s", e.what());
            return FAILED;
        } catch (...) {
            OP_LOGE(GetOpName(op_dst).c_str(), "get unknown exception, please check compile info json.");
            return FAILED;
        }
    }
    op_dst.SetAttr("pixel", pixel_value);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("AdaCast")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::AdaCast"), ge::AscendString("ai.onnx::9::AdaCast"),
                   ge::AscendString("ai.onnx::10::AdaCast"), ge::AscendString("ai.onnx::11::AdaCast"),
                   ge::AscendString("ai.onnx::12::AdaCast"), ge::AscendString("ai.onnx::13::AdaCast"),
                   ge::AscendString("ai.onnx::14::AdaCast"), ge::AscendString("ai.onnx::15::AdaCast"),
                   ge::AscendString("ai.onnx::16::AdaCast"), ge::AscendString("ai.onnx::17::AdaCast"),
                   ge::AscendString("ai.onnx::18::AdaCast")})
    .ParseParamsByOperatorFn(ParseParamsAdaCast)
    .ImplyType(ImplyType::TVM);
} // namespace domi
