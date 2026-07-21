/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "nlohmann/json.hpp"

#include <vector>

namespace domi {
static void GetAttrListFromJson(nlohmann::json& attr, std::vector<float>& val)
{
    for (size_t i = 0; i < attr["floats"].size(); ++i) {
        val.push_back(attr["floats"][i].get<float>());
    }
}

using NodeProto = ge::onnx::NodeProto;
static Status ParseOnnxParamsBucketize(const ge::Operator& op_src, ge::Operator& op_dest)
{
    std::vector<float> boundaries;
    ge::AscendString attrs_string;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("attribute", attrs_string)) {
        try {
            nlohmann::json attrs = nlohmann::json::parse(attrs_string.GetString());
            for (nlohmann::json& attr : attrs["attribute"]) {
                if (attr["name"] == "boundaries") {
                    GetAttrListFromJson(attr, boundaries);
                }
            }
        } catch (...) {
            OP_LOGE(GetOpName(op_dest).c_str(), "get unknown exception, please check compile info json.");
            return FAILED;
        }
    }
    op_dest.SetAttr("boundaries", boundaries);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Bucketize")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Bucketize"), ge::AscendString("ai.onnx::9::Bucketize"),
                   ge::AscendString("ai.onnx::10::Bucketize"), ge::AscendString("ai.onnx::11::Bucketize"),
                   ge::AscendString("ai.onnx::12::Bucketize"), ge::AscendString("ai.onnx::13::Bucketize"),
                   ge::AscendString("ai.onnx::14::Bucketize"), ge::AscendString("ai.onnx::15::Bucketize"),
                   ge::AscendString("ai.onnx::16::Bucketize")})
    .ParseParamsByOperatorFn(ParseOnnxParamsBucketize)
    .ImplyType(ImplyType::AI_CPU);
} // namespace domi
