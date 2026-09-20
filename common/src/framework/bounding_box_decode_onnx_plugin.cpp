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
const int kTypeFloat = 1;

template <typename T>
void GetAttrListFromJson(json& attr, std::vector<T>& val, std::string& dtype)
{
    int num = attr[dtype].size();
    for (int i = 0; i < num; ++i) {
        val.push_back(attr[dtype][i].get<T>());
    }
}

static Status ParseBoundingBoxDecodeAttr(json& attr, std::vector<int64_t>& max_shape, std::vector<float>& means,
                                         std::vector<float>& stds, ge::Operator& op_dest)
{
    std::string dtype;
    if (attr["name"] == "max_shape") {
        dtype = "ints";
        GetAttrListFromJson(attr, max_shape, dtype);
    } else if (attr["name"] == "means") {
        dtype = "floats";
        GetAttrListFromJson(attr, means, dtype);
    } else if (attr["name"] == "stds") {
        dtype = "floats";
        GetAttrListFromJson(attr, stds, dtype);
    } else if (attr["name"] == "wh_ratio_clip" && attr["type"] == kTypeFloat) {
        // float type in json has accuracy loss, so we use string type to store it
        float wh_ratio_clip = 0.016f;
        if (attr.contains("f")) {
            std::string wh_ratio_clip_str = attr["f"];
            if (!StrToFloat(wh_ratio_clip_str, wh_ratio_clip)) {
                OP_LOGE("bounding_box_decode", "invalid wh_ratio_clip value: %s", wh_ratio_clip_str.c_str());
                return FAILED;
            }
        } else {
            // GE 序列化 float 属性时会省略值为 0 的 "f" 字段，此时 wh_ratio_clip 实际值为 0
            wh_ratio_clip = 0.0f;
        }
        op_dest.SetAttr("wh_ratio_clip", wh_ratio_clip);
    }
    return SUCCESS;
}

static Status ParseParamsBoundingBoxDecode(const ge::Operator& op_src, ge::Operator& op_dest)
{
    std::vector<int64_t> max_shape;
    std::vector<float> means;
    std::vector<float> stds;
    ge::AscendString attrs_string;
    try {
        if (ge::GRAPH_SUCCESS == op_src.GetAttr("attribute", attrs_string)) {
            json attrs = json::parse(attrs_string.GetString());
            for (json attr : attrs["attribute"]) {
                if (ParseBoundingBoxDecodeAttr(attr, max_shape, means, stds, op_dest) != SUCCESS) {
                    return FAILED;
                }
            }
        }
        if (max_shape.empty() || means.empty() || stds.empty()) {
            return FAILED;
        }

        op_dest.SetAttr("max_shape", max_shape);
        op_dest.SetAttr("means", means);
        op_dest.SetAttr("stds", stds);
    } catch (const json::parse_error& e) {
        OP_LOGE("bounding_box_decode", "JSON parse error: %s", e.what());
        return FAILED;
    } catch (const json::exception& e) {
        OP_LOGE("bounding_box_decode", "JSON processing error: %s", e.what());
        return FAILED;
    }

    return SUCCESS;
}

static Status ParseParamsBoundingBoxDecodeV2(const ge::Operator& op_src, ge::Operator& op_dest)
{
    std::vector<int> max_shape;
    std::map<std::string, float> attr_map = {{"means0", 0.0f}, {"means1", 0.0f}, {"means2", 0.0f},
                                             {"means3", 0.0f}, {"stds0", 1.0f},  {"stds1", 1.0f},
                                             {"stds2", 1.0f},  {"stds3", 1.0f},  {"wh_ratio_clip", 0.016f}};
    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            for (json& attr : attrs["attribute"]) {
                std::string attr_name = attr.value("name", "");
                if (attr_map.find(attr_name) != attr_map.end()) {
                    if (attr.contains("f")) {
                        std::string value_str = attr["f"];
                        if (!StrToFloat(value_str, attr_map[attr_name])) {
                            OP_LOGE(GetOpName(op_dest).c_str(), "invalid %s value: %s", attr_name.c_str(),
                                    value_str.c_str());
                            return FAILED;
                        }
                    } else {
                        // GE 序列化 float 属性时会省略值为 0 的 "f" 字段，此时属性值为 0
                        attr_map[attr_name] = 0.0f;
                    }
                } else if (attr_name == "max_shapes" && attr.contains("ints")) {
                    for (json& v : attr["ints"]) {
                        max_shape.push_back(v.get<int>());
                    }
                }
            }
        }
    } catch (const json::parse_error& e) {
        OP_LOGE("bounding_box_decode", "JSON parse error: %s", e.what());
        return FAILED;
    } catch (const json::exception& e) {
        OP_LOGE("bounding_box_decode", "JSON processing error: %s", e.what());
        return FAILED;
    }

    if (max_shape.empty()) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr max_shape.");
        return FAILED;
    }
    std::vector<float> means;
    std::vector<float> stds;
    means.push_back(attr_map["means0"]);
    means.push_back(attr_map["means1"]);
    means.push_back(attr_map["means2"]);
    means.push_back(attr_map["means3"]);
    stds.push_back(attr_map["stds0"]);
    stds.push_back(attr_map["stds1"]);
    stds.push_back(attr_map["stds2"]);
    stds.push_back(attr_map["stds3"]);

    op_dest.SetAttr("max_shape", max_shape);
    op_dest.SetAttr("means", means);
    op_dest.SetAttr("stds", stds);
    op_dest.SetAttr("wh_ratio_clip", attr_map["wh_ratio_clip"]);
    return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("BoundingBoxDecode")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::BoundingBoxDecode"), ge::AscendString("ai.onnx::9::BoundingBoxDecode"),
         ge::AscendString("ai.onnx::10::BoundingBoxDecode"), ge::AscendString("ai.onnx::11::BoundingBoxDecode"),
         ge::AscendString("ai.onnx::12::BoundingBoxDecode"), ge::AscendString("ai.onnx::13::BoundingBoxDecode"),
         ge::AscendString("ai.onnx::14::BoundingBoxDecode"), ge::AscendString("ai.onnx::15::BoundingBoxDecode"),
         ge::AscendString("ai.onnx::16::BoundingBoxDecode"), ge::AscendString("ai.onnx::17::BoundingBoxDecode"),
         ge::AscendString("ai.onnx::18::BoundingBoxDecode")})
    .ParseParamsByOperatorFn(ParseParamsBoundingBoxDecode)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("BoundingBoxDecode")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::NPUBoundingBoxDecode"), ge::AscendString("ai.onnx::12::NPUBoundingBoxDecode"),
         ge::AscendString("ai.onnx::13::NPUBoundingBoxDecode"), ge::AscendString("ai.onnx::14::NPUBoundingBoxDecode"),
         ge::AscendString("ai.onnx::15::NPUBoundingBoxDecode"), ge::AscendString("ai.onnx::16::NPUBoundingBoxDecode"),
         ge::AscendString("ai.onnx::17::NPUBoundingBoxDecode"), ge::AscendString("ai.onnx::18::NPUBoundingBoxDecode"),
         "npu::1::NPUBoundingBoxDecode"})
    .ParseParamsByOperatorFn(ParseParamsBoundingBoxDecodeV2)
    .ImplyType(ImplyType::TVM);
} // namespace domi
