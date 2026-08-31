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

static constexpr int64_t FIXED_SHIFT_VALUE_DEFAULT = 42;
static constexpr int64_t FIXED_SHIFT_VALUE_MAX = 43;
static constexpr int64_t FIXED_SHIFT_VALUE_MIN = 34;

// 将输出 0 的 format 设置为 NCHW，解耦 protobuf 后内联原 onnx_common.h 中 ChangeFormatFromOnnx 逻辑
static Status ChangeOutputFormatToNchw(ge::Operator& op)
{
    ge::TensorDesc org_tensor_y = op.GetOutputDesc(0);
    org_tensor_y.SetOriginFormat(ge::FORMAT_NCHW);
    org_tensor_y.SetFormat(ge::FORMAT_NCHW);
    if (op.UpdateOutputDesc(0U, org_tensor_y) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "change output format failed.");
        return FAILED;
    }
    return SUCCESS;
}

static Status ParseParamsMatMul(const ge::Operator& op_src, ge::Operator& op_dest)
{
    // add the attr to support the custom matmul transpose fusion
    bool trans_a = false;
    bool trans_b = false;
    int64_t fixed_shift_value = FIXED_SHIFT_VALUE_DEFAULT;
    int64_t enable_uncache = 0;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    std::string attr_name = attr.value("name", "");
                    if (attr_name == "transA" && attr.contains("i") && attr["i"].get<int64_t>() != 0) {
                        trans_a = true;
                    }
                    if (attr_name == "transB" && attr.contains("i") && attr["i"].get<int64_t>() != 0) {
                        trans_b = true;
                    }
                    if (attr_name == "fixed_shift_value" && attr.contains("i")) {
                        int64_t shifted = attr["i"].get<int64_t>();
                        if (shifted <= FIXED_SHIFT_VALUE_MAX && shifted >= FIXED_SHIFT_VALUE_MIN) {
                            fixed_shift_value = shifted;
                        } else {
                            OP_LOGW(GetOpName(op_dest).c_str(),
                                    "fixed_shift_value %ld is out of range [%ld, %ld], use default %ld.", shifted,
                                    FIXED_SHIFT_VALUE_MIN, FIXED_SHIFT_VALUE_MAX, FIXED_SHIFT_VALUE_DEFAULT);
                        }
                    }
                    if (attr_name == "enable_uncache" && attr.contains("i")) {
                        enable_uncache = attr["i"].get<int64_t>();
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

    op_dest.SetAttr("enable_uncache", enable_uncache);
    op_dest.SetAttr("adj_x1", trans_a);
    op_dest.SetAttr("adj_x2", trans_b);
    op_dest.SetAttr("fixed_shift_value", fixed_shift_value);
    if (ChangeOutputFormatToNchw(op_dest) != SUCCESS) {
        OP_LOGE(GetOpName(op_dest).c_str(), "failed to change format.");
        return FAILED;
    }
    return SUCCESS;
}

// register MatMul op info to GE
REGISTER_CUSTOM_OP("BatchMatMulV2")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::MatMul"), ge::AscendString("ai.onnx::9::MatMul"),
                   ge::AscendString("ai.onnx::10::MatMul"), ge::AscendString("ai.onnx::11::MatMul"),
                   ge::AscendString("ai.onnx::12::MatMul"), ge::AscendString("ai.onnx::13::MatMul"),
                   ge::AscendString("ai.onnx::14::MatMul"), ge::AscendString("ai.onnx::15::MatMul"),
                   ge::AscendString("ai.onnx::16::MatMul"), ge::AscendString("ai.onnx::17::MatMul"),
                   ge::AscendString("ai.onnx::18::MatMul")})
    .ParseParamsByOperatorFn(ParseParamsMatMul)
    .ImplyType(ImplyType::TVM);
} // namespace domi
