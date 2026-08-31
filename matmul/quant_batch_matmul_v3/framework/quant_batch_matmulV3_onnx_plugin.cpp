/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmulV3_onnx_plugin.cpp
 * \brief
 */

#include "plugin_util.h"
#include "register/register.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsQuantBatchMatMulV3(const ge::Operator& op_src, ge::Operator& opDst)
{
    int a_dtype = 1;
    bool trans_x1 = false;
    bool trans_x2 = false;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    std::string attr_name = attr.value("name", "");
                    if (attr_name == "dtype" && attr.contains("i")) {
                        a_dtype = attr["i"].get<int>();
                        continue;
                    }
                    if (attr_name == "transpose_x1" && attr.contains("i")) {
                        if (attr["i"].get<int>() == 1) {
                            trans_x1 = true;
                        }
                        continue;
                    }
                    if (attr_name == "transpose_x2" && attr.contains("i")) {
                        if (attr["i"].get<int>() == 1) {
                            trans_x2 = true;
                        }
                        continue;
                    }
                }
            }
        } catch (const nlohmann::json::exception& e) {
            OP_LOGE(GetOpName(opDst).c_str(), "JSON parse error: %s", e.what());
            return FAILED;
        } catch (...) {
            OP_LOGE(GetOpName(opDst).c_str(), "get unknown exception, please check compile info json.");
            return FAILED;
        }
    }
    opDst.SetAttr("dtype", a_dtype);
    opDst.SetAttr("transpose_x1", trans_x1);
    opDst.SetAttr("transpose_x2", trans_x2);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("QuantBatchMatmulV3")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::QuantBatchMatMul"), ge::AscendString("ai.onnx::9::QuantBatchMatMul"),
                   ge::AscendString("ai.onnx::10::QuantBatchMatMul"), ge::AscendString("ai.onnx::11::QuantBatchMatMul"),
                   ge::AscendString("ai.onnx::12::QuantBatchMatMul"), ge::AscendString("ai.onnx::13::QuantBatchMatMul"),
                   ge::AscendString("ai.onnx::14::QuantBatchMatMul"), ge::AscendString("ai.onnx::15::QuantBatchMatMul"),
                   ge::AscendString("ai.onnx::16::QuantBatchMatMul")})
    .ParseParamsByOperatorFn(ParseParamsQuantBatchMatMulV3)
    .ImplyType(ImplyType::TVM);
} // namespace domi
