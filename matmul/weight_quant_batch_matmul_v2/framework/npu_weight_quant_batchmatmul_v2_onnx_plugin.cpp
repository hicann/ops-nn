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
 * \file npu_weight_quant_batchmatmul_v2_onnx_plugin.cpp
 * \brief
 */

#include "plugin_util.h"
#include "register/register.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsWeightBatchQuantMatMulV2(const ge::Operator& op_src, ge::Operator& opDst)
{
    int antiquant_group_size = 0;
    int dtype = -1;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    std::string attr_name = attr.value("name", "");
                    if (attr_name == "antiquant_group_size" && attr.contains("i")) {
                        antiquant_group_size = attr["i"].get<int>();
                        continue;
                    }
                    if (attr_name == "dtype" && attr.contains("i")) {
                        dtype = attr["i"].get<int>();
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
    // onnx doesn't have transpose attr
    opDst.SetAttr("transpose_x", false);
    opDst.SetAttr("transpose_weight", false);
    opDst.SetAttr("antiquant_group_size", antiquant_group_size);
    opDst.SetAttr("dtype", dtype);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("WeightQuantBatchMatmulV2")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::WeightQuantBatchMatmulV2"),
                   ge::AscendString("ai.onnx::11::NPUWeightQuantBatchMatmulV2"),
                   ge::AscendString("ai.onnx::12::NPUWeightQuantBatchMatmulV2"),
                   ge::AscendString("ai.onnx::13::NPUWeightQuantBatchMatmulV2"),
                   ge::AscendString("ai.onnx::14::NPUWeightQuantBatchMatmulV2"),
                   ge::AscendString("ai.onnx::15::NPUWeightQuantBatchMatmulV2"),
                   ge::AscendString("ai.onnx::16::NPUWeightQuantBatchMatmulV2"),
                   ge::AscendString("ai.onnx::17::NPUWeightQuantBatchMatmulV2"),
                   ge::AscendString("ai.onnx::18::NPUWeightQuantBatchMatmulV2"),
                   ge::AscendString("ai.onnx::19::NPUWeightQuantBatchMatmulV2")})
    .ParseParamsByOperatorFn(ParseParamsWeightBatchQuantMatMulV2)
    .ImplyType(ImplyType::TVM);
} // namespace domi
