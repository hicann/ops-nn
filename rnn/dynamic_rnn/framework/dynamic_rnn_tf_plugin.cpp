/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_rnn_tf_plugin.cpp
 * \brief
 */

#include <cstring>
#include <string>
#include <vector>

#include "register/register.h"
#include "framework/plugin_util.h"
#include "graph/operator.h"
#include "log/log.h"

namespace domi {
uint32_t wRnnInputPosition = 1;
static const char* const kForgetBias = "lstm_cell/add/y";
static const char* const kTransposeNode = "Transpose";

static Status DynamicRNNParserParams(const std::vector<ge::Operator>& inside_nodes, ge::Operator& op)
{
    OP_LOGI(GetOpName(op).c_str(), "Enter DynamicRNN fusion parser.");

    // 基于融合子图算子恢复旧 NodeDef 解析语义（node_def.pb.h 在新工具链已不存在，改用算子版融合解析）：
    // - 子图含 Transpose 节点 → time_major=false
    // - 名为 lstm_cell/add/y 的 Const 节点 → 从 value 张量解析 forget_bias
    bool time_major = true;
    float forget_bias = 0.0f;
    for (const auto& node : inside_nodes) {
        ge::AscendString node_type;
        if (node.GetOpType(node_type) == ge::GRAPH_SUCCESS && std::string(node_type.GetString()) == kTransposeNode) {
            time_major = false;
        }

        ge::AscendString node_name;
        if (node.GetName(node_name) != ge::GRAPH_SUCCESS) {
            continue;
        }
        if (std::string(node_name.GetString()).find(kForgetBias) == std::string::npos) {
            continue;
        }
        ge::Tensor const_value;
        if (node.GetAttr("value", const_value) != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "parse forget_bias from const node %s failed", node_name.GetString());
            return PARAM_INVALID;
        }
        const uint8_t* value_data = const_value.GetData();
        const size_t value_size = const_value.GetSize();
        if (value_data == nullptr || value_size < sizeof(float)) {
            OP_LOGE(GetOpName(op).c_str(), "parse forget_bias from const node %s failed", node_name.GetString());
            return PARAM_INVALID;
        }
        (void)memcpy(&forget_bias, value_data, sizeof(float));
    }
    op.SetAttr("time_major", time_major);
    op.SetAttr("forget_bias", forget_bias);
    OP_LOGD(GetOpName(op).c_str(), "parser stage set DynamicRNN's attr time_major is %s forget_bias is %.1f",
            time_major ? "true" : "false", forget_bias);

    ge::TensorDesc input_desc = op.GetInputDesc(wRnnInputPosition);
    input_desc.SetOriginFormat(ge::FORMAT_HWCN);
    input_desc.SetFormat(ge::FORMAT_HWCN);

    if (op.UpdateInputDesc(wRnnInputPosition, input_desc) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Update input desc fail, index:%u.", wRnnInputPosition);
        return FAILED;
    }

    return SUCCESS;
}

// 来自 dynamic_rnn_plugin.cc：TF 原生 DynamicRNN
REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicRNN")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .FusionParseParamsFn(DynamicRNNParserParams)
    .ImplyType(ImplyType::TVM);

// 来自 dynamic_rnn_tf_plugin.cc：DynamicRnn（保持原独立注册语义，不附加融合解析）
REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicRnn")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);
} // namespace domi
