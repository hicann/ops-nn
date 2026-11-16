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
 * \file block_lstm_tf_plugin.cpp
 * \brief TensorFlow BlockLSTM adapter that expands to a DynamicRNN subgraph.
 */
#include <string>
#include <utility>
#include <vector>

#include "framework/plugin_util.h"
#include "register/register.h"
#include "rnn/dynamic_rnn/op_graph/dynamic_rnn_proto.h"
#include "stub_ops.h"

namespace domi {
using namespace ge;

static Status ParseParamsRNN(const ge::Operator& opSrc, ge::Operator& op)
{
    (void)op.SetAttr("original_type", std::string("BlockLSTM"));

    float forgetBias = 0.0F;
    if (opSrc.GetAttr("forget_bias", forgetBias) != ge::GRAPH_SUCCESS) {
        OP_LOGD(GetOpName(op).c_str(), "attr forget_bias is absent, using default 0.0.");
    }
    (void)op.SetAttr("forget_bias", forgetBias);

    float cellClip = 3.0F;
    if (opSrc.GetAttr("cell_clip", cellClip) != ge::GRAPH_SUCCESS) {
        OP_LOGD(GetOpName(op).c_str(), "attr cell_clip is absent, using default 3.0.");
    }
    (void)op.SetAttr("cell_clip", cellClip);

    bool usePeephole = false;
    if (opSrc.GetAttr("use_peephole", usePeephole) != ge::GRAPH_SUCCESS) {
        OP_LOGD(GetOpName(op).c_str(), "attr use_peephole is absent, using default false.");
    }
    (void)op.SetAttr("use_peephole", usePeephole);

    return SUCCESS;
}

static Status ParseOpToGraphRNN(const ge::Operator& op, ge::Graph& graph)
{
    const std::string name = GetOpName(op).empty() ? "block_lstm" : GetOpName(op);
    auto data0 = ge::op::Data((name + "_seq_len_max").c_str()).set_attr_index(0);
    auto data1 = ge::op::Data((name + "_x").c_str()).set_attr_index(1);
    auto data2 = ge::op::Data((name + "_cs_prev").c_str()).set_attr_index(2);
    auto data3 = ge::op::Data((name + "_h_prev").c_str()).set_attr_index(3);
    auto data4 = ge::op::Data((name + "_w").c_str()).set_attr_index(4);
    auto data8 = ge::op::Data((name + "_b").c_str()).set_attr_index(8);

    float forgetBias = 0.0F;
    if (op.GetAttr("forget_bias", forgetBias) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr forget_bias failed.");
        return FAILED;
    }

    float cellClip = 3.0F;
    if (op.GetAttr("cell_clip", cellClip) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr cell_clip failed.");
        return FAILED;
    }

    bool usePeephole = false;
    if (op.GetAttr("use_peephole", usePeephole) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr use_peephole failed.");
        return FAILED;
    }

    auto cast = ge::op::Cast((name + "_cast").c_str()).set_input_x(data0).set_attr_dst_type(ge::DT_INT32);
    auto rnn = ge::op::DynamicRNN((name + "_DynamicRNN").c_str());
    rnn.set_input_x(data1)
        .set_input_w(data4)
        .set_input_b(data8)
        .set_input_seq_length(cast)
        .set_input_init_h(data3)
        .set_input_init_c(data2)
        .set_attr_forget_bias(forgetBias)
        .set_attr_cell_clip(cellClip)
        .set_attr_use_peephole(usePeephole)
        .set_attr_cell_type(std::string("BLOCKLSTM"));

    std::vector<ge::Operator> inputs{data1, data4, data8, data0, data3, data2};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(rnn, std::vector<size_t>{3U, 2U, 5U, 6U, 4U, 7U, 1U});
    (void)graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BlockLSTM")
    .ParseParamsByOperatorFn(ParseParamsRNN)
    .ParseOpToGraphFn(ParseOpToGraphRNN)
    .ImplyType(ImplyType::TVM);
} // namespace domi
