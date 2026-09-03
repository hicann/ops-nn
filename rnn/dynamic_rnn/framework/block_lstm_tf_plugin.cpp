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
 * \brief
 */
#include <string>
#include <vector>
#include <map>
#include "register/register.h"
#include "framework/plugin_util.h"
#include "graph/operator.h"
#include "stub_ops.h"
#include "rnn/dynamic_rnn/op_graph/dynamic_rnn_proto.h"

#include "log/log.h"
#include "graph/types.h"

namespace domi {
using namespace ge;

static Status ParseParamsRNN(const ge::Operator& op_src, ge::Operator& op)
{
    // Set original_type
    op.SetAttr("original_type", "BlockLSTM");

    // 未调用 AutoMappingByOpFn，TF 源节点属性不会自动映射到目标算子；
    // 因此从 op_src（携带 TF 源节点属性）显式解析后写入目标算子，缺失时保持默认值。
    float forget_bias = 0.0f;
    (void)op_src.GetAttr("forget_bias", forget_bias);
    (void)op.SetAttr("forget_bias", forget_bias);

    float cell_clip = 3.0f;
    (void)op_src.GetAttr("cell_clip", cell_clip);
    (void)op.SetAttr("cell_clip", cell_clip);

    bool use_peephole = false;
    (void)op_src.GetAttr("use_peephole", use_peephole);
    (void)op.SetAttr("use_peephole", use_peephole);

    return SUCCESS;
}

static Status ParseOpToGraphRNN(const ge::Operator& op, ge::Graph& graph)
{
    ge::Operator data_0 = op::Data("seq_len_max").set_attr_index(0);
    ge::Operator data_1 = op::Data("x").set_attr_index(1);
    ge::Operator data_2 = op::Data("cs_prev").set_attr_index(2);
    ge::Operator data_3 = op::Data("h_prev").set_attr_index(3);
    ge::Operator data_4 = op::Data("w").set_attr_index(4);
    ge::Operator data_8 = op::Data("b").set_attr_index(8);

    float forget_bias = 0.0;
    if (op.GetAttr("forget_bias", forget_bias) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr forget_bias failed.");
        return FAILED;
    }

    float cell_clip = 3.0;
    if (op.GetAttr("cell_clip", cell_clip) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr cell_clip failed.");
        return FAILED;
    }

    bool use_peephole = false;
    if (op.GetAttr("use_peephole", use_peephole) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr use_peephole failed.");
        return FAILED;
    }

    auto cast = op::Cast().set_input_x(data_0).set_attr_dst_type(3);
    auto rnn = op::DynamicRNN()
                   .set_input_x(data_1)
                   .set_input_w(data_4)
                   .set_input_b(data_8)
                   .set_input_seq_length(cast)
                   .set_input_init_h(data_3)
                   .set_input_init_c(data_2)
                   .set_attr_forget_bias(forget_bias)
                   .set_attr_cell_clip(cell_clip)
                   .set_attr_use_peephole(use_peephole)
                   .set_attr_cell_type("BLOCKLSTM");

    std::vector<ge::Operator> inputs{data_1, data_4, data_8, data_0, data_3, data_2};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(rnn, vector<std::size_t>{3, 2, 5, 6, 4, 7, 1});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

// register BlockLSTM op info to GE
REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BlockLSTM")
    .ParseParamsByOperatorFn(ParseParamsRNN)
    .ParseOpToGraphFn(ParseOpToGraphRNN)
    .ImplyType(ImplyType::TVM);
} // namespace domi
