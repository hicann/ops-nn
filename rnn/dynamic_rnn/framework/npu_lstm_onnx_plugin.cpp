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
 * \file npu_lstm_plugin.cc
 * \brief onnx plugin for custom operator npu_lstm
 */

#include "onnx_common.h"
#include "rnn/dynamic_rnn/op_graph/dynamic_rnn_proto.h"

namespace {
const int OUTPUT_SIZE = 8;
}
namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNpuLSTM(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    std::string direction = "UNIDIRECTIONAL";
    std::string gate_order = "ifjo";
    bool train = false;
    bool flag_seq = false;
    for (auto& attr : node->attribute()) {
        if (attr.name() == "direction" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
            direction = "REDIRECTIONAL";
        } else if (attr.name() == "flagSeq" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
            flag_seq = true;
        } else if (attr.name() == "train" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
            train = true;
        }
    }
    op_dest.SetAttr("direction", direction);
    op_dest.SetAttr("gate_order", gate_order);
    op_dest.SetAttr("is_training", train);
    op_dest.SetAttr("flag_seq", flag_seq);
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("original_type", "npu::1::NPULstm");

    auto input_size = node->input_size();
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", OUTPUT_SIZE);

    return SUCCESS;
}

static Status ParseOpToGraphNpuLSTM(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name, direction, gate_order;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }
    if (op.GetAttr("direction", direction) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get direction from op failed.");
        return FAILED;
    }
    if (op.GetAttr("gate_order", gate_order) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get gate_order from op failed.");
        return FAILED;
    }
    bool train, flag_seq;
    if (op.GetAttr("is_training", train) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get is_training from op failed.");
        return FAILED;
    }
    if (op.GetAttr("flag_seq", flag_seq) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get flag_seq from op failed.");
        return FAILED;
    }

    auto data_x = ge::op::Data((ori_name + "_x").c_str()).set_attr_index(0);
    auto data_weight = ge::op::Data((ori_name + "_weight").c_str()).set_attr_index(1);
    auto data_bias = ge::op::Data((ori_name + "_bias").c_str()).set_attr_index(2);
    auto data_seq_length = ge::op::Data((ori_name + "_seq_length").c_str()).set_attr_index(3);
    auto data_h = ge::op::Data((ori_name + "_h").c_str()).set_attr_index(4);
    auto data_c = ge::op::Data((ori_name + "_c").c_str()).set_attr_index(5);
    auto dynamic_rnn = ge::op::DynamicRNN((ori_name + "_DynamicRnn").c_str());
    if (flag_seq == true) {
        dynamic_rnn.set_input_x(data_x)
            .set_input_w(data_weight)
            .set_input_b(data_bias)
            .set_input_seq_length(data_seq_length)
            .set_input_init_h(data_h)
            .set_input_init_c(data_c)
            .set_attr_direction(direction)
            .set_attr_gate_order(gate_order)
            .set_attr_is_training(train);
    } else {
        auto identity = ge::op::Identity((ori_name + "_identity").c_str()).set_input_x(data_seq_length);
        dynamic_rnn.set_input_x(data_x)
            .set_input_w(data_weight)
            .set_input_b(data_bias)
            .set_input_init_h(data_h)
            .set_input_init_c(data_c)
            .set_attr_direction(direction)
            .set_attr_gate_order(gate_order)
            .set_attr_is_training(train);
    }

    std::vector<ge::Operator> inputs{data_x, data_weight, data_bias, data_seq_length, data_h, data_c};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexes;
    for (size_t i = 0; i < OUTPUT_SIZE; i++) {
        output_indexes.emplace_back(dynamic_rnn, std::vector<size_t>{i});
    }
    graph.SetInputs(inputs).SetOutputs(output_indexes);
    return SUCCESS;
}

// register npu_lstm op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPULstm"), ge::AscendString("ai.onnx::11::NPULstm"),
                   ge::AscendString("ai.onnx::12::NPULstm"), ge::AscendString("ai.onnx::13::NPULstm"),
                   ge::AscendString("ai.onnx::14::NPULstm"), ge::AscendString("ai.onnx::15::NPULstm"),
                   ge::AscendString("ai.onnx::16::NPULstm"), ge::AscendString("ai.onnx::17::NPULstm"),
                   ge::AscendString("ai.onnx::18::NPULstm")})
    .ParseParamsFn(ParseParamsNpuLSTM)
    .ParseOpToGraphFn(ParseOpToGraphNpuLSTM)
    .ImplyType(ImplyType::TVM);
} // namespace domi
