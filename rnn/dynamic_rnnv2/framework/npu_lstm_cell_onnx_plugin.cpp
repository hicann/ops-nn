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
#include "rnn/dynamic_rnnv2/op_graph/dynamic_rnnv2_proto.h"

namespace {
const int OUTPUT_SIZE = 8;
}
namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNpuLSTMCell(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    std::string direction = "UNIDIRECTIONAL";
    std::string gate_order = "ifco";
    bool train = false;
    bool flag_bias = true;

    for (auto& attr : node->attribute()) {
        if (attr.name() == "hasBias" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 0) {
            flag_bias = false;
        }
    }

    op_dest.SetAttr("direction", direction);
    op_dest.SetAttr("gate_order", gate_order);
    op_dest.SetAttr("is_training", train);
    op_dest.SetAttr("flag_bias", flag_bias);
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("original_type", "npu::1::NPULstmCell");

    auto input_size = node->input_size();
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", OUTPUT_SIZE);

    return SUCCESS;
}

static Status ParseOpToGraphNpuLSTMCell(const ge::Operator& op, ge::Graph& graph)
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
    bool train;
    if (op.GetAttr("is_training", train) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get is_training from op failed.");
        return FAILED;
    }

    bool flag_bias;
    if (op.GetAttr("flag_bias", flag_bias) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get flag_bias from op failed.");
        return FAILED;
    }

    auto data_x = ge::op::Data((ori_name + "_x").c_str()).set_attr_index(0);
    auto data_weight_input = ge::op::Data((ori_name + "_weight_input").c_str()).set_attr_index(1);
    auto data_weight_hidden = ge::op::Data((ori_name + "_weight_hidden").c_str()).set_attr_index(2);
    auto data_h = ge::op::Data((ori_name + "_h").c_str()).set_attr_index(3);
    auto data_c = ge::op::Data((ori_name + "_c").c_str()).set_attr_index(4);
    auto data_bias_input = ge::op::Data((ori_name + "_bias_input").c_str()).set_attr_index(5);
    auto data_bias_hidden = ge::op::Data((ori_name + "_bias_hidden").c_str()).set_attr_index(6);

    auto dynamic_rnn_v2 = ge::op::DynamicRNNV2((ori_name + "_DynamicRNNV2").c_str());
    auto add_op = ge::op::Add((ori_name + "_Bias_Add").c_str());

    std::vector<int32_t> axis = {0};
    auto tensor_zero = Vec2Tensor(axis, {1}, ge::DT_INT32);
    auto const_zero = ge::op::Const((ori_name + "_Const_zero").c_str()).set_attr_value(tensor_zero);
    auto expand_x_op = ge::op::ExpandDims((ori_name + "_ExpandDims_x").c_str());
    auto expand_h_op = ge::op::ExpandDims((ori_name + "_ExpandDims_h").c_str());
    auto expand_c_op = ge::op::ExpandDims((ori_name + "_ExpandDims_c").c_str());

    expand_x_op.set_input_x(data_x).set_input_axis(const_zero);
    expand_h_op.set_input_x(data_h).set_input_axis(const_zero);
    expand_c_op.set_input_x(data_c).set_input_axis(const_zero);

    dynamic_rnn_v2.set_input_x(expand_x_op)
        .set_input_weight_input(data_weight_input)
        .set_input_weight_hidden(data_weight_hidden)
        .set_input_init_h(expand_h_op)
        .set_input_init_c(expand_c_op)
        .set_attr_direction(direction)
        .set_attr_gate_order(gate_order)
        .set_attr_is_training(train);

    if (flag_bias) {
        add_op.set_input_x1(data_bias_input).set_input_x2(data_bias_hidden);
        dynamic_rnn_v2.set_input_b(add_op);
    } else {
        auto identity_bi = ge::op::Identity((ori_name + "_identity_bi").c_str()).set_input_x(data_bias_input);
        auto identity_bh = ge::op::Identity((ori_name + "_identity_bh").c_str()).set_input_x(data_bias_hidden);
    }

    std::vector<ge::Operator> inputs{data_x, data_weight_input, data_weight_hidden, data_h,
                                     data_c, data_bias_input,   data_bias_hidden};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexes;
    for (size_t i = 0; i < OUTPUT_SIZE; i++) {
        output_indexes.emplace_back(dynamic_rnn_v2, std::vector<size_t>{i});
    }
    graph.SetInputs(inputs).SetOutputs(output_indexes);
    return SUCCESS;
}

// register npu_lstm op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPULstmCell"), ge::AscendString("ai.onnx::11::NPULstmCell"),
                   ge::AscendString("ai.onnx::12::NPULstmCell"), ge::AscendString("ai.onnx::13::NPULstmCell"),
                   ge::AscendString("ai.onnx::14::NPULstmCell"), ge::AscendString("ai.onnx::15::NPULstmCell"),
                   ge::AscendString("ai.onnx::16::NPULstmCell"), ge::AscendString("ai.onnx::17::NPULstmCell"),
                   ge::AscendString("ai.onnx::18::NPULstmCell")})
    .ParseParamsFn(ParseParamsNpuLSTMCell)
    .ParseOpToGraphFn(ParseOpToGraphNpuLSTMCell)
    .ImplyType(ImplyType::TVM);
} // namespace domi
