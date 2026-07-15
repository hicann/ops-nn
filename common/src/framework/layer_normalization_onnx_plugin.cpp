/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "op_nn_proto_extend.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace {
constexpr int INPUT_NUM = 3;
constexpr int OUTPUT_NUM = 3;
constexpr int REQUIRED_INPUT_NUM = 2;
} // namespace

namespace domi {
static Status ParseParamsLayerNormalization(const Message* op_src, ge::Operator& op_dest)
{
    op_dest.DynamicInputRegister("input", INPUT_NUM);
    op_dest.DynamicOutputRegister("output", OUTPUT_NUM);

    op_dest.SetAttr("original_type", "ai.onnx::17::LayerNormalization");

    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    bool no_beta = false;
    int input_num = node->input_size();
    if (input_num == REQUIRED_INPUT_NUM) {
        no_beta = true;
    }

    int begin_norm_axis = -1;
    float epsilon = 0.00001;
    int stash_type = 1;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
            begin_norm_axis = attr.i();
        }
        if (attr.name() == "epsilon" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            epsilon = attr.f();
        }
        if (attr.name() == "stash_type" && attr.type() == ge::onnx::AttributeProto::INT) {
            stash_type = attr.i();
        }
    }
    op_dest.SetAttr("begin_norm_axis", begin_norm_axis);
    op_dest.SetAttr("epsilon", epsilon);
    op_dest.SetAttr("stash_type", stash_type);

    op_dest.SetAttr("no_beta", no_beta);
    op_dest.SetAttr("name", node->name());

    return SUCCESS;
}

static Status ParseOpToGraphLayerNormalization(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    int begin_norm_axis = -1;
    if (op.GetAttr("begin_norm_axis", begin_norm_axis) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get begin_norm_axis from op failed.");
        return FAILED;
    }

    float epsilon = 0.00001;
    if (op.GetAttr("epsilon", epsilon) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get epsilon from op failed.");
        return FAILED;
    }

    int stash_type = 1;
    if (op.GetAttr("stash_type", stash_type) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get stash_type from op failed.");
        return FAILED;
    }

    auto data0 = op::Data(ori_name + "_x").set_attr_index(0);
    auto identity_0 = op::Identity(ori_name + "identity_0").set_input_x(data0);
    auto shape_op = op::Shape(ori_name + "_shape").set_input_x(identity_0).set_attr_dtype(ge::DT_INT32);
    auto data1 = op::Data(ori_name + "_gamma").set_attr_index(1);
    auto identity_1 = op::Identity(ori_name + "identity_1").set_input_x(data1);
    auto data1_op = op::BroadcastTo(ori_name + "_broadcastto1").set_input_x(identity_1).set_input_shape(shape_op);

    bool no_beta = false;
    op.GetAttr("no_beta", no_beta);
    Operator beta_op;
    if (no_beta) {
        float beta = 0.0;
        ge::Tensor scalar_beta = CreateScalar(beta, ge::DT_FLOAT);
        beta_op = op::Const(ori_name + "_beta").set_attr_value(scalar_beta);
    } else {
        beta_op = op::Data(ori_name + "_beta").set_attr_index(REQUIRED_INPUT_NUM);
    }

    auto identity_2 = op::Identity(ori_name + "identity_2").set_input_x(beta_op);
    auto beta1_op = op::BroadcastTo(ori_name + "_broadcastto2").set_input_x(identity_2).set_input_shape(shape_op);

    auto layer_norm = op::LayerNorm(ori_name + "_LayerNorm")
                          .set_input_x(identity_0)
                          .set_input_gamma(data1_op)
                          .set_input_beta(beta1_op)
                          .set_attr_begin_norm_axis(begin_norm_axis)
                          .set_attr_epsilon(epsilon);

    std::vector<Operator> inputs{data0, data1, beta_op};
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;

    auto variance_add = op::Adds(ori_name + "_Adds").set_input_x(layer_norm, 2).set_attr_value(epsilon);
    auto inv_std_dev = op::Rsqrt(ori_name + "_Rsqrt").set_input_x(variance_add);
    Operator inv_std_dev_op;
    Operator mean_op;
    if (stash_type) {
        auto identity_3 = op::Identity(ori_name + "identity_3").set_input_x(inv_std_dev);
        auto identity_4 = op::Identity(ori_name + "identity_4").set_input_x(layer_norm, 1);
        inv_std_dev_op = op::Cast(ori_name + "_cast1").set_input_x(identity_3).set_attr_dst_type(ge::DT_FLOAT);
        mean_op = op::Cast(ori_name + "_cast2").set_input_x(identity_4).set_attr_dst_type(ge::DT_FLOAT);
    } else {
        inv_std_dev_op = op::Identity(ori_name + "_Identity3").set_input_x(inv_std_dev);
        mean_op = op::Identity(ori_name + "_Identity4").set_input_x(layer_norm, 1);
    }

    output_indexs.emplace_back(layer_norm, vector<std::size_t>{0});
    output_indexs.emplace_back(mean_op, vector<std::size_t>{0});
    output_indexs.emplace_back(inv_std_dev_op, vector<std::size_t>{0});

    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::17::LayerNormalization", "ai.onnx::18::LayerNormalization"})
    .ParseParamsFn(ParseParamsLayerNormalization)
    .ParseOpToGraphFn(ParseOpToGraphLayerNormalization)
    .ImplyType(ImplyType::TVM);
} // namespace domi
