/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <limits>
#include "onnx_common.h"
#include "op_nn_proto_extend.h"
using namespace ge;

namespace domi {
static Status ParseParamsGemm(const Message* op_src, ge::Operator& op_dest)
{
    ge::AscendString op_name;
    if (op_dest.GetName(op_name) != ge::GRAPH_SUCCESS) {
        OP_LOGE("", "failed to get op_name");
        return FAILED;
    }
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(op_name.GetString(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    bool trans_a = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "transA" && attr.i() != 0) {
            trans_a = true;
            break;
        }
    }
    bool trans_b = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "transB" && attr.i() != 0) {
            trans_b = true;
            break;
        }
    }
    float alpha_value = 1.0;
    float beta_value = 1.0;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "alpha" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            alpha_value = attr.f();
        }
        if (attr.name() == "beta" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            beta_value = attr.f();
        }
    }

    op_dest.SetAttr("alpha", alpha_value);
    op_dest.SetAttr("beta", beta_value);
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("transpose_x1", trans_a);
    op_dest.SetAttr("transpose_x2", trans_b);
    int input_size = node->input_size();
    op_dest.SetAttr("input_size", input_size);

    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", 1);
    op_dest.SetAttr("original_type", "ai.onnx::14::Gemm");

    return SUCCESS;
}

static Status ParseOpToGraphGemm(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    bool trans_a = false;
    bool trans_b = false;
    float alpha_value = 1.0;
    float beta_value = 1.0;
    op.GetAttr("transpose_x1", trans_a);
    op.GetAttr("transpose_x2", trans_b);
    op.GetAttr("alpha", alpha_value);
    op.GetAttr("beta", beta_value);

    auto data_0 = op::Data(ori_name + "_data0").set_attr_index(0);
    auto data_1 = op::Data(ori_name + "_data1").set_attr_index(1);

    auto matmul_op = op::MatMulV2(ori_name + "_MatMulV2")
                         .set_input_x1(data_0)
                         .set_input_x2(data_1)
                         .set_attr_transpose_x1(trans_a)
                         .set_attr_transpose_x2(trans_b);

    int input_size = 0;
    int len_size = 3;
    op.GetAttr("input_size", input_size);
    std::vector<ge::Operator> inputs;
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    if (input_size == len_size) {
        auto data_2 = op::Data(ori_name + "_data2").set_attr_index(2);
        inputs = {data_0, data_1, data_2};
        if (std::abs(alpha_value - 1.0f) <= std::numeric_limits<float>::epsilon() &&
            std::abs(beta_value - 1.0f) <= std::numeric_limits<float>::epsilon()) {
            auto matmul_add_op = op::Add(ori_name + "_add").set_input_x1(matmul_op).set_input_x2(data_2);
            output_indexs.emplace_back(matmul_add_op, std::vector<std::size_t>{0});
        } else {
            auto mul1_op = op::Muls(ori_name + "_Muls1").set_input_x(matmul_op).set_attr_value(alpha_value);
            auto mul2_op = op::Muls(ori_name + "_Muls2").set_input_x(data_2).set_attr_value(beta_value);
            auto matmul_add_op = op::Add(ori_name + "_add").set_input_x1(mul1_op).set_input_x2(mul2_op);
            output_indexs.emplace_back(matmul_add_op, std::vector<std::size_t>{0});
        }
    } else {
        inputs = {data_0, data_1};
        output_indexs.emplace_back(matmul_op, std::vector<std::size_t>{0});
    }
    graph.SetInputs(inputs).SetOutputs(output_indexs);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Gemm"), ge::AscendString("ai.onnx::9::Gemm"),
                   ge::AscendString("ai.onnx::10::Gemm"), ge::AscendString("ai.onnx::11::Gemm"),
                   ge::AscendString("ai.onnx::12::Gemm"), ge::AscendString("ai.onnx::13::Gemm"),
                   ge::AscendString("ai.onnx::14::Gemm"), ge::AscendString("ai.onnx::15::Gemm"),
                   ge::AscendString("ai.onnx::16::Gemm"), ge::AscendString("ai.onnx::17::Gemm"),
                   ge::AscendString("ai.onnx::18::Gemm")})
    .ParseParamsFn(ParseParamsGemm)
    .ParseOpToGraphFn(ParseOpToGraphGemm)
    .ImplyType(ImplyType::TVM);
} // namespace domi
