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

namespace domi {
static Status ParseParamsTopK(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("TopK", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int axis = -1;
    bool sorted = true;
    bool largest = true;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "sorted" && attr.type() == ge::onnx::AttributeProto::INT) {
            sorted = static_cast<bool>(attr.i());
        } else if (attr.name() == "largest" && attr.type() == ge::onnx::AttributeProto::INT) {
            largest = static_cast<bool>(attr.i());
        } else if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
            axis = attr.i();
        }
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("sorted", sorted);
    op_dest.SetAttr("dim", axis);
    op_dest.SetAttr("largest", largest);

    return SUCCESS;
}

static Status ParseParamsTopKV9(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("TopK", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    // 1.add dynamic input and out
    int op_input_size = node->input_size();
    int op_output_size = node->output_size();
    op_dest.DynamicInputRegister("x", op_input_size);
    op_dest.DynamicOutputRegister("y", op_output_size);

    // 2.set original_type
    op_dest.SetAttr("original_type", "ai.onnx::9::TopK");

    int axis = -1;
    int64_t data = -1;

    // 3.set attr if needed
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "k" && attr.type() == ge::onnx::AttributeProto::INT) {
            data = attr.i();
        }
        if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
            axis = attr.i();
        }
    }
    if (data == -1) {
        OP_LOGE("TopK", "onnx TopK has no K attr.");
        return PARAM_INVALID;
    }

    const ge::Tensor scalar_k = CreateScalar(data, ge::DT_INT32);
    op_dest.SetAttr("k", scalar_k);
    op_dest.SetAttr("dim", axis);
    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

static Status ParseOpToGraphTopKV9(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    ge::Tensor k_value;
    if (op.GetAttr("k", k_value) != SUCCESS) {
        OP_LOGE("TopK", "get value from op failed");
        return FAILED;
    }

    int dim = -1;
    if (op.GetAttr("dim", dim) != SUCCESS) {
        OP_LOGE("TopK", "get dim from op failed");
        return FAILED;
    }

    auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(k_value);
    auto topk = op::TopK((ori_name + "_TopK").c_str()).set_input_x(data0).set_input_k(data1).set_attr_dim(dim);
    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
    output_indexs.emplace_back(topk, vector<std::size_t>{0});
    output_indexs.emplace_back(topk, vector<std::size_t>{1});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(std::vector<ge::AscendString>{ge::AscendString("ai.onnx::8::TopK"),
                                                ge::AscendString("ai.onnx::9::TopK")})
    .ParseParamsFn(ParseParamsTopKV9)
    .ParseOpToGraphFn(ParseOpToGraphTopKV9)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("TopK")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::10::TopK"),
         ge::AscendString("ai.onnx::11::TopK"),
         ge::AscendString("ai.onnx::12::TopK"),
         ge::AscendString("ai.onnx::13::TopK"),
         ge::AscendString("ai.onnx::14::TopK"),
         ge::AscendString("ai.onnx::15::TopK"),
         ge::AscendString("ai.onnx::16::TopK"),
         ge::AscendString("ai.onnx::17::TopK"),
         ge::AscendString("ai.onnx::18::TopK")})
    .ParseParamsFn(ParseParamsTopK)
    .ImplyType(ImplyType::TVM);

} // namespace domi
