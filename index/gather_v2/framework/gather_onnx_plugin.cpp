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
 * \file gather_plugin.cpp
 * \brief
 */

#include "onnx_common.h"
#include "index/gather_v2/op_graph/gather_v2_proto.h"

using namespace ge;
namespace domi {

static const int DEFAULT_AXIS = 0;

static Status ParseParamsGather(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int input = 2;
    int output = 1;
    op_dest.SetAttr("original_type", "ai.onnx::11::Gather");
    op_dest.DynamicInputRegister("x", input);
    op_dest.DynamicOutputRegister("y", output);

    int32_t axis_val = DEFAULT_AXIS;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
            axis_val = attr.i();
        }
    }

    ge::Tensor scalar_axis = CreateScalar(axis_val, ge::DT_INT32);
    op_dest.SetAttr("constant_values", scalar_axis);
    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

static Status ParseOpToGraphGather(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

    ge::Tensor const_value;
    if (op.GetAttr("constant_values", const_value) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get const_value from op failed");
        return FAILED;
    }

    auto const_op = op::Const((ori_name + "_const_data").c_str()).set_attr_value(const_value);

    auto gather2v_op = op::GatherV2((ori_name + "_gatherv2").c_str())
                           .set_input_x(data0)
                           .set_input_indices(data1)
                           .set_input_axis(const_op)
                           .set_attr_negative_index_support(true);

    std::vector<ge::Operator> inputs = {data0, data1, const_op};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(gather2v_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::Gather"), 
         ge::AscendString("ai.onnx::9::Gather"),
         ge::AscendString("ai.onnx::10::Gather"), 
         ge::AscendString("ai.onnx::11::Gather"),

         ge::AscendString("ai.onnx::12::Gather"), 
         ge::AscendString("ai.onnx::13::Gather"),
         ge::AscendString("ai.onnx::14::Gather"), 
         ge::AscendString("ai.onnx::15::Gather"),

         ge::AscendString("ai.onnx::16::Gather"), 
         ge::AscendString("ai.onnx::17::Gather"),
         ge::AscendString("ai.onnx::18::Gather")})
    .ParseParamsFn(ParseParamsGather)
    .ParseOpToGraphFn(ParseOpToGraphGather)
    .ImplyType(ImplyType::TVM);
} // namespace domi
