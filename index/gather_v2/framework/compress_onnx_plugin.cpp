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
 * \file compress_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "index/gather_v2/op_graph/gather_v2_proto.h"

using namespace ge;
namespace domi {

static Status parseParamsCompress(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int input = 3;
    int output = 1;
    op_dest.SetAttr("original_type", "ai.onnx::11::Compress");
    op_dest.DynamicInputRegister("x", input);
    op_dest.DynamicOutputRegister("y", output);
    int32_t axis_val = 0;
    int32_t need_flatten = 1;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
            axis_val = attr.i();
            need_flatten = 0;
        }
    }

    ge::Tensor scalar_axis = CreateScalar(axis_val, DT_INT32);

    op_dest.SetAttr("need_flatten", need_flatten);
    op_dest.SetAttr("axis_value", scalar_axis);
    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

static Status ParseOpToGraphCompress(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

    int need_flatten = 0;
    if (op.GetAttr("need_flatten", need_flatten) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get the switch of whether inserting flatten from op failed");
        return FAILED;
    }

    ge::Tensor const_value;
    if (op.GetAttr("axis_value", const_value) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get const_value from op failed");
        return FAILED;
    }

    auto const_op = op::Const((ori_name + "_axis_data").c_str()).set_attr_value(const_value);

    ge::Operator compress;
    auto where = op::Where((ori_name + "_compress_condition").c_str()).set_input_x(data1);

    ge::Operator::OpListInt axis = {1};
    auto squeeze_where = op::Squeeze((ori_name + "_squeeze_where").c_str()).set_input_x(where).set_attr_axis(axis);

    if (need_flatten) {
        auto flatten = op::Flatten((ori_name + "_compress_flatten").c_str()).set_input_x(data0).set_attr_axis(0);
        auto squeeze = op::Squeeze((ori_name + "_squeeze").c_str()).set_input_x(flatten).set_attr_axis(0);
        compress = op::GatherV2((ori_name + "_compress").c_str())
                       .set_input_x(squeeze)
                       .set_input_indices(squeeze_where)
                       .set_input_axis(const_op);
    } else {
        compress = op::GatherV2((ori_name + "_compress").c_str())
                       .set_input_x(data0)
                       .set_input_indices(squeeze_where)
                       .set_input_axis(const_op);
    }

    std::vector<ge::Operator> inputs = {data0, data1, const_op};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(compress, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {
            ge::AscendString("ai.onnx::9::Compress"), 
            ge::AscendString("ai.onnx::10::Compress"),
            ge::AscendString("ai.onnx::11::Compress"), 
            ge::AscendString("ai.onnx::12::Compress"),
            ge::AscendString("ai.onnx::13::Compress"), 
            ge::AscendString("ai.onnx::14::Compress"),
            ge::AscendString("ai.onnx::15::Compress"), 
            ge::AscendString("ai.onnx::16::Compress"),
            ge::AscendString("ai.onnx::17::Compress"), 
            ge::AscendString("ai.onnx::18::Compress")})
    .ParseParamsFn(parseParamsCompress)
    .ParseOpToGraphFn(ParseOpToGraphCompress)
    .ImplyType(ImplyType::TVM);
} // namespace domi
