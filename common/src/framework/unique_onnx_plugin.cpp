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
 * \file unique_plugin.cc
 * \brief onnx plugin for Unique operator, but not all functions are supported.
 *        Only support same case with UnqiueWithCountsAndSorting, so onnx's axis attribute
 *        and indices output are currently unavailable for use.
 */
#include "onnx_common.h"
#include "op_nn_proto_extend.h"
#include "control/shape/op_graph/shape_proto.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsUnique(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    bool sorted = true;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "sorted" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 0) {
            sorted = false;
        }
    }
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("sorted", sorted);
    op_dest.SetAttr("original_type", "ai.onnx::11::Unique");

    auto input_size = node->input_size();
    auto output_size = node->output_size();
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);
    return SUCCESS;
}

static Status ParseOpToGraphUnique(const ge::Operator& op, ge::Graph& graph)
{
    bool sorted;
    std::string name;

    if (op.GetAttr("name", name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }
    if (op.GetAttr("sorted", sorted) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get sorted attribute from op failed.");
        return FAILED;
    }

    auto data_x = ge::op::Data((name + "_x").c_str()).set_attr_index(0);
    auto unique_with_counts_and_sorting =
      ge::op::UniqueWithCountsAndSorting((name + "_UniqueWithCountsAndSorting").c_str());
    unique_with_counts_and_sorting.set_input_x(data_x)
        .set_attr_return_inverse(true)
        .set_attr_return_counts(true)
        .set_attr_sorted(sorted);

    auto shape = ge::op::Shape((name + "_shape").c_str()).set_input_x_by_name(unique_with_counts_and_sorting, "y");
    auto empty = ge::op::Empty((name + "_empty").c_str()).set_input_shape(shape)
                                                         .set_attr_dtype(ge::DT_INT64)
                                                         .set_attr_init(true);

    std::vector<ge::Operator> inputs{data_x};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(unique_with_counts_and_sorting, std::vector<size_t>{0});
    outputs.emplace_back(empty, std::vector<size_t>{0});
    outputs.emplace_back(unique_with_counts_and_sorting, std::vector<size_t>{1});
    outputs.emplace_back(unique_with_counts_and_sorting, std::vector<size_t>{2});

    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register Unique op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::Unique"),
         ge::AscendString("ai.onnx::12::Unique"),
         ge::AscendString("ai.onnx::13::Unique"),
         ge::AscendString("ai.onnx::14::Unique"),
         ge::AscendString("ai.onnx::15::Unique"),
         ge::AscendString("ai.onnx::16::Unique"),
         ge::AscendString("ai.onnx::17::Unique"),
         ge::AscendString("ai.onnx::18::Unique")})
    .ParseParamsFn(ParseParamsUnique)
    .ParseOpToGraphFn(ParseOpToGraphUnique)
    .ImplyType(ImplyType::TVM);
} // namespace domi
