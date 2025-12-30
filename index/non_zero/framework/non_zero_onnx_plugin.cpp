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
 * \file nonzero_plugin.cc
 * \brief
 */
#include "onnx_common.h"
#include "index/non_zero/op_graph/non_zero_proto.h"

using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsNonZero(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    op_dest.SetAttr("name", node->name());
    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    op_dest.SetAttr("original_type", "ai.onnx::11::NonZero");

    return SUCCESS;
}

static Status ParseOpToGraphNonZero(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data = op::Data((ori_name + "_data").c_str()).set_attr_index(0);
    std::vector<Operator> inputs{data};
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;

    auto non_zero = op::NonZero((ori_name + "_NonZero").c_str()).set_input_x(data).set_attr_transpose(true);

    output_indexs.emplace_back(non_zero, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {
            ge::AscendString("ai.onnx::9::NonZero"), 
            ge::AscendString("ai.onnx::10::NonZero"),
         
            ge::AscendString("ai.onnx::11::NonZero"), 
            ge::AscendString("ai.onnx::12::NonZero"),
         
            ge::AscendString("ai.onnx::13::NonZero"), 
            ge::AscendString("ai.onnx::14::NonZero"),
         
            ge::AscendString("ai.onnx::15::NonZero"), 
            ge::AscendString("ai.onnx::16::NonZero"),
         
            ge::AscendString("ai.onnx::17::NonZero"), 
            ge::AscendString("ai.onnx::18::NonZero")
        })
    .ParseParamsFn(ParseParamsNonZero)
    .ParseOpToGraphFn(ParseOpToGraphNonZero)
    .ImplyType(ImplyType::TVM);
} // namespace domi
