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
 * \file global_max_pool_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "op_nn_proto_extend.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsGlobalMaxPool(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int input_size = node->input_size();
    int output_size = node->output_size();

    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("original_type", "ai.onnx::11::GlobalMaxPool");

    return SUCCESS;
}

static Status ParseOpToGraphGlobalMaxPool(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    bool keep_dims = true;

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);

    std::vector<int> axes = {-1, 2};
    auto tensor_axes = Vec2Tensor(axes, {2}, ge::DT_INT32);
    auto const_axes = ge::op::Const((ori_name + "_Const_0").c_str()).set_attr_value(tensor_axes);
    auto global_max_pool = ge::op::ReduceMax((ori_name + "_RedeceMax").c_str())
                               .set_input_x(data0)
                               .set_input_axes(const_axes)
                               .set_attr_keep_dims(keep_dims);

    std::vector<ge::Operator> inputs{data0};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(global_max_pool, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::GlobalMaxPool"),
         ge::AscendString("ai.onnx::9::GlobalMaxPool"),
         ge::AscendString("ai.onnx::10::GlobalMaxPool"),
         ge::AscendString("ai.onnx::11::GlobalMaxPool"),
         ge::AscendString("ai.onnx::12::GlobalMaxPool"),
         ge::AscendString("ai.onnx::13::GlobalMaxPool"),
         ge::AscendString("ai.onnx::14::GlobalMaxPool"),
         ge::AscendString("ai.onnx::15::GlobalMaxPool"),
         ge::AscendString("ai.onnx::16::GlobalMaxPool"),
         ge::AscendString("ai.onnx::17::GlobalMaxPool"),
         ge::AscendString("ai.onnx::18::GlobalMaxPool")})
    .ParseParamsFn(ParseParamsGlobalMaxPool)
    .ParseOpToGraphFn(ParseOpToGraphGlobalMaxPool)
    .ImplyType(ImplyType::TVM);
} // namespace domi
