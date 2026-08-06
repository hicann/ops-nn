/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "index/scatter_nd_add/op_graph/scatter_nd_add_proto.h"
#include "index/scatter_nd_max/op_graph/scatter_nd_max_proto.h"
#include "index/scatter_nd_min/op_graph/scatter_nd_min_proto.h"
#include "index/scatter_nd_update/op_graph/scatter_nd_update_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;

// 解析 ONNX 属性，并设置中间节点的输入输出和临时属性
static Status ParseParamsScatterND(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("ScatterND", "reinterpret_cast op_src to NodeProto failed.");
        return FAILED;
    }

    // 设置 3 个动态输入和 1 个动态输出
    op_dest.DynamicInputRegister("data", 1);
    op_dest.DynamicInputRegister("indices", 1);
    op_dest.DynamicInputRegister("updates", 1);
    op_dest.DynamicOutputRegister("y", 1);

    // 提取并保存 reduction 属性，供 ParseOpToGraphFn 使用
    std::string reduction = "none";
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "reduction" && attr.type() == ge::onnx::AttributeProto::STRING) {
            reduction = attr.s();
        }
    }

    op_dest.SetAttr("original_type", "ai.onnx::16::ScatterND");
    op_dest.SetAttr("reduction", reduction);
    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

// 构建 TensorMove + ScatterNdXX 子图
static Status ParseOpToGraphScatterND(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE("ScatterND", "get name from op failed.");
        return FAILED;
    }

    std::string reduction = "none";
    if (op.GetAttr("reduction", reduction) != SUCCESS) {
        OP_LOGW("ScatterND", "get reduction from op failed, use default 'none'.");
    }

    // 创建子图的 3 个输入 Data 节点，索引对应 ONNX 的输入顺序
    ge::Operator data_in = op::Data(ori_name + "_data").set_attr_index(0);
    ge::Operator indices_in = op::Data(ori_name + "_indices").set_attr_index(1);
    ge::Operator updates_in = op::Data(ori_name + "_updates").set_attr_index(2);

    std::vector<Operator> inputs{data_in, indices_in, updates_in};
    std::vector<std::pair<Operator, std::vector<size_t>>> outputs;

    // 使用 TensorMove 将 data 搬运/拷贝到输出 buffer，作为 Scatter 的初始数据
    auto tensor_move = op::TensorMove(ori_name + "_TensorMove").set_input_x(data_in);
    if (reduction == "none") {
        auto scatter_update = op::ScatterNdUpdate(ori_name + "_ScatterNdUpdate")
                                  .set_input_var(tensor_move)
                                  .set_input_indices(indices_in)
                                  .set_input_updates(updates_in);
        outputs.emplace_back(scatter_update, std::vector<std::size_t>{0});
    } else if (reduction == "add") {
        auto scatter_add = op::ScatterNdAdd(ori_name + "_ScatterNdAdd")
                               .set_input_var(tensor_move)
                               .set_input_indices(indices_in)
                               .set_input_updates(updates_in);
        outputs.emplace_back(scatter_add, std::vector<std::size_t>{0});
    } else if (reduction == "max") {
        auto scatter_max = op::ScatterNdMax(ori_name + "_ScatterNdMax")
                               .set_input_var(tensor_move)
                               .set_input_indices(indices_in)
                               .set_input_updates(updates_in);
        outputs.emplace_back(scatter_max, std::vector<std::size_t>{0});
    } else if (reduction == "min") {
        auto scatter_min = op::ScatterNdMin(ori_name + "_ScatterNdMin")
                               .set_input_var(tensor_move)
                               .set_input_indices(indices_in)
                               .set_input_updates(updates_in);
        outputs.emplace_back(scatter_min, std::vector<std::size_t>{0});
    } else if (reduction == "mul") {
        OP_LOGE("ScatterND", "ScatterND with reduction='mul' is not supported in this plugin. ");
        return FAILED;
    } else {
        OP_LOGE("ScatterND", "Unsupported reduction mode: %s", reduction.c_str());
        return FAILED;
    }

    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::ScatterND", "ai.onnx::9::ScatterND", "ai.onnx::10::ScatterND", "ai.onnx::11::ScatterND",
                   "ai.onnx::12::ScatterND", "ai.onnx::13::ScatterND", "ai.onnx::14::ScatterND",
                   "ai.onnx::15::ScatterND", "ai.onnx::16::ScatterND", "ai.onnx::17::ScatterND",
                   "ai.onnx::18::ScatterND"})
    .ParseParamsFn(ParseParamsScatterND)
    .ParseOpToGraphFn(ParseOpToGraphScatterND)
    .ImplyType(ImplyType::TVM);
} //  namespace domi
