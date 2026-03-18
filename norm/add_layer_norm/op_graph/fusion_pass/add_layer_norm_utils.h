/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_ADD_LAYER_NORM_FUSION_UTILS_H
#define NN_ADD_LAYER_NORM_FUSION_UTILS_H
#include "ge/ge_utils.h"

namespace ops {
static bool IsScaler(const Shape& shape)
{
    return shape.GetDims().size() == 0;
}

static bool IsDynamicShape(const Shape& shape)
{
    return shape.GetShapeSize() == -1;
}

static void GetInputsInfo(
    const std::vector<SubgraphInput>& subgraph_inputs, std::vector<Shape>& input_shapes,
    std::vector<DataType>& input_dtpyes, std::vector<Format>& input_formats)
{
    for (const auto& subgraph_input : subgraph_inputs) {
        auto match_node = subgraph_input.GetAllInputs().at(0);
        TensorDesc tensor_desc;
        AscendString node_type;
        match_node.node.GetType(node_type);
        match_node.node.GetInputDesc(match_node.index, tensor_desc);
        input_shapes.emplace_back(tensor_desc.GetShape());
        input_dtpyes.emplace_back(tensor_desc.GetDataType());
        input_formats.emplace_back(tensor_desc.GetFormat());
    }
}

static Status InferShape(const GraphUniqPtr& replace_graph, const std::vector<SubgraphInput>& subgraph_inputs)
{
    OPS_LOG_D(pass_name.c_str(), "Begin infershape for replacements.");
    std::vector<Shape> input_shapes;
    for (const auto& subgraph_input : subgraph_inputs) {
        auto match_node = subgraph_input.GetAllInputs().at(0);
        TensorDesc tensor_desc;
        match_node.node.GetInputDesc(match_node.index, tensor_desc);
        input_shapes.emplace_back(tensor_desc.GetShape());
    }
    return GeUtils::InferShape(*replace_graph, input_shapes);
}

static bool IsAdd2InputValid(
    const GNode& add1_node, const GNode& add2_node,
    const AscendString& add1_node_name, const AscendString& add2_node_name, const std::string& pass_name)
{
    TensorDesc add1_input0_desc;
    TensorDesc add1_input1_desc;
    add1_node.GetInputDesc(0, add1_input0_desc);
    add1_node.GetInputDesc(1, add1_input1_desc);
    auto add1_input0_size = add1_input0_desc.GetShape().GetShapeSize();
    auto add1_input1_size = add1_input1_desc.GetShape().GetShapeSize();
    auto add1_inputx_desc = (add1_input0_size > add1_input1_size) ? add1_input0_desc : add1_input1_desc;
    
    TensorDesc add2_inputx_desc;
    auto [input_node0,_] = add2_node.GetInDataNodesAndPortIndexs(0);
    AscendString add2_node_type;
    input_node0->GetType(add2_node_type);
    if (add2_node_type == "Add") {
        add2_node.GetInputDesc(1, add2_inputx_desc);
    }else {
        add2_node.GetInputDesc(0, add2_inputx_desc);
    }
    //check shape
    if (!(add1_inputx_desc.GetShape().GetDims() == add2_inputx_desc.GetShape().GetDims())) {
        OPS_LOG_D(
            pass_name.c_str(), "only support %s input0 and %s input0 have same shape.",
            add1_node_name.GetString(), add2_node_name.GetString());
        return false;
    }
    //check dtype
    if (!(add1_inputx_desc.GetDataType() == add2_inputx_desc.GetDataType())) {
        OPS_LOG_D(
            pass_name.c_str(),
            "Only support %s inputs and %s inputs have same dtype, but now Add1_input0_dtype is %d, Add2_input0_dtype is %d",
            add1_node_name.GetString(), add2_node_name.GetString(), add1_inputx_desc.GetDataType(),
            add2_inputx_desc.GetDataType());
        return false;
    }
    return true;
}

static bool IsCast1HasControlEdge(
    const std::unique_ptr<MatchResult>& match_result, int64_t cast1_capture_idx, const std::string& pass_name)
{
    NodeIo cast1_node;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(cast1_capture_idx, cast1_node) != SUCCESS, false, pass_name.c_str(),
        "get cast node failed.");
    AscendString cast1_node_name;
    cast1_node.node.GetName(cast1_node_name);
    if (cast1_node.node.GetInControlNodes().size() != 0 || cast1_node.node.GetOutControlNodes().size() != 0) {
        OPS_LOG_D(kPassName.c_str(), "%s node can not have control edge", cast1_node_name.GetString());
        return true;
    }
    return false;
}

static void UpdateAddLayerNormFormat(const Format layernorm_node_format, GNode& addlayernorm_node)
{
    for (int64_t i = 0; i < 5; i++) {
        TensorDesc addlayernorm_input_desc;
        addlayernorm_node.GetInputDesc(i, addlayernorm_input_desc);
        addlayernorm_input_desc.SetFormat(layernorm_node_format);
        addlayernorm_node.UpdateInputDesc(i, addlayernorm_input_desc);
    }
}
} // namespace ops


#endif // NN_ADD_LAYER_NORM_FUSION_UTILS_H