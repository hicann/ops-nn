/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \brief dynamic_quant_update_scatter_v2_fusion_pass.cpp
 * \details fusion pass
 *                         x                            x  indices  In0  In1  In2
 *                         |                            |     |      |    |    |
 *          Const0    DynamicQuantV2                    |     |      |    |    |
 *                \       /|\                           |     |      |    |    |
 *                 Reshape0| \                --->     DynamicQuantUpdateScatterV2
 *                      /  | Neg indices                  |         |        |
 *        Const1   Bitcast0|   \ * /|                     |         |        |
 *              \     /    | *  \ / |                    var   var_scale var_offset
 *           In0 Reshape1 *| In1 X  | In2
 *             \    /  *   |  | / \ |  |
 *       Const2 Scatter0 Scatter1 Scatter2
 *           \    |        |        |
 *            Reshape2     |        |
 *                |        |        |
 *     Const3 Bitcast1 Unsqueeze0 Unsqueeze1
 *           \    |        |        |
 *            Reshape3     |        |
 *                |        |        |
 *               var   var_scale var_offset
 */

#include "dynamic_quant_update_scatter_v2_fusion_pass.h"

#include <string>

#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "ge/ge_utils.h"
#include "common/inc/error_util.h"
#include "platform/platform_info.h"
#include "compliant_node_builder.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {

static constexpr uint32_t IDX_0 = 0;
static constexpr uint32_t IDX_1 = 1;
static constexpr uint32_t IDX_2 = 2;
static constexpr uint32_t IDX_3 = 3;
static constexpr uint32_t IDX_4 = 4;
static constexpr uint32_t IDX_5 = 5;
static constexpr uint32_t IDX_8 = 8;
static constexpr int64_t FULL_LOAD_H_SIZE = 14960;

static const std::string FUSED_OP_TYPE = "DynamicQuantUpdateScatterV2";
static const std::string PASS_NAME = "DynamicQuantUpdateScatterV2FusionPass";

using UniqueGraphPtr = std::unique_ptr<Graph>;

static es::EsTensorHolder BuildPatternReshapeNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x, const es::EsTensorHolder& shape)
{
    static int counter = 0;
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    std::string name = "Reshape_" + std::to_string(counter++);
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Reshape")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .Name(name.c_str())
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *shape.GetProducer(), shape.GetProducerOutIndex(), node, 1);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildPatternConstNode(es::EsGraphBuilder& builder)
{
    static int counter = 0;
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    std::string name = "Const_" + std::to_string(counter++);
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Const")
        .Name(name.c_str())
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildNegNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x)
{
    static int counter = 0;
    std::string name = "Neg_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Neg")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildUnsqueezeNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x)
{
    static int counter = 0;
    std::string name = "Unsqueeze_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Unsqueeze")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildIdentityNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x)
{
    static int counter = 0;
    std::string name = "Identity_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Identity")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildBitcastNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x, DataType dt)
{
    static int counter = 0;
    std::string name = "Bitcast_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Bitcast")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"type", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
             es::CreateFrom(static_cast<int64_t>(dt))}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

struct DynamicQuantV2Result {
    es::EsTensorHolder y;
    es::EsTensorHolder scale;
    es::EsTensorHolder offset;
};

static DynamicQuantV2Result BuildDynamicQuantV2Node(es::EsGraphBuilder& builder, const es::EsTensorHolder& x)
{
    static int counter = 0;
    std::string name = "DynamicQuantV2_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("DynamicQuantV2")
        .Name(name.c_str())
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"smooth_scales", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"group_index", es::CompliantNodeBuilder::kEsIrInputOptional, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"scale", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"offset", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return {
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 2))
    };
}

static es::EsTensorHolder BuildScatterNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& var,
    const es::EsTensorHolder& indices, const es::EsTensorHolder& updates, const char* reduce)
{
    static int counter = 0;
    std::string name = "Scatter_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Scatter")
        .Name(name.c_str())
        .IrDefInputs({
            {"var", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"updates", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({{"var", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"reduce", es::CompliantNodeBuilder::kEsAttrOptional, "String",
             es::CreateFrom(ge::AscendString(reduce))},
            {"axis", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
             es::CreateFrom(static_cast<int64_t>(0))}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *var.GetProducer(), var.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *indices.GetProducer(), indices.GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *updates.GetProducer(), updates.GetProducerOutIndex(), node, 2);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static void MakePatternForDynamicQuantUpdateScatterV2(
    std::vector<PatternUniqPtr>& pattern_graphs, bool has_identity)
{
    std::string pattern_name = has_identity ? "patternDQUSv2WithIdentity" : "patternDQUSv2";
    auto graph_builder = es::EsGraphBuilder(pattern_name.c_str());
    auto x = graph_builder.CreateInput(0);
    auto indices = graph_builder.CreateInput(1);
    auto In0 = graph_builder.CreateInput(2);
    auto In1 = graph_builder.CreateInput(3);
    auto In2 = graph_builder.CreateInput(4);

    // When has_identity, insert Identity between Data inputs and Scatter var port
    auto scatter_var0 = has_identity ? BuildIdentityNode(graph_builder, In0) : In0;
    auto scatter_var1 = has_identity ? BuildIdentityNode(graph_builder, In1) : In1;
    auto scatter_var2 = has_identity ? BuildIdentityNode(graph_builder, In2) : In2;

    auto dqv2 = BuildDynamicQuantV2Node(graph_builder, x);

    // Pre-Scatter path: DQV2:y → Reshape → Bitcast → Reshape → Reshape → Scatter
    // The actual graph has 2 Reshapes after Bitcast (dynamic_const shape + static Const shape)
    auto y = BuildPatternReshapeNode(graph_builder, dqv2.y, BuildPatternConstNode(graph_builder));
    y = BuildBitcastNode(graph_builder, y, DT_INT32);
    y = BuildPatternReshapeNode(graph_builder, y, BuildPatternConstNode(graph_builder));  // dynamic shape reshape
    y = BuildPatternReshapeNode(graph_builder, y, BuildPatternConstNode(graph_builder));  // static shape reshape
    y = BuildScatterNode(graph_builder, scatter_var0, indices, y, "update");

    // Post-Scatter path: Scatter → Reshape → Bitcast → Reshape
    y = BuildPatternReshapeNode(graph_builder, y, BuildPatternConstNode(graph_builder));
    y = BuildBitcastNode(graph_builder, y, DT_INT4);
    y = BuildPatternReshapeNode(graph_builder, y, BuildPatternConstNode(graph_builder)); // var output

    auto scale = BuildScatterNode(graph_builder, scatter_var1, indices, dqv2.scale, "update");
    scale = BuildUnsqueezeNode(graph_builder, scale); // var_scale

    auto offset = BuildNegNode(graph_builder, dqv2.offset);
    offset = BuildScatterNode(graph_builder, scatter_var2, indices, offset, "update");
    offset = BuildUnsqueezeNode(graph_builder, offset); // var_offset

    std::vector<es::EsTensorHolder> pattern_outputs;
    pattern_outputs.emplace_back(std::move(y));
    pattern_outputs.emplace_back(std::move(scale));
    pattern_outputs.emplace_back(std::move(offset));
    auto graph = graph_builder.BuildAndReset(pattern_outputs);
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern_graphs.emplace_back(std::move(pattern));
}

static bool IsAllInputDtypeRight(const std::vector<DataType>& input_dtypes)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Begin check all input dtype");
    DataType x_dtype = input_dtypes[0];
    DataType indices_dtype = input_dtypes[1];
    DataType input0_dtype = input_dtypes[2];
    DataType input1_dtype = input_dtypes[3];
    DataType input2_dtype = input_dtypes[4];
    OP_LOGE_IF(
        (x_dtype != DT_FLOAT16) && (x_dtype != DT_BF16), false, PASS_NAME.c_str(), "x dtype is not right.");
    OP_LOGE_IF(
        (indices_dtype != DT_INT32), false, PASS_NAME.c_str(), "indices dtype is not right.");
    OP_LOGE_IF(
        (input0_dtype != DT_INT32), false, PASS_NAME.c_str(), "var dtype is not right.");
    OP_LOGE_IF(
        (input1_dtype != DT_FLOAT), false, PASS_NAME.c_str(), "var_scale dtype is not right.");
    OP_LOGE_IF(
        (input2_dtype != DT_FLOAT), false, PASS_NAME.c_str(), "var_offset dtype is not right.");
    return true;
}

static bool IsAllOutputDtypeRight(const std::vector<DataType>& output_dtypes)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Begin check all output dtype");
    DataType quant_out_dtype = output_dtypes[0];
    DataType var_out_dtype = output_dtypes[1];
    OP_LOGE_IF(
        (quant_out_dtype != DT_INT4), false, PASS_NAME.c_str(), "it is not int4 dynamic quant.");
    OP_LOGE_IF(
        (var_out_dtype != DT_INT4), false, PASS_NAME.c_str(), "var out dtype is not right.");
    return true;
}

static bool IsTargetPlatform()
{
    OPS_LOG_I(PASS_NAME.c_str(), "Begin check platform information");
    PlatformInfo platform_info;
    OptionalInfo optional_info;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
        false, PASS_NAME.c_str(), "Get platform_info failed.");

    const string soc = platform_info.str_info.short_soc_version;
    bool is_platform910b = soc == "Ascend910b";
    bool is_platform910_93 = soc == "Ascend910_93";
    OPS_LOG_I(PASS_NAME.c_str(), "Platform short soc: %s", soc.c_str());
    OP_LOGE_IF(
        !is_platform910b && !is_platform910_93, false, PASS_NAME.c_str(),
        "Platform is not support, only work on 910b or 910_93.");
    return true;
}

static bool IsAllNodeNotNull(const std::vector<NodeIo>& input_nodes)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Begin check if all node not null.");

    AscendString node_type;
    for (const auto& input_node : input_nodes) {
        input_node.node.GetType(node_type);
        OP_LOGE_IF(
            &input_node.node == nullptr, false, PASS_NAME.c_str(), "node %s can't be null.", node_type.GetString());
    }
    OPS_LOG_I(PASS_NAME.c_str(), "All nodes not null.");
    return true;
}

static bool IsAllInputShapeRight(std::vector<Shape>& input_shapes)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Begin check all input shape.");
    Shape x_shape = input_shapes[0];
    Shape indices_shape = input_shapes[1];
    Shape input0_shape = input_shapes[2];
    Shape input1_shape = input_shapes[3];
    Shape input2_shape = input_shapes[4];

    OP_LOGE_IF(x_shape.GetDims().size() != IDX_3, false, PASS_NAME.c_str(), "x shape is not right.");
    OP_LOGE_IF(
        ((x_shape.GetDims().at(IDX_2) % IDX_8) != 0) || (x_shape.GetDims().at(IDX_2) > FULL_LOAD_H_SIZE),
        false, PASS_NAME.c_str(), "h size is not right.");
    OP_LOGE_IF(input0_shape.GetDims().size() != IDX_4, false, PASS_NAME.c_str(), "var shape is not right.");
    OP_LOGE_IF(
        (input0_shape.GetDims().at(IDX_2) != IDX_1) || ((input0_shape.GetDims().at(IDX_3) * IDX_8) != x_shape.GetDims().at(IDX_2)),
        false, PASS_NAME.c_str(), "var size is not right.");
    return true;
}

static Status InferShape(UniqueGraphPtr& replaceGraph, const std::vector<SubgraphInput>& subgraph_inputs)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Begin infershape for replacements.");
    std::vector<ge::TensorDesc> input_descs;
    for (const auto& subgraph_input : subgraph_inputs) {
        auto match_node = subgraph_input.GetAllInputs().at(0);
        TensorDesc tensor_desc;
        match_node.node.GetInputDesc(match_node.index, tensor_desc);
        input_descs.emplace_back(tensor_desc);
    }

    // Update Data nodes with full TensorDesc (dtype + format + shape)
    for (auto node : replaceGraph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type != "Data") {
            continue;
        }
        int64_t index = -1;
        node.GetAttr("index", index);
        if (index >= 0 && index < static_cast<int64_t>(input_descs.size())) {
            node.UpdateInputDesc(0, input_descs[index]);
            node.UpdateOutputDesc(0, input_descs[index]);
        }
    }

    std::vector<Shape> input_shapes;
    for (const auto& desc : input_descs) {
        input_shapes.emplace_back(desc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, input_shapes);
}

static void GetInputShapeAndDtype(
    std::vector<Shape>& input_shapes, std::vector<DataType>& input_dtypes, const std::vector<NodeIo>& input_nodes)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Begin get input_shape and input_dtype.");
    for (const auto& input_node : input_nodes) {
        AscendString node_type;
        input_node.node.GetType(node_type);
        TensorDesc tensor_desc;
        input_node.node.GetInputDesc(input_node.index, tensor_desc);
        if (node_type == "DynamicQuantV2") {
            // x input to DynamicQuantV2
            input_shapes[0] = tensor_desc.GetShape();
            input_dtypes[0] = tensor_desc.GetDataType();
        } else if (node_type == "Scatter" && input_node.index == 1) {
            // indices input (shared across all Scatters, port 1)
            input_shapes[1] = tensor_desc.GetShape();
            input_dtypes[1] = tensor_desc.GetDataType();
        } else if (node_type == "Scatter" && input_node.index == 0) {
            // var input directly to Scatter (no Identity case)
            input_shapes[2] = tensor_desc.GetShape();
            input_dtypes[2] = tensor_desc.GetDataType();
        } else if (node_type == "Identity") {
            // Identity before Scatter var input (inplace scatter case)
            // These carry the var/var_scale/var_offset shapes
            if (input_shapes[2].GetDims().empty()) {
                input_shapes[2] = tensor_desc.GetShape();
                input_dtypes[2] = tensor_desc.GetDataType();
            } else if (input_shapes[3].GetDims().empty()) {
                input_shapes[3] = tensor_desc.GetShape();
                input_dtypes[3] = tensor_desc.GetDataType();
            } else if (input_shapes[4].GetDims().empty()) {
                input_shapes[4] = tensor_desc.GetShape();
                input_dtypes[4] = tensor_desc.GetDataType();
            }
        }
    }
}

static void GetOutputDtype(
    std::vector<DataType>& output_dtypes, const std::vector<NodeIo>& input_nodes, const std::vector<NodeIo>& output_nodes)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Begin get output_dtype.");
    for (const auto& input_node : input_nodes) {
        AscendString node_type;
        input_node.node.GetType(node_type);
        TensorDesc tensor_desc;
        input_node.node.GetOutputDesc(input_node.index, tensor_desc);
        if (node_type == "DynamicQuantV2") {
            if (input_node.index == 0) {
                output_dtypes[0] = tensor_desc.GetDataType();
                break;
            }
        }
    }
    for (const auto& output_node : output_nodes) {
        AscendString node_type;
        output_node.node.GetType(node_type);
        TensorDesc tensor_desc;
        output_node.node.GetOutputDesc(output_node.index, tensor_desc);
        if (node_type == "Reshape") {
            // Check if this Reshape's data input (port 0) comes from Bitcast
            auto prev_data_node = output_node.node.GetInDataNodesAndPortIndexs(0);
            AscendString prev_node_name;
            prev_data_node.first->GetType(prev_node_name);
            if (prev_node_name == "Bitcast") {
                output_dtypes[1] = tensor_desc.GetDataType();
            }
        }
    }
}

std::vector<PatternUniqPtr> DynamicQuantUpdateScatterV2FusionPass::Patterns()
{
    OPS_LOG_I(PASS_NAME.c_str(), "Define pattern for DynamicQuantUpdateScatterV2Pass");
    std::vector<PatternUniqPtr> pattern_graphs;

    // Register both variants: with Identity (inplace scatter) and without
    MakePatternForDynamicQuantUpdateScatterV2(pattern_graphs, true);
    MakePatternForDynamicQuantUpdateScatterV2(pattern_graphs, false);

    return pattern_graphs;
}

bool DynamicQuantUpdateScatterV2FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Enter MeetRequirements");
    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);
    std::vector<SubgraphOutput> subgraph_outputs;
    match_result->ToSubgraphBoundary()->GetAllOutputs(subgraph_outputs);

    std::vector<NodeIo> input_nodes;
    for (const auto& subgraph_input : subgraph_inputs) {
        auto boundary_node = subgraph_input.GetAllInputs().at(0);
        input_nodes.emplace_back(boundary_node);
    }

    std::vector<NodeIo> output_nodes;
    for (const auto& subgraph_output : subgraph_outputs) {
        NodeIo node_o;
        subgraph_output.GetOutput(node_o);
        output_nodes.emplace_back(node_o);
    }

    if (!IsAllNodeNotNull(input_nodes)) {
        OPS_LOG_I(PASS_NAME.c_str(), "Check All NodeNotNull Fail");
        return false;
    }

    // 获取所有input tensor 的shape和dtype
    std::vector<Shape> input_shapes;
    input_shapes.resize(5);
    std::vector<DataType> input_dtypes;
    input_dtypes.resize(5);
    GetInputShapeAndDtype(input_shapes, input_dtypes, input_nodes);

    // 获取DynamicQuantV2和Reshape3 output tensor 的dytpe
    std::vector<DataType> output_dtypes;
    output_dtypes.resize(2);
    GetOutputDtype(output_dtypes, input_nodes, output_nodes);

    if (!IsTargetPlatform()) {
        OPS_LOG_I(PASS_NAME.c_str(), "CheckPlatform Fail");
        return false;
    }

    if (!IsAllInputShapeRight(input_shapes)) {
        OPS_LOG_I(PASS_NAME.c_str(), "CheckAllInputShape Fail");
        return false;
    }

    if (!IsAllInputDtypeRight(input_dtypes)) {
        OPS_LOG_I(PASS_NAME.c_str(), "CheckAllInputDtype Fail");
        return false;
    }

    if (!IsAllOutputDtypeRight(output_dtypes)) {
        OPS_LOG_I(PASS_NAME.c_str(), "CheckAllOutputDtype Fail");
        return false;
    }
    return true;
}

UniqueGraphPtr DynamicQuantUpdateScatterV2FusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    auto replace_graph_builder = es::EsGraphBuilder("replacement");
    auto r_x = replace_graph_builder.CreateInput(0);
    auto r_indices = replace_graph_builder.CreateInput(1);
    auto r_In0 = replace_graph_builder.CreateInput(2);
    auto r_In1 = replace_graph_builder.CreateInput(3);
    auto r_In2 = replace_graph_builder.CreateInput(4);

    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);

    es::DynamicQuantUpdateScatterV2Output dynamic_quant_update_scatter_v2 = es::DynamicQuantUpdateScatterV2(r_x, r_indices, r_In0, r_In1, r_In2);

    std::vector<es::EsTensorHolder> replace_outputs;
    replace_outputs.emplace_back(dynamic_quant_update_scatter_v2.ref_var);
    replace_outputs.emplace_back(dynamic_quant_update_scatter_v2.ref_var_scale);
    replace_outputs.emplace_back(dynamic_quant_update_scatter_v2.ref_var_offset);

    UniqueGraphPtr replaceGraph = replace_graph_builder.BuildAndReset(replace_outputs);

    OP_LOGW_IF(
        InferShape(replaceGraph, subgraph_inputs) != SUCCESS, PASS_NAME.c_str(), "Infershape for replacement failed.");
    return replaceGraph;
}

REG_FUSION_PASS(DynamicQuantUpdateScatterV2FusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ops
