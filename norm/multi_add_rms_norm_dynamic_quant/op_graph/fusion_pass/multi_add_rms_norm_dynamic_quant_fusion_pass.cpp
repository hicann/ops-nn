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
 * \file multi_add_rms_norm_dynamic_quant_fusion_pass.cpp
 * \brief quant fusion for multiaddrmsnorm
 *
 * Function: Define fusion multiple Add and AddRmsNorm and dynamicquant pattern;
 * Fusion pattern:
 *                                    ...
 *                     x1_0  x1_1 or Add_1
 *                       \   /                                  (x1: 1-5 list of tensor)
 *        gamma    x2    Add_0  (dynamic, 0-4 Add op)             x1     x2     gamma     s1     s2
 *            \    |    /                                           \     |      /      /       /
 *            AddRmsNorm                              ==>            MultiAddRmsNormDynamicQuant
 *           /     |                                               /     /     |     |     \     \
 *          x  (Reshape)                                  (Reshape) (Reshape)  |  (Reshape) \     \
 *                 |                                           /      /        |     |       \     \
 *              y--+      (s1)   (s2)                        y1      y2        x     y      scale1  scale2
 *                 |     *        *
 *               /   \*          *                        Note: Either all or none Reshape(s) are present.
 *              /  *  \         *
 *       DynamicQuant  (DynamicQuant) (1-2 dynamic DQ)
 *       /         \      /        \
 *     y1       scale1  y2       scale2
 *
 */

#include "multi_add_rms_norm_dynamic_quant_fusion_pass.h"

#include <string>
#include <vector>

#include "ge/ge_utils.h"
#include "common/inc/error_util.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "compliant_node_builder.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {

static const std::string FUSED_OP_TYPE = "MultiAddRmsNormDynamicQuant";
static const std::string SOC_ASCEND910B = "Ascend910b";
static const std::string SOC_ASCEND910_93 = "Ascend910_93";

using UniqueGraphPtr = std::unique_ptr<Graph>;
static const std::string PASS_NAME = "MultiAddRmsNormDynamicQuantFusionPass";

static es::EsTensorHolder BuildPatternConstNode(es::EsGraphBuilder& builder)
{
    static int counter = 0;
    std::string name = "Const_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Const")
        .Name(name.c_str())
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildPatternReshapeNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x, const es::EsTensorHolder& shape)
{
    static int counter = 0;
    std::string name = "Reshape_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Reshape")
        .Name(name.c_str())
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *shape.GetProducer(), shape.GetProducerOutIndex(), node, 1);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static constexpr int64_t OUTPUT_IDX_Y1 = 0;
static constexpr int64_t OUTPUT_IDX_Y2 = 1;
static constexpr int64_t OUTPUT_IDX_X = 2;
static constexpr int64_t OUTPUT_IDX_Y = 3;
static constexpr int64_t OUTPUT_IDX_SCALE1 = 4;
static constexpr int64_t OUTPUT_IDX_SCALE2 = 5;

static bool IsTargetPlatform()
{
    OPS_LOG_I(PASS_NAME.c_str(), "Begin check platform information");
    PlatformInfo platform_info;
    OptionalInfo optional_info;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
        false, PASS_NAME.c_str(), "Get platform_info failed.");

    const string soc = platform_info.str_info.short_soc_version;
    bool is_platform910b = soc == SOC_ASCEND910B;
    bool is_platform910_93 = soc == SOC_ASCEND910_93;
    OPS_LOG_I(PASS_NAME.c_str(), "Platform short soc: %s", soc.c_str());
    OP_LOGE_IF(
        !is_platform910b && !is_platform910_93, false, PASS_NAME.c_str(),
        "Platform is not support, only work on 910b or 910_93.");
    return true;
}

static bool IsAllNodeNotNull(const std::vector<NodeIo>& input_nodes)
{
    AscendString node_type;
    for (const auto& input_node : input_nodes) {
        OPS_LOG_I(PASS_NAME.c_str(), "Check if node not null.");
        input_node.node.GetType(node_type);
        OP_LOGE_IF(
            &input_node.node == nullptr, false, PASS_NAME.c_str(), "node %s can't be null.", node_type.GetString());
    }
    OPS_LOG_I(PASS_NAME.c_str(), "All nodes not null.");
    return true;
}

static Status InferShape(UniqueGraphPtr& replace_graph, const std::vector<SubgraphInput>& subgraph_inputs)
{
    std::vector<ge::TensorDesc> input_descs;
    for (const auto& subgraph_input : subgraph_inputs) {
        auto match_node = subgraph_input.GetAllInputs().at(0);
        TensorDesc tensor_desc;
        match_node.node.GetInputDesc(match_node.index, tensor_desc);
        input_descs.emplace_back(tensor_desc);
    }

    // Update Data nodes in replacement graph with full TensorDesc (shape + dtype + format)
    for (auto node : replace_graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type != "Data") {
            continue;
        }
        int64_t index = -1;
        node.GetAttr("index", index);
        if (index < 0 || index >= static_cast<int64_t>(input_descs.size())) {
            continue;
        }
        node.UpdateOutputDesc(0, input_descs[index]);
        node.UpdateInputDesc(0, input_descs[index]);
    }

    std::vector<Shape> input_shapes;
    for (const auto& desc : input_descs) {
        input_shapes.emplace_back(desc.GetShape());
    }
    return GeUtils::InferShape(*replace_graph, input_shapes);
}

struct AddRmsNormResult {
    es::EsTensorHolder x;
    es::EsTensorHolder y;
    es::EsTensorHolder rstd;
};

static AddRmsNormResult BuildAddRmsNormNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x1,
    const es::EsTensorHolder& x2, const es::EsTensorHolder& gamma)
{
    static int counter = 0;
    std::string name = "AddRmsNorm_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("AddRmsNorm")
        .Name(name.c_str())
        .IrDefInputs({
            {"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"gamma", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"rstd", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"x", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .IrDefAttrs({
            {"epsilon", es::CompliantNodeBuilder::kEsAttrOptional, "Float",
             es::CreateFrom(1e-6f)}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *gamma.GetProducer(), gamma.GetProducerOutIndex(), node, 2);
    return {
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 2)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1))
    };
}

struct DynamicQuantResult {
    es::EsTensorHolder y;
    es::EsTensorHolder scale;
};

static DynamicQuantResult BuildDynamicQuantNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x)
{
    static int counter = 0;
    std::string name = "DynamicQuant_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("DynamicQuant")
        .Name(name.c_str())
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"smooth_scales", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"group_index", es::CompliantNodeBuilder::kEsIrInputOptional, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"scale", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return {
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1))
    };
}

static DynamicQuantResult BuildDynamicQuantNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x, const es::EsTensorHolder& smooth)
{
    static int counter = 0;
    std::string name = "DynamicQuant_s_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("DynamicQuant")
        .Name(name.c_str())
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"smooth_scales", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"group_index", es::CompliantNodeBuilder::kEsIrInputOptional, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"scale", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *smooth.GetProducer(), smooth.GetProducerOutIndex(), node, 1);
    return {
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1))
    };
}

static es::EsTensorHolder BuildAddChain(
    es::EsGraphBuilder& graph_bdr, es::EsGraphBuilder& dummy_bdr, int64_t p_num_add_nodes)
{
    auto x1_0 = graph_bdr.CreateInput(0);
    auto x1_1 = p_num_add_nodes > 0 ? graph_bdr.CreateInput(1) : dummy_bdr.CreateInput(0);
    auto x1_2 = p_num_add_nodes > 1 ? graph_bdr.CreateInput(2) : dummy_bdr.CreateInput(1);
    auto x1_3 = p_num_add_nodes > 2 ? graph_bdr.CreateInput(3) : dummy_bdr.CreateInput(2);
    auto x1_4 = p_num_add_nodes > 3 ? graph_bdr.CreateInput(4) : dummy_bdr.CreateInput(3);

    if (p_num_add_nodes == 0) {
        return x1_0;
    }
    auto x1 = es::Add(x1_0, x1_1);
    if (p_num_add_nodes == 1) {
        return x1;
    }
    x1 = es::Add(x1, x1_2);
    if (p_num_add_nodes == 2) {
        return x1;
    }
    x1 = es::Add(x1, x1_3);
    if (p_num_add_nodes == 3) {
        return x1;
    }
    return es::Add(x1, x1_4);
}

static void MakePatternForMultiAddRmsNormDynamicQuant(
    std::vector<PatternUniqPtr>& pattern_graphs, int64_t p_num_add_nodes, int64_t p_num_smooth, bool p_has_reshape)
{
    std::string pattern_name = "pattern" + std::to_string(p_num_add_nodes) + "Add" +
        std::to_string(p_num_smooth) + "DQ" + (p_has_reshape ? "Reshape" : "");
    auto graph_bdr = es::EsGraphBuilder(pattern_name.c_str());
    auto dummy_bdr = es::EsGraphBuilder("dummy");

    auto x1 = BuildAddChain(graph_bdr, dummy_bdr, p_num_add_nodes);
    auto x2 = graph_bdr.CreateInput(1 + p_num_add_nodes);
    auto gamma = graph_bdr.CreateInput(2 + p_num_add_nodes);

    auto smooth1 = p_num_smooth > 0 ? graph_bdr.CreateInput(3 + p_num_add_nodes) : dummy_bdr.CreateInput(4);
    auto smooth2 = p_num_smooth > 1 ? graph_bdr.CreateInput(4 + p_num_add_nodes) : dummy_bdr.CreateInput(5);

    auto arnResult = BuildAddRmsNormNode(graph_bdr, x1, x2, gamma);
    auto y_reshape = p_has_reshape ? BuildPatternReshapeNode(graph_bdr, arnResult.y, BuildPatternConstNode(graph_bdr))
                                   : arnResult.y;

    auto dq1Result = p_num_smooth == 0 ? BuildDynamicQuantNode(graph_bdr, y_reshape)
                                       : BuildDynamicQuantNode(graph_bdr, y_reshape, smooth1);

    std::vector<es::EsTensorHolder> outputs;
    outputs.emplace_back(std::move(dq1Result.y));
    if (p_num_smooth == 2) {
        auto dq2Result = BuildDynamicQuantNode(graph_bdr, y_reshape, smooth2);
        outputs.emplace_back(std::move(dq2Result.y));
        outputs.emplace_back(std::move(arnResult.x));
        outputs.emplace_back(std::move(y_reshape));
        outputs.emplace_back(std::move(dq1Result.scale));
        outputs.emplace_back(std::move(dq2Result.scale));
    } else {
        outputs.emplace_back(std::move(arnResult.x));
        outputs.emplace_back(std::move(y_reshape));
        outputs.emplace_back(std::move(dq1Result.scale));
    }

    auto graph = graph_bdr.BuildAndReset(outputs);
    pattern_graphs.emplace_back(std::make_unique<Pattern>(std::move(*graph)));
}

static std::tuple<int64_t, int64_t, bool> GetMatchedNodes(
    const std::vector<NodeIo>& input_nodes, std::vector<NodeIo>& matched_adds,
    std::vector<NodeIo>& matched_addrns)
{
    OPS_LOG_I(PASS_NAME.c_str(), "GetMatchedNodes for specify subgraph.");
    int64_t p_add_num = 0;
    int64_t p_smooth_num = 0;
    bool has_reshape = false;
    for (const auto& input_node : input_nodes) {
        AscendString node_type;
        input_node.node.GetType(node_type);
        TensorDesc tensor_desc;
        input_node.node.GetInputDesc(input_node.index, tensor_desc);
        if (node_type == "Add") {
            p_add_num++;
            matched_adds.emplace_back(input_node);
        }
        if (node_type == "AddRmsNorm") {
            matched_addrns.emplace_back(input_node);
        }
        if (node_type == "DynamicQuant") {
            p_smooth_num++;
            auto prev_node = input_node.node.GetInDataNodesAndPortIndexs(0);
            if (prev_node.first != nullptr) {
                AscendString prev_node_name;
                prev_node.first->GetType(prev_node_name);
                if (prev_node_name == "Reshape") {
                    has_reshape = true;
                }
            }
        }
    }
    return std::make_tuple(p_add_num, p_smooth_num, has_reshape);
}

static bool IsMaxAdd(const std::vector<NodeIo>& input_nodes)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Check IsMaxAdd for specify subgraph.");
    std::vector<NodeIo> local_matched_adds;
    for (const auto& input_node : input_nodes) {
        AscendString node_type;
        input_node.node.GetType(node_type);
        if (node_type == "Add") {
            local_matched_adds.emplace_back(input_node);
        }
    }
    for (const auto& add_node : local_matched_adds) {
        auto prev_second_node = add_node.node.GetInDataNodesAndPortIndexs(1);
        AscendString prev_node_name;
        prev_second_node.first->GetType(prev_node_name);
        if (prev_node_name == "Add") {
            return false;
        }
    }
    return true;
}

static bool IsMaxSmooth(const std::vector<NodeIo>& input_nodes, int64_t& patern_smooth_num)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Check IsMaxSmooth for specify subgraph.");
    int64_t smooth_cnt = 0;
    for (const auto& input_node : input_nodes) {
        AscendString node_type;
        input_node.node.GetType(node_type);
        if (node_type == "DynamicQuant") {
            smooth_cnt++;
            // check if preceding node is AddRmsNorm (unmatched external DQ)
            auto prev_second_node = input_node.node.GetInDataNodesAndPortIndexs(1);
            AscendString prev_node_name;
            prev_second_node.first->GetType(prev_node_name);
            if (prev_node_name == "AddRmsNorm") {
                return false;
            }
        }
    }
    patern_smooth_num = smooth_cnt;
    return true;
}

static bool IsMaxSubGraph(
    const std::vector<NodeIo>& input_nodes, int64_t& patern_add_num, int64_t& patern_smooth_num, bool& out_has_reshape)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Check IsMaxSubGraph for specify subgraph.");
    std::vector<NodeIo> matched_adds;
    std::vector<NodeIo> matched_addrns;

    bool has_reshape = false;
    std::tie(patern_add_num, patern_smooth_num, has_reshape) =
        GetMatchedNodes(input_nodes, matched_adds, matched_addrns);

    if (!IsMaxAdd(input_nodes)) {
        OP_LOGW_IF(true, PASS_NAME.c_str(), "Check IsMaxAdd failed.");
        return false;
    }
    if (!IsMaxSmooth(input_nodes, patern_smooth_num)) {
        OP_LOGW_IF(true, PASS_NAME.c_str(), "Check IsMaxSmooth failed.");
        return false;
    }

    out_has_reshape = has_reshape;
    return true;
}

static GNode BuildFusedNode(
    es::EsGraphBuilder& replace_bdr, std::vector<es::EsTensorHolder>& inputs,
    int64_t add_num)
{
    auto graph = replace_bdr.GetCGraphBuilder()->GetGraph();

    GNode node = es::CompliantNodeBuilder(graph)
        .OpType(FUSED_OP_TYPE.c_str())
        .Name(FUSED_OP_TYPE.c_str())
        .IrDefInputs({
            {"x1", es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
            {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"gamma", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"smooth_scale1", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"smooth_scale2", es::CompliantNodeBuilder::kEsIrInputOptional, ""}
        })
        .IrDefOutputs({
            {"y1", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"y2", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"x", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"scale1", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"scale2", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .IrDefAttrs({
            {"epsilon", es::CompliantNodeBuilder::kEsAttrOptional, "Float",
             es::CreateFrom(1e-6f)}
        })
        .InstanceDynamicInputNum("x1", static_cast<int32_t>(add_num + 1))
        .Build();

    for (size_t i = 0; i < inputs.size(); i++) {
        es::AddEdgeAndUpdatePeerDesc(
            *graph, *inputs[i].GetProducer(), inputs[i].GetProducerOutIndex(),
            node, static_cast<int32_t>(i));
    }

    return node;
}

static UniqueGraphPtr BuildReplaceGraph(
    es::EsGraphBuilder& replace_bdr, GNode& fused_node,
    int64_t smooth_num, bool has_reshape, const es::EsTensorHolder& reshape_shape)
{
    (void)has_reshape;
    (void)reshape_shape;
    auto get_output = [&](int64_t idx) {
        return es::EsTensorHolder(
            replace_bdr.GetCGraphBuilder()->GetTensorHolderFromNode(fused_node, idx));
    };

    // Output fused node results directly without Reshape.
    // The fused op's InferShape on host side will compute the correct output shapes.
    std::vector<es::EsTensorHolder> outputs;
    outputs.emplace_back(get_output(OUTPUT_IDX_Y1));
    if (smooth_num == 2) {
        outputs.emplace_back(get_output(OUTPUT_IDX_Y2));
    }
    outputs.emplace_back(get_output(OUTPUT_IDX_X));
    outputs.emplace_back(get_output(OUTPUT_IDX_Y));
    outputs.emplace_back(get_output(OUTPUT_IDX_SCALE1));
    if (smooth_num == 2) {
        outputs.emplace_back(get_output(OUTPUT_IDX_SCALE2));
    }

    return replace_bdr.BuildAndReset(outputs);
}

std::vector<PatternUniqPtr> MultiAddRmsNormDynamicQuantFusionPass::Patterns()
{
    OPS_LOG_I(PASS_NAME.c_str(), "Define pattern for MultiAddRmsNormDynamicQuantFusionPass");
    std::vector<PatternUniqPtr> pattern_graphs;

    for (int64_t p_num_smooth = MAX_SMOOTH_NUM; p_num_smooth >= 0; p_num_smooth--) {
        for (int64_t p_num_add_nodes = MAX_ADD_NUM; p_num_add_nodes >= 0; p_num_add_nodes--) {
            MakePatternForMultiAddRmsNormDynamicQuant(pattern_graphs, p_num_add_nodes, p_num_smooth, true);
            MakePatternForMultiAddRmsNormDynamicQuant(pattern_graphs, p_num_add_nodes, p_num_smooth, false);
        }
    }

    return pattern_graphs;
}

bool MultiAddRmsNormDynamicQuantFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Enter MeetRequirements");
    if (!IsTargetPlatform()) {
        OPS_LOG_I(PASS_NAME.c_str(), "CheckPlatform Fail");
        return false;
    }

    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);
    std::vector<SubgraphOutput> subgraph_outputs;
    match_result->ToSubgraphBoundary()->GetAllOutputs(subgraph_outputs);

    std::vector<NodeIo> input_nodes;
    for (const auto& subgraph_input : subgraph_inputs) {
        auto boundary_node = subgraph_input.GetAllInputs().at(0);
        input_nodes.emplace_back(boundary_node);
    }

    if (!IsAllNodeNotNull(input_nodes)) {
        OPS_LOG_I(PASS_NAME.c_str(), "Check All NodeNotNull Fail");
        return false;
    }
    // check if num of Add nodes properly matched
    int64_t patern_add_num = 0;
    int64_t patern_smooth_num = 0;
    bool out_has_reshape = false;
    if (!IsMaxSubGraph(input_nodes, patern_add_num, patern_smooth_num, out_has_reshape)) {
        OPS_LOG_I(PASS_NAME.c_str(), "Check IsMaxSubGraph Fail");
        return false;
    }

    this->add_num_ = patern_add_num > 0 ? patern_add_num - 1 : 0;
    this->smooth_num_ = patern_smooth_num;
    this->has_reshape_ = out_has_reshape;
    return true;
}

UniqueGraphPtr MultiAddRmsNormDynamicQuantFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    auto replace_bdr = es::EsGraphBuilder("replacement");

    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);

    // Create exactly subgraph_inputs.size() inputs to match boundary inputs count.
    // Order: x1[0..add_num], x2, gamma, smooth_scale1?, smooth_scale2?
    // Any extra boundary inputs (e.g. Const for Reshape shape) are created but unused by fused node.
    const int64_t num_boundary_inputs = static_cast<int64_t>(subgraph_inputs.size());
    std::vector<es::EsTensorHolder> all_inputs;
    for (int64_t i = 0; i < num_boundary_inputs; i++) {
        all_inputs.push_back(replace_bdr.CreateInput(i));
    }

    // Collect inputs for fused node: x1[0..add_num] + x2 + gamma + smooth_scale(s)
    std::vector<es::EsTensorHolder> fused_inputs;
    int64_t fused_idx = 0;
    for (int64_t i = 0; i <= this->add_num_; i++) {
        fused_inputs.push_back(all_inputs[fused_idx++]);
    }
    fused_inputs.push_back(all_inputs[fused_idx++]); // x2
    fused_inputs.push_back(all_inputs[fused_idx++]); // gamma
    if (this->smooth_num_ > 0) {
        fused_inputs.push_back(all_inputs[fused_idx++]); // smooth_scale1
    }
    if (this->smooth_num_ > 1) {
        fused_inputs.push_back(all_inputs[fused_idx++]); // smooth_scale2
    }

    es::EsTensorHolder reshape_shape; // unused placeholder
    GNode fused_node = BuildFusedNode(replace_bdr, fused_inputs, this->add_num_);
    UniqueGraphPtr replace_graph = BuildReplaceGraph(
        replace_bdr, fused_node, this->smooth_num_, this->has_reshape_, reshape_shape);

    OP_LOGW_IF(
        InferShape(replace_graph, subgraph_inputs) != SUCCESS, PASS_NAME.c_str(),
        "Infershape for replacement failed.");
    return std::move(replace_graph);
}

REG_FUSION_PASS(MultiAddRmsNormDynamicQuantFusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ops
