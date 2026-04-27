/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "common/inc/error_util.h"
#include "prelu_grad_fusion_pass.h"
#include "es_nn_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "compliant_node_builder.h"

using namespace ge;
using namespace fe;
using namespace fusion;

/**
 * @brief Define fusion PReluGradUpdate pattern
 * @details Split PReluGrad into PReluGradUpdate and PReluGradReduce
 *                  dy, x, weight                dy, x, weight
 *                    |       |                     |       |
 *                  PReluGrad        ==>      PReluGradUpdate
 *                   /    \                           /    \
 *                 dx     da                      dx    update
 *                                                  |
 *                                            PReluGradReduce
 *                                                  |
 *                                                 da
 */

namespace ops {

static const std::string PASS_NAME = "PReluGradFusionPass";
const int64_t CAPTURE_TENSOR_IDX_PRELU_GRAD = 0l;

std::vector<es::EsTensorHolder> CreatePatternPreluGrad(
    es::EsGraphBuilder &graph_builder, const es::EsTensorHolder &dy,
    const es::EsTensorHolder &x, const es::EsTensorHolder &weight)
{
    auto *graph = graph_builder.GetCGraphBuilder()->GetGraph();
    auto prelu_grad = es::CompliantNodeBuilder(graph)
        .OpType("PReluGrad")
        .Name("PReluGrad")
        .IrDefInputs({
            {"dy", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"weight", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({
            {"dx", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"da", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        })
        .Build();
    std::vector<es::EsTensorHolder> default_res = {es::EsTensorHolder(), es::EsTensorHolder()};
    OP_LOGE_IF(
        es::AddEdgeAndUpdatePeerDesc(*graph, *dy.GetProducer(), dy.GetProducerOutIndex(), prelu_grad, 0) != GRAPH_SUCCESS,
        default_res, PASS_NAME.c_str(), "AddEdgeAndUpdatePeerDesc failed.");
    OP_LOGE_IF(
        es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), prelu_grad, 1) != GRAPH_SUCCESS,
        default_res, PASS_NAME.c_str(), "AddEdgeAndUpdatePeerDesc failed.");
    OP_LOGE_IF(
        es::AddEdgeAndUpdatePeerDesc(*graph, *weight.GetProducer(), weight.GetProducerOutIndex(), prelu_grad, 2) != GRAPH_SUCCESS,
        default_res, PASS_NAME.c_str(), "AddEdgeAndUpdatePeerDesc failed.");
    
    auto *dx_holder = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(prelu_grad, 0);
    auto *da_holder = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(prelu_grad, 1);
    return {es::EsTensorHolder(dx_holder), es::EsTensorHolder(da_holder)};
}

std::vector<es::EsTensorHolder> BuildUpdateNode(
    es::EsGraphBuilder &graph_builder, const es::EsTensorHolder &r_dy,
    const es::EsTensorHolder &r_x, const es::EsTensorHolder &r_weight)
{
    auto replace_graph = graph_builder.GetCGraphBuilder()->GetGraph();
    auto prelu_grad_update = es::CompliantNodeBuilder(replace_graph)
        .OpType("PReluGradUpdate")
        .Name("PReluGradUpdate")
        .IrDefInputs({
            {"dy", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"weight", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({
            {"dx", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"update", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        })
        .Build();
    // Connect input to ReduceSum node using AddEdgeAndUpdatePeerDesc
    es::AddEdgeAndUpdatePeerDesc(*replace_graph, *r_dy.GetProducer(), r_dy.GetProducerOutIndex(), prelu_grad_update, 0);
    es::AddEdgeAndUpdatePeerDesc(*replace_graph, *r_x.GetProducer(), r_x.GetProducerOutIndex(), prelu_grad_update, 1);
    es::AddEdgeAndUpdatePeerDesc(*replace_graph, *r_weight.GetProducer(), r_weight.GetProducerOutIndex(), prelu_grad_update, 2);
    
    auto res = {es::EsTensorHolder(graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(prelu_grad_update, 0)),
                es::EsTensorHolder(graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(prelu_grad_update, 1))};
    return res;
}

es::EsTensorHolder BuildReduceNode(
    es::EsGraphBuilder &graph_builder, const es::EsTensorHolder &r_dy,
    const es::EsTensorHolder &r_x, const es::EsTensorHolder &r_weight,
    const es::EsTensorHolder &update_holder)
{
    auto replace_graph = graph_builder.GetCGraphBuilder()->GetGraph();
    auto prelu_grad_reduce = es::CompliantNodeBuilder(replace_graph)
        .OpType("PReluGradReduce")
        .Name("PReluGradReduce")
        .IrDefInputs({
            {"dy", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"weight", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"update", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({
            {"da", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        })
        .Build();

    es::AddEdgeAndUpdatePeerDesc(*replace_graph, *r_dy.GetProducer(), r_dy.GetProducerOutIndex(), prelu_grad_reduce, 0);
    es::AddEdgeAndUpdatePeerDesc(*replace_graph, *r_x.GetProducer(), r_x.GetProducerOutIndex(), prelu_grad_reduce, 1);
    es::AddEdgeAndUpdatePeerDesc(*replace_graph, *r_weight.GetProducer(), r_weight.GetProducerOutIndex(), prelu_grad_reduce, 2);
    es::AddEdgeAndUpdatePeerDesc(*replace_graph, *update_holder.GetProducer(), update_holder.GetProducerOutIndex(), prelu_grad_reduce, 3);

    return es::EsTensorHolder(graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(prelu_grad_reduce, 0));
}

std::vector<PatternUniqPtr> PReluGradFusionPass::Patterns()
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Patterns for PReluGradFusionPass");
    std::vector<PatternUniqPtr> pattern_graphs;
    auto graph_builder = es::EsGraphBuilder(PASS_NAME.c_str());

    auto dy = graph_builder.CreateInput(0);
    auto x = graph_builder.CreateInput(1);
    auto weight = graph_builder.CreateInput(2);

    auto prelu_grad_out = CreatePatternPreluGrad(graph_builder, dy, x, weight);
    auto graph = graph_builder.BuildAndReset(prelu_grad_out);
    auto pattern = std::make_unique<Pattern>(std::move(*graph));

    pattern->CaptureTensor({*dy.GetProducer(), 0})
            .CaptureTensor({*x.GetProducer(), 0})
            .CaptureTensor({*weight.GetProducer(), 0})
            .CaptureTensor({*prelu_grad_out[0].GetProducer(), 0})
            .CaptureTensor({*prelu_grad_out[1].GetProducer(), 0});
    pattern_graphs.emplace_back(std::move(pattern));
    return pattern_graphs;
}

static bool IsTargetPlatform()
{
    PlatformInfo platform_info;
    OptionalInfo optional_info;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
        false, PASS_NAME.c_str(), "Get platform_info failed.");
    const std::string soc = platform_info.str_info.short_soc_version;
    bool is_platform950 = (soc == "Ascend950");
    OPS_LOG_D(PASS_NAME.c_str(), "Platform short soc: %s", soc.c_str());
    if (!is_platform950) {
        OPS_LOG_D(PASS_NAME.c_str(), "Platform is not support, only work on Ascend950.");
        return false;
    }
    return true;
}

bool PReluGradFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter MeetRequirements for PReluGradFusionPass");

    if (!IsTargetPlatform()) {
        OPS_LOG_D(PASS_NAME.c_str(), "Check platform fail");
        return false;
    }

    return true;
}

static void SetSubNodeDesc(GNode& update_node, GNode& reduce_node,
    es::EsTensorHolder& dy, es::EsTensorHolder& x, es::EsTensorHolder& weight)
{
    TensorDesc dy_desc;
    TensorDesc x_desc;
    TensorDesc w_desc;
    dy.GetProducer()->GetOutputDesc(0, dy_desc);
    x.GetProducer()->GetOutputDesc(0, x_desc);
    weight.GetProducer()->GetOutputDesc(0, w_desc);
    
    update_node.UpdateInputDesc(0, dy_desc);
    update_node.UpdateInputDesc(1, x_desc);
    update_node.UpdateInputDesc(2, w_desc);
    update_node.UpdateOutputDesc(0, x_desc);
    update_node.UpdateOutputDesc(1, x_desc);

    reduce_node.UpdateInputDesc(0, dy_desc);
    reduce_node.UpdateInputDesc(1, x_desc);
    reduce_node.UpdateInputDesc(2, w_desc);
    reduce_node.UpdateInputDesc(3, x_desc);
    reduce_node.UpdateOutputDesc(0, w_desc);
} 

GraphUniqPtr PReluGradFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Replacement for PReluGradFusionPass");
    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);

    std::vector<Shape> input_shapes;
    std::vector<DataType> input_dtypes;
    std::vector<Format> input_formats;
    GetInputsInfo(subgraph_inputs, input_shapes, input_dtypes, input_formats);

    auto replace_graph_builder = es::EsGraphBuilder("replacement");
    auto r_dy = replace_graph_builder.CreateInput(0, "dy", input_dtypes[0], input_formats[0], input_shapes[0].GetDims());
    auto r_x = replace_graph_builder.CreateInput(1, "x", input_dtypes[1], input_formats[1], input_shapes[1].GetDims());
    auto r_weight = replace_graph_builder.CreateInput(2, "weight", input_dtypes[2], input_formats[2], input_shapes[2].GetDims());
    auto replace_graph = replace_graph_builder.GetCGraphBuilder()->GetGraph();

    auto p_relu_grad_update_out = BuildUpdateNode(replace_graph_builder, r_dy, r_x, r_weight);
    auto p_relu_grad_reduce_out = BuildReduceNode(replace_graph_builder, r_dy, r_x, r_weight, p_relu_grad_update_out[1]);

    GraphUniqPtr res_graph = replace_graph_builder.BuildAndReset({es::EsTensorHolder(p_relu_grad_update_out[0]),
                                                                  es::EsTensorHolder(p_relu_grad_reduce_out)});

    GNode update_node = *p_relu_grad_update_out[0].GetProducer();
    GNode reduce_node = *p_relu_grad_reduce_out.GetProducer();
    SetSubNodeDesc(update_node, reduce_node, r_dy, r_x, r_weight);
    if (InferShape(res_graph, subgraph_inputs) != SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }

    return std::move(res_graph);
}

static void GetInputsInfo(
    const std::vector<SubgraphInput>& subgraph_inputs, std::vector<Shape>& input_shape,
    std::vector<DataType>& input_dtype, std::vector<Format>& input_format)
{
    for (const auto& sub_input : subgraph_inputs) {
        auto match_node = sub_input.GetAllInputs().at(0);
        TensorDesc tensor_desc;
        match_node.node.GetInputDesc(match_node.index, tensor_desc);
        input_shape.emplace_back(tensor_desc.GetShape());
        input_dtype.emplace_back(tensor_desc.GetDataType());
        input_format.emplace_back(tensor_desc.GetFormat());
    }
}

static Status InferShape(const GraphUniqPtr& replace_graph, const std::vector<SubgraphInput>& subgraph_inputs)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Begin infershape for replacements.");
    std::vector<Shape> input_shapes;
    for (const auto& subgraph_input : subgraph_inputs) {
        auto match_node = subgraph_input.GetAllInputs().at(0);
        TensorDesc tensor_desc;
        match_node.node.GetInputDesc(match_node.index, tensor_desc);
        input_shapes.emplace_back(tensor_desc.GetShape());
    }
    return GeUtils::InferShape(*replace_graph, input_shapes);
}

REG_FUSION_PASS(PReluGradFusionPass).Stage(CustomPassStage::kAfterInferShape);
} // namespace ops
