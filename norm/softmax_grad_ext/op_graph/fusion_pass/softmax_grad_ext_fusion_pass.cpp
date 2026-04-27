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
#include "softmax_grad_ext_fusion_pass.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "ge/compliant_node_builder.h"
#include "es_nn_ops.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {
namespace {
const std::string kPassName = "SoftmaxGradExtFusionPass";
const std::string kPassNameV2 = "SoftmaxGradExtV2FusionPass";

const int64_t kCaptureMulIdx = 0;
const int64_t kCaptureMul1Idx = 1;
const int64_t kCaptureMulGradIdx = 2;
const int64_t kCaptureSumIdx = 3;
const int64_t kCaptureSubIdx = 4;

bool IsUnknownShape(const std::vector<int64_t>& dims)
{
    for (auto dim : dims) {
        if (dim == -1) {
            return true;
        }
    }
    return false;
}

void GetInputsInfo(
    const std::vector<SubgraphInput>& subgraph_inputs,
    std::vector<Shape>& input_shapes,
    std::vector<DataType>& input_dtypes,
    std::vector<Format>& input_formats)
{
    for (const auto& subgraph_input : subgraph_inputs) {
        auto match_node = subgraph_input.GetAllInputs().at(0);
        TensorDesc tensor_desc;
        match_node.node.GetInputDesc(match_node.index, tensor_desc);
        input_shapes.emplace_back(tensor_desc.GetShape());
        input_dtypes.emplace_back(tensor_desc.GetDataType());
        input_formats.emplace_back(tensor_desc.GetFormat());
    }
}

Status InferShape(
    const GraphUniqPtr& replace_graph,
    const std::vector<SubgraphInput>& subgraph_inputs)
{
    OPS_LOG_D(kPassName.c_str(), "Begin infershape for replacements.");
    std::vector<Shape> input_shapes;
    for (const auto& subgraph_input : subgraph_inputs) {
        auto match_node = subgraph_input.GetAllInputs().at(0);
        TensorDesc tensor_desc;
        match_node.node.GetInputDesc(match_node.index, tensor_desc);
        input_shapes.emplace_back(tensor_desc.GetShape());
    }
    return GeUtils::InferShape(*replace_graph, input_shapes);
}

// Helper function to build TwoInputOneOutput node using CompliantNodeBuilder
es::EsTensorHolder BuildTwoInOneOutNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& input0,
    const es::EsTensorHolder& input1, const char *op_type)
{
    ge::Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType(op_type)
        .Name(op_type)
        .IrDefInputs({
            {"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .Build();
    
    // Connect input to two input one output node using AddEdgeAndUpdatePeerDesc
    es::AddEdgeAndUpdatePeerDesc(*graph, *input0.GetProducer(), input0.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *input1.GetProducer(), input1.GetProducerOutIndex(), node, 1);
    
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

// Helper function to build ReduceSum node using CompliantNodeBuilder
// Note: EsReduceSum not available in library, must use CompliantNodeBuilder
es::EsTensorHolder BuildReduceSum(es::EsGraphBuilder& builder, const es::EsTensorHolder& input, int64_t axis, bool keep_dims)
{
    ge::Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    
    GNode reduce_sum_node = es::CompliantNodeBuilder(graph)
        .OpType("ReduceSum")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .IrDefAttrs({
            {"axes", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt", es::CreateFrom(std::vector<int64_t>{axis})},
            {"keep_dims", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(keep_dims)}
        })
        .Build();
    
    // Connect input to ReduceSum node using AddEdgeAndUpdatePeerDesc
    GNode* input_node = input.GetProducer();
    int32_t input_out_idx = input.GetProducerOutIndex();
    es::AddEdgeAndUpdatePeerDesc(*graph, *input_node, input_out_idx, reduce_sum_node, 0);
    
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(reduce_sum_node, 0));
}

es::EsTensorHolder BuildSoftmaxGradExt(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& r_input0,
    const es::EsTensorHolder& r_input1, const es::EsTensorHolder& r_input2,
    int64_t axis_value, bool keep_dims)
{
    ge::Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    
    GNode softmax_grad_ext_node = es::CompliantNodeBuilder(graph)
        .OpType("SoftmaxGradExt")
        .IrDefInputs({
            {"grad", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .IrDefAttrs({
            {"axes", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(axis_value)},
            {"keep_dims", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(keep_dims)}
        })
        .Build();

    // Connect inputs to SoftmaxGradExt node
    es::AddEdgeAndUpdatePeerDesc(*graph, *r_input0.GetProducer(), r_input0.GetProducerOutIndex(), softmax_grad_ext_node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *r_input1.GetProducer(), r_input1.GetProducerOutIndex(), softmax_grad_ext_node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *r_input2.GetProducer(), r_input2.GetProducerOutIndex(), softmax_grad_ext_node, 2);
    
    return es::EsTensorHolder(
        builder.GetCGraphBuilder()->GetTensorHolderFromNode(softmax_grad_ext_node, 0));
}

// Create pattern for SoftmaxGradExtFusionPass
// Pattern: mul = Mul(input0, input1), sum = ReduceSum(mul), sub = Sub(input0, sum),
//          mul1 = Mul(input2, input1), mulGrad = Mul(mul1, sub)
// Note: Use operator overloads for Mul/Sub (EsMul, EsSub available via operator overload), 
//       CompliantNodeBuilder for ReduceSum (EsReduceSum not available)
PatternUniqPtr MakePatternSoftmaxGradExt()
{
    auto graph_builder = es::EsGraphBuilder(kPassName.c_str());
    auto input0 = graph_builder.CreateInput(0, "input0");
    auto input1 = graph_builder.CreateInput(1, "input1");
    auto input2 = graph_builder.CreateInput(2, "input2");
    
    // Use operator overloads for Mul/Sub (internally uses EsMul, EsSub)
    auto mul = BuildTwoInOneOutNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildReduceSum(graph_builder, mul, -1, true); 
    auto sub = BuildTwoInOneOutNode(graph_builder, input0, sum, "Sub");
    auto mul1 = BuildTwoInOneOutNode(graph_builder, input2, input1, "Mul");
    auto mulGrad = BuildTwoInOneOutNode(graph_builder, mul1, sub, "Mul");

    auto graph = graph_builder.BuildAndReset({mulGrad});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    
    pattern->CaptureTensor({*mul.GetProducer(), 0})      // index 0
             .CaptureTensor({*mul1.GetProducer(), 0})    // index 1
             .CaptureTensor({*mulGrad.GetProducer(), 0}) // index 2
             .CaptureTensor({*sum.GetProducer(), 0})     // index 3
             .CaptureTensor({*sub.GetProducer(), 0});    // index 4
    
    return pattern;
}

// Create patterns for SoftmaxGradExtV2FusionPass (4 variants)
// variant 0: mul1 = Mul(input1, sub), mulGrad = Mul(mul1, input2)
// variant 1: mul1 = Mul(sub, input1), mulGrad = Mul(mul1, input2)
// variant 2: mul1 = Mul(input1, sub), mulGrad = Mul(input2, mul1)
// variant 3: mul1 = Mul(sub, input1), mulGrad = Mul(input2, mul1)
PatternUniqPtr MakePatternSoftmaxGradExtV2(int variant)
{
    std::string pattern_name = kPassNameV2 + "_" + std::to_string(variant);
    auto graph_builder = es::EsGraphBuilder(pattern_name.c_str());
    auto input0 = graph_builder.CreateInput(0, "input0");
    auto input1 = graph_builder.CreateInput(1, "input1");
    auto input2 = graph_builder.CreateInput(2, "input2");
    
    auto mul = BuildTwoInOneOutNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildReduceSum(graph_builder, mul, -1, true);
    auto sub = BuildTwoInOneOutNode(graph_builder, input0, sum, "Sub");
    
    es::EsTensorHolder mul1;
    es::EsTensorHolder mulGrad;
    
    switch (variant) {
        case 0:
            mul1 = BuildTwoInOneOutNode(graph_builder, input1, sub, "Mul");
            mulGrad = BuildTwoInOneOutNode(graph_builder, mul1, input2, "Mul");
            break;
        case 1:
            mul1 = BuildTwoInOneOutNode(graph_builder, sub, input1, "Mul");
            mulGrad = BuildTwoInOneOutNode(graph_builder, mul1, input2, "Mul");
            break;
        case 2:
            mul1 = BuildTwoInOneOutNode(graph_builder, input1, sub, "Mul");
            mulGrad = BuildTwoInOneOutNode(graph_builder, input2, mul1, "Mul");
            break;
        case 3:
            mul1 = BuildTwoInOneOutNode(graph_builder, sub, input1, "Mul");
            mulGrad = BuildTwoInOneOutNode(graph_builder, input2, mul1, "Mul");
            break;
        default:
            mul1 = BuildTwoInOneOutNode(graph_builder, input1, sub, "Mul");
            mulGrad = BuildTwoInOneOutNode(graph_builder, mul1, input2, "Mul");
            break;
    }
    
    auto graph = graph_builder.BuildAndReset({mulGrad});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    
    pattern->CaptureTensor({*mul.GetProducer(), 0})
             .CaptureTensor({*mul1.GetProducer(), 0})
             .CaptureTensor({*mulGrad.GetProducer(), 0})
             .CaptureTensor({*sum.GetProducer(), 0})
             .CaptureTensor({*sub.GetProducer(), 0});
    
    return pattern;
}

bool CheckInputsShapeValid(const std::unique_ptr<MatchResult>& match_result)
{
    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);
    
    for (const auto& input : subgraph_inputs) {
        auto match_node = input.GetAllInputs().at(0);
        TensorDesc tensor_desc;
        match_node.node.GetInputDesc(match_node.index, tensor_desc);
        Shape shape = tensor_desc.GetShape();
        if (IsUnknownShape(shape.GetDims())) {
            OPS_LOG_D(kPassName.c_str(), "Input has unknown shape, skip fusion.");
            return false;
        }
    }
    return true;
}

bool GetAxisFromReduceSum(const GNode& sum_node, int64_t& axis_value, bool& keep_dims)
{
    // Try to get axis from attribute first
    std::vector<int64_t> axes;
    if (sum_node.GetAttr("axes", axes) == SUCCESS && axes.size() == 1) {
        axis_value = axes[0];
    } else if (sum_node.GetAttr("axis", axis_value) == SUCCESS) {
        // Some versions use "axis" instead of "axes"
    } else {
        OPS_LOG_D(kPassName.c_str(), "Failed to get axis from ReduceSum node.");
        return false;
    }
    
    if (sum_node.GetAttr("keep_dims", keep_dims) != SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get keep_dims from ReduceSum node.");
        return false;
    }
    
    return true;
}

GraphUniqPtr SoftmaxGradExtReplacementCommon(const std::unique_ptr<MatchResult>& match_result, string pass_name)
{
    OPS_LOG_D(pass_name.c_str(), "Enter Replacement for SoftmaxGradExt FusionPass");
    
    // Get subgraph inputs
    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);

    std::vector<Shape> input_shapes;
    std::vector<DataType> input_dtypes;
    std::vector<Format> input_formats;
    GetInputsInfo(subgraph_inputs, input_shapes, input_dtypes, input_formats);
    
    // Get ReduceSum node to extract axis and keep_dims
    NodeIo sum_node_io;
    if (match_result->GetCapturedTensor(kCaptureSumIdx, sum_node_io) != SUCCESS) {
        OPS_LOG_E(pass_name.c_str(), "Failed to get sum node.");
        return nullptr;
    }

    int64_t axis_value = -1;
    bool keep_dims = true;
    if (!GetAxisFromReduceSum(sum_node_io.node, axis_value, keep_dims)) {
        OPS_LOG_E(pass_name.c_str(), "Failed to get axis attributes.");
        return nullptr;
    }
    
    OPS_LOG_D(pass_name.c_str(), "Axis: %ld, keep_dims: %d", axis_value, keep_dims);
    
    // Build replacement graph
    auto replace_graph_builder = es::EsGraphBuilder("replacement");
    auto r_input0 = replace_graph_builder.CreateInput(
        0, "grad", input_dtypes[0], input_formats[0], input_shapes[0].GetDims());
    auto r_input1 = replace_graph_builder.CreateInput(
        1, "x1", input_dtypes[1], input_formats[1], input_shapes[1].GetDims());
    auto r_input2 = replace_graph_builder.CreateInput(
        2, "x2", input_dtypes[2], input_formats[2], input_shapes[2].GetDims());

    auto softmax_grad_ext = BuildSoftmaxGradExt(
        replace_graph_builder, r_input0, r_input1, r_input2, axis_value, keep_dims);

    GraphUniqPtr replace_graph = replace_graph_builder.BuildAndReset({softmax_grad_ext});
    
    // Must call InferShape
    if (InferShape(replace_graph, subgraph_inputs) != SUCCESS) {
        OPS_LOG_E(pass_name.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }
    
    return replace_graph;
}

} // namespace

// ==================== SoftmaxGradExtFusionPass ====================

std::vector<PatternUniqPtr> SoftmaxGradExtFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for SoftmaxGradExtFusionPass");
    std::vector<PatternUniqPtr> patterns;
    patterns.emplace_back(MakePatternSoftmaxGradExt());
    return patterns;
}

static bool IsTargetPlatform()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        false, kPassName.c_str(), "Get platformInfo failed.");

    const std::string soc = platformInfo.str_info.short_soc_version;
    bool isSoc950 = (soc == "Ascend950");
    OPS_LOG_D(kPassName.c_str(), "Platform short soc: %s", soc.c_str());
    if (!isSoc950) {
        OPS_LOG_D(kPassName.c_str(), "Platform is not support, only support Ascend950.");
        return false;
    }
    return true;
}

bool SoftmaxGradExtFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for SoftmaxGradExtFusionPass");

    if (!IsTargetPlatform()) {
        OPS_LOG_D(kPassName.c_str(), "Check platform fail");
        return false;
    }
    
    if (!CheckInputsShapeValid(match_result)) {
        return false;
    }
    
    return true;
}

GraphUniqPtr SoftmaxGradExtFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    return SoftmaxGradExtReplacementCommon(match_result, kPassName);
}

// ==================== SoftmaxGradExtV2FusionPass ====================

std::vector<PatternUniqPtr> SoftmaxGradExtV2FusionPass::Patterns()
{
    OPS_LOG_D(kPassNameV2.c_str(), "Enter Patterns for SoftmaxGradExtV2FusionPass");
    std::vector<PatternUniqPtr> patterns;
    for (int i = 0; i < 4; ++i) {
        patterns.emplace_back(MakePatternSoftmaxGradExtV2(i));
    }
    return patterns;
}

bool SoftmaxGradExtV2FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassNameV2.c_str(), "Enter MeetRequirements for SoftmaxGradExtV2FusionPass");
    if (!IsTargetPlatform()) {
        OPS_LOG_D(kPassName.c_str(), "Check platform fail");
        return false;
    }

    if (!CheckInputsShapeValid(match_result)) {
        return false;
    }
    
    return true;
}

GraphUniqPtr SoftmaxGradExtV2FusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    return SoftmaxGradExtReplacementCommon(match_result, kPassNameV2);
}

REG_FUSION_PASS(SoftmaxGradExtFusionPass).Stage(CustomPassStage::kAfterInferShape);
REG_FUSION_PASS(SoftmaxGradExtV2FusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ops