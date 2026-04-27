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
 * \file ascend_quant_v2_scatter_fusion_pass.cpp
 * \brief AscendQuantV2 + Scatter fusion pass
 *
 *        x, scale, offset      var, indices, updates, scale, offset
 *               |                          |
 *               |                          |
 *         ascend_quant_v2          quant_update_scatter
 *               |                          |
 *   var indices   |                          |
 *        \     \   |                          |
 *          \    \  |                          |
 *           \   scatter   ======>>>>          var
 *               |
 *               |
 *             var
 *
 * The key transformation:
 * - AscendQuantV2 has 2-3 inputs: x, scale, offset(optional)
 * - Scatter has 3 inputs: var, indices, updates(from AscendQuantV2 output)
 * - QuantUpdateScatter has 4-5 inputs: var, indices, x, scale, offset(optional)
 * - Attributes: reduce="update", axis, quant_axis=-1, reciprocal_scale=true
 */

#include <vector>
#include <string>
#include <algorithm>
#include "common/inc/error_util.h"
#include "ascend_quant_v2_scatter_fusion_pass.h"
#include "es_nn_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "compliant_node_builder.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {

static const std::string kPassName = "AscendQuantV2ScatterFusionPass";
static const std::string kPatternQuantV2 = "AscendQuantV2";
static const std::string kPatternScatter = "Scatter";
static const std::string kFusionType = "QuantUpdateScatter";

static const int64_t kCaptureIdxQuantV2Node = 0L;
static const int64_t kCaptureIdxScatterNode = 1L;
static const int64_t kCaptureIdxVarNode = 2L;
static const int64_t kCaptureIdxIndicesNode = 3L;

static const int32_t kInputNumTwo = 2;
static const int32_t kInputNumThree = 3;
static const int32_t kAttrNeg = -1;

// Supported quant output data types
static bool IsSupportedQuantOutputDtype(DataType dtype) {
    static const std::vector<DataType> supportedDtypes = {
        DT_INT8,
        DT_HIFLOAT8,
        DT_FLOAT8_E5M2,
        DT_FLOAT8_E4M3FN
    };
    return std::find(supportedDtypes.begin(), supportedDtypes.end(), dtype) != supportedDtypes.end();
}

// Helper function to get inputs info
static void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs,
                          std::vector<Shape>& inputShapes,
                          std::vector<DataType>& inputDtypes,
                          std::vector<Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

// Helper function to infer shape
static Status InferShape(const std::unique_ptr<Graph>& replaceGraph,
                         const std::vector<SubgraphInput>& subgraphInputs)
{
    OPS_LOG_D(kPassName.c_str(), "Begin infershape for replacement.");
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

// Helper function to set default tensor desc for pattern nodes
static void SetTensorDescForNode(GNode& node, DataType dtype = DT_FLOAT16, Format format = FORMAT_ND)
{
    TensorDesc outputDesc;
    node.GetOutputDesc(0, outputDesc);
    outputDesc.SetDataType(dtype);
    outputDesc.SetShape(Shape({1}));
    outputDesc.SetFormat(format);
    node.UpdateOutputDesc(0, outputDesc);
}

es::EsTensorHolder BuildAscendQuantV2(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x,
                                      const es::EsTensorHolder& scale)
{
    auto graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();

    // Build AscendQuantV2 node
    auto quantV2Node = es::CompliantNodeBuilder(graphPtr)
        .OpType("AscendQuantV2")
        .Name("AscendQuantV2")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"scale", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"offset", es::CompliantNodeBuilder::kEsIrInputOptional, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .Build();

    SetTensorDescForNode(quantV2Node, DT_INT8);

    // Connect inputs to AscendQuantV2
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), x.GetProducerOutIndex(), quantV2Node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *scale.GetProducer(), scale.GetProducerOutIndex(), quantV2Node, 1);
    
    return es::EsTensorHolder(
        graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(quantV2Node, 0));
}

es::EsTensorHolder BuildScatter(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& var,
                                      const es::EsTensorHolder& indices, const es::EsTensorHolder& ascend_quant_v2)
{
    auto graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();
    // Build Scatter node (updates from AscendQuantV2 output)
    auto scatterNode = es::CompliantNodeBuilder(graphPtr)
        .OpType("Scatter")
        .Name("Scatter")
        .IrDefInputs({
            {"var", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"updates", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({
            {"var", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .IrDefAttrs({
            {"reduce", es::CompliantNodeBuilder::kEsAttrOptional, "String", es::CreateFrom(ge::AscendString("update"))},
            {"axis", es::CompliantNodeBuilder::kEsAttrOptional, "Int", es::CreateFrom(static_cast<int64_t>(0))}
        })
        .Build();
    SetTensorDescForNode(scatterNode, DT_FLOAT16);

    // Connect inputs to Scatter
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *var.GetProducer(), var.GetProducerOutIndex(), scatterNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *indices.GetProducer(), indices.GetProducerOutIndex(), scatterNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *ascend_quant_v2.GetProducer(), ascend_quant_v2.GetProducerOutIndex(), scatterNode, 2);

    // Build graph with scatter output
    auto scatterOutput = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(scatterNode, 0));
    return scatterOutput;
}

std::vector<PatternUniqPtr> AscendQuantV2ScatterFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for AscendQuantV2ScatterFusionPass");
    std::vector<PatternUniqPtr> patternGraphs;

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    // Create input nodes
    auto var = graphBuilder.CreateInput(0);      // Scatter input 0 (var)
    auto indices = graphBuilder.CreateInput(1);   // Scatter input 1 (indices)
    auto x = graphBuilder.CreateInput(2);         // AscendQuantV2 input 0 (x)
    auto scale = graphBuilder.CreateInput(3);     // AscendQuantV2 input 1 (scale)
    // Set default tensor desc for input nodes
    SetTensorDescForNode(*var.GetProducer(), DT_FLOAT16);
    SetTensorDescForNode(*indices.GetProducer(), DT_INT32);
    SetTensorDescForNode(*x.GetProducer(), DT_FLOAT16);
    SetTensorDescForNode(*scale.GetProducer(), DT_FLOAT);

    auto ascend_quant_v2 = BuildAscendQuantV2(graphBuilder, x, scale);
    auto scatter = BuildScatter(graphBuilder, var, indices, ascend_quant_v2);

    auto graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{scatter});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    // Capture nodes for later use
    pattern->CaptureTensor({*ascend_quant_v2.GetProducer(), 0});     // index 0: AscendQuantV2 node
    pattern->CaptureTensor({*scatter.GetProducer(), 0});    // index 1: Scatter node
    pattern->CaptureTensor({*var.GetProducer(), 0});   // index 2: var input
    pattern->CaptureTensor({*indices.GetProducer(), 0}); // index 3: indices input

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

static bool IsTargetPlatform()
{
    PlatformInfo platform_info;
    OptionalInfo optional_info;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
        false, kPassName.c_str(), "Get platform_info failed.");
    const std::string soc = platform_info.str_info.short_soc_version;
    bool is_platform950 = (soc == "Ascend950");
    OPS_LOG_D(kPassName.c_str(), "Platform short soc: %s", soc.c_str());
    if (!is_platform950) {
        OPS_LOG_D(kPassName.c_str(), "Platform is not support, only work on Ascend950.");
        return false;
    }
    return true;
}

static bool CheckQuantAttr(NodeIo& quantV2NodeIo)
{
    // 5. Check AscendQuantV2 attributes
    bool sqrtMode = true;
    if (quantV2NodeIo.node.GetAttr("sqrt_mode", sqrtMode) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get AscendQuantV2 sqrt_mode attribute.");
        return false;
    }
    if (sqrtMode) {
        OPS_LOG_D(kPassName.c_str(), "AscendQuantV2 sqrt_mode is true, skip.");
        return false;
    }

    ge::AscendString roundModeStr;
    if (quantV2NodeIo.node.GetAttr("round_mode", roundModeStr) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get AscendQuantV2 round_mode attribute.");
        return false;
    }
    std::string roundMode = roundModeStr.GetString();
    if (roundMode != "round") {
        OPS_LOG_D(kPassName.c_str(), "AscendQuantV2 round_mode %s not satisfied, need 'round'.", roundMode.c_str());
        return false;
    }

    // 6. Check AscendQuantV2 has only one output
    size_t quantV2OutputCount = quantV2NodeIo.node.GetOutputsSize();
    if (quantV2OutputCount != 1) {
        OPS_LOG_D(kPassName.c_str(), "AscendQuantV2 output count %zu not equal to 1.", quantV2OutputCount);
        return false;
    }

    // 7. Check AscendQuantV2 input count
    auto quantV2InputSize = quantV2NodeIo.node.GetInputsSize();
    if (quantV2InputSize < kInputNumTwo || quantV2InputSize > kInputNumThree) {
        OPS_LOG_D(kPassName.c_str(), "AscendQuantV2 input size %d not in range [2, 3].", quantV2InputSize);
        return false;
    }

    // 8. Check optional offset input
    if (quantV2InputSize == kInputNumThree) {
        TensorDesc quantV2Input2Desc;
        quantV2NodeIo.node.GetInputDesc(2, quantV2Input2Desc);
        DataType input2Dtype = quantV2Input2Desc.GetDataType();
        if (input2Dtype != DT_BF16 && input2Dtype != DT_INT32) {
            OPS_LOG_D(kPassName.c_str(), "AscendQuantV2 input2 dtype %d not satisfied.", input2Dtype);
            return false;
        }
    }
    return true;
}

static bool CheckQuant(NodeIo& quantV2NodeIo)
{
    AscendString quantV2NodeType;
    quantV2NodeIo.node.GetType(quantV2NodeType);
    std::string quantV2TypeStr = quantV2NodeType.GetString();
    if (quantV2TypeStr != "AscendQuantV2") {
        OPS_LOG_D(kPassName.c_str(), "Node type %s is not AscendQuantV2, skip.", quantV2TypeStr.c_str());
        return false;
    }

    // 2. Check AscendQuantV2 output dtype
    TensorDesc quantV2OutputDesc;
    quantV2NodeIo.node.GetOutputDesc(0, quantV2OutputDesc);
    DataType quantV2OutDtype = quantV2OutputDesc.GetDataType();
    if (!IsSupportedQuantOutputDtype(quantV2OutDtype)) {
        OPS_LOG_D(kPassName.c_str(), "AscendQuantV2 output dtype %d not supported.", quantV2OutDtype);
        return false;
    }

    // 3. Check AscendQuantV2 input dtype
    TensorDesc quantV2Input0Desc;
    quantV2NodeIo.node.GetInputDesc(0, quantV2Input0Desc);
    DataType input0Dtype = quantV2Input0Desc.GetDataType();
    if (input0Dtype != DT_FLOAT16 && input0Dtype != DT_BF16) {
        OPS_LOG_D(kPassName.c_str(), "AscendQuantV2 input0 dtype %d not satisfied.", input0Dtype);
        return false;
    }

    TensorDesc quantV2Input1Desc;
    quantV2NodeIo.node.GetInputDesc(1, quantV2Input1Desc);
    DataType input1Dtype = quantV2Input1Desc.GetDataType();
    if (input1Dtype != DT_BF16 && input1Dtype != DT_FLOAT) {
        OPS_LOG_D(kPassName.c_str(), "AscendQuantV2 input1 dtype %d not satisfied.", input1Dtype);
        return false;
    }

    if (!CheckQuantAttr(quantV2NodeIo)) {
        OPS_LOG_E(kPassName.c_str(), "Failed to check quant attr.");
        return false;
    }
    return true;
}

static bool CheckScatter(NodeIo& scatterNodeIo)
{
    AscendString scatterNodeType;
    scatterNodeIo.node.GetType(scatterNodeType);
    std::string scatterTypeStr = scatterNodeType.GetString();
    if (scatterTypeStr != "Scatter") {
        OPS_LOG_D(kPassName.c_str(), "Node type %s is not Scatter, skip.", scatterTypeStr.c_str());
        return false;
    }

    // Check Scatter attributes
    ge::AscendString reduceStr;
    if (scatterNodeIo.node.GetAttr("reduce", reduceStr) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get Scatter reduce attribute.");
        return false;
    }
    std::string reduce = reduceStr.GetString();
    if (reduce != "update") {
        OPS_LOG_D(kPassName.c_str(), "Scatter reduce %s not satisfied, need 'update'.", reduce.c_str());
        return false;
    }

    int64_t axisVal = 0;
    if (scatterNodeIo.node.GetAttr("axis", axisVal) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get Scatter axis attribute.");
        return false;
    }
    int32_t axis = static_cast<int32_t>(axisVal);
    // Note: Original code rejects axis == -1 or axis == 0
    if (axis == -1 || axis == 0) {
        OPS_LOG_D(kPassName.c_str(), "Scatter axis %d equals -1 or 0, skip.", axis);
        return false;
    }
    return true;
}

bool AscendQuantV2ScatterFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for AscendQuantV2ScatterFusionPass");

    if (!IsTargetPlatform()) {
        OPS_LOG_D(kPassName.c_str(), "Check platform fail");
        return false;
    }

    // 1. Get matched nodes
    NodeIo quantV2NodeIo;
    NodeIo scatterNodeIo;
    if (match_result->GetCapturedTensor(kCaptureIdxQuantV2Node, quantV2NodeIo) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to get AscendQuantV2 node.");
        return false;
    }
    if (match_result->GetCapturedTensor(kCaptureIdxScatterNode, scatterNodeIo) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to get Scatter node.");
        return false;
    }

    if (!CheckQuant(quantV2NodeIo)) {
        OPS_LOG_E(kPassName.c_str(), "Failed to check quant node.");
        return false;
    }

    if (!CheckScatter(scatterNodeIo)) {
        OPS_LOG_E(kPassName.c_str(), "Failed to check scatter node.");
        return false;
    }

    return true;
}

es::EsTensorHolder BuildQuantUpdateScatter(
    es::EsGraphBuilder& graphBuilder, std::vector<es::EsTensorHolder>& inputs,
    bool hasOffset, std::string reduce, int32_t axis)
{
    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("QuantUpdateScatter")
        .Name("QuantUpdateScatter")
        .IrDefInputs({
            {"var", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"updates", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"quant_scales", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"quant_zero_points", es::CompliantNodeBuilder::kEsIrInputOptional, ""}
        })
        .IrDefOutputs({
            {"var", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .IrDefAttrs({
            {"reduce", es::CompliantNodeBuilder::kEsAttrRequired, "String", es::CreateFrom(ge::AscendString(reduce.c_str()))},
            {"axis", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(static_cast<int64_t>(axis))},
            {"quant_axis", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(static_cast<int64_t>(kAttrNeg))},
            {"reciprocal_scale", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(true)}
        })
        .Build();

    // Connect inputs
    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[0].GetProducer(), inputs[0].GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[1].GetProducer(), inputs[1].GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[2].GetProducer(), inputs[2].GetProducerOutIndex(), node, 2);
    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[3].GetProducer(), inputs[3].GetProducerOutIndex(), node, 3);

    if (hasOffset) {
        // Connect offset input (index 4)
        es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[4].GetProducer(), inputs[4].GetProducerOutIndex(), node, 4);
    }

    return es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static std::vector<es::EsTensorHolder> CreateInputs(
    es::EsGraphBuilder& replaceGraphBuilder, std::vector<Shape> inputShapes,
    std::vector<DataType> inputDtypes, std::vector<Format> inputFormats, bool hasOffset)
{
    // Create inputs
    std::vector<int64_t> varDims;
    for (size_t i = 0; i < inputShapes[0].GetDimNum(); i++) {
        varDims.push_back(inputShapes[0].GetDim(i));
    }
    auto rVar = replaceGraphBuilder.CreateInput(0, "var", inputDtypes[0], inputFormats[0], varDims);

    std::vector<int64_t> indicesDims;
    for (size_t i = 0; i < inputShapes[1].GetDimNum(); i++) {
        indicesDims.push_back(inputShapes[1].GetDim(i));
    }
    auto rIndices = replaceGraphBuilder.CreateInput(1, "indices", inputDtypes[1], inputFormats[1], indicesDims);

    std::vector<int64_t> xDims;
    for (size_t i = 0; i < inputShapes[2].GetDimNum(); i++) {
        xDims.push_back(inputShapes[2].GetDim(i));
    }
    auto rX = replaceGraphBuilder.CreateInput(2, "x", inputDtypes[2], inputFormats[2], xDims);

    std::vector<int64_t> scaleDims;
    for (size_t i = 0; i < inputShapes[3].GetDimNum(); i++) {
        scaleDims.push_back(inputShapes[3].GetDim(i));
    }
    auto rScale = replaceGraphBuilder.CreateInput(3, "scale", inputDtypes[3], inputFormats[3], scaleDims);

    es::EsTensorHolder rOffset;
    if (hasOffset) {
        std::vector<int64_t> offsetDims;
        for (size_t i = 0; i < inputShapes[4].GetDimNum(); i++) {
            offsetDims.push_back(inputShapes[4].GetDim(i));
        }
        rOffset = replaceGraphBuilder.CreateInput(4, "offset", inputDtypes[4], inputFormats[4], offsetDims);
    }

    std::vector<es::EsTensorHolder> res = {rVar, rIndices, rX, rScale, rOffset};
    return res;
}

std::unique_ptr<Graph> AscendQuantV2ScatterFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for AscendQuantV2ScatterFusionPass");

    // 1. Get inputs info
    std::vector<SubgraphInput> subgraphInputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    // 2. Get matched nodes
    NodeIo quantV2NodeIo;
    NodeIo scatterNodeIo;
    if (match_result->GetCapturedTensor(kCaptureIdxQuantV2Node, quantV2NodeIo) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to get AscendQuantV2 node in Replacement.");
        return nullptr;
    }
    if (match_result->GetCapturedTensor(kCaptureIdxScatterNode, scatterNodeIo) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to get Scatter node in Replacement.");
        return nullptr;
    }

    // 3. Get attributes from Scatter
    ge::AscendString reduceAttrStr;
    scatterNodeIo.node.GetAttr("reduce", reduceAttrStr);
    std::string reduce = reduceAttrStr.GetString();
    int64_t axisAttrVal = 0;
    scatterNodeIo.node.GetAttr("axis", axisAttrVal);
    int32_t axis = static_cast<int32_t>(axisAttrVal);

    // 4. Check if offset is present
    bool hasOffset = (quantV2NodeIo.node.GetInputsSize() == kInputNumThree);

    // 5. Build replacement graph
    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto replaceGraphPtr = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();

    auto inputs = CreateInputs(replaceGraphBuilder, inputShapes, inputDtypes, inputFormats, hasOffset);

    auto quant_update_scatter = BuildQuantUpdateScatter(
        replaceGraphBuilder, inputs, hasOffset, reduce, axis);
    auto replaceGraph = replaceGraphBuilder.BuildAndReset({quant_update_scatter});

    // Infer shape
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }

    OPS_LOG_I(kPassName.c_str(), "AscendQuantV2ScatterFusionPass fusion success!");
    return replaceGraph;
}

REG_FUSION_PASS(AscendQuantV2ScatterFusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ops