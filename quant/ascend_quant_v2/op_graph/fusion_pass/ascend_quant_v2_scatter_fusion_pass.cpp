/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
 * - Attributes: reduce="update", axis, quant_axis=-1, reciprocal_scale=true, round_mode
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

static const std::string PASS_NAME = "AscendQuantV2ScatterFusionPass";

static const size_t CAPTURE_IDX_QUANT_V2_NODE = 0L;
static const size_t CAPTURE_IDX_SCATTER_NODE = 1L;

static const size_t QUANT_INPUT_COUNT_WITHOUT_OFFSET = 2;
static const size_t QUANT_INPUT_COUNT_WITH_OFFSET = 3;
static const int64_t QUANT_AXIS_NEGATIVE_ONE = -1L;

// AscendQuantV2 node port indices (IrDefInputs: x=0, scale=1, offset=2)
static constexpr int32_t QUANT_PORT_X = 0;
static constexpr int32_t QUANT_PORT_SCALE = 1;
static constexpr int32_t QUANT_PORT_OFFSET = 2;

// Scatter node port indices (IrDefInputs: var=0, indices=1, updates=2)
static constexpr int32_t SCATTER_PORT_VAR = 0;
static constexpr int32_t SCATTER_PORT_INDICES = 1;
static constexpr int32_t SCATTER_PORT_UPDATES = 2;

// QuantUpdateScatter node port indices (IrDefInputs: var=0, indices=1, updates=2, quant_scales=3, quant_zero_points=4)
static constexpr int32_t QUS_PORT_VAR = 0;
static constexpr int32_t QUS_PORT_INDICES = 1;
static constexpr int32_t QUS_PORT_UPDATES = 2;
static constexpr int32_t QUS_PORT_QUANT_SCALES = 3;
static constexpr int32_t QUS_PORT_QUANT_ZERO_POINTS = 4;

struct QuantUpdateScatterAttrs {
    std::string reduce;
    int64_t axis;
    std::string roundMode;
};

// Scatter attribute values
static constexpr int64_t SCATTER_DEFAULT_AXIS = 0L;
static constexpr int64_t SCATTER_AXIS_VALUE_ZERO = 0L;

// Supported quant output data types
static bool IsSupportedQuantOutputDtype(DataType dtype)
{
    static const std::vector<DataType> supportedDtypes = {DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN};
    return std::find(supportedDtypes.begin(), supportedDtypes.end(), dtype) != supportedDtypes.end();
}

// Helper function to get inputs info
static void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs, std::vector<Shape>& inputShapes,
                          std::vector<DataType>& inputDtypes, std::vector<Format>& inputFormats)
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
static Status InferShape(const std::unique_ptr<Graph>& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Begin infershape for replacement.");
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
                                      const es::EsTensorHolder& scale, const es::EsTensorHolder* offset = nullptr)
{
    auto graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();
    // Build AscendQuantV2 node
    auto quantV2Node = es::CompliantNodeBuilder(graphPtr)
                           .OpType("AscendQuantV2")
                           .Name("AscendQuantV2")
                           .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"scale", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"offset", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                           .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .Build();

    SetTensorDescForNode(quantV2Node, DT_INT8);

    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), x.GetProducerOutIndex(), quantV2Node, QUANT_PORT_X);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *scale.GetProducer(), scale.GetProducerOutIndex(), quantV2Node,
                                 QUANT_PORT_SCALE);
    if (offset != nullptr) {
        es::AddEdgeAndUpdatePeerDesc(*graphPtr, *offset->GetProducer(), offset->GetProducerOutIndex(), quantV2Node,
                                     QUANT_PORT_OFFSET);
    }

    return es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(quantV2Node, 0));
}

es::EsTensorHolder BuildScatter(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& var,
                                const es::EsTensorHolder& indices, const es::EsTensorHolder& ascend_quant_v2)
{
    auto graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();
    // Build Scatter node (updates from AscendQuantV2 output)
    auto scatterNode = es::CompliantNodeBuilder(graphPtr)
                           .OpType("Scatter")
                           .Name("Scatter")
                           .IrDefInputs({{"var", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"updates", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"var", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({{"reduce", es::CompliantNodeBuilder::kEsAttrOptional, "String",
                                         es::CreateFrom(ge::AscendString("update"))},
                                        {"axis", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                                         es::CreateFrom(SCATTER_DEFAULT_AXIS)}})
                           .Build();
    SetTensorDescForNode(scatterNode, DT_FLOAT16);

    // Connect inputs to Scatter
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *var.GetProducer(), var.GetProducerOutIndex(), scatterNode,
                                 SCATTER_PORT_VAR);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *indices.GetProducer(), indices.GetProducerOutIndex(), scatterNode,
                                 SCATTER_PORT_INDICES);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *ascend_quant_v2.GetProducer(), ascend_quant_v2.GetProducerOutIndex(),
                                 scatterNode, SCATTER_PORT_UPDATES);

    // Build graph with scatter output
    auto scatterOutput = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(scatterNode, 0));
    return scatterOutput;
}

static void MakePatternForAscendQuantV2Scatter(std::vector<PatternUniqPtr>& patternGraphs, bool hasOffset)
{
    auto graphBuilder = es::EsGraphBuilder(PASS_NAME.c_str());
    auto var = graphBuilder.CreateInput(QUS_PORT_VAR);
    auto indices = graphBuilder.CreateInput(QUS_PORT_INDICES);
    auto x = graphBuilder.CreateInput(QUS_PORT_UPDATES);
    auto scale = graphBuilder.CreateInput(QUS_PORT_QUANT_SCALES);
    SetTensorDescForNode(*var.GetProducer(), DT_FLOAT16);
    SetTensorDescForNode(*indices.GetProducer(), DT_INT32);
    SetTensorDescForNode(*x.GetProducer(), DT_FLOAT16);
    SetTensorDescForNode(*scale.GetProducer(), DT_FLOAT);

    es::EsTensorHolder ascendQuantV2;
    if (hasOffset) {
        auto offset = graphBuilder.CreateInput(QUS_PORT_QUANT_ZERO_POINTS);
        SetTensorDescForNode(*offset.GetProducer(), DT_BF16);
        ascendQuantV2 = BuildAscendQuantV2(graphBuilder, x, scale, &offset);
    } else {
        ascendQuantV2 = BuildAscendQuantV2(graphBuilder, x, scale);
    }

    auto scatter = BuildScatter(graphBuilder, var, indices, ascendQuantV2);
    auto graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{scatter});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*ascendQuantV2.GetProducer(), 0});
    pattern->CaptureTensor({*scatter.GetProducer(), 0});
    pattern->CaptureTensor({*var.GetProducer(), 0});
    pattern->CaptureTensor({*indices.GetProducer(), 0});

    patternGraphs.emplace_back(std::move(pattern));
}

std::vector<PatternUniqPtr> AscendQuantV2ScatterFusionPass::Patterns()
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Patterns for AscendQuantV2ScatterFusionPass");
    std::vector<PatternUniqPtr> patternGraphs;

    MakePatternForAscendQuantV2Scatter(patternGraphs, false);
    MakePatternForAscendQuantV2Scatter(patternGraphs, true);
    return patternGraphs;
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

static bool CheckQuantAttr(NodeIo& quantV2NodeIo)
{
    // 5. Check AscendQuantV2 attributes
    bool sqrtMode = true;
    if (quantV2NodeIo.node.GetAttr("sqrt_mode", sqrtMode) != GRAPH_SUCCESS) {
        OPS_LOG_D(PASS_NAME.c_str(), "Failed to get AscendQuantV2 sqrt_mode attribute.");
        return false;
    }
    if (sqrtMode) {
        OPS_LOG_D(PASS_NAME.c_str(), "AscendQuantV2 sqrt_mode is true, skip.");
        return false;
    }

    ge::AscendString roundModeStr;
    if (quantV2NodeIo.node.GetAttr("round_mode", roundModeStr) != GRAPH_SUCCESS) {
        OPS_LOG_D(PASS_NAME.c_str(), "Failed to get AscendQuantV2 round_mode attribute.");
        return false;
    }
    std::string roundMode = roundModeStr.GetString();
    if (roundMode != "round") {
        OPS_LOG_D(PASS_NAME.c_str(), "AscendQuantV2 round_mode %s not satisfied, need 'round'.", roundMode.c_str());
        return false;
    }

    // 6. Check AscendQuantV2 has only one output
    size_t quantV2OutputCount = quantV2NodeIo.node.GetOutputsSize();
    if (quantV2OutputCount != 1) {
        OPS_LOG_D(PASS_NAME.c_str(), "AscendQuantV2 output count %zu not equal to 1.", quantV2OutputCount);
        return false;
    }

    // 7. Check AscendQuantV2 input count
    auto quantV2InputSize = quantV2NodeIo.node.GetInputsSize();
    if (quantV2InputSize < QUANT_INPUT_COUNT_WITHOUT_OFFSET || quantV2InputSize > QUANT_INPUT_COUNT_WITH_OFFSET) {
        OPS_LOG_D(PASS_NAME.c_str(), "AscendQuantV2 input size %d not in range [2, 3].", quantV2InputSize);
        return false;
    }

    // 8. Check optional offset input
    if (quantV2InputSize == QUANT_INPUT_COUNT_WITH_OFFSET) {
        TensorDesc quantV2Input2Desc;
        quantV2NodeIo.node.GetInputDesc(QUANT_PORT_OFFSET, quantV2Input2Desc);
        DataType input2Dtype = quantV2Input2Desc.GetDataType();
        if (input2Dtype != DT_BF16 && input2Dtype != DT_INT32) {
            OPS_LOG_D(PASS_NAME.c_str(), "AscendQuantV2 input2 dtype %d not satisfied.", input2Dtype);
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
        OPS_LOG_D(PASS_NAME.c_str(), "Node type %s is not AscendQuantV2, skip.", quantV2TypeStr.c_str());
        return false;
    }

    // 2. Check AscendQuantV2 output dtype
    TensorDesc quantV2OutputDesc;
    quantV2NodeIo.node.GetOutputDesc(0, quantV2OutputDesc);
    DataType quantV2OutDtype = quantV2OutputDesc.GetDataType();
    if (!IsSupportedQuantOutputDtype(quantV2OutDtype)) {
        OPS_LOG_D(PASS_NAME.c_str(), "AscendQuantV2 output dtype %d not supported.", quantV2OutDtype);
        return false;
    }

    // 3. Check AscendQuantV2 input dtype
    TensorDesc quantV2Input0Desc;
    quantV2NodeIo.node.GetInputDesc(QUANT_PORT_X, quantV2Input0Desc);
    DataType input0Dtype = quantV2Input0Desc.GetDataType();
    if (input0Dtype != DT_FLOAT16 && input0Dtype != DT_BF16) {
        OPS_LOG_D(PASS_NAME.c_str(), "AscendQuantV2 input0 dtype %d not satisfied.", input0Dtype);
        return false;
    }

    TensorDesc quantV2Input1Desc;
    quantV2NodeIo.node.GetInputDesc(QUANT_PORT_SCALE, quantV2Input1Desc);
    DataType input1Dtype = quantV2Input1Desc.GetDataType();
    if (input1Dtype != DT_BF16 && input1Dtype != DT_FLOAT) {
        OPS_LOG_D(PASS_NAME.c_str(), "AscendQuantV2 input1 dtype %d not satisfied.", input1Dtype);
        return false;
    }

    if (!CheckQuantAttr(quantV2NodeIo)) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to check quant attr.");
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
        OPS_LOG_D(PASS_NAME.c_str(), "Node type %s is not Scatter, skip.", scatterTypeStr.c_str());
        return false;
    }

    // Check Scatter attributes
    ge::AscendString reduceStr;
    if (scatterNodeIo.node.GetAttr("reduce", reduceStr) != GRAPH_SUCCESS) {
        OPS_LOG_D(PASS_NAME.c_str(), "Failed to get Scatter reduce attribute.");
        return false;
    }
    std::string reduce = reduceStr.GetString();
    if (reduce != "update") {
        OPS_LOG_D(PASS_NAME.c_str(), "Scatter reduce %s not satisfied, need 'update'.", reduce.c_str());
        return false;
    }

    int64_t axisVal = 0;
    if (scatterNodeIo.node.GetAttr("axis", axisVal) != GRAPH_SUCCESS) {
        OPS_LOG_D(PASS_NAME.c_str(), "Failed to get Scatter axis attribute.");
        return false;
    }
    if (axisVal == QUANT_AXIS_NEGATIVE_ONE || axisVal == SCATTER_AXIS_VALUE_ZERO) {
        OPS_LOG_D(PASS_NAME.c_str(), "Scatter axis %ld equals -1 or 0, skip.", axisVal);
        return false;
    }
    return true;
}

bool AscendQuantV2ScatterFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter MeetRequirements for AscendQuantV2ScatterFusionPass");

    if (!IsTargetPlatform()) {
        OPS_LOG_D(PASS_NAME.c_str(), "Check platform fail");
        return false;
    }

    // 1. Get matched nodes
    NodeIo quantV2NodeIo;
    NodeIo scatterNodeIo;
    if (match_result->GetCapturedTensor(CAPTURE_IDX_QUANT_V2_NODE, quantV2NodeIo) != SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to get AscendQuantV2 node.");
        return false;
    }
    if (match_result->GetCapturedTensor(CAPTURE_IDX_SCATTER_NODE, scatterNodeIo) != SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to get Scatter node.");
        return false;
    }

    if (!CheckQuant(quantV2NodeIo)) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to check quant node.");
        return false;
    }

    if (!CheckScatter(scatterNodeIo)) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to check scatter node.");
        return false;
    }

    return true;
}

es::EsTensorHolder BuildQuantUpdateScatter(es::EsGraphBuilder& graphBuilder, std::vector<es::EsTensorHolder>& inputs,
                                           bool hasOffset, const QuantUpdateScatterAttrs& attrs)
{
    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("QuantUpdateScatter")
                     .Name("QuantUpdateScatter")
                     .IrDefInputs({{"var", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"updates", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"quant_scales", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"quant_zero_points", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                     .IrDefOutputs({{"var", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs({{"reduce", es::CompliantNodeBuilder::kEsAttrRequired, "String",
                                   es::CreateFrom(ge::AscendString(attrs.reduce.c_str()))},
                                  {"axis", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(attrs.axis)},
                                  {"quant_axis", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                   es::CreateFrom(QUANT_AXIS_NEGATIVE_ONE)},
                                  {"reciprocal_scale", es::CompliantNodeBuilder::kEsAttrRequired, "Bool",
                                   es::CreateFrom(true)},
                                  {"round_mode", es::CompliantNodeBuilder::kEsAttrOptional, "String",
                                   es::CreateFrom(ge::AscendString(attrs.roundMode.c_str()))}})
                     .Build();

    // Connect inputs
    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[QUS_PORT_VAR].GetProducer(),
                                 inputs[QUS_PORT_VAR].GetProducerOutIndex(), node, QUS_PORT_VAR);
    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[QUS_PORT_INDICES].GetProducer(),
                                 inputs[QUS_PORT_INDICES].GetProducerOutIndex(), node, QUS_PORT_INDICES);
    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[QUS_PORT_UPDATES].GetProducer(),
                                 inputs[QUS_PORT_UPDATES].GetProducerOutIndex(), node, QUS_PORT_UPDATES);
    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[QUS_PORT_QUANT_SCALES].GetProducer(),
                                 inputs[QUS_PORT_QUANT_SCALES].GetProducerOutIndex(), node, QUS_PORT_QUANT_SCALES);

    if (hasOffset) {
        // Connect offset input (index 4)
        es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[QUS_PORT_QUANT_ZERO_POINTS].GetProducer(),
                                     inputs[QUS_PORT_QUANT_ZERO_POINTS].GetProducerOutIndex(), node,
                                     QUS_PORT_QUANT_ZERO_POINTS);
    }

    return es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static std::vector<es::EsTensorHolder> CreateInputs(es::EsGraphBuilder& replaceGraphBuilder,
                                                    std::vector<Shape> inputShapes, std::vector<DataType> inputDtypes,
                                                    std::vector<Format> inputFormats, bool hasOffset)
{
    // Create inputs
    std::vector<int64_t> varDims;
    for (size_t i = 0; i < inputShapes[QUS_PORT_VAR].GetDimNum(); i++) {
        varDims.push_back(inputShapes[QUS_PORT_VAR].GetDim(i));
    }
    auto rVar = replaceGraphBuilder.CreateInput(QUS_PORT_VAR, "var", inputDtypes[QUS_PORT_VAR],
                                                inputFormats[QUS_PORT_VAR], varDims);

    std::vector<int64_t> indicesDims;
    for (size_t i = 0; i < inputShapes[QUS_PORT_INDICES].GetDimNum(); i++) {
        indicesDims.push_back(inputShapes[QUS_PORT_INDICES].GetDim(i));
    }
    auto rIndices = replaceGraphBuilder.CreateInput(QUS_PORT_INDICES, "indices", inputDtypes[QUS_PORT_INDICES],
                                                    inputFormats[QUS_PORT_INDICES], indicesDims);

    std::vector<int64_t> xDims;
    for (size_t i = 0; i < inputShapes[QUS_PORT_UPDATES].GetDimNum(); i++) {
        xDims.push_back(inputShapes[QUS_PORT_UPDATES].GetDim(i));
    }
    auto rX = replaceGraphBuilder.CreateInput(QUS_PORT_UPDATES, "x", inputDtypes[QUS_PORT_UPDATES],
                                              inputFormats[QUS_PORT_UPDATES], xDims);

    std::vector<int64_t> scaleDims;
    for (size_t i = 0; i < inputShapes[QUS_PORT_QUANT_SCALES].GetDimNum(); i++) {
        scaleDims.push_back(inputShapes[QUS_PORT_QUANT_SCALES].GetDim(i));
    }
    auto rScale = replaceGraphBuilder.CreateInput(QUS_PORT_QUANT_SCALES, "scale", inputDtypes[QUS_PORT_QUANT_SCALES],
                                                  inputFormats[QUS_PORT_QUANT_SCALES], scaleDims);

    es::EsTensorHolder rOffset;
    if (hasOffset) {
        std::vector<int64_t> offsetDims;
        for (size_t i = 0; i < inputShapes[QUS_PORT_QUANT_ZERO_POINTS].GetDimNum(); i++) {
            offsetDims.push_back(inputShapes[QUS_PORT_QUANT_ZERO_POINTS].GetDim(i));
        }
        rOffset = replaceGraphBuilder.CreateInput(QUS_PORT_QUANT_ZERO_POINTS, "offset",
                                                  inputDtypes[QUS_PORT_QUANT_ZERO_POINTS],
                                                  inputFormats[QUS_PORT_QUANT_ZERO_POINTS], offsetDims);
    }

    std::vector<es::EsTensorHolder> res = {rVar, rIndices, rX, rScale, rOffset};
    return res;
}

std::unique_ptr<Graph> AscendQuantV2ScatterFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Replacement for AscendQuantV2ScatterFusionPass");

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
    if (match_result->GetCapturedTensor(CAPTURE_IDX_QUANT_V2_NODE, quantV2NodeIo) != SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to get AscendQuantV2 node in Replacement.");
        return nullptr;
    }
    if (match_result->GetCapturedTensor(CAPTURE_IDX_SCATTER_NODE, scatterNodeIo) != SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to get Scatter node in Replacement.");
        return nullptr;
    }

    // 3. Get attributes from Scatter
    ge::AscendString reduceAttrStr;
    scatterNodeIo.node.GetAttr("reduce", reduceAttrStr);
    std::string reduce = reduceAttrStr.GetString();
    int64_t axisAttrVal = 0;
    scatterNodeIo.node.GetAttr("axis", axisAttrVal);

    // 4. Check if offset is present
    bool hasOffset = (quantV2NodeIo.node.GetInputsSize() == QUANT_INPUT_COUNT_WITH_OFFSET);

    // 5. Map round_mode based on output dtype: HIFLOAT8 needs "round", others use "rint"
    TensorDesc quantV2OutDesc;
    quantV2NodeIo.node.GetOutputDesc(0, quantV2OutDesc);
    DataType quantV2OutDtype = quantV2OutDesc.GetDataType();
    std::string roundModeForQUS = (quantV2OutDtype == DT_HIFLOAT8) ? "round" : "rint";

    // 6. Build replacement graph
    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto replaceGraphPtr = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();

    auto inputs = CreateInputs(replaceGraphBuilder, inputShapes, inputDtypes, inputFormats, hasOffset);

    auto quant_update_scatter = BuildQuantUpdateScatter(replaceGraphBuilder, inputs, hasOffset,
                                                        QuantUpdateScatterAttrs{reduce, axisAttrVal, roundModeForQUS});
    auto replaceGraph = replaceGraphBuilder.BuildAndReset({quant_update_scatter});
    // Infer shape
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }

    OPS_LOG_I(PASS_NAME.c_str(), "AscendQuantV2ScatterFusionPass fusion success!");
    return replaceGraph;
}

REG_FUSION_PASS(AscendQuantV2ScatterFusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ops