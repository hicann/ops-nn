/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_transpose_fusion_pass.cpp
 * \brief QuantBatchMatmulV4 Transpose Fusion Pass
 *
 * Fusion pattern: bypass Transpose/Reshape nodes before QuantBatchMatmulV4's x2 and x2_scale inputs
 * by flipping the transpose_x2 attribute.
 *
 * parse nodes matched in mapping and call graph DoFusion. fusion rule like this:
 *
 *
 *                     x2          x2_scale
 *                      |             /
 *            x1     transpose    transpose                    x1          x2     x2_scale
 *             \        |           /                             \         |          /
 *              \       |          /                               \        |         /
 *               quantBatchMatmulV4                                 quantBatchMatmulV4
 *                      |                    -------->                      |
 *                      |                                                   |
 *                      |                                                   |
 *                     out                                                 out
 *
 *
 *                      x2          x2_scale
 *                      |             /
 *            x1     transpose      reshape                       x1          x2     x2_scale
 *             \        |           /                             \         |          /
 *              \       |          /                               \        |         /
 *               quantBatchMatmulV4                                 quantBatchMatmulV4
 *                      |                    -------->                      |
 *                      |                                                   |
 *                      |                                                   |
 *                     out                                                 out
 *
 *                                    x2_scale
 *                                    |    \
 *                                    |     \
 *                                    |     shape
 *                                    |       /
 *                                    |    gather
 *                                    |     /
 *                      x2            |   pack
 *                      |             |   /
 *            x1     transpose      reshape                       x1          x2     x2_scale
 *             \        |           /                             \         |          /
 *              \       |          /                               \        |         /
 *               quantBatchMatmulV4                                 quantBatchMatmulV4
 *                      |                    -------->                      |
 *                      |                                                   |
 *                      |                                                   |
 *                     out                                                 out
 *
 */

#include <algorithm>
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"
#include "common/op_graph/fusion_pass/weight_quant_fusion_utils.h"
#include "quant_batch_matmul_v4_transpose_fusion_pass.h"

namespace ops {
using namespace weight_quant;
static const char* kFusedOpType = "QuantBatchMatmulV4TransposeFusionPass";
constexpr int64_t kX1 = 0;
constexpr int64_t kX2 = 1;
constexpr int64_t kX2Scale = 4;
constexpr int64_t kMatMulDimNum = 2;
static const std::vector<ge::DataType> kLegalX1 = {ge::DT_FLOAT8_E4M3FN};
static const std::vector<ge::DataType> kLegalX2 = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT};
static const std::vector<ge::DataType> kLegalOut = {ge::DT_BF16, ge::DT_FLOAT16};

static bool Legal(ge::DataType dtype, const std::vector<ge::DataType>& legalDtypes)
{
    return std::find(legalDtypes.begin(), legalDtypes.end(), dtype) != legalDtypes.end();
}

static bool UpInputDesc(const ge::GNode& n, int64_t i, ge::TensorDesc& desc)
{
    auto peer = n.GetInDataNodesAndPortIndexs(i);
    if (!peer.first) {
        return false;
    }
    ge::TensorDesc inDesc;
    if (peer.first->GetOutputDesc(peer.second, inDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    desc = inDesc;
    return true;
}

static bool IsReshapeValid(const ge::GNode& n)
{
    auto pack = GetInputNode(n, 1);
    ge::AscendString nodeType;
    if (!pack || pack->GetType(nodeType) != ge::GRAPH_SUCCESS || nodeType != "Pack") {
        return false;
    }
    auto gather = GetInputNode(*pack, 0);
    ge::AscendString gatherType;
    if (!gather || gather->GetType(gatherType) != ge::GRAPH_SUCCESS || gatherType != "Gather") {
        return false;
    }
    auto shape = GetInputNode(*gather, 0);
    ge::AscendString shapeType;
    return shape && shape->GetType(shapeType) == ge::GRAPH_SUCCESS && shapeType == "Shape";
}

static bool IsSimpleReshape(const ge::GNode& n)
{
    ge::TensorDesc inputDesc;
    if (!UpInputDesc(n, 0, inputDesc)) {
        return false;
    }
    auto shape = inputDesc.GetShape();
    if (shape.GetDimNum() != kMatMulDimNum) {
        return false;
    }
    return shape.GetDim(0) == 1 || shape.GetDim(1) == 1;
}

static bool X2ScaleOk(const ge::GNode& n)
{
    auto scaleInput = GetInputNode(n, kX2Scale);
    if (!scaleInput) {
        return false;
    }
    ge::AscendString nodeType;
    if (scaleInput->GetType(nodeType) != ge::GRAPH_SUCCESS) {
        return false;
    }
    if (IsTrans(nodeType)) {
        return true;
    }
    if (nodeType == "Reshape") {
        if (!IsSimpleReshape(*scaleInput)) {
            return false;
        }
        auto pack = GetInputNode(*scaleInput, 1);
        if (pack != nullptr) {
            ge::AscendString packType;
            if (pack->GetType(packType) == ge::GRAPH_SUCCESS && packType == "Pack") {
                return IsReshapeValid(*scaleInput);
            }
        }
        return true;
    }
    return true;
}

static bool CheckDtype(const ge::GNode& n)
{
    int64_t dtype = static_cast<int64_t>(ge::DT_BF16);
    if (n.GetAttr("dtype", dtype) != ge::GRAPH_SUCCESS || !Legal(static_cast<ge::DataType>(dtype), kLegalOut)) {
        return false;
    }
    ge::TensorDesc x1Desc;
    if (n.GetInputDesc(kX1, x1Desc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    auto x1DataType = x1Desc.GetDataType();
    if (x1DataType != ge::DT_UNDEFINED && !Legal(x1DataType, kLegalX1)) {
        return false;
    }
    auto x1Shape = x1Desc.GetOriginShape();
    if (x1Shape.GetDimNum() != kMatMulDimNum) {
        return false;
    }
    ge::TensorDesc x2Desc;
    if (n.GetInputDesc(kX2, x2Desc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    auto x2DataType = x2Desc.GetDataType();
    if (x2DataType != ge::DT_UNDEFINED && !Legal(x2DataType, kLegalX2)) {
        return false;
    }
    auto x2Shape = x2Desc.GetOriginShape();
    if (x2Shape.GetDimNum() != kMatMulDimNum) {
        return false;
    }
    return true;
}

static bool IsTarget(const ge::GNode& n)
{
    ge::AscendString nodeType;
    if (n.GetType(nodeType) != ge::GRAPH_SUCCESS || nodeType != "QuantBatchMatmulV4") {
        return false;
    }
    if (!CheckDtype(n)) {
        OP_LOGW(kFusedOpType, "CheckDtype failed for QuantBatchMatmulV4 node");
        return false;
    }
    auto x2Node = GetInputNode(n, kX2);
    ge::AscendString x2NodeType;
    if (!x2Node || x2Node->GetType(x2NodeType) != ge::GRAPH_SUCCESS || !IsTrans(x2NodeType)) {
        OP_LOGD(kFusedOpType, "x2 input is not a Transpose node");
        return false;
    }
    if (!X2ScaleOk(n)) {
        OP_LOGW(kFusedOpType, "X2Scale check failed");
        return false;
    }
    return true;
}

static bool ExtractPackGatherShapeChain(const ge::GNode& reshapeNode, ge::GNodePtr& packNode, ge::GNodePtr& gatherNode,
                                        ge::GNodePtr& shapeNode)
{
    packNode = GetInputNode(reshapeNode, 1);
    if (!packNode) {
        OP_LOGW(kFusedOpType, "Failed to get pack node");
        return false;
    }

    gatherNode = GetInputNode(*packNode, 0);
    if (!gatherNode) {
        OP_LOGW(kFusedOpType, "Failed to get gather node");
        return false;
    }

    shapeNode = GetInputNode(*gatherNode, 0);
    if (!shapeNode) {
        OP_LOGW(kFusedOpType, "Failed to get shape node");
        return false;
    }

    return true;
}

static bool RemovePackGatherShapeChain(const ge::GraphPtr& graph, const ge::GNodePtr& reshapeNode)
{
    if (!reshapeNode) {
        return true;
    }

    ge::AscendString reshapeType;
    if (reshapeNode->GetType(reshapeType) != ge::GRAPH_SUCCESS || reshapeType != "Reshape") {
        return true;
    }

    if (!IsReshapeValid(*reshapeNode)) {
        return true;
    }

    OP_LOGD(kFusedOpType, "Removing Pack-Gather-Shape chain for dynamic reshape");

    ge::GNodePtr packNode, gatherNode, shapeNode;
    if (!ExtractPackGatherShapeChain(*reshapeNode, packNode, gatherNode, shapeNode)) {
        return false;
    }

    if (graph->RemoveEdge(*packNode, 0, *reshapeNode, 1) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove edge between pack and reshape");
        return false;
    }

    if (graph->RemoveEdge(*gatherNode, 0, *packNode, 0) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove edge between gather and pack");
        return false;
    }

    if (graph->RemoveEdge(*shapeNode, 0, *gatherNode, 0) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove edge between shape and gather");
        return false;
    }

    auto [dataNode, dataOutputIdx] = shapeNode->GetInDataNodesAndPortIndexs(0);
    if (dataNode) {
        if (graph->RemoveEdge(*dataNode, dataOutputIdx, *shapeNode, 0) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to remove edge between data and shape");
            return false;
        }
    }

    if (graph->RemoveNode(*packNode) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove pack node");
        return false;
    }

    if (graph->RemoveNode(*gatherNode) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove gather node");
        return false;
    }

    if (graph->RemoveNode(*shapeNode) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove shape node");
        return false;
    }

    OP_LOGD(kFusedOpType, "Successfully removed Pack-Gather-Shape chain");
    return true;
}

static bool RelinkNode(const ge::GraphPtr& graph, const ge::GNodePtr& nodeToRemove, ge::GNode& targetNode,
                       int64_t targetInputIdx)
{
    OP_LOGD(kFusedOpType, "Relinking input %ld of target node", targetInputIdx);
    if (!RemovePackGatherShapeChain(graph, nodeToRemove)) {
        return false;
    }

    ge::TensorDesc dataDesc;
    if (nodeToRemove->GetInputDesc(0, dataDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get input desc of node to relink");
        return false;
    }

    auto [dataNode, dataOutputIdx] = nodeToRemove->GetInDataNodesAndPortIndexs(0);
    if (!dataNode) {
        OP_LOGW(kFusedOpType, "Failed to get input data node of node to relink");
        return false;
    }

    if (graph->RemoveEdge(*nodeToRemove, 0, targetNode, targetInputIdx) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove edge from node to remove to target");
        return false;
    }

    if (graph->AddDataEdge(*dataNode, dataOutputIdx, targetNode, targetInputIdx) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to add data edge from data node to target");
        return false;
    }

    if (targetNode.UpdateInputDesc(targetInputIdx, dataDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to update input desc of target node");
        return false;
    }

    OP_LOGD(kFusedOpType, "Successfully relinked input %ld", targetInputIdx);
    return true;
}

static bool RemoveFusedNode(const ge::GraphPtr& graph, const ge::GNodePtr& nodeToRemove)
{
    if (HasOtherConsumers(nodeToRemove)) {
        OP_LOGW(kFusedOpType, "Node has other consumers, skip removal");
        return true;
    }

    size_t inputSize = nodeToRemove->GetInputsSize();
    for (size_t i = 0; i < inputSize; ++i) {
        auto [inNode, inOutputIdx] = nodeToRemove->GetInDataNodesAndPortIndexs(i);
        if (!inNode) {
            continue;
        }
        if (graph->RemoveEdge(*inNode, inOutputIdx, *nodeToRemove, i) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to remove edge from input node at index %zu", i);
            return false;
        }
    }

    if (graph->RemoveNode(*nodeToRemove) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove node");
        return false;
    }

    OP_LOGD(kFusedOpType, "Successfully removed fused node");
    return true;
}

static void CollectFusionNodes(const ge::GNode& qbmmNode, const ge::GNodePtr& x2Trans, const ge::GNodePtr& x2ScaleNode,
                               bool hasX2Scale, std::vector<ge::GNode>& nodesBeforeFuse,
                               std::vector<ge::GNodePtr>& nodesToRemove)
{
    nodesBeforeFuse = {qbmmNode};
    nodesToRemove.clear();
    AddNodeToRemove(nodesBeforeFuse, nodesToRemove, x2Trans);
    if (hasX2Scale) {
        AddNodeToRemove(nodesBeforeFuse, nodesToRemove, x2ScaleNode);
    }
}

static bool RelinkFusionNodes(const ge::GraphPtr& graph, ge::GNode& qbmmNode, const ge::GNodePtr& x2Trans,
                              const ge::GNodePtr& x2ScaleNode, bool hasX2Scale)
{
    if (!RelinkNode(graph, x2Trans, qbmmNode, kX2)) {
        OP_LOGW(kFusedOpType, "Failed to relink x2 transpose node");
        return false;
    }
    if (hasX2Scale && !RelinkNode(graph, x2ScaleNode, qbmmNode, kX2Scale)) {
        OP_LOGW(kFusedOpType, "Failed to relink x2_scale node");
        return false;
    }
    return true;
}

static bool RemoveFusionNodes(const ge::GraphPtr& graph, const std::vector<ge::GNodePtr>& nodesToRemove)
{
    for (const auto& node : nodesToRemove) {
        if (!RemoveFusedNode(graph, node)) {
            OP_LOGW(kFusedOpType, "Failed to remove fused node");
            return false;
        }
    }
    return true;
}

static ge::Status ProcessSingleNodeFusion(const ge::GraphPtr& graph, ge::GNode& n, ge::CustomPassContext& passContext)
{
    auto x2Trans = GetInputNode(n, kX2);
    if (!x2Trans) {
        OP_LOGW(kFusedOpType, "Failed to get x2 transpose node");
        return ge::FAILED;
    }

    auto x2ScaleNode = GetInputNode(n, kX2Scale);
    ge::AscendString x2ScaleType;
    bool hasX2ScaleTranspose = x2ScaleNode && x2ScaleNode->GetType(x2ScaleType) == ge::GRAPH_SUCCESS &&
                               IsTrans(x2ScaleType);
    bool hasX2ScaleReshape = x2ScaleNode && x2ScaleNode->GetType(x2ScaleType) == ge::GRAPH_SUCCESS &&
                             x2ScaleType == "Reshape";
    bool hasX2Scale = hasX2ScaleTranspose || hasX2ScaleReshape;

    OP_LOGD(kFusedOpType, "hasX2ScaleTranspose: %d, hasX2ScaleReshape: %d", hasX2ScaleTranspose, hasX2ScaleReshape);

    std::vector<ge::GNode> nodesBeforeFuse;
    std::vector<ge::GNodePtr> nodesToRemove;
    CollectFusionNodes(n, x2Trans, x2ScaleNode, hasX2Scale, nodesBeforeFuse, nodesToRemove);

    if (!FlipTransposeAttr(kFusedOpType, n, "transpose_x2")) {
        return ge::FAILED;
    }

    if (!RelinkFusionNodes(graph, n, x2Trans, x2ScaleNode, hasX2Scale)) {
        return ge::FAILED;
    }

    ReportTransposeFusion(kFusedOpType, nodesBeforeFuse, n, passContext);

    if (!RemoveFusionNodes(graph, nodesToRemove)) {
        return ge::FAILED;
    }

    return ge::SUCCESS;
}

ge::Status QuantBatchMatmulV4TransposeFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    if (!IsGeVersionSupported()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(kFusedOpType, "Graph is null or invalid, skip fusion pass.");
        return ge::GRAPH_NOT_CHANGED;
    }
    OP_LOGD(kFusedOpType, "Enter QuantBatchMatmulV4TransposeFusionPass");

    std::vector<ge::GNode> tg;
    for (auto& n : graph->GetDirectNode()) {
        if (IsTarget(n)) {
            tg.emplace_back(n);
        }
    }

    if (tg.empty()) {
        OP_LOGD(kFusedOpType, "No matched QuantBatchMatmulV4 node found, exit fusion pass");
        return ge::GRAPH_NOT_CHANGED;
    }

    OP_LOGD(kFusedOpType, "Found %zu QuantBatchMatmulV4 nodes to fuse", tg.size());
    bool changed = false;
    for (auto& n : tg) {
        OP_LOGD(kFusedOpType, "Processing QuantBatchMatmulV4 node");
        auto status = ProcessSingleNodeFusion(graph, n, passContext);
        if (status == ge::SUCCESS) {
            changed = true;
            continue;
        }
        if (status != ge::GRAPH_NOT_CHANGED) {
            return status;
        }
    }

    OP_LOGD(kFusedOpType, "QuantBatchMatmulV4TransposeFusionPass completed successfully");
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(QuantBatchMatmulV4TransposeFusionPass)
    .Stage(IsGeVersionSupported() ? ge::CustomPassStage::kCompatibleInherited : ge::CustomPassStage::kAfterInferShape);
#endif

} // namespace ops
