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
 * \file weight_quant_batch_matmul_v2_transpose_fusion_pass.cpp
 * \brief WeightQuantBatchMatmulV2 Transpose Fusion Pass
 *
 * Fusion pattern: bypass Transpose/Reshape nodes before WeightQuantBatchMatmulV2's inputs
 * by flipping the transpose_x / transpose_weight attributes.
 *
 * parse nodes matched in mapping and call graph DoFusion. fusion rule like this:
 *
 *        x        weight   antiquant_scale  antiquant_offset
 *        |          |          |               |
 *    transpose  transpose  transpose/       transpose/
 *        |          |      reshape          reshape
 *        |          |          |               |
 *        +----------+----------+---------------+
 *                   |
 *          WeightQuantBatchMatmulV2
 *                   |
 *                  out
 *
 *            -------->
 *
 *        x        weight   antiquant_scale  antiquant_offset
 *        |          |          |               |
 *        +----------+----------+---------------+
 *                   |
 *          WeightQuantBatchMatmulV2
 *        (flip transpose_x / transpose_weight)
 *                   |
 *                  out
 *
 */

#include <algorithm>
#include <array>
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"
#include "common/op_graph/fusion_pass/weight_quant_fusion_utils.h"
#include "platform/platform_info.h"
#include "weight_quant_batch_matmul_v2_transpose_fusion_pass.h"

namespace ops {
using namespace weight_quant;

static const char* kFusedOpType = "WeightQuantBatchMatmulV2TransposeFusionPass";
static const char* kFusedOpTypeNZ = "WeightQuantBatchMatmulV2TransposeNZFusionPass";

constexpr int64_t kX = 0;
constexpr int64_t kWeight = 1;
constexpr int64_t kAntiQuantScale = 2;
constexpr int64_t kAntiQuantOffset = 3;
constexpr int64_t kMatMulDimNum = 2;

bool WeightQuantBatchMatmulV2TransposeFusionPass::CheckPlatForm()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        ge::SUCCESS) {
        OP_LOGW(kFusedOpType, "Fail to get platform info");
        return false;
    }
    bool supportL0c2out = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_fix_pipe_l0c2out") !=
                          platformInfo.ai_core_intrinsic_dtype_map.end();
    if (!supportL0c2out) {
        OP_LOGW(kFusedOpType, "The pattern need support l0c2out");
        return false;
    }
    return true;
}

bool WeightQuantBatchMatmulV2TransposeNZFusionPass::CheckPlatForm()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        ge::SUCCESS) {
        OP_LOGW(kFusedOpTypeNZ, "Fail to get platform info");
        return false;
    }
    bool supportL0c2out = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_fix_pipe_l0c2out") !=
                          platformInfo.ai_core_intrinsic_dtype_map.end();
    if (supportL0c2out) {
        OP_LOGW(kFusedOpTypeNZ, "The pattern need not support l0c2out");
        return false;
    }
    return true;
}

static bool UpInputDesc(const ge::GNode& n, int64_t i, ge::TensorDesc& desc)
{
    auto peer = n.GetInDataNodesAndPortIndexs(i);
    if (!peer.first) {
        return false;
    }
    ge::TensorDesc inDesc;
    if (peer.first->GetInputDesc(0, inDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    desc = inDesc;
    return true;
}

static bool IsSimpleReshape(const ge::GNode& reshapeNode, const ge::GNode& qbmmNode, int64_t qbmmPort)
{
    ge::TensorDesc inputDesc;
    if (!UpInputDesc(reshapeNode, 0, inputDesc)) {
        return false;
    }
    auto inputShape = inputDesc.GetShape();
    ge::TensorDesc outputDesc;
    if (qbmmNode.GetInputDesc(qbmmPort, outputDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    auto outputShape = outputDesc.GetShape();
    if (inputShape.GetDimNum() == 1 && outputShape.GetDimNum() == 1) {
        return true;
    }
    if (inputShape.GetDimNum() != kMatMulDimNum && outputShape.GetDimNum() != kMatMulDimNum) {
        return false;
    }
    if (inputShape.GetDimNum() != kMatMulDimNum) {
        return false;
    }
    return inputShape.GetDim(0) == 1 || inputShape.GetDim(1) == 1;
}

static bool CheckDtype(const ge::GNode& n)
{
    ge::TensorDesc xDesc;
    if (n.GetInputDesc(kX, xDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGD(kFusedOpType, "Failed to get x input desc");
        return false;
    }
    ge::DataType xDtype = xDesc.GetDataType();
    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16) {
        OP_LOGD(kFusedOpType, "Input x dtype is %d, must be FLOAT16 or BF16", static_cast<int>(xDtype));
        return false;
    }

    ge::TensorDesc wDesc;
    if (n.GetInputDesc(kWeight, wDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGD(kFusedOpType, "Failed to get weight input desc");
        return false;
    }
    ge::DataType wDtype = wDesc.GetDataType();
    if (wDtype != ge::DT_INT4 && wDtype != ge::DT_INT8 && wDtype != ge::DT_INT32 && wDtype != ge::DT_FLOAT &&
        wDtype != ge::DT_FLOAT4_E2M1 && wDtype != ge::DT_FLOAT8_E4M3FN && wDtype != ge::DT_HIFLOAT8) {
        OP_LOGD(kFusedOpType, "Input weight dtype is %d, must be INT4/INT8/INT32/FP32/FP4/FP8/HIFLOAT8",
                static_cast<int>(wDtype));
        return false;
    }

    ge::TensorDesc outDesc;
    if (n.GetOutputDesc(0, outDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGD(kFusedOpType, "Failed to get output desc");
        return false;
    }
    ge::DataType outDtype = outDesc.GetDataType();
    if (outDtype != ge::DT_FLOAT16 && outDtype != ge::DT_BF16 && outDtype != ge::DT_INT8) {
        OP_LOGD(kFusedOpType, "Output dtype is %d, must be FLOAT16/BF16/INT8", static_cast<int>(outDtype));
        return false;
    }

    return true;
}

static bool CheckShape(const ge::GNode& n)
{
    ge::TensorDesc xDesc;
    if (n.GetInputDesc(kX, xDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGD(kFusedOpType, "Failed to get x input desc for shape check");
        return false;
    }
    auto xShape = xDesc.GetOriginShape();
    if (xShape.GetDimNum() != kMatMulDimNum) {
        OP_LOGD(kFusedOpType, "Input x shape must be 2D, got %zu", xShape.GetDimNum());
        return false;
    }

    ge::TensorDesc wDesc;
    if (n.GetInputDesc(kWeight, wDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGD(kFusedOpType, "Failed to get weight input desc for shape check");
        return false;
    }
    auto wShape = wDesc.GetOriginShape();
    if (wShape.GetDimNum() != kMatMulDimNum) {
        OP_LOGD(kFusedOpType, "Input weight shape must be 2D, got %zu", wShape.GetDimNum());
        return false;
    }

    return true;
}

static bool IsTarget(const ge::GNode& n)
{
    ge::AscendString t;
    if (n.GetType(t) != ge::GRAPH_SUCCESS || t != "WeightQuantBatchMatmulV2") {
        return false;
    }
    if (!CheckDtype(n)) {
        OP_LOGW(kFusedOpType, "CheckDtype failed for WeightQuantBatchMatmulV2 node");
        return false;
    }
    if (!CheckShape(n)) {
        OP_LOGW(kFusedOpType, "CheckShape failed for WeightQuantBatchMatmulV2 node");
        return false;
    }

    auto xNode = GetInputNode(n, kX);
    if (!xNode) {
        OP_LOGW(kFusedOpType, "the input node x of WeightQuantBatchMatmulV2 is null");
        return false;
    }
    auto wNode = GetInputNode(n, kWeight);
    if (!wNode) {
        OP_LOGW(kFusedOpType, "the input node weight of WeightQuantBatchMatmulV2 is null");
        return false;
    }

    ge::AscendString xType;
    bool hasXTranspose = xNode->GetType(xType) == ge::GRAPH_SUCCESS && IsTrans(xType);
    ge::AscendString wType;
    bool hasWeightTranspose = wNode->GetType(wType) == ge::GRAPH_SUCCESS && IsTrans(wType);
    if (!hasXTranspose && !hasWeightTranspose) {
        OP_LOGW(kFusedOpType, "there is not transpose before input x and weight");
        return false;
    }

    return true;
}

static std::vector<ge::GNodePtr> CollectOrphanConstNodes(const ge::GNodePtr& nodeToRemove)
{
    std::vector<ge::GNodePtr> orphanConstNodes;
    size_t inputSize = nodeToRemove->GetInputsSize();
    for (size_t i = 1; i < inputSize; ++i) {
        auto [inNode, inOutputIdx] = nodeToRemove->GetInDataNodesAndPortIndexs(i);
        if (!inNode) {
            continue;
        }
        ge::AscendString inType;
        if (inNode->GetType(inType) == ge::GRAPH_SUCCESS && (inType == "Const" || inType == "Data")) {
            orphanConstNodes.push_back(inNode);
        }
    }
    return orphanConstNodes;
}

static bool DisconnectNodeInputs(const ge::GraphPtr& graph, const ge::GNodePtr& node)
{
    size_t inputSize = node->GetInputsSize();
    for (size_t i = 0; i < inputSize; ++i) {
        auto [inNode, inOutputIdx] = node->GetInDataNodesAndPortIndexs(i);
        if (!inNode) {
            continue;
        }
        if (graph->RemoveEdge(*inNode, inOutputIdx, *node, i) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to remove edge from input node at index %zu", i);
            return false;
        }
    }
    return true;
}

static void CleanupOrphanConstNodes(const ge::GraphPtr& graph, const std::vector<ge::GNodePtr>& orphanConstNodes)
{
    for (const auto& constNode : orphanConstNodes) {
        if (constNode->GetOutDataNodesAndPortIndexs(0).empty()) {
            if (graph->RemoveNode(*constNode) != ge::GRAPH_SUCCESS) {
                OP_LOGW(kFusedOpType, "Failed to remove orphan const node");
            }
        }
    }
}

static ge::Status FuseXTranspose(const ge::GNode& n, std::vector<ge::GNode>& nodesBeforeFuse,
                                 std::vector<ge::GNodePtr>& nodesToRemove)
{
    auto xTrans = GetInputNode(n, kX);
    ge::AscendString xType;
    bool hasXTranspose = xTrans && xTrans->GetType(xType) == ge::GRAPH_SUCCESS && IsTrans(xType);
    if (!hasXTranspose) {
        return ge::SUCCESS;
    }

    AddNodeToRemove(nodesBeforeFuse, nodesToRemove, xTrans);
    OP_LOGD(kFusedOpType, "Detected x transpose node for fusion");
    return ge::SUCCESS;
}

static ge::Status FuseScaleOffsetTranspose(const ge::GNode& n, int64_t idx, std::vector<ge::GNode>& nodesBeforeFuse,
                                           std::vector<ge::GNodePtr>& nodesToRemove)
{
    auto node = GetInputNode(n, idx);
    if (!node) {
        if (idx == kAntiQuantScale) {
            OP_LOGW(kFusedOpType, "anti-quant scale node is null, abort fusion");
            return ge::GRAPH_NOT_CHANGED;
        }
        return ge::SUCCESS;
    }
    ge::AscendString t;
    if (node->GetType(t) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "failed to get type of anti-quant scale/offset node, abort fusion");
        return ge::GRAPH_NOT_CHANGED;
    }
    if (IsTrans(t)) {
        AddNodeToRemove(nodesBeforeFuse, nodesToRemove, node);
        OP_LOGD(kFusedOpType, "Detected anti-quant scale/offset transpose node for fusion");
        return ge::SUCCESS;
    }
    if (t == "Reshape") {
        if (!IsSimpleReshape(*node, n, idx)) {
            OP_LOGW(kFusedOpType, "anti-quant scale/offset reshape is not simple, abort fusion");
            return ge::GRAPH_NOT_CHANGED;
        }
        AddNodeToRemove(nodesBeforeFuse, nodesToRemove, node);
        OP_LOGD(kFusedOpType, "Detected anti-quant scale/offset reshape node for fusion");
        return ge::SUCCESS;
    }
    return ge::SUCCESS;
}

static ge::Status FuseWeightTranspose(const ge::GNode& n, std::vector<ge::GNode>& nodesBeforeFuse,
                                      std::vector<ge::GNodePtr>& nodesToRemove)
{
    auto wTrans = GetInputNode(n, kWeight);
    ge::AscendString wType;
    bool hasWeightTranspose = wTrans && wTrans->GetType(wType) == ge::GRAPH_SUCCESS && IsTrans(wType);
    if (!hasWeightTranspose) {
        return ge::SUCCESS;
    }

    AddNodeToRemove(nodesBeforeFuse, nodesToRemove, wTrans);
    OP_LOGD(kFusedOpType, "Detected weight transpose node for fusion");

    auto status = FuseScaleOffsetTranspose(n, kAntiQuantScale, nodesBeforeFuse, nodesToRemove);
    if (status != ge::SUCCESS) {
        return status;
    }
    status = FuseScaleOffsetTranspose(n, kAntiQuantOffset, nodesBeforeFuse, nodesToRemove);
    if (status != ge::SUCCESS) {
        return status;
    }

    return ge::SUCCESS;
}

struct TransNodeMapping {
    ge::GNodePtr transNode;
    ge::GNodePtr srcNode;
    int64_t srcPort;
    ge::TensorDesc srcDesc;
    std::vector<int64_t> targetPorts;
};

static bool IsSameNode(const ge::GNodePtr& a, const ge::GNodePtr& b)
{
    if (!a || !b) {
        return false;
    }
    if (a.get() == b.get()) {
        return true;
    }
    ge::AscendString nameA, nameB;
    if (a->GetName(nameA) != ge::GRAPH_SUCCESS || b->GetName(nameB) != ge::GRAPH_SUCCESS) {
        return false;
    }
    return nameA == nameB;
}

static ge::Status BuildTransNodeMappings(const ge::GNode& n, const std::vector<ge::GNodePtr>& nodesToRemove,
                                         std::vector<TransNodeMapping>& mappings)
{
    constexpr std::array<int64_t, 4> inputPorts = {kX, kWeight, kAntiQuantScale, kAntiQuantOffset};
    for (const auto& transNode : nodesToRemove) {
        TransNodeMapping m;
        m.transNode = transNode;
        for (int64_t port : inputPorts) {
            if (IsSameNode(GetInputNode(n, port), transNode)) {
                m.targetPorts.push_back(port);
            }
        }
        if (m.targetPorts.empty()) {
            continue;
        }
        auto [srcNode, srcPort] = transNode->GetInDataNodesAndPortIndexs(0);
        if (!srcNode) {
            OP_LOGW(kFusedOpType, "Failed to get source node");
            return ge::GRAPH_NOT_CHANGED;
        }
        m.srcNode = srcNode;
        m.srcPort = srcPort;
        if (transNode->GetInputDesc(0, m.srcDesc) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to get input desc of node to relink");
            return ge::GRAPH_NOT_CHANGED;
        }
        mappings.push_back(std::move(m));
    }
    return ge::SUCCESS;
}

static ge::Status UpdateAndRewireEdges(const ge::GraphPtr& graph, ge::GNode& n,
                                       const std::vector<TransNodeMapping>& mappings)
{
    for (const auto& m : mappings) {
        for (int64_t port : m.targetPorts) {
            if (n.UpdateInputDesc(port, m.srcDesc) != ge::GRAPH_SUCCESS) {
                OP_LOGW(kFusedOpType, "Failed to update input desc for port %ld", port);
                return ge::FAILED;
            }
        }
        for (int64_t port : m.targetPorts) {
            auto status = graph->RemoveEdge(*m.transNode, 0, n, port);
            if (status != ge::GRAPH_SUCCESS && status != ge::GRAPH_NOT_CHANGED) {
                OP_LOGW(kFusedOpType, "Failed to remove edge from transpose to port %ld", port);
                return ge::FAILED;
            }
        }
        for (int64_t port : m.targetPorts) {
            if (graph->AddDataEdge(*m.srcNode, m.srcPort, n, port) != ge::GRAPH_SUCCESS) {
                OP_LOGW(kFusedOpType, "Failed to add edge to port %ld", port);
                return ge::FAILED;
            }
        }
    }
    return ge::SUCCESS;
}

static ge::Status RemoveFusedNodes(const ge::GraphPtr& graph, const std::vector<ge::GNodePtr>& nodesToRemove)
{
    std::vector<ge::GNodePtr> orphanConstNodes;
    for (const auto& node : nodesToRemove) {
        if (HasOtherConsumers(node)) {
            OP_LOGW(kFusedOpType, "Node has other consumers, skip removal");
            continue;
        }
        auto orphans = CollectOrphanConstNodes(node);
        orphanConstNodes.insert(orphanConstNodes.end(), orphans.begin(), orphans.end());
        if (!DisconnectNodeInputs(graph, node)) {
            return ge::FAILED;
        }
        if (graph->RemoveNode(*node) != ge::GRAPH_SUCCESS) {
            OP_LOGW(kFusedOpType, "Failed to remove node");
            return ge::FAILED;
        }
    }
    CleanupOrphanConstNodes(graph, orphanConstNodes);
    return ge::SUCCESS;
}

static ge::Status FlipTransposeAttrsForMappings(ge::GNode& n, const std::vector<TransNodeMapping>& mappings)
{
    for (const auto& m : mappings) {
        for (int64_t port : m.targetPorts) {
            if (port == kX && !FlipTransposeAttr(kFusedOpType, n, "transpose_x")) {
                return ge::GRAPH_NOT_CHANGED;
            }
            if (port == kWeight && !FlipTransposeAttr(kFusedOpType, n, "transpose_weight")) {
                return ge::GRAPH_NOT_CHANGED;
            }
        }
    }
    return ge::SUCCESS;
}

static ge::Status CommitFusion(const ge::GraphPtr& graph, ge::GNode& n, const std::vector<ge::GNode>& nodesBeforeFuse,
                               const std::vector<ge::GNodePtr>& nodesToRemove, ge::CustomPassContext& passContext)
{
    std::vector<TransNodeMapping> mappings;
    auto status = BuildTransNodeMappings(n, nodesToRemove, mappings);
    if (status != ge::SUCCESS) {
        return status;
    }
    status = FlipTransposeAttrsForMappings(n, mappings);
    if (status != ge::SUCCESS) {
        return status;
    }
    status = UpdateAndRewireEdges(graph, n, mappings);
    if (status != ge::SUCCESS) {
        return status;
    }
    ReportTransposeFusion(kFusedOpType, nodesBeforeFuse, n, passContext);
    return RemoveFusedNodes(graph, nodesToRemove);
}

static ge::Status ProcessSingleNodeFusion(const ge::GraphPtr& graph, ge::GNode& n, ge::CustomPassContext& passContext)
{
    std::vector<ge::GNode> nodesBeforeFuse;
    std::vector<ge::GNodePtr> nodesToRemove;
    nodesBeforeFuse.emplace_back(n);

    auto status = FuseXTranspose(n, nodesBeforeFuse, nodesToRemove);
    if (status != ge::SUCCESS) {
        return status;
    }

    status = FuseWeightTranspose(n, nodesBeforeFuse, nodesToRemove);
    if (status != ge::SUCCESS) {
        return status;
    }

    if (nodesToRemove.empty()) {
        return ge::GRAPH_NOT_CHANGED;
    }

    return CommitFusion(graph, n, nodesBeforeFuse, nodesToRemove, passContext);
}

ge::Status WeightQuantBatchMatmulV2TransposeFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    if (!IsGeVersionSupported()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!CheckPlatForm()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(kFusedOpType, "Graph is null or invalid, skip fusion pass.");
        return ge::GRAPH_NOT_CHANGED;
    }
    OP_LOGD(kFusedOpType, "Enter WeightQuantBatchMatmulV2TransposeFusionPass");

    std::vector<ge::GNode> tg;
    for (auto& n : graph->GetDirectNode()) {
        if (IsTarget(n)) {
            tg.emplace_back(n);
        }
    }

    if (tg.empty()) {
        OP_LOGD(kFusedOpType, "No matched WeightQuantBatchMatmulV2 node found, exit fusion pass");
        return ge::GRAPH_NOT_CHANGED;
    }

    OP_LOGD(kFusedOpType, "Found %zu WeightQuantBatchMatmulV2 nodes to fuse", tg.size());
    bool changed = false;
    for (auto& n : tg) {
        OP_LOGD(kFusedOpType, "Processing WeightQuantBatchMatmulV2 node");
        auto status = ProcessSingleNodeFusion(graph, n, passContext);
        if (status == ge::SUCCESS) {
            changed = true;
            continue;
        }
        if (status != ge::GRAPH_NOT_CHANGED) {
            return status;
        }
    }

    OP_LOGD(kFusedOpType, "WeightQuantBatchMatmulV2TransposeFusionPass completed successfully");
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(WeightQuantBatchMatmulV2TransposeFusionPass)
    .Stage(IsGeVersionSupported() ? ge::CustomPassStage::kCompatibleInherited : ge::CustomPassStage::kAfterInferShape);
REG_FUSION_PASS(WeightQuantBatchMatmulV2TransposeNZFusionPass)
    .Stage(IsGeVersionSupported() ? ge::CustomPassStage::kCompatibleInherited : ge::CustomPassStage::kAfterInferShape);
#endif

} // namespace ops
