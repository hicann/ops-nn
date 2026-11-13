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
 * \file fixpipe_pre_quant_pass.cc
 * \brief Cube + AscendDequant + (relu) + AscendQuant fusion pass
 */

#include "fusion_pass/fixpipe_pre_quant_pass.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "graph/operator_factory.h"
#include "version/ge-compiler_version.h"

namespace ops {
namespace {
const char* const kFixpipePreQuantModule = "FixPipePreQuant";
// legacy pass name, kept for the log/report alignment
const char* const kFixpipePreQuantPassName = "FIXPIPEAPREQUANTFUSIONPASS";
// VectorMulScalar host op, keep consistent with the postcube add input strategies
const std::string kVectorMulScalar = "VectorMulScalar";
// VectorMulScalar host op attrs, keep consistent with the legacy pass
const ge::AscendString kAscendSingleOpAsc("_is_single_op");
// the "scale"/"offset" attr of the enhanced dequant, the postcube fusion reads the offset
// attr to decide whether the requant is absorbed (see PrepareFusionAttributes of cube_utils)
const ge::AscendString kAscendQuantScaleAsc("scale");
// attr names of the cloned nodes
const char* const kSqrtModeAttr = "sqrt_mode";
const char* const kReluFlagAttr = "relu_flag";
const char* const kDtypeAttr = "dtype";
// the fixpipe multiple output platform config key
const char* const kFixpipeMultiOutputKey = "Intrinsic_fix_pipe_support_multiple_output";
// relu candidates whose out data nodes are limited during the multi out judge
constexpr int MULTI_OUT_RELU_LIMIT = 2;

// collect all data successors of the node on every output port (equals GetOutDataNodes)
std::vector<ge::GNodePtr> GetAllOutDataNodes(const ge::GNodePtr& node)
{
    std::vector<ge::GNodePtr> outNodes;
    if (node == nullptr) {
        return outNodes;
    }
    for (size_t idx = 0; idx < node->GetOutputsSize(); ++idx) {
        const auto outDataNodes = node->GetOutDataNodesAndPortIndexs(static_cast<int32_t>(idx));
        for (const auto& outNode : outDataNodes) {
            if (outNode.first != nullptr) {
                outNodes.push_back(outNode.first);
            }
        }
    }
    return outNodes;
}

// count the data and control successors of the node (equals GetOutAllNodes().size())
size_t GetOutAllNodesSize(const ge::GNodePtr& node)
{
    if (node == nullptr) {
        return 0UL;
    }
    size_t outNodeSize = 0UL;
    for (size_t idx = 0; idx < node->GetOutputsSize(); ++idx) {
        outNodeSize += node->GetOutDataNodesAndPortIndexs(static_cast<int32_t>(idx)).size();
    }
    outNodeSize += node->GetOutControlNodes().size();
    return outNodeSize;
}

// bfs along the data edges: whether the node `from` can reach the node `to`
// (equals FusionCycleDetector::IsConnected of the legacy framework)
bool IsNodeReachable(const ge::GNodePtr& from, const ge::GNodePtr& to)
{
    if (from == nullptr || to == nullptr || from == to) {
        return false;
    }
    std::set<ge::GNodePtr> visited;
    std::deque<ge::GNodePtr> nodeQueue;
    nodeQueue.push_back(from);
    visited.insert(from);
    while (!nodeQueue.empty()) {
        const ge::GNodePtr curNode = nodeQueue.front();
        nodeQueue.pop_front();
        const auto successors = GetAllOutDataNodes(curNode);
        for (const auto& successor : successors) {
            if (successor == to) {
                return true;
            }
            if (visited.insert(successor).second) {
                nodeQueue.push_back(successor);
            }
        }
    }
    return false;
}

// build the match op type list of the pass (equals kSupportFixpipeCubeTypeVec)
} // namespace

bool FIXPIPEAPREQUANTFUSIONPASS::ReadConfig(const ge::CustomPassContext& context)
{
    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfos, optionalInfos) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_W(kFixpipePreQuantModule, "Fail to get platform info without soc version.");
        return false;
    }
    mFixpipeMap_.clear();
    mUnits_.clear();
    configRelus_.clear();
    std::map<std::string, std::vector<std::string>> dependsUnits;
    PostCubeComm::ReadPlatFormConfig(context, false, mUnits_, dependsUnits, mFixpipeMap_);
    if ((std::find(mUnits_.begin(), mUnits_.end(), kPreConv) == mUnits_.end()) ||
        (std::find(mUnits_.begin(), mUnits_.end(), kPreAct) == mUnits_.end())) {
        return false;
    }
    const auto& actunitNodesInfo = mFixpipeMap_[kPreAct];
    for (const auto& iter : actunitNodesInfo) {
        configRelus_.insert(iter.first);
    }
    return true;
}

const std::map<std::string, std::map<std::string, std::vector<CONFIGDTYPE>>>&
FIXPIPEAPREQUANTFUSIONPASS::GetFixPipeMap() const
{
    return mFixpipeMap_;
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::JudgeQuantReluValue(const ge::GNodePtr& dequantNode,
                                                                const ge::GNodePtr& reluNode,
                                                                const ge::GNodePtr& quantNode) const
{
    if (!PostCubeComm::CheckConstValueData(dequantNode)) {
        return ge::GRAPH_FAILED;
    }
    if (IsDumpableOp(quantNode)) {
        return ge::GRAPH_FAILED;
    }
    float quantScale = 0.0f;
    float offset = 0.0f;
    (void)GNodeGetAttr(quantNode, ATTR_SCALE, quantScale);
    (void)GNodeGetAttr(quantNode, ATTR_OFFSET, offset);
    if (quantScale < 0.0f) {
        return ge::GRAPH_FAILED;
    }
    const std::string reluType = GNodeGetType(reluNode).GetString();
    if (reluType == LEAKY_RELU) {
        float attrNegativeSlopeA = 0.0f;
        (void)GNodeGetAttr(reluNode, ATTR_NEGATIVE_SLOPE, attrNegativeSlopeA);
        if (attrNegativeSlopeA < 0.0f) {
            return ge::GRAPH_FAILED;
        }
    }
    if (reluType == kPRelu) {
        if (!PostCubeComm::CheckConstValueData(reluNode)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (reluType == RELU6) {
        if (quantScale * kRelu6Value + offset < 0.0f) {
            return ge::GRAPH_FAILED;
        }
        if (IsTinyPlatform() && offset < 0.0f) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void FIXPIPEAPREQUANTFUSIONPASS::Pattern0ParseForRelu(const ge::GNodePtr& dequantNode, const ge::GNodePtr& reluNode,
                                                      std::vector<ge::GNodePtr>& reluquants) const
{
    const auto reluOutNodes = GetAllOutDataNodes(reluNode);
    for (const auto& nodeNext : reluOutNodes) {
        if (nodeNext == nullptr) {
            continue;
        }
        OPS_LOG_D(kFixpipePreQuantModule, "relu next node = quant node's name = %s type = %s",
                  GNodeGetName(nodeNext).GetString(), GNodeGetType(nodeNext).GetString());
        if (GNodeGetType(nodeNext).GetString() != ASCEND_QUANT) {
            continue;
        }
        if (JudgeQuantReluValue(dequantNode, reluNode, nodeNext) != ge::GRAPH_SUCCESS) {
            continue;
        }
        reluquants.push_back(nodeNext);
    }
}

bool FIXPIPEAPREQUANTFUSIONPASS::NeedCheckCycle(const ge::GNodePtr& node) const
{
    if (node == nullptr) {
        return false;
    }
    const std::string opType = GNodeGetType(node).GetString();
    if (opType != ELTWISE && opType != kAdd && opType != kSub) {
        return false;
    }
    // only the op type supported by the fixpipe units of the platform needs the cycle check
    // (equals the func_list scan of the legacy needCheckCycle)
    for (const auto& unitIter : mFixpipeMap_) {
        if (unitIter.second.find(opType) != unitIter.second.end()) {
            return true;
        }
    }
    return false;
}

bool FIXPIPEAPREQUANTFUSIONPASS::IsFixpipeSupportMultiOutput() const
{
    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfos, optionalInfos) !=
        ge::GRAPH_SUCCESS) {
        return false;
    }
    std::string multiOutputStr;
    if (!platformInfos.GetPlatformRes("AICoreintrinsicDtypeMap", kFixpipeMultiOutputKey, multiOutputStr) ||
        multiOutputStr.empty()) {
        return false;
    }
    return multiOutputStr == "true";
}

bool FIXPIPEAPREQUANTFUSIONPASS::CheckEltwiseCycleExist(const ge::GNodePtr& node) const
{
    if (node == nullptr) {
        return false;
    }
    const auto outNodes = GetAllOutDataNodes(node);
    if (outNodes.size() <= 1UL) {
        return false;
    }
    // equals !PlatformUtils::Instance().IsFixpipeSupportMultiOutput() of the legacy framework
    if (!IsFixpipeSupportMultiOutput()) {
        return true;
    }
    for (const auto& outNode : outNodes) {
        if (!NeedCheckCycle(outNode)) {
            continue;
        }
        for (const auto& otherNode : outNodes) {
            if (otherNode == outNode) {
                continue;
            }
            if (IsNodeReachable(outNode, otherNode)) {
                return true;
            }
        }
    }
    return false;
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::Pattern0Parse(
    const ge::GNodePtr& dequantNode, std::vector<ge::GNodePtr>& quants,
    std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus) const
{
    if (PostCubeComm::GetOutDataNodesSize(dequantNode) > kPostCubeNodeLimited) {
        return ge::GRAPH_FAILED;
    }
    const auto dequantOutNodes = GetAllOutDataNodes(dequantNode);
    for (const auto& nodePtr : dequantOutNodes) {
        if (nodePtr == nullptr) {
            continue;
        }
        const std::string nextNodeType = GNodeGetType(nodePtr).GetString();
        OPS_LOG_D(kFixpipePreQuantModule, "dequant next node = quant node's name = %s type = %s",
                  GNodeGetName(nodePtr).GetString(), nextNodeType.c_str());
        if ((kCandidateReluTypes.count(nextNodeType) != 0U) && (configRelus_.count(nextNodeType) != 0U)) {
            std::vector<ge::GNodePtr> reluquants;
            OPS_LOG_D(kFixpipePreQuantModule, "next node = quant node's name = %s type = %s",
                      GNodeGetName(nodePtr).GetString(), nextNodeType.c_str());
            if (CheckEltwiseCycleExist(nodePtr)) {
                return ge::GRAPH_FAILED;
            }
            OPS_LOG_D(kFixpipePreQuantModule, "after cycleexit next node = quant node's name = %s type = %s",
                      GNodeGetName(nodePtr).GetString(), nextNodeType.c_str());
            Pattern0ParseForRelu(dequantNode, nodePtr, reluquants);
            relus.emplace(std::make_pair(nodePtr, reluquants));
        } else if (nextNodeType == ASCEND_QUANT) {
            if (IsDumpableOp(nodePtr)) {
                return ge::GRAPH_FAILED;
            }
            quants.push_back(nodePtr);
        } else if (CheckEltwiseCycleExist(dequantNode)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::GNodePtr FIXPIPEAPREQUANTFUSIONPASS::CloneNode(ge::Graph& graph, const ge::GNodePtr& srcNode,
                                                   const std::string& nodeName) const
{
    if (srcNode == nullptr) {
        return nullptr;
    }
    const std::string srcType = GNodeGetType(srcNode).GetString();
    ge::Operator op = ge::OperatorFactory::CreateOperator(nodeName.c_str(), srcType.c_str());
    ge::GNode newNode = graph.AddNodeByOp(op);
    ge::AscendString newType;
    if (newNode.GetType(newType) != ge::GRAPH_SUCCESS || newType.GetString() == nullptr ||
        srcType != std::string(newType.GetString())) {
        OPS_LOG_E(kFixpipePreQuantModule, "Clone node [%s] with type [%s] failed.", nodeName.c_str(), srcType.c_str());
        return nullptr;
    }
    // copy the explicit attrs of the source node (sqrt_mode/relu_flag/negative_slope/dtype ...)
    bool boolValue = false;
    if (srcNode->GetAttr(ge::AscendString(kSqrtModeAttr), boolValue) == ge::GRAPH_SUCCESS) {
        (void)newNode.SetAttr(ge::AscendString(kSqrtModeAttr), boolValue);
    }
    boolValue = false;
    if (srcNode->GetAttr(ge::AscendString(kReluFlagAttr), boolValue) == ge::GRAPH_SUCCESS) {
        (void)newNode.SetAttr(ge::AscendString(kReluFlagAttr), boolValue);
    }
    int64_t dtypeValue = 0;
    if (srcNode->GetAttr(ge::AscendString(kDtypeAttr), dtypeValue) == ge::GRAPH_SUCCESS) {
        (void)newNode.SetAttr(ge::AscendString(kDtypeAttr), dtypeValue);
    }
    float negativeSlope = 0.0f;
    if (srcNode->GetAttr(ge::AscendString(ATTR_NEGATIVE_SLOPE.c_str()), negativeSlope) == ge::GRAPH_SUCCESS) {
        (void)newNode.SetAttr(ge::AscendString(ATTR_NEGATIVE_SLOPE.c_str()), negativeSlope);
    }
    // copy the input/output tensor desc of the source node (equals CloneOpDesc)
    for (size_t idx = 0; idx < srcNode->GetInputsSize(); ++idx) {
        ge::TensorDesc inputDesc;
        if (srcNode->GetInputDesc(static_cast<int32_t>(idx), inputDesc) == ge::GRAPH_SUCCESS) {
            (void)newNode.UpdateInputDesc(static_cast<int32_t>(idx), inputDesc);
        }
    }
    for (size_t idx = 0; idx < srcNode->GetOutputsSize(); ++idx) {
        ge::TensorDesc outputDesc;
        if (srcNode->GetOutputDesc(static_cast<int32_t>(idx), outputDesc) == ge::GRAPH_SUCCESS) {
            (void)newNode.UpdateOutputDesc(static_cast<int32_t>(idx), outputDesc);
        }
    }
    return graph.FindNodeByName(nodeName.c_str());
}

ge::GNodePtr FIXPIPEAPREQUANTFUSIONPASS::DealDequantInput(const ge::GNodePtr& enhancedDequantNode, ge::Graph& graph,
                                                          const ge::GNodePtr& dequantNode,
                                                          const ge::GNodePtr& quantNode,
                                                          std::vector<ge::GNodePtr>& newNodes) const
{
    float scaleTmp = 0.0f;
    (void)GNodeGetAttr(quantNode, ATTR_SCALE, scaleTmp);
    float offsetA = 0.0f;
    (void)GNodeGetAttr(quantNode, ATTR_OFFSET, offsetA);
    if (PostCubeComm::CheckPeerOutNode(dequantNode, 1) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    ge::TensorDesc dequantScaleDesc;
    if (dequantNode->GetInputDesc(1, dequantScaleDesc) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    const auto scaleInput = dequantNode->GetInDataNodesAndPortIndexs(1);
    if (scaleInput.first == nullptr) {
        return nullptr;
    }
    // build the VectorMulScalar host op which pre-multiplies the quant scale on the dequant scale
    const std::string newOpName = "VectorMulScalar" + std::string(GNodeGetName(dequantNode).GetString()) + "_" +
                                  std::to_string(GetAtomicId());
    ge::GNode newOpNode = ge::es::CompliantNodeBuilder(&graph)
                              .OpType(kVectorMulScalar.c_str())
                              .Name(newOpName.c_str())
                              .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                              .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .Build();
    bool isSingleOp = true;
    (void)newOpNode.SetAttr(kAscendSingleOpAsc, isSingleOp);
    ge::TensorDesc curTensorDesc = dequantScaleDesc;
    curTensorDesc.SetDataType(ge::DT_FLOAT);
    (void)newOpNode.UpdateOutputDesc(0, curTensorDesc);
    (void)newOpNode.UpdateInputDesc(0, dequantScaleDesc);
    float scaleAttr = scaleTmp;
    (void)newOpNode.SetAttr(kAscendScaleAsc, scaleAttr);
    ge::GNodePtr newOpNodePtr = graph.FindNodeByName(newOpName.c_str());
    if (newOpNodePtr == nullptr) {
        return nullptr;
    }
    // the cloned dequant reads the float scale and outputs the dtype/shape of the quant output
    ge::TensorDesc enhancedScaleDesc = dequantScaleDesc;
    enhancedScaleDesc.SetDataType(ge::DT_FLOAT);
    (void)enhancedDequantNode->UpdateInputDesc(1, enhancedScaleDesc);
    ge::TensorDesc enhancedOutDesc;
    ge::TensorDesc quantOutDesc;
    if (enhancedDequantNode->GetOutputDesc(0, enhancedOutDesc) != ge::GRAPH_SUCCESS ||
        quantNode->GetOutputDesc(0, quantOutDesc) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    enhancedOutDesc.SetDataType(quantOutDesc.GetDataType());
    enhancedOutDesc.SetShape(quantOutDesc.GetShape());
    (void)enhancedDequantNode->UpdateOutputDesc(0, enhancedOutDesc);
    OPS_LOG_D(kFixpipePreQuantModule, "scale = %f offset = %f", scaleTmp, offsetA);
    float offsetAttr = offsetA;
    (void)enhancedDequantNode->SetAttr(kAscendOffsetAsc, offsetAttr);
    (void)enhancedDequantNode->SetAttr(kAscendQuantScaleAsc, scaleAttr);
    // scale input -> VectorMulScalar -> cloned dequant input(1)
    if (graph.AddDataEdge(*scaleInput.first, scaleInput.second, newOpNode, 0) != ge::GRAPH_SUCCESS ||
        graph.AddDataEdge(newOpNode, 0, *enhancedDequantNode, 1) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    newNodes.push_back(newOpNodePtr);
    return enhancedDequantNode;
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::RelinkQuantConsumers(ge::Graph& graph, const ge::GNodePtr& quantNode,
                                                                 const ge::GNodePtr& newProducer) const
{
    const auto quantConsumers = quantNode->GetOutDataNodesAndPortIndexs(0);
    if (!quantConsumers.empty()) {
        OPS_LOG_I(kFixpipePreQuantModule, "The size of node is [%zu].", quantConsumers.size());
        for (const auto& consumer : quantConsumers) {
            if (consumer.first == nullptr) {
                continue;
            }
            (void)graph.RemoveEdge(*quantNode, 0, *consumer.first, consumer.second);
            if (graph.AddDataEdge(*newProducer, 0, *consumer.first, consumer.second) != ge::GRAPH_SUCCESS) {
                OPS_LOG_E(kFixpipePreQuantModule,
                          "Add edge from fused node:%s's index to fusion node:%s's 1st "
                          "index failed.",
                          GNodeGetName(quantNode).GetString(), GNodeGetName(newProducer).GetString());
                return ge::GRAPH_FAILED;
            }
            OPS_LOG_D(kFixpipePreQuantModule, "Add edge from fused node:%s's 1st index to fusion node:%s's 1st ",
                      GNodeGetName(quantNode).GetString(), GNodeGetName(newProducer).GetString());
        }
    }
    // the control successors are relinked to the new producer, the stale control edges of the
    // quant node are cleaned up together with the node removal
    const auto quantControlConsumers = quantNode->GetOutControlNodes();
    if (!quantControlConsumers.empty()) {
        OPS_LOG_I(kFixpipePreQuantModule, "The size of node is [%zu].", quantControlConsumers.size());
        for (const auto& consumer : quantControlConsumers) {
            if (consumer == nullptr) {
                continue;
            }
            if (graph.AddControlEdge(*newProducer, *consumer) != ge::GRAPH_SUCCESS) {
                OPS_LOG_E(kFixpipePreQuantModule,
                          "Add edge from fused node:%s's index to fusion node:%s's 1st "
                          "index failed.",
                          GNodeGetName(quantNode).GetString(), GNodeGetName(newProducer).GetString());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::RelinkReluWithQuant(ge::Graph& graph, const ge::GNodePtr& headNode,
                                                                const ge::GNodePtr& enhancedDequantNode,
                                                                const ge::GNodePtr& newReluNode,
                                                                const ge::GNodePtr& reluNode,
                                                                const ge::GNodePtr& quantNode) const
{
    if (graph.AddDataEdge(*headNode, 0, *enhancedDequantNode, 0) != ge::GRAPH_SUCCESS ||
        graph.AddDataEdge(*enhancedDequantNode, 0, *newReluNode, 0) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ge::TensorDesc headOutDesc;
    if (headNode->GetOutputDesc(0, headOutDesc) == ge::GRAPH_SUCCESS) {
        (void)enhancedDequantNode->UpdateInputDesc(0, headOutDesc);
    }
    ge::TensorDesc enhancedOutDesc;
    if (enhancedDequantNode->GetOutputDesc(0, enhancedOutDesc) == ge::GRAPH_SUCCESS) {
        (void)newReluNode->UpdateInputDesc(0, enhancedOutDesc);
    }
    if (GNodeGetType(reluNode).GetString() == kPRelu) {
        const auto reluWeightInput = reluNode->GetInDataNodesAndPortIndexs(1);
        if (reluWeightInput.first == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (graph.AddDataEdge(*reluWeightInput.first, reluWeightInput.second, *newReluNode, 1) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        ge::TensorDesc weightDesc;
        if (reluNode->GetInputDesc(1, weightDesc) == ge::GRAPH_SUCCESS) {
            (void)newReluNode->UpdateInputDesc(1, weightDesc);
        }
        if (PostCubeComm::HasControlEdge(reluWeightInput.first, reluNode)) {
            if (graph.AddControlEdge(*reluWeightInput.first, *newReluNode) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    // the stale relu -> quant edges are cleaned up together with the node removal
    (void)graph.RemoveEdge(*reluNode, 0, *quantNode, 0);
    return RelinkQuantConsumers(graph, quantNode, newReluNode);
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::RelinkDequantWithQuant(ge::Graph& graph, const ge::GNodePtr& headNode,
                                                                   const ge::GNodePtr& enhancedDequantNode,
                                                                   const ge::GNodePtr& dequantNode,
                                                                   const ge::GNodePtr& quantNode) const
{
    if (graph.AddDataEdge(*headNode, 0, *enhancedDequantNode, 0) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ge::TensorDesc headOutDesc;
    if (headNode->GetOutputDesc(0, headOutDesc) == ge::GRAPH_SUCCESS) {
        (void)enhancedDequantNode->UpdateInputDesc(0, headOutDesc);
    }
    if (PostCubeComm::HasControlEdge(headNode, dequantNode)) {
        if (graph.AddControlEdge(*headNode, *enhancedDequantNode) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    // the stale dequant -> quant edges are cleaned up together with the node removal
    (void)graph.RemoveEdge(*dequantNode, 0, *quantNode, 0);
    return RelinkQuantConsumers(graph, quantNode, enhancedDequantNode);
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::DeleteNode(ge::Graph& graph, const ge::GNodePtr& toDeleteNode) const
{
    if (toDeleteNode == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (graph.RemoveNode(*toDeleteNode, true) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::DealQuant(ge::Graph& graph, std::vector<ge::GNodePtr>& quantNodes) const
{
    if (quantNodes.empty()) {
        OPS_LOG_D(kFixpipePreQuantModule, "quant_nodes is empty");
        return ge::GRAPH_SUCCESS;
    }
    for (const auto& quantNode : quantNodes) {
        if (DeleteNode(graph, quantNode) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::DealRelus(ge::Graph& graph, const ge::GNodePtr& dequantNode,
                                                      std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus) const
{
    for (auto iter = relus.begin(); iter != relus.end(); ++iter) {
        const ge::GNodePtr& reluNode = iter->first;
        std::vector<ge::GNodePtr>& quantnodes = iter->second;
        if (PostCubeComm::GetOutDataNodesSize(reluNode) == 0U) {
            // the stale dequant -> relu edges are cleaned up together with the node removal
            (void)graph.RemoveEdge(*dequantNode, 0, *reluNode, 0);
        }
        for (const auto& quantNode : quantnodes) {
            if (DeleteNode(graph, quantNode) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
        if (PostCubeComm::GetOutDataNodesSize(reluNode) == 0U) {
            if (DeleteNode(graph, reluNode) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::DeleteDequant(ge::Graph& graph, const ge::GNodePtr& dequantNode) const
{
    if (PostCubeComm::GetOutDataNodesSize(dequantNode) == 0U) {
        if (DeleteNode(graph, dequantNode) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::DealDequant(const ge::GNodePtr& headNode, ge::Graph& graph,
                                                        const ge::GNodePtr& dequantNode,
                                                        std::vector<ge::GNodePtr>& quants,
                                                        std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus,
                                                        std::vector<ge::GNodePtr>& dequants,
                                                        std::vector<ge::GNodePtr>& newNodes)
{
    const std::string dequantName = GNodeGetName(dequantNode).GetString();
    bool needJudge = true;
    uint32_t requantIdx = 0U;
    for (const auto& quantNode : quants) {
        const std::string enhancedName = dequantName + "_requant_" + std::to_string(requantIdx++);
        ge::GNodePtr enhancedDequantNode = CloneNode(graph, dequantNode, enhancedName);
        if (enhancedDequantNode == nullptr) {
            return ge::GRAPH_FAILED;
        }
        (void)enhancedDequantNode->SetAttr(kAscendNeedJudgeDtypeAsc, needJudge);
        ge::GNodePtr enhancedNode = DealDequantInput(enhancedDequantNode, graph, dequantNode, quantNode, newNodes);
        if (enhancedNode == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (RelinkDequantWithQuant(graph, headNode, enhancedDequantNode, dequantNode, quantNode) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        newNodes.push_back(enhancedDequantNode);
        dequants.push_back(enhancedDequantNode);
    }
    for (auto iter = relus.begin(); iter != relus.end(); ++iter) {
        const ge::GNodePtr& reluNode = iter->first;
        std::vector<ge::GNodePtr>& quantnodes = iter->second;
        const std::string reluName = GNodeGetName(reluNode).GetString();
        for (const auto& quantNode : quantnodes) {
            const std::string enhancedName = dequantName + "_requant_" + std::to_string(requantIdx++);
            ge::GNodePtr enhancedDequantNode = CloneNode(graph, dequantNode, enhancedName);
            if (enhancedDequantNode == nullptr) {
                return ge::GRAPH_FAILED;
            }
            (void)enhancedDequantNode->SetAttr(kAscendNeedJudgeDtypeAsc, needJudge);
            const std::string newReluName = reluName + "_requant_" + std::to_string(requantIdx++);
            ge::GNodePtr newReluNode = CloneNode(graph, reluNode, newReluName);
            if (newReluNode == nullptr) {
                return ge::GRAPH_FAILED;
            }
            (void)newReluNode->SetAttr(kAscendNeedJudgeDtypeAsc, needJudge);
            // the cloned relu outputs the dtype/shape of the quant output (equals the legacy desc patch)
            ge::TensorDesc newReluOutDesc;
            ge::TensorDesc quantOutDesc;
            if (newReluNode->GetOutputDesc(0, newReluOutDesc) != ge::GRAPH_SUCCESS ||
                quantNode->GetOutputDesc(0, quantOutDesc) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            newReluOutDesc.SetDataType(quantOutDesc.GetDataType());
            newReluOutDesc.SetShape(quantOutDesc.GetShape());
            (void)newReluNode->UpdateOutputDesc(0, newReluOutDesc);
            ge::GNodePtr enhancedNode = DealDequantInput(enhancedDequantNode, graph, dequantNode, quantNode, newNodes);
            if (enhancedNode == nullptr) {
                return ge::GRAPH_FAILED;
            }
            if (RelinkReluWithQuant(graph, headNode, enhancedDequantNode, newReluNode, reluNode, quantNode) !=
                ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            newNodes.push_back(enhancedDequantNode);
            newNodes.push_back(newReluNode);
            dequants.push_back(enhancedDequantNode);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::ReDequantProcess(const ge::GNodePtr& headNode, ge::Graph& graph,
                                                             const ge::GNodePtr& dequantNode,
                                                             std::vector<ge::GNodePtr>& quants,
                                                             std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus,
                                                             std::vector<ge::GNodePtr>& newNodes)
{
    OPS_LOG_D(kFixpipePreQuantModule, "Start to do general requant fusion pass.");
    std::vector<ge::GNodePtr> dequants;
    if (DealDequant(headNode, graph, dequantNode, quants, relus, dequants, newNodes) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kFixpipePreQuantModule, "[GraphOpt][fixpipeprequant][ReqntPcs] deal dequant nodes failed");
        return ge::GRAPH_FAILED;
    }
    if (DealRelus(graph, dequantNode, relus) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kFixpipePreQuantModule, "[GraphOpt][fixpipeprequant][ReqntPcs] deal relu nodes failed");
        return ge::GRAPH_FAILED;
    }
    (void)DeleteDequant(graph, dequantNode);
    if (DealQuant(graph, quants) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kFixpipePreQuantModule, "[GraphOpt][fixpipeprequant][ReqntPcs] deal quant nodes failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FIXPIPEAPREQUANTFUSIONPASS::CheckQuantsAndrelusSize(const ge::GNodePtr& dequantNode,
                                                         const std::vector<ge::GNodePtr>& quants,
                                                         const std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus)
{
    const auto quantsSize = quants.size();
    const auto relusSize = relus.size();
    if (quantsSize == 0UL && relusSize == 0UL) {
        OPS_LOG_D(kFixpipePreQuantModule, "quants and relus all empty");
        return false;
    }
    if (quantsSize + relusSize > 1UL) {
        OPS_LOG_D(kFixpipePreQuantModule, "quants size = %zu reluquant size = %zu", quantsSize, relusSize);
        return false;
    }
    const auto dequantOutnodeSize = GetOutAllNodesSize(dequantNode);
    if (quantsSize == 1UL) {
        if (dequantOutnodeSize != 1UL) {
            OPS_LOG_D(kFixpipePreQuantModule, "quants size = 1 dequant_node->GetOutAllNodes = %zu", dequantOutnodeSize);
            return false;
        }
        OPS_LOG_D(kFixpipePreQuantModule, "quants size = 1 dequant_node->GetOutAllNodes = 1");
        return true;
    }
    if (relusSize == 1UL) {
        if (dequantOutnodeSize != 1UL) {
            OPS_LOG_D(kFixpipePreQuantModule, "relus size = 1 dequant_node->GetOutAllNodes = %zu", dequantOutnodeSize);
            return false;
        }
        for (const auto& iter : relus) {
            if (GetOutAllNodesSize(iter.first) != 1UL) {
                OPS_LOG_D(kFixpipePreQuantModule, "relus size = 1 relu[0]->GetOutAllNodes = %zu",
                          GetOutAllNodesSize(iter.first));
                return false;
            }
            if (iter.second.size() != 1UL) {
                OPS_LOG_D(kFixpipePreQuantModule, "relus size = 1 relu[0]->quants = %zu", iter.second.size());
                return false;
            }
        }
    }
    return true;
}

bool FIXPIPEAPREQUANTFUSIONPASS::IsCubeVectorCombineFuse() const
{
    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfos, optionalInfos) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_D(kFixpipePreQuantModule, "Can't get platform info when check cube_vector_combine.");
        return false;
    }
    std::string cubeVectorCombineStr;
    platformInfos.GetPlatformRes("SoCInfo", "cube_vector_combine", cubeVectorCombineStr);
    if (cubeVectorCombineStr.empty()) {
        OPS_LOG_D(kFixpipePreQuantModule, "get cube_vector_combine failed");
        return false;
    }
    return cubeVectorCombineStr == "fuse";
}

bool FIXPIPEAPREQUANTFUSIONPASS::IsTinyPlatform() const
{
    // equals UtilPlatform::isTiny of the legacy framework: Intrinsic_set_l1 has f16 but no f32
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optiCompilationInfo) !=
        ge::GRAPH_SUCCESS) {
        return false;
    }
    const auto& instrinsicMap = platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_set_l1"];
    const bool supportFp16Flag = std::find(instrinsicMap.begin(), instrinsicMap.end(), "f16") != instrinsicMap.end();
    const bool supportFp32Flag = std::find(instrinsicMap.begin(), instrinsicMap.end(), "f32") != instrinsicMap.end();
    return supportFp16Flag && !supportFp32Flag;
}

bool FIXPIPEAPREQUANTFUSIONPASS::JudgeMultiOutNode(const ge::GNodePtr& headNode, const ge::GNodePtr& dequantNode,
                                                   const std::vector<ge::GNodePtr>& quants,
                                                   const std::map<ge::GNodePtr, std::vector<ge::GNodePtr>>& relus) const
{
    if (headNode == nullptr) {
        OPS_LOG_D(kFixpipePreQuantModule, "Head node opdesc is nullptr");
        return false;
    }
    for (auto iter = relus.begin(); iter != relus.end(); ++iter) {
        const ge::GNodePtr& reluNode = iter->first;
        const std::vector<ge::GNodePtr>& reluQuantNodes = iter->second;
        if ((PostCubeComm::GetOutDataNodesSize(reluNode) > static_cast<uint32_t>(MULTI_OUT_RELU_LIMIT)) &&
            (reluQuantNodes.size() > 1UL)) {
            OPS_LOG_D(kFixpipePreQuantModule,
                      "Relu[%s:%s] GetOutDataNodesSize[%u], relu_quant_nodes size[%zu], "
                      "can't fusion",
                      GNodeGetName(reluNode).GetString(), GNodeGetType(reluNode).GetString(),
                      PostCubeComm::GetOutDataNodesSize(reluNode), reluQuantNodes.size());
            return false;
        }
    }
    if (PostCubeComm::CheckPostCubeAbilityAttr(headNode, PostCubeAbilityType::SupportMultipleOutput)) {
        // Conv2DTranspose without the fuse soc does not support the multi output fixpipe fusion
        if (GNodeGetType(headNode).GetString() == kConv2DTranspose && !IsCubeVectorCombineFuse()) {
            OPS_LOG_D(kFixpipePreQuantModule,
                      "Conv2DTranspose head [%s] multi-output needs fuse soc, skip requant "
                      "absorb by ability attr.",
                      GNodeGetName(headNode).GetString());
            return CheckQuantsAndrelusSize(dequantNode, quants, relus);
        }
        OPS_LOG_D(kFixpipePreQuantModule, "Conv2d head canne have multioutput");
        return true;
    }
    return CheckQuantsAndrelusSize(dequantNode, quants, relus);
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::HandlePattern0(const ge::GNodePtr& headNode, ge::Graph& graph,
                                                           const ge::GNodePtr& dequantNode,
                                                           std::vector<ge::GNodePtr>& newNodes)
{
    std::vector<ge::GNodePtr> quants;
    std::map<ge::GNodePtr, std::vector<ge::GNodePtr>> relus;
    if (Pattern0Parse(dequantNode, quants, relus) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(kFixpipePreQuantModule, "Do not need do requant pattern0 here, dequant name [%s]",
                  GNodeGetName(dequantNode).GetString());
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!JudgeMultiOutNode(headNode, dequantNode, quants, relus)) {
        OPS_LOG_D(kFixpipePreQuantModule, "Node[%s, %s]: quants size = %zu, reluquant size = %zu",
                  GNodeGetName(headNode).GetString(), GNodeGetType(headNode).GetString(), quants.size(), relus.size());
        return ge::GRAPH_NOT_CHANGED;
    }
    const ge::graphStatus retSwapForbidden = PostCubeComm::GetConv2DReluSwapForbiddenFlag(headNode);
    if (retSwapForbidden == ge::GRAPH_SUCCESS) {
        // PATTERN 1: conv2d+dequant+srelu/lrelu/relu6+quant scene not change
        // PATTERN 2: conv2d+dequant+quant scene add VectorMulScalar and requant node (no relu, bypass)
        for (auto it = relus.begin(); it != relus.end(); ++it) {
            const std::string reluType = GNodeGetType(it->first).GetString();
            if (reluType != RELU && reluType != LEAKY_RELU) {
                OPS_LOG_D(kFixpipePreQuantModule,
                          "Special relu exist in match list, but forbidden to swap position "
                          "with quant, dequant name [%s]",
                          GNodeGetName(dequantNode).GetString());
                return ge::GRAPH_NOT_CHANGED;
            }
        }
    } else if (retSwapForbidden == ge::GRAPH_NOT_CHANGED) { // platforminfo get failed
        return ge::GRAPH_NOT_CHANGED;
    }
    const ge::graphStatus ret = ReDequantProcess(headNode, graph, dequantNode, quants, relus, newNodes);
    if (ret != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kFixpipePreQuantModule,
                  "[GraphOpt][fixpipeprequant][HdlPtn0] requant fusion pass fail, dequant "
                  "node [%s].",
                  GNodeGetName(dequantNode).GetString());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FIXPIPEAPREQUANTFUSIONPASS::IsDumpableOp(const ge::GNodePtr& node)
{
    if (node == nullptr) {
        return false;
    }
    bool isDumpAble = false;
    return node->GetAttr(ge::AscendString(kAttrDumpAble.c_str()), isDumpAble) == ge::GRAPH_SUCCESS && isDumpAble;
}

uint32_t FIXPIPEAPREQUANTFUSIONPASS::GetAtomicId()
{
    static std::atomic<uint32_t> atomicId{0U};
    return atomicId.fetch_add(1U);
}

ge::graphStatus FIXPIPEAPREQUANTFUSIONPASS::FusionImpl(const ge::GNodePtr& headNode, ge::Graph& graph,
                                                       std::vector<ge::GNodePtr>& newNodes)
{
    ge::GNodePtr curHeadNode = headNode;
    const ge::GNodePtr mergeNode = PostCubeComm::GetMergeNodeByCube(curHeadNode);
    if (mergeNode != nullptr) {
        curHeadNode = mergeNode;
        OPS_LOG_D(kFixpipePreQuantModule, "Head node has replaced with merge node[%s, %s].",
                  GNodeGetName(curHeadNode).GetString(), GNodeGetType(curHeadNode).GetString());
    }
    // equals the GetOutAllNodes walk of the legacy pass: both the data and control successors
    // are scanned, the dedup keeps one HandlePattern0 run per dequant node
    std::vector<ge::GNodePtr> headOutNodes = GetAllOutDataNodes(curHeadNode);
    const auto headControlNodes = curHeadNode->GetOutControlNodes();
    for (const auto& controlNode : headControlNodes) {
        if (controlNode != nullptr &&
            std::find(headOutNodes.begin(), headOutNodes.end(), controlNode) == headOutNodes.end()) {
            headOutNodes.push_back(controlNode);
        }
    }
    std::vector<ge::GNodePtr> tmpNodes;
    for (const auto& nodeOutput : headOutNodes) {
        if (nodeOutput == nullptr) {
            continue;
        }
        if (GNodeGetType(nodeOutput).GetString() != ASCEND_DEQUANT) {
            continue;
        }
        if (HandlePattern0(curHeadNode, graph, nodeOutput, tmpNodes) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    if (tmpNodes.empty()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    newNodes.insert(newNodes.end(), tmpNodes.begin(), tmpNodes.end());
    return ge::GRAPH_SUCCESS;
}

bool FIXPIPEAPREQUANTFUSIONPASS::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    (void)matchResult;
    // Run is fully rewritten, the hook never takes effect
    return true;
}

ge::fusion::GraphUniqPtr FIXPIPEAPREQUANTFUSIONPASS::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    (void)matchResult;
    // Run is fully rewritten, the hook never takes effect
    return nullptr;
}

std::vector<ge::fusion::PatternUniqPtr> FIXPIPEAPREQUANTFUSIONPASS::Patterns()
{
    // Run is fully rewritten as a graph surgery flow which walks the graph by itself, no
    // pattern graph is needed and the hook never takes effect
    return {};
}

ge::Status FIXPIPEAPREQUANTFUSIONPASS::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    OPS_LOG_D(kFixpipePreQuantModule, "Begin to do %s.", kFixpipePreQuantPassName);
    if (graph == nullptr) {
        OPS_LOG_E(kFixpipePreQuantModule, "Input graph is nullptr.");
        return ge::GRAPH_FAILED;
    }
    // equals the ReadConfig check of the legacy Fusion(): no pre_conv/pre_act unit, no fusion
    if (!ReadConfig(passContext)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    // collect the matched cube nodes (equals the kPatternConv of the legacy DefinePatterns)
    const std::set<std::string> matchTypes(kSupportPostCubeCubeTypeVec.begin(), kSupportPostCubeCubeTypeVec.end());
    std::vector<ge::GNodePtr> headNodes;
    for (const auto& node : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if (node.GetType(nodeType) != ge::GRAPH_SUCCESS || nodeType.GetString() == nullptr) {
            continue;
        }
        if (matchTypes.count(std::string(nodeType.GetString())) == 0U) {
            continue;
        }
        ge::AscendString nodeName;
        if (node.GetName(nodeName) != ge::GRAPH_SUCCESS || nodeName.GetString() == nullptr) {
            continue;
        }
        ge::GNodePtr nodePtr = graph->FindNodeByName(nodeName);
        if (nodePtr == nullptr) {
            continue;
        }
        // equals the GetFixpipeUtilFuncSupportFlag check of the legacy Fusion()
        if (PostCubeComm::GetPostCubeUtilFuncSupportFlag(nodePtr) != ge::GRAPH_SUCCESS) {
            continue;
        }
        headNodes.push_back(nodePtr);
    }
    if (headNodes.empty()) {
        OPS_LOG_D(kFixpipePreQuantModule, "No matched cube node, exit.");
        return ge::GRAPH_NOT_CHANGED;
    }
    int32_t effectTimes = 0;
    for (const auto& headNode : headNodes) {
        std::vector<ge::GNodePtr> newNodes;
        const ge::graphStatus ret = FusionImpl(headNode, *graph, newNodes);
        if (ret == ge::GRAPH_SUCCESS) {
            effectTimes++;
        } else if (ret == ge::GRAPH_FAILED) {
            // equals the FAILED propagation of the legacy Fusion()
            return ge::GRAPH_FAILED;
        }
    }
    OPS_LOG_D(kFixpipePreQuantModule, "%s completed, effect times = %d.", kFixpipePreQuantPassName, effectTimes);
    return effectTimes != 0 ? ge::GRAPH_SUCCESS : ge::GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(FIXPIPEAPREQUANTFUSIONPASS).Stage(ge::CustomPassStage::kCompatibleInherited);
#endif

} // namespace ops
