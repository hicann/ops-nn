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
 * \file fixpipe_relu_quant_pass.cc
 * \brief Cube + relu + AscendQuant pass
 */

#include "fusion_pass/fixpipe_relu_quant_pass.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "graph/operator_factory.h"
#include "version/ge-compiler_version.h"

namespace ops {
namespace {
const char* const kFixpipeReluQuantModule = "FixPipeReluQuant";
// legacy pass name, kept for the log/report alignment
const char* const kFixpipeReluQuantPassName = "FIXPIPEARELUQUANTFUSIONPASS";
// the relu candidate types of the legacy pattern: {Relu, PRelu, LeakyRelu, Relu6}
// built on first use instead of at static init: the ops::RELU/kPRelu/LEAKY_RELU/RELU6
// globals live in cube_common.cc which is linked after this object, so copying them
// during the dlopen static init (before cube_common.cc is initialized) would read
// zero-initialized std::string objects and segfault
const std::set<std::string>& GetReluCandidateTypes()
{
    static const std::set<std::string> kReluCandidateTypes = {RELU, kPRelu, LEAKY_RELU, RELU6};
    return kReluCandidateTypes;
}
// the fixpipe multiple output platform config key
const char* const kFixpipeMultiOutputKey = "Intrinsic_fix_pipe_support_multiple_output";
// the quant attrs compared by the matched relu reuse
const char* const kSqrtModeAttr = "sqrt_mode";

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

bool FloatEqual(const float left, const float right) { return std::fabs(left - right) < 1e-6F; }
} // namespace

bool FIXPIPEARELUQUANTFUSIONPASS::NeedCheckCycle(const ge::GNodePtr& node) const
{
    if (node == nullptr) {
        return false;
    }
    const std::string opType = GNodeGetType(node).GetString();
    if (opType != ELTWISE && opType != kAdd && opType != kSub) {
        return false;
    }
    // reads the fixpipe dtype map of the platform directly (equals the legacy needCheckCycle):
    // only the op type supported by some unit func_list needs the cycle check
    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfos, optionalInfos) !=
        ge::GRAPH_SUCCESS) {
        return false;
    }
    const auto fixpipeDtypeMap = optionalInfos.GetFixPipeDtypeMap();
    for (const auto& unitIter : fixpipeDtypeMap) {
        const auto& funcIter = unitIter.second;
        for (const auto& funcName : funcIter) {
            const auto item = kPostCubeInstruction2OpTypeMap.find(funcName);
            if (item != kPostCubeInstruction2OpTypeMap.end() && item->second == opType) {
                return true;
            }
        }
    }
    return false;
}

bool FIXPIPEARELUQUANTFUSIONPASS::IsFixpipeSupportMultiOutput() const
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

bool FIXPIPEARELUQUANTFUSIONPASS::CheckEltwiseCycleExist(const ge::GNodePtr& node) const
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

ge::graphStatus FIXPIPEARELUQUANTFUSIONPASS::JudgeQuantReluValue(const ge::GNodePtr& reluNode,
                                                                 const ge::GNodePtr& quantNode) const
{
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
    }
    return ge::GRAPH_SUCCESS;
}

ge::GNodePtr FIXPIPEARELUQUANTFUSIONPASS::FindMatchedReluNode(const ge::GNodePtr& quantNode,
                                                              const std::map<ge::GNodePtr, ge::GNodePtr>& reluMap) const
{
    float oriScale = 1.0f;
    float scale = 1.0f;
    float oriOffset = 0.0f;
    float offset = 0.0f;
    bool oriSqrtMode = false;
    bool sqrtMode = false;
    ge::TensorDesc oriQuantOutDesc;
    if (quantNode->GetOutputDesc(0, oriQuantOutDesc) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    const ge::DataType oriDataType = oriQuantOutDesc.GetDataType();
    (void)GNodeGetAttr(quantNode, ATTR_SCALE, oriScale);
    (void)GNodeGetAttr(quantNode, ATTR_OFFSET, oriOffset);
    (void)GNodeGetAttr(quantNode, kSqrtModeAttr, oriSqrtMode);
    for (const auto& item : reluMap) {
        ge::TensorDesc quantOutDesc;
        if (item.first->GetOutputDesc(0, quantOutDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        (void)GNodeGetAttr(item.first, ATTR_SCALE, scale);
        (void)GNodeGetAttr(item.first, ATTR_OFFSET, offset);
        (void)GNodeGetAttr(item.first, kSqrtModeAttr, sqrtMode);
        if (FloatEqual(oriScale, scale) && FloatEqual(oriOffset, offset) && (oriSqrtMode == sqrtMode) &&
            (oriDataType == quantOutDesc.GetDataType())) {
            return item.second;
        }
    }
    return nullptr;
}

ge::graphStatus FIXPIPEARELUQUANTFUSIONPASS::RelinkEdge(ge::Graph& graph, const ge::GNodePtr& cubeNode,
                                                        const ge::GNodePtr& reluNode, const ge::GNodePtr& quantNode,
                                                        const ge::GNodePtr& newReluNode,
                                                        std::map<ge::GNodePtr, ge::GNodePtr>& reluMap) const
{
    if (graph.RemoveEdge(*reluNode, 0, *quantNode, 0) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kFixpipeReluQuantModule,
                  "[GraphFusion][FixpipeFusion][RelinkEdge] RemoveEdge between [%s] and "
                  "[%s] failed.",
                  GNodeGetName(reluNode).GetString(), GNodeGetName(quantNode).GetString());
        return ge::GRAPH_FAILED;
    }
    if (graph.AddDataEdge(*cubeNode, 0, *quantNode, 0) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kFixpipeReluQuantModule,
                  "[GraphFusion][FixpipeFusion][RelinkEdge] AddEdge between [%s] and [%s] "
                  "failed.",
                  GNodeGetName(cubeNode).GetString(), GNodeGetName(quantNode).GetString());
        return ge::GRAPH_FAILED;
    }
    const auto quantConsumers = quantNode->GetOutDataNodesAndPortIndexs(0);
    for (const auto& consumer : quantConsumers) {
        if (consumer.first == nullptr) {
            continue;
        }
        (void)graph.RemoveEdge(*quantNode, 0, *consumer.first, consumer.second);
        if (graph.AddDataEdge(*newReluNode, 0, *consumer.first, consumer.second) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kFixpipeReluQuantModule,
                      "[GraphFusion][FixpipeFusion][RelinkEdge] AddEdge between [%s] and "
                      "[%s] failed.",
                      GNodeGetName(newReluNode).GetString(), GNodeGetName(consumer.first).GetString());
            return ge::GRAPH_FAILED;
        }
    }
    if (graph.AddDataEdge(*quantNode, 0, *newReluNode, 0) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kFixpipeReluQuantModule,
                  "[GraphFusion][FixpipeFusion][RelinkEdge] AddEdge between [%s] and [%s] "
                  "failed.",
                  GNodeGetName(quantNode).GetString(), GNodeGetName(newReluNode).GetString());
        return ge::GRAPH_FAILED;
    }
    reluMap.insert(std::make_pair(quantNode, newReluNode));
    if (GNodeGetType(reluNode).GetString() == kPRelu) {
        if (reluNode->GetInputsSize() < 2UL || newReluNode->GetInputsSize() < 2UL) {
            OPS_LOG_E(kFixpipeReluQuantModule,
                      "[GraphFusion][FixpipeFusion][RelinkEdge] PRelu node[%s] input size "
                      "is less than 2.",
                      GNodeGetName(reluNode).GetString());
            return ge::GRAPH_FAILED;
        }
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
    }
    return ge::GRAPH_SUCCESS;
}

ge::GNodePtr FIXPIPEARELUQUANTFUSIONPASS::CreateReluNode(ge::Graph& graph, const ge::GNodePtr& reluNode,
                                                         const ge::GNodePtr& quantNode,
                                                         std::vector<ge::GNodePtr>& newNodes) const
{
    bool needJudge = true;
    const std::string reluName = GNodeGetName(reluNode).GetString();
    const std::string reluType = GNodeGetType(reluNode).GetString();
    std::string newReluName = reluName + "_fixpipe_" + std::to_string(GetAtomicId());
    if (newReluName.size() > kMaxOpNmaLen) {
        newReluName = reluType + "_fixpipe_" + std::to_string(GetAtomicId());
    }
    // clone the relu node with the same type/attrs/input-output desc (equals CopyOpDesc)
    ge::Operator op = ge::OperatorFactory::CreateOperator(newReluName.c_str(), reluType.c_str());
    ge::GNode newReluNode = graph.AddNodeByOp(op);
    ge::AscendString newType;
    if (newReluNode.GetType(newType) != ge::GRAPH_SUCCESS || newType.GetString() == nullptr ||
        reluType != std::string(newType.GetString())) {
        OPS_LOG_E(kFixpipeReluQuantModule, "Clone node [%s] with type [%s] failed.", newReluName.c_str(),
                  reluType.c_str());
        return nullptr;
    }
    float negativeSlope = 0.0f;
    if (reluNode->GetAttr(ge::AscendString(ATTR_NEGATIVE_SLOPE.c_str()), negativeSlope) == ge::GRAPH_SUCCESS) {
        (void)newReluNode.SetAttr(ge::AscendString(ATTR_NEGATIVE_SLOPE.c_str()), negativeSlope);
    }
    for (size_t idx = 0; idx < reluNode->GetInputsSize(); ++idx) {
        ge::TensorDesc inputDesc;
        if (reluNode->GetInputDesc(static_cast<int32_t>(idx), inputDesc) == ge::GRAPH_SUCCESS) {
            (void)newReluNode.UpdateInputDesc(static_cast<int32_t>(idx), inputDesc);
        }
    }
    for (size_t idx = 0; idx < reluNode->GetOutputsSize(); ++idx) {
        ge::TensorDesc outputDesc;
        if (reluNode->GetOutputDesc(static_cast<int32_t>(idx), outputDesc) == ge::GRAPH_SUCCESS) {
            (void)newReluNode.UpdateOutputDesc(static_cast<int32_t>(idx), outputDesc);
        }
    }
    (void)newReluNode.SetAttr(kAscendNeedJudgeDtypeAsc, needJudge);
    // the input/output desc takes the quant output desc (equals the legacy desc patch)
    ge::TensorDesc quantOutDesc;
    if (quantNode->GetOutputDesc(0, quantOutDesc) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    ge::TensorDesc newReluInputDesc;
    ge::TensorDesc newReluOutputDesc;
    if (newReluNode.GetInputDesc(0, newReluInputDesc) != ge::GRAPH_SUCCESS ||
        newReluNode.GetOutputDesc(0, newReluOutputDesc) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    newReluInputDesc.SetDataType(quantOutDesc.GetDataType());
    newReluInputDesc.SetShape(quantOutDesc.GetShape());
    newReluInputDesc.SetOriginShape(quantOutDesc.GetOriginShape());
    (void)newReluNode.UpdateInputDesc(0, newReluInputDesc);
    newReluOutputDesc.SetDataType(quantOutDesc.GetDataType());
    newReluOutputDesc.SetShape(quantOutDesc.GetShape());
    newReluOutputDesc.SetOriginShape(quantOutDesc.GetOriginShape());
    (void)newReluNode.UpdateOutputDesc(0, newReluOutputDesc);
    ge::GNodePtr newReluNodePtr = graph.FindNodeByName(newReluName.c_str());
    if (newReluNodePtr == nullptr) {
        return nullptr;
    }
    newNodes.push_back(newReluNodePtr);
    return newReluNodePtr;
}

ge::graphStatus FIXPIPEARELUQUANTFUSIONPASS::DeleteQuantNode(ge::Graph& graph, const ge::GNodePtr& quantNode,
                                                             const ge::GNodePtr& matchedReluNode) const
{
    const auto quantConsumers = quantNode->GetOutDataNodesAndPortIndexs(0);
    for (const auto& consumer : quantConsumers) {
        if (consumer.first == nullptr) {
            continue;
        }
        (void)graph.RemoveEdge(*quantNode, 0, *consumer.first, consumer.second);
        if (graph.AddDataEdge(*matchedReluNode, 0, *consumer.first, consumer.second) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kFixpipeReluQuantModule, "[GraphFusion][FixpipeFusion][RemoveNode] Relink node[%s] failed.",
                      GNodeGetName(quantNode).GetString());
            return ge::GRAPH_FAILED;
        }
    }
    if (graph.RemoveNode(*quantNode, true) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kFixpipeReluQuantModule, "[GraphFusion][FixpipeFusion][RemoveNode] Remove node[%s] failed.",
                  GNodeGetName(quantNode).GetString());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIXPIPEARELUQUANTFUSIONPASS::ReluQuantFusionImpl(ge::Graph& graph, const ge::GNodePtr& cubeNode,
                                                                 const ge::GNodePtr& reluNode,
                                                                 std::vector<ge::GNodePtr>& newNodes) const
{
    std::map<ge::GNodePtr, ge::GNodePtr> reluMap;
    const auto reluOutNodes = GetAllOutDataNodes(reluNode);
    for (const auto& quantNode : reluOutNodes) {
        if (quantNode == nullptr || GNodeGetType(quantNode).GetString() != ASCEND_QUANT ||
            JudgeQuantReluValue(reluNode, quantNode) != ge::GRAPH_SUCCESS) {
            continue;
        }
        const ge::GNodePtr matchedReluNode = FindMatchedReluNode(quantNode, reluMap);
        if (matchedReluNode != nullptr) {
            if (DeleteQuantNode(graph, quantNode, matchedReluNode) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        } else {
            // create the quant op and the relu op, add edge
            const ge::GNodePtr newReluNode = CreateReluNode(graph, reluNode, quantNode, newNodes);
            if (newReluNode == nullptr ||
                RelinkEdge(graph, cubeNode, reluNode, quantNode, newReluNode, reluMap) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (PostCubeComm::GetOutDataNodesSize(reluNode) == 0U) {
                (void)graph.RemoveNode(*reluNode, true);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::Status FIXPIPEARELUQUANTFUSIONPASS::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    OPS_LOG_D(kFixpipeReluQuantModule, "Begin to do %s.", kFixpipeReluQuantPassName);
    if (graph == nullptr) {
        OPS_LOG_E(kFixpipeReluQuantModule, "Input graph is nullptr.");
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
        if (nodePtr != nullptr) {
            headNodes.push_back(nodePtr);
        }
    }
    if (headNodes.empty()) {
        OPS_LOG_D(kFixpipeReluQuantModule, "No matched cube node, exit.");
        return ge::GRAPH_NOT_CHANGED;
    }
    int32_t effectTimes = 0;
    for (const auto& headNode : headNodes) {
        // equals the GetFixpipeUtilFuncSupportFlag check of the legacy Fusion()
        if (PostCubeComm::GetPostCubeUtilFuncSupportFlag(headNode) != ge::GRAPH_SUCCESS) {
            continue; // platforminfo get failed or not in fixpipeUtilFunc op white list
        }
        // equals the kPatternRelu walk of the legacy pattern (cube -> relu -> quant)
        const auto cubeOutNodes = GetAllOutDataNodes(headNode);
        for (const auto& reluNode : cubeOutNodes) {
            if (reluNode == nullptr || GetReluCandidateTypes().count(GNodeGetType(reluNode).GetString()) == 0U) {
                continue;
            }
            // equals the legacy Fusion() checks
            // platform cannot get situation already deal in ReadConfig
            if (PostCubeComm::GetConv2DReluSwapForbiddenFlag(headNode) == ge::GRAPH_SUCCESS) {
                const std::string reluType = GNodeGetType(reluNode).GetString();
                if (reluType != RELU && reluType != LEAKY_RELU) {
                    continue; // Conv2D not support relu quant swap in this condition
                }
            }
            if (PostCubeComm::GetOutDataNodesSize(headNode) > 1U && PostCubeComm::GetOutDataNodesSize(reluNode) > 1U) {
                continue;
            }
            if (configRelus_.count(GNodeGetType(reluNode).GetString()) == 0U) {
                continue;
            }
            int32_t quantCnt = 0;
            const auto reluOutNodes = GetAllOutDataNodes(reluNode);
            for (const auto& quantNode : reluOutNodes) {
                if (quantNode != nullptr && GNodeGetType(quantNode).GetString() == ASCEND_QUANT) {
                    quantCnt++;
                }
            }
            // equals the kPatternQuant of the legacy pattern: the relu without any quant
            // successor is not matched by the pattern at all
            if (quantCnt < 1) {
                continue;
            }
            if (quantCnt > 1) {
                OPS_LOG_D(kFixpipeReluQuantModule, "Node[%s, %s] does not support multiple quant output.",
                          GNodeGetName(headNode).GetString(), GNodeGetType(headNode).GetString());
                continue;
            }
            if ((PostCubeComm::GetOutDataNodesSize(headNode) > 1U ||
                 PostCubeComm::GetOutDataNodesSize(reluNode) > 1U) &&
                !PostCubeComm::CheckPostCubeAbilityAttr(headNode, PostCubeAbilityType::SupportMultipleOutput)) {
                OPS_LOG_D(kFixpipeReluQuantModule, "Node[%s, %s] does not support fixpipe multiple output.",
                          GNodeGetName(headNode).GetString(), GNodeGetType(headNode).GetString());
                continue;
            }
            if (CheckEltwiseCycleExist(headNode) || CheckEltwiseCycleExist(reluNode)) {
                continue;
            }
            std::vector<ge::GNodePtr> newNodes;
            if (ReluQuantFusionImpl(*graph, headNode, reluNode, newNodes) == ge::GRAPH_FAILED) {
                // equals the FAILED propagation of the legacy Fusion()
                return ge::GRAPH_FAILED;
            }
            effectTimes++;
        }
    }
    OPS_LOG_D(kFixpipeReluQuantModule, "%s completed, effect times = %d.", kFixpipeReluQuantPassName, effectTimes);
    return effectTimes != 0 ? ge::GRAPH_SUCCESS : ge::GRAPH_NOT_CHANGED;
}

uint32_t FIXPIPEARELUQUANTFUSIONPASS::GetAtomicId()
{
    static std::atomic<uint32_t> atomicId{0U};
    return atomicId.fetch_add(1U);
}

#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(FIXPIPEARELUQUANTFUSIONPASS).Stage(ge::CustomPassStage::kCompatibleInherited);
#endif

} // namespace ops
