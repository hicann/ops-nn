/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv3d_quant_process_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/tensor_desc_origin_dtype_api.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace ge;

namespace {
const std::string FUSION_NAME = "Conv3DQuantProcessFusionPass";
const ge::AscendString PAD = "Pad";

bool IsNodeType(const ge::GNodePtr& node, const ge::AscendString& expectType)
{
    if (node == nullptr) {
        return false;
    }
    ge::AscendString type;
    FUSION_PASS_CHECK_NOLOG(node->GetType(type) != GRAPH_SUCCESS, return false);
    return type == expectType;
}
} // namespace

ge::Status Conv3DQuantProcessFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    if (graph == nullptr || !graph->IsValid()) {
        return GRAPH_NOT_CHANGED;
    }
    passContext.SetPassName(FUSION_NAME.c_str());

    std::vector<GNode> cubeNodes;
    if (!ConvFusionUtilsPass::GetMatchedNodes(graph, cubeNodes, {CONV3D})) {
        return FAILED;
    }
    if (cubeNodes.empty()) {
        return GRAPH_NOT_CHANGED;
    }

    int32_t effectTimes = 0;
    for (auto& cubeNode : cubeNodes) {
        InitMember();
        if (!CheckQuantChain(cubeNode)) {
            continue;
        }
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvDescInfo(cubeNode, convDescInfo), return FAILED);

        std::vector<GNode> nodesBeforeFuse;
        if (quantNode_ != nullptr) {
            nodesBeforeFuse.emplace_back(*quantNode_);
        }
        nodesBeforeFuse.emplace_back(cubeNode);
        if (dequantNode_ != nullptr) {
            nodesBeforeFuse.emplace_back(*dequantNode_);
        }
        if (padNode_ != nullptr) {
            nodesBeforeFuse.emplace_back(*padNode_);
        }
        FUSION_PASS_CHECK_NOLOG(!ReportMatchNodes(nodesBeforeFuse, passContext), return FAILED);
        if (!CanFuseNodes(nodesBeforeFuse)) {
            continue;
        }

        ge::GNodePtr cubeNodePtr = std::make_shared<ge::GNode>(cubeNode);
        std::vector<ge::GNodePtr> fusionNodes;
        std::vector<ge::GNodePtr> nodesToRemove;
        auto status = OptimizeMainProc(graph, quantNode_, dequantNode_, cubeNodePtr, fusionNodes, nodesToRemove);
        if (status == FAILED) {
            return FAILED;
        }
        if (fusionNodes.empty()) {
            continue;
        }

        for (auto& nodeToRemove : nodesToRemove) {
            FUSION_PASS_CHECK(graph->RemoveNode(*nodeToRemove) != GRAPH_SUCCESS,
                              OP_LOGE(FUSION_NAME, "Remove node failed."), return FAILED);
        }
        std::vector<GNode> nodesForReport;
        nodesForReport.emplace_back(cubeNode);
        if (padNode_ != nullptr) {
            nodesForReport.emplace_back(*padNode_);
        }
        std::vector<GNode> nodesAfterFuse;
        for (auto& fusedNode : fusionNodes) {
            nodesAfterFuse.emplace_back(*fusedNode);
        }
        FUSION_PASS_CHECK_NOLOG(!ReportFuseNodes(nodesForReport, nodesAfterFuse, passContext), return FAILED);
        PrintGraphStructure();
        effectTimes++;
    }
    return effectTimes != 0 ? SUCCESS : GRAPH_NOT_CHANGED;
}

void Conv3DQuantProcessFusionPass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    convDescInfo = ConvDescInfo();
    quantNode_ = nullptr;
    dequantNode_ = nullptr;
    padNode_ = nullptr;
}

bool Conv3DQuantProcessFusionPass::CheckQuantChain(const ge::GNode& cubeNode)
{
    padNode_ = nullptr;
    auto in0Pair = cubeNode.GetInDataNodesAndPortIndexs(INPUT_FMAP_INDEX);
    ge::GNodePtr in0Node = in0Pair.first;
    if (IsNodeType(in0Node, ASCEND_QUANT)) {
        quantNode_ = in0Node;
    } else if (IsNodeType(in0Node, PAD)) {
        auto quantPair = in0Node->GetInDataNodesAndPortIndexs(INPUT_FMAP_INDEX);
        FUSION_PASS_CHECK(!IsNodeType(quantPair.first, ASCEND_QUANT),
                          OP_LOGD(FUSION_NAME, "Pad input producer is not AscendQuant, no fusion."), return false);
        quantNode_ = quantPair.first;
        padNode_ = in0Node;
    } else {
        OP_LOGD(FUSION_NAME, "Conv3D input producer is not AscendQuant or Pad, no fusion.");
        return false;
    }

    dequantNode_ = nullptr;
    for (auto& consumerPair : cubeNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX)) {
        if (consumerPair.first == nullptr || consumerPair.second != INPUT_FMAP_INDEX) {
            continue;
        }
        if (IsNodeType(consumerPair.first, ASCEND_DEQUANT)) {
            dequantNode_ = consumerPair.first;
            break;
        }
    }
    FUSION_PASS_CHECK(dequantNode_ == nullptr,
                      OP_LOGD(FUSION_NAME, "Conv3D output consumer is not AscendDequant, no fusion."), return false);
    return true;
}

ge::Status Conv3DQuantProcessFusionPass::GetCoValue(const ge::GNodePtr& cubeNode, int64_t& co)
{
    ge::TensorDesc filterDesc;
    FUSION_PASS_CHECK(cubeNode->GetInputDesc(static_cast<int32_t>(GetWeightInputIndex()), filterDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get filter desc failed."), return FAILED);

    ge::Format filterFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(static_cast<int32_t>(filterDesc.GetFormat())));
    if (filterFormat == FORMAT_ND) {
        return SUCCESS;
    }
    std::vector<int64_t> filterDims = filterDesc.GetShape().GetDims();
    int32_t indexCo = ops::GetAxisIndexByFormat(filterFormat, "N");
    if (indexCo < 0) {
        OP_LOGE(FUSION_NAME, "Node[%s] index_co is negative, check filter format.", convDescInfo.nodeNameStr.c_str());
        return FAILED;
    }
    if (indexCo >= static_cast<int32_t>(filterDims.size())) {
        OP_LOGE(FUSION_NAME, "Node[%s] index_co is larger than the size of filter dims.",
                convDescInfo.nodeNameStr.c_str());
        return FAILED;
    }
    co = filterDims[indexCo];
    return SUCCESS;
}

ge::Status Conv3DQuantProcessFusionPass::SetDataTypeOfNodes(const ge::GNodePtr& cubeNode)
{
    if (SetCubeNodeDataType(cubeNode, ge::DT_FLOAT, ge::DT_FLOAT16) != SUCCESS) {
        OP_LOGE(FUSION_NAME, "Set cube node[%s] data type failed.", convDescInfo.nodeNameStr.c_str());
        return FAILED;
    }

    if (padNode_ != nullptr && SetPadNodeDataType(*padNode_) != SUCCESS) {
        OP_LOGE(FUSION_NAME, "Set pad node data type failed.");
        return FAILED;
    }
    return SUCCESS;
}

ge::GNodePtr Conv3DQuantProcessFusionPass::GetCubeNodeInputNode(const ge::GNodePtr& cubeNode) const
{
    return padNode_ != nullptr ? padNode_ : cubeNode;
}

ge::Status Conv3DQuantProcessFusionPass::SetPadNodeDataType(ge::GNode& padNode) const
{
    ge::TensorDesc padInDesc;
    FUSION_PASS_CHECK(padNode.GetInputDesc(INPUT_FMAP_INDEX, padInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get pad input desc failed."), return FAILED);
    padInDesc.SetDataType(ge::DT_FLOAT16);
    SetOriginDataType(padInDesc, ge::DT_FLOAT16);
    FUSION_PASS_CHECK(padNode.UpdateInputDesc(INPUT_FMAP_INDEX, padInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Fail to update input desc of pad node."), return FAILED);

    ge::TensorDesc padOutDesc;
    FUSION_PASS_CHECK(padNode.GetOutputDesc(OUTPUT_INDEX, padOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get pad output desc failed."), return FAILED);
    padOutDesc.SetDataType(ge::DT_FLOAT16);
    SetOriginDataType(padOutDesc, ge::DT_FLOAT16);
    FUSION_PASS_CHECK(padNode.UpdateOutputDesc(OUTPUT_INDEX, padOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Fail to update output desc of pad node."), return FAILED);

    OP_LOGD(FUSION_NAME, "Data type of Pad has been updated to float16 during quant rollback.");
    return SUCCESS;
}

bool Conv3DQuantProcessFusionPass::MeetRequirements(const ge::GNode& convNode)
{
    (void)convNode;
    return true;
}

std::set<ge::AscendString> Conv3DQuantProcessFusionPass::GetNodeTypes() const { return {CONV3D}; }

void Conv3DQuantProcessFusionPass::PrintGraphStructure() const
{
    OP_LOGI(FUSION_NAME, "%s %s done.", convDescInfo.nodeNameStr.c_str(), FUSION_NAME.c_str());
}

#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(Conv3DQuantProcessFusionPass).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace Ops
