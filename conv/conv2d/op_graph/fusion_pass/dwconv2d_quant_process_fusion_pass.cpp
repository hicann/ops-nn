/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dwconv2d_quant_process_fusion_pass.h"

#include <limits>

#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace ge;

namespace {
constexpr size_t LAST_AXIS_INDEX = 3;

const std::string FUSION_NAME = "DWConv2DQuantProcessFusionPass";
const ge::AscendString PAD = "Pad";
const std::string DN2NZ_INTRINSIC = "Intrinsic_data_move_out2l1_dn2nz";

bool IsNodeType(const ge::GNodePtr& node, const ge::AscendString& expectType)
{
    if (node == nullptr) {
        return false;
    }
    ge::AscendString type;
    FUSION_PASS_CHECK_NOLOG(node->GetType(type) != GRAPH_SUCCESS, return false);
    return type == expectType;
}

bool PadShapeTo4Dim(ge::Format filterFormat, const std::vector<int64_t>& filterDims, std::vector<int64_t>& filterDims4D)
{
    size_t sizeOfFilter = filterDims.size();
    for (size_t i = 0; i <= LAST_AXIS_INDEX; ++i) {
        if (i < sizeOfFilter) {
            filterDims4D.emplace_back(filterDims.at(i));
        } else if (filterFormat == ge::FORMAT_NCHW) {
            filterDims4D.emplace_back(1);
        } else if (filterFormat == ge::FORMAT_HWCN) {
            filterDims4D.insert(filterDims4D.cbegin(), 1);
        } else if (filterFormat == ge::FORMAT_NHWC) {
            filterDims4D.insert(filterDims4D.cbegin() + 1, 1);
        } else if (filterFormat == ge::FORMAT_ND) {
            filterDims4D.emplace_back(0);
        } else {
            OP_LOGE(FUSION_NAME, "Format [%s] can not pad shape.", GeFormatToString(filterFormat).c_str());
            return false;
        }
    }
    return true;
}

bool CheckInt64MulOverflow(int64_t left, int64_t right)
{
    if (left == 0 || right == 0) {
        return false;
    }
    if (left > 0) {
        return right > 0 ? left > std::numeric_limits<int64_t>::max() / right :
                           right < std::numeric_limits<int64_t>::min() / left;
    }
    return right > 0 ? left < std::numeric_limits<int64_t>::min() / right :
                       right < std::numeric_limits<int64_t>::max() / left;
}
} // namespace

ge::Status DWConv2DQuantProcessFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    if (graph == nullptr || !graph->IsValid()) {
        return GRAPH_NOT_CHANGED;
    }
    passContext.SetPassName(FUSION_NAME.c_str());

    std::vector<GNode> cubeNodes;
    if (!ConvFusionUtilsPass::GetMatchedNodes(graph, cubeNodes, {DEPTHWISE_CONV2D})) {
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

void DWConv2DQuantProcessFusionPass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    convDescInfo = ConvDescInfo();
    quantNode_ = nullptr;
    dequantNode_ = nullptr;
    padNode_ = nullptr;
}

bool DWConv2DQuantProcessFusionPass::CheckQuantChain(const ge::GNode& cubeNode)
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
        OP_LOGD(FUSION_NAME, "DepthwiseConv2D input producer is not AscendQuant or Pad, no fusion.");
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
                      OP_LOGD(FUSION_NAME, "DepthwiseConv2D output consumer is not AscendDequant, no fusion."),
                      return false);
    return true;
}

ge::Status DWConv2DQuantProcessFusionPass::GetCoValue(const ge::GNodePtr& cubeNode, int64_t& co)
{
    ge::TensorDesc filterDesc;
    FUSION_PASS_CHECK(cubeNode->GetInputDesc(static_cast<int32_t>(GetWeightInputIndex()), filterDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get filter desc failed."), return FAILED);

    ge::Format filterFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(static_cast<int32_t>(filterDesc.GetFormat())));
    if (filterFormat == FORMAT_ND) {
        return SUCCESS;
    }
    int64_t groups = 1;
    (void)cubeNode->GetAttr(GROUPS, groups);
    std::vector<int64_t> filterDims4D;
    FUSION_PASS_CHECK_NOLOG(!PadShapeTo4Dim(filterFormat, filterDesc.GetShape().GetDims(), filterDims4D),
                            return FAILED);
    FUSION_PASS_CHECK(filterDims4D.empty(),
                      OP_LOGE(FUSION_NAME, "Node[%s] filter dims4d is empty.", convDescInfo.nodeNameStr.c_str()),
                      return FAILED);
    int32_t indexCo = ops::GetAxisIndexByFormat(filterFormat, "N");
    if (indexCo < 0) {
        OP_LOGE(FUSION_NAME, "Node[%s] index_co is negative, check filter format.", convDescInfo.nodeNameStr.c_str());
        return FAILED;
    }
    if (indexCo >= static_cast<int32_t>(filterDims4D.size())) {
        OP_LOGE(FUSION_NAME, "Node[%s] index_co is larger than the size of filter dims.",
                convDescInfo.nodeNameStr.c_str());
        return FAILED;
    }
    FUSION_PASS_CHECK(CheckInt64MulOverflow(filterDims4D[indexCo], groups),
                      OP_LOGE(FUSION_NAME, "Node[%s] int64 mul overflow.", convDescInfo.nodeNameStr.c_str()),
                      return FAILED);
    co = filterDims4D[indexCo] * groups;
    return SUCCESS;
}

ge::Status DWConv2DQuantProcessFusionPass::SetDataTypeOfNodes(const ge::GNodePtr& cubeNode)
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

ge::GNodePtr DWConv2DQuantProcessFusionPass::GetCubeNodeInputNode(const ge::GNodePtr& cubeNode) const
{
    return padNode_ != nullptr ? padNode_ : cubeNode;
}

ge::Status DWConv2DQuantProcessFusionPass::SetPadNodeDataType(ge::GNode& padNode) const
{
    ge::TensorDesc padInDesc;
    FUSION_PASS_CHECK(padNode.GetInputDesc(INPUT_FMAP_INDEX, padInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get pad input desc failed."), return FAILED);
    padInDesc.SetDataType(ge::DT_FLOAT16);
#if GE_COMPILER_VERSION_NUM >= 90300000U
    padInDesc.SetOriginDataType(ge::DT_FLOAT16);
#endif
    FUSION_PASS_CHECK(padNode.UpdateInputDesc(INPUT_FMAP_INDEX, padInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Fail to update input desc of pad node."), return FAILED);

    ge::TensorDesc padOutDesc;
    FUSION_PASS_CHECK(padNode.GetOutputDesc(OUTPUT_INDEX, padOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get pad output desc failed."), return FAILED);
    padOutDesc.SetDataType(ge::DT_FLOAT16);
#if GE_COMPILER_VERSION_NUM >= 90300000U
    padOutDesc.SetOriginDataType(ge::DT_FLOAT16);
#endif
    FUSION_PASS_CHECK(padNode.UpdateOutputDesc(OUTPUT_INDEX, padOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Fail to update output desc of pad node."), return FAILED);

    OP_LOGD(FUSION_NAME, "Data type of Pad has been updated to float16 during quant rollback.");
    return SUCCESS;
}

bool DWConv2DQuantProcessFusionPass::MeetRequirements(const ge::GNode& convNode)
{
    (void)convNode;
    return true;
}

bool DWConv2DQuantProcessFusionPass::CheckOpSupported(const ge::GNode& node) const
{
    // Ascend950 包不携带depthwise TBE 实现，恒失败并走 quant rollback，
    // 此处以 dn2nz intrinsic 识别该 soc 家族保持同语义；GE 提供普通模式 CheckSupported 接口后应移除。
    (void)node;
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != 0U) {
        OP_LOGD(FUSION_NAME, "Get platform info failed, keep cube node supported.");
        return true;
    }
    if (platformInfo.ai_core_intrinsic_dtype_map.find(DN2NZ_INTRINSIC) !=
        platformInfo.ai_core_intrinsic_dtype_map.end()) {
        OP_LOGD(FUSION_NAME, "Cube node is unsupported on soc without legacy depthwise impl, quant rollback.");
        return false;
    }
    return true;
}

std::set<ge::AscendString> DWConv2DQuantProcessFusionPass::GetNodeTypes() const { return {DEPTHWISE_CONV2D}; }

void DWConv2DQuantProcessFusionPass::PrintGraphStructure() const
{
    OP_LOGI(FUSION_NAME, "%s %s done.", convDescInfo.nodeNameStr.c_str(), FUSION_NAME.c_str());
}

#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(DWConv2DQuantProcessFusionPass).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace Ops
