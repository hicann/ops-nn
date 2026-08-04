/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_unsqueeze_conv2d_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "graph/graph.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

#if GE_COMPILER_VERSION_NUM >= 90100000U
#include "ge/fusion/graph_fuse_inspector_utils.h"
#endif

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace QuantUnsqueezeConv2dFusion;
using namespace ge;

namespace {
ge::GNodePtr FindOutNodeByType(const GNode& node, const AscendString& expectType)
{
    for (auto& outPair : node.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX)) {
        AscendString type;
        FUSION_PASS_CHECK_NOLOG(outPair.first->GetType(type) != GRAPH_SUCCESS, continue);
        if (type == expectType) {
            return outPair.first;
        }
    }
    return nullptr;
}

bool IsNodeType(const GNodePtr& node, const AscendString& expectType)
{
    if (node == nullptr) {
        return false;
    }
    AscendString type;
    FUSION_PASS_CHECK_NOLOG(node->GetType(type) != GRAPH_SUCCESS, return false);
    return type == expectType;
}
} // namespace

void QuantUnsqueezeConv2DFusionPass::InitMember()
{
    broadcastFlag = false;
    broadcastNode = nullptr;
    dequantNode = nullptr;
    quantNode = nullptr;
    squeezeNode = nullptr;
    unsqueezeNode = nullptr;
}

bool QuantUnsqueezeConv2DFusionPass::CheckSocCapability() const
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optiInfo) !=
        PLATFORM_INFO_OK) {
        OP_LOGD(FUSION_NAME, "Get platform info failed, not support fusion.");
        return false;
    }
    const auto& intrMap = platformInfo.ai_core_intrinsic_dtype_map;
    bool ub2ub = intrMap.find(INTRINSIC_CONV_UB_TO_UB) != intrMap.end();
    bool dn2nz = intrMap.find(INTRINSIC_DATA_MOVE_OUT2L1_DN2NZ) != intrMap.end();
    if (!ub2ub && !dn2nz) {
        OP_LOGD(FUSION_NAME, "Neither conv_ub2ub nor dn2nz instr is supported, not support fusion.");
        return false;
    }
    return true;
}

bool QuantUnsqueezeConv2DFusionPass::CheckMatchStructure(const GNode& matchNode)
{
    auto unsqueezePair = matchNode.GetInDataNodesAndPortIndexs(INPUT_FMAP_INDEX);
    unsqueezeNode = unsqueezePair.first;
    FUSION_PASS_CHECK(!IsNodeType(unsqueezeNode, UNSQUEEZE),
                      OP_LOGD(convDescInfo.nodeNameStr, "Conv2D input producer is not Unsqueeze, no fusion."),
                      return false);

    auto quantPair = unsqueezeNode->GetInDataNodesAndPortIndexs(INPUT_INDEX_0);
    quantNode = quantPair.first;
    FUSION_PASS_CHECK(!IsNodeType(quantNode, ASCEND_QUANT),
                      OP_LOGD(convDescInfo.nodeNameStr, "Unsqueeze input producer is not AscendQuant, no fusion."),
                      return false);

    squeezeNode = FindOutNodeByType(matchNode, SQUEEZE);
    FUSION_PASS_CHECK(squeezeNode == nullptr,
                      OP_LOGD(convDescInfo.nodeNameStr, "Conv2D output consuimer is not Squeeze, no fusion."),
                      return false);

    dequantNode = FindOutNodeByType(*squeezeNode, ASCEND_DEQUANT);
    FUSION_PASS_CHECK(dequantNode == nullptr,
                      OP_LOGD(convDescInfo.nodeNameStr, "Squeeze output consumer is not AscendDequant, no fusion."),
                      return false);
    return true;
}

bool QuantUnsqueezeConv2DFusionPass::CheckNodeNoControlAnchor(const GNode& node) const
{
    return node.GetInControlNodes().empty() && node.GetOutControlNodes().empty();
}

bool QuantUnsqueezeConv2DFusionPass::CheckUnsqueezeDim() const
{
    TensorDesc desc;
    FUSION_PASS_CHECK(unsqueezeNode->GetInputDesc(INPUT_INDEX_0, desc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Unsqueeze input desc failed."), return false);
    int64_t dimSize = static_cast<int64_t>(desc.GetShape().GetDimNum());
    if (dimSize != CONV1D_DIM_SIZE) {
        OP_LOGD(convDescInfo.nodeNameStr, "Unsqueeze input dim size [%lld] is invalid, no fusion.", dimSize);
        return false;
    }

    return true;
}

bool QuantUnsqueezeConv2DFusionPass::MeetRequirements(const GNode& convNode)
{
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do QuantUnsqueezeConv2DFusionPass.");
    FUSION_PASS_CHECK_NOLOG(!CheckSocCapability(), return false);

    FUSION_PASS_CHECK(!CheckNodeNoControlAnchor(*quantNode),
                      OP_LOGD(convDescInfo.nodeNameStr, "Quant node has control anchor, no fusion."), return false);
    FUSION_PASS_CHECK(!CheckNodeNoControlAnchor(*unsqueezeNode),
                      OP_LOGD(convDescInfo.nodeNameStr, "Unsqueeze node has control anchor, no fusion."), return false);
    FUSION_PASS_CHECK(!CheckNodeNoControlAnchor(convNode),
                      OP_LOGD(convDescInfo.nodeNameStr, "Conv2D node has control anchor, no fusion."), return false);
    FUSION_PASS_CHECK(!CheckNodeNoControlAnchor(*squeezeNode),
                      OP_LOGD(convDescInfo.nodeNameStr, "Squeeze node has control anchor, no fusion."), return false);
    FUSION_PASS_CHECK(!CheckNodeNoControlAnchor(*dequantNode),
                      OP_LOGD(convDescInfo.nodeNameStr, "Dequant node has control anchor, no fusion."), return false);

    FUSION_PASS_CHECK_NOLOG(!CheckUnsqueezeDim(), return false);

    return true;
}

std::set<AscendString> QuantUnsqueezeConv2DFusionPass::GetNodeTypes() const { return {CONV2D}; }

void QuantUnsqueezeConv2DFusionPass::PrintGraphStructure() const
{
    OP_LOGI(convDescInfo.nodeNameStr, "QuantUnsqueezeConv2D fusion done: unsqueeze->quant->conv2d->dequant%s->squeeze.",
            broadcastFlag ? "->broadcast" : "");
}

bool QuantUnsqueezeConv2DFusionPass::UpdateQuantUnsqueezeDesc()
{
    TensorDesc quantIn, quantOut, unsqueezeIn, unsqueezeOut;
    FUSION_PASS_CHECK_NOLOG(quantNode->GetInputDesc(INPUT_INDEX_0, quantIn) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(quantNode->GetOutputDesc(OUTPUT_INDEX, quantOut) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(unsqueezeNode->GetInputDesc(INPUT_INDEX_0, unsqueezeIn) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(unsqueezeNode->GetOutputDesc(OUTPUT_INDEX, unsqueezeOut) != GRAPH_SUCCESS, return false);

    Shape outShape = unsqueezeOut.GetShape();
    int64_t realDimCnt = static_cast<int64_t>(outShape.GetDimNum());
    quantIn.SetShape(outShape);
    quantIn.SetOriginShape(outShape);
    quantIn.SetRealDimCnt(realDimCnt);
    quantOut.SetShape(outShape);
    quantOut.SetOriginShape(outShape);
    quantOut.SetRealDimCnt(realDimCnt);
    unsqueezeIn.SetDataType(DT_FLOAT16);
    unsqueezeOut.SetDataType(DT_FLOAT16);

    FUSION_PASS_CHECK_NOLOG(quantNode->UpdateInputDesc(INPUT_INDEX_0, quantIn) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(quantNode->UpdateOutputDesc(OUTPUT_INDEX, quantOut) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(unsqueezeNode->UpdateInputDesc(INPUT_INDEX_0, unsqueezeIn) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(unsqueezeNode->UpdateOutputDesc(OUTPUT_INDEX, unsqueezeOut) != GRAPH_SUCCESS, return false);
    return true;
}

bool QuantUnsqueezeConv2DFusionPass::UpdateSqueezeDequantDesc()
{
    TensorDesc squeezeIn, squeezeOut, dequantIn, dequantOut;
    FUSION_PASS_CHECK_NOLOG(squeezeNode->GetInputDesc(INPUT_INDEX_0, squeezeIn) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(squeezeNode->GetOutputDesc(OUTPUT_INDEX, squeezeOut) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(dequantNode->GetInputDesc(INPUT_INDEX_0, dequantIn) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(dequantNode->GetOutputDesc(OUTPUT_INDEX, dequantOut) != GRAPH_SUCCESS, return false);

    Shape shape = squeezeIn.GetShape();
    int64_t realDimCnt = static_cast<int64_t>(shape.GetDimNum());
    dequantIn.SetShape(shape);
    dequantIn.SetOriginShape(shape);
    dequantIn.SetRealDimCnt(realDimCnt);
    dequantOut.SetShape(shape);
    dequantOut.SetOriginShape(shape);
    dequantOut.SetRealDimCnt(realDimCnt);
    squeezeIn.SetDataType(DT_FLOAT16);
    squeezeOut.SetDataType(DT_FLOAT16);

    FUSION_PASS_CHECK_NOLOG(dequantNode->UpdateInputDesc(INPUT_INDEX_0, dequantIn) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(dequantNode->UpdateOutputDesc(OUTPUT_INDEX, dequantOut) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(squeezeNode->UpdateInputDesc(INPUT_INDEX_0, squeezeIn) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(squeezeNode->UpdateOutputDesc(OUTPUT_INDEX, squeezeOut) != GRAPH_SUCCESS, return false);

    if (broadcastFlag) {
        TensorDesc bIn, bOut;
        FUSION_PASS_CHECK_NOLOG(broadcastNode->GetInputDesc(INPUT_INDEX_0, bIn) != GRAPH_SUCCESS, return false);
        FUSION_PASS_CHECK_NOLOG(broadcastNode->GetOutputDesc(OUTPUT_INDEX, bOut) != GRAPH_SUCCESS, return false);
        bIn.SetShape(shape);
        bIn.SetOriginShape(shape);
        bIn.SetRealDimCnt(realDimCnt);
        bOut.SetShape(shape);
        bOut.SetOriginShape(shape);
        bOut.SetRealDimCnt(realDimCnt);
        FUSION_PASS_CHECK_NOLOG(broadcastNode->UpdateInputDesc(INPUT_INDEX_0, bIn) != GRAPH_SUCCESS, return false);
        FUSION_PASS_CHECK_NOLOG(broadcastNode->UpdateOutputDesc(OUTPUT_INDEX, bOut) != GRAPH_SUCCESS, return false);
    }
    return true;
}

Status QuantUnsqueezeConv2DFusionPass::RelinkQuantUnsqueezeEdges(Graph& graph)
{
    auto [preNode, preOutPort] = quantNode->GetInDataNodesAndPortIndexs(INPUT_INDEX_0);
    FUSION_PASS_CHECK(preNode == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Get quant input producer failed."),
                      return FAILED);

    FUSION_PASS_CHECK(graph.RemoveEdge(*preNode, preOutPort, *quantNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Remove preNode-quant edge failed."), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveEdge(*quantNode, OUTPUT_INDEX, *unsqueezeNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Remove quant-unsqueeze edge failed."), return FAILED);

    auto consumers = unsqueezeNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (auto& [consumer, inPort] : consumers) {
        FUSION_PASS_CHECK(graph.RemoveEdge(*unsqueezeNode, OUTPUT_INDEX, *consumer, inPort) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Remove unsqueeze-outnode edge failed."), return FAILED);
        FUSION_PASS_CHECK(graph.AddDataEdge(*quantNode, OUTPUT_INDEX, *consumer, inPort) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Add quant-outnode edge failed."), return FAILED);
    }

    FUSION_PASS_CHECK(graph.AddDataEdge(*unsqueezeNode, OUTPUT_INDEX, *quantNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Add unsqueeze-quant edge failed."), return FAILED);
    FUSION_PASS_CHECK(graph.AddDataEdge(*preNode, preOutPort, *unsqueezeNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Add preNode-unsqueeze edge failed."), return FAILED);
    return SUCCESS;
}

Status QuantUnsqueezeConv2DFusionPass::RelinkSqueezeDequantEdges(Graph& graph, GNode& convNode)
{
    FUSION_PASS_CHECK(graph.RemoveEdge(convNode, OUTPUT_INDEX, *squeezeNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Remove conv-squeeze edge failed."), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveEdge(*squeezeNode, OUTPUT_INDEX, *dequantNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Remove squeeze-dequant edge failed."), return FAILED);

    GNodePtr linkNode = broadcastFlag ? broadcastNode : dequantNode;
    auto consumers = linkNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (auto& [consumer, inPort] : consumers) {
        FUSION_PASS_CHECK(graph.RemoveEdge(*linkNode, OUTPUT_INDEX, *consumer, inPort) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Remove linkNode-outnode edge failed."), return FAILED);
        FUSION_PASS_CHECK(graph.AddDataEdge(*squeezeNode, OUTPUT_INDEX, *consumer, inPort) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Add squeeze-outnode edge failed."), return FAILED);
    }

    FUSION_PASS_CHECK(graph.AddDataEdge(convNode, OUTPUT_INDEX, *dequantNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Add conv-dequant edge failed."), return FAILED);
    FUSION_PASS_CHECK(graph.AddDataEdge(*linkNode, OUTPUT_INDEX, *squeezeNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Add linkNode-squeeze edge failed."), return FAILED);
    return SUCCESS;
}

Status QuantUnsqueezeConv2DFusionPass::ConvFusionPreImpl(GraphPtr& graph, GNode& convNode,
                                                         CustomPassContext& passContext)
{
    auto consumers = dequantNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    if (consumers.size() != SINGLE_CONSUMER_CNT) {
        OP_LOGD(convDescInfo.nodeNameStr, "Dequant consumers size [%zu] is not 1, no broadcast.", consumers.size());
        return SUCCESS;
    }

    AscendString type;
    FUSION_PASS_CHECK(consumers[FIRST_CONSUMER_IDX].first == nullptr,
                      OP_LOGD(convDescInfo.nodeNameStr, "Dequant consumer node is null, no broadcast."),
                      return SUCCESS);
    FUSION_PASS_CHECK(consumers[FIRST_CONSUMER_IDX].first->GetType(type) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get dequant consumer type failed."), return FAILED);
    if (BROADCAST_WHITELIST.count(type) != 0) {
        broadcastFlag = true;
        broadcastNode = consumers[FIRST_CONSUMER_IDX].first;
        OP_LOGD(convDescInfo.nodeNameStr, "Detect broadcast node [%s].", type.GetString());
    }
    return SUCCESS;
}

bool QuantUnsqueezeConv2DFusionPass::ConvFusionReplaceImpl(GraphPtr& graph, GNode& convNode,
                                                           CustomPassContext& passContext)
{
    std::vector<GNode> nodesBeforeFuse = {*quantNode, *unsqueezeNode, convNode, *squeezeNode, *dequantNode};
    if (broadcastFlag && broadcastNode != nullptr) {
        nodesBeforeFuse.emplace_back(*broadcastNode);
    }

    AscendString failedReason;
#if GE_COMPILER_VERSION_NUM >= 90100000U
    FUSION_PASS_CHECK(!ge::fusion::GraphFuseInspectorUtils::CanFuse(nodesBeforeFuse, failedReason),
                      OP_LOGD(convDescInfo.nodeNameStr, "CanFuse failed, reason: %s.", failedReason.GetString()),
                      return false);
#endif

    FUSION_PASS_CHECK(!UpdateQuantUnsqueezeDesc(),
                      OP_LOGE(convDescInfo.nodeNameStr, "Update quant/unsqueeze desc failed."), return false);
    FUSION_PASS_CHECK(RelinkQuantUnsqueezeEdges(*graph) != SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Relink quant/unsqueeze edges failed."), return false);
    FUSION_PASS_CHECK(!UpdateSqueezeDequantDesc(),
                      OP_LOGE(convDescInfo.nodeNameStr, "Update squeeze/dequant desc failed."), return false);
    FUSION_PASS_CHECK(RelinkSqueezeDequantEdges(*graph, convNode) != SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Relink squeeze/dequant edges failed."), return false);
#if GE_COMPILER_VERSION_NUM >= 90100000U
    FUSION_PASS_CHECK(
        ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, nodesBeforeFuse, passContext) != SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "ReportFuse failed."), return false);
#endif
    return true;
}
} // namespace Ops
