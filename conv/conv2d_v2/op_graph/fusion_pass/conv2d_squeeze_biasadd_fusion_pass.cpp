/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv2d_squeeze_biasadd_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "graph/graph.h"
#include "graph/utils/type_utils.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

#if GE_COMPILER_VERSION_NUM >= 90100000U
#include "ge/fusion/graph_fuse_inspector_utils.h"
#endif

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace Conv2DSqueezeBiasaddFusion;
using namespace ge;
using namespace fusion;

namespace {
GNodePtr FindOutNodeByType(const GNode& node, const int32_t outputIndex, const AscendString& expectType)
{
    for (auto& outPair : node.GetOutDataNodesAndPortIndexs(outputIndex)) {
        AscendString type;
        FUSION_PASS_CHECK_NOLOG(outPair.first == nullptr, continue);
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

void Conv2DSqueezeBiasaddFusionPass::InitMember()
{
    convNode = nullptr;
    biasaddNode = nullptr;
    squeezeNodeName = "";
    convOutputShape = {};
    convOutputDesc = {};
    convDescInfo = ConvDescInfo();
}

bool Conv2DSqueezeBiasaddFusionPass::CheckMatchStructure(const GNode& matchNode)
{
    auto inPair = matchNode.GetInDataNodesAndPortIndexs(SQUEEZE_INPUT_INDEX);
    FUSION_PASS_CHECK(!IsNodeType(inPair.first, CONV2D), OP_LOGD(FUSION_NAME, "squeeze input is not Conv2D."),
                      return false);
    convNode = inPair.first;

    biasaddNode = FindOutNodeByType(matchNode, SQUEEZE_OUTPUT_INDEX, BIASADD);
    FUSION_PASS_CHECK(biasaddNode == nullptr, OP_LOGD(FUSION_NAME, "squeeze output consumer is not BiasAdd."),
                      return false);
    return true;
}

bool Conv2DSqueezeBiasaddFusionPass::MeetRequirements(const GNode& matchNode)
{
    FUSION_PASS_CHECK(convNode == nullptr, OP_LOGD(FUSION_NAME, "conv node is null."), return false);
    FUSION_PASS_CHECK(biasaddNode == nullptr, OP_LOGD(FUSION_NAME, "biasadd node is null."), return false);

    auto squeezeInPair = biasaddNode->GetInDataNodesAndPortIndexs(BIASADD_X_INPUT_INDEX);
    FUSION_PASS_CHECK(!IsNodeType(squeezeInPair.first, SQUEEZE),
                      OP_LOGD(FUSION_NAME, "biasadd input 0 producer is not Squeeze."), return false);

    AscendString matchNodeName;
    FUSION_PASS_CHECK_NOLOG(matchNode.GetName(matchNodeName) != GRAPH_SUCCESS, return false);
    squeezeNodeName = matchNodeName;

    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvDescInfo(*convNode, convDescInfo), return false);
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do %s.", FUSION_NAME.c_str());

    FUSION_PASS_CHECK(convNode->GetOutputDesc(OUTPUT_INDEX, convOutputDesc) != GRAPH_SUCCESS,
                      OP_LOGW(convDescInfo.nodeNameStr, "get conv output desc failed."), return false);

    TensorDesc biasDesc;
    FUSION_PASS_CHECK(biasaddNode->GetInputDesc(BIASADD_BIAS_INPUT_INDEX, biasDesc) != GRAPH_SUCCESS,
                      OP_LOGW(convDescInfo.nodeNameStr, "get biasadd bias desc failed."), return false);
    auto biasShape = biasDesc.GetShape().GetDims();
    FUSION_PASS_CHECK(biasShape.size() != BIAS_DIM,
                      OP_LOGD(convDescInfo.nodeNameStr, "bias dim is %zu, not 1, no fusion.", biasShape.size()),
                      return false);

    TensorDesc biasaddXDesc;
    FUSION_PASS_CHECK(biasaddNode->GetInputDesc(BIASADD_X_INPUT_INDEX, biasaddXDesc) != GRAPH_SUCCESS,
                      OP_LOGW(convDescInfo.nodeNameStr, "get biasadd x desc failed."), return false);
    FUSION_PASS_CHECK(ConvFusionUtilsPass::IsUnknownShape(biasaddXDesc),
                      OP_LOGD(convDescInfo.nodeNameStr, "biasadd x is unknown shape, no fusion."), return false);
    FUSION_PASS_CHECK(ConvFusionUtilsPass::IsUnknownShape(biasDesc),
                      OP_LOGD(convDescInfo.nodeNameStr, "biasadd bias is unknown shape, no fusion."), return false);

    auto biasProducerPair = biasaddNode->GetInDataNodesAndPortIndexs(BIASADD_BIAS_INPUT_INDEX);
    FUSION_PASS_CHECK(biasProducerPair.first == nullptr, OP_LOGD(convDescInfo.nodeNameStr, "bias producer is null."),
                      return false);
    AscendString biasProducerType;
    FUSION_PASS_CHECK(biasProducerPair.first->GetType(biasProducerType) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "get bias producer type failed."), return false);
    FUSION_PASS_CHECK(biasProducerType == VARIABLE,
                      OP_LOGD(convDescInfo.nodeNameStr, "training mode (Variable bias), no fusion."), return false);

    return true;
}

std::set<AscendString> Conv2DSqueezeBiasaddFusionPass::GetNodeTypes() const { return {SQUEEZE}; }

void Conv2DSqueezeBiasaddFusionPass::PrintGraphStructure() const
{
    AscendString convName;
    convNode->GetName(convName);
    AscendString biasaddName;
    biasaddNode->GetName(biasaddName);
    OP_LOGI(convDescInfo.nodeNameStr,
            "%s: conv[%s] -> squeeze[%s] -> biasadd[%s] reordered to conv -> biasadd -> squeeze.", FUSION_NAME.c_str(),
            convName.GetString(), squeezeNodeName.GetString(), biasaddName.GetString());
}

bool Conv2DSqueezeBiasaddFusionPass::ConvFusionReplaceImpl(GraphPtr& graph, GNode& matchNode,
                                                           CustomPassContext& passContext)
{
    std::vector<GNode> nodesBeforeFuse = {*convNode, matchNode, *biasaddNode};

    AscendString failedReason;
#if GE_COMPILER_VERSION_NUM >= 90100000U
    FUSION_PASS_CHECK(!ge::fusion::GraphFuseInspectorUtils::CanFuse(nodesBeforeFuse, failedReason),
                      OP_LOGD(convDescInfo.nodeNameStr, "CanFuse failed, reason: %s.", failedReason.GetString()),
                      return false);
#endif

    FUSION_PASS_CHECK(!UpdateBiasAddDesc(), OP_LOGE(convDescInfo.nodeNameStr, "Update biasadd desc failed."),
                      return false);
    FUSION_PASS_CHECK(RelinkEdges(*graph, matchNode) != SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Relink edges failed."), return false);
#if GE_COMPILER_VERSION_NUM >= 90100000U
    FUSION_PASS_CHECK(
        ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, nodesBeforeFuse, passContext) != SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "ReportFuse failed."), return false);
#endif
    return true;
}

bool Conv2DSqueezeBiasaddFusionPass::UpdateBiasAddDesc()
{
    convOutputShape = convOutputDesc.GetShape();

    TensorDesc biasaddXDesc;
    FUSION_PASS_CHECK(biasaddNode->GetInputDesc(BIASADD_X_INPUT_INDEX, biasaddXDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "get biasadd x desc failed during replace."), return false);
    biasaddXDesc.SetShape(convOutputShape);
    biasaddXDesc.SetOriginShape(convOutputShape);
    FUSION_PASS_CHECK(biasaddNode->UpdateInputDesc(BIASADD_X_INPUT_INDEX, biasaddXDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "update biasadd x desc failed."), return false);

    TensorDesc biasaddYDesc;
    FUSION_PASS_CHECK(biasaddNode->GetOutputDesc(OUTPUT_INDEX, biasaddYDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "get biasadd y desc failed during replace."), return false);
    biasaddYDesc.SetShape(convOutputShape);
    biasaddYDesc.SetOriginShape(convOutputShape);
    FUSION_PASS_CHECK(biasaddNode->UpdateOutputDesc(OUTPUT_INDEX, biasaddYDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "update biasadd y desc failed."), return false);
    return true;
}

Status Conv2DSqueezeBiasaddFusionPass::RelinkEdges(Graph& graph, GNode& squeezeNode)
{
    FUSION_PASS_CHECK(graph.RemoveEdge(*convNode, OUTPUT_INDEX, squeezeNode, SQUEEZE_INPUT_INDEX) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Remove conv-squeeze edge failed."), return FAILED);
    FUSION_PASS_CHECK(
        graph.RemoveEdge(squeezeNode, SQUEEZE_OUTPUT_INDEX, *biasaddNode, BIASADD_X_INPUT_INDEX) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Remove squeeze-biasadd edge failed."), return FAILED);

    auto consumers = biasaddNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (auto& [consumer, inPort] : consumers) {
        FUSION_PASS_CHECK_NOLOG(consumer == nullptr, continue);
        FUSION_PASS_CHECK(graph.RemoveEdge(*biasaddNode, OUTPUT_INDEX, *consumer, inPort) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Remove biasadd-outnode edge failed."), return FAILED);
        FUSION_PASS_CHECK(graph.AddDataEdge(squeezeNode, SQUEEZE_OUTPUT_INDEX, *consumer, inPort) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Add squeeze-outnode edge failed."), return FAILED);
    }

    FUSION_PASS_CHECK(graph.AddDataEdge(*convNode, OUTPUT_INDEX, *biasaddNode, BIASADD_X_INPUT_INDEX) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Add conv-biasadd edge failed."), return FAILED);
    FUSION_PASS_CHECK(graph.AddDataEdge(*biasaddNode, OUTPUT_INDEX, squeezeNode, SQUEEZE_INPUT_INDEX) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Add biasadd-squeeze edge failed."), return FAILED);
    return SUCCESS;
}
#if GE_COMPILER_VERSION_NUM >= 90100000U
REG_FUSION_PASS(Conv2DSqueezeBiasaddFusionPass).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace Ops
