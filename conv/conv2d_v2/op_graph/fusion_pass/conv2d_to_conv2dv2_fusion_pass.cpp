/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv2d_to_conv2dv2_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace Conv2dToConv2dV2Fusion;
using namespace ge;
using namespace fusion;

void Conv2dToConv2dV2FusionPass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    convDescInfo = ConvDescInfo();
}

bool Conv2dToConv2dV2FusionPass::MeetRequirements([[maybe_unused]] const GNode& convNode)
{
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSocList(SUPPORT_SOC_LIST, npuArch, true),
                      OP_LOGD(convDescInfo.nodeNameStr, "Current soc not supported, no fusion."), return false);

    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do Conv2dToConv2dV2FusionPass.");

    return true;
}

std::set<AscendString> Conv2dToConv2dV2FusionPass::GetNodeTypes() const { return {ConvFusionUtils::CONV2D}; }

void Conv2dToConv2dV2FusionPass::PrintGraphStructure() const
{
    OP_LOGI(convDescInfo.nodeNameStr, "graph structure: cube node: [%s] -> Conv2DV2.",
            convDescInfo.nodeNameStr.c_str());
}

bool Conv2dToConv2dV2FusionPass::AdjustLayout(const GraphPtr& graph, GNode& fusedNode)
{
    if (!ConvFusionUtilsPass::IsFuseSoc()) {
        OP_LOGD(convDescInfo.nodeNameStr, "Current soc not support adjust layout, skip.");
        return true;
    }
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AdjustConvInputLayout(graph, fusedNode), return false);

    auto outNodes = fusedNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK(outNodes.size() != 1, OP_LOGD(convDescInfo.nodeNameStr, "Output consumers != 1, skip transpose."),
                      return true);
    auto postNode = outNodes[0].first;
    FUSION_PASS_CHECK(postNode == nullptr, OP_LOGD(convDescInfo.nodeNameStr, "Output node is nullptr, skip transpose."),
                      return true);
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::CheckTransposeValid(*postNode), return true);

    GNode transposeNode = *postNode;
    TensorDesc transOutDesc;
    FUSION_PASS_CHECK(transposeNode.GetOutputDesc(OUTPUT_INDEX, transOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Transpose output desc failed."), return false);
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::RemoveTransposeNode(graph, transposeNode), return false);
    FUSION_PASS_CHECK(fusedNode.UpdateOutputDesc(OUTPUT_INDEX, transOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update fused node output desc failed."), return false);
    OP_LOGD(convDescInfo.nodeNameStr, "AdjustLayout post transpose success.");
    return true;
}

std::unique_ptr<SubgraphBoundary> Conv2dToConv2dV2FusionPass::ConstructBoundary(const GNode& convNode)
{
    auto boundary = std::make_unique<SubgraphBoundary>();
    for (size_t i = 0; i < convNode.GetInputsSize(); ++i) {
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, convNode, static_cast<int64_t>(i),
                                                                       static_cast<int64_t>(i)),
                                return nullptr);
    }
    FUSION_PASS_CHECK_NOLOG(
        !ConvFusionUtilsPass::AddSubgraphOutput(boundary, convNode, static_cast<int64_t>(OUTPUT_INDEX),
                                                static_cast<int64_t>(OUTPUT_INDEX)),
        return nullptr);
    return boundary;
}

GraphUniqPtr Conv2dToConv2dV2FusionPass::Replacement(const GNode& convNode)
{
    auto graphBuilder = es::EsGraphBuilder("replacement");

    ConvBaseAttrs baseAttrs;
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvBaseAttr(convNode, baseAttrs, convDescInfo), return nullptr);

    auto [fmap, filter] = graphBuilder.CreateInputs<REQUIRED_INPUT_NUMS>();
    auto bias = convDescInfo.hasBias ? graphBuilder.CreateInput(static_cast<int64_t>(INPUT_BIAS_INDEX)) : nullptr;

    auto conv2dV2 = es::Conv2DV2(fmap, filter, bias, nullptr, baseAttrs.strides, baseAttrs.pads, baseAttrs.dilations,
                                 baseAttrs.groups, baseAttrs.dataFormat.GetString(), baseAttrs.offsetX,
                                 baseAttrs.padMode.GetString(), baseAttrs.enableHf32);

    auto conv2dV2Node = conv2dV2.GetProducer();
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::UpdateInputDesc(conv2dV2Node, convDescInfo), return nullptr);
    FUSION_PASS_CHECK(conv2dV2Node->UpdateOutputDesc(OUTPUT_INDEX, convDescInfo.outputDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update Conv2DV2 output tensor desc failed."), return nullptr);
    FUSION_PASS_CHECK(conv2dV2Node->SetAttr(OP_IMPL_MODE_ENUM, baseAttrs.opImplModeEnum) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set _op_impl_mode_enum for Conv2DV2 failed."), return nullptr);

    int64_t fixedShiftValue = 0;
    (void)convNode.GetAttr(FIXED_SHIFT_VALUE.GetString(), fixedShiftValue);
    if (fixedShiftValue == 0) {
        fixedShiftValue = FIXED_SHIFT_VALUE_DEFAULT_A16W16;
    }
    FUSION_PASS_CHECK(conv2dV2Node->SetAttr(FIXED_SHIFT_VALUE.GetString(), fixedShiftValue) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set fixed_shift_value for Conv2DV2 failed."), return nullptr);

    (void)ConvFusionUtilsPass::CopyConvOtherAttrs(convNode, *conv2dV2Node);

    return graphBuilder.BuildAndReset({conv2dV2});
}

bool Conv2dToConv2dV2FusionPass::ConvFusionReplaceImpl(GraphPtr& graph, GNode& convNode, CustomPassContext& passContext)
{
    FUSION_PASS_CHECK(graph == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Graph is nullptr."), return false);

    auto boundary = ConstructBoundary(convNode);
    FUSION_PASS_CHECK(boundary == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Construct boundary failed."),
                      return false);

    auto replacement = Replacement(convNode);
    FUSION_PASS_CHECK(replacement == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Construct replacement failed."),
                      return false);

    auto conv2dV2NamesBefore = ConvFusionUtilsPass::CollectNodeTypeNames(graph, ConvFusionUtils::CONV2DV2);

    FUSION_PASS_CHECK(!ReplaceSubgraphWithFallback(boundary, replacement, passContext),
                      OP_LOGE(convDescInfo.nodeNameStr, "SubgraphRewriter::Replace failed."), return false);

    GNode newConv2dV2Node;
    FUSION_PASS_CHECK(
        !ConvFusionUtilsPass::FindNewNodeOfType(graph, ConvFusionUtils::CONV2DV2, conv2dV2NamesBefore, newConv2dV2Node),
        OP_LOGE(convDescInfo.nodeNameStr, "Find new Conv2DV2 node after replace failed."), return false);

    FUSION_PASS_CHECK_NOLOG(!AdjustLayout(graph, newConv2dV2Node), return false);

    OP_LOGD(convDescInfo.nodeNameStr, "Conv2dToConv2dV2FusionPass replace success.");
    return true;
}

#if GE_COMPILER_VERSION_NUM >= 90100000U
REG_FUSION_PASS(Conv2dToConv2dV2FusionPass).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace Ops
