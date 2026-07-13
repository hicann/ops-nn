/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv2d_backprop_filter_to_v3_fusion_pass.h"

namespace ops {
namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t FILTER_SIZE_INDEX = 1;
constexpr size_t CONV2D_BP_FILTER_INPUT_NUM = 3;
} // namespace

AscendString Conv2DBackpropFilterToV3FusionPass::GetNodeType() const { return CONV2D_BP_FILTER_TO_V3_PASS; }

bool Conv2DBackpropFilterToV3FusionPass::GetNodeAttrs(const ge::GNode& node)
{
    OP_CHECK_IF(!ConvBackpropFusionBasePass::GetNodeAttrs(node),
                OP_LOGE(GetNodeType().GetString(), "Base GetNodeAttrs failed"), return false);
    ConvBackpropFusionUtilsPass::ExpandAttrs(convBpAttr.strides, convBpAttr.pads, convBpAttr.dilations,
                                             convBpAttr.dataFormat);
    return true;
}

bool Conv2DBackpropFilterToV3FusionPass::BuildConv3DBackpropFilterNode(EsGraphBuilder& builder,
                                                                       EsTensorHolder& iFilterSize,
                                                                       UnsqueezeNodeInfo& unsqueezeXInfo,
                                                                       UnsqueezeNodeInfo& unsqueezeGradOutputInfo,
                                                                       const std::string& nodeNamePrefix,
                                                                       GNode& outNode, TensorDesc& output3DDesc)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(graph == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Get graph failed in BuildConv3DBackpropFilterNode"), return false);

    ConvBackpropFusionUtilsPass::ExpandOutputDesc(outputDesc, output3DDesc);

    std::string conv3dBpFilterName = nodeNamePrefix + "_to_Conv3DBackpropFilter";
    outNode = CompliantNodeBuilder(graph)
                  .OpType("Conv3DBackpropFilter")
                  .Name(conv3dBpFilterName.c_str())
                  .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                {"filter_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                  .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                  .Build();

    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, unsqueezeXInfo.node, 0, outNode, static_cast<int32_t>(X_INDEX)) !=
                    GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for unsqueeze x failed"), return false);

    auto* filterSizeProducer = iFilterSize.GetProducer();
    OP_CHECK_IF(filterSizeProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "Filter_size producer is nullptr"),
                return false);
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *filterSizeProducer, iFilterSize.GetProducerOutIndex(), outNode,
                                         static_cast<int32_t>(FILTER_SIZE_INDEX)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for filter_size failed"), return false);

    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, unsqueezeGradOutputInfo.node, 0, outNode,
                                         static_cast<int32_t>(OUT_BACKPROP_INDEX)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for unsqueeze grad_output failed"), return false);

    SetConv3DBackpropFilterAttrsAndDescs(outNode, unsqueezeXInfo.outDesc, unsqueezeGradOutputInfo.outDesc,
                                         output3DDesc);
    return true;
}

void Conv2DBackpropFilterToV3FusionPass::SetConv3DBackpropFilterAttrsAndDescs(
    GNode& outNode, const TensorDesc& unsqueezeXOutDesc, const TensorDesc& unsqueezeGradOutputOutDesc,
    const TensorDesc& output3DDesc)
{
    SetNodeAttrs(outNode);

    outNode.UpdateInputDesc(X_INDEX, unsqueezeXOutDesc);
    outNode.UpdateInputDesc(FILTER_SIZE_INDEX, input1Desc);
    outNode.UpdateInputDesc(OUT_BACKPROP_INDEX, unsqueezeGradOutputOutDesc);
    outNode.UpdateOutputDesc(OUTPUT_INDEX, output3DDesc);
}

GraphUniqPtr Conv2DBackpropFilterToV3FusionPass::Replacement(const GNode& convBpFilterNode)
{
    OP_LOGD(GetNodeType().GetString(), "Replacement start");

    OP_CHECK_IF(!GetNodeDesc(convBpFilterNode), OP_LOGE(GetNodeType().GetString(), "GetNodeDesc failed"),
                return nullptr);

    OP_CHECK_IF(!GetNodeAttrs(convBpFilterNode), OP_LOGE(GetNodeType().GetString(), "GetNodeAttrs failed"),
                return nullptr);

    auto builder = EsGraphBuilder("replacement");
    auto [iX, iFilterSize, iGradOutput] = builder.CreateInputs<CONV2D_BP_FILTER_INPUT_NUM>();

    std::string nodeNamePrefix;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::GetNodeName(convBpFilterNode, nodeNamePrefix),
                OP_LOGE(GetNodeType().GetString(), "Get node name failed"), return nullptr);

    UnsqueezeNodeInfo unsqueezeXInfo;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateUnsqueezeNode(
                    builder, iX, input0Desc, nodeNamePrefix + "_Unsqueeze_0", unsqueezeXInfo, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Create unsqueeze x node failed"), return nullptr);

    UnsqueezeNodeInfo unsqueezeGradOutputInfo;
    OP_CHECK_IF(
        !ConvBackpropFusionUtilsPass::CreateUnsqueezeNode(
            builder, iGradOutput, input2Desc, nodeNamePrefix + "_Unsqueeze_1", unsqueezeGradOutputInfo, GetNodeType()),
        OP_LOGE(GetNodeType().GetString(), "Create unsqueeze grad_output node failed"), return nullptr);

    GNode conv3dBpFilterNode;
    TensorDesc output3DDesc;
    OP_CHECK_IF(!BuildConv3DBackpropFilterNode(builder, iFilterSize, unsqueezeXInfo, unsqueezeGradOutputInfo,
                                               nodeNamePrefix, conv3dBpFilterNode, output3DDesc),
                OP_LOGE(GetNodeType().GetString(), "Build Conv3DBackpropFilter node failed"), return nullptr);

    GNode squeezeNode;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::BuildSqueezeNode(builder, conv3dBpFilterNode, output3DDesc, outputDesc,
                                                               nodeNamePrefix, squeezeNode, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Build Squeeze node failed"), return nullptr);

    auto squeezeOutput = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(squeezeNode, OUTPUT_INDEX));
    std::vector<EsTensorHolder> outputs = {squeezeOutput};
    OP_LOGD(GetNodeType().GetString(), "Conv2DBackpropFilter trans to Conv3DBackpropFilter fusion success!");
    return builder.BuildAndReset(outputs);
}

REG_DECOMPOSE_PASS(Conv2DBackpropFilterToV3FusionPass, {CONV2D_BACKPROP_FILTER})
    .Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
