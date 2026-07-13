/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv2d_backprop_input_to_v2_fusion_pass.h"

namespace ops {
namespace {
const AscendString PASS_NAME = "Conv2DBackpropInputToV2FusionPass";
const AscendString CONV2D_BACKPROP_INPUT = "Conv2DBackpropInput";

constexpr size_t INPUT_SIZE_INDEX = 0;
constexpr size_t FILTER_INDEX = 1;
constexpr size_t GRAD_OUTPUT_INDEX = 2;
constexpr size_t CONV2D_BP_INPUT_INPUT_NUM = 3;

} // namespace

AscendString Conv2DBackpropInputToV2FusionPass::GetNodeType() const { return PASS_NAME; }

bool Conv2DBackpropInputToV2FusionPass::GetNodeAttrs(const ge::GNode& node)
{
    OP_CHECK_IF(!ConvBackpropFusionBasePass::GetNodeAttrs(node),
                OP_LOGE(GetNodeType().GetString(), "Base GetNodeAttrs failed"), return false);

    ConvBackpropFusionUtilsPass::ExpandAttrs(convBpAttr.strides, convBpAttr.pads, convBpAttr.dilations,
                                             convBpAttr.dataFormat);

    return true;
}

bool Conv2DBackpropInputToV2FusionPass::BuildConv3DBackpropInputNode(
    EsGraphBuilder& builder, const GNode& convBpInputNode, EsTensorHolder& iInputSize,
    UnsqueezeNodeInfo& unsqueezeFilterInfo, UnsqueezeNodeInfo& unsqueezeGradOutputInfo,
    const std::string& nodeNamePrefix, GNode& outNode, TensorDesc& output3DDesc)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(graph == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Get graph failed in BuildConv3DBackpropInputNode"), return false);

    ConvBackpropFusionUtilsPass::ExpandOutputDesc(outputDesc, output3DDesc);

    std::string conv3dBpInputName = nodeNamePrefix + "_to_Conv3DBackpropInput";
    outNode = CompliantNodeBuilder(graph)
                  .OpType("Conv3DBackpropInput")
                  .Name(conv3dBpInputName.c_str())
                  .IrDefInputs({{"input_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                  .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                  .Build();

    auto* inputSizeProducer = iInputSize.GetProducer();
    OP_CHECK_IF(inputSizeProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "Input_size producer is nullptr"),
                return false);
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *inputSizeProducer, iInputSize.GetProducerOutIndex(), outNode,
                                         static_cast<int32_t>(INPUT_SIZE_INDEX)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for input_size failed"), return false);

    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, unsqueezeFilterInfo.node, 0, outNode,
                                         static_cast<int32_t>(FILTER_INDEX)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for unsqueeze filter failed"), return false);

    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, unsqueezeGradOutputInfo.node, 0, outNode,
                                         static_cast<int32_t>(GRAD_OUTPUT_INDEX)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for unsqueeze grad_output failed"), return false);

    SetConv3DBackpropInputAttrsAndDescs(outNode, unsqueezeFilterInfo.outDesc, unsqueezeGradOutputInfo.outDesc,
                                        output3DDesc);

    return true;
}

void Conv2DBackpropInputToV2FusionPass::SetConv3DBackpropInputAttrsAndDescs(
    GNode& outNode, const TensorDesc& unsqueezeFilterOutDesc, const TensorDesc& unsqueezeGradOutputOutDesc,
    const TensorDesc& output3DDesc)
{
    SetNodeAttrs(outNode);

    outNode.UpdateInputDesc(INPUT_SIZE_INDEX, input0Desc);
    outNode.UpdateInputDesc(FILTER_INDEX, unsqueezeFilterOutDesc);
    outNode.UpdateInputDesc(GRAD_OUTPUT_INDEX, unsqueezeGradOutputOutDesc);
    outNode.UpdateOutputDesc(OUTPUT_INDEX, output3DDesc);
}

GraphUniqPtr Conv2DBackpropInputToV2FusionPass::Replacement(const GNode& convBpInputNode)
{
    OP_LOGD(GetNodeType().GetString(), "Replacement start");

    OP_CHECK_IF(!GetNodeDesc(convBpInputNode), OP_LOGE(GetNodeType().GetString(), "GetNodeDesc failed"),
                return nullptr);

    OP_CHECK_IF(!GetNodeAttrs(convBpInputNode), OP_LOGE(GetNodeType().GetString(), "GetNodeAttrs failed"),
                return nullptr);

    auto builder = EsGraphBuilder("replacement");
    auto [iInputSize, iFilter, iGradOutput] = builder.CreateInputs<CONV2D_BP_INPUT_INPUT_NUM>();

    std::string nodeNamePrefix;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::GetNodeName(convBpInputNode, nodeNamePrefix),
                OP_LOGE(GetNodeType().GetString(), "Get node name failed"), return nullptr);

    UnsqueezeNodeInfo unsqueezeFilterInfo;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateUnsqueezeNode(
                    builder, iFilter, input1Desc, nodeNamePrefix + "_Unsqueeze_0", unsqueezeFilterInfo, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Create unsqueeze filter node failed"), return nullptr);

    UnsqueezeNodeInfo unsqueezeGradOutputInfo;
    OP_CHECK_IF(
        !ConvBackpropFusionUtilsPass::CreateUnsqueezeNode(
            builder, iGradOutput, input2Desc, nodeNamePrefix + "_Unsqueeze_1", unsqueezeGradOutputInfo, GetNodeType()),
        OP_LOGE(GetNodeType().GetString(), "Create unsqueeze grad_output node failed"), return nullptr);

    GNode conv3dBpInputNode;
    TensorDesc output3DDesc;
    OP_CHECK_IF(!BuildConv3DBackpropInputNode(builder, convBpInputNode, iInputSize, unsqueezeFilterInfo,
                                              unsqueezeGradOutputInfo, nodeNamePrefix, conv3dBpInputNode, output3DDesc),
                OP_LOGE(GetNodeType().GetString(), "Build Conv3DBackpropInput node failed"), return nullptr);

    GNode squeezeNode;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::BuildSqueezeNode(builder, conv3dBpInputNode, output3DDesc, outputDesc,
                                                               nodeNamePrefix, squeezeNode, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Build Squeeze node failed"), return nullptr);

    auto squeezeOutput = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(squeezeNode, OUTPUT_INDEX));
    std::vector<EsTensorHolder> outputs = {squeezeOutput};
    OP_LOGD(GetNodeType().GetString(), "Conv2DBackpropInput trans to Conv3DBackpropInput fusion success!");
    return builder.BuildAndReset(outputs);
}

REG_DECOMPOSE_PASS(Conv2DBackpropInputToV2FusionPass, {CONV2D_BACKPROP_INPUT})
    .Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
