/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv2d_transpose_to_v2_fusion_pass.h"

namespace ops {
namespace {
const AscendString PASS_NAME = "Conv2DTransposeToV2FusionPass";
const AscendString CONV2D_TRANSPOSE = "Conv2DTranspose";

constexpr size_t INPUT_SIZE_INDEX = 0;
constexpr size_t X_INDEX = 1;
constexpr size_t FILTER_INDEX = 2;
constexpr size_t BIAS_INDEX = 3;
constexpr size_t OFFSET_W_INDEX = 4;
constexpr size_t CONV2D_TRANSPOSE_INPUT_NUM = 3;

constexpr int64_t DEFAULT_OFFSET_X = 0;

} // namespace

AscendString Conv2DTransposeToV2FusionPass::GetNodeType() const { return PASS_NAME; }

bool Conv2DTransposeToV2FusionPass::IsInputInt8(const GNode& matchedNode)
{
    TensorDesc xDesc;
    OP_CHECK_IF(matchedNode.GetInputDesc(X_INDEX, xDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Failed to get x desc."), return false);
    TensorDesc filterDesc;
    OP_CHECK_IF(matchedNode.GetInputDesc(FILTER_INDEX, filterDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Failed to get filter desc."), return false);
    OP_CHECK_IF(xDesc.GetDataType() == DT_INT8 || filterDesc.GetDataType() == DT_INT8,
                OP_LOGD(GetNodeType().GetString(), "Input dtype is INT8, skip fusion."), return true);
    return false;
}

bool Conv2DTransposeToV2FusionPass::MeetRequirements(const GNode& matchedNode)
{
    OP_LOGD(GetNodeType().GetString(), "Enter MeetRequirements");

    OP_CHECK_IF(!ConvBackpropFusionBasePass::CheckSocAndIntrinsic(),
                OP_LOGD(GetNodeType().GetString(), "SOC check failed."), return false);

    OP_CHECK_IF(IsInputInt8(matchedNode), OP_LOGD(GetNodeType().GetString(), "INT8 dtype detected, skip fusion."),
                return false);

    return true;
}

bool Conv2DTransposeToV2FusionPass::GetNodeDesc(const GNode& node)
{
    OP_CHECK_IF(!ConvBackpropFusionBasePass::GetNodeDesc(node),
                OP_LOGE(GetNodeType().GetString(), "Base GetNodeDesc failed"), return false);

    biasDesc = TensorDesc();
    offsetWDesc = TensorDesc();

    if (node.GetInputsSize() > BIAS_INDEX) {
        node.GetInputDesc(BIAS_INDEX, biasDesc);
    }
    if (node.GetInputsSize() > OFFSET_W_INDEX) {
        node.GetInputDesc(OFFSET_W_INDEX, offsetWDesc);
    }

    return true;
}

bool Conv2DTransposeToV2FusionPass::GetNodeAttrs(const GNode& node)
{
    OP_CHECK_IF(!ConvBackpropFusionBasePass::GetNodeAttrs(node),
                OP_LOGE(GetNodeType().GetString(), "Base GetNodeAttrs failed"), return false);
    outputPadding = {0, 0, 0, 0};
    node.GetAttr("output_padding", outputPadding);
    offsetX = DEFAULT_OFFSET_X;
    node.GetAttr("offset_x", offsetX);

    ConvBackpropFusionUtilsPass::ExpandAttrs(convBpAttr.strides, convBpAttr.pads, convBpAttr.dilations,
                                             convBpAttr.dataFormat, &outputPadding);

    return true;
}

bool Conv2DTransposeToV2FusionPass::ConnectOptionalInput(EsGraphBuilder& builder, size_t nodeInputsSize,
                                                         size_t inputIndex, const TensorDesc& inputDesc, GNode& outNode)
{
    if (nodeInputsSize <= inputIndex) {
        OP_LOGD(GetNodeType().GetString(), "nodeInputsSize %d <= inputIndex %d, skip.", nodeInputsSize, inputIndex);
        return true;
    }

    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(graph == nullptr, OP_LOGE(GetNodeType().GetString(), "Get graph failed in ConnectOptionalInput"),
                return false);
    auto rInput = builder.CreateInput(static_cast<int64_t>(inputIndex));
    auto* producer = rInput.GetProducer();
    OP_CHECK_IF(producer == nullptr, OP_LOGE(GetNodeType().GetString(), "Optional input producer is nullptr"),
                return false);
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *producer, rInput.GetProducerOutIndex(), outNode,
                                         static_cast<int32_t>(inputIndex)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for optional input failed"), return false);
    outNode.UpdateInputDesc(inputIndex, inputDesc);

    return true;
}

void Conv2DTransposeToV2FusionPass::SetConv3DTransposeAttrsAndDescs(GNode& outNode, const TensorDesc& unsqueezeXOutDesc,
                                                                    const TensorDesc& unsqueezeFilterOutDesc,
                                                                    const TensorDesc& output3DDesc)
{
    SetNodeAttrs(outNode);
    outNode.SetAttr("output_padding", outputPadding);
    outNode.SetAttr("offset_x", offsetX);

    outNode.UpdateInputDesc(INPUT_SIZE_INDEX, input0Desc);
    outNode.UpdateInputDesc(X_INDEX, unsqueezeXOutDesc);
    outNode.UpdateInputDesc(FILTER_INDEX, unsqueezeFilterOutDesc);
    outNode.UpdateOutputDesc(OUTPUT_INDEX, output3DDesc);
}

bool Conv2DTransposeToV2FusionPass::BuildConv3DTransposeNode(EsGraphBuilder& builder, const GNode& convTransposeNode,
                                                             EsTensorHolder& iInputSize,
                                                             UnsqueezeNodeInfo& unsqueezeXInfo,
                                                             UnsqueezeNodeInfo& unsqueezeFilterInfo,
                                                             const std::string& nodeNamePrefix, GNode& outNode,
                                                             TensorDesc& output3DDesc)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(graph == nullptr, OP_LOGE(GetNodeType().GetString(), "Get graph failed in BuildConv3DTransposeNode"),
                return false);

    ConvBackpropFusionUtilsPass::ExpandOutputDesc(outputDesc, output3DDesc);

    size_t nodeInputsSize = convTransposeNode.GetInputsSize();
    std::string conv3dTransposeName = nodeNamePrefix + "_to_Conv3DTranspose";

    outNode = CompliantNodeBuilder(graph)
                  .OpType("Conv3DTranspose")
                  .Name(conv3dTransposeName.c_str())
                  .IrDefInputs({{"input_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
                                {"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""}})
                  .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                  .Build();

    auto* inputSizeProducer = iInputSize.GetProducer();
    OP_CHECK_IF(inputSizeProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "Input_size producer is nullptr"),
                return false);
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *inputSizeProducer, iInputSize.GetProducerOutIndex(), outNode,
                                         static_cast<int32_t>(INPUT_SIZE_INDEX)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for input_size failed"), return false);

    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, unsqueezeXInfo.node, 0, outNode, static_cast<int32_t>(X_INDEX)) !=
                    GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for unsqueeze x failed"), return false);

    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, unsqueezeFilterInfo.node, 0, outNode,
                                         static_cast<int32_t>(FILTER_INDEX)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge for unsqueeze filter failed"), return false);

    OP_CHECK_IF(!ConnectOptionalInput(builder, nodeInputsSize, BIAS_INDEX, biasDesc, outNode),
                OP_LOGE(GetNodeType().GetString(), "Connect bias input failed"), return false);
    OP_CHECK_IF(!ConnectOptionalInput(builder, nodeInputsSize, OFFSET_W_INDEX, offsetWDesc, outNode),
                OP_LOGE(GetNodeType().GetString(), "Connect offset_w input failed"), return false);

    SetConv3DTransposeAttrsAndDescs(outNode, unsqueezeXInfo.outDesc, unsqueezeFilterInfo.outDesc, output3DDesc);

    return true;
}

GraphUniqPtr Conv2DTransposeToV2FusionPass::Replacement(const GNode& convTransposeNode)
{
    OP_LOGD(GetNodeType().GetString(), "Replacement start");

    OP_CHECK_IF(!GetNodeDesc(convTransposeNode), OP_LOGE(GetNodeType().GetString(), "GetNodeDesc failed"),
                return nullptr);

    OP_CHECK_IF(!GetNodeAttrs(convTransposeNode), OP_LOGE(GetNodeType().GetString(), "GetNodeAttrs failed"),
                return nullptr);

    auto builder = EsGraphBuilder("replacement");
    auto [iInputSize, iX, iFilter] = builder.CreateInputs<CONV2D_TRANSPOSE_INPUT_NUM>();

    std::string nodeNamePrefix;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::GetNodeName(convTransposeNode, nodeNamePrefix),
                OP_LOGE(GetNodeType().GetString(), "Get node name failed"), return nullptr);

    UnsqueezeNodeInfo unsqueezeXInfo;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateUnsqueezeNode(
                    builder, iX, input1Desc, nodeNamePrefix + "_Unsqueeze_0", unsqueezeXInfo, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Create unsqueeze x node failed"), return nullptr);

    UnsqueezeNodeInfo unsqueezeFilterInfo;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateUnsqueezeNode(
                    builder, iFilter, input2Desc, nodeNamePrefix + "_Unsqueeze_1", unsqueezeFilterInfo, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Create unsqueeze filter node failed"), return nullptr);

    GNode conv3dTransposeNode;
    TensorDesc output3DDesc;
    OP_CHECK_IF(!BuildConv3DTransposeNode(builder, convTransposeNode, iInputSize, unsqueezeXInfo, unsqueezeFilterInfo,
                                          nodeNamePrefix, conv3dTransposeNode, output3DDesc),
                OP_LOGE(GetNodeType().GetString(), "Build Conv3DTranspose node failed"), return nullptr);

    GNode squeezeNode;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::BuildSqueezeNode(builder, conv3dTransposeNode, output3DDesc, outputDesc,
                                                               nodeNamePrefix, squeezeNode, GetNodeType()),
                OP_LOGE(GetNodeType().GetString(), "Build Squeeze node failed"), return nullptr);

    auto squeezeOutput = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(squeezeNode, OUTPUT_INDEX));
    std::vector<EsTensorHolder> outputs = {squeezeOutput};
    OP_LOGD(GetNodeType().GetString(), "Conv2DTranspose trans to Conv3DTranspose fusion success!");
    return builder.BuildAndReset(outputs);
}

REG_DECOMPOSE_PASS(Conv2DTransposeToV2FusionPass, {CONV2D_TRANSPOSE}).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
