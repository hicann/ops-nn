/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv3d_dequant_to_quantconv3d_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "es_nn_ops.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace Conv3DDequantToQuantconv3DFusion;
using namespace ge;
using namespace fusion;

std::unique_ptr<SubgraphBoundary> Conv3DDequantToQuantConv3DFusionPass::ConstructBoundary(const GNode &convNode)
{
    FUSION_PASS_CHECK_NOLOG(!GetPostCubeNodes(convNode), return nullptr);

    std::unique_ptr<SubgraphBoundary> boundary = std::make_unique<SubgraphBoundary>();
    for (size_t index = 0; index < REQUIRED_INPUT_NUMS; ++index) {
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, convNode, static_cast<int64_t>(index),
            static_cast<int64_t>(index)), return nullptr);
    }

    if (convDescInfo.hasBias) {
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, convNode,
            INPUT_BIAS_INDEX, QUANTCONV3D_CONV3D_INPUT_BIAS_INDEX), return nullptr);
    }

    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, *postCubeNode,
        POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, QUANTCONV3D_CONV3D_INPUT_SCALE_INDEX), return nullptr);

    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphOutput(boundary, *postCubeNode,
        static_cast<int64_t>(OUTPUT_INDEX), static_cast<int64_t>(OUTPUT_INDEX)), return nullptr);

    return boundary;
}

bool Conv3DDequantToQuantConv3DFusionPass::PostCubeFusionImpl(GraphPtr& graph, GNode &convNode,
    const CustomPassContext &pass_context)
{
    GNodePtr nodePtr = ConvFusionUtilsPass::GetNodePtr(convNode, convDescInfo);
    FUSION_PASS_CHECK_NOLOG(nodePtr == nullptr, return false);

    ops::PostCubeUtils postCubeUtils;
    // [Step 1] Determine which nodes in the subsequent nodes satisfy PostCube hardware unit
    FUSION_PASS_CHECK(postCubeUtils.GetPostCubeNodeList(nodePtr, pass_context) != GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "GetPostCubeNodeList failed, no fusion."), return false);

    // [Step 2] To customize the selection of the PostCube fusion range
    SelectPostCubePassByWhiteList(postCubeUtils.m_matchpasses_);

    // [Step 3] PostCube tool method selects 1 PostCube paths
    FUSION_PASS_CHECK(postCubeUtils.SelectPostCubeNodeList(false) != GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "SelectPostCubeNodeList failed, no fusion."), return false);

    // [Step 4] Create the PostCube operator node and modify the graph
    std::vector<GNodePtr> newNodes;
    FUSION_PASS_CHECK(postCubeUtils.CreatePostCubeNode(convDescInfo.nodeNameStr, *graph, newNodes) != GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "CreatePostCubeNode failed, no fusion."), return false);

    return true;
}

void Conv3DDequantToQuantConv3DFusionPass::InitMember()
{
    postCubeNode = nullptr;
}

bool Conv3DDequantToQuantConv3DFusionPass::MeetRequirements(const GNode &convNode)
{
    // Check cur node's dtypes whether it is supported.
    std::vector<DataType> convDtypes = {convDescInfo.fmapDtype, convDescInfo.filterDtype, convDescInfo.outputDtype};
    if (convDescInfo.hasBias) {
        convDtypes.emplace_back(convDescInfo.biasDtype);
    }
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSupportList<DataType>(CONV_SUPPORT_DTYPES, convDtypes),
        OP_LOGD(convDescInfo.nodeNameStr, "Conv3D dtype not supported, no fusion."), return false);

    // Check cur node's formats whether it is supported.
    std::vector<Format> convFormats = {convDescInfo.fmapFormat, convDescInfo.filterFormat, convDescInfo.outputFormat};
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSupportList<Format>(CONV_SUPPORT_FORMATS_DAV_3510, convFormats),
        OP_LOGD(convDescInfo.nodeNameStr, "Conv3D format not supported, no fusion."), return false);

    auto outputNodes = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);

    FUSION_PASS_CHECK(outputNodes.size() != SINGLE_OUTPUTNUM,
        OP_LOGD(convDescInfo.nodeNameStr, "Only support conv3d single output."), return false);

    FUSION_PASS_CHECK(outputNodes[0].first == nullptr,
        OP_LOGD(convDescInfo.nodeNameStr, "Conv3d node has no output nodes."), return false);

    AscendString nodeType;
    FUSION_PASS_CHECK(outputNodes[0].first->GetType(nodeType) != ge::GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "Failed to get node type."), return false);

    FUSION_PASS_CHECK(nodeType != ConvFusionUtils::ASCEND_DEQUANT,
        OP_LOGD(convDescInfo.nodeNameStr, "Node type is [%s], only support dequant", nodeType.GetString()),
        return false);

    bool sqrtMode = false;
    FUSION_PASS_CHECK(outputNodes[0].first->GetAttr(SQRT_MODE, sqrtMode) != ge::GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "Get sqrt_mode failed."), return false);
    FUSION_PASS_CHECK(sqrtMode, OP_LOGD(convDescInfo.nodeNameStr, "Only support sqrt_mode is false."), return false);

    bool reluFlag = false;
    FUSION_PASS_CHECK(outputNodes[0].first->GetAttr(RELU_FLAG, reluFlag) != ge::GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "Get relu_flag failed."), return false);
    FUSION_PASS_CHECK(reluFlag, OP_LOGD(convDescInfo.nodeNameStr, "Only support relu_flag is false."), return false);

    return true;
}

AscendString Conv3DDequantToQuantConv3DFusionPass::GetNodeType() const
{
    return ConvFusionUtils::CONV3D;
}

std::map<std::string, NpuArch> Conv3DDequantToQuantConv3DFusionPass::GetSocSupportList() const
{
    return SUPPORT_SOC_LIST;
}

GraphUniqPtr Conv3DDequantToQuantConv3DFusionPass::Replacement(const GNode &convNode)
{
    auto replacement_graph_builder = es::EsGraphBuilder("replacement");

    ConvBaseAttrs baseAttrs;
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvBaseAttr(convNode, baseAttrs, convDescInfo), return nullptr);

    TensorDesc postCubeOutDesc;
    FUSION_PASS_CHECK(postCubeNode->GetOutputDesc(OUTPUT_INDEX, postCubeOutDesc) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Get fxipipe out tensor desc failed."), return nullptr);
    auto outDtype = postCubeOutDesc.GetDataType();

    auto [fmap, filter] = replacement_graph_builder.CreateInputs<REQUIRED_INPUT_NUMS>();
    auto scale = replacement_graph_builder.CreateInput(static_cast<int64_t>(QUANTCONV3D_CONV3D_INPUT_SCALE_INDEX));
    auto bias = convDescInfo.hasBias ?
        replacement_graph_builder.CreateInput(static_cast<int64_t>(QUANTCONV3D_CONV3D_INPUT_BIAS_INDEX)) : nullptr;

    auto quantConv3D = es::QuantConv3D(fmap, filter, scale, bias, nullptr,
        outDtype, baseAttrs.strides, baseAttrs.pads, baseAttrs.dilations, baseAttrs.groups,
        baseAttrs.dataFormat.GetString(), baseAttrs.offsetX, RINT.GetString(), baseAttrs.padMode.GetString());

    auto quantConv3DNode = quantConv3D.GetProducer();
    FUSION_PASS_CHECK_NOLOG(!UpdateQuantConv3DDesc(quantConv3DNode, postCubeOutDesc), return nullptr);

    std::vector<es::EsTensorHolder> replaceOutput = {quantConv3D};

    return replacement_graph_builder.BuildAndReset(replaceOutput);
}

bool Conv3DDequantToQuantConv3DFusionPass::GetPostCubeNodes(const GNode &convNode)
{
    auto outputNodes = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);

    FUSION_PASS_CHECK(outputNodes.size() != SINGLE_OUTPUTNUM,
        OP_LOGE(convDescInfo.nodeNameStr, "After create Fixpipe, Conv3D output num is not one."), return false);

    postCubeNode = outputNodes[0].first;
    FUSION_PASS_CHECK(postCubeNode == nullptr,
        OP_LOGE(convDescInfo.nodeNameStr, "After create PostCube, Conv3D out nodes is nullptr."), return false);

    AscendString nodeType;
    FUSION_PASS_CHECK(postCubeNode->GetType(nodeType) != ge::GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "After create PostCube, cannot get output node name."), return false);

    FUSION_PASS_CHECK(nodeType != POST_CUBE_OP,
        OP_LOGE(convDescInfo.nodeNameStr,
                "After create PostCube, node type is [%s], not PostCube", nodeType.GetString()), return false);

    return true;
}

void Conv3DDequantToQuantConv3DFusionPass::SelectPostCubePassByWhiteList(std::vector<ops::PostCubePassInfo> &matchVec)
{
    std::vector<ops::PostCubePassInfo> tmpPasses(matchVec);
    matchVec.clear();
    for (auto &tmpPass : tmpPasses) {
        if (tmpPass.m_opnodes.size() < FUSION_LIST_LENGTH) {
            continue;
        }

        AscendString nodeType;
        auto& conv3dNode = tmpPass.m_opnodes[CONV3D_INDEX];
        FUSION_PASS_CHECK(conv3dNode.GetNode()->GetType(nodeType) != ge::GRAPH_SUCCESS,
            OP_LOGD(convDescInfo.nodeNameStr, "Get conv3d node type failed."), return);
        if (nodeType != CONV3D) {
            continue;
        }

        auto& dequantNode = tmpPass.m_opnodes[ASCEND_DEQUANT_INDEX];
        FUSION_PASS_CHECK(dequantNode.GetNode()->GetType(nodeType) != ge::GRAPH_SUCCESS,
            OP_LOGD(convDescInfo.nodeNameStr, "Get dequant node type failed."), return);
        if (nodeType != ConvFusionUtils::ASCEND_DEQUANT) {
            continue;
        }

        matchVec.push_back(tmpPass);
        return;
    }
}

bool Conv3DDequantToQuantConv3DFusionPass::UpdateQuantConv3DDesc(GNode *quantConv3D, TensorDesc &postCubeOutDesc)
{
    if (convDescInfo.hasBias) {
        FUSION_PASS_CHECK(
            quantConv3D->UpdateInputDesc(QUANTCONV3D_CONV3D_INPUT_BIAS_INDEX, convDescInfo.biasDesc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Update bias tensor desc failed."), return false);
    }
    convDescInfo.hasBias = false;
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::UpdateInputDesc(quantConv3D, convDescInfo), return false);

    FUSION_PASS_CHECK(quantConv3D->UpdateOutputDesc(OUTPUT_INDEX, postCubeOutDesc) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Update QuantConv3D out tensor desc failed."), return false);

    TensorDesc scaleTensorDesc;
    FUSION_PASS_CHECK(postCubeNode->GetInputDesc(POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, scaleTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Get PostCube scale tensor desc failed."), return false);
    scaleTensorDesc.SetFormat(ge::FORMAT_ND);
    scaleTensorDesc.SetOriginFormat(ge::FORMAT_ND);
    FUSION_PASS_CHECK(
        quantConv3D->UpdateInputDesc(QUANTCONV3D_CONV3D_INPUT_SCALE_INDEX, scaleTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Update quant scale tensor desc failed."), return false);

    return true;
}

} // namespace ops