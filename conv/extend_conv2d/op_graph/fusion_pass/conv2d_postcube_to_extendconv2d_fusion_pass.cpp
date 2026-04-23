/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv2d_postcube_to_extendconv2d_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "es_nn_ops.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace Conv2DPostCubeToExtendConv2DFusion;
using namespace ge;
using namespace fusion;

std::unique_ptr<SubgraphBoundary> Conv2DPostCubeToExtendConv2DFusionPass::ConstructBoundary(const GNode &convNode)
{
    FUSION_PASS_CHECK_NOLOG(!GetPostCubeNodes(convNode), return nullptr);
    FUSION_PASS_CHECK_NOLOG(!CheckDescInfo(), return nullptr);

    std::unique_ptr<SubgraphBoundary> boundary = std::make_unique<SubgraphBoundary>();
    for (size_t index = 0; index < convNode.GetInputsSize(); ++index) {
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, convNode, static_cast<int64_t>(index),
            static_cast<int64_t>(index)), return nullptr);
    }

    FUSION_PASS_CHECK_NOLOG(!AddScaleReluToBoundAry(boundary), return nullptr);

    if (outputCase == OutputCase::DUAL_POST_CUBE || outputCase == OutputCase::SINGLE) {
        for (size_t index = 0; index < postCubeNodes.size(); index++) {
            GNodePtr postCubeNode = postCubeNodes[index];
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphOutput(boundary, *postCubeNode,
                OUTPUT_0_INDEX, static_cast<int64_t>(index)), return nullptr);
        }
    } else if (outputCase == OutputCase::OTHER_POST_CUBE || outputCase == OutputCase::POST_CUBE_OTHER) {
        // Add PostCube node to boundary
        int64_t postCubeIndex = outputCase != OutputCase::OTHER_POST_CUBE ? OUTPUT_0_INDEX : OUTPUT_1_INDEX;
        int64_t otherIndex = outputCase != OutputCase::OTHER_POST_CUBE ? OUTPUT_1_INDEX : OUTPUT_0_INDEX;

        GNodePtr postCubeNode = postCubeNodes[0];
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphOutput(boundary, *postCubeNode,
            static_cast<int64_t>(OUTPUT_INDEX), postCubeIndex), return nullptr);
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphOutput(boundary, convNode,
            static_cast<int64_t>(OUTPUT_INDEX), otherIndex), return nullptr);
    }

    return boundary;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::PostCubeFusionImpl(GraphPtr &graph, GNode &convNode,
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

    // [Step 3] PostCube tool method selects 1~2 PostCube paths
    FUSION_PASS_CHECK(postCubeUtils.SelectPostCubeNodeList(true) != GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "SelectPostCubeNodeList failed, no fusion."), return false);

    // [Step 4] Create the PostCube operator node and modify the graph
    std::vector<GNodePtr> newNodes;
    FUSION_PASS_CHECK(postCubeUtils.CreatePostCubeNode(convDescInfo.nodeNameStr, *graph, newNodes) != GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "CreatePostCubeNode failed, no fusion."), return false);

    return true;
}

void Conv2DPostCubeToExtendConv2DFusionPass::InitMember()
{
    postCubeFusionOps.clear();
    postCubeNodes.clear();
    otherNodes.clear();
    outputCase = OutputCase::SINGLE;
    hasScale0 = false;
    hasScale1 = false;
    hasRelu0 = false;
    hasRelu1 = false;
    graphIndex = REQUIRED_INPUT_NUMS;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::MeetRequirements(const GNode &convNode)
{
    FUSION_PASS_CHECK(convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX).empty(),
        OP_LOGD(convDescInfo.nodeNameStr, "Conv2D out nodes is empty, don't need fusion process."), return false);

    // Check cur node's dtypes whether it is supported.
    std::vector<DataType> convDtypes = {convDescInfo.fmapDtype, convDescInfo.filterDtype, convDescInfo.outputDtype};
    if (convDescInfo.hasBias) {
        convDtypes.emplace_back(convDescInfo.biasDtype);
    }
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSupportList<DataType>(CONV_SUPPORT_DTYPES, convDtypes),
        OP_LOGD(convDescInfo.nodeNameStr, "Conv2D dtype not supported, no fusion."), return false);

    // Check cur node's formats whether it is supported.
    std::vector<Format> convFormats = {convDescInfo.fmapFormat, convDescInfo.filterFormat, convDescInfo.outputFormat};
    auto convSupportFormats = npuArch == NpuArch::DAV_5102 ?
        CONV_SUPPORT_FORMATS_DAV_5102 : CONV_SUPPORT_FORMATS_DAV_3510;
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSupportList<Format>(convSupportFormats, convFormats),
        OP_LOGD(convDescInfo.nodeNameStr, "Conv2D format not supported, no fusion."), return false);

    return true;
}

AscendString Conv2DPostCubeToExtendConv2DFusionPass::GetNodeType() const
{
    return ConvFusionUtils::CONV2D;
}

std::map<std::string, NpuArch> Conv2DPostCubeToExtendConv2DFusionPass::GetSocSupportList() const
{
    return SUPPORT_SOC_LIST;
}

void Conv2DPostCubeToExtendConv2DFusionPass::PrintGraphStructure() const
{
    if (postCubeFusionOps.empty()) {
        return;
    }

    auto fusionList0 = ConvFusionUtilsPass::ListToAscendString(postCubeFusionOps.front());
    auto fusionList1 = ConvFusionUtilsPass::ListToAscendString(postCubeFusionOps.back());

    std::stringstream logStr;
    logStr << "graph structure: cube node name: [ " << convDescInfo.nodeNameStr << " ], ";
    if (outputCase == OutputCase::SINGLE) {
        logStr << "y0" << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList0.GetString() << "].";
    } else if (outputCase == OutputCase::DUAL_POST_CUBE) {
        logStr << "y0" << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList0.GetString() << "], ";
        logStr << "y1" << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList1.GetString() << "].";
    } else if (outputCase == OutputCase::POST_CUBE_OTHER) {
        logStr << "y0" << " output: [" << convDescInfo.nodeNameStr << "], ";
        logStr << "y1" << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList1.GetString() << "].";
    } else if (outputCase == OutputCase::OTHER_POST_CUBE) {
        logStr << "y0" << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList0.GetString() << "], ";
        logStr << "y1" << " output: [" << convDescInfo.nodeNameStr << "].";
    }

    OP_LOGI(convDescInfo.nodeNameStr, "%s", logStr.str().c_str());
}

GraphUniqPtr Conv2DPostCubeToExtendConv2DFusionPass::Replacement(const GNode &convNode)
{
    auto replace_graph_builder = es::EsGraphBuilder("replacement");

    ConvBaseAttrs baseAttrs;
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvBaseAttr(convNode, baseAttrs, convDescInfo), return nullptr);

    auto [fmap, filter] = replace_graph_builder.CreateInputs<REQUIRED_INPUT_NUMS>();
    auto bias = convDescInfo.hasBias ?
        replace_graph_builder.CreateInput(graphIndex++) : nullptr;
    auto scale0 = hasScale0 ?
        replace_graph_builder.CreateInput(graphIndex++) : nullptr;
    auto relu0 = hasRelu0 ?
        replace_graph_builder.CreateInput(graphIndex++) : nullptr;
    auto scale1 = hasScale1 ?
        replace_graph_builder.CreateInput(graphIndex++) : nullptr;
    auto relu1 = hasRelu1 ?
        replace_graph_builder.CreateInput(graphIndex++) : nullptr;

    bool enableRelu0 = outputCase != OutputCase::OTHER_POST_CUBE &&
        !postCubeFusionOps.empty() && IsReluEnable(postCubeFusionOps.front());
    bool enableRelu1 = (outputCase == OutputCase::DUAL_POST_CUBE || outputCase == OutputCase::OTHER_POST_CUBE) &&
        !postCubeFusionOps.empty() && IsReluEnable(postCubeFusionOps.back());
    bool dualOutput = outputCase != OutputCase::SINGLE;

    // fmap, filter, bias, offset_w, scale0, relu_weight0, clip_value0, scale1, relu_weight1, clip_value1
    auto extendConv2D = es::ExtendConv2D(fmap, filter, bias, nullptr, scale0, relu0, nullptr, scale1, relu1, nullptr,
        baseAttrs.strides, baseAttrs.pads, baseAttrs.dilations, baseAttrs.groups, baseAttrs.dataFormat.GetString(),
        baseAttrs.offsetX, RINT.GetString(), baseAttrs.padMode.GetString(), baseAttrs.enableHf32,
        enableRelu0, enableRelu1, dualOutput);

    // Multi TensorHolder get from es::Op, it's GetProducer() point to one node.
    auto output0 = extendConv2D.y0.GetProducer();
    FUSION_PASS_CHECK_NOLOG(!UpdateExtendConv2DDesc(output0), return nullptr);

    std::vector<es::EsTensorHolder> replaceOutput = {extendConv2D.y0};
    if (dualOutput) {
        replaceOutput.emplace_back(extendConv2D.y1);
    }
    return replace_graph_builder.BuildAndReset(replaceOutput);
}

bool Conv2DPostCubeToExtendConv2DFusionPass::AddScaleReluToBoundAry(std::unique_ptr<SubgraphBoundary> &boundary)
{
    // Add quant_scale_0 and relu_weight_0 to boundary
    if (outputCase != OutputCase::OTHER_POST_CUBE) {
        if (IsScaleEnable(postCubeFusionOps.front())) {
            GNodePtr postCubeNode = postCubeNodes.front();
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, *postCubeNode,
                POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, EXTENDCONV2D_QUANT_SCALE_0_INDEX), return false);
            hasScale0 = true;
        } else if (IsReluEnable(postCubeFusionOps.front(), Conv2DPostCubeToExtendConv2DFusion::LEAKY_RELU)) {
            GNodePtr postCubeNode = postCubeNodes.front();
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, *postCubeNode,
                POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX, EXTENDCONV2D_RELU_WEIGHT_0_INDEX), return false);
            hasRelu0 = true;
        }
    }

    // Add quant_scale_1 and relu_weight_1 to boundary
    if (outputCase == OutputCase::OTHER_POST_CUBE || outputCase == OutputCase::DUAL_POST_CUBE) {
        if (IsScaleEnable(postCubeFusionOps.back())) {
            GNodePtr postCubeNode = postCubeNodes.back();
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, *postCubeNode,
                POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, EXTENDCONV2D_QUANT_SCALE_1_INDEX), return false);
            hasScale1 = true;
        } else if (IsReluEnable(postCubeFusionOps.back(), Conv2DPostCubeToExtendConv2DFusion::LEAKY_RELU)) {
            GNodePtr postCubeNode = postCubeNodes.back();
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, *postCubeNode,
                POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX, EXTENDCONV2D_RELU_WEIGHT_1_INDEX), return false);
            hasRelu1 = true;
        }
    }

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::CheckConvPostCubeDtype(const GNodePtr postCubeNode) const
{
    TensorDesc postCubeInDesc;
    FUSION_PASS_CHECK(postCubeNode->GetInputDesc(0, postCubeInDesc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get fxipipe in tensor desc failed."), return false);
    DataType postCubeInDtype = postCubeInDesc.GetDataType();

    TensorDesc postCubeOutDesc;
    FUSION_PASS_CHECK(postCubeNode->GetOutputDesc(OUTPUT_INDEX, postCubeOutDesc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get fxipipe out tensor desc failed."), return false);
    DataType postCubeOutDtype = postCubeOutDesc.GetDataType();

    std::vector<DataType> checkDtypes = {convDescInfo.fmapDtype, convDescInfo.filterDtype,
        postCubeInDtype, postCubeOutDtype};
    OP_LOGD(convDescInfo.nodeNameStr, "Current dtypes: fmap is %s weight is %s postCubeIn is %s postCubeOut is %s.",
        TypeUtils::DataTypeToSerialString(convDescInfo.fmapDtype).c_str(),
        TypeUtils::DataTypeToSerialString(convDescInfo.filterDtype).c_str(),
        TypeUtils::DataTypeToSerialString(postCubeInDtype).c_str(),
        TypeUtils::DataTypeToSerialString(postCubeOutDtype).c_str());

    auto supportedDtypes = npuArch == NpuArch::DAV_5102 ?
        SUPPORTED_DTYPES_WITH_POST_CUBE_DAV_5102 : SUPPORTED_DTYPES_WITH_POST_CUBE_DAV_3510;
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSupportList<DataType>(supportedDtypes, checkDtypes),
        OP_LOGE(convDescInfo.nodeNameStr, "Current dtype not supported."), return false);

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::CheckDescInfo()
{
    for (auto postCubeNode : postCubeNodes) {
        FUSION_PASS_CHECK_NOLOG(!CheckConvPostCubeDtype(postCubeNode), return false);
        // Check PostCube not supported case.
        FUSION_PASS_CHECK_NOLOG(!CheckSupportPostCubeCase(postCubeNode), return false);
    }

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::CheckSupportPostCubeCase(const GNodePtr postCubeNode)
{
    std::vector<AscendString> postCubeFusionOp;
    FUSION_PASS_CHECK(postCubeNode->GetAttr(FUSION_OP_LIST, postCubeFusionOp) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Get fusion op list from PostCube unsuccessfully."), return false);

    postCubeFusionOps.emplace_back(postCubeFusionOp);

    AscendString supportedListStr = ConvFusionUtilsPass::ListToAscendString(SUPPORTED_NODE_TYPES);
    for (auto &node : postCubeFusionOp) {
        if (std::find(SUPPORTED_NODE_TYPES.begin(), SUPPORTED_NODE_TYPES.end(), node) == SUPPORTED_NODE_TYPES.end()) {
            OP_LOGE(convDescInfo.nodeNameStr, "PostCube uint not supported: %s, only support [%s].",
                node.GetString(), supportedListStr.GetString());
            return false;
        }
    }

    if (std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(), ConvFusionUtils::ASCEND_DEQUANT) !=
        postCubeFusionOp.end() || std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(),
        Conv2DPostCubeToExtendConv2DFusion::ASCEND_REQUANT) != postCubeFusionOp.end()) {
        TensorDesc quantScale0Desc;
        FUSION_PASS_CHECK(
            postCubeNode->GetInputDesc(POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, quantScale0Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get fxipipe quant_scale_0 tensor desc failed."), return false);

        // check PostCube unit input format: quant_scale_0
        Format quantScale0Format = quantScale0Desc.GetFormat();
        if (quantScale0Format != FORMAT_ND && quantScale0Format != FORMAT_NCHW) {
            OP_LOGE(convDescInfo.nodeNameStr,
                "PostCube node quant_scale_0 format %s not supported, should be in [ND, NCHW].",
                TypeUtils::FormatToSerialString(quantScale0Format).c_str());
            return false;
        }

        // check PostCube unit input dtype: quant_scale_0
        DataType quantScale0dtype = quantScale0Desc.GetDataType();
        if (quantScale0dtype != ge::DT_UINT64 && quantScale0dtype != ge::DT_INT64) {
            OP_LOGE(convDescInfo.nodeNameStr,
                "PostCube node quant_scale_0 dtype %s not supported, should be in [uint64, int64].",
                TypeUtils::DataTypeToSerialString(quantScale0dtype).c_str());
            return false;
        }
    }

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::GetPostCubeNodes(const GNode &convNode)
{
    auto convOutputNodes = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK(convOutputNodes.empty(),
        OP_LOGE(convDescInfo.nodeNameStr, "Conv2D out nodes is empty after PostCube fusion."), return false);

    for (auto &outputNode : convOutputNodes) {
        GNodePtr nodePtr = outputNode.first;
        FUSION_PASS_CHECK(nodePtr == nullptr,
            OP_LOGE(convDescInfo.nodeNameStr, "After create PostCube, Conv2D out nodes is nullptr."), return false);

        AscendString curNodeType;
        FUSION_PASS_CHECK(nodePtr->GetType(curNodeType) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get node type failed."), return false);
        if (curNodeType == POST_CUBE_OP) {
            postCubeNodes.emplace_back(nodePtr);
        } else {
            otherNodes.emplace_back(outputNode);
        }
    }

    FUSION_PASS_CHECK(postCubeNodes.empty(),
        OP_LOGE(convDescInfo.nodeNameStr, "Get PostCube nodes failed."), return false);

    AscendString firstNodeType;
    convOutputNodes[0].first->GetType(firstNodeType);
    bool isPostCubeFirst = firstNodeType == POST_CUBE_OP;

    if ((postCubeNodes.size() > DUAL_OUTPUTNUM) || (!otherNodes.empty() && postCubeNodes.size() == DUAL_OUTPUTNUM)) {
        OP_LOGE(convDescInfo.nodeNameStr, "Unsupported multiple outputs with %zu PostCube and %zu other nodes.",
            postCubeNodes.size(), otherNodes.size());
        return false;
    }

    if (postCubeNodes.size() == DUAL_OUTPUTNUM) {
        outputCase = OutputCase::DUAL_POST_CUBE;
    } else if (isPostCubeFirst && !otherNodes.empty()) {
        outputCase = OutputCase::POST_CUBE_OTHER;
    } else if (!isPostCubeFirst && !otherNodes.empty()) {
        outputCase = OutputCase::OTHER_POST_CUBE;
    } else {
        outputCase = OutputCase::SINGLE;
    }

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::IsReluEnable(const std::vector<AscendString> &postCubeFusionOp,
    const AscendString &opType) const
{
    if (postCubeFusionOp.empty()) {
        return false;
    }

    bool ret = false;
    if (opType == "default") {
        ret = std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(), Conv2DPostCubeToExtendConv2DFusion::RELU) !=
            postCubeFusionOp.end() || std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(),
            Conv2DPostCubeToExtendConv2DFusion::LEAKY_RELU) != postCubeFusionOp.end();
    } else {
        ret = std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(), opType) != postCubeFusionOp.end();
    }

    return ret;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::IsScaleEnable(const std::vector<AscendString> &postCubeFusionOp) const
{
    if (postCubeFusionOp.empty()) {
        return false;
    }

    return std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(), ConvFusionUtils::ASCEND_QUANT) !=
        postCubeFusionOp.end() || std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(),
        ConvFusionUtils::ASCEND_DEQUANT) != postCubeFusionOp.end() || std::find(postCubeFusionOp.begin(),
        postCubeFusionOp.end(), ASCEND_REQUANT) != postCubeFusionOp.end();
}

void Conv2DPostCubeToExtendConv2DFusionPass::SelectPostCubePassByWhiteList(
    std::vector<ops::PostCubePassInfo> &matchLists) const
{
    std::vector<ops::PostCubePassInfo> tmpLists(matchLists);
    matchLists.clear();
    for (auto &postCubePass : tmpLists) {
        std::vector<AscendString> nodeTypeList;
        for (uint32_t index = 0; index < postCubePass.m_opnodes.size(); ++index) {
            auto curNode = postCubePass.m_opnodes[index].GetNode();
            FUSION_PASS_CHECK(curNode == nullptr,
                OP_LOGD(convDescInfo.nodeNameStr, "Node is nullptr in postCubePass."), return);
            AscendString nodeType;
            curNode->GetType(nodeType);
            nodeTypeList.push_back(nodeType);
            if (std::find(SUPPORTED_NODE_TYPES.begin(), SUPPORTED_NODE_TYPES.end(), nodeType) ==
                SUPPORTED_NODE_TYPES.end()) {
                OP_LOGD(convDescInfo.nodeNameStr,
                        "Node[%s] is not supported, removing cur list.", nodeType.GetString());
                break;
            }

            if (index == postCubePass.m_opnodes.size() - 1) {
                AscendString nodeTypeListStr = ConvFusionUtilsPass::ListToAscendString(nodeTypeList);
                if (std::find(nodeTypeList.begin(), nodeTypeList.end(),
                    Conv2DPostCubeToExtendConv2DFusion::LEAKY_RELU) != nodeTypeList.end() &&
                    std::find(nodeTypeList.begin(), nodeTypeList.end(),
                    ConvFusionUtils::ASCEND_DEQUANT) != nodeTypeList.end()) {
                    OP_LOGD(convDescInfo.nodeNameStr,
                            "List[%s] is not supported, removing.", nodeTypeListStr.GetString());
                    break;
                }

                matchLists.push_back(postCubePass);
                OP_LOGD(convDescInfo.nodeNameStr, "PostCubePass after selected by Valid list is %s",
                    nodeTypeListStr.GetString());
            }
        }
    }
}

bool Conv2DPostCubeToExtendConv2DFusionPass::UpdateExtendConv2DDesc(GNode *extendConv2D) const
{
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::UpdateInputDesc(extendConv2D, convDescInfo), return false);

    if (outputCase != OutputCase::OTHER_POST_CUBE) {
        TensorDesc output0Desc;
        FUSION_PASS_CHECK(postCubeNodes.front()->GetOutputDesc(OUTPUT_0_INDEX, output0Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get PostCube0 output tensor desc failed."), return false);
        FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(OUTPUT_0_INDEX, output0Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output0 tensor desc failed."), return false);

        if (hasScale0) {
            FUSION_PASS_CHECK_NOLOG(!UpdateScaleReluDesc(postCubeNodes.front(), extendConv2D,
                POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, EXTENDCONV2D_QUANT_SCALE_0_INDEX, SCALE_0), return false);
        }
        if (hasRelu0) {
            FUSION_PASS_CHECK_NOLOG(!UpdateScaleReluDesc(postCubeNodes.front(), extendConv2D,
                POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX, EXTENDCONV2D_RELU_WEIGHT_0_INDEX, RELU_WEIGHT_0), return false);
        }

        if (outputCase == OutputCase::SINGLE) {
            FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(OUTPUT_1_INDEX, output0Desc) != GRAPH_SUCCESS,
                OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output0 tensor desc failed."), return false);
        }
    }

    if (outputCase == OutputCase::DUAL_POST_CUBE || outputCase == OutputCase::OTHER_POST_CUBE) {
        TensorDesc output1Desc;
        FUSION_PASS_CHECK(postCubeNodes.back()->GetOutputDesc(OUTPUT_0_INDEX, output1Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get PostCube1 output tensor desc failed."), return false);
        FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(OUTPUT_1_INDEX, output1Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output1 tensor desc failed."), return false);

        if (hasScale1) {
            FUSION_PASS_CHECK_NOLOG(!UpdateScaleReluDesc(postCubeNodes.back(), extendConv2D,
                POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, EXTENDCONV2D_QUANT_SCALE_1_INDEX, SCALE_1), return false);
        }
        if (hasRelu1) {
            FUSION_PASS_CHECK_NOLOG(!UpdateScaleReluDesc(postCubeNodes.back(), extendConv2D,
                POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX, EXTENDCONV2D_RELU_WEIGHT_1_INDEX, RELU_WEIGHT_1), return false);
        }
    }

    if (outputCase == OutputCase::POST_CUBE_OTHER || outputCase == OutputCase::OTHER_POST_CUBE) {
        int32_t index = outputCase == OutputCase::OTHER_POST_CUBE ? OUTPUT_0_INDEX : OUTPUT_1_INDEX;
        FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(index, convDescInfo.outputDesc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output%d tensor desc failed.", index),
            return false);
    }

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::UpdateScaleReluDesc(GNodePtr postCube, GNode *extendConv2D,
    const int32_t getIndex, const int32_t updateIndex, const AscendString &name) const
{
    TensorDesc tensorDesc;
    FUSION_PASS_CHECK(postCube->GetInputDesc(getIndex, tensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Get %s tensor desc failed.", name.GetString()), return false);
    // Currently, conv only supports per-tensor quantization scenarios, but onnx plugin set format to NCHW
    tensorDesc.SetFormat(ge::FORMAT_ND);
    tensorDesc.SetOriginFormat(ge::FORMAT_ND);
    FUSION_PASS_CHECK(extendConv2D->UpdateInputDesc(updateIndex, tensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Update %s tensor desc failed.", name.GetString()), return false);

    return true;
}

} // namespace Ops