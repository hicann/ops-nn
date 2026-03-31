/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv2d_fixpipe_to_extendconv2d_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "es_nn_ops.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace Conv2DFixpipeToExtendConv2DFusion;
using namespace ge;
using namespace fusion;

std::unique_ptr<SubgraphBoundary> Conv2DFixPipeToExtendConv2DFusionPass::ConstructBoundary(const GNode &convNode)
{
    FUSION_PASS_CHECK_NOLOG(!GetFixpipeNodes(convNode), return nullptr);
    FUSION_PASS_CHECK_NOLOG(!CheckDescInfo(), return nullptr);

    std::unique_ptr<SubgraphBoundary> boundary = std::make_unique<SubgraphBoundary>();
    for (size_t index = 0; index < convNode.GetInputsSize(); ++index) {
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, convNode, static_cast<int64_t>(index),
            static_cast<int64_t>(index)), return nullptr);
    }

    FUSION_PASS_CHECK_NOLOG(!AddScaleReluToBoundAry(boundary), return nullptr);

    if (outputCase == OutputCase::DUAL_FIXPIPE || outputCase == OutputCase::SINGLE) {
        for (size_t index = 0; index < fixpipeNodes.size(); index++) {
            GNodePtr fixpipeNode = fixpipeNodes[index];
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphOutput(boundary, *fixpipeNode,
                OUTPUT_0_INDEX, static_cast<int64_t>(index)), return nullptr);
        }
    } else if (outputCase == OutputCase::OTHER_FIXPIPE || outputCase == OutputCase::FIXPIPE_OTHER) {
        // Add fixpipe to boundary
        int64_t fipxipeIndex = outputCase != OutputCase::OTHER_FIXPIPE ? OUTPUT_0_INDEX : OUTPUT_1_INDEX;
        int64_t otherIndex = outputCase != OutputCase::OTHER_FIXPIPE ? OUTPUT_1_INDEX : OUTPUT_0_INDEX;

        GNodePtr fixpipeNode = fixpipeNodes[0];
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphOutput(boundary, *fixpipeNode,
            static_cast<int64_t>(OUTPUT_INDEX), fipxipeIndex), return nullptr);
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphOutput(boundary, convNode,
            static_cast<int64_t>(OUTPUT_INDEX), otherIndex), return nullptr);
    }

    return boundary;
}

bool Conv2DFixPipeToExtendConv2DFusionPass::FixpipeFusionImpl(GraphPtr &graph, GNode &convNode,
    const CustomPassContext &pass_context)
{
    GNodePtr nodePtr = ConvFusionUtilsPass::GetNodePtr(convNode, convDescInfo);
    FUSION_PASS_CHECK_NOLOG(nodePtr == nullptr, return false);

    ops::FixpipeUtils fixpipeUtils;
    // [Step 1] Determine which nodes in the subsequent nodes satisfy fixpipe hardware unit
    FUSION_PASS_CHECK(fixpipeUtils.GetFixpipeNodeList(nodePtr, pass_context) != GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "GetFixpipeNodeList failed, no fusion."), return false);

    // [Step 2] To customize the selection of the fixpipe fusion range
    SelectFixpipePassByWhiteList(fixpipeUtils.m_matchpasses_);

    // [Step 3] Fixpipe tool method selects 1~2 Fixpipe paths
    FUSION_PASS_CHECK(fixpipeUtils.SelectFixpipeNodeList(true) != GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "SelectFixpipeNodeList failed, no fusion."), return false);

    // [Step 4] Create the Fixpipe operator node and modify the graph
    std::vector<GNodePtr> newNodes;
    FUSION_PASS_CHECK(fixpipeUtils.CreateFixpipeNode(convDescInfo.nodeNameStr, *graph, newNodes) != GRAPH_SUCCESS,
        OP_LOGD(convDescInfo.nodeNameStr, "CreateFixpipeNode failed, no fusion."), return false);

    return true;
}

void Conv2DFixPipeToExtendConv2DFusionPass::InitMember()
{
    fixpipeFusionOps.clear();
    fixpipeNodes.clear();
    otherNodes.clear();
    outputCase = OutputCase::SINGLE;
    hasScale0 = false;
    hasScale1 = false;
    hasRelu0 = false;
    hasRelu1 = false;
    graphIndex = REQUIRED_INPUT_NUMS;
}

bool Conv2DFixPipeToExtendConv2DFusionPass::MeetRequirements(const GNode &convNode)
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

AscendString Conv2DFixPipeToExtendConv2DFusionPass::GetNodeType() const
{
    return ConvFusionUtils::CONV2D;
}

std::map<std::string, NpuArch> Conv2DFixPipeToExtendConv2DFusionPass::GetSocSupportList() const
{
    return SUPPORT_SOC_LIST;
}

void Conv2DFixPipeToExtendConv2DFusionPass::PrintGraphStructure() const
{
    if (fixpipeFusionOps.empty()) {
        return;
    }

    auto fusionList0 = ConvFusionUtilsPass::ListToAscendString(fixpipeFusionOps.front());
    auto fusionList1 = ConvFusionUtilsPass::ListToAscendString(fixpipeFusionOps.back());

    std::stringstream logStr;
    logStr << "graph structure: cube node name: [ " << convDescInfo.nodeNameStr << " ], ";
    if (outputCase == OutputCase::SINGLE) {
        logStr << "y0" << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList0.GetString() << "].";
    } else if (outputCase == OutputCase::DUAL_FIXPIPE) {
        logStr << "y0" << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList0.GetString() << "], ";
        logStr << "y1" << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList1.GetString() << "].";
    } else if (outputCase == OutputCase::FIXPIPE_OTHER) {
        logStr << "y0" << " output: [" << convDescInfo.nodeNameStr << "], ";
        logStr << "y1" << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList1.GetString() << "].";
    } else if (outputCase == OutputCase::OTHER_FIXPIPE) {
        logStr << "y0" << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList0.GetString() << "], ";
        logStr << "y1" << " output: [" << convDescInfo.nodeNameStr << "].";
    }

    OP_LOGI(convDescInfo.nodeNameStr, "%s", logStr.str().c_str());
}

GraphUniqPtr Conv2DFixPipeToExtendConv2DFusionPass::Replacement(const GNode &convNode)
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

    bool enableRelu0 = outputCase != OutputCase::OTHER_FIXPIPE &&
        !fixpipeFusionOps.empty() && IsReluEnable(fixpipeFusionOps.front());
    bool enableRelu1 = (outputCase == OutputCase::DUAL_FIXPIPE || outputCase == OutputCase::OTHER_FIXPIPE) &&
        !fixpipeFusionOps.empty() && IsReluEnable(fixpipeFusionOps.back());
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

bool Conv2DFixPipeToExtendConv2DFusionPass::AddScaleReluToBoundAry(std::unique_ptr<SubgraphBoundary> &boundary)
{
    // Add quant_scale_0 and relu_weight_0 to boundary
    if (outputCase != OutputCase::OTHER_FIXPIPE) {
        if (IsScaleEnable(fixpipeFusionOps.front())) {
            GNodePtr fixpipeNode = fixpipeNodes.front();
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, *fixpipeNode,
                FIXPIPE_INPUT_QUANT_SCALE_0_INDEX, EXTENDCONV2D_QUANT_SCALE_0_INDEX), return false);
            hasScale0 = true;
        } else if (IsReluEnable(fixpipeFusionOps.front(), Conv2DFixpipeToExtendConv2DFusion::LEAKY_RELU)) {
            GNodePtr fixpipeNode = fixpipeNodes.front();
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, *fixpipeNode,
                FIXPIPE_INPUT_RELU_WEIGHT_0_INDEX, EXTENDCONV2D_RELU_WEIGHT_0_INDEX), return false);
            hasRelu0 = true;
        }
    }

    // Add quant_scale_1 and relu_weight_1 to boundary
    if (outputCase == OutputCase::OTHER_FIXPIPE || outputCase == OutputCase::DUAL_FIXPIPE) {
        if (IsScaleEnable(fixpipeFusionOps.back())) {
            GNodePtr fixpipeNode = fixpipeNodes.back();
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, *fixpipeNode,
                FIXPIPE_INPUT_QUANT_SCALE_0_INDEX, EXTENDCONV2D_QUANT_SCALE_1_INDEX), return false);
            hasScale1 = true;
        } else if (IsReluEnable(fixpipeFusionOps.back(), Conv2DFixpipeToExtendConv2DFusion::LEAKY_RELU)) {
            GNodePtr fixpipeNode = fixpipeNodes.back();
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, *fixpipeNode,
                FIXPIPE_INPUT_RELU_WEIGHT_0_INDEX, EXTENDCONV2D_RELU_WEIGHT_1_INDEX), return false);
            hasRelu1 = true;
        }
    }

    return true;
}

bool Conv2DFixPipeToExtendConv2DFusionPass::CheckConvFixpipeDtype(const GNodePtr fixpipeNode) const
{
    TensorDesc fixpipeInDesc;
    FUSION_PASS_CHECK(fixpipeNode->GetInputDesc(0, fixpipeInDesc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get fxipipe in tensor desc failed."), return false);
    DataType fixpipeInDtype = fixpipeInDesc.GetDataType();

    TensorDesc fixpipeOutDesc;
    FUSION_PASS_CHECK(fixpipeNode->GetOutputDesc(OUTPUT_INDEX, fixpipeOutDesc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get fxipipe out tensor desc failed."), return false);
    DataType fixpipeOutDtype = fixpipeOutDesc.GetDataType();

    std::vector<DataType> checkDtypes = {convDescInfo.fmapDtype, convDescInfo.filterDtype,
        fixpipeInDtype, fixpipeOutDtype};
    OP_LOGD(convDescInfo.nodeNameStr, "Current dtypes: fmap is %s weight is %s fixpipeIn is %s fixpipeOut is %s.",
        TypeUtils::DataTypeToSerialString(convDescInfo.fmapDtype).c_str(),
        TypeUtils::DataTypeToSerialString(convDescInfo.filterDtype).c_str(),
        TypeUtils::DataTypeToSerialString(fixpipeInDtype).c_str(),
        TypeUtils::DataTypeToSerialString(fixpipeOutDtype).c_str());

    auto supportedDtypes = npuArch == NpuArch::DAV_5102 ?
        SUPPORTED_DTYPES_WITH_FIXPIPE_DAV_5102 : SUPPORTED_DTYPES_WITH_FIXPIPE_DAV_3510;
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSupportList<DataType>(supportedDtypes, checkDtypes),
        OP_LOGE(convDescInfo.nodeNameStr, "Current dtype not supported."), return false);

    return true;
}

bool Conv2DFixPipeToExtendConv2DFusionPass::CheckDescInfo()
{
    for (auto fixpipeNode : fixpipeNodes) {
        FUSION_PASS_CHECK_NOLOG(!CheckConvFixpipeDtype(fixpipeNode), return false);
        // Check fixpipe not supported case.
        FUSION_PASS_CHECK_NOLOG(!CheckSupportFixpipeCase(fixpipeNode), return false);
    }

    return true;
}

bool Conv2DFixPipeToExtendConv2DFusionPass::CheckSupportFixpipeCase(const GNodePtr fixpipeNode)
{
    std::vector<AscendString> fixpipeFusionOp;
    FUSION_PASS_CHECK(fixpipeNode->GetAttr(FUSION_OP_LIST, fixpipeFusionOp) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Get fusion op list from FixPipe unsuccessfully."), return false);

    fixpipeFusionOps.emplace_back(fixpipeFusionOp);

    AscendString supportedListStr = ConvFusionUtilsPass::ListToAscendString(SUPPORTED_NODE_TYPES);
    for (auto &node : fixpipeFusionOp) {
        if (std::find(SUPPORTED_NODE_TYPES.begin(), SUPPORTED_NODE_TYPES.end(), node) == SUPPORTED_NODE_TYPES.end()) {
            OP_LOGE(convDescInfo.nodeNameStr, "Fixpipe uint not supported: %s, only support [%s].",
                node.GetString(), supportedListStr.GetString());
            return false;
        }
    }

    if (std::find(fixpipeFusionOp.begin(), fixpipeFusionOp.end(), ConvFusionUtils::ASCEND_DEQUANT) !=
        fixpipeFusionOp.end() || std::find(fixpipeFusionOp.begin(), fixpipeFusionOp.end(),
        Conv2DFixpipeToExtendConv2DFusion::ASCEND_REQUANT) != fixpipeFusionOp.end()) {
        TensorDesc quantScale0Desc;
        FUSION_PASS_CHECK(
            fixpipeNode->GetInputDesc(FIXPIPE_INPUT_QUANT_SCALE_0_INDEX, quantScale0Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get fxipipe quant_scale_0 tensor desc failed."), return false);

        // check fixpipe unit input format: quant_scale_0
        Format quantScale0Format = quantScale0Desc.GetFormat();
        if (quantScale0Format != FORMAT_ND && quantScale0Format != FORMAT_NCHW) {
            OP_LOGE(convDescInfo.nodeNameStr,
                "Fixpipe node quant_scale_0 format %s not supported, should be in [ND, NCHW].",
                TypeUtils::FormatToSerialString(quantScale0Format).c_str());
            return false;
        }

        // check fixpipe unit input dtype: quant_scale_0
        DataType quantScale0dtype = quantScale0Desc.GetDataType();
        if (quantScale0dtype != ge::DT_UINT64 && quantScale0dtype != ge::DT_INT64) {
            OP_LOGE(convDescInfo.nodeNameStr,
                "Fixpipe node quant_scale_0 dtype %s not supported, should be in [uint64, int64].",
                TypeUtils::DataTypeToSerialString(quantScale0dtype).c_str());
            return false;
        }
    }

    return true;
}

bool Conv2DFixPipeToExtendConv2DFusionPass::GetFixpipeNodes(const GNode &convNode)
{
    auto convOutputNodes = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK(convOutputNodes.empty(),
        OP_LOGE(convDescInfo.nodeNameStr, "Conv2D out nodes is empty after fixpipe fusion."), return false);

    for (auto &outputNode : convOutputNodes) {
        GNodePtr nodePtr = outputNode.first;
        FUSION_PASS_CHECK(nodePtr == nullptr,
            OP_LOGE(convDescInfo.nodeNameStr, "After create Fixpipe, Conv2D out nodes is nullptr."), return false);

        AscendString curNodeType;
        FUSION_PASS_CHECK(nodePtr->GetType(curNodeType) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get node type failed."), return false);
        if (curNodeType == FIXPIPE) {
            fixpipeNodes.emplace_back(nodePtr);
        } else {
            otherNodes.emplace_back(outputNode);
        }
    }

    FUSION_PASS_CHECK(fixpipeNodes.empty(),
        OP_LOGE(convDescInfo.nodeNameStr, "Get FixPipe nodes failed."), return false);

    AscendString firstNodeType;
    convOutputNodes[0].first->GetType(firstNodeType);
    bool isFixpipeFirst = firstNodeType == FIXPIPE;

    if ((fixpipeNodes.size() > DUAL_OUTPUTNUM) || (!otherNodes.empty() && fixpipeNodes.size() == DUAL_OUTPUTNUM)) {
        OP_LOGE(convDescInfo.nodeNameStr, "Unsupported multiple outputs with %zu fixpipe and %zu other nodes.",
            fixpipeNodes.size(), otherNodes.size());
        return false;
    }

    if (fixpipeNodes.size() == DUAL_OUTPUTNUM) {
        outputCase = OutputCase::DUAL_FIXPIPE;
    } else if (isFixpipeFirst && !otherNodes.empty()) {
        outputCase = OutputCase::FIXPIPE_OTHER;
    } else if (!isFixpipeFirst && !otherNodes.empty()) {
        outputCase = OutputCase::OTHER_FIXPIPE;
    } else {
        outputCase = OutputCase::SINGLE;
    }

    return true;
}

bool Conv2DFixPipeToExtendConv2DFusionPass::IsReluEnable(const std::vector<AscendString> &fixpipeFusionOp,
    const AscendString &opType) const
{
    if (fixpipeFusionOp.empty()) {
        return false;
    }
    
    bool ret = false;
    if (opType == "default") {
        ret = std::find(fixpipeFusionOp.begin(), fixpipeFusionOp.end(), Conv2DFixpipeToExtendConv2DFusion::RELU) !=
            fixpipeFusionOp.end() || std::find(fixpipeFusionOp.begin(), fixpipeFusionOp.end(),
            Conv2DFixpipeToExtendConv2DFusion::LEAKY_RELU) != fixpipeFusionOp.end();
    } else {
        ret = std::find(fixpipeFusionOp.begin(), fixpipeFusionOp.end(), opType) != fixpipeFusionOp.end();
    }

    return ret;
}

bool Conv2DFixPipeToExtendConv2DFusionPass::IsScaleEnable(const std::vector<AscendString> &fixpipeFusionOp) const
{
    if (fixpipeFusionOp.empty()) {
        return false;
    }
    
    return std::find(fixpipeFusionOp.begin(), fixpipeFusionOp.end(), ConvFusionUtils::ASCEND_QUANT) !=
        fixpipeFusionOp.end() || std::find(fixpipeFusionOp.begin(), fixpipeFusionOp.end(),
        ConvFusionUtils::ASCEND_DEQUANT) != fixpipeFusionOp.end() || std::find(fixpipeFusionOp.begin(),
        fixpipeFusionOp.end(), ASCEND_REQUANT) != fixpipeFusionOp.end();
}

void Conv2DFixPipeToExtendConv2DFusionPass::SelectFixpipePassByWhiteList(
    std::vector<ops::FixPipePassInfo> &matchLists) const
{
    std::vector<ops::FixPipePassInfo> tmpLists(matchLists);
    matchLists.clear();
    for (auto &fixpipePass : tmpLists) {
        std::vector<AscendString> nodeTypeList;
        for (uint32_t index = 0; index < fixpipePass.m_opnodes.size(); ++index) {
            auto curNode = fixpipePass.m_opnodes[index].GetNode();
            FUSION_PASS_CHECK(curNode == nullptr,
                OP_LOGD(convDescInfo.nodeNameStr, "Node is nullptr in fixpipePass."), return);
            AscendString nodeType;
            curNode->GetType(nodeType);
            nodeTypeList.push_back(nodeType);
            if (std::find(SUPPORTED_NODE_TYPES.begin(), SUPPORTED_NODE_TYPES.end(), nodeType) ==
                SUPPORTED_NODE_TYPES.end()) {
                OP_LOGD(convDescInfo.nodeNameStr,
                        "Node[%s] is not supported, removing cur list.", nodeType.GetString());
                break;
            }

            if (index == fixpipePass.m_opnodes.size() - 1) {
                AscendString nodeTypeListStr = ConvFusionUtilsPass::ListToAscendString(nodeTypeList);
                if (std::find(nodeTypeList.begin(), nodeTypeList.end(),
                    Conv2DFixpipeToExtendConv2DFusion::LEAKY_RELU) != nodeTypeList.end() &&
                    std::find(nodeTypeList.begin(), nodeTypeList.end(),
                    ConvFusionUtils::ASCEND_DEQUANT) != nodeTypeList.end()) {
                    OP_LOGD(convDescInfo.nodeNameStr,
                            "List[%s] is not supported, removing.", nodeTypeListStr.GetString());
                    break;
                }

                matchLists.push_back(fixpipePass);
                OP_LOGD(convDescInfo.nodeNameStr, "FixpipePass after selected by Valid list is %s",
                    nodeTypeListStr.GetString());
            }
        }
    }
}

bool Conv2DFixPipeToExtendConv2DFusionPass::UpdateExtendConv2DDesc(GNode *extendConv2D) const
{
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::UpdateInputDesc(extendConv2D, convDescInfo), return false);

    if (outputCase != OutputCase::OTHER_FIXPIPE) {
        TensorDesc output0Desc;
        FUSION_PASS_CHECK(fixpipeNodes.front()->GetOutputDesc(OUTPUT_0_INDEX, output0Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get fixpipe0 output tensor desc failed."), return false);
        FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(OUTPUT_0_INDEX, output0Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output0 tensor desc failed."), return false);
        
        if (hasScale0) {
            FUSION_PASS_CHECK_NOLOG(!UpdateScaleReluDesc(fixpipeNodes.front(), extendConv2D,
                FIXPIPE_INPUT_QUANT_SCALE_0_INDEX, EXTENDCONV2D_QUANT_SCALE_0_INDEX, SCALE_0), return false);
        }
        if (hasRelu0) {
            FUSION_PASS_CHECK_NOLOG(!UpdateScaleReluDesc(fixpipeNodes.front(), extendConv2D,
                FIXPIPE_INPUT_RELU_WEIGHT_0_INDEX, EXTENDCONV2D_RELU_WEIGHT_0_INDEX, RELU_WEIGHT_0), return false);
        }

        if (outputCase == OutputCase::SINGLE) {
            FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(OUTPUT_1_INDEX, output0Desc) != GRAPH_SUCCESS,
                OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output0 tensor desc failed."), return false);
        }
    }

    if (outputCase == OutputCase::DUAL_FIXPIPE || outputCase == OutputCase::OTHER_FIXPIPE) {
        TensorDesc output1Desc;
        FUSION_PASS_CHECK(fixpipeNodes.back()->GetOutputDesc(OUTPUT_0_INDEX, output1Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get fixpipe1 output tensor desc failed."), return false);
        FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(OUTPUT_1_INDEX, output1Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output1 tensor desc failed."), return false);

        if (hasScale1) {
            FUSION_PASS_CHECK_NOLOG(!UpdateScaleReluDesc(fixpipeNodes.back(), extendConv2D,
                FIXPIPE_INPUT_QUANT_SCALE_0_INDEX, EXTENDCONV2D_QUANT_SCALE_1_INDEX, SCALE_1), return false);
        }
        if (hasRelu1) {
            FUSION_PASS_CHECK_NOLOG(!UpdateScaleReluDesc(fixpipeNodes.back(), extendConv2D,
                FIXPIPE_INPUT_RELU_WEIGHT_0_INDEX, EXTENDCONV2D_RELU_WEIGHT_1_INDEX, RELU_WEIGHT_1), return false);
        }
    }

    if (outputCase == OutputCase::FIXPIPE_OTHER || outputCase == OutputCase::OTHER_FIXPIPE) {
        int32_t index = outputCase == OutputCase::OTHER_FIXPIPE ? OUTPUT_0_INDEX : OUTPUT_1_INDEX;
        FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(index, convDescInfo.outputDesc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output%d tensor desc failed.", index),
            return false);
    }

    return true;
}

bool Conv2DFixPipeToExtendConv2DFusionPass::UpdateScaleReluDesc(GNodePtr fixpipe, GNode *extendConv2D,
    const int32_t getIndex, const int32_t updateIndex, const AscendString &name) const
{
    TensorDesc tensorDesc;
    FUSION_PASS_CHECK(fixpipe->GetInputDesc(getIndex, tensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Get %s tensor desc failed.", name.GetString()), return false);
    // Currently, conv only supports per-tensor quantization scenarios, but onnx plugin set format to NCHW
    tensorDesc.SetFormat(ge::FORMAT_ND);
    tensorDesc.SetOriginFormat(ge::FORMAT_ND);
    FUSION_PASS_CHECK(extendConv2D->UpdateInputDesc(updateIndex, tensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Update %s tensor desc failed.", name.GetString()), return false);

    return true;
}

} // namespace Ops