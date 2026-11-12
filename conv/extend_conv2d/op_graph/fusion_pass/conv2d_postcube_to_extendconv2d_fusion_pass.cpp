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

#include <algorithm>
#include <sstream>

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace Conv2DPostCubeToExtendConv2DFusion;
using namespace ge;
using namespace fusion;

namespace {
constexpr uint32_t FILTER_DIM_NUM = 4;
constexpr uint32_t NCHW_N_IDX = 0;
constexpr uint32_t NCHW_C_IDX = 1;
constexpr uint32_t NCHW_H_IDX = 2;
constexpr uint32_t NCHW_W_IDX = 3;
constexpr uint32_t NHWC_N_IDX = 0;
constexpr uint32_t NHWC_H_IDX = 1;
constexpr uint32_t NHWC_W_IDX = 2;
constexpr uint32_t NHWC_C_IDX = 3;
constexpr int64_t FRACTAL_Z_C0_INT8 = 32;
constexpr int64_t FRACTAL_Z_C0_FP16 = 16;
constexpr int64_t FRACTAL_Z_N0 = 16;
const ge::AscendString VECTOR_MUL_SCALAR_OP = "VectorMulScalar";
} // namespace

void Conv2DPostCubeToExtendConv2DFusionPass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    convDescInfo = ConvDescInfo();
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

bool Conv2DPostCubeToExtendConv2DFusionPass::MeetRequirements(const GNode& convNode)
{
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::CheckSocList(SUPPORT_SOC_LIST, npuArch, true),
                      OP_LOGD(convDescInfo.nodeNameStr, "Current soc not supported, no fusion."), return false);

    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do Conv2DPostCubeToExtendConv2DFusionPass.");

    FUSION_PASS_CHECK(convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX).empty(),
                      OP_LOGD(convDescInfo.nodeNameStr, "Conv2D out nodes are empty, no fusion needed."), return false);

    return true;
}

std::set<AscendString> Conv2DPostCubeToExtendConv2DFusionPass::GetNodeTypes() const { return {CONV2D}; }

void Conv2DPostCubeToExtendConv2DFusionPass::ScanLeakyReluSlope(const std::vector<ops::PostCubePassInfo>& matchLists,
                                                                float& slope, bool& hasLeakyRelu) const
{
    for (const auto& matchPass : matchLists) {
        for (const auto& nodeInfo : matchPass.m_opnodes) {
            FUSION_PASS_CHECK_NOLOG(nodeInfo.GetNode() == nullptr, continue);
            AscendString nodeType;
            FUSION_PASS_CHECK_NOLOG(nodeInfo.GetNode()->GetType(nodeType) != GRAPH_SUCCESS, continue);
            if (nodeType == Conv2DPostCubeToExtendConv2DFusion::LEAKY_RELU) {
                (void)ops::GNodeGetAttr(nodeInfo.GetNode(), ops::ATTR_NEGATIVE_SLOPE, slope);
                hasLeakyRelu = true;
            }
        }
    }
}

void Conv2DPostCubeToExtendConv2DFusionPass::PatchVectorMulScalarAttr(const GNode& convNode, float slope) const
{
    auto convOutputNodes = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (auto& outputNode : convOutputNodes) {
        FUSION_PASS_CHECK_NOLOG(outputNode.first == nullptr, continue);
        AscendString outputType;
        FUSION_PASS_CHECK_NOLOG(outputNode.first->GetType(outputType) != GRAPH_SUCCESS, continue);
        if (outputType != POST_CUBE_OP) {
            continue;
        }
        GNode postCubeNode = *outputNode.first;
        auto producer = postCubeNode.GetInDataNodesAndPortIndexs(POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX).first;
        FUSION_PASS_CHECK_NOLOG(producer == nullptr, continue);
        AscendString producerType;
        FUSION_PASS_CHECK_NOLOG(producer->GetType(producerType) != GRAPH_SUCCESS, continue);
        if (producerType != VECTOR_MUL_SCALAR_OP) {
            continue;
        }
        GNode vectorMulScalarNode = *producer;
        FUSION_PASS_CHECK(
            vectorMulScalarNode.SetAttr(ge::AscendString(ops::kAttrScale.c_str()), slope) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Set scalar attr for VectorMulScalar failed."), return);
        TensorDesc outputDesc;
        if (vectorMulScalarNode.GetOutputDesc(OUTPUT_INDEX, outputDesc) == GRAPH_SUCCESS) {
            (void)postCubeNode.UpdateInputDesc(POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX, outputDesc);
        }
        OP_LOGD(convDescInfo.nodeNameStr, "Fix VectorMulScalar scalar attr with leakyrelu slope [%f].", slope);
    }
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
        logStr << "y0"
               << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList0.GetString() << "].";
    } else if (outputCase == OutputCase::DUAL_POST_CUBE) {
        logStr << "y0"
               << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList0.GetString() << "], ";
        logStr << "y1"
               << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList1.GetString() << "].";
    } else if (outputCase == OutputCase::POST_CUBE_OTHER) {
        logStr << "y0"
               << " output: [" << convDescInfo.nodeNameStr << "], ";
        logStr << "y1"
               << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList1.GetString() << "].";
    } else if (outputCase == OutputCase::OTHER_POST_CUBE) {
        logStr << "y0"
               << " output: [" << convDescInfo.nodeNameStr << ", " << fusionList0.GetString() << "], ";
        logStr << "y1"
               << " output: [" << convDescInfo.nodeNameStr << "].";
    }

    OP_LOGI(convDescInfo.nodeNameStr, "%s", logStr.str().c_str());
}

Status Conv2DPostCubeToExtendConv2DFusionPass::ConvFusionPreImpl(GraphPtr& graph, GNode& convNode,
                                                                 CustomPassContext& passContext)
{
    GNodePtr nodePtr = ConvFusionUtilsPass::GetNodePtr(convNode, convDescInfo);
    FUSION_PASS_CHECK(nodePtr == nullptr, OP_LOGD(convDescInfo.nodeNameStr, "GetNodePtr failed, no fusion."),
                      return CONV_NOT_CHANGED);

    ops::PostCubeUtils postCubeUtils;
    FUSION_PASS_CHECK(postCubeUtils.GetPostCubeNodeList(nodePtr, passContext) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "GetPostCubeNodeList failed, no fusion."),
                      return CONV_NOT_CHANGED);

    SelectPostCubePassByWhiteList(postCubeUtils.m_matchpasses_);

    FUSION_PASS_CHECK(postCubeUtils.SelectPostCubeNodeList(true) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "SelectPostCubeNodeList failed, no fusion."),
                      return CONV_NOT_CHANGED);

    if (npuArch == Ops::Base::DAV_5162) {
        auto convOutputNodes = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
        FUSION_PASS_CHECK(convOutputNodes.size() != 1,
                          OP_LOGD(convDescInfo.nodeNameStr, "DAV_5162 only support single output, skip fusion."),
                          return CONV_NOT_CHANGED);
    }

    float leakySlope = 0.0f;
    bool hasLeakyRelu = false;
    ScanLeakyReluSlope(postCubeUtils.m_matchpasses_, leakySlope, hasLeakyRelu);

    std::vector<GNodePtr> newNodes;
    FUSION_PASS_CHECK(postCubeUtils.CreatePostCubeNode(convDescInfo.nodeNameStr, *graph, newNodes) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "CreatePostCubeNode failed, no fusion."),
                      return CONV_NOT_CHANGED);

    if (hasLeakyRelu) {
        PatchVectorMulScalarAttr(convNode, leakySlope);
    }

    return SUCCESS;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::ConvFusionReplaceImpl(GraphPtr& graph, GNode& convNode,
                                                                   CustomPassContext& passContext)
{
    FUSION_PASS_CHECK(graph == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Graph is nullptr."), return false);

    auto boundary = ConstructBoundary(convNode);
    FUSION_PASS_CHECK(boundary == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Construct boundary failed."),
                      return false);

    auto replacement = Replacement(convNode);
    FUSION_PASS_CHECK(replacement == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Construct replacement failed."),
                      return false);

    auto extendConv2DNamesBefore = ConvFusionUtilsPass::CollectNodeTypeNames(graph, EXTENDCONV2D);

    FUSION_PASS_CHECK(!ReplaceSubgraphWithFallback(boundary, replacement, passContext),
                      OP_LOGE(convDescInfo.nodeNameStr, "SubgraphRewriter::Replace failed."), return false);

    GNode extendConv2DNode;
    FUSION_PASS_CHECK(
        !ConvFusionUtilsPass::FindNewNodeOfType(graph, EXTENDCONV2D, extendConv2DNamesBefore, extendConv2DNode),
        OP_LOGE(convDescInfo.nodeNameStr, "Find new ExtendConv2D node after replace failed."), return false);

    FUSION_PASS_CHECK_NOLOG(!AdjustLayout(graph, extendConv2DNode), return false);
    FUSION_PASS_CHECK_NOLOG(!InsertA16W8WeightRearrange(graph, extendConv2DNode), return false);

    OP_LOGD(convDescInfo.nodeNameStr, "Conv2DPostCubeToExtendConv2DFusionPass replace success.");
    return true;
}

std::unique_ptr<SubgraphBoundary> Conv2DPostCubeToExtendConv2DFusionPass::ConstructBoundary(const GNode& convNode)
{
    FUSION_PASS_CHECK_NOLOG(!GetPostCubeNodes(convNode), return nullptr);
    FUSION_PASS_CHECK_NOLOG(!CheckDescInfo(), return nullptr);

    std::unique_ptr<SubgraphBoundary> boundary = std::make_unique<SubgraphBoundary>();
    for (size_t index = 0; index < convNode.GetInputsSize(); ++index) {
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphInput(boundary, convNode, static_cast<int64_t>(index),
                                                                       static_cast<int64_t>(index)),
                                return nullptr);
    }

    FUSION_PASS_CHECK_NOLOG(!AddScaleReluToBoundAry(boundary), return nullptr);

    if (outputCase == OutputCase::DUAL_POST_CUBE || outputCase == OutputCase::SINGLE) {
        for (size_t index = 0; index < postCubeNodes.size(); index++) {
            GNodePtr postCubeNode = postCubeNodes[index];
            FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphOutput(boundary, *postCubeNode, OUTPUT_INDEX,
                                                                            static_cast<int64_t>(index)),
                                    return nullptr);
        }
    } else if (outputCase == OutputCase::OTHER_POST_CUBE || outputCase == OutputCase::POST_CUBE_OTHER) {
        // Add PostCube node to boundary
        int64_t postCubeIndex = outputCase != OutputCase::OTHER_POST_CUBE ? OUTPUT_INDEX : OUTPUT_1_INDEX;
        int64_t otherIndex = outputCase != OutputCase::OTHER_POST_CUBE ? OUTPUT_1_INDEX : OUTPUT_INDEX;

        GNodePtr postCubeNode = postCubeNodes[0];
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AddSubgraphOutput(
                                    boundary, *postCubeNode, static_cast<int64_t>(OUTPUT_INDEX), postCubeIndex),
                                return nullptr);
        FUSION_PASS_CHECK_NOLOG(
            !ConvFusionUtilsPass::AddSubgraphOutput(boundary, convNode, static_cast<int64_t>(OUTPUT_INDEX), otherIndex),
            return nullptr);
    }

    return boundary;
}

GraphUniqPtr Conv2DPostCubeToExtendConv2DFusionPass::Replacement(const GNode& convNode)
{
    auto graphBuilder = es::EsGraphBuilder("replacement");

    ConvBaseAttrs baseAttrs;
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvBaseAttr(convNode, baseAttrs, convDescInfo), return nullptr);
    int64_t opImplModeEnum = 0;
    (void)convNode.GetAttr(OP_IMPL_MODE_ENUM, opImplModeEnum);
    baseAttrs.enableHf32 = std::any_of(
        HF32_PRECISION_MODES_INT.begin(), HF32_PRECISION_MODES_INT.end(),
        [&opImplModeEnum](const int64_t& targetPrecisionModeInt) { return opImplModeEnum == targetPrecisionModeInt; });

    auto [fmap, filter] = graphBuilder.CreateInputs<REQUIRED_INPUT_NUMS>();
    auto bias = convDescInfo.hasBias ? graphBuilder.CreateInput(graphIndex++) : nullptr;
    auto scale0 = hasScale0 ? graphBuilder.CreateInput(graphIndex++) : nullptr;
    auto relu0 = hasRelu0 ? graphBuilder.CreateInput(graphIndex++) : nullptr;
    auto scale1 = hasScale1 ? graphBuilder.CreateInput(graphIndex++) : nullptr;
    auto relu1 = hasRelu1 ? graphBuilder.CreateInput(graphIndex++) : nullptr;

    bool enableRelu0 = outputCase != OutputCase::OTHER_POST_CUBE && !postCubeFusionOps.empty() &&
                       IsReluEnable(postCubeFusionOps.front());
    bool enableRelu1 = (outputCase == OutputCase::DUAL_POST_CUBE || outputCase == OutputCase::OTHER_POST_CUBE) &&
                       !postCubeFusionOps.empty() && IsReluEnable(postCubeFusionOps.back());
    bool dualOutput = outputCase != OutputCase::SINGLE;

    // fmap, filter, bias, offset_w, scale0, relu_weight0, clip_value0, scale1, relu_weight1, clip_value1
    auto extendConv2D = es::ExtendConv2D(
        fmap, filter, bias, nullptr, scale0, relu0, nullptr, scale1, relu1, nullptr, baseAttrs.strides, baseAttrs.pads,
        baseAttrs.dilations, baseAttrs.groups, baseAttrs.dataFormat.GetString(), baseAttrs.offsetX, RINT.GetString(),
        baseAttrs.padMode.GetString(), baseAttrs.enableHf32, enableRelu0, enableRelu1, dualOutput);

    // Multi TensorHolder get from es::Op, it's GetProducer() point to one node.
    auto output0 = extendConv2D.y0.GetProducer();
    FUSION_PASS_CHECK_NOLOG(!UpdateExtendConv2DDesc(output0), return nullptr);

    int64_t fixedShiftValue = 0;
    (void)convNode.GetAttr(FIXED_SHIFT_VALUE.GetString(), fixedShiftValue);
    if (fixedShiftValue == 0 && convDescInfo.fmapDtype != DT_INT8) {
        fixedShiftValue = (convDescInfo.filterDtype == DT_INT8) ? FIXED_SHIFT_VALUE_DEFAULT_A16W8 :
                                                                  FIXED_SHIFT_VALUE_DEFAULT_A16W16;
    }
    FUSION_PASS_CHECK(output0->SetAttr(FIXED_SHIFT_VALUE.GetString(), fixedShiftValue) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set fixed_shift_value for ExtendConv2D failed."),
                      return nullptr);

    (void)ConvFusionUtilsPass::CopyConvOtherAttrs(convNode, *output0);

    std::vector<es::EsTensorHolder> replaceOutput = {extendConv2D.y0};
    if (dualOutput) {
        replaceOutput.emplace_back(extendConv2D.y1);
    }
    return graphBuilder.BuildAndReset(replaceOutput);
}

bool Conv2DPostCubeToExtendConv2DFusionPass::AddScaleReluToBoundAry(std::unique_ptr<SubgraphBoundary>& boundary)
{
    // Add quant_scale_0 and relu_weight_0 to boundary
    if (outputCase != OutputCase::OTHER_POST_CUBE) {
        if (IsScaleEnable(postCubeFusionOps.front())) {
            GNodePtr postCubeNode = postCubeNodes.front();
            FUSION_PASS_CHECK_NOLOG(
                !ConvFusionUtilsPass::AddSubgraphInput(boundary, *postCubeNode, POST_CUBE_INPUT_QUANT_SCALE_0_INDEX,
                                                       EXTENDCONV2D_QUANT_SCALE_0_INDEX),
                return false);
            hasScale0 = true;
        }
        if (IsReluEnable(postCubeFusionOps.front(), Conv2DPostCubeToExtendConv2DFusion::LEAKY_RELU)) {
            GNodePtr postCubeNode = postCubeNodes.front();
            FUSION_PASS_CHECK_NOLOG(
                !ConvFusionUtilsPass::AddSubgraphInput(boundary, *postCubeNode, POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX,
                                                       EXTENDCONV2D_RELU_WEIGHT_0_INDEX),
                return false);
            hasRelu0 = true;
        }
    }

    // Add quant_scale_1 and relu_weight_1 to boundary
    if (outputCase == OutputCase::OTHER_POST_CUBE || outputCase == OutputCase::DUAL_POST_CUBE) {
        if (IsScaleEnable(postCubeFusionOps.back())) {
            GNodePtr postCubeNode = postCubeNodes.back();
            FUSION_PASS_CHECK_NOLOG(
                !ConvFusionUtilsPass::AddSubgraphInput(boundary, *postCubeNode, POST_CUBE_INPUT_QUANT_SCALE_0_INDEX,
                                                       EXTENDCONV2D_QUANT_SCALE_1_INDEX),
                return false);
            hasScale1 = true;
        }
        if (IsReluEnable(postCubeFusionOps.back(), Conv2DPostCubeToExtendConv2DFusion::LEAKY_RELU)) {
            GNodePtr postCubeNode = postCubeNodes.back();
            FUSION_PASS_CHECK_NOLOG(
                !ConvFusionUtilsPass::AddSubgraphInput(boundary, *postCubeNode, POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX,
                                                       EXTENDCONV2D_RELU_WEIGHT_1_INDEX),
                return false);
            hasRelu1 = true;
        }
    }

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::CheckDescInfo()
{
    for (auto postCubeNode : postCubeNodes) {
        // Check PostCube not supported case.
        FUSION_PASS_CHECK_NOLOG(!CheckSupportPostCubeCase(postCubeNode), return false);
    }

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::CheckSupportPostCubeCase(const GNodePtr postCubeNode)
{
    std::vector<AscendString> postCubeFusionOp;
    FUSION_PASS_CHECK(postCubeNode->GetAttr(FUSION_OP_LIST, postCubeFusionOp) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get fusion op list from PostCube unsuccessfully."),
                      return false);

    postCubeFusionOps.emplace_back(postCubeFusionOp);

    AscendString supportedListStr = ConvFusionUtilsPass::ListToAscendString(SUPPORTED_NODE_TYPES);
    for (auto& node : postCubeFusionOp) {
        if (std::find(SUPPORTED_NODE_TYPES.begin(), SUPPORTED_NODE_TYPES.end(), node) == SUPPORTED_NODE_TYPES.end()) {
            std::stringstream reason;
            reason << "In the fusion pass Conv2DPostCubeToExtendConv2DFusionPass, "
                   << "the output node connected to this node can only be of the following types: "
                   << std::string(supportedListStr.GetString());
            OP_LOGE_FOR_INVALID_GRAPH_NODE(convDescInfo.nodeNameStr.c_str(), reason.str());
            return false;
        }
    }

    if (std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(), ConvFusionUtils::ASCEND_DEQUANT) !=
            postCubeFusionOp.end() ||
        std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(),
                  Conv2DPostCubeToExtendConv2DFusion::ASCEND_REQUANT) != postCubeFusionOp.end()) {
        TensorDesc quantScale0Desc;
        FUSION_PASS_CHECK(
            postCubeNode->GetInputDesc(POST_CUBE_INPUT_QUANT_SCALE_0_INDEX, quantScale0Desc) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Get PostCube quant_scale_0 tensor desc failed."), return false);

        // check PostCube unit input format: quant_scale_0
        Format quantScale0Format = quantScale0Desc.GetFormat();
        if (quantScale0Format != FORMAT_ND && quantScale0Format != FORMAT_NCHW) {
            std::string correctFormat = GeFormatToString(FORMAT_ND) + ", " + GeFormatToString(FORMAT_NCHW);
            OP_LOGE_FOR_INVALID_FORMAT(convDescInfo.nodeNameStr.c_str(), "scale0",
                                       GeFormatToString(quantScale0Format).c_str(), correctFormat.c_str());
            return false;
        }

        // check PostCube unit input dtype: quant_scale_0
        DataType quantScale0dtype = quantScale0Desc.GetDataType();
        if (quantScale0dtype != ge::DT_UINT64 && quantScale0dtype != ge::DT_INT64) {
            OP_LOGD(convDescInfo.nodeNameStr,
                    "PostCube node quant scale 0 dtype %s not supported, should be in [uint64, int64].",
                    GeDtypeToString(quantScale0dtype).c_str());
        }
    }

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::GetPostCubeNodes(const GNode& convNode)
{
    auto convOutputNodes = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK(convOutputNodes.empty(),
                      OP_LOGE(convDescInfo.nodeNameStr, "Conv2D out nodes are empty after PostCube fusion."),
                      return false);

    for (auto& outputNode : convOutputNodes) {
        GNodePtr nodePtr = outputNode.first;
        FUSION_PASS_CHECK(nodePtr == nullptr,
                          OP_LOGE(convDescInfo.nodeNameStr, "After PostCube creation, Conv2D out node is nullptr."),
                          return false);

        AscendString curNodeType;
        FUSION_PASS_CHECK(nodePtr->GetType(curNodeType) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get node type failed."), return false);
        if (curNodeType == POST_CUBE_OP) {
            postCubeNodes.emplace_back(nodePtr);
        } else {
            otherNodes.emplace_back(outputNode);
        }
    }

    FUSION_PASS_CHECK(postCubeNodes.empty(), OP_LOGE(convDescInfo.nodeNameStr, "Get PostCube nodes failed."),
                      return false);

    AscendString firstNodeType;
    FUSION_PASS_CHECK(convOutputNodes[0].first->GetType(firstNodeType) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get node type failed."), return false);
    bool isPostCubeFirst = firstNodeType == POST_CUBE_OP;

    if ((postCubeNodes.size() > DUAL_OUTPUTNUM) || (!otherNodes.empty() && postCubeNodes.size() == DUAL_OUTPUTNUM)) {
        AscendString tmpPostCube = ConvFusionUtilsPass::ListToAscendString(POST_CUBE_NODE_TYPES);
        std::string supportedPostCubeString(tmpPostCube.GetString());
        std::stringstream reason;
        reason << "In the fusion pass Conv2DPostCubeToExtendConv2DFusionPass, "
               << "the output node combination connected to this node can only be: "
               << "1. One postCube node and N non-postCube nodes; "
               << "2. Two postCube nodes. "
               << "The postCube node types include: " << supportedPostCubeString;
        OP_LOGE_FOR_INVALID_GRAPH_NODE(convDescInfo.nodeNameStr.c_str(), reason.str());
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

bool Conv2DPostCubeToExtendConv2DFusionPass::IsReluEnable(const std::vector<AscendString>& postCubeFusionOp,
                                                          const AscendString& opType) const
{
    if (postCubeFusionOp.empty()) {
        return false;
    }

    bool ret = false;
    if (opType == "default") {
        ret = std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(), Conv2DPostCubeToExtendConv2DFusion::RELU) !=
                  postCubeFusionOp.end() ||
              std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(),
                        Conv2DPostCubeToExtendConv2DFusion::LEAKY_RELU) != postCubeFusionOp.end();
    } else {
        ret = std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(), opType) != postCubeFusionOp.end();
    }

    return ret;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::IsScaleEnable(const std::vector<AscendString>& postCubeFusionOp) const
{
    if (postCubeFusionOp.empty()) {
        return false;
    }

    return std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(), ConvFusionUtils::ASCEND_QUANT) !=
               postCubeFusionOp.end() ||
           std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(), ConvFusionUtils::ASCEND_DEQUANT) !=
               postCubeFusionOp.end() ||
           std::find(postCubeFusionOp.begin(), postCubeFusionOp.end(), ASCEND_REQUANT) != postCubeFusionOp.end();
}

void Conv2DPostCubeToExtendConv2DFusionPass::SelectPostCubePassByWhiteList(
    std::vector<ops::PostCubePassInfo>& matchLists) const
{
    const bool isDn2NzSoc = ConvFusionUtilsPass::GetArchKey() == ConvFusionUtils::NPU_ARCH_KEY_3510;
    std::vector<ops::PostCubePassInfo> tmpLists(matchLists);
    matchLists.clear();
    for (auto& postCubePass : tmpLists) {
        std::vector<AscendString> nodeTypeList;
        for (uint32_t index = 0; index < postCubePass.m_opnodes.size(); ++index) {
            auto curNode = postCubePass.m_opnodes[index].GetNode();
            FUSION_PASS_CHECK(curNode == nullptr, OP_LOGD(convDescInfo.nodeNameStr, "Node is nullptr in postCubePass."),
                              return);
            AscendString nodeType;
            curNode->GetType(nodeType);
            nodeTypeList.push_back(nodeType);
            if (std::find(SUPPORTED_NODE_TYPES.begin(), SUPPORTED_NODE_TYPES.end(), nodeType) ==
                SUPPORTED_NODE_TYPES.end()) {
                OP_LOGD(convDescInfo.nodeNameStr, "Node[%s] is not supported, removing cur list.",
                        nodeType.GetString());
                break;
            }

            if (isDn2NzSoc && nodeType == Conv2DPostCubeToExtendConv2DFusion::RELU &&
                index == Conv2DPostCubeToExtendConv2DFusion::RELU_NODE_INDEX_IN_PASS &&
                postCubePass.m_opnodes.size() == Conv2DPostCubeToExtendConv2DFusion::DN2NZ_UNSUPPORT_PASS_NODE_NUM) {
                OP_LOGD(convDescInfo.nodeNameStr,
                        "Conv2D + node[%s] fusion path is not supported on dn2nz soc, removing cur list.",
                        nodeType.GetString());
                break;
            }

            if (index == postCubePass.m_opnodes.size() - 1) {
                matchLists.push_back(postCubePass);
                AscendString nodeTypeListStr = ConvFusionUtilsPass::ListToAscendString(nodeTypeList);
                OP_LOGD(convDescInfo.nodeNameStr, "PostCubePass selected by valid list is %s",
                        nodeTypeListStr.GetString());
            }
        }
    }
}

bool Conv2DPostCubeToExtendConv2DFusionPass::UpdateExtendConv2DDesc(GNode* extendConv2D) const
{
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::UpdateInputDesc(extendConv2D, convDescInfo), return false);

    if (outputCase != OutputCase::OTHER_POST_CUBE) {
        TensorDesc output0Desc;
        FUSION_PASS_CHECK(postCubeNodes.front()->GetOutputDesc(OUTPUT_INDEX, output0Desc) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get PostCube0 output tensor desc failed."), return false);
        FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(OUTPUT_INDEX, output0Desc) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output0 tensor desc failed."),
                          return false);

        if (hasScale0) {
            FUSION_PASS_CHECK_NOLOG(
                !UpdateScaleReluDesc(postCubeNodes.front(), extendConv2D, POST_CUBE_INPUT_QUANT_SCALE_0_INDEX,
                                     EXTENDCONV2D_QUANT_SCALE_0_INDEX, SCALE_0),
                return false);
        }
        if (hasRelu0) {
            FUSION_PASS_CHECK_NOLOG(
                !UpdateScaleReluDesc(postCubeNodes.front(), extendConv2D, POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX,
                                     EXTENDCONV2D_RELU_WEIGHT_0_INDEX, RELU_WEIGHT_0),
                return false);
        }

        if (outputCase == OutputCase::SINGLE) {
            FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(OUTPUT_1_INDEX, output0Desc) != GRAPH_SUCCESS,
                              OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output1 tensor desc failed."),
                              return false);
        }
    }

    if (outputCase == OutputCase::DUAL_POST_CUBE || outputCase == OutputCase::OTHER_POST_CUBE) {
        TensorDesc output1Desc;
        FUSION_PASS_CHECK(postCubeNodes.back()->GetOutputDesc(OUTPUT_INDEX, output1Desc) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get PostCube1 output tensor desc failed."), return false);
        FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(OUTPUT_1_INDEX, output1Desc) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output1 tensor desc failed."),
                          return false);

        if (hasScale1) {
            FUSION_PASS_CHECK_NOLOG(
                !UpdateScaleReluDesc(postCubeNodes.back(), extendConv2D, POST_CUBE_INPUT_QUANT_SCALE_0_INDEX,
                                     EXTENDCONV2D_QUANT_SCALE_1_INDEX, SCALE_1),
                return false);
        }
        if (hasRelu1) {
            FUSION_PASS_CHECK_NOLOG(
                !UpdateScaleReluDesc(postCubeNodes.back(), extendConv2D, POST_CUBE_INPUT_RELU_WEIGHT_0_INDEX,
                                     EXTENDCONV2D_RELU_WEIGHT_1_INDEX, RELU_WEIGHT_1),
                return false);
        }
    }

    if (outputCase == OutputCase::POST_CUBE_OTHER || outputCase == OutputCase::OTHER_POST_CUBE) {
        int32_t index = outputCase == OutputCase::OTHER_POST_CUBE ? OUTPUT_INDEX : OUTPUT_1_INDEX;
        FUSION_PASS_CHECK(extendConv2D->UpdateOutputDesc(index, convDescInfo.outputDesc) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output%d tensor desc failed.", index),
                          return false);
    }

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::UpdateScaleReluDesc(GNodePtr postCube, GNode* extendConv2D,
                                                                 const int32_t getIndex, const int32_t updateIndex,
                                                                 const AscendString& name) const
{
    TensorDesc tensorDesc;
    if (postCube->GetInputDesc(getIndex, tensorDesc) != GRAPH_SUCCESS) {
        Shape shape;
        tensorDesc.Update(shape, FORMAT_ND, DT_FLOAT);
    }
    tensorDesc.SetFormat(ge::FORMAT_ND);
    tensorDesc.SetOriginFormat(ge::FORMAT_ND);
    FUSION_PASS_CHECK(extendConv2D->UpdateInputDesc(updateIndex, tensorDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update %s tensor desc failed.", name.GetString()),
                      return false);

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::AdjustLayout(const GraphPtr& graph, GNode& fusedNode)
{
    if (!ConvFusionUtilsPass::IsFuseSoc() || npuArch == Ops::Base::DAV_5162) {
        OP_LOGD(convDescInfo.nodeNameStr, "Current soc not support adjust layout, skip.");
        return true;
    }
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::AdjustConvInputLayout(graph, fusedNode), return false);

    FUSION_PASS_CHECK(outputCase != OutputCase::SINGLE,
                      OP_LOGD(convDescInfo.nodeNameStr, "Multi output, skip post transpose."), return true);

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
                      OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output0 desc failed."), return false);
    if (fusedNode.GetOutputsSize() > 1) {
        FUSION_PASS_CHECK(fusedNode.UpdateOutputDesc(OUTPUT_1_INDEX, transOutDesc) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Update ExtendConv2D output1 desc failed."), return false);
    }
    OP_LOGD(convDescInfo.nodeNameStr, "AdjustLayout post transpose success.");
    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::SetWeightFractalZDesc(TensorDesc& weightDesc) const
{
    auto oriDims = weightDesc.GetOriginShape().GetDims();
    FUSION_PASS_CHECK(
        oriDims.size() != FILTER_DIM_NUM,
        OP_LOGE(convDescInfo.nodeNameStr, "Filter origin shape size %zu is not %u.", oriDims.size(), FILTER_DIM_NUM),
        return false);

    Format oriFormat = weightDesc.GetOriginFormat();
    uint32_t nIdx = 0;
    uint32_t cIdx = 0;
    uint32_t hIdx = 0;
    uint32_t wIdx = 0;
    if (oriFormat == FORMAT_NCHW) {
        nIdx = NCHW_N_IDX;
        cIdx = NCHW_C_IDX;
        hIdx = NCHW_H_IDX;
        wIdx = NCHW_W_IDX;
        weightDesc.SetOriginFormat(FORMAT_NCHW);
    } else if (oriFormat == FORMAT_NHWC) {
        nIdx = NHWC_N_IDX;
        cIdx = NHWC_C_IDX;
        hIdx = NHWC_H_IDX;
        wIdx = NHWC_W_IDX;
        weightDesc.SetOriginFormat(FORMAT_NHWC);
    } else {
        OP_LOGE(convDescInfo.nodeNameStr, "Filter origin format is not supported, only support NCHW/NHWC.");
        return false;
    }

    weightDesc.SetFormat(FORMAT_FRACTAL_Z);
    int64_t c0 = (weightDesc.GetDataType() == DT_INT8 || weightDesc.GetDataType() == DT_UINT8) ? FRACTAL_Z_C0_INT8 :
                                                                                                 FRACTAL_Z_C0_FP16;
    std::vector<int64_t> fzShape(FILTER_DIM_NUM);
    fzShape[0] = oriDims[hIdx] * oriDims[wIdx] * ((oriDims[cIdx] + c0 - 1) / c0);
    fzShape[1] = (oriDims[nIdx] + FRACTAL_Z_N0 - 1) / FRACTAL_Z_N0;
    fzShape[2] = FRACTAL_Z_N0;
    fzShape[3] = c0;
    weightDesc.SetShape(Shape(fzShape));

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::InsertWeightReFormatOp(const GraphPtr& graph, GNode& extendConv2DNode,
                                                                    int64_t groups)
{
    auto weightPair = extendConv2DNode.GetInDataNodesAndPortIndexs(INPUT_FILTER_INDEX);
    FUSION_PASS_CHECK(weightPair.first == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Get weight producer failed."),
                      return false);
    GNode weightNode = *weightPair.first;
    int32_t weightPort = weightPair.second;

    AscendString weightType;
    FUSION_PASS_CHECK(weightNode.GetType(weightType) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get weight node type failed."), return false);
    if (weightType == REFORMAT_OP) {
        OP_LOGD(convDescInfo.nodeNameStr, "Weight has already been reformatted, no need to insert again.");
        return true;
    }

    TensorDesc reformatInDesc;
    FUSION_PASS_CHECK(weightNode.GetOutputDesc(weightPort, reformatInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get weight output desc failed."), return false);
    TensorDesc reformatOutDesc = reformatInDesc;
    FUSION_PASS_CHECK_NOLOG(!SetWeightFractalZDesc(reformatInDesc), return false);
    reformatInDesc.SetFormat(static_cast<Format>(
        ge::GetFormatFromSub(static_cast<int32_t>(FORMAT_FRACTAL_Z), static_cast<int32_t>(groups))));
    FUSION_PASS_CHECK_NOLOG(!SetWeightFractalZDesc(reformatOutDesc), return false);

    std::string reformatName = convDescInfo.nodeNameStr + "_const_fold_reformat";
    GNode reformatNode = es::CompliantNodeBuilder(graph.get())
                             .OpType(REFORMAT_OP.GetString())
                             .Name(reformatName.c_str())
                             .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();

    FUSION_PASS_CHECK(reformatNode.UpdateInputDesc(INPUT_INDEX_0, reformatInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update reformat input desc failed."), return false);
    FUSION_PASS_CHECK(reformatNode.UpdateOutputDesc(OUTPUT_INDEX, reformatOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update reformat output desc failed."), return false);
    FUSION_PASS_CHECK(reformatNode.SetAttr(GROUPS, groups) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set groups for reformat node failed."), return false);

    FUSION_PASS_CHECK(graph->RemoveEdge(weightNode, weightPort, extendConv2DNode, INPUT_FILTER_INDEX) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Remove edge between weight node and extendconv2d failed."),
                      return false);
    FUSION_PASS_CHECK(graph->AddDataEdge(weightNode, weightPort, reformatNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Add edge between weight node and reformat node failed."),
                      return false);
    FUSION_PASS_CHECK(
        graph->AddDataEdge(reformatNode, OUTPUT_INDEX, extendConv2DNode, INPUT_FILTER_INDEX) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Add edge between reformat node and extendconv2d failed."), return false);
    FUSION_PASS_CHECK(extendConv2DNode.UpdateInputDesc(INPUT_FILTER_INDEX, reformatOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update extendconv2d weight input desc failed."), return false);

    OP_LOGI(convDescInfo.nodeNameStr,
            "Insert ReFormat node between weight node and ExtendConv2D, physical NCHW/NHWC to FRACTAL_Z "
            "conversion is delegated to ReFormat.");

    return true;
}

bool Conv2DPostCubeToExtendConv2DFusionPass::InsertA16W8WeightRearrange(const GraphPtr& graph, GNode& extendConv2DNode)
{
    if (convDescInfo.fmapDtype == DT_INT8 || convDescInfo.filterDtype != DT_INT8) {
        OP_LOGD(convDescInfo.nodeNameStr, "No need to insert a16w8 weight rearrange op.");
        return true;
    }

    auto weightPreNode = extendConv2DNode.GetInDataNodesAndPortIndexs(INPUT_FILTER_INDEX).first;
    FUSION_PASS_CHECK(weightPreNode == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Get weight producer failed."),
                      return false);
    AscendString weightPreType;
    FUSION_PASS_CHECK(weightPreNode->GetType(weightPreType) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get weight producer type failed."), return false);
    if (weightPreType == REARRANGE_A16W8_WEIGHT_FZ_OP) {
        OP_LOGD(convDescInfo.nodeNameStr, "Weight has already been rearranged, no need to insert again.");
        return true;
    }

    int64_t groups = 1;
    (void)extendConv2DNode.GetAttr(GROUPS, groups);

    if (ConvFusionUtilsPass::IsFuseSoc()) {
        FUSION_PASS_CHECK_NOLOG(!InsertWeightReFormatOp(graph, extendConv2DNode, groups), return false);
    }

    TensorDesc weightDesc;
    FUSION_PASS_CHECK(extendConv2DNode.GetInputDesc(INPUT_FILTER_INDEX, weightDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get weight desc failed."), return false);

    std::string rearrangeName = convDescInfo.nodeNameStr + "_rearrange_a16w8_weight";
    GNode rearrangeNode = es::CompliantNodeBuilder(graph.get())
                              .OpType(REARRANGE_A16W8_WEIGHT_FZ_OP.GetString())
                              .Name(rearrangeName.c_str())
                              .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                              .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .Build();

    TensorDesc rearrangeInDesc = weightDesc;
    FUSION_PASS_CHECK(rearrangeNode.UpdateInputDesc(INPUT_INDEX_0, rearrangeInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update rearrange input desc failed."), return false);

    TensorDesc rearrangeOutDesc = weightDesc;
    rearrangeOutDesc.SetDataType(DT_INT8);
    FUSION_PASS_CHECK(rearrangeNode.UpdateOutputDesc(OUTPUT_INDEX, rearrangeOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update rearrange output desc failed."), return false);

    FUSION_PASS_CHECK(rearrangeNode.SetAttr(GROUPS, groups) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set groups for rearrange node failed."), return false);

    auto weightPair = extendConv2DNode.GetInDataNodesAndPortIndexs(INPUT_FILTER_INDEX);
    FUSION_PASS_CHECK(weightPair.first == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Get weight producer failed."),
                      return false);
    auto weightProducer = weightPair.first;
    int32_t weightPort = weightPair.second;

    FUSION_PASS_CHECK(
        graph->RemoveEdge(*weightProducer, weightPort, extendConv2DNode, INPUT_FILTER_INDEX) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Remove old weight edge failed."), return false);
    FUSION_PASS_CHECK(graph->AddDataEdge(*weightProducer, weightPort, rearrangeNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Add weight->rearrange edge failed."), return false);
    FUSION_PASS_CHECK(
        graph->AddDataEdge(rearrangeNode, OUTPUT_INDEX, extendConv2DNode, INPUT_FILTER_INDEX) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Add rearrange->extendconv2d edge failed."), return false);

    FUSION_PASS_CHECK(extendConv2DNode.UpdateInputDesc(INPUT_FILTER_INDEX, rearrangeOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update weight input desc failed."), return false);

    OP_LOGD(convDescInfo.nodeNameStr, "Insert a16w8 weight rearrange node success.");
    return true;
}

using Conv2DFixPipeToExtendConv2DFusionPass = Conv2DPostCubeToExtendConv2DFusionPass;
#if GE_COMPILER_VERSION_NUM >= 90100000U
REG_FUSION_PASS(Conv2DFixPipeToExtendConv2DFusionPass).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace Ops
