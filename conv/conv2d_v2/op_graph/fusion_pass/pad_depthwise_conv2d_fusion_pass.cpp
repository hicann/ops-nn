/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pad_depthwise_conv2d_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "graph/graph.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

#if GE_COMPILER_VERSION_NUM >= 90100000U
#include "ge/fusion/graph_fuse_inspector_utils.h"
#endif

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace PadDepthwiseConv2dFusion;
using namespace ge;
using namespace fusion;

void PadDepthwiseConv2dFusionPass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    convDescInfo = ConvDescInfo();
    padNode = nullptr;
    padXProducer = nullptr;
    padXProducerOutIdx = 0;
    paddings.clear();
    pads.clear();
    paddingsT = 0;
    paddingsB = 0;
    paddingsL = 0;
    paddingsR = 0;
    isAscend950 = false;
}

bool PadDepthwiseConv2dFusionPass::CheckMatchStructure(const GNode& matchNode)
{
    auto padPair = matchNode.GetInDataNodesAndPortIndexs(CONV_FMAP_INPUT_INDEX);
    padNode = padPair.first;
    FUSION_PASS_CHECK(padNode == nullptr, OP_LOGD(FUSION_NAME, "Get input producer of DepthwiseConv2D failed."),
                      return false);

    AscendString padType;
    FUSION_PASS_CHECK(padNode->GetType(padType) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get DepthwiseConv2D input producer type failed."), return false);
    FUSION_PASS_CHECK(padType != PAD, OP_LOGD(FUSION_NAME, "DepthwiseConv2D input producer is not Pad, no fusion."),
                      return false);

    auto padXPair = padNode->GetInDataNodesAndPortIndexs(PAD_X_INPUT_INDEX);
    padXProducer = padXPair.first;
    padXProducerOutIdx = padXPair.second;
    FUSION_PASS_CHECK(padXProducer == nullptr, OP_LOGD(FUSION_NAME, "Get Pad input producer failed."), return false);
    return true;
}

bool PadDepthwiseConv2dFusionPass::IsAscend950() const
{
    NpuArch curArch = NpuArch::DAV_RESV;
    return ConvFusionUtilsPass::CheckSocList(SUPPORT_SOC_LIST, curArch) && curArch == NpuArch::DAV_3510;
}

bool PadDepthwiseConv2dFusionPass::CheckPadDynamicShape() const
{
    TensorDesc inputDesc;
    FUSION_PASS_CHECK(padNode->GetInputDesc(PAD_X_INPUT_INDEX, inputDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Pad input desc failed."), return false);
    FUSION_PASS_CHECK(ConvFusionUtilsPass::IsUnknownShape(inputDesc),
                      OP_LOGD(convDescInfo.nodeNameStr, "Pad input is unknown shape, no fusion."), return false);

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(padNode->GetOutputDesc(OUTPUT_INDEX, outputDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Pad output desc failed."), return false);
    FUSION_PASS_CHECK(ConvFusionUtilsPass::IsUnknownShape(outputDesc),
                      OP_LOGD(convDescInfo.nodeNameStr, "Pad output is unknown shape, no fusion."), return false);
    return true;
}

bool PadDepthwiseConv2dFusionPass::GetPaddingsFromConst()
{
    paddings.clear();
    Tensor paddingsTensor;
    FUSION_PASS_CHECK(padNode->GetInputConstData(PAD_PADDINGS_INPUT_INDEX, paddingsTensor) != GRAPH_SUCCESS,
                      OP_LOGW(convDescInfo.nodeNameStr, "Get const value of paddings failed."), return false);

    TensorDesc paddingsDesc;
    FUSION_PASS_CHECK(padNode->GetInputDesc(PAD_PADDINGS_INPUT_INDEX, paddingsDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get paddings input desc failed."), return false);
    FUSION_PASS_CHECK(paddingsTensor.GetData() == nullptr,
                      OP_LOGW(convDescInfo.nodeNameStr, "Get paddings const data failed."), return false);

    std::vector<int64_t> padValue;
    const void* dataPtr = paddingsTensor.GetData();
    if (paddingsDesc.GetDataType() == ge::DT_INT32) {
        const int32_t* int32Ptr = static_cast<const int32_t*>(dataPtr);
        size_t size = paddingsTensor.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i) {
            padValue.push_back(static_cast<int64_t>(int32Ptr[i]));
        }
    } else if (paddingsDesc.GetDataType() == ge::DT_INT64) {
        const int64_t* int64Ptr = static_cast<const int64_t*>(dataPtr);
        size_t size = paddingsTensor.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < size; ++i) {
            padValue.push_back(int64Ptr[i]);
        }
    } else {
        OP_LOGW(convDescInfo.nodeNameStr, "Padding dtype is not int32 or int64, can not fusion.");
        return false;
    }

    for (size_t i = 1; i < padValue.size(); i += static_cast<size_t>(DIRECTION_COUNT)) {
        paddings.push_back({padValue[i - 1], padValue[i]});
    }
    FUSION_PASS_CHECK(paddings.size() < static_cast<size_t>(DIM_NUM4),
                      OP_LOGI(convDescInfo.nodeNameStr, "The number of paddings not valid, can not fusion."),
                      return false);
    for (size_t i = 0; i < static_cast<size_t>(DIM_NUM4); ++i) {
        FUSION_PASS_CHECK(paddings[i].size() < static_cast<size_t>(DIRECTION_COUNT),
                          OP_LOGI(convDescInfo.nodeNameStr, "The number of paddings not valid, can not fusion."),
                          return false);
    }
    return true;
}

bool PadDepthwiseConv2dFusionPass::ExtractPaddingByFormat()
{
    TensorDesc padInDesc;
    FUSION_PASS_CHECK(padNode->GetInputDesc(PAD_X_INPUT_INDEX, padInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Pad input desc failed."), return false);

    Format primaryFormat = padInDesc.GetOriginFormat();
    if (primaryFormat == ge::FORMAT_NCHW) {
        FUSION_PASS_CHECK(paddings[NCHW_NC_PAIR_INDEX][0] != 0 || paddings[NCHW_NC_PAIR_INDEX][1] != 0 ||
                              paddings[NCHW_C_AXIS_INDEX][0] != 0 || paddings[NCHW_C_AXIS_INDEX][1] != 0,
                          OP_LOGI(convDescInfo.nodeNameStr, "Pad and DepthwiseConv2d fusion can only on H and W."),
                          return false);
        paddingsT = paddings[NCHW_PAD_H_INDEX][0];
        paddingsB = paddings[NCHW_PAD_H_INDEX][1];
        paddingsL = paddings[NCHW_PAD_W_INDEX][0];
        paddingsR = paddings[NCHW_PAD_W_INDEX][1];
    } else if (primaryFormat == ge::FORMAT_NHWC) {
        FUSION_PASS_CHECK(paddings[NHWC_N_AXIS_INDEX][0] != 0 || paddings[NHWC_N_AXIS_INDEX][1] != 0 ||
                              paddings[NHWC_C_AXIS_INDEX][0] != 0 || paddings[NHWC_C_AXIS_INDEX][1] != 0,
                          OP_LOGI(convDescInfo.nodeNameStr, "Pad and DepthwiseConv2d fusion can only on H and W."),
                          return false);
        paddingsT = paddings[NHWC_PAD_H_INDEX][0];
        paddingsB = paddings[NHWC_PAD_H_INDEX][1];
        paddingsL = paddings[NHWC_PAD_W_INDEX][0];
        paddingsR = paddings[NHWC_PAD_W_INDEX][1];
    } else {
        OP_LOGI(convDescInfo.nodeNameStr, "Pad input format is not NCHW or NHWC, can not fusion.");
        return false;
    }
    return true;
}

bool PadDepthwiseConv2dFusionPass::CheckPaddingRange() const
{
    FUSION_PASS_CHECK(
        paddingsT < MIN_PADDING_VALUE || paddingsT > MAX_PADDING_VALUE || paddingsB < MIN_PADDING_VALUE ||
            paddingsB > MAX_PADDING_VALUE || paddingsL < MIN_PADDING_VALUE || paddingsL > MAX_PADDING_VALUE ||
            paddingsR < MIN_PADDING_VALUE || paddingsR > MAX_PADDING_VALUE,
        OP_LOGI(convDescInfo.nodeNameStr, "Paddings value not in [0,255], can not fusion."), return false);
    return true;
}

bool PadDepthwiseConv2dFusionPass::CheckFilterVsPadding(const GNode& convNode) const
{
    (void)convNode;
    // filter desc 由基类 Run() 中的 GetConvDescInfo 填充（conv_fusion_base_pass.cpp:63）
    const TensorDesc& filterDesc = convDescInfo.filterDesc;
    FUSION_PASS_CHECK(filterDesc.GetShape().GetDimNum() < static_cast<size_t>(DIM_NUM4),
                      OP_LOGI(convDescInfo.nodeNameStr, "Filter shape is not 4D, can not fusion."), return false);

    Format filterFormat = filterDesc.GetOriginFormat();
    if (filterFormat == ge::FORMAT_NCHW) {
        FUSION_PASS_CHECK(filterDesc.GetShape().GetDim(NCHW_H_INDEX) <= paddingsT ||
                              filterDesc.GetShape().GetDim(NCHW_H_INDEX) <= paddingsB,
                          OP_LOGI(convDescInfo.nodeNameStr, "Filter_H more than pad_H, can not fusion."), return false);
    } else if (filterFormat == ge::FORMAT_HWCN) {
        FUSION_PASS_CHECK(filterDesc.GetShape().GetDim(HWCN_H_INDEX) <= paddingsT ||
                              filterDesc.GetShape().GetDim(HWCN_H_INDEX) <= paddingsB,
                          OP_LOGI(convDescInfo.nodeNameStr, "Filter_H more than pad_H, can not fusion."), return false);
    } else {
        OP_LOGI(convDescInfo.nodeNameStr, "Filter format is not NCHW or HWCN, can not fusion.");
        return false;
    }
    return true;
}

bool PadDepthwiseConv2dFusionPass::CheckPadOutputsAllDepthwise() const
{
    auto padConsumers = padNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (const auto& consumerPair : padConsumers) {
        FUSION_PASS_CHECK(consumerPair.first == nullptr,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get Pad output consumer failed."), return false);
        AscendString consumerType;
        FUSION_PASS_CHECK(consumerPair.first->GetType(consumerType) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get Pad output consumer type failed."), return false);
        FUSION_PASS_CHECK(consumerType != DEPTHWISE_CONV2D,
                          OP_LOGI(convDescInfo.nodeNameStr, "Output node is not DepthwiseConv2D, can not fusion."),
                          return false);
    }
    return true;
}

bool PadDepthwiseConv2dFusionPass::BuildPadsVector()
{
    pads.clear();
    pads.push_back(paddingsT);
    pads.push_back(paddingsB);
    pads.push_back(paddingsL);
    pads.push_back(paddingsR);
    return true;
}

bool PadDepthwiseConv2dFusionPass::MeetRequirements(const GNode& convNode)
{
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do PadDepthwiseConv2dFusionPass.");

    isAscend950 = IsAscend950();

    FUSION_PASS_CHECK_NOLOG(!CheckPadDynamicShape(), return false);

    AscendString paddingMode;
    FUSION_PASS_CHECK(convNode.GetAttr(PADDING, paddingMode) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get DepthwiseConv2D padding attr failed."), return false);
    FUSION_PASS_CHECK(
        paddingMode != VALID_PADDING,
        OP_LOGI(convDescInfo.nodeNameStr, "PadDepthwiseConv2dFusion can only support VALID padding mode."),
        return false);

    int64_t convCount = 0;
    auto padConsumers = padNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (const auto& consumerPair : padConsumers) {
        if (consumerPair.first == nullptr) {
            continue;
        }
        AscendString consumerType;
        FUSION_PASS_CHECK(consumerPair.first->GetType(consumerType) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get Pad output consumer type failed."), return false);
        if (consumerType == DEPTHWISE_CONV2D) {
            convCount++;
        }
    }
    FUSION_PASS_CHECK(
        convCount > 1,
        OP_LOGI(convDescInfo.nodeNameStr, "Pad node has multiple depthwise_conv2d outputs, can not fusion."),
        return false);

    FUSION_PASS_CHECK_NOLOG(!GetPaddingsFromConst(), return false);
    FUSION_PASS_CHECK_NOLOG(!ExtractPaddingByFormat(), return false);
    FUSION_PASS_CHECK_NOLOG(!CheckPaddingRange(), return false);

    if (!isAscend950) {
        FUSION_PASS_CHECK_NOLOG(!CheckFilterVsPadding(convNode), return false);
    }

    FUSION_PASS_CHECK(!padNode->GetOutControlNodes().empty(),
                      OP_LOGI(convDescInfo.nodeNameStr, "Pad node has control edge, can not fusion."), return false);
    FUSION_PASS_CHECK_NOLOG(!CheckPadOutputsAllDepthwise(), return false);
    FUSION_PASS_CHECK_NOLOG(!BuildPadsVector(), return false);

    return true;
}

std::set<AscendString> PadDepthwiseConv2dFusionPass::GetNodeTypes() const { return {DEPTHWISE_CONV2D}; }

void PadDepthwiseConv2dFusionPass::PrintGraphStructure() const
{
    OP_LOGI(convDescInfo.nodeNameStr, "PadDepthwiseConv2d fusion success: pads=[%lld,%lld,%lld,%lld].", paddingsT,
            paddingsB, paddingsL, paddingsR);
}

bool PadDepthwiseConv2dFusionPass::ConvFusionReplaceImpl(GraphPtr& graph, GNode& convNode,
                                                         CustomPassContext& passContext)
{
    FUSION_PASS_CHECK(graph == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Graph is nullptr."), return false);

    TensorDesc padXDesc;
    FUSION_PASS_CHECK(padNode->GetInputDesc(PAD_X_INPUT_INDEX, padXDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Pad input desc failed."), return false);

    std::vector<GNode> nodesBeforeFuse = {convNode, *padNode};
    AscendString failedReason;
#if GE_COMPILER_VERSION_NUM >= 90100000U
    FUSION_PASS_CHECK(!ge::fusion::GraphFuseInspectorUtils::CanFuse(nodesBeforeFuse, failedReason),
                      OP_LOGD(convDescInfo.nodeNameStr, "CanFuse failed, reason: %s.", failedReason.GetString()),
                      return false);
#endif

    FUSION_PASS_CHECK(convNode.UpdateInputDesc(CONV_FMAP_INPUT_INDEX, padXDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update DepthwiseConv2D input desc failed."), return false);

    FUSION_PASS_CHECK(graph->RemoveEdge(*padNode, OUTPUT_INDEX, convNode, CONV_FMAP_INPUT_INDEX) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Remove Pad-DepthwiseConv2D edge failed."), return false);
    FUSION_PASS_CHECK(
        graph->AddDataEdge(*padXProducer, padXProducerOutIdx, convNode, CONV_FMAP_INPUT_INDEX) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Add edge between Pad input producer and DepthwiseConv2D failed."),
        return false);

    FUSION_PASS_CHECK(convNode.SetAttr(PADS, pads) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set pads attr failed."), return false);
    AscendString samePadding = SAME_PADDING;
    FUSION_PASS_CHECK(convNode.SetAttr(PADDING, samePadding) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set padding attr failed."), return false);

#if GE_COMPILER_VERSION_NUM >= 90100000U
    FUSION_PASS_CHECK(
        ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, nodesBeforeFuse, passContext) != SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "ReportFuse failed."), return false);
#endif

    auto remainingConsumers = padNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (const auto& consumerPair : remainingConsumers) {
        FUSION_PASS_CHECK(consumerPair.first == nullptr,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get Pad output consumer failed."), return false);
        FUSION_PASS_CHECK(
            graph->RemoveEdge(*padNode, OUTPUT_INDEX, *consumerPair.first, consumerPair.second) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Remove Pad output edge failed."), return false);
    }

    FUSION_PASS_CHECK(graph->RemoveNode(*padNode) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Remove Pad node failed."), return false);

    return true;
}

#if GE_COMPILER_VERSION_NUM >= 90100000U
REG_FUSION_PASS(PadDepthwiseConv2dFusionPass).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace Ops
