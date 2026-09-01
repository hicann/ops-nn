/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pad_conv2d_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "graph/graph.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

#if GE_COMPILER_VERSION_NUM >= 90100000U
#include "ge/fusion/graph_fuse_inspector_utils.h"
#endif

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace PadConv2dFusion;
using namespace ge;

void PadConv2dFusionPass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    convDescInfo = ConvDescInfo();
    hasDn2Nz = false;
    padNode = nullptr;
    paddings.clear();
    combinedPads.clear();
    backpropFilterNode = nullptr;
    backpropInputNode = nullptr;
    sliceNode = nullptr;
}

bool PadConv2dFusionPass::CheckMatchStructure(const GNode& matchNode)
{
    auto padPair = matchNode.GetInDataNodesAndPortIndexs(INPUT_FMAP_INDEX);
    FUSION_PASS_CHECK_NOLOG(padPair.first == nullptr, return false);
    padNode = padPair.first;

    AscendString padType;
    FUSION_PASS_CHECK_NOLOG(padNode->GetType(padType) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK(padType != PAD_OP && padType != PADV3_OP,
                      OP_LOGD(FUSION_NAME, "Conv2D input producer is not Pad or PadV3, no fusion."), return false);

    return true;
}

std::set<AscendString> PadConv2dFusionPass::GetNodeTypes() const { return {CONV2D}; }

void PadConv2dFusionPass::InitPlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        PLATFORM_INFO_OK) {
        OP_LOGW(convDescInfo.nodeNameStr, "Get platform info failed, keep default arch and no dn2nz.");
        return;
    }

    const auto& intrinsicMap = platformInfo.ai_core_intrinsic_dtype_map;
    hasDn2Nz = intrinsicMap.find(DN2NZ_INTRINSIC) != intrinsicMap.end();

    auto socIter = SOC_LIST.find(platformInfo.str_info.short_soc_version);
    if (socIter != SOC_LIST.end()) {
        npuArch = socIter->second;
    }
}

bool PadConv2dFusionPass::CheckPadControlEdges(const GNode& convNode) const
{
    auto outCtrlNodes = padNode->GetOutControlNodes();
    FUSION_PASS_CHECK_NOLOG(outCtrlNodes.empty(), return true);

    AscendString convName;
    FUSION_PASS_CHECK(convNode.GetName(convName) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Conv2D name failed."), return false);
    AscendString dwName;
    if (backpropFilterNode != nullptr) {
        FUSION_PASS_CHECK(backpropFilterNode->GetName(dwName) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get Conv2DBackpropFilterD name failed."), return false);
    }

    for (const auto& ctrlNode : outCtrlNodes) {
        FUSION_PASS_CHECK(ctrlNode == nullptr,
                          OP_LOGD(convDescInfo.nodeNameStr, "Pad control edge peer is null, no fusion."), return false);
        AscendString ctrlName;
        AscendString ctrlType;
        FUSION_PASS_CHECK(ctrlNode->GetName(ctrlName) != GRAPH_SUCCESS || ctrlNode->GetType(ctrlType) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get Pad control edge peer name or type failed."),
                          return false);
        bool linkToFusedCube = ctrlName == convName || (backpropFilterNode != nullptr && ctrlName == dwName) ||
                               ctrlType == CONV2D_BACKPROP_INPUT_D;
        FUSION_PASS_CHECK(
            !linkToFusedCube,
            OP_LOGD(convDescInfo.nodeNameStr, "Pad control edge links to [%s], no fusion.", ctrlName.GetString()),
            return false);
    }
    return true;
}

bool PadConv2dFusionPass::CheckPadDynamicShape() const
{
    for (size_t idx = 0; idx < padNode->GetInputsSize(); ++idx) {
        TensorDesc inputDesc;
        FUSION_PASS_CHECK(padNode->GetInputDesc(static_cast<int32_t>(idx), inputDesc) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get Pad input desc %zu failed.", idx), return false);
        FUSION_PASS_CHECK(ConvFusionUtilsPass::IsUnknownShape(inputDesc),
                          OP_LOGD(convDescInfo.nodeNameStr, "Pad input %zu is unknown shape, no fusion.", idx),
                          return false);
    }

    for (size_t idx = 0; idx < padNode->GetOutputsSize(); ++idx) {
        TensorDesc outputDesc;
        FUSION_PASS_CHECK(padNode->GetOutputDesc(static_cast<int32_t>(idx), outputDesc) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get Pad output desc %zu failed.", idx), return false);
        FUSION_PASS_CHECK(ConvFusionUtilsPass::IsUnknownShape(outputDesc),
                          OP_LOGD(convDescInfo.nodeNameStr, "Pad output %zu is unknown shape, no fusion.", idx),
                          return false);
    }
    return true;
}

bool PadConv2dFusionPass::ValidatePadTopology(const GNode& convNode)
{
    int64_t convCount = 0;
    int64_t dwCount = 0;
    for (const auto& consumerPair : padNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX)) {
        FUSION_PASS_CHECK(consumerPair.first == nullptr,
                          OP_LOGD(convDescInfo.nodeNameStr, "Pad consumer is null, no fusion."), return false);
        AscendString consumerType;
        FUSION_PASS_CHECK(consumerPair.first->GetType(consumerType) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Get Pad consumer type failed."), return false);
        if (consumerType == CONV2D) {
            convCount++;
        } else if (consumerType == CONV2D_BACKPROP_FILTER_D) {
            dwCount++;
            backpropFilterNode = consumerPair.first;
        } else {
            OP_LOGD(convDescInfo.nodeNameStr, "Pad consumer type [%s] is not cube node, no fusion.",
                    consumerType.GetString());
            return false;
        }
    }
    FUSION_PASS_CHECK(
        convCount > SINGLE_CUBE_CNT || dwCount > SINGLE_CUBE_CNT,
        OP_LOGD(convDescInfo.nodeNameStr, "Pad has %ld Conv2D and %ld dw consumers, no fusion.", convCount, dwCount),
        return false);

    FUSION_PASS_CHECK_NOLOG(!CheckPadControlEdges(convNode), return false);
    FUSION_PASS_CHECK_NOLOG(!CheckPadDynamicShape(), return false);

    return true;
}

bool PadConv2dFusionPass::CheckPadV3ConstantValue() const
{
    Tensor constTensor;
    FUSION_PASS_CHECK_NOLOG(padNode->GetInputConstData(CONSTANT_VALUES_INPUT_INDEX, constTensor) != GRAPH_SUCCESS,
                            return true);

    FUSION_PASS_CHECK(constTensor.GetDataType() != DT_FLOAT,
                      OP_LOGD(convDescInfo.nodeNameStr, "PadV3 constant_values dtype is not float, no fusion."),
                      return false);

    const uint8_t* constData = constTensor.GetData();
    size_t valueNum = constTensor.GetSize() / sizeof(float);
    FUSION_PASS_CHECK(constData == nullptr || valueNum == 0,
                      OP_LOGD(convDescInfo.nodeNameStr, "PadV3 constant_values is empty, no fusion."), return false);

    const float* floatData = static_cast<const float*>(static_cast<const void*>(constData));
    FUSION_PASS_CHECK(
        static_cast<int32_t>(floatData[INDEX_0]) != 0,
        OP_LOGD(convDescInfo.nodeNameStr, "PadV3 constant_values [%f] is not 0, no fusion.", floatData[INDEX_0]),
        return false);
    return true;
}

bool PadConv2dFusionPass::ExtractPaddingsData(const Tensor& padTensor, std::vector<int64_t>& padValue) const
{
    const uint8_t* padData = padTensor.GetData();
    FUSION_PASS_CHECK(padData == nullptr, OP_LOGD(convDescInfo.nodeNameStr, "Paddings data is null, no fusion."),
                      return false);

    DataType padDtype = padTensor.GetDataType();
    if (padDtype == DT_INT32) {
        const int32_t* intData = static_cast<const int32_t*>(static_cast<const void*>(padData));
        size_t valueSize = padTensor.GetSize() / sizeof(int32_t);
        for (size_t idx = 0; idx < valueSize; ++idx) {
            padValue.emplace_back(static_cast<int64_t>(intData[idx]));
        }
        return true;
    }
    if (padDtype == DT_INT64) {
        const int64_t* longData = static_cast<const int64_t*>(static_cast<const void*>(padData));
        size_t valueSize = padTensor.GetSize() / sizeof(int64_t);
        for (size_t idx = 0; idx < valueSize; ++idx) {
            padValue.emplace_back(longData[idx]);
        }
        return true;
    }

    OP_LOGD(convDescInfo.nodeNameStr, "Paddings dtype [%s] is not int32 or int64, no fusion.",
            GeDtypeToString(padDtype).c_str());
    return false;
}

bool PadConv2dFusionPass::CheckPadV3AndExtractPaddings()
{
    AscendString padType;
    FUSION_PASS_CHECK(padNode->GetType(padType) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Pad node type failed."), return false);

    bool paddingsContiguous = true;
    if (padType == PADV3_OP) {
        AscendString padMode;
        FUSION_PASS_CHECK(padNode->GetAttr(MODE, padMode) != GRAPH_SUCCESS || padMode != CONSTANT,
                          OP_LOGD(convDescInfo.nodeNameStr, "PadV3 mode is not constant, no fusion."), return false);
        FUSION_PASS_CHECK_NOLOG(!CheckPadV3ConstantValue(), return false);
        padNode->GetAttr(PADDINGS_CONTIGUOUS, paddingsContiguous);
    }

    Tensor padTensor;
    FUSION_PASS_CHECK(padNode->GetInputConstData(PADDINGS_INPUT_INDEX, padTensor) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "Get paddings const data failed, no fusion."), return false);
    std::vector<int64_t> padValue;
    FUSION_PASS_CHECK_NOLOG(!ExtractPaddingsData(padTensor, padValue), return false);

    if (paddingsContiguous) {
        for (size_t idx = 1; idx < padValue.size(); idx += DIRECTION_COUNT) {
            paddings.emplace_back(std::vector<int64_t>{padValue[idx - 1], padValue[idx]});
        }
    } else {
        size_t rank = padValue.size() / DIRECTION_COUNT;
        for (size_t idx = 0; idx < rank; ++idx) {
            paddings.emplace_back(std::vector<int64_t>{padValue[idx], padValue[idx + rank]});
        }
    }
    return true;
}

bool PadConv2dFusionPass::CheckFilterHeight() const
{
    FUSION_PASS_CHECK_NOLOG(npuArch == NpuArch::DAV_3510, return true);

    Format filterFormat = convDescInfo.filterDesc.GetOriginFormat();
    size_t filterHIdx = INDEX_0;
    if (filterFormat == FORMAT_NCHW) {
        filterHIdx = INDEX_2;
    } else if (filterFormat != FORMAT_HWCN) {
        return true;
    }

    std::vector<int64_t> filterShape = convDescInfo.filterDesc.GetOriginShape().GetDims();
    FUSION_PASS_CHECK(filterShape.size() != DIM_NUM4, OP_LOGD(convDescInfo.nodeNameStr, "Filter is not 4D, no fusion."),
                      return false);
    FUSION_PASS_CHECK(
        filterShape[filterHIdx] <= combinedPads[INDEX_0] || filterShape[filterHIdx] <= combinedPads[INDEX_1],
        OP_LOGD(convDescInfo.nodeNameStr, "Filter H [%ld] is not greater than pad H [%ld, %ld], no fusion.",
                filterShape[filterHIdx], combinedPads[INDEX_0], combinedPads[INDEX_1]),
        return false);
    return true;
}

bool PadConv2dFusionPass::ValidateAndComputePads(const GNode& convNode)
{
    std::vector<int64_t> convPads;
    FUSION_PASS_CHECK(convNode.GetAttr(PADS, convPads) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "Get Conv2D pads failed, no fusion."), return false);
    FUSION_PASS_CHECK(convPads.size() != DIM_NUM4,
                      OP_LOGD(convDescInfo.nodeNameStr, "Conv2D pads size is not 4, no fusion."), return false);
    FUSION_PASS_CHECK(paddings.size() < DIM_NUM4,
                      OP_LOGD(convDescInfo.nodeNameStr, "Paddings size [%zu] is invalid, no fusion.", paddings.size()),
                      return false);

    for (size_t idx = INDEX_0; idx < DIM_NUM4; ++idx) {
        FUSION_PASS_CHECK(convPads[idx] < MIN_PADDING_VALUE || paddings[idx].size() < DIRECTION_COUNT,
                          OP_LOGD(convDescInfo.nodeNameStr, "Conv2D pads or paddings %zu is invalid, no fusion.", idx),
                          return false);
    }

    TensorDesc padInputDesc;
    FUSION_PASS_CHECK(padNode->GetInputDesc(INPUT_FMAP_INDEX, padInputDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Pad input desc failed."), return false);
    Format padFormat = padInputDesc.GetOriginFormat();
    size_t cIdx = INDEX_1;
    size_t hIdx = INDEX_2;
    size_t wIdx = INDEX_3;
    if (padFormat == FORMAT_NHWC) {
        hIdx = INDEX_1;
        wIdx = INDEX_2;
        cIdx = INDEX_3;
    } else if (padFormat != FORMAT_NCHW) {
        OP_LOGD(convDescInfo.nodeNameStr, "Pad input format is not NCHW or NHWC, no fusion.");
        return false;
    }

    FUSION_PASS_CHECK(paddings[INDEX_0][INDEX_0] != 0 || paddings[INDEX_0][INDEX_1] != 0 ||
                          paddings[cIdx][INDEX_0] != 0 || paddings[cIdx][INDEX_1] != 0,
                      OP_LOGD(convDescInfo.nodeNameStr, "Pad on batch or channel axis is not 0, no fusion."),
                      return false);

    std::vector<int64_t> padSides = {paddings[hIdx][INDEX_0], paddings[hIdx][INDEX_1], paddings[wIdx][INDEX_0],
                                     paddings[wIdx][INDEX_1]};
    for (size_t idx = INDEX_0; idx < DIM_NUM4; ++idx) {
        combinedPads.emplace_back(padSides[idx] + convPads[idx]);
        FUSION_PASS_CHECK(
            padSides[idx] < MIN_PADDING_VALUE || combinedPads[idx] > MAX_PADDING_VALUE,
            OP_LOGD(convDescInfo.nodeNameStr, "Pad value [%ld] < %ld or combined pad [%ld] > %ld, no fusion.",
                    padSides[idx], MIN_PADDING_VALUE, combinedPads[idx], MAX_PADDING_VALUE),
            return false);
    }

    FUSION_PASS_CHECK_NOLOG(!CheckFilterHeight(), return false);
    return true;
}

bool PadConv2dFusionPass::DiscoverAndCheckBackward()
{
    FUSION_PASS_CHECK_NOLOG(backpropFilterNode == nullptr, return true);
    FUSION_PASS_CHECK(npuArch == NpuArch::DAV_3510,
                      OP_LOGD(convDescInfo.nodeNameStr, "Backward cube node is not supported on this soc, no fusion."),
                      return false);

    auto bnGradPair = backpropFilterNode->GetInDataNodesAndPortIndexs(DW_OUT_BACKPROP_INDEX);
    FUSION_PASS_CHECK_NOLOG(bnGradPair.first == nullptr, return true);
    AscendString bnGradType;
    FUSION_PASS_CHECK_NOLOG(bnGradPair.first->GetType(bnGradType) != GRAPH_SUCCESS, return true);
    FUSION_PASS_CHECK_NOLOG(bnGradType != BN_TRAINING_REDUCE_GRAD, return true);

    for (const auto& consumerPair : bnGradPair.first->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX)) {
        AscendString consumerType;
        if (consumerPair.first == nullptr || consumerPair.first->GetType(consumerType) != GRAPH_SUCCESS) {
            continue;
        }
        if (consumerType == CONV2D_BACKPROP_INPUT_D) {
            backpropInputNode = consumerPair.first;
        }
    }
    FUSION_PASS_CHECK_NOLOG(backpropInputNode == nullptr, return true);

    auto dxConsumers = backpropInputNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK_NOLOG(dxConsumers.size() != SINGLE_CONSUMER_CNT, return true);
    FUSION_PASS_CHECK_NOLOG(dxConsumers[INDEX_0].first == nullptr, return true);
    AscendString dxConsumerType;
    FUSION_PASS_CHECK_NOLOG(dxConsumers[INDEX_0].first->GetType(dxConsumerType) != GRAPH_SUCCESS, return true);
    if (dxConsumerType == SLICE_OP || dxConsumerType == SLICE_D_OP) {
        sliceNode = dxConsumers[INDEX_0].first;
    }
    return true;
}

bool PadConv2dFusionPass::MeetRequirements(const GNode& convNode)
{
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do PadConv2dFusionPass.");
    InitPlatform();

    FUSION_PASS_CHECK_NOLOG(!ValidatePadTopology(convNode), return false);
    FUSION_PASS_CHECK_NOLOG(!CheckPadV3AndExtractPaddings(), return false);
    FUSION_PASS_CHECK_NOLOG(!ValidateAndComputePads(convNode), return false);
    FUSION_PASS_CHECK_NOLOG(!DiscoverAndCheckBackward(), return false);

    return true;
}

void PadConv2dFusionPass::PrintGraphStructure() const
{
    OP_LOGI(FUSION_NAME,
            "%s PadConv2dFusionPass done, pads [%ld, %ld, %ld, %ld], dn2nz [%d], with dw [%d], "
            "with slice [%d].",
            convDescInfo.nodeNameStr.c_str(), combinedPads[INDEX_0], combinedPads[INDEX_1], combinedPads[INDEX_2],
            combinedPads[INDEX_3], static_cast<int32_t>(hasDn2Nz), static_cast<int32_t>(backpropFilterNode != nullptr),
            static_cast<int32_t>(sliceNode != nullptr));
}

bool PadConv2dFusionPass::HandleBackwardPath(Graph& graph)
{
    FUSION_PASS_CHECK_NOLOG(sliceNode == nullptr, return true);

    TensorDesc sliceOutDesc;
    FUSION_PASS_CHECK(sliceNode->GetOutputDesc(OUTPUT_INDEX, sliceOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Slice output desc failed."), return false);
    FUSION_PASS_CHECK(backpropInputNode->UpdateOutputDesc(OUTPUT_INDEX, sliceOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update dx output desc failed."), return false);
    FUSION_PASS_CHECK(graph.RemoveEdge(*backpropInputNode, OUTPUT_INDEX, *sliceNode, INPUT_FMAP_INDEX) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Remove dx-Slice edge failed."), return false);

    std::vector<int64_t> dxPads = combinedPads;
    AscendString dxPadding = SAME;
    std::vector<int64_t> inputSize = sliceOutDesc.GetShape().GetDims();
    FUSION_PASS_CHECK(backpropInputNode->SetAttr(PADS, dxPads) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set dx pads failed."), return false);
    FUSION_PASS_CHECK(backpropInputNode->SetAttr(PADDING, dxPadding) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set dx padding failed."), return false);
    FUSION_PASS_CHECK(backpropInputNode->SetAttr(INPUT_SIZE, inputSize) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set dx input_size failed."), return false);

    for (const auto& consumerPair : sliceNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX)) {
        FUSION_PASS_CHECK(consumerPair.first == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Slice consumer is null."),
                          return false);
        FUSION_PASS_CHECK(
            graph.RemoveEdge(*sliceNode, OUTPUT_INDEX, *consumerPair.first, consumerPair.second) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Remove Slice-consumer edge failed."), return false);
        FUSION_PASS_CHECK(graph.AddDataEdge(*backpropInputNode, OUTPUT_INDEX, *consumerPair.first,
                                            consumerPair.second) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Add dx-consumer edge failed."), return false);
    }

    return true;
}

bool PadConv2dFusionPass::SetPaddingAttrs(GNode& cubeNode) const
{
    std::vector<int64_t> cubePads = combinedPads;
    FUSION_PASS_CHECK(cubeNode.SetAttr(PADS, cubePads) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set cube node pads failed."), return false);

    AscendString cubeType;
    FUSION_PASS_CHECK(cubeNode.GetType(cubeType) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get cube node type failed."), return false);

    AscendString paddingValue = SAME;
    if (hasDn2Nz && cubeType == CONV2D) {
        paddingValue = EXPLICIT;
        AscendString autoPadValue = NOTSET;
        FUSION_PASS_CHECK(cubeNode.SetAttr(AUTO_PAD, autoPadValue) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Set cube node auto_pad failed."), return false);
    }
    FUSION_PASS_CHECK(cubeNode.SetAttr(PADDING, paddingValue) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set cube node padding failed."), return false);
    return true;
}

bool PadConv2dFusionPass::UpdateCubeNodes(Graph& graph, GNode& convNode)
{
    TensorDesc padInputDesc;
    FUSION_PASS_CHECK(padNode->GetInputDesc(INPUT_FMAP_INDEX, padInputDesc) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Pad input desc failed."), return false);

    auto padInputPair = padNode->GetInDataNodesAndPortIndexs(INPUT_FMAP_INDEX);
    FUSION_PASS_CHECK(padInputPair.first == nullptr,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get Pad input producer failed."), return false);

    std::vector<GNode> cubeNodes = {convNode};
    if (backpropFilterNode != nullptr) {
        cubeNodes.emplace_back(*backpropFilterNode);
    }

    for (GNode& cubeNode : cubeNodes) {
        FUSION_PASS_CHECK(cubeNode.UpdateInputDesc(INPUT_FMAP_INDEX, padInputDesc) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Update cube node input desc failed."), return false);
        FUSION_PASS_CHECK(graph.RemoveEdge(*padNode, OUTPUT_INDEX, cubeNode, INPUT_FMAP_INDEX) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Remove Pad-cube edge failed."), return false);
        FUSION_PASS_CHECK(
            graph.AddDataEdge(*padInputPair.first, padInputPair.second, cubeNode, INPUT_FMAP_INDEX) != GRAPH_SUCCESS,
            OP_LOGE(convDescInfo.nodeNameStr, "Add producer-cube edge failed."), return false);
        FUSION_PASS_CHECK_NOLOG(!SetPaddingAttrs(cubeNode), return false);
    }
    return true;
}

bool PadConv2dFusionPass::ConvFusionReplaceImpl(GraphPtr& graph, GNode& convNode, CustomPassContext& passContext)
{
    FUSION_PASS_CHECK(graph == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Graph is nullptr."), return false);

    std::vector<GNode> nodesBeforeFuse = {convNode, *padNode};
    if (backpropFilterNode != nullptr) {
        nodesBeforeFuse.emplace_back(*backpropFilterNode);
    }
    if (sliceNode != nullptr) {
        nodesBeforeFuse.emplace_back(*backpropInputNode);
        nodesBeforeFuse.emplace_back(*sliceNode);
    }

    AscendString failedReason;
#if GE_COMPILER_VERSION_NUM >= 90100000U
    FUSION_PASS_CHECK(!ge::fusion::GraphFuseInspectorUtils::CanFuse(nodesBeforeFuse, failedReason),
                      OP_LOGD(convDescInfo.nodeNameStr, "CanFuse failed, reason: %s.", failedReason.GetString()),
                      return false);
#endif

    FUSION_PASS_CHECK_NOLOG(!HandleBackwardPath(*graph), return false);
    FUSION_PASS_CHECK_NOLOG(!UpdateCubeNodes(*graph, convNode), return false);

    std::vector<GNode> nodesAfterFuse = {convNode};
    if (backpropFilterNode != nullptr) {
        nodesAfterFuse.emplace_back(*backpropFilterNode);
    }
    if (backpropInputNode != nullptr) {
        nodesAfterFuse.emplace_back(*backpropInputNode);
    }
#if GE_COMPILER_VERSION_NUM >= 90100000U
    FUSION_PASS_CHECK(
        ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, nodesAfterFuse, passContext) != SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "ReportFuse failed."), return false);
#endif

    if (sliceNode != nullptr) {
        FUSION_PASS_CHECK(graph->RemoveNode(*sliceNode) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "Remove Slice node failed."), return false);
    }
    FUSION_PASS_CHECK(graph->RemoveNode(*padNode) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Remove Pad node failed."), return false);
    return true;
}

} // namespace Ops
