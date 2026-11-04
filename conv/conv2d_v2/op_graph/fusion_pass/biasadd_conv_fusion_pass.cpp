/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file biasadd_conv_fusion_pass.cpp
 *  \brief conv + biasadd/add --> conv with bias
 */
#include "biasadd_conv_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace BiasaddConvFusion;
using namespace ConvFusionUtils;
using namespace ge;
using namespace fe;
using namespace fusion;

namespace {

void SyncOriginDataType(TensorDesc& desc)
{
#if GE_COMPILER_VERSION_NUM >= 90300000U
    if (desc.GetOriginDataType() == DT_UNDEFINED) {
        desc.SetOriginDataType(desc.GetDataType());
    }
#endif
}

bool IsConstCandidate(const GNode& consumer, int32_t inputIndex)
{
    const auto source = consumer.GetInDataNodesAndPortIndexs(inputIndex).first;
    if (source == nullptr) {
        return false;
    }
    AscendString sourceType;
    source->GetType(sourceType);
    if (sourceType == CONST) {
        return true;
    }
    if (sourceType != DATA) {
        return false;
    }
    const auto parent = source->GetInDataNodesAndPortIndexs(0).first;
    if (parent != nullptr) {
        AscendString parentType;
        parent->GetType(parentType);
        return parentType == CONST;
    }

    int64_t parentIndex = -1;
    if (source->GetAttr(AscendString(ATTR_PARENT_NODE_INDEX.c_str()), parentIndex) != GRAPH_SUCCESS) {
        return false;
    }
    Tensor probe;
    return consumer.GetInputConstData(inputIndex, probe) == GRAPH_SUCCESS;
}

// Conv2D/DepthwiseConv2D/Conv3D require the bias channel dim to match Cout
bool CheckCoutMatch(const TensorDesc& convOutputDesc, const AscendString& convType, int64_t biasDim)
{
    size_t channelIndex = 0U;
    if (convType == CONV2D || convType == DEPTHWISE_CONV2D) {
        if (convOutputDesc.GetOriginFormat() == FORMAT_NCHW) {
            channelIndex = 1U;
        } else if (convOutputDesc.GetOriginFormat() == FORMAT_NHWC) {
            channelIndex = NHWC_CHANNEL_AXIS;
        } else if (convOutputDesc.GetOriginFormat() == FORMAT_HWCN) {
            channelIndex = HWCN_CHANNEL_AXIS;
        } else {
            return false;
        }
    } else if (convType == CONV3D) {
        if (convOutputDesc.GetOriginFormat() == FORMAT_NCDHW) {
            channelIndex = 1U;
        } else if (convOutputDesc.GetOriginFormat() == FORMAT_NDHWC) {
            channelIndex = NDHWC_CHANNEL_AXIS;
        } else {
            return false;
        }
    } else {
        return true;
    }
    const auto outputDims = convOutputDesc.GetOriginShape().GetDims();
    return channelIndex < outputDims.size() && biasDim == outputDims[channelIndex];
}

bool NormalizeConstShape(const TensorDesc& convOutputDesc, const TensorDesc& biasDesc, const AscendString& convType,
                         std::vector<int64_t>& normalizedShape)
{
    const auto dims = biasDesc.GetShape().GetDims();
    if (dims.empty()) {
        return false;
    }
    if (dims.size() == 1U) {
        if (!CheckCoutMatch(convOutputDesc, convType, dims[0])) {
            return false;
        }
        normalizedShape = dims;
        return true;
    }
    int64_t oneCount = 0;
    int64_t lastNonOne = 1;
    for (const auto dim : dims) {
        if (dim == -1 || dim == UNKNOWN_DIM_SENTINEL) {
            return false;
        }
        if (dim == 1) {
            ++oneCount;
        } else {
            lastNonOne = dim;
        }
    }
    if (oneCount < MIN_ONE_DIM_COUNT) {
        return false;
    }
    if (!CheckCoutMatch(convOutputDesc, convType, lastNonOne)) {
        return false;
    }
    normalizedShape = {lastNonOne};
    return true;
}

bool CheckConstBias(const GNode& conv, const Tensor& biasTensor, const AscendString& convType)
{
    TensorDesc convOutputDesc;
    FUSION_PASS_CHECK(conv.GetOutputDesc(0, convOutputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get conv output desc failed, no fusion."), return false);
    std::vector<int64_t> normalizedShape;
    FUSION_PASS_CHECK(!NormalizeConstShape(convOutputDesc, biasTensor.GetTensorDesc(), convType, normalizedShape),
                      OP_LOGD(FUSION_NAME, "Const bias shape, Cout or format does not meet fusion requirements."),
                      return false);
    return true;
}

bool CheckPlaceholderBias(const GNode& conv, const GNode& addNode, const AscendString& convType)
{
    const auto source = addNode.GetInDataNodesAndPortIndexs(ADD_BIAS_INPUT_INDEX).first;
    FUSION_PASS_CHECK(convType == CONV3D, OP_LOGD(FUSION_NAME, "Conv3D does not support Placeholder fusion."),
                      return false);
    AscendString sourceType;
    TensorDesc desc;
    source->GetType(sourceType);
    FUSION_PASS_CHECK(sourceType != DATA, OP_LOGD(FUSION_NAME, "Placeholder bias is not direct Data, no fusion."),
                      return false);
    FUSION_PASS_CHECK(source->GetOutputDesc(0, desc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get Placeholder bias desc failed, no fusion."), return false);
    const auto dims = desc.GetShape().GetDims();
    FUSION_PASS_CHECK(dims.size() != 1U, OP_LOGD(FUSION_NAME, "Placeholder bias rank is not one, no fusion."),
                      return false);
    FUSION_PASS_CHECK(dims[0] == -1 || dims[0] == UNKNOWN_DIM_SENTINEL,
                      OP_LOGD(FUSION_NAME, "Placeholder bias has unknown dimension, no fusion."), return false);

    TensorDesc convOutputDesc;
    FUSION_PASS_CHECK(conv.GetOutputDesc(0, convOutputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get conv output desc failed, no fusion."), return false);
    FUSION_PASS_CHECK(!CheckCoutMatch(convOutputDesc, convType, dims[0]),
                      OP_LOGD(FUSION_NAME, "Placeholder bias length mismatches Cout, no fusion."), return false);
    OP_LOGD(FUSION_NAME, "Platform supports direct Conv and Placeholder bias fusion.");
    return true;
}

bool CheckReshapeBias(const GNode& conv, const GNode& reshape)
{
    const auto biasSource = reshape.GetInDataNodesAndPortIndexs(0).first;
    FUSION_PASS_CHECK(biasSource == nullptr, OP_LOGD(FUSION_NAME, "Reshape bias source node is null, no fusion."),
                      return false);

    FUSION_PASS_CHECK(!IsConstCandidate(reshape, RESHAPE_BIAS_INPUT_INDEX),
                      OP_LOGD(FUSION_NAME, "Reshape bias is not a const candidate, no fusion."), return false);
    const auto shapeSource = reshape.GetInDataNodesAndPortIndexs(RESHAPE_SHAPE_INPUT_INDEX).first;
    if (shapeSource != nullptr) {
        FUSION_PASS_CHECK(IsConstCandidate(reshape, RESHAPE_SHAPE_INPUT_INDEX),
                          OP_LOGD(FUSION_NAME, "Reshape shape is an additional Const candidate, no fusion."),
                          return false);
    }
    Tensor tensor;
    TensorDesc reshapeInputDesc;
    TensorDesc biasOutputDesc;
    FUSION_PASS_CHECK(reshape.GetInputConstData(RESHAPE_BIAS_INPUT_INDEX, tensor) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get Reshape bias Const data failed, no fusion."), return false);
    FUSION_PASS_CHECK(biasSource->GetOutputDesc(0, biasOutputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get Reshape bias source desc failed, no fusion."), return false);

    if (reshape.GetInputDesc(RESHAPE_BIAS_INPUT_INDEX, reshapeInputDesc) != GRAPH_SUCCESS ||
        reshapeInputDesc.GetShape().GetDims().empty()) {
        reshapeInputDesc = biasOutputDesc;
    }
    const auto dims = reshapeInputDesc.GetShape().GetDims();
    const auto dtype = biasOutputDesc.GetDataType();
    FUSION_PASS_CHECK(dims.size() != 1U, OP_LOGD(FUSION_NAME, "Reshape bias rank is not one, no fusion."),
                      return false);
    FUSION_PASS_CHECK(dtype != DT_FLOAT16 && dtype != DT_FLOAT,
                      OP_LOGD(FUSION_NAME, "Reshape bias dtype is not FP16 or FP32, no fusion."), return false);

    TensorDesc convOutputDesc;
    FUSION_PASS_CHECK(conv.GetOutputDesc(0, convOutputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get conv output desc failed, no fusion."), return false);
    const auto convOutputFormat = convOutputDesc.GetOriginFormat();
    if (dims[0] != -1 && dims[0] != UNKNOWN_DIM_SENTINEL &&
        (convOutputFormat == FORMAT_NCDHW || convOutputFormat == FORMAT_NDHWC)) {
        FUSION_PASS_CHECK(!CheckCoutMatch(convOutputDesc, CONV3D, dims[0]),
                          OP_LOGD(FUSION_NAME, "Reshape bias length mismatches Cout, no fusion."), return false);
    }
    return true;
}

bool QueryPlatform(PlatformInfo& platformInfo)
{
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OP_LOGD(FUSION_NAME, "Get platform info failed, no fusion."), return false);
    return true;
}

bool SupportsFixPipe(const PlatformInfo& platformInfo)
{
    return platformInfo.ai_core_intrinsic_dtype_map.find(INTRINSIC_FIX_PIPE_L0C2OUT) !=
           platformInfo.ai_core_intrinsic_dtype_map.end();
}

bool GetReplacementBiasShape(const GNode& conv, const GNode& biasConsumer, int32_t biasInputIndex,
                             std::vector<int64_t>& normalizedShape)
{
    if (!IsConstCandidate(biasConsumer, biasInputIndex)) {
        return true;
    }
    AscendString convType;
    TensorDesc convOutputDesc;
    Tensor tensor;
    FUSION_PASS_CHECK(conv.GetType(convType) != GRAPH_SUCCESS, OP_LOGD(FUSION_NAME, "Get conv type failed."),
                      return false);
    FUSION_PASS_CHECK(conv.GetOutputDesc(0, convOutputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get conv output desc failed."), return false);
    FUSION_PASS_CHECK(biasConsumer.GetInputConstData(biasInputIndex, tensor) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get Const bias data failed."), return false);

    NormalizeConstShape(convOutputDesc, tensor.GetTensorDesc(), convType, normalizedShape);
    return true;
}

bool PrepareTargetBiasDesc(const TensorDesc& convSideInputDesc, GNode& biasSource,
                           const std::vector<int64_t>& normalizedShape, const TensorDesc& firstInputDesc,
                           TensorDesc& targetBiasDesc)
{
    TensorDesc sourceDesc;
    FUSION_PASS_CHECK(biasSource.GetOutputDesc(0, sourceDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get bias source output desc failed."), return false);
    const auto biasShape = normalizedShape.empty() ? sourceDesc.GetShape().GetDims() : normalizedShape;
    sourceDesc.SetShape(Shape(biasShape));
    sourceDesc.SetOriginShape(Shape(biasShape));
    sourceDesc.SetFormat(static_cast<Format>(GetPrimaryFormat(convSideInputDesc.GetFormat())));
    sourceDesc.SetOriginFormat(convSideInputDesc.GetOriginFormat());
    SyncOriginDataType(sourceDesc);
    FUSION_PASS_CHECK(biasSource.UpdateOutputDesc(0, sourceDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Update bias source output desc failed."), return false);
    targetBiasDesc = sourceDesc;
    targetBiasDesc.SetOriginFormat(firstInputDesc.GetOriginFormat());
    SyncOriginDataType(targetBiasDesc);
    return true;
}

const std::vector<std::string> legacyOutputTypeOrder = {"BiasAdd", "Add"};

std::pair<int32_t, std::string> GetLegacyOrderKey(const GNode& convNode)
{
    AscendString convName;
    (void)convNode.GetName(convName);
    std::string consumerType;
    std::string consumerName = (convName.GetString() == nullptr) ? std::string() : convName.GetString();

    const auto consumers = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (const auto& consumer : consumers) {
        if (consumer.first == nullptr) {
            continue;
        }
        AscendString type;
        AscendString name;
        if (consumer.first->GetType(type) == GRAPH_SUCCESS && consumer.first->GetName(name) == GRAPH_SUCCESS &&
            name.GetString() != nullptr) {
            consumerType = (type.GetString() == nullptr) ? std::string() : type.GetString();
            consumerName = name.GetString();
            break;
        }
    }
    // Types outside the legacy pattern output list never anchor a legacy match; sort them
    // last (they are rejected in CheckMatchStructure anyway, order is irrelevant).
    auto rank = static_cast<int32_t>(
        std::find(legacyOutputTypeOrder.begin(), legacyOutputTypeOrder.end(), consumerType) -
        legacyOutputTypeOrder.begin());
    return {rank, consumerName};
}

} // namespace

void AABiasaddConvFusion::InitMember()
{
    addNode = nullptr;
    reshapeNode = nullptr;
    addNodeName.clear();
    convInputPortOfAdd = -1;
    biasInputIndex = -1;
    isReversedAdd = false;
    isReshapeMode = false;
    reshapeHasOtherConsumer = false;
}

std::set<AscendString> AABiasaddConvFusion::GetNodeTypes() const
{
    std::set<AscendString> types;
    for (const auto& type : convTypes) {
        types.insert(AscendString(type.c_str()));
    }
    return types;
}

// Keeps the legacy pass visible processing order: sorted by the (pattern type rank,
// name) of the Add/BiasAdd node consuming each conv. stable_sort preserves the graph
// order for convs sharing one consumer (both rejected later, order irrelevant).
void AABiasaddConvFusion::OrderMatchedNodes(std::vector<GNode>& matchedNodes)
{
    std::stable_sort(matchedNodes.begin(), matchedNodes.end(), [](const GNode& lhs, const GNode& rhs) {
        return GetLegacyOrderKey(lhs) < GetLegacyOrderKey(rhs);
    });
}

void AABiasaddConvFusion::PrintGraphStructure() const {}

bool AABiasaddConvFusion::ResolveMatchStructure(const AscendString& addType)
{
    FUSION_PASS_CHECK(convInputPortOfAdd < 0, OP_LOGD(FUSION_NAME, "Get conv input port of add failed, no fusion."),
                      return false);
    const bool isAdd = addType == ADD;
    if (addType == BIASADD) {
        FUSION_PASS_CHECK(convInputPortOfAdd != ADD_CONV_INPUT_INDEX,
                          OP_LOGD(FUSION_NAME, "Reversed BiasAdd input order is not supported, no fusion."),
                          return false);
    } else {
        FUSION_PASS_CHECK(
            !isAdd, OP_LOGD(FUSION_NAME, "Conv consumer type %s is not Add/BiasAdd, no fusion.", addType.GetString()),
            return false);
    }
    isReversedAdd = isAdd && convInputPortOfAdd != ADD_CONV_INPUT_INDEX;
    // Reversed Add swaps the two ports, so the bias occupies the port the conv uses in normal order.
    biasInputIndex = isReversedAdd ? ADD_CONV_INPUT_INDEX : ADD_BIAS_INPUT_INDEX;

    // R mode: Conv3D + Add whose bias side is a Reshape, normal input order only
    if (isReshapeMode) {
        const auto reshapeBias = reshapeNode->GetInDataNodesAndPortIndexs(RESHAPE_BIAS_INPUT_INDEX);
        FUSION_PASS_CHECK(reshapeBias.first == nullptr, OP_LOGD(FUSION_NAME, "Reshape bias source is null, no fusion."),
                          return false);
        const auto reshapeConsumers = reshapeNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
        for (const auto& consumer : reshapeConsumers) {
            if (consumer.first == nullptr) {
                continue;
            }
            // Compare by node name: the graph API hands out distinct GNode copies, so the
            // shared-pointer identity of the same logical node is not stable.
            AscendString consumerName;
            if (consumer.first->GetName(consumerName) != GRAPH_SUCCESS || addNodeName != consumerName.GetString()) {
                reshapeHasOtherConsumer = true;
                break;
            }
        }
    }
    return true;
}

bool AABiasaddConvFusion::CheckMatchStructure(const GNode& matchNode)
{
    FUSION_PASS_CHECK(matchNode.GetInDataNodesAndPortIndexs(INPUT_FMAP_INDEX).first == nullptr ||
                          matchNode.GetInDataNodesAndPortIndexs(INPUT_FILTER_INDEX).first == nullptr,
                      OP_LOGD(FUSION_NAME, "Conv fmap/filter input is missing, no fusion."), return false);

    std::set<std::string> convConsumerNames;
    for (int32_t outIndex = 0; outIndex < static_cast<int32_t>(matchNode.GetOutputsSize()); ++outIndex) {
        for (const auto& consumer : matchNode.GetOutDataNodesAndPortIndexs(outIndex)) {
            if (consumer.first == nullptr) {
                continue;
            }
            AscendString consumerName;
            if (consumer.first->GetName(consumerName) == GRAPH_SUCCESS) {
                convConsumerNames.insert(consumerName.GetString());
            }
        }
    }
    FUSION_PASS_CHECK(convConsumerNames.size() != SINGLE_CONSUMER_COUNT,
                      OP_LOGD(FUSION_NAME, "Conv output consumer count %zu != 1, no fusion.", convConsumerNames.size()),
                      return false);
    const auto convConsumers = matchNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK(convConsumers.empty(), OP_LOGD(FUSION_NAME, "Conv consumer is null, no fusion."), return false);
    addNode = convConsumers[0].first;
    convInputPortOfAdd = convConsumers[0].second;
    FUSION_PASS_CHECK(addNode == nullptr, OP_LOGD(FUSION_NAME, "Conv consumer node is null, no fusion."), return false);
    AscendString addType;
    FUSION_PASS_CHECK(addNode->GetType(addType) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get add node type failed, no fusion."), return false);
    int32_t connectedAddInputs = 0;
    for (int32_t index = 0; index < static_cast<int32_t>(addNode->GetInputsSize()); ++index) {
        if (addNode->GetInDataNodesAndPortIndexs(index).first != nullptr) {
            ++connectedAddInputs;
        }
    }
    FUSION_PASS_CHECK(connectedAddInputs != ADD_INPUT_NUMS,
                      OP_LOGD(FUSION_NAME, "Add connected input count %d != 2, no fusion.", connectedAddInputs),
                      return false);
    AscendString addName;
    FUSION_PASS_CHECK(addNode->GetName(addName) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get add node name failed, no fusion."), return false);
    addNodeName = addName.GetString();

    AscendString convType;
    FUSION_PASS_CHECK(matchNode.GetType(convType) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get conv type failed, no fusion."), return false);
    isReshapeMode = convType == CONV3D && addType == ADD;
    if (isReshapeMode) {
        const auto biasCandidate = addNode->GetInDataNodesAndPortIndexs(ADD_BIAS_INPUT_INDEX).first;
        AscendString biasCandidateType;
        FUSION_PASS_CHECK(biasCandidate == nullptr || biasCandidate->GetType(biasCandidateType) != GRAPH_SUCCESS,
                          OP_LOGD(FUSION_NAME, "Get add bias side type failed, no fusion."), return false);
        isReshapeMode = biasCandidate != nullptr && biasCandidateType == RESHAPE;
        // R mode requires both Reshape inputs (bias and shape) connected, matching the pattern
        // boundary; a shape-less Reshape falls back to the ordinary path where a Reshape bias
        // side can never fuse.
        if (isReshapeMode && (biasCandidate->GetInDataNodesAndPortIndexs(RESHAPE_BIAS_INPUT_INDEX).first == nullptr ||
                              biasCandidate->GetInDataNodesAndPortIndexs(RESHAPE_SHAPE_INPUT_INDEX).first == nullptr)) {
            isReshapeMode = false;
        }
        reshapeNode = isReshapeMode ? biasCandidate : nullptr;
    }
    return ResolveMatchStructure(addType);
}

bool AABiasaddConvFusion::MeetRequirements(const GNode& convNode)
{
    const char* candidateType = isReshapeMode ? (reshapeHasOtherConsumer ? "R dual-output" : "R single-output") :
                                                (isReversedAdd ? "ordinary reversed" : "ordinary");
    OP_LOGD(FUSION_NAME, "Begin to check BiasaddConv fusion requirements, candidate is %s.", candidateType);
    FUSION_PASS_CHECK(!(convNode.GetInDataNodesAndPortIndexs(INPUT_BIAS_INDEX).first == nullptr),
                      OP_LOGD(FUSION_NAME, "Conv already has bias, no fusion."), return false);
    PlatformInfo platformInfo;
    FUSION_PASS_CHECK_NOLOG(!QueryPlatform(platformInfo), return false);
    if (isReshapeMode) {
        return CheckReshapeBias(convNode, *reshapeNode);
    }
    AscendString convType;
    convNode.GetType(convType);
    const auto bias = addNode->GetInDataNodesAndPortIndexs(biasInputIndex).first;
    FUSION_PASS_CHECK(bias == nullptr, OP_LOGD(FUSION_NAME, "Bias source node is null, no fusion."), return false);
    TensorDesc biasDesc;
    FUSION_PASS_CHECK(bias->GetOutputDesc(0, biasDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get bias output desc failed, no fusion."), return false);
    FUSION_PASS_CHECK(supportedBiasDtypes.find(biasDesc.GetDataType()) == supportedBiasDtypes.end(),
                      OP_LOGD(FUSION_NAME, "Bias dtype is not supported, no fusion."), return false);
    AscendString biasType;
    bias->GetType(biasType);
    if (biasType == VARIABLE) {
        FUSION_PASS_CHECK(isReversedAdd,
                          OP_LOGD(FUSION_NAME, "Reversed Add does not support Variable bias, no fusion."),
                          return false);
        AscendString addType;
        addNode->GetType(addType);
        FUSION_PASS_CHECK(addType != BIASADD, OP_LOGD(FUSION_NAME, "Variable bias only supports BiasAdd, no fusion."),
                          return false);
        OP_LOGD(FUSION_NAME, "Variable bias meets requirements, fuse in training mode.");
        return true;
    }

    if (IsConstCandidate(*addNode, biasInputIndex)) {
        Tensor biasTensor;
        if (addNode->GetInputConstData(biasInputIndex, biasTensor) == GRAPH_SUCCESS) {
            return CheckConstBias(convNode, biasTensor, convType);
        }
    }

    FUSION_PASS_CHECK(isReversedAdd,
                      OP_LOGD(FUSION_NAME, "Reversed Add only supports Const candidate bias, no fusion."),
                      return false);
    const bool supportsFixPipe = SupportsFixPipe(platformInfo);
    if (supportsFixPipe) {
        return CheckPlaceholderBias(convNode, *addNode, convType);
    }
    return false;
}

bool AABiasaddConvFusion::LinkBiasSourceToConv(Graph& graph, GNode& convNode)
{
    GNode& biasConsumer = isReshapeMode ? *reshapeNode : *addNode;
    const int32_t biasInputIdx = isReshapeMode ? RESHAPE_BIAS_INPUT_INDEX : biasInputIndex;
    std::vector<int64_t> normalizedShape;
    FUSION_PASS_CHECK(!GetReplacementBiasShape(convNode, biasConsumer, biasInputIdx, normalizedShape),
                      OP_LOGE(FUSION_NAME, "Get replacement bias shape failed."), return false);
    const auto biasPair = biasConsumer.GetInDataNodesAndPortIndexs(biasInputIdx);
    FUSION_PASS_CHECK(biasPair.first == nullptr, OP_LOGE(FUSION_NAME, "Bias source node is null."), return false);
    TensorDesc convSideDesc;
    FUSION_PASS_CHECK(addNode->GetInputDesc(convInputPortOfAdd, convSideDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get add conv side desc failed."), return false);
    TensorDesc firstInputDesc;
    FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_FMAP_INDEX, firstInputDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get conv fmap desc failed."), return false);
    TensorDesc targetBiasDesc;
    FUSION_PASS_CHECK(
        !PrepareTargetBiasDesc(convSideDesc, *biasPair.first, normalizedShape, firstInputDesc, targetBiasDesc),
        OP_LOGE(FUSION_NAME, "Prepare target bias desc failed."), return false);
    FUSION_PASS_CHECK(graph.AddDataEdge(*biasPair.first, biasPair.second, convNode, INPUT_BIAS_INDEX) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Link bias source to conv failed."), return false);
    FUSION_PASS_CHECK(convNode.UpdateInputDesc(INPUT_BIAS_INDEX, targetBiasDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Update conv bias desc failed."), return false);
    bool hasBias = true;
    FUSION_PASS_CHECK(convNode.SetAttr(HAS_BIAS_ATTR, hasBias) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Set has_bias attr on conv failed."), return false);
    return true;
}

bool AABiasaddConvFusion::RelinkAddConsumersToConv(Graph& graph, GNode& convNode)
{
    const auto consumers = addNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (const auto& consumer : consumers) {
        FUSION_PASS_CHECK(consumer.first == nullptr, OP_LOGE(FUSION_NAME, "Add consumer is null."), return false);
        FUSION_PASS_CHECK(graph.RemoveEdge(*addNode, OUTPUT_INDEX, *consumer.first, consumer.second) != GRAPH_SUCCESS,
                          OP_LOGE(FUSION_NAME, "Remove edge between add and consumer failed."), return false);
        FUSION_PASS_CHECK(graph.AddDataEdge(convNode, OUTPUT_INDEX, *consumer.first, consumer.second) != GRAPH_SUCCESS,
                          OP_LOGE(FUSION_NAME, "Add edge between conv and add consumer failed."), return false);
    }
    return true;
}

bool AABiasaddConvFusion::MigrateStreamSwitchControlEdges(Graph& graph, GNode& convNode)
{
    for (const auto& controlInput : addNode->GetInControlNodes()) {
        if (controlInput == nullptr) {
            continue;
        }
        AscendString controlType;
        if (controlInput->GetType(controlType) != GRAPH_SUCCESS || controlType != STREAM_SWITCH) {
            continue;
        }
        FUSION_PASS_CHECK(graph.AddControlEdge(*controlInput, convNode) != GRAPH_SUCCESS,
                          OP_LOGE(FUSION_NAME, "Migrate StreamSwitch control edge to conv failed."), return false);
    }
    return true;
}

bool AABiasaddConvFusion::ConvFusionReplaceImpl(GraphPtr& graph, GNode& convNode, CustomPassContext& passContext)
{
    FUSION_PASS_CHECK(graph == nullptr, OP_LOGE(FUSION_NAME, "%s graph is nullptr.", convDescInfo.nodeNameStr.c_str()),
                      return false);

    // Removed node first: the legacy pattern framework records the fused (removed)
    // output node before the surviving conv, and the om graph keeps that order in
    // _datadump_original_op_names/_original_op_attrs_in_fusion_process.
    std::vector<GNode> nodesBeforeFuse = {*addNode, convNode};
    FUSION_PASS_CHECK_NOLOG(!ReportMatchNodes(nodesBeforeFuse, passContext), return false);
    FUSION_PASS_CHECK_NOLOG(!CanFuseNodes(nodesBeforeFuse), return false);

    FUSION_PASS_CHECK(!LinkBiasSourceToConv(*graph, convNode), OP_LOGE(FUSION_NAME, "Link bias source to conv failed."),
                      return false);
    FUSION_PASS_CHECK(!RelinkAddConsumersToConv(*graph, convNode),
                      OP_LOGE(FUSION_NAME, "Relink add consumers to conv failed."), return false);
    FUSION_PASS_CHECK(!MigrateStreamSwitchControlEdges(*graph, convNode),
                      OP_LOGE(FUSION_NAME, "Migrate StreamSwitch control edges failed."), return false);

    std::vector<GNode> nodesAfterFuse = {convNode};
    FUSION_PASS_CHECK_NOLOG(!ReportFuseNodes(nodesBeforeFuse, nodesAfterFuse, passContext), return false);

    if (!ConvFusionUtilsPass::CopyDumpOutputMapping(*addNode, convNode)) {
        OP_LOGE(FUSION_NAME, "Copy datadump output mapping from %s failed.", addNodeName.c_str());
    }
    FUSION_PASS_CHECK(graph->RemoveNode(*addNode) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Remove fused add node failed."), return false);
    if (isReshapeMode && !reshapeHasOtherConsumer) {
        FUSION_PASS_CHECK(graph->RemoveNode(*reshapeNode) != GRAPH_SUCCESS,
                          OP_LOGE(FUSION_NAME, "Remove orphaned reshape node failed."), return false);
    }

    AscendString convName;
    convNode.GetName(convName);
    OP_LOGD(FUSION_NAME, "BiasaddConv fusion success for node %s.", convName.GetString());
    return true;
}

#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(AABiasaddConvFusion).Stage(CustomPassStage::kCompatibleInherited);
#endif

} // namespace Ops
