/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv_fusion_utils_pass.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "platform/platform_info.h"
#include "runtime/runtime/base.h"
#include "version/ge-compiler_version.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace ConvFusionUtils {
using namespace ge;
using namespace fe;
using namespace fusion;
using conv_arch::GetNpuArchKey;
using conv_arch::IsCubeVectorFuseSoc;

bool ConvFusionUtilsPass::AddSubgraphInput(std::unique_ptr<SubgraphBoundary>& boundary, const GNode& node,
                                           const int64_t subgraphIndex, const int64_t boundaryIndex)
{
    SubgraphInput subgraphInput;
    FUSION_PASS_CHECK(subgraphInput.AddInput({node, subgraphIndex}) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Create SubgraphInput failed."), return false);

    FUSION_PASS_CHECK(boundary->AddInput(boundaryIndex, std::move(subgraphInput)) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "SubgraphBoundary add SubgraphInput failed."), return false);

    return true;
}

bool ConvFusionUtilsPass::AddSubgraphOutput(std::unique_ptr<SubgraphBoundary>& boundary, const GNode& node,
                                            const int64_t subgraphIndex, const int64_t boundaryIndex)
{
    SubgraphOutput subgraphOutput;
    FUSION_PASS_CHECK(subgraphOutput.SetOutput({node, subgraphIndex}) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Create SubgraphOutput failed."), return false);

    FUSION_PASS_CHECK(boundary->AddOutput(boundaryIndex, std::move(subgraphOutput)) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "SubgraphBoundary add SubgraphOutput failed."), return false);

    return true;
}

bool ConvFusionUtilsPass::IsFuseSoc()
{
    fe::PlatFormInfos platFormInfos;
    fe::OptionalInfos optionalInfos;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platFormInfos, optionalInfos) !=
        GRAPH_SUCCESS) {
        OP_LOGW(UTIL_NAME, "Get PlatFormInfos failed.");
        return false;
    }
    return conv_arch::IsCubeVectorFuseSoc(platFormInfos);
}

bool ConvFusionUtilsPass::CheckSocList(const std::map<std::string, NpuArch>& socList, NpuArch& npuArch,
                                       bool supportFuse)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != GRAPH_SUCCESS,
        OP_LOGW(UTIL_NAME, "Get platform_info failed."), return false);
    const std::string soc = platformInfo.str_info.short_soc_version;

    auto it = socList.find(soc);
    if (it != socList.end()) {
        npuArch = it->second;
        OP_LOGD(UTIL_NAME, "Current NpuArch is DAV_%u.", npuArch);
        return true;
    }

    if (supportFuse && IsFuseSoc()) {
        OP_LOGD(UTIL_NAME, "Current platform is cube_vector_combine=fuse.");
        return true;
    }

    OP_LOGD(UTIL_NAME, "Current soc %s not in check list %s.", soc.c_str(), SocListToString(socList).c_str());
    return false;
}

const std::string& ConvFusionUtilsPass::GetArchKey()
{
    fe::PlatFormInfos platFormInfos;
    fe::OptionalInfos optionalInfos;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platFormInfos, optionalInfos) !=
        GRAPH_SUCCESS) {
        OP_LOGW(UTIL_NAME, "Get PlatFormInfos failed, fallback to 3510.");
        return NPU_ARCH_KEY_3510;
    }
    return GetNpuArchKey(platFormInfos);
}

bool ConvFusionUtilsPass::GetConvBaseAttr(const GNode& convNode, ConvBaseAttrs& baseAttrs,
                                          const ConvDescInfo& convDescInfo)
{
    baseAttrs = ConvBaseAttrs();
    FUSION_PASS_CHECK(convNode.GetAttr(STRIDES, baseAttrs.strides) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get strides from %s failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);

    FUSION_PASS_CHECK(convNode.GetAttr(PADS, baseAttrs.pads) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get pads from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);

    FUSION_PASS_CHECK(convNode.GetAttr(DILATIONS, baseAttrs.dilations) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get dilations from %s failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);

    FUSION_PASS_CHECK(convNode.GetAttr(GROUPS, baseAttrs.groups) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get groups from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);

    FUSION_PASS_CHECK(convNode.GetAttr(DATA_FORMAT, baseAttrs.dataFormat) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get data_format from %s failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);

    FUSION_PASS_CHECK(convNode.GetAttr(OFFSET_X, baseAttrs.offsetX) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get offset_x from %s failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);

    convNode.GetAttr(PADDING, baseAttrs.padding);
    AscendString attrAutoPad = "";
    convNode.GetAttr(AUTO_PAD, attrAutoPad);

    AscendString tmpPadMode = baseAttrs.padding.GetLength() != 0 ?
                                  baseAttrs.padding :
                                  (attrAutoPad.GetLength() != 0 ? attrAutoPad : "NOTSET");
    baseAttrs.padMode = SPECIFIC_PAD_LIST.count(tmpPadMode) != 0 ? "SPECIFIC" : tmpPadMode;

    convNode.GetAttr(OP_IMPL_MODE_ENUM, baseAttrs.opImplModeEnum);
    int64_t opImplModeEnum = baseAttrs.opImplModeEnum;
    bool isHf32OpImplMode = std::any_of(
        HF32_PRECISION_MODES_INT.begin(), HF32_PRECISION_MODES_INT.end(),
        [&opImplModeEnum](const int64_t& targetPrecisionModeInt) { return opImplModeEnum == targetPrecisionModeInt; });
    baseAttrs.enableHf32 = (convDescInfo.fmapDtype == DataType::DT_FLOAT && isHf32OpImplMode);
    baseAttrs.opImplModeEnum = 0x1;

    return true;
}

namespace {
std::string DimsToString(const std::vector<int64_t>& dims)
{
    std::string res = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        res += std::to_string(dims[i]);
        if (i + 1 < dims.size()) {
            res += ", ";
        }
    }
    res += "]";
    return res;
}

void FillConvDescMetaFromDescs(ConvDescInfo& convDescInfo)
{
    convDescInfo.fmapDtype = convDescInfo.fmapDesc.GetDataType();
    convDescInfo.filterDtype = convDescInfo.filterDesc.GetDataType();
    convDescInfo.outputDtype = convDescInfo.outputDesc.GetDataType();

    convDescInfo.fmapFormat = convDescInfo.fmapDesc.GetFormat();
    convDescInfo.filterFormat = convDescInfo.filterDesc.GetFormat();
    convDescInfo.outputFormat = convDescInfo.outputDesc.GetFormat();

    convDescInfo.fmapShape = convDescInfo.fmapDesc.GetShape().GetDims();
    convDescInfo.filterShape = convDescInfo.filterDesc.GetShape().GetDims();
    convDescInfo.outputShape = convDescInfo.outputDesc.GetShape().GetDims();

    if (convDescInfo.hasBias) {
        convDescInfo.biasDtype = convDescInfo.biasDesc.GetDataType();
        convDescInfo.biasFormat = convDescInfo.biasDesc.GetFormat();
        convDescInfo.biasShape = convDescInfo.biasDesc.GetShape().GetDims();
    }
}

bool FillConvTensorDescs(const GNode& convNode, ConvDescInfo& convDescInfo)
{
    FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_FMAP_INDEX, convDescInfo.fmapDesc) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get %s fmap tensor desc failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);
    FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_FILTER_INDEX, convDescInfo.filterDesc) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get %s filter tensor desc failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);
    if (convDescInfo.hasBias) {
        FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_BIAS_INDEX, convDescInfo.biasDesc) != GRAPH_SUCCESS,
                          OP_LOGE(UTIL_NAME, "Get %s bias tensor desc failed.", convDescInfo.nodeNameStr.c_str()),
                          return false);
    }
    FUSION_PASS_CHECK(convNode.GetOutputDesc(OUTPUT_INDEX, convDescInfo.outputDesc) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get %s output tensor desc failed.", convDescInfo.nodeNameStr.c_str()),
                      return false);
    return true;
}
} // namespace

bool ConvFusionUtilsPass::GetConvDescInfo(const GNode& convNode, ConvDescInfo& convDescInfo)
{
    convDescInfo = ConvDescInfo();
    FUSION_PASS_CHECK(convNode.GetName(convDescInfo.nodeName) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get node name failed."), return false);
    convDescInfo.nodeNameStr = convDescInfo.nodeName.GetString();

    AscendString nodeType;
    FUSION_PASS_CHECK(convNode.GetType(nodeType) != GRAPH_SUCCESS, OP_LOGE(UTIL_NAME, "Get node type failed."),
                      return false);
    if (CONV_OP_LIST.count(nodeType) == 0) {
        OP_LOGD(UTIL_NAME, "%s is not conv type op, skip GetConvDescInfo.", nodeType.GetString());
        return true;
    }

    convDescInfo.hasBias = convNode.GetInputsSize() == CONV_COUNT_PARAMS_BIAS;
    FUSION_PASS_CHECK_NOLOG(!FillConvTensorDescs(convNode, convDescInfo), return false);
    FillConvDescMetaFromDescs(convDescInfo);
    PrintConvDescInfo(convDescInfo);
    return true;
}

bool ConvFusionUtilsPass::GetMatchedNodes(const GraphPtr& graph, std::vector<GNode>& matchedNodes,
                                          const std::set<AscendString>& nodeTypes)
{
    for (auto& node : graph->GetDirectNode()) {
        AscendString curNodeType;
        FUSION_PASS_CHECK(node.GetType(curNodeType) != GRAPH_SUCCESS,
                          OP_LOGE(UTIL_NAME, "GetType in GetMatchedNodes failed."), return false);

        if (nodeTypes.count(curNodeType) != 0) {
            matchedNodes.emplace_back(node);
        }
    }

    return true;
}

GNodePtr ConvFusionUtilsPass::GetNodePtr(const GNode& node, const ConvDescInfo& convDescInfo)
{
    auto convOutputNodes = node.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK(convOutputNodes.empty(),
                      OP_LOGD(UTIL_NAME, "%s out nodes is empty.", convDescInfo.nodeNameStr.c_str()), return nullptr);

    GNodePtr outNodePtr = convOutputNodes[0].first;
    int32_t outNodePortIndex = convOutputNodes[0].second;

    GNodePtr nodePtr = outNodePtr->GetInDataNodesAndPortIndexs(outNodePortIndex).first;
    FUSION_PASS_CHECK(nodePtr == nullptr,
                      OP_LOGD(UTIL_NAME, "%s get in data nodes failed.", convDescInfo.nodeNameStr.c_str()),
                      return nullptr);

    AscendString name;
    FUSION_PASS_CHECK(nodePtr->GetName(name) != GRAPH_SUCCESS,
                      OP_LOGD(UTIL_NAME, "%s get node name by node ptr failed.", convDescInfo.nodeNameStr.c_str()),
                      return nullptr);

    FUSION_PASS_CHECK(name != convDescInfo.nodeName,
                      OP_LOGD(UTIL_NAME, "%s get node ptr failed.", convDescInfo.nodeNameStr.c_str()), return nullptr);

    return nodePtr;
}

bool ConvFusionUtilsPass::IsUnknownShape(const TensorDesc& tensorDesc)
{
    auto dims = tensorDesc.GetShape().GetDims();
    for (auto dim : dims) {
        if (dim < 0) {
            return true;
        }
    }
    return false;
}

AscendString ConvFusionUtilsPass::ListToAscendString(const std::vector<AscendString>& strList)
{
    std::string res = "";
    for (size_t index = 0; index < strList.size(); ++index) {
        res += strList[index].GetString();
        if (index != strList.size() - 1) {
            res += ", ";
        }
    }

    return AscendString(res.c_str());
}

void ConvFusionUtilsPass::PrintConvDescInfo(const ConvDescInfo& convDescInfo)
{
    std::string convDtypeInfo = "fmap is " + TypeUtils::DataTypeToSerialString(convDescInfo.fmapDtype) + " filter is " +
                                TypeUtils::DataTypeToSerialString(convDescInfo.filterDtype) + " output is " +
                                TypeUtils::DataTypeToSerialString(convDescInfo.outputDtype);

    std::string convFormatInfo = "fmap is " + TypeUtils::FormatToSerialString(convDescInfo.fmapFormat) + " filter is " +
                                 TypeUtils::FormatToSerialString(convDescInfo.filterFormat) + " output is " +
                                 TypeUtils::FormatToSerialString(convDescInfo.outputFormat);

    std::string convShapeInfo = "fmap is " + DimsToString(convDescInfo.fmapShape) + " filter is " +
                                DimsToString(convDescInfo.filterShape) + " output is " +
                                DimsToString(convDescInfo.outputShape);

    if (convDescInfo.hasBias) {
        convDtypeInfo += " bias is " + TypeUtils::DataTypeToSerialString(convDescInfo.biasDtype);
        convFormatInfo += " bias is " + TypeUtils::FormatToSerialString(convDescInfo.biasFormat);
        convShapeInfo += " bias is " + DimsToString(convDescInfo.biasShape);
    }

    OP_LOGD(UTIL_NAME, "%s dtype: %s", convDescInfo.nodeNameStr.c_str(), convDtypeInfo.c_str());
    OP_LOGD(UTIL_NAME, "%s format: %s", convDescInfo.nodeNameStr.c_str(), convFormatInfo.c_str());
    OP_LOGD(UTIL_NAME, "%s shape: %s", convDescInfo.nodeNameStr.c_str(), convShapeInfo.c_str());
}

bool ConvFusionUtilsPass::UpdateInputDesc(GNode* convNode, const ConvDescInfo& convDescInfo)
{
    FUSION_PASS_CHECK(convNode == nullptr, OP_LOGE(UTIL_NAME, "Node is nullptr, update input desc failed."),
                      return false);

    AscendString nodeName = "";
    FUSION_PASS_CHECK(convNode->GetName(nodeName) != GRAPH_SUCCESS, OP_LOGE(UTIL_NAME, "Get node name failed."),
                      return false);

    FUSION_PASS_CHECK(convNode->UpdateInputDesc(INPUT_FMAP_INDEX, convDescInfo.fmapDesc) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Update %s fmap tensor desc failed.", nodeName.GetString()), return false);

    FUSION_PASS_CHECK(convNode->UpdateInputDesc(INPUT_FILTER_INDEX, convDescInfo.filterDesc) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Update %s filter tensor desc failed.", nodeName.GetString()), return false);

    if (convDescInfo.hasBias) {
        FUSION_PASS_CHECK(convNode->UpdateInputDesc(INPUT_BIAS_INDEX, convDescInfo.biasDesc) != GRAPH_SUCCESS,
                          OP_LOGE(UTIL_NAME, "Update %s bias tensor desc failed.", nodeName.GetString()), return false);
    }

    return true;
}

bool ConvFusionUtilsPass::BuildConv2dNode(Graph* graph, const std::string& nodeName,
                                          const std::vector<es::EsTensorHolder>& inputs, GNode& conv2dNode)
{
    const auto& fmap = inputs[INPUT_FMAP_INDEX];
    const auto& filter = inputs[INPUT_FILTER_INDEX];

    FUSION_PASS_CHECK(
        fmap.GetProducer() == nullptr || filter.GetProducer() == nullptr ||
            (inputs.size() == CONV_COUNT_PARAMS_BIAS && inputs[INPUT_BIAS_INDEX].GetProducer() == nullptr),
        OP_LOGE(UTIL_NAME, "input producer is nullptr for node %s.", nodeName.c_str()), return false);

    conv2dNode = es::CompliantNodeBuilder(graph)
                     .OpType(CONV2D.GetString())
                     .Name(nodeName.c_str())
                     .IrDefInputs({
                         {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                         {"filter", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                         {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                         {"offset_w", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                     })
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();

    FUSION_PASS_CHECK(graph->AddDataEdge(*fmap.GetProducer(), fmap.GetProducerOutIndex(), conv2dNode,
                                         INPUT_FMAP_INDEX) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Add fmap edge failed for node %s.", nodeName.c_str()), return false);
    FUSION_PASS_CHECK(graph->AddDataEdge(*filter.GetProducer(), filter.GetProducerOutIndex(), conv2dNode,
                                         INPUT_FILTER_INDEX) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Add filter edge failed for node %s.", nodeName.c_str()), return false);

    if (inputs.size() == CONV_COUNT_PARAMS_BIAS) {
        const auto& bias = inputs[INPUT_BIAS_INDEX];
        FUSION_PASS_CHECK(graph->AddDataEdge(*bias.GetProducer(), bias.GetProducerOutIndex(), conv2dNode,
                                             INPUT_BIAS_INDEX) != GRAPH_SUCCESS,
                          OP_LOGE(UTIL_NAME, "Add bias edge failed for node %s.", nodeName.c_str()), return false);
    }

    return true;
}

std::string GeFormatToString(const ge::Format& geFormat)
{
    return std::string(ge::TypeUtils::FormatToAscendString(geFormat).GetString());
}

std::string GeDtypeToString(const ge::DataType& geDtype)
{
    return std::string(ge::TypeUtils::DataTypeToAscendString(geDtype).GetString());
}

bool ConvFusionUtilsPass::CheckTransposeValid(const GNode& transposeNode, bool isPreFusion)
{
    AscendString nodeType;
    FUSION_PASS_CHECK(transposeNode.GetType(nodeType) != GRAPH_SUCCESS,
                      OP_LOGD(UTIL_NAME.c_str(), "Get node type failed."), return false);
    if (nodeType != TRANSPOSE_OP) {
        return false;
    }

    if (isPreFusion) {
        auto outNodes = transposeNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
        FUSION_PASS_CHECK(outNodes.size() != 1, OP_LOGD(UTIL_NAME.c_str(), "Pre-fusion transpose output size != 1."),
                          return false);
    }

    TensorDesc inDesc;
    TensorDesc outDesc;
    FUSION_PASS_CHECK(transposeNode.GetInputDesc(INPUT_INDEX_0, inDesc) != GRAPH_SUCCESS,
                      OP_LOGD(UTIL_NAME.c_str(), "Get transpose input desc failed."), return false);
    FUSION_PASS_CHECK(transposeNode.GetOutputDesc(OUTPUT_INDEX, outDesc) != GRAPH_SUCCESS,
                      OP_LOGD(UTIL_NAME.c_str(), "Get transpose output desc failed."), return false);

    auto inFormat = static_cast<Format>(ge::GetPrimaryFormat(static_cast<int32_t>(inDesc.GetFormat())));
    auto outFormat = static_cast<Format>(ge::GetPrimaryFormat(static_cast<int32_t>(outDesc.GetFormat())));
    bool formatMatch = (inFormat == FORMAT_NCHW && outFormat == FORMAT_NHWC) ||
                       (inFormat == FORMAT_NHWC && outFormat == FORMAT_NCHW);
    DataType inDtype = inDesc.GetDataType();
    DataType outDtype = outDesc.GetDataType();
    bool dtypeValid = (inDtype == DT_FLOAT16 || inDtype == DT_FLOAT) &&
                      (outDtype == DT_FLOAT16 || outDtype == DT_FLOAT);
    if (!dtypeValid || !formatMatch) {
        OP_LOGD(UTIL_NAME.c_str(), "Transpose node dtype or format not valid, keep transpose.");
        return false;
    }
    return true;
}

bool ConvFusionUtilsPass::RemoveTransposeNode(const GraphPtr& graph, GNode& transposeNode)
{
    auto inPair = transposeNode.GetInDataNodesAndPortIndexs(INPUT_INDEX_0);
    FUSION_PASS_CHECK(inPair.first == nullptr, OP_LOGE(UTIL_NAME.c_str(), "Get transpose input producer failed."),
                      return false);
    auto producer = inPair.first;
    int32_t producerPort = inPair.second;

    auto outNodes = transposeNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);

    FUSION_PASS_CHECK(graph->RemoveEdge(*producer, producerPort, transposeNode, INPUT_INDEX_0) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME.c_str(), "Remove transpose input edge failed."), return false);

    for (auto& outPair : outNodes) {
        if (outPair.first == nullptr) {
            continue;
        }
        FUSION_PASS_CHECK(
            graph->RemoveEdge(transposeNode, OUTPUT_INDEX, *outPair.first, outPair.second) != GRAPH_SUCCESS,
            OP_LOGE(UTIL_NAME.c_str(), "Remove transpose output edge failed."), return false);
        FUSION_PASS_CHECK(graph->AddDataEdge(*producer, producerPort, *outPair.first, outPair.second) != GRAPH_SUCCESS,
                          OP_LOGE(UTIL_NAME.c_str(), "Add bypass edge failed."), return false);
    }

    FUSION_PASS_CHECK(graph->RemoveNode(transposeNode) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME.c_str(), "Remove transpose node failed."), return false);
    return true;
}

bool ConvFusionUtilsPass::AdjustConvInputLayout(const GraphPtr& graph, GNode& convNode)
{
    auto inPair = convNode.GetInDataNodesAndPortIndexs(INPUT_FMAP_INDEX);
    FUSION_PASS_CHECK(inPair.first == nullptr, OP_LOGD(UTIL_NAME.c_str(), "Get conv input producer failed."),
                      return true);

    GNode preNode = *inPair.first;
    if (!CheckTransposeValid(preNode, true)) {
        return true;
    }

    TensorDesc transInDesc;
    TensorDesc transOutDesc;
    (void)preNode.GetInputDesc(INPUT_INDEX_0, transInDesc);
    (void)preNode.GetOutputDesc(OUTPUT_INDEX, transOutDesc);
    auto transInFormat = static_cast<Format>(ge::GetPrimaryFormat(static_cast<int32_t>(transInDesc.GetFormat())));
    auto transOutFormat = static_cast<Format>(ge::GetPrimaryFormat(static_cast<int32_t>(transOutDesc.GetFormat())));

    FUSION_PASS_CHECK_NOLOG(!RemoveTransposeNode(graph, preNode), return false);

    FUSION_PASS_CHECK(convNode.UpdateInputDesc(INPUT_FMAP_INDEX, transInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME.c_str(), "Update conv input desc failed."), return false);

    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    bool hasStrides = convNode.GetAttr(STRIDES, strides) == GRAPH_SUCCESS && strides.size() == 4;
    bool hasDilations = convNode.GetAttr(DILATIONS, dilations) == GRAPH_SUCCESS && dilations.size() == 4;
    if (transInFormat == FORMAT_NCHW && transOutFormat == FORMAT_NHWC) {
        if (hasStrides) {
            strides = ReorderNhwcToNchw(strides);
            FUSION_PASS_CHECK(convNode.SetAttr(STRIDES, strides) != GRAPH_SUCCESS,
                              OP_LOGE(UTIL_NAME.c_str(), "Set strides after reorder failed."), return false);
        }
        if (hasDilations) {
            dilations = ReorderNhwcToNchw(dilations);
            FUSION_PASS_CHECK(convNode.SetAttr(DILATIONS, dilations) != GRAPH_SUCCESS,
                              OP_LOGE(UTIL_NAME.c_str(), "Set dilations after reorder failed."), return false);
        }
    } else if (transInFormat == FORMAT_NHWC && transOutFormat == FORMAT_NCHW) {
        if (hasStrides) {
            strides = ReorderNchwToNhwc(strides);
            FUSION_PASS_CHECK(convNode.SetAttr(STRIDES, strides) != GRAPH_SUCCESS,
                              OP_LOGE(UTIL_NAME.c_str(), "Set strides after reorder failed."), return false);
        }
        if (hasDilations) {
            dilations = ReorderNchwToNhwc(dilations);
            FUSION_PASS_CHECK(convNode.SetAttr(DILATIONS, dilations) != GRAPH_SUCCESS,
                              OP_LOGE(UTIL_NAME.c_str(), "Set dilations after reorder failed."), return false);
        }
    }
    OP_LOGD(UTIL_NAME.c_str(), "AdjustConvInputLayout pre transpose success.");
    return true;
}

std::vector<int64_t> ConvFusionUtilsPass::ReorderNchwToNhwc(const std::vector<int64_t>& attr)
{
    return {attr[0], attr[2], attr[3], attr[1]};
}

std::vector<int64_t> ConvFusionUtilsPass::ReorderNhwcToNchw(const std::vector<int64_t>& attr)
{
    return {attr[0], attr[3], attr[1], attr[2]};
}

bool ConvFusionUtilsPass::CopyConvOtherAttrs(const GNode& srcNode, GNode& dstNode)
{
    const std::vector<AscendString> strAttrNames = {"_user_stream_label",  "_user_stream_priority",
                                                    "_super_kernel_scope", "_super_kernel_options",
                                                    "_op_aicore_num",      "_op_vectorcore_num"};
    for (const auto& attrName : strAttrNames) {
        AscendString attrValue;
        if (srcNode.GetAttr(attrName, attrValue) == GRAPH_SUCCESS) {
            (void)dstNode.SetAttr(attrName, attrValue);
        }
    }

    int64_t enableUncache = 0;
    if (srcNode.GetAttr("enable_uncache", enableUncache) == GRAPH_SUCCESS) {
        (void)dstNode.SetAttr("enable_uncache", enableUncache);
    }

    return true;
}

std::set<std::string> ConvFusionUtilsPass::CollectNodeTypeNames(const GraphPtr& graph, const AscendString& nodeType)
{
    std::set<std::string> names;
    for (auto& node : graph->GetDirectNode()) {
        AscendString curType;
        if (node.GetType(curType) != GRAPH_SUCCESS || curType != nodeType) {
            continue;
        }
        AscendString curName;
        if (node.GetName(curName) == GRAPH_SUCCESS) {
            names.insert(curName.GetString());
        }
    }
    return names;
}

bool ConvFusionUtilsPass::CopyDumpOutputMapping(const GNode& source, GNode& target)
{
    const auto readText = [&source](const char* key, AscendString& text) {
        AttrValue value;
        return source.GetOutputAttr(key, OUTPUT_INDEX, value) == GRAPH_SUCCESS &&
               value.GetAttrValue(text) == GRAPH_SUCCESS && text.GetString() != nullptr;
    };
    const auto writeText = [&target](const char* key, const AscendString& text) {
        return target.SetOutputAttr(key, OUTPUT_INDEX, es::CreateFrom(text)) == GRAPH_SUCCESS;
    };
    AscendString name;
    if (!readText("_datadump_origin_name", name) || name.GetString()[0] == '\0') {
        std::vector<AscendString> names;
        if (source.GetAttr("_datadump_original_op_names", names) == GRAPH_SUCCESS && !names.empty()) {
            name = names.front();
        } else if (source.GetName(name) != GRAPH_SUCCESS) {
            return false;
        }
    }
    AttrValue value;
    int64_t index = 0;
    if (source.GetOutputAttr("_datadump_origin_output_index", OUTPUT_INDEX, value) == GRAPH_SUCCESS) {
        (void)value.GetAttrValue(index);
    }
    TensorDesc desc;
    FUSION_PASS_CHECK(source.GetOutputDesc(OUTPUT_INDEX, desc) != GRAPH_SUCCESS,
                      OP_LOGE(UTIL_NAME, "Get dump source output desc failed."), return false);
    AscendString format;
    if (!readText("_datadump_origin_format", format) ||
        TypeUtils::SerialStringToFormat(format.GetString()) == FORMAT_RESERVED) {
        format = AscendString(TypeUtils::FormatToSerialString(desc.GetOriginFormat()).c_str());
    }
    AscendString dtype;
    if (!readText("_datadump_origin_data_type", dtype) ||
        TypeUtils::SerialStringToDataType(dtype.GetString()) == DT_UNDEFINED) {
#if GE_COMPILER_VERSION_NUM >= 90300000U
        const auto originType = desc.GetOriginDataType();
#else
        const auto originType = DT_UNDEFINED;
#endif
        dtype = AscendString(originType == DT_UNDEFINED ? "RESERVED" :
                                                          TypeUtils::DataTypeToSerialString(originType).c_str());
    }
    FUSION_PASS_CHECK(!writeText("_datadump_origin_name", name) ||
                          target.SetOutputAttr("_datadump_origin_output_index", OUTPUT_INDEX, es::CreateFrom(index)) !=
                              GRAPH_SUCCESS ||
                          !writeText("_datadump_origin_data_type", dtype) ||
                          !writeText("_datadump_origin_format", format),
                      OP_LOGE(UTIL_NAME, "Write dump output mapping failed."), return false);
    return true;
}

bool ConvFusionUtilsPass::FindNewNodeOfType(const GraphPtr& graph, const AscendString& nodeType,
                                            const std::set<std::string>& namesBefore, GNode& newNode)
{
    for (auto& node : graph->GetDirectNode()) {
        AscendString curType;
        if (node.GetType(curType) != GRAPH_SUCCESS || curType != nodeType) {
            continue;
        }
        AscendString curName;
        if (node.GetName(curName) != GRAPH_SUCCESS) {
            continue;
        }
        if (namesBefore.find(curName.GetString()) == namesBefore.end()) {
            newNode = node;
            return true;
        }
    }
    return false;
}

} // namespace ConvFusionUtils
} // namespace Conv
} // namespace NN
} // namespace Ops
