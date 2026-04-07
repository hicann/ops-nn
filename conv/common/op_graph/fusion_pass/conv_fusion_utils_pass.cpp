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
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "runtime/runtime/base.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace ConvFusionUtils {
using namespace ge;
using namespace fe;
using namespace fusion;

namespace {
constexpr int32_t BASE = 10;
constexpr int32_t NPU_ARCH_VAL_MAX_LEN = 32;
}

bool ConvFusionUtilsPass::AddSubgraphInput(std::unique_ptr<SubgraphBoundary> &boundary, const GNode &node,
    const int64_t subgraphIndex, const int64_t boundaryIndex)
{
    SubgraphInput subgraphInput;
    FUSION_PASS_CHECK(subgraphInput.AddInput({node, subgraphIndex}) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Create SubgraphInput failed."), return false);

    FUSION_PASS_CHECK(boundary->AddInput(boundaryIndex, std::move(subgraphInput)) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "SubgraphBoundary add SubgraphInput failed."), return false);

    return true;
}

bool ConvFusionUtilsPass::AddSubgraphOutput(std::unique_ptr<SubgraphBoundary> &boundary, const GNode &node,
    const int64_t subgraphIndex, const int64_t boundaryIndex)
{
    SubgraphOutput subgraphOutput;
    FUSION_PASS_CHECK(subgraphOutput.SetOutput({node, subgraphIndex}) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Create SubgraphOutput failed."), return false);

    FUSION_PASS_CHECK(boundary->AddOutput(boundaryIndex, std::move(subgraphOutput)) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "SubgraphBoundary add SubgraphOutput failed."), return false);

    return true;
}

bool ConvFusionUtilsPass::CheckSocSupport(const std::map<std::string, NpuArch> &supportSocList, NpuArch &npuArch)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get platform_info failed."), return false);
    const std::string soc = platformInfo.str_info.short_soc_version;

    FUSION_PASS_CHECK(supportSocList.find(soc) == supportSocList.end(),
        OP_LOGE(UTIL_NAME, "Current soc %s not supported.", soc.c_str()), return false);

    npuArch = supportSocList.at(soc);
    OP_LOGD(UTIL_NAME, "Current NpuArch is DAV_%u.", npuArch);

    return true;
}

bool ConvFusionUtilsPass::GetConvBaseAttr(const GNode &convNode, ConvBaseAttrs &baseAttrs,
    const ConvDescInfo &convDescInfo)
{
    baseAttrs = ConvBaseAttrs();
    FUSION_PASS_CHECK(convNode.GetAttr(STRIDES, baseAttrs.strides) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get strides from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);

    FUSION_PASS_CHECK(convNode.GetAttr(PADS, baseAttrs.pads) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get pads from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);

    FUSION_PASS_CHECK(convNode.GetAttr(DILATIONS, baseAttrs.dilations) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get dilations from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);

    FUSION_PASS_CHECK(convNode.GetAttr(GROUPS, baseAttrs.groups) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get groups from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);

    FUSION_PASS_CHECK(convNode.GetAttr(DATA_FORMAT, baseAttrs.dataFormat) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get data_format from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);

    FUSION_PASS_CHECK(convNode.GetAttr(OFFSET_X, baseAttrs.offsetX) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get offset_x from %s failed.", convDescInfo.nodeNameStr.c_str()), return false);

    AscendString attrPadding = "";
    convNode.GetAttr(PADDING, attrPadding);
    AscendString attrAutoPad = "";
    convNode.GetAttr(AUTO_PAD, attrAutoPad);

    AscendString tmpPadMode = attrPadding.GetLength() != 0 ? attrPadding :
        (attrAutoPad.GetLength() != 0 ? attrAutoPad : "NOTSET");
    baseAttrs.padMode = SPECIFIC_PAD_LIST.count(tmpPadMode) != 0 ? "SPECIFIC" : tmpPadMode;

    convNode.GetAttr(OP_IMPL_MODE_ENUM, baseAttrs.opImplModeEnum);
    int64_t opImplModeEnum = baseAttrs.opImplModeEnum;
    bool isHf32OpImplMode = std::any_of(HF32_PRECISION_MODES_INT.begin(), HF32_PRECISION_MODES_INT.end(),
                                        [&opImplModeEnum](const int64_t& targetPrecisionModeInt) {
                                            return opImplModeEnum == targetPrecisionModeInt;
                                        });
    baseAttrs.enableHf32 = (convDescInfo.fmapDtype == DataType::DT_FLOAT && isHf32OpImplMode);

    // Set Node's implMode to default
    baseAttrs.opImplModeEnum = 0x1;
    FUSION_PASS_CHECK(convNode.SetAttr(OP_IMPL_MODE_ENUM, baseAttrs.opImplModeEnum) != GRAPH_SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "Set _op_impl_mode_enum for %s failed.", convDescInfo.nodeNameStr.c_str()),
        return false);

    return true;
}

bool ConvFusionUtilsPass::GetConvDescInfo(const GNode &convNode, ConvDescInfo &convDescInfo)
{
    convDescInfo = ConvDescInfo();
    FUSION_PASS_CHECK(convNode.GetName(convDescInfo.nodeName) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get node name failed."), return false);
    convDescInfo.nodeNameStr = convDescInfo.nodeName.GetString();

    convDescInfo.hasBias = convNode.GetInputsSize() == CONV_COUNT_PARAMS_BIAS;

    FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_FMAP_INDEX, convDescInfo.fmapDesc) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get %s fmap tensor desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_FILTER_INDEX, convDescInfo.filterDesc) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get %s filter tensor desc failed.", convDescInfo.nodeNameStr.c_str()),
        return false);
    if (convDescInfo.hasBias) {
        FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_BIAS_INDEX, convDescInfo.biasDesc) != GRAPH_SUCCESS,
            OP_LOGE(UTIL_NAME, "Get %s bias tensor desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    }
    FUSION_PASS_CHECK(convNode.GetOutputDesc(OUTPUT_INDEX, convDescInfo.outputDesc) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get %s output tensor desc failed.", convDescInfo.nodeNameStr.c_str()), return false);

    convDescInfo.fmapDtype = convDescInfo.fmapDesc.GetDataType();
    convDescInfo.filterDtype = convDescInfo.filterDesc.GetDataType();
    if (convDescInfo.hasBias) {
        convDescInfo.biasDtype = convDescInfo.biasDesc.GetDataType();
    }
    convDescInfo.outputDtype = convDescInfo.outputDesc.GetDataType();

    convDescInfo.fmapFormat = convDescInfo.fmapDesc.GetFormat();
    convDescInfo.filterFormat = convDescInfo.filterDesc.GetFormat();
    if (convDescInfo.hasBias) {
        convDescInfo.biasFormat = convDescInfo.biasDesc.GetFormat();
    }
    convDescInfo.outputFormat = convDescInfo.outputDesc.GetFormat();

    PrintConvDescInfo(convDescInfo);

    return true;
}

bool ConvFusionUtilsPass::GetMatchedNodes(const GraphPtr &graph, std::vector<GNode> &matchedNodes,
    const AscendString &nodeType)
{
    for (auto &node : graph->GetDirectNode()) {
        AscendString curNodeType;
        FUSION_PASS_CHECK(node.GetType(curNodeType) != GRAPH_SUCCESS,
            OP_LOGE(UTIL_NAME, "GetType in GetMatchedNodes failed."), return false);

        if (curNodeType == nodeType) {
            matchedNodes.emplace_back(node);
        }
    }

    return true;
}

GNodePtr ConvFusionUtilsPass::GetNodePtr(const GNode &node, const ConvDescInfo &convDescInfo)
{
    auto convOutputNodes = node.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK(convOutputNodes.empty(),
        OP_LOGD(UTIL_NAME, "%s out nodes is empty.", convDescInfo.nodeNameStr.c_str()), return nullptr);

    GNodePtr outNodePtr = convOutputNodes[0].first;
    int32_t outNodePortIndex = convOutputNodes[0].second;

    GNodePtr nodePtr = outNodePtr->GetInDataNodesAndPortIndexs(outNodePortIndex).first;
    FUSION_PASS_CHECK(nodePtr == nullptr,
        OP_LOGD(UTIL_NAME, "%s get in data nodes failed.", convDescInfo.nodeNameStr.c_str()), return nullptr);

    AscendString name;
    FUSION_PASS_CHECK(nodePtr->GetName(name) != GRAPH_SUCCESS,
        OP_LOGD(UTIL_NAME, "%s get node name by node ptr failed.", convDescInfo.nodeNameStr.c_str()), return nullptr);

    FUSION_PASS_CHECK(name != convDescInfo.nodeName,
        OP_LOGD(UTIL_NAME, "%s get node ptr failed.", convDescInfo.nodeNameStr.c_str()), return nullptr);

    return nodePtr;
}

AscendString ConvFusionUtilsPass::ListToAscendString(const std::vector<AscendString> &strList)
{
    std::string res = "";
    for (uint32_t index = 0; index < strList.size(); ++index) {
        res += strList[index].GetString();
        if (index != strList.size() - 1) {
            res += ", ";
        }
    }

    return AscendString(res.c_str());
}

void ConvFusionUtilsPass::PrintConvDescInfo(const ConvDescInfo &convDescInfo)
{
    std::string convDtypeInfo = "fmap is " + TypeUtils::DataTypeToSerialString(convDescInfo.fmapDtype) +
                                " filter is " + TypeUtils::DataTypeToSerialString(convDescInfo.filterDtype) +
                                " output is " + TypeUtils::DataTypeToSerialString(convDescInfo.outputDtype);

    std::string convFormatInfo = "fmap is " + TypeUtils::FormatToSerialString(convDescInfo.fmapFormat) +
                                 " filter is " + TypeUtils::FormatToSerialString(convDescInfo.filterFormat) +
                                 " output is " + TypeUtils::FormatToSerialString(convDescInfo.outputFormat);

    if (convDescInfo.hasBias) {
        convDtypeInfo += " bias is " + TypeUtils::DataTypeToSerialString(convDescInfo.biasDtype);
        convFormatInfo += " bias is " + TypeUtils::FormatToSerialString(convDescInfo.biasFormat);
    }

    OP_LOGD(UTIL_NAME, "%s dtype: %s", convDescInfo.nodeNameStr.c_str(), convDtypeInfo.c_str());
    OP_LOGD(UTIL_NAME, "%s format: %s", convDescInfo.nodeNameStr.c_str(), convFormatInfo.c_str());
}

bool ConvFusionUtilsPass::UpdateInputDesc(GNode *convNode, const ConvDescInfo &convDescInfo)
{
    FUSION_PASS_CHECK(convNode == nullptr,
        OP_LOGE(UTIL_NAME, "Node is nullptr, update input desc failed."), return false);

    AscendString nodeName = "";
    FUSION_PASS_CHECK(convNode->GetName(nodeName) != GRAPH_SUCCESS,
        OP_LOGE(UTIL_NAME, "Get node name failed."), return false);

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

} // namespace ConvFusionUtilsPass
} // namespace Conv
} // namespace NN
} // namespace Ops