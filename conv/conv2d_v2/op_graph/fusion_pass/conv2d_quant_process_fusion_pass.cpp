/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv2d_quant_process_fusion_pass.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "graph/attr_value.h"
#include "graph/graph.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace Conv2dQuantProcessFusion;
namespace {
using ops::QuantProcessMode;

bool IsPadNode(const ge::GNodePtr& node) { return IsNodeOfType(node, PAD) || IsNodeOfType(node, PADV3); }

bool IsTinyProduct()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optiInfo) != 0) {
        return false;
    }
    const auto& intrinsicMap = platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_set_l1"];
    bool supportFp16 = std::find(intrinsicMap.begin(), intrinsicMap.end(), "f16") != intrinsicMap.end();
    bool supportFp32 = std::find(intrinsicMap.begin(), intrinsicMap.end(), "f32") != intrinsicMap.end();
    return supportFp16 && !supportFp32;
}

bool ResolveAippArch(NpuArch& npuArch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optiInfo) != 0) {
        return false;
    }
    static const std::map<std::string, NpuArch> socToArch = {
        {"Ascend310P", NpuArch::DAV_2002},   {"Ascend910", NpuArch::DAV_1001}, {"Ascend910B", NpuArch::DAV_2201},
        {"Ascend910_93", NpuArch::DAV_2201}, {"Ascend950", NpuArch::DAV_3510}, {"Ascend950PR", NpuArch::DAV_3510},
        {"Ascend950DT", NpuArch::DAV_3510}};
    auto iter = socToArch.find(platformInfo.str_info.short_soc_version);
    if (iter == socToArch.end()) {
        npuArch = NpuArch::DAV_RESV;
        return true;
    }
    npuArch = iter->second;
    return true;
}
} // namespace

void Conv2DQuantProcessFusionPass::InitMember()
{
    dequantNode = nullptr;
    padNode = nullptr;
    quantNode = nullptr;
    aippNode = nullptr;
    aippFirstConvType = AIPPFirstConvType::NONE_TYPE;
    meanAipp = 0.0F;
    scaleRate = 0.0F;
    aippChnInfo = AippChnInfo();
    quantProcessMode = QuantProcessMode::QUANT_UNDEFINED;
}

// 输入 0（可隔 Pad）回溯到 AscendQuant；输出侧找 AscendDequant
bool Conv2DQuantProcessFusionPass::CheckMatchStructure(const ge::GNode& matchNode)
{
    padNode = nullptr;
    auto convInputPair = matchNode.GetInDataNodesAndPortIndexs(INPUT_INDEX_0);
    if (IsPadNode(convInputPair.first)) {
        padNode = convInputPair.first;
        auto padInputPair = padNode->GetInDataNodesAndPortIndexs(INPUT_INDEX_0);
        convInputPair = padInputPair;
    }
    FUSION_PASS_CHECK(!IsNodeOfType(convInputPair.first, ASCEND_QUANT),
                      OP_LOGD(convDescInfo.nodeNameStr, "Conv2D input producer is not AscendQuant, no fusion."),
                      return false);
    quantNode = convInputPair.first;

    auto consumers = matchNode.GetOutDataNodesAndPortIndexs(0);
    for (auto& [consumer, inPort] : consumers) {
        (void)inPort;
        if (IsNodeOfType(consumer, ASCEND_DEQUANT)) {
            dequantNode = consumer;
            break;
        }
    }
    FUSION_PASS_CHECK(dequantNode == nullptr,
                      OP_LOGD(convDescInfo.nodeNameStr, "Conv2D output consumer is not AscendDequant, no fusion."),
                      return false);
    return true;
}

// 老 SetPadNodeDataType：rollback 时 Pad 双向 desc 改 fp16
ge::Status Conv2DQuantProcessFusionPass::SetPadNodeDataType(ge::GNode& padNodeRef)
{
    ge::TensorDesc inDesc;
    ge::TensorDesc outDesc;
    FUSION_PASS_CHECK(padNodeRef.GetInputDesc(0, inDesc) != ge::GRAPH_SUCCESS ||
                          padNodeRef.GetOutputDesc(0, outDesc) != ge::GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get pad desc failed."), return ge::FAILED);
    inDesc.SetDataType(ge::DT_FLOAT16);
    outDesc.SetDataType(ge::DT_FLOAT16);
    FUSION_PASS_CHECK(padNodeRef.UpdateInputDesc(0, inDesc) != ge::GRAPH_SUCCESS ||
                          padNodeRef.UpdateOutputDesc(0, outDesc) != ge::GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Update pad desc to float16 failed."), return ge::FAILED);
    return ge::SUCCESS;
}

ge::Status Conv2DQuantProcessFusionPass::GetCoValue(const ge::GNodePtr& cubeNode, int64_t& co)
{
    ge::TensorDesc filterDesc;
    if (cubeNode->GetInputDesc(static_cast<int32_t>(INPUT_FILTER_INDEX), filterDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGE(convDescInfo.nodeNameStr, "Get conv filter desc failed.");
        return ge::FAILED;
    }
    static const std::map<ge::Format, int32_t> axisOfCo = {
        {ge::Format::FORMAT_NCHW, 0}, {ge::Format::FORMAT_NHWC, 3}, {ge::Format::FORMAT_HWCN, 3}};
    const ge::Format filterFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(static_cast<int32_t>(filterDesc.GetFormat())));
    auto iter = axisOfCo.find(filterFormat);
    if (iter == axisOfCo.end()) {
        co = 1;
        return ge::SUCCESS;
    }
    const auto& filterDims = filterDesc.GetShape().GetDims();
    if (static_cast<size_t>(iter->second) >= filterDims.size()) {
        OP_LOGE(convDescInfo.nodeNameStr, "Filter dim num is less than Co axis index.");
        return ge::FAILED;
    }
    co = filterDims[static_cast<size_t>(iter->second)];
    return ge::SUCCESS;
}

// 基类 cube 全 desc 改 fp16 + conv2d 特化的 Pad 修正
ge::Status Conv2DQuantProcessFusionPass::SetDataTypeOfNodes(const ge::GNodePtr& cubeNode)
{
    if (BiasOptimizeQuantRollbackBase::SetDataTypeOfNodes(cubeNode) != ge::SUCCESS) {
        OP_LOGE(convDescInfo.nodeNameStr, "Set cube node data type failed.");
        return ge::FAILED;
    }
    if (padNode != nullptr && SetPadNodeDataType(*padNode) != ge::SUCCESS) {
        OP_LOGE(convDescInfo.nodeNameStr, "Update pad node dtype failed.");
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

// quant 与 cube 间隔 Pad 时，摘 quant 边重接到 Pad
ge::GNodePtr Conv2DQuantProcessFusionPass::GetCubeNodeInputNode(const ge::GNodePtr& cubeNode) const
{
    auto convInputPair = cubeNode->GetInDataNodesAndPortIndexs(INPUT_INDEX_0);
    if (IsPadNode(convInputPair.first)) {
        return convInputPair.first;
    }
    return cubeNode;
}

// 基类默认判定（不支持→ROLLBACK；C unknown→跳过；
// C>16 或不支持 fp16→BIAS_OPTIMIZE）+ conv2d 特化
ge::Status Conv2DQuantProcessFusionPass::GetQuantProcessMode(const ge::GNodePtr& quantNode,
                                                             const ge::GNodePtr& cubeNode,
                                                             QuantProcessMode& quantProcessMode)
{
    if (BiasOptimizeQuantRollbackBase::GetQuantProcessMode(quantNode, cubeNode, quantProcessMode) != ge::SUCCESS) {
        OP_LOGE(convDescInfo.nodeNameStr, "Base quant process mode decide failed.");
        return ge::FAILED;
    }
    if (quantProcessMode == QuantProcessMode::QUANT_UNDEFINED) {
        OP_LOGD(convDescInfo.nodeNameStr, "Quant input is unknown shape, skip fusion.");
        return ge::SUCCESS;
    }
    if (quantProcessMode == QuantProcessMode::QUANT_ROLLBACK && IsTinyProduct()) {
        quantProcessMode = QuantProcessMode::BIAS_OPTIMIZE;
    }
    if (quantProcessMode == QuantProcessMode::QUANT_ROLLBACK && aippFirstConvType != AIPPFirstConvType::NONE_TYPE) {
        quantProcessMode = QuantProcessMode::BIAS_OPTIMIZE;
    }
    return ge::SUCCESS;
}

bool Conv2DQuantProcessFusionPass::MeetRequirements(const ge::GNode& convNode)
{
    (void)convNode;
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do Conv2DQuantProcessFusionPass.");
    if (quantNode == nullptr || dequantNode == nullptr) {
        return false;
    }
    auto cubeNodePtr = dequantNode->GetInDataNodesAndPortIndexs(INPUT_INDEX_0).first;
    if (cubeNodePtr == nullptr) {
        return false;
    }
    if (GetQuantProcessMode(quantNode, cubeNodePtr, quantProcessMode) != ge::SUCCESS) {
        return false;
    }
    return quantProcessMode != QuantProcessMode::QUANT_UNDEFINED;
}

std::set<ge::AscendString> Conv2DQuantProcessFusionPass::GetNodeTypes() const { return {CONV2D}; }

void Conv2DQuantProcessFusionPass::PrintGraphStructure() const
{
    OP_LOGI(convDescInfo.nodeNameStr, "Conv2DQuantProcess fusion done: mode [%d], with pad [%d].",
            static_cast<int32_t>(quantProcessMode), padNode != nullptr);
}

// 开关值经 atc --compression_optimize_conf（ge.compressionOptimizeConf 指向的配置文件）传入，
// "enable_first_layer_quantization:true"
bool Conv2DQuantProcessFusionPass::IsFirstLayerQuantEnabled(const ge::CustomPassContext& passContext) const
{
    ge::AscendString confPath;
    if (passContext.GetOptionValue(ge::AscendString("ge.compressionOptimizeConf"), confPath) != ge::GRAPH_SUCCESS ||
        confPath.GetString() == nullptr || confPath.GetString()[0] == '\0') {
        OP_LOGD(convDescInfo.nodeNameStr,
                "Option [ge.compressionOptimizeConf] not set, first layer quantization disabled.");
        return false;
    }
    std::ifstream confFile(confPath.GetString());
    FUSION_PASS_CHECK(!confFile.is_open(),
                      OP_LOGD(convDescInfo.nodeNameStr, "Open conf file [%s] failed.", confPath.GetString()),
                      return false);
    std::string line;
    while (std::getline(confFile, line)) {
        const size_t pos = line.find(':');
        if (pos == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t\r\n") + 1);
        if (key == OPTION_FIRST_LAYER_QUANT && value == "true") {
            return true;
        }
    }
    return false;
}

extern "C" ge::graphStatus _ZNK2ge9AttrValue17GetNamedAttrValueERKNS_12AscendStringERS0_(
    const ge::AttrValue& aippValue, const ge::AscendString& attrName, ge::AttrValue& attrValue) __attribute__((weak));

namespace {
ge::Status GetAippInt64Value(const ge::AttrValue& aippValue, const std::string& key, int64_t& value)
{
    ge::AttrValue subValue;
    if (_ZNK2ge9AttrValue17GetNamedAttrValueERKNS_12AscendStringERS0_(aippValue, ge::AscendString(key.c_str()),
                                                                      subValue) != ge::GRAPH_SUCCESS) {
        return ge::FAILED;
    }
    return subValue.GetAttrValue(value) == ge::GRAPH_SUCCESS ? ge::SUCCESS : ge::FAILED;
}

// min_chn/var_reci_chn 为 vector<float> 时取首元素，标量直接读
ge::Status GetAippFloatValue(const ge::AttrValue& aippValue, const std::string& key, ge::float32_t& value)
{
    ge::AttrValue subValue;
    if (_ZNK2ge9AttrValue17GetNamedAttrValueERKNS_12AscendStringERS0_(aippValue, ge::AscendString(key.c_str()),
                                                                      subValue) != ge::GRAPH_SUCCESS) {
        return ge::FAILED;
    }
    if (subValue.GetAttrValue(value) == ge::GRAPH_SUCCESS) {
        return ge::SUCCESS;
    }
    std::vector<ge::float32_t> valueVec;
    if (subValue.GetAttrValue(valueVec) != ge::GRAPH_SUCCESS || valueVec.empty()) {
        return ge::FAILED;
    }
    value = valueVec.front();
    return ge::SUCCESS;
}
} // namespace

ge::Status Conv2DQuantProcessFusionPass::GetAippCfgInfo(const ge::GNode& aippNodeRef)
{
    FUSION_PASS_CHECK(
        _ZNK2ge9AttrValue17GetNamedAttrValueERKNS_12AscendStringERS0_ == nullptr,
        OP_LOGD(convDescInfo.nodeNameStr, "Runtime ge does not export AttrValue::GetNamedAttrValue, no aipp adjust."),
        return ge::FAILED);
    ge::AttrValue aippValue;
    if (aippNodeRef.GetAttr(ge::AscendString(ATTR_NAME_AIPP.c_str()), aippValue) != ge::GRAPH_SUCCESS) {
        OP_LOGD(convDescInfo.nodeNameStr, "Get aipp attr failed.");
        return ge::FAILED;
    }
    for (int32_t chnIdx = 0; chnIdx < K_MAX_CHN_NUM; ++chnIdx) {
        int64_t meanChn = 0;
        ge::float32_t minChn = 0.0F;
        ge::float32_t varReciChn = 0.0F;
        FUSION_PASS_CHECK(
            GetAippInt64Value(aippValue, K_AIPP_KEY[static_cast<size_t>(chnIdx)][0], meanChn) != ge::SUCCESS,
            OP_LOGD(convDescInfo.nodeNameStr, "Get aipp mean_chn_%d failed.", chnIdx), return ge::FAILED);
        FUSION_PASS_CHECK(
            GetAippFloatValue(aippValue, K_AIPP_KEY[static_cast<size_t>(chnIdx)][1], minChn) != ge::SUCCESS,
            OP_LOGD(convDescInfo.nodeNameStr, "Get aipp min_chn_%d failed.", chnIdx), return ge::FAILED);
        FUSION_PASS_CHECK(
            GetAippFloatValue(aippValue, K_AIPP_KEY[static_cast<size_t>(chnIdx)][2], varReciChn) != ge::SUCCESS,
            OP_LOGD(convDescInfo.nodeNameStr, "Get aipp var_reci_chn_%d failed.", chnIdx), return ge::FAILED);
        aippChnInfo.meanChnVec.emplace_back(meanChn);
        aippChnInfo.minChnVec.emplace_back(minChn);
        aippChnInfo.varReciChnVec.emplace_back(varReciChn);
    }
    return ge::SUCCESS;
}

bool Conv2DQuantProcessFusionPass::CheckAippVarChn() const
{
    if (aippChnInfo.varReciChnVec.empty()) {
        return false;
    }
    for (size_t i = 1; i < aippChnInfo.varReciChnVec.size(); ++i) {
        if (!FloatEqual(aippChnInfo.varReciChnVec[i], aippChnInfo.varReciChnVec[0])) {
            OP_LOGD(convDescInfo.nodeNameStr, "Aipp not all var channel's value is equal.");
            return false;
        }
    }
    return true;
}

// 按 AIPP var_reci 均值改写 quant 的 scale/offset，记录 scale_rate 供 dequant/host op
ge::Status Conv2DQuantProcessFusionPass::QuantParamRevise()
{
    FUSION_PASS_CHECK(dequantNode == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "Dequant node is null."),
                      return ge::FAILED);
    FUSION_PASS_CHECK(aippChnInfo.varReciChnVec.empty() || aippChnInfo.meanChnVec.empty(),
                      OP_LOGE(convDescInfo.nodeNameStr, "Aipp channel info is empty."), return ge::FAILED);
    ge::float32_t qScale = 0.0F;
    FUSION_PASS_CHECK(quantNode->GetAttr(ATTR_SCALE.c_str(), qScale) != ge::GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get quant scale attr failed."), return ge::FAILED);
    FUSION_PASS_CHECK(FloatEqual(qScale, 0.0F), OP_LOGE(convDescInfo.nodeNameStr, "Quant scale is 0."),
                      return ge::FAILED);
    ge::float32_t newScale = FloatEqual(aippChnInfo.varReciChnVec[0], 0.0F) ? 0.0F :
                                                                              1.0F / aippChnInfo.varReciChnVec[0];
    FUSION_PASS_CHECK(quantNode->SetAttr(ATTR_SCALE.c_str(), newScale) != ge::GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set quant scale attr failed."), return ge::FAILED);
    ge::float32_t newOffset = 0.0F;
    for (size_t i = 0; i < aippChnInfo.meanChnVec.size(); ++i) {
        newOffset += aippChnInfo.meanChnVec[i] + aippChnInfo.minChnVec[i];
    }
    if (!aippChnInfo.meanChnVec.empty()) {
        newOffset /= static_cast<ge::float32_t>(aippChnInfo.meanChnVec.size());
    }
    meanAipp = newOffset;
    newOffset = newOffset - static_cast<ge::float32_t>(K_BASE_OFFSET);
    FUSION_PASS_CHECK(quantNode->SetAttr(ATTR_OFFSET.c_str(), newOffset) != ge::GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set quant offset attr failed."), return ge::FAILED);
    scaleRate = newScale / qScale;
    FUSION_PASS_CHECK(dequantNode->SetAttr(K_QUANT_SCALE_RATE.c_str(), scaleRate) != ge::GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set dequant scale rate attr failed."), return ge::FAILED);
    return ge::SUCCESS;
}

ge::Status Conv2DQuantProcessFusionPass::ConvFusionPreImpl(ge::GraphPtr& graph, ge::GNode& convNode,
                                                           ge::CustomPassContext& passContext)
{
    (void)graph;
    (void)convNode;
    if (!IsFirstLayerQuantEnabled(passContext)) {
        return ge::SUCCESS;
    }
    if (quantNode == nullptr || quantNode->GetInputsSize() != 1) {
        OP_LOGD(convDescInfo.nodeNameStr, "Quant node in data size is not one, no aipp adjust.");
        return ge::SUCCESS;
    }
    auto preNode = quantNode->GetInDataNodesAndPortIndexs(INPUT_INDEX_0).first;
    if (!IsNodeOfType(preNode, AIPP)) {
        OP_LOGD(convDescInfo.nodeNameStr, "Quant input producer is not Aipp, no aipp adjust.");
        return ge::SUCCESS;
    }
    NpuArch npuArch = NpuArch::DAV_RESV;
    if (!ResolveAippArch(npuArch)) {
        return ge::SUCCESS;
    }
    if (npuArch == NpuArch::DAV_2002) {
        OP_LOGD(convDescInfo.nodeNameStr, "Current arch not support aipp adjust yet.");
        return ge::SUCCESS;
    }
    if (npuArch != NpuArch::DAV_1001 && npuArch != NpuArch::DAV_2201) {
        OP_LOGD(convDescInfo.nodeNameStr, "Current arch no need adjust aipp param.");
        aippFirstConvType = AIPPFirstConvType::NO_NEED_ADJUST_TYPE;
        return ge::SUCCESS;
    }
    FUSION_PASS_CHECK(GetAippCfgInfo(*preNode) != ge::GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "Get aipp cfg info failed, no aipp adjust."),
                      return ge::SUCCESS);
    if (!CheckAippVarChn()) {
        return ge::SUCCESS;
    }
    aippNode = preNode;
    bool needAdjust = true;
    FUSION_PASS_CHECK(quantNode->SetAttr(K_AIPP_QUANT_NEED_ADJUST.c_str(), needAdjust) != ge::GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Set quant aipp adjust flag failed."), return ge::FAILED);
    aippFirstConvType = AIPPFirstConvType::NEED_ADJUST_TYPE;
    FUSION_PASS_CHECK(QuantParamRevise() != ge::SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "Quant param revise failed."), return ge::FAILED);
    return ge::SUCCESS;
}

// NEED_ADJUST 时回写 AIPP mean 向量 + host op 的 scale_rate
void Conv2DQuantProcessFusionPass::PostProcAfterOptimize(const std::vector<ge::GNode>& nodesAfterFuse)
{
    if (aippFirstConvType != AIPPFirstConvType::NEED_ADJUST_TYPE || aippNode == nullptr) {
        return;
    }
    std::vector<int64_t> aippMeanVec;
    for (size_t i = 0; i < aippChnInfo.meanChnVec.size(); ++i) {
        int64_t meanTmp = static_cast<int64_t>(std::round(
                              static_cast<double>(aippChnInfo.meanChnVec[i] + aippChnInfo.minChnVec[i]) - meanAipp)) +
                          K_BASE_OFFSET;
        aippMeanVec.emplace_back(meanTmp);
    }
    if (aippNode->SetAttr(K_AIPP_QUANT_MEAN_VEC.c_str(), aippMeanVec) != ge::GRAPH_SUCCESS) {
        OP_LOGE(convDescInfo.nodeNameStr, "Aipp set mean attr failed.");
        return;
    }
    for (const auto& node : nodesAfterFuse) {
        ge::AscendString nodeType;
        FUSION_PASS_CHECK_NOLOG(node.GetType(nodeType) != ge::GRAPH_SUCCESS, continue);
        if (nodeType != ge::AscendString("QuantBiasOptimization")) {
            continue;
        }
        if (node.SetAttr(K_QUANT_SCALE_RATE.c_str(), scaleRate) != ge::GRAPH_SUCCESS) {
            OP_LOGE(convDescInfo.nodeNameStr, "Set bias host cpu op scale rate failed.");
        }
        break;
    }
}

bool Conv2DQuantProcessFusionPass::ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& convNode,
                                                         ge::CustomPassContext& passContext)
{
    ge::AscendString convNodeName;
    (void)convNode.GetName(convNodeName);
    OP_LOGI(convDescInfo.nodeNameStr,
            "New op_graph fusion pass Conv2DQuantProcessFusionPass execute fusion on node [%s].",
            convNodeName.GetString() == nullptr ? "" : convNodeName.GetString());
    std::vector<ge::GNode> nodesBeforeFuse = {convNode};
    if (quantNode != nullptr) {
        nodesBeforeFuse.emplace_back(*quantNode);
    }
    if (padNode != nullptr) {
        nodesBeforeFuse.emplace_back(*padNode);
    }
    if (dequantNode != nullptr) {
        nodesBeforeFuse.emplace_back(*dequantNode);
    }
    FUSION_PASS_CHECK_NOLOG(!CanFuseNodes(nodesBeforeFuse), return false);

    auto cubeNodePtr = dequantNode->GetInDataNodesAndPortIndexs(INPUT_INDEX_0).first;
    FUSION_PASS_CHECK(cubeNodePtr == nullptr,
                      OP_LOGE(convDescInfo.nodeNameStr, "Get conv GNodePtr from dequant input failed."), return false);
    std::vector<ge::GNodePtr> fusionNodes;
    std::vector<ge::GNodePtr> nodesToRemove;
    if (OptimizeMainProc(graph, quantNode, dequantNode, cubeNodePtr, fusionNodes, nodesToRemove) != ge::SUCCESS) {
        OP_LOGE(convDescInfo.nodeNameStr, "Quant process fusion failed.");
        return false;
    }

    std::vector<ge::GNode> nodesAfterFuse = {convNode};
    for (const auto& fusedNode : fusionNodes) {
        if (fusedNode != nullptr) {
            nodesAfterFuse.emplace_back(*fusedNode);
        }
    }
    PostProcAfterOptimize(nodesAfterFuse);
    // 基类不再删节点（nodesToRemove 交由调用方处理），Report 时 before 节点仍图上有效；
    // 删除须在 Report 之后：GE ReportFuse 的 ConvertGNodes 对已删节点 weak_ptr 悬空会失败
    FUSION_PASS_CHECK_NOLOG(!ReportFuseNodes(nodesBeforeFuse, nodesAfterFuse, passContext), return false);
    for (const auto& nodeToRemove : nodesToRemove) {
        if (nodeToRemove != nullptr && graph->RemoveNode(*nodeToRemove) != ge::GRAPH_SUCCESS) {
            OP_LOGE(convDescInfo.nodeNameStr, "Remove node failed after report fuse.");
            return false;
        }
    }
    return true;
}

} // namespace Ops
