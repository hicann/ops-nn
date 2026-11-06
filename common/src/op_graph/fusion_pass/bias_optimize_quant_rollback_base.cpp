/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fusion_pass/bias_optimize_quant_rollback_base.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "error_util.h"
#include "ge/compliant_node_builder.h"
#include "ge/ge_api_error_codes.h"
#include "ge/ge_utils.h"
#include "graph/attr_value.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"

namespace ops {
namespace {
constexpr const char* const kLogTag = "BiasOptQuant";

constexpr const char* const kEnterOpType = "Enter";
constexpr const char* const kDataOpType = "Data";
constexpr const char* const kRefDataOpType = "RefData";
constexpr const char* const kConstOpType = "Const";
constexpr const char* const kPlaceHolderOpType = "PlaceHolder";
constexpr const char* const kConstantOpType = "Constant";
constexpr const char* const kAttrParentNodeIndex = "_parent_node_index";
constexpr const char* const kAttrIndex = "index";
constexpr const char* const kAttrNameGroups = "groups";
constexpr const char* const kAttrTransposeX2 = "transpose_x2";
constexpr const char* const kAttrAdjX2 = "adj_x2";
constexpr const char* const kAttrTranspose = "transpose";
constexpr const char* const kAttrWeightTranspose = "weight_transpose";
constexpr const char* const kIsComeFromConstOp = "_is_come_from_const_op";
constexpr const char* const kAttrCubeOpType = "cube_op_type";
constexpr const char* const kAttrCinCoutReverse = "quant_cin_cout_reverse";
constexpr const char* const kAttrOffset = "offset";
constexpr const char* const kAttrScale = "scale";
constexpr const char* const kAttrOffsetX = "offset_x";
constexpr const char* const kAttrDstType = "dst_type";
constexpr const char* const kStrSocVersion = "soc_version";
constexpr const char* const kQuantWeightRollBack = "QuantWeightRollBack";
constexpr const char* const kQuantBiasRollBack = "QuantBiasRollBack";
constexpr const char* const kQuantBiasOptimization = "QuantBiasOptimization";
constexpr const char* const kCubeQuantRollBackOutput = "cube_quant_roll_back_output";
constexpr const char* const kCubeFilter = "cube_filter";
constexpr const char* const kCubeBias = "cube_bias";
constexpr const char* const kCubeOptimizationBias = "cube_optimization_bias";
constexpr const char* const kCubeOptimizationFilter = "cube_optimization_filter";
constexpr const char* const kIrAttrValueTypeTensor = "VT_TENSOR";
constexpr const char* const kIrOutputName = "y";
constexpr const char* const kIrInputScaleName = "dequant_scale";

constexpr int64_t kGroupsDefaultValue = 1;
constexpr int64_t kFp16C0DefaultValue = 16;
constexpr size_t kLastAxisIndex = 3U;
constexpr size_t kAicVersionParts = 3U;
constexpr size_t kMknPartsNum = 3U;
constexpr size_t kAicVersionPartIndex = 2U;
constexpr size_t kMinInputsWithWeight = 2U;
constexpr int64_t kMaxBiasElementNum = INT64_MAX / static_cast<int64_t>(sizeof(int32_t));
constexpr uint32_t kMaxSubGraphTraceDepth = 16U;

const std::vector<std::vector<std::string>> kTensorNameOfHostCpuOp = {{kCubeFilter},
                                                                      {kCubeBias},
                                                                      {kCubeOptimizationBias, kCubeOptimizationFilter},
                                                                      {kCubeOptimizationBias, kCubeOptimizationFilter},
                                                                      {kCubeFilter},
                                                                      {kCubeBias}};

const std::vector<std::vector<uint32_t>> kOriginalConvAnchorIndex = {{1U}, {2U}, {2U, 1U}, {3U, 2U}, {2U}, {3U}};

constexpr int32_t kNchwDimN = 0;
constexpr int32_t kNchwDimC = 1;
constexpr int32_t kNchwDimH = 2;
constexpr int32_t kNchwDimW = 3;
constexpr int32_t kHwcnDimH = 0;
constexpr int32_t kHwcnDimW = 1;
constexpr int32_t kHwcnDimC = 2;
constexpr int32_t kHwcnDimN = 3;
constexpr int32_t kNhwcDimN = 0;
constexpr int32_t kNhwcDimH = 1;
constexpr int32_t kNhwcDimW = 2;
constexpr int32_t kNhwcDimC = 3;
constexpr int32_t NDHWC_DIM_N = 0;
constexpr int32_t NDHWC_DIM_H = 2;
constexpr int32_t NDHWC_DIM_W = 3;
constexpr int32_t NDHWC_DIM_C = 4;
constexpr int32_t NCDHW_DIM_N = 0;
constexpr int32_t NCDHW_DIM_C = 1;
constexpr int32_t NCDHW_DIM_H = 3;
constexpr int32_t NCDHW_DIM_W = 4;
constexpr int32_t DHWCN_DIM_H = 1;
constexpr int32_t DHWCN_DIM_W = 2;
constexpr int32_t DHWCN_DIM_C = 3;
constexpr int32_t DHWCN_DIM_N = 4;

const std::map<ge::Format, std::map<std::string, int32_t>> kAxisIndexOfFormat = {
    {ge::FORMAT_NCHW, {{"N", kNchwDimN}, {"C", kNchwDimC}, {"H", kNchwDimH}, {"W", kNchwDimW}}},
    {ge::FORMAT_HWCN, {{"N", kHwcnDimN}, {"C", kHwcnDimC}, {"H", kHwcnDimH}, {"W", kHwcnDimW}}},
    {ge::FORMAT_NHWC, {{"N", kNhwcDimN}, {"C", kNhwcDimC}, {"H", kNhwcDimH}, {"W", kNhwcDimW}}},
    {ge::FORMAT_NDHWC, {{"N", NDHWC_DIM_N}, {"C", NDHWC_DIM_C}, {"H", NDHWC_DIM_H}, {"W", NDHWC_DIM_W}}},
    {ge::FORMAT_NCDHW, {{"N", NCDHW_DIM_N}, {"C", NCDHW_DIM_C}, {"H", NCDHW_DIM_H}, {"W", NCDHW_DIM_W}}},
    {ge::FORMAT_DHWCN, {{"N", DHWCN_DIM_N}, {"C", DHWCN_DIM_C}, {"H", DHWCN_DIM_H}, {"W", DHWCN_DIM_W}}}};
std::atomic<uint64_t> gHostCpuAtomicId{0};

uint64_t GetHostCpuAtomicId() { return gHostCpuAtomicId.fetch_add(1U); }

void SetOriginDtype(ge::TensorDesc& desc, const ge::DataType& dstDtype)
{
#if GE_COMPILER_VERSION_NUM >= 90300000U
    if (desc.GetOriginDataType() != dstDtype) {
        desc.SetOriginDataType(dstDtype);
    }
#endif
}

ge::DataType GetOriginDtype(const ge::TensorDesc& desc)
{
#if GE_COMPILER_VERSION_NUM >= 90300000U
    return desc.GetOriginDataType();
#else
    return desc.GetDataType();
#endif
}

std::string NodeName(const ge::GNode& node)
{
    ge::AscendString name;
    if (node.GetName(name) != ge::GRAPH_SUCCESS) {
        return "";
    }
    return (name.GetString() == nullptr) ? "" : name.GetString();
}

std::string NodeType(const ge::GNode& node)
{
    ge::AscendString type;
    if (node.GetType(type) != ge::GRAPH_SUCCESS) {
        return "";
    }
    return (type.GetString() == nullptr) ? "" : type.GetString();
}

bool CheckInt64MulOverflow(const int64_t a, const int64_t b)
{
    if (a > 0 && b > 0 && a > INT64_MAX / b) {
        return false;
    }
    if (a > 0 && b < 0 && b < INT64_MIN / a) {
        return false;
    }
    if (a < 0 && b > 0 && a < INT64_MIN / b) {
        return false;
    }
    if (a < 0 && b < 0 && b < INT64_MAX / a) {
        return false;
    }
    return true;
}

ge::Status PadShapeTo4Dim(const ge::Format& filterFormat, const std::vector<int64_t>& filterDims,
                          std::vector<int64_t>& filterDims4D)
{
    const size_t sizeOfFilter = filterDims.size();
    for (size_t i = 0U; i <= kLastAxisIndex; ++i) {
        if (i < sizeOfFilter) {
            filterDims4D.emplace_back(filterDims.at(i));
            continue;
        }
        if (filterFormat == ge::FORMAT_NCHW) {
            filterDims4D.emplace_back(1);
        } else if (filterFormat == ge::FORMAT_HWCN) {
            filterDims4D.insert(filterDims4D.cbegin(), 1);
        } else if (filterFormat == ge::FORMAT_NHWC) {
            filterDims4D.insert(filterDims4D.cbegin() + 1, 1);
        } else if (filterFormat == ge::FORMAT_ND) {
            filterDims4D.emplace_back(0);
        } else {
            OPS_LOG_E(kLogTag, "Format %s can not pad shape.",
                      ge::TypeUtils::FormatToSerialString(filterFormat).c_str());
            return ge::FAILED;
        }
    }
    return ge::SUCCESS;
}

int64_t ParseInt64(const std::string& str)
{
    errno = 0;
    const long long value = std::strtoll(str.c_str(), nullptr, 10);
    if (errno == ERANGE) {
        return 0;
    }
    return static_cast<int64_t>(value);
}

std::vector<std::string> SplitString(const std::string& str, const char sep)
{
    std::vector<std::string> parts;
    std::string item;
    std::istringstream stream(str);
    while (std::getline(stream, item, sep)) {
        parts.push_back(item);
    }
    return parts;
}

int64_t GetC0OfFloat16()
{
    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfos, optionalInfos) != 0U) {
        return kFp16C0DefaultValue;
    }
    std::map<std::string, std::string> mknMap;
    if (!platformInfos.GetPlatformRes("MKN", mknMap) || mknMap.empty()) {
        return kFp16C0DefaultValue;
    }
    std::string mknValue;
    if (mknMap.count("float16") != 0U) {
        mknValue = mknMap.at("float16");
    } else if (mknMap.count("default") != 0U) {
        mknValue = mknMap.at("default");
    } else {
        return kFp16C0DefaultValue;
    }
    const std::vector<std::string> parts = SplitString(mknValue, ',');
    if (parts.size() != kMknPartsNum) {
        return kFp16C0DefaultValue;
    }
    const int64_t c0Value = ParseInt64(parts[1]);
    return (c0Value > 0) ? c0Value : kFp16C0DefaultValue;
}

std::string GetIsaArchVersionStr()
{
    static const std::map<int32_t, std::string> kIsaVersionStrMap = {{100, "v100"}, {200, "v200"}, {202, "v200"},
                                                                     {210, "v200"}, {220, "v220"}, {300, "v300"},
                                                                     {310, "v300"}, {350, "v350"}};
    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfos, optionalInfos) != 0U) {
        return "";
    }
    std::string aicVersionStr;
    if (!platformInfos.GetPlatformRes("version", "AIC_version", aicVersionStr) || aicVersionStr.empty()) {
        return "";
    }
    const std::vector<std::string> parts = SplitString(aicVersionStr, '-');
    if (parts.size() < kAicVersionParts) {
        return "";
    }
    const int32_t aicVersion = static_cast<int32_t>(ParseInt64(parts[kAicVersionPartIndex]));
    auto iter = kIsaVersionStrMap.find(aicVersion);
    return (iter != kIsaVersionStrMap.end()) ? iter->second : "";
}

struct NodeDescDtype {
    uint32_t index;
    bool isInput;
    ge::DataType dtype;
    ge::DataType originDtype;
};

void ReplaceNodeDtypes(const ge::GNodePtr& node, const ge::DataType& dtype, std::vector<NodeDescDtype>& records)
{
    records.clear();
    if (dtype == ge::DT_UNDEFINED) {
        return;
    }
    for (uint32_t i = 0U; i < node->GetInputsSize(); ++i) {
        ge::TensorDesc tensorDesc;
        if (node->GetInputDesc(static_cast<int32_t>(i), tensorDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        records.push_back({i, true, tensorDesc.GetDataType(), GetOriginDtype(tensorDesc)});
        tensorDesc.SetDataType(dtype);
        SetOriginDtype(tensorDesc, dtype);
        (void)node->UpdateInputDesc(static_cast<int32_t>(i), tensorDesc);
    }
    for (uint32_t i = 0U; i < node->GetOutputsSize(); ++i) {
        ge::TensorDesc tensorDesc;
        if (node->GetOutputDesc(static_cast<int32_t>(i), tensorDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        records.push_back({i, false, tensorDesc.GetDataType(), GetOriginDtype(tensorDesc)});
        tensorDesc.SetDataType(dtype);
        SetOriginDtype(tensorDesc, dtype);
        (void)node->UpdateOutputDesc(static_cast<int32_t>(i), tensorDesc);
    }
}

void RecoverNodeDtypes(const ge::GNodePtr& node, const std::vector<NodeDescDtype>& records)
{
    for (const auto& record : records) {
        const auto index = static_cast<int32_t>(record.index);
        ge::TensorDesc tensorDesc;
        const bool isInput = record.isInput;
        if (isInput) {
            if (node->GetInputDesc(index, tensorDesc) != ge::GRAPH_SUCCESS) {
                continue;
            }
        } else if (node->GetOutputDesc(index, tensorDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        tensorDesc.SetDataType(record.dtype);
        SetOriginDtype(tensorDesc, record.originDtype);
        const ge::graphStatus updateRet = isInput ? node->UpdateInputDesc(index, tensorDesc) :
                                                    node->UpdateOutputDesc(index, tensorDesc);
        if (updateRet != ge::GRAPH_SUCCESS) {
            OPS_LOG_W(kLogTag, "Recover dtype of %s index %d failed after dtype probe.", isInput ? "input" : "output",
                      index);
        }
    }
}

bool IsConstNode(const ge::GNodePtr& node)
{
    const std::string type = NodeType(*node);
    return type == kConstOpType || type == kConstantOpType;
}

ge::GNodePtr FindParentNodeOfSubGraphData(const ge::Graph& graph, const std::string& dataNodeName)
{
    for (const auto& node : graph.GetAllNodes()) {
        if (NodeName(node) == dataNodeName) {
            return graph.GetParentNode();
        }
    }
    for (const auto& subGraph : graph.GetAllSubgraphs()) {
        if (subGraph == nullptr) {
            continue;
        }
        const ge::GNodePtr parentNode = FindParentNodeOfSubGraphData(*subGraph, dataNodeName);
        if (parentNode != nullptr) {
            return parentNode;
        }
    }
    return nullptr;
}

ge::GNodePtr TraceNodeToConst(const ge::Graph& rootGraph, const ge::GNodePtr& node, const uint32_t depth)
{
    if (node == nullptr || depth > kMaxSubGraphTraceDepth) {
        return nullptr;
    }
    if (IsConstNode(node)) {
        return node;
    }
    if (NodeType(*node) == kPlaceHolderOpType) {
        const auto phInput = node->GetInDataNodesAndPortIndexs(0);
        if (phInput.first == nullptr) {
            return nullptr;
        }
        return TraceNodeToConst(rootGraph, phInput.first, depth + 1U);
    }
    if (NodeType(*node) != kDataOpType) {
        return nullptr;
    }
    int64_t parentIndex = -1;
    if (node->GetAttr(ge::AscendString(kAttrParentNodeIndex), parentIndex) != ge::GRAPH_SUCCESS) {
        if (node->GetAttr(ge::AscendString(kAttrIndex), parentIndex) != ge::GRAPH_SUCCESS) {
            return nullptr;
        }
    }
    const ge::GNodePtr parentNode = FindParentNodeOfSubGraphData(rootGraph, NodeName(*node));
    if (parentNode == nullptr || parentIndex < 0) {
        return nullptr;
    }
    const auto producer = parentNode->GetInDataNodesAndPortIndexs(static_cast<int32_t>(parentIndex));
    if (producer.first == nullptr) {
        return nullptr;
    }
    return TraceNodeToConst(rootGraph, producer.first, depth + 1U);
}

ge::GNodePtr IsPeerOutConst(const ge::GNodePtr& enterNode, const ge::Graph& rootGraph)
{
    const auto peer = enterNode->GetInDataNodesAndPortIndexs(0);
    if (peer.first == nullptr) {
        return nullptr;
    }
    if (IsConstNode(peer.first)) {
        return peer.first;
    }
    if (NodeType(*peer.first) == kDataOpType || NodeType(*peer.first) == kPlaceHolderOpType) {
        return TraceNodeToConst(rootGraph, peer.first, 0U);
    }
    return nullptr;
}

bool HasInControlEdgeFrom(const ge::GNode& node, const std::string& srcName)
{
    for (const auto& inCtrlNode : node.GetInControlNodes()) {
        if (inCtrlNode == nullptr) {
            continue;
        }
        if (NodeName(*inCtrlNode) == srcName) {
            return true;
        }
    }
    return false;
}

ge::GNodePtr CreateHostOpNode(ge::Graph& graph, const std::string& opType,
                              const std::vector<std::string>& tensorDescNameList)
{
    std::stringstream opName;
    opName << opType << "_" << GetHostCpuAtomicId();
    std::vector<ge::es::CompliantNodeBuilder::IrInputDef> inputDefs;
    for (const auto& name : tensorDescNameList) {
        inputDefs.push_back({name, ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""});
    }
    inputDefs.push_back({kIrInputScaleName, ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""});
    ge::GNode hostNode = ge::es::CompliantNodeBuilder(&graph)
                             .OpType(opType.c_str())
                             .Name(opName.str().c_str())
                             .IrDefInputs(inputDefs)
                             .IrDefOutputs(
                                 {{kCubeQuantRollBackOutput, ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
    if (NodeName(hostNode).empty()) {
        OPS_LOG_E(kLogTag, "Create host op node[%s] by CompliantNodeBuilder failed.", opName.str().c_str());
        return nullptr;
    }
    return std::make_shared<ge::GNode>(hostNode);
}
} // namespace

int32_t GetAxisIndexByFormat(const ge::Format& format, const std::string& axis)
{
    auto iter = kAxisIndexOfFormat.find(format);
    if (iter == kAxisIndexOfFormat.end()) {
        OPS_LOG_W(kLogTag, "Format %s is not supported.", ge::TypeUtils::FormatToSerialString(format).c_str());
        return -1;
    }
    auto axisIter = iter->second.find(axis);
    if (axisIter == iter->second.end()) {
        OPS_LOG_W(kLogTag, "Axis %s is not supported.", axis.c_str());
        return -1;
    }
    return axisIter->second;
}

ge::Status BiasOptimizeQuantRollbackBase::OptimizeMainProc(ge::GraphPtr& graph, const ge::GNodePtr& quantNode,
                                                           const ge::GNodePtr& dequantNode,
                                                           const ge::GNodePtr& cubeNode,
                                                           std::vector<ge::GNodePtr>& fusionNodes,
                                                           std::vector<ge::GNodePtr>& nodesToRemove)
{
    nodesToRemove_.clear();
    if (graph == nullptr) {
        OPS_LOG_E(kLogTag, "Graph is null, fusion failed.");
        return ge::FAILED;
    }
    if (quantNode == nullptr) {
        OPS_LOG_E(kLogTag, "QuantNode is null, fusion failed.");
        return ge::FAILED;
    }
    if (dequantNode == nullptr) {
        OPS_LOG_E(kLogTag, "DequantNode is null, fusion failed.");
        return ge::FAILED;
    }
    if (cubeNode == nullptr) {
        OPS_LOG_E(kLogTag, "CubeNode is null, fusion failed.");
        return ge::FAILED;
    }
    if (SetGroupInfoForAscendWeightQuantNodes(cubeNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Set group info for AscendWeightQuant node failed.");
        return ge::FAILED;
    }
    QuantProcessMode quantProcessMode = QuantProcessMode::QUANT_UNDEFINED;
    if (GetQuantProcessMode(quantNode, cubeNode, quantProcessMode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Get quant process failed.");
        return ge::FAILED;
    }
    if (quantProcessMode == QuantProcessMode::QUANT_UNDEFINED) {
        OPS_LOG_I(kLogTag, "BiasOptimizeQuantRollback pass skips node[%s] for unknown shape.",
                  NodeName(*cubeNode).c_str());
        nodesToRemove = nodesToRemove_;
        return ge::SUCCESS;
    }
    if (quantProcessMode == QuantProcessMode::BIAS_OPTIMIZE) {
        if (BiasOptimize(*graph, cubeNode, dequantNode, quantNode, fusionNodes) != ge::SUCCESS) {
            OPS_LOG_E(kLogTag, "Node[%s] bias optimize failed.", NodeName(*cubeNode).c_str());
            return ge::FAILED;
        }
        OPS_LOG_D(kLogTag, "Node[%s] bias optimize successfully.", NodeName(*cubeNode).c_str());
        nodesToRemove = nodesToRemove_;
        return ge::SUCCESS;
    }
    if (QuantRollback(*graph, cubeNode, dequantNode, quantNode, fusionNodes) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Node[%s] quant rollback failed.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    if (std::find(fusionNodes.begin(), fusionNodes.end(), cubeNode) == fusionNodes.end()) {
        fusionNodes.push_back(cubeNode);
    }
    OPS_LOG_D(kLogTag, "Node[%s] quant rollback successfully.", NodeName(*cubeNode).c_str());
    nodesToRemove = nodesToRemove_;
    return ge::SUCCESS;
}

bool BiasOptimizeQuantRollbackBase::CheckOpSupported(const ge::GNode& node) const { return true; }

ge::Status BiasOptimizeQuantRollbackBase::GetCoValue(const ge::GNodePtr& cubeNode, int64_t& co)
{
    ge::TensorDesc weightDesc;
    if (cubeNode->GetInputDesc(static_cast<int32_t>(GetWeightInputIndex()), weightDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Node[%s] get weight input desc failed.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    const ge::Format filterFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(static_cast<int32_t>(weightDesc.GetFormat())));
    const std::vector<int64_t> filterDims = weightDesc.GetShape().GetDims();
    if (filterFormat == ge::FORMAT_ND) {
        return ge::SUCCESS;
    }
    int64_t groups = kGroupsDefaultValue;
    (void)cubeNode->GetAttr(ge::AscendString(kAttrNameGroups), groups);
    std::vector<int64_t> filterDims4D;
    if (PadShapeTo4Dim(filterFormat, filterDims, filterDims4D) != ge::SUCCESS || filterDims4D.empty()) {
        OPS_LOG_E(kLogTag, "Node[%s] filter_dims4_d is empty.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }

    const int32_t indexCo = GetAxisIndexByFormat(filterFormat, "C");
    if (indexCo < 0) {
        OPS_LOG_E(kLogTag, "Node[%s] index_co is negative, check filter format.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    if (static_cast<int64_t>(indexCo) >= static_cast<int64_t>(filterDims4D.size())) {
        OPS_LOG_E(kLogTag, "Node[%s] index_co is not less than the size of filter dims.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    if (!CheckInt64MulOverflow(filterDims4D[static_cast<size_t>(indexCo)], groups)) {
        OPS_LOG_E(kLogTag, "Int64 %ld and %ld multiplication can result in overflow!",
                  filterDims4D[static_cast<size_t>(indexCo)], groups);
        return ge::FAILED;
    }
    co = filterDims4D[static_cast<size_t>(indexCo)] * groups;
    return ge::SUCCESS;
}

void BiasOptimizeQuantRollbackBase::SetCinCoutReverse(const ge::GNodePtr& node)
{
    bool cinCoutReverse = false;
    if (node->SetAttr(ge::AscendString(kAttrCinCoutReverse), cinCoutReverse) != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(kLogTag, "Node[%s] set attr cin_cout_reverse failed.", NodeName(*node).c_str());
    }
}

ge::Status BiasOptimizeQuantRollbackBase::SetBiasName(const std::string& biasName)
{
    bias_name_ = biasName;
    return ge::SUCCESS;
}

bool BiasOptimizeQuantRollbackBase::IsCubeNodeSupported(const ge::GNodePtr& cubeNode) const
{
    const bool isSupported = CheckOpSupported(*cubeNode);
    OPS_LOG_D(kLogTag, "Check support result of node[%s, %s] is %d.", NodeName(*cubeNode).c_str(),
              NodeType(*cubeNode).c_str(), static_cast<int32_t>(isSupported));
    return isSupported;
}

bool BiasOptimizeQuantRollbackBase::IsCubeSupportDtype(const ge::GNodePtr& cubeNode, const ge::DataType& dtype) const
{
    if (cubeNode == nullptr) {
        OPS_LOG_I(kLogTag, "Cube node is null.");
        return false;
    }
    std::vector<NodeDescDtype> records;
    ReplaceNodeDtypes(cubeNode, dtype, records);
    const bool isSupported = CheckOpSupported(*cubeNode);
    OPS_LOG_D(kLogTag, "Data type is %s by cube node[%s, %s].", isSupported ? "supported" : "not supported",
              NodeName(*cubeNode).c_str(), NodeType(*cubeNode).c_str());
    RecoverNodeDtypes(cubeNode, records);
    return isSupported;
}

bool BiasOptimizeQuantRollbackBase::IsWeightTranspose(const ge::GNodePtr& cubeNode)
{
    bool transposeX2 = false;
    bool adjX2 = false;
    bool transpose = false;
    (void)cubeNode->GetAttr(ge::AscendString(kAttrTransposeX2), transposeX2);
    (void)cubeNode->GetAttr(ge::AscendString(kAttrAdjX2), adjX2);
    (void)cubeNode->GetAttr(ge::AscendString(kAttrTranspose), transpose);
    return transposeX2 || adjX2 || transpose;
}

void BiasOptimizeQuantRollbackBase::SetNodeOutputDtype(const ge::GNodePtr& node, const ge::DataType& checkDtype,
                                                       const ge::DataType& targetDtype)
{
    if (node == nullptr) {
        return;
    }
    for (uint32_t i = 0U; i < node->GetOutputsSize(); ++i) {
        ge::TensorDesc outTensor;
        if (node->GetOutputDesc(static_cast<int32_t>(i), outTensor) != ge::GRAPH_SUCCESS) {
            continue;
        }
        if (outTensor.GetDataType() == checkDtype) {
            outTensor.SetDataType(targetDtype);
        }
        if (GetOriginDtype(outTensor) == checkDtype) {
            SetOriginDtype(outTensor, targetDtype);
        }
        (void)node->UpdateOutputDesc(static_cast<int32_t>(i), outTensor);
    }
}

ge::Status BiasOptimizeQuantRollbackBase::GetQuantProcessMode(const ge::GNodePtr& quantNode,
                                                              const ge::GNodePtr& cubeNode,
                                                              QuantProcessMode& quantProcessMode)
{
    if (!IsCubeNodeSupported(cubeNode)) {
        quantProcessMode = QuantProcessMode::QUANT_ROLLBACK;
        return ge::SUCCESS;
    }
    ge::TensorDesc inputDesc;
    if (quantNode->GetInputDesc(0, inputDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Node[%s] get input desc failed.", NodeName(*quantNode).c_str());
        return ge::FAILED;
    }
    const ge::Format inputFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(static_cast<int32_t>(inputDesc.GetFormat())));
    const int32_t indexCi = GetAxisIndexByFormat(inputFormat, "C");
    if (indexCi < 0) {
        OPS_LOG_E(kLogTag, "Can not get C index of format %s.",
                  ge::TypeUtils::FormatToSerialString(inputFormat).c_str());
        return ge::FAILED;
    }
    const ge::Shape inputShape = inputDesc.GetShape();
    if (static_cast<int64_t>(indexCi) >= static_cast<int64_t>(inputShape.GetDimNum())) {
        OPS_LOG_W(kLogTag, "C index %d is not less than dim num %zu of quant input.", indexCi, inputShape.GetDimNum());
    }
    const int64_t cDim = inputShape.GetDim(static_cast<size_t>(indexCi));
    if (cDim == ge::UNKNOWN_DIM || cDim == ge::UNKNOWN_DIM_NUM) {
        OPS_LOG_W(kLogTag, "Bias optimize cannot be applied when the input shape is unknown.");
        quantProcessMode = QuantProcessMode::QUANT_UNDEFINED;
        return ge::SUCCESS;
    }
    const int64_t fp16C0Value = GetC0OfFloat16();
    const bool isCubeSupportFloat16 = IsCubeSupportDtype(cubeNode, ge::DT_FLOAT16);
    if (cDim > fp16C0Value || !isCubeSupportFloat16) {
        quantProcessMode = QuantProcessMode::BIAS_OPTIMIZE;
    } else {
        quantProcessMode = QuantProcessMode::QUANT_ROLLBACK;
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::JudgeDeqscaleShape(const ge::GNodePtr& dequantNode) const
{
    OPS_LOG_D(kLogTag, "Current ISA version is [%s].", GetIsaArchVersionStr().c_str());
    ge::Tensor deqScaleTensor;
    if (dequantNode->GetInputConstData(static_cast<int32_t>(DEQUANT_SCALE_INPUT_INDEX), deqScaleTensor) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Node[%s] deq_scale is not const or get const data failed.", NodeName(*dequantNode).c_str());
        return ge::FAILED;
    }
    if (deqScaleTensor.GetData() == nullptr) {
        OPS_LOG_E(kLogTag, "Node[%s] deqScale data is nullptr.", NodeName(*dequantNode).c_str());
        return ge::FAILED;
    }
    const ge::TensorDesc& deqScaleDesc = deqScaleTensor.GetTensorDesc();
    if (deqScaleDesc.GetShape().GetDimNum() != 1U) {
        OPS_LOG_E(kLogTag, "Node[%s] deq_scale shape error, dim num is %zu.", NodeName(*dequantNode).c_str(),
                  deqScaleDesc.GetShape().GetDimNum());
        return ge::FAILED;
    }
    OPS_LOG_D(kLogTag, "Node[%s] bias optimize, deq_scale_tensor dim is %ld.", NodeName(*dequantNode).c_str(),
              deqScaleDesc.GetShape().GetDim(0));
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::RemoveEnter(ge::Graph& graph, const ge::GNodePtr& node) const
{
    for (uint32_t i = 0U; i < node->GetInputsSize(); ++i) {
        const auto peer = node->GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (peer.first == nullptr) {
            continue;
        }
        const ge::GNodePtr preNode = peer.first;
        if (NodeType(*preNode) != kEnterOpType) {
            continue;
        }
        const size_t outDataNum = preNode->GetOutDataNodesAndPortIndexs(0).size() +
                                  preNode->GetOutControlNodes().size();
        const bool hasSingleInput = (preNode->GetInDataNodesAndPortIndexs(0).first != nullptr) &&
                                    preNode->GetInControlNodes().empty();
        if (outDataNum != 1U || !hasSingleInput) {
            continue;
        }
        const ge::GNodePtr srcNode = IsPeerOutConst(preNode, graph);
        if (srcNode == nullptr) {
            return ge::SUCCESS;
        }
        const auto enterInput = preNode->GetInDataNodesAndPortIndexs(0);
        ge::GNodePtr connectSrc = enterInput.first;
        int32_t connectPort = enterInput.second;
        ge::GNodePtr placeHolderToRemove = nullptr;
        if (connectSrc != nullptr && NodeType(*connectSrc) == kPlaceHolderOpType) {
            const auto phInput = connectSrc->GetInDataNodesAndPortIndexs(0);
            if (phInput.first != nullptr) {
                placeHolderToRemove = connectSrc;
                connectSrc = phInput.first;
                connectPort = phInput.second;
            }
        }
        OPS_LOG_D(kLogTag, "Start removing enter node[%s].", NodeName(*preNode).c_str());
        if (graph.RemoveEdge(*enterInput.first, enterInput.second, *preNode, 0) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Fail to remove edge between node[%s] and node[%s].",
                      NodeName(*enterInput.first).c_str(), NodeName(*preNode).c_str());
            return ge::FAILED;
        }
        if (graph.RemoveEdge(*preNode, peer.second, *node, static_cast<int32_t>(i)) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Fail to remove edge between node[%s] and node[%s].", NodeName(*preNode).c_str(),
                      NodeName(*node).c_str());
            return ge::FAILED;
        }
        if (graph.AddDataEdge(*connectSrc, connectPort, *node, static_cast<int32_t>(i)) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Fail to add edge for the input data anchor of node[%s].", NodeName(*node).c_str());
            return ge::FAILED;
        }
        nodesToRemove_.push_back(preNode);
        if (placeHolderToRemove != nullptr) {
            nodesToRemove_.push_back(placeHolderToRemove);
        }
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::SetQuantParameters(const ge::GNodePtr& cubeNode,
                                                             const ge::GNodePtr& quantNode) const
{
    ge::float32_t offsetA = 0.0F;
    (void)quantNode->GetAttr(ge::AscendString(kAttrOffset), offsetA);
    if (!std::isfinite(offsetA) || offsetA > static_cast<ge::float32_t>(INT32_MAX) ||
        offsetA < static_cast<ge::float32_t>(INT32_MIN)) {
        OPS_LOG_E(kLogTag, "Node[%s] quant offset[%f] is invalid.", NodeName(*quantNode).c_str(), offsetA);
        return ge::FAILED;
    }
    offsetA = static_cast<ge::float32_t>(static_cast<int32_t>(offsetA));
    int64_t offsetTemp = static_cast<int64_t>(offsetA);
    if (cubeNode->SetAttr(ge::AscendString(kAttrOffsetX), offsetTemp) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Set conv offset failed for node[%s].", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

ge::Tensor BiasOptimizeQuantRollbackBase::CreateBiasTensor(const ge::Shape& biasShape) const
{
    const int64_t size = biasShape.GetShapeSize();
    if (size <= 0 || size > kMaxBiasElementNum) {
        OPS_LOG_W(kLogTag, "Bias size %ld is invalid.", size);
        return ge::Tensor();
    }
    std::vector<uint8_t> biasData(static_cast<size_t>(size) * sizeof(int32_t), 0U);
    const ge::TensorDesc biasDesc(biasShape, ge::FORMAT_ND, ge::DT_INT32);
    return ge::Tensor(biasDesc, biasData);
}

ge::GNodePtr BiasOptimizeQuantRollbackBase::CreateBiasNode(ge::Graph& graph, const ge::Tensor& biasTensor,
                                                           const ge::GNodePtr& cubeNode) const
{
    const std::string constOpName = NodeName(*cubeNode) + "_quant_bias";
    ge::AttrValue attrValue;
    if (attrValue.SetAttrValue(biasTensor) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Fail to create bias attr value for node[%s].", NodeName(*cubeNode).c_str());
        return nullptr;
    }
    ge::GNode constNode = ge::es::CompliantNodeBuilder(&graph)
                              .OpType(kConstOpType)
                              .Name(constOpName.c_str())
                              .IrDefOutputs({{kIrOutputName, ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .IrDefAttrs({{"value", ge::es::CompliantNodeBuilder::kEsAttrOptional,
                                            kIrAttrValueTypeTensor, attrValue}})
                              .InstanceOutputShape(kIrOutputName, biasTensor.GetTensorDesc().GetShape().GetDims())
                              .InstanceOutputDataType(kIrOutputName, ge::DT_INT32)
                              .InstanceOutputFormat(kIrOutputName, ge::FORMAT_ND)
                              .Build();
    if (NodeName(constNode).empty()) {
        OPS_LOG_E(kLogTag, "Fail to create const op desc for node[%s].", NodeName(*cubeNode).c_str());
        return nullptr;
    }
    return std::make_shared<ge::GNode>(constNode);
}

ge::Status BiasOptimizeQuantRollbackBase::UpdateBiasOutputDesc(const ge::GNodePtr& cubeNode, const ge::Shape& shape,
                                                               const ge::Format& format, const uint32_t index) const
{
    const auto biasPeer = cubeNode->GetInDataNodesAndPortIndexs(static_cast<int32_t>(index));
    if (biasPeer.first == nullptr) {
        OPS_LOG_E(kLogTag, "Node[%s] bias input peer is null.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    ge::TensorDesc biasOutputDesc;
    if (biasPeer.first->GetOutputDesc(0, biasOutputDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Get output desc of bias node[%s] failed.", NodeName(*biasPeer.first).c_str());
        return ge::FAILED;
    }
    biasOutputDesc.SetShape(shape);
    biasOutputDesc.SetOriginFormat(format);
    biasOutputDesc.SetOriginShape(shape);
    SetOriginDtype(biasOutputDesc, ge::DT_INT32);
    biasOutputDesc.SetDataType(ge::DT_INT32);
    if (biasPeer.first->UpdateOutputDesc(0, biasOutputDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Update output desc of bias node[%s] failed.", NodeName(*biasPeer.first).c_str());
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::UpdateCubeInputDesc(const ge::GNodePtr& cubeNode, const ge::Shape& shape,
                                                              const ge::Format& format, const uint32_t index) const
{
    ge::TensorDesc biasDesc;
    if (cubeNode->GetInputDesc(static_cast<int32_t>(index), biasDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(kLogTag, "Cube node[%s] input %u desc is unset, use default desc.", NodeName(*cubeNode).c_str(),
                  index);
    }
    biasDesc.SetShape(shape);
    biasDesc.SetOriginFormat(format);
    biasDesc.SetOriginShape(shape);
    SetOriginDtype(biasDesc, ge::DT_INT32);
    biasDesc.SetDataType(ge::DT_INT32);
    if (cubeNode->UpdateInputDesc(static_cast<int32_t>(index), biasDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Update bias input desc of cube node[%s] failed.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::CreateBiasInput(ge::Graph& graph, const ge::GNodePtr& cubeNode,
                                                          const int64_t& co)
{
    ge::Shape biasShape({co});
    return CreateBiasInput(graph, cubeNode, biasShape);
}

ge::Status BiasOptimizeQuantRollbackBase::CreateBiasInput(ge::Graph& graph, const ge::GNodePtr& cubeNode,
                                                          const ge::Shape& biasShape)
{
    OPS_LOG_D(kLogTag, "Cube node[%s] has no bias, create bias and set data.", NodeName(*cubeNode).c_str());
    if (RemoveEnter(graph, cubeNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Remove enter node failed for node[%s].", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    const ge::Tensor biasTensor = CreateBiasTensor(biasShape);
    if (biasTensor.GetData() == nullptr) {
        return ge::FAILED;
    }
    const ge::GNodePtr constNode = CreateBiasNode(graph, biasTensor, cubeNode);
    if (constNode == nullptr) {
        OPS_LOG_E(kLogTag, "Fail to add const node for cube node[%s].", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    if (graph.AddDataEdge(*constNode, 0, *cubeNode, static_cast<int32_t>(GetBiasInputIndex())) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Fail to link const node with cube node[%s].", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    ge::TensorDesc inputDesc0;
    if (cubeNode->GetInputDesc(0, inputDesc0) != ge::GRAPH_SUCCESS) {
        return ge::FAILED;
    }
    const ge::Format inputDesc0OriginFormat = inputDesc0.GetOriginFormat();
    const auto ub = UpdateBiasOutputDesc(cubeNode, biasShape, inputDesc0OriginFormat, GetBiasInputIndex());
    if (ub != ge::SUCCESS) {
        return ge::FAILED;
    }
    const auto uc = UpdateCubeInputDesc(cubeNode, biasShape, inputDesc0OriginFormat, GetBiasInputIndex());
    if (uc != ge::SUCCESS) {
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

bool BiasOptimizeQuantRollbackBase::IsCubeHasBiasInput(const ge::GNodePtr& cubeNode) const
{
    if (cubeNode == nullptr) {
        OPS_LOG_W(kLogTag, "Cube node is null.");
        return false;
    }
    if (cubeNode->GetInputsSize() < static_cast<size_t>(GetInputSizeContainsBias())) {
        return false;
    }
    const auto biasPeer = cubeNode->GetInDataNodesAndPortIndexs(static_cast<int32_t>(GetBiasInputIndex()));
    return biasPeer.first != nullptr;
}

bool BiasOptimizeQuantRollbackBase::IsCubeNeedBiasInput(const ge::GNodePtr& cubeNode)
{
    return !IsCubeHasBiasInput(cubeNode);
}

ge::Status BiasOptimizeQuantRollbackBase::ProcessBiasInput(ge::Graph& graph, const ge::GNodePtr& cubeNode)
{
    if (IsCubeNeedBiasInput(cubeNode)) {
        int64_t co = 1;
        if (GetCoValue(cubeNode, co) != ge::SUCCESS) {
            OPS_LOG_E(kLogTag, "Get node[%s] co value failed.", NodeName(*cubeNode).c_str());
            return ge::FAILED;
        }
        if (CreateBiasInput(graph, cubeNode, co) != ge::SUCCESS) {
            OPS_LOG_E(kLogTag, "Create cube node[%s] bias input failed.", NodeName(*cubeNode).c_str());
            return ge::FAILED;
        }
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::BiasOptimize(ge::Graph& graph, const ge::GNodePtr& cubeNode,
                                                       const ge::GNodePtr& dequantNode, const ge::GNodePtr& quantNode,
                                                       std::vector<ge::GNodePtr>& fusionNodes)
{
    if (RemoveEnter(graph, dequantNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Remove enter node failed.");
        return ge::FAILED;
    }
    if (JudgeDeqscaleShape(dequantNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Judge node[%s] deq_scale failed.", NodeName(*dequantNode).c_str());
        return ge::FAILED;
    }
    if (SetQuantParameters(cubeNode, quantNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Set quant paras to cube node[%s] failed.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    if (ProcessBiasInput(graph, cubeNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Process node[%s] bias failed.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    PatternNodes patternNodes = {cubeNode, dequantNode, quantNode, nullptr, nullptr};
    if (GetWeightNodesOfCubeNode(cubeNode, patternNodes.weightConstNode, patternNodes.ascendWeightQuantNode) !=
        ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Get weight_const node and ascend_weight_quant node of cube node[%s] failed.",
                  NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    if (CreateNewHostCpuOp(kQuantBiasOptimization, patternNodes, graph, GetBiasOptimizationModeIndex(), fusionNodes) !=
        ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Create host op failed.");
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::RemoveInputEdgeAndSingleConstInput(ge::Graph& graph, const ge::GNodePtr& node,
                                                                             PeerOutAnchors& peerOutAnchorsOfNode) const
{
    peerOutAnchorsOfNode.clear();
    for (uint32_t i = 0U; i < node->GetInputsSize(); ++i) {
        const auto peer = node->GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (peer.first == nullptr) {
            continue;
        }
        peerOutAnchorsOfNode.emplace_back(peer);
        if (graph.RemoveEdge(*peer.first, peer.second, *node, static_cast<int32_t>(i)) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Remove input data edge of node[%s] failed.", NodeName(*node).c_str());
            return ge::FAILED;
        }
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::LinkOutputEdgeWithoutControl(
    ge::Graph& graph, const ge::GNodePtr& oldNode, const ge::GNodePtr& newNode, const std::string& cubeName,
    const PeerOutAnchors& peerOutAnchorsOfOldNode) const
{
    ge::GNodePtr srcNode = nullptr;
    int32_t srcPort = 0;
    if (peerOutAnchorsOfOldNode.empty()) {
        const auto inPeer = oldNode->GetInDataNodesAndPortIndexs(0);
        srcNode = inPeer.first;
        srcPort = inPeer.second;
    } else {
        srcNode = peerOutAnchorsOfOldNode[0].first;
        srcPort = peerOutAnchorsOfOldNode[0].second;
    }
    if (srcNode == nullptr) {
        OPS_LOG_E(kLogTag, "New out data anchor is null for node[%s] and node[%s].", NodeName(*oldNode).c_str(),
                  NodeName(*newNode).c_str());
        return ge::FAILED;
    }
    const bool isQuantType = NodeType(*oldNode) == ASCEND_QUANT;
    for (uint32_t outIdx = 0U; outIdx < oldNode->GetOutputsSize(); ++outIdx) {
        const auto consumers = oldNode->GetOutDataNodesAndPortIndexs(static_cast<int32_t>(outIdx));
        for (const auto& consumer : consumers) {
            if (consumer.first == nullptr) {
                continue;
            }
            if (isQuantType && NodeName(*consumer.first) != cubeName) {
                continue;
            }
            if (graph.RemoveEdge(*oldNode, static_cast<int32_t>(outIdx), *consumer.first, consumer.second) !=
                    ge::GRAPH_SUCCESS ||
                graph.AddDataEdge(*srcNode, srcPort, *consumer.first, consumer.second) != ge::GRAPH_SUCCESS) {
                OPS_LOG_E(kLogTag, "Replace out data anchor of node[%s] failed.", NodeName(*oldNode).c_str());
                return ge::FAILED;
            }
        }
    }
    const std::string oldNodeName = NodeName(*oldNode);
    for (auto& candidate : graph.GetAllNodes()) {
        if (isQuantType && NodeName(candidate) != cubeName) {
            continue;
        }
        if (!HasInControlEdgeFrom(candidate, oldNodeName)) {
            continue;
        }
        bool removed = false;
        for (uint32_t port = 0U; port < oldNode->GetOutputsSize(); ++port) {
            if (graph.RemoveEdge(*oldNode, static_cast<int32_t>(port), candidate, -1) == ge::GRAPH_SUCCESS) {
                removed = true;
                break;
            }
        }
        if (removed && graph.AddControlEdge(*srcNode, candidate) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Replace input control anchor of node[%s] failed.", NodeName(candidate).c_str());
            return ge::FAILED;
        }
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::SetHostOpNodeDescs(const ge::GNodePtr& hostNode,
                                                             const PatternNodes& patternNode,
                                                             const std::vector<uint32_t>& anchorIndexlist,
                                                             const std::vector<std::string>& tensorDescNameList,
                                                             ge::TensorDesc& outputDesc) const
{
    const size_t inputsSize = patternNode.cubeNode->GetInputsSize();
    for (size_t i = 0U; i < anchorIndexlist.size(); ++i) {
        const uint32_t weightAnchorIndex = anchorIndexlist[i];
        if (weightAnchorIndex >= inputsSize) {
            OPS_LOG_E(kLogTag, "Weight index %u is not less than cube node[%s] input size %zu.", weightAnchorIndex,
                      NodeName(*patternNode.cubeNode).c_str(), inputsSize);
            return ge::FAILED;
        }
        if (i >= tensorDescNameList.size()) {
            OPS_LOG_E(kLogTag, "Tensor index %zu is not less than tensor name size %zu.", i, tensorDescNameList.size());
            return ge::FAILED;
        }
        ge::TensorDesc filterOrBiasDesc;
        if (weightAnchorIndex == GetWeightInputIndex()) {
            if (patternNode.weightConstNode->GetOutputDesc(0, filterOrBiasDesc) != ge::GRAPH_SUCCESS) {
                OPS_LOG_E(kLogTag, "Get output desc of weight const node[%s] failed.",
                          NodeName(*patternNode.weightConstNode).c_str());
                return ge::FAILED;
            }
        } else if (patternNode.cubeNode->GetInputDesc(static_cast<int32_t>(weightAnchorIndex), filterOrBiasDesc) !=
                   ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Get input desc %u of cube node[%s] failed.", weightAnchorIndex,
                      NodeName(*patternNode.cubeNode).c_str());
            return ge::FAILED;
        }
        OPS_LOG_I(kLogTag, "Tensor index[%zu]'s name is [%s].", i, tensorDescNameList[i].c_str());
        if (hostNode->UpdateInputDesc(static_cast<int32_t>(i), filterOrBiasDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Update input desc %zu of host node[%s] failed.", i, NodeName(*hostNode).c_str());
            return ge::FAILED;
        }
        if (i == 0U) {
            outputDesc = filterOrBiasDesc;
        }
    }
    ge::TensorDesc scaleDesc;
    if (patternNode.dequantNode->GetInputDesc(static_cast<int32_t>(DEQUANT_SCALE_INPUT_INDEX), scaleDesc) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Get deq_scale desc of dequant node[%s] failed.",
                  NodeName(*patternNode.dequantNode).c_str());
        return ge::FAILED;
    }
    if (hostNode->UpdateInputDesc(static_cast<int32_t>(anchorIndexlist.size()), scaleDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Update scale input desc of host node[%s] failed.", NodeName(*hostNode).c_str());
        return ge::FAILED;
    }
    for (uint32_t i = 0U; i < hostNode->GetInputsSize(); ++i) {
        ge::TensorDesc tensorDesc;
        if (hostNode->GetInputDesc(static_cast<int32_t>(i), tensorDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        tensorDesc.SetOriginFormat(
            static_cast<ge::Format>(ge::GetPrimaryFormat(static_cast<int32_t>(tensorDesc.GetFormat()))));
        tensorDesc.SetOriginShape(tensorDesc.GetShape());
        SetOriginDtype(tensorDesc, tensorDesc.GetDataType());
        (void)hostNode->UpdateInputDesc(static_cast<int32_t>(i), tensorDesc);
    }
    ge::TensorDesc hostOutputDesc = outputDesc;
    const ge::Format hostOriginFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(static_cast<int32_t>(outputDesc.GetFormat())));
    hostOutputDesc.SetOriginFormat(hostOriginFormat);
    hostOutputDesc.SetOriginShape(outputDesc.GetShape());
    SetOriginDtype(hostOutputDesc, outputDesc.GetDataType());
    if (hostNode->UpdateOutputDesc(0, hostOutputDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Update output desc of host node[%s] failed.", NodeName(*hostNode).c_str());
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::LinkHostOpEdges(ge::Graph& graph, const PatternNodes& patternNode,
                                                          const ge::GNodePtr& hostNode,
                                                          const std::vector<uint32_t>& anchorIndexlist) const
{
    for (size_t i = 0U; i < anchorIndexlist.size(); ++i) {
        if (LinkHostOpEdge(graph, patternNode, hostNode, anchorIndexlist[i], i) != ge::SUCCESS) {
            OPS_LOG_E(kLogTag, "Failed to add edge for host node[%s, %s], cube node is %s.",
                      NodeName(*hostNode).c_str(), NodeType(*hostNode).c_str(),
                      NodeName(*patternNode.cubeNode).c_str());
            return ge::FAILED;
        }
    }
    const auto scalePeer = patternNode.dequantNode->GetInDataNodesAndPortIndexs(
        static_cast<int32_t>(DEQUANT_SCALE_INPUT_INDEX));
    if (scalePeer.first == nullptr) {
        OPS_LOG_E(kLogTag, "Get deq_scale peer of dequant node[%s] failed.",
                  NodeName(*patternNode.dequantNode).c_str());
        return ge::FAILED;
    }
    const int32_t scaleIndex = static_cast<int32_t>(anchorIndexlist.size());
    OPS_LOG_I(kLogTag, "Node[%s]: scale index is %d.", NodeName(*hostNode).c_str(), scaleIndex);
    if (graph.AddDataEdge(*scalePeer.first, scalePeer.second, *hostNode, scaleIndex) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Fail to add edge between dequant node[%s] and host node[%s].",
                  NodeName(*patternNode.dequantNode).c_str(), NodeName(*hostNode).c_str());
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::SetHostOpNodeAttrs(const std::string& opType, const PatternNodes& patternNode,
                                                             const ge::GNodePtr& hostNode)
{
    SetCinCoutReverse(hostNode);
    int64_t groups = kGroupsDefaultValue;
    (void)patternNode.cubeNode->GetAttr(ge::AscendString(kAttrNameGroups), groups);
    (void)hostNode->SetAttr(ge::AscendString(kAttrNameGroups), groups);
    bool isComeFromConstOp = true;
    (void)hostNode->SetAttr(ge::AscendString(kIsComeFromConstOp), isComeFromConstOp);
    ge::AscendString cubeOpType(NodeType(*patternNode.cubeNode).c_str());
    (void)hostNode->SetAttr(ge::AscendString(kAttrCubeOpType), cubeOpType);
    const bool rollBackScenario = (opType == kQuantWeightRollBack || opType == kQuantBiasRollBack);
    if (rollBackScenario) {
        ge::float32_t scaleA = 0.0F;
        (void)patternNode.quantNode->GetAttr(ge::AscendString(kAttrScale), scaleA);
        (void)hostNode->SetAttr(ge::AscendString(kAttrScale), scaleA);
    }
    if (opType == kQuantBiasOptimization) {
        const std::string socVersion = GetIsaArchVersionStr();
        ge::AscendString socVersionAsc(socVersion.c_str());
        (void)hostNode->SetAttr(ge::AscendString(kStrSocVersion), socVersionAsc);
        ge::float32_t offsetA = 0.0F;
        (void)patternNode.quantNode->GetAttr(ge::AscendString(kAttrOffset), offsetA);
        (void)hostNode->SetAttr(ge::AscendString(kAttrOffset), offsetA);
    }
    if (SetInt4OrInt8Flag(patternNode.ascendWeightQuantNode, hostNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Set int4 or int8 flag for roll back node[%s] failed.", NodeName(*hostNode).c_str());
        return ge::FAILED;
    }
    if (IsWeightTranspose(patternNode.cubeNode)) {
        bool weightTranspose = true;
        (void)hostNode->SetAttr(ge::AscendString(kAttrWeightTranspose), weightTranspose);
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::CreateNewHostCpuOp(const std::string& opType, const PatternNodes& patternNode,
                                                             ge::Graph& graph, const uint32_t mode,
                                                             std::vector<ge::GNodePtr>& fusNodes)
{
    if (patternNode.cubeNode == nullptr || patternNode.dequantNode == nullptr || patternNode.quantNode == nullptr) {
        OPS_LOG_E(kLogTag, "Pattern node is null, create host op failed.");
        return ge::FAILED;
    }
    if (mode >= static_cast<uint32_t>(HostOpMode::MODE_BOTTOM) ||
        kTensorNameOfHostCpuOp.size() < static_cast<size_t>(HostOpMode::MODE_BOTTOM)) {
        OPS_LOG_E(kLogTag, "Mode %u is invalid for node[%s] and tensor size is %zu.", mode,
                  NodeName(*patternNode.cubeNode).c_str(), kTensorNameOfHostCpuOp.size());
        return ge::FAILED;
    }
    if (patternNode.weightConstNode == nullptr) {
        OPS_LOG_E(kLogTag, "Weight_const_node is null.");
        return ge::FAILED;
    }
    OPS_LOG_I(kLogTag, "Create host op[type:%s] for cube node %s and dequant node %s in mode %u.", opType.c_str(),
              NodeName(*patternNode.cubeNode).c_str(), NodeName(*patternNode.dequantNode).c_str(), mode);
    const std::vector<uint32_t> anchorIndexlist = kOriginalConvAnchorIndex[mode];
    const std::vector<std::string> tensorDescNameList = kTensorNameOfHostCpuOp[mode];
    if (anchorIndexlist.empty() || tensorDescNameList.empty()) {
        OPS_LOG_E(kLogTag, "Tensor name size %zu or anchor index size %zu is empty.", tensorDescNameList.size(),
                  anchorIndexlist.size());
        return ge::FAILED;
    }
    const ge::GNodePtr hostNode = CreateHostOpNode(graph, opType, tensorDescNameList);
    if (hostNode == nullptr) {
        return ge::FAILED;
    }
    ge::TensorDesc outputDesc;
    if (SetHostOpNodeDescs(hostNode, patternNode, anchorIndexlist, tensorDescNameList, outputDesc) != ge::SUCCESS) {
        return ge::FAILED;
    }
    fusNodes.push_back(hostNode);
    if (LinkHostOpEdges(graph, patternNode, hostNode, anchorIndexlist) != ge::SUCCESS) {
        return ge::FAILED;
    }
    if (SetHostOpNodeAttrs(opType, patternNode, hostNode) != ge::SUCCESS) {
        return ge::FAILED;
    }
    OPS_LOG_D(kLogTag, "Add node[%s] into the graph for cube node[%s] successfully.", NodeName(*hostNode).c_str(),
              NodeName(*patternNode.cubeNode).c_str());
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::LinkHostOpEdge(ge::Graph& graph, const PatternNodes& patternNode,
                                                         const ge::GNodePtr& hostNode, const uint32_t& anchorIndex,
                                                         const size_t& index) const
{
    const auto convWeightPeer = patternNode.cubeNode->GetInDataNodesAndPortIndexs(static_cast<int32_t>(anchorIndex));
    if (convWeightPeer.first == nullptr) {
        OPS_LOG_E(kLogTag, "Cube node[%s] input %u peer is null.", NodeName(*patternNode.cubeNode).c_str(),
                  anchorIndex);
        return ge::FAILED;
    }
    if (anchorIndex == GetWeightInputIndex()) {
        if (graph.AddDataEdge(*patternNode.weightConstNode, 0, *hostNode, static_cast<int32_t>(index)) !=
            ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag,
                      "Failed to add edge between weight const node and host op[%s], cube node[%s], "
                      "input index %zu.",
                      NodeName(*hostNode).c_str(), NodeName(*patternNode.cubeNode).c_str(), index);
            return ge::FAILED;
        }
    } else if (graph.AddDataEdge(*convWeightPeer.first, convWeightPeer.second, *hostNode,
                                 static_cast<int32_t>(index)) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Failed to add edge between cube input[%s] and host op[%s], input index %zu.",
                  NodeName(*patternNode.cubeNode).c_str(), NodeName(*hostNode).c_str(), index);
        return ge::FAILED;
    }
    if (index == 0U) {
        if (graph.RemoveEdge(*convWeightPeer.first, convWeightPeer.second, *patternNode.cubeNode,
                             static_cast<int32_t>(anchorIndex)) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Failed to remove edge between weight and cube node[%s], input index %zu.",
                      NodeName(*patternNode.cubeNode).c_str(), index);
            return ge::FAILED;
        }
        if (graph.AddDataEdge(*hostNode, 0, *patternNode.cubeNode, static_cast<int32_t>(anchorIndex)) !=
            ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Failed to add edge between host op[%s] and cube node[%s], input index %zu.",
                      NodeName(*hostNode).c_str(), NodeName(*patternNode.cubeNode).c_str(), index);
            return ge::FAILED;
        }
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::SetGroupInfoForAscendWeightQuantNodes(const ge::GNodePtr& cubeNode) const
{
    std::vector<ge::GNodePtr> allInputNodes;
    for (uint32_t i = 0U; i < cubeNode->GetInputsSize(); ++i) {
        const auto peer = cubeNode->GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (peer.first != nullptr) {
            allInputNodes.push_back(peer.first);
        }
    }
    if (allInputNodes.size() < kMinInputsWithWeight) {
        OPS_LOG_D(kLogTag, "Cube node[%s] does not have a weight node, no need to set group info.",
                  NodeName(*cubeNode).c_str());
        return ge::SUCCESS;
    }
    const ge::GNodePtr ascendWeightQuantNode = allInputNodes.at(1U);
    if (ascendWeightQuantNode == nullptr) {
        OPS_LOG_E(kLogTag, "Get AscendWeightQuant node failed.");
        return ge::FAILED;
    }
    int64_t groupsValue = kGroupsDefaultValue;
    (void)cubeNode->GetAttr(ge::AscendString(kAttrNameGroups), groupsValue);
    (void)ascendWeightQuantNode->SetAttr(ge::AscendString(kAttrNameGroups), groupsValue);
    OPS_LOG_D(kLogTag, "Set group info for ascend_weight_quant node[%s] successfully, group value is %ld.",
              NodeName(*ascendWeightQuantNode).c_str(), groupsValue);
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::SetCubeNodeDataType(const ge::GNodePtr& cubeNode,
                                                              const ge::DataType& dataType,
                                                              const ge::DataType& targetDataType) const
{
    for (uint32_t i = 0U; i < cubeNode->GetInputsSize(); ++i) {
        ge::TensorDesc inputDesc;
        if (cubeNode->GetInputDesc(static_cast<int32_t>(i), inputDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        inputDesc.SetDataType(targetDataType);
        SetOriginDtype(inputDesc, targetDataType);
        if (cubeNode->UpdateInputDesc(static_cast<int32_t>(i), inputDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Fail to update input desc of cube node[%s].", NodeName(*cubeNode).c_str());
            return ge::FAILED;
        }
        if (i == 0U) {
            continue;
        }
        const auto peer = cubeNode->GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (peer.first == nullptr) {
            continue;
        }
        ge::TensorDesc peerOutputDesc;
        if (peer.first->GetOutputDesc(peer.second, peerOutputDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Fail to get output desc of input node[%s].", NodeName(*peer.first).c_str());
            return ge::FAILED;
        }
        peerOutputDesc.SetDataType(dataType);
        SetOriginDtype(peerOutputDesc, dataType);
        if (peer.first->UpdateOutputDesc(peer.second, peerOutputDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Fail to update output desc of input node[%s].", NodeName(*peer.first).c_str());
            return ge::FAILED;
        }
    }
    ge::TensorDesc outputDesc;
    if (cubeNode->GetOutputDesc(0, outputDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Fail to get output desc of cube node[%s].", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    outputDesc.SetDataType(targetDataType);
    SetOriginDtype(outputDesc, targetDataType);
    if (cubeNode->UpdateOutputDesc(0, outputDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kLogTag, "Fail to update output desc of cube node[%s].", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::SetDataTypeOfNodes(const ge::GNodePtr& cubeNode)
{
    if (SetCubeNodeDataType(cubeNode, ge::DT_FLOAT, ge::DT_FLOAT16) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Set cube node[%s] data type failed.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    OPS_LOG_D(kLogTag, "Set cube node[%s] data type successfully.", NodeName(*cubeNode).c_str());
    return ge::SUCCESS;
}

ge::GNodePtr BiasOptimizeQuantRollbackBase::GetCubeNodeInputNode(const ge::GNodePtr& cubeNode) const
{
    return cubeNode;
}

ge::Status BiasOptimizeQuantRollbackBase::GetWeightNodesOfCubeNode(const ge::GNodePtr& cubeNode,
                                                                   ge::GNodePtr& weightConstNode,
                                                                   ge::GNodePtr& ascendWeightQuantNode) const
{
    const auto cubeWeightPeer = cubeNode->GetInDataNodesAndPortIndexs(static_cast<int32_t>(GetWeightInputIndex()));
    if (cubeWeightPeer.first == nullptr) {
        OPS_LOG_E(kLogTag, "Cube node[%s] weight input peer is null.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    const ge::GNodePtr cubeWeightInputNode = cubeWeightPeer.first;
    const std::string cubeWeightInputType = NodeType(*cubeWeightInputNode);
    const bool isRootGraphData = (cubeWeightInputType == kDataOpType) || (cubeWeightInputType == kRefDataOpType);
    ge::GNodePtr firstInputNode = nullptr;
    if (!isRootGraphData && cubeWeightInputNode->GetInputsSize() > 0U) {
        firstInputNode = cubeWeightInputNode->GetInDataNodesAndPortIndexs(0).first;
    }
    if (firstInputNode == nullptr) {
        OPS_LOG_D(kLogTag, "In caffe case, set ascend_weight_quant node to nullptr.");
        ascendWeightQuantNode = nullptr;
        weightConstNode = cubeWeightInputNode;
        return ge::SUCCESS;
    }
    ascendWeightQuantNode = cubeWeightInputNode;
    weightConstNode = firstInputNode;
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::WeightRollbackDeleteIsolateNodes(
    ge::Graph& graph, const ge::GNodePtr& ascendWeightQuantNode) const
{
    if (ascendWeightQuantNode == nullptr) {
        return ge::SUCCESS;
    }
    const auto offsetPeer = ascendWeightQuantNode->GetInDataNodesAndPortIndexs(1);
    if (offsetPeer.first == nullptr) {
        OPS_LOG_E(kLogTag, "Node[%s] offset const peer is null.", NodeName(*ascendWeightQuantNode).c_str());
        return ge::FAILED;
    }
    const ge::GNodePtr offsetConstNode = offsetPeer.first;
    if (ascendWeightQuantNode->GetOutDataNodesAndPortIndexs(0).empty()) {
        (void)graph.RemoveEdge(*offsetPeer.first, offsetPeer.second, *ascendWeightQuantNode, 1);
        if (offsetConstNode->GetOutDataNodesAndPortIndexs(0).empty()) {
            nodesToRemove_.push_back(offsetConstNode);
        }
        nodesToRemove_.push_back(ascendWeightQuantNode);
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::SetInt4OrInt8Flag(const ge::GNodePtr& ascendWeightQuantNode,
                                                            const ge::GNodePtr& quantHostCpuOpNode) const
{
    int64_t dstType = static_cast<int64_t>(ge::DT_INT8);
    if (ascendWeightQuantNode != nullptr) {
        (void)ascendWeightQuantNode->GetAttr(ge::AscendString(kAttrDstType), dstType);
    }
    (void)quantHostCpuOpNode->SetAttr(ge::AscendString(kAttrDstType), dstType);
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::DoFusion(ge::Graph& graph, const ge::GNodePtr& cubeNode,
                                                   const ge::GNodePtr& quantNode, const ge::GNodePtr& dequantNode,
                                                   std::vector<ge::GNodePtr>& fusNodes)
{
    if (cubeNode == nullptr) {
        OPS_LOG_E(kLogTag, "CubeNode is null, fusion failed.");
        return ge::FAILED;
    }
    if (quantNode == nullptr) {
        OPS_LOG_E(kLogTag, "QuantNode is null, fusion failed.");
        return ge::FAILED;
    }
    if (dequantNode == nullptr) {
        OPS_LOG_E(kLogTag, "DequantNode is null, fusion failed.");
        return ge::FAILED;
    }
    PatternNodes patternNodes = {cubeNode, dequantNode, quantNode, nullptr, nullptr};
    if (GetWeightNodesOfCubeNode(cubeNode, patternNodes.weightConstNode, patternNodes.ascendWeightQuantNode) !=
        ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Get weight_const node and ascend_weight_quant node of cube node[%s] failed.",
                  NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    if (CreateNewHostCpuOp(kQuantWeightRollBack, patternNodes, graph, GetWeightRollBackModeIndex(), fusNodes) !=
        ge::SUCCESS) {
        return ge::FAILED;
    }
    if (IsCubeHasBiasInput(cubeNode)) {
        if (CreateNewHostCpuOp(kQuantBiasRollBack, patternNodes, graph, GetBiasRollBackModeIndex(), fusNodes) !=
            ge::SUCCESS) {
            return ge::FAILED;
        }
    }
    if (WeightRollbackDeleteIsolateNodes(graph, patternNodes.ascendWeightQuantNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "In quant_roll_back mode, failed to delete AscendWeightQuant and offset_const nodes.");
        return ge::FAILED;
    }
    return SetDataTypeOfNodes(cubeNode);
}

ge::Status BiasOptimizeQuantRollbackBase::ChangeQuantNodeEdge(ge::Graph& graph, const ge::GNodePtr& cubeNode,
                                                              const ge::GNodePtr& quantNode) const
{
    PeerOutAnchors peerOutAnchorsOfNode;
    const ge::GNodePtr nodeAfterQuant = GetCubeNodeInputNode(cubeNode);
    if (nodeAfterQuant == nullptr) {
        OPS_LOG_E(kLogTag, "Get node after quant failed.");
        return ge::FAILED;
    }
    const auto inputPeer = quantNode->GetInDataNodesAndPortIndexs(0);
    if (inputPeer.first == nullptr) {
        OPS_LOG_E(kLogTag, "Quant node[%s] input peer is null.", NodeName(*quantNode).c_str());
        return ge::FAILED;
    }
    if (quantNode->GetOutDataNodesAndPortIndexs(0).size() == 1U) {
        if (RemoveInputEdgeAndSingleConstInput(graph, quantNode, peerOutAnchorsOfNode) != ge::SUCCESS) {
            OPS_LOG_E(kLogTag, "Remove input edge of quant node[%s] failed.", NodeName(*quantNode).c_str());
            return ge::FAILED;
        }
    } else {
        if (graph.RemoveEdge(*quantNode, 0, *nodeAfterQuant, 0) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Fail to remove edge between node[%s] and node[%s].", NodeName(*quantNode).c_str(),
                      NodeName(*nodeAfterQuant).c_str());
            return ge::FAILED;
        }
        if (graph.AddDataEdge(*inputPeer.first, inputPeer.second, *nodeAfterQuant, 0) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kLogTag, "Fail to add edge for the input data anchor of node[%s].",
                      NodeName(*nodeAfterQuant).c_str());
            return ge::FAILED;
        }
    }
    const std::string nodeName = NodeName(*nodeAfterQuant);
    if (LinkOutputEdgeWithoutControl(graph, quantNode, inputPeer.first, nodeName, peerOutAnchorsOfNode) !=
        ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Link output edge of quant node[%s] failed.", NodeName(*quantNode).c_str());
        return ge::FAILED;
    }
    if (quantNode->GetOutDataNodesAndPortIndexs(0).empty()) {
        nodesToRemove_.push_back(quantNode);
    }
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::ChangeDequantNodeEdge(ge::Graph& graph, const ge::GNodePtr& cubeNode,
                                                                const ge::GNodePtr& dequantNode)
{
    PeerOutAnchors peerOutAnchorsOfNode;
    const std::string cubeName = NodeName(*cubeNode);
    if (RemoveInputEdgeAndSingleConstInput(graph, dequantNode, peerOutAnchorsOfNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Remove input edge of dequant node[%s] failed.", NodeName(*dequantNode).c_str());
        return ge::FAILED;
    }
    if (LinkOutputEdgeWithoutControl(graph, dequantNode, cubeNode, cubeName, peerOutAnchorsOfNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Link output edge of dequant node[%s] failed.", NodeName(*dequantNode).c_str());
        return ge::FAILED;
    }
    nodesToRemove_.push_back(dequantNode);
    return ge::SUCCESS;
}

ge::Status BiasOptimizeQuantRollbackBase::QuantRollback(ge::Graph& graph, const ge::GNodePtr& cubeNode,
                                                        const ge::GNodePtr& dequantNode, const ge::GNodePtr& quantNode,
                                                        std::vector<ge::GNodePtr>& fusionNodes)
{
    OPS_LOG_D(kLogTag, "Begin to do quant rollback fusion on cube node[%s, %s].", NodeName(*cubeNode).c_str(),
              NodeType(*cubeNode).c_str());
    if (DoFusion(graph, cubeNode, quantNode, dequantNode, fusionNodes) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Failed to do fusion.");
        return ge::FAILED;
    }
    if (ChangeQuantNodeEdge(graph, cubeNode, quantNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Cube node[%s]: change quant node edge failed.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    OPS_LOG_D(kLogTag, "Cube node[%s]: change quant node edge successfully.", NodeName(*cubeNode).c_str());
    if (ChangeDequantNodeEdge(graph, cubeNode, dequantNode) != ge::SUCCESS) {
        OPS_LOG_E(kLogTag, "Cube node[%s]: change dequant node edge failed.", NodeName(*cubeNode).c_str());
        return ge::FAILED;
    }
    OPS_LOG_D(kLogTag, "Cube node[%s]: change dequant node edge successfully.", NodeName(*cubeNode).c_str());
    return ge::SUCCESS;
}
} // namespace ops
