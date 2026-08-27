/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_add_layer_norm_fusion_pass.cpp
 * \brief AddLayerNorm --> InplaceAddLayerNorm (graph base route)
 *
 * 使用 graph_metadef 的原生 Graph/GNode 接口改图。
 */
#include "inplace_add_layer_norm_fusion_pass.h"

#include <dlfcn.h>

#include <cstdlib>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#include "common/inc/error_util.h"
#include "graph/operator_factory.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
namespace {
// GetOptionValue @since 9.0.0，按 D4 走 dlsym。
const char* const kGetOptionValueSymbol = "_ZNK2ge17CustomPassContext14GetOptionValueERKNS_12AscendStringERS1_";
using GetOptionValueFn = graphStatus (*)(const void*, const AscendString&, AscendString&);

// GE 库以 RTLD_GLOBAL 加载，故查全局符号表；
GetOptionValueFn ResolveGetOptionValue()
{
    static GetOptionValueFn fn = reinterpret_cast<GetOptionValueFn>(dlsym(RTLD_DEFAULT, kGetOptionValueSymbol));
    return fn;
}

const std::string kPassName = "ZInplaceAddLayerNormFusionPass";

const char* const kAddLayerNormType = "AddLayerNorm";
const char* const kInplaceAddLayerNormType = "InplaceAddLayerNorm";
const char* const kNewNodeNameSuffix = "_inplace";

// IR 位序：输入 x1 x2 gamma beta [bias]；输出 y/x1' mean rstd x/x2'
const int32_t kInputIdxX1 = 0;
const int32_t kInputIdxX2 = 1;
const int32_t kInputIdxGamma = 2;
const size_t kInputNumWithoutBias = 4U;
const size_t kInputNumWithBias = 5U;
const size_t kOutputNum = 4U;

// 只有 x1/x2 会被原地写回，独占性只需检查这两个输入
const int32_t kInputRefNums = 2;

const int64_t kMainPartsNums = 4L;
const int64_t kL2UpperFactor = 2L;
const int64_t kL2LowerFactor = 2L;

const int64_t kUnknownDim = -1L;
const int64_t kUnknownRank = -2L;

const int64_t kReduceAxisValue = 5120L;

const char* const kAttrEpsilon = "epsilon";
const char* const kAttrAdditionalOutput = "additional_output";
const char* const kAttrContinuousOutput = "continuous_output";
const char* const kOptionGraphRunMode = "ge.graphRunMode";
const int32_t kGraphRunModeTrain = 1;
const int32_t kDecimalBase = 10;

const int32_t kMinGeCompilerVersion = 90000000;

// 含未知维或未知秩即动态 shape。GE 未提供作用于对外 Shape 的等价接口，自备。
bool IsDynamicShape(const Shape& shape)
{
    const size_t dimNum = shape.GetDimNum();
    for (size_t i = 0U; i < dimNum; ++i) {
        const int64_t dim = shape.GetDim(i);
        if ((dim == kUnknownDim) || (dim == kUnknownRank)) {
            return true;
        }
    }
    return false;
}

enum class SceneCheckResult {
    INFERENCE,
    TRAIN,
    API_UNAVAILABLE // 运行时 < 9.0.0
};

// 守卫 G1：训练场景不做原地改写；读不到 option 则返回 API_UNAVAILABLE 保持静默。
SceneCheckResult CheckScene(CustomPassContext& passContext)
{
    // 运行期check
    int32_t runtimeVersion = 0;
    char geCompilerName[] = "ge_compiler";
    (void)aclsysGetVersionNum(geCompilerName, &runtimeVersion);
    if (runtimeVersion > 0 && runtimeVersion < kMinGeCompilerVersion) {
        return SceneCheckResult::API_UNAVAILABLE;
    }

    // 符号可达性
    const GetOptionValueFn getOptionValue = ResolveGetOptionValue();
    if (getOptionValue == nullptr) {
        return SceneCheckResult::API_UNAVAILABLE;
    }

    AscendString value;
    if (getOptionValue(&passContext, AscendString(kOptionGraphRunMode), value) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Option %s is not set, treat as inference scene.", kOptionGraphRunMode);
        return SceneCheckResult::INFERENCE;
    }
    const char* modeStr = value.GetString();
    if (modeStr == nullptr) {
        return SceneCheckResult::INFERENCE;
    }
    if (static_cast<int32_t>(std::strtol(modeStr, nullptr, kDecimalBase)) == kGraphRunModeTrain) {
        return SceneCheckResult::TRAIN;
    }
    return SceneCheckResult::INFERENCE;
}

// 守卫 G2：平台check。
bool IsSupportedPlatform(int64_t& l2Size)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        ge::SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Get platform info failed, skip.");
        return false;
    }

    const std::string curSoc = platformInfo.str_info.short_soc_version;
    static const std::set<std::string> kSupportSoc = {"Ascend910B", "Ascend910_93", "Ascend950"};
    if (kSupportSoc.count(curSoc) == 0U) {
        OPS_LOG_D(kPassName.c_str(), "Platform %s is not supported, skip.", curSoc.c_str());
        return false;
    }

    l2Size = static_cast<int64_t>(platformInfo.soc_info.l2_size);
    return true;
}

// 守卫 G4/G5：x1、x2 必须被本节点独占消费，否则原地写回会踩踏别人的数据。
bool IsInplaceSafe(const GNode& node)
{
    bool isContinuousOutput = false;
    (void)node.GetAttr(AscendString(kAttrContinuousOutput), isContinuousOutput);

    for (int32_t inputIdx = 0; inputIdx < kInputRefNums; ++inputIdx) {
        const std::pair<GNodePtr, int32_t> peer = node.GetInDataNodesAndPortIndexs(inputIdx);
        const GNodePtr producer = peer.first;
        if (producer == nullptr) {
            OPS_LOG_D(kPassName.c_str(), "Input %d has no producer, skip.", inputIdx);
            return false;
        }

        if (producer->GetOutDataNodesAndPortIndexs(peer.second).size() > 1U) {
            OPS_LOG_D(kPassName.c_str(), "Input %d is shared by other consumers, skip.", inputIdx);
            return false;
        }

        if (producer->GetOutputsSize() > 1U && isContinuousOutput) {
            OPS_LOG_D(kPassName.c_str(), "Producer of input %d has continuous multi-outputs, skip.", inputIdx);
            return false;
        }
    }
    return true;
}

// 守卫 G6：原地收益与 L2 容量相关。太大装不下，太小不值得。
bool IsShapeSupport(const GNode& node, int64_t l2Size)
{
    TensorDesc x1Desc;
    if (node.GetInputDesc(kInputIdxX1, x1Desc) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Get x1 input desc failed, skip.");
        return false;
    }
    const Shape x1Shape = x1Desc.GetShape();

    if (IsDynamicShape(x1Shape)) {
        const size_t dimNum = x1Shape.GetDimNum();
        if (dimNum == 0U) {
            OPS_LOG_D(kPassName.c_str(), "Dynamic shape with zero dim, skip.");
            return false;
        }
        if (x1Shape.GetDim(dimNum - 1U) != kReduceAxisValue) {
            OPS_LOG_D(kPassName.c_str(), "Dynamic shape last dim is not %ld, skip.", kReduceAxisValue);
            return false;
        }
        return true;
    }

    if (l2Size <= 0L) {
        OPS_LOG_D(kPassName.c_str(), "Invalid l2_size %ld, skip.", l2Size);
        return false;
    }

    const int64_t x1Size = x1Shape.GetShapeSize() * static_cast<int64_t>(GetSizeByDataType(x1Desc.GetDataType()));
    if (x1Size > l2Size * kL2UpperFactor) {
        OPS_LOG_D(kPassName.c_str(), "Input x1 size %ld far exceeds l2 %ld, skip.", x1Size, l2Size);
        return false;
    }
    if (x1Size * kMainPartsNums < l2Size / kL2LowerFactor) {
        OPS_LOG_D(kPassName.c_str(), "Input x1 size %ld far below l2 %ld, skip.", x1Size, l2Size);
        return false;
    }
    return true;
}

// 守卫 G7：x1/x2/gamma 必须同 dtype，混合类型下原地语义不成立。
bool IsSameInputDataType(const GNode& node)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc gammaDesc;
    if (node.GetInputDesc(kInputIdxX1, x1Desc) != GRAPH_SUCCESS ||
        node.GetInputDesc(kInputIdxX2, x2Desc) != GRAPH_SUCCESS ||
        node.GetInputDesc(kInputIdxGamma, gammaDesc) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Get input desc for dtype check failed, skip.");
        return false;
    }

    const DataType x1Dtype = x1Desc.GetDataType();
    if (x1Dtype != x2Desc.GetDataType() || x1Dtype != gammaDesc.GetDataType()) {
        OPS_LOG_D(kPassName.c_str(), "Inputs x1/x2/gamma have different dtypes, skip.");
        return false;
    }
    return true;
}

// 逐节点守卫链。全局守卫（场景、平台）在 Run 入口只判一次。
bool MeetGuards(const GNode& node, int64_t l2Size)
{
    const size_t inputNum = node.GetInputsSize();
    if (inputNum != kInputNumWithoutBias && inputNum != kInputNumWithBias) {
        OPS_LOG_D(kPassName.c_str(), "Unexpected input num %zu, skip.", inputNum);
        return false;
    }
    if (node.GetOutputsSize() != kOutputNum) {
        OPS_LOG_D(kPassName.c_str(), "Unexpected output num %zu, skip.", node.GetOutputsSize());
        return false;
    }
    if (!IsInplaceSafe(node)) {
        return false;
    }
    if (!IsShapeSupport(node, l2Size)) {
        return false;
    }
    if (!IsSameInputDataType(node)) {
        return false;
    }

    OPS_LOG_D(kPassName.c_str(), "All guards passed, input num %zu.", inputNum);
    return true;
}

bool CreateInplaceNode(Graph& graph, const GNode& oldNode, GNode& newNode)
{
    AscendString oldName;
    if (oldNode.GetName(oldName) != GRAPH_SUCCESS || oldName.GetString() == nullptr) {
        return false;
    }
    const std::string newName = std::string(oldName.GetString()) + kNewNodeNameSuffix;

    Operator op = OperatorFactory::CreateOperator(newName.c_str(), kInplaceAddLayerNormType);
    newNode = graph.AddNodeByOp(op);

    AscendString newType;
    if (newNode.GetType(newType) != GRAPH_SUCCESS || newType != AscendString(kInplaceAddLayerNormType)) {
        OPS_LOG_E(kPassName.c_str(), "Create %s node failed.", kInplaceAddLayerNormType);
        return false;
    }

    // attr 原样透传，读不到则保留 IR 默认值。
    float32_t epsilon = 1e-5F;
    if (oldNode.GetAttr(AscendString(kAttrEpsilon), epsilon) == GRAPH_SUCCESS) {
        (void)newNode.SetAttr(AscendString(kAttrEpsilon), epsilon);
    }
    bool additionalOutput = false;
    if (oldNode.GetAttr(AscendString(kAttrAdditionalOutput), additionalOutput) == GRAPH_SUCCESS) {
        (void)newNode.SetAttr(AscendString(kAttrAdditionalOutput), additionalOutput);
    }
    return true;
}

// 输入逐位接线并拷贝 TensorDesc。bias 缺省（4 输入）时第 5 个端口保持未接。
bool RewireInputs(Graph& graph, const GNode& oldNode, GNode& newNode, size_t inputNum)
{
    for (size_t i = 0U; i < inputNum; ++i) {
        const int32_t idx = static_cast<int32_t>(i);
        const std::pair<GNodePtr, int32_t> peer = oldNode.GetInDataNodesAndPortIndexs(idx);
        if (peer.first == nullptr) {
            continue;
        }
        if (graph.AddDataEdge(*peer.first, peer.second, newNode, idx) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Add input edge %zu failed.", i);
            return false;
        }
        TensorDesc desc;
        if (oldNode.GetInputDesc(idx, desc) == GRAPH_SUCCESS) {
            (void)newNode.UpdateInputDesc(idx, desc);
        }
    }
    return true;
}

// 输出逐位改接并拷贝 TensorDesc。替换前后 shape/dtype 完全相同，直接拷贝，无需 InferShape。
bool RewireOutputs(Graph& graph, GNode& oldNode, GNode& newNode)
{
    for (size_t j = 0U; j < kOutputNum; ++j) {
        const int32_t idx = static_cast<int32_t>(j);
        TensorDesc desc;
        if (oldNode.GetOutputDesc(idx, desc) == GRAPH_SUCCESS) {
            (void)newNode.UpdateOutputDesc(idx, desc);
        }

        // 该输出的所有下游都要改接，漏一个就断图
        const std::vector<std::pair<GNodePtr, int32_t>> consumers = oldNode.GetOutDataNodesAndPortIndexs(idx);
        for (const std::pair<GNodePtr, int32_t>& consumer : consumers) {
            if (consumer.first == nullptr) {
                continue;
            }
            if (graph.RemoveEdge(oldNode, idx, *consumer.first, consumer.second) != GRAPH_SUCCESS) {
                OPS_LOG_E(kPassName.c_str(), "Remove output edge %zu failed.", j);
                return false;
            }
            if (graph.AddDataEdge(newNode, idx, *consumer.first, consumer.second) != GRAPH_SUCCESS) {
                OPS_LOG_E(kPassName.c_str(), "Add output edge %zu failed.", j);
                return false;
            }
        }
    }
    return true;
}

// 控制边搬运。
bool RewireControlEdges(Graph& graph, const GNode& oldNode, GNode& newNode)
{
    for (const GNodePtr& src : oldNode.GetInControlNodes()) {
        if (src == nullptr) {
            continue;
        }
        if (graph.AddControlEdge(*src, newNode) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Add in-control edge failed.");
            return false;
        }
    }
    for (const GNodePtr& dst : oldNode.GetOutControlNodes()) {
        if (dst == nullptr) {
            continue;
        }
        if (graph.AddControlEdge(newNode, *dst) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Add out-control edge failed.");
            return false;
        }
    }
    return true;
}

bool RemoveOldNode(Graph& graph, GNode& oldNode, size_t inputNum)
{
    for (size_t i = 0U; i < inputNum; ++i) {
        const int32_t idx = static_cast<int32_t>(i);
        const std::pair<GNodePtr, int32_t> peer = oldNode.GetInDataNodesAndPortIndexs(idx);
        if (peer.first == nullptr) {
            continue;
        }
        if (graph.RemoveEdge(*peer.first, peer.second, oldNode, idx) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Remove input edge %zu failed.", i);
            return false;
        }
    }
    if (graph.RemoveNode(oldNode) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Remove old node failed.");
        return false;
    }
    return true;
}

// 单节点替换：建点 -> 接输入 -> 改接输出 -> 搬控制边 -> 断旧边删旧点
bool ReplaceOneNode(Graph& graph, GNode& oldNode)
{
    const size_t inputNum = oldNode.GetInputsSize();

    GNode newNode;
    if (!CreateInplaceNode(graph, oldNode, newNode)) {
        return false;
    }
    if (!RewireInputs(graph, oldNode, newNode, inputNum)) {
        return false;
    }
    if (!RewireOutputs(graph, oldNode, newNode)) {
        return false;
    }
    if (!RewireControlEdges(graph, oldNode, newNode)) {
        return false;
    }
    if (!RemoveOldNode(graph, oldNode, inputNum)) {
        return false;
    }

    OPS_LOG_D(kPassName.c_str(), "Replaced, with_bias=%d.", static_cast<int32_t>(inputNum == kInputNumWithBias));
    return true;
}
} // namespace

Status ZInplaceAddLayerNormFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    if (graph == nullptr) {
        return GRAPH_NOT_CHANGED;
    }

    // 全局守卫：与具体节点无关，只判一次
    const SceneCheckResult scene = CheckScene(passContext);
    if (scene == SceneCheckResult::API_UNAVAILABLE) {
        // 兼容，保持静默
        OPS_LOG_D(kPassName.c_str(), "GetOptionValue is unavailable below CANN 9.0.0, stay silent.");
        return GRAPH_NOT_CHANGED;
    }
    if (scene == SceneCheckResult::TRAIN) {
        OPS_LOG_D(kPassName.c_str(), "Train mode is not supported, skip.");
        return GRAPH_NOT_CHANGED;
    }

    int64_t l2Size = 0L;
    if (!IsSupportedPlatform(l2Size)) {
        return GRAPH_NOT_CHANGED;
    }

    // 扫图取候选。
    std::vector<GNode> candidates;
    for (auto& node : graph->GetDirectNode()) {
        AscendString nodeType;
        if (node.GetType(nodeType) != GRAPH_SUCCESS) {
            continue;
        }
        if (nodeType != AscendString(kAddLayerNormType)) {
            continue;
        }
        candidates.emplace_back(node);
    }
    if (candidates.empty()) {
        OPS_LOG_D(kPassName.c_str(), "No %s node found.", kAddLayerNormType);
        return GRAPH_NOT_CHANGED;
    }

    // 替换失败时整图回退，避免留下半改的图
    Graph originGraph = *graph;
    bool changed = false;
    for (auto& node : candidates) {
        if (!MeetGuards(node, l2Size)) {
            continue;
        }

        if (!ReplaceOneNode(*graph, node)) {
            OPS_LOG_E(kPassName.c_str(), "Replacement failed, rollback whole graph.");
            passContext.SetErrorMessage(AscendString("ZInplaceAddLayerNormFusionPass replacement failed."));
            *graph = originGraph;
            return FAILED;
        }
        changed = true;
    }

    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}

#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90000000)
REG_FUSION_PASS(ZInplaceAddLayerNormFusionPass).Stage(CustomPassStage::kAfterBuiltinFusionPass);
#endif
} // namespace ops
