/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "batch_matmulv2_quant_process_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "common/inc/error_util.h"
#include "version/ge-compiler_version.h"

namespace ops {
namespace {
/*
 *  fusion pattern（融合前）
 *           input
 *               \
 *             AscendQuant
 *                \
 *                 v
 *     weights--->BatchMatmulV2--->AscendDequant--->output
 *                ^          ^
 *               /          /
 *              /          /
 *           bias      deq_scale
 *
 *  融合后（BIAS_OPTIMIZE, 平台支持量化组合）:
 *  host 算子编译期把 FP 域 weight/bias 预量化为 int8/int32, quant/dequant 保留原位
 *
 *           input
 *               \
 *             AscendQuant
 *                \
 *                 v
 *    QuantBiasOptimization--->BatchMatmulV2--->AscendDequant--->output
 *      ^        ^        \        ^    ^
 *      |        |         \      /     |
 *   weights   bias(补零)   +---->+   deq_scale
 *                          (接管 x2/bias 位)
 */
constexpr const char* kPassName = "BatchMatmulV2QuantProcessFusionPass";
constexpr int32_t kCubeXInputPort = 0;    // BatchMatMulV2 x1 口，AscendQuant 挂接位置
constexpr int32_t kDequantXInputPort = 0; // AscendDequant x 口
constexpr int32_t kOutputPort = 0;
constexpr size_t kWeightDimSize = 2U;          // 权重维数
constexpr size_t kWeightWithBatchDimSize = 3U; // 带 batch 维的权重维数

bool HasType(const ge::GNodePtr& node, const ge::AscendString& type)
{
    ge::AscendString nodeType;
    return node != nullptr && node->GetType(nodeType) == ge::GRAPH_SUCCESS && nodeType == type;
}

std::string NodeNameOf(const ge::GNode& node)
{
    ge::AscendString name;
    if (node.GetName(name) != ge::GRAPH_SUCCESS || name.GetString() == nullptr) {
        return "";
    }
    return name.GetString();
}

ge::GNodePtr GetDequantConsumer(const ge::GNode& cube)
{
    for (const auto& consumer : cube.GetOutDataNodesAndPortIndexs(kOutputPort)) {
        if (consumer.second == kDequantXInputPort && HasType(consumer.first, ASCEND_DEQUANT)) {
            return consumer.first;
        }
    }
    return nullptr;
}

// 对齐 canndev ge::GeShape::IsUnknownShape：任一维为负值即视为未知 shape
bool IsUnknownShape(const ge::Shape& shape)
{
    const auto dims = shape.GetDims();
    return std::any_of(dims.begin(), dims.end(), [](const int64_t dim) {
        return dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM || dim < 0;
    });
}

// 老 UtilPlatform::isTiny：Intrinsic_set_l1 支持 f16 不支持 f32（Tiny 产品 L1 无 fp32 能力），
// 数据源与 conv2d_quant_process_fusion_pass 的 IsTinyProduct 同为 PlatformInfoManager。
bool IsTinyProduct()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optiInfo) != 0) {
        return false;
    }
    const auto& intrinsicMap = platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_set_l1"];
    const bool supportFp16 = std::find(intrinsicMap.begin(), intrinsicMap.end(), "f16") != intrinsicMap.end();
    const bool supportFp32 = std::find(intrinsicMap.begin(), intrinsicMap.end(), "f32") != intrinsicMap.end();
    return supportFp16 && !supportFp32;
}

// 老 UtilPlatform::isNano：UB block size 为 16 的 Nano 产品特性。
bool IsNanoProduct()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optiInfo) != 0) {
        return false;
    }
    constexpr uint64_t kNanoBlockSize = 16U;
    return platformInfo.ai_core_spec.ubblock_size == kNanoBlockSize;
}
} // namespace

ge::Status BatchMatmulV2QuantProcessFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    OPS_LOG_I(kPassName, "[OPS_NN_BATCHMATMULV2_QUANT_PROCESS] pass invoked.");
    if (graph == nullptr || !graph->IsValid()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    bool changed = false;
    for (auto cube : graph->GetAllNodes()) {
        auto cubeNode = std::make_shared<ge::GNode>(cube);
        if (!HasType(cubeNode, kOpTypeBatchMatMulV2)) {
            continue;
        }
        const auto quantInput = cube.GetInDataNodesAndPortIndexs(kCubeXInputPort);
        if (quantInput.second != kOutputPort || !HasType(quantInput.first, ASCEND_QUANT)) {
            continue;
        }
        const auto dequantNode = GetDequantConsumer(cube);
        if (dequantNode == nullptr) {
            continue;
        }
        std::vector<ge::GNodePtr> fusionNodes;
        std::vector<ge::GNodePtr> nodesToRemove;
        const std::vector<ge::GNode> matchedNodes = {*quantInput.first, *dequantNode, *cubeNode};
        weightFormatChanged_ = false; // 每个 cube 处理前重置（串行无并发）
        // 匹配到量化链即报 match（含 UNDEFINED/double-quant 让位等主动跳过场景, 对齐旧框架
        // match_times 全托管语义——json 中始终有条目, 不再出现整 pass 静默）
        ReportMatch(matchedNodes, passContext, kPassName);
        FUSION_PASS_CHECK(
            OptimizeMainProc(graph, quantInput.first, dequantNode, cubeNode, fusionNodes, nodesToRemove) != ge::SUCCESS,
            OPS_LOG_E(kPassName, "Quant process failed for node[%s].", NodeNameOf(cube).c_str()), return ge::FAILED);
        // effect 上报统一收口：载体 = 新增融合节点（BIAS_OPTIMIZE/ROLLBACK）；UNDEFINED 跳过路径
        // （含 double-quant 让位与纯 unknown-shape）原版旧框架按"返回 SUCCESS 即记 effect=1"语义
        // （原始算子实测 UNDEFINED 路径 effect=1）, 载体退化为原节点集合。FAILED 不报。
        // 上报须在节点删除之前：GE ReportFuse 的 ConvertGNodes 对已删节点 weak_ptr 悬空会失败。
        std::vector<ge::GNode> nodesAfterFuse;
        if (fusionNodes.empty()) {
            nodesAfterFuse = matchedNodes;
        } else {
            nodesAfterFuse.reserve(fusionNodes.size());
            for (const auto& node : fusionNodes) {
                if (node != nullptr) {
                    nodesAfterFuse.emplace_back(*node);
                }
            }
        }
        ReportFusion(matchedNodes, nodesAfterFuse, passContext, kPassName);
        if (weightFormatChanged_) {
            OPS_LOG_I(kPassName, "[OPS_NN_BATCHMATMULV2_QUANT_PROCESS] changed weight format for double-quant.");
        } else if (!fusionNodes.empty()) {
            OPS_LOG_I(kPassName, "[OPS_NN_BATCHMATMULV2_QUANT_PROCESS] optimized BatchMatMulV2 quant chain.");
        }
        for (const auto& node : nodesToRemove) {
            FUSION_PASS_CHECK(
                node == nullptr || graph->RemoveNode(*node) != ge::GRAPH_SUCCESS,
                OPS_LOG_E(kPassName, "Remove node of node[%s] failed after quant process.", NodeNameOf(cube).c_str()),
                return ge::FAILED);
        }
        changed = changed || !nodesAfterFuse.empty();
    }
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}

std::vector<int64_t> BatchMatmulV2QuantProcessFusionPass::GetBiasShape(const ge::GNodePtr& cubeNode) const
{
    ge::TensorDesc weightDesc;
    FUSION_PASS_CHECK(
        cubeNode->GetInputDesc(static_cast<int32_t>(GetWeightInputIndex()), weightDesc) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "Node[%s] get weight input desc failed.", NodeNameOf(*cubeNode).c_str()), return {});
    const auto filterDims = weightDesc.GetShape().GetDims();
    // 防护：Tiny 产品放行 <2 维权重后，下方 dims[size-2] 为负索引（原版为 UB）；
    // 此处安全返回空 shape，由调用方 ProcessBiasInput 走 FAILED 路径。
    if (filterDims.size() < kWeightDimSize) {
        OPS_LOG_W(kPassName, "Node[%s] weight dim num %zu less than 2, bias shape is undefined.",
                  NodeNameOf(*cubeNode).c_str(), filterDims.size());
        return {};
    }
    std::vector<int64_t> biasShape;
    if (filterDims.size() >= kWeightWithBatchDimSize) {
        int64_t batch = 1;
        for (size_t i = 0; i < filterDims.size() - kWeightDimSize; ++i) {
            batch *= filterDims[i];
        }
        biasShape.emplace_back(batch);
    }
    const int64_t co = IsWeightTranspose(cubeNode) ? filterDims[filterDims.size() - 2U] :
                                                     filterDims[filterDims.size() - 1U];
    biasShape.emplace_back(co);
    return biasShape;
}

ge::Status BatchMatmulV2QuantProcessFusionPass::ProcessBiasInput(ge::Graph& graph, const ge::GNodePtr& cubeNode)
{
    if (IsCubeNeedBiasInput(cubeNode)) {
        const std::vector<int64_t> biasShape = GetBiasShape(cubeNode);
        FUSION_PASS_CHECK(biasShape.empty(),
                          OPS_LOG_E(kPassName, "Get node[%s] bias shape failed.", NodeNameOf(*cubeNode).c_str()),
                          return ge::FAILED);
        FUSION_PASS_CHECK(
            CreateBiasInput(graph, cubeNode, ge::Shape(biasShape)) != ge::SUCCESS,
            OPS_LOG_E(kPassName, "Create cube node[%s] bias input failed.", NodeNameOf(*cubeNode).c_str()),
            return ge::FAILED);
    }
    return ge::SUCCESS;
}

ge::Status BatchMatmulV2QuantProcessFusionPass::GetQuantProcessMode(const ge::GNodePtr& quantNode,
                                                                    const ge::GNodePtr& cubeNode,
                                                                    QuantProcessMode& quantProcessMode)
{
    (void)quantNode;
    if (!IsCubeNodeSupported(cubeNode)) {
        quantProcessMode = QuantProcessMode::QUANT_ROLLBACK;
        return ge::SUCCESS;
    }
    const auto weightPeer = cubeNode->GetInDataNodesAndPortIndexs(static_cast<int32_t>(GetWeightInputIndex()));
    if (weightPeer.first != nullptr && HasType(weightPeer.first, ASCEND_QUANT)) {
        OPS_LOG_D(kPassName, "Node[%s] weight node is ascend quant.", NodeNameOf(*cubeNode).c_str());
        quantProcessMode = QuantProcessMode::QUANT_UNDEFINED;
        ge::TensorDesc weightDesc;
        if (cubeNode->GetInputDesc(static_cast<int32_t>(GetWeightInputIndex()), weightDesc) == ge::GRAPH_SUCCESS) {
            weightDesc.SetOriginFormat(ge::FORMAT_HWCN);
            weightDesc.SetFormat(ge::FORMAT_HWCN);
            (void)cubeNode->UpdateInputDesc(static_cast<int32_t>(GetWeightInputIndex()), weightDesc);
            // 图已变更（weight Format 置 HWCN），Run 依据此标记补报 effect（原始算子该场景 effect=1）
            weightFormatChanged_ = true;
        }
        return ge::SUCCESS;
    }
    ge::TensorDesc weightDesc;
    FUSION_PASS_CHECK(
        cubeNode->GetInputDesc(static_cast<int32_t>(GetWeightInputIndex()), weightDesc) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "Node[%s] get weight input desc failed.", NodeNameOf(*cubeNode).c_str()),
        return ge::FAILED);
    // 老 canndev 逐字三分支语义：标准平台权重必须 2 维；Nano 产品（ubblock_size==16）允许 >=2 维
    // （3 维走 GetBiasShape 的 batch 维路径）；Tiny 产品（L1 仅 f16）不做维数限制。
    const size_t weightDimNum = weightDesc.GetShape().GetDims().size();
    FUSION_PASS_CHECK(
        weightDimNum != kWeightDimSize && !IsTinyProduct() && !IsNanoProduct(),
        OPS_LOG_E(kPassName, "Dim size of weight should equal 2 in this platform, but now is %zu.", weightDimNum),
        return ge::FAILED);
    FUSION_PASS_CHECK(
        weightDimNum < kWeightDimSize && IsNanoProduct(),
        OPS_LOG_E(kPassName, "Dim size of weight should >= 2 in nano platform, but now is %zu.", weightDimNum),
        return ge::FAILED);
    if (IsUnknownShape(weightDesc.GetShape())) {
        OPS_LOG_W(kPassName, "Quant process fusion cannot be applied for unknown shape.");
        quantProcessMode = QuantProcessMode::QUANT_UNDEFINED;
        return ge::SUCCESS;
    }
    quantProcessMode = QuantProcessMode::BIAS_OPTIMIZE;
    return ge::SUCCESS;
}

// 基类依赖 903+ 的 GE 特性（SetOriginDataType 等), 低于 903 的编译环境不注册本 pass
#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(BatchMatmulV2QuantProcessFusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
#endif
} // namespace ops
