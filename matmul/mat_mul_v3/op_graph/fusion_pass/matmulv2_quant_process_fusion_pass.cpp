/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmulv2_quant_process_fusion_pass.h"

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
 *     weights--->MatmulV2--->AscendDequant--->output
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
 *    QuantBiasOptimization--->MatmulV2--->AscendDequant--->output
 *      ^        ^        \        ^   ^
 *      |        |         \      /    |
 *   weights   bias(补零)   +---->+  deq_scale
 *                          (接管 x2/bias 位)
 */
constexpr const char* kPassName = "MatmulV2QuantProcessFusionPass";
constexpr int32_t kCubeXInputPort = 0;    // MatMulV2 x1 口，AscendQuant 挂接位置
constexpr int32_t kDequantXInputPort = 0; // AscendDequant x 口
constexpr int32_t kOutputPort = 0;
constexpr size_t kWeightDimSize = 2U; // MatMulV2 权重维数

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

// 对齐 canndev UnknownShapeUtils::IsUnknownShapeValue：负值视为未知维度
bool IsUnknownDim(const int64_t dim) { return dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM || dim < 0; }
} // namespace

ge::Status MatmulV2QuantProcessFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    OPS_LOG_I(kPassName, "[OPS_NN_MATMULV2_QUANT_PROCESS] pass invoked.");
    if (graph == nullptr || !graph->IsValid()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    bool changed = false;
    for (auto cube : graph->GetAllNodes()) {
        auto cubeNode = std::make_shared<ge::GNode>(cube);
        if (!HasType(cubeNode, kOpTypeMatMulV2)) {
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
        // 匹配到量化链即报 match（含 UNDEFINED/主动跳过场景, 对齐旧框架 match_times 全托管语义）
        ReportMatch(matchedNodes, passContext, kPassName);
        FUSION_PASS_CHECK(
            OptimizeMainProc(graph, quantInput.first, dequantNode, cubeNode, fusionNodes, nodesToRemove) != ge::SUCCESS,
            OPS_LOG_E(kPassName, "Quant process failed for node[%s].", NodeNameOf(cube).c_str()), return ge::FAILED);
        // effect 上报统一收口：载体 = 新增融合节点（BIAS_OPTIMIZE/ROLLBACK）；UNDEFINED 跳过
        // （unknown-shape）原版旧框架按"返回 SUCCESS 即记 effect=1"语义（原始算子实测同）,
        // 载体退化为原节点集合。FAILED 不报。上报须在节点删除之前：GE ReportFuse 的
        // ConvertGNodes 对已删节点 weak_ptr 悬空会失败（conv 家族先例）。
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
        if (fusionNodes.empty()) {
            OPS_LOG_I(kPassName, "[OPS_NN_MATMULV2_QUANT_PROCESS] skipped quant chain for unknown shape.");
        } else {
            OPS_LOG_I(kPassName, "[OPS_NN_MATMULV2_QUANT_PROCESS] optimized MatMulV2 quant chain.");
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

ge::Status MatmulV2QuantProcessFusionPass::GetCoValue(const ge::GNodePtr& cubeNode, int64_t& co)
{
    ge::TensorDesc weightDesc;
    FUSION_PASS_CHECK(
        cubeNode->GetInputDesc(static_cast<int32_t>(GetWeightInputIndex()), weightDesc) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "Node[%s] get weight input desc failed.", NodeNameOf(*cubeNode).c_str()),
        return ge::FAILED);
    const auto filterDims = weightDesc.GetShape().GetDims();
    if (filterDims.size() == kWeightDimSize) {
        co = IsWeightTranspose(cubeNode) ? filterDims[0] : filterDims[1];
    }
    return ge::SUCCESS;
}

ge::Status MatmulV2QuantProcessFusionPass::GetQuantProcessMode(const ge::GNodePtr& quantNode,
                                                               const ge::GNodePtr& cubeNode,
                                                               QuantProcessMode& quantProcessMode)
{
    if (!IsCubeNodeSupported(cubeNode)) {
        quantProcessMode = QuantProcessMode::QUANT_ROLLBACK;
        return ge::SUCCESS;
    }
    ge::TensorDesc inputDesc;
    FUSION_PASS_CHECK(quantNode->GetInputDesc(kCubeXInputPort, inputDesc) != ge::GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Node[%s] get quant input desc failed.", NodeNameOf(*quantNode).c_str()),
                      return ge::FAILED);
    const auto inputDims = inputDesc.GetShape().GetDims();
    // 维数不足时按未知维度处理，对齐旧 GeShape::GetDim 越界返回 -1 的行为
    const auto dimAt = [&inputDims](const size_t idx) -> int64_t {
        return (idx < inputDims.size()) ? inputDims[idx] : ge::UNKNOWN_DIM;
    };
    if (IsUnknownDim(dimAt(0)) || IsUnknownDim(dimAt(1))) {
        OPS_LOG_W(kPassName, "Quant process fusion cannot be applied for unknown shape.");
        quantProcessMode = QuantProcessMode::QUANT_UNDEFINED;
        return ge::SUCCESS;
    }
    quantProcessMode = QuantProcessMode::BIAS_OPTIMIZE;
    return ge::SUCCESS;
}

// 基类依赖 903+ 的 GE 特性（SetOriginDataType 等), 低于 903 的编译环境不注册本 pass
#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(MatmulV2QuantProcessFusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
#endif
} // namespace ops
