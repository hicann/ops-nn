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
 * \file batch_matmul_transpose_fusion_pass.cpp
 * \brief BatchMatMul Transpose Fusion Pass
 *
 * Do Transpose + MatMul Fusion:
 *
 *                     input1
 *                        |
 *            input0   transpose             input0   input1
 *               \      /          ====>        \      /
 *                batchmatmul                    batchmatmul
 *                  |                               |
 *                  out                            out
 *
 * The pass finds Transpose/TransposeD nodes feeding into MatMul/MatMulV2/
 * BatchMatMul/BatchMatMulV2 inputs. If the transpose only swaps the last two
 * dimensions, it is removed and the matmul transpose attribute is toggled.
 */

#include "batch_matmul_transpose_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "version/ge-compiler_version.h"
#include "acl/acl_rt.h"

using namespace ge;
using namespace fe;

namespace ge {
namespace fusion {
class GraphFuseInspectorUtils {
public:
    static Status ReportFuse(const std::vector<GNode>& nodesBeforeFuse, const std::vector<GNode>& nodesAfterFuse,
                             CustomPassContext& ctx) __attribute__((weak));
};
} // namespace fusion
} // namespace ge

namespace ops {
namespace {

constexpr char kPassName[] = "BatchMatMulTransposeFusionPass";
constexpr char kOpTypeTranspose[] = "Transpose";
constexpr char kOpTypeTransposeD[] = "TransposeD";
constexpr int32_t kDimLimit = 2;

bool IsTargetVersion()
{
    int32_t version = 0;
    if (aclsysGetVersionNum("ge-compiler", &version) != ACL_SUCCESS) {
        OPS_LOG_W(kPassName, "Failed to get ge-compiler version, skip fusion.");
        return false;
    }
    return version >= kTargetGeCompilerVersion;
}

bool IsMatMulType(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS) {
        return false;
    }
    return opType == kOpTypeMatMul || opType == kOpTypeMatMulV2 || opType == kOpTypeBatchMatMul ||
           opType == kOpTypeBatchMatMulV2;
}

bool IsTransposeType(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS) {
        return false;
    }
    return opType == kOpTypeTranspose || opType == kOpTypeTransposeD;
}

bool IsBatchOp(const AscendString& opType) { return opType == kOpTypeBatchMatMul || opType == kOpTypeBatchMatMulV2; }

GNodePtr GetInputNode(const GNode& node, int32_t port) { return node.GetInDataNodesAndPortIndexs(port).first; }

bool GetTransposePerm(const GNode& transposeNode, std::vector<int32_t>& permValue)
{
    AscendString transType;
    if (transposeNode.GetType(transType) != GRAPH_SUCCESS) {
        return false;
    }

    std::vector<int64_t> permInt64;
    if (transType == kOpTypeTransposeD) {
        if (transposeNode.GetAttr("perm", permInt64) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName, "Get attr perm failed.");
            return false;
        }
    } else {
        if (!GetTransposePermFromConst(transposeNode, permInt64)) {
            return false;
        }
    }

    for (auto v : permInt64) {
        permValue.push_back(static_cast<int32_t>(v));
    }
    return true;
}

// 校验 Transpose 节点是否可融合：
// - x1 侧（inputIndex==0）：transpose 输出只能有 1 个消费者（严格独占）
// - x2 侧（inputIndex==1）：若 dtype 是 fp16/bf16，允许 transpose 输出有多个消费者，
// - 只允许交换最后两维的 transpose（batch 维不可变），这样的 perm 可用 adj_x/transpose_x 属性吸收
bool CheckTransposeFusion(const GNode& transposeNode, int32_t inputIndex, DataType dataDtype)
{
    AscendString transType;
    FUSION_PASS_CHECK(transposeNode.GetType(transType) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get transpose node type failed."), return false);
    FUSION_PASS_CHECK(transType != kOpTypeTransposeD && transType != kOpTypeTranspose,
                      OPS_LOG_W(kPassName, "Transpose type is %s.", transType.GetString()), return false);

    auto outConsumers = transposeNode.GetOutDataNodesAndPortIndexs(0);
    FUSION_PASS_CHECK(inputIndex == 0 && outConsumers.size() != 1,
                      OPS_LOG_W(kPassName, "when input index is 0, Transpose output is not 1."), return false);
    FUSION_PASS_CHECK(inputIndex == 1 && outConsumers.size() != 1 && dataDtype != DT_FLOAT16 && dataDtype != DT_BF16,
                      OPS_LOG_W(kPassName, "input index is 1 and dtype isn't fp16 or bf16, transpose out is not 1."),
                      return false);

    auto outControlNodes = transposeNode.GetOutControlNodes();
    FUSION_PASS_CHECK(!outControlNodes.empty(),
                      OPS_LOG_W(kPassName, "Transpose has output control edge, fusion is not supported."),
                      return false);

    TensorDesc transInputDesc;
    FUSION_PASS_CHECK(transposeNode.GetInputDesc(0, transInputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "The opdesc of transpose is null."), return false);

    std::vector<int32_t> permValue;
    FUSION_PASS_CHECK(!GetTransposePerm(transposeNode, permValue), OPS_LOG_W(kPassName, "Get transpose perm failed."),
                      return false);

    int32_t permLen = static_cast<int32_t>(permValue.size());
    auto inputDims = transInputDesc.GetShape().GetDims();
    int32_t inputDimNum = static_cast<int32_t>(inputDims.size());
    FUSION_PASS_CHECK(
        permLen != inputDimNum || permLen < kDimLimit,
        OPS_LOG_W(kPassName,
                  "perm value dim should be equal to input dim and must be >= 2, now perm value len is %d, "
                  "input shape len is %d.",
                  permLen, inputDimNum),
        return false);

    // batch 维度（前 permLen-2 个维度）不允许被 transpose
    for (int i = 0; i < permLen - kDimLimit; i++) {
        FUSION_PASS_CHECK(permValue[i] != i, OPS_LOG_W(kPassName, "batch dim is transposed."), return false);
    }

    // 只允许最后两维互换：perm[len-1]==len-2 且 perm[len-2]==len-1
    return permValue[permLen - 1] == permLen - kDimLimit && permValue[permLen - kDimLimit] == permLen - 1;
}

// 翻转 matmul 的 transpose 属性，用属性吸收外部 Transpose 节点的语义。
// attrPrefix 为 "adj_x"（BatchMatMul 系列）或 "transpose_x"（MatMul 系列），
// 拼接 dataIndex+1 得到完整属性名（如 adj_x1、transpose_x2）。
bool ToggleTransposeAttr(GNode& matmulNode, int32_t dataIndex, const char* attrPrefix)
{
    std::string attrName = std::string(attrPrefix) + std::to_string(dataIndex + 1);
    bool adj = false;
    FUSION_PASS_CHECK(matmulNode.GetAttr(attrName.c_str(), adj) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul attr %s failed.", attrName.c_str()), return false);
    adj = !adj;
    FUSION_PASS_CHECK(matmulNode.SetAttr(attrName.c_str(), adj) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Set matmul attr %s failed.", attrName.c_str()), return false);
    return true;
}

// 翻转 matmul 输入 desc 的最后两维 shape 和 shapeRange，
// 与 adj_x 属性翻转配套：adj_x 翻转后 matmul 内部会转置输入，因此输入 desc 的 shape 也要对应翻转。
bool SwapLastTwoDims(TensorDesc& desc)
{
    auto dims = desc.GetShape().GetDims();
    FUSION_PASS_CHECK(dims.size() < static_cast<size_t>(kDimLimit),
                      OPS_LOG_W(kPassName, "MatMul input shape dim must be >= 2."), return false);
    std::swap(dims[dims.size() - 1], dims[dims.size() - static_cast<size_t>(kDimLimit)]);
    Shape shape(dims);
    desc.SetShape(shape);
    desc.SetOriginShape(shape);

    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    desc.GetShapeRange(shapeRange);
    if (!shapeRange.empty()) {
        FUSION_PASS_CHECK(shapeRange.size() != dims.size(),
                          OPS_LOG_W(kPassName, "MatMul input shape dim and range dim are not equal."), return false);
        std::swap(shapeRange[shapeRange.size() - 1], shapeRange[shapeRange.size() - static_cast<size_t>(kDimLimit)]);
        desc.SetShapeRange(shapeRange);
    }
    return true;
}

bool RelinkEdge(const GraphPtr& graph, GNode& transposeNode, GNode& matmulNode, int32_t dataIndex)
{
    auto [inputNode, inputPort] = transposeNode.GetInDataNodesAndPortIndexs(0);
    FUSION_PASS_CHECK(inputNode == nullptr, OPS_LOG_E(kPassName, "Transpose peer out anchor is null."), return false);

    FUSION_PASS_CHECK(graph->RemoveEdge(transposeNode, 0, matmulNode, dataIndex) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "RemoveEdge transpose-->matmul failed."), return false);
    FUSION_PASS_CHECK(graph->AddDataEdge(*inputNode, inputPort, matmulNode, dataIndex) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "AddEdge peer-->matmul failed."), return false);
    return true;
}

// 对 transpose 的所有 MatMul 消费者执行属性翻转和边重连。
// needRemoveTranspose 默认 true，以下情况置 false（不移除 transpose 节点）：
// 1. 消费者为 null 或类型获取失败
// 2. 消费者不全是 MatMul 类型，或输入端口不匹配（只要有一个不满足即跳过该消费者）
// 这些情况下 transpose 被其他非 MatMul 节点使用，不能移除，
// 但已匹配的 MatMul 消费者仍会执行属性翻转和边重连。
bool DoTransposeFusion(const GraphPtr& graph, GNode& transposeNode, int32_t dataIndex, const char* attrPrefix,
                       bool& needRemoveTranspose)
{
    auto outConsumers = transposeNode.GetOutDataNodesAndPortIndexs(0);
    needRemoveTranspose = true;

    std::vector<GNodePtr> matmulConsumers;
    for (auto& [consumer, inputPort] : outConsumers) {
        if (consumer == nullptr) {
            OPS_LOG_W(kPassName, "Consumer node is null, skip.");
            needRemoveTranspose = false;
            continue;
        }
        AscendString consumerType;
        if (consumer->GetType(consumerType) != GRAPH_SUCCESS) {
            needRemoveTranspose = false;
            continue;
        }
        bool isMatMul = consumerType == kOpTypeMatMul || consumerType == kOpTypeMatMulV2 ||
                        consumerType == kOpTypeBatchMatMul || consumerType == kOpTypeBatchMatMulV2;
        if (!isMatMul || inputPort != dataIndex) {
            needRemoveTranspose = false;
            continue;
        }
        matmulConsumers.emplace_back(consumer);
    }

    for (auto& matmulNode : matmulConsumers) {
        FUSION_PASS_CHECK(!ToggleTransposeAttr(*matmulNode, dataIndex, attrPrefix),
                          OPS_LOG_E(kPassName, "Toggle transpose attr failed for index %d.", dataIndex), return false);

        TensorDesc inputDesc;
        FUSION_PASS_CHECK(matmulNode->GetInputDesc(dataIndex, inputDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Get input desc failed for index %d.", dataIndex), return false);
        FUSION_PASS_CHECK(!SwapLastTwoDims(inputDesc),
                          OPS_LOG_E(kPassName, "Swap last two dims failed for index %d.", dataIndex), return false);
        FUSION_PASS_CHECK(matmulNode->UpdateInputDesc(dataIndex, inputDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Update input desc failed for index %d.", dataIndex), return false);

        FUSION_PASS_CHECK(!RelinkEdge(graph, transposeNode, *matmulNode, dataIndex),
                          OPS_LOG_E(kPassName, "Relink edge failed for index %d.", dataIndex), return false);
    }

    return true;
}

bool RemoveNodeFully(const GraphPtr& graph, const GNodePtr& node)
{
    if (node == nullptr) {
        return true;
    }

    size_t inputSize = node->GetInputsSize();
    for (size_t i = 0; i < inputSize; ++i) {
        auto [srcNode, srcPort] = node->GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (srcNode != nullptr) {
            FUSION_PASS_CHECK(graph->RemoveEdge(*srcNode, srcPort, *node, static_cast<int32_t>(i)) != GRAPH_SUCCESS,
                              OPS_LOG_E(kPassName, "Failed to remove input edge at index %zu.", i), return false);
        }
    }

    FUSION_PASS_CHECK(graph->RemoveNode(*node) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Failed to remove node."),
                      return false);
    return true;
}

void ReportTransposeFusion(const std::vector<GNode>& nodesBeforeFuse, const GNode& fusedNode,
                           CustomPassContext& passContext)
{
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse == nullptr) {
        return;
    }
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, {fusedNode}, passContext) != SUCCESS) {
        OPS_LOG_W(kPassName, "Failed to report fusion result.");
    }
}

Status ProcessNode(const GraphPtr& graph, GNode& matmulNode, CustomPassContext& passContext,
                   const PlatformInfo& platformInfo)
{
    TensorDesc inputDesc1;
    if (matmulNode.GetInputDesc(1, inputDesc1) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName, "The input desc is null.");
        return GRAPH_NOT_CHANGED;
    }

    DataType dataDtype = inputDesc1.GetDataType();
    if (dataDtype != DT_FLOAT16 && dataDtype != DT_FLOAT && dataDtype != DT_BF16) {
        OPS_LOG_W(kPassName, "Transpose MatMul Fusion only supports float16 and float32 and bfloat16.");
        return GRAPH_NOT_CHANGED;
    }

    AscendString opType;
    FUSION_PASS_CHECK(matmulNode.GetType(opType) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Get matmul node type failed."),
                      return GRAPH_NOT_CHANGED);
    // BatchMatMul/BatchMatMulV2 用 "adj_x" 前缀，MatMul/MatMulV2 用 "transpose_x" 前缀
    const char* attrPrefix = IsBatchOp(opType) ? "adj_x" : "transpose_x";

    std::vector<GNode> nodesBeforeFuse = {matmulNode};
    std::vector<GNodePtr> nodesToRemove;
    bool isFusionTranspose = false;

    for (int32_t inputIndex = 0; inputIndex <= 1; inputIndex++) {
        auto transposeNode = GetInputNode(matmulNode, inputIndex);
        if (transposeNode == nullptr) {
            OPS_LOG_W(kPassName, "Transpose node of input %d is null, skip.", inputIndex);
            continue;
        }

        if (!CheckTransposeFusion(*transposeNode, inputIndex, dataDtype)) {
            continue;
        }

        OPS_LOG_D(kPassName, "Start to do transpose fusion for input %d.", inputIndex);
        bool needRemoveTranspose = false;
        if (!DoTransposeFusion(graph, *transposeNode, inputIndex, attrPrefix, needRemoveTranspose)) {
            return GRAPH_FAILED;
        }

        if (needRemoveTranspose) {
            nodesBeforeFuse.push_back(*transposeNode);
            nodesToRemove.push_back(transposeNode);
        }
        isFusionTranspose = true;
    }

    if (!isFusionTranspose) {
        OPS_LOG_W(kPassName, "No need to do transpose fusion.");
        return GRAPH_NOT_CHANGED;
    }

    // 必须在删除旧节点之前上报融合结果，因为 ReportFuse 要求 nodesBeforeFuse 中的节点仍属于当前图
    ReportTransposeFusion(nodesBeforeFuse, matmulNode, passContext);

    for (auto& node : nodesToRemove) {
        if (!RemoveNodeFully(graph, node)) {
            OPS_LOG_W(kPassName, "Failed to remove transpose node.");
        }
    }

    OPS_LOG_I(kPassName, "Succeeded in executing fusion pass.");
    return SUCCESS;
}

} // namespace

Status BatchMatMulTransposeFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    OPS_LOG_D(kPassName, "Enter BatchMatMulTransposeFusionPass.");
    if (graph == nullptr || !graph->IsValid()) {
        OPS_LOG_W(kPassName, "Graph is null or invalid, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    if (!IsTargetVersion()) {
        return GRAPH_NOT_CHANGED;
    }

    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OPS_LOG_W(kPassName, "Can't get platformInfo.");
        return GRAPH_NOT_CHANGED;
    }
    if (!IsSupportL12BtBf16(platformInfo)) {
        OPS_LOG_W(kPassName, "Not supported on this platform.");
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> targetNodes;
    for (auto& node : graph->GetDirectNode()) {
        if (IsMatMulType(node)) {
            targetNodes.emplace_back(node);
        }
    }
    if (targetNodes.empty()) {
        return GRAPH_NOT_CHANGED;
    }

    passContext.SetPassName(kPassName);
    bool changed = false;
    for (auto& node : targetNodes) {
        auto status = ProcessNode(graph, node, passContext, platformInfo);
        if (status == SUCCESS) {
            changed = true;
        } else if (status != GRAPH_NOT_CHANGED) {
            return status;
        }
    }

    OPS_LOG_D(kPassName, "Leave BatchMatMulTransposeFusionPass.");
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}

// 满足目标版本时用 kCompatibleInherited（InferShape 前执行，与旧框架一致），
// 不满足时降级到 kAfterInferShape，保证旧版本 CANN 兼容性。
#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(BatchMatMulTransposeFusionPass)
    .Stage(IsTargetVersion() ? ge::CustomPassStage::kCompatibleInherited : ge::CustomPassStage::kAfterInferShape);
#endif

} // namespace ops
