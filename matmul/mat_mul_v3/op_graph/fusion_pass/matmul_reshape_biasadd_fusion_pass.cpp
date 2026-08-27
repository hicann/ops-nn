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
 * \file matmul_reshape_biasadd_fusion_pass.cpp
 * \brief matmul reshape biasadd fusion pass (matmul/matmulv2 --> reshape --> biasadd/add)
 *
 * 融合规则：将 MatMul/MatMulV2 → Reshape → BiasAdd/Add 融合为
 *           带 has_bias=true 属性的 MatMul/MatMulV2（bias 作为第三个输入），
 *           并移除 BiasAdd/Add 节点，Reshape 的输出直连下游节点。
 *
 *     x1   x2      bias               x1   x2  bias
 *      \  /        /                    \  |  /
 *     MatMul      /                   MatMul(has_bias=true)
 *        \       /                        |
 *     Reshape   /             ====>       |
 *        \     /                          |
 *     BiasAdd/Add                         |
 *          |                              |
 *        output                         output
 */

#include "matmul_reshape_biasadd_fusion_pass.h"

#include <cstdint>
#include <vector>

#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "version/ge-compiler_version.h"
#include "acl/acl_rt.h"

using namespace ge;
using namespace fe;

namespace ops {
namespace {

constexpr char kPassName[] = "MatMulReshapeBiasAddFusionPass";
constexpr int64_t k2D = 2;
constexpr char kOpTypeBiasAdd[] = "BiasAdd";
constexpr char kOpTypeAdd[] = "Add";
constexpr char kOpTypeReshape[] = "Reshape";
constexpr char kAttrHasBias[] = "has_bias";
constexpr int32_t kMinGeCompilerVersion = 90100000;

bool IsTargetVersion()
{
    int32_t version = 0;
    FUSION_PASS_CHECK(aclsysGetVersionNum("ge-compiler", &version) != ACL_SUCCESS,
                      OPS_LOG_W(kPassName, "Failed to get ge-compiler version, skip fusion."), return false);
    return version >= kMinGeCompilerVersion;
}

bool IsType(const GNode& node, const char* type)
{
    AscendString opType;
    return node.GetType(opType) == GRAPH_SUCCESS && opType == type;
}

bool IsType(const GNodePtr& nodePtr, const char* type)
{
    if (nodePtr == nullptr) {
        return false;
    }
    AscendString opType;
    return nodePtr->GetType(opType) == GRAPH_SUCCESS && opType == type;
}

bool IsMatMulOrMatMulV2Type(const GNodePtr& nodePtr)
{
    return IsType(nodePtr, kOpTypeMatMul) || IsType(nodePtr, kOpTypeMatMulV2);
}

// 解析 Add/BiasAdd 节点的输入端口，确定哪个端口接 Reshape、哪个接 bias。
// BiasAdd 端口固定（input0=feature, input1=bias），无需交换；
// Add 两个输入对等，需运行时判断：若 input0 不是 Reshape 则交换端口索引。
bool ResolveFusionPorts(const GNode& addOpNode, int32_t& reshapeInputIdx, int32_t& biasInputIdx)
{
    AscendString addType;
    if (addOpNode.GetType(addType) != GRAPH_SUCCESS) {
        return false;
    }
    reshapeInputIdx = 0;
    biasInputIdx = 1;
    if (addType == kOpTypeAdd) {
        auto in0OpNodePtr = addOpNode.GetInDataNodesAndPortIndexs(0).first;
        if (!IsType(in0OpNodePtr, kOpTypeReshape)) {
            reshapeInputIdx = 1;
            biasInputIdx = 0;
        }
    }
    return true;
}

bool CalcBiasLength(const std::vector<int64_t>& biasShapeVec, int64_t& biasLength)
{
    if (biasShapeVec.empty()) {
        OPS_LOG_W(kPassName, "Bias shape is empty.");
        return false;
    }
    biasLength = 1;
    for (auto dim : biasShapeVec) {
        FUSION_PASS_CHECK(dim <= 0, OPS_LOG_W(kPassName, "Bias output shape is invalid."), return false);
        FUSION_PASS_CHECK(biasLength > INT64_MAX / dim, OPS_LOG_W(kPassName, "Bias output shape size exceeds int64."),
                          return false);
        biasLength *= dim;
    }
    return true;
}

// Add 分支的 bias shape 校验：bias 可能是多维，从最后一个维度向前逐维与 reshape 输出比对，
// 确保每维都相等（不支持 broadcast），同时 bias 总长度必须等于 MatMul 输出的 n 维。
bool ValidateAddBiasShape(const GNode& biasDataNode, const Shape& reshapeOutShape, const Shape& matmulOutShape)
{
    TensorDesc biasOutDesc;
    FUSION_PASS_CHECK(biasDataNode.GetOutputDesc(0, biasOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get bias output desc failed."), return false);
    auto biasShapeVec = biasOutDesc.GetShape().GetDims();
    auto reshapeShapeVec = reshapeOutShape.GetDims();

    FUSION_PASS_CHECK(reshapeShapeVec.size() < biasShapeVec.size() || biasShapeVec.empty(),
                      OPS_LOG_W(kPassName, "Reshape output dim num should be larger than bias dim num, "
                                           "and bias dim num cannot be 0."),
                      return false);

    int64_t biasLength = 1;
    FUSION_PASS_CHECK(!CalcBiasLength(biasShapeVec, biasLength), OPS_LOG_W(kPassName, "Calc bias length failed."),
                      return false);
    FUSION_PASS_CHECK(biasLength != matmulOutShape.GetDim(1),
                      OPS_LOG_W(kPassName, "Bias dim is not equal to matmul output n dim."), return false);
    for (size_t i = 0; i < biasShapeVec.size(); i++) {
        FUSION_PASS_CHECK(biasShapeVec[biasShapeVec.size() - i - 1] != reshapeShapeVec[reshapeShapeVec.size() - i - 1],
                          OPS_LOG_W(kPassName, "Bias shape cannot be broadcast to reshape output shape."),
                          return false);
    }
    return true;
}

// BiasAdd 分支的 bias shape 校验：bias 通常为 1D，只需校验总长度等于 reshape 输出最后一维。
bool ValidateBiasAddBiasShape(const GNode& biasDataNode, const Shape& reshapeOutShape)
{
    TensorDesc biasOutDesc;
    FUSION_PASS_CHECK(biasDataNode.GetOutputDesc(0, biasOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get bias output desc failed."), return false);
    auto biasShapeVec = biasOutDesc.GetShape().GetDims();

    int64_t biasLength = 1;
    FUSION_PASS_CHECK(!CalcBiasLength(biasShapeVec, biasLength), OPS_LOG_W(kPassName, "Calc bias length failed."),
                      return false);
    FUSION_PASS_CHECK(biasLength != reshapeOutShape.GetDim(reshapeOutShape.GetDimNum() - 1),
                      OPS_LOG_W(kPassName, "Bias dim is not equal to matmul output n dim."), return false);
    return true;
}

// 校验 MatMul 节点是否满足融合条件：
// - 输出只能有 1 个消费者（Reshape），输入只能有 2 个（无 bias），输出必须为 2D
// - 不支持动态 shape（需读取 dim(1) 的具体值校验 bias 长度）
// - 不支持空 tensor（dim==0 时 bias 加法无意义）
bool ValidateMatMulNode(const GNode& matmulOpNode, Shape& matmulOutShape)
{
    auto matmulOutPairs = matmulOpNode.GetOutDataNodesAndPortIndexs(0);
    FUSION_PASS_CHECK(matmulOutPairs.size() != 1,
                      OPS_LOG_W(kPassName, "MatMul node should only have 1 output, actual %zu.", matmulOutPairs.size()),
                      return false);

    FUSION_PASS_CHECK(
        matmulOpNode.GetInputsSize() != static_cast<size_t>(kBaseNodeNum),
        OPS_LOG_W(kPassName, "MatMul node should have 2 inputs, actual %zu.", matmulOpNode.GetInputsSize()),
        return false);

    TensorDesc matmulOutDesc;
    FUSION_PASS_CHECK(matmulOpNode.GetOutputDesc(0, matmulOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul output desc failed."), return false);
    matmulOutShape = matmulOutDesc.GetShape();
    FUSION_PASS_CHECK(
        matmulOutShape.GetDimNum() != static_cast<size_t>(k2D),
        OPS_LOG_W(kPassName, "The output dim of matmul node must be 2, actual %zu.", matmulOutShape.GetDimNum()),
        return false);
    FUSION_PASS_CHECK(matmulOutShape.GetDim(0) < 0 || matmulOutShape.GetDim(1) < 0,
                      OPS_LOG_W(kPassName, "MatmulReshapeBiasAddFusion cannot be applied to unknown shape."),
                      return false);
    FUSION_PASS_CHECK(matmulOutShape.GetDim(0) == 0 || matmulOutShape.GetDim(1) == 0,
                      OPS_LOG_W(kPassName, "MatmulReshapeBiasAddFusion cannot be applied to empty tensor."),
                      return false);
    return true;
}

// 校验 Reshape 节点：输出最后一维必须等于 MatMul 输出的 n 维
bool ValidateReshapeNode(const GNode& reshapeOpNode, const Shape& matmulOutShape, Shape& reshapeOutShape,
                         TensorDesc& reshapeOutDesc)
{
    auto reshapeOutPairs = reshapeOpNode.GetOutDataNodesAndPortIndexs(0);
    FUSION_PASS_CHECK(reshapeOutPairs.size() != 1,
                      OPS_LOG_W(kPassName, "Fusion of Reshape with over 1 output is not supported."), return false);

    FUSION_PASS_CHECK(reshapeOpNode.GetOutputDesc(0, reshapeOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get reshape output desc failed."), return false);
    reshapeOutShape = reshapeOutDesc.GetShape();
    FUSION_PASS_CHECK(reshapeOutShape.GetDimNum() < 1,
                      OPS_LOG_W(kPassName, "The dim num of reshape output shape must be at least 1."), return false);
    FUSION_PASS_CHECK(matmulOutShape.GetDim(1) != reshapeOutShape.GetDim(reshapeOutShape.GetDimNum() - 1),
                      OPS_LOG_W(kPassName, "MatmulReshapeBiasAddFusion is only supported when the last dim of matmul "
                                           "output is not split during fusion with bias_add node."),
                      return false);
    return true;
}

bool ValidateAddOpNode(const GNode& addOpNode, const GNode& biasDataNode, const TensorDesc& reshapeOutDesc)
{
    FUSION_PASS_CHECK(
        addOpNode.GetOutputsSize() != 1,
        OPS_LOG_W(kPassName, "BiasAdd node should only have 1 output, actual %zu.", addOpNode.GetOutputsSize()),
        return false);

    FUSION_PASS_CHECK(biasDataNode.GetOutputsSize() != 1,
                      OPS_LOG_W(kPassName, "Fusion of Bias with over 1 output is not supported."), return false);

    TensorDesc addOutDesc;
    FUSION_PASS_CHECK(addOpNode.GetOutputDesc(0, addOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get biasadd output desc failed."), return false);
    FUSION_PASS_CHECK(reshapeOutDesc.GetDataType() != addOutDesc.GetDataType(),
                      OPS_LOG_W(kPassName, "The output dtype of reshape differs from that of biasadd."), return false);
    return true;
}

bool ValidateBiasShape(const GNode& addOpNode, const GNode& biasDataNode, const Shape& reshapeOutShape,
                       const Shape& matmulOutShape)
{
    AscendString addType;
    FUSION_PASS_CHECK(addOpNode.GetType(addType) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Get add op type failed."),
                      return false);
    if (addType == kOpTypeAdd) {
        return ValidateAddBiasShape(biasDataNode, reshapeOutShape, matmulOutShape);
    }
    return ValidateBiasAddBiasShape(biasDataNode, reshapeOutShape);
}

bool ValidateFusionPreconditions(const GNode& matmulOpNode, const GNode& reshapeOpNode, const GNode& addOpNode,
                                 const GNode& biasDataNode)
{
    Shape matmulOutShape;
    FUSION_PASS_CHECK(!ValidateMatMulNode(matmulOpNode, matmulOutShape),
                      OPS_LOG_W(kPassName, "Validate matmul node failed."), return false);

    Shape reshapeOutShape;
    TensorDesc reshapeOutDesc;
    FUSION_PASS_CHECK(!ValidateReshapeNode(reshapeOpNode, matmulOutShape, reshapeOutShape, reshapeOutDesc),
                      OPS_LOG_W(kPassName, "Validate reshape node failed."), return false);

    FUSION_PASS_CHECK(!ValidateAddOpNode(addOpNode, biasDataNode, reshapeOutDesc),
                      OPS_LOG_W(kPassName, "Validate add op node failed."), return false);

    FUSION_PASS_CHECK(!ValidateBiasShape(addOpNode, biasDataNode, reshapeOutShape, matmulOutShape),
                      OPS_LOG_W(kPassName, "Validate bias shape failed."), return false);
    return true;
}

Status PrepareFusion(const GNode& addOpNode, int32_t& reshapeInputIdx, int32_t& biasInputIdx,
                     GNodePtr& reshapeOpNodePtr, GNodePtr& biasDataNodePtr, GNodePtr& matmulOpNodePtr)
{
    FUSION_PASS_CHECK(!ResolveFusionPorts(addOpNode, reshapeInputIdx, biasInputIdx),
                      OPS_LOG_W(kPassName, "Resolve fusion ports failed, skip fusion."), return GRAPH_NOT_CHANGED);

    reshapeOpNodePtr = addOpNode.GetInDataNodesAndPortIndexs(reshapeInputIdx).first;
    biasDataNodePtr = addOpNode.GetInDataNodesAndPortIndexs(biasInputIdx).first;
    FUSION_PASS_CHECK(reshapeOpNodePtr == nullptr || biasDataNodePtr == nullptr,
                      OPS_LOG_W(kPassName, "Reshape or bias input node is null."), return GRAPH_NOT_CHANGED);
    FUSION_PASS_CHECK(!IsType(reshapeOpNodePtr, kOpTypeReshape),
                      OPS_LOG_W(kPassName, "Input node is not reshape type, skip fusion."), return GRAPH_NOT_CHANGED);

    auto [matmulInPtr, matmulInPort] = reshapeOpNodePtr->GetInDataNodesAndPortIndexs(0);
    matmulOpNodePtr = matmulInPtr;
    FUSION_PASS_CHECK(matmulOpNodePtr == nullptr, OPS_LOG_W(kPassName, "Matmul input node of reshape is null."),
                      return GRAPH_NOT_CHANGED);
    FUSION_PASS_CHECK(!IsMatMulOrMatMulV2Type(matmulOpNodePtr),
                      OPS_LOG_W(kPassName, "Reshape input is not matmul type, skip fusion."), return GRAPH_NOT_CHANGED);

    FUSION_PASS_CHECK(!ValidateFusionPreconditions(*matmulOpNodePtr, *reshapeOpNodePtr, addOpNode, *biasDataNodePtr),
                      OPS_LOG_W(kPassName, "Validate fusion preconditions failed, skip fusion."),
                      return GRAPH_NOT_CHANGED);
    return SUCCESS;
}

Status AddBiasToMatMulNode(const GraphPtr& graph, GNode& matmulOpNode, GNode& biasDataNode)
{
    TensorDesc biasDesc;
    FUSION_PASS_CHECK(biasDataNode.GetOutputDesc(0, biasDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get bias output desc failed."), return GRAPH_FAILED);

    FUSION_PASS_CHECK(
        graph->AddDataEdge(biasDataNode, 0, matmulOpNode, static_cast<int32_t>(kBiasInputIdx)) != GRAPH_SUCCESS,
        OPS_LOG_W(kPassName, "AddDataEdge for bias to matmul port %d failed, skip fusion.",
                  static_cast<int32_t>(kBiasInputIdx)),
        return GRAPH_NOT_CHANGED);
    FUSION_PASS_CHECK(
        matmulOpNode.UpdateInputDesc(static_cast<int32_t>(kBiasInputIdx), biasDesc) != GRAPH_SUCCESS,
        OPS_LOG_E(kPassName, "UpdateInputDesc for bias port %d failed.", static_cast<int32_t>(kBiasInputIdx)),
        return GRAPH_FAILED);

    bool hasBias = true;
    FUSION_PASS_CHECK(matmulOpNode.SetAttr(kAttrHasBias, hasBias) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "SetAttr has_bias failed."), return GRAPH_FAILED);

    AscendString matmulName;
    FUSION_PASS_CHECK(matmulOpNode.GetName(matmulName) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul node name failed."), return GRAPH_FAILED);
    OPS_LOG_I(kPassName, "Added bias to matmul node, name=%s.", matmulName.GetString());
    return SUCCESS;
}

Status UpdateReshapeOutputFormat(const GNode& addOpNode, GNode& reshapeOpNode)
{
    TensorDesc addOutDesc;
    TensorDesc reshapeOutDesc;
    FUSION_PASS_CHECK(addOpNode.GetOutputDesc(0, addOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get add output desc failed."), return GRAPH_FAILED);
    FUSION_PASS_CHECK(reshapeOpNode.GetOutputDesc(0, reshapeOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get reshape output desc failed."), return GRAPH_FAILED);

    auto addOutFormat = addOutDesc.GetFormat();
    auto reshapeOutFormat = reshapeOutDesc.GetFormat();
    if (addOutFormat != reshapeOutFormat) {
        OPS_LOG_D(kPassName, "The output format of biasadd and reshape is different.");
        reshapeOutDesc.SetFormat(addOutFormat);
        FUSION_PASS_CHECK(reshapeOpNode.UpdateOutputDesc(0, reshapeOutDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Update reshape output desc failed."), return GRAPH_FAILED);
    }
    return SUCCESS;
}

Status RelinkOutputEdges(const GraphPtr& graph, GNode& addOpNode, GNode& reshapeOpNode)
{
    auto outPairs = addOpNode.GetOutDataNodesAndPortIndexs(0);
    for (auto& [dstOpNodePtr, dstPort] : outPairs) {
        if (dstOpNodePtr == nullptr) {
            OPS_LOG_W(kPassName, "Output consumer node is null, skip.");
            continue;
        }
        GNode dstOpNode = *dstOpNodePtr;
        FUSION_PASS_CHECK(graph->RemoveEdge(addOpNode, 0, dstOpNode, dstPort) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Remove edge addOpNode-->dst failed."), return GRAPH_FAILED);
        FUSION_PASS_CHECK(graph->AddDataEdge(reshapeOpNode, 0, dstOpNode, dstPort) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Add edge reshapeOpNode-->dst failed."), return GRAPH_FAILED);
    }
    return SUCCESS;
}

// 将 Add/BiasAdd 节点的入控制边迁移到 MatMul（src→MatMul），
// 出控制边迁移到 MatMul（MatMul→dst），保证融合后控制依赖关系不变。
Status TransferCtrlEdges(const GraphPtr& graph, const GNode& addOpNode, GNode& matmulOpNode)
{
    for (auto& srcNodePtr : addOpNode.GetInControlNodes()) {
        if (srcNodePtr != nullptr) {
            FUSION_PASS_CHECK(graph->AddControlEdge(*srcNodePtr, matmulOpNode) != GRAPH_SUCCESS,
                              OPS_LOG_E(kPassName, "Add control edge src-->matmul failed."), return GRAPH_FAILED);
        }
    }
    for (auto& dstNodePtr : addOpNode.GetOutControlNodes()) {
        if (dstNodePtr != nullptr) {
            FUSION_PASS_CHECK(graph->AddControlEdge(matmulOpNode, *dstNodePtr) != GRAPH_SUCCESS,
                              OPS_LOG_E(kPassName, "Add control edge matmul-->dst failed."), return GRAPH_FAILED);
        }
    }
    return SUCCESS;
}

Status RemoveFusedNodes(const GraphPtr& graph, GNode& addOpNode, GNode& reshapeOpNode, GNode& biasDataNode,
                        int32_t reshapeInputIdx, int32_t biasInputIdx)
{
    FUSION_PASS_CHECK(graph->RemoveEdge(reshapeOpNode, 0, addOpNode, reshapeInputIdx) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Remove edge reshapeOpNode-->addOpNode failed."), return GRAPH_FAILED);
    FUSION_PASS_CHECK(graph->RemoveEdge(biasDataNode, 0, addOpNode, biasInputIdx) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Remove edge biasDataNode-->addOpNode failed."), return GRAPH_FAILED);
    FUSION_PASS_CHECK(graph->RemoveNode(addOpNode) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Remove addOpNode failed."),
                      return GRAPH_FAILED);
    return SUCCESS;
}

Status CommitFusion(const GraphPtr& graph, GNode& addOpNode, int32_t reshapeInputIdx, int32_t biasInputIdx,
                    CustomPassContext& passContext)
{
    auto reshapeOpNodePtr = addOpNode.GetInDataNodesAndPortIndexs(reshapeInputIdx).first;
    auto biasDataNodePtr = addOpNode.GetInDataNodesAndPortIndexs(biasInputIdx).first;
    FUSION_PASS_CHECK(reshapeOpNodePtr == nullptr || biasDataNodePtr == nullptr,
                      OPS_LOG_E(kPassName, "Reshape or bias node is null."), return GRAPH_FAILED);

    GNode reshapeOpNode = *reshapeOpNodePtr;
    GNode biasDataNode = *biasDataNodePtr;

    auto [matmulOpNodePtr, matmulPort] = reshapeOpNode.GetInDataNodesAndPortIndexs(0);
    FUSION_PASS_CHECK(matmulOpNodePtr == nullptr, OPS_LOG_E(kPassName, "Matmul node is null."), return GRAPH_FAILED);

    GNode matmulOpNode = *matmulOpNodePtr;

    AscendString matmulName;
    FUSION_PASS_CHECK(matmulOpNode.GetName(matmulName) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul node name failed."), return GRAPH_FAILED);

    auto status = AddBiasToMatMulNode(graph, matmulOpNode, biasDataNode);
    FUSION_PASS_CHECK(status != SUCCESS, OPS_LOG_W(kPassName, "AddBiasToMatMulNode failed, skip fusion."),
                      return GRAPH_NOT_CHANGED);

    FUSION_PASS_CHECK(RelinkOutputEdges(graph, addOpNode, reshapeOpNode) != SUCCESS,
                      OPS_LOG_E(kPassName, "RelinkOutputEdges failed."), return GRAPH_FAILED);

    FUSION_PASS_CHECK(UpdateReshapeOutputFormat(addOpNode, reshapeOpNode) != SUCCESS,
                      OPS_LOG_E(kPassName, "UpdateReshapeOutputFormat failed."), return GRAPH_FAILED);

    FUSION_PASS_CHECK(TransferCtrlEdges(graph, addOpNode, matmulOpNode) != SUCCESS,
                      OPS_LOG_E(kPassName, "TransferCtrlEdges failed."), return GRAPH_FAILED);

    std::vector<GNode> nodesBeforeFuse = {matmulOpNode, addOpNode};
    // 必须在删除旧节点之前上报融合结果，因为 ReportFuse 要求 nodesBeforeFuse 中的节点仍属于当前图，
    // 删除后再调用会报 "nodes belong to different graphs" 错误。
    ReportFusion(nodesBeforeFuse, {matmulOpNode}, passContext, kPassName);

    FUSION_PASS_CHECK(
        RemoveFusedNodes(graph, addOpNode, reshapeOpNode, biasDataNode, reshapeInputIdx, biasInputIdx) != SUCCESS,
        OPS_LOG_E(kPassName, "RemoveFusedNodes failed."), return GRAPH_FAILED);

    OPS_LOG_I(kPassName, "Matmul reshape biasadd fusion success! matmul=%s.", matmulName.GetString());
    return SUCCESS;
}

Status FuseOneReshapeBiasAddNode(const GraphPtr& graph, GNode& addOpNode, CustomPassContext& passContext)
{
    int32_t reshapeInputIdx = 0;
    int32_t biasInputIdx = 0;
    GNodePtr reshapeOpNodePtr = nullptr;
    GNodePtr biasDataNodePtr = nullptr;
    GNodePtr matmulOpNodePtr = nullptr;

    auto status = PrepareFusion(addOpNode, reshapeInputIdx, biasInputIdx, reshapeOpNodePtr, biasDataNodePtr,
                                matmulOpNodePtr);
    if (status != SUCCESS) {
        return status;
    }
    return CommitFusion(graph, addOpNode, reshapeInputIdx, biasInputIdx, passContext);
}

} // namespace

Status MatMulReshapeBiasAddFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    OPS_LOG_D(kPassName, "Begin to do MatMulReshapeBiasAddFusionPass Run.");
    FUSION_PASS_CHECK(graph == nullptr || !graph->IsValid(),
                      OPS_LOG_W(kPassName, "Graph is null or invalid, skip fusion pass."), return GRAPH_NOT_CHANGED);

    FUSION_PASS_CHECK(!IsTargetVersion(), OPS_LOG_W(kPassName, "Not target version, skip fusion pass."),
                      return GRAPH_NOT_CHANGED);

    passContext.SetPassName(kPassName);

    std::vector<GNode> addOpNodes;
    for (auto& node : graph->GetDirectNode()) {
        if (IsType(node, kOpTypeBiasAdd) || IsType(node, kOpTypeAdd)) {
            addOpNodes.emplace_back(node);
        }
    }
    FUSION_PASS_CHECK(addOpNodes.empty(), OPS_LOG_W(kPassName, "No BiasAdd/Add node, skip fusion pass."),
                      return GRAPH_NOT_CHANGED);

    bool changed = false;
    for (auto& addOpNode : addOpNodes) {
        auto status = FuseOneReshapeBiasAddNode(graph, addOpNode, passContext);
        if (status == SUCCESS) {
            changed = true;
            continue;
        }
        FUSION_PASS_CHECK(status != GRAPH_NOT_CHANGED, OPS_LOG_E(kPassName, "Fuse node failed, status=%d.", status),
                          return status);
    }

    OPS_LOG_D(kPassName, "Exit MatMulReshapeBiasAddFusionPass.");
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}

// 满足目标版本时用 kCompatibleInherited（InferShape 前执行，与旧框架一致），
// 不满足时降级到 kAfterInferShape，保证旧版本 CANN 兼容性。
#if GE_COMPILER_VERSION_NUM >= 90100000 // mirrors kMinGeCompilerVersion
REG_FUSION_PASS(MatMulReshapeBiasAddFusionPass)
    .Stage(IsTargetVersion() ? CustomPassStage::kCompatibleInherited : CustomPassStage::kAfterInferShape);
#endif

} // namespace ops
