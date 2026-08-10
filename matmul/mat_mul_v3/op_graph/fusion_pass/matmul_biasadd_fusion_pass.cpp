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
 * \file matmul_biasadd_fusion_pass.cpp
 * \brief matmul biasadd fusion pass (matmul/batchmatmul + biasadd/add --> matmul/batchmatmul with has_bias)
 *
 * 融合规则：将 MatMul/MatMulV2/BatchMatMul/BatchMatMulV2 + BiasAdd/Add 融合为
 *           带 has_bias=true 属性的 MatMul/MatMulV2/BatchMatMul/BatchMatMulV2（bias 作为第三个输入）。
 *
 *     x1   x2                 x1   x2  bias
 *      \  /                     \  |  /
 *    MatMul   bias              MatMul(has_bias=true)
 *        \   /                      |
 *      BiasAdd/Add   ====>          |
 *          |                        |
 *        output                   output
 */

#include "matmul_biasadd_fusion_pass.h"

#include <cstdint>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "ge/compliant_node_builder.h"
#include "version/ge-compiler_version.h"
#include "acl/acl_rt.h"

using namespace ge;
using namespace fe;

namespace ops {
namespace {

constexpr char kPassName[] = "MatMulBiasAddFusionPass";
constexpr int64_t k2D = 2;
constexpr char kOpTypeBiasAdd[] = "BiasAdd";
constexpr char kOpTypeAdd[] = "Add";
constexpr char kAttrHasBias[] = "has_bias";

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

bool IsBatchMatMulType(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS) {
        return false;
    }
    return opType == kOpTypeBatchMatMul || opType == kOpTypeBatchMatMulV2;
}

bool IsBiasAddOrAddType(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS) {
        return false;
    }
    return opType == kOpTypeBiasAdd || opType == kOpTypeAdd;
}

bool IsType(const GNodePtr& nodePtr, const char* type)
{
    if (nodePtr == nullptr) {
        return false;
    }
    AscendString opType;
    return nodePtr->GetType(opType) == GRAPH_SUCCESS && opType == type;
}

bool IsMatMulType(const GNodePtr& nodePtr)
{
    return IsType(nodePtr, kOpTypeMatMul) || IsType(nodePtr, kOpTypeMatMulV2) || IsType(nodePtr, kOpTypeBatchMatMul) ||
           IsType(nodePtr, kOpTypeBatchMatMulV2);
}

bool IsBatchOpType(const AscendString& opTypeStr)
{
    return opTypeStr == kOpTypeBatchMatMul || opTypeStr == kOpTypeBatchMatMulV2;
}

bool HasOffsetWInput(const AscendString& opTypeStr)
{
    return opTypeStr == kOpTypeMatMulV2 || opTypeStr == kOpTypeBatchMatMulV2;
}

bool CheckAddDtype(const GNode& matmulOpNode, const GNode& addOpNode)
{
    TensorDesc input0Desc;
    TensorDesc input1Desc;
    FUSION_PASS_CHECK(addOpNode.GetInputDesc(0, input0Desc) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName, "inputDesc0 is null"), return false);
    FUSION_PASS_CHECK(addOpNode.GetInputDesc(1, input1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName, "inputDesc1 is null"), return false);

    auto firstShape = input0Desc.GetShape().GetDims();
    auto secondShape = input1Desc.GetShape().GetDims();
    if (firstShape.size() != 1 && secondShape.size() != 1) {
        OPS_LOG_I(kPassName, "Added input is not equaled to 1");
        return false;
    }

    int64_t biasDim = 0;
    int64_t inputNdim = 0;
    bool isBatch = IsBatchMatMulType(matmulOpNode);
    if (secondShape.size() == 1) {
        if (!isBatch) {
            if (firstShape.size() != static_cast<size_t>(k2D)) {
                OPS_LOG_I(kPassName, "Matmul output shape no match.");
                return false;
            }
        } else if (firstShape.empty()) {
            OPS_LOG_I(kPassName, "Matmul output shape is empty, skip fusion.");
            return false;
        }
        biasDim = secondShape[0];
        inputNdim = firstShape[firstShape.size() - 1];
    } else {
        if (!isBatch) {
            if (secondShape.size() != static_cast<size_t>(k2D)) {
                OPS_LOG_I(kPassName, "Matmul output shape no match.");
                return false;
            }
        } else if (secondShape.empty()) {
            OPS_LOG_I(kPassName, "Matmul output shape is empty, skip fusion.");
            return false;
        }
        biasDim = firstShape[0];
        inputNdim = secondShape[secondShape.size() - 1];
    }

    if (biasDim > 0 && inputNdim > 0 && biasDim != inputNdim) {
        OPS_LOG_I(kPassName, "bias shape %lld, is not equal to input second dim %lld.", static_cast<long long>(biasDim),
                  static_cast<long long>(inputNdim));
        return false;
    }
    return true;
}

bool ResolveFusionPorts(const GNode& addOpNode, int32_t& matmulInputIdx, int32_t& biasInputIdx)
{
    AscendString biasAddType;
    if (addOpNode.GetType(biasAddType) != GRAPH_SUCCESS) {
        return false;
    }
    matmulInputIdx = 0;
    biasInputIdx = 1;
    if (biasAddType == kOpTypeAdd) {
        auto in0OpNode = addOpNode.GetInDataNodesAndPortIndexs(0).first;
        if (!IsMatMulType(in0OpNode)) {
            matmulInputIdx = 1;
            biasInputIdx = 0;
        }
    }
    return true;
}

bool ValidateFusionPreconditions(const GNode& matmulOpNode, const GNode& addOpNode, const GNode& biasDataNode)
{
    AscendString biasAddType;
    if (addOpNode.GetType(biasAddType) != GRAPH_SUCCESS) {
        return false;
    }
    if (biasAddType == kOpTypeAdd) {
        if (!CheckAddDtype(matmulOpNode, addOpNode)) {
            OPS_LOG_D(kPassName, "Add dtype check failed, skip fusion.");
            return false;
        }
    }

    if (matmulOpNode.GetInputsSize() != static_cast<size_t>(kBaseNodeNum)) {
        OPS_LOG_I(kPassName, "MatMul node should have 2 inputs, actual %zu.", matmulOpNode.GetInputsSize());
        return false;
    }

    auto matmulOutNodes = matmulOpNode.GetOutDataNodesAndPortIndexs(0);
    if (matmulOutNodes.size() != 1) {
        OPS_LOG_I(kPassName, "MatMul node should only have 1 output, actual %zu.", matmulOutNodes.size());
        return false;
    }

    if (addOpNode.GetOutputsSize() != 1) {
        OPS_LOG_I(kPassName, "BiasAdd node should only have 1 output, actual %zu.", addOpNode.GetOutputsSize());
        return false;
    }

    if (biasDataNode.GetOutputsSize() != 1) {
        OPS_LOG_I(kPassName, "Bias node should only have 1 output, skip fusion.");
        return false;
    }

    return true;
}

void LinkMatMulEdges(Graph& rawGraph, const GNode& matmulOpNode, GNode& biasDataNode, GNode& newNode)
{
    for (int64_t idx = 0; idx < kBaseNodeNum; idx++) {
        auto [srcPtr, srcPort] = matmulOpNode.GetInDataNodesAndPortIndexs(static_cast<int32_t>(idx));
        if (srcPtr != nullptr) {
            ge::es::AddEdgeAndUpdatePeerDesc(rawGraph, *srcPtr, srcPort, newNode, static_cast<int32_t>(idx));
        }
    }
    ge::es::AddEdgeAndUpdatePeerDesc(rawGraph, biasDataNode, 0, newNode, static_cast<int32_t>(kBiasInputIdx));
}

void CopyNodeAttrsAndDescs(const GNode& matmulOpNode, GNode& biasDataNode, const AscendString& opTypeStr,
                           GNode& newNode)
{
    bool isBatch = IsBatchOpType(opTypeStr);
    const char* transAttr1 = isBatch ? kAttrAdjX1 : kAttrTransposeX1;
    const char* transAttr2 = isBatch ? kAttrAdjX2 : kAttrTransposeX2;

    bool transX1 = false;
    bool transX2 = false;
    if (matmulOpNode.GetAttr(transAttr1, transX1) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName, "Get %s attr failed, use default false.", transAttr1);
        transX1 = false;
    }
    if (matmulOpNode.GetAttr(transAttr2, transX2) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName, "Get %s attr failed, use default false.", transAttr2);
        transX2 = false;
    }
    newNode.SetAttr(transAttr1, transX1);
    newNode.SetAttr(transAttr2, transX2);
    CopyOtherAttrs(matmulOpNode, newNode, kPassName);

    TensorDesc desc;
    for (int64_t idx = 0; idx < kBaseNodeNum; idx++) {
        if (matmulOpNode.GetInputDesc(idx, desc) == GRAPH_SUCCESS) {
            newNode.UpdateInputDesc(idx, desc);
        }
    }
    TensorDesc biasDesc;
    if (biasDataNode.GetOutputDesc(0, biasDesc) == GRAPH_SUCCESS) {
        newNode.UpdateInputDesc(static_cast<int32_t>(kBiasInputIdx), biasDesc);
    }
    TensorDesc outputDesc;
    if (matmulOpNode.GetOutputDesc(0, outputDesc) == GRAPH_SUCCESS) {
        newNode.UpdateOutputDesc(0, outputDesc);
    }
}

Status CreateMatMulNodeWithBias(const GraphPtr& graph, const GNode& matmulOpNode, GNode& biasDataNode, GNode& newNode)
{
    AscendString opTypeStr;
    FUSION_PASS_CHECK(matmulOpNode.GetType(opTypeStr) != GRAPH_SUCCESS, OPS_LOG_E(kPassName, "Get matmul type failed."),
                      return GRAPH_FAILED);

    AscendString matmulName;
    FUSION_PASS_CHECK(matmulOpNode.GetName(matmulName) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul name failed."), return GRAPH_FAILED);

    AscendString targetOpTypeStr = opTypeStr;
    if (opTypeStr == kOpTypeBatchMatMul) {
        targetOpTypeStr = kOpTypeBatchMatMulV2;
        OPS_LOG_I(kPassName, "Switch BatchMatMul to BatchMatMulV2 for bias fusion, name=%s.", matmulName.GetString());
    }

    bool isBatch = IsBatchOpType(targetOpTypeStr);
    bool hasOffsetW = HasOffsetWInput(targetOpTypeStr);
    const char* transAttr1 = isBatch ? kAttrAdjX1 : kAttrTransposeX1;
    const char* transAttr2 = isBatch ? kAttrAdjX2 : kAttrTransposeX2;

    std::vector<ge::es::CompliantNodeBuilder::IrInputDef> inputs = {
        {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
    };
    if (hasOffsetW) {
        inputs.push_back({"offset_w", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""});
    }

    std::vector<ge::es::CompliantNodeBuilder::IrAttrDef> attrs = {
        {transAttr1, ge::es::CompliantNodeBuilder::kEsAttrRequired, "Bool", ge::es::CreateFrom(false)},
        {transAttr2, ge::es::CompliantNodeBuilder::kEsAttrRequired, "Bool", ge::es::CreateFrom(false)},
    };
    if (hasOffsetW) {
        attrs.push_back({"offset_x", ge::es::CompliantNodeBuilder::kEsAttrOptional, "Int", AttrValue()});
    }

    auto* rawGraph = graph.get();
    newNode = ge::es::CompliantNodeBuilder(rawGraph)
                  .OpType(targetOpTypeStr.GetString())
                  .Name(matmulName.GetString())
                  .IrDefInputs(inputs)
                  .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                  .IrDefAttrs(attrs)
                  .Build();

    LinkMatMulEdges(*rawGraph, matmulOpNode, biasDataNode, newNode);
    CopyNodeAttrsAndDescs(matmulOpNode, biasDataNode, targetOpTypeStr, newNode);

    bool hasBias = true;
    newNode.SetAttr(kAttrHasBias, hasBias);
    OPS_LOG_I(kPassName, "Created new matmul node with bias, name=%s, type=%s.", matmulName.GetString(),
              targetOpTypeStr.GetString());
    return SUCCESS;
}

Status PrepareFusion(const GNode& addOpNode, int32_t& matmulInputIdx, int32_t& biasInputIdx)
{
    if (!ResolveFusionPorts(addOpNode, matmulInputIdx, biasInputIdx)) {
        OPS_LOG_D(kPassName, "Resolve fusion ports failed, skip fusion.");
        return GRAPH_NOT_CHANGED;
    }

    auto matmulOpNodePtr = addOpNode.GetInDataNodesAndPortIndexs(matmulInputIdx).first;
    auto biasDataNodePtr = addOpNode.GetInDataNodesAndPortIndexs(biasInputIdx).first;
    if (matmulOpNodePtr == nullptr || biasDataNodePtr == nullptr) {
        OPS_LOG_D(kPassName, "matmul or bias input node is null.");
        return GRAPH_NOT_CHANGED;
    }
    if (!IsMatMulType(matmulOpNodePtr)) {
        OPS_LOG_D(kPassName, "Input node is not matmul type, skip fusion.");
        return GRAPH_NOT_CHANGED;
    }

    if (!ValidateFusionPreconditions(*matmulOpNodePtr, addOpNode, *biasDataNodePtr)) {
        OPS_LOG_D(kPassName, "Validate fusion preconditions failed, skip fusion.");
        return GRAPH_NOT_CHANGED;
    }
    return SUCCESS;
}

void RelinkOutputEdges(const GraphPtr& graph, GNode& addOpNode, GNode& newMatmulOpNode)
{
    auto outPairs = addOpNode.GetOutDataNodesAndPortIndexs(0);
    for (auto& [dstOpNodePtr, dstPort] : outPairs) {
        if (dstOpNodePtr == nullptr) {
            continue;
        }
        GNode dstOpNode = *dstOpNodePtr;
        graph->RemoveEdge(addOpNode, 0, dstOpNode, dstPort);
        graph->AddDataEdge(newMatmulOpNode, 0, dstOpNode, dstPort);
    }
}

void TransferCtrlEdges(const GraphPtr& graph, const GNode& addOpNode, GNode& newMatmulOpNode)
{
    for (auto& srcNodePtr : addOpNode.GetInControlNodes()) {
        if (srcNodePtr != nullptr) {
            graph->AddControlEdge(*srcNodePtr, newMatmulOpNode);
        }
    }
    for (auto& dstNodePtr : addOpNode.GetOutControlNodes()) {
        if (dstNodePtr != nullptr) {
            graph->AddControlEdge(newMatmulOpNode, *dstNodePtr);
        }
    }
}

void RemoveFusedNodes(const GraphPtr& graph, GNode& oldMatmulOpNode, GNode& addOpNode, GNode& biasDataNode,
                      int32_t matmulInputIdx, int32_t biasInputIdx)
{
    graph->RemoveEdge(oldMatmulOpNode, 0, addOpNode, matmulInputIdx);
    graph->RemoveEdge(biasDataNode, 0, addOpNode, biasInputIdx);
    graph->RemoveNode(addOpNode);

    for (int64_t idx = 0; idx < kBaseNodeNum; idx++) {
        auto [srcPtr, srcPort] = oldMatmulOpNode.GetInDataNodesAndPortIndexs(static_cast<int32_t>(idx));
        if (srcPtr != nullptr) {
            graph->RemoveEdge(*srcPtr, srcPort, oldMatmulOpNode, static_cast<int32_t>(idx));
        }
    }
    graph->RemoveNode(oldMatmulOpNode);
}

Status CommitFusion(const GraphPtr& graph, GNode& addOpNode, int32_t matmulInputIdx, int32_t biasInputIdx,
                    CustomPassContext& passContext)
{
    auto matmulOpNodePtr = addOpNode.GetInDataNodesAndPortIndexs(matmulInputIdx).first;
    auto biasDataNodePtr = addOpNode.GetInDataNodesAndPortIndexs(biasInputIdx).first;
    FUSION_PASS_CHECK(matmulOpNodePtr == nullptr || biasDataNodePtr == nullptr,
                      OPS_LOG_E(kPassName, "matmul or bias node is null."), return GRAPH_FAILED);

    GNode oldMatmulOpNode = *matmulOpNodePtr;
    GNode biasDataNode = *biasDataNodePtr;

    AscendString matmulName;
    FUSION_PASS_CHECK(oldMatmulOpNode.GetName(matmulName) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul name failed."), return GRAPH_FAILED);

    GNode newMatmulOpNode;
    FUSION_PASS_CHECK(CreateMatMulNodeWithBias(graph, oldMatmulOpNode, biasDataNode, newMatmulOpNode) != SUCCESS,
                      OPS_LOG_E(kPassName, "Create matmul node with bias failed."), return GRAPH_FAILED);

    TensorDesc outputDesc;
    if (addOpNode.GetOutputDesc(0, outputDesc) == GRAPH_SUCCESS) {
        newMatmulOpNode.UpdateOutputDesc(0, outputDesc);
    }

    RelinkOutputEdges(graph, addOpNode, newMatmulOpNode);

    TransferCtrlEdges(graph, addOpNode, newMatmulOpNode);

    std::vector<GNode> nodesBeforeFuse = {oldMatmulOpNode, addOpNode};
    ReportFusion(nodesBeforeFuse, {newMatmulOpNode}, passContext, kPassName);

    RemoveFusedNodes(graph, oldMatmulOpNode, addOpNode, biasDataNode, matmulInputIdx, biasInputIdx);

    OPS_LOG_I(kPassName, "matmul biasadd fusion success! matmul=%s.", matmulName.GetString());
    return SUCCESS;
}

Status FuseOneBiasAddNode(const GraphPtr& graph, GNode& addOpNode, CustomPassContext& passContext)
{
    int32_t matmulInputIdx = 0;
    int32_t biasInputIdx = 0;
    auto status = PrepareFusion(addOpNode, matmulInputIdx, biasInputIdx);
    if (status != SUCCESS) {
        return status;
    }
    return CommitFusion(graph, addOpNode, matmulInputIdx, biasInputIdx, passContext);
}

} // namespace

Status MatMulBiasAddFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    OPS_LOG_D(kPassName, "Begin to do MatMulBiasAddFusionPass Run.");
    if (graph == nullptr || !graph->IsValid()) {
        OPS_LOG_W(kPassName, "Graph is null or invalid, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    if (!IsTargetVersion()) {
        return GRAPH_NOT_CHANGED;
    }

    passContext.SetPassName(kPassName);

    std::vector<GNode> addOpNodes;
    for (auto& node : graph->GetDirectNode()) {
        if (IsBiasAddOrAddType(node)) {
            addOpNodes.emplace_back(node);
        }
    }
    if (addOpNodes.empty()) {
        OPS_LOG_D(kPassName, "No BiasAdd/Add node, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    bool changed = false;
    for (auto& addOpNode : addOpNodes) {
        auto status = FuseOneBiasAddNode(graph, addOpNode, passContext);
        if (status == SUCCESS) {
            changed = true;
            continue;
        }
        if (status != GRAPH_NOT_CHANGED) {
            return status;
        }
    }

    OPS_LOG_D(kPassName, "Exit MatMulBiasAddFusionPass.");
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(MatMulBiasAddFusionPass)
    .Stage(IsTargetVersion() ? CustomPassStage::kCompatibleInherited : CustomPassStage::kAfterInferShape);
#endif

} // namespace ops
