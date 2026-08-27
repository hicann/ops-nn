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
 * \file matmul_reshape_transpose_fusion_pass.cpp
 * \brief the pass will effect if the graph pattern is as follows:
 *                     input1
 *                       |
 *                    transpose             input1
 *                       |                     |
 *            input0   reshape              reshape  input0
 *               \      /          ====>        \      /
 *                matmul                         matmul
 *                  |                               |
 *                reshape                        reshape
 *                  |                               |
 *               transpose                         out
 *                  |
 *                 out
 */

#include "matmul_reshape_transpose_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "platform/platform_info.h"
#include "graph/tensor.h"

using namespace ge;
using namespace ge::fusion;

namespace ops {
namespace {

constexpr char kPassName[] = "MatmulReshapeTransposeFusionPass";
constexpr int32_t kGeCompilerVersion900 = 90000000;
constexpr char kOpTypeMatMul[] = "MatMul";
constexpr char kOpTypeMatMulV2[] = "MatMulV2";
constexpr char kOpTypeReshape[] = "Reshape";
constexpr char kOpTypeTranspose[] = "Transpose";
constexpr char kOpTypeTransposeD[] = "TransposeD";
constexpr char kOpTypeConst[] = "Const";
constexpr char kOpTypeConstant[] = "Constant";

constexpr int32_t kIdx0 = 0;
constexpr int32_t kIdx1 = 1;
constexpr uint32_t kDimNumTwo = 2;
constexpr uint32_t kDimNumThree = 3;

// ========== 辅助函数 ==========

bool IsType(const GNodePtr& node, const char* type)
{
    if (node == nullptr) {
        return false;
    }
    AscendString nodeType;
    return node->GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

bool IsTransposeType(const GNodePtr& node) { return IsType(node, kOpTypeTranspose) || IsType(node, kOpTypeTransposeD); }

bool IsMatMulType(const GNodePtr& node) { return IsType(node, kOpTypeMatMul) || IsType(node, kOpTypeMatMulV2); }

bool IsUnknownShape(const Shape& shape)
{
    const auto dims = shape.GetDims();
    return std::any_of(dims.begin(), dims.end(), [](const int64_t dim) {
        return dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM || dim < 0;
    });
}

// ========== 平台检查 ==========

bool CheckPlatformSupport()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                          platformInfo, optionalInfo) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName, "Failed to get platform info."), return false);
    bool supportL0c2Out = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_fix_pipe_l0c2out") !=
                          platformInfo.ai_core_intrinsic_dtype_map.end();
    FUSION_PASS_CHECK(!supportL0c2Out, OPS_LOG_I(kPassName, "The platform is not supported, skip fusion."),
                      return false);
    return true;
}

// ========== 检查逻辑 ==========

bool CheckNodesBeforeMatmul(const GNodePtr& preTransposeNode, const GNodePtr& preReshapeNode)
{
    // 检查前置 TransposeD
    auto transOutNodes = preTransposeNode->GetOutDataNodesAndPortIndexs(kIdx0);
    if (transOutNodes.size() > 1) {
        OPS_LOG_D(kPassName, "preTransposeNode links to more than one node.");
        return false;
    }
    TensorDesc inDescTrans;
    TensorDesc outDescTrans;
    FUSION_PASS_CHECK(preTransposeNode->GetInputDesc(kIdx0, inDescTrans) != GRAPH_SUCCESS ||
                          preTransposeNode->GetOutputDesc(kIdx0, outDescTrans) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName, "The inputDesc/outputDesc of preTransposeNode is null."), return false);
    FUSION_PASS_CHECK(inDescTrans.GetDataType() != DT_FLOAT, OPS_LOG_D(kPassName, "input dtype is not float."),
                      return false);
    FUSION_PASS_CHECK(IsUnknownShape(inDescTrans.GetShape()), OPS_LOG_D(kPassName, "input shape is dynamic."),
                      return false);
    ge::Shape inShapeTrans = inDescTrans.GetOriginShape();
    ge::Shape outShapeTrans = outDescTrans.GetOriginShape();
    FUSION_PASS_CHECK(inShapeTrans.GetDimNum() != kDimNumThree || inShapeTrans.GetDims()[kDimNumThree - 1] != 1,
                      OPS_LOG_D(kPassName, "shape dim of transpose is not 3, or last dim of transpose is not 1."),
                      return false);
    FUSION_PASS_CHECK(inShapeTrans.GetDims()[0] != outShapeTrans.GetDims()[1] ||
                          inShapeTrans.GetDims()[1] != outShapeTrans.GetDims()[0],
                      OPS_LOG_D(kPassName, "perm of preTransposeNode is not {1,0,2}."), return false);

    // 检查前置 Reshape
    TensorDesc outDescRes;
    FUSION_PASS_CHECK(preReshapeNode->GetOutputDesc(kIdx0, outDescRes) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName, "The outputDesc of preReshapeNode is null."), return false);
    ge::Shape outShapeRes = outDescRes.GetOriginShape();
    FUSION_PASS_CHECK(outShapeRes.GetDimNum() != kDimNumTwo || outShapeRes.GetDims()[0] != outShapeTrans.GetDims()[0] ||
                          outShapeRes.GetDims()[1] != outShapeTrans.GetDims()[1],
                      OPS_LOG_D(kPassName, "reshape node pattern is not to reduce last dim 1 as expected."),
                      return false);
    return true;
}

bool CheckMatmulNode(const GNodePtr& matmulNode)
{
    FUSION_PASS_CHECK(!IsMatMulType(matmulNode), OPS_LOG_D(kPassName, "node is not matmul."), return false);
    auto [x2InputNode, _] = matmulNode->GetInDataNodesAndPortIndexs(kIdx1);
    FUSION_PASS_CHECK(!IsType(x2InputNode, kOpTypeReshape),
                      OPS_LOG_D(kPassName, "The input node of matmul x2 is not reshape."), return false);
    auto mmOutNodes = matmulNode->GetOutDataNodesAndPortIndexs(kIdx0);
    if (mmOutNodes.size() > 1) {
        OPS_LOG_D(kPassName, "matmulNode links to more than one node.");
        return false;
    }
    bool transA = false;
    bool transB = false;
    FUSION_PASS_CHECK(matmulNode->GetAttr("transpose_x1", transA) != GRAPH_SUCCESS ||
                          matmulNode->GetAttr("transpose_x2", transB) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName, "GetBool transpose_x1/x2 failed!"), return false);
    FUSION_PASS_CHECK(transA || transB, OPS_LOG_D(kPassName, "transpose flag is not as expected."), return false);
    TensorDesc inDescMatmul;
    FUSION_PASS_CHECK(matmulNode->GetInputDesc(kIdx0, inDescMatmul) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName, "The inputDesc of matmulNode is null."), return false);
    FUSION_PASS_CHECK(inDescMatmul.GetShape().GetDimNum() != kDimNumTwo,
                      OPS_LOG_D(kPassName, "input shape dim is not 2."), return false);
    return true;
}

bool CheckNodesAfterMatmul(const GNodePtr& suffReshapeNode, const GNodePtr& suffTransposeNode)
{
    auto suffResOutNodes = suffReshapeNode->GetOutDataNodesAndPortIndexs(kIdx0);
    if (suffResOutNodes.size() > 1) {
        OPS_LOG_D(kPassName, "suffReshapeNode links to more than one node.");
        return false;
    }
    TensorDesc inDescRes;
    TensorDesc outDescRes;
    FUSION_PASS_CHECK(suffReshapeNode->GetInputDesc(kIdx0, inDescRes) != GRAPH_SUCCESS ||
                          suffReshapeNode->GetOutputDesc(kIdx0, outDescRes) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName, "The inputDesc/outputDesc of suffReshapeNode is null."), return false);
    ge::Shape inShapeRes = inDescRes.GetOriginShape();
    ge::Shape outShapeRes = outDescRes.GetOriginShape();
    FUSION_PASS_CHECK(outShapeRes.GetDimNum() != kDimNumThree || outShapeRes.GetDims()[0] != inShapeRes.GetDims()[0] ||
                          outShapeRes.GetDims()[1] != inShapeRes.GetDims()[1] || outShapeRes.GetDims()[kDimNumTwo] != 1,
                      OPS_LOG_D(kPassName, "reshape node pattern is not to add last dim 1 as expected."), return false);
    auto suffTransOutNodes = suffTransposeNode->GetOutDataNodesAndPortIndexs(kIdx0);
    FUSION_PASS_CHECK(suffTransOutNodes.size() > 1,
                      OPS_LOG_D(kPassName, "suffTransposeNode links to more than one node."), return false);
    TensorDesc outDescTrans;
    FUSION_PASS_CHECK(suffTransposeNode->GetOutputDesc(kIdx0, outDescTrans) != GRAPH_SUCCESS,
                      OPS_LOG_W(kPassName, "The outputDesc of suffTransposeNode is null."), return false);
    ge::Shape outShapeTrans = outDescTrans.GetOriginShape();
    FUSION_PASS_CHECK(outShapeTrans.GetDims()[0] != outShapeRes.GetDims()[1] ||
                          outShapeTrans.GetDims()[1] != outShapeRes.GetDims()[0],
                      OPS_LOG_D(kPassName, "perm of suffTransposeNode is not {1,0,2}."), return false);
    return true;
}

// ========== 图修改逻辑 ==========

bool ChangeConstNode(const GNodePtr& reshapeNode)
{
    TensorDesc constDescRes;
    FUSION_PASS_CHECK(reshapeNode->GetInputDesc(kIdx1, constDescRes) != GRAPH_SUCCESS,
                      OPS_LOG_D(kPassName, "Reshape has no const input at index 1."), return true);
    TensorDesc outReshapeDescRes;
    FUSION_PASS_CHECK(reshapeNode->GetOutputDesc(kIdx0, outReshapeDescRes) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get output desc of reshape node."), return false);
    auto [constNode, _] = reshapeNode->GetInDataNodesAndPortIndexs(kIdx1);
    FUSION_PASS_CHECK(constNode == nullptr, OPS_LOG_D(kPassName, "const node is null."), return true);
    AscendString constType;
    FUSION_PASS_CHECK(constNode->GetType(constType) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get type of const node."), return false);
    FUSION_PASS_CHECK(constType != kOpTypeConst && constType != kOpTypeConstant,
                      OPS_LOG_D(kPassName, "The right input node of reshape node is not a const node."), return true);
    // 被共享时不修改原 const 的 value，避免影响其他消费者
    size_t totalOutEdges = 0;
    for (int32_t outIdx = 0; outIdx < static_cast<int32_t>(constNode->GetOutputsSize()); outIdx++) {
        totalOutEdges += constNode->GetOutDataNodesAndPortIndexs(outIdx).size();
    }
    if (totalOutEdges > 1) {
        OPS_LOG_D(kPassName, "Const node is shared by multiple nodes, skip modifying.");
        return true;
    }
    ge::DataType dataDtype = constDescRes.GetDataType();
    int64_t dimNum = static_cast<int64_t>(outReshapeDescRes.GetOriginShape().GetDimNum());
    Tensor tensor;
    FUSION_PASS_CHECK(constNode->GetAttr("value", tensor) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get value attr from const node."), return false);
    TensorDesc tensorDesc(ge::Shape({dimNum}), FORMAT_ND, dataDtype);
    tensor.SetTensorDesc(tensorDesc);
    if (dataDtype == DT_INT32) {
        std::vector<int32_t> constValue;
        for (int64_t i = 0; i < dimNum; i++) {
            constValue.push_back(static_cast<int32_t>(outReshapeDescRes.GetOriginShape().GetDim(i)));
        }
        tensor.SetData(reinterpret_cast<const uint8_t*>(constValue.data()), constValue.size() * sizeof(int32_t));
    } else {
        std::vector<int64_t> constValue;
        for (int64_t i = 0; i < dimNum; i++) {
            constValue.push_back(outReshapeDescRes.GetOriginShape().GetDim(i));
        }
        tensor.SetData(reinterpret_cast<const uint8_t*>(constValue.data()), constValue.size() * sizeof(int64_t));
    }
    FUSION_PASS_CHECK(constNode->SetAttr("value", tensor) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to set value attr to const node."), return false);
    return true;
}

bool ProcessPreNodes(const GraphPtr& graph, const GNodePtr& preTransposeNode, const GNodePtr& preReshapeNode)
{
    auto [srcNode, srcPort] = preTransposeNode->GetInDataNodesAndPortIndexs(kIdx0);
    FUSION_PASS_CHECK(srcNode == nullptr, OPS_LOG_E(kPassName, "Failed to get input node of preTransposeNode."),
                      return false);
    TensorDesc resInDesc;
    preReshapeNode->GetInputDesc(kIdx0, resInDesc);
    std::vector<int64_t> newInShapeRes = {resInDesc.GetOriginShape().GetDims()[1],
                                          resInDesc.GetOriginShape().GetDims()[0], 1};
    FUSION_PASS_CHECK(graph->RemoveEdge(*srcNode, srcPort, *preTransposeNode, kIdx0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Remove edge from previous node to preTransposeNode failed."), return false);
    FUSION_PASS_CHECK(graph->RemoveEdge(*preTransposeNode, kIdx0, *preReshapeNode, kIdx0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Remove edge from preTransposeNode to preReshapeNode failed."),
                      return false);
    FUSION_PASS_CHECK(graph->RemoveNode(*preTransposeNode) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Remove preTransposeNode failed."), return false);
    TensorDesc newInDesc = resInDesc;
    newInDesc.SetShape(ge::Shape(newInShapeRes));
    newInDesc.SetOriginShape(ge::Shape(newInShapeRes));
    FUSION_PASS_CHECK(preReshapeNode->UpdateInputDesc(kIdx0, newInDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update preReshapeNode input desc failed."), return false);
    std::vector<int64_t> newOutShapeRes = {newInShapeRes[0], newInShapeRes[1]};
    TensorDesc newOutDesc;
    preReshapeNode->GetOutputDesc(kIdx0, newOutDesc);
    newOutDesc.SetShape(ge::Shape(newOutShapeRes));
    newOutDesc.SetOriginShape(ge::Shape(newOutShapeRes));
    FUSION_PASS_CHECK(preReshapeNode->UpdateOutputDesc(kIdx0, newOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update preReshapeNode output desc failed."), return false);
    FUSION_PASS_CHECK(graph->AddDataEdge(*srcNode, srcPort, *preReshapeNode, kIdx0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Add edge from previous node to preReshapeNode failed."), return false);
    FUSION_PASS_CHECK(!ChangeConstNode(preReshapeNode),
                      OPS_LOG_E(kPassName, "Change the const input node of the preReshapeNode failed."), return false);
    OPS_LOG_D(kPassName, "End to process nodes before matmul.");
    return true;
}

bool ProcessMatmulNode(const GraphPtr& graph, const GNodePtr& matmulNode)
{
    auto [x1SrcNode, x1SrcPort] = matmulNode->GetInDataNodesAndPortIndexs(kIdx0);
    auto [x2SrcNode, x2SrcPort] = matmulNode->GetInDataNodesAndPortIndexs(kIdx1);
    FUSION_PASS_CHECK(x1SrcNode == nullptr || x2SrcNode == nullptr,
                      OPS_LOG_E(kPassName, "inDataAnchor of matmul input node is null."), return false);
    FUSION_PASS_CHECK(graph->RemoveEdge(*x1SrcNode, x1SrcPort, *matmulNode, kIdx0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Remove edge from x1 source to matmul failed."), return false);
    FUSION_PASS_CHECK(graph->RemoveEdge(*x2SrcNode, x2SrcPort, *matmulNode, kIdx1) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Remove edge from x2 source to matmul failed."), return false);
    // 交换 x1/x2 输入：x2 来源 → matmul x1, x1 来源 → matmul x2
    FUSION_PASS_CHECK(graph->AddDataEdge(*x2SrcNode, x2SrcPort, *matmulNode, kIdx0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Add edge from x2 source to matmul x1 failed."), return false);
    FUSION_PASS_CHECK(graph->AddDataEdge(*x1SrcNode, x1SrcPort, *matmulNode, kIdx1) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Add edge from x1 source to matmul x2 failed."), return false);
    bool transB = true;
    FUSION_PASS_CHECK(matmulNode->SetAttr("transpose_x2", transB) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Set transpose_x2 failed."), return false);
    TensorDesc x1SrcOutDesc;
    FUSION_PASS_CHECK(x1SrcNode->GetOutputDesc(x1SrcPort, x1SrcOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 source output desc failed."), return false);
    TensorDesc x2SrcOutDesc;
    FUSION_PASS_CHECK(x2SrcNode->GetOutputDesc(x2SrcPort, x2SrcOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 source output desc failed."), return false);
    FUSION_PASS_CHECK(matmulNode->UpdateInputDesc(kIdx0, x2SrcOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update matmul x1 desc failed."), return false);
    FUSION_PASS_CHECK(matmulNode->UpdateInputDesc(kIdx1, x1SrcOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update matmul x2 desc failed."), return false);
    TensorDesc outDesc;
    FUSION_PASS_CHECK(matmulNode->GetOutputDesc(kIdx0, outDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul output desc failed."), return false);
    auto outDims = outDesc.GetOriginShape().GetDims();
    if (outDims.size() >= kDimNumTwo) {
        std::vector<int64_t> newOutDims = {outDims[1], outDims[0]};
        outDesc.SetShape(ge::Shape(newOutDims));
        outDesc.SetOriginShape(ge::Shape(newOutDims));
        FUSION_PASS_CHECK(matmulNode->UpdateOutputDesc(kIdx0, outDesc) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Update matmul output desc failed."), return false);
    }
    OPS_LOG_D(kPassName, "End to process matmul node.");
    return true;
}

bool ProcessSuffNodes(const GraphPtr& graph, const GNodePtr& suffReshapeNode, const GNodePtr& suffTransposeNode)
{
    auto suffTransOutNodes = suffTransposeNode->GetOutDataNodesAndPortIndexs(kIdx0);
    if (suffTransOutNodes.empty()) {
        OPS_LOG_E(kPassName, "suffTransposeNode has no output node.");
        return false;
    }
    auto [outNode, outPort] = suffTransOutNodes[0];
    FUSION_PASS_CHECK(graph->RemoveEdge(*suffReshapeNode, kIdx0, *suffTransposeNode, kIdx0) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Remove edge from suffReshapeNode to suffTransposeNode failed."),
                      return false);
    FUSION_PASS_CHECK(graph->RemoveEdge(*suffTransposeNode, kIdx0, *outNode, outPort) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Remove edge from suffTransposeNode to outNode failed."), return false);
    FUSION_PASS_CHECK(graph->RemoveNode(*suffTransposeNode) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Remove suffTransposeNode failed."), return false);
    // 更新 suffReshapeNode 的输入/输出 desc（交换前两维）
    TensorDesc inDescRes;
    suffReshapeNode->GetInputDesc(kIdx0, inDescRes);
    auto inDims = inDescRes.GetOriginShape().GetDims();
    if (inDims.size() >= kDimNumTwo) {
        std::vector<int64_t> newInDims = {inDims[1], inDims[0]};
        inDescRes.SetShape(ge::Shape(newInDims));
        inDescRes.SetOriginShape(ge::Shape(newInDims));
        FUSION_PASS_CHECK(suffReshapeNode->UpdateInputDesc(kIdx0, inDescRes) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Update suffReshapeNode input desc failed."), return false);
    }
    TensorDesc outDescRes;
    suffReshapeNode->GetOutputDesc(kIdx0, outDescRes);
    auto outDims = outDescRes.GetOriginShape().GetDims();
    if (outDims.size() >= kDimNumThree) {
        std::vector<int64_t> newOutDims = {outDims[1], outDims[0], 1};
        outDescRes.SetShape(ge::Shape(newOutDims));
        outDescRes.SetOriginShape(ge::Shape(newOutDims));
        FUSION_PASS_CHECK(suffReshapeNode->UpdateOutputDesc(kIdx0, outDescRes) != GRAPH_SUCCESS,
                          OPS_LOG_E(kPassName, "Update suffReshapeNode output desc failed."), return false);
    }
    FUSION_PASS_CHECK(graph->AddDataEdge(*suffReshapeNode, kIdx0, *outNode, outPort) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Add edge from suffReshapeNode to outNode failed."), return false);
    FUSION_PASS_CHECK(!ChangeConstNode(suffReshapeNode),
                      OPS_LOG_E(kPassName, "Change the const input node of the suffReshapeNode failed."), return false);
    OPS_LOG_D(kPassName, "End to process nodes after matmul.");
    return true;
}

// ========== 主融合逻辑 ==========

struct MatmulChain {
    GNodePtr matmulNode;
    GNodePtr suffReshapeNode;
    GNodePtr suffTransposeNode;
};

bool CollectMatmulChains(const GNodePtr& preReshapeNode, std::vector<MatmulChain>& chains)
{
    auto resOutNodes = preReshapeNode->GetOutDataNodesAndPortIndexs(kIdx0);
    if (resOutNodes.empty()) {
        return false;
    }
    for (auto& [mmNode, mmPort] : resOutNodes) {
        if (!CheckMatmulNode(mmNode)) {
            return false;
        }
        auto mmOutNodes = mmNode->GetOutDataNodesAndPortIndexs(kIdx0);
        if (mmOutNodes.size() != 1) {
            return false;
        }
        GNodePtr suffReshapeNode = mmOutNodes[0].first;
        if (!IsType(suffReshapeNode, kOpTypeReshape)) {
            return false;
        }
        auto suffResOutNodes = suffReshapeNode->GetOutDataNodesAndPortIndexs(kIdx0);
        if (suffResOutNodes.size() != 1) {
            return false;
        }
        GNodePtr suffTransposeNode = suffResOutNodes[0].first;
        if (!IsTransposeType(suffTransposeNode)) {
            return false;
        }
        if (!CheckNodesAfterMatmul(suffReshapeNode, suffTransposeNode)) {
            return false;
        }
        chains.push_back({mmNode, suffReshapeNode, suffTransposeNode});
    }
    return true;
}

bool ExecuteFusion(const GraphPtr& graph, const GNodePtr& preTransposePtr, const GNodePtr& preReshapeNode,
                   const std::vector<MatmulChain>& chains, CustomPassContext& passContext)
{
    // ReportFuse（在删除节点之前调用，收集所有融合前后节点）
    std::vector<GNode> nodesBeforeFuse = {*preTransposePtr};
    std::vector<GNode> nodesAfterFuse = {*preReshapeNode};
    for (auto& chain : chains) {
        nodesBeforeFuse.push_back(*chain.matmulNode);
        nodesBeforeFuse.push_back(*chain.suffTransposeNode);
        nodesAfterFuse.push_back(*chain.matmulNode);
        nodesAfterFuse.push_back(*chain.suffReshapeNode);
    }
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse != nullptr) {
        if (ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, nodesAfterFuse, passContext) !=
            GRAPH_SUCCESS) {
            OPS_LOG_W(kPassName, "Failed to report fusion result.");
        }
    }

    // 处理前置节点（删除 Transpose，更新 Reshape，只需做一次）
    if (!ProcessPreNodes(graph, preTransposePtr, preReshapeNode)) {
        return false;
    }
    // 遍历处理每个 MatMul 及其后置节点
    for (auto& chain : chains) {
        if (!ProcessMatmulNode(graph, chain.matmulNode)) {
            return false;
        }
        if (!ProcessSuffNodes(graph, chain.suffReshapeNode, chain.suffTransposeNode)) {
            return false;
        }
    }

    OPS_LOG_D(kPassName, "End to do MatmulReshapeTransposeFusionPass replacement.");
    return true;
}

bool TryFuseOneChain(const GraphPtr& graph, GNode& node, CustomPassContext& passContext)
{
    AscendString nodeType;
    node.GetType(nodeType);
    if (nodeType != kOpTypeTranspose && nodeType != kOpTypeTransposeD) {
        return false;
    }

    GNodePtr preTransposePtr = std::make_shared<GNode>(node);

    auto transOutNodes = preTransposePtr->GetOutDataNodesAndPortIndexs(kIdx0);
    if (transOutNodes.size() != 1) {
        return false;
    }
    GNodePtr preReshapeNode = transOutNodes[0].first;
    if (!IsType(preReshapeNode, kOpTypeReshape)) {
        return false;
    }
    if (!CheckNodesBeforeMatmul(preTransposePtr, preReshapeNode)) {
        return false;
    }

    std::vector<MatmulChain> chains;
    if (!CollectMatmulChains(preReshapeNode, chains)) {
        return false;
    }

    OPS_LOG_D(kPassName, "Begin to do MatmulReshapeTransposeFusionPass replacement, chains=%zu.", chains.size());

    return ExecuteFusion(graph, preTransposePtr, preReshapeNode, chains, passContext);
}

} // namespace

Status MatmulReshapeTransposeFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    OPS_LOG_D(kPassName, "Enter MatmulReshapeTransposeFusionPass.");
    passContext.SetPassName(kPassName);

    FUSION_PASS_CHECK(GetGeCompilerVersionNum() < kGeCompilerVersion900,
                      OPS_LOG_D(kPassName, "GE runtime < 9.0.0, skip fusion."), return GRAPH_NOT_CHANGED);

    FUSION_PASS_CHECK(!CheckPlatformSupport(), OPS_LOG_D(kPassName, "Platform not supported, skip fusion."),
                      return GRAPH_NOT_CHANGED);

    bool changed = false;
    std::vector<GNode> nodes = graph->GetDirectNode();
    for (auto& node : nodes) {
        if (TryFuseOneChain(graph, node, passContext)) {
            changed = true;
        }
    }

    OPS_LOG_D(kPassName, "Exit MatmulReshapeTransposeFusionPass, changed=%d.", changed ? 1 : 0);
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}

REG_FUSION_PASS(MatmulReshapeTransposeFusionPass).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
