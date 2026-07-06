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
 * \file quant_batch_matmul_v3_transpose_fusion_pass.cpp
 * \brief 删除QuantBatchMatmulV3前的类transpose节点，并设置transpose_x1/transpose_x2属性。
 *
 * 融合规则：
 *
 * 直接输入场景：
 *
 *      x1或x2             x1_scale或x2_scale               x1或x2          x1_scale或x2_scale
 *          |                       |                           |                    |
 *      Transpose             Transpose/Reshape                 |                    |
 *          |                       |                           |                    |
 *          +---------- QuantBatchMatmulV3  -------->           +---- QuantBatchMatmulV3
 *                             |                                                   |
 *                            out                                                 out
 *
 * Bitcast输入场景：
 *
 *      x1或x2             x1_scale或x2_scale               x1或x2          x1_scale或x2_scale
 *          |                       |                           |                    |
 *      Transpose             Transpose/Reshape                 |                    |
 *          |                       |                           |                    |
 *       Bitcast                 Bitcast                     Bitcast              Bitcast
 *          |                       |                           |                    |
 *          +---------- QuantBatchMatmulV3  -------->           +---- QuantBatchMatmulV3
 *                             |                                                   |
 *                            out                                                 out
 */

#include "quant_batch_matmul_v3_transpose_fusion_pass.h"

#include <algorithm>
#include <array>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"
#include "acl/acl_rt.h"

using namespace ge;
using namespace ge::fusion;

namespace ops {
namespace {
constexpr char PASS_NAME[] = "QuantBatchMatmulV3TransposeFusionPass";
constexpr char LIMIT_PASS_NAME[] = "QuantBatchMatmulV3TransposeLimitFusionPass";
constexpr char OP_TYPE_QUANTBMMV3[] = "QuantBatchMatmulV3";
constexpr char OP_TYPE_TRANSPOSE[] = "Transpose";
constexpr char OP_TYPE_TRANSPOSED[] = "TransposeD";
constexpr char OP_TYPE_RESHAPE[] = "Reshape";
constexpr char OP_TYPE_BITCAST[] = "Bitcast";

constexpr int X1_INDEX = 0;
constexpr int X2_INDEX = 1;
constexpr int X2_SCALE_INDEX = 2;
constexpr int X1_SCALE_INDEX = 5;
constexpr int MINI_SHAPE_LEN = 2;
constexpr int MX_SCALE_LEN = 3;
constexpr int INNER_SHAPE_LIMIT = 65535;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;

bool IsTargetVersion()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge-compiler", &version);
    if (version >= 90100000) {
        return true;
    }
    return false;
}

bool IsType(const GNodePtr& node, const char* type)
{
    if (node == nullptr) {
        return false;
    }
    AscendString nodeType;
    return node->GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

bool IsTransposeType(const GNodePtr& node)
{
    return IsType(node, OP_TYPE_TRANSPOSE) || IsType(node, OP_TYPE_TRANSPOSED);
}

// 边方向：srcNode:srcOutputPort  --->  dstNode:dstInputPort
// 返回srcNode，并可选地将源节点输出端口写入srcOutputPort。
GNodePtr GetInputNode(const GNode& dstNode, int64_t dstInputPort, int64_t* srcOutputPort = nullptr)
{
    auto [srcNode, resolvedSrcOutputPort] = dstNode.GetInDataNodesAndPortIndexs(dstInputPort);
    if (srcOutputPort != nullptr) {
        *srcOutputPort = resolvedSrcOutputPort;
    }
    return srcNode;
}

bool IsLegalDataType(DataType dtype, const std::vector<DataType>& legalDtypes)
{
    return std::find(legalDtypes.begin(), legalDtypes.end(), dtype) != legalDtypes.end();
}

bool IsUnknownShape(const Shape& shape)
{
    const auto dims = shape.GetDims();
    return std::any_of(dims.begin(), dims.end(), [](const int64_t dim) {
        return dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM || dim < 0;
    });
}

void GetPlatformSupport(bool& supportL12btBf16, bool& supportMmadS8S4)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get platform info.");
        return;
    }

    auto l12btIter = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_data_move_l12bt");
    supportL12btBf16 = l12btIter != platformInfo.ai_core_intrinsic_dtype_map.end() &&
                       std::find(l12btIter->second.begin(), l12btIter->second.end(), "bf16") != l12btIter->second.end();
    auto mmadIter = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_mmad");
    supportMmadS8S4 = mmadIter != platformInfo.ai_core_intrinsic_dtype_map.end() &&
                      std::find(mmadIter->second.begin(), mmadIter->second.end(), "s8s4") != mmadIter->second.end();
}

bool CheckNodeDtype(const GNode& nodeQuantBmmv3)
{
    static const std::vector<DataType> legalInputDtypes = {DT_INT8,        DT_INT4,     DT_FLOAT8_E4M3FN,
                                                           DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT4_E2M1};
    static const std::vector<DataType> legalOutDtypes = {DT_INT8, DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT};

    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    if (nodeQuantBmmv3.GetInputDesc(X1_INDEX, x1Desc) != GRAPH_SUCCESS ||
        nodeQuantBmmv3.GetInputDesc(X2_INDEX, x2Desc) != GRAPH_SUCCESS ||
        nodeQuantBmmv3.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get QuantBatchMatmulV3 input or output desc.");
        return false;
    }
    if (!IsLegalDataType(x1Desc.GetDataType(), legalInputDtypes)) {
        OP_LOGW(PASS_NAME, "X1 dtype is %s, which must be INT8/INT4/FLOAT8_E4M3/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1.",
                ge::TypeUtils::DataTypeToSerialString(x1Desc.GetDataType()).c_str());
        return false;
    }
    if (!IsLegalDataType(x2Desc.GetDataType(), legalInputDtypes)) {
        OP_LOGW(PASS_NAME, "X2 dtype is %s, which must be INT8/INT4/FLOAT8_E4M3/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1.",
                ge::TypeUtils::DataTypeToSerialString(x2Desc.GetDataType()).c_str());
        return false;
    }
    if (!IsLegalDataType(outputDesc.GetDataType(), legalOutDtypes)) {
        OP_LOGW(PASS_NAME, "Out dtype is %s, which must be INT8/FLOAT16/BFLOAT16/INT32/FLOAT.",
                ge::TypeUtils::DataTypeToSerialString(outputDesc.GetDataType()).c_str());
        return false;
    }
    return true;
}

bool CheckNodeShape(const GNode& nodeQuantBmmv3)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    if (nodeQuantBmmv3.GetInputDesc(X1_INDEX, x1Desc) != GRAPH_SUCCESS ||
        nodeQuantBmmv3.GetInputDesc(X2_INDEX, x2Desc) != GRAPH_SUCCESS) {
        return false;
    }
    return x1Desc.GetOriginShape().GetDimNum() >= MINI_SHAPE_LEN &&
           x2Desc.GetOriginShape().GetDimNum() >= MINI_SHAPE_LEN;
}

bool IsDynamicNode(const GNode& nodeQuantBmmv3)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    if (nodeQuantBmmv3.GetInputDesc(X1_INDEX, x1Desc) != GRAPH_SUCCESS ||
        nodeQuantBmmv3.GetInputDesc(X2_INDEX, x2Desc) != GRAPH_SUCCESS) {
        return false;
    }

    const bool nodeIsDynamic = IsUnknownShape(x1Desc.GetShape()) || IsUnknownShape(x2Desc.GetShape());
    OP_LOGD(PASS_NAME, "isDynamic %d", nodeIsDynamic);
    return nodeIsDynamic;
}

bool IsReshapeTrans(const TensorDesc& inputDesc, const GNodePtr& nodePerInput, bool isDynamic)
{
    if (!IsType(nodePerInput, OP_TYPE_RESHAPE)) {
        return false;
    }
    if (isDynamic) {
        return true;
    }

    auto shape = inputDesc.GetShape();
    if (shape.GetDimNum() < MINI_SHAPE_LEN) {
        return false;
    }

    TensorDesc reshapeInputDesc;
    TensorDesc reshapeOutputDesc;
    if (nodePerInput->GetInputDesc(0, reshapeInputDesc) != GRAPH_SUCCESS ||
        nodePerInput->GetOutputDesc(0, reshapeOutputDesc) != GRAPH_SUCCESS) {
        return false;
    }
    auto shapeInput = reshapeInputDesc.GetShape();
    auto shapeOut = reshapeOutputDesc.GetShape();
    // mx量化场景中，FLOAT8_E8M0 scale使用(m, ceil(k / 64), 2)或(ceil(k / 64), n, 2)等3维shape。
    // 当前两维任意一维为1时，交换这两维可以用Reshape表示。
    if (inputDesc.GetDataType() == DT_FLOAT8_E8M0 && shapeInput.GetDimNum() == MX_SCALE_LEN &&
        shapeOut.GetDimNum() == MX_SCALE_LEN && shapeInput.GetDim(0) == shapeOut.GetDim(1) &&
        shapeInput.GetDim(1) == shapeOut.GetDim(0) && (shapeInput.GetDim(0) == 1 || shapeInput.GetDim(1) == 1)) {
        return true;
    }

    // 普通tensor只有在最后两维存在1时，最后两维transpose才可以等价为Reshape。
    return shape.GetDim(shape.GetDimNum() - LAST_FIRST_DIM_INDEX) == 1 ||
           shape.GetDim(shape.GetDimNum() - LAST_SECOND_DIM_INDEX) == 1;
}

bool IsNodeEqualTrans(const TensorDesc& inputDesc, const GNodePtr& inputNode, bool isDynamic)
{
    if (IsTransposeType(inputNode)) {
        return true;
    }
    return IsReshapeTrans(inputDesc, inputNode, isDynamic);
}

GNodePtr GetTransposeCandidateNode(const GNode& dstNode, int64_t dstInputPort, bool isBitcastPattern)
{
    auto srcNode = GetInputNode(dstNode, dstInputPort);
    if (isBitcastPattern && IsType(srcNode, OP_TYPE_BITCAST)) {
        return GetInputNode(*srcNode, 0);
    }
    return srcNode;
}

bool IsBitcastPattern(const GNode& nodeQuantBmmv3, bool isDynamic)
{
    static const std::array<int64_t, 4> quantDstInputPorts = {X1_INDEX, X2_INDEX, X1_SCALE_INDEX, X2_SCALE_INDEX};
    for (auto quantDstInputPort : quantDstInputPorts) {
        auto srcNode = GetInputNode(nodeQuantBmmv3, quantDstInputPort);
        if (!IsType(srcNode, OP_TYPE_BITCAST)) {
            continue;
        }
        // 低比特量化图中Bitcast可能插在Transpose/Reshape之后。
        // 这里检查Bitcast之前的节点，判断该输入是否为transpose模式。
        auto srcNodeBeforeBitcast = GetInputNode(*srcNode, 0);
        if (quantDstInputPort == X1_SCALE_INDEX || quantDstInputPort == X2_SCALE_INDEX) {
            // scale张量在特定shape场景下可能使用等价于Transpose的Reshape。
            TensorDesc inputDesc;
            if (nodeQuantBmmv3.GetInputDesc(quantDstInputPort, inputDesc) == GRAPH_SUCCESS &&
                IsNodeEqualTrans(inputDesc, srcNodeBeforeBitcast, isDynamic)) {
                return true;
            }
        } else if (IsTransposeType(srcNodeBeforeBitcast)) {
            // Bitcast模式下，x1/x2数据输入只需要识别真实Transpose节点。
            return true;
        }
    }
    return false;
}

bool CheckQuantBatchMatmulV3(const GNode& nodeQuantBmmv3)
{
    AscendString nodeType;
    return nodeQuantBmmv3.GetType(nodeType) == GRAPH_SUCCESS && nodeType == OP_TYPE_QUANTBMMV3;
}

bool CheckTranspose(const GNode& nodeQuantBmmv3, bool isBitcastPattern)
{
    auto nodeX1 = GetTransposeCandidateNode(nodeQuantBmmv3, X1_INDEX, isBitcastPattern);
    auto nodeX2 = GetTransposeCandidateNode(nodeQuantBmmv3, X2_INDEX, isBitcastPattern);
    if (nodeX1 == nullptr || nodeX2 == nullptr) {
        return false;
    }
    return IsTransposeType(nodeX1) || IsTransposeType(nodeX2);
}

bool GetIntrinsicsLimit(const GNode& nodeQuantBmmv3, bool supportL12btBf16, bool limitPass, bool& isDelTransx1,
                        bool& isDelTransx2)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    if (nodeQuantBmmv3.GetInputDesc(X1_INDEX, x1Desc) != GRAPH_SUCCESS ||
        nodeQuantBmmv3.GetInputDesc(X2_INDEX, x2Desc) != GRAPH_SUCCESS) {
        return false;
    }
    auto x1Shape = x1Desc.GetOriginShape();
    auto x2Shape = x2Desc.GetOriginShape();
    if (x1Shape.GetDimNum() < MINI_SHAPE_LEN || x2Shape.GetDimNum() < MINI_SHAPE_LEN) {
        return false;
    }

    // inner表示最后一维，outer表示倒数第二维。
    const auto x1Inner = x1Shape.GetDim(x1Shape.GetDimNum() - LAST_FIRST_DIM_INDEX);
    const auto x1Outer = x1Shape.GetDim(x1Shape.GetDimNum() - LAST_SECOND_DIM_INDEX);
    const auto x2Inner = x2Shape.GetDim(x2Shape.GetDimNum() - LAST_FIRST_DIM_INDEX);
    const auto x2Outer = x2Shape.GetDim(x2Shape.GetDimNum() - LAST_SECOND_DIM_INDEX);
    if (limitPass) {
        if (supportL12btBf16) {
            return true;
        }
        // 禁止关闭融合规则的场景，考虑客户可能会将可关闭融合规则关闭，当内轴 > 65535时, 且外轴 < 65535时一定融合
        isDelTransx1 = x1Outer > 0 && x1Outer <= INNER_SHAPE_LIMIT && x1Inner > INNER_SHAPE_LIMIT;
        isDelTransx2 = x2Outer > 0 && x2Outer <= INNER_SHAPE_LIMIT && x2Inner > INNER_SHAPE_LIMIT;
        return true;
    }

    // 普通融合pass在outer不超过65535或平台支持l12bt bf16时允许删除transpose。
    isDelTransx1 = x1Outer <= INNER_SHAPE_LIMIT || supportL12btBf16;
    // x2为FRACTAL_NZ格式时也允许删除transpose。
    isDelTransx2 = x2Outer <= INNER_SHAPE_LIMIT || supportL12btBf16 || x2Desc.GetFormat() == FORMAT_FRACTAL_NZ;
    return true;
}

void GetTransposeNode(const GNode& nodeQuantBmmv3, bool isBitcastPattern, bool isDelTransx1, bool isDelTransx2,
                      GNodePtr& nodeTransX1, GNodePtr& nodeTransX2)
{
    auto nodeX1 = GetTransposeCandidateNode(nodeQuantBmmv3, X1_INDEX, isBitcastPattern);
    auto nodeX2 = GetTransposeCandidateNode(nodeQuantBmmv3, X2_INDEX, isBitcastPattern);
    if (isDelTransx1 && IsTransposeType(nodeX1)) {
        nodeTransX1 = nodeX1;
    }
    if (isDelTransx2 && IsTransposeType(nodeX2)) {
        nodeTransX2 = nodeX2;
    }
}

bool GetScaleTransNode(const GNode& nodeQuantBmmv3, const GNodePtr& nodeTransX1, const GNodePtr& nodeTransX2,
                       bool isBitcastPattern, bool isDynamic, GNodePtr& nodeTransX1Scale, GNodePtr& nodeTransX2Scale)
{
    if (nodeTransX1 == nullptr && nodeTransX2 == nullptr) {
        return true;
    }

    auto nodeX1Scale = GetTransposeCandidateNode(nodeQuantBmmv3, X1_SCALE_INDEX, isBitcastPattern);
    auto nodeX2Scale = GetTransposeCandidateNode(nodeQuantBmmv3, X2_SCALE_INDEX, isBitcastPattern);

    TensorDesc x1ScaleDesc;
    if (nodeTransX1 != nullptr && nodeX1Scale != nullptr &&
        nodeQuantBmmv3.GetInputDesc(X1_SCALE_INDEX, x1ScaleDesc) == GRAPH_SUCCESS &&
        IsNodeEqualTrans(x1ScaleDesc, nodeX1Scale, isDynamic)) {
        nodeTransX1Scale = nodeX1Scale;
    }

    TensorDesc x2ScaleDesc;
    if (nodeTransX2 != nullptr && nodeX2Scale != nullptr &&
        nodeQuantBmmv3.GetInputDesc(X2_SCALE_INDEX, x2ScaleDesc) == GRAPH_SUCCESS &&
        IsNodeEqualTrans(x2ScaleDesc, nodeX2Scale, isDynamic)) {
        nodeTransX2Scale = nodeX2Scale;
    }
    return true;
}

// 删除节点及其输入边，调用前要求该节点已经没有输出边。
bool RemoveNode(const GraphPtr& graph, const GNodePtr& dstNode)
{
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(PASS_NAME, "Graph is null or invalid when removing node.");
        return false;
    }
    if (dstNode == nullptr || !dstNode->GetOutDataNodesAndPortIndexs(0).empty()) {
        return true;
    }

    const auto inputSize = dstNode->GetInputsSize();
    for (size_t dstInputPort = 0; dstInputPort < inputSize; ++dstInputPort) {
        int64_t srcOutputPort = 0;
        auto srcNode = GetInputNode(*dstNode, static_cast<int64_t>(dstInputPort), &srcOutputPort);
        if (srcNode == nullptr) {
            continue;
        }
        // 删除源节点到当前待删除节点的输入边。
        if (graph->RemoveEdge(*srcNode, srcOutputPort, *dstNode, dstInputPort) != GRAPH_SUCCESS) {
            OP_LOGW(PASS_NAME, "Failed to remove input edge before removing transpose-like node.");
            return false;
        }
    }
    // 输入边清理完成后，删除当前无输出使用的节点。
    if (graph->RemoveNode(*dstNode) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to remove unused transpose-like node.");
        return false;
    }
    return true;
}

// 绕过类transpose节点：普通场景重连到QuantBatchMatmulV3，bitcast场景重连到Bitcast。
bool RelinkNode(const GraphPtr& graph, GNode& nodeQuantBmmv3, const GNodePtr& nodeTrans, int64_t quantDstInputPort,
                bool isBitcastPattern)
{
    if (nodeTrans == nullptr) {
        return true;
    }
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(PASS_NAME, "Graph is null or invalid when relinking node.");
        return false;
    }

    int64_t srcOutputPort = 0;
    auto srcNode = GetInputNode(*nodeTrans, 0, &srcOutputPort);
    if (srcNode == nullptr) {
        OP_LOGW(PASS_NAME, "Failed to get input node of transpose-like node.");
        return false;
    }

    TensorDesc inputDesc;
    if (srcNode->GetOutputDesc(srcOutputPort, inputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get source output desc.");
        return false;
    }

    auto candidateDstNode = GetInputNode(nodeQuantBmmv3, quantDstInputPort);
    const bool relinkToBitcast = isBitcastPattern && IsType(candidateDstNode, OP_TYPE_BITCAST);
    auto& dstNode = relinkToBitcast ? *candidateDstNode : nodeQuantBmmv3;
    const int64_t dstInputPort = relinkToBitcast ? 0 : quantDstInputPort;

    // 先删除类transpose节点到后继节点的输出边。
    if (graph->RemoveEdge(*nodeTrans, 0, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to remove edge from transpose-like node.");
        return false;
    }
    // 再把类transpose节点的源节点直接接到后继节点。
    if (graph->AddDataEdge(*srcNode, srcOutputPort, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to add edge from source node.");
        return false;
    }

    if (relinkToBitcast) {
        TensorDesc quantInputDesc;
        if (nodeQuantBmmv3.GetInputDesc(quantDstInputPort, quantInputDesc) != GRAPH_SUCCESS) {
            OP_LOGW(PASS_NAME, "Failed to get QuantBatchMatmulV3 input desc.");
            return false;
        }
        // Bitcast场景下先更新Bitcast输入desc，使其匹配类transpose节点的输入。
        if (dstNode.UpdateInputDesc(dstInputPort, inputDesc) != GRAPH_SUCCESS) {
            OP_LOGW(PASS_NAME, "Failed to update Bitcast input desc.");
            return false;
        }
        // 后续还会更新QuantBatchMatmulV3输入desc，这里保留Bitcast输出侧dtype。
        inputDesc.SetDataType(quantInputDesc.GetDataType());
    }
    if (nodeQuantBmmv3.UpdateInputDesc(quantDstInputPort, inputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update QuantBatchMatmulV3 input desc.");
        return false;
    }
    return true;
}

bool SetTransposeAttr(GNode& node, const char* attrName)
{
    bool currentAttrValue = false;
    (void)node.GetAttr(attrName, currentAttrValue);
    bool newAttrValue = !currentAttrValue;
    if (node.SetAttr(attrName, newAttrValue) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to set transpose attr.");
        return false;
    }
    return true;
}

void ReportTransposeFusion(const std::vector<GNode>& nodesBeforeFuse, const GNode& nodeQuantBmmv3,
                           CustomPassContext& passContext)
{
    if (GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, {nodeQuantBmmv3}, passContext) != SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to report fusion result.");
    }
}

void AddNodeToRemove(std::vector<GNode>& nodesBeforeFuse, std::vector<GNodePtr>& nodesToRemove, const GNodePtr& node)
{
    if (node == nullptr || std::find(nodesToRemove.begin(), nodesToRemove.end(), node) != nodesToRemove.end()) {
        return;
    }
    nodesToRemove.emplace_back(node);
    nodesBeforeFuse.emplace_back(*node);
}

void CollectFusionNodes(const GNode& nodeQuantBmmv3, const std::array<GNodePtr, 4>& transNodes,
                        std::vector<GNode>& nodesBeforeFuse, std::vector<GNodePtr>& nodesToRemove)
{
    nodesBeforeFuse = {nodeQuantBmmv3};
    nodesToRemove.clear();
    for (const auto& node : transNodes) {
        AddNodeToRemove(nodesBeforeFuse, nodesToRemove, node);
    }
}

Status CheckFusionPreconditions(const GNode& nodeQuantBmmv3, bool& isDynamic, bool& isBitcastPattern,
                                bool& supportL12btBf16)
{
    if (!CheckQuantBatchMatmulV3(nodeQuantBmmv3)) {
        return GRAPH_NOT_CHANGED;
    }
    bool supportMmadS8S4 = false;
    GetPlatformSupport(supportL12btBf16, supportMmadS8S4);
    isBitcastPattern = IsBitcastPattern(nodeQuantBmmv3, isDynamic);
    if (supportMmadS8S4 && isBitcastPattern) {
        OP_LOGW(PASS_NAME, "Bitcast transpose fusion is not supported on MMAD S8S4 platform.");
        return GRAPH_NOT_CHANGED;
    }
    if (!CheckNodeDtype(nodeQuantBmmv3) || !CheckNodeShape(nodeQuantBmmv3)) {
        return GRAPH_NOT_CHANGED;
    }
    isDynamic = IsDynamicNode(nodeQuantBmmv3);
    if (!CheckTranspose(nodeQuantBmmv3, isBitcastPattern)) {
        return GRAPH_NOT_CHANGED;
    }
    return SUCCESS;
}

Status FindTransposeNodes(const GNode& nodeQuantBmmv3, bool supportL12btBf16, bool limitPass, bool isBitcastPattern,
                          bool isDynamic, std::array<GNodePtr, 4>& transNodes)
{
    constexpr size_t nodeX1Index = 0;
    constexpr size_t nodeX2Index = 1;
    constexpr size_t nodeX1ScaleIndex = 2;
    constexpr size_t nodeX2ScaleIndex = 3;
    bool isDelTransx1 = false;
    bool isDelTransx2 = false;
    if (!GetIntrinsicsLimit(nodeQuantBmmv3, supportL12btBf16, limitPass, isDelTransx1, isDelTransx2)) {
        return GRAPH_NOT_CHANGED;
    }
    if (!isDelTransx1 && !isDelTransx2) {
        return GRAPH_NOT_CHANGED;
    }

    GetTransposeNode(nodeQuantBmmv3, isBitcastPattern, isDelTransx1, isDelTransx2, transNodes[nodeX1Index],
                     transNodes[nodeX2Index]);
    if (transNodes[nodeX1Index] == nullptr && transNodes[nodeX2Index] == nullptr) {
        return GRAPH_NOT_CHANGED;
    }
    if (!GetScaleTransNode(nodeQuantBmmv3, transNodes[nodeX1Index], transNodes[nodeX2Index], isBitcastPattern,
                           isDynamic, transNodes[nodeX1ScaleIndex], transNodes[nodeX2ScaleIndex])) {
        return GRAPH_NOT_CHANGED;
    }
    return SUCCESS;
}

Status PrepareFusion(const GNode& nodeQuantBmmv3, bool& isDynamic, bool limitPass, bool& isBitcastPattern,
                     std::array<GNodePtr, 4>& transNodes)
{
    bool supportL12btBf16 = false;
    auto status = CheckFusionPreconditions(nodeQuantBmmv3, isDynamic, isBitcastPattern, supportL12btBf16);
    if (status != SUCCESS) {
        return status;
    }
    return FindTransposeNodes(nodeQuantBmmv3, supportL12btBf16, limitPass, isBitcastPattern, isDynamic, transNodes);
}

bool SetTransposeAttrs(GNode& nodeQuantBmmv3, const std::array<GNodePtr, 4>& transNodes)
{
    constexpr size_t nodeX1Index = 0;
    constexpr size_t nodeX2Index = 1;
    if (transNodes[nodeX1Index] != nullptr && !SetTransposeAttr(nodeQuantBmmv3, "transpose_x1")) {
        return false;
    }
    if (transNodes[nodeX2Index] != nullptr && !SetTransposeAttr(nodeQuantBmmv3, "transpose_x2")) {
        return false;
    }
    return true;
}

bool RelinkFusionNodes(const GraphPtr& graph, GNode& nodeQuantBmmv3, bool isBitcastPattern,
                       const std::array<GNodePtr, 4>& transNodes)
{
    constexpr std::array<int64_t, 4> inputPorts = {X1_INDEX, X2_INDEX, X1_SCALE_INDEX, X2_SCALE_INDEX};
    for (size_t nodeIndex = 0; nodeIndex < transNodes.size(); ++nodeIndex) {
        if (!RelinkNode(graph, nodeQuantBmmv3, transNodes[nodeIndex], inputPorts[nodeIndex], isBitcastPattern)) {
            return false;
        }
    }
    return true;
}

bool RemoveFusionNodes(const GraphPtr& graph, const std::vector<GNodePtr>& nodesToRemove)
{
    for (const auto& nodeToRemove : nodesToRemove) {
        if (!RemoveNode(graph, nodeToRemove)) {
            return false;
        }
    }
    return true;
}

Status CommitFusion(const GraphPtr& graph, GNode& nodeQuantBmmv3, bool isBitcastPattern,
                    const std::array<GNodePtr, 4>& transNodes, CustomPassContext& passContext)
{
    std::vector<GNode> nodesBeforeFuse;
    std::vector<GNodePtr> nodesToRemove;
    CollectFusionNodes(nodeQuantBmmv3, transNodes, nodesBeforeFuse, nodesToRemove);

    if (!SetTransposeAttrs(nodeQuantBmmv3, transNodes)) {
        return GRAPH_NOT_CHANGED;
    }
    if (!RelinkFusionNodes(graph, nodeQuantBmmv3, isBitcastPattern, transNodes)) {
        return GRAPH_FAILED;
    }
    ReportTransposeFusion(nodesBeforeFuse, nodeQuantBmmv3, passContext);
    if (!RemoveFusionNodes(graph, nodesToRemove)) {
        return GRAPH_FAILED;
    }
    return SUCCESS;
}

Status Fusion(const GraphPtr& graph, GNode& nodeQuantBmmv3, bool& isDynamic, bool limitPass,
              CustomPassContext& passContext)
{
    bool isBitcastPattern = false;
    std::array<GNodePtr, 4> transNodes = {};
    const auto status = PrepareFusion(nodeQuantBmmv3, isDynamic, limitPass, isBitcastPattern, transNodes);
    if (status != SUCCESS) {
        return status;
    }
    return CommitFusion(graph, nodeQuantBmmv3, isBitcastPattern, transNodes, passContext);
}

Status RunQuantBatchMatmulV3TransposeFusion(const GraphPtr& graph, bool limitPass, CustomPassContext& passContext)
{
    OP_LOGD(limitPass ? LIMIT_PASS_NAME : PASS_NAME, "Enter QuantBatchMatmulV3 transpose fusion pass.");
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(limitPass ? LIMIT_PASS_NAME : PASS_NAME, "Graph is null or invalid, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }
    // 只缓存QuantBatchMatmulV3节点，避免融合删除Transpose节点后继续访问失效GNode。
    std::vector<GNode> nodeQuantBmmv3List;
    for (auto& node : graph->GetDirectNode()) {
        if (CheckQuantBatchMatmulV3(node)) {
            nodeQuantBmmv3List.emplace_back(node);
        }
    }
    if (nodeQuantBmmv3List.empty()) {
        OP_LOGD(limitPass ? LIMIT_PASS_NAME : PASS_NAME, "No QuantBatchMatmulV3 node, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    passContext.SetPassName(limitPass ? LIMIT_PASS_NAME : PASS_NAME);
    bool isDynamic = false;
    bool changed = false;
    for (auto& nodeQuantBmmv3 : nodeQuantBmmv3List) {
        auto status = Fusion(graph, nodeQuantBmmv3, isDynamic, limitPass, passContext);
        if (status == SUCCESS) {
            changed = true;
            continue;
        }
        if (status != GRAPH_NOT_CHANGED) {
            return status;
        }
    }
    OP_LOGD(limitPass ? LIMIT_PASS_NAME : PASS_NAME, "Exit QuantBatchMatmulV3 transpose fusion pass.");
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}
} // namespace

Status QuantBatchMatmulV3TransposeFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    if (!IsTargetVersion()) {
        return GRAPH_NOT_CHANGED;
    }
    return RunQuantBatchMatmulV3TransposeFusion(graph, false, passContext);
}

Status QuantBatchMatmulV3TransposeLimitFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    if (!IsTargetVersion()) {
        return GRAPH_NOT_CHANGED;
    }
    return RunQuantBatchMatmulV3TransposeFusion(graph, true, passContext);
}
#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(QuantBatchMatmulV3TransposeFusionPass)
    .Stage(IsTargetVersion() ? CustomPassStage::kCompatibleInherited : CustomPassStage::kAfterInferShape);
REG_FUSION_PASS(QuantBatchMatmulV3TransposeLimitFusionPass)
    .Stage(IsTargetVersion() ? CustomPassStage::kCompatibleInherited : CustomPassStage::kAfterInferShape);
#endif

} // namespace ops
