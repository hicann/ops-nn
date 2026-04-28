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
 * \file indexbytensor_static_fusion_pass.cpp
 * \brief indexbytensor_static_fusion_pass
 * static: IndexByTensor => Index;
 * dynamic: skip, do nothing;
 *
 * 使用 FusionBasePass + 手动构建 SubgraphBoundary 方式处理动态输入数量。
 * 参考 example: 2_move_relu_before_concat_pass
 */

#include <vector>
#include <string>
#include <algorithm>

#include "es_nn_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "ge/compliant_node_builder.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/graph_rewriter.h"
#include "ge/ge_utils.h"
#include "ge/fusion/pass/decompose_pass.h"
#include "common/inc/error_util.h"
#include "indexbytensor_static_fusion_pass.h"

using namespace ge;
using namespace ge::fusion;
using namespace fe;

namespace ops {

static const std::string kPassName = "IndexByTensorStaticFusionPass";
static const size_t kMinInputSize = 2; // IndexByTensor 至少需要 x + 1个indices

static bool IsDynamicShape(const std::vector<int64_t>& dims)
{
    return std::any_of(dims.begin(), dims.end(), [](int64_t d) { return d < 0; });
}

// 判断节点是否为 IndexByTensor 且满足替换条件
static bool MeetRequirements(const GNode& node)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for IndexByTensorStaticFusionPass");

    AscendString nodeType;
    if (node.GetType(nodeType) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get node type.");
        return false;
    }
    if (nodeType != "IndexByTensor") {
        return false;
    }

    TensorDesc xDesc;
    if (node.GetInputDesc(0, xDesc) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get input x desc.");
        return false;
    }
    auto xDims = xDesc.GetShape().GetDims();
    if (IsDynamicShape(xDims)) {
        OPS_LOG_D(kPassName.c_str(), "Skip because of dynamic input shape.");
        return false;
    }

    TensorDesc yDesc;
    if (node.GetOutputDesc(0, yDesc) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get output y desc.");
        return false;
    }
    auto yDims = yDesc.GetShape().GetDims();
    if (IsDynamicShape(yDims)) {
        OPS_LOG_D(kPassName.c_str(), "Skip because of dynamic output shape.");
        return false;
    }

    std::vector<int64_t> indicesMask;
    if (node.GetAttr("indices_mask", indicesMask) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get indices_mask attribute.");
        return false;
    }

    if (node.GetInputsSize() < kMinInputSize) {
        OPS_LOG_D(kPassName.c_str(), "IndexByTensor should have at least %zu inputs, got %zu.",
                  kMinInputSize, node.GetInputsSize());
        return false;
    }

    return true;
}

// 设置单个 producer 的输出描述
static void SetProducerOutputDesc(GNode* producer, const DataType& dtype,
                                   const Shape& shape, const Format& format)
{
    if (producer == nullptr) {
        return;
    }
    TensorDesc desc;
    producer->GetOutputDesc(0, desc);
    desc.SetDataType(dtype);
    desc.SetShape(shape);
    desc.SetFormat(format);
    desc.SetOriginShape(shape);
    desc.SetOriginFormat(format);
    producer->UpdateOutputDesc(0, desc);
}

// 收集所有 indices 输入的描述信息
struct IndicesDescInfo {
    std::vector<Shape> shapes;
    std::vector<DataType> dtypes;
    std::vector<Format> formats;
};

static IndicesDescInfo CollectIndicesDescs(const GNode& node, size_t indicesCount)
{
    IndicesDescInfo info;
    info.shapes.resize(indicesCount);
    info.dtypes.resize(indicesCount);
    info.formats.resize(indicesCount);
    for (size_t i = 0; i < indicesCount; i++) {
        TensorDesc idxDesc;
        node.GetInputDesc(static_cast<uint32_t>(1 + i), idxDesc);
        info.shapes[i] = idxDesc.GetShape();
        info.dtypes[i] = idxDesc.GetDataType();
        info.formats[i] = idxDesc.GetFormat();
    }
    return info;
}

// 构建替换子图: IndexByTensor -> Index
static GraphUniqPtr Replacement(const GNode& matchedNode)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for IndexByTensorStaticFusionPass");

    TensorDesc xDesc;
    matchedNode.GetInputDesc(0, xDesc);
    Shape xShape = xDesc.GetShape();
    DataType xDtype = xDesc.GetDataType();
    Format xFormat = xDesc.GetFormat();

    size_t indicesCount = matchedNode.GetInputsSize() - 1;
    auto indicesInfo = CollectIndicesDescs(matchedNode, indicesCount);

    std::vector<int64_t> indicesMask;
    if (matchedNode.GetAttr("indices_mask", indicesMask) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to get indices_mask attribute in Replacement.");
        return nullptr;
    }

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto rX = replaceGraphBuilder.CreateInput(0, "x", xDtype, xFormat, xShape.GetDims());

    std::vector<es::EsTensorHolder> rIndices;
    for (size_t i = 0; i < indicesCount; i++) {
        std::string idxName = "indices" + std::to_string(i);
        auto rIdx = replaceGraphBuilder.CreateInput(static_cast<int64_t>(i + 1), idxName.c_str(),
                                                     indicesInfo.dtypes[i], indicesInfo.formats[i],
                                                     indicesInfo.shapes[i].GetDims());
        rIndices.push_back(rIdx);
    }

    auto rIndexedSizes = replaceGraphBuilder.CreateVector(indicesMask);
    auto rIndexedStrides = replaceGraphBuilder.CreateVector(indicesMask);
    auto indexOutput = es::Index(rX, rIndexedSizes, rIndexedStrides, rIndices);

    SetProducerOutputDesc(rX.GetProducer(), xDtype, xShape, xFormat);
    for (size_t i = 0; i < rIndices.size(); i++) {
        SetProducerOutputDesc(rIndices[i].GetProducer(),
            indicesInfo.dtypes[i], indicesInfo.shapes[i], indicesInfo.formats[i]);
    }

    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({indexOutput});
    if (replaceGraph == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "Failed to build replacement graph.");
        return nullptr;
    }

    std::vector<Shape> allInputShapes = {xShape};
    for (size_t i = 0; i < indicesCount; i++) {
        allInputShapes.push_back(indicesInfo.shapes[i]);
    }
    if (GeUtils::InferShape(*replaceGraph, allInputShapes) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }

    return replaceGraph;
}

// 构造被替换的子图边界（手动处理动态输入数量）
static std::unique_ptr<SubgraphBoundary> ConstructSubgraphBoundary(const GNode& node)
{
    auto inputSize = node.GetInputsSize();
    std::unique_ptr<SubgraphBoundary> boundary = std::make_unique<SubgraphBoundary>();

    // 动态添加所有输入到 boundary
    for (size_t inputIdx = 0; inputIdx < inputSize; ++inputIdx) {
        SubgraphInput subgraphInput;
        subgraphInput.AddInput({node, static_cast<int64_t>(inputIdx)});
        if (boundary->AddInput(static_cast<int64_t>(inputIdx), std::move(subgraphInput)) != SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Failed to add input %zu to boundary.", inputIdx);
            return nullptr;
        }
    }

    // IndexByTensor 单输出
    SubgraphOutput subgraphOutput({node, 0});
    if (boundary->AddOutput(0, std::move(subgraphOutput)) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "Failed to add output to boundary.");
        return nullptr;
    }

    return boundary;
}

// 对单个 IndexByTensor 节点执行替换
static bool ReplaceIndexByTensor(const GNode& node)
{
    auto replacement = Replacement(node);
    if (replacement == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "Failed to create replacement graph.");
        return false;
    }

    auto boundary = ConstructSubgraphBoundary(node);
    if (boundary == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "Failed to construct subgraph boundary.");
        return false;
    }

    if (SubgraphRewriter::Replace(*boundary, *replacement) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "SubgraphRewriter::Replace failed.");
        return false;
    }

    OPS_LOG_D(kPassName.c_str(), "IndexByTensorStaticFusionPass replacement succeeded.");
    return true;
}

// 查找图中所有满足条件的 IndexByTensor 节点
static bool FindIndexByTensorNodes(const GraphPtr& graph, std::vector<GNode>& matchedNodes)
{
    for (auto& node : graph->GetDirectNode()) {
        AscendString nodeType;
        if (node.GetType(nodeType) != GRAPH_SUCCESS) {
            OPS_LOG_D(kPassName.c_str(), "GetType failed during node traversal.");
            continue;
        }
        if (nodeType == "IndexByTensor" && MeetRequirements(node)) {
            matchedNodes.emplace_back(node);
        }
    }
    return true;
}

Status IndexByTensorStaticFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    OPS_LOG_D(kPassName.c_str(), "Enter IndexByTensorStaticFusionPass::Run");

    // 备份原图用于回退
    Graph originGraph = *graph;

    // 遍历节点获取符合条件的 IndexByTensor 节点
    std::vector<GNode> matchedNodes;
    if (!FindIndexByTensorNodes(graph, matchedNodes)) {
        OPS_LOG_E(kPassName.c_str(), "FindIndexByTensorNodes failed.");
        *graph = originGraph;
        return FAILED;
    }

    if (matchedNodes.empty()) {
        OPS_LOG_D(kPassName.c_str(), "No IndexByTensor nodes found.");
        return GRAPH_NOT_CHANGED;
    }

    // 遍历每个符合条件的节点进行替换
    for (auto& node : matchedNodes) {
        if (!ReplaceIndexByTensor(node)) {
            OPS_LOG_E(kPassName.c_str(), "ReplaceIndexByTensor failed.");
            *graph = originGraph;
            return FAILED;
        }
    }

    return SUCCESS;
}

REG_FUSION_PASS(IndexByTensorStaticFusionPass).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
