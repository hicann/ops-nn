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
 * \file tensor_scatter_add_fusion_pass.cpp
 * \brief TensorScatterAdd / ScatterNonAliasingAdd --> TensorMove + ScatterNdAdd
 *
 *   x        indices    updates           x        indices  updates
 *    \         |          /                |           |       /
 *     \        |         /                 |           |      /
 *   TensorScatterAdd           -->    TensorMove      |     /
 *   or ScatterNonAliasingAdd              |           |    /
 *            |                            |           |   /
 *            y                        ScatterNdAdd(use_locking=false)
 *                                         |
 *                                         y
 */

#include "tensor_scatter_add_fusion_pass.h"
#include "es_nn_ops.h"
#include "compliant_node_builder.h"
#include "common/inc/error_util.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "ge/es_graph_builder.h"
#include <set>

using namespace ge;
using namespace fe;
using namespace fusion;

namespace OPS {
namespace NN {
namespace {

const std::string PASS_NAME = "TensorScatterAddFusionPass";
const int64_t CAPTURE_IDX_OUTPUT = 0l;
const std::set<std::string> SUPPORTED_OP_TYPES = {"TensorScatterAdd", "ScatterNonAliasingAdd"};

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------
bool IsTargetPlatform()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        false, PASS_NAME.c_str(), "Get platformInfo failed.");
    const std::string soc = platformInfo.str_info.short_soc_version;
    bool isSupported = (soc == "Ascend910_93") || (soc == "Ascend950");
    if (!isSupported) {
        OPS_LOG_D(PASS_NAME.c_str(), "Platform %s is not supported.", soc.c_str());
        return false;
    }
    return true;
}

static void GetInputsInfo(
    const std::vector<SubgraphInput>& subgraphInputs,
    std::vector<Shape>& inputShapes,
    std::vector<DataType>& inputDtypes,
    std::vector<Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

static void UpdateInputFormat(GNode& node, uint32_t idx, Format format)
{
    TensorDesc desc;
    node.GetInputDesc(idx, desc);
    desc.SetFormat(format);
    node.UpdateInputDesc(idx, desc);
}

static Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Begin infershape for replacement.");
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// 构建单个 Pattern：使用 CompliantNodeBuilder（方案B，TensorScatterAdd 无 ES API）
// ---------------------------------------------------------------------------
static PatternUniqPtr MakePattern(const std::string& opType)
{
    auto graphBuilder = es::EsGraphBuilder((PASS_NAME + "_" + opType).c_str());
    auto x = graphBuilder.CreateInput(0);
    auto indices = graphBuilder.CreateInput(1);
    auto updates = graphBuilder.CreateInput(2);

    ge::Graph* graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();

    GNode opNode = es::CompliantNodeBuilder(graphPtr)
        .OpType(opType.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                       {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                       {"updates", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();

    GNode xNode = *x.GetProducer();
    GNode indicesNode = *indices.GetProducer();
    GNode updatesNode = *updates.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, xNode, 0, opNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, indicesNode, 0, opNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, updatesNode, 0, opNode, 2);

    es::EsTensorHolder output(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(opNode, 0));

    auto graph = graphBuilder.BuildAndReset({output});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});
    return pattern;
}

// ===========================================================================
// Patterns — 方案B：CompliantNodeBuilder 构建 Pattern
// ===========================================================================
std::vector<PatternUniqPtr> TensorScatterAddFusionPass::Patterns()
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Patterns for TensorScatterAddFusionPass");
    std::vector<PatternUniqPtr> patterns;
    patterns.emplace_back(MakePattern("TensorScatterAdd"));
    patterns.emplace_back(MakePattern("ScatterNonAliasingAdd"));
    return patterns;
}

// ===========================================================================
// MeetRequirements — 校验平台 + 匹配节点 OpType
// ===========================================================================
bool TensorScatterAddFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter MeetRequirements for TensorScatterAddFusionPass");

    if (!IsTargetPlatform()) {
        OPS_LOG_D(PASS_NAME.c_str(), "Platform check failed, skip fusion.");
        return false;
    }

    NodeIo captured;
    OP_LOGE_IF(
        matchResult->GetCapturedTensor(CAPTURE_IDX_OUTPUT, captured) != SUCCESS,
        false, PASS_NAME.c_str(), "Failed to get captured tensor.");

    // 校验匹配节点的 OpType
    AscendString nodeType;
    captured.node.GetType(nodeType);
    std::string typeStr = nodeType.GetString();
    if (SUPPORTED_OP_TYPES.find(typeStr) == SUPPORTED_OP_TYPES.end()) {
        OPS_LOG_D(PASS_NAME.c_str(), "Matched node type %s is not supported, skip.", typeStr.c_str());
        return false;
    }

    return true;
}

// ===========================================================================
// Replacement — 构建替换图：TensorMove + ScatterNdAdd
// ===========================================================================
GraphUniqPtr TensorScatterAddFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Replacement for TensorScatterAddFusionPass");

    // 1. 获取子图边界的所有输入信息
    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    // 2. 创建替换图输入
    auto builder = es::EsGraphBuilder("replacement");
    auto rX = builder.CreateInput(
        0, "x", inputDtypes[0], inputFormats[0], inputShapes[0].GetDims());
    auto rIndices = builder.CreateInput(
        1, "indices", inputDtypes[1], inputFormats[1], inputShapes[1].GetDims());
    auto rUpdates = builder.CreateInput(
        2, "updates", inputDtypes[2], inputFormats[2], inputShapes[2].GetDims());

    ge::Graph* graphPtr = builder.GetCGraphBuilder()->GetGraph();

    // 3. TensorMove 节点（仓内无 ES API，使用 CompliantNodeBuilder）
    GNode tensorMoveNode = es::CompliantNodeBuilder(graphPtr)
        .OpType("TensorMove")
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputShape("y", inputShapes[0].GetDims())
        .InstanceOutputDataType("y", inputDtypes[0])
        .InstanceOutputFormat("y", inputFormats[0])
        .Build();

    // 4. 连边：x -> TensorMove
    GNode xNode = *rX.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, xNode, 0, tensorMoveNode, 0);

    // 4.1 刷新 TensorMove 节点输入的 Format（InferShape 不推导 Format）
    UpdateInputFormat(tensorMoveNode, 0, inputFormats[0]);

    // 5. TensorMove 输出转为 EsTensorHolder，用于 ScatterNdAdd ES API
    es::EsTensorHolder tensorMoveOutput(
        builder.GetCGraphBuilder()->GetTensorHolderFromNode(tensorMoveNode, 0));

    // 6. ScatterNdAdd 节点（使用 ES API）
    auto scatterNdAddOutput = es::ScatterNdAdd(tensorMoveOutput, rIndices, rUpdates, false);

    // 7. 刷新 ScatterNdAdd 节点所有输入的 Format（InferShape 不推导 Format）
    GNode scatterNdAddNode = *scatterNdAddOutput.GetProducer();
    UpdateInputFormat(scatterNdAddNode, 0, inputFormats[0]);
    UpdateInputFormat(scatterNdAddNode, 1, inputFormats[1]);
    UpdateInputFormat(scatterNdAddNode, 2, inputFormats[2]);

    // 8. 构建替换图
    GraphUniqPtr replaceGraph = builder.BuildAndReset({scatterNdAddOutput});
    if (replaceGraph == nullptr) {
        OPS_LOG_E(PASS_NAME.c_str(), "BuildAndReset returned nullptr.");
        return nullptr;
    }

    // 9. InferShape
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OPS_LOG_W(PASS_NAME.c_str(), "InferShape failed, continue with manual desc.");
    }

    OPS_LOG_D(PASS_NAME.c_str(), "Replacement graph built successfully.");
    return replaceGraph;
}

REG_FUSION_PASS(TensorScatterAddFusionPass).Stage(CustomPassStage::kCompatibleInherited);
} // namespace NN
} // namespace OPS
