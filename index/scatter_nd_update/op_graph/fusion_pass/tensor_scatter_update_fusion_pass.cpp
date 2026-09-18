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
 * \file tensor_scatter_update_fusion_pass.cpp
 * \brief TensorScatterUpdate --> TensorMove + ScatterNdUpdate
 *
 *
 *   x        indices    updates           x    indices updates
 *    \         |          /               |        /   /
 *     \        |         /                |       /   /
 *      tensor_scatter_update      -->tensor_move /   /
 *              |                          |     /   /
 *              |                          |    /   /
 *              y                     scatter_nd_update
 *                                         |
 *                                         |
 *                                         y
 */

#include "tensor_scatter_update_fusion_pass.h"
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

const std::string PASS_NAME = "TensorScatterUpdateFusionPass";
const int64_t CAPTURE_IDX_OUTPUT = 0l;
const std::string SUPPORTED_OP_TYPE = "TensorScatterUpdate";
constexpr int32_t kPortSelf = 0;
constexpr int32_t kPortIndices = 1;
constexpr int32_t kPortUpdates = 2;
constexpr int32_t PARAM_NUM = 3;
constexpr int64_t NANO_BLOCK_SIZE = 16;

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

static bool IsRegbasePlatform()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OPS_LOG_D(PASS_NAME.c_str(), "Get platformInfo failed.");
        return false;
    }
    const std::string socVersion = platformInfo.str_info.short_soc_version;
    bool isRegbase = (socVersion == "Ascend950" || socVersion == "MC62CM12A");
    OPS_LOG_D(PASS_NAME.c_str(), "Platform short soc: %s, is_regbase: %d", socVersion.c_str(), isRegbase);
    return isRegbase;
}

static bool IsNanoPlatform()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OPS_LOG_D(PASS_NAME.c_str(), "Get platformInfo failed.");
        return false;
    }

    bool isNano = static_cast<int64_t>(platformInfo.ai_core_spec.ubblock_size) == NANO_BLOCK_SIZE;
    OPS_LOG_D(PASS_NAME.c_str(), "UB block size: %lu, is_nano: %d", platformInfo.ai_core_spec.ubblock_size, isNano);
    return isNano;
}

static void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs, std::vector<Shape>& inputShapes,
                          std::vector<DataType>& inputDtypes, std::vector<Format>& inputFormats)
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

static bool CheckScatterNdUpdateSupported(const std::vector<DataType>& inputDtypes,
                                          const std::vector<Format>& inputFormats,
                                          const std::vector<Shape>& inputShapes)
{
    auto builder = es::EsGraphBuilder("probe");
    auto rX = builder.CreateInput(0, "x", inputDtypes[0], inputFormats[0], inputShapes[0].GetDims());
    auto rIndices = builder.CreateInput(1, "indices", inputDtypes[1], inputFormats[1], inputShapes[1].GetDims());
    auto rUpdates = builder.CreateInput(2, "updates", inputDtypes[2], inputFormats[2], inputShapes[2].GetDims());

    auto scatterNdUpdateOutput = es::ScatterNdUpdate(rX, rIndices, rUpdates, false);
    GNode scatterNdUpdateNode = *scatterNdUpdateOutput.GetProducer();
    UpdateInputFormat(scatterNdUpdateNode, kPortSelf, inputFormats[0]);
    UpdateInputFormat(scatterNdUpdateNode, kPortIndices, inputFormats[1]);
    UpdateInputFormat(scatterNdUpdateNode, kPortUpdates, inputFormats[2]);

    GraphUniqPtr probeGraph = builder.BuildAndReset({scatterNdUpdateOutput});
    if (probeGraph == nullptr) {
        OPS_LOG_W(PASS_NAME.c_str(), "Failed to build probe graph, skip fusion.");
        return false;
    }

    if (GeUtils::InferShape(*probeGraph, inputShapes) != SUCCESS) {
        OPS_LOG_W(PASS_NAME.c_str(), "InferShape for probe graph failed, skip fusion.");
        return false;
    }

    bool isOpSupported = false;
    AscendString unsupportedReason;
    if (GeUtils::CheckNodeSupportOnAicore(scatterNdUpdateNode, isOpSupported, unsupportedReason) != SUCCESS) {
        OPS_LOG_W(PASS_NAME.c_str(), "CheckNodeSupportOnAicore returned error, skip fusion.");
        return false;
    }
    if (!isOpSupported) {
        OPS_LOG_D(PASS_NAME.c_str(), "ScatterNdUpdate not supported on AICore: %s, skip fusion.",
                  unsupportedReason.GetString());
        return false;
    }
    return true;
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
// 构建单个 Pattern：使用 CompliantNodeBuilder（方案B，TensorScatterUpdate 无 ES API）
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
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, xNode, 0, opNode, kPortSelf);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, indicesNode, 0, opNode, kPortIndices);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, updatesNode, 0, opNode, kPortUpdates);

    es::EsTensorHolder output(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(opNode, 0));

    auto graph = graphBuilder.BuildAndReset({output});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});
    return pattern;
}

// ===========================================================================
// Patterns — 方案B：CompliantNodeBuilder 构建 Pattern
// ===========================================================================
std::vector<PatternUniqPtr> TensorScatterUpdateFusionPass::Patterns()
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Patterns for TensorScatterUpdateFusionPass");
    std::vector<PatternUniqPtr> patterns;
    patterns.emplace_back(MakePattern("TensorScatterUpdate"));
    return patterns;
}

// ===========================================================================
// MeetRequirements — 校验平台 + 匹配节点 OpType
// ===========================================================================
bool TensorScatterUpdateFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter MeetRequirements for TensorScatterUpdateFusionPass");

    NodeIo captured;
    OP_LOGE_IF(matchResult->GetCapturedTensor(CAPTURE_IDX_OUTPUT, captured) != SUCCESS, false, PASS_NAME.c_str(),
               "Failed to get captured tensor.");

    // 校验匹配节点的 OpType
    GNode sourceNode = captured.node;
    AscendString nodeType;
    sourceNode.GetType(nodeType);
    if (std::string(nodeType.GetString()) != SUPPORTED_OP_TYPE) {
        OPS_LOG_D(PASS_NAME.c_str(), "Captured node type %s is not TensorScatterUpdate, skip.", nodeType.GetString());
        return false;
    }

    TensorDesc inputDesc0;
    OP_LOGE_IF(sourceNode.GetInputDesc(0, inputDesc0) != SUCCESS, false, PASS_NAME.c_str(), "Get input x desc failed.");
    bool isRegbase = IsRegbasePlatform();
    if (isRegbase && (inputDesc0.GetDataType() == ge::DT_STRING || inputDesc0.GetDataType() == ge::DT_COMPLEX128)) {
        OPS_LOG_D(SUPPORTED_OP_TYPE.c_str(),
                  "In regbase arch, TensorScatterUpdateFusionPass not support string or complex128, not changed.");
        return false;
    }
    if (!isRegbase && inputDesc0.GetDataType() == ge::DT_BOOL) {
        OPS_LOG_D(SUPPORTED_OP_TYPE.c_str(), "TensorScatterUpdateFusionPass not support bool, not changed.");
        return false;
    }

    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    std::vector<Shape> inputShapes;
    for (int32_t i = 0; i < PARAM_NUM; i++) {
        TensorDesc desc;
        OP_LOGE_IF(sourceNode.GetInputDesc(i, desc) != SUCCESS, false, PASS_NAME.c_str(),
                   "Failed to get input desc %d.", i);
        inputDtypes.emplace_back(desc.GetDataType());
        inputFormats.emplace_back(desc.GetFormat());
        inputShapes.emplace_back(desc.GetShape());
    }

    if (!CheckScatterNdUpdateSupported(inputDtypes, inputFormats, inputShapes)) {
        OPS_LOG_D(PASS_NAME.c_str(), "ScatterNdUpdate not supported on AICore for current dtype/format, skip fusion.");
        return false;
    }

    return true;
}

// ===========================================================================
// Replacement — 构建替换图：TensorMove + ScatterNdUpdate
// ===========================================================================
GraphUniqPtr TensorScatterUpdateFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Replacement for TensorScatterUpdateFusionPass");

    // 1. 获取子图边界的所有输入信息
    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    // 2. 创建替换图输入
    auto builder = es::EsGraphBuilder("replacement");
    auto rX = builder.CreateInput(0, "x", inputDtypes[0], inputFormats[0], inputShapes[0].GetDims());
    auto rIndices = builder.CreateInput(1, "indices", inputDtypes[1], inputFormats[1], inputShapes[1].GetDims());
    auto rUpdates = builder.CreateInput(2, "updates", inputDtypes[2], inputFormats[2], inputShapes[2].GetDims());

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

    // 5. TensorMove 输出转为 EsTensorHolder，用于 ScatterNdUpdate ES API
    es::EsTensorHolder tensorMoveOutput(builder.GetCGraphBuilder()->GetTensorHolderFromNode(tensorMoveNode, 0));

    // 6. ScatterNdUpdate 节点（使用 ES API）
    auto scatterNdUpdateOutput = es::ScatterNdUpdate(tensorMoveOutput, rIndices, rUpdates, false);

    // 7. 刷新 ScatterNdUpdate 节点所有输入的 Format（InferShape 不推导 Format）
    GNode scatterNdUpdateNode = *scatterNdUpdateOutput.GetProducer();
    UpdateInputFormat(scatterNdUpdateNode, kPortSelf, inputFormats[0]);
    UpdateInputFormat(scatterNdUpdateNode, kPortIndices, inputFormats[1]);
    UpdateInputFormat(scatterNdUpdateNode, kPortUpdates, inputFormats[2]);

    // 8. Nano 平台：在 ScatterNdUpdate 输出后再插入一个 TensorMove
    //    非 Nano: x -> TensorMove -> ScatterNdUpdate -> y
    //    Nano:    x -> TensorMove -> ScatterNdUpdate -> TensorMove -> y
    es::EsTensorHolder finalOutput = scatterNdUpdateOutput;
    if (IsNanoPlatform()) {
        OPS_LOG_D(PASS_NAME.c_str(), "Nano platform detected, inserting TensorMove after ScatterNdUpdate.");
        GNode tensorMoveAfterNode = es::CompliantNodeBuilder(graphPtr)
                                        .Name("tensor_move_after_scatter_nd_update")
                                        .OpType("TensorMove")
                                        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                        .InstanceOutputShape("y", inputShapes[0].GetDims())
                                        .InstanceOutputDataType("y", inputDtypes[0])
                                        .InstanceOutputFormat("y", inputFormats[0])
                                        .Build();
        es::AddEdgeAndUpdatePeerDesc(*graphPtr, scatterNdUpdateNode, 0, tensorMoveAfterNode, 0);
        UpdateInputFormat(tensorMoveAfterNode, 0, inputFormats[0]);
        finalOutput = es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(tensorMoveAfterNode, 0));
    }

    // 9. 构建替换图
    GraphUniqPtr replaceGraph = builder.BuildAndReset({finalOutput});
    if (replaceGraph == nullptr) {
        OPS_LOG_E(PASS_NAME.c_str(), "BuildAndReset returned nullptr.");
        return nullptr;
    }

    // 10. InferShape
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OPS_LOG_W(PASS_NAME.c_str(), "InferShape failed, continue with manual desc.");
    }

    OPS_LOG_D(PASS_NAME.c_str(), "Replacement graph built successfully.");
    return replaceGraph;
}

REG_FUSION_PASS(TensorScatterUpdateFusionPass).Stage(CustomPassStage::kCompatibleInherited);
} // namespace NN
} // namespace OPS
