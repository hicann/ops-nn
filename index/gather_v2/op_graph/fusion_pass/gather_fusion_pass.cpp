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
 * \file gather_fusion_pass.cpp
 * \brief Gather --> GatherV2 fusion pass (regbase only)
 *
 *       x     indices                  x     indices   axis(=0)
 *        \      /                       \      |       /
 *        Gather            -->           GatherV2
 *          |                                |
 *          y                                y
 */

#include "gather_fusion_pass.h"
#include "es_nn_ops.h"
#include "compliant_node_builder.h"
#include "common/inc/error_util.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "ge/es_graph_builder.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace OPS {
namespace NN {
namespace {

const std::string PASS_NAME = "GatherToGatherV2FusionPass";
const int64_t CAPTURE_IDX_OUTPUT = 0l;
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
// Pattern：匹配 Gather 算子（CompliantNodeBuilder，Gather 无 ES API）
// ---------------------------------------------------------------------------
std::vector<PatternUniqPtr> GatherToGatherV2FusionPass::Patterns()
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Patterns for GatherToGatherV2FusionPass");
    std::vector<PatternUniqPtr> patterns;

    auto graphBuilder = es::EsGraphBuilder(PASS_NAME.c_str());
    auto x = graphBuilder.CreateInput(0);
    auto indices = graphBuilder.CreateInput(1);

    ge::Graph* graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();

    GNode opNode = es::CompliantNodeBuilder(graphPtr)
        .OpType("Gather")
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                       {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"batch_dims", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
             es::CreateFrom(static_cast<int64_t>(0))},
            {"is_preprocessed", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
             es::CreateFrom(false)},
            {"negative_index_support", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
             es::CreateFrom(false)}
        })
        .Build();

    GNode xNode = *x.GetProducer();
    GNode indicesNode = *indices.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, xNode, 0, opNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, indicesNode, 0, opNode, 1);

    es::EsTensorHolder output(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(opNode, 0));

    auto graph = graphBuilder.BuildAndReset({output});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});
    patterns.emplace_back(std::move(pattern));

    return patterns;
}

// ---------------------------------------------------------------------------
// MeetRequirements：校验平台
// ---------------------------------------------------------------------------
bool GatherToGatherV2FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter MeetRequirements");

    if (!IsTargetPlatform()) {
        OPS_LOG_D(PASS_NAME.c_str(), "Platform check failed, skip fusion.");
        return false;
    }

    NodeIo captured;
    OP_LOGE_IF(
        matchResult->GetCapturedTensor(CAPTURE_IDX_OUTPUT, captured) != SUCCESS,
        false, PASS_NAME.c_str(), "Failed to get captured tensor.");

    AscendString nodeType;
    captured.node.GetType(nodeType);
    std::string typeStr = nodeType.GetString();
    if (typeStr != "Gather") {
        OPS_LOG_D(PASS_NAME.c_str(), "Matched node type %s is not Gather, skip.", typeStr.c_str());
        return false;
    }

    return true;
}
// ---------------------------------------------------------------------------
// Replacement：构建替换图 GatherV2 (x, indices, axis=0)
// ---------------------------------------------------------------------------
GraphUniqPtr GatherToGatherV2FusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Replacement");

    // 1. 获取匹配节点信息
    NodeIo captured;
    if (matchResult->GetCapturedTensor(CAPTURE_IDX_OUTPUT, captured) != SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "Failed to get captured tensor.");
        return nullptr;
    }

    // 2. 获取原始 Gather 节点的属性
    int64_t batchDims = 0;
    bool isPreprocessed = false;
    bool negativeIndexSupport = false;
    captured.node.GetAttr("batch_dims", batchDims);
    captured.node.GetAttr("is_preprocessed", isPreprocessed);
    captured.node.GetAttr("negative_index_support", negativeIndexSupport);

    // 3. 获取子图边界输入信息
    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    // 4. 创建替换图
    auto builder = es::EsGraphBuilder("replacement");
    auto rX = builder.CreateInput(
        0, "x", inputDtypes[0], inputFormats[0], inputShapes[0].GetDims());
    auto rIndices = builder.CreateInput(
        1, "indices", inputDtypes[1], inputFormats[1], inputShapes[1].GetDims());

    // 5. 创建 axis 标量常量节点 (值为 0, 类型 INT64)
    auto rAxis = builder.CreateScalar(static_cast<int64_t>(0));

    // 6. 创建 GatherV2 节点（使用 ES API）
    auto gatherv2Output = es::GatherV2(rX, rIndices, rAxis,
                                         batchDims, isPreprocessed, negativeIndexSupport);

    // 6.1 刷新 GatherV2 节点的 Format（InferShape 不推导 Format，需手动刷新）
    GNode gatherv2Node = *gatherv2Output.GetProducer();
    // input[0]: x
    TensorDesc gv2Input0Desc;
    gatherv2Node.GetInputDesc(0, gv2Input0Desc);
    gv2Input0Desc.SetFormat(inputFormats[0]);
    gatherv2Node.UpdateInputDesc(0, gv2Input0Desc);
    // input[1]: indices
    TensorDesc gv2Input1Desc;
    gatherv2Node.GetInputDesc(1, gv2Input1Desc);
    gv2Input1Desc.SetFormat(inputFormats[1]);
    gatherv2Node.UpdateInputDesc(1, gv2Input1Desc);

    // 7. 构建替换图
    GraphUniqPtr replaceGraph = builder.BuildAndReset({gatherv2Output});
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

REG_FUSION_PASS(GatherToGatherV2FusionPass).Stage(CustomPassStage::kCompatibleInherited);
} // namespace NN
} // namespace OPS
