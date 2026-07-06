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
 * \file inplace_add_fusion_pass.cpp
 * \brief InplaceAdd fusion pass (InplaceAdd --> TensorMove & ScatterAdd)
 */

#include "inplace_add_fusion_pass.h"

#include <algorithm>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"
#include "acl/acl_rt.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {
namespace {
const std::string kPassName = "AInplaceAddFusionPass";
const char* kInplaceAddType = "InplaceAdd";
const char* kNotSupportSoc = "Ascend310";
const int64_t kCaptureInplaceAdd = 0L;
const int32_t kInplaceAddInputNum = 3;
// Version compat (D1): kCompatibleInherited is a GE 9.0.0 (90000000) Stage enum.
const int32_t kGeCompilerVersion900 = 90000000;

// Runtime GE version. Returns 0 if query fails (treated as legacy / not-support).
int32_t GetGeCompilerVersionNum()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    return version;
}

// Version compat (D1): kCompatibleInherited is a GE 9.0.0 enum.
// - compile with GE < 9.0.0: enum undefined, register to legacy kBeforeInferShape so it still builds.
// - compile with GE >= 9.0.0: at runtime, use kCompatibleInherited on 9.x; on a 8.5.0 runtime fall
//   back to the legacy kBeforeInferShape stage and run empty (MeetRequirements returns false).
CustomPassStage GetInplaceAddPassStage()
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90000000)
    if (GetGeCompilerVersionNum() >= kGeCompilerVersion900) {
        return CustomPassStage::kCompatibleInherited;
    }
    return CustomPassStage::kBeforeInferShape;
#else
    return CustomPassStage::kBeforeInferShape;
#endif
}

// ScatterAdd var/updates supported dtypes, aligned with ScatterAdd IR definition.
bool IsSupportedDataDtype(const DataType dtype)
{
    static const std::initializer_list<DataType> kSupportedDataDtypes = {DT_FLOAT16, DT_FLOAT, DT_INT32,
                                                                         DT_INT8,    DT_UINT8, DT_BF16};
    return std::find(kSupportedDataDtypes.begin(), kSupportedDataDtypes.end(), dtype) != kSupportedDataDtypes.end();
}

// ScatterAdd indices supported dtypes (IndexNumberType: int32/int64).
bool IsSupportedIndicesDtype(const DataType dtype)
{
    static const std::initializer_list<DataType> kSupportedIndicesDtypes = {DT_INT32, DT_INT64};
    return std::find(kSupportedIndicesDtypes.begin(), kSupportedIndicesDtypes.end(), dtype) !=
           kSupportedIndicesDtypes.end();
}

// The original pass only excludes Ascend310; every other platform keeps the fusion.
bool IsSupportSoc()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGW(kPassName.c_str(), "Get platform info failed, not fusion.");
        return false;
    }
    const std::string curSoc = platformInfo.str_info.short_soc_version;
    OPS_LOG_D(kPassName.c_str(), "cur_soc is %s", curSoc.c_str());
    if (curSoc == kNotSupportSoc) {
        OPS_LOG_D(kPassName.c_str(), "cur_soc %s not support.", curSoc.c_str());
        return false;
    }
    return true;
}

void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs, std::vector<Shape>& inputShapes,
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

// InferShape does not deduce Format, refresh the input Format manually.
void UpdateInputFormat(GNode& node, uint32_t idx, Format format)
{
    TensorDesc desc;
    node.GetInputDesc(idx, desc);
    desc.SetFormat(format);
    node.UpdateInputDesc(idx, desc);
}

Status InferShape(const std::unique_ptr<Graph>& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    OPS_LOG_D(kPassName.c_str(), "Begin infershape for replacement.");
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

// InplaceAdd has no ES API, build the pattern node via CompliantNodeBuilder.
// IR: InplaceAdd(x, indices, v) -> y
es::EsTensorHolder CreatePatternInplaceAdd(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x,
                                           const es::EsTensorHolder& indices, const es::EsTensorHolder& v)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto inplaceAdd = es::CompliantNodeBuilder(graph)
                          .OpType(kInplaceAddType)
                          .Name((std::string(kInplaceAddType) + "Pattern").c_str())
                          .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"v", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();

    OP_LOGE_IF(
        es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), inplaceAdd, 0) != GRAPH_SUCCESS,
        es::EsTensorHolder(), kPassName.c_str(), "AddEdge for InplaceAdd input x failed.");
    OP_LOGE_IF(es::AddEdgeAndUpdatePeerDesc(*graph, *indices.GetProducer(), indices.GetProducerOutIndex(), inplaceAdd,
                                            1) != GRAPH_SUCCESS,
               es::EsTensorHolder(), kPassName.c_str(), "AddEdge for InplaceAdd input indices failed.");
    OP_LOGE_IF(
        es::AddEdgeAndUpdatePeerDesc(*graph, *v.GetProducer(), v.GetProducerOutIndex(), inplaceAdd, 2) != GRAPH_SUCCESS,
        es::EsTensorHolder(), kPassName.c_str(), "AddEdge for InplaceAdd input v failed.");

    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(inplaceAdd, 0);
    return es::EsTensorHolder(yHolder);
}
} // namespace
std::vector<PatternUniqPtr> AInplaceAddFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for AInplaceAddFusionPass.");
    std::vector<PatternUniqPtr> patternGraphs;

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto x = graphBuilder.CreateInput(0, "x");
    auto indices = graphBuilder.CreateInput(1, "indices");
    auto v = graphBuilder.CreateInput(2, "v");
    auto y = CreatePatternInplaceAdd(graphBuilder, x, indices, v);
    auto graph = graphBuilder.BuildAndReset({y});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*y.GetProducer(), 0});
    patternGraphs.emplace_back(std::move(pattern));

    OPS_LOG_D(kPassName.c_str(), "End Patterns for AInplaceAddFusionPass.");
    return patternGraphs;
}

bool AInplaceAddFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for AInplaceAddFusionPass.");

    // Version compat (D1): this pass registers on kCompatibleInherited (GE 9.0.0). On a GE 8.5.0
    // runtime it is registered to the legacy kBeforeInferShape stage and must run empty here.
    if (GetGeCompilerVersionNum() < kGeCompilerVersion900) {
        OPS_LOG_D(kPassName.c_str(), "GE runtime < 9.0.0, skip fusion (compat empty run).");
        return false;
    }

    // Platform check: keep the original behavior, only exclude Ascend310.
    if (!IsSupportSoc()) {
        return false;
    }

    NodeIo inplaceAddIo;
    OP_LOGE_IF(match_result->GetCapturedTensor(kCaptureInplaceAdd, inplaceAddIo) != SUCCESS, false, kPassName.c_str(),
               "Failed to get captured InplaceAdd node.");
    GNode sourceNode = inplaceAddIo.node;

    AscendString nodeType;
    sourceNode.GetType(nodeType);
    if (std::string(nodeType.GetString()) != kInplaceAddType) {
        OPS_LOG_D(kPassName.c_str(), "Captured node type %s is not InplaceAdd, skip.", nodeType.GetString());
        return false;
    }

    // The fused ScatterAdd must support the dtypes of var(x)/indices/updates(v).
    if (sourceNode.GetInputsSize() != static_cast<size_t>(kInplaceAddInputNum)) {
        OPS_LOG_D(kPassName.c_str(), "InplaceAdd input num %zu is not %d, skip.", sourceNode.GetInputsSize(),
                  kInplaceAddInputNum);
        return false;
    }

    TensorDesc xDesc;
    OP_LOGE_IF(sourceNode.GetInputDesc(0, xDesc) != SUCCESS, false, kPassName.c_str(), "Get input x desc failed.");
    TensorDesc indicesDesc;
    OP_LOGE_IF(sourceNode.GetInputDesc(1, indicesDesc) != SUCCESS, false, kPassName.c_str(),
               "Get input indices desc failed.");
    TensorDesc vDesc;
    OP_LOGE_IF(sourceNode.GetInputDesc(2, vDesc) != SUCCESS, false, kPassName.c_str(), "Get input v desc failed.");

    if (!IsSupportedDataDtype(xDesc.GetDataType()) || !IsSupportedDataDtype(vDesc.GetDataType()) ||
        !IsSupportedIndicesDtype(indicesDesc.GetDataType())) {
        OPS_LOG_D(kPassName.c_str(), "ScatterAdd does not support input dtype (x:%d, indices:%d, v:%d), skip.",
                  xDesc.GetDataType(), indicesDesc.GetDataType(), vDesc.GetDataType());
        return false;
    }

    OPS_LOG_D(kPassName.c_str(), "End MeetRequirements for AInplaceAddFusionPass.");
    return true;
}

std::unique_ptr<Graph> AInplaceAddFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for AInplaceAddFusionPass.");

    std::vector<SubgraphInput> subgraphInputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);
    OP_LOGE_IF(inputShapes.size() != static_cast<size_t>(kInplaceAddInputNum), nullptr, kPassName.c_str(),
               "Replacement got %zu boundary inputs, expect %d.", inputShapes.size(), kInplaceAddInputNum);

    NodeIo inplaceAddIo;
    OP_LOGE_IF(match_result->GetCapturedTensor(kCaptureInplaceAdd, inplaceAddIo) != SUCCESS, nullptr, kPassName.c_str(),
               "Failed to get captured InplaceAdd node in Replacement.");
    GNode sourceNode = inplaceAddIo.node;

    TensorDesc outputDesc;
    OP_LOGE_IF(sourceNode.GetOutputDesc(0, outputDesc) != SUCCESS, nullptr, kPassName.c_str(),
               "Get InplaceAdd output desc failed.");

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto repX = replaceGraphBuilder.CreateInput(0, "x", inputDtypes[0], inputFormats[0], inputShapes[0].GetDims());
    auto repIndices = replaceGraphBuilder.CreateInput(1, "indices", inputDtypes[1], inputFormats[1],
                                                      inputShapes[1].GetDims());
    auto repUpdates = replaceGraphBuilder.CreateInput(2, "updates", inputDtypes[2], inputFormats[2],
                                                      inputShapes[2].GetDims());

    auto* graph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();

    // TensorMove is a math op with no ES API in this op_graph build, build it via CompliantNodeBuilder.
    GNode tensorMoveNode = es::CompliantNodeBuilder(graph)
                               .OpType("TensorMove")
                               .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                               .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                               .InstanceOutputShape("y", inputShapes[0].GetDims())
                               .InstanceOutputDataType("y", inputDtypes[0])
                               .InstanceOutputFormat("y", inputFormats[0])
                               .Build();

    // x -> TensorMove
    GNode xNode = *repX.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graph, xNode, 0, tensorMoveNode, 0);
    UpdateInputFormat(tensorMoveNode, 0, inputFormats[0]);

    // ScatterAdd(var, indices, updates, use_locking=false), var comes from TensorMove output.
    es::EsTensorHolder tensorMoveOut(
        replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(tensorMoveNode, 0));
    auto scatterAddOut = es::ScatterAdd(tensorMoveOut, repIndices, repUpdates, false);

    GNode scatterAddNode = *scatterAddOut.GetProducer();
    UpdateInputFormat(scatterAddNode, 0, inputFormats[0]);
    UpdateInputFormat(scatterAddNode, 1, inputFormats[1]);
    UpdateInputFormat(scatterAddNode, 2, inputFormats[2]);
    scatterAddNode.UpdateOutputDesc(0, outputDesc);

    // Mark this ScatterAdd as the inplace fusion variant, same as the original pass.
    AscendString inplaceOptionKey = "fusion_op_build_options";
    AscendString inplaceOptionValue = "{\"is_inplace\",True}";
    scatterAddNode.SetAttr(inplaceOptionKey, inplaceOptionValue);

    auto replaceGraph = replaceGraphBuilder.BuildAndReset({scatterAddOut});
    if (replaceGraph == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "BuildAndReset returned nullptr.");
        return nullptr;
    }

    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OPS_LOG_W(kPassName.c_str(), "InferShape for replacement failed, continue with manual desc.");
    }

    OPS_LOG_D(kPassName.c_str(), "End Replacement for AInplaceAddFusionPass.");
    return replaceGraph;
}

REG_FUSION_PASS(AInplaceAddFusionPass).Stage(GetInplaceAddPassStage());
} // namespace ops