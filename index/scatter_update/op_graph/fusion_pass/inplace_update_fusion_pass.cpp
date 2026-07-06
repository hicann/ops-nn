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
 * \file inplace_update_fusion_pass.cpp
 * \brief InplaceUpdate fusion pass (InplaceUpdate --> TensorMove & ScatterUpdate)
 */

#include "inplace_update_fusion_pass.h"

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
const std::string kPassName = "AInplaceUpdateFusionPass";
const char* kInplaceUpdateType = "InplaceUpdate";
const char* kNotSupportSoc = "Ascend310";
const int64_t kCaptureInplaceUpdate = 0L;
const int32_t kInplaceUpdateInputNum = 3;

// ScatterUpdate var/updates supported dtypes, aligned with ScatterUpdate IR definition.
bool IsSupportedDataDtype(const DataType dtype)
{
    static const std::initializer_list<DataType> kSupportedDataDtypes = {
        DT_FLOAT16, DT_FLOAT,  DT_INT32,  DT_INT8,        DT_UINT8,         DT_BF16,
        DT_INT64,   DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0};
    return std::find(kSupportedDataDtypes.begin(), kSupportedDataDtypes.end(), dtype) != kSupportedDataDtypes.end();
}

// ScatterUpdate indices supported dtypes (IndexNumberType: int32/int64).
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

// InplaceUpdate has no ES API, build the pattern node via CompliantNodeBuilder.
// IR: InplaceUpdate(x, indices, v) -> y
es::EsTensorHolder CreatePatternInplaceUpdate(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x,
                                              const es::EsTensorHolder& indices, const es::EsTensorHolder& v)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto inplaceUpdate = es::CompliantNodeBuilder(graph)
                             .OpType(kInplaceUpdateType)
                             .Name((std::string(kInplaceUpdateType) + "Pattern").c_str())
                             .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                           {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                           {"v", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();

    OP_LOGE_IF(es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), inplaceUpdate, 0) !=
                   GRAPH_SUCCESS,
               es::EsTensorHolder(), kPassName.c_str(), "AddEdge for InplaceUpdate input x failed.");
    OP_LOGE_IF(es::AddEdgeAndUpdatePeerDesc(*graph, *indices.GetProducer(), indices.GetProducerOutIndex(),
                                            inplaceUpdate, 1) != GRAPH_SUCCESS,
               es::EsTensorHolder(), kPassName.c_str(), "AddEdge for InplaceUpdate input indices failed.");
    OP_LOGE_IF(es::AddEdgeAndUpdatePeerDesc(*graph, *v.GetProducer(), v.GetProducerOutIndex(), inplaceUpdate, 2) !=
                   GRAPH_SUCCESS,
               es::EsTensorHolder(), kPassName.c_str(), "AddEdge for InplaceUpdate input v failed.");

    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(inplaceUpdate, 0);
    return es::EsTensorHolder(yHolder);
}
} // namespace

std::vector<PatternUniqPtr> AInplaceUpdateFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for AInplaceUpdateFusionPass.");
    std::vector<PatternUniqPtr> patternGraphs;

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto x = graphBuilder.CreateInput(0, "x");
    auto indices = graphBuilder.CreateInput(1, "indices");
    auto v = graphBuilder.CreateInput(2, "v");
    auto y = CreatePatternInplaceUpdate(graphBuilder, x, indices, v);
    auto graph = graphBuilder.BuildAndReset({y});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*y.GetProducer(), 0});
    patternGraphs.emplace_back(std::move(pattern));

    OPS_LOG_D(kPassName.c_str(), "End Patterns for AInplaceUpdateFusionPass.");
    return patternGraphs;
}

bool AInplaceUpdateFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for AInplaceUpdateFusionPass.");

    // Compat (D1): on an 8.5.0 runtime the pass is registered to the old stage kBeforeInferShape
    // (see GetInplaceUpdateFusionPassStage). kCompatibleInherited only exists from 9.0.0, so when
    // running on an older runtime we no-op here to keep the overall-silent behavior.
    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < 90000000) {
        OPS_LOG_D(kPassName.c_str(), "ge_compiler runtime %d < 90000000, skip fusion (overall-silent).",
                  geCompilerVersion);
        return false;
    }

    // Platform check: keep the original behavior, only exclude Ascend310.
    if (!IsSupportSoc()) {
        return false;
    }

    NodeIo inplaceUpdateIo;
    OP_LOGE_IF(match_result->GetCapturedTensor(kCaptureInplaceUpdate, inplaceUpdateIo) != SUCCESS, false,
               kPassName.c_str(), "Failed to get captured InplaceUpdate node.");
    GNode sourceNode = inplaceUpdateIo.node;

    AscendString nodeType;
    sourceNode.GetType(nodeType);
    if (std::string(nodeType.GetString()) != kInplaceUpdateType) {
        OPS_LOG_D(kPassName.c_str(), "Captured node type %s is not InplaceUpdate, skip.", nodeType.GetString());
        return false;
    }

    // The fused ScatterUpdate must support the dtypes of var(x)/indices/updates(v).
    if (sourceNode.GetInputsSize() != static_cast<size_t>(kInplaceUpdateInputNum)) {
        OPS_LOG_D(kPassName.c_str(), "InplaceUpdate input num %zu is not %d, skip.", sourceNode.GetInputsSize(),
                  kInplaceUpdateInputNum);
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
        OPS_LOG_D(kPassName.c_str(), "ScatterUpdate does not support input dtype (x:%d, indices:%d, v:%d), skip.",
                  xDesc.GetDataType(), indicesDesc.GetDataType(), vDesc.GetDataType());
        return false;
    }

    OPS_LOG_D(kPassName.c_str(), "End MeetRequirements for AInplaceUpdateFusionPass.");
    return true;
}

std::unique_ptr<Graph> AInplaceUpdateFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for AInplaceUpdateFusionPass.");

    std::vector<SubgraphInput> subgraphInputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);
    OP_LOGE_IF(inputShapes.size() != static_cast<size_t>(kInplaceUpdateInputNum), nullptr, kPassName.c_str(),
               "Replacement got %zu boundary inputs, expect %d.", inputShapes.size(), kInplaceUpdateInputNum);

    NodeIo inplaceUpdateIo;
    OP_LOGE_IF(match_result->GetCapturedTensor(kCaptureInplaceUpdate, inplaceUpdateIo) != SUCCESS, nullptr,
               kPassName.c_str(), "Failed to get captured InplaceUpdate node in Replacement.");
    GNode sourceNode = inplaceUpdateIo.node;

    TensorDesc outputDesc;
    OP_LOGE_IF(sourceNode.GetOutputDesc(0, outputDesc) != SUCCESS, nullptr, kPassName.c_str(),
               "Get InplaceUpdate output desc failed.");

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

    // ScatterUpdate(var, indices, updates, use_locking=false), var comes from TensorMove output.
    es::EsTensorHolder tensorMoveOut(
        replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(tensorMoveNode, 0));
    auto scatterUpdateOut = es::ScatterUpdate(tensorMoveOut, repIndices, repUpdates, false);

    GNode scatterUpdateNode = *scatterUpdateOut.GetProducer();
    UpdateInputFormat(scatterUpdateNode, 0, inputFormats[0]);
    UpdateInputFormat(scatterUpdateNode, 1, inputFormats[1]);
    UpdateInputFormat(scatterUpdateNode, 2, inputFormats[2]);
    scatterUpdateNode.UpdateOutputDesc(0, outputDesc);

    // Mark this ScatterUpdate as the inplace fusion variant, same as the original pass.
    AscendString inplaceOptionKey = "fusion_op_build_options";
    AscendString inplaceOptionValue = "{\"is_inplace\",True}";
    scatterUpdateNode.SetAttr(inplaceOptionKey, inplaceOptionValue);

    auto replaceGraph = replaceGraphBuilder.BuildAndReset({scatterUpdateOut});
    if (replaceGraph == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "BuildAndReset returned nullptr.");
        return nullptr;
    }

    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OPS_LOG_W(kPassName.c_str(), "InferShape for replacement failed, continue with manual desc.");
    }

    OPS_LOG_D(kPassName.c_str(), "End Replacement for AInplaceUpdateFusionPass.");
    return replaceGraph;
}

// Compat (D1): kCompatibleInherited is born in 9.0.0 (GE_COMPILER_VERSION_NUM 90000000).
// To keep compiling on toolkit 8.5.0 (old header without this enum), guard the whole
// registration with the compile macro. To keep a 9.x-compiled .so runnable on an 8.5.0
// runtime (which does not know kCompatibleInherited), pick the stage at runtime: fall back
// to an old stage on 8.5.0 and make MeetRequirements no-op there (overall-silent strategy).
#if GE_COMPILER_VERSION_NUM >= 90000000
namespace {
CustomPassStage GetInplaceUpdateFusionPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= 90000000) {
        return CustomPassStage::kCompatibleInherited; // expected stage on 9.x runtime
    }
    return CustomPassStage::kBeforeInferShape; // old stage on 8.5.0 runtime, used for no-op
}
} // namespace
REG_FUSION_PASS(AInplaceUpdateFusionPass).Stage(GetInplaceUpdateFusionPassStage());
#else
// 8.5.0 compile: kCompatibleInherited enum does not exist, register to the 8.5.0 stage
// kAfterInferShape. This pass uses only 8.5.0-compatible APIs internally, so 8.5.0 compile
// + 8.5.0 runtime fuses normally (aligned with inplace_add / inplace_sub).
REG_FUSION_PASS(AInplaceUpdateFusionPass).Stage(CustomPassStage::kAfterInferShape);
#endif
} // namespace ops