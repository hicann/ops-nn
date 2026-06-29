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
 * \file sorted_sparse_segment_mean_grad_fusion_pass.cpp
 * \brief fusion pass (SparseSegmentMeanGrad --> Sort + SortedSparseSegmentMeanGrad)
 *                                                                indices
 *  grad    indices   segment_ids  output_dim0                       |
 *    \        |          /          /                              Sort
 *          SparseSegmentMeanGrad                                 /      \
 *                |                            grad   sorted_indices  pre_location_indices  segment_ids  output_dim0
 *              output                ======>>>    \        \                  /                /             /
 *                                                        SortedSparseSegmentMeanGrad
 *                                                                    |
 *                                                                  output
 */

#include "sorted_sparse_segment_mean_grad_fusion_pass.h"

#include <array>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {
namespace {
const std::string kPassName = "SortedSparseSegmentMeanGradFusionPass";
const char *kSourceOpType = "SparseSegmentMeanGrad";

// SparseSegmentMeanGrad IR input index
const int64_t kSrcGradIndex = 0L;
const int64_t kSrcIndicesIndex = 1L;
const int64_t kSrcSegmentIdsIndex = 2L;
const int64_t kSrcOutputDim0Index = 3L;
const int64_t kSrcInputNum = 4L;

// captured tensor index in pattern
const int64_t kCaptureSource = 0L;

// Sort attribute, consistent with the original fusion pass
const int64_t kSortAxis = -1L;
const bool kSortDescending = false;
const bool kSortStable = false;

bool IsSupportedGradDtype(const DataType dtype)
{
    static const std::initializer_list<DataType> kSupportedDtypes = {DT_FLOAT, DT_FLOAT16, DT_BF16};
    return std::find(kSupportedDtypes.begin(), kSupportedDtypes.end(), dtype) != kSupportedDtypes.end();
}

// The original pass only fuses on regbase arch (IsRegBase, i.e. NpuArch DAV_3510 / DAV_5102). On the
// new framework these map to short soc versions Ascend950 / MC62CM12A respectively, consistent with the
// other passes migrated from IsRegBase (e.g. max_pool_v3).
inline bool IsRegbasePlatform()
{
    PlatformInfo info;
    OptionalInfo optInfo;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(info, optInfo) != SUCCESS,
        false, kPassName.c_str(), "Get platform_info failed.");
    const std::string socVersion = info.str_info.short_soc_version;
    bool isRegbasePlatform = (socVersion == "Ascend950" || socVersion == "MC62CM12A");
    OPS_LOG_D(kPassName.c_str(), "Platform short soc: %s, is_regbase: %d", socVersion.c_str(), isRegbasePlatform);
    return isRegbasePlatform;
}

// Build the SparseSegmentMeanGrad pattern node via CompliantNodeBuilder (no ES API for the source op).
es::EsTensorHolder CreatePatternSource(es::EsGraphBuilder &graphBuilder, const std::string &nodeName,
    const std::array<es::EsTensorHolder, kSrcInputNum> &inputs)
{
    auto *graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto sourceNode = es::CompliantNodeBuilder(graph)
                          .OpType(kSourceOpType)
                          .Name(nodeName.c_str())
                          .IrDefInputs({
                              {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"segment_ids", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"output_dim0", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
                          })
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();

    for (int64_t i = 0; i < kSrcInputNum; ++i) {
        OP_LOGE_IF(
            es::AddEdgeAndUpdatePeerDesc(*graph, *inputs[i].GetProducer(), inputs[i].GetProducerOutIndex(),
                sourceNode, static_cast<int32_t>(i)) != GRAPH_SUCCESS,
            es::EsTensorHolder(), kPassName.c_str(), "AddEdgeAndUpdatePeerDesc failed for input %ld.", i);
    }

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(sourceNode, 0);
    return es::EsTensorHolder(yHolder);
}

// Pass is registered at kAfterInferShape and the replacement contains an intermediate Sort node,
// so the replacement graph shapes must be inferred explicitly.
Status InferShape(const GraphUniqPtr &replaceGraph, const std::vector<SubgraphInput> &subgraphInputs)
{
    OPS_LOG_D(kPassName.c_str(), "Begin infershape for replacement.");
    std::vector<Shape> inputShapes;
    for (const auto &subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}
} // namespace

std::vector<PatternUniqPtr> SortedSparseSegmentMeanGradFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for SortedSparseSegmentMeanGradFusionPass");
    std::vector<PatternUniqPtr> patterns;

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    std::array<es::EsTensorHolder, kSrcInputNum> inputs;
    for (int64_t i = 0; i < kSrcInputNum; ++i) {
        inputs[i] = graphBuilder.CreateInput(static_cast<int32_t>(i));
    }
    auto y = CreatePatternSource(graphBuilder, std::string(kSourceOpType) + "Pattern", inputs);
    auto graph = graphBuilder.BuildAndReset({y});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*y.GetProducer(), kCaptureSource});
    patterns.emplace_back(std::move(pattern));

    OPS_LOG_D(kPassName.c_str(), "End Patterns for SortedSparseSegmentMeanGradFusionPass");
    return patterns;
}

bool SortedSparseSegmentMeanGradFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for SortedSparseSegmentMeanGradFusionPass");

    if (!IsRegbasePlatform()) {
        OPS_LOG_D(kPassName.c_str(), "Not regbase arch, no need fusion.");
        return false;
    }

    NodeIo sourceIo;
    OP_LOGE_IF(match_result->GetCapturedTensor(kCaptureSource, sourceIo) != SUCCESS, false, kPassName.c_str(),
        "Failed to get captured source tensor.");

    TensorDesc gradDesc;
    OP_LOGE_IF(sourceIo.node.GetInputDesc(kSrcGradIndex, gradDesc) != SUCCESS, false, kPassName.c_str(),
        "Failed to get grad input desc.");
    if (!IsSupportedGradDtype(gradDesc.GetDataType())) {
        OPS_LOG_D(kPassName.c_str(), "grad dtype %d is not supported.", gradDesc.GetDataType());
        return false;
    }

    return true;
}

std::unique_ptr<Graph> SortedSparseSegmentMeanGradFusionPass::Replacement(
    const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for SortedSparseSegmentMeanGradFusionPass");

    NodeIo sourceIo;
    OP_LOGE_IF(match_result->GetCapturedTensor(kCaptureSource, sourceIo) != SUCCESS, nullptr, kPassName.c_str(),
        "Failed to get captured source tensor.");
    GNode sourceNode = sourceIo.node;

    std::vector<SubgraphInput> subgraphInputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    // Collect source input descs.
    TensorDesc gradDesc;
    TensorDesc indicesDesc;
    TensorDesc segmentIdsDesc;
    TensorDesc outputDim0Desc;
    OP_LOGE_IF(sourceNode.GetInputDesc(kSrcGradIndex, gradDesc) != SUCCESS, nullptr, kPassName.c_str(),
        "Failed to get grad input desc.");
    OP_LOGE_IF(sourceNode.GetInputDesc(kSrcIndicesIndex, indicesDesc) != SUCCESS, nullptr, kPassName.c_str(),
        "Failed to get indices input desc.");
    OP_LOGE_IF(sourceNode.GetInputDesc(kSrcSegmentIdsIndex, segmentIdsDesc) != SUCCESS, nullptr, kPassName.c_str(),
        "Failed to get segment_ids input desc.");
    OP_LOGE_IF(sourceNode.GetInputDesc(kSrcOutputDim0Index, outputDim0Desc) != SUCCESS, nullptr, kPassName.c_str(),
        "Failed to get output_dim0 input desc.");

    TensorDesc outputDesc;
    OP_LOGE_IF(sourceNode.GetOutputDesc(0, outputDesc) != SUCCESS, nullptr, kPassName.c_str(),
        "Failed to get output desc.");

    // Sort.y2 dtype follows the indices dtype: int32 -> INT32, otherwise INT64. Same as the original pass.
    DataType y2Dtype = (indicesDesc.GetDataType() == DT_INT32) ? DT_INT32 : DT_INT64;

    auto graphBuilder = es::EsGraphBuilder("replacement");
    auto grad = graphBuilder.CreateInput(0, "x", gradDesc.GetDataType(), gradDesc.GetFormat(),
        gradDesc.GetShape().GetDims());
    auto indices = graphBuilder.CreateInput(1, "indices", indicesDesc.GetDataType(), indicesDesc.GetFormat(),
        indicesDesc.GetShape().GetDims());
    auto segmentIds = graphBuilder.CreateInput(2, "segment_ids", segmentIdsDesc.GetDataType(),
        segmentIdsDesc.GetFormat(), segmentIdsDesc.GetShape().GetDims());
    auto outputDim0 = graphBuilder.CreateInput(3, "output_dim0", outputDim0Desc.GetDataType(),
        outputDim0Desc.GetFormat(), outputDim0Desc.GetShape().GetDims());

    // Step 1: Sort the indices, producing sorted_indices (y1) and pre_location_indices (y2).
    auto sortOut = es::Sort(indices, kSortAxis, kSortDescending, kSortStable, y2Dtype);

    // Step 2: feed grad, sorted_indices, pre_location_indices, segment_ids, output_dim0 into the target op.
    auto y = es::SortedSparseSegmentMeanGrad(grad, sortOut.y1, sortOut.y2, segmentIds, outputDim0);

    // Restore output desc/dtype on the target node.
    GNode targetNode = *y.GetProducer();
    targetNode.UpdateOutputDesc(0, outputDesc);

    auto replaceGraph = graphBuilder.BuildAndReset({y});

    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }

    OPS_LOG_D(kPassName.c_str(), "End Replacement for SortedSparseSegmentMeanGradFusionPass");
    return replaceGraph;
}

REG_FUSION_PASS(SortedSparseSegmentMeanGradFusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ops