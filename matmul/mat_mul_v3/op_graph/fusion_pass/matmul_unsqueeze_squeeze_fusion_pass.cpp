/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_unsqueeze_squeeze_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "platform/platform_info.h"
#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "ge/ge_utils.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;

namespace ops {
namespace {

constexpr char kPassName[] = "MatMulUnsqueezeSqueezeFusionPass";
constexpr char kOpTypeAscendDequant[] = "AscendDequant";
constexpr int64_t kSqueezeAxisOffsetX1 = 2;
constexpr int64_t kGeCompilerVersion900 = 90000000;

struct FusionParams {
    size_t dimNumX1 = 0;
    size_t dimNumX2 = 0;
    bool bothOneDim = false;
    bool needSqueeze = false;
    bool isBatch = false;
    bool transX1 = false;
    bool transX2 = false;
};

EsTensorHolder CreateUnsqueezeNode(EsGraphBuilder& builder, const EsTensorHolder& input, int64_t axis)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    std::string nodeName = "unsqueeze_" + std::to_string(axis);
    auto node = CompliantNodeBuilder(graph)
                    .OpType("Unsqueeze")
                    .Name(nodeName.c_str())
                    .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({{"axes", CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                  CreateFrom(std::vector<int64_t>{axis})}})
                    .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), node, 0);

    TensorDesc inDesc;
    input.GetProducer()->GetOutputDesc(input.GetProducerOutIndex(), inDesc);
    auto inDims = inDesc.GetShape().GetDims();
    inDims.insert(inDims.begin() + axis, 1);
    TensorDesc outDesc(ge::Shape(inDims), inDesc.GetFormat(), inDesc.GetDataType());
    outDesc.SetOriginShape(ge::Shape(inDims));
    outDesc.SetOriginFormat(inDesc.GetFormat());
    node.UpdateOutputDesc(0, outDesc);

    auto* yHolder = builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0);
    return EsTensorHolder(yHolder);
}

EsTensorHolder CreateSqueezeNode(EsGraphBuilder& builder, const EsTensorHolder& input, int64_t axis)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = CompliantNodeBuilder(graph)
                    .OpType("Squeeze")
                    .Name("squeeze")
                    .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({{"axis", CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                  CreateFrom(std::vector<int64_t>{axis})}})
                    .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), node, 0);

    TensorDesc inDesc;
    input.GetProducer()->GetOutputDesc(input.GetProducerOutIndex(), inDesc);
    auto inDims = inDesc.GetShape().GetDims();
    if (axis >= 0 && axis < static_cast<int64_t>(inDims.size())) {
        inDims.erase(inDims.begin() + axis);
    }
    TensorDesc outDesc(ge::Shape(inDims), inDesc.GetFormat(), inDesc.GetDataType());
    outDesc.SetOriginShape(ge::Shape(inDims));
    outDesc.SetOriginFormat(inDesc.GetFormat());
    node.UpdateOutputDesc(0, outDesc);

    auto* yHolder = builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0);
    return EsTensorHolder(yHolder);
}

EsTensorHolder CreateAscendDequantNode(EsGraphBuilder& builder, const EsTensorHolder& input,
                                       const EsTensorHolder& deqScale, bool sqrtMode, bool reluFlag, int64_t dtype)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = CompliantNodeBuilder(graph)
                    .OpType(kOpTypeAscendDequant)
                    .Name("dequant")
                    .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"deq_scale", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({{"sqrt_mode", CompliantNodeBuilder::kEsAttrOptional, "Bool", CreateFrom(sqrtMode)},
                                 {"relu_flag", CompliantNodeBuilder::kEsAttrOptional, "Bool", CreateFrom(reluFlag)},
                                 {"dtype", CompliantNodeBuilder::kEsAttrOptional, "Int", CreateFrom(dtype)}})
                    .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *deqScale.GetProducer(), deqScale.GetProducerOutIndex(), node, 1);
    auto* yHolder = builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0);
    return EsTensorHolder(yHolder);
}

Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    std::vector<ge::Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        const auto& allInputs = subgraphInput.GetAllInputs();
        if (allInputs.empty()) {
            continue;
        }
        auto matchNode = allInputs[0];
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

bool IsBatchMatMulType(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS) {
        return false;
    }
    return opType == kOpTypeBatchMatMul || opType == kOpTypeBatchMatMulV2;
}

constexpr int32_t kMatMulIrInputNum = 3;
constexpr int32_t kBatchMatMulIrInputNum = 2;

int64_t GetIrInputNum(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS) {
        return kFourInputNum;
    }
    if (opType == kOpTypeMatMul) {
        return std::max(static_cast<int64_t>(node.GetInputsSize()), static_cast<int64_t>(kMatMulIrInputNum));
    }
    if (opType == kOpTypeBatchMatMul) {
        return std::max(static_cast<int64_t>(node.GetInputsSize()), static_cast<int64_t>(kBatchMatMulIrInputNum));
    }
    return kFourInputNum;
}

bool GetMatchedNodeAttrs(const GNode& matchedMatmulNode, bool& isBatch, bool& transX1, bool& transX2)
{
    isBatch = IsBatchMatMulType(matchedMatmulNode);
    const char* transStrX1 = isBatch ? "adj_x1" : "transpose_x1";
    const char* transStrX2 = isBatch ? "adj_x2" : "transpose_x2";
    if (matchedMatmulNode.GetAttr(transStrX1, transX1) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName, "Get %s from node failed.", transStrX1);
        return false;
    }
    if (matchedMatmulNode.GetAttr(transStrX2, transX2) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName, "Get %s from node failed.", transStrX2);
        return false;
    }
    return true;
}

TensorDesc CalcMatMulOutputDesc(size_t dimNumX1, size_t dimNumX2, bool bothOneDim, const TensorDesc& srcOutputDesc)
{
    auto matmulOutDims = srcOutputDesc.GetShape().GetDims();
    if (matmulOutDims.empty()) {
        OPS_LOG_E(kPassName, "Matmul output dims is empty.");
        return srcOutputDesc;
    }

    TensorDesc matmulOutDesc;
    matmulOutDesc.SetDataType(srcOutputDesc.GetDataType());
    matmulOutDesc.SetFormat(srcOutputDesc.GetFormat());
    matmulOutDesc.SetOriginFormat(srcOutputDesc.GetOriginFormat());

    if (bothOneDim) {
        matmulOutDims = {1, 1};
    } else if (dimNumX1 == 1) {
        matmulOutDims.insert(matmulOutDims.cend() - 1, 1);
    } else if (dimNumX2 == 1) {
        matmulOutDims.insert(matmulOutDims.cend(), 1);
    }
    matmulOutDesc.SetShape(ge::Shape(matmulOutDims));
    matmulOutDesc.SetOriginShape(ge::Shape(matmulOutDims));
    return matmulOutDesc;
}

EsTensorHolder CreateReplaceInput(EsGraphBuilder& builder, int64_t idx, const char* name, const TensorDesc& srcDesc)
{
    auto input = builder.CreateInput(idx, name, srcDesc.GetDataType(), srcDesc.GetFormat(),
                                     srcDesc.GetShape().GetDims());
    GNode inputNode = *input.GetProducer();
    TensorDesc desc;
    inputNode.GetOutputDesc(0, desc);
    desc.SetDataType(srcDesc.GetDataType());
    desc.SetFormat(srcDesc.GetFormat());
    desc.SetShape(srcDesc.GetShape());
    desc.SetOriginShape(srcDesc.GetShape());
    desc.SetOriginFormat(srcDesc.GetFormat());
    inputNode.UpdateOutputDesc(0, desc);
    return input;
}

int64_t CreateReplaceInputs(const GNode& matchedMatmulNode, EsGraphBuilder& builder, EsTensorHolder replaceInput[])
{
    const char* inputNames[] = {"x1", "x2", "bias", "offset_w"};
    int64_t irInputNum = GetIrInputNum(matchedMatmulNode);
    TensorDesc matchedInputDesc;
    int64_t createIdx = 0;
    int64_t loopCount = std::min(irInputNum, kFourInputNum);
    for (int64_t idx = 0; idx < loopCount; idx++) {
        if (matchedMatmulNode.GetInputDesc(idx, matchedInputDesc) != GRAPH_SUCCESS) {
            continue;
        }
        replaceInput[idx] = CreateReplaceInput(builder, createIdx, inputNames[idx], matchedInputDesc);
        createIdx++;
    }
    return createIdx;
}

int64_t CalcSqueezeAxis(size_t dimNumX1, size_t dimNumYNew)
{
    return dimNumX1 == 1 ? static_cast<int64_t>(dimNumYNew) - kSqueezeAxisOffsetX1 :
                           static_cast<int64_t>(dimNumYNew) - 1;
}

bool IsSupportL0c2out(const PlatformInfo& platformInfo)
{
    return platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_fix_pipe_l0c2out") !=
           platformInfo.ai_core_intrinsic_dtype_map.end();
}

bool IsMatMulDequantScenario(const GNode& matchedMatmulNode, bool supportL0c2out)
{
    if (!supportL0c2out) {
        return false;
    }
    auto outNodes = matchedMatmulNode.GetOutDataNodesAndPortIndexs(0);
    if (outNodes.size() != 1 || outNodes[0].first == nullptr) {
        return false;
    }
    AscendString outType;
    if (outNodes[0].first->GetType(outType) != GRAPH_SUCCESS) {
        return false;
    }
    return outType == kOpTypeAscendDequant;
}

EsTensorHolder BuildDequantReplacement(EsGraphBuilder& builder, const GNode& matchedDequantNode,
                                       const EsTensorHolder& rMatMul, int64_t deqScaleIdx)
{
    TensorDesc deqScaleDesc;
    FUSION_PASS_CHECK(matchedDequantNode.GetInputDesc(1, deqScaleDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get deq_scale input desc failed."), return nullptr);

    auto rDeqScale = CreateReplaceInput(builder, deqScaleIdx, "deq_scale", deqScaleDesc);

    bool sqrtMode = false;
    bool reluFlag = false;
    int64_t deqDtype = static_cast<int64_t>(DT_FLOAT16);
    matchedDequantNode.GetAttr("sqrt_mode", sqrtMode);
    matchedDequantNode.GetAttr("relu_flag", reluFlag);
    matchedDequantNode.GetAttr("dtype", deqDtype);

    auto rDequant = CreateAscendDequantNode(builder, rMatMul, rDeqScale, sqrtMode, reluFlag, deqDtype);
    CopyOtherAttrs(matchedDequantNode, *rDequant.GetProducer(), kPassName);

    TensorDesc dequantOutDesc;
    FUSION_PASS_CHECK(rMatMul.GetProducer()->GetOutputDesc(0, dequantOutDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get matmul output desc for dequant failed."), return nullptr);
    rDequant.GetProducer()->UpdateOutputDesc(0, dequantOutDesc);
    return rDequant;
}

PatternUniqPtr BuildDequantPattern(const std::string& patternName, const char* opType, int64_t inputCount)
{
    auto graphBuilder = EsGraphBuilder(patternName.c_str());
    auto x1 = graphBuilder.CreateInput(kX1InputIdx);
    auto x2 = graphBuilder.CreateInput(kX2InputIdx);

    EsTensorHolder bias = nullptr;
    EsTensorHolder offsetW = nullptr;
    int64_t inputIdx = kBiasInputIdx;
    if (inputCount >= kThreeInputNum) {
        bias = graphBuilder.CreateInput(inputIdx++);
    }
    if (inputCount >= kFourInputNum) {
        offsetW = graphBuilder.CreateInput(inputIdx++);
    }
    auto matmulY = CreateMatMulLikeNode(graphBuilder, opType, x1, x2, bias, offsetW);

    auto deqScale = graphBuilder.CreateInput(inputIdx);
    auto dequantY = CreateAscendDequantNode(graphBuilder, matmulY, deqScale, false, false,
                                            static_cast<int64_t>(DT_FLOAT16));

    auto graph = graphBuilder.BuildAndReset({dequantY});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*matmulY.GetProducer(), 0});
    pattern->CaptureTensor({*dequantY.GetProducer(), 0});
    return pattern;
}

std::vector<PatternUniqPtr> BuildDequantPatterns(const std::string& prefix)
{
    struct DequantOpConfig {
        const char* suffix;
        const char* opType;
        int64_t maxInputCount;
    };
    const DequantOpConfig opConfigs[] = {
        {"_matmul", kOpTypeMatMul, kThreeInputNum},
        {"_matmulv2", kOpTypeMatMulV2, kFourInputNum},
        {"_batchmatmul", kOpTypeBatchMatMul, kThreeInputNum},
        {"_batchmatmulv2", kOpTypeBatchMatMulV2, kFourInputNum},
    };
    std::vector<PatternUniqPtr> patterns;
    for (const auto& opCfg : opConfigs) {
        for (int64_t inputCount = kBaseNodeNum; inputCount <= opCfg.maxInputCount; inputCount++) {
            patterns.emplace_back(BuildDequantPattern(
                prefix + opCfg.suffix + "_" + std::to_string(inputCount) + "in_dequant", opCfg.opType, inputCount));
        }
    }
    return patterns;
}

GNode GetMatchedDequantNode(const std::unique_ptr<MatchResult>& matchResult, bool& isDequant)
{
    isDequant = matchResult->GetMatchedNodes().size() > 1;
    if (!isDequant) {
        return GNode();
    }
    NodeIo dequantIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(1, dequantIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get dequant captured tensor."), return GNode());
    return dequantIo.node;
}

bool GetMatchedNodeDescs(const GNode& node, TensorDesc& x1Desc, TensorDesc& x2Desc, TensorDesc& outputDesc)
{
    FUSION_PASS_CHECK(node.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return false);
    FUSION_PASS_CHECK(node.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return false);
    FUSION_PASS_CHECK(node.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output desc failed."), return false);
    return true;
}

EsTensorHolder BuildMatMulNode(EsGraphBuilder& builder, const GNode& matchedMatmulNode, const char* opType,
                               const FusionParams& params, const TensorDesc& outputDesc, int64_t& nextInputIdx)
{
    EsTensorHolder replaceInput[kFourInputNum];
    nextInputIdx = CreateReplaceInputs(matchedMatmulNode, builder, replaceInput);
    EsTensorHolder rX1 = replaceInput[kX1InputIdx];
    EsTensorHolder rX2 = replaceInput[kX2InputIdx];
    if (params.dimNumX1 == 1) {
        rX1 = CreateUnsqueezeNode(builder, rX1, 0);
    }
    if (params.dimNumX2 == 1) {
        rX2 = CreateUnsqueezeNode(builder, rX2, 1);
    }
    auto rMatMul = CreateMatMulLikeNode(builder, opType, rX1, rX2, replaceInput[kBiasInputIdx],
                                        replaceInput[kOffsetWInputIdx]);
    GNode matmulNode = *rMatMul.GetProducer();
    matmulNode.UpdateOutputDesc(0,
                                CalcMatMulOutputDesc(params.dimNumX1, params.dimNumX2, params.bothOneDim, outputDesc));
    const char* transAttr1 = params.isBatch ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = params.isBatch ? "adj_x2" : "transpose_x2";
    bool transX1 = params.transX1;
    bool transX2 = params.transX2;
    matmulNode.SetAttr(transAttr1, transX1);
    matmulNode.SetAttr(transAttr2, transX2);
    int64_t offsetX = 0;
    if (matchedMatmulNode.GetAttr(kAttrOffsetX, offsetX) == GRAPH_SUCCESS) {
        matmulNode.SetAttr(kAttrOffsetX, offsetX);
    }
    CopyOtherAttrs(matchedMatmulNode, matmulNode, kPassName);
    return rMatMul;
}

} // namespace

std::vector<PatternUniqPtr> MatMulUnsqueezeSqueezeFusionPass::Patterns()
{
    using PatternBuilder = std::vector<PatternUniqPtr> (*)(const std::string&);
    const PatternBuilder builders[] = {BuildDequantPatterns, BuildMatMulPatterns, BuildMatMulV2Patterns,
                                       BuildBatchMatMulPatterns, BuildBatchMatMulV2Patterns};
    std::vector<PatternUniqPtr> patternGraphs;
    for (auto builder : builders) {
        auto result = builder("pattern");
        patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(result.begin()),
                             std::make_move_iterator(result.end()));
    }
    return patternGraphs;
}

bool MatMulUnsqueezeSqueezeFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Begin to do MatMulUnsqueezeSqueezeFusionPass MeetRequirements.");
    if (GetGeCompilerVersionNum() < kGeCompilerVersion900) {
        OPS_LOG_D(kPassName, "GE runtime < 9.0.0, skip fusion (compat empty run).");
        return false;
    }
    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(0, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor."), return false);
    GNode matchedMatmulNode = nodeIo.node;

    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OPS_LOG_E(kPassName, "Get platform_info failed."), return false);
    bool isDequant = IsMatMulDequantScenario(matchedMatmulNode, IsSupportL0c2out(platformInfo));
    auto matchedNodes = matchResult->GetMatchedNodes();
    bool isDequantPattern = matchedNodes.size() > 1;
    if (isDequant != isDequantPattern) {
        return false;
    }

    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedMatmulNode.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return false);
    FUSION_PASS_CHECK(matchedMatmulNode.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return false);
    FUSION_PASS_CHECK(matchedMatmulNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output desc failed."), return false);

    size_t dimNumX1 = x1Desc.GetShape().GetDimNum();
    size_t dimNumX2 = x2Desc.GetShape().GetDimNum();
    size_t dimNumY = outputDesc.GetShape().GetDimNum();
    if (dimNumX1 != 1 && dimNumX2 != 1 && dimNumY != 1) {
        OPS_LOG_D(kPassName,
                  "Neither x1 nor x2 is 1-dim and output is not 1-dim, skip fusion. dimX1=%zu dimX2=%zu dimY=%zu",
                  dimNumX1, dimNumX2, dimNumY);
        return false;
    }
    if (dimNumY == 1 && dimNumX1 != 1 && dimNumX2 != 1 && !isDequant) {
        OPS_LOG_D(kPassName, "Output is 1-dim but inputs are not 1-dim and no dequant, skip fusion.");
        return false;
    }

    return true;
}

GraphUniqPtr MatMulUnsqueezeSqueezeFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Begin to do MatMulUnsqueezeSqueezeFusionPass Replacement.");
    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(0, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor."), return nullptr);
    GNode matchedMatmulNode = nodeIo.node;

    bool isDequant = false;
    GNode matchedDequantNode = GetMatchedDequantNode(matchResult, isDequant);

    AscendString opTypeStr;
    FUSION_PASS_CHECK(matchedMatmulNode.GetType(opTypeStr) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get op type."), return nullptr);

    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(!GetMatchedNodeDescs(matchedMatmulNode, x1Desc, x2Desc, outputDesc),
                      OPS_LOG_E(kPassName, "Get matched node descs failed."), return nullptr);

    FusionParams params;
    params.dimNumX1 = x1Desc.GetShape().GetDimNum();
    params.dimNumX2 = x2Desc.GetShape().GetDimNum();
    params.bothOneDim = (params.dimNumX1 == 1 && params.dimNumX2 == 1);
    params.needSqueeze = !params.bothOneDim && (params.dimNumX1 == 1 || params.dimNumX2 == 1);
    FUSION_PASS_CHECK(!GetMatchedNodeAttrs(matchedMatmulNode, params.isBatch, params.transX1, params.transX2),
                      OPS_LOG_E(kPassName, "Get matched node attrs failed."), return nullptr);

    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);
    auto builder = es::EsGraphBuilder("replacement");
    int64_t nextInputIdx = 0;
    auto rMatMul = BuildMatMulNode(builder, matchedMatmulNode, opTypeStr.GetString(), params, outputDesc, nextInputIdx);

    EsTensorHolder rY = rMatMul;
    if (isDequant) {
        rY = BuildDequantReplacement(builder, matchedDequantNode, rMatMul, nextInputIdx);
        FUSION_PASS_CHECK(rY.GetCTensorHolder() == nullptr, OPS_LOG_E(kPassName, "Build dequant failed."),
                          return nullptr);
    }
    if (params.needSqueeze) {
        TensorDesc squeezeInputDesc;
        rY.GetProducer()->GetOutputDesc(0, squeezeInputDesc);
        int64_t squeezeAxis = CalcSqueezeAxis(params.dimNumX1, squeezeInputDesc.GetShape().GetDimNum());
        rY = CreateSqueezeNode(builder, rY, squeezeAxis);
        rY.GetProducer()->UpdateOutputDesc(0, outputDesc);
    }

    GraphUniqPtr replaceGraph = builder.BuildAndReset({rY});
    FUSION_PASS_CHECK(InferShape(replaceGraph, subgraphInputs) != SUCCESS, OPS_LOG_E(kPassName, "InferShape failed."),
                      return nullptr);
    return replaceGraph;
}

REG_FUSION_PASS(MatMulUnsqueezeSqueezeFusionPass).Stage(GetCompatPassStage());

} // namespace ops
