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
 *                                        a            b
 *                                        |            |
 *                                     (cast_a)    (cast_b)  cast int8->fp16 if output dtype is fp32
 *  a  b  c  alpha beta                     \        /
 *  \  \  |    /    /                         matmul    alpha    c     beta
 *        gemm               ===>                \       /        \     /
 *          |                                       mul             mul
 *        output                                       \           /
 *                                                          add
 *                                                            |
 *                                                         output
 */

#include "gemm_to_matmul_fusion_pass.h"

#include <cstdint>
#include <string>
#include <vector>

#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "ge/compliant_node_builder.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace ops {
namespace {

constexpr char kPassName[] = "GemmToMatmulFusionPass";
constexpr char kOpTypeGemm[] = "GEMM";
constexpr int32_t kGemmInputAIdx = 0;
constexpr int32_t kGemmInputBIdx = 1;
constexpr int32_t kGemmInputCIdx = 2;
constexpr int32_t kGemmInputAlphaIdx = 3;
constexpr int32_t kGemmInputBetaIdx = 4;
constexpr int32_t kGemmOutputYIdx = 0;
constexpr char kIntrinsicFixPipeL0c2Out[] = "Intrinsic_fix_pipe_l0c2out";
constexpr int32_t kGeCompilerVersion900 = 90000000;
constexpr size_t kGemmMatrixDimNum = 2;
constexpr size_t kGemmScalarDimNum = 1;

es::EsTensorHolder CreateCastNode(const es::EsTensorHolder& input, const GNode& matchedNode, int64_t inputIdx,
                                  DataType dstDtype)
{
    auto castOutput = es::Cast(input, dstDtype);
    GNode castNode = *castOutput.GetProducer();
    TensorDesc castInDesc;
    matchedNode.GetInputDesc(inputIdx, castInDesc);
    castNode.UpdateInputDesc(0, castInDesc);
    TensorDesc castOutDesc;
    castNode.GetOutputDesc(0, castOutDesc);
    castOutDesc.SetShape(castInDesc.GetShape());
    castOutDesc.SetOriginShape(castInDesc.GetShape());
    castOutDesc.SetDataType(dstDtype);
    castNode.UpdateOutputDesc(0, castOutDesc);
    return castOutput;
}

Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    std::vector<ge::Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        ge::TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

es::EsTensorHolder CreateMulNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x1,
                                 const es::EsTensorHolder& x2, const std::string& name)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto mulNode = es::CompliantNodeBuilder(graph)
                       .OpType("Mul")
                       .Name(name.c_str())
                       .IrDefInputs({
                           {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
                       })
                       .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), mulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), mulNode, 1);
    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0);
    return es::EsTensorHolder(yHolder);
}

bool ValidateGemmNode(const GNode& matchedNode)
{
    TensorDesc aDesc;
    TensorDesc bDesc;
    TensorDesc cDesc;
    TensorDesc alphaDesc;
    TensorDesc betaDesc;
    TensorDesc outDesc;
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kGemmInputAIdx, aDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get gemm input a desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kGemmInputBIdx, bDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get gemm input b desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kGemmInputCIdx, cDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get gemm input c desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kGemmInputAlphaIdx, alphaDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get gemm input alpha desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kGemmInputBetaIdx, betaDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get gemm input beta desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(kGemmOutputYIdx, outDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get gemm output desc failed."), return false);

    if (aDesc.GetShape().GetDimNum() != kGemmMatrixDimNum || bDesc.GetShape().GetDimNum() != kGemmMatrixDimNum ||
        cDesc.GetShape().GetDimNum() != kGemmMatrixDimNum) {
        OPS_LOG_D(kPassName, "Gemm input a/b/c dim num[%zu] [%zu] [%zu] is not 2, skip fusion.",
                  aDesc.GetShape().GetDimNum(), bDesc.GetShape().GetDimNum(), cDesc.GetShape().GetDimNum());
        return false;
    }
    if (alphaDesc.GetShape().GetDimNum() != kGemmScalarDimNum || betaDesc.GetShape().GetDimNum() != kGemmScalarDimNum) {
        OPS_LOG_D(kPassName, "Gemm input alpha/beta dim num[%zu] [%zu] is not 1, skip fusion.",
                  alphaDesc.GetShape().GetDimNum(), betaDesc.GetShape().GetDimNum());
        return false;
    }

    ge::DataType outDtype = outDesc.GetDataType();
    if (outDtype == DT_INT8) {
        OPS_LOG_I(kPassName, "Output dtype is INT8, MatMulV2 does not support INT8 output, skip fusion.");
        return false;
    }

    return true;
}

} // namespace

std::vector<PatternUniqPtr> GemmToMatmulFusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patterns;
    auto graphBuilder = es::EsGraphBuilder("pattern");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto a = graphBuilder.CreateInput(kGemmInputAIdx);
    auto b = graphBuilder.CreateInput(kGemmInputBIdx);
    auto c = graphBuilder.CreateInput(kGemmInputCIdx);
    auto alpha = graphBuilder.CreateInput(kGemmInputAlphaIdx);
    auto beta = graphBuilder.CreateInput(kGemmInputBetaIdx);

    auto gemmNode = es::CompliantNodeBuilder(graph)
                        .OpType(kOpTypeGemm)
                        .Name(kOpTypeGemm)
                        .IrDefInputs({
                            {"a", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"b", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"c", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"alpha", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"beta", CompliantNodeBuilder::kEsIrInputRequired, ""},
                        })
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .IrDefAttrs({
                            {"transpose_a", CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
                            {"transpose_b", CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
                        })
                        .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *a.GetProducer(), a.GetProducerOutIndex(), gemmNode, kGemmInputAIdx);
    AddEdgeAndUpdatePeerDesc(*graph, *b.GetProducer(), b.GetProducerOutIndex(), gemmNode, kGemmInputBIdx);
    AddEdgeAndUpdatePeerDesc(*graph, *c.GetProducer(), c.GetProducerOutIndex(), gemmNode, kGemmInputCIdx);
    AddEdgeAndUpdatePeerDesc(*graph, *alpha.GetProducer(), alpha.GetProducerOutIndex(), gemmNode, kGemmInputAlphaIdx);
    AddEdgeAndUpdatePeerDesc(*graph, *beta.GetProducer(), beta.GetProducerOutIndex(), gemmNode, kGemmInputBetaIdx);

    auto output = es::EsTensorHolder(
        graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(gemmNode, kGemmOutputYIdx));
    auto patternGraph = graphBuilder.BuildAndReset({output});
    auto pattern = std::make_unique<Pattern>(std::move(*patternGraph));
    pattern->CaptureTensor({*output.GetProducer(), kCaptureTensorIdx});
    patterns.emplace_back(std::move(pattern));
    return patterns;
}

bool GemmToMatmulFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Enter GemmToMatmulFusionPass MeetRequirements.");

    if (GetGeCompilerVersionNum() < kGeCompilerVersion900) {
        OPS_LOG_D(kPassName, "GE runtime < 9.0.0, skip fusion.");
        return false;
    }

    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor."), return false);
    GNode matchedNode = nodeIo.node;

    if (!ValidateGemmNode(matchedNode)) {
        return false;
    }

    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OPS_LOG_E(kPassName, "Can't get platformInfo."), return false);

    const auto& intrinsicMap = platformInfo.ai_core_intrinsic_dtype_map;
    if (intrinsicMap.find(kIntrinsicFixPipeL0c2Out) == intrinsicMap.end()) {
        OPS_LOG_D(kPassName, "Platform does not have Intrinsic_fix_pipe_l0c2out, skip fusion.");
        return false;
    }
    return true;
}

std::unique_ptr<Graph> GemmToMatmulFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Enter Replacement, begin building replacement graph.");
    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor in Replacement."), return nullptr);
    GNode matchedNode = nodeIo.node;

    bool transposeA = false;
    bool transposeB = false;
    matchedNode.GetAttr("transpose_a", transposeA);
    matchedNode.GetAttr("transpose_b", transposeB);

    TensorDesc aInDesc;
    TensorDesc bInDesc;
    TensorDesc cInDesc;
    TensorDesc alphaInDesc;
    TensorDesc betaInDesc;
    TensorDesc outDesc;
    matchedNode.GetInputDesc(kGemmInputAIdx, aInDesc);
    matchedNode.GetInputDesc(kGemmInputBIdx, bInDesc);
    matchedNode.GetInputDesc(kGemmInputCIdx, cInDesc);
    matchedNode.GetInputDesc(kGemmInputAlphaIdx, alphaInDesc);
    matchedNode.GetInputDesc(kGemmInputBetaIdx, betaInDesc);
    matchedNode.GetOutputDesc(kGemmOutputYIdx, outDesc);

    ge::DataType aDtype = aInDesc.GetDataType();
    ge::DataType bDtype = bInDesc.GetDataType();
    ge::DataType outDtype = outDesc.GetDataType();
    bool needInputCast = (aDtype == DT_INT8 && bDtype == DT_INT8 && outDtype == DT_FLOAT);

    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);
    if (subgraphInputs.size() < static_cast<size_t>(kGemmInputBetaIdx + 1)) {
        OPS_LOG_E(kPassName, "Subgraph inputs size[%zu] less than expected.", subgraphInputs.size());
        return nullptr;
    }

    auto builder = es::EsGraphBuilder("replacement");

    auto rA = builder.CreateInput(kGemmInputAIdx, "a", aInDesc.GetDataType(), aInDesc.GetFormat(),
                                  aInDesc.GetShape().GetDims());
    auto rB = builder.CreateInput(kGemmInputBIdx, "b", bInDesc.GetDataType(), bInDesc.GetFormat(),
                                  bInDesc.GetShape().GetDims());
    auto rC = builder.CreateInput(kGemmInputCIdx, "c", cInDesc.GetDataType(), cInDesc.GetFormat(),
                                  cInDesc.GetShape().GetDims());
    auto rAlpha = builder.CreateInput(kGemmInputAlphaIdx, "alpha", alphaInDesc.GetDataType(), alphaInDesc.GetFormat(),
                                      alphaInDesc.GetShape().GetDims());
    auto rBeta = builder.CreateInput(kGemmInputBetaIdx, "beta", betaInDesc.GetDataType(), betaInDesc.GetFormat(),
                                     betaInDesc.GetShape().GetDims());

    es::EsTensorHolder matmulA = rA;
    es::EsTensorHolder matmulB = rB;
    ge::TensorDesc matmulAInDesc = aInDesc;
    ge::TensorDesc matmulBInDesc = bInDesc;
    if (needInputCast) {
        OPS_LOG_D(kPassName, "Creating Cast nodes for int8->fp16.");
        matmulA = CreateCastNode(rA, matchedNode, kGemmInputAIdx, DT_FLOAT16);
        matmulB = CreateCastNode(rB, matchedNode, kGemmInputBIdx, DT_FLOAT16);
        matmulAInDesc.SetDataType(DT_FLOAT16);
        matmulBInDesc.SetDataType(DT_FLOAT16);
    }

    auto matmulOutput = CreateMatMulLikeNode(builder, kOpTypeMatMulV2, matmulA, matmulB, nullptr, nullptr);
    GNode matmulNode = *matmulOutput.GetProducer();
    matmulNode.UpdateInputDesc(kX1InputIdx, matmulAInDesc);
    matmulNode.UpdateInputDesc(kX2InputIdx, matmulBInDesc);
    matmulNode.UpdateOutputDesc(0, outDesc);
    matmulNode.SetAttr("transpose_x1", transposeA);
    matmulNode.SetAttr("transpose_x2", transposeB);
    int64_t offsetX = 0;
    matmulNode.SetAttr("offset_x", offsetX);
    FUSION_PASS_CHECK(!CopyOtherAttrs(matchedNode, matmulNode, kPassName),
                      OPS_LOG_E(kPassName, "Copy other attrs failed."), return nullptr);

    auto mul1Output = CreateMulNode(builder, matmulOutput, rAlpha, "gemm_mul_alpha");
    GNode mul1Node = *mul1Output.GetProducer();
    mul1Node.UpdateInputDesc(0, outDesc);
    mul1Node.UpdateInputDesc(1, alphaInDesc);
    mul1Node.UpdateOutputDesc(0, outDesc);

    auto mul2Output = CreateMulNode(builder, rC, rBeta, "gemm_mul_beta");
    GNode mul2Node = *mul2Output.GetProducer();
    mul2Node.UpdateInputDesc(0, cInDesc);
    mul2Node.UpdateInputDesc(1, betaInDesc);
    mul2Node.UpdateOutputDesc(0, cInDesc);

    auto addOutput = es::Add(mul1Output, mul2Output);
    GNode addNode = *addOutput.GetProducer();
    addNode.UpdateInputDesc(0, outDesc);
    addNode.UpdateInputDesc(1, cInDesc);
    addNode.UpdateOutputDesc(0, outDesc);

    auto result = builder.BuildAndReset({addOutput});
    if (result == nullptr) {
        OPS_LOG_E(kPassName, "Build replacement graph failed.");
        return nullptr;
    }

    if (InferShape(result, subgraphInputs) != SUCCESS) {
        OPS_LOG_E(kPassName, "InferShape failed.");
        return nullptr;
    }

    for (auto& node : result->GetAllNodes()) {
        ge::AscendString nodeType;
        node.GetType(nodeType);
        if (nodeType == "MatMulV2") {
            ge::TensorDesc outputDesc;
            if (node.GetOutputDesc(0, outputDesc) == ge::GRAPH_SUCCESS) {
                outputDesc.SetDataType(outDtype);
                node.UpdateOutputDesc(0, outputDesc);
            }
        } else if (nodeType == "Mul") {
            ge::TensorDesc inputDesc;
            if (node.GetInputDesc(0, inputDesc) == ge::GRAPH_SUCCESS) {
                inputDesc.SetDataType(outDtype);
                node.UpdateInputDesc(0, inputDesc);
            }
        }
    }

    OPS_LOG_I(kPassName, "Do GemmToMatmulFusionPass success, replacement graph built with %zu nodes.",
              result->GetAllNodes().size());
    return result;
}

REG_FUSION_PASS(GemmToMatmulFusionPass).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
