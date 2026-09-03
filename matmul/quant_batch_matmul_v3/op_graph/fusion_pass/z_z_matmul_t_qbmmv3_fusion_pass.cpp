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
 * \file z_z_matmul_t_qbmmv3_fusion_pass.cpp
 * \brief 将MatMul/MatMulV2/MatMulV3/BatchMatMul/BatchMatMulV2/BatchMatMulV3转换为QuantBatchMatmulV3。
 *
 * 融合规则：
 *
 *      x1       x2       bias(可选)                      x1       x2       bias(可选)   scale=1.0(const)
 *       |        |          |                             |        |          |              |
 *       +--------+----------+                             +--------+----------+--------------+
 *                |                                                      |
 *           MatMul系列            -------->                      QuantBatchMatmulV3
 *                |                                                      |
 *               out                                                    out
 */

#include "z_z_matmul_t_qbmmv3_fusion_pass.h"

#include <cstdint>
#include <string>
#include <vector>

#include "es_nn_ops.h"
#include "platform/platform_info.h"
#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "ge/es_graph_builder.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;

namespace ops {
namespace {

constexpr char kPassName[] = "ZZMatMulTOQBMMV3FusionPass";
constexpr char kOpTypeMatMulV3[] = "MatMulV3";
constexpr char kOpTypeBatchMatMulV3[] = "BatchMatMulV3";
constexpr int64_t kV3ScaleIdx = 2;
constexpr int64_t kV3BiasIdx = 4;
constexpr uint64_t kScaleOneValue = 0x3F800000;
constexpr int32_t kGeCompilerVersion900 = 90000000;

bool IsBatchMatMulType(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS) {
        return false;
    }
    return opType == kOpTypeBatchMatMul || opType == kOpTypeBatchMatMulV2 || opType == kOpTypeBatchMatMulV3;
}

PatternUniqPtr BuildV3PatternWithInputCount(const std::string& patternName, const char* opType, int64_t inputCount)
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
        offsetW = graphBuilder.CreateInput(inputIdx);
    }

    EsTensorHolder y;
    if (strcmp(opType, kOpTypeBatchMatMulV3) == 0) {
        y = BatchMatMulV3(x1, x2, bias, offsetW);
    } else {
        y = MatMulV3(x1, x2, bias, offsetW);
    }

    auto graph = graphBuilder.BuildAndReset({y});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*y.GetProducer(), kCaptureTensorIdx});
    return pattern;
}

std::vector<PatternUniqPtr> BuildMatMulV3Patterns(const std::string& prefix)
{
    std::vector<PatternUniqPtr> patterns;
    patterns.emplace_back(BuildV3PatternWithInputCount(prefix + "_matmulv3_2in", kOpTypeMatMulV3, kBaseNodeNum));
    patterns.emplace_back(BuildV3PatternWithInputCount(prefix + "_matmulv3_3in", kOpTypeMatMulV3, kThreeInputNum));
    patterns.emplace_back(BuildV3PatternWithInputCount(prefix + "_matmulv3_4in", kOpTypeMatMulV3, kFourInputNum));
    return patterns;
}

std::vector<PatternUniqPtr> BuildBatchMatMulV3Patterns(const std::string& prefix)
{
    std::vector<PatternUniqPtr> patterns;
    patterns.emplace_back(
        BuildV3PatternWithInputCount(prefix + "_batchmatmulv3_2in", kOpTypeBatchMatMulV3, kBaseNodeNum));
    patterns.emplace_back(
        BuildV3PatternWithInputCount(prefix + "_batchmatmulv3_3in", kOpTypeBatchMatMulV3, kThreeInputNum));
    patterns.emplace_back(
        BuildV3PatternWithInputCount(prefix + "_batchmatmulv3_4in", kOpTypeBatchMatMulV3, kFourInputNum));
    return patterns;
}

bool ValidateNodeInputs(const GNode& matchedNode)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output desc failed."), return false);
    return true;
}

bool CheckInputDtype(const GNode& matchedNode)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc yDesc;
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return true);
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return true);
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, yDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output desc failed."), return true);
    DataType x1Dtype = x1Desc.GetDataType();
    DataType x2Dtype = x2Desc.GetDataType();
    DataType yDtype = yDesc.GetDataType();

    bool notHif8 = (x1Dtype != DT_HIFLOAT8);

    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OPS_LOG_W(kPassName, "Get platform info failed.");
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (soc != "Ascend950") {
        return notHif8;
    }
    bool notSupportInt8ToInt32 = !(x1Dtype == DT_INT8 && x2Dtype == DT_INT8 && yDtype == DT_INT32);
    return notHif8 && notSupportInt8ToInt32;
}

bool GetTransposeAttrs(const GNode& matchedNode, bool& transX1, bool& transX2)
{
    bool isBatch = IsBatchMatMulType(matchedNode);
    const char* transStrX1 = isBatch ? "adj_x1" : "transpose_x1";
    const char* transStrX2 = isBatch ? "adj_x2" : "transpose_x2";
    if (matchedNode.GetAttr(transStrX1, transX1) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName, "Get %s from node failed.", transStrX1);
        return false;
    }
    if (matchedNode.GetAttr(transStrX2, transX2) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName, "Get %s from node failed.", transStrX2);
        return false;
    }
    return true;
}

} // namespace

std::vector<PatternUniqPtr> ZZMatMulTOQBMMV3FusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patternGraphs;
    auto matMulPatterns = BuildMatMulPatterns("pattern");
    auto matMulV2Patterns = BuildMatMulV2Patterns("pattern");
    auto batchMatMulPatterns = BuildBatchMatMulPatterns("pattern");
    auto batchMatMulV2Patterns = BuildBatchMatMulV2Patterns("pattern");
    auto matMulV3Patterns = BuildMatMulV3Patterns("pattern");
    auto batchMatMulV3Patterns = BuildBatchMatMulV3Patterns("pattern");
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(matMulPatterns.begin()),
                         std::make_move_iterator(matMulPatterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(matMulV2Patterns.begin()),
                         std::make_move_iterator(matMulV2Patterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(batchMatMulPatterns.begin()),
                         std::make_move_iterator(batchMatMulPatterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(batchMatMulV2Patterns.begin()),
                         std::make_move_iterator(batchMatMulV2Patterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(matMulV3Patterns.begin()),
                         std::make_move_iterator(matMulV3Patterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(batchMatMulV3Patterns.begin()),
                         std::make_move_iterator(batchMatMulV3Patterns.end()));
    return patternGraphs;
}

bool ZZMatMulTOQBMMV3FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Begin to do ZZMatMulTOQBMMV3FusionPass MeetRequirements.");
    if (GetGeCompilerVersionNum() < kGeCompilerVersion900) {
        OPS_LOG_D(kPassName, "GE runtime < 9.0.0, skip fusion (compat empty run).");
        return false;
    }
    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor."), return false);
    GNode matchedNode = nodeIo.node;
    FUSION_PASS_CHECK(!ValidateNodeInputs(matchedNode), OPS_LOG_E(kPassName, "Validate node inputs failed."),
                      return false);
    FUSION_PASS_CHECK(CheckInputDtype(matchedNode), OPS_LOG_I(kPassName, "Input type is not hif8 or int8."),
                      return false);

    return true;
}

std::unique_ptr<Graph> ZZMatMulTOQBMMV3FusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Begin to do ZZMatMulTOQBMMV3FusionPass Replacement.");
    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor in Replacement."), return nullptr);
    GNode matchedNode = nodeIo.node;

    bool transX1 = false;
    bool transX2 = false;
    FUSION_PASS_CHECK(!GetTransposeAttrs(matchedNode, transX1, transX2),
                      OPS_LOG_E(kPassName, "Get transpose attrs failed."), return nullptr);

    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return nullptr);
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return nullptr);
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output tensor desc failed."), return nullptr);
    int64_t dtype = static_cast<int64_t>(outputDesc.GetDataType());

    auto replaceGraphBuilder = EsGraphBuilder("replacement");
    auto rX1 = replaceGraphBuilder.CreateInput(kX1InputIdx);
    auto rX2 = replaceGraphBuilder.CreateInput(kX2InputIdx);
    rX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    rX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto rScale = replaceGraphBuilder.CreateConst(std::vector<uint64_t>{kScaleOneValue}, std::vector<int64_t>{1});

    EsTensorHolder rBias = nullptr;
    TensorDesc biasDesc;
    if (matchedNode.GetInputDesc(kBiasInputIdx, biasDesc) == GRAPH_SUCCESS) {
        rBias = replaceGraphBuilder.CreateInput(kBiasInputIdx);
        rBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    }

    int64_t groupSize = 0;
    auto rY = QuantBatchMatmulV3(rX1, rX2, rScale, nullptr, rBias, nullptr, dtype, transX1, transX2, groupSize);

    GNode quantBatchMatmulV3Node = *rY.GetProducer();
    quantBatchMatmulV3Node.UpdateInputDesc(kX1InputIdx, x1Desc);
    quantBatchMatmulV3Node.UpdateInputDesc(kX2InputIdx, x2Desc);
    TensorDesc scaleDesc(Shape({1}), FORMAT_ND, DT_UINT64);
    scaleDesc.SetOriginFormat(FORMAT_ND);
    scaleDesc.SetOriginShape(Shape({1}));
    quantBatchMatmulV3Node.UpdateInputDesc(kV3ScaleIdx, scaleDesc);
    if (rBias.GetCTensorHolder() != nullptr) {
        quantBatchMatmulV3Node.UpdateInputDesc(kV3BiasIdx, biasDesc);
    }
    quantBatchMatmulV3Node.UpdateOutputDesc(0, outputDesc);

    FUSION_PASS_CHECK(!CopyOtherAttrs(matchedNode, quantBatchMatmulV3Node, kPassName),
                      OPS_LOG_E(kPassName, "Copy other attrs failed."), return nullptr);

    return replaceGraphBuilder.BuildAndReset({rY});
}

REG_FUSION_PASS(ZZMatMulTOQBMMV3FusionPass).Stage(GetCompatPassStage());

} // namespace ops
