/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_to_matmul_v3_fusion_pass.h"

#include <cstdint>
#include <string>
#include <vector>

#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;

namespace ops {
namespace {

constexpr char kPassName[] = "MatMulToMatmulV3FusionPass";
constexpr int32_t kV3InputNum = 4;
constexpr int32_t kV3OffsetWIdx = 3;
constexpr int32_t kMatMulIrInputNum = 3;
constexpr int32_t kBatchMatMulIrInputNum = 2;
constexpr uint64_t kHf32EnableBit = 0x40UL;

bool IsBatchMatMulType(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS) {
        return false;
    }
    return opType == kOpTypeBatchMatMul || opType == kOpTypeBatchMatMulV2;
}

int64_t GetIrInputNum(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS) {
        return kV3InputNum;
    }
    if (opType == kOpTypeMatMul) {
        return std::max(static_cast<int64_t>(node.GetInputsSize()), static_cast<int64_t>(kMatMulIrInputNum));
    }
    if (opType == kOpTypeBatchMatMul) {
        return std::max(static_cast<int64_t>(node.GetInputsSize()), static_cast<int64_t>(kBatchMatMulIrInputNum));
    }
    return kV3InputNum;
}

void GetInputsInfo(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs, std::vector<DataType>& inputDtypes)
{
    for (const auto& subgraphInput : subgraphInputs) {
        const auto& allInputs = subgraphInput.GetAllInputs();
        if (allInputs.empty()) {
            continue;
        }
        auto matchNode = allInputs[0];
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputDtypes.emplace_back(tensorDesc.GetDataType());
    }
}

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

bool GetMatchedNodeAttrs(const GNode& matchedNode, bool& isBatch, bool& transX1, bool& transX2, int64_t& opImplModeEnum,
                         bool& enableHf32)
{
    isBatch = IsBatchMatMulType(matchedNode);
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
    opImplModeEnum = 0;
    if (matchedNode.GetAttr("_op_impl_mode_enum", opImplModeEnum) != SUCCESS) {
        OPS_LOG_D(kPassName, "Can't get opImplModeEnum.");
    }
    enableHf32 = (static_cast<uint64_t>(opImplModeEnum) & kHf32EnableBit) != 0UL;
    return true;
}

void CreateV3InputsAndCast(const GNode& matchedNode, es::EsGraphBuilder& builder, DataType x1Dtype, DataType x2Dtype,
                           es::EsTensorHolder& rX1, es::EsTensorHolder& rX2, es::EsTensorHolder v3Input[],
                           TensorDesc v3InputDesc[], TensorDesc& outputDesc)
{
    TensorDesc matchedInputDesc;
    int createIndex = 0;
    int64_t inputNum = GetIrInputNum(matchedNode);
    for (int64_t v3Idx = 0; v3Idx < inputNum; v3Idx++) {
        if (matchedNode.GetInputDesc(v3Idx, matchedInputDesc) == GRAPH_SUCCESS) {
            v3InputDesc[v3Idx] = matchedInputDesc;
            v3Input[v3Idx] = builder.CreateInput(createIndex++);
        }
    }
    rX1 = v3Input[kX1InputIdx];
    rX2 = v3Input[kX2InputIdx];
    if (x1Dtype == DT_INT8 && x2Dtype != DT_INT8) {
        rX1 = CreateCastNode(v3Input[kX1InputIdx], matchedNode, kX1InputIdx, x2Dtype);
        v3InputDesc[kX1InputIdx].SetDataType(x2Dtype);
        outputDesc.SetDataType(x2Dtype);
    } else if (x1Dtype != DT_INT8 && x2Dtype == DT_INT8) {
        rX2 = CreateCastNode(v3Input[kX2InputIdx], matchedNode, kX2InputIdx, x1Dtype);
        v3InputDesc[kX2InputIdx].SetDataType(x1Dtype);
        outputDesc.SetDataType(x1Dtype);
    }
}

es::EsTensorHolder BuildV3Op(bool isBatch, const es::EsTensorHolder& rX1, const es::EsTensorHolder& rX2,
                             const es::EsTensorHolder v3Input[], bool transX1, bool transX2, int64_t opImplModeEnum,
                             bool enableHf32)
{
    int64_t offsetX = 0;
    if (isBatch) {
        return es::BatchMatMulV3(rX1, rX2, v3Input[kBiasInputIdx], v3Input[kV3OffsetWIdx], transX1, transX2, offsetX,
                                 enableHf32);
    }
    return es::MatMulV3(rX1, rX2, v3Input[kBiasInputIdx], v3Input[kV3OffsetWIdx], transX1, transX2, offsetX,
                        opImplModeEnum);
}

void UpdateV3NodeDescs(const GNode& matchedNode, GNode& v3Node, const TensorDesc v3InputDesc[],
                       const TensorDesc& outputDesc)
{
    TensorDesc matchedInputDesc;
    int64_t inputNum = GetIrInputNum(matchedNode);
    for (int64_t v3Idx = 0; v3Idx < inputNum; v3Idx++) {
        if (matchedNode.GetInputDesc(v3Idx, matchedInputDesc) == GRAPH_SUCCESS) {
            v3Node.UpdateInputDesc(v3Idx, v3InputDesc[v3Idx]);
        }
    }
    v3Node.UpdateOutputDesc(0, outputDesc);
}

bool CheckInputDtype(const GNode& nodeMatmul)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    nodeMatmul.GetInputDesc(kX1InputIdx, x1Desc);
    nodeMatmul.GetInputDesc(kX2InputIdx, x2Desc);
    DataType x1Dtype = x1Desc.GetDataType();
    DataType x2Dtype = x2Desc.GetDataType();
    const std::vector<DataType> allowedDtypes = {DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8};
    return std::find(allowedDtypes.begin(), allowedDtypes.end(), x1Dtype) != allowedDtypes.end() &&
           std::find(allowedDtypes.begin(), allowedDtypes.end(), x2Dtype) != allowedDtypes.end();
}

bool ValidateNodeInputs(const ge::GNode& matchedNode)
{
    TensorDesc x1Desc;
    TensorDesc x2Desc;
    TensorDesc outputTensor;
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kX1InputIdx, x1Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x1 input desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetInputDesc(kX2InputIdx, x2Desc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get x2 input desc failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputTensor) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output desc failed."), return false);
    return true;
}
} // namespace

std::vector<PatternUniqPtr> MatMulToMatmulV3FusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patternGraphs;
    auto matMulPatterns = BuildMatMulPatterns("pattern");
    auto matMulV2Patterns = BuildMatMulV2Patterns("pattern");
    auto batchMatMulPatterns = BuildBatchMatMulPatterns("pattern");
    auto batchMatMulV2Patterns = BuildBatchMatMulV2Patterns("pattern");
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(matMulPatterns.begin()),
                         std::make_move_iterator(matMulPatterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(matMulV2Patterns.begin()),
                         std::make_move_iterator(matMulV2Patterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(batchMatMulPatterns.begin()),
                         std::make_move_iterator(batchMatMulPatterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(batchMatMulV2Patterns.begin()),
                         std::make_move_iterator(batchMatMulV2Patterns.end()));
    return patternGraphs;
}

bool MatMulToMatmulV3FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Begin to do MatMulToMatmulV3FusionPass MeetRequirements.");
    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor."), return false);
    GNode matchedNode = nodeIo.node;
    FUSION_PASS_CHECK(!ValidateNodeInputs(matchedNode), OPS_LOG_E(kPassName, "Validate node inputs failed."),
                      return false);

    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OPS_LOG_D(kPassName, "Can't get platformInfo."), return false);

    bool supportL12btBf16 = IsSupportL12BtBf16(platformInfo);
    FUSION_PASS_CHECK(!supportL12btBf16, OPS_LOG_D(kPassName, "This fusion pass is not supported on this platform."),
                      return false);

    FUSION_PASS_CHECK(
        !CheckInputDtype(matchedNode),
        OPS_LOG_W(kPassName, "Input data type must be FP16, BF16, FP32 or INT8 when TransferOpTypeV1V2ToV3."),
        return false);

    return true;
}

std::unique_ptr<Graph> MatMulToMatmulV3FusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor in Replacement."), return nullptr);
    GNode matchedNode = nodeIo.node;

    bool isBatch = false;
    bool transX1 = false;
    bool transX2 = false;
    bool enableHf32 = false;
    int64_t opImplModeEnum = 0;
    FUSION_PASS_CHECK(!GetMatchedNodeAttrs(matchedNode, isBatch, transX1, transX2, opImplModeEnum, enableHf32),
                      OPS_LOG_E(kPassName, "Get matched node attrs failed."), return nullptr);

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);
    std::vector<DataType> inputDtypes;
    GetInputsInfo(subgraphInputs, inputDtypes);
    FUSION_PASS_CHECK(inputDtypes.size() < kBaseNodeNum,
                      OPS_LOG_E(kPassName, "inputDtypes size[%zu] less than 2.", inputDtypes.size()), return nullptr);

    TensorDesc matchedOutputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, matchedOutputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output tensor desc failed."), return nullptr);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    es::EsTensorHolder rX1;
    es::EsTensorHolder rX2;
    es::EsTensorHolder v3Input[kV3InputNum];
    TensorDesc v3InputDesc[kV3InputNum];
    CreateV3InputsAndCast(matchedNode, replaceGraphBuilder, inputDtypes[0], inputDtypes[1], rX1, rX2, v3Input,
                          v3InputDesc, matchedOutputDesc);

    auto rY = BuildV3Op(isBatch, rX1, rX2, v3Input, transX1, transX2, opImplModeEnum, enableHf32);

    GNode v3Node = *rY.GetProducer();
    UpdateV3NodeDescs(matchedNode, v3Node, v3InputDesc, matchedOutputDesc);
    FUSION_PASS_CHECK(!CopyOtherAttrs(matchedNode, v3Node, kPassName), OPS_LOG_E(kPassName, "Copy other attrs failed."),
                      return nullptr);

    return replaceGraphBuilder.BuildAndReset({rY});
}

REG_FUSION_PASS(MatMulToMatmulV3FusionPass).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
