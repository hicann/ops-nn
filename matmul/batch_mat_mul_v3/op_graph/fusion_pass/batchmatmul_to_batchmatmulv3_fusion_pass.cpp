/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batchmatmul_to_batchmatmulv3_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"

#include "platform/platform_info.h"
#include "common/inc/error_util.h"
#include "common/op_graph/fusion_pass/matmul_fusion_utils_pass.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;

namespace ops {
namespace {

constexpr char kPassName[] = "BatchMatMulToBatchMatmulV3FusionPass";
constexpr int32_t kV3InputNum = 4;
constexpr int32_t kV3OffsetWIdx = 3;

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
    bool transX1 = false;
    bool transX2 = false;
    FUSION_PASS_CHECK(matchedNode.GetAttr("adj_x1", transX1) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get adj_x1 failed."), return false);
    FUSION_PASS_CHECK(matchedNode.GetAttr("adj_x2", transX2) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get adj_x2 failed."), return false);
    return true;
}
} // namespace

std::vector<PatternUniqPtr> BatchMatMulToBatchMatmulV3FusionPass::Patterns()
{
    std::vector<PatternUniqPtr> patternGraphs;
    auto batchMatMulPatterns = BuildBatchMatMulPatterns("pattern");
    auto batchMatMulV2Patterns = BuildBatchMatMulV2Patterns("pattern");
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(batchMatMulPatterns.begin()),
                         std::make_move_iterator(batchMatMulPatterns.end()));
    patternGraphs.insert(patternGraphs.end(), std::make_move_iterator(batchMatMulV2Patterns.begin()),
                         std::make_move_iterator(batchMatMulV2Patterns.end()));
    return patternGraphs;
}

bool BatchMatMulToBatchMatmulV3FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Begin to do BatchMatMulToBatchMatmulV3FusionPass MeetRequirements.");
    FUSION_PASS_CHECK(!IsDav3510Platform(kPassName),
                      OPS_LOG_D(kPassName, "Current platform is not Dav3510, skip this fusion pass."), return false);
    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor."), return false);
    GNode matchedNode = nodeIo.node;

    FUSION_PASS_CHECK(!ValidateNodeInputs(matchedNode), OPS_LOG_E(kPassName, "Validate node inputs failed."),
                      return false);

    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
                          SUCCESS,
                      OPS_LOG_I(kPassName, "Can't get platformInfo."), return false);
    bool supportL12btBf16 = IsSupportL12BtBf16(platformInfo);

    if (!supportL12btBf16) {
        OPS_LOG_E(kPassName, "Current platform does not support l12bt bf16, can not do fusion.");
        return false;
    }

    FUSION_PASS_CHECK(
        !CheckInputDtype(matchedNode),
        OPS_LOG_W(kPassName, "Input data type must be FP16, BF16, FP32 or INT8 when TransferOpTypeV1V2ToV3."),
        return false);

    return true;
}

std::unique_ptr<Graph>
BatchMatMulToBatchMatmulV3FusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    NodeIo nodeIo;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != SUCCESS,
                      OPS_LOG_E(kPassName, "Failed to get captured tensor in Replacement."), return nullptr);
    GNode matchedNode = nodeIo.node;

    bool transX1 = false;
    bool transX2 = false;
    FUSION_PASS_CHECK(matchedNode.GetAttr("adj_x1", transX1) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get adj_x1 from BatchMatMul failed."), return nullptr);
    FUSION_PASS_CHECK(matchedNode.GetAttr("adj_x2", transX2) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get adj_x2 from BatchMatMul failed."), return nullptr);

    bool enableHf32 = false;
    int64_t opImplModeEnum = 0;
    if (matchedNode.GetAttr("_op_impl_mode_enum", opImplModeEnum) == SUCCESS) {
        enableHf32 = (static_cast<uint64_t>(opImplModeEnum) & 0x40UL) != 0UL;
    }

    int64_t offsetX = 0;
    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");

    TensorDesc matchedInputDesc;
    TensorDesc v3InputDesc[kV3InputNum];
    es::EsTensorHolder v3Input[kV3InputNum];
    int createIndex = 0;
    int32_t inputNum = static_cast<int32_t>(matchedNode.GetInputsSize());
    for (int32_t v3Idx = 0; v3Idx < inputNum; v3Idx++) {
        if (matchedNode.GetInputDesc(v3Idx, matchedInputDesc) == GRAPH_SUCCESS) {
            v3InputDesc[v3Idx] = matchedInputDesc;
            v3Input[v3Idx] = replaceGraphBuilder.CreateInput(createIndex++);
        }
    }

    auto rY = es::BatchMatMulV3(v3Input[kX1InputIdx], v3Input[kX2InputIdx], v3Input[kBiasInputIdx],
                                v3Input[kV3OffsetWIdx], transX1, transX2, offsetX, enableHf32);

    GNode v3Node = *rY.GetProducer();
    for (int32_t v3Idx = 0; v3Idx < inputNum; v3Idx++) {
        if (matchedNode.GetInputDesc(v3Idx, matchedInputDesc) == GRAPH_SUCCESS) {
            v3Node.UpdateInputDesc(v3Idx, v3InputDesc[v3Idx]);
        }
    }

    TensorDesc outputDesc;
    FUSION_PASS_CHECK(matchedNode.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Get output tensor desc failed."), return nullptr);
    FUSION_PASS_CHECK(v3Node.UpdateOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OPS_LOG_E(kPassName, "Update output tensor desc failed."), return nullptr);

    FUSION_PASS_CHECK(!CopyOtherAttrs(matchedNode, v3Node, kPassName), OPS_LOG_E(kPassName, "Copy other attrs failed."),
                      return nullptr);

    return replaceGraphBuilder.BuildAndReset({rY});
}

REG_FUSION_PASS(BatchMatMulToBatchMatmulV3FusionPass)
    .Stage(IsDav3510Platform(kPassName) ? CustomPassStage::kCompatibleInherited : CustomPassStage::kAfterInferShape);
} // namespace ops