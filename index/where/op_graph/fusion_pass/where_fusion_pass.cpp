/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "es_nn_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"

#include "common/inc/error_util.h"
#include "where_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;

/**
 * @brief Define fusion where pattern
 * @details Only support 910_93 and Ascend950
 *                  input                    input
 *                    |          ==>           |
 *                  Where                   Nonzero
 *
 */

namespace ops {

static const std::initializer_list<DataType> NONZERO_DTYPE_SUPPORT_LIST = {DT_FLOAT,  DT_FLOAT16, DT_BF16,   DT_INT8,
                                                                           DT_INT16,  DT_INT32,   DT_INT64,  DT_UINT8,
                                                                           DT_UINT16, DT_UINT32,  DT_UINT64, DT_BOOL};

const std::string FUSION_PASS_NAME = "WhereFusionPass";
const int64_t CAPTURE_TENSOR_IDX_INPUT = 0l;

std::vector<PatternUniqPtr> WhereFusionPass::Patterns()
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Patterns for WhereFusionPass");
    std::vector<PatternUniqPtr> patternGraphs;
    auto graphBuilder = es::EsGraphBuilder(FUSION_PASS_NAME.c_str());

    auto x = graphBuilder.CreateInput(0);

    auto y = es::Where(x);
    auto graph = graphBuilder.BuildAndReset({y});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*x.GetProducer(), 0});

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool WhereFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for WhereFusionPass");

    PlatformInfo platform_info;
    OptionalInfo optional_info;
    OP_LOGE_IF(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
        false, FUSION_PASS_NAME.c_str(), "Get platform_info failed.");
    const std::string soc = platform_info.str_info.short_soc_version;
    bool is_platform910_93 = (soc == "Ascend910_93");
    bool is_platform950 = (soc == "Ascend950");
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Platform short soc: %s", soc.c_str());
    if (!(is_platform910_93 || is_platform950)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Only support Ascend910_93 and Ascend950");
        return false;
    }

    NodeIo input;
    OP_LOGE_IF(
        match_result->GetCapturedTensor(CAPTURE_TENSOR_IDX_INPUT, input) != SUCCESS, false, FUSION_PASS_NAME,
        "Failed to GetCaptrue tensor");

    auto inputNode = input.node;
    AscendString nodeName;
    inputNode.GetName(nodeName);

    TensorDesc inputDesc;
    inputNode.GetInputDesc(0, inputDesc);

    // The fusion pass does not support more than 2 inputs
    if (inputNode.GetInputsSize() > 1) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "%s does not support more than 2 inputs", nodeName.GetString());
        return false;
    }

    // The fusion pass does not support the input dtype not in dtypes of nonzero.
    DataType inputDtype = inputDesc.GetDataType();
    auto dtypeIter = std::find(NONZERO_DTYPE_SUPPORT_LIST.begin(), NONZERO_DTYPE_SUPPORT_LIST.end(), inputDtype);

    if (dtypeIter == NONZERO_DTYPE_SUPPORT_LIST.end()) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "%s does not support input dytpe %d.", nodeName.GetString(), inputDtype);
        return false;
    }

    return true;
}

GraphUniqPtr WhereFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement for WhereFusionPass");
    std::vector<SubgraphInput> subGraphInputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subGraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtpyes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subGraphInputs, inputShapes, inputDtpyes, inputFormats);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto nonzeroInput =
        replaceGraphBuilder.CreateInput(0, "x", inputDtpyes[0], inputFormats[0], inputShapes[0].GetDims());

    ge::Graph* graphPtr = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();
    GNode nonZero = es::CompliantNodeBuilder(graphPtr)
                    .OpType("NonZero")
                    .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();

    GNode xNode = *nonzeroInput.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, xNode, 0, nonZero, 0);
    es::EsTensorHolder nonzeroOutput(replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(nonZero, 0));

    // refresh format
    GNode nonzeroNode = *nonzeroOutput.GetProducer();
    auto nonzeroNodeFormat = inputFormats[0];
    TensorDesc nonzeroInputDesc;
    nonzeroNode.GetInputDesc(0, nonzeroInputDesc);
    nonzeroInputDesc.SetFormat(nonzeroNodeFormat);
    nonzeroNode.UpdateInputDesc(0, nonzeroInputDesc);

    // repalce output
    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({nonzeroOutput});

    // todo：exce infershape to shape/dtype
    if (InferShape(replaceGraph, subGraphInputs) != SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Infershape for replacement failed.");
        return nullptr;
    }
    return replaceGraph;
}

static void GetInputsInfo(
    const std::vector<SubgraphInput>& subGraphInputs, std::vector<Shape>& inputShapes,
    std::vector<DataType>& inputDtpyes, std::vector<Format>& inputFormats)
{
    for (const auto& subGraphInput : subGraphInputs) {
        auto matchNode = subGraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        AscendString nodeType;
        matchNode.node.GetType(nodeType);
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtpyes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

static Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subGraphInputs)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Begin infershape for replacements.");
    std::vector<Shape> inputShapes;
    for (const auto& subGraphInput : subGraphInputs) {
        auto matchNode = subGraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

REG_FUSION_PASS(WhereFusionPass).Stage(CustomPassStage::kCompatibleInherited);
} // namespace ops