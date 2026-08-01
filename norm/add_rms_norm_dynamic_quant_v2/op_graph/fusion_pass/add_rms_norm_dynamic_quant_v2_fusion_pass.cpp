/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file add_rms_norm_dynamic_quant_v2_fusion_pass.cpp
 * \brief Fusion pass for AddRmsNorm + Cast + DynamicQuant -> AddRmsNormDynamicQuantV2.
 *
 * Fusion pattern:
 *          x1   x2   gamma     smooth1
 *           \   |    /          |
 *           AddRmsNorm          |
 *          /    |    \          |
 *         x     y      \        |
 *               |       \       |
 *              Cast    DynamicQuant
 *               |       /         \
 *              y3     y1        scale1
 *
 *  ==>  x1, x2, gamma, smooth1  --> AddRmsNormDynamicQuantV2
 *       outputs: y1, y3(Cast), y4(AddRmsNorm.y), x, scale1
 */

#include "add_rms_norm_dynamic_quant_v2_fusion_pass.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <string>
#include <vector>

#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "compliant_node_builder.h"
#include "ge/ge_utils.h"
#include "external/ge_common/ge_api_types.h"
#include "common/inc/error_util.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {
namespace {
using UniqueGraphPtr = std::unique_ptr<Graph>;
constexpr char kPassName[] = "AddRmsNormDynamicQuantV2FusionPass";
constexpr char kAddRmsNormType[] = "AddRmsNorm";
constexpr char kCastType[] = "Cast";
constexpr char kDynamicQuantType[] = "DynamicQuant";
constexpr char kPatternName[] = "AddRmsNormDynamicQuantV2FusionPass";
constexpr char kPatternSuffixWithSmooth1[] = "WithSmooth1";
constexpr char kPatternSuffixWithSmooth2[] = "WithSmooth2";

constexpr int64_t kAddRmsNormCaptureIdx = 0;
constexpr int64_t kCastCaptureIdx = 1;
constexpr int64_t kDynamicQuant1CaptureIdx = 2;
constexpr int64_t kDynamicQuant2CaptureIdx = 3;

constexpr int32_t kAddRmsNormX1Idx = 0;
constexpr int32_t kAddRmsNormX2Idx = 1;
constexpr int32_t kAddRmsNormGammaIdx = 2;
constexpr int32_t kAddRmsNormYOutIdx = 0;
constexpr int32_t kAddRmsNormRstdOutIdx = 1;
constexpr int32_t kAddRmsNormXOutIdx = 2;

constexpr int32_t kCastInputIdx = 0;
constexpr int32_t kCastOutputIdx = 0;

constexpr int32_t kDynamicQuantInputIdx = 0;
constexpr int32_t kDynamicQuantSmoothScalesIdx = 1;
constexpr int32_t kDynamicQuantYOutIdx = 0;
constexpr int32_t kDynamicQuantScaleOutIdx = 1;

constexpr int32_t kFusedX1Idx = 0;
constexpr int32_t kFusedX2Idx = 1;
constexpr int32_t kFusedGammaIdx = 2;
constexpr int32_t kFusedSmoothScale1Idx = 3;
constexpr int32_t kFusedSmoothScale2Idx = 4;
constexpr int32_t kFusedBetaIdx = 5;

constexpr int32_t kFusedY1OutIdx = 0;
constexpr int32_t kFusedY2OutIdx = 1;
constexpr int32_t kFusedY3OutIdx = 2;
constexpr int32_t kFusedY4OutIdx = 3;
constexpr int32_t kFusedXOutIdx = 4;
constexpr int32_t kFusedScale1OutIdx = 5;
constexpr int32_t kFusedScale2OutIdx = 6;

constexpr int32_t NUM_TWO = 2;
constexpr int32_t NUM_THREE = 3;

const std::vector<DataType> kFp16Bf16Dtypes = {DT_FLOAT16, DT_BF16};
const std::vector<DataType> kFp32Dtypes = {DT_FLOAT};
const std::vector<DataType> kQuantOutDtypes = {DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN};

static bool IsDtypeSupported(DataType dtype, const std::vector<DataType>& supportedList)
{
    return std::find(supportedList.begin(), supportedList.end(), dtype) != supportedList.end();
}

static bool IsTargetPlatform()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    OP_LOGE_IF(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
               false, kPassName, "Get platform_info failed.");
    const std::string soc = platformInfo.str_info.short_soc_version;
    const bool isPlatform910B = (soc == "Ascend910B");
    const bool isPlatform950 = (soc == "Ascend950");
    OPS_LOG_D(kPassName, "Platform short soc: %s", soc.c_str());
    if (!isPlatform910B && !isPlatform950) {
        OPS_LOG_D(kPassName, "Platform %s is not supported, only Ascend910B and Ascend950.", soc.c_str());
        return false;
    }
    return true;
}

static bool InferShape(const UniqueGraphPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    OPS_LOG_D(kPassName, "Begin infershape for replacements.");
    std::vector<TensorDesc> inputDescs;
    for (const auto& subgraphInput : subgraphInputs) {
        const auto allInputs = subgraphInput.GetAllInputs();
        if (allInputs.empty()) {
            OPS_LOG_E(kPassName, "subgraph input is empty.");
            return FAILED;
        }
        TensorDesc tensorDesc;
        const auto matchNode = allInputs.at(0);
        if (matchNode.node.GetInputDesc(matchNode.index, tensorDesc) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName, "get subgraph input desc failed.");
            return FAILED;
        }
        inputDescs.emplace_back(tensorDesc);
    }

    // Update Data nodes in replacement graph with full TensorDesc (shape + dtype + format)
    for (auto node : (*replaceGraph).GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type != "Data") {
            continue;
        }
        int64_t index = -1;
        node.GetAttr("index", index);
        if (index < 0 || index >= static_cast<int64_t>(inputDescs.size())) {
            continue;
        }
        node.UpdateOutputDesc(0, inputDescs[index]);
        node.UpdateInputDesc(0, inputDescs[index]);
    }

    std::vector<Shape> inputShapes;
    for (const auto& desc : inputDescs) {
        inputShapes.emplace_back(desc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

static bool IsAddRmsNormYConsumersValid(const GNode& addRmsNormNode, int32_t& dynamicQuantCount)
{
    auto yConsumers = addRmsNormNode.GetOutDataNodesAndPortIndexs(kAddRmsNormYOutIdx);
    int32_t castCount = 0;
    dynamicQuantCount = 0;
    int32_t netOutputCount = 0;
    for (const auto& [consumerNode, port] : yConsumers) {
        (void)port;
        AscendString nodeType;
        consumerNode->GetType(nodeType);
        const std::string typeStr = nodeType.GetString();
        if (typeStr == kCastType) {
            ++castCount;
        } else if (typeStr == kDynamicQuantType) {
            ++dynamicQuantCount;
        } else if (typeStr == "NetOutput") {
            ++netOutputCount;
        } else {
            OPS_LOG_D(kPassName, "AddRmsNorm output y has unsupported consumer type %s.", typeStr.c_str());
            return false;
        }
    }
    if (castCount != 1 || (dynamicQuantCount != 2 && dynamicQuantCount != 1)) {
        OPS_LOG_D(kPassName, "AddRmsNorm output y must have one Cast and one or two DynamicQuant consumers.");
        return false;
    }
    return true;
}

static bool IsCastDtypeValid(const GNode& castNode)
{
    TensorDesc inputDesc;
    castNode.GetInputDesc(kCastInputIdx, inputDesc);
    TensorDesc outputDesc;
    castNode.GetOutputDesc(kCastOutputIdx, outputDesc);
    if (!IsDtypeSupported(inputDesc.GetDataType(), kFp16Bf16Dtypes)) {
        OPS_LOG_D(kPassName, "Cast input dtype %d is not supported, only fp16/bf16.", inputDesc.GetDataType());
        return false;
    }
    if (!IsDtypeSupported(outputDesc.GetDataType(), kFp32Dtypes)) {
        OPS_LOG_D(kPassName, "Cast output dtype %d is not supported, only fp32.", outputDesc.GetDataType());
        return false;
    }
    return true;
}

static bool GetDynamicQuantDstType(const GNode& dynamicQuantNode, int64_t& dstType)
{
    if (dynamicQuantNode.GetAttr("dst_type", dstType) != SUCCESS) {
        OPS_LOG_D(kPassName, "DynamicQuant node has no dst_type attr, use default DT_INT8.");
    }
    return true;
}

static bool IsDynamicQuantDtypeValid(const GNode& dynamicQuantNode)
{
    TensorDesc inputDesc;
    dynamicQuantNode.GetInputDesc(kDynamicQuantInputIdx, inputDesc);
    TensorDesc outputDesc;
    dynamicQuantNode.GetOutputDesc(kDynamicQuantYOutIdx, outputDesc);
    if (!IsDtypeSupported(inputDesc.GetDataType(), kFp16Bf16Dtypes)) {
        OPS_LOG_D(kPassName, "DynamicQuant input dtype %d is not supported, only fp16/bf16.", inputDesc.GetDataType());
        return false;
    }
    if (!IsDtypeSupported(outputDesc.GetDataType(), kQuantOutDtypes)) {
        OPS_LOG_D(kPassName, "DynamicQuant output dtype %d is not supported.", outputDesc.GetDataType());
        return false;
    }
    int64_t dstType = static_cast<int64_t>(DT_INT8);
    (void)GetDynamicQuantDstType(dynamicQuantNode, dstType);
    if (outputDesc.GetDataType() != static_cast<DataType>(dstType)) {
        OPS_LOG_D(kPassName, "DynamicQuant output dtype %d does not match dst_type %ld.", outputDesc.GetDataType(),
                  dstType);
        return false;
    }
    return true;
}

static bool IsSmoothScalesDtypeValid(const GNode& addRmsNormNode, const GNode& dynamicQuantNode,
                                     int32_t smoothScalesIdx)
{
    TensorDesc smoothScalesDesc;
    if (dynamicQuantNode.GetInputDesc(smoothScalesIdx, smoothScalesDesc) != SUCCESS) {
        return true;
    }
    TensorDesc x1Desc;
    addRmsNormNode.GetInputDesc(kAddRmsNormX1Idx, x1Desc);
    if (smoothScalesDesc.GetDataType() != x1Desc.GetDataType()) {
        OPS_LOG_D(kPassName,
                  "AddRmsNorm x1 dtype %d and DynamicQuant smooth_scales dtype %d are different, do not fuse.",
                  x1Desc.GetDataType(), smoothScalesDesc.GetDataType());
        return false;
    }
    return true;
}

struct AddRmsNormBuildResult {
    es::EsTensorHolder y;
    es::EsTensorHolder rstd;
    es::EsTensorHolder x;
};

static AddRmsNormBuildResult BuildAddRmsNormNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x1,
                                                 const es::EsTensorHolder& x2, const es::EsTensorHolder& gamma)
{
    static int counter = 0;
    const std::string name = "AddRmsNorm_" + std::to_string(counter++);
    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType(kAddRmsNormType)
                     .Name(name.c_str())
                     .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"gamma", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"rstd", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"x", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs(
                         {{"epsilon", es::CompliantNodeBuilder::kEsAttrOptional, "Float", es::CreateFrom(1e-6f)}})
                     .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), node, kAddRmsNormX1Idx);
    es::AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), node, kAddRmsNormX2Idx);
    es::AddEdgeAndUpdatePeerDesc(*graph, *gamma.GetProducer(), gamma.GetProducerOutIndex(), node, kAddRmsNormGammaIdx);
    return {es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, kAddRmsNormYOutIdx)),
            es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, kAddRmsNormRstdOutIdx)),
            es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, kAddRmsNormXOutIdx))};
}

struct DynamicQuantBuildResult {
    es::EsTensorHolder y;
    es::EsTensorHolder scale;
};

static DynamicQuantBuildResult BuildDynamicQuantNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x)
{
    static int counter = 0;
    const std::string name = "DynamicQuant_" + std::to_string(counter++);
    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType(kDynamicQuantType)
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"smooth_scales", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                   {"group_index", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"scale", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, kDynamicQuantInputIdx);
    return {
        es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, kDynamicQuantYOutIdx)),
        es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, kDynamicQuantScaleOutIdx))};
}

static DynamicQuantBuildResult BuildDynamicQuantNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x,
                                                     const es::EsTensorHolder& smoothScales)
{
    static int counter = 0;
    const std::string name = "DynamicQuant_s_" + std::to_string(counter++);
    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType(kDynamicQuantType)
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"smooth_scales", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"group_index", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"scale", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, kDynamicQuantInputIdx);
    es::AddEdgeAndUpdatePeerDesc(*graph, *smoothScales.GetProducer(), smoothScales.GetProducerOutIndex(), node,
                                 kDynamicQuantSmoothScalesIdx);
    return {
        es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, kDynamicQuantYOutIdx)),
        es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, kDynamicQuantScaleOutIdx))};
}

static GNode BuildFusedNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x1,
                            const es::EsTensorHolder& x2, const es::EsTensorHolder& gamma,
                            const es::EsTensorHolder& smoothScale1, const es::EsTensorHolder& smoothScale2,
                            const es::EsTensorHolder& beta, float epsilon, int64_t dstType)
{
    static int counter = 0;
    const std::string name = "AddRmsNormDynamicQuantV2_" + std::to_string(counter++);
    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("AddRmsNormDynamicQuantV2")
                     .Name(name.c_str())
                     .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"gamma", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"smooth_scale1", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                   {"smooth_scale2", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                   {"beta", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                     .IrDefOutputs({{"y1", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"y2", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"y3", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"y4", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"x", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"scale1", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"scale2", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs(
                         {{"epsilon", es::CompliantNodeBuilder::kEsAttrOptional, "Float", es::CreateFrom(epsilon)},
                          {"output_mask", es::CompliantNodeBuilder::kEsAttrOptional, "ListBool",
                           es::CreateFrom(std::vector<bool>{})},
                          {"dst_type", es::CompliantNodeBuilder::kEsAttrOptional, "Int", es::CreateFrom(dstType)}})
                     .Build();

    es::AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), node, kFusedX1Idx);
    es::AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), node, kFusedX2Idx);
    es::AddEdgeAndUpdatePeerDesc(*graph, *gamma.GetProducer(), gamma.GetProducerOutIndex(), node, kFusedGammaIdx);
    if (smoothScale1.GetProducer() != nullptr) {
        es::AddEdgeAndUpdatePeerDesc(*graph, *smoothScale1.GetProducer(), smoothScale1.GetProducerOutIndex(), node,
                                     kFusedSmoothScale1Idx);
    }
    if (smoothScale2.GetProducer() != nullptr) {
        es::AddEdgeAndUpdatePeerDesc(*graph, *smoothScale2.GetProducer(), smoothScale2.GetProducerOutIndex(), node,
                                     kFusedSmoothScale2Idx);
    }
    if (beta.GetProducer() != nullptr) {
        es::AddEdgeAndUpdatePeerDesc(*graph, *beta.GetProducer(), beta.GetProducerOutIndex(), node, kFusedBetaIdx);
    }

    return node;
}

PatternUniqPtr MakePatternDoubleDynamicQuant()
{
    std::string patternName = std::string(kPatternName) + kPatternSuffixWithSmooth1 + kPatternSuffixWithSmooth2;
    auto graphBuilder = es::EsGraphBuilder(patternName.c_str());
    auto inputs = graphBuilder.CreateInputs<5>();

    auto addRmsNormOut = BuildAddRmsNormNode(graphBuilder, inputs[0], inputs[1], inputs[2]);
    auto castY = es::Cast(addRmsNormOut.y, DT_FLOAT);
    auto dynamicQuantOut1 = BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, inputs[3]);
    auto dynamicQuantOut2 = BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, inputs[4]);

    std::vector<es::EsTensorHolder> outputs = {dynamicQuantOut1.y,    dynamicQuantOut2.y, castY,
                                               addRmsNormOut.y,       addRmsNormOut.x,    dynamicQuantOut1.scale,
                                               dynamicQuantOut2.scale};

    auto graph = graphBuilder.BuildAndReset(outputs);
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*addRmsNormOut.y.GetProducer(), kAddRmsNormYOutIdx})
        .CaptureTensor({*castY.GetProducer(), kCastOutputIdx})
        .CaptureTensor({*dynamicQuantOut1.y.GetProducer(), kDynamicQuantYOutIdx})
        .CaptureTensor({*dynamicQuantOut2.y.GetProducer(), kDynamicQuantYOutIdx});
    return pattern;
}

PatternUniqPtr MakePatternSingleDynamicQuant(bool hasSmooth1)
{
    std::string patternName = kPatternName;
    patternName = hasSmooth1 ? patternName + kPatternSuffixWithSmooth1 : patternName;
    auto graphBuilder = es::EsGraphBuilder(patternName.c_str());
    auto inputs = graphBuilder.CreateInputs<4>();
    if (!hasSmooth1) {
        inputs[3] = nullptr;
    }

    auto addRmsNormOut = BuildAddRmsNormNode(graphBuilder, inputs[0], inputs[1], inputs[2]);
    auto castY = es::Cast(addRmsNormOut.y, DT_FLOAT);
    auto dynamicQuantOut1 = hasSmooth1 ? BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y, inputs[3]) :
                                         BuildDynamicQuantNode(graphBuilder, addRmsNormOut.y);

    std::vector<es::EsTensorHolder> outputs = {dynamicQuantOut1.y, castY, addRmsNormOut.y, addRmsNormOut.x,
                                               dynamicQuantOut1.scale};

    auto graph = graphBuilder.BuildAndReset(outputs);
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*addRmsNormOut.y.GetProducer(), kAddRmsNormYOutIdx})
        .CaptureTensor({*castY.GetProducer(), kCastOutputIdx})
        .CaptureTensor({*dynamicQuantOut1.y.GetProducer(), kDynamicQuantYOutIdx});
    return pattern;
}

static UniqueGraphPtr BuildReplaceGraph(es::EsGraphBuilder& replaceGraphBuilder, GNode& fusedNode,
                                        bool hasDynamicQuant2)
{
    auto get_output = [&](int64_t idx) {
        return es::EsTensorHolder(replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(fusedNode, idx));
    };

    std::vector<es::EsTensorHolder> outputs;
    outputs.emplace_back(get_output(kFusedY1OutIdx));
    if (hasDynamicQuant2) {
        outputs.emplace_back(get_output(kFusedY2OutIdx));
    }
    outputs.emplace_back(get_output(kFusedY3OutIdx));
    outputs.emplace_back(get_output(kFusedY4OutIdx));
    outputs.emplace_back(get_output(kFusedXOutIdx));
    outputs.emplace_back(get_output(kFusedScale1OutIdx));
    if (hasDynamicQuant2) {
        outputs.emplace_back(get_output(kFusedScale2OutIdx));
    }

    return replaceGraphBuilder.BuildAndReset(outputs);
}
} // namespace

std::vector<PatternUniqPtr> AddRmsNormDynamicQuantV2FusionPass::Patterns()
{
    OPS_LOG_D(kPassName, "Enter Patterns for AddRmsNormDynamicQuantV2FusionPass");
    std::vector<PatternUniqPtr> patternGraphs;
    patternGraphs.emplace_back(MakePatternDoubleDynamicQuant());
    patternGraphs.emplace_back(MakePatternSingleDynamicQuant(false));
    patternGraphs.emplace_back(MakePatternSingleDynamicQuant(true));
    OPS_LOG_D(kPassName, "Define Patterns success, size = %zu.", patternGraphs.size());
    return patternGraphs;
}

bool AddRmsNormDynamicQuantV2FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Enter MeetRequirements for AddRmsNormDynamicQuantV2FusionPass");
    if (!IsTargetPlatform()) {
        return false;
    }

    NodeIo addRmsNormNodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kAddRmsNormCaptureIdx, addRmsNormNodeIo) != SUCCESS, false, kPassName,
               "Failed to get AddRmsNorm node.");
    NodeIo castNodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kCastCaptureIdx, castNodeIo) != SUCCESS, false, kPassName,
               "Failed to get Cast node.");
    NodeIo dynamicQuantNode1Io;
    OP_LOGE_IF(matchResult->GetCapturedTensor(kDynamicQuant1CaptureIdx, dynamicQuantNode1Io) != SUCCESS, false,
               kPassName, "Failed to get DynamicQuant1 node.");

    // Determine early whether this is a single or double DQ pattern match
    NodeIo dynamicQuantNode2Io;
    const bool hasDynamicQuant2 = (matchResult->GetCapturedTensor(kDynamicQuant2CaptureIdx, dynamicQuantNode2Io) ==
                                   SUCCESS);

    const auto& addRmsNormNode = addRmsNormNodeIo.node;
    const auto& castNode = castNodeIo.node;
    const auto& dynamicQuantNode1 = dynamicQuantNode1Io.node;

    TensorDesc x1Desc;
    addRmsNormNode.GetInputDesc(kAddRmsNormX1Idx, x1Desc);
    int64_t dstType1 = static_cast<int64_t>(DT_INT8);
    (void)GetDynamicQuantDstType(dynamicQuantNode1, dstType1);
    if (!IsDtypeSupported(x1Desc.GetDataType(), kFp16Bf16Dtypes)) {
        return false;
    }
    int32_t dynamicQuantCount = 0;
    if (!IsAddRmsNormYConsumersValid(addRmsNormNode, dynamicQuantCount)) {
        return false;
    }
    // Maximality check: if AddRmsNorm.y has 2 DynamicQuant consumers but the current
    // pattern only captured 1, reject this match so the double-DQ pattern can match instead.
    // Without this guard, a single-DQ pattern would fuse only one DQ and leave the other
    // as a standalone node fed by the fused op's output.
    int32_t capturedDqCount = hasDynamicQuant2 ? 2 : 1;
    if (dynamicQuantCount > capturedDqCount) {
        OPS_LOG_D(kPassName,
                  "AddRmsNorm has %d DynamicQuant consumers but current pattern captured %d, "
                  "skip to allow larger pattern to match.",
                  dynamicQuantCount, capturedDqCount);
        return false;
    }
    if (!IsCastDtypeValid(castNode)) {
        return false;
    }
    if (!IsDynamicQuantDtypeValid(dynamicQuantNode1)) {
        return false;
    }
    if (!IsSmoothScalesDtypeValid(addRmsNormNode, dynamicQuantNode1, kDynamicQuantSmoothScalesIdx)) {
        return false;
    }

    if (hasDynamicQuant2) {
        const auto& dynamicQuantNode2 = dynamicQuantNode2Io.node;
        if (!IsDynamicQuantDtypeValid(dynamicQuantNode2)) {
            return false;
        }
        int64_t dstType2 = static_cast<int64_t>(DT_INT8);
        (void)GetDynamicQuantDstType(dynamicQuantNode2, dstType2);
        if (dstType1 != dstType2) {
            return false;
        }
        if (!IsSmoothScalesDtypeValid(addRmsNormNode, dynamicQuantNode2, kDynamicQuantSmoothScalesIdx)) {
            return false;
        }
    }
    return true;
}

GraphUniqPtr AddRmsNormDynamicQuantV2FusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OPS_LOG_D(kPassName, "Enter Replacement for AddRmsNormDynamicQuantV2FusionPass");
    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    NodeIo addRmsNormNodeIo;
    if (matchResult->GetCapturedTensor(kAddRmsNormCaptureIdx, addRmsNormNodeIo) != SUCCESS) {
        return nullptr;
    }
    NodeIo castNodeIo;
    if (matchResult->GetCapturedTensor(kCastCaptureIdx, castNodeIo) != SUCCESS) {
        return nullptr;
    }
    NodeIo dynamicQuantNode1Io;
    if (matchResult->GetCapturedTensor(kDynamicQuant1CaptureIdx, dynamicQuantNode1Io) != SUCCESS) {
        return nullptr;
    }
    NodeIo dynamicQuantNode2Io;
    const bool hasDynamicQuant2 = (matchResult->GetCapturedTensor(kDynamicQuant2CaptureIdx, dynamicQuantNode2Io) ==
                                   SUCCESS);

    const auto& addRmsNormNode = addRmsNormNodeIo.node;
    const auto& castNode = castNodeIo.node;
    const auto& dynamicQuantNode1 = dynamicQuantNode1Io.node;
    const auto& dynamicQuantNode2 = dynamicQuantNode2Io.node;

    float epsilon = 1e-6f;
    addRmsNormNode.GetAttr("epsilon", epsilon);

    // Determine smooth presence from boundary input count (robust against captured-node clones
    // whose optional input edges may not be fully materialized, which would cause GetInputDesc
    // to fail and make us under-create Data nodes).
    //   double-DQ pattern: 5 boundary inputs (x1, x2, gamma, smooth1, smooth2)
    //   single-DQ + smooth: 4 boundary inputs
    //   single-DQ no smooth: 3 boundary inputs
    const int64_t numBoundaryInputs = static_cast<int64_t>(subgraphInputs.size());
    const bool hasSmoothScale1 = (numBoundaryInputs >= 4);
    const bool hasSmoothScale2 = hasDynamicQuant2;

    TensorDesc x1Desc;
    addRmsNormNode.GetInputDesc(kAddRmsNormX1Idx, x1Desc);
    TensorDesc x2Desc;
    addRmsNormNode.GetInputDesc(kAddRmsNormX2Idx, x2Desc);
    TensorDesc gammaDesc;
    addRmsNormNode.GetInputDesc(kAddRmsNormGammaIdx, gammaDesc);

    auto getInputDescFromBoundary = [&](int64_t boundaryIdx, TensorDesc& desc) -> bool {
        if (boundaryIdx >= numBoundaryInputs) {
            return false;
        }
        const auto& allInputs = subgraphInputs[boundaryIdx].GetAllInputs();
        if (allInputs.empty()) {
            return false;
        }
        const auto& consumer = allInputs.at(0);
        return (consumer.node.GetInputDesc(consumer.index, desc) == GRAPH_SUCCESS);
    };

    TensorDesc smoothScale1Desc;
    if (hasSmoothScale1) {
        getInputDescFromBoundary(3, smoothScale1Desc);
    }
    TensorDesc smoothScale2Desc;
    if (hasSmoothScale2) {
        getInputDescFromBoundary(numBoundaryInputs - 1, smoothScale2Desc);
    }

    int64_t dstType = static_cast<int64_t>(DT_INT8);
    (void)GetDynamicQuantDstType(dynamicQuantNode1, dstType);

    // Always create exactly numBoundaryInputs Data nodes so the framework's boundary-to-replacement
    // input mapping by index stays consistent. Under-creating Data nodes leaves later boundary
    // inputs unmapped, which surfaces as standalone DynamicQuant nodes fed by the fused op output.
    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    std::vector<es::EsTensorHolder> allInputs;
    allInputs.reserve(static_cast<size_t>(numBoundaryInputs));
    auto buildInput = [&](int64_t idx, const char* name, const TensorDesc& desc) {
        return replaceGraphBuilder.CreateInput(idx, name, desc.GetDataType(), desc.GetFormat(),
                                               desc.GetShape().GetDims());
    };
    allInputs.emplace_back(buildInput(0, "x1", x1Desc));
    allInputs.emplace_back(buildInput(1, "x2", x2Desc));
    allInputs.emplace_back(buildInput(2, "gamma", gammaDesc));
    if (hasSmoothScale1) {
        allInputs.emplace_back(buildInput(3, "smooth_scale1", smoothScale1Desc));
    }
    if (hasSmoothScale2) {
        const int64_t s2Idx = hasSmoothScale1 ? 4 : 3;
        allInputs.emplace_back(buildInput(s2Idx, "smooth_scale2", smoothScale2Desc));
    }
    // Defensive: if the framework reports more boundary inputs than expected, create placeholders
    // so the mapping doesn't drift (e.g. Const nodes feeding Reshape inside the pattern).
    while (static_cast<int64_t>(allInputs.size()) < numBoundaryInputs) {
        const int64_t idx = static_cast<int64_t>(allInputs.size());
        TensorDesc fallbackDesc;
        getInputDescFromBoundary(idx, fallbackDesc);
        allInputs.emplace_back(replaceGraphBuilder.CreateInput(
            idx, "extra", fallbackDesc.GetDataType(), fallbackDesc.GetFormat(), fallbackDesc.GetShape().GetDims()));
    }

    es::EsTensorHolder rSmoothScale1 = hasSmoothScale1 ? allInputs[3] : es::EsTensorHolder();
    es::EsTensorHolder rSmoothScale2 = hasSmoothScale2 ? allInputs[hasSmoothScale1 ? 4 : 3] : es::EsTensorHolder();
    es::EsTensorHolder rBeta;

    GNode fusedOut = BuildFusedNode(replaceGraphBuilder, allInputs[0], allInputs[1], allInputs[2], rSmoothScale1,
                                    rSmoothScale2, rBeta, epsilon, dstType);

    fusedOut.UpdateInputDesc(kFusedX1Idx, x1Desc);
    fusedOut.UpdateInputDesc(kFusedX2Idx, x2Desc);
    fusedOut.UpdateInputDesc(kFusedGammaIdx, gammaDesc);
    if (hasSmoothScale1) {
        fusedOut.UpdateInputDesc(kFusedSmoothScale1Idx, smoothScale1Desc);
    }
    if (hasSmoothScale2) {
        fusedOut.UpdateInputDesc(kFusedSmoothScale2Idx, smoothScale2Desc);
    }

    TensorDesc y1Desc;
    dynamicQuantNode1.GetOutputDesc(kDynamicQuantYOutIdx, y1Desc);
    fusedOut.UpdateOutputDesc(kFusedY1OutIdx, y1Desc);

    if (hasDynamicQuant2) {
        TensorDesc y2Desc;
        dynamicQuantNode2.GetOutputDesc(kDynamicQuantYOutIdx, y2Desc);
        fusedOut.UpdateOutputDesc(kFusedY2OutIdx, y2Desc);
    }

    TensorDesc scale1Desc;
    dynamicQuantNode1.GetOutputDesc(kDynamicQuantScaleOutIdx, scale1Desc);
    fusedOut.UpdateOutputDesc(kFusedScale1OutIdx, scale1Desc);

    if (hasDynamicQuant2) {
        TensorDesc scale2Desc;
        dynamicQuantNode2.GetOutputDesc(kDynamicQuantScaleOutIdx, scale2Desc);
        fusedOut.UpdateOutputDesc(kFusedScale2OutIdx, scale2Desc);
    }

    TensorDesc y3Desc;
    castNode.GetOutputDesc(kCastOutputIdx, y3Desc);
    fusedOut.UpdateOutputDesc(kFusedY3OutIdx, y3Desc);

    TensorDesc y4Desc;
    addRmsNormNode.GetOutputDesc(kAddRmsNormYOutIdx, y4Desc);
    fusedOut.UpdateOutputDesc(kFusedY4OutIdx, y4Desc);

    TensorDesc xOutDesc;
    addRmsNormNode.GetOutputDesc(kAddRmsNormXOutIdx, xOutDesc);
    fusedOut.UpdateOutputDesc(kFusedXOutIdx, xOutDesc);

    UniqueGraphPtr replaceGraph = BuildReplaceGraph(replaceGraphBuilder, fusedOut, hasDynamicQuant2);
    const auto inferStatus = InferShape(replaceGraph, subgraphInputs);
    if (inferStatus != SUCCESS) {
        return nullptr;
    }
    return replaceGraph;
}

REG_FUSION_PASS(AddRmsNormDynamicQuantV2FusionPass).Stage(CustomPassStage::kAfterInferShape);
} // namespace ops
