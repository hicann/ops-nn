/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "a_conv2d_mul_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "ge/compliant_node_builder.h"
#include "graph/graph.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

#if GE_COMPILER_VERSION_NUM >= 90100000U
#include "ge/fusion/graph_fuse_inspector_utils.h"
#endif

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace AConv2dMulFusionConsts;
using namespace ge;

void AConv2dMulFusion::InitMember()
{
    mulNode = nullptr;
    scaleNode = nullptr;
    filterNode = nullptr;
    biasNode = nullptr;
    mulNonConstInputIdx = -1;
    mulConstInputIdx = -1;
    isDav3510 = false;
    npuArch = NpuArch::DAV_RESV;
    insertedMulNodes.clear();
}

bool AConv2dMulFusion::CheckMatchStructure(const GNode& matchNode)
{
    auto consumers = matchNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (auto& consumer : consumers) {
        if (consumer.first == nullptr) {
            continue;
        }
        AscendString type;
        FUSION_PASS_CHECK_NOLOG(consumer.first->GetType(type) != GRAPH_SUCCESS, continue);
        if (type == MUL) {
            mulNode = consumer.first;
            break;
        }
    }
    FUSION_PASS_CHECK(mulNode == nullptr, OP_LOGD(FUSION_NAME, "conv output has no Mul consumer, no fusion."),
                      return false);

    int32_t connectedInputs = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(mulNode->GetInputsSize()); ++i) {
        if (mulNode->GetInDataNodesAndPortIndexs(i).first != nullptr) {
            connectedInputs++;
        }
    }
    FUSION_PASS_CHECK(connectedInputs != MUL_INPUT_NUMS,
                      OP_LOGD(FUSION_NAME, "mul connected input count %d != 2, no fusion.", connectedInputs),
                      return false);
    return true;
}

bool AConv2dMulFusion::DetermineMulInputIndices()
{
    int32_t constCnt = 0;
    int32_t nonConstCnt = 0;
    for (int32_t i = 0; i < MUL_INPUT_NUMS; ++i) {
        auto inPair = mulNode->GetInDataNodesAndPortIndexs(i);
        FUSION_PASS_CHECK(inPair.first == nullptr,
                          OP_LOGD(convDescInfo.nodeNameStr, "mul input %d is null, no fusion.", i), return false);
        AscendString type;
        FUSION_PASS_CHECK(inPair.first->GetType(type) != GRAPH_SUCCESS,
                          OP_LOGD(convDescInfo.nodeNameStr, "cannot get mul input %d type, no fusion.", i),
                          return false);
        if (type == CONST) {
            mulConstInputIdx = i;
            scaleNode = inPair.first;
            constCnt++;
        } else {
            mulNonConstInputIdx = i;
            nonConstCnt++;
        }
    }
    FUSION_PASS_CHECK(constCnt != 1 || nonConstCnt != 1,
                      OP_LOGD(convDescInfo.nodeNameStr, "mul must have 1 const and 1 non-const input, no fusion."),
                      return false);
    return true;
}

bool AConv2dMulFusion::CheckInputConst(const GNode& convNode, int32_t inputIdx, const char* weightName,
                                       GNodePtr& weightNode)
{
    auto weightPair = convNode.GetInDataNodesAndPortIndexs(inputIdx);
    FUSION_PASS_CHECK(weightPair.first == nullptr,
                      OP_LOGD(convDescInfo.nodeNameStr, "%s input is null, no fusion.", weightName), return false);
    AscendString weightType;
    FUSION_PASS_CHECK(weightPair.first->GetType(weightType) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "cannot get %s type, no fusion.", weightName), return false);
    FUSION_PASS_CHECK(weightType != CONST, OP_LOGD(convDescInfo.nodeNameStr, "%s is not Const, no fusion.", weightName),
                      return false);
    weightNode = weightPair.first;
    return true;
}

bool AConv2dMulFusion::CheckWeightConst(const GNode& convNode)
{
    FUSION_PASS_CHECK_NOLOG(!CheckInputConst(convNode, INPUT_FILTER_INDEX, "filter", filterNode), return false);
    if (convDescInfo.hasBias) {
        FUSION_PASS_CHECK_NOLOG(!CheckInputConst(convNode, INPUT_BIAS_INDEX, "bias", biasNode), return false);
    }
    return true;
}

bool AConv2dMulFusion::CheckScaleShape(const GNode& convNode) const
{
    AscendString convType;
    FUSION_PASS_CHECK(convNode.GetType(convType) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "can not get conv type, no fusion."), return false);
    if (convType == CONV2D) {
        return CheckScaleShapeConv2d();
    }
    return CheckScaleShapeConv3d();
}

bool AConv2dMulFusion::CheckScaleShapeConv2d() const
{
    const auto& outShape = convDescInfo.outputShape;
    FUSION_PASS_CHECK(static_cast<int32_t>(outShape.size()) != CONV2D_DIM_SIZE,
                      OP_LOGD(convDescInfo.nodeNameStr, "conv output dim %zu != 4, no fusion.", outShape.size()),
                      return false);
    Format outFormat = convDescInfo.outputFormat;
    FUSION_PASS_CHECK(outFormat != FORMAT_NCHW && outFormat != FORMAT_NHWC,
                      OP_LOGD(convDescInfo.nodeNameStr, "conv output format not NCHW/NHWC, no fusion."), return false);
    int64_t outputC = (outFormat == FORMAT_NCHW) ? outShape[NCHW_C_POSITION] : outShape[NHWC_C_POSITION];

    TensorDesc scaleDesc;
    FUSION_PASS_CHECK(scaleNode->GetOutputDesc(OUTPUT_INDEX, scaleDesc) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "can not get scale desc, no fusion."), return false);
    FUSION_PASS_CHECK(ConvFusionUtilsPass::IsUnknownShape(scaleDesc),
                      OP_LOGD(convDescInfo.nodeNameStr, "scale has unknown shape, no fusion."), return false);
    auto scaleDims = scaleDesc.GetShape().GetDims();
    if (scaleDims.empty()) {
        return true;
    }
    if (scaleDims.size() == 1) {
        FUSION_PASS_CHECK(scaleDims[0] != 1 && scaleDims[0] != outputC,
                          OP_LOGD(convDescInfo.nodeNameStr, "scale shape is not scalar or channel-wise, no fusion."),
                          return false);
        FUSION_PASS_CHECK(
            isDav3510 && outFormat == FORMAT_NCHW && scaleDims[0] == outputC,
            OP_LOGD(convDescInfo.nodeNameStr, "channel-wise scale not supported on dav3510 NCHW, no fusion."),
            return false);
        return true;
    }
    OP_LOGD(convDescInfo.nodeNameStr, "scale dim %zu not supported, no fusion.", scaleDims.size());
    return false;
}

bool AConv2dMulFusion::CheckScaleShapeConv3d() const
{
    const auto& outShape = convDescInfo.outputShape;
    FUSION_PASS_CHECK(static_cast<int32_t>(outShape.size()) != CONV3D_DIM_SIZE,
                      OP_LOGD(convDescInfo.nodeNameStr, "conv output dim %zu != 5, no fusion.", outShape.size()),
                      return false);
    Format outFormat = convDescInfo.outputFormat;
    int64_t outputC;
    FUSION_PASS_CHECK(isDav3510 && outFormat != FORMAT_NDHWC,
                      OP_LOGD(convDescInfo.nodeNameStr, "conv3d output format not NDHWC on dav3510, no fusion."),
                      return false);
    outputC = outShape[NDHWC_C_POSITION];
    if (outputC < 0) {
        OP_LOGD(convDescInfo.nodeNameStr, "conv output channel is %ld, which is unknown shape, no fusion.", outputC);
        return false;
    }
    TensorDesc scaleDesc;
    FUSION_PASS_CHECK(scaleNode->GetOutputDesc(OUTPUT_INDEX, scaleDesc) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "can not get scale desc, no fusion."), return false);
    FUSION_PASS_CHECK(ConvFusionUtilsPass::IsUnknownShape(scaleDesc),
                      OP_LOGD(convDescInfo.nodeNameStr, "scale has unknown shape, no fusion."), return false);
    auto scaleDims = scaleDesc.GetShape().GetDims();

    if (scaleDims.size() == 1) {
        FUSION_PASS_CHECK(scaleDims[0] != outputC, OP_LOGD(convDescInfo.nodeNameStr, "scale 1D C mismatch, no fusion."),
                          return false);
        return true;
    }
    if (scaleDims.size() == static_cast<size_t>(CONV3D_DIM_SIZE)) {
        FUSION_PASS_CHECK(
            scaleDims[0] != 1 || scaleDims[1] != 1 || scaleDims[2] != 1 || scaleDims[3] != 1 || scaleDims[4] != outputC,
            OP_LOGD(convDescInfo.nodeNameStr, "scale 5D not channel-wise, no fusion."), return false);
        return true;
    }
    OP_LOGD(convDescInfo.nodeNameStr, "scale dim %zu not 1 or 5, no fusion.", scaleDims.size());
    return false;
}

bool AConv2dMulFusion::MeetRequirements(const GNode& convNode)
{
    isDav3510 = ConvFusionUtilsPass::CheckSocList(ND_SOC_LIST, npuArch);

    auto convConsumers = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK(
        convConsumers.size() != SINGLE_CONSUMER_CNT,
        OP_LOGD(convDescInfo.nodeNameStr, "conv output consumer count %zu != 1, no fusion.", convConsumers.size()),
        return false);

    AscendString convType;
    FUSION_PASS_CHECK(convNode.GetType(convType) != GRAPH_SUCCESS,
                      OP_LOGD(convDescInfo.nodeNameStr, "can not get conv type, no fusion."), return false);
    if (convType == CONV3D) {
        auto mulConsumers = mulNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
        FUSION_PASS_CHECK(mulConsumers.size() > SINGLE_CONSUMER_CNT,
                          OP_LOGD(convDescInfo.nodeNameStr, "conv3d mul output consumer count %zu > 1, no fusion.",
                                  mulConsumers.size()),
                          return false);
    }

    FUSION_PASS_CHECK_NOLOG(!DetermineMulInputIndices(), return false);
    FUSION_PASS_CHECK_NOLOG(!CheckWeightConst(convNode), return false);
    FUSION_PASS_CHECK_NOLOG(!CheckScaleShape(convNode), return false);
    return true;
}

std::set<AscendString> AConv2dMulFusion::GetNodeTypes() const { return {CONV2D, CONV3D}; }

void AConv2dMulFusion::PrintGraphStructure() const {}

bool AConv2dMulFusion::CreateScaleMulNode(Graph& graph, const AscendString& name, const TensorDesc& dataDesc,
                                          const TensorDesc& outDesc, GNode& scaleMulNode) const
{
    scaleMulNode = es::CompliantNodeBuilder(&graph)
                       .OpType(MUL.GetString())
                       .Name(name.GetString())
                       .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                       .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();
    TensorDesc scaleDesc;
    FUSION_PASS_CHECK_NOLOG(scaleNode->GetOutputDesc(OUTPUT_INDEX, scaleDesc) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(scaleMulNode.UpdateInputDesc(mulNonConstInputIdx, dataDesc) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(scaleMulNode.UpdateInputDesc(mulConstInputIdx, scaleDesc) != GRAPH_SUCCESS, return false);
    FUSION_PASS_CHECK_NOLOG(scaleMulNode.UpdateOutputDesc(OUTPUT_INDEX, outDesc) != GRAPH_SUCCESS, return false);
    return true;
}

bool AConv2dMulFusion::RelinkConvOutputToMulConsumers(Graph& graph, GNode& convNode)
{
    auto consumers = mulNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    for (auto& [consumer, inPort] : consumers) {
        FUSION_PASS_CHECK(consumer == nullptr, OP_LOGE(convDescInfo.nodeNameStr, "mul's consumer is null."),
                          return false);
        FUSION_PASS_CHECK(graph.RemoveEdge(*mulNode, OUTPUT_INDEX, *consumer, inPort) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "remove edge between mul and mul's consumer failed."),
                          return false);
        FUSION_PASS_CHECK(graph.AddDataEdge(convNode, OUTPUT_INDEX, *consumer, inPort) != GRAPH_SUCCESS,
                          OP_LOGE(convDescInfo.nodeNameStr, "add edge between conv and mul's consumer failed."),
                          return false);
    }
    return true;
}

bool AConv2dMulFusion::InsertWeightMul(Graph& graph, GNode& convNode, const GNodePtr& weightNode,
                                       int32_t weightInputIdx, const std::string& nameSuffix)
{
    TensorDesc weightDataDesc;
    FUSION_PASS_CHECK_NOLOG(weightNode->GetOutputDesc(OUTPUT_INDEX, weightDataDesc) != GRAPH_SUCCESS, return false);

    AscendString mulName;
    FUSION_PASS_CHECK_NOLOG(mulNode->GetName(mulName) != GRAPH_SUCCESS, return false);
    std::string weightMulName = std::string(mulName.GetString()) + nameSuffix;
    const TensorDesc& weightDesc = (weightInputIdx == INPUT_FILTER_INDEX) ? convDescInfo.filterDesc :
                                                                            convDescInfo.biasDesc;
    GNode weightMulNode;
    FUSION_PASS_CHECK_NOLOG(
        !CreateScaleMulNode(graph, AscendString(weightMulName.c_str()), weightDataDesc, weightDesc, weightMulNode),
        return false);

    FUSION_PASS_CHECK(graph.RemoveEdge(*weightNode, OUTPUT_INDEX, convNode, weightInputIdx) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "remove edge between weight and conv failed."), return false);
    FUSION_PASS_CHECK(graph.AddDataEdge(*weightNode, OUTPUT_INDEX, weightMulNode, mulNonConstInputIdx) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "add edge between weight and weightMul failed."), return false);
    FUSION_PASS_CHECK(graph.AddDataEdge(*scaleNode, OUTPUT_INDEX, weightMulNode, mulConstInputIdx) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "add edge between scale and weightMul failed."), return false);
    FUSION_PASS_CHECK(graph.AddDataEdge(weightMulNode, OUTPUT_INDEX, convNode, weightInputIdx) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "add edge between weightMul and conv failed."), return false);
    insertedMulNodes.emplace_back(weightMulNode);
    return true;
}

bool AConv2dMulFusion::ConvFusionReplaceImpl(GraphPtr& graph, GNode& convNode, CustomPassContext& passContext)
{
    FUSION_PASS_CHECK(graph == nullptr, OP_LOGE(FUSION_NAME, "%s graph is nullptr.", convDescInfo.nodeNameStr.c_str()),
                      return false);

    std::vector<GNode> nodesBeforeFuse = {convNode, *mulNode};
    AscendString failedReason;
#if GE_COMPILER_VERSION_NUM >= 90100000U
    FUSION_PASS_CHECK(!ge::fusion::GraphFuseInspectorUtils::CanFuse(nodesBeforeFuse, failedReason),
                      OP_LOGD(convDescInfo.nodeNameStr, "CanFuse failed, reason: %s.", failedReason.GetString()),
                      return false);
#endif

    FUSION_PASS_CHECK(!RelinkConvOutputToMulConsumers(*graph, convNode),
                      OP_LOGE(convDescInfo.nodeNameStr, "relink conv output to mul's consumers failed."), return false);
    FUSION_PASS_CHECK(!InsertWeightMul(*graph, convNode, filterNode, INPUT_FILTER_INDEX, FILTER_MUL_NAME_SUFFIX),
                      OP_LOGE(convDescInfo.nodeNameStr, "insert filterMul failed."), return false);
    if (convDescInfo.hasBias) {
        FUSION_PASS_CHECK(!InsertWeightMul(*graph, convNode, biasNode, INPUT_BIAS_INDEX, BIAS_MUL_NAME_SUFFIX),
                          OP_LOGE(convDescInfo.nodeNameStr, "insert biasMul failed."), return false);
    }

#if GE_COMPILER_VERSION_NUM >= 90100000U
    std::vector<GNode> nodesAfterFuse = {convNode};
    nodesAfterFuse.insert(nodesAfterFuse.end(), insertedMulNodes.begin(), insertedMulNodes.end());
    FUSION_PASS_CHECK(
        ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, nodesAfterFuse, passContext) != SUCCESS,
        OP_LOGE(convDescInfo.nodeNameStr, "ReportFuse failed."), return false);
#endif

    FUSION_PASS_CHECK(graph->RemoveNode(*mulNode) != GRAPH_SUCCESS,
                      OP_LOGE(convDescInfo.nodeNameStr, "remove original mul node failed."), return false);
    return true;
}

#if GE_COMPILER_VERSION_NUM >= 90100000U
REG_FUSION_PASS(AConv2dMulFusion).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace Ops
