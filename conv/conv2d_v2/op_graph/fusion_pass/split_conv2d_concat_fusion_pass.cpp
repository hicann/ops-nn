/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_conv2d_concat_fusion_pass.h"

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "ge/compliant_node_builder.h"
#include "graph/utils/type_utils.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace SplitConv2dConcatFusion;
using namespace ge;
using namespace fusion;

void ASplitConv2dConcatPass::InitMember()
{
    splitNodeName = "";
    convNodes.clear();
    concatNode = nullptr;
    fmapDesc = TensorDesc();
    outputDesc = TensorDesc();
    splitDimOriginFormat = FORMAT_ND;
    groups = 0;
    hasBias = false;
    isConcatV2 = false;
    isSplitV = false;
}

int32_t ASplitConv2dConcatPass::GetSplitDataInputIdx() const
{
    return isSplitV ? SPLITV_DATA_INPUT_IDX : SPLIT_DATA_INPUT_IDX;
}

int32_t ASplitConv2dConcatPass::GetSplitDimInputIdx() const
{
    return isSplitV ? SPLITV_DIM_INPUT_IDX : SPLIT_DIM_INPUT_IDX;
}

int32_t ASplitConv2dConcatPass::GetConcatDimInputIdx() const
{
    return isConcatV2 ? static_cast<int32_t>(convNodes.size()) : CONCAT_DIM_INPUT_IDX;
}

int32_t ASplitConv2dConcatPass::GetFormatAxisPos(Format format, char axisChar) const
{
    std::string fmtStr = TypeUtils::FormatToAscendString(format).GetString();
    size_t found = fmtStr.find(axisChar);
    return found == std::string::npos ? 0 : static_cast<int32_t>(found);
}

bool ASplitConv2dConcatPass::GetAxisValue(const GNode& ownerNode, int32_t inputIdx, int32_t& axisVal) const
{
    Tensor axisTensor;
    if (ownerNode.GetInputConstData(inputIdx, axisTensor) == GRAPH_SUCCESS && axisTensor.GetData() != nullptr) {
        axisVal = *reinterpret_cast<const int32_t*>(axisTensor.GetData());
        return true;
    }
    auto inPair = ownerNode.GetInDataNodesAndPortIndexs(inputIdx);
    FUSION_PASS_CHECK(inPair.first == nullptr, OP_LOGD(splitNodeName.GetString(), "axis input node is null."),
                      return false);
    FUSION_PASS_CHECK(inPair.first->GetAttr(AscendString("value"), axisTensor) != GRAPH_SUCCESS,
                      OP_LOGD(splitNodeName.GetString(), "get const value attr failed."), return false);
    auto dataPtr = axisTensor.GetData();
    FUSION_PASS_CHECK(dataPtr == nullptr, OP_LOGD(splitNodeName.GetString(), "axis tensor data is null."),
                      return false);
    axisVal = *reinterpret_cast<const int32_t*>(dataPtr);
    return true;
}

bool ASplitConv2dConcatPass::CheckConvWeight(const GNode& convNode, ConvBranchBaseline& baseline, bool isFirst)
{
    TensorDesc weightDesc;
    FUSION_PASS_CHECK(convNode.GetInputDesc(INPUT_FILTER_INDEX, weightDesc) != GRAPH_SUCCESS,
                      OP_LOGD(splitNodeName.GetString(), "get weight desc failed."), return false);
    FUSION_PASS_CHECK(ConvFusionUtilsPass::IsUnknownShape(weightDesc),
                      OP_LOGD(splitNodeName.GetString(), "unknown shape not supported."), return false);
    std::vector<int64_t> weightShape = weightDesc.GetOriginShape().GetDims();
    FUSION_PASS_CHECK(weightShape.empty(), OP_LOGD(splitNodeName.GetString(), "weight shape is empty."), return false);
    Format weightFormat = weightDesc.GetOriginFormat();
    if (isFirst) {
        baseline.weightShape = weightShape;
        baseline.weightFormat = weightFormat;
        FUSION_PASS_CHECK(baseline.weightFormat != FORMAT_HWCN && baseline.weightFormat != FORMAT_NCHW,
                          OP_LOGD(splitNodeName.GetString(), "weight format only support HWCN or NCHW."), return false);
        return true;
    }
    FUSION_PASS_CHECK(weightShape != baseline.weightShape,
                      OP_LOGD(splitNodeName.GetString(), "weight shape inconsistent."), return false);
    FUSION_PASS_CHECK(weightFormat != baseline.weightFormat,
                      OP_LOGD(splitNodeName.GetString(), "weight format inconsistent."), return false);
    return true;
}

bool ASplitConv2dConcatPass::CheckConvOutputToConcat(const GNode& convNode, ConvBranchBaseline& baseline, bool isFirst)
{
    auto convOutputs = convNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK(convOutputs.size() != SINGLE_REF_CNT,
                      OP_LOGD(splitNodeName.GetString(), "conv2d output is multi-refer."), return false);
    GNodePtr outNode = convOutputs[FIRST_BRANCH_IDX].first;
    FUSION_PASS_CHECK(outNode == nullptr, OP_LOGD(splitNodeName.GetString(), "conv output node is null."),
                      return false);
    AscendString outType;
    FUSION_PASS_CHECK(outNode->GetType(outType) != GRAPH_SUCCESS,
                      OP_LOGD(splitNodeName.GetString(), "get conv output type failed."), return false);
    FUSION_PASS_CHECK(outType != CONCAT && outType != CONCAT_V2,
                      OP_LOGD(splitNodeName.GetString(), "conv output is not Concat/ConcatV2."), return false);
    if (isFirst) {
        concatNode = outNode;
        isConcatV2 = (outType == CONCAT_V2);
        FUSION_PASS_CHECK(concatNode->GetName(baseline.concatName) != GRAPH_SUCCESS,
                          OP_LOGD(splitNodeName.GetString(), "get concat name failed."), return false);
        return true;
    }
    AscendString curName;
    FUSION_PASS_CHECK(outNode->GetName(curName) != GRAPH_SUCCESS,
                      OP_LOGD(splitNodeName.GetString(), "get current concat name failed."), return false);
    FUSION_PASS_CHECK(curName != baseline.concatName,
                      OP_LOGD(splitNodeName.GetString(), "conv outputs go to different concat."), return false);
    return true;
}

bool ASplitConv2dConcatPass::CheckConvNonFmapInputs(const GNode& convNode, bool isFirst) const
{
    for (size_t idx = static_cast<size_t>(INPUT_FILTER_INDEX); idx < convNode.GetInputsSize(); ++idx) {
        auto inputPair = convNode.GetInDataNodesAndPortIndexs(static_cast<int32_t>(idx));
        if (inputPair.first == nullptr) {
            continue;
        }
        AscendString inType;
        FUSION_PASS_CHECK(inputPair.first->GetType(inType) != GRAPH_SUCCESS,
                          OP_LOGD(splitNodeName.GetString(), "get conv input type failed."), return false);
        FUSION_PASS_CHECK(ALLOWED_CONST_LIST.find(inType) == ALLOWED_CONST_LIST.end(),
                          OP_LOGD(splitNodeName.GetString(), "conv input %zu is not const type.", idx), return false);
        if (isFirst) {
            TensorDesc bDesc;
            FUSION_PASS_CHECK(convNode.GetInputDesc(static_cast<int32_t>(idx), bDesc) != GRAPH_SUCCESS,
                              OP_LOGD(splitNodeName.GetString(), "get conv input %zu desc failed.", idx), return false);
            FUSION_PASS_CHECK(DATA_TYPE_IN.find(bDesc.GetDataType()) == DATA_TYPE_IN.end(),
                              OP_LOGD(splitNodeName.GetString(), "conv input %zu dtype not supported.", idx),
                              return false);
        }
    }
    return true;
}

bool ASplitConv2dConcatPass::CheckOneConvBranch(const GNode& convNode, ConvBranchBaseline& baseline, bool isFirst)
{
    size_t inputCnt = convNode.GetInputsSize();
    FUSION_PASS_CHECK(inputCnt < REQUIRED_INPUT_NUMS, OP_LOGD(splitNodeName.GetString(), "conv2d inputs less than 2."),
                      return false);
    if (isFirst) {
        baseline.inputCnt = inputCnt;
        hasBias = inputCnt == CONV_COUNT_PARAMS_BIAS;
    } else {
        FUSION_PASS_CHECK(inputCnt != baseline.inputCnt,
                          OP_LOGD(splitNodeName.GetString(), "conv2d input count inconsistent."), return false);
    }
    FUSION_PASS_CHECK_NOLOG(!CheckConvWeight(convNode, baseline, isFirst), return false);
    FUSION_PASS_CHECK_NOLOG(!CheckConvOutputToConcat(convNode, baseline, isFirst), return false);
    FUSION_PASS_CHECK_NOLOG(!CheckConvNonFmapInputs(convNode, isFirst), return false);
    return true;
}

bool ASplitConv2dConcatPass::AnalyzeMidLayer(const GNode& splitNode)
{
    ConvBranchBaseline baseline;
    for (size_t i = 0; i < splitNode.GetOutputsSize(); ++i) {
        auto consumers = splitNode.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
        FUSION_PASS_CHECK(consumers.empty(), OP_LOGD(splitNodeName.GetString(), "split output %zu has no consumer.", i),
                          return false);
        FUSION_PASS_CHECK(consumers.size() != SINGLE_REF_CNT,
                          OP_LOGD(splitNodeName.GetString(), "split output %zu has multi consumers.", i), return false);
        GNode convNode = *consumers[FIRST_BRANCH_IDX].first;
        AscendString convType;
        FUSION_PASS_CHECK(convNode.GetType(convType) != GRAPH_SUCCESS,
                          OP_LOGD(splitNodeName.GetString(), "get conv type failed."), return false);
        FUSION_PASS_CHECK(convType != CONV2D, OP_LOGD(splitNodeName.GetString(), "split output %zu is not Conv2D.", i),
                          return false);
        FUSION_PASS_CHECK(!CheckOneConvBranch(convNode, baseline, i == FIRST_BRANCH_IDX),
                          OP_LOGD(splitNodeName.GetString(), "check conv branch %zu failed.", i), return false);
        convNodes.push_back(convNode);
    }
    return true;
}

bool ASplitConv2dConcatPass::CheckConcatDataInputs() const
{
    FUSION_PASS_CHECK(concatNode->GetInputsSize() != convNodes.size() + CONCAT_DIM_EXTRA_INPUT_CNT,
                      OP_LOGD(splitNodeName.GetString(), "concat input count mismatch."), return false);
    size_t convCnt = 0;
    for (size_t i = 0; i < concatNode->GetInputsSize(); ++i) {
        if (static_cast<int32_t>(i) == GetConcatDimInputIdx()) {
            continue;
        }
        auto inPair = concatNode->GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        FUSION_PASS_CHECK(inPair.first == nullptr,
                          OP_LOGD(splitNodeName.GetString(), "concat data input %zu is null.", i), return false);
        AscendString inType;
        FUSION_PASS_CHECK(inPair.first->GetType(inType) != GRAPH_SUCCESS,
                          OP_LOGD(splitNodeName.GetString(), "get concat input type failed."), return false);
        if (inType == CONV2D) {
            convCnt++;
        }
    }
    FUSION_PASS_CHECK(convCnt != convNodes.size(),
                      OP_LOGD(splitNodeName.GetString(), "concat conv input count mismatch."), return false);
    return true;
}

bool ASplitConv2dConcatPass::VerifySptCcatAxis(const GNode& splitNode)
{
    auto splitDimPair = splitNode.GetInDataNodesAndPortIndexs(GetSplitDimInputIdx());
    FUSION_PASS_CHECK(splitDimPair.first == nullptr, OP_LOGD(splitNodeName.GetString(), "split dim input is null."),
                      return false);
    AscendString splitDimType;
    FUSION_PASS_CHECK(splitDimPair.first->GetType(splitDimType) != GRAPH_SUCCESS,
                      OP_LOGD(splitNodeName.GetString(), "get split dim type failed."), return false);
    FUSION_PASS_CHECK(splitDimType != CONST, OP_LOGD(splitNodeName.GetString(), "split dim is not Const."),
                      return false);
    TensorDesc splitDimOutDesc;
    FUSION_PASS_CHECK(splitDimPair.first->GetOutputDesc(OUTPUT_INDEX, splitDimOutDesc) != GRAPH_SUCCESS,
                      OP_LOGD(splitNodeName.GetString(), "get split dim output desc failed."), return false);
    splitDimOriginFormat = splitDimOutDesc.GetOriginFormat();

    int32_t splitAxis = 0;
    FUSION_PASS_CHECK(!GetAxisValue(splitNode, GetSplitDimInputIdx(), splitAxis),
                      OP_LOGD(splitNodeName.GetString(), "get split axis failed."), return false);

    int32_t concatAxis = 0;
    FUSION_PASS_CHECK(!GetAxisValue(*concatNode, GetConcatDimInputIdx(), concatAxis),
                      OP_LOGD(splitNodeName.GetString(), "get concat axis failed."), return false);
    FUSION_PASS_CHECK(concatAxis != splitAxis, OP_LOGD(splitNodeName.GetString(), "split axis != concat axis."),
                      return false);

    Format fmapOriginFormat = convDescInfo.fmapDesc.GetOriginFormat();
    std::string fmtStr = TypeUtils::FormatToAscendString(fmapOriginFormat).GetString();
    FUSION_PASS_CHECK(fmtStr.find('C') == std::string::npos, OP_LOGD(splitNodeName.GetString(), "format has no C dim."),
                      return false);
    int32_t axisPos = GetFormatAxisPos(fmapOriginFormat, 'C');
    if (concatAxis == AXIS_FROM_END) {
        FUSION_PASS_CHECK(axisPos != NHWC_C_POSITION, OP_LOGD(splitNodeName.GetString(), "axis=-1 only for NHWC."),
                          return false);
        return true;
    }
    FUSION_PASS_CHECK(concatAxis != axisPos, OP_LOGD(splitNodeName.GetString(), "axis not on C dim."), return false);
    return true;
}

bool ASplitConv2dConcatPass::CheckFormatsConsistent(const GNode& splitNode)
{
    FUSION_PASS_CHECK(splitNode.GetInputDesc(GetSplitDataInputIdx(), fmapDesc) != GRAPH_SUCCESS,
                      OP_LOGD(splitNodeName.GetString(), "get split data desc failed."), return false);
    FUSION_PASS_CHECK(concatNode->GetOutputDesc(OUTPUT_INDEX, outputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(splitNodeName.GetString(), "get concat output desc failed."), return false);
    Format splitFmt = fmapDesc.GetOriginFormat();
    FUSION_PASS_CHECK(splitFmt != outputDesc.GetOriginFormat(),
                      OP_LOGD(splitNodeName.GetString(), "split format is not equal to concat."), return false);
    FUSION_PASS_CHECK(splitFmt != convDescInfo.fmapDesc.GetOriginFormat(),
                      OP_LOGD(splitNodeName.GetString(), "split format is not equal to conv2d."), return false);
    return true;
}

bool ASplitConv2dConcatPass::CheckMatchStructure(const GNode& matchNode)
{
    auto consumers = matchNode.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(FIRST_BRANCH_IDX));
    FUSION_PASS_CHECK(consumers.empty(), OP_LOGD(FUSION_NAME, "split first output has no consumer."), return false);
    GNodePtr convNode = consumers[FIRST_BRANCH_IDX].first;
    FUSION_PASS_CHECK(convNode == nullptr, OP_LOGD(FUSION_NAME, "split first output consumer is null."), return false);
    AscendString convType;
    FUSION_PASS_CHECK(convNode->GetType(convType) != GRAPH_SUCCESS, OP_LOGD(FUSION_NAME, "get conv type failed."),
                      return false);
    FUSION_PASS_CHECK(convType != CONV2D, OP_LOGD(FUSION_NAME, "split first output is not Conv2D."), return false);

    auto convOutputs = convNode->GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    FUSION_PASS_CHECK(convOutputs.size() != SINGLE_REF_CNT, OP_LOGD(FUSION_NAME, "conv2d output is multi-refer."),
                      return false);
    GNodePtr outNode = convOutputs[FIRST_BRANCH_IDX].first;
    FUSION_PASS_CHECK(outNode == nullptr, OP_LOGD(FUSION_NAME, "conv output node is null."), return false);
    AscendString outType;
    FUSION_PASS_CHECK(outNode->GetType(outType) != GRAPH_SUCCESS, OP_LOGD(FUSION_NAME, "get conv output type failed."),
                      return false);
    FUSION_PASS_CHECK(outType != CONCAT && outType != CONCAT_V2,
                      OP_LOGD(FUSION_NAME, "conv2d is not connected to Concat/ConcatV2."), return false);
    return true;
}

bool ASplitConv2dConcatPass::MeetRequirements(const GNode& splitNode)
{
    FUSION_PASS_CHECK(splitNode.GetName(splitNodeName) != GRAPH_SUCCESS, OP_LOGD(FUSION_NAME, "get split name failed."),
                      return false);
    OP_LOGD(splitNodeName.GetString(), "Begin to do ASplitConv2dConcatPass.");
    AscendString splitType;
    FUSION_PASS_CHECK(splitNode.GetType(splitType) != GRAPH_SUCCESS, OP_LOGD(FUSION_NAME, "get split type failed."),
                      return false);
    isSplitV = (splitType == SPLIT_V);

    FUSION_PASS_CHECK_NOLOG(!AnalyzeMidLayer(splitNode), return false);
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvDescInfo(convNodes[FIRST_BRANCH_IDX], convDescInfo),
                            return false);
    FUSION_PASS_CHECK_NOLOG(!CheckConcatDataInputs(), return false);
    FUSION_PASS_CHECK_NOLOG(!VerifySptCcatAxis(splitNode), return false);
    FUSION_PASS_CHECK_NOLOG(!CheckFormatsConsistent(splitNode), return false);
    return true;
}

std::set<AscendString> ASplitConv2dConcatPass::GetNodeTypes() const { return {SPLIT, SPLIT_V}; }

Status ASplitConv2dConcatPass::SafeRemoveConstNode(GraphPtr& graph, GNode& constNode) const
{
    auto consumers = constNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX);
    if (consumers.size() != SINGLE_REF_CNT) {
        OP_LOGD(splitNodeName.GetString(), "const node has %zu consumers, skip remove.", consumers.size());
        return SUCCESS;
    }
    FUSION_PASS_CHECK(graph->RemoveNode(constNode) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "remove const node failed."), return FAILED);
    return SUCCESS;
}

Status ASplitConv2dConcatPass::ConvFusionPreImpl(GraphPtr& graph, GNode& splitNode, CustomPassContext& passContext)
{
    groups = static_cast<int64_t>(convNodes.size());
    auto dimPair = splitNode.GetInDataNodesAndPortIndexs(GetSplitDimInputIdx());
    if (dimPair.first != nullptr) {
        GNode dimNode = *dimPair.first;
        FUSION_PASS_CHECK(SafeRemoveConstNode(graph, dimNode) != SUCCESS,
                          OP_LOGE(splitNodeName.GetString(), "remove split dim node failed."), return FAILED);
    }
    if (isSplitV) {
        auto sizePair = splitNode.GetInDataNodesAndPortIndexs(SPLITV_SIZE_SPLITS_INPUT_IDX);
        if (sizePair.first != nullptr) {
            GNode sizeNode = *sizePair.first;
            FUSION_PASS_CHECK(SafeRemoveConstNode(graph, sizeNode) != SUCCESS,
                              OP_LOGE(splitNodeName.GetString(), "remove size_splits node failed."), return FAILED);
        }
    }
    auto concatDimPair = concatNode->GetInDataNodesAndPortIndexs(GetConcatDimInputIdx());
    if (concatDimPair.first != nullptr) {
        GNode concatDimNode = *concatDimPair.first;
        FUSION_PASS_CHECK(SafeRemoveConstNode(graph, concatDimNode) != SUCCESS,
                          OP_LOGE(splitNodeName.GetString(), "remove concat dim node failed."), return FAILED);
    }
    return SUCCESS;
}

bool ASplitConv2dConcatPass::ConvFusionReplaceImpl(GraphPtr& graph, GNode& splitNode, CustomPassContext& passContext)
{
    return DefaultConvFusionReplaceImpl(splitNode, passContext);
}

std::unique_ptr<SubgraphBoundary> ASplitConv2dConcatPass::ConstructBoundary(const GNode& splitNode)
{
    auto boundary = std::make_unique<SubgraphBoundary>();
    FUSION_PASS_CHECK_NOLOG(
        !ConvFusionUtilsPass::AddSubgraphInput(boundary, splitNode, GetSplitDataInputIdx(), BOUNDARY_FMAP_INPUT_IDX),
        return nullptr);
    for (size_t i = 0; i < convNodes.size(); ++i) {
        FUSION_PASS_CHECK_NOLOG(
            !ConvFusionUtilsPass::AddSubgraphInput(boundary, convNodes[i], INPUT_FILTER_INDEX,
                                                   BOUNDARY_FILTER_INPUT_BASE + static_cast<int64_t>(i)),
            return nullptr);
    }
    if (hasBias) {
        for (size_t i = 0; i < convNodes.size(); ++i) {
            FUSION_PASS_CHECK_NOLOG(
                !ConvFusionUtilsPass::AddSubgraphInput(
                    boundary, convNodes[i], INPUT_BIAS_INDEX,
                    static_cast<int64_t>(convNodes.size()) + BOUNDARY_FILTER_INPUT_BASE + static_cast<int64_t>(i)),
                return nullptr);
        }
    }
    FUSION_PASS_CHECK_NOLOG(
        !ConvFusionUtilsPass::AddSubgraphOutput(boundary, *concatNode, OUTPUT_INDEX, BOUNDARY_OUTPUT_IDX),
        return nullptr);
    return boundary;
}

bool ASplitConv2dConcatPass::BuildHostConcatNode(es::EsGraphBuilder& graphBuilder,
                                                 const std::vector<es::EsTensorHolder>& inputs, int32_t axis,
                                                 const AscendString& name, GNode& hostConcatNode) const
{
    Graph* geGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    FUSION_PASS_CHECK(geGraph == nullptr, OP_LOGE(splitNodeName.GetString(), "get replacement graph failed."),
                      return false);

    std::vector<es::CompliantNodeBuilder::IrInputDef> concatInputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
        concatInputs.push_back({"x", es::CompliantNodeBuilder::kEsIrInputDynamic, ""});
    }
    concatInputs.push_back({"concat_dim", es::CompliantNodeBuilder::kEsIrInputRequired, ""});

    hostConcatNode = es::CompliantNodeBuilder(geGraph)
                         .OpType(CONCAT_HOST_OP.GetString())
                         .Name(name.GetString())
                         .IrDefInputs(concatInputs)
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .InstanceDynamicInputNum("x", static_cast<int32_t>(inputs.size()))
                         .Build();

    int64_t nAttr = static_cast<int64_t>(inputs.size());
    FUSION_PASS_CHECK(hostConcatNode.SetAttr(ATTR_N, nAttr) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "set N attr for %s failed.", name.GetString()), return false);

    for (size_t i = 0; i < inputs.size(); ++i) {
        FUSION_PASS_CHECK(geGraph->AddDataEdge(*inputs[i].GetProducer(), inputs[i].GetProducerOutIndex(),
                                               hostConcatNode, static_cast<int32_t>(i)) != GRAPH_SUCCESS,
                          OP_LOGE(splitNodeName.GetString(), "add host concat data edge failed."), return false);
    }

    auto axisHolder = graphBuilder.CreateConst(std::vector<int32_t>{axis}, {CONCAT_DIM_SHAPE_SIZE}, DT_INT32,
                                               splitDimOriginFormat);
    int32_t dimIdx = static_cast<int32_t>(inputs.size());
    FUSION_PASS_CHECK(geGraph->AddDataEdge(*axisHolder.GetProducer(), axisHolder.GetProducerOutIndex(), hostConcatNode,
                                           dimIdx) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "add host concat dim edge failed."), return false);
    return true;
}

bool ASplitConv2dConcatPass::ExpandNDimInDesc(TensorDesc& desc) const
{
    std::vector<int64_t> shape = desc.GetOriginShape().GetDims();
    FUSION_PASS_CHECK(shape.empty(), OP_LOGE(splitNodeName.GetString(), "desc shape is empty."), return false);
    size_t pos = 0;
    if (shape.size() == CONCAT_SHAPE_SIZE) {
        pos = static_cast<size_t>(GetFormatAxisPos(desc.GetOriginFormat(), 'N'));
    }
    FUSION_PASS_CHECK(pos >= shape.size(), OP_LOGE(splitNodeName.GetString(), "N axis pos out of range."),
                      return false);
    shape[pos] *= groups;
    desc.SetShape(Shape(shape));
    desc.SetOriginShape(Shape(shape));
    return true;
}

bool ASplitConv2dConcatPass::UpdateHostConcatDescs(GNode& hostConcatNode, const TensorDesc& sampleDesc) const
{
    for (size_t i = 0; i < convNodes.size(); ++i) {
        TensorDesc inDesc = sampleDesc;
        FUSION_PASS_CHECK(hostConcatNode.UpdateInputDesc(static_cast<int32_t>(i), inDesc) != GRAPH_SUCCESS,
                          OP_LOGE(splitNodeName.GetString(), "update host concat input desc failed."), return false);
    }

    TensorDesc outDesc = sampleDesc;
    FUSION_PASS_CHECK_NOLOG(!ExpandNDimInDesc(outDesc), return false);
    FUSION_PASS_CHECK(hostConcatNode.UpdateOutputDesc(OUTPUT_INDEX, outDesc) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "update host concat output desc failed."), return false);

    TensorDesc dimDesc(Shape({CONCAT_DIM_SHAPE_SIZE}), splitDimOriginFormat, DT_INT32);
    dimDesc.SetOriginFormat(splitDimOriginFormat);
    dimDesc.SetOriginShape(Shape({CONCAT_DIM_SHAPE_SIZE}));
    FUSION_PASS_CHECK(hostConcatNode.UpdateInputDesc(static_cast<int32_t>(convNodes.size()), dimDesc) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "update host concat dim desc failed."), return false);
    return true;
}

bool ASplitConv2dConcatPass::SetGroupConvAttrs(GNode& groupConv)
{
    ConvBaseAttrs baseAttrs;
    FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvBaseAttr(convNodes[FIRST_BRANCH_IDX], baseAttrs, convDescInfo),
                            return false);
    FUSION_PASS_CHECK(groupConv.SetAttr(STRIDES, baseAttrs.strides) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "set strides failed."), return false);
    FUSION_PASS_CHECK(groupConv.SetAttr(PADS, baseAttrs.pads) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "set pads failed."), return false);
    FUSION_PASS_CHECK(groupConv.SetAttr(DILATIONS, baseAttrs.dilations) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "set dilations failed."), return false);
    FUSION_PASS_CHECK(groupConv.SetAttr(GROUPS, groups) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "set groups failed."), return false);
    FUSION_PASS_CHECK(groupConv.SetAttr(DATA_FORMAT, baseAttrs.dataFormat) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "set data_format failed."), return false);
    FUSION_PASS_CHECK(groupConv.SetAttr(OFFSET_X, baseAttrs.offsetX) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "set offset_x failed."), return false);
    if (baseAttrs.padding.GetLength() != 0) {
        FUSION_PASS_CHECK(groupConv.SetAttr(PADDING, baseAttrs.padding) != GRAPH_SUCCESS,
                          OP_LOGE(splitNodeName.GetString(), "set padding failed."), return false);
    }
    int64_t opImplModeEnum = 0;
    if (convNodes[FIRST_BRANCH_IDX].GetAttr(OP_IMPL_MODE_ENUM, opImplModeEnum) == GRAPH_SUCCESS) {
        FUSION_PASS_CHECK(groupConv.SetAttr(OP_IMPL_MODE_ENUM, opImplModeEnum) != GRAPH_SUCCESS,
                          OP_LOGE(splitNodeName.GetString(), "set _op_impl_mode_enum failed."), return false);
    }
    return true;
}

bool ASplitConv2dConcatPass::UpdateGroupConvDescs(GNode& groupConv)
{
    FUSION_PASS_CHECK(groupConv.UpdateInputDesc(INPUT_FMAP_INDEX, fmapDesc) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "update group conv fmap desc failed."), return false);

    TensorDesc filterDesc = convDescInfo.filterDesc;
    FUSION_PASS_CHECK_NOLOG(!ExpandNDimInDesc(filterDesc), return false);
    FUSION_PASS_CHECK(groupConv.UpdateInputDesc(INPUT_FILTER_INDEX, filterDesc) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "update group conv filter desc failed."), return false);

    if (hasBias) {
        TensorDesc biasDesc = convDescInfo.biasDesc;
        FUSION_PASS_CHECK_NOLOG(!ExpandNDimInDesc(biasDesc), return false);
        FUSION_PASS_CHECK(groupConv.UpdateInputDesc(INPUT_BIAS_INDEX, biasDesc) != GRAPH_SUCCESS,
                          OP_LOGE(splitNodeName.GetString(), "update group conv bias desc failed."), return false);
    }
    FUSION_PASS_CHECK(groupConv.UpdateOutputDesc(OUTPUT_INDEX, outputDesc) != GRAPH_SUCCESS,
                      OP_LOGE(splitNodeName.GetString(), "update group conv output desc failed."), return false);
    return true;
}

bool ASplitConv2dConcatPass::BuildGroupConvNode(es::EsGraphBuilder& graphBuilder,
                                                const std::vector<es::EsTensorHolder>& inputs, GNode& groupConv)
{
    Graph* geGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    FUSION_PASS_CHECK(geGraph == nullptr, OP_LOGE(splitNodeName.GetString(), "get replacement graph failed."),
                      return false);
    FUSION_PASS_CHECK(!ConvFusionUtilsPass::BuildConv2dNode(geGraph, "group_conv2d", inputs, groupConv),
                      OP_LOGE(splitNodeName.GetString(), "build group conv2d failed."), return false);
    FUSION_PASS_CHECK_NOLOG(!SetGroupConvAttrs(groupConv), return false);
    FUSION_PASS_CHECK_NOLOG(!UpdateGroupConvDescs(groupConv), return false);
    return true;
}

GraphUniqPtr ASplitConv2dConcatPass::Replacement(const GNode& splitNode)
{
    auto graphBuilder = es::EsGraphBuilder("replacement");
    auto fmap = graphBuilder.CreateInput(BOUNDARY_FMAP_INPUT_IDX);
    std::vector<es::EsTensorHolder> filters;
    std::vector<es::EsTensorHolder> biases;
    for (size_t i = 0; i < convNodes.size(); ++i) {
        filters.push_back(graphBuilder.CreateInput(BOUNDARY_FILTER_INPUT_BASE + static_cast<int64_t>(i)));
    }
    if (hasBias) {
        for (size_t i = 0; i < convNodes.size(); ++i) {
            biases.push_back(graphBuilder.CreateInput(static_cast<int64_t>(convNodes.size()) +
                                                      BOUNDARY_FILTER_INPUT_BASE + static_cast<int64_t>(i)));
        }
    }

    GNode filterConcatNode;
    int32_t weightAxisPos = GetFormatAxisPos(convDescInfo.filterDesc.GetOriginFormat(), 'N');
    FUSION_PASS_CHECK(
        !BuildHostConcatNode(graphBuilder, filters, weightAxisPos, AscendString("weight_concat"), filterConcatNode),
        OP_LOGE(splitNodeName.GetString(), "build weight host concat failed."), return nullptr);
    FUSION_PASS_CHECK(!UpdateHostConcatDescs(filterConcatNode, convDescInfo.filterDesc),
                      OP_LOGE(splitNodeName.GetString(), "update weight host concat desc failed."), return nullptr);

    auto* filterTh = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(filterConcatNode, OUTPUT_INDEX);
    FUSION_PASS_CHECK(filterTh == nullptr, OP_LOGE(splitNodeName.GetString(), "get filter concat holder failed."),
                      return nullptr);
    std::vector<es::EsTensorHolder> convInputs = {fmap, es::EsTensorHolder(filterTh)};

    if (hasBias) {
        GNode biasConcatNode;
        FUSION_PASS_CHECK(
            !BuildHostConcatNode(graphBuilder, biases, BIAS_AXIS, AscendString("bias_concat"), biasConcatNode),
            OP_LOGE(splitNodeName.GetString(), "build bias host concat failed."), return nullptr);
        FUSION_PASS_CHECK(!UpdateHostConcatDescs(biasConcatNode, convDescInfo.biasDesc),
                          OP_LOGE(splitNodeName.GetString(), "update bias host concat desc failed."), return nullptr);
        auto* biasTh = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(biasConcatNode, OUTPUT_INDEX);
        FUSION_PASS_CHECK(biasTh == nullptr, OP_LOGE(splitNodeName.GetString(), "get bias concat holder failed."),
                          return nullptr);
        convInputs.emplace_back(es::EsTensorHolder(biasTh));
    }

    GNode groupConv;
    FUSION_PASS_CHECK(!BuildGroupConvNode(graphBuilder, convInputs, groupConv),
                      OP_LOGE(splitNodeName.GetString(), "build group conv failed."), return nullptr);
    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(groupConv, OUTPUT_INDEX);
    FUSION_PASS_CHECK(yHolder == nullptr, OP_LOGE(splitNodeName.GetString(), "get group conv output holder failed."),
                      return nullptr);
    return graphBuilder.BuildAndReset({es::EsTensorHolder(yHolder)});
}

void ASplitConv2dConcatPass::PrintGraphStructure() const
{
    OP_LOGI(splitNodeName.GetString(), "split_conv2d_concat fusion success: groups=%lld.", groups);
}
} // namespace Ops
