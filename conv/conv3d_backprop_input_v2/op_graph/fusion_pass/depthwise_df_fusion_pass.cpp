/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "depthwise_df_fusion_pass.h"

#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace ops {
using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace ConvBackpropFusionUtils;
namespace {

const AscendString PASS_NAME = "DepthwiseDfFusionPass";
const AscendString DEPTHWISE_D = "DepthwiseConv2DBackpropInputD";
const AscendString DEPTHWISE_DYN = "DepthwiseConv2DBackpropInput";

constexpr uint32_t FILTER_INDEX_D = 0U;
constexpr uint32_t FILTER_INDEX_DYN = 1U;
constexpr uint32_t GRAD_OUTPUT_INDEX_D = 1U;
constexpr uint32_t GRAD_OUTPUT_INDEX_DYN = 2U;
constexpr int32_t GRAD_OUTPUT_DIM = 4;
constexpr int64_t GROUP_MAX_RANGE = 65535;
constexpr size_t FILTER_DIM_EXPECT = 4;
constexpr int32_t INPUT_SIZE_INDEX = 0;

const std::vector<int32_t> TRANSPOSE_PERM = {1, 0, 2, 3};

const std::map<ge::Format, std::vector<int32_t>> FORMAT_TO_NCHW_DIM_MAP = {{ge::FORMAT_NCHW, {0, 1, 2, 3}},
                                                                           {ge::FORMAT_HWCN, {3, 2, 0, 1}}};

constexpr const char* DEPTHWISE_PREFIX = "Depthwise";

} // anonymous namespace

AscendString DepthwiseDfFusionPass::GetNodeType() const { return PASS_NAME; }

bool DepthwiseDfFusionPass::GetResizeDepthwiseFilterShape(std::vector<int64_t>& resizeShape)
{
    const TensorDesc& filterDesc = isDynamic ? input1Desc : input0Desc;
    const auto& oriShape = filterDesc.GetShape().GetDims();
    ge::Format format = filterDesc.GetOriginFormat();
    if (oriShape.size() != FILTER_DIM_EXPECT) {
        OP_LOGE(GetNodeType().GetString(), "filter dim only supports 4 dims, got %zu.", oriShape.size());
        return false;
    }
    const auto& dimVec = FORMAT_TO_NCHW_DIM_MAP.at(format);
    if (format == ge::FORMAT_NCHW) {
        resizeShape = {oriShape[dimVec[0]] * oriShape[dimVec[1]], 1, oriShape[dimVec[2]], oriShape[dimVec[3]]};
    } else {
        resizeShape = {oriShape[dimVec[2]], oriShape[dimVec[3]], 1, oriShape[dimVec[0]] * oriShape[dimVec[1]]};
    }
    return true;
}

bool DepthwiseDfFusionPass::ValidateArch35Descs()
{
    const TensorDesc& filterDesc = isDynamic ? input1Desc : input0Desc;
    const TensorDesc& dedyDesc = isDynamic ? input2Desc : input1Desc;
    auto filterOriShapeDims = filterDesc.GetOriginShape().GetDims();
    for (auto filterDim : filterOriShapeDims) {
        OP_CHECK_IF(filterDim <= 0,
                    OP_LOGE(GetNodeType().GetString(), "%ld in filter shape should be positive.", filterDim),
                    return false);
    }
    auto dedyOriShape = dedyDesc.GetOriginShape();
    size_t dedyDims = dedyOriShape.GetDims().size();
    OP_CHECK_IF(dedyDims != static_cast<size_t>(GRAD_OUTPUT_DIM) &&
                    (dedyDims != UNKNOWN_RANK_DIM || dedyOriShape.GetDim(0) != UNKNOWN_RANK_DIM_VALUE),
                OP_LOGE(GetNodeType().GetString(), "out_backprop dims should be 4 or shape is [-2], but got dims=%zu",
                        dedyDims),
                return false);
    ge::Format gradOutputFormat = dedyDesc.GetOriginFormat();
    ge::Format outputFormat = outputDesc.GetOriginFormat();
    if (outputFormat != gradOutputFormat) {
        OP_LOGE(GetNodeType().GetString(), "output origin format and out_backprop origin format are not consistent.");
        return false;
    }
    return true;
}

bool DepthwiseDfFusionPass::ValidateFilterDesc(const GNode& matchedNode)
{
    if (isArch35) {
        OP_CHECK_IF(!ValidateArch35Descs(), OP_LOGE(GetNodeType().GetString(), "ValidateArch35Descs failed"),
                    return false);
    }
    int64_t groups = 0;
    OP_CHECK_IF(matchedNode.GetAttr("groups", groups) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Failed to get groups attr"), return false);
    OP_CHECK_IF(groups <= 0 || groups > GROUP_MAX_RANGE,
                OP_LOGE(GetNodeType().GetString(), "groups=%ld should be in range [1, 65535].", groups), return false);
    return true;
}

bool DepthwiseDfFusionPass::MeetRequirements(const GNode& matchedNode)
{
    AscendString matchedTypeAsc;
    OP_CHECK_IF(matchedNode.GetType(matchedTypeAsc) != GRAPH_SUCCESS,
                OP_LOGD(GetNodeType().GetString(), "GetType failed"), return false);
    isDynamic = (matchedTypeAsc.GetString() == std::string(DEPTHWISE_DYN.GetString()));
    isArch35 = ConvBackpropFusionUtilsPass::IsArch35();
    OP_CHECK_IF(!GetNodeDesc(matchedNode), OP_LOGE(GetNodeType().GetString(), "GetNodeDesc failed"), return false);
    OP_CHECK_IF(!ValidateFilterDesc(matchedNode), OP_LOGD(GetNodeType().GetString(), "ValidateFilterDesc failed"),
                return false);
    const TensorDesc& filterDesc = isDynamic ? input1Desc : input0Desc;
    ge::Format originFormat = filterDesc.GetOriginFormat();
    OP_CHECK_IF(FORMAT_TO_NCHW_DIM_MAP.find(originFormat) == FORMAT_TO_NCHW_DIM_MAP.end(),
                OP_LOGE(GetNodeType().GetString(), "filter origin format only supports NCHW or HWCN, got %d",
                        static_cast<int>(originFormat)),
                return false);
    OP_LOGD(GetNodeType().GetString(), "MeetRequirements passed");
    return true;
}

bool DepthwiseDfFusionPass::GetNodeAttrs(const GNode& node)
{
    OP_CHECK_IF(!ConvBackpropFusionBasePass::GetNodeAttrs(node),
                OP_LOGE(GetNodeType().GetString(), "Base GetNodeAttrs failed"), return false);
    if (!isDynamic) {
        OP_CHECK_IF(node.GetAttr("input_size", convBpAttr.input_size) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Failed to get input_size attr"), return false);
    }
    return true;
}

void DepthwiseDfFusionPass::SetNodeAttrs(GNode& outNode)
{
    ConvBackpropFusionBasePass::SetNodeAttrs(outNode);
    if (!isDynamic) {
        if (outNode.SetAttr("input_size", convBpAttr.input_size) != GRAPH_SUCCESS) {
            OP_LOGD(GetNodeType().GetString(), "Set input_size attr failed");
        }
    }
}

void DepthwiseDfFusionPass::CreateBoundaryInputs(EsGraphBuilder& builder, EsTensorHolder& iFilterHolder,
                                                 EsTensorHolder& iGradOutputHolder, EsTensorHolder& iInputSizeHolder)
{
    uint32_t filterIndex = isDynamic ? FILTER_INDEX_DYN : FILTER_INDEX_D;
    uint32_t gradOutputIndex = isDynamic ? GRAD_OUTPUT_INDEX_DYN : GRAD_OUTPUT_INDEX_D;
    const TensorDesc& filterDesc = isDynamic ? input1Desc : input0Desc;
    const TensorDesc& gradOutputDesc = isDynamic ? input2Desc : input1Desc;
    if (isDynamic) {
        iInputSizeHolder = builder.CreateInput(INPUT_SIZE_INDEX);
        ConvBackpropFusionUtilsPass::SetPlaceholderDesc(iInputSizeHolder, TENSOR_DEFAULT_OUTPUT_INDEX, input0Desc);
    }
    iFilterHolder = builder.CreateInput(static_cast<int64_t>(filterIndex));
    ConvBackpropFusionUtilsPass::SetPlaceholderDesc(iFilterHolder, TENSOR_DEFAULT_OUTPUT_INDEX, filterDesc);
    iGradOutputHolder = builder.CreateInput(static_cast<int64_t>(gradOutputIndex));
    ConvBackpropFusionUtilsPass::SetPlaceholderDesc(iGradOutputHolder, TENSOR_DEFAULT_OUTPUT_INDEX, gradOutputDesc);
}

bool DepthwiseDfFusionPass::BuildOptionalTranspose(EsGraphBuilder& builder, const std::string& nodeNamePrefix,
                                                   const EsTensorHolder& iFilterHolder,
                                                   EsTensorHolder& reshapeInputHolder, TensorDesc& reshapeInputDesc)
{
    const TensorDesc& filterDesc = isDynamic ? input1Desc : input0Desc;
    ge::Format originFormat = filterDesc.GetOriginFormat();
    reshapeInputHolder = iFilterHolder;
    reshapeInputDesc = filterDesc;
    if (originFormat != ge::FORMAT_NCHW) {
        return true;
    }
    TensorDesc transposeOutDesc;
    EsTensorHolder transposeOut;
    if (isArch35) {
        TransposeNodeConfig config = TransposeNodeConfig::Create(iFilterHolder, TRANSPOSE_PERM,
                                                                 nodeNamePrefix + "/Transpose", originFormat);
        OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateTransposeNode(builder, config, transposeOut, transposeOutDesc,
                                                                      GetNodeType()),
                    OP_LOGE(GetNodeType().GetString(), "Create Transpose failed"), return false);
    } else {
        TransposeNodeConfig config = TransposeNodeConfig::Create(iFilterHolder, TRANSPOSE_PERM,
                                                                 nodeNamePrefix + "/TransposeD", originFormat);
        OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateTransposeDNode(builder, config, transposeOut, transposeOutDesc,
                                                                       filterDesc, GetNodeType()),
                    OP_LOGE(GetNodeType().GetString(), "Create TransposeD failed"), return false);
    }
    reshapeInputHolder = transposeOut;
    reshapeInputDesc = transposeOutDesc;
    OP_LOGD(GetNodeType().GetString(), "Insert Transpose before filter.");
    return true;
}

bool DepthwiseDfFusionPass::BuildReshapeNode(EsGraphBuilder& builder, const std::vector<int64_t>& filterResetShape,
                                             const std::string& nodeNamePrefix,
                                             const EsTensorHolder& reshapeInputHolder,
                                             const TensorDesc& reshapeInputDesc, EsTensorHolder& reshapeOutput,
                                             TensorDesc& targetFilterDesc)
{
    const TensorDesc& filterDesc = isDynamic ? input1Desc : input0Desc;
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto* reshapeInProducer = reshapeInputHolder.GetProducer();
    OP_CHECK_IF(reshapeInProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "Reshape input producer nullptr"),
                return false);
    TensorDesc reshapeOutDesc(reshapeInputDesc);
    reshapeOutDesc.SetShape(ge::Shape(filterResetShape));
    reshapeOutDesc.SetOriginShape(ge::Shape(filterResetShape));
    std::string reshapeName = nodeNamePrefix + "/Reshape";
    GNode reshapeNode = CompliantNodeBuilder(graph)
                            .OpType("Reshape")
                            .Name(reshapeName.c_str())
                            .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                            .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .Build();
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *reshapeInProducer, reshapeInputHolder.GetProducerOutIndex(),
                                         reshapeNode, 0) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge to reshape x failed"), return false);
    OP_CHECK_IF(reshapeNode.UpdateInputDesc(0, reshapeInputDesc) != GRAPH_SUCCESS ||
                    reshapeNode.UpdateOutputDesc(0, reshapeOutDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update reshape node desc failed"), return false);
    std::vector<int64_t> shapeAttr = filterResetShape;
    OP_CHECK_IF(reshapeNode.SetAttr("shape", shapeAttr) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Set shape attr failed"), return false);
    reshapeOutput = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(reshapeNode, OUTPUT_INDEX));
    targetFilterDesc = filterDesc;
    targetFilterDesc.SetShape(ge::Shape(filterResetShape));
    targetFilterDesc.SetOriginShape(ge::Shape(filterResetShape));
    return true;
}

GraphUniqPtr DepthwiseDfFusionPass::BuildDynamicTargetNode(EsGraphBuilder& builder, const std::string& targetOpType,
                                                           const std::string& targetNodeName,
                                                           const EsTensorHolder& iInputSizeHolder,
                                                           const EsTensorHolder& iGradOutputHolder,
                                                           const EsTensorHolder& reshapeOutput,
                                                           const TensorDesc& targetFilterDesc)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    const std::string outputName = isArch35 ? "y" : "input_grad";
    GNode targetNode = CompliantNodeBuilder(graph)
                           .OpType(targetOpType.c_str())
                           .Name(targetNodeName.c_str())
                           .IrDefInputs({{"input_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{outputName, CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .Build();
    auto* inputSizeProducer = iInputSizeHolder.GetProducer();
    OP_CHECK_IF(inputSizeProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "input_size producer nullptr"),
                return nullptr);
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *inputSizeProducer, iInputSizeHolder.GetProducerOutIndex(), targetNode,
                                         INPUT_SIZE_INDEX) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge input_size failed"), return nullptr);
    auto* reshapeProducer = reshapeOutput.GetProducer();
    OP_CHECK_IF(reshapeProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "reshape output producer nullptr"),
                return nullptr);
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *reshapeProducer, reshapeOutput.GetProducerOutIndex(), targetNode,
                                         static_cast<int32_t>(FILTER_INDEX_DYN)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge filter to target failed"), return nullptr);
    auto* gradOutProducer = iGradOutputHolder.GetProducer();
    OP_CHECK_IF(gradOutProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "grad out producer nullptr"),
                return nullptr);
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *gradOutProducer, iGradOutputHolder.GetProducerOutIndex(), targetNode,
                                         static_cast<int32_t>(GRAD_OUTPUT_INDEX_DYN)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge grad output to target failed"), return nullptr);
    SetNodeAttrs(targetNode);
    OP_CHECK_IF(targetNode.SetAttr("from_depthwise", convBpAttr.from_depthwise) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Set from_depthwise attr failed"), return nullptr);
    OP_CHECK_IF(targetNode.UpdateInputDesc(INPUT_SIZE_INDEX, input0Desc) != GRAPH_SUCCESS ||
                    targetNode.UpdateInputDesc(FILTER_INDEX_DYN, targetFilterDesc) != GRAPH_SUCCESS ||
                    targetNode.UpdateInputDesc(GRAD_OUTPUT_INDEX_DYN, input2Desc) != GRAPH_SUCCESS ||
                    targetNode.UpdateOutputDesc(OUTPUT_INDEX, outputDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update target node desc failed"), return nullptr);
    auto targetOutput = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(targetNode, OUTPUT_INDEX));
    OP_LOGD(GetNodeType().GetString(), "DepthwiseDF Replacement success (type=%s)", targetOpType.c_str());
    return builder.BuildAndReset(std::vector<EsTensorHolder>{targetOutput});
}

GraphUniqPtr DepthwiseDfFusionPass::BuildStaticTargetNode(EsGraphBuilder& builder, const std::string& targetOpType,
                                                          const std::string& targetNodeName,
                                                          const EsTensorHolder& iGradOutputHolder,
                                                          const EsTensorHolder& reshapeOutput,
                                                          const TensorDesc& targetFilterDesc)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode targetNode = CompliantNodeBuilder(graph)
                           .OpType(targetOpType.c_str())
                           .Name(targetNodeName.c_str())
                           .IrDefInputs({{"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"input_grad", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .Build();
    auto* reshapeProducer = reshapeOutput.GetProducer();
    OP_CHECK_IF(reshapeProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "reshape output producer nullptr"),
                return nullptr);
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *reshapeProducer, reshapeOutput.GetProducerOutIndex(), targetNode,
                                         static_cast<int32_t>(FILTER_INDEX_D)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge filter to target failed"), return nullptr);
    auto* gradOutProducer = iGradOutputHolder.GetProducer();
    OP_CHECK_IF(gradOutProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "grad out producer nullptr"),
                return nullptr);
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *gradOutProducer, iGradOutputHolder.GetProducerOutIndex(), targetNode,
                                         static_cast<int32_t>(GRAD_OUTPUT_INDEX_D)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge grad output to target failed"), return nullptr);
    SetNodeAttrs(targetNode);
    OP_CHECK_IF(targetNode.SetAttr("from_depthwise", convBpAttr.from_depthwise) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Set from_depthwise attr failed"), return nullptr);
    OP_CHECK_IF(targetNode.UpdateInputDesc(FILTER_INDEX_D, targetFilterDesc) != GRAPH_SUCCESS ||
                    targetNode.UpdateInputDesc(GRAD_OUTPUT_INDEX_D, input1Desc) != GRAPH_SUCCESS ||
                    targetNode.UpdateOutputDesc(OUTPUT_INDEX, outputDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update target node desc failed"), return nullptr);
    auto targetOutput = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(targetNode, OUTPUT_INDEX));
    OP_LOGD(GetNodeType().GetString(), "DepthwiseDF Replacement success (D, type=%s)", targetOpType.c_str());
    return builder.BuildAndReset(std::vector<EsTensorHolder>{targetOutput});
}

GraphUniqPtr DepthwiseDfFusionPass::Replacement(const GNode& matchedNode)
{
    OP_LOGD(GetNodeType().GetString(), "Replacement start");
    AscendString matchedTypeAsc;
    OP_CHECK_IF(matchedNode.GetType(matchedTypeAsc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "GetType failed"), return nullptr);
    std::string matchedType(matchedTypeAsc.GetString());
    OP_CHECK_IF(!GetNodeAttrs(matchedNode), OP_LOGE(GetNodeType().GetString(), "GetNodeAttrs failed"), return nullptr);
    std::string nodeNamePrefix;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::GetNodeName(matchedNode, nodeNamePrefix),
                OP_LOGE(GetNodeType().GetString(), "GetNodeName failed"), return nullptr);
    std::vector<int64_t> filterResetShape;
    OP_CHECK_IF(!GetResizeDepthwiseFilterShape(filterResetShape),
                OP_LOGE(GetNodeType().GetString(), "Compute filter resize shape failed"), return nullptr);
    auto builder = EsGraphBuilder("replacement");
    OP_CHECK_IF(builder.GetCGraphBuilder()->GetGraph() == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Get graph failed"), return nullptr);
    EsTensorHolder iFilterHolder, iGradOutputHolder, iInputSizeHolder;
    CreateBoundaryInputs(builder, iFilterHolder, iGradOutputHolder, iInputSizeHolder);
    EsTensorHolder reshapeInputHolder;
    TensorDesc reshapeInputDesc;
    OP_CHECK_IF(!BuildOptionalTranspose(builder, nodeNamePrefix, iFilterHolder, reshapeInputHolder, reshapeInputDesc),
                OP_LOGE(GetNodeType().GetString(), "BuildOptionalTranspose failed"), return nullptr);
    EsTensorHolder reshapeOutput;
    TensorDesc targetFilterDesc;
    OP_CHECK_IF(!BuildReshapeNode(builder, filterResetShape, nodeNamePrefix, reshapeInputHolder, reshapeInputDesc,
                                  reshapeOutput, targetFilterDesc),
                OP_LOGE(GetNodeType().GetString(), "BuildReshapeNode failed"), return nullptr);
    std::string targetOpType = matchedType;
    if (isArch35) {
        targetOpType.erase(0, strlen(DEPTHWISE_PREFIX));
    }
    std::string targetNodeName = nodeNamePrefix + "/dx";
    if (isDynamic) {
        return BuildDynamicTargetNode(builder, targetOpType, targetNodeName, iInputSizeHolder, iGradOutputHolder,
                                      reshapeOutput, targetFilterDesc);
    }
    return BuildStaticTargetNode(builder, targetOpType, targetNodeName, iGradOutputHolder, reshapeOutput,
                                 targetFilterDesc);
}

const std::vector<AscendString> kMatchOpTypes = {DEPTHWISE_D, DEPTHWISE_DYN};

REG_DECOMPOSE_PASS(DepthwiseDfFusionPass, kMatchOpTypes).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
