/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "depthwise_dw_mul_fusion_pass.h"

#include <cstring>
#include <string>
#include <vector>

namespace ops {
using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace ConvBackpropFusionUtils;
namespace {

const AscendString PASS_NAME = "DepthwiseDwMulFusionPass";
const AscendString DEPTHWISE_D = "DepthwiseConv2DBackpropFilterD";
const AscendString DEPTHWISE_DYN = "DepthwiseConv2DBackpropFilter";

constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t FILTER_SIZE_INDEX_DYN = 1U;
constexpr uint32_t OUT_BACKPROP_INDEX_DYN = 2U;
constexpr uint32_t OUT_BACKPROP_INDEX_D = 1U;
constexpr int64_t GROUP_MAX_RANGE = 65535;
constexpr size_t FILTER_DIM_EXPECT = 4;

const std::vector<int32_t> TRANSPOSE_PERM = {1, 0, 2, 3};

// NCHW格式的维度索引
constexpr int32_t N_DIM_NCHW_INDEX = 0;
constexpr int32_t C_DIM_NCHW_INDEX = 1;
constexpr int32_t H_DIM_NCHW_INDEX = 2;
constexpr int32_t W_DIM_NCHW_INDEX = 3;

// HWCN格式的维度索引
constexpr int32_t H_DIM_HWCN_INDEX = 0;
constexpr int32_t W_DIM_HWCN_INDEX = 1;
constexpr int32_t C_DIM_HWCN_INDEX = 2;
constexpr int32_t N_DIM_HWCN_INDEX = 3;

constexpr const char* DEPTHWISE_PREFIX = "Depthwise";

} // anonymous namespace

AscendString DepthwiseDwMulFusionPass::GetNodeType() const { return PASS_NAME; }

bool DepthwiseDwMulFusionPass::GetResizeDepthwiseFilterShape(std::vector<int64_t>& resizeShape)
{
    ge::Format format = outputDesc.GetOriginFormat();
    std::vector<int64_t> oriShape = outputDesc.GetShape().GetDims();
    if (oriShape.size() != FILTER_DIM_EXPECT) {
        OP_LOGE(GetNodeType().GetString(), "filter dim only supports 4 dims, got %zu.", oriShape.size());
        return false;
    }
    if (format == ge::FORMAT_NCHW) {
        resizeShape = {oriShape[N_DIM_NCHW_INDEX] * oriShape[C_DIM_NCHW_INDEX], 1, oriShape[H_DIM_NCHW_INDEX],
                       oriShape[W_DIM_NCHW_INDEX]};
    } else {
        resizeShape = {oriShape[H_DIM_HWCN_INDEX], oriShape[W_DIM_HWCN_INDEX], 1,
                       oriShape[N_DIM_HWCN_INDEX] * oriShape[C_DIM_HWCN_INDEX]};
    }
    return true;
}

bool DepthwiseDwMulFusionPass::ValidateFilterDesc(const GNode& matchedNode)
{
    std::vector<int64_t> filterSize;
    if (isDynamic) {
        filterSize = outputDesc.GetShape().GetDims();
    } else {
        OP_CHECK_IF(matchedNode.GetAttr("filter_size", filterSize) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Failed to get filter_size attr"), return false);
    }

    OP_CHECK_IF(filterSize.size() != FILTER_DIM_EXPECT,
                OP_LOGE(GetNodeType().GetString(), "filter dim only supports 4 dims, got %zu.", filterSize.size()),
                return false);
    if (isArch35) {
        for (auto filterDim : filterSize) {
            OP_CHECK_IF(filterDim <= 0,
                        OP_LOGE(GetNodeType().GetString(), "%ld in filter shape should be positive.", filterDim),
                        return false);
        }
    }

    int64_t groups = 0;
    OP_CHECK_IF(matchedNode.GetAttr("groups", groups) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Failed to get groups attr"), return false);
    OP_CHECK_IF(groups <= 0 || groups > GROUP_MAX_RANGE,
                OP_LOGE(GetNodeType().GetString(), "groups=%ld should be in range [1, 65535].", groups), return false);
    return true;
}

bool DepthwiseDwMulFusionPass::MeetRequirements(const GNode& matchedNode)
{
    AscendString matchedTypeAsc;
    OP_CHECK_IF(matchedNode.GetType(matchedTypeAsc) != GRAPH_SUCCESS,
                OP_LOGD(GetNodeType().GetString(), "GetType failed"), return false);
    isDynamic = (matchedTypeAsc.GetString() == std::string(DEPTHWISE_DYN.GetString()));
    isArch35 = ConvBackpropFusionUtilsPass::IsArch35();
    OP_CHECK_IF(!GetNodeDesc(matchedNode), OP_LOGE(GetNodeType().GetString(), "GetNodeDesc failed"), return false);
    OP_CHECK_IF(!ValidateFilterDesc(matchedNode), OP_LOGD(GetNodeType().GetString(), "ValidateFilterDesc failed"),
                return false);
    ge::Format originFormat = outputDesc.GetOriginFormat();
    OP_CHECK_IF(originFormat != ge::FORMAT_NCHW && originFormat != ge::FORMAT_HWCN,
                OP_LOGE(GetNodeType().GetString(), "filter origin format only supports NCHW or HWCN, got %d",
                        static_cast<int>(originFormat)),
                return false);
    OP_LOGD(GetNodeType().GetString(), "MeetRequirements passed");
    return true;
}

void DepthwiseDwMulFusionPass::CreateBoundaryInputs(EsGraphBuilder& builder, EsTensorHolder& iXHolder,
                                                    EsTensorHolder& iFilterSizeHolder,
                                                    EsTensorHolder& iGradOutputHolder)
{
    iXHolder = builder.CreateInput(X_INDEX);
    ConvBackpropFusionUtilsPass::SetPlaceholderDesc(iXHolder, TENSOR_DEFAULT_OUTPUT_INDEX, input0Desc);
    if (isDynamic) {
        iFilterSizeHolder = builder.CreateInput(FILTER_SIZE_INDEX_DYN);
        ConvBackpropFusionUtilsPass::SetPlaceholderDesc(iFilterSizeHolder, TENSOR_DEFAULT_OUTPUT_INDEX, input1Desc);
        iGradOutputHolder = builder.CreateInput(OUT_BACKPROP_INDEX_DYN);
        ConvBackpropFusionUtilsPass::SetPlaceholderDesc(iGradOutputHolder, TENSOR_DEFAULT_OUTPUT_INDEX, input2Desc);
    } else {
        iGradOutputHolder = builder.CreateInput(OUT_BACKPROP_INDEX_D);
        ConvBackpropFusionUtilsPass::SetPlaceholderDesc(iGradOutputHolder, TENSOR_DEFAULT_OUTPUT_INDEX, input1Desc);
    }
}

bool DepthwiseDwMulFusionPass::BuildTargetGNode(EsGraphBuilder& builder, const std::string& targetOpType,
                                                const std::string& targetNodeName, const EsTensorHolder& iXHolder,
                                                const EsTensorHolder& iFilterSizeHolder,
                                                const EsTensorHolder& iGradOutputHolder, GNode& targetNode) const
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(graph == nullptr, OP_LOGE(GetNodeType().GetString(), "Get graph failed"), return false);
    if (isDynamic) {
        targetNode = CompliantNodeBuilder(graph)
                         .OpType(targetOpType.c_str())
                         .Name(targetNodeName.c_str())
                         .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"filter_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
        auto* filterSizeProducer = iFilterSizeHolder.GetProducer();
        OP_CHECK_IF(filterSizeProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "filter_size producer nullptr"),
                    return false);
        OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *filterSizeProducer, iFilterSizeHolder.GetProducerOutIndex(),
                                             targetNode, static_cast<int32_t>(FILTER_SIZE_INDEX_DYN)) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Add edge filter_size failed"), return false);
        auto* gradOutProducer = iGradOutputHolder.GetProducer();
        OP_CHECK_IF(gradOutProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "grad out producer nullptr"),
                    return false);
        OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *gradOutProducer, iGradOutputHolder.GetProducerOutIndex(),
                                             targetNode, static_cast<int32_t>(OUT_BACKPROP_INDEX_DYN)) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Add edge out_backprop failed"), return false);
    } else {
        targetNode = CompliantNodeBuilder(graph)
                         .OpType(targetOpType.c_str())
                         .Name(targetNodeName.c_str())
                         .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
        auto* gradOutProducer = iGradOutputHolder.GetProducer();
        OP_CHECK_IF(gradOutProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "grad out producer nullptr"),
                    return false);
        OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *gradOutProducer, iGradOutputHolder.GetProducerOutIndex(),
                                             targetNode, static_cast<int32_t>(OUT_BACKPROP_INDEX_D)) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Add edge out_backprop failed"), return false);
    }
    auto* xProducer = iXHolder.GetProducer();
    OP_CHECK_IF(xProducer == nullptr, OP_LOGE(GetNodeType().GetString(), "x producer nullptr"), return false);
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, *xProducer, iXHolder.GetProducerOutIndex(), targetNode,
                                         static_cast<int32_t>(X_INDEX)) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge x failed"), return false);
    return true;
}

bool DepthwiseDwMulFusionPass::UpdateTargetNodeDescs(GNode& targetNode, TensorDesc& targetOutDesc)
{
    SetNodeAttrs(targetNode);
    convBpAttr.from_depthwise = true;
    OP_CHECK_IF(targetNode.SetAttr("from_depthwise", convBpAttr.from_depthwise) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Set from_depthwise attr failed"), return false);

    targetOutDesc = TensorDesc(outputDesc);
    std::vector<int64_t> filterResetShape;
    ge::Format originFormat = outputDesc.GetOriginFormat();
    OP_CHECK_IF(!GetResizeDepthwiseFilterShape(filterResetShape),
                OP_LOGE(GetNodeType().GetString(), "Compute filter reset shape failed"), return false);
    targetOutDesc.SetShape(ge::Shape(filterResetShape));
    targetOutDesc.SetOriginShape(ge::Shape(filterResetShape));

    if (!isDynamic) {
        OP_CHECK_IF(targetNode.SetAttr("filter_size", filterResetShape) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Set filter_size attr failed"), return false);
    }

    OP_CHECK_IF(targetNode.UpdateInputDesc(X_INDEX, input0Desc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update target input x desc failed"), return false);
    if (isDynamic) {
        OP_CHECK_IF(targetNode.UpdateInputDesc(FILTER_SIZE_INDEX_DYN, input1Desc) != GRAPH_SUCCESS ||
                        targetNode.UpdateInputDesc(OUT_BACKPROP_INDEX_DYN, input2Desc) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Update target input desc failed"), return false);
    } else {
        OP_CHECK_IF(targetNode.UpdateInputDesc(OUT_BACKPROP_INDEX_D, input1Desc) != GRAPH_SUCCESS,
                    OP_LOGE(GetNodeType().GetString(), "Update target input desc failed"), return false);
    }
    OP_CHECK_IF(targetNode.UpdateOutputDesc(OUTPUT_INDEX, targetOutDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update target output desc failed"), return false);
    return true;
}

bool DepthwiseDwMulFusionPass::BuildTargetNode(EsGraphBuilder& builder, const std::string& targetOpType,
                                               const std::string& targetNodeName, const EsTensorHolder& iXHolder,
                                               const EsTensorHolder& iFilterSizeHolder,
                                               const EsTensorHolder& iGradOutputHolder, GNode& targetNode,
                                               TensorDesc& targetOutDesc)
{
    OP_CHECK_IF(!BuildTargetGNode(builder, targetOpType, targetNodeName, iXHolder, iFilterSizeHolder, iGradOutputHolder,
                                  targetNode),
                OP_LOGE(GetNodeType().GetString(), "BuildTargetGNode failed"), return false);
    OP_CHECK_IF(!UpdateTargetNodeDescs(targetNode, targetOutDesc),
                OP_LOGE(GetNodeType().GetString(), "UpdateTargetNodeDescs failed"), return false);
    OP_LOGD(GetNodeType().GetString(), "Build target node success (type=%s)", targetOpType.c_str());
    return true;
}

bool DepthwiseDwMulFusionPass::BuildReshapeNode(EsGraphBuilder& builder, const std::string& nodeNamePrefix,
                                                GNode& targetNode, const std::vector<int64_t>& reshapeOutShape,
                                                const TensorDesc& reshapeInDesc, EsTensorHolder& reshapeOutput,
                                                TensorDesc& reshapeOutDesc) const
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    OP_CHECK_IF(graph == nullptr, OP_LOGE(GetNodeType().GetString(), "Get graph failed for reshape"), return false);

    std::string reshapeName = nodeNamePrefix + "/Reshape";
    GNode reshapeNode = CompliantNodeBuilder(graph)
                            .OpType("Reshape")
                            .Name(reshapeName.c_str())
                            .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                            .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .Build();
    OP_CHECK_IF(AddEdgeAndUpdatePeerDesc(*graph, targetNode, OUTPUT_INDEX, reshapeNode, 0) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Add edge to reshape failed"), return false);

    reshapeOutDesc = TensorDesc(reshapeInDesc);
    reshapeOutDesc.SetShape(ge::Shape(reshapeOutShape));
    reshapeOutDesc.SetOriginShape(ge::Shape(reshapeOutShape));

    OP_CHECK_IF(reshapeNode.UpdateInputDesc(0, reshapeInDesc) != GRAPH_SUCCESS ||
                    reshapeNode.UpdateOutputDesc(0, reshapeOutDesc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Update reshape node desc failed"), return false);
    std::vector<int64_t> shapeAttr = reshapeOutShape;
    OP_CHECK_IF(reshapeNode.SetAttr("shape", shapeAttr) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "Set shape attr failed"), return false);

    reshapeOutput = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(reshapeNode, OUTPUT_INDEX));
    OP_LOGD(GetNodeType().GetString(), "Build reshape node success");
    return true;
}

bool DepthwiseDwMulFusionPass::BuildOptionalTranspose(EsGraphBuilder& builder, ge::Format originFormat,
                                                      const std::string& nodeNamePrefix,
                                                      const EsTensorHolder& reshapeOutput,
                                                      const TensorDesc& reshapeOutDesc, TensorDesc& finalOutDesc,
                                                      EsTensorHolder& finalOutput) const
{
    if (isArch35) {
        TransposeNodeConfig config = TransposeNodeConfig::Create(reshapeOutput, TRANSPOSE_PERM,
                                                                 nodeNamePrefix + "/Transpose", originFormat);
        OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateTransposeNode(builder, config, finalOutput, finalOutDesc,
                                                                      GetNodeType()),
                    OP_LOGE(GetNodeType().GetString(), "Create Transpose failed"), return false);
    } else {
        TransposeNodeConfig config = TransposeNodeConfig::Create(reshapeOutput, TRANSPOSE_PERM,
                                                                 nodeNamePrefix + "/TransposeD", originFormat);
        OP_CHECK_IF(!ConvBackpropFusionUtilsPass::CreateTransposeDNode(builder, config, finalOutput, finalOutDesc,
                                                                       reshapeOutDesc, GetNodeType()),
                    OP_LOGE(GetNodeType().GetString(), "Create TransposeD failed"), return false);
    }
    OP_LOGD(GetNodeType().GetString(), "Insert Transpose after reshape.");
    return true;
}

GraphUniqPtr DepthwiseDwMulFusionPass::Replacement(const GNode& matchedNode)
{
    OP_LOGD(GetNodeType().GetString(), "Replacement start");
    AscendString matchedTypeAsc;
    OP_CHECK_IF(matchedNode.GetType(matchedTypeAsc) != GRAPH_SUCCESS,
                OP_LOGE(GetNodeType().GetString(), "GetType failed"), return nullptr);
    std::string targetOpType(matchedTypeAsc.GetString());
    OP_CHECK_IF(!GetNodeAttrs(matchedNode), OP_LOGE(GetNodeType().GetString(), "GetNodeAttrs failed"), return nullptr);
    std::string nodeNamePrefix;
    OP_CHECK_IF(!ConvBackpropFusionUtilsPass::GetNodeName(matchedNode, nodeNamePrefix),
                OP_LOGE(GetNodeType().GetString(), "GetNodeName failed"), return nullptr);
    ge::Format originFormat = outputDesc.GetOriginFormat();
    auto builder = EsGraphBuilder("replacement");
    OP_CHECK_IF(builder.GetCGraphBuilder()->GetGraph() == nullptr,
                OP_LOGE(GetNodeType().GetString(), "Get graph failed"), return nullptr);
    EsTensorHolder iXHolder, iFilterSizeHolder, iGradOutputHolder;
    CreateBoundaryInputs(builder, iXHolder, iFilterSizeHolder, iGradOutputHolder);
    targetOpType.erase(0, strlen(DEPTHWISE_PREFIX));
    GNode targetNode;
    TensorDesc targetOutDesc;
    OP_CHECK_IF(!BuildTargetNode(builder, targetOpType, nodeNamePrefix + "/dw", iXHolder, iFilterSizeHolder,
                                 iGradOutputHolder, targetNode, targetOutDesc),
                OP_LOGE(GetNodeType().GetString(), "BuildTargetNode failed"), return nullptr);
    std::vector<int64_t> reshapeOutShape(outputDesc.GetShape().GetDims());
    if (originFormat == ge::FORMAT_NCHW) {
        std::swap(reshapeOutShape[0], reshapeOutShape[1]);
    }
    EsTensorHolder reshapeOutput;
    TensorDesc reshapeOutDesc;
    OP_CHECK_IF(!BuildReshapeNode(builder, nodeNamePrefix, targetNode, reshapeOutShape, targetOutDesc, reshapeOutput,
                                  reshapeOutDesc),
                OP_LOGE(GetNodeType().GetString(), "BuildReshapeNode failed"), return nullptr);
    EsTensorHolder finalOutput;
    TensorDesc finalOutDesc;
    if (originFormat == ge::FORMAT_NCHW) {
        OP_CHECK_IF(!BuildOptionalTranspose(builder, originFormat, nodeNamePrefix, reshapeOutput, reshapeOutDesc,
                                            finalOutDesc, finalOutput),
                    OP_LOGE(GetNodeType().GetString(), "BuildOptionalTranspose failed"), return nullptr);
    } else {
        finalOutput = reshapeOutput;
    }
    OP_LOGD(GetNodeType().GetString(), "DepthwiseDwMul Replacement success");
    return builder.BuildAndReset(std::vector<EsTensorHolder>{finalOutput});
}

const std::vector<AscendString> kMatchOpTypes = {DEPTHWISE_D, DEPTHWISE_DYN};

REG_DECOMPOSE_PASS(DepthwiseDwMulFusionPass, kMatchOpTypes).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
