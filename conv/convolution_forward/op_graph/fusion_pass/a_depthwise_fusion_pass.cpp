/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "a_depthwise_fusion_pass.h"

#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
#include "register/register_custom_pass.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace ADepthwiseFusion;
using namespace ge;
using namespace fusion;

void ADepthwiseFusionPass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    convDescInfo = ConvDescInfo();
    reshapeCounter = 0;
}

bool ADepthwiseFusionPass::MeetRequirements(const GNode &depthwiseNode)
{
    OP_LOGD(convDescInfo.nodeNameStr, "Begin to do ADepthwiseFusionPass.");

    auto filterShape = convDescInfo.filterDesc.GetOriginShape().GetDims();
    FUSION_PASS_CHECK(filterShape.size() != static_cast<size_t>(FILTER_SHAPE_SIZE),
        OP_LOGD(FUSION_NAME, "%s filter is not 4D, not changed.", convDescInfo.nodeNameStr.c_str()),
        return false);

    auto filterFormat = convDescInfo.filterDesc.GetOriginFormat();
    int64_t filterC = 0;
    if (filterFormat == Format::FORMAT_HWCN) {
        filterC = filterShape[INDEX_2];
    } else if (filterFormat == Format::FORMAT_NCHW) {
        filterC = filterShape[INDEX_1];
    } else {
        OP_LOGD(FUSION_NAME, "%s filter format is not NCHW or HWCN, not changed.",
            convDescInfo.nodeNameStr.c_str());
        return false;
    }

    FUSION_PASS_CHECK(filterC == ALREADY_CHANGED_C,
        OP_LOGD(FUSION_NAME, "%s filter Cin [%ld] of DepthwiseConv2D is equal to 1, graph will not be changed.",
            convDescInfo.nodeNameStr.c_str(), filterC),
        return false);

    return true;
}

AscendString ADepthwiseFusionPass::GetNodeType() const
{
    return DEPTHWISE_CONV2D;
}

void ADepthwiseFusionPass::PrintGraphStructure() const
{
    OP_LOGI(FUSION_NAME, "%s ADepthwiseFusionPass done, inserted reshape count [%ld].",
        convDescInfo.nodeNameStr.c_str(), reshapeCounter);
}

Status ADepthwiseFusionPass::ConvFusionPreImpl(GraphPtr &graph, GNode &depthwiseNode,
    const CustomPassContext &pass_context)
{
    FUSION_PASS_CHECK(graph == nullptr,
        OP_LOGE(FUSION_NAME, "%s graph is nullptr.", convDescInfo.nodeNameStr.c_str()), return FAILED);

    GNodePtr filterProducer = depthwiseNode.GetInDataNodesAndPortIndexs(FILTER_INPUT_INDEX).first;
    FUSION_PASS_CHECK(filterProducer == nullptr,
        OP_LOGE(FUSION_NAME, "%s get filter producer node failed.", convDescInfo.nodeNameStr.c_str()),
        return FAILED);

    AscendString producerType;
    FUSION_PASS_CHECK(filterProducer->GetType(producerType) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s get filter producer type failed.", convDescInfo.nodeNameStr.c_str()), return FAILED);

    if (producerType == ASCEND_WEIGHT_QUANT) {
        FUSION_PASS_CHECK_NOLOG(!DealQuantNodeCase(*graph, *filterProducer, depthwiseNode), return FAILED);
    } else {
        FUSION_PASS_CHECK_NOLOG(!InsertReshapeForConsumers(*graph, *filterProducer, TARGET_TYPES), return FAILED);
    }
    return SUCCESS;
}

bool ADepthwiseFusionPass::DealQuantNodeCase(Graph &graph, GNode &quantNode, GNode &depthwiseNode)
{
    FUSION_PASS_CHECK_NOLOG(!UpdateQuantAndDepthwiseFilterDesc(quantNode, depthwiseNode), return false);

    for (size_t inIdx = 0; inIdx < quantNode.GetInputsSize(); ++inIdx) {
        auto upstreamPair = quantNode.GetInDataNodesAndPortIndexs(static_cast<int32_t>(inIdx));
        GNodePtr upstreamProducer = upstreamPair.first;
        if (upstreamProducer == nullptr) {
            continue;
        }
        TensorDesc upstreamOutDesc;
        FUSION_PASS_CHECK(upstreamProducer->GetOutputDesc(upstreamPair.second, upstreamOutDesc) != GRAPH_SUCCESS,
            OP_LOGE(FUSION_NAME, "%s get upstream producer output desc failed.", convDescInfo.nodeNameStr.c_str()),
            return false);
        Format upstreamFormat = upstreamOutDesc.GetOriginFormat();
        if (upstreamFormat != Format::FORMAT_HWCN && upstreamFormat != Format::FORMAT_NCHW) {
            continue;
        }
        FUSION_PASS_CHECK_NOLOG(!InsertReshapeForConsumers(graph, *upstreamProducer, TARGET_TYPES_QUANT), return false);
    }
    return true;
}

bool ADepthwiseFusionPass::UpdateQuantAndDepthwiseFilterDesc(GNode &quantNode, GNode &depthwiseNode) const
{
    TensorDesc filterDesc;
    FUSION_PASS_CHECK(depthwiseNode.GetInputDesc(FILTER_INPUT_INDEX, filterDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s get depthwise filter input desc failed.", convDescInfo.nodeNameStr.c_str()),
        return false);

    FilterShapeResult shapeResult;
    FUSION_PASS_CHECK(!ComputeFilterShapes(filterDesc.GetOriginShape().GetDims(), filterDesc.GetOriginFormat(),
        shapeResult), OP_LOGE(FUSION_NAME, "%s compute quant filter shapes failed.", convDescInfo.nodeNameStr.c_str()),
        return false);

    TensorDesc quantOutDesc;
    FUSION_PASS_CHECK(quantNode.GetOutputDesc(OUTPUT_INDEX, quantOutDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s get quant node output desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    SetTensorDescShape(quantOutDesc, shapeResult.newShape);
    FUSION_PASS_CHECK(quantNode.UpdateOutputDesc(OUTPUT_INDEX, quantOutDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s update quant node output desc failed.", convDescInfo.nodeNameStr.c_str()),
        return false);

    SetTensorDescShape(filterDesc, shapeResult.newShape);
    FUSION_PASS_CHECK(depthwiseNode.UpdateInputDesc(FILTER_INPUT_INDEX, filterDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s update depthwise filter input desc failed.", convDescInfo.nodeNameStr.c_str()),
        return false);

    return true;
}

bool ADepthwiseFusionPass::InsertReshapeForConsumers(Graph &graph, GNode &producer,
    const std::set<AscendString> &targetTypes)
{
    for (size_t outIdx = 0; outIdx < producer.GetOutputsSize(); ++outIdx) {
        std::vector<int64_t> curPreShape;
        std::vector<int64_t> curNewShape;
        bool resolved = false;
        FUSION_PASS_CHECK_NOLOG(!ResolveReshapeShapes(producer, static_cast<int32_t>(outIdx), resolved, curPreShape,
            curNewShape), return false);
        if (!resolved) {
            continue;
        }

        auto consumers = producer.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(outIdx));
        for (auto &consumerPair : consumers) {
            if (consumerPair.first == nullptr) {
                continue;
            }
            GNode &consumer = *consumerPair.first;
            int32_t consumerInIdx = consumerPair.second;

            AscendString consumerType;
            FUSION_PASS_CHECK(consumer.GetType(consumerType) != GRAPH_SUCCESS,
                OP_LOGE(FUSION_NAME, "%s get consumer type failed.", convDescInfo.nodeNameStr.c_str()), return false);
            if (targetTypes.find(consumerType) == targetTypes.end()) {
                continue;
            }

            TensorDesc consumerInDesc;
            FUSION_PASS_CHECK(consumer.GetInputDesc(consumerInIdx, consumerInDesc) != GRAPH_SUCCESS,
                OP_LOGE(FUSION_NAME, "%s get consumer input desc failed.", convDescInfo.nodeNameStr.c_str()),
                return false);

            ReshapeDescInfo descInfo = BuildReshapeDescInfo(consumerInDesc, curPreShape, curNewShape);
            ReshapeEdgeInfo edge{graph, producer, consumer, static_cast<int32_t>(outIdx), consumerInIdx};
            FUSION_PASS_CHECK_NOLOG(!InsertReshapeBetween(edge, descInfo), return false);
            FUSION_PASS_CHECK_NOLOG(!UpdateConsumerInputShape(consumer, consumerInIdx, consumerInDesc, curNewShape),
                return false);
        }
    }
    return true;
}

bool ADepthwiseFusionPass::ResolveReshapeShapes(GNode &producer, int32_t outIdx, bool &resolved,
    std::vector<int64_t> &curPreShape, std::vector<int64_t> &curNewShape)
{
    resolved = false;
    TensorDesc producerOutDesc;
    FUSION_PASS_CHECK(producer.GetOutputDesc(outIdx, producerOutDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s get producer output desc failed.", convDescInfo.nodeNameStr.c_str()), return false);

    FilterShapeResult shapeResult;
    FUSION_PASS_CHECK_NOLOG(!ComputeFilterShapes(producerOutDesc.GetOriginShape().GetDims(),
        producerOutDesc.GetOriginFormat(), shapeResult), return true);
    curPreShape = shapeResult.preShape;
    curNewShape = shapeResult.newShape;
    resolved = true;
    return true;
}

ReshapeDescInfo ADepthwiseFusionPass::BuildReshapeDescInfo(const TensorDesc &consumerInDesc,
    const std::vector<int64_t> &curPreShape, const std::vector<int64_t> &curNewShape) const
{
    ReshapeDescInfo descInfo;
    descInfo.inDesc = consumerInDesc;
    SetTensorDescShape(descInfo.inDesc, curPreShape);
    descInfo.outDesc = consumerInDesc;
    SetTensorDescShape(descInfo.outDesc, curNewShape);
    descInfo.shapeAttr = curNewShape;
    return descInfo;
}

bool ADepthwiseFusionPass::UpdateConsumerInputShape(GNode &consumer, int32_t consumerInIdx,
    const TensorDesc &consumerInDesc, const std::vector<int64_t> &newShape) const
{
    TensorDesc updatedInDesc = consumerInDesc;
    SetTensorDescShape(updatedInDesc, newShape);
    FUSION_PASS_CHECK(consumer.UpdateInputDesc(consumerInIdx, updatedInDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s update consumer input desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    return true;
}

bool ADepthwiseFusionPass::InsertReshapeBetween(const ReshapeEdgeInfo &edge, const ReshapeDescInfo &descInfo)
{
    FUSION_PASS_CHECK(edge.graph.RemoveEdge(edge.src, edge.srcOutIdx, edge.dst, edge.dstInIdx) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s remove edge failed.", convDescInfo.nodeNameStr.c_str()), return false);

    AscendString srcName;
    FUSION_PASS_CHECK(edge.src.GetName(srcName) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s get src node name failed.", convDescInfo.nodeNameStr.c_str()), return false);

    GNode reshapeNode;
    FUSION_PASS_CHECK_NOLOG(!CreateReshapeNode(edge.graph, srcName, descInfo, reshapeNode), return false);

    FUSION_PASS_CHECK(
        edge.graph.AddDataEdge(edge.src, edge.srcOutIdx, reshapeNode, RESHAPE_INPUT_INDEX) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s add edge src->reshape failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(
        edge.graph.AddDataEdge(reshapeNode, RESHAPE_OUTPUT_INDEX, edge.dst, edge.dstInIdx) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s add edge reshape->dst failed.", convDescInfo.nodeNameStr.c_str()), return false);
    return true;
}

bool ADepthwiseFusionPass::CreateReshapeNode(Graph &graph, const AscendString &producerName,
    const ReshapeDescInfo &descInfo, GNode &reshapeNode)
{
    reshapeCounter++;
    std::string reshapeName = std::string(producerName.GetString()) + "/Reshape_" + std::to_string(reshapeCounter);

    reshapeNode = es::CompliantNodeBuilder(&graph)
                      .OpType(RESHAPE_OP.GetString())
                      .Name(reshapeName.c_str())
                      .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                      .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                      .Build();

    AscendString reshapeType;
    FUSION_PASS_CHECK(reshapeNode.GetType(reshapeType) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s get reshape node type failed.", convDescInfo.nodeNameStr.c_str()), return false);

    FUSION_PASS_CHECK(reshapeNode.UpdateInputDesc(RESHAPE_INPUT_INDEX, descInfo.inDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s update reshape input desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    FUSION_PASS_CHECK(reshapeNode.UpdateOutputDesc(RESHAPE_OUTPUT_INDEX, descInfo.outDesc) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s update reshape output desc failed.", convDescInfo.nodeNameStr.c_str()), return false);
    std::vector<int64_t> attrValue = descInfo.shapeAttr;
    FUSION_PASS_CHECK(reshapeNode.SetAttr(SHAPE_ATTR, attrValue) != GRAPH_SUCCESS,
        OP_LOGE(FUSION_NAME, "%s set reshape shape attr failed.", convDescInfo.nodeNameStr.c_str()), return false);
    return true;
}

bool ADepthwiseFusionPass::ComputeFilterShapes(const std::vector<int64_t> &filterShape, Format originFormat,
    FilterShapeResult &result) const
{
    if (originFormat == Format::FORMAT_HWCN) {
        result.h = filterShape[INDEX_0];
        result.w = filterShape[INDEX_1];
        result.c = filterShape[INDEX_2];
        result.n = filterShape[INDEX_3];
        result.preShape = {result.h, result.w, result.c, result.n};
        result.newShape = {result.h, result.w, ALREADY_CHANGED_C, result.c * result.n};
        return true;
    } else if (originFormat == Format::FORMAT_NCHW) {
        result.n = filterShape[INDEX_0];
        result.c = filterShape[INDEX_1];
        result.h = filterShape[INDEX_2];
        result.w = filterShape[INDEX_3];
        result.preShape = {result.n, result.c, result.h, result.w};
        result.newShape = {result.n * result.c, ALREADY_CHANGED_C, result.h, result.w};
        return true;
    }

    return false;
}

void ADepthwiseFusionPass::SetTensorDescShape(TensorDesc &desc, const std::vector<int64_t> &dims) const
{
    desc.SetShape(Shape(dims));
    desc.SetOriginShape(Shape(dims));
}

} // namespace Ops
