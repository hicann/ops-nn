/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "compliant_node_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/biasadd_conv_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;
using namespace Ops::NN::Conv;
using namespace BiasaddConvFusion;

namespace {

constexpr size_t FLOAT16_ELEMENT_BYTES = 2U;
constexpr size_t FLOAT_ELEMENT_BYTES = 4U;

static bool TryMultiplySize(size_t left, size_t right, size_t& result)
{
    if (right != 0U && left > std::numeric_limits<size_t>::max() / right) {
        return false;
    }
    result = left * right;
    return true;
}

static void SetOutputDesc(GNode& node, int32_t outIndex, const std::vector<int64_t>& dims, Format format,
                          DataType dtype)
{
    TensorDesc desc(Shape(dims), format, dtype);
    desc.SetOriginShape(Shape(dims));
    desc.SetOriginFormat(format);
    node.UpdateOutputDesc(outIndex, desc);
}

static void SetInputDesc(GNode& node, int32_t inIndex, const std::vector<int64_t>& dims, Format format, DataType dtype)
{
    TensorDesc desc(Shape(dims), format, dtype);
    desc.SetOriginShape(Shape(dims));
    desc.SetOriginFormat(format);
    node.UpdateInputDesc(inIndex, desc);
}

static es::EsTensorHolder BuildConstNode(es::EsGraphBuilder& builder, const std::string& name,
                                         const std::vector<int64_t>& dims, Format format, DataType dtype)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Const")
                     .Name(name.c_str())
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    SetOutputDesc(node, 0, dims, format, dtype);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildConstNodeWithValue(es::EsGraphBuilder& builder, const std::string& name,
                                                  const std::vector<int64_t>& dims, Format format, DataType dtype)
{
    auto holder = BuildConstNode(builder, name, dims, format, dtype);
    auto node = holder.GetProducer();
    TensorDesc desc;
    if (node == nullptr || node->GetOutputDesc(0, desc) != GRAPH_SUCCESS) {
        return holder;
    }

    size_t elementCount = 1U;
    for (const int64_t dim : dims) {
        if (dim <= 0) {
            elementCount = 0U;
            break;
        }
        if (!TryMultiplySize(elementCount, static_cast<size_t>(dim), elementCount)) {
            return es::EsTensorHolder();
        }
    }
    const size_t elementBytes = (dtype == DT_FLOAT16) ? FLOAT16_ELEMENT_BYTES : FLOAT_ELEMENT_BYTES;
    size_t byteCount = 0U;
    if (!TryMultiplySize(elementCount, elementBytes, byteCount)) {
        return es::EsTensorHolder();
    }
    std::vector<uint8_t> bytes(byteCount, 0U);
    Tensor value(desc, bytes);
    node->SetAttr("value", value);
    return holder;
}

struct ConstPayloadConfig {
    std::vector<int64_t> dims;
    Format format;
    DataType dtype;
    size_t elementCount;
};

static es::EsTensorHolder BuildConstNodeWithForcedPayload(es::EsGraphBuilder& builder, const std::string& name,
                                                          const ConstPayloadConfig& config)
{
    auto holder = BuildConstNode(builder, name, config.dims, config.format, config.dtype);
    auto node = holder.GetProducer();
    TensorDesc desc;
    if (node == nullptr || node->GetOutputDesc(0, desc) != GRAPH_SUCCESS) {
        return holder;
    }
    const size_t elementBytes = config.dtype == DT_FLOAT16 ? FLOAT16_ELEMENT_BYTES : FLOAT_ELEMENT_BYTES;
    size_t byteCount = 0U;
    if (!TryMultiplySize(config.elementCount, elementBytes, byteCount)) {
        return es::EsTensorHolder();
    }
    std::vector<uint8_t> bytes(byteCount, 0U);
    Tensor value(desc, bytes);
    node->SetAttr("value", value);
    return holder;
}

static es::EsTensorHolder BuildConstNodeWithOutputAndValueDescs(es::EsGraphBuilder& builder, const std::string& name,
                                                                const TensorDesc& outputDesc,
                                                                const TensorDesc& valueDesc, size_t elementCount)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Const")
                     .Name(name.c_str())
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    if (node.UpdateOutputDesc(0, outputDesc) != GRAPH_SUCCESS) {
        return es::EsTensorHolder();
    }
    const size_t elementBytes = valueDesc.GetDataType() == DT_FLOAT16 ? FLOAT16_ELEMENT_BYTES : FLOAT_ELEMENT_BYTES;
    size_t byteCount = 0U;
    if (!TryMultiplySize(elementCount, elementBytes, byteCount)) {
        return es::EsTensorHolder();
    }
    Tensor value(valueDesc, std::vector<uint8_t>(byteCount, 0U));
    if (node.SetAttr("value", value) != GRAPH_SUCCESS) {
        return es::EsTensorHolder();
    }
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildVariableNode(es::EsGraphBuilder& builder, const std::string& name,
                                            const std::vector<int64_t>& dims, Format format, DataType dtype)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Variable")
                     .Name(name.c_str())
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    SetOutputDesc(node, 0, dims, format, dtype);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

struct ConvNodeConfig {
    std::string type;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::string dataFormat;
};

static es::EsTensorHolder BuildConvNode(es::EsGraphBuilder& builder, const std::string& name,
                                        const es::EsTensorHolder& x, const es::EsTensorHolder& filter,
                                        const ConvNodeConfig& config)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType(config.type.c_str())
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"filter", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                   {"offset_w", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();

    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *filter.GetProducer(), filter.GetProducerOutIndex(), node, 1);

    auto strides = config.strides;
    auto pads = config.pads;
    node.SetAttr("strides", strides);
    node.SetAttr("pads", pads);
    AscendString fmtStr(config.dataFormat.c_str());
    node.SetAttr("data_format", fmtStr);

    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

struct ConvNodeInputs {
    const es::EsTensorHolder& x;
    const es::EsTensorHolder& filter;
    const es::EsTensorHolder& bias;
};

static es::EsTensorHolder BuildConvNodeWithBias(es::EsGraphBuilder& builder, const std::string& name,
                                                const ConvNodeInputs& inputs, const ConvNodeConfig& config)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType(config.type.c_str())
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"filter", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                   {"offset_w", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();

    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs.x.GetProducer(), inputs.x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs.filter.GetProducer(), inputs.filter.GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *inputs.bias.GetProducer(), inputs.bias.GetProducerOutIndex(), node, 2);

    auto strides = config.strides;
    auto pads = config.pads;
    node.SetAttr("strides", strides);
    node.SetAttr("pads", pads);
    AscendString fmtStr(config.dataFormat.c_str());
    node.SetAttr("data_format", fmtStr);

    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildAddNode(es::EsGraphBuilder& builder, const std::string& name,
                                       const es::EsTensorHolder& x1, const es::EsTensorHolder& x2)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Add")
                     .Name(name.c_str())
                     .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();

    es::AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), node, 1);

    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildBiasAddNode(es::EsGraphBuilder& builder, const std::string& name,
                                           const es::EsTensorHolder& x, const es::EsTensorHolder& bias)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("BiasAdd")
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"bias", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();

    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *bias.GetProducer(), bias.GetProducerOutIndex(), node, 1);

    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildReshapeNode(es::EsGraphBuilder& builder, const std::string& name,
                                           const es::EsTensorHolder& x, const es::EsTensorHolder& shape)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Reshape")
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();

    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *shape.GetProducer(), shape.GetProducerOutIndex(), node, 1);

    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildIdentityNode(es::EsGraphBuilder& builder, const std::string& name,
                                            const es::EsTensorHolder& input)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Identity")
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), node, 0) !=
        GRAPH_SUCCESS) {
        return es::EsTensorHolder();
    }
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

struct UnaryNodeConfig {
    std::string name;
    std::string type;
    std::vector<int64_t> outputDims;
    Format outputFormat;
    DataType outputDtype;
};

static es::EsTensorHolder BuildUnaryNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& input,
                                         const UnaryNodeConfig& config)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    if (graph == nullptr || input.GetProducer() == nullptr) {
        return es::EsTensorHolder();
    }
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType(config.type.c_str())
                     .Name(config.name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), node, 0) !=
        GRAPH_SUCCESS) {
        return es::EsTensorHolder();
    }
    SetOutputDesc(node, 0, config.outputDims, config.outputFormat, config.outputDtype);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

struct RConsumerOutputs {
    es::EsTensorHolder reshape;
    es::EsTensorHolder bias;
    es::EsTensorHolder shape;
};

struct RConsumerDims {
    std::vector<int64_t> output;
    std::vector<int64_t> bias;
    std::vector<int64_t> shape;
};

static RConsumerOutputs BuildRConsumers(es::EsGraphBuilder& builder, const es::EsTensorHolder& reshape,
                                        const es::EsTensorHolder& bias, const es::EsTensorHolder& shape,
                                        const RConsumerDims& dims)
{
    return {
        BuildUnaryNode(builder, reshape, {"reshape_other_consumer", "Identity", dims.output, FORMAT_ND, DT_FLOAT16}),
        BuildUnaryNode(builder, bias, {"bias_other_consumer", "Identity", dims.bias, FORMAT_ND, DT_FLOAT16}),
        BuildUnaryNode(builder, shape, {"shape_other_consumer", "Identity", dims.shape, FORMAT_ND, DT_INT64})};
}

static GNode BuildStreamSwitchNode(es::EsGraphBuilder& builder, const std::string& name,
                                   const es::EsTensorHolder& input)
{
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("StreamSwitch")
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), node, 0);
    return node;
}

static TensorDesc MakeP0TensorDesc(const std::vector<int64_t>& dims, Format format, DataType dtype)
{
    TensorDesc desc(Shape(dims), format, dtype);
    desc.SetOriginShape(Shape(dims));
    desc.SetOriginFormat(format);
    return desc;
}

static Tensor MakeP0ConstTensor(const TensorDesc& desc, size_t elementCount)
{
    std::vector<float> values(elementCount, 1.0F);
    size_t byteCount = 0U;
    if (!TryMultiplySize(values.size(), sizeof(float), byteCount)) {
        return Tensor();
    }
    std::vector<uint8_t> bytes(byteCount);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return Tensor(desc, bytes);
}

static GNode AddP0SourceNode(Graph& graph, const std::string& name, const std::string& type, const TensorDesc& desc,
                             const Tensor* constValue = nullptr)
{
    es::CompliantNodeBuilder::AnyTypeOperator sourceOp(name.c_str(), type.c_str());
    sourceOp.OutputRegister("y");
    sourceOp.UpdateOutputDesc(static_cast<uint32_t>(0), desc);
    if (constValue != nullptr) {
        Tensor value = *constValue;
        sourceOp.SetAttr("value", value);
    }
    return graph.AddNodeByOp(sourceOp);
}

struct P0ParentGraphCase {
    std::shared_ptr<Graph> parentGraph;
    std::shared_ptr<Graph> childGraph;
};

struct P0GraphDescs {
    TensorDesc input;
    TensorDesc filter;
    TensorDesc bias;
    TensorDesc shape;
    TensorDesc output;
};

struct P0ParentConstConfig {
    std::vector<int64_t> biasDims;
    std::string sourceType;
    bool hasValue;
};

static bool AddP0ParentConstCall(Graph& graph, const TensorDesc& biasDesc, const P0ParentConstConfig& config,
                                 GNode& parentNode)
{
    const size_t elementCount = config.biasDims.empty() ?
                                    0U :
                                    static_cast<size_t>(config.biasDims.back() > 0 ? config.biasDims.back() : 0);
    const Tensor parentConstTensor = MakeP0ConstTensor(biasDesc, elementCount);
    const Tensor* value = config.sourceType == "Const" && config.hasValue ? &parentConstTensor : nullptr;
    GNode parentSource = AddP0SourceNode(graph, "parent_bias_source", config.sourceType, biasDesc, value);

    es::CompliantNodeBuilder::AnyTypeOperator parentOp("parent_call", "PartitionedCall");
    parentOp.DynamicInputRegister("args", 1);
    parentOp.DynamicOutputRegister("output", 1);
    parentOp.SubgraphRegister("f", false);
    parentOp.SubgraphCountRegister("f", 1);
    parentOp.UpdateInputDesc(static_cast<uint32_t>(0), biasDesc);
    parentOp.UpdateOutputDesc(static_cast<uint32_t>(0), biasDesc);
    parentOp.SetSubgraphInstanceName(0, "p0_child_graph");
    parentNode = graph.AddNodeByOp(parentOp);
    return graph.AddDataEdge(parentSource, 0, parentNode, 0) == GRAPH_SUCCESS;
}

static bool AddP0Conv2DNode(Graph& graph, const P0GraphDescs& descs, GNode& convNode)
{
    GNode xNode = AddP0SourceNode(graph, "x", "Data", descs.input);
    GNode filterNode = AddP0SourceNode(graph, "filter", "Data", descs.filter);
    es::CompliantNodeBuilder::AnyTypeOperator convOp("conv", "Conv2D");
    convOp.InputRegister("x");
    convOp.InputRegister("filter");
    convOp.OptionalInputRegister("bias");
    convOp.OptionalInputRegister("offset_w");
    convOp.OutputRegister("y");
    convOp.UpdateInputDesc(static_cast<uint32_t>(0), descs.input);
    convOp.UpdateInputDesc(static_cast<uint32_t>(1), descs.filter);
    convOp.UpdateOutputDesc(static_cast<uint32_t>(0), descs.output);
    std::vector<int64_t> strides{1, 2, 2, 1};
    std::vector<int64_t> pads{1, 1, 1, 1};
    convOp.SetAttr("strides", strides);
    convOp.SetAttr("pads", pads);
    AscendString dataFormat("NHWC");
    convOp.SetAttr("data_format", dataFormat);
    convNode = graph.AddNodeByOp(convOp);
    return graph.AddDataEdge(xNode, 0, convNode, 0) == GRAPH_SUCCESS &&
           graph.AddDataEdge(filterNode, 0, convNode, 1) == GRAPH_SUCCESS;
}

static bool AddP0AddNode(Graph& graph, const P0GraphDescs& descs, GNode& first, GNode& second, GNode& addNode)
{
    es::CompliantNodeBuilder::AnyTypeOperator addOp("add", "Add");
    addOp.InputRegister("x1");
    addOp.InputRegister("x2");
    addOp.OutputRegister("y");
    addOp.UpdateInputDesc(static_cast<uint32_t>(0), descs.output);
    addOp.UpdateInputDesc(static_cast<uint32_t>(1), descs.bias);
    addOp.UpdateOutputDesc(static_cast<uint32_t>(0), descs.output);
    addNode = graph.AddNodeByOp(addOp);
    return graph.AddDataEdge(first, 0, addNode, 0) == GRAPH_SUCCESS &&
           graph.AddDataEdge(second, 0, addNode, 1) == GRAPH_SUCCESS;
}

static bool AttachP0ChildGraph(GNode& parentNode, Graph& childGraph, std::shared_ptr<Graph>& attachedChild)
{
    return parentNode.SetSubgraph("f", childGraph) == GRAPH_SUCCESS &&
           parentNode.GetSubgraph(0, attachedChild) == GRAPH_SUCCESS;
}

static Tensor MakeP0Int64ConstTensor(const TensorDesc& desc, size_t elementCount)
{
    std::vector<int64_t> values(elementCount, 1);
    size_t byteCount = 0U;
    if (!TryMultiplySize(values.size(), sizeof(int64_t), byteCount)) {
        return Tensor();
    }
    std::vector<uint8_t> bytes(byteCount);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return Tensor(desc, bytes);
}

static P0ParentGraphCase BuildP0ParentConstMappedConvAddGraph(const std::vector<int64_t>& biasDims,
                                                              const std::string& parentSourceType = "Const",
                                                              bool parentHasValue = true)
{
    const P0GraphDescs descs{MakeP0TensorDesc({1, 56, 56, 64}, FORMAT_NHWC, DT_FLOAT),
                             MakeP0TensorDesc({64, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT),
                             MakeP0TensorDesc(biasDims, FORMAT_NHWC, DT_FLOAT), TensorDesc(),
                             MakeP0TensorDesc({1, 28, 28, 64}, FORMAT_NHWC, DT_FLOAT)};
    auto parentGraph = std::make_shared<Graph>("p0_parent_const_graph");
    GNode parentNode;
    if (!AddP0ParentConstCall(*parentGraph, descs.bias, {biasDims, parentSourceType, parentHasValue}, parentNode)) {
        return {nullptr, nullptr};
    }
    Graph childGraph("p0_child_graph");
    GNode biasNode = AddP0SourceNode(childGraph, "bias_data", "Data", descs.bias);
    int64_t parentIndex = 0;
    biasNode.SetAttr("_parent_node_index", parentIndex);
    GNode convNode;
    GNode addNode;
    if (!AddP0Conv2DNode(childGraph, descs, convNode) ||
        !AddP0AddNode(childGraph, descs, convNode, biasNode, addNode) ||
        childGraph.SetOutputs({{addNode, 0}}) != GRAPH_SUCCESS) {
        return {nullptr, nullptr};
    }
    std::shared_ptr<Graph> attachedChild;
    return AttachP0ChildGraph(parentNode, childGraph, attachedChild) ? P0ParentGraphCase{parentGraph, attachedChild} :
                                                                       P0ParentGraphCase{nullptr, nullptr};
}

struct P0ReshapeMapping {
    bool biasMapped;
    bool shapeMapped;
};

struct P0ReshapeSources {
    GNode bias;
    GNode shape;
};

static bool AddP0ReshapeParentCall(Graph& graph, const P0GraphDescs& descs, const P0ReshapeMapping& mapping,
                                   GNode& parentNode)
{
    const Tensor biasTensor = MakeP0ConstTensor(descs.bias, 4U);
    const Tensor shapeTensor = MakeP0Int64ConstTensor(descs.shape, 5U);
    GNode parentBias;
    GNode parentShape;
    if (mapping.biasMapped) {
        parentBias = AddP0SourceNode(graph, "parent_bias_const", "Const", descs.bias, &biasTensor);
    }
    if (mapping.shapeMapped) {
        parentShape = AddP0SourceNode(graph, "parent_shape_const", "Const", descs.shape, &shapeTensor);
    }

    es::CompliantNodeBuilder::AnyTypeOperator parentOp("parent_call", "PartitionedCall");
    const int32_t parentInputCount = static_cast<int32_t>(mapping.biasMapped) +
                                     static_cast<int32_t>(mapping.shapeMapped);
    parentOp.DynamicInputRegister("args", parentInputCount);
    parentOp.DynamicOutputRegister("output", 1);
    parentOp.SubgraphRegister("f", false);
    parentOp.SubgraphCountRegister("f", 1);
    int32_t parentInputIndex = 0;
    if (mapping.biasMapped) {
        parentOp.UpdateInputDesc(static_cast<uint32_t>(parentInputIndex++), descs.bias);
    }
    if (mapping.shapeMapped) {
        parentOp.UpdateInputDesc(static_cast<uint32_t>(parentInputIndex++), descs.shape);
    }
    parentOp.UpdateOutputDesc(static_cast<uint32_t>(0), descs.bias);
    parentOp.SetSubgraphInstanceName(0, "p0_r_child_graph");
    parentNode = graph.AddNodeByOp(parentOp);
    parentInputIndex = 0;
    if (mapping.biasMapped && graph.AddDataEdge(parentBias, 0, parentNode, parentInputIndex++) != GRAPH_SUCCESS) {
        return false;
    }
    if (mapping.shapeMapped && graph.AddDataEdge(parentShape, 0, parentNode, parentInputIndex) != GRAPH_SUCCESS) {
        return false;
    }
    return true;
}

static P0ReshapeSources AddP0ReshapeChildSources(Graph& graph, const P0GraphDescs& descs,
                                                 const P0ReshapeMapping& mapping)
{
    const Tensor biasTensor = MakeP0ConstTensor(descs.bias, 4U);
    const Tensor shapeTensor = MakeP0Int64ConstTensor(descs.shape, 5U);
    P0ReshapeSources sources;
    if (mapping.biasMapped) {
        sources.bias = AddP0SourceNode(graph, "bias_data", "Data", descs.bias);
        int64_t biasParentIndex = 0;
        sources.bias.SetAttr("_parent_node_index", biasParentIndex);
    } else {
        sources.bias = AddP0SourceNode(graph, "bias_const", "Const", descs.bias, &biasTensor);
    }
    if (mapping.shapeMapped) {
        sources.shape = AddP0SourceNode(graph, "shape_data", "Data", descs.shape);
        int64_t shapeParentIndex = mapping.biasMapped ? 1 : 0;
        sources.shape.SetAttr("_parent_node_index", shapeParentIndex);
    } else {
        sources.shape = AddP0SourceNode(graph, "shape_const", "Const", descs.shape, &shapeTensor);
    }
    return sources;
}

static bool AddP0Conv3DNode(Graph& graph, const P0GraphDescs& descs, GNode& convNode)
{
    GNode xNode = AddP0SourceNode(graph, "x", "Data", descs.input);
    GNode filterNode = AddP0SourceNode(graph, "filter", "Data", descs.filter);
    es::CompliantNodeBuilder::AnyTypeOperator convOp("conv3d", "Conv3D");
    convOp.InputRegister("x");
    convOp.InputRegister("filter");
    convOp.OptionalInputRegister("bias");
    convOp.OutputRegister("y");
    convOp.UpdateInputDesc(static_cast<uint32_t>(0), descs.input);
    convOp.UpdateInputDesc(static_cast<uint32_t>(1), descs.filter);
    convOp.UpdateOutputDesc(static_cast<uint32_t>(0), descs.output);
    convOp.SetAttr("strides", std::vector<int64_t>{1, 1, 1, 1, 1});
    convOp.SetAttr("pads", std::vector<int64_t>{0, 0, 0, 0, 0, 0});
    AscendString dataFormat("NCDHW");
    convOp.SetAttr("data_format", dataFormat);
    convNode = graph.AddNodeByOp(convOp);
    return graph.AddDataEdge(xNode, 0, convNode, 0) == GRAPH_SUCCESS &&
           graph.AddDataEdge(filterNode, 0, convNode, 1) == GRAPH_SUCCESS;
}

static bool AddP0ReshapeNode(Graph& graph, const P0GraphDescs& descs, P0ReshapeSources& sources, GNode& reshapeNode)
{
    es::CompliantNodeBuilder::AnyTypeOperator reshapeOp("reshape", "Reshape");
    reshapeOp.InputRegister("x");
    reshapeOp.InputRegister("shape");
    reshapeOp.OutputRegister("y");
    reshapeOp.UpdateInputDesc(static_cast<uint32_t>(0), descs.bias);
    reshapeOp.UpdateInputDesc(static_cast<uint32_t>(1), descs.shape);
    reshapeOp.UpdateOutputDesc(static_cast<uint32_t>(0), descs.bias);
    reshapeNode = graph.AddNodeByOp(reshapeOp);
    return graph.AddDataEdge(sources.bias, 0, reshapeNode, 0) == GRAPH_SUCCESS &&
           graph.AddDataEdge(sources.shape, 0, reshapeNode, 1) == GRAPH_SUCCESS;
}

static P0ParentGraphCase BuildP0ParentConstMappedConv3DReshapeGraph(bool biasMapped = true, bool shapeMapped = true)
{
    const P0GraphDescs descs{MakeP0TensorDesc({1, 4, 8, 8, 8}, FORMAT_ND, DT_FLOAT),
                             MakeP0TensorDesc({4, 4, 1, 1, 1}, FORMAT_NCDHW, DT_FLOAT),
                             MakeP0TensorDesc({4}, FORMAT_ND, DT_FLOAT), MakeP0TensorDesc({5}, FORMAT_ND, DT_INT64),
                             MakeP0TensorDesc({1, 4, 8, 8, 8}, FORMAT_ND, DT_FLOAT)};
    const P0ReshapeMapping mapping{biasMapped, shapeMapped};
    auto parentGraph = std::make_shared<Graph>("p0_parent_r_shape_graph");
    GNode parentNode;
    if (!AddP0ReshapeParentCall(*parentGraph, descs, mapping, parentNode)) {
        return {nullptr, nullptr};
    }
    Graph childGraph("p0_r_child_graph");
    P0ReshapeSources sources = AddP0ReshapeChildSources(childGraph, descs, mapping);
    GNode convNode;
    GNode reshapeNode;
    GNode addNode;
    if (!AddP0Conv3DNode(childGraph, descs, convNode) || !AddP0ReshapeNode(childGraph, descs, sources, reshapeNode) ||
        !AddP0AddNode(childGraph, descs, convNode, reshapeNode, addNode) ||
        childGraph.SetOutputs({{addNode, 0}}) != GRAPH_SUCCESS) {
        return {nullptr, nullptr};
    }
    std::shared_ptr<Graph> attachedChild;
    return AttachP0ChildGraph(parentNode, childGraph, attachedChild) ? P0ParentGraphCase{parentGraph, attachedChild} :
                                                                       P0ParentGraphCase{nullptr, nullptr};
}

static std::shared_ptr<Graph> BuildP0DirectDataConvAddGraph(const std::string& convType, const std::string& addType,
                                                            const std::vector<int64_t>& biasDims)
{
    const bool isConv3D = convType == "Conv3D";
    const std::vector<int64_t> dimsX = isConv3D ? std::vector<int64_t>{1, 4, 8, 8, 8} :
                                                  std::vector<int64_t>{1, 8, 8, 64};
    const std::vector<int64_t> dimsFilter = isConv3D ? std::vector<int64_t>{4, 4, 1, 1, 1} :
                                                       std::vector<int64_t>{64, 1, 1, 64};
    const std::vector<int64_t> dimsOut = isConv3D ? std::vector<int64_t>{1, 4, 8, 8, 8} :
                                                    std::vector<int64_t>{1, 4, 4, 64};
    const Format dataFormat = isConv3D ? FORMAT_NCDHW : FORMAT_NHWC;
    const std::string dataFormatName = isConv3D ? "NCDHW" : "NHWC";
    auto graphBuilder = es::EsGraphBuilder("p0_direct_data_conv_add");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, dataFormat, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, dataFormat, DT_FLOAT);
    auto bias = graphBuilder.CreateInput(1, "bias_data", DT_FLOAT, FORMAT_ND, biasDims);
    auto convOutput = BuildConvNode(
        graphBuilder, "conv", x, filter,
        {convType, isConv3D ? std::vector<int64_t>{1, 1, 1, 1, 1} : std::vector<int64_t>{1, 2, 2, 1},
         isConv3D ? std::vector<int64_t>{0, 0, 0, 0, 0, 0} : std::vector<int64_t>{1, 1, 1, 1}, dataFormatName});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, dataFormat, DT_FLOAT);
    auto addOutput = (addType == "BiasAdd") ? BuildBiasAddNode(graphBuilder, "add", convOutput, bias) :
                                              BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, dataFormat, DT_FLOAT);
    return graphBuilder.BuildAndReset({addOutput});
}

static std::shared_ptr<Graph> BuildP0DirectVariableConvBiasAddGraph(const std::string& convType,
                                                                    const std::vector<int64_t>& biasDims)
{
    const bool isConv3D = convType == "Conv3D";
    const std::vector<int64_t> dimsX = isConv3D ? std::vector<int64_t>{1, 4, 8, 8, 8} :
                                                  std::vector<int64_t>{1, 8, 8, 64};
    const std::vector<int64_t> dimsFilter = isConv3D ? std::vector<int64_t>{4, 4, 1, 1, 1} :
                                                       std::vector<int64_t>{64, 1, 1, 64};
    const std::vector<int64_t> dimsOut = isConv3D ? std::vector<int64_t>{1, 4, 8, 8, 8} :
                                                    std::vector<int64_t>{1, 4, 4, 64};
    const Format dataFormat = isConv3D ? FORMAT_NCDHW : FORMAT_NHWC;
    const std::string dataFormatName = isConv3D ? "NCDHW" : "NHWC";
    auto graphBuilder = es::EsGraphBuilder("p0_direct_variable_conv_biasadd");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, dataFormat, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, dataFormat, DT_FLOAT);
    auto bias = BuildVariableNode(graphBuilder, "bias_variable", biasDims, FORMAT_ND, DT_FLOAT);
    auto convOutput = BuildConvNode(
        graphBuilder, "conv", x, filter,
        {convType, isConv3D ? std::vector<int64_t>{1, 1, 1, 1, 1} : std::vector<int64_t>{1, 2, 2, 1},
         isConv3D ? std::vector<int64_t>{0, 0, 0, 0, 0, 0} : std::vector<int64_t>{1, 1, 1, 1}, dataFormatName});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, dataFormat, DT_FLOAT);
    auto addOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, dataFormat, DT_FLOAT);
    return graphBuilder.BuildAndReset({addOutput});
}

static std::shared_ptr<Graph> BuildP0DirectConstRGraph(const std::vector<int64_t>& biasDims,
                                                       DataType biasDtype = DT_FLOAT, bool connectShape = true)
{
    const std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    const std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    const std::vector<int64_t> dimsShape{5};
    const std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};
    auto graphBuilder = es::EsGraphBuilder("p0_direct_const_r_graph");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithForcedPayload(graphBuilder, "bias_const", {biasDims, FORMAT_ND, biasDtype, 4U});
    auto shape = graphBuilder.CreateInput(1, "shape_data", DT_INT64, FORMAT_ND, dimsShape);
    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    SetOutputDesc(*reshapeOutput.GetProducer(), 0, dimsOut, FORMAT_ND, biasDtype);
    if (!connectShape) {
        auto rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
        if (rawGraph->RemoveEdge(*shape.GetProducer(), shape.GetProducerOutIndex(), *reshapeOutput.GetProducer(), 1) !=
            GRAPH_SUCCESS) {
            return nullptr;
        }
    }
    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    return graphBuilder.BuildAndReset({addOutput});
}

struct DirectConstConvAddConfig {
    std::string convType;
    std::string addType;
    std::vector<int64_t> biasDims;
    std::vector<int64_t> outputDims;
    Format outputFormat;
    std::string dataFormatName;
};

static std::shared_ptr<Graph> BuildP0DirectConstConvAddGraph(const DirectConstConvAddConfig& config)
{
    const bool isConv3D = config.convType == "Conv3D";
    const std::vector<int64_t> dimsFilter = isConv3D ? std::vector<int64_t>{64, 1, 1, 1, 1} :
                                                       std::vector<int64_t>{64, 1, 1, 64};
    auto graphBuilder = es::EsGraphBuilder("p0_direct_const_conv_add_graph");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, config.outputFormat, config.outputDims);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, config.outputFormat, DT_FLOAT);
    auto bias = BuildConstNodeWithForcedPayload(graphBuilder, "bias_const",
                                                {config.biasDims, FORMAT_ND, DT_FLOAT, 64U});
    auto convOutput = BuildConvNode(
        graphBuilder, "conv", x, filter,
        {config.convType, isConv3D ? std::vector<int64_t>{1, 1, 1, 1, 1} : std::vector<int64_t>{1, 1, 1, 1},
         isConv3D ? std::vector<int64_t>{0, 0, 0, 0, 0, 0} : std::vector<int64_t>{0, 0, 0, 0}, config.dataFormatName});
    SetOutputDesc(*convOutput.GetProducer(), 0, config.outputDims, config.outputFormat, DT_FLOAT);
    auto addOutput = config.addType == "BiasAdd" ? BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias) :
                                                   BuildAddNode(graphBuilder, "add", convOutput, bias);
    SetOutputDesc(*addOutput.GetProducer(), 0, config.outputDims, config.outputFormat, DT_FLOAT);
    return graphBuilder.BuildAndReset({addOutput});
}

struct DescSnapshot {
    bool readable = false;
    std::vector<int64_t> shape;
    std::vector<int64_t> originShape;
    int64_t dataType = 0;
    int64_t format = 0;
    int64_t originFormat = 0;
};

struct EdgeSnapshot {
    bool connected = false;
    std::string sourceName;
    std::string sourceType;
    int32_t sourceOutputIndex = -1;
};

struct NodeSnapshot {
    std::string name;
    std::string type;
    std::vector<EdgeSnapshot> inputs;
    std::vector<DescSnapshot> inputDescs;
    std::vector<DescSnapshot> outputDescs;
    std::vector<std::string> outputConsumers;
    std::vector<std::string> controlInputs;
    std::vector<std::string> controlOutputs;
    std::vector<std::pair<std::string, std::string>> attrs;
};

struct GraphSnapshot {
    bool valid = false;
    std::vector<NodeSnapshot> nodes;
    std::vector<std::string> leafOutputs;
};

static std::string GetNodeNameForSnapshot(const GNodePtr& node)
{
    if (node == nullptr) {
        return "<null>";
    }
    AscendString name;
    return node->GetName(name) == GRAPH_SUCCESS ? name.GetString() : "<unnamed>";
}

static std::string GetNodeTypeForSnapshot(const GNodePtr& node)
{
    if (node == nullptr) {
        return "<null>";
    }
    AscendString type;
    return node->GetType(type) == GRAPH_SUCCESS ? type.GetString() : "<unknown>";
}

static std::string GetNodeNameForSnapshot(const GNode& node)
{
    AscendString name;
    return node.GetName(name) == GRAPH_SUCCESS ? name.GetString() : "<unnamed>";
}

static std::string GetNodeTypeForSnapshot(const GNode& node)
{
    AscendString type;
    return node.GetType(type) == GRAPH_SUCCESS ? type.GetString() : "<unknown>";
}

static DescSnapshot CaptureDescSnapshot(const GNode& node, bool isInput, int32_t index)
{
    DescSnapshot result;
    TensorDesc desc;
    const auto status = isInput ? node.GetInputDesc(index, desc) : node.GetOutputDesc(index, desc);
    if (status != GRAPH_SUCCESS) {
        return result;
    }
    result.readable = true;
    result.shape = desc.GetShape().GetDims();
    result.originShape = desc.GetOriginShape().GetDims();
    result.dataType = static_cast<int64_t>(desc.GetDataType());
    result.format = static_cast<int64_t>(desc.GetFormat());
    result.originFormat = static_cast<int64_t>(desc.GetOriginFormat());
    return result;
}

static bool EqualDescSnapshot(const DescSnapshot& lhs, const DescSnapshot& rhs)
{
    return lhs.readable == rhs.readable && lhs.shape == rhs.shape && lhs.originShape == rhs.originShape &&
           lhs.dataType == rhs.dataType && lhs.format == rhs.format && lhs.originFormat == rhs.originFormat;
}

static std::string DescSnapshotText(const TensorDesc& desc)
{
    std::string text = "shape=";
    for (const auto dim : desc.GetShape().GetDims()) {
        text += std::to_string(dim) + ",";
    }
    text += ";origin=";
    for (const auto dim : desc.GetOriginShape().GetDims()) {
        text += std::to_string(dim) + ",";
    }
    text += ";dtype=" + std::to_string(static_cast<int64_t>(desc.GetDataType()));
    text += ";format=" + std::to_string(static_cast<int64_t>(desc.GetFormat()));
    text += ";originFormat=" + std::to_string(static_cast<int64_t>(desc.GetOriginFormat()));
    return text;
}

static std::string TensorSnapshotText(const Tensor& tensor)
{
    std::string text = DescSnapshotText(tensor.GetTensorDesc());
    text += ";size=" + std::to_string(tensor.GetSize()) + ";data=";
    const auto* data = static_cast<const uint8_t*>(tensor.GetData());
    for (size_t index = 0; data != nullptr && index < tensor.GetSize(); ++index) {
        text += std::to_string(static_cast<uint32_t>(data[index])) + ",";
    }
    return text;
}

static bool CaptureAttrSnapshot(const GNode& node, const std::string& name, std::string& value)
{
    const AscendString attrName(name.c_str());

    std::vector<int64_t> intValues;
    if (node.GetAttr(attrName, intValues) == GRAPH_SUCCESS) {
        value.clear();
        for (const auto item : intValues) {
            value += std::to_string(item) + ",";
        }
        return true;
    }
    int64_t intValue = 0;
    if (node.GetAttr(attrName, intValue) == GRAPH_SUCCESS) {
        value = std::to_string(intValue);
        return true;
    }
    bool boolValue = false;
    if (node.GetAttr(attrName, boolValue) == GRAPH_SUCCESS) {
        value = boolValue ? "true" : "false";
        return true;
    }
    AscendString stringValue;
    if (node.GetAttr(attrName, stringValue) == GRAPH_SUCCESS) {
        value = stringValue.GetString();
        return true;
    }
    Tensor tensorValue;
    if (node.GetAttr(attrName, tensorValue) == GRAPH_SUCCESS) {
        value = TensorSnapshotText(tensorValue);
        return true;
    }
    return false;
}

static void CaptureNodeInputs(const GNode& node, NodeSnapshot& result)
{
    for (size_t index = 0; index < node.GetInputsSize(); ++index) {
        const auto source = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(index));
        EdgeSnapshot edge;
        edge.connected = source.first != nullptr;
        edge.sourceName = GetNodeNameForSnapshot(source.first);
        edge.sourceType = GetNodeTypeForSnapshot(source.first);
        edge.sourceOutputIndex = source.second;
        result.inputs.emplace_back(edge);
        result.inputDescs.emplace_back(CaptureDescSnapshot(node, true, static_cast<int32_t>(index)));
    }
}

static void CaptureNodeOutputs(const GNode& node, NodeSnapshot& result)
{
    for (size_t index = 0; index < node.GetOutputsSize(); ++index) {
        const int32_t outputIndex = static_cast<int32_t>(index);
        result.outputDescs.emplace_back(CaptureDescSnapshot(node, false, outputIndex));
        const auto consumers = node.GetOutDataNodesAndPortIndexs(outputIndex);
        for (const auto& consumer : consumers) {
            result.outputConsumers.emplace_back(GetNodeNameForSnapshot(consumer.first) + ":" +
                                                GetNodeTypeForSnapshot(consumer.first) + ":" +
                                                std::to_string(consumer.second));
        }
    }
    std::sort(result.outputConsumers.begin(), result.outputConsumers.end());
}

static void CaptureNodeControlEdges(const GNode& node, NodeSnapshot& result)
{
    for (const auto& controlInput : node.GetInControlNodes()) {
        result.controlInputs.emplace_back(GetNodeNameForSnapshot(controlInput) + ":" +
                                          GetNodeTypeForSnapshot(controlInput));
    }
    for (const auto& controlOutput : node.GetOutControlNodes()) {
        result.controlOutputs.emplace_back(GetNodeNameForSnapshot(controlOutput) + ":" +
                                           GetNodeTypeForSnapshot(controlOutput));
    }
    std::sort(result.controlInputs.begin(), result.controlInputs.end());
    std::sort(result.controlOutputs.begin(), result.controlOutputs.end());
}

static void CaptureNodeAttrs(const GNode& node, NodeSnapshot& result)
{
    const std::vector<std::string> attrNames = {"strides",     "pads",     "dilations",          "groups",
                                                "data_format", "offset_x", "_op_impl_mode_enum", "padding",
                                                "auto_pad",    "value"};
    for (const auto& attrName : attrNames) {
        std::string value;
        if (CaptureAttrSnapshot(node, attrName, value)) {
            result.attrs.emplace_back(attrName, value);
        }
    }
}

static NodeSnapshot CaptureNodeSnapshot(const GNode& node)
{
    NodeSnapshot result;
    result.name = GetNodeNameForSnapshot(node);
    result.type = GetNodeTypeForSnapshot(node);
    CaptureNodeInputs(node, result);
    CaptureNodeOutputs(node, result);
    CaptureNodeControlEdges(node, result);
    CaptureNodeAttrs(node, result);
    return result;
}

static bool EqualNodeSnapshot(const NodeSnapshot& lhs, const NodeSnapshot& rhs)
{
    if (lhs.name != rhs.name || lhs.type != rhs.type || lhs.inputs.size() != rhs.inputs.size() ||
        lhs.inputDescs.size() != rhs.inputDescs.size() || lhs.outputDescs.size() != rhs.outputDescs.size() ||
        lhs.outputConsumers != rhs.outputConsumers || lhs.controlInputs != rhs.controlInputs ||
        lhs.controlOutputs != rhs.controlOutputs || lhs.attrs != rhs.attrs) {
        return false;
    }
    for (size_t index = 0; index < lhs.inputs.size(); ++index) {
        const auto& left = lhs.inputs[index];
        const auto& right = rhs.inputs[index];
        if (left.connected != right.connected || left.sourceName != right.sourceName ||
            left.sourceType != right.sourceType || left.sourceOutputIndex != right.sourceOutputIndex ||
            !EqualDescSnapshot(lhs.inputDescs[index], rhs.inputDescs[index])) {
            return false;
        }
    }
    for (size_t index = 0; index < lhs.outputDescs.size(); ++index) {
        if (!EqualDescSnapshot(lhs.outputDescs[index], rhs.outputDescs[index])) {
            return false;
        }
    }
    return true;
}

static GraphSnapshot CaptureGraphSnapshot(const std::shared_ptr<Graph>& graph)
{
    GraphSnapshot result;
    if (graph == nullptr) {
        return result;
    }
    result.valid = true;

    for (const auto& node : graph->GetAllNodes()) {
        result.nodes.emplace_back(CaptureNodeSnapshot(node));
        for (size_t outputIndex = 0; outputIndex < node.GetOutputsSize(); ++outputIndex) {
            if (node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(outputIndex)).empty()) {
                result.leafOutputs.emplace_back(GetNodeNameForSnapshot(node) + ":" + std::to_string(outputIndex));
            }
        }
    }
    std::sort(result.nodes.begin(), result.nodes.end(),
              [](const NodeSnapshot& lhs, const NodeSnapshot& rhs) { return lhs.name < rhs.name; });
    std::sort(result.leafOutputs.begin(), result.leafOutputs.end());
    return result;
}

static bool EqualGraphSnapshot(const GraphSnapshot& lhs, const GraphSnapshot& rhs)
{
    if (lhs.valid != rhs.valid || lhs.nodes.size() != rhs.nodes.size() || lhs.leafOutputs != rhs.leafOutputs) {
        return false;
    }
    for (size_t index = 0; index < lhs.nodes.size(); ++index) {
        if (!EqualNodeSnapshot(lhs.nodes[index], rhs.nodes[index])) {
            return false;
        }
    }
    return true;
}

static void ExpectGraphUnchanged(const GraphSnapshot& before, const std::shared_ptr<Graph>& graph)
{
    const auto after = CaptureGraphSnapshot(graph);
    EXPECT_TRUE(EqualGraphSnapshot(before, after));
}

static bool IsSupportedConvTypeForTest(const std::string& type)
{
    return type == "Conv2D" || type == "DepthwiseConv2D" || type == "Conv3D";
}

static void ExpectFusionTargetGraph(const std::shared_ptr<Graph>& graph)
{
    ASSERT_NE(graph, nullptr);
    bool foundTarget = false;

    for (const auto& node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) != GRAPH_SUCCESS || !IsSupportedConvTypeForTest(type.GetString())) {
            continue;
        }
        const auto xSource = node.GetInDataNodesAndPortIndexs(0);
        const auto filterSource = node.GetInDataNodesAndPortIndexs(1);
        const auto biasSource = node.GetInDataNodesAndPortIndexs(2);
        if (xSource.first == nullptr || filterSource.first == nullptr || biasSource.first == nullptr) {
            continue;
        }
        foundTarget = true;
        EXPECT_NE(xSource.first, nullptr);
        EXPECT_NE(filterSource.first, nullptr);
        EXPECT_NE(biasSource.first, nullptr);
        TensorDesc biasDesc;
        EXPECT_EQ(node.GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
        EXPECT_FALSE(biasDesc.GetShape().GetDims().empty());
    }
    EXPECT_TRUE(foundTarget);
}

static void ExpectTensorDescEqual(const TensorDesc& actual, const TensorDesc& expected)
{
    EXPECT_EQ(actual.GetShape().GetDims(), expected.GetShape().GetDims());
    EXPECT_EQ(actual.GetOriginShape().GetDims(), expected.GetOriginShape().GetDims());
    EXPECT_EQ(actual.GetDataType(), expected.GetDataType());
    EXPECT_EQ(actual.GetFormat(), expected.GetFormat());
    EXPECT_EQ(actual.GetOriginFormat(), expected.GetOriginFormat());
}

static bool FindFirstNodeByType(const std::shared_ptr<Graph>& graph, const std::string& nodeType, GNode& result);

static void ExpectFusedConvDescs(const std::shared_ptr<Graph>& graph, const TensorDesc& xDesc,
                                 const TensorDesc& filterDesc, const TensorDesc& biasDesc, const TensorDesc& outputDesc)
{
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    TensorDesc actual;
    ASSERT_EQ(fusedConv.GetInputDesc(0, actual), GRAPH_SUCCESS);
    ExpectTensorDescEqual(actual, xDesc);
    ASSERT_EQ(fusedConv.GetInputDesc(1, actual), GRAPH_SUCCESS);
    ExpectTensorDescEqual(actual, filterDesc);
    ASSERT_EQ(fusedConv.GetInputDesc(2, actual), GRAPH_SUCCESS);
    ExpectTensorDescEqual(actual, biasDesc);
    ASSERT_EQ(fusedConv.GetOutputDesc(0, actual), GRAPH_SUCCESS);
    ExpectTensorDescEqual(actual, outputDesc);
}

static int CountNodes(const std::shared_ptr<Graph>& graph, const std::string& nodeType)
{
    if (graph == nullptr) {
        return 0;
    }
    std::shared_ptr<Graph> graphPtr = graph;
    return test_conv_fusion_framework::GraphChecker::CountNodes(graphPtr, nodeType);
}

static bool FindFirstNodeByType(const std::shared_ptr<Graph>& graph, const std::string& nodeType, GNode& result)
{
    if (graph == nullptr) {
        return false;
    }
    std::shared_ptr<Graph> graphPtr = graph;
    return test_conv_fusion_framework::GraphChecker::FindFirstNodeByOpType(graphPtr, nodeType, result);
}

static bool HasControlPathWithinTwoHops(const GNode& source, const GNode& target)
{
    const std::string targetName = GetNodeNameForSnapshot(target);
    for (const auto& firstHop : source.GetOutControlNodes()) {
        if (GetNodeNameForSnapshot(firstHop) == targetName) {
            return true;
        }
        for (const auto& secondHop : firstHop->GetOutControlNodes()) {
            if (GetNodeNameForSnapshot(secondHop) == targetName) {
                return true;
            }
        }
    }
    return false;
}

static void ExpectInputSourceType(const GNode& node, const char* expectedType)
{
    const auto input = node.GetInDataNodesAndPortIndexs(0);
    ASSERT_NE(input.first, nullptr);
    AscendString sourceType;
    ASSERT_EQ(input.first->GetType(sourceType), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceType.GetString(), expectedType);
}

static void ExpectRConsumerTypes(const RConsumerOutputs& consumers)
{
    ASSERT_NE(consumers.reshape.GetProducer(), nullptr);
    ASSERT_NE(consumers.bias.GetProducer(), nullptr);
    ASSERT_NE(consumers.shape.GetProducer(), nullptr);
    ExpectInputSourceType(*consumers.reshape.GetProducer(), "Reshape");
    ExpectInputSourceType(*consumers.bias.GetProducer(), "Const");
    ExpectInputSourceType(*consumers.shape.GetProducer(), "Data");
}

struct ConvAddState {
    AscendString convXName;
    AscendString convFilterName;
    AscendString addConvName;
    AscendString addBiasName;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    TensorDesc addOutputDesc;
};

static void CaptureConvAddState(const GNode& conv, const GNode& add, ConvAddState& state)
{
    const auto convX = conv.GetInDataNodesAndPortIndexs(0);
    const auto convFilter = conv.GetInDataNodesAndPortIndexs(1);
    const auto addConv = add.GetInDataNodesAndPortIndexs(0);
    const auto addBias = add.GetInDataNodesAndPortIndexs(1);
    ASSERT_NE(convX.first, nullptr);
    ASSERT_NE(convFilter.first, nullptr);
    ASSERT_NE(addConv.first, nullptr);
    ASSERT_NE(addBias.first, nullptr);
    ASSERT_EQ(convX.first->GetName(state.convXName), GRAPH_SUCCESS);
    ASSERT_EQ(convFilter.first->GetName(state.convFilterName), GRAPH_SUCCESS);
    ASSERT_EQ(addConv.first->GetName(state.addConvName), GRAPH_SUCCESS);
    ASSERT_EQ(addBias.first->GetName(state.addBiasName), GRAPH_SUCCESS);
    ASSERT_EQ(conv.GetAttr("strides", state.strides), GRAPH_SUCCESS);
    ASSERT_EQ(conv.GetAttr("pads", state.pads), GRAPH_SUCCESS);
    ASSERT_EQ(add.GetOutputDesc(0, state.addOutputDesc), GRAPH_SUCCESS);
}

static void ExpectConvAddState(const std::shared_ptr<Graph>& graph, const ConvAddState& expected)
{
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    GNode conv;
    GNode add;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", conv));
    ASSERT_TRUE(FindFirstNodeByType(graph, "Add", add));
    ConvAddState actual;
    CaptureConvAddState(conv, add, actual);
    EXPECT_EQ(actual.convXName, expected.convXName);
    EXPECT_EQ(actual.convFilterName, expected.convFilterName);
    EXPECT_EQ(actual.addConvName, expected.addConvName);
    EXPECT_EQ(actual.addBiasName, expected.addBiasName);
    EXPECT_EQ(actual.strides, expected.strides);
    EXPECT_EQ(actual.pads, expected.pads);
    EXPECT_EQ(actual.addOutputDesc.GetShape().GetDims(), expected.addOutputDesc.GetShape().GetDims());
    EXPECT_EQ(actual.addOutputDesc.GetDataType(), expected.addOutputDesc.GetDataType());
    EXPECT_EQ(actual.addOutputDesc.GetFormat(), expected.addOutputDesc.GetFormat());
}

struct GlobalPlatformState {
    std::map<std::string, PlatformInfo> platformInfoMap;
    OptionalInfo optionalInfo;
};

GlobalPlatformState& GetSavedPlatformState()
{
    static GlobalPlatformState savedState = [] {
        GlobalPlatformState state;
        state.platformInfoMap = PlatformInfoManager::Instance().platform_info_map_;
        state.optionalInfo = PlatformInfoManager::Instance().opti_compilation_info_;
        return state;
    }();
    return savedState;
}

void SaveGlobalPlatformState() { (void)GetSavedPlatformState(); }

void RestoreGlobalPlatformState()
{
    const GlobalPlatformState& savedState = GetSavedPlatformState();
    PlatformInfoManager::Instance().platform_info_map_ = savedState.platformInfoMap;
    OptionalInfo optionalInfo = savedState.optionalInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

} // namespace

class AABiasaddConvFusionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AABiasaddConvFusionTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AABiasaddConvFusionTest TearDown" << std::endl; }

    void SetUp() override { SaveGlobalPlatformState(); }

    void SetUpFixpipePlatform()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
        optiInfo.soc_version = "Ascend910B";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiInfo);
    }

    void SetUpNonFixpipePlatform()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiInfo;
        platformInfo.soc_info.ai_core_cnt = 32;
        platformInfo.str_info.short_soc_version = "Ascend910A";
        optiInfo.soc_version = "Ascend910A";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiInfo);
    }

    void ClearPlatform()
    {
        PlatformInfoManager::Instance().platform_info_map_.clear();
        OptionalInfo emptyOptionalInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(emptyOptionalInfo);
    }

    void TearDown() override { RestoreGlobalPlatformState(); }

    void RunPass(std::shared_ptr<Graph>& graph, Status expectedStatus)
    {
        if (graph == nullptr) {
            ADD_FAILURE() << "Graph construction failed";
            return;
        }
        const GraphSnapshot before = expectedStatus == GRAPH_NOT_CHANGED ? CaptureGraphSnapshot(graph) :
                                                                           GraphSnapshot();

        CustomPassContext passContext;
        passContext.SetPassName(FUSION_NAME.c_str());
        AABiasaddConvFusion pass;
        Status status = pass.Run(graph, passContext);
        if (expectedStatus == GRAPH_NOT_CHANGED) {
            EXPECT_TRUE(status == GRAPH_NOT_CHANGED || status == CONV_NOT_CHANGED) << "status=" << status;
        } else {
            EXPECT_EQ(status, expectedStatus);
        }
        if (expectedStatus == GRAPH_NOT_CHANGED) {
            ExpectGraphUnchanged(before, graph);
        } else if (expectedStatus == SUCCESS) {
            ExpectFusionTargetGraph(graph);
        }
    }
};

TEST_F(AABiasaddConvFusionTest, conv2dBiasAddConstSuccess)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_const_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    auto biasSrc = fusedConv.GetInDataNodesAndPortIndexs(2);
    EXPECT_NE(biasSrc.first, nullptr);

    std::vector<int64_t> strides;
    fusedConv.GetAttr("strides", strides);
    EXPECT_EQ(strides, (std::vector<int64_t>{1, 2, 2, 1}));

    std::vector<int64_t> pads;
    fusedConv.GetAttr("pads", pads);
    EXPECT_EQ(pads, (std::vector<int64_t>{1, 1, 1, 1}));

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dBiasAddConstSuccessNoFixpipe)
{
    SetUpNonFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_const_nofixpipe");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dAddVariableFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_add_variable_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildVariableNode(graphBuilder, "filter_var", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildVariableNode(graphBuilder, "bias_var", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dWithBiasFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_with_bias_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto convBias = BuildConstNode(graphBuilder, "conv_bias", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto addBias = BuildConstNode(graphBuilder, "add_bias", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNodeWithBias(graphBuilder, "conv", {x, filter, convBias},
                                            {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, addBias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    GNode convNode2;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", convNode2));
    auto existingBias = convNode2.GetInDataNodesAndPortIndexs(2);
    EXPECT_NE(existingBias.first, nullptr);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dBiasAddDataNoFixpipeFail)
{
    SetUpNonFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_data_nofixpipe_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = graphBuilder.CreateInput(1, "bias_data", DT_FLOAT, FORMAT_ND, dimsBias);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dBiasAddDataFixpipeSuccess)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_data_fixpipe_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = graphBuilder.CreateInput(1, "bias_data", DT_FLOAT, FORMAT_ND, dimsBias);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dAddWrongShapeFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{1};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_add_wrong_shape_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNode(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv3dAddReshapeDataFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    std::vector<int64_t> dimsFilter{4, 16, 1, 1, 1};
    std::vector<int64_t> dimsBiasData{4, 4, 64, 4096};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{4, 4, 64, 64, 64};

    auto graphBuilder = es::EsGraphBuilder("conv3d_add_reshape_data_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto biasData = graphBuilder.CreateInput(1, "bias_data", DT_FLOAT, FORMAT_NCHW, dimsBiasData);
    auto shape = BuildConstNode(graphBuilder, "shape", dimsShape, FORMAT_ND, DT_INT64);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", biasData, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dAddConstSuccess)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_add_const_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, depthwiseConv2dBiasAddConstSuccess)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 1};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 56, 56, 64};

    auto graphBuilder = es::EsGraphBuilder("depthwise_conv2d_biasadd_const_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "depthwise_conv", x, filter,
                                    {"DepthwiseConv2D", {1, 1, 1, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "DepthwiseConv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv3dCoutMismatchMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    std::vector<int64_t> dimsFilter{4, 16, 1, 1, 1};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{4, 4, 64, 64, 64};

    auto graphBuilder = es::EsGraphBuilder("conv3d_cout_mismatch");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, depthwiseConv2dGroupsAttrPreserved)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 1};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 56, 56, 64};

    auto graphBuilder = es::EsGraphBuilder("depthwise_conv2d_groups_preserved");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "depthwise_conv", x, filter,
                                    {"DepthwiseConv2D", {1, 1, 1, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    int64_t sourceGroups = 64;
    ASSERT_EQ(convNode->SetAttr("groups", sourceGroups), GRAPH_SUCCESS);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "DepthwiseConv2D", fusedConv));
    int64_t groups = 0;
    ASSERT_EQ(fusedConv.GetAttr("groups", groups), GRAPH_SUCCESS);
    EXPECT_EQ(groups, 64);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv3dBiasAddConstSuccess)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    std::vector<int64_t> dimsFilter{4, 16, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsOut{4, 4, 64, 64, 64};

    auto graphBuilder = es::EsGraphBuilder("conv3d_biasadd_const_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv3D", fusedConv));
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dMultiDimBiasInsufficientOnesFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBadBias{1, 1, 64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_multi_dim_bias_insufficient_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNode(graphBuilder, "bias_bad", dimsBadBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0Conv3dAddReshapeDataBiasMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    std::vector<int64_t> dimsFilter{4, 16, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{4, 4, 64, 64, 64};

    auto graphBuilder = es::EsGraphBuilder("conv3d_add_reshape_framework_limitation");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = graphBuilder.CreateInput(1, "filter", DT_FLOAT, FORMAT_NCDHW, dimsFilter);
    auto bias = graphBuilder.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, dimsBias);
    auto shape = graphBuilder.CreateInput(3, "shape", DT_INT64, FORMAT_ND, dimsShape);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_ND, DT_FLOAT16);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0Conv3dAddReshapeBothConstMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    std::vector<int64_t> dimsFilter{4, 16, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{4, 4, 64, 64, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_conv3d_add_reshape_both_const");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNode(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT16);
    auto shape = BuildConstNode(graphBuilder, "shape_const", dimsShape, FORMAT_ND, DT_INT64);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_ND, DT_FLOAT16);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv3dAddReshapeVariableFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    std::vector<int64_t> dimsFilter{4, 16, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{4, 4, 64, 64, 64};

    auto graphBuilder = es::EsGraphBuilder("conv3d_add_reshape_variable_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildVariableNode(graphBuilder, "bias_var", dimsBias, FORMAT_ND, DT_FLOAT16);
    auto shape = BuildConstNode(graphBuilder, "shape", dimsShape, FORMAT_ND, DT_INT64);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_ND, DT_FLOAT16);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv3dAddReshapeWrongDtypeFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    std::vector<int64_t> dimsFilter{4, 16, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{4, 4, 64, 64, 64};

    auto graphBuilder = es::EsGraphBuilder("conv3d_add_reshape_wrong_dtype_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNode(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_INT32);
    auto shape = BuildConstNode(graphBuilder, "shape", dimsShape, FORMAT_ND, DT_INT64);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_ND, DT_INT32);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, platformQueryFailureFail)
{
    PlatformInfoManager::Instance().platform_info_map_.clear();
    OptionalInfo emptyOptionalInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(emptyOptionalInfo);

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("platform_query_failure_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
}

TEST_F(AABiasaddConvFusionTest, platformQueryFailureVariableFail)
{
    PlatformInfoManager::Instance().platform_info_map_.clear();
    OptionalInfo emptyOptionalInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(emptyOptionalInfo);

    auto graph = BuildP0DirectVariableConvBiasAddGraph("Conv2D", {64});
    ASSERT_NE(graph, nullptr);
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
}

TEST_F(AABiasaddConvFusionTest, platformQueryFailureReshapeFail)
{
    PlatformInfoManager::Instance().platform_info_map_.clear();
    OptionalInfo emptyOptionalInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(emptyOptionalInfo);

    auto graph = BuildP0DirectConstRGraph({4}, DT_FLOAT16, true);
    ASSERT_NE(graph, nullptr);
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
}

TEST_F(AABiasaddConvFusionTest, conv2dBiasAddVariableSuccess)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_variable_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildVariableNode(graphBuilder, "bias_var", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dBiasAddVariableWrongDtypeFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_variable_wrong_dtype_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildVariableNode(graphBuilder, "bias_var", dimsBias, FORMAT_NHWC, DT_HIFLOAT8);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dBiasAddConstWrongDtypeFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_const_wrong_dtype_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_HIFLOAT8);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dBiasAddDataWrongDtypeFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_data_wrong_dtype_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = graphBuilder.CreateInput(1, "bias_data", DT_HIFLOAT8, FORMAT_ND, dimsBias);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, convAttrAndNamePreservation)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv_attr_name_preservation");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "my_conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, SUCCESS);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));

    std::vector<int64_t> strides;
    fusedConv.GetAttr("strides", strides);
    EXPECT_EQ(strides, (std::vector<int64_t>{1, 2, 2, 1}));

    std::vector<int64_t> pads;
    fusedConv.GetAttr("pads", pads);
    EXPECT_EQ(pads, (std::vector<int64_t>{1, 1, 1, 1}));

    AscendString dataFormat;
    fusedConv.GetAttr("data_format", dataFormat);
    EXPECT_STREQ(dataFormat.GetString(), "NHWC");

    AscendString convName;
    ASSERT_EQ(fusedConv.GetName(convName), GRAPH_SUCCESS);
    std::string nameStr = convName.GetString();
    EXPECT_TRUE(nameStr.find("my_conv") != std::string::npos);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dBiasAddMultiConsumersSuccess)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_multi_consumers");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto squareOutput = BuildUnaryNode(graphBuilder, biasaddOutput,
                                       {"square", "Square", dimsOut, FORMAT_NHWC, DT_FLOAT});
    auto asinOutput = BuildUnaryNode(graphBuilder, biasaddOutput, {"asin", "Asin", dimsOut, FORMAT_NHWC, DT_FLOAT});
    auto squareNode = squareOutput.GetProducer();
    auto asinNode = asinOutput.GetProducer();
    ASSERT_NE(squareNode, nullptr);
    ASSERT_NE(asinNode, nullptr);

    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({squareOutput, asinOutput});
    RunPass(graphPtr, SUCCESS);

    EXPECT_EQ(CountNodes(graphPtr, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graphPtr, "Conv2D"), 1);
    EXPECT_EQ(CountNodes(graphPtr, "Square"), 1);
    EXPECT_EQ(CountNodes(graphPtr, "Asin"), 1);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graphPtr, "Conv2D", fusedConv));
    ExpectInputSourceType(*squareNode, "Conv2D");
    ExpectInputSourceType(*asinNode, "Conv2D");

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dMultiConsumersFail)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_multi_consumers_fail");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNode(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode extraNode = es::CompliantNodeBuilder(graph)
                          .OpType("Square")
                          .Name("extra_consumer")
                          .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *convNode, 0, extraNode, 0);
    SetOutputDesc(extraNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto extraOutput = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(extraNode, 0));

    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({biasaddOutput, extraOutput});
    RunPass(graphPtr, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graphPtr, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graphPtr, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dAddInputDescMismatchMustFuse)
{
    SetUpFixpipePlatform();

    auto graph = BuildP0DirectConstConvAddGraph({"Conv2D", "BiasAdd", {64}, {1, 4, 4, 64}, FORMAT_NHWC, "NHWC"});
    ASSERT_NE(graph, nullptr);

    GNode addNode;
    GNode convNode;
    for (const auto& node : graph->GetAllNodes()) {
        AscendString name;
        ASSERT_EQ(node.GetName(name), GRAPH_SUCCESS);
        if (name == "biasadd") {
            addNode = node;
        } else if (name == "conv") {
            convNode = node;
        }
    }
    TensorDesc convOutputDesc;
    ASSERT_EQ(convNode.GetOutputDesc(0, convOutputDesc), GRAPH_SUCCESS);
    SetInputDesc(addNode, 0, {1, 4, 4, 64}, FORMAT_NCHW, DT_FLOAT);

    CustomPassContext passContext;
    passContext.SetPassName(FUSION_NAME.c_str());
    AABiasaddConvFusion pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    TensorDesc fusedOutputDesc;
    ASSERT_EQ(fusedConv.GetOutputDesc(0, fusedOutputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(fusedOutputDesc.GetFormat(), convOutputDesc.GetFormat());
    EXPECT_EQ(fusedOutputDesc.GetShape().GetDims(), convOutputDesc.GetShape().GetDims());

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0AddInputOrderBiasAddConstMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_add_input_order_biasadd_const");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNode(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", bias, convOutput);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dAddReversedConstMustFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_add_reversed_const");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", bias, convOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    TensorDesc fusedBiasDesc;
    ASSERT_EQ(fusedConv.GetInputDesc(2, fusedBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(fusedBiasDesc.GetShape().GetDims(), dimsBias);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dAddReversedVariableMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_add_reversed_variable");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildVariableNode(graphBuilder, "bias_var", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", bias, convOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, conv2dAddReversedDataMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_add_reversed_data");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = graphBuilder.CreateInput(1, "bias_data", DT_FLOAT, FORMAT_ND, dimsBias);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", bias, convOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0Conv2dAddNchwConst1dFuses)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 64, 56, 56};
    std::vector<int64_t> dimsFilter{64, 64, 1, 1};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 64, 28, 28};

    auto graphBuilder = es::EsGraphBuilder("p0_conv2d_add_nchw_const1d");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NCHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 1, 2, 2}, {0, 0, 0, 0}, "NCHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NCHW, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NCHW, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    TensorDesc fusedBiasDesc;
    ASSERT_EQ(fusedConv.GetInputDesc(2, fusedBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(fusedBiasDesc.GetShape().GetDims(), dimsBias);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0Conv2dAddConst5dFuses)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{1, 1, 1, 1, 64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_conv2d_add_const5d");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    TensorDesc fusedBiasDesc;
    ASSERT_EQ(fusedConv.GetInputDesc(2, fusedBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(fusedBiasDesc.GetShape().GetDims(), (std::vector<int64_t>{64}));

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0Conv2dAddConst6dFuses)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{1, 1, 1, 1, 4, 64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_conv2d_add_const6d");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    TensorDesc fusedBiasDesc;
    ASSERT_EQ(fusedConv.GetInputDesc(2, fusedBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(fusedBiasDesc.GetShape().GetDims(), (std::vector<int64_t>{64}));

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0Conv2dBiasAddMissingCdimMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28};

    auto graphBuilder = es::EsGraphBuilder("p0_conv2d_biasadd_missing_cdim");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0DepthwiseConv2dBiasAddHighRankNegativeDimMustNotFuse)
{
    SetUpFixpipePlatform();

    for (const auto& biasDims : std::vector<std::vector<int64_t>>{{1, 1, -1, 64}, {1, 1, -2, 64}}) {
        std::vector<int64_t> dimsX{1, 56, 56, 64};
        std::vector<int64_t> dimsFilter{64, 1, 1, 64};
        std::vector<int64_t> dimsOut{1, 56, 56, 64};

        auto graphBuilder = es::EsGraphBuilder("p0_dw_biasadd_high_rank_negative");
        auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
        auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
        auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", biasDims, FORMAT_NHWC, DT_FLOAT);

        auto convOutput = BuildConvNode(graphBuilder, "dw", x, filter,
                                        {"DepthwiseConv2D", {1, 1, 1, 1}, {1, 1, 1, 1}, "NHWC"});
        auto convNode = convOutput.GetProducer();
        SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

        auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
        auto biasaddNode = biasaddOutput.GetProducer();
        SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
        RunPass(graph, GRAPH_NOT_CHANGED);

        EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
        EXPECT_EQ(CountNodes(graph, "DepthwiseConv2D"), 1);
    }

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0Conv3dBiasAddHighRankInsufficientOnesMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{1, 1, 2, 64};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};

    auto graphBuilder = es::EsGraphBuilder("p0_conv3d_biasadd_insufficient_ones");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCDHW, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0DepthwiseConv2dBiasAddDataNonFixpipeMustNotFuse)
{
    SetUpNonFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 56, 56, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_dw_biasadd_data_nonfixpipe");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = graphBuilder.CreateInput(1, "bias_data", DT_FLOAT, FORMAT_ND, dimsBias);

    auto convOutput = BuildConvNode(graphBuilder, "dw", x, filter,
                                    {"DepthwiseConv2D", {1, 1, 1, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graph, "DepthwiseConv2D"), 1);
}

TEST_F(AABiasaddConvFusionTest, p0ReshapeModeMissingPayloadMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};

    auto graphBuilder = es::EsGraphBuilder("p0_r_mode_missing_payload");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = graphBuilder.CreateInput(1, "filter", DT_FLOAT, FORMAT_NCDHW, dimsFilter);
    auto bias = BuildConstNode(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT);
    auto shape = graphBuilder.CreateInput(2, "shape_data", DT_INT64, FORMAT_ND, dimsShape);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RDualOutputConstShapeMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};

    auto graphBuilder = es::EsGraphBuilder("p0_r_dual_const_shape");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCDHW, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT16);
    auto shape = BuildConstNode(graphBuilder, "shape_const", dimsShape, FORMAT_ND, DT_INT64);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT16);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    const RConsumerOutputs consumers = BuildRConsumers(graphBuilder, reshapeOutput, bias, shape,
                                                       {dimsOut, dimsBias, dimsShape});

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {addOutput, consumers.reshape, consumers.bias, consumers.shape});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RDualOutputReshapeAttrPreservation)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};

    auto graphBuilder = es::EsGraphBuilder("p0_r_dual_reshape_attr_preservation");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = graphBuilder.CreateInput(1, "filter", DT_FLOAT, FORMAT_NCDHW, dimsFilter);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT16);
    auto shape = graphBuilder.CreateInput(2, "shape_data", DT_INT64, FORMAT_ND, dimsShape);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT16);
    int64_t axisValue = -1;
    int64_t numAxesValue = 1;
    ASSERT_EQ(reshapeNode->SetAttr("axis", axisValue), GRAPH_SUCCESS);
    ASSERT_EQ(reshapeNode->SetAttr("num_axes", numAxesValue), GRAPH_SUCCESS);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);

    const RConsumerOutputs consumers = BuildRConsumers(graphBuilder, reshapeOutput, bias, shape,
                                                       {dimsOut, dimsBias, dimsShape});

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {addOutput, consumers.reshape, consumers.bias, consumers.shape});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    GNode targetReshape;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Reshape", targetReshape));
    int64_t axis = 0;
    int64_t numAxes = 0;
    ASSERT_EQ(targetReshape.GetAttr("axis", axis), GRAPH_SUCCESS);
    ASSERT_EQ(targetReshape.GetAttr("num_axes", numAxes), GRAPH_SUCCESS);
    EXPECT_EQ(axis, -1);
    EXPECT_EQ(numAxes, 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConstRank4AllOnesCoutMismatchMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{1, 1, 1, 1};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_const_rank4_all_ones");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_rank4", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConstMissingPayloadMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_const_missing_payload");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNode(graphBuilder, "bias_without_payload", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    Tensor biasTensor;
    ASSERT_NE(addNode->GetInputConstData(1, biasTensor), GRAPH_SUCCESS);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0VariableDynamicShapeMustFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsDynamicBias{-1};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_variable_dynamic_shape");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildVariableNode(graphBuilder, "bias_dynamic", dimsDynamicBias, FORMAT_ND, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    TensorDesc fusedBiasDesc;
    ASSERT_EQ(fusedConv.GetInputDesc(2, fusedBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(fusedBiasDesc.GetShape().GetDims(), dimsDynamicBias);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConvDynamicInputShapeMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsDynamicX{-1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_conv_dynamic_input_shape");
    auto x = graphBuilder.CreateInput(0, "x_dynamic", DT_FLOAT, FORMAT_NHWC, dimsDynamicX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNode(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ParentConstDataRankOneMustFuseAsConst)
{
    SetUpFixpipePlatform();

    auto graphCase = BuildP0ParentConstMappedConvAddGraph({64});
    ASSERT_NE(graphCase.parentGraph, nullptr);
    auto graph = graphCase.childGraph;
    ASSERT_NE(graph, nullptr);
    ASSERT_NE(graph->GetParentNode(), nullptr);

    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    ExpectFusionTargetGraph(graph);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ParentConstDataCoutMismatchMustNotFuse)
{
    SetUpFixpipePlatform();

    auto graphCase = BuildP0ParentConstMappedConvAddGraph({32});
    ASSERT_NE(graphCase.parentGraph, nullptr);
    auto graph = graphCase.childGraph;
    ASSERT_NE(graph, nullptr);
    ASSERT_NE(graph->GetParentNode(), nullptr);

    GNode addNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Add", addNode));
    Tensor parentConstTensor;
    ASSERT_EQ(addNode.GetInputConstData(1, parentConstTensor), GRAPH_SUCCESS);
    EXPECT_EQ(parentConstTensor.GetTensorDesc().GetShape().GetDims(), (std::vector<int64_t>{32}));

    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeReshapeMultiConsumerMustFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    std::vector<int64_t> dimsFilter{4, 16, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{4, 4, 64, 64, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_r_mode_reshape_multi_consumer");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = graphBuilder.CreateInput(1, "filter", DT_FLOAT, FORMAT_NCDHW, dimsFilter);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT16);
    auto shape = graphBuilder.CreateInput(2, "shape_data", DT_INT64, FORMAT_ND, dimsShape);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_ND, DT_FLOAT16);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    const RConsumerOutputs consumers = BuildRConsumers(graphBuilder, reshapeOutput, bias, shape,
                                                       {dimsOut, dimsBias, dimsShape});

    Tensor biasTensor;
    ASSERT_EQ(reshapeNode->GetInputConstData(0, biasTensor), GRAPH_SUCCESS);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {addOutput, consumers.reshape, consumers.bias, consumers.shape});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    ExpectRConsumerTypes(consumers);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RDualOutputReshapeInputDescsMustBeSet)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    std::vector<int64_t> dimsFilter{4, 16, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{4, 4, 64, 64, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_r_dual_reshape_input_descs");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = graphBuilder.CreateInput(1, "filter", DT_FLOAT, FORMAT_NCDHW, dimsFilter);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT16);
    auto shape = graphBuilder.CreateInput(2, "shape_data", DT_INT64, FORMAT_ND, dimsShape);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetInputDesc(*reshapeNode, 0, dimsBias, FORMAT_ND, DT_FLOAT16);
    SetInputDesc(*reshapeNode, 1, dimsShape, FORMAT_ND, DT_INT64);
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_ND, DT_FLOAT16);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    const RConsumerOutputs consumers = BuildRConsumers(graphBuilder, reshapeOutput, bias, shape,
                                                       {dimsOut, dimsBias, dimsShape});

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {addOutput, consumers.reshape, consumers.bias, consumers.shape});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    GNode targetReshape;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Reshape", targetReshape));
    TensorDesc targetBiasDesc;
    TensorDesc targetShapeDesc;
    ASSERT_EQ(targetReshape.GetInputDesc(0, targetBiasDesc), GRAPH_SUCCESS);
    ASSERT_EQ(targetReshape.GetInputDesc(1, targetShapeDesc), GRAPH_SUCCESS);
    EXPECT_EQ(targetBiasDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(targetShapeDesc.GetDataType(), DT_INT64);
    EXPECT_EQ(targetShapeDesc.GetShape().GetDims(), dimsShape);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0StreamSwitchAddControlEdgeMustFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_conv_control_edge");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode streamSwitch = BuildStreamSwitchNode(graphBuilder, "stream_switch", x);
    SetInputDesc(streamSwitch, 0, dimsX, FORMAT_NHWC, DT_FLOAT);
    SetOutputDesc(streamSwitch, 0, dimsX, FORMAT_NHWC, DT_FLOAT);
    ASSERT_EQ(rawGraph->AddControlEdge(streamSwitch, *addNode), GRAPH_SUCCESS);
    ASSERT_EQ(addNode->GetInControlNodes().size(), 1U);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(CountNodes(graph, "StreamSwitch"), 1);

    GNode targetConv;
    GNode finalStreamSwitch;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", targetConv));
    ASSERT_TRUE(FindFirstNodeByType(graph, "StreamSwitch", finalStreamSwitch));
    EXPECT_TRUE(HasControlPathWithinTwoHops(finalStreamSwitch, targetConv));

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConvOutputDescMissingMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_conv_output_desc_missing");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    TensorDesc convOutputDesc;
    ASSERT_EQ(convNode->GetOutputDesc(0, convOutputDesc), GRAPH_SUCCESS);
    ASSERT_TRUE(convOutputDesc.GetShape().GetDims().empty());

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    ConvAddState before;
    CaptureConvAddState(*convNode, *addNode, before);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    ExpectConvAddState(graph, before);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0UnsupportedConvTypeMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_unsupported_conv_type");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv_v2", x, filter,
                                    {"Conv2DV2", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2DV2"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0UnsupportedAddTypeMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_unsupported_add_type");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode subNode = es::CompliantNodeBuilder(rawGraph)
                        .OpType("Sub")
                        .Name("sub_add")
                        .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    es::AddEdgeAndUpdatePeerDesc(*rawGraph, *convNode, 0, subNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*rawGraph, *bias.GetProducer(), 0, subNode, 1);
    SetOutputDesc(subNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto subOutput = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(subNode, 0));

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({subOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Sub"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConvMissingFilterMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_conv_missing_filter");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    auto rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    ASSERT_EQ(rawGraph->RemoveEdge(*filter.GetProducer(), filter.GetProducerOutIndex(), *convNode, 1), GRAPH_SUCCESS);
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0AddMissingInputMustNotFuse)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_add_missing_input");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode addNode = es::CompliantNodeBuilder(rawGraph)
                        .OpType("Add")
                        .Name("add_missing_input")
                        .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    es::AddEdgeAndUpdatePeerDesc(*rawGraph, *convNode, 0, addNode, 0);
    SetOutputDesc(addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto addOutput = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addNode, 0));

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0DirectDataRankOrDynamicMustNotFuse)
{
    SetUpFixpipePlatform();
    for (const auto& biasDims : std::vector<std::vector<int64_t>>{{2, 32}, {-1}}) {
        auto graph = BuildP0DirectDataConvAddGraph("Conv2D", "BiasAdd", biasDims);
        ASSERT_NE(graph, nullptr);
        RunPass(graph, GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
        EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    }
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0Conv3dDirectDataMustNotFuse)
{
    SetUpFixpipePlatform();
    auto graph = BuildP0DirectDataConvAddGraph("Conv3D", "BiasAdd", {4});
    ASSERT_NE(graph, nullptr);
    RunPass(graph, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0FixpipeAddDataMustFuse)
{
    SetUpFixpipePlatform();
    auto graph = BuildP0DirectDataConvAddGraph("Conv2D", "Add", {64});
    ASSERT_NE(graph, nullptr);
    RunPass(graph, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    auto biasSource = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(biasSource.first, nullptr);
    AscendString biasType;
    ASSERT_EQ(biasSource.first->GetType(biasType), GRAPH_SUCCESS);
    EXPECT_STREQ(biasType.GetString(), "Data");
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0AddExtraInputMustNotFuse)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};
    auto graphBuilder = es::EsGraphBuilder("p0_add_extra_input");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto extra = BuildConstNodeWithValue(graphBuilder, "extra", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode addNode = es::CompliantNodeBuilder(rawGraph)
                        .OpType("Add")
                        .Name("add_extra_input")
                        .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"x3", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    es::AddEdgeAndUpdatePeerDesc(*rawGraph, *convNode, 0, addNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*rawGraph, *bias.GetProducer(), 0, addNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*rawGraph, *extra.GetProducer(), 0, addNode, 2);
    SetOutputDesc(addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto addOutput = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addNode, 0));
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0AddAmbiguousConvInputsMustNotFuse)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};
    auto graphBuilder = es::EsGraphBuilder("p0_add_ambiguous_conv_inputs");
    auto x0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto x1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter0 = BuildConstNodeWithValue(graphBuilder, "filter0", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto filter1 = BuildConstNodeWithValue(graphBuilder, "filter1", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto conv0 = BuildConvNode(graphBuilder, "conv0", x0, filter0, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto conv1 = BuildConvNode(graphBuilder, "conv1", x1, filter1, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    SetOutputDesc(*conv0.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    SetOutputDesc(*conv1.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", conv0, conv1);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 2);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeDirectConstBiasDataShapeMustFuse)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};
    auto graphBuilder = es::EsGraphBuilder("p0_r_direct_const_data_shape");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias", dimsBias, FORMAT_ND, DT_FLOAT);
    auto shape = graphBuilder.CreateInput(1, "shape", DT_INT64, FORMAT_ND, dimsShape);
    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    SetOutputDesc(*reshapeOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv3D", fusedConv));
    auto biasSource = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(biasSource.first, nullptr);
    AscendString biasType;
    ASSERT_EQ(biasSource.first->GetType(biasType), GRAPH_SUCCESS);
    EXPECT_STREQ(biasType.GetString(), "Const");
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeConstBiasVariableShapeMustFuse)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};

    auto graphBuilder = es::EsGraphBuilder("p0_r_const_bias_variable_shape");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias", dimsBias, FORMAT_ND, DT_FLOAT);
    auto shape = BuildVariableNode(graphBuilder, "shape_variable", dimsShape, FORMAT_ND, DT_INT64);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    SetOutputDesc(*reshapeOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv3D", fusedConv));
    EXPECT_NE(fusedConv.GetInDataNodesAndPortIndexs(2).first, nullptr);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeDynamicShapeDescMustFuse)
{
    SetUpFixpipePlatform();
    auto graph = BuildP0DirectConstRGraph({4}, DT_FLOAT, true);
    ASSERT_NE(graph, nullptr);

    GNode shapeNode;
    GNode reshapeNode;
    for (const auto& node : graph->GetAllNodes()) {
        AscendString name;
        ASSERT_EQ(node.GetName(name), GRAPH_SUCCESS);
        if (name == "shape_data") {
            shapeNode = node;
        } else if (name == "reshape") {
            reshapeNode = node;
        }
    }
    SetOutputDesc(shapeNode, 0, {-1}, FORMAT_ND, DT_INT64);
    SetInputDesc(reshapeNode, 1, {-1}, FORMAT_ND, DT_INT64);

    RunPass(graph, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeMissingReshapeOutputDescMustFuse)
{
    SetUpFixpipePlatform();
    auto graph = BuildP0DirectConstRGraph({4}, DT_FLOAT, true);
    ASSERT_NE(graph, nullptr);

    GNode reshapeNode;
    GNode addNode;
    for (const auto& node : graph->GetAllNodes()) {
        AscendString name;
        ASSERT_EQ(node.GetName(name), GRAPH_SUCCESS);
        if (name == "reshape") {
            reshapeNode = node;
        } else if (name == "add") {
            addNode = node;
        }
    }
    ASSERT_EQ(reshapeNode.UpdateOutputDesc(0, TensorDesc()), GRAPH_SUCCESS);
    SetInputDesc(addNode, 1, {1, 4, 8, 8, 8}, FORMAT_ND, DT_FLOAT);

    RunPass(graph, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeReshapeAddInputOrderMustNotFuse)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};
    auto graphBuilder = es::EsGraphBuilder("p0_r_swapped_add_inputs");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias", dimsBias, FORMAT_ND, DT_FLOAT);
    auto shape = graphBuilder.CreateInput(1, "shape", DT_INT64, FORMAT_ND, dimsShape);
    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    SetOutputDesc(*reshapeOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", reshapeOutput, convOutput);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeParentConstShapeMustNotFuseOnCurrentGe)
{
    SetUpFixpipePlatform();
    auto graphCase = BuildP0ParentConstMappedConv3DReshapeGraph(false, true);
    ASSERT_NE(graphCase.parentGraph, nullptr);
    auto graph = graphCase.childGraph;
    ASSERT_NE(graph, nullptr);
    ASSERT_NE(graph->GetParentNode(), nullptr);
    RunPass(graph, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeRank4BiasMustNotFuse)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{1, 1, 1, 1};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};
    auto graphBuilder = es::EsGraphBuilder("p0_r_rank4_bias");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_rank4", dimsBias, FORMAT_ND, DT_FLOAT);
    auto shape = graphBuilder.CreateInput(1, "shape", DT_INT64, FORMAT_ND, dimsShape);
    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    SetOutputDesc(*reshapeOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeDynamicBiasMustFuse)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{-1};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};
    auto graphBuilder = es::EsGraphBuilder("p0_r_dynamic_bias");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_dynamic", dimsBias, FORMAT_ND, DT_FLOAT);
    auto shape = graphBuilder.CreateInput(1, "shape", DT_INT64, FORMAT_ND, dimsShape);
    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    SetOutputDesc(*reshapeOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_ND, DT_FLOAT);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 0);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv3D", fusedConv));
    EXPECT_NE(fusedConv.GetInDataNodesAndPortIndexs(2).first, nullptr);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConstHighRankBiasShapeMustBeNormalized)
{
    SetUpFixpipePlatform();
    auto graph = BuildP0DirectConstConvAddGraph(
        {"Conv3D", "Add", {1, 1, 1, 1, 64}, {1, 4, 4, 4, 64}, FORMAT_NDHWC, "NDHWC"});
    ASSERT_NE(graph, nullptr);
    RunPass(graph, SUCCESS);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv3D", fusedConv));
    TensorDesc fusedBiasDesc;
    ASSERT_EQ(fusedConv.GetInputDesc(2, fusedBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(fusedBiasDesc.GetShape().GetDims(), (std::vector<int64_t>{64}));
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConvImplModeAttrMustBePreserved)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 8, 8, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 4, 4, 64};
    auto graphBuilder = es::EsGraphBuilder("p0_conv_impl_mode_attr");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    int64_t implMode = 64;
    ASSERT_EQ(convNode->SetAttr("_op_impl_mode_enum", implMode), GRAPH_SUCCESS);
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, SUCCESS);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    int64_t fusedImplMode = 0;
    ASSERT_EQ(fusedConv.GetAttr("_op_impl_mode_enum", fusedImplMode), GRAPH_SUCCESS);
    EXPECT_EQ(fusedImplMode, implMode);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0VariableSupportedConvTypesMustFuse)
{
    SetUpFixpipePlatform();
    for (const auto& convType : std::vector<std::string>{"DepthwiseConv2D", "Conv3D"}) {
        auto graph = BuildP0DirectVariableConvBiasAddGraph(convType, {4});
        ASSERT_NE(graph, nullptr);
        RunPass(graph, SUCCESS);
        EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
        EXPECT_EQ(CountNodes(graph, convType), 1);
    }
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0VariableAddInputOrderMustNotFuse)
{
    SetUpFixpipePlatform();
    const std::vector<int64_t> dimsX{1, 8, 8, 64};
    const std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    const std::vector<int64_t> dimsOut{1, 4, 4, 64};
    auto graphBuilder = es::EsGraphBuilder("p0_variable_swapped_add");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildVariableNode(graphBuilder, "bias_variable", {64}, FORMAT_ND, DT_FLOAT);
    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {0, 0, 0, 0}, "NHWC"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto swappedOutput = BuildBiasAddNode(graphBuilder, "biasadd", bias, convOutput);
    SetOutputDesc(*swappedOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({swappedOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConstRankMatrixMustNotFuse)
{
    SetUpFixpipePlatform();
    for (const auto& biasDims : std::vector<std::vector<int64_t>>{{}, {1, 64}, {1, 1, 64}}) {
        auto graph = BuildP0DirectConstConvAddGraph({"Conv2D", "Add", biasDims, {1, 4, 4, 64}, FORMAT_NHWC, "NHWC"});
        ASSERT_NE(graph, nullptr);
        RunPass(graph, GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodes(graph, "Add"), 1);
    }
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConstNegativeShapeMustNotFuse)
{
    SetUpFixpipePlatform();
    for (const auto& biasDims : std::vector<std::vector<int64_t>>{{-1}, {-2}, {-3}}) {
        auto graph = BuildP0DirectConstConvAddGraph({"Conv2D", "Add", biasDims, {1, 4, 4, 64}, FORMAT_NHWC, "NHWC"});
        ASSERT_NE(graph, nullptr);
        RunPass(graph, GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodes(graph, "Add"), 1);
    }
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConstMultipleNonOneDimensionsUseLastDimension)
{
    SetUpFixpipePlatform();
    auto graph = BuildP0DirectConstConvAddGraph(
        {"Conv2D", "BiasAdd", {1, 1, 1, 1, 4, 64}, {1, 4, 4, 64}, FORMAT_NHWC, "NHWC"});
    ASSERT_NE(graph, nullptr);
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    TensorDesc fusedBiasDesc;
    ASSERT_EQ(fusedConv.GetInputDesc(2, fusedBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(fusedBiasDesc.GetShape().GetDims(), (std::vector<int64_t>{64}));
    EXPECT_EQ(fusedBiasDesc.GetOriginShape().GetDims(), (std::vector<int64_t>{64}));
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConstCoutFormatMatrix)
{
    SetUpFixpipePlatform();
    struct FormatCase {
        Format format;
        std::string name;
        std::vector<int64_t> outputDims;
        bool shouldFuse;
    };
    const std::vector<FormatCase> cases = {{FORMAT_NCHW, "NCHW", {1, 64, 4, 4}, true},
                                           {FORMAT_NHWC, "NHWC", {1, 4, 4, 64}, true},
                                           {FORMAT_HWCN, "HWCN", {4, 4, 64, 1}, true},
                                           {FORMAT_ND, "ND", {1, 4, 4, 64}, false}};
    for (const auto& formatCase : cases) {
        auto graph = BuildP0DirectConstConvAddGraph(
            {"Conv2D", "BiasAdd", {64}, formatCase.outputDims, formatCase.format, formatCase.name});
        ASSERT_NE(graph, nullptr);
        RunPass(graph, formatCase.shouldFuse ? SUCCESS : GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodes(graph, "BiasAdd"), formatCase.shouldFuse ? 0 : 1);
    }
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0DataNonConstParentMustFuseAsPlaceholder)
{
    SetUpFixpipePlatform();
    auto graphCase = BuildP0ParentConstMappedConvAddGraph({64}, "Identity");
    ASSERT_NE(graphCase.parentGraph, nullptr);
    auto graph = graphCase.childGraph;
    ASSERT_NE(graph, nullptr);
    RunPass(graph, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    const auto biasSource = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(biasSource.first, nullptr);
    AscendString biasType;
    ASSERT_EQ(biasSource.first->GetType(biasType), GRAPH_SUCCESS);
    EXPECT_STREQ(biasType.GetString(), "Data");
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ParentConstInvalidShapeMustNotFuse)
{
    SetUpFixpipePlatform();
    for (const auto& biasDims : std::vector<std::vector<int64_t>>{{-1}, {1, 1, 64}}) {
        auto graphCase = BuildP0ParentConstMappedConvAddGraph(biasDims);
        ASSERT_NE(graphCase.parentGraph, nullptr);
        auto graph = graphCase.childGraph;
        ASSERT_NE(graph, nullptr);
        RunPass(graph, GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodes(graph, "Add"), 1);
    }
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ParentConstPayloadMissingFusesAsPlaceholder)
{
    SetUpFixpipePlatform();

    auto graphCase = BuildP0ParentConstMappedConvAddGraph({64}, "Const", false);
    ASSERT_NE(graphCase.parentGraph, nullptr);
    auto graph = graphCase.childGraph;
    ASSERT_NE(graph, nullptr);

    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0DirectConstCoutMismatchWithPayloadMustNotFuse)
{
    SetUpFixpipePlatform();

    auto graph = BuildP0DirectConstConvAddGraph({"Conv2D", "Add", {32}, {1, 4, 4, 64}, FORMAT_NHWC, "NHWC"});
    ASSERT_NE(graph, nullptr);

    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RParentConstSourceCombinationsMustNotFuse)
{
    SetUpFixpipePlatform();
    for (const auto& sourceCase : std::vector<std::pair<bool, bool>>{{true, false}, {true, true}}) {
        auto graphCase = BuildP0ParentConstMappedConv3DReshapeGraph(sourceCase.first, sourceCase.second);
        ASSERT_NE(graphCase.parentGraph, nullptr);
        auto graph = graphCase.childGraph;
        ASSERT_NE(graph, nullptr);
        RunPass(graph, GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodes(graph, "Add"), 1);
        EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    }
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RShapeInputMissingMustNotFuse)
{
    SetUpFixpipePlatform();

    auto graph = BuildP0DirectConstRGraph({4}, DT_FLOAT16, false);
    ASSERT_NE(graph, nullptr);

    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0AddGraphOutputMustFuse)
{
    SetUpFixpipePlatform();
    auto graph = BuildP0DirectConstConvAddGraph({"Conv2D", "Add", {64}, {1, 4, 4, 64}, FORMAT_NHWC, "NHWC"});
    ASSERT_NE(graph, nullptr);
    RunPass(graph, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    const auto consumers = fusedConv.GetOutDataNodesAndPortIndexs(0);
    ASSERT_EQ(consumers.size(), 1U);
    ASSERT_NE(consumers[0].first, nullptr);
    AscendString consumerType;
    ASSERT_EQ(consumers[0].first->GetType(consumerType), GRAPH_SUCCESS);
    EXPECT_STREQ(consumerType.GetString(), "NetOutput");
    EXPECT_EQ(consumers[0].second, 0);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0TensorDescFieldsMustBePreserved)
{
    SetUpFixpipePlatform();
    const TensorDesc xDesc = MakeP0TensorDesc({1, 8, 8, 64}, FORMAT_NHWC, DT_FLOAT);
    const TensorDesc filterDesc = MakeP0TensorDesc({64, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT);
    const TensorDesc biasDesc = MakeP0TensorDesc({64}, FORMAT_NHWC, DT_FLOAT);
    const TensorDesc outputDesc = MakeP0TensorDesc({1, 4, 4, 64}, FORMAT_NHWC, DT_FLOAT);
    auto graphBuilder = es::EsGraphBuilder("p0_tensor_desc_fields");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, xDesc.GetShape().GetDims());
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", filterDesc.GetShape().GetDims(), FORMAT_NHWC,
                                          DT_FLOAT);
    auto bias = BuildConstNodeWithForcedPayload(graphBuilder, "bias",
                                                {biasDesc.GetShape().GetDims(), FORMAT_NHWC, DT_FLOAT, 64U});
    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode convNode = es::CompliantNodeBuilder(graph)
                         .OpType("Conv2D")
                         .Name("conv_desc")
                         .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"filter", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
    ASSERT_EQ(es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), convNode, 0),
              GRAPH_SUCCESS);
    ASSERT_EQ(es::AddEdgeAndUpdatePeerDesc(*graph, *filter.GetProducer(), filter.GetProducerOutIndex(), convNode, 1),
              GRAPH_SUCCESS);
    ASSERT_EQ(convNode.UpdateInputDesc(0, xDesc), GRAPH_SUCCESS);
    ASSERT_EQ(convNode.UpdateInputDesc(1, filterDesc), GRAPH_SUCCESS);
    std::vector<int64_t> strides{1, 2, 2, 1};
    std::vector<int64_t> pads{0, 0, 0, 0};
    AscendString dataFormat("NHWC");
    convNode.SetAttr("strides", strides);
    convNode.SetAttr("pads", pads);
    convNode.SetAttr("data_format", dataFormat);
    SetOutputDesc(convNode, 0, outputDesc.GetShape().GetDims(), FORMAT_NHWC, DT_FLOAT);
    auto convOutput = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(convNode, 0));
    auto addOutput = BuildAddNode(graphBuilder, "add_desc", convOutput, bias);
    ASSERT_EQ(addOutput.GetProducer()->UpdateInputDesc(0, outputDesc), GRAPH_SUCCESS);
    SetOutputDesc(*addOutput.GetProducer(), 0, outputDesc.GetShape().GetDims(), FORMAT_NHWC, DT_FLOAT);
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({addOutput});
    RunPass(graphPtr, SUCCESS);
    ExpectFusedConvDescs(graphPtr, xDesc, filterDesc, biasDesc, outputDesc);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0NonDirectConvAddMustNotFuse)
{
    SetUpFixpipePlatform();
    const std::vector<int64_t> dimsX{1, 8, 8, 64};
    const std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    const std::vector<int64_t> dimsBias{64};
    const std::vector<int64_t> dimsOut{1, 4, 4, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_non_direct_conv_add");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {0, 0, 0, 0}, "NHWC"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto intermediateOutput = BuildIdentityNode(graphBuilder, "conv_intermediate", convOutput);
    ASSERT_NE(intermediateOutput.GetProducer(), nullptr);
    SetOutputDesc(*intermediateOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", intermediateOutput, bias);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(CountNodes(graph, "Identity"), 1);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0UnsupportedBiasIntermediateMustNotFuse)
{
    SetUpFixpipePlatform();
    const std::vector<int64_t> dimsX{1, 8, 8, 64};
    const std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    const std::vector<int64_t> dimsBias{64};
    const std::vector<int64_t> dimsOut{1, 4, 4, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_unsupported_bias_intermediate");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto biasIntermediate = BuildIdentityNode(graphBuilder, "bias_intermediate", bias);
    ASSERT_NE(biasIntermediate.GetProducer(), nullptr);
    SetOutputDesc(*biasIntermediate.GetProducer(), 0, dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {0, 0, 0, 0}, "NHWC"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto biasAddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, biasIntermediate);
    SetOutputDesc(*biasAddOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasAddOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(CountNodes(graph, "Identity"), 1);
    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 1);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0ConstShapeMismatchUsesShapeForLegacyCompatibility)
{
    SetUpFixpipePlatform();
    const std::vector<int64_t> dimsX{1, 8, 8, 64};
    const std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    const std::vector<int64_t> dimsOut{1, 4, 4, 64};
    const TensorDesc biasDesc = [&]() {
        TensorDesc desc(Shape({64}), FORMAT_ND, DT_FLOAT);
        desc.SetOriginShape(Shape({1, 64}));
        desc.SetOriginFormat(FORMAT_ND);
        return desc;
    }();

    es::EsGraphBuilder graphBuilder("p0_const_shape_origin_shape_mismatch");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithOutputAndValueDescs(graphBuilder, "bias", biasDesc, biasDesc, 64U);
    ASSERT_NE(bias.GetProducer(), nullptr);
    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {0, 0, 0, 0}, "NHWC"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto addOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    ASSERT_NE(graph, nullptr);
    TensorDesc encodedBiasDesc;
    ASSERT_EQ(bias.GetProducer()->GetOutputDesc(0, encodedBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(encodedBiasDesc.GetShape().GetDims(), (std::vector<int64_t>{64}));
    EXPECT_EQ(encodedBiasDesc.GetOriginShape().GetDims(), (std::vector<int64_t>{1, 64}));
    RunPass(graph, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0PlaceholderShapeMismatchUsesShapeForLegacyCompatibility)
{
    SetUpFixpipePlatform();
    const std::vector<int64_t> dimsX{1, 8, 8, 64};
    const std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    const std::vector<int64_t> dimsOut{1, 4, 4, 64};
    const TensorDesc biasDesc = [&]() {
        TensorDesc desc(Shape({64}), FORMAT_ND, DT_FLOAT);
        desc.SetOriginShape(Shape({1, 64}));
        desc.SetOriginFormat(FORMAT_ND);
        return desc;
    }();

    es::EsGraphBuilder graphBuilder("p0_placeholder_shape_origin_shape_mismatch");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = graphBuilder.CreateInput(1, "bias_data", DT_FLOAT, FORMAT_ND, {64});
    ASSERT_NE(bias.GetProducer(), nullptr);
    ASSERT_EQ(bias.GetProducer()->UpdateOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {0, 0, 0, 0}, "NHWC"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    ASSERT_NE(graph, nullptr);
    TensorDesc encodedBiasDesc;
    ASSERT_EQ(bias.GetProducer()->GetOutputDesc(0, encodedBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(encodedBiasDesc.GetShape().GetDims(), (std::vector<int64_t>{64}));
    EXPECT_EQ(encodedBiasDesc.GetOriginShape().GetDims(), (std::vector<int64_t>{1, 64}));
    RunPass(graph, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0SharedConstSourceDescriptorMutationIsObservable)
{
    SetUpFixpipePlatform();
    const std::vector<int64_t> dimsX{1, 8, 8, 64};
    const std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    const std::vector<int64_t> dimsBias{1, 1, 1, 1, 64};
    const std::vector<int64_t> dimsOut{1, 4, 4, 64};
    TensorDesc biasDesc(Shape(dimsBias), FORMAT_ND, DT_FLOAT);
    biasDesc.SetOriginShape(Shape(dimsBias));
    biasDesc.SetOriginFormat(FORMAT_ND);

    es::EsGraphBuilder graphBuilder("p0_shared_const_source_descriptor");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithOutputAndValueDescs(graphBuilder, "shared_bias", biasDesc, biasDesc, 64U);
    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {0, 0, 0, 0}, "NHWC"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto addOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    const TensorDesc addInputDesc = MakeP0TensorDesc(dimsOut, FORMAT_NHWC, DT_FLOAT);
    ASSERT_EQ(addOutput.GetProducer()->UpdateInputDesc(0, addInputDesc), GRAPH_SUCCESS);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto bypassOutput = BuildIdentityNode(graphBuilder, "bias_bypass", bias);
    ASSERT_NE(bypassOutput.GetProducer(), nullptr);
    SetOutputDesc(*bypassOutput.GetProducer(), 0, dimsBias, FORMAT_ND, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput, bypassOutput});
    ASSERT_NE(graph, nullptr);
    RunPass(graph, SUCCESS);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    TensorDesc targetBiasDesc;
    ASSERT_EQ(fusedConv.GetInputDesc(2, targetBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(targetBiasDesc.GetShape().GetDims(), (std::vector<int64_t>{64}));

    const auto source = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(source.first, nullptr);
    TensorDesc sourceDesc;
    ASSERT_EQ(source.first->GetOutputDesc(source.second, sourceDesc), GRAPH_SUCCESS);
    const std::vector<int64_t> normalizedBiasShape{64};
    EXPECT_EQ(sourceDesc.GetShape().GetDims(), normalizedBiasShape);
    EXPECT_EQ(sourceDesc.GetOriginShape().GetDims(), normalizedBiasShape);
    EXPECT_EQ(sourceDesc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(sourceDesc.GetOriginFormat(), FORMAT_NHWC);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0VariableBiasOriginMetadataIsObservable)
{
    SetUpFixpipePlatform();
    const std::vector<int64_t> dimsX{1, 8, 8, 64};
    const std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    const std::vector<int64_t> dimsOut{1, 4, 4, 64};
    TensorDesc biasDesc(Shape({64}), FORMAT_ND, DT_FLOAT);
    biasDesc.SetOriginShape(Shape({1, 64}));
    biasDesc.SetOriginFormat(FORMAT_NCHW);

    es::EsGraphBuilder graphBuilder("p0_variable_bias_origin_metadata");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildVariableNode(graphBuilder, "bias_variable", {64}, FORMAT_ND, DT_FLOAT);
    ASSERT_NE(bias.GetProducer(), nullptr);
    ASSERT_EQ(bias.GetProducer()->UpdateOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {0, 0, 0, 0}, "NHWC"});
    const TensorDesc xDesc = MakeP0TensorDesc(dimsX, FORMAT_NHWC, DT_FLOAT);
    const TensorDesc filterDesc = MakeP0TensorDesc(dimsFilter, FORMAT_NHWC, DT_FLOAT);
    ASSERT_EQ(convOutput.GetProducer()->UpdateInputDesc(0, xDesc), GRAPH_SUCCESS);
    ASSERT_EQ(convOutput.GetProducer()->UpdateInputDesc(1, filterDesc), GRAPH_SUCCESS);
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto addOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    const TensorDesc addInputDesc = MakeP0TensorDesc(dimsOut, FORMAT_NHWC, DT_FLOAT);
    ASSERT_EQ(addOutput.GetProducer()->UpdateInputDesc(0, addInputDesc), GRAPH_SUCCESS);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    ASSERT_NE(graph, nullptr);
    RunPass(graph, SUCCESS);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    TensorDesc targetBiasDesc;
    ASSERT_EQ(fusedConv.GetInputDesc(2, targetBiasDesc), GRAPH_SUCCESS);
    const std::vector<int64_t> normalizedBiasShape{64};
    EXPECT_EQ(targetBiasDesc.GetShape().GetDims(), normalizedBiasShape);
    EXPECT_EQ(targetBiasDesc.GetOriginShape().GetDims(), normalizedBiasShape);
    EXPECT_EQ(targetBiasDesc.GetOriginFormat(), FORMAT_NHWC);
    const auto source = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(source.first, nullptr);
    TensorDesc sourceDesc;
    ASSERT_EQ(source.first->GetOutputDesc(source.second, sourceDesc), GRAPH_SUCCESS);
    EXPECT_EQ(sourceDesc.GetShape().GetDims(), normalizedBiasShape);
    EXPECT_EQ(sourceDesc.GetOriginShape().GetDims(), normalizedBiasShape);
    EXPECT_EQ(sourceDesc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(sourceDesc.GetOriginFormat(), FORMAT_NHWC);
    ClearPlatform();
}

struct AcceptanceSourceRef {
    GNode node;
    int32_t outputIndex = 0;
};

struct AcceptanceGraphCase {
    std::unique_ptr<es::EsGraphBuilder> builder;
    Graph* graph = nullptr;
    std::shared_ptr<Graph> result;
    AcceptanceSourceRef x;
    AcceptanceSourceRef biasData;
    AcceptanceSourceRef shapeInput;
    GNode filter;
    GNode bias;
    GNode convBias;
    GNode addBias;
    GNode shape;
    GNode conv;
    GNode reshape;
    GNode add;
    GNode square;
    GNode asin;
    GNode identity;
};

struct AcceptanceNodeConfig {
    std::string name;
    std::string type;
    TensorDesc desc;
    std::vector<uint8_t> value;
    bool hasValue = false;
};

static void BuildAcceptanceSource(Graph& graph, const AcceptanceNodeConfig& config, GNode& node)
{
    node = es::CompliantNodeBuilder(&graph)
               .OpType(config.type.c_str())
               .Name(config.name.c_str())
               .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
               .Build();
    ASSERT_EQ(node.UpdateOutputDesc(0, config.desc), GRAPH_SUCCESS);
    if (config.hasValue) {
        Tensor value(config.desc, config.value);
        ASSERT_EQ(node.SetAttr("value", value), GRAPH_SUCCESS);
    }
}

struct AcceptanceConvConfig {
    std::string name;
    std::string type;
    AcceptanceSourceRef x;
    AcceptanceSourceRef filter;
    AcceptanceSourceRef bias;
    bool hasBias = false;
    TensorDesc outputDesc;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::string dataFormat;
};

static void BuildAcceptanceConv(Graph& graph, const AcceptanceConvConfig& config, GNode& node)
{
    node = es::CompliantNodeBuilder(&graph)
               .OpType(config.type.c_str())
               .Name(config.name.c_str())
               .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"filter", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
               .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
               .Build();
    GNode x = config.x.node;
    GNode filter = config.filter.node;
    ASSERT_EQ(es::AddEdgeAndUpdatePeerDesc(graph, x, config.x.outputIndex, node, 0), GRAPH_SUCCESS);
    ASSERT_EQ(es::AddEdgeAndUpdatePeerDesc(graph, filter, config.filter.outputIndex, node, 1), GRAPH_SUCCESS);
    if (config.hasBias) {
        GNode bias = config.bias.node;
        ASSERT_EQ(es::AddEdgeAndUpdatePeerDesc(graph, bias, config.bias.outputIndex, node, 2), GRAPH_SUCCESS);
    }
    auto strides = config.strides;
    auto pads = config.pads;
    AscendString dataFormat(config.dataFormat.c_str());
    ASSERT_EQ(node.SetAttr("strides", strides), GRAPH_SUCCESS);
    ASSERT_EQ(node.SetAttr("pads", pads), GRAPH_SUCCESS);
    ASSERT_EQ(node.SetAttr("data_format", dataFormat), GRAPH_SUCCESS);
    ASSERT_EQ(node.UpdateOutputDesc(0, config.outputDesc), GRAPH_SUCCESS);
}

struct AcceptanceBinaryConfig {
    std::string name;
    std::string type;
    AcceptanceSourceRef first;
    AcceptanceSourceRef second;
    TensorDesc outputDesc;
};

static void BuildAcceptanceBinary(Graph& graph, const AcceptanceBinaryConfig& config, GNode& node)
{
    es::CompliantNodeBuilder builder(&graph);
    builder.OpType(config.type.c_str()).Name(config.name.c_str());
    if (config.type == "BiasAdd") {
        builder.IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"bias", es::CompliantNodeBuilder::kEsIrInputRequired, ""}});
    } else {
        builder.IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}});
    }
    node = builder.IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}}).Build();
    GNode first = config.first.node;
    GNode second = config.second.node;
    ASSERT_EQ(es::AddEdgeAndUpdatePeerDesc(graph, first, config.first.outputIndex, node, 0), GRAPH_SUCCESS);
    ASSERT_EQ(es::AddEdgeAndUpdatePeerDesc(graph, second, config.second.outputIndex, node, 1), GRAPH_SUCCESS);
    ASSERT_EQ(node.UpdateOutputDesc(0, config.outputDesc), GRAPH_SUCCESS);
}

struct AcceptanceReshapeConfig {
    std::string name;
    AcceptanceSourceRef input;
    AcceptanceSourceRef shape;
    TensorDesc outputDesc;
};

static void BuildAcceptanceReshape(Graph& graph, const AcceptanceReshapeConfig& config, GNode& node)
{
    node = es::CompliantNodeBuilder(&graph)
               .OpType("Reshape")
               .Name(config.name.c_str())
               .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
               .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
               .Build();
    GNode input = config.input.node;
    GNode shape = config.shape.node;
    ASSERT_EQ(es::AddEdgeAndUpdatePeerDesc(graph, input, config.input.outputIndex, node, 0), GRAPH_SUCCESS);
    ASSERT_EQ(es::AddEdgeAndUpdatePeerDesc(graph, shape, config.shape.outputIndex, node, 1), GRAPH_SUCCESS);
    ASSERT_EQ(node.UpdateOutputDesc(0, config.outputDesc), GRAPH_SUCCESS);
}

struct AcceptanceUnaryConfig {
    std::string name;
    std::string type;
    AcceptanceSourceRef input;
    TensorDesc outputDesc;
};

static void BuildAcceptanceUnary(Graph& graph, const AcceptanceUnaryConfig& config, GNode& node)
{
    node = es::CompliantNodeBuilder(&graph)
               .OpType(config.type.c_str())
               .Name(config.name.c_str())
               .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
               .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
               .Build();
    GNode source = config.input.node;
    ASSERT_EQ(es::AddEdgeAndUpdatePeerDesc(graph, source, config.input.outputIndex, node, 0), GRAPH_SUCCESS);
    ASSERT_EQ(node.UpdateOutputDesc(0, config.outputDesc), GRAPH_SUCCESS);
}

static void SetAcceptanceInput(const es::EsTensorHolder& holder, AcceptanceSourceRef& source)
{
    ASSERT_NE(holder.GetProducer(), nullptr);
    source = {*holder.GetProducer(), holder.GetProducerOutIndex()};
}

static void FinalizeAcceptanceGraph(es::EsGraphBuilder& builder, const std::vector<GNode>& outputNodes,
                                    std::shared_ptr<Graph>& result)
{
    std::vector<es::EsTensorHolder> outputs;
    for (const auto& outputNode : outputNodes) {
        outputs.emplace_back(builder.GetCGraphBuilder()->GetTensorHolderFromNode(outputNode, 0));
    }
    result = builder.BuildAndReset(outputs);
    ASSERT_NE(result, nullptr);
}

static void RunAcceptancePass(std::shared_ptr<Graph>& graph, Status expectedStatus)
{
    ASSERT_NE(graph, nullptr);
    CustomPassContext passContext;
    passContext.SetPassName("AABiasaddConvFusion");
    AABiasaddConvFusion pass;
    const Status runStatus = pass.Run(graph, passContext);
    if (expectedStatus == GRAPH_NOT_CHANGED) {
        ASSERT_TRUE(runStatus == GRAPH_NOT_CHANGED || runStatus == CONV_NOT_CHANGED) << "status=" << runStatus;
    } else {
        ASSERT_EQ(runStatus, expectedStatus);
    }
}

static void BuildLegacyConstBiasCase(AcceptanceGraphCase& graphCase, const std::string& graphName,
                                     const std::string& prefix)
{
    const std::vector<int64_t> dimsX{1, 56, 56, 64};
    const TensorDesc filterDesc = MakeP0TensorDesc({64, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT);
    const TensorDesc biasDesc = MakeP0TensorDesc({64}, FORMAT_NHWC, DT_FLOAT);
    const TensorDesc outputDesc = MakeP0TensorDesc({1, 28, 28, 64}, FORMAT_NHWC, DT_FLOAT);
    graphCase.builder = std::make_unique<es::EsGraphBuilder>(graphName.c_str());
    auto x = graphCase.builder->CreateInput(0, (prefix + "_x").c_str(), DT_FLOAT, FORMAT_NHWC, dimsX);
    SetAcceptanceInput(x, graphCase.x);
    graphCase.graph = graphCase.builder->GetCGraphBuilder()->GetGraph();
    ASSERT_NE(graphCase.graph, nullptr);
    BuildAcceptanceSource(
        *graphCase.graph,
        {prefix + "_filter", "Const", filterDesc, std::vector<uint8_t>(64U * 64U * sizeof(float), 0U), true},
        graphCase.filter);
    BuildAcceptanceSource(*graphCase.graph,
                          {prefix + "_bias", "Const", biasDesc, std::vector<uint8_t>(64U * sizeof(float), 0U), true},
                          graphCase.bias);
    BuildAcceptanceConv(*graphCase.graph,
                        {prefix + "_conv",
                         "Conv2D",
                         graphCase.x,
                         {graphCase.filter, 0},
                         {},
                         false,
                         outputDesc,
                         {1, 2, 2, 1},
                         {1, 1, 1, 1},
                         "NHWC"},
                        graphCase.conv);
    BuildAcceptanceBinary(*graphCase.graph,
                          {prefix + "_biasadd", "BiasAdd", {graphCase.conv, 0}, {graphCase.bias, 0}, outputDesc},
                          graphCase.add);
    BuildAcceptanceUnary(*graphCase.graph, {prefix + "_square", "Square", {graphCase.add, 0}, outputDesc},
                         graphCase.square);
    BuildAcceptanceUnary(*graphCase.graph, {prefix + "_asin", "Asin", {graphCase.add, 0}, outputDesc}, graphCase.asin);
    FinalizeAcceptanceGraph(*graphCase.builder, {graphCase.square, graphCase.asin}, graphCase.result);
}

static void VerifyLegacyConstBiasTopology(const AcceptanceGraphCase& graphCase, const std::string& prefix)
{
    int32_t convCount = 0;
    int32_t biasAddCount = 0;
    GNode fusedConv;
    for (const auto& node : graphCase.result->GetAllNodes()) {
        AscendString type;
        ASSERT_EQ(node.GetType(type), GRAPH_SUCCESS);
        if (type == "Conv2D") {
            ++convCount;
            fusedConv = node;
        } else if (type == "BiasAdd") {
            ++biasAddCount;
        }
    }
    EXPECT_EQ(convCount, 1);
    EXPECT_EQ(biasAddCount, 0);
    const auto xSource = fusedConv.GetInDataNodesAndPortIndexs(0);
    const auto filterSource = fusedConv.GetInDataNodesAndPortIndexs(1);
    const auto biasSource = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(xSource.first, nullptr);
    ASSERT_NE(filterSource.first, nullptr);
    ASSERT_NE(biasSource.first, nullptr);
    AscendString sourceName;
    ASSERT_EQ(xSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_x").c_str());
    ASSERT_EQ(filterSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_filter").c_str());
    ASSERT_EQ(biasSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_bias").c_str());
}

static void VerifyLegacyConstBiasConsumersAndAttrs(const AcceptanceGraphCase& graphCase)
{
    const auto squareSource = graphCase.square.GetInDataNodesAndPortIndexs(0);
    const auto asinSource = graphCase.asin.GetInDataNodesAndPortIndexs(0);
    ASSERT_NE(squareSource.first, nullptr);
    ASSERT_NE(asinSource.first, nullptr);
    AscendString sourceType;
    ASSERT_EQ(squareSource.first->GetType(sourceType), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceType.GetString(), "Conv2D");
    ASSERT_EQ(asinSource.first->GetType(sourceType), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceType.GetString(), "Conv2D");
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graphCase.result, "Conv2D", fusedConv));
    std::vector<int64_t> actualStrides;
    std::vector<int64_t> actualPads;
    ASSERT_EQ(fusedConv.GetAttr("strides", actualStrides), GRAPH_SUCCESS);
    ASSERT_EQ(fusedConv.GetAttr("pads", actualPads), GRAPH_SUCCESS);
    EXPECT_EQ(actualStrides, (std::vector<int64_t>{1, 2, 2, 1}));
    EXPECT_EQ(actualPads, (std::vector<int64_t>{1, 1, 1, 1}));
}

struct Acceptance2DBinaryConfig {
    std::string graphName;
    std::string prefix;
    std::string filterType;
    bool filterHasValue;
    std::string biasType;
    bool biasHasValue;
    bool biasIsInput;
    std::string addType;
    std::vector<int64_t> biasDims;
    Format biasFormat;
    size_t biasElementCount;
};

static void BuildAcceptance2DBinaryCase(AcceptanceGraphCase& graphCase, const Acceptance2DBinaryConfig& config)
{
    const std::vector<int64_t> dimsX{1, 56, 56, 64};
    const TensorDesc filterDesc = MakeP0TensorDesc({64, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT);
    const TensorDesc biasDesc = MakeP0TensorDesc(config.biasDims, config.biasFormat, DT_FLOAT);
    const TensorDesc outputDesc = MakeP0TensorDesc({1, 28, 28, 64}, FORMAT_NHWC, DT_FLOAT);
    graphCase.builder = std::make_unique<es::EsGraphBuilder>(config.graphName.c_str());
    auto x = graphCase.builder->CreateInput(0, (config.prefix + "_x").c_str(), DT_FLOAT, FORMAT_NHWC, dimsX);
    SetAcceptanceInput(x, graphCase.x);
    if (config.biasIsInput) {
        auto bias = graphCase.builder->CreateInput(1, (config.prefix + "_bias_data").c_str(), DT_FLOAT,
                                                   config.biasFormat, config.biasDims);
        SetAcceptanceInput(bias, graphCase.biasData);
    }
    graphCase.graph = graphCase.builder->GetCGraphBuilder()->GetGraph();
    ASSERT_NE(graphCase.graph, nullptr);
    BuildAcceptanceSource(*graphCase.graph,
                          {config.prefix + "_filter", config.filterType, filterDesc,
                           std::vector<uint8_t>(64U * 64U * sizeof(float), 0U), config.filterHasValue},
                          graphCase.filter);
    size_t biasByteCount = 0U;
    ASSERT_TRUE(TryMultiplySize(config.biasElementCount, sizeof(float), biasByteCount));
    if (!config.biasIsInput) {
        BuildAcceptanceSource(*graphCase.graph,
                              {config.prefix + "_bias", config.biasType, biasDesc,
                               std::vector<uint8_t>(biasByteCount, 0U), config.biasHasValue},
                              graphCase.bias);
    }
    const AcceptanceSourceRef biasSource = config.biasIsInput ? graphCase.biasData :
                                                                AcceptanceSourceRef{graphCase.bias, 0};
    BuildAcceptanceConv(*graphCase.graph,
                        {config.prefix + "_conv",
                         "Conv2D",
                         graphCase.x,
                         {graphCase.filter, 0},
                         {},
                         false,
                         outputDesc,
                         {1, 2, 2, 1},
                         {1, 1, 1, 1},
                         "NHWC"},
                        graphCase.conv);
    BuildAcceptanceBinary(*graphCase.graph,
                          {config.prefix + (config.addType == "Add" ? "_add" : "_biasadd"),
                           config.addType,
                           {graphCase.conv, 0},
                           biasSource,
                           outputDesc},
                          graphCase.add);
    FinalizeAcceptanceGraph(*graphCase.builder, {graphCase.add}, graphCase.result);
}

static void VerifyUnfused2DBinaryCase(const AcceptanceGraphCase& graphCase, const std::string& prefix,
                                      const std::string& addType, const std::string& expectedBiasName)
{
    int32_t convCount = 0;
    int32_t addCount = 0;
    for (const auto& node : graphCase.result->GetAllNodes()) {
        AscendString type;
        ASSERT_EQ(node.GetType(type), GRAPH_SUCCESS);
        convCount += (type == "Conv2D");
        addCount += (type == addType.c_str());
    }
    EXPECT_EQ(convCount, 1);
    EXPECT_EQ(addCount, 1);
    EXPECT_EQ(graphCase.conv.GetInDataNodesAndPortIndexs(2).first, nullptr);
    const auto addConvSource = graphCase.add.GetInDataNodesAndPortIndexs(0);
    const auto addBiasSource = graphCase.add.GetInDataNodesAndPortIndexs(1);
    ASSERT_NE(addConvSource.first, nullptr);
    ASSERT_NE(addBiasSource.first, nullptr);
    AscendString sourceName;
    ASSERT_EQ(addConvSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_conv").c_str());
    ASSERT_EQ(addBiasSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), expectedBiasName.c_str());
}

static void VerifyFused2DBinaryCase(const AcceptanceGraphCase& graphCase, const std::string& prefix,
                                    const std::string& expectedBiasName, const std::string& removedAddType)
{
    int32_t convCount = 0;
    int32_t addCount = 0;
    GNode fusedConv;
    for (const auto& node : graphCase.result->GetAllNodes()) {
        AscendString type;
        ASSERT_EQ(node.GetType(type), GRAPH_SUCCESS);
        if (type == "Conv2D") {
            ++convCount;
            fusedConv = node;
        }
        addCount += (type == removedAddType.c_str());
    }
    EXPECT_EQ(convCount, 1);
    EXPECT_EQ(addCount, 0);
    const auto xSource = fusedConv.GetInDataNodesAndPortIndexs(0);
    const auto filterSource = fusedConv.GetInDataNodesAndPortIndexs(1);
    const auto biasSource = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(xSource.first, nullptr);
    ASSERT_NE(filterSource.first, nullptr);
    ASSERT_NE(biasSource.first, nullptr);
    AscendString sourceName;
    ASSERT_EQ(xSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_x").c_str());
    ASSERT_EQ(filterSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_filter").c_str());
    ASSERT_EQ(biasSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), expectedBiasName.c_str());
}

static void BuildAcceptanceConvWithBiasCase(AcceptanceGraphCase& graphCase, const std::string& graphName,
                                            const std::string& prefix)
{
    const std::vector<int64_t> dimsX{1, 56, 56, 64};
    const TensorDesc filterDesc = MakeP0TensorDesc({64, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT);
    const TensorDesc biasDesc = MakeP0TensorDesc({64}, FORMAT_NHWC, DT_FLOAT);
    const TensorDesc outputDesc = MakeP0TensorDesc({1, 28, 28, 64}, FORMAT_NHWC, DT_FLOAT);
    graphCase.builder = std::make_unique<es::EsGraphBuilder>(graphName.c_str());
    auto x = graphCase.builder->CreateInput(0, (prefix + "_x").c_str(), DT_FLOAT, FORMAT_NHWC, dimsX);
    SetAcceptanceInput(x, graphCase.x);
    graphCase.graph = graphCase.builder->GetCGraphBuilder()->GetGraph();
    ASSERT_NE(graphCase.graph, nullptr);
    const std::vector<uint8_t> filterBytes(64U * 64U * sizeof(float), 0U);
    const std::vector<uint8_t> biasBytes(64U * sizeof(float), 0U);
    BuildAcceptanceSource(*graphCase.graph, {prefix + "_filter", "Const", filterDesc, filterBytes, true},
                          graphCase.filter);
    BuildAcceptanceSource(*graphCase.graph, {prefix + "_conv_bias", "Const", biasDesc, biasBytes, true},
                          graphCase.convBias);
    BuildAcceptanceSource(*graphCase.graph, {prefix + "_add_bias", "Const", biasDesc, biasBytes, true},
                          graphCase.addBias);
    BuildAcceptanceConv(*graphCase.graph,
                        {prefix + "_conv",
                         "Conv2D",
                         graphCase.x,
                         {graphCase.filter, 0},
                         {graphCase.convBias, 0},
                         true,
                         outputDesc,
                         {1, 2, 2, 1},
                         {1, 1, 1, 1},
                         "NHWC"},
                        graphCase.conv);
    BuildAcceptanceBinary(*graphCase.graph,
                          {prefix + "_biasadd", "BiasAdd", {graphCase.conv, 0}, {graphCase.addBias, 0}, outputDesc},
                          graphCase.add);
    FinalizeAcceptanceGraph(*graphCase.builder, {graphCase.add}, graphCase.result);
}

static void VerifyAcceptanceConvWithBiasCase(const AcceptanceGraphCase& graphCase, const std::string& prefix)
{
    int32_t convCount = 0;
    int32_t addCount = 0;
    for (const auto& node : graphCase.result->GetAllNodes()) {
        AscendString type;
        ASSERT_EQ(node.GetType(type), GRAPH_SUCCESS);
        convCount += (type == "Conv2D");
        addCount += (type == "BiasAdd");
    }
    EXPECT_EQ(convCount, 1);
    EXPECT_EQ(addCount, 1);
    const auto convBiasSource = graphCase.conv.GetInDataNodesAndPortIndexs(2);
    const auto addConvSource = graphCase.add.GetInDataNodesAndPortIndexs(0);
    const auto addBiasSource = graphCase.add.GetInDataNodesAndPortIndexs(1);
    ASSERT_NE(convBiasSource.first, nullptr);
    ASSERT_NE(addConvSource.first, nullptr);
    ASSERT_NE(addBiasSource.first, nullptr);
    AscendString sourceName;
    ASSERT_EQ(convBiasSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_conv_bias").c_str());
    ASSERT_EQ(addConvSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_conv").c_str());
    ASSERT_EQ(addBiasSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_add_bias").c_str());
}

static void BuildAcceptanceLegacyReshapeSources(AcceptanceGraphCase& graphCase, const std::string& prefix,
                                                const TensorDesc& filterDesc, const TensorDesc& shapeDesc)
{
    BuildAcceptanceSource(
        *graphCase.graph,
        {prefix + "_filter", "Const", filterDesc, std::vector<uint8_t>(4U * 16U * sizeof(float), 0U), true},
        graphCase.filter);
    const std::vector<int64_t> shapeValues{4, 4, 64, 64, 64};
    size_t shapeByteCount = 0U;
    ASSERT_TRUE(TryMultiplySize(shapeValues.size(), sizeof(int64_t), shapeByteCount));
    std::vector<uint8_t> shapeBytes(shapeByteCount, 0U);
    std::memcpy(shapeBytes.data(), shapeValues.data(), shapeBytes.size());
    BuildAcceptanceSource(*graphCase.graph, {prefix + "_shape", "Const", shapeDesc, shapeBytes, true}, graphCase.shape);
}

static void BuildAcceptanceLegacyReshapeDataCase(AcceptanceGraphCase& graphCase, const std::string& graphName,
                                                 const std::string& prefix)
{
    const std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    const std::vector<int64_t> dimsBiasData{4, 4, 64, 4096};
    const TensorDesc filterDesc = MakeP0TensorDesc({4, 16, 1, 1, 1}, FORMAT_NCDHW, DT_FLOAT);
    const TensorDesc shapeDesc = MakeP0TensorDesc({5}, FORMAT_ND, DT_INT64);
    const TensorDesc outputDesc = MakeP0TensorDesc({4, 4, 64, 64, 64}, FORMAT_ND, DT_FLOAT);
    graphCase.builder = std::make_unique<es::EsGraphBuilder>(graphName.c_str());
    auto x = graphCase.builder->CreateInput(0, (prefix + "_x").c_str(), DT_FLOAT, FORMAT_ND, dimsX);
    auto bias = graphCase.builder->CreateInput(1, (prefix + "_bias_data").c_str(), DT_FLOAT, FORMAT_NCHW, dimsBiasData);
    SetAcceptanceInput(x, graphCase.x);
    SetAcceptanceInput(bias, graphCase.biasData);
    graphCase.graph = graphCase.builder->GetCGraphBuilder()->GetGraph();
    ASSERT_NE(graphCase.graph, nullptr);
    BuildAcceptanceLegacyReshapeSources(graphCase, prefix, filterDesc, shapeDesc);
    BuildAcceptanceConv(*graphCase.graph,
                        {prefix + "_conv3d",
                         "Conv3D",
                         graphCase.x,
                         {graphCase.filter, 0},
                         {},
                         false,
                         outputDesc,
                         {1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         "NCDHW"},
                        graphCase.conv);
    BuildAcceptanceReshape(*graphCase.graph,
                           {prefix + "_reshape", graphCase.biasData, {graphCase.shape, 0}, outputDesc},
                           graphCase.reshape);
    BuildAcceptanceBinary(*graphCase.graph,
                          {prefix + "_add", "Add", {graphCase.conv, 0}, {graphCase.reshape, 0}, outputDesc},
                          graphCase.add);
    FinalizeAcceptanceGraph(*graphCase.builder, {graphCase.add}, graphCase.result);
}

static void VerifyAcceptanceLegacyReshapeDataCase(const AcceptanceGraphCase& graphCase, const std::string& prefix)
{
    int32_t convCount = 0;
    int32_t reshapeCount = 0;
    int32_t addCount = 0;
    for (const auto& node : graphCase.result->GetAllNodes()) {
        AscendString type;
        ASSERT_EQ(node.GetType(type), GRAPH_SUCCESS);
        convCount += (type == "Conv3D");
        reshapeCount += (type == "Reshape");
        addCount += (type == "Add");
    }
    EXPECT_EQ(convCount, 1);
    EXPECT_EQ(reshapeCount, 1);
    EXPECT_EQ(addCount, 1);
    const auto reshapeBiasSource = graphCase.reshape.GetInDataNodesAndPortIndexs(0);
    const auto reshapeShapeSource = graphCase.reshape.GetInDataNodesAndPortIndexs(1);
    const auto addConvSource = graphCase.add.GetInDataNodesAndPortIndexs(0);
    const auto addReshapeSource = graphCase.add.GetInDataNodesAndPortIndexs(1);
    ASSERT_NE(reshapeBiasSource.first, nullptr);
    ASSERT_NE(reshapeShapeSource.first, nullptr);
    ASSERT_NE(addConvSource.first, nullptr);
    ASSERT_NE(addReshapeSource.first, nullptr);
    AscendString sourceName;
    ASSERT_EQ(reshapeBiasSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_bias_data").c_str());
    ASSERT_EQ(reshapeShapeSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_shape").c_str());
    ASSERT_EQ(addConvSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_conv3d").c_str());
    ASSERT_EQ(addReshapeSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_reshape").c_str());
}

struct AcceptancePatternConfig {
    std::string graphName;
    std::string prefix;
    std::string convType;
    std::string addType;
    std::string biasType;
    std::vector<int64_t> inputDims;
    std::vector<int64_t> filterDims;
    std::vector<int64_t> biasDims;
    std::vector<int64_t> outputDims;
    Format inputFormat;
    Format filterFormat;
    Format biasFormat;
    Format outputFormat;
    size_t filterElementCount;
    size_t biasElementCount;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::string dataFormat;
};

static void BuildAcceptancePatternSources(AcceptanceGraphCase& graphCase, const AcceptancePatternConfig& config,
                                          const TensorDesc& filterDesc, const TensorDesc& biasDesc)
{
    size_t filterByteCount = 0U;
    size_t biasByteCount = 0U;
    ASSERT_TRUE(TryMultiplySize(config.filterElementCount, sizeof(float), filterByteCount));
    ASSERT_TRUE(TryMultiplySize(config.biasElementCount, sizeof(float), biasByteCount));
    BuildAcceptanceSource(
        *graphCase.graph,
        {config.prefix + "_filter", "Const", filterDesc, std::vector<uint8_t>(filterByteCount, 0U), true},
        graphCase.filter);
    const bool biasHasValue = config.biasType == "Const";
    BuildAcceptanceSource(
        *graphCase.graph,
        {config.prefix + "_bias", config.biasType, biasDesc, std::vector<uint8_t>(biasByteCount, 0U), biasHasValue},
        graphCase.bias);
}

static void BuildAcceptancePatternCase(AcceptanceGraphCase& graphCase, const AcceptancePatternConfig& config)
{
    const TensorDesc filterDesc = MakeP0TensorDesc(config.filterDims, config.filterFormat, DT_FLOAT);
    const TensorDesc biasDesc = MakeP0TensorDesc(config.biasDims, config.biasFormat, DT_FLOAT);
    const TensorDesc outputDesc = MakeP0TensorDesc(config.outputDims, config.outputFormat, DT_FLOAT);
    graphCase.builder = std::make_unique<es::EsGraphBuilder>(config.graphName.c_str());
    auto x = graphCase.builder->CreateInput(0, (config.prefix + "_x").c_str(), DT_FLOAT, config.inputFormat,
                                            config.inputDims);
    SetAcceptanceInput(x, graphCase.x);
    graphCase.graph = graphCase.builder->GetCGraphBuilder()->GetGraph();
    ASSERT_NE(graphCase.graph, nullptr);
    BuildAcceptancePatternSources(graphCase, config, filterDesc, biasDesc);
    BuildAcceptanceConv(*graphCase.graph,
                        {config.prefix + "_conv",
                         config.convType,
                         graphCase.x,
                         {graphCase.filter, 0},
                         {},
                         false,
                         outputDesc,
                         config.strides,
                         config.pads,
                         config.dataFormat},
                        graphCase.conv);
    BuildAcceptanceBinary(*graphCase.graph,
                          {config.prefix + (config.addType == "Add" ? "_add" : "_biasadd"),
                           config.addType,
                           {graphCase.conv, 0},
                           {graphCase.bias, 0},
                           outputDesc},
                          graphCase.add);
    FinalizeAcceptanceGraph(*graphCase.builder, {graphCase.add}, graphCase.result);
}

static void VerifyAcceptancePatternCase(const AcceptanceGraphCase& graphCase, const std::string& prefix,
                                        const std::string& convType, const std::string& addType)
{
    int32_t convCount = 0;
    int32_t addCount = 0;
    GNode fusedConv;
    for (const auto& node : graphCase.result->GetAllNodes()) {
        AscendString type;
        ASSERT_EQ(node.GetType(type), GRAPH_SUCCESS);
        if (type == convType.c_str()) {
            ++convCount;
            fusedConv = node;
        } else if (type == addType.c_str()) {
            ++addCount;
        }
    }
    EXPECT_EQ(convCount, 1);
    EXPECT_EQ(addCount, 0);
    const auto xSource = fusedConv.GetInDataNodesAndPortIndexs(0);
    const auto filterSource = fusedConv.GetInDataNodesAndPortIndexs(1);
    const auto biasSource = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(xSource.first, nullptr);
    ASSERT_NE(filterSource.first, nullptr);
    ASSERT_NE(biasSource.first, nullptr);
    AscendString sourceName;
    ASSERT_EQ(xSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_x").c_str());
    ASSERT_EQ(filterSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_filter").c_str());
    ASSERT_EQ(biasSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_bias").c_str());
}

static void BuildAcceptanceRCase(AcceptanceGraphCase& graphCase, const std::string& graphName,
                                 const std::string& prefix, bool dualOutput)
{
    const std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    const TensorDesc filterDesc = MakeP0TensorDesc({4, 4, 1, 1, 1}, FORMAT_NCDHW, DT_FLOAT);
    const TensorDesc biasDesc = MakeP0TensorDesc({4}, FORMAT_ND, DT_FLOAT);
    const TensorDesc outputDesc = MakeP0TensorDesc({1, 4, 8, 8, 8}, FORMAT_ND, DT_FLOAT);
    graphCase.builder = std::make_unique<es::EsGraphBuilder>(graphName.c_str());
    auto x = graphCase.builder->CreateInput(0, (prefix + "_x").c_str(), DT_FLOAT, FORMAT_ND, dimsX);
    auto shape = graphCase.builder->CreateInput(1, (prefix + "_shape").c_str(), DT_INT64, FORMAT_ND, {5});
    SetAcceptanceInput(x, graphCase.x);
    SetAcceptanceInput(shape, graphCase.shapeInput);
    graphCase.graph = graphCase.builder->GetCGraphBuilder()->GetGraph();
    ASSERT_NE(graphCase.graph, nullptr);
    BuildAcceptanceSource(
        *graphCase.graph,
        {prefix + "_filter", "Const", filterDesc, std::vector<uint8_t>(16U * sizeof(float), 0U), true},
        graphCase.filter);
    BuildAcceptanceSource(*graphCase.graph,
                          {prefix + "_bias", "Const", biasDesc, std::vector<uint8_t>(4U * sizeof(float), 0U), true},
                          graphCase.bias);
    BuildAcceptanceConv(*graphCase.graph,
                        {prefix + "_conv",
                         "Conv3D",
                         graphCase.x,
                         {graphCase.filter, 0},
                         {},
                         false,
                         outputDesc,
                         {1, 1, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         "NCDHW"},
                        graphCase.conv);
    BuildAcceptanceReshape(*graphCase.graph,
                           {prefix + "_reshape", {graphCase.bias, 0}, graphCase.shapeInput, outputDesc},
                           graphCase.reshape);
    BuildAcceptanceBinary(*graphCase.graph,
                          {prefix + "_add", "Add", {graphCase.conv, 0}, {graphCase.reshape, 0}, outputDesc},
                          graphCase.add);
    if (dualOutput) {
        BuildAcceptanceUnary(*graphCase.graph, {prefix + "_identity", "Identity", {graphCase.reshape, 0}, outputDesc},
                             graphCase.identity);
        FinalizeAcceptanceGraph(*graphCase.builder, {graphCase.add, graphCase.identity}, graphCase.result);
    } else {
        FinalizeAcceptanceGraph(*graphCase.builder, {graphCase.add}, graphCase.result);
    }
}

static void VerifyAcceptanceRSingleCase(const AcceptanceGraphCase& graphCase, const std::string& prefix)
{
    int32_t convCount = 0;
    int32_t addCount = 0;
    int32_t reshapeCount = 0;
    GNode fusedConv;
    for (const auto& node : graphCase.result->GetAllNodes()) {
        AscendString type;
        ASSERT_EQ(node.GetType(type), GRAPH_SUCCESS);
        if (type == "Conv3D") {
            ++convCount;
            fusedConv = node;
        } else if (type == "Add") {
            ++addCount;
        } else if (type == "Reshape") {
            ++reshapeCount;
        }
    }
    EXPECT_EQ(convCount, 1);
    EXPECT_EQ(addCount, 0);
    EXPECT_EQ(reshapeCount, 0);
    const auto xSource = fusedConv.GetInDataNodesAndPortIndexs(0);
    const auto filterSource = fusedConv.GetInDataNodesAndPortIndexs(1);
    const auto biasSource = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(xSource.first, nullptr);
    ASSERT_NE(filterSource.first, nullptr);
    ASSERT_NE(biasSource.first, nullptr);
    AscendString sourceName;
    ASSERT_EQ(xSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_x").c_str());
    ASSERT_EQ(filterSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_filter").c_str());
    ASSERT_EQ(biasSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_bias").c_str());
}

static void VerifyAcceptanceRDualTopology(const AcceptanceGraphCase& graphCase, GNode& fusedConv, GNode& targetReshape)
{
    int32_t convCount = 0;
    int32_t addCount = 0;
    int32_t reshapeCount = 0;
    for (const auto& node : graphCase.result->GetAllNodes()) {
        AscendString type;
        ASSERT_EQ(node.GetType(type), GRAPH_SUCCESS);
        if (type == "Conv3D") {
            ++convCount;
            fusedConv = node;
        } else if (type == "Add") {
            ++addCount;
        } else if (type == "Reshape") {
            ++reshapeCount;
            targetReshape = node;
        }
    }
    EXPECT_EQ(convCount, 1);
    EXPECT_EQ(addCount, 0);
    EXPECT_EQ(reshapeCount, 1);
}

static void VerifyAcceptanceRDualSources(const AcceptanceGraphCase& graphCase, const std::string& prefix,
                                         const GNode& fusedConv, const GNode& targetReshape)
{
    const auto convBiasSource = fusedConv.GetInDataNodesAndPortIndexs(2);
    const auto reshapeBiasSource = targetReshape.GetInDataNodesAndPortIndexs(0);
    const auto reshapeShapeSource = targetReshape.GetInDataNodesAndPortIndexs(1);
    const auto identitySource = graphCase.identity.GetInDataNodesAndPortIndexs(0);
    ASSERT_NE(convBiasSource.first, nullptr);
    ASSERT_NE(reshapeBiasSource.first, nullptr);
    ASSERT_NE(reshapeShapeSource.first, nullptr);
    ASSERT_NE(identitySource.first, nullptr);
    AscendString sourceName;
    ASSERT_EQ(convBiasSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_bias").c_str());
    ASSERT_EQ(reshapeBiasSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_bias").c_str());
    ASSERT_EQ(reshapeShapeSource.first->GetName(sourceName), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceName.GetString(), (prefix + "_shape").c_str());
    AscendString sourceType;
    ASSERT_EQ(identitySource.first->GetType(sourceType), GRAPH_SUCCESS);
    EXPECT_STREQ(sourceType.GetString(), "Reshape");
}

class BiasaddConvFusionAcceptanceTest : public testing::Test {
protected:
    void SetUp() override
    {
        SaveGlobalPlatformState();
        PlatformInfoManager::Instance().platform_info_map_.clear();
        OptionalInfo emptyOptionalInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(emptyOptionalInfo);
        PlatformInfo platformInfo;
        OptionalInfo optionalInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
        optionalInfo.soc_version = "Ascend910B";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
    }

    void TearDown() override { RestoreGlobalPlatformState(); }
};

class BiasaddConvFusionAcceptanceNonFixpipeTest : public testing::Test {
protected:
    void SetUp() override
    {
        SaveGlobalPlatformState();
        PlatformInfoManager::Instance().platform_info_map_.clear();
        OptionalInfo emptyOptionalInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(emptyOptionalInfo);
        PlatformInfo platformInfo;
        OptionalInfo optionalInfo;
        platformInfo.soc_info.ai_core_cnt = 32;
        platformInfo.str_info.short_soc_version = "Ascend910A";
        optionalInfo.soc_version = "Ascend910A";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
    }

    void TearDown() override { RestoreGlobalPlatformState(); }
};

TEST_F(BiasaddConvFusionAcceptanceNonFixpipeTest, acceptanceLegacyConstBiasNonFixpipeFuses)
{
    AcceptanceGraphCase graphCase;
    BuildLegacyConstBiasCase(graphCase, "acceptance_l01_const_nonfixpipe", "acceptance_l01");
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, SUCCESS);
    VerifyLegacyConstBiasTopology(graphCase, "acceptance_l01");
    VerifyLegacyConstBiasConsumersAndAttrs(graphCase);
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptanceLegacyConstBiasFixpipeFuses)
{
    AcceptanceGraphCase graphCase;
    BuildLegacyConstBiasCase(graphCase, "acceptance_l02_const_fixpipe", "acceptance_l02");
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, SUCCESS);
    VerifyLegacyConstBiasTopology(graphCase, "acceptance_l02");
    VerifyLegacyConstBiasConsumersAndAttrs(graphCase);
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptanceLegacyVariableAddDoesNotFuse)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptance2DBinaryCase(graphCase, {"acceptance_l03_variable_add",
                                            "acceptance_l03",
                                            "Variable",
                                            false,
                                            "Variable",
                                            false,
                                            false,
                                            "Add",
                                            {64},
                                            FORMAT_NHWC,
                                            64U});
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, GRAPH_NOT_CHANGED);
    VerifyUnfused2DBinaryCase(graphCase, "acceptance_l03", "Add", "acceptance_l03_bias");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptanceLegacyConvWithBiasDoesNotFuse)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptanceConvWithBiasCase(graphCase, "acceptance_l04_conv_with_bias", "acceptance_l04");
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, GRAPH_NOT_CHANGED);
    VerifyAcceptanceConvWithBiasCase(graphCase, "acceptance_l04");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptanceLegacyReshapeDataBiasDoesNotFuse)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptanceLegacyReshapeDataCase(graphCase, "acceptance_l05_r_data_bias", "acceptance_l05");
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, GRAPH_NOT_CHANGED);
    VerifyAcceptanceLegacyReshapeDataCase(graphCase, "acceptance_l05");
}

TEST_F(BiasaddConvFusionAcceptanceNonFixpipeTest, acceptanceLegacyDirectDataNonFixpipeDoesNotFuse)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptance2DBinaryCase(graphCase, {"acceptance_l06_data_nonfixpipe",
                                            "acceptance_l06",
                                            "Const",
                                            true,
                                            "Data",
                                            false,
                                            true,
                                            "BiasAdd",
                                            {64},
                                            FORMAT_ND,
                                            64U});
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, GRAPH_NOT_CHANGED);
    VerifyUnfused2DBinaryCase(graphCase, "acceptance_l06", "BiasAdd", "acceptance_l06_bias_data");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptanceLegacyDirectDataFixpipeFuses)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptance2DBinaryCase(graphCase, {"acceptance_l07_data_fixpipe",
                                            "acceptance_l07",
                                            "Const",
                                            true,
                                            "Data",
                                            false,
                                            true,
                                            "BiasAdd",
                                            {64},
                                            FORMAT_ND,
                                            64U});
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, SUCCESS);
    VerifyFused2DBinaryCase(graphCase, "acceptance_l07", "acceptance_l07_bias_data", "BiasAdd");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptanceLegacyConstCoutMismatchDoesNotFuse)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptance2DBinaryCase(graphCase, {"acceptance_l08_cout_mismatch",
                                            "acceptance_l08",
                                            "Const",
                                            true,
                                            "Const",
                                            true,
                                            false,
                                            "Add",
                                            {1},
                                            FORMAT_ND,
                                            1U});
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, GRAPH_NOT_CHANGED);
    VerifyUnfused2DBinaryCase(graphCase, "acceptance_l08", "Add", "acceptance_l08_bias");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptancePatternConv2dAddConstFuses)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptancePatternCase(graphCase, {"acceptance_p01_conv2d_add",
                                           "acceptance_p01",
                                           "Conv2D",
                                           "Add",
                                           "Const",
                                           {1, 56, 56, 64},
                                           {64, 1, 1, 64},
                                           {64},
                                           {1, 28, 28, 64},
                                           FORMAT_NHWC,
                                           FORMAT_NHWC,
                                           FORMAT_NHWC,
                                           FORMAT_NHWC,
                                           4096U,
                                           64U,
                                           {1, 2, 2, 1},
                                           {1, 1, 1, 1},
                                           "NHWC"});
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, SUCCESS);
    VerifyAcceptancePatternCase(graphCase, "acceptance_p01", "Conv2D", "Add");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptancePatternDepthwiseAddConstFuses)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptancePatternCase(graphCase, {"acceptance_p02_depthwise_add",
                                           "acceptance_p02",
                                           "DepthwiseConv2D",
                                           "Add",
                                           "Const",
                                           {1, 56, 56, 64},
                                           {64, 1, 1, 1},
                                           {64},
                                           {1, 56, 56, 64},
                                           FORMAT_NHWC,
                                           FORMAT_NHWC,
                                           FORMAT_NHWC,
                                           FORMAT_NHWC,
                                           64U,
                                           64U,
                                           {1, 1, 1, 1},
                                           {1, 1, 1, 1},
                                           "NHWC"});
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, SUCCESS);
    VerifyAcceptancePatternCase(graphCase, "acceptance_p02", "DepthwiseConv2D", "Add");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptancePatternDepthwiseBiasAddVariableFuses)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptancePatternCase(graphCase, {"acceptance_p03_depthwise_biasadd_variable",
                                           "acceptance_p03",
                                           "DepthwiseConv2D",
                                           "BiasAdd",
                                           "Variable",
                                           {1, 56, 56, 64},
                                           {64, 1, 1, 1},
                                           {64},
                                           {1, 56, 56, 64},
                                           FORMAT_NHWC,
                                           FORMAT_NHWC,
                                           FORMAT_ND,
                                           FORMAT_NHWC,
                                           64U,
                                           64U,
                                           {1, 1, 1, 1},
                                           {1, 1, 1, 1},
                                           "NHWC"});
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, SUCCESS);
    VerifyAcceptancePatternCase(graphCase, "acceptance_p03", "DepthwiseConv2D", "BiasAdd");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptancePatternConv3dAddConstFuses)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptancePatternCase(graphCase, {"acceptance_p04_conv3d_add",
                                           "acceptance_p04",
                                           "Conv3D",
                                           "Add",
                                           "Const",
                                           {4, 16, 64, 64, 64},
                                           {4, 16, 1, 1, 1},
                                           {4},
                                           {4, 4, 64, 64, 64},
                                           FORMAT_ND,
                                           FORMAT_NCDHW,
                                           FORMAT_ND,
                                           FORMAT_NCDHW,
                                           64U,
                                           4U,
                                           {1, 1, 1, 1, 1},
                                           {0, 0, 0, 0, 0, 0},
                                           "NCDHW"});
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, SUCCESS);
    VerifyAcceptancePatternCase(graphCase, "acceptance_p04", "Conv3D", "Add");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptancePatternConv3dBiasAddConstFuses)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptancePatternCase(graphCase, {"acceptance_p05_conv3d_biasadd",
                                           "acceptance_p05",
                                           "Conv3D",
                                           "BiasAdd",
                                           "Const",
                                           {4, 16, 64, 64, 64},
                                           {4, 16, 1, 1, 1},
                                           {4},
                                           {4, 4, 64, 64, 64},
                                           FORMAT_ND,
                                           FORMAT_NCDHW,
                                           FORMAT_ND,
                                           FORMAT_NCDHW,
                                           64U,
                                           4U,
                                           {1, 1, 1, 1, 1},
                                           {0, 0, 0, 0, 0, 0},
                                           "NCDHW"});
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, SUCCESS);
    VerifyAcceptancePatternCase(graphCase, "acceptance_p05", "Conv3D", "BiasAdd");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptancePatternRSingleFuses)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptanceRCase(graphCase, "acceptance_p06_r_single_output", "acceptance_p06", false);
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, SUCCESS);
    VerifyAcceptanceRSingleCase(graphCase, "acceptance_p06");
}

TEST_F(BiasaddConvFusionAcceptanceTest, acceptancePatternRDoubleFusesAndKeepsBypass)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptanceRCase(graphCase, "acceptance_p07_r_dual_output", "acceptance_p07", true);
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, SUCCESS);
    GNode fusedConv;
    GNode targetReshape;
    VerifyAcceptanceRDualTopology(graphCase, fusedConv, targetReshape);
    VerifyAcceptanceRDualSources(graphCase, "acceptance_p07", fusedConv, targetReshape);
}

TEST_F(BiasaddConvFusionAcceptanceNonFixpipeTest, acceptanceLegacyVariableAddNonFixpipeDoesNotFuse)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptance2DBinaryCase(graphCase, {"acceptance_l09_variable_add",
                                            "acceptance_l09",
                                            "Variable",
                                            false,
                                            "Variable",
                                            false,
                                            false,
                                            "Add",
                                            {64},
                                            FORMAT_NHWC,
                                            64U});
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, GRAPH_NOT_CHANGED);
    VerifyUnfused2DBinaryCase(graphCase, "acceptance_l09", "Add", "acceptance_l09_bias");
}

TEST_F(BiasaddConvFusionAcceptanceNonFixpipeTest, acceptanceLegacyConvWithBiasNonFixpipeDoesNotFuse)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptanceConvWithBiasCase(graphCase, "acceptance_l10_conv_with_bias", "acceptance_l10");
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, GRAPH_NOT_CHANGED);
    VerifyAcceptanceConvWithBiasCase(graphCase, "acceptance_l10");
}

TEST_F(BiasaddConvFusionAcceptanceNonFixpipeTest, acceptanceLegacyReshapeDataBiasNonFixpipeDoesNotFuse)
{
    AcceptanceGraphCase graphCase;
    BuildAcceptanceLegacyReshapeDataCase(graphCase, "acceptance_l11_r_data_bias", "acceptance_l11");
    ASSERT_NE(graphCase.result, nullptr);
    RunAcceptancePass(graphCase.result, GRAPH_NOT_CHANGED);
    VerifyAcceptanceLegacyReshapeDataCase(graphCase, "acceptance_l11");
}

TEST_F(AABiasaddConvFusionTest, p0FmapDescMismatchFusesWithProducerEdgeKept)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_fmap_desc_mismatch");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto absOut = BuildUnaryNode(graphBuilder, x, {"Abs_0", "Abs", dimsX, FORMAT_NHWC, DT_FLOAT});
    auto convOutput = BuildConvNode(graphBuilder, "my_conv", absOut, filter,
                                    {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    SetInputDesc(*convNode, 0, dimsX, FORMAT_NHWC, DT_FLOAT16);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "Cast"), 0);
    EXPECT_EQ(CountNodes(graph, "TransData"), 0);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    AscendString fusedName;
    ASSERT_EQ(fusedConv.GetName(fusedName), GRAPH_SUCCESS);
    EXPECT_STREQ(fusedName.GetString(), "my_conv");
    const auto xSrc = fusedConv.GetInDataNodesAndPortIndexs(0);
    ASSERT_NE(xSrc.first, nullptr);
    AscendString xSrcName;
    ASSERT_EQ(xSrc.first->GetName(xSrcName), GRAPH_SUCCESS);
    EXPECT_STREQ(xSrcName.GetString(), "Abs_0");

    TensorDesc xDescAfter;
    ASSERT_EQ(fusedConv.GetInputDesc(0, xDescAfter), GRAPH_SUCCESS);
    EXPECT_EQ(xDescAfter.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(xDescAfter.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(xDescAfter.GetShape().GetDims(), dimsX);

    TensorDesc filterDescAfter;
    ASSERT_EQ(fusedConv.GetInputDesc(1, filterDescAfter), GRAPH_SUCCESS);
    EXPECT_EQ(filterDescAfter.GetDataType(), DT_FLOAT);

    const auto biasSrc = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(biasSrc.first, nullptr);
    AscendString biasSrcName;
    ASSERT_EQ(biasSrc.first->GetName(biasSrcName), GRAPH_SUCCESS);
    EXPECT_STREQ(biasSrcName.GetString(), "bias_const");
    bool hasBias = false;
    ASSERT_EQ(fusedConv.GetAttr(HAS_BIAS_ATTR, hasBias), GRAPH_SUCCESS);
    EXPECT_TRUE(hasBias);

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0FusedConvOutputDirectlyFeedsAddConsumers)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_add_multi_consumer_adjacency");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filter, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, bias);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto consumer1 = BuildUnaryNode(graphBuilder, addOutput,
                                    {"requant_like_1", "Identity", dimsOut, FORMAT_NHWC, DT_FLOAT});
    auto consumer2 = BuildUnaryNode(graphBuilder, addOutput,
                                    {"requant_like_2", "Identity", dimsOut, FORMAT_NHWC, DT_FLOAT});

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({consumer1, consumer2});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Cast"), 0);
    EXPECT_EQ(CountNodes(graph, "TransData"), 0);

    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv2D", fusedConv));
    AscendString fusedName;
    ASSERT_EQ(fusedConv.GetName(fusedName), GRAPH_SUCCESS);
    const std::string fusedNameStr = fusedName.GetString();
    const auto consumers = fusedConv.GetOutDataNodesAndPortIndexs(0);
    ASSERT_EQ(consumers.size(), 2U);
    for (const auto& consumer : consumers) {
        AscendString consumerName;
        ASSERT_EQ(consumer.first->GetName(consumerName), GRAPH_SUCCESS);
        const std::string nameStr = consumerName.GetString();
        EXPECT_TRUE(nameStr == "requant_like_1" || nameStr == "requant_like_2");
        const auto consumerInputs = consumer.first->GetInDataNodesAndPortIndexs(0);
        ASSERT_NE(consumerInputs.first, nullptr);
        AscendString consumerSrcName;
        ASSERT_EQ(consumerInputs.first->GetName(consumerSrcName), GRAPH_SUCCESS);
        EXPECT_STREQ(consumerSrcName.GetString(), fusedNameStr.c_str());
    }

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RDualOutputReshapeInputsFollowProducerDescs)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{4, 16, 64, 64, 64};
    std::vector<int64_t> dimsFilter{4, 16, 1, 1, 1};
    std::vector<int64_t> dimsBias{4};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{4, 4, 64, 64, 64};

    auto graphBuilder = es::EsGraphBuilder("p0_r_dual_producer_descs");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto filter = graphBuilder.CreateInput(1, "filter", DT_FLOAT, FORMAT_NCDHW, dimsFilter);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_ND, DT_FLOAT16);
    auto shape = graphBuilder.CreateInput(2, "shape_data", DT_INT64, FORMAT_ND, dimsShape);

    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    auto reshapeNode = reshapeOutput.GetProducer();
    SetInputDesc(*reshapeNode, 0, dimsBias, FORMAT_ND, DT_FLOAT16);
    SetInputDesc(*reshapeNode, 1, dimsShape, FORMAT_ND, DT_INT64);
    SetOutputDesc(*reshapeNode, 0, dimsOut, FORMAT_ND, DT_FLOAT16);

    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    auto addNode = addOutput.GetProducer();
    SetOutputDesc(*addNode, 0, dimsOut, FORMAT_ND, DT_FLOAT);

    const RConsumerOutputs consumers = BuildRConsumers(graphBuilder, reshapeOutput, bias, shape,
                                                       {dimsOut, dimsBias, dimsShape});

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(
        {addOutput, consumers.reshape, consumers.bias, consumers.shape});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    EXPECT_EQ(CountNodes(graph, "Cast"), 0);
    EXPECT_EQ(CountNodes(graph, "TransData"), 0);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);

    GNode targetReshape;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Reshape", targetReshape));
    AscendString targetReshapeName;
    ASSERT_EQ(targetReshape.GetName(targetReshapeName), GRAPH_SUCCESS);
    EXPECT_STREQ(targetReshapeName.GetString(), "reshape");
    const auto reshapeBiasSrc = targetReshape.GetInDataNodesAndPortIndexs(0);
    ASSERT_NE(reshapeBiasSrc.first, nullptr);
    AscendString reshapeBiasSrcName;
    ASSERT_EQ(reshapeBiasSrc.first->GetName(reshapeBiasSrcName), GRAPH_SUCCESS);
    EXPECT_STREQ(reshapeBiasSrcName.GetString(), "bias_const");

    TensorDesc reshapeBiasDesc;
    ASSERT_EQ(targetReshape.GetInputDesc(0, reshapeBiasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(reshapeBiasDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(reshapeBiasDesc.GetShape().GetDims(), dimsBias);
    TensorDesc reshapeShapeDesc;
    ASSERT_EQ(targetReshape.GetInputDesc(1, reshapeShapeDesc), GRAPH_SUCCESS);
    EXPECT_EQ(reshapeShapeDesc.GetDataType(), DT_INT64);
    EXPECT_EQ(reshapeShapeDesc.GetShape().GetDims(), dimsShape);
    TensorDesc reshapeOutputDesc;
    ASSERT_EQ(targetReshape.GetOutputDesc(0, reshapeOutputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(reshapeOutputDesc.GetDataType(), DT_FLOAT16);

    GNode targetConv;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Conv3D", targetConv));
    AscendString targetConvName;
    ASSERT_EQ(targetConv.GetName(targetConvName), GRAPH_SUCCESS);
    EXPECT_STREQ(targetConvName.GetString(), "conv3d");
    const auto convBiasSrc = targetConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(convBiasSrc.first, nullptr);
    AscendString convBiasSrcName;
    ASSERT_EQ(convBiasSrc.first->GetName(convBiasSrcName), GRAPH_SUCCESS);
    EXPECT_STREQ(convBiasSrcName.GetString(), "bias_const");

    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeCoutMismatchOnSpatialAxisMustNotFuse)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{8};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};
    auto graphBuilder = es::EsGraphBuilder("p0_r_cout_mismatch_waxis");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCDHW, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_waxis", dimsBias, FORMAT_ND, DT_FLOAT);
    auto shape = graphBuilder.CreateInput(1, "shape", DT_INT64, FORMAT_ND, dimsShape);
    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);
    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    SetOutputDesc(*reshapeOutput.GetProducer(), 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    EXPECT_EQ(CountNodes(graph, "Conv3D"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0RModeCoutMismatchNoAxisMatchMustNotFuse)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 4, 8, 8, 8};
    std::vector<int64_t> dimsFilter{4, 4, 1, 1, 1};
    std::vector<int64_t> dimsBias{3};
    std::vector<int64_t> dimsShape{5};
    std::vector<int64_t> dimsOut{1, 4, 8, 8, 8};
    auto graphBuilder = es::EsGraphBuilder("p0_r_cout_mismatch_none");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCDHW, dimsX);
    auto filter = BuildConstNodeWithValue(graphBuilder, "filter", dimsFilter, FORMAT_NCDHW, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_none", dimsBias, FORMAT_ND, DT_FLOAT);
    auto shape = graphBuilder.CreateInput(1, "shape", DT_INT64, FORMAT_ND, dimsShape);
    auto convOutput = BuildConvNode(graphBuilder, "conv3d", x, filter,
                                    {"Conv3D", {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, "NCDHW"});
    SetOutputDesc(*convOutput.GetProducer(), 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);
    auto reshapeOutput = BuildReshapeNode(graphBuilder, "reshape", bias, shape);
    SetOutputDesc(*reshapeOutput.GetProducer(), 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);
    auto addOutput = BuildAddNode(graphBuilder, "add", convOutput, reshapeOutput);
    SetOutputDesc(*addOutput.GetProducer(), 0, dimsOut, FORMAT_NCDHW, DT_FLOAT);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({addOutput});
    RunPass(graph, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0SubgraphDataParentConstConv3DMustFuse)
{
    SetUpFixpipePlatform();
    const P0GraphDescs descs{MakeP0TensorDesc({1, 4, 8, 8, 8}, FORMAT_NCDHW, DT_FLOAT),
                             MakeP0TensorDesc({4, 4, 1, 1, 1}, FORMAT_NCDHW, DT_FLOAT),
                             MakeP0TensorDesc({4}, FORMAT_ND, DT_FLOAT), TensorDesc(),
                             MakeP0TensorDesc({1, 4, 8, 8, 8}, FORMAT_NCDHW, DT_FLOAT)};
    auto parentGraph = std::make_shared<Graph>("p0_subg_const_parent");
    GNode parentNode;
    ASSERT_TRUE(AddP0ParentConstCall(*parentGraph, descs.bias, {{4}, "Const", true}, parentNode));
    Graph childGraph("p0_child_graph");
    GNode biasNode = AddP0SourceNode(childGraph, "bias_data", "Data", descs.bias);
    int64_t biasParentIndex = 0;
    biasNode.SetAttr("_parent_node_index", biasParentIndex);
    GNode convNode;
    ASSERT_TRUE(AddP0Conv3DNode(childGraph, descs, convNode));
    GNode biasAddNode;
    es::CompliantNodeBuilder::AnyTypeOperator biasAddOp("biasadd", "BiasAdd");
    biasAddOp.InputRegister("x");
    biasAddOp.InputRegister("bias");
    biasAddOp.OutputRegister("y");
    biasAddOp.UpdateInputDesc(0U, descs.output);
    biasAddOp.UpdateInputDesc(1U, descs.bias);
    biasAddOp.UpdateOutputDesc(0U, descs.output);
    biasAddNode = childGraph.AddNodeByOp(biasAddOp);
    ASSERT_EQ(childGraph.AddDataEdge(convNode, 0, biasAddNode, 0), GRAPH_SUCCESS);
    ASSERT_EQ(childGraph.AddDataEdge(biasNode, 0, biasAddNode, 1), GRAPH_SUCCESS);
    ASSERT_EQ(childGraph.SetOutputs({{biasAddNode, 0}}), GRAPH_SUCCESS);
    std::shared_ptr<Graph> attachedChild;
    ASSERT_TRUE(AttachP0ChildGraph(parentNode, childGraph, attachedChild));
    ASSERT_NE(attachedChild->GetParentNode(), nullptr);

    RunPass(attachedChild, SUCCESS);
    EXPECT_EQ(CountNodes(attachedChild, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(attachedChild, "Conv3D"), 1);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(attachedChild, "Conv3D", fusedConv));
    const auto biasSrc = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(biasSrc.first, nullptr);
    AscendString biasSrcName;
    ASSERT_EQ(biasSrc.first->GetName(biasSrcName), GRAPH_SUCCESS);
    EXPECT_STREQ(biasSrcName.GetString(), "bias_data");
    bool hasBias = false;
    ASSERT_EQ(fusedConv.GetAttr(HAS_BIAS_ATTR, hasBias), GRAPH_SUCCESS);
    EXPECT_TRUE(hasBias);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0SubgraphDataParentNonConstConv3DMustNotFuse)
{
    SetUpFixpipePlatform();
    const P0GraphDescs descs{MakeP0TensorDesc({1, 4, 8, 8, 8}, FORMAT_NCDHW, DT_FLOAT),
                             MakeP0TensorDesc({4, 4, 1, 1, 1}, FORMAT_NCDHW, DT_FLOAT),
                             MakeP0TensorDesc({4}, FORMAT_ND, DT_FLOAT), TensorDesc(),
                             MakeP0TensorDesc({1, 4, 8, 8, 8}, FORMAT_NCDHW, DT_FLOAT)};

    auto parentGraph = std::make_shared<Graph>("p0_subg_plain_parent");
    GNode parentBias;
    parentBias = AddP0SourceNode(*parentGraph, "parent_bias_data", "Data", descs.bias);
    es::CompliantNodeBuilder::AnyTypeOperator pcOp("pc", "PartitionedCall");
    pcOp.DynamicInputRegister("args", 1);
    pcOp.DynamicOutputRegister("output", 1);
    pcOp.SubgraphRegister("f", false);
    pcOp.SubgraphCountRegister("f", 1);
    pcOp.UpdateInputDesc(0U, descs.bias);
    pcOp.UpdateOutputDesc(0U, descs.output);
    pcOp.SetSubgraphInstanceName(0U, "p0_child_graph");
    GNode parentNode = parentGraph->AddNodeByOp(pcOp);
    ASSERT_EQ(parentGraph->AddDataEdge(parentBias, 0, parentNode, 0), GRAPH_SUCCESS);

    Graph childGraph("p0_child_graph");
    GNode biasNode = AddP0SourceNode(childGraph, "bias_data", "Data", descs.bias);
    int64_t biasParentIndex = 0;
    biasNode.SetAttr("_parent_node_index", biasParentIndex);
    GNode convNode;
    ASSERT_TRUE(AddP0Conv3DNode(childGraph, descs, convNode));
    es::CompliantNodeBuilder::AnyTypeOperator biasAddOp("biasadd", "BiasAdd");
    biasAddOp.InputRegister("x");
    biasAddOp.InputRegister("bias");
    biasAddOp.OutputRegister("y");
    biasAddOp.UpdateInputDesc(0U, descs.output);
    biasAddOp.UpdateInputDesc(1U, descs.bias);
    biasAddOp.UpdateOutputDesc(0U, descs.output);
    GNode biasAddNode = childGraph.AddNodeByOp(biasAddOp);
    ASSERT_EQ(childGraph.AddDataEdge(convNode, 0, biasAddNode, 0), GRAPH_SUCCESS);
    ASSERT_EQ(childGraph.AddDataEdge(biasNode, 0, biasAddNode, 1), GRAPH_SUCCESS);
    ASSERT_EQ(childGraph.SetOutputs({{biasAddNode, 0}}), GRAPH_SUCCESS);
    std::shared_ptr<Graph> attachedChild;
    ASSERT_TRUE(AttachP0ChildGraph(parentNode, childGraph, attachedChild));
    ASSERT_NE(attachedChild->GetParentNode(), nullptr);

    RunPass(attachedChild, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(attachedChild, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(attachedChild, "Conv3D"), 1);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0NestedSubgraphDataParentConstConv3DMustFuse)
{
    SetUpFixpipePlatform();
    const P0GraphDescs descs{MakeP0TensorDesc({1, 4, 8, 8, 8}, FORMAT_NCDHW, DT_FLOAT),
                             MakeP0TensorDesc({4, 4, 1, 1, 1}, FORMAT_NCDHW, DT_FLOAT),
                             MakeP0TensorDesc({4}, FORMAT_ND, DT_FLOAT), TensorDesc(),
                             MakeP0TensorDesc({1, 4, 8, 8, 8}, FORMAT_NCDHW, DT_FLOAT)};

    auto outerGraph = std::make_shared<Graph>("p0_nested_outer");
    GNode outerConst;
    ASSERT_TRUE(AddP0ParentConstCall(*outerGraph, descs.bias, {{4}, "Const", true}, outerConst));

    Graph middleGraph("p0_middle");
    GNode middleBias = AddP0SourceNode(middleGraph, "bias_mid", "Data", descs.bias);
    int64_t midParentIndex = 0;
    middleBias.SetAttr("_parent_node_index", midParentIndex);
    es::CompliantNodeBuilder::AnyTypeOperator innerPcOp("inner_pc", "PartitionedCall");
    innerPcOp.DynamicInputRegister("args", 1);
    innerPcOp.DynamicOutputRegister("output", 1);
    innerPcOp.SubgraphRegister("g", false);
    innerPcOp.SubgraphCountRegister("g", 1);
    innerPcOp.UpdateInputDesc(0U, descs.bias);
    innerPcOp.UpdateOutputDesc(0U, descs.output);
    innerPcOp.SetSubgraphInstanceName(0U, "p0_inner");
    GNode innerPc = middleGraph.AddNodeByOp(innerPcOp);
    ASSERT_EQ(middleGraph.AddDataEdge(middleBias, 0, innerPc, 0), GRAPH_SUCCESS);
    ASSERT_EQ(middleGraph.SetOutputs({{innerPc, 0}}), GRAPH_SUCCESS);

    Graph innerGraph("p0_inner");
    GNode biasNode = AddP0SourceNode(innerGraph, "bias_data", "Data", descs.bias);
    int64_t biasParentIndex = 0;
    biasNode.SetAttr("_parent_node_index", biasParentIndex);
    GNode convNode;
    ASSERT_TRUE(AddP0Conv3DNode(innerGraph, descs, convNode));
    es::CompliantNodeBuilder::AnyTypeOperator biasAddOp("biasadd", "BiasAdd");
    biasAddOp.InputRegister("x");
    biasAddOp.InputRegister("bias");
    biasAddOp.OutputRegister("y");
    biasAddOp.UpdateInputDesc(0U, descs.output);
    biasAddOp.UpdateInputDesc(1U, descs.bias);
    biasAddOp.UpdateOutputDesc(0U, descs.output);
    GNode biasAddNode = innerGraph.AddNodeByOp(biasAddOp);
    ASSERT_EQ(innerGraph.AddDataEdge(convNode, 0, biasAddNode, 0), GRAPH_SUCCESS);
    ASSERT_EQ(innerGraph.AddDataEdge(biasNode, 0, biasAddNode, 1), GRAPH_SUCCESS);
    ASSERT_EQ(innerGraph.SetOutputs({{biasAddNode, 0}}), GRAPH_SUCCESS);
    std::shared_ptr<Graph> attachedInner;
    ASSERT_EQ(innerPc.SetSubgraph("g", innerGraph), GRAPH_SUCCESS);
    ASSERT_EQ(innerPc.GetSubgraph(0, attachedInner), GRAPH_SUCCESS);
    std::shared_ptr<Graph> attachedMiddle;
    ASSERT_TRUE(AttachP0ChildGraph(outerConst, middleGraph, attachedMiddle));

    RunPass(attachedInner, SUCCESS);
    EXPECT_EQ(CountNodes(attachedInner, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(attachedInner, "Conv3D"), 1);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(attachedInner, "Conv3D", fusedConv));
    const auto biasSrc = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(biasSrc.first, nullptr);
    AscendString biasSrcName;
    ASSERT_EQ(biasSrc.first->GetName(biasSrcName), GRAPH_SUCCESS);
    EXPECT_STREQ(biasSrcName.GetString(), "bias_data");
    bool hasBias = false;
    ASSERT_EQ(fusedConv.GetAttr(HAS_BIAS_ATTR, hasBias), GRAPH_SUCCESS);
    EXPECT_TRUE(hasBias);
    ClearPlatform();
}

TEST_F(AABiasaddConvFusionTest, p0SubgraphDataParentConstConv2dNoFixpipeMustFuse)
{
    SetUpNonFixpipePlatform();
    const P0GraphDescs descs{MakeP0TensorDesc({1, 56, 56, 64}, FORMAT_NHWC, DT_FLOAT),
                             MakeP0TensorDesc({64, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT),
                             MakeP0TensorDesc({64}, FORMAT_NHWC, DT_FLOAT), TensorDesc(),
                             MakeP0TensorDesc({1, 28, 28, 64}, FORMAT_NHWC, DT_FLOAT)};
    auto parentGraph = std::make_shared<Graph>("p0_subg_c2d_parent");
    GNode parentNode;
    ASSERT_TRUE(AddP0ParentConstCall(*parentGraph, descs.bias, {{64}, "Const", true}, parentNode));
    Graph childGraph("p0_child_graph");
    GNode biasNode = AddP0SourceNode(childGraph, "bias_data", "Data", descs.bias);
    int64_t biasParentIndex = 0;
    biasNode.SetAttr("_parent_node_index", biasParentIndex);
    GNode convNode;
    GNode addNode;
    ASSERT_TRUE(AddP0Conv2DNode(childGraph, descs, convNode) &&
                AddP0AddNode(childGraph, descs, convNode, biasNode, addNode));
    ASSERT_EQ(childGraph.SetOutputs({{addNode, 0}}), GRAPH_SUCCESS);
    std::shared_ptr<Graph> attachedChild;
    ASSERT_TRUE(AttachP0ChildGraph(parentNode, childGraph, attachedChild));

    RunPass(attachedChild, SUCCESS);
    EXPECT_EQ(CountNodes(attachedChild, "Add"), 0);
    EXPECT_EQ(CountNodes(attachedChild, "Conv2D"), 1);
    GNode fusedConv;
    ASSERT_TRUE(FindFirstNodeByType(attachedChild, "Conv2D", fusedConv));
    const auto biasSrc = fusedConv.GetInDataNodesAndPortIndexs(2);
    ASSERT_NE(biasSrc.first, nullptr);
    AscendString biasSrcName;
    ASSERT_EQ(biasSrc.first->GetName(biasSrcName), GRAPH_SUCCESS);
    EXPECT_STREQ(biasSrcName.GetString(), "bias_data");
    ClearPlatform();
}

class OrderPeerBiasaddConvFusion : public AABiasaddConvFusion {
public:
    void OrderMatchedNodesForTest(std::vector<GNode>& matchedNodes) { OrderMatchedNodes(matchedNodes); }
};

static bool FindNodeByName(const std::shared_ptr<Graph>& graph, const std::string& name, GNode& result)
{
    for (auto& node : graph->GetDirectNode()) {
        AscendString curName;
        if (node.GetName(curName) == GRAPH_SUCCESS && curName.GetString() != nullptr && name == curName.GetString()) {
            result = node;
            return true;
        }
    }
    return false;
}

static std::vector<std::string> GetMatchedNodeNamesInOrder(const std::shared_ptr<Graph>& graph)
{
    std::vector<GNode> matchedNodes;
    ConvFusionUtils::ConvFusionUtilsPass::GetMatchedNodes(graph, matchedNodes, {AscendString("Conv2D")});
    OrderPeerBiasaddConvFusion peer;
    peer.OrderMatchedNodesForTest(matchedNodes);
    std::vector<std::string> names;
    for (auto& node : matchedNodes) {
        AscendString name;
        if (node.GetName(name) == GRAPH_SUCCESS && name.GetString() != nullptr) {
            names.emplace_back(name.GetString());
        }
    }
    return names;
}

TEST_F(AABiasaddConvFusionTest, legacyOrderTwoBranchesSortedByConsumerName)
{
    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("order_two_branches");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    // Graph-node-order first branch whose consumer name sorts later: convolution_1 -> BiasAdd_1.
    auto filter1 = BuildConstNode(graphBuilder, "filter_1", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias1 = BuildConstNodeWithValue(graphBuilder, "bias_1", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto conv1Out = BuildConvNode(graphBuilder, "convolution_1", x, filter1,
                                  {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    SetOutputDesc(*conv1Out.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto biasadd1Out = BuildBiasAddNode(graphBuilder, "BiasAdd_1", conv1Out, bias1);
    SetOutputDesc(*biasadd1Out.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    // Graph-node-order second branch whose consumer name sorts first: convolution -> BiasAdd.
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto convOut = BuildConvNode(graphBuilder, "convolution", x, filter,
                                 {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    SetOutputDesc(*convOut.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto biasaddOut = BuildBiasAddNode(graphBuilder, "BiasAdd", convOut, bias);
    SetOutputDesc(*biasaddOut.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasadd1Out, biasaddOut});
    ASSERT_NE(graph, nullptr);

    std::vector<GNode> matchedNodes;
    ASSERT_TRUE(ConvFusionUtils::ConvFusionUtilsPass::GetMatchedNodes(graph, matchedNodes, {AscendString("Conv2D")}));
    ASSERT_EQ(matchedNodes.size(), 2U);
    AscendString graphFirstName;
    ASSERT_EQ(matchedNodes[0].GetName(graphFirstName), GRAPH_SUCCESS);
    EXPECT_STREQ(graphFirstName.GetString(), "convolution_1");

    EXPECT_EQ(GetMatchedNodeNamesInOrder(graph), (std::vector<std::string>{"convolution", "convolution_1"}));
}

TEST_F(AABiasaddConvFusionTest, legacyOrderBiasAddConsumerBeforeAddConsumer)
{
    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("order_mixed_consumers");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    // Graph-node-order first: conv_b whose consumer is BiasAdd "zzz_biasadd".
    auto filterB = BuildConstNode(graphBuilder, "filter_b", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto biasB = BuildConstNodeWithValue(graphBuilder, "bias_b", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto convBOut = BuildConvNode(graphBuilder, "conv_b", x, filterB, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    SetOutputDesc(*convBOut.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto biasaddOut = BuildBiasAddNode(graphBuilder, "zzz_biasadd", convBOut, biasB);
    SetOutputDesc(*biasaddOut.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    // Graph-node-order second: conv_a whose consumer is Add "aaa_add".
    auto filterA = BuildConstNode(graphBuilder, "filter_a", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto biasA = BuildConstNodeWithValue(graphBuilder, "bias_a", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto convAOut = BuildConvNode(graphBuilder, "conv_a", x, filterA, {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    SetOutputDesc(*convAOut.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto addOut = BuildAddNode(graphBuilder, "aaa_add", convAOut, biasA);
    SetOutputDesc(*addOut.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasaddOut, addOut});
    ASSERT_NE(graph, nullptr);

    // Legacy BiasaddFusion pattern declares output types {BiasAdd, Add} in definition
    // order (std::vector, not name-sorted): BiasAdd consumers are visited first.
    EXPECT_EQ(GetMatchedNodeNamesInOrder(graph), (std::vector<std::string>{"conv_b", "conv_a"}));
}

TEST_F(AABiasaddConvFusionTest, twoBranchesFusedWithDumpMappingInherited)
{
    SetUpFixpipePlatform();
    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("two_branches_dump_mapping");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter1 = BuildConstNode(graphBuilder, "filter_1", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias1 = BuildConstNodeWithValue(graphBuilder, "bias_1", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto conv1Out = BuildConvNode(graphBuilder, "convolution_1", x, filter1,
                                  {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    SetOutputDesc(*conv1Out.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto biasadd1Out = BuildBiasAddNode(graphBuilder, "BiasAdd_1", conv1Out, bias1);
    SetOutputDesc(*biasadd1Out.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto filter = BuildConstNode(graphBuilder, "filter", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto convOut = BuildConvNode(graphBuilder, "convolution", x, filter,
                                 {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    SetOutputDesc(*convOut.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto biasaddOut = BuildBiasAddNode(graphBuilder, "BiasAdd", convOut, bias);
    SetOutputDesc(*biasaddOut.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({biasadd1Out, biasaddOut});
    RunPass(graph, SUCCESS);

    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graph, "Conv2D"), 2);
    const std::vector<std::pair<std::string, std::string>> convToPatternOutput = {{"convolution", "BiasAdd"},
                                                                                  {"convolution_1", "BiasAdd_1"}};
    for (const auto& item : convToPatternOutput) {
        GNode fusedConv;
        ASSERT_TRUE(FindNodeByName(graph, item.first, fusedConv));
        const auto biasSrc = fusedConv.GetInDataNodesAndPortIndexs(2);
        ASSERT_NE(biasSrc.first, nullptr);
        AttrValue value;
        ASSERT_EQ(fusedConv.GetOutputAttr("_datadump_origin_name", 0, value), GRAPH_SUCCESS);
        AscendString dumpName;
        ASSERT_EQ(value.GetAttrValue(dumpName), GRAPH_SUCCESS);
        EXPECT_STREQ(dumpName.GetString(), item.second.c_str());
        ASSERT_EQ(fusedConv.GetOutputAttr("_datadump_origin_output_index", 0, value), GRAPH_SUCCESS);
        int64_t dumpIndex = -1;
        ASSERT_EQ(value.GetAttrValue(dumpIndex), GRAPH_SUCCESS);
        EXPECT_EQ(dumpIndex, 0);
        EXPECT_EQ(fusedConv.GetOutputAttr("_datadump_origin_format", 0, value), GRAPH_SUCCESS);
        EXPECT_EQ(fusedConv.GetOutputAttr("_datadump_origin_data_type", 0, value), GRAPH_SUCCESS);
    }
    ClearPlatform();
}

// ==========================================================================================
// 成环候选（issue: while 循环体 conv1d+BiasAdd 融合成环）：filter 生产链来自 BiasAdd 输出下游，
// {BiasAdd, Conv2D} 融合成单节点后经外部路径回到自身 -> CanFuse 拒绝。
// 对齐老 pattern 框架语义：仅跳过该候选（GRAPH_NOT_CHANGED），不得整 pass FAILED。
// ==========================================================================================
TEST_F(AABiasaddConvFusionTest, conv2dBiasAddCycleCandidateSkip)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_cycle_skip");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto bias = BuildConstNodeWithValue(graphBuilder, "bias_const", dimsBias, FORMAT_NHWC, DT_FLOAT);

    auto filterSource = graphBuilder.CreateInput(1, "filter_src", DT_FLOAT, FORMAT_NHWC, dimsFilter);
    auto filterOutput = BuildUnaryNode(graphBuilder, filterSource,
                                       {"filter_square", "Square", dimsFilter, FORMAT_NHWC, DT_FLOAT});

    auto convOutput = BuildConvNode(graphBuilder, "conv", x, filterOutput,
                                    {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    auto convNode = convOutput.GetProducer();
    SetOutputDesc(*convNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto biasaddOutput = BuildBiasAddNode(graphBuilder, "biasadd", convOutput, bias);
    auto biasaddNode = biasaddOutput.GetProducer();
    SetOutputDesc(*biasaddNode, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    // 成环：断开占位 filter 边，biasadd 输出 -> Square(输出 filter 形状) -> conv.filter
    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode cycleNode = es::CompliantNodeBuilder(graph)
                          .OpType("Square")
                          .Name("cycle_filter")
                          .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *biasaddNode, 0, cycleNode, 0);
    SetOutputDesc(cycleNode, 0, dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto cycleOutput = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(cycleNode, 0));
    ASSERT_EQ(graph->RemoveEdge(*filterOutput.GetProducer(), filterOutput.GetProducerOutIndex(), *convNode, 1),
              GRAPH_SUCCESS);
    es::AddEdgeAndUpdatePeerDesc(*graph, cycleNode, 0, *convNode, 1);

    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({biasaddOutput});
    RunPass(graphPtr, GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graphPtr, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graphPtr, "Conv2D"), 1);

    ClearPlatform();
}

// 成环候选 + 正常候选共存：正常链正常融合，成环链跳过，pass 整体 SUCCESS
TEST_F(AABiasaddConvFusionTest, conv2dBiasAddCycleSkipWithNormalCandidate)
{
    SetUpFixpipePlatform();

    std::vector<int64_t> dimsX{1, 56, 56, 64};
    std::vector<int64_t> dimsFilter{64, 1, 1, 64};
    std::vector<int64_t> dimsBias{64};
    std::vector<int64_t> dimsOut{1, 28, 28, 64};

    auto graphBuilder = es::EsGraphBuilder("conv2d_biasadd_cycle_with_normal");

    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto filter1 = BuildConstNode(graphBuilder, "filter1", dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto bias1 = BuildConstNodeWithValue(graphBuilder, "bias1", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto conv1Output = BuildConvNode(graphBuilder, "conv1", x1, filter1,
                                     {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    SetOutputDesc(*conv1Output.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto biasadd1Output = BuildBiasAddNode(graphBuilder, "biasadd1", conv1Output, bias1);
    SetOutputDesc(*biasadd1Output.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_NHWC, dimsX);
    auto bias2 = BuildConstNodeWithValue(graphBuilder, "bias2", dimsBias, FORMAT_NHWC, DT_FLOAT);
    auto filter2Source = graphBuilder.CreateInput(2, "filter2_src", DT_FLOAT, FORMAT_NHWC, dimsFilter);
    auto filter2Output = BuildUnaryNode(graphBuilder, filter2Source,
                                        {"filter2_square", "Square", dimsFilter, FORMAT_NHWC, DT_FLOAT});
    auto conv2Output = BuildConvNode(graphBuilder, "conv2", x2, filter2Output,
                                     {"Conv2D", {1, 2, 2, 1}, {1, 1, 1, 1}, "NHWC"});
    SetOutputDesc(*conv2Output.GetProducer(), 0, dimsOut, FORMAT_NHWC, DT_FLOAT);
    auto biasadd2Output = BuildBiasAddNode(graphBuilder, "biasadd2", conv2Output, bias2);
    auto biasadd2Node = biasadd2Output.GetProducer();
    SetOutputDesc(*biasadd2Node, 0, dimsOut, FORMAT_NHWC, DT_FLOAT);

    auto graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode cycleNode = es::CompliantNodeBuilder(graph)
                          .OpType("Square")
                          .Name("cycle_filter2")
                          .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *biasadd2Node, 0, cycleNode, 0);
    SetOutputDesc(cycleNode, 0, dimsFilter, FORMAT_NHWC, DT_FLOAT);
    auto cycleOutput = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(cycleNode, 0));
    ASSERT_EQ(graph->RemoveEdge(*filter2Output.GetProducer(), filter2Output.GetProducerOutIndex(),
                                *conv2Output.GetProducer(), 1),
              GRAPH_SUCCESS);
    es::AddEdgeAndUpdatePeerDesc(*graph, cycleNode, 0, *conv2Output.GetProducer(), 1);

    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({biasadd1Output, biasadd2Output});
    RunPass(graphPtr, SUCCESS);

    EXPECT_EQ(CountNodes(graphPtr, "BiasAdd"), 1);
    EXPECT_EQ(CountNodes(graphPtr, "Conv2D"), 2);

    ClearPlatform();
}
