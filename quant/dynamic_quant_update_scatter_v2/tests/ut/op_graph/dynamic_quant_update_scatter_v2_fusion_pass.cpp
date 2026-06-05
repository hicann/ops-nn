/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "array_ops.h"
#include "nn_quantize.h"
#include "experiment_ops.h"
#include "matrix_calculation_ops.h"
#include "fusion_ops.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph_dsl/graph_checker.h"
#include "graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "gtest/gtest.h"
#include "graph_builder_utils.h"
#include "platform_info_stub.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"
#include "util/util.h"
#include "platform/platform_info.h"
#include "common/configuration.h"
#include "common/utils/ut_op_util.h"
#include "graph/utils/attr_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "dynamic_quant_update_scatter_v2_fusion_pass.h"
#include "compliant_node_builder.h"

using namespace ge;
using namespace op;

static es::EsTensorHolder BuildConstNode(es::EsGraphBuilder& builder)
{
    static int counter = 0;
    std::string name = "Const_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Const")
        .Name(name.c_str())
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildReshapeNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x, const es::EsTensorHolder& shape)
{
    static int counter = 0;
    std::string name = "Reshape_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Reshape")
        .Name(name.c_str())
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *shape.GetProducer(), shape.GetProducerOutIndex(), node, 1);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildBitcastNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x, DataType dt)
{
    static int counter = 0;
    std::string name = "Bitcast_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Bitcast")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"type", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
             es::CreateFrom(static_cast<int64_t>(dt))}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildNegNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x)
{
    static int counter = 0;
    std::string name = "Neg_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Neg")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildUnsqueezeNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x)
{
    static int counter = 0;
    std::string name = "Unsqueeze_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Unsqueeze")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

struct DynamicQuantV2Result {
    es::EsTensorHolder y;
    es::EsTensorHolder scale;
    es::EsTensorHolder offset;
};

static DynamicQuantV2Result BuildDynamicQuantV2Node(es::EsGraphBuilder& builder, const es::EsTensorHolder& x)
{
    static int counter = 0;
    std::string name = "DynamicQuantV2_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("DynamicQuantV2")
        .Name(name.c_str())
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"smooth_scales", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"group_index", es::CompliantNodeBuilder::kEsIrInputOptional, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"scale", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"offset", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return {
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 2))
    };
}

static es::EsTensorHolder BuildScatterNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& var,
    const es::EsTensorHolder& indices, const es::EsTensorHolder& updates, const char* reduce)
{
    static int counter = 0;
    std::string name = "Scatter_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Scatter")
        .Name(name.c_str())
        .IrDefInputs({
            {"var", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"updates", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({{"var", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"reduce", es::CompliantNodeBuilder::kEsAttrOptional, "String",
             es::CreateFrom(ge::AscendString(reduce))},
            {"axis", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
             es::CreateFrom(static_cast<int64_t>(0))}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *var.GetProducer(), var.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *indices.GetProducer(), indices.GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *updates.GetProducer(), updates.GetProducerOutIndex(), node, 2);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

class dynamic_quant_update_scatter_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "dynamic_quant_update_scatter_v2_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "dynamic_quant_update_scatter_v2_test TearDown" << std::endl;
    }
};

TEST_F(dynamic_quant_update_scatter_v2_test, dynamic_quant_update_scatter_v2_fusion_93)
{
    std::cout << "DynamicQuantUpdateScatterV2FusionPass Test 1" << std::endl;

    std::vector<int64_t> dims_x{192, 1, 128};
    std::vector<int64_t> dims_indices{192};
    std::vector<int64_t> dims_input0{192, 1024, 1, 16};
    std::vector<int64_t> dims_input1{192, 1024};
    std::vector<int64_t> dims_input2{192, 1024};
    Shape shape_x(dims_x);
    Shape shape_indices(dims_indices);
    Shape shape_input0(dims_input0);
    Shape shape_input1(dims_input1);
    Shape shape_input2(dims_input2);

    dlog_setlevel(-1, 0, 0);
    auto graph_builder = es::EsGraphBuilder("dynamic_quant_update_scatter_v2_fusion_third_patten");
    auto x = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto indices = graph_builder.CreateInput(1, "indices", DT_INT32, FORMAT_ND, shape_x.GetDims());
    auto input0 = graph_builder.CreateInput(2, "input0", DT_INT32, FORMAT_ND, shape_input0.GetDims());
    auto input1 = graph_builder.CreateInput(3, "input1", DT_FLOAT, FORMAT_ND, shape_input1.GetDims());
    auto input2 = graph_builder.CreateInput(4, "input2", DT_FLOAT, FORMAT_ND, shape_input2.GetDims());

    auto dynamic_quant_v2 = BuildDynamicQuantV2Node(graph_builder, x);
    auto reshape0 = BuildReshapeNode(graph_builder, dynamic_quant_v2.y, BuildConstNode(graph_builder));
    auto bitcast0 = BuildBitcastNode(graph_builder, reshape0, DT_INT32);
    auto reshape1 = BuildReshapeNode(graph_builder, bitcast0, BuildConstNode(graph_builder));
    auto scatter0 = BuildScatterNode(graph_builder, input0, indices, reshape1, "update");
    auto reshape2 = BuildReshapeNode(graph_builder, scatter0, BuildConstNode(graph_builder));
    auto bitcast1 = BuildBitcastNode(graph_builder, reshape2, DT_INT4);
    auto reshape3 = BuildReshapeNode(graph_builder, bitcast1, BuildConstNode(graph_builder));

    auto scatter1 = BuildScatterNode(graph_builder, input1, indices, dynamic_quant_v2.scale, "update");
    auto unsqueeze0 = BuildUnsqueezeNode(graph_builder, scatter1);

    auto neg = BuildNegNode(graph_builder, dynamic_quant_v2.offset);
    auto scatter2 = BuildScatterNode(graph_builder, input2, indices, neg, "update");
    auto unsqueeze1 = BuildUnsqueezeNode(graph_builder, scatter2);

    // x
    TensorDesc dynamicquantv2_input_0_desc;
    dynamic_quant_v2.y.GetProducer()->GetInputDesc(0, dynamicquantv2_input_0_desc);
    dynamicquantv2_input_0_desc.SetDataType(DT_FLOAT16);
    dynamicquantv2_input_0_desc.SetShape(shape_x);
    dynamic_quant_v2.y.GetProducer()->UpdateInputDesc(0, dynamicquantv2_input_0_desc);

    // indices
    TensorDesc scatter0_input_1_desc;
    scatter0.GetProducer()->GetInputDesc(1, scatter0_input_1_desc);
    scatter0_input_1_desc.SetDataType(DT_INT32);
    scatter0_input_1_desc.SetShape(shape_indices);
    scatter0.GetProducer()->UpdateInputDesc(1, scatter0_input_1_desc);

    TensorDesc scatter1_input_1_desc;
    scatter1.GetProducer()->GetInputDesc(1, scatter1_input_1_desc);
    scatter1_input_1_desc.SetDataType(DT_INT32);
    scatter1_input_1_desc.SetShape(shape_indices);
    scatter1.GetProducer()->UpdateInputDesc(1, scatter1_input_1_desc);

    TensorDesc scatter2_input_1_desc;
    scatter2.GetProducer()->GetInputDesc(1, scatter2_input_1_desc);
    scatter2_input_1_desc.SetDataType(DT_INT32);
    scatter2_input_1_desc.SetShape(shape_indices);
    scatter2.GetProducer()->UpdateInputDesc(1, scatter2_input_1_desc);

    // input0
    TensorDesc scatter0_input_0_desc;
    scatter0.GetProducer()->GetInputDesc(0, scatter0_input_0_desc);
    scatter0_input_0_desc.SetDataType(DT_INT32);
    scatter0_input_0_desc.SetShape(shape_input0);
    scatter0.GetProducer()->UpdateInputDesc(0, scatter0_input_0_desc);

    // input1
    TensorDesc scatter1_input_0_desc;
    scatter1.GetProducer()->GetInputDesc(0, scatter1_input_0_desc);
    scatter1_input_0_desc.SetDataType(DT_FLOAT);
    scatter1_input_0_desc.SetShape(shape_input1);
    scatter1.GetProducer()->UpdateInputDesc(0, scatter1_input_0_desc);
    
    // input2
    TensorDesc scatter2_input_0_desc;
    scatter2.GetProducer()->GetInputDesc(0, scatter2_input_0_desc);
    scatter2_input_0_desc.SetDataType(DT_FLOAT);
    scatter2_input_0_desc.SetShape(shape_input2);
    scatter2.GetProducer()->UpdateInputDesc(0, scatter2_input_0_desc);

    std::vector<es::EsTensorHolder> graph_outputs;
    graph_outputs.emplace_back(std::move(reshape3));
    graph_outputs.emplace_back(std::move(unsqueeze0));
    graph_outputs.emplace_back(std::move(unsqueeze1));
    std::shared_ptr<Graph> graph = graph_builder.Build(graph_outputs);
    CustomPassContext pass_context;

    ops::DynamicQuantUpdateScatterV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);

    bool findDynamicQuantUpdateScatterV2 = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        std::cout<<"GName"<<type.GetString()<<std::endl;
        if (type == "DynamicQuantUpdateScatterV2") {
            findDynamicQuantUpdateScatterV2 = true;
        }
    }
    EXPECT_EQ(findDynamicQuantUpdateScatterV2, true);
}

TEST_F(dynamic_quant_update_scatter_v2_test, test_3)
{
    ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
    auto x = builder.AddNode("X", "Data", 0, 1, {192, 1, 128}, ge::FORMAT_ND, ge::DT_FLOAT16);
    auto indices = builder.AddNode("Indices", "Data", 0, 1, {192}, ge::FORMAT_ND, ge::DT_INT32);
    auto input0 = builder.AddNode("Input0", "Data", 0, 1, {192, 1024, 1, 32}, ge::FORMAT_ND, ge::DT_INT32);
    auto input1 = builder.AddNode("Input1", "Data", 0, 1, {192, 1024}, ge::FORMAT_ND, ge::DT_FLOAT);
    auto input2 = builder.AddNode("Input2", "Data", 0, 1, {192, 1024}, ge::FORMAT_ND, ge::DT_FLOAT);
    auto dynamicQuantV2 = builder.AddNode(
        "DynamicQuantV2", "DynamicQuantV2", {{Format::FORMAT_ND, ge::DT_FLOAT16, {192, 1, 128}}},
        {{Format::FORMAT_ND, ge::DT_INT8, {192, 1, 128}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}});
    auto const0 = builder.AddNode("Const0", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT32, {4}}});
    auto const1 = builder.AddNode("Const1", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT64, {4}}});
    auto const2 = builder.AddNode("Const2", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT64, {3}}});
    auto const3 = builder.AddNode("Const3", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT64, {3}}});
    auto reshape0 = builder.AddNode(
        "Reshape0", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT8, {192, 1, 128}}, {Format::FORMAT_ND, ge::DT_INT32, {4}}},
        {{Format::FORMAT_ND, ge::DT_INT8, {192, 1, 32, 4}}});
    auto bitcast0 = builder.AddNode(
        "Bitcast0", "Bitcast", {{Format::FORMAT_ND, ge::DT_INT8, {192, 1, 32, 4}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1, 32}}});
    auto reshape1 = builder.AddNode(
        "Reshape1", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1, 32}}, {Format::FORMAT_ND, ge::DT_INT64, {4}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1, 1, 32}}});
    auto scatter0 = builder.AddNode(
        "Scatter0", "Scatter",
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 1, 32}},
         {Format::FORMAT_ND, ge::DT_INT32, {192}},
         {Format::FORMAT_ND, ge::DT_INT32, {192, 1, 1, 32}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 1, 32}}});
    auto reshape2 = builder.AddNode(
        "Reshape2", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 1, 32}}, {Format::FORMAT_ND, ge::DT_INT64, {3}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 32}}});
    auto bitcast1 = builder.AddNode(
        "Bitcast1", "Bitcast", {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 32}}},
        {{Format::FORMAT_ND, ge::DT_INT8, {192, 1024, 32, 4}}});
    auto reshape3 = builder.AddNode(
        "Reshape3", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT8, {192, 1024, 32, 4}}, {Format::FORMAT_ND, ge::DT_INT64, {3}}},
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1024, 128}}});
    auto scatter1 = builder.AddNode(
        "Scatter1", "Scatter",
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}},
         {Format::FORMAT_ND, ge::DT_INT32, {192}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}});
    auto unsqueeze0 = builder.AddNode(
        "Unsqueeze0", "Unsqueeze", {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {1, 192, 1024}}});
    auto neg = builder.AddNode(
        "Neg", "Neg", {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}}, {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}});
    auto scatter2 = builder.AddNode(
        "Scatter2", "Scatter",
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}},
         {Format::FORMAT_ND, ge::DT_INT32, {192}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}});
    auto unsqueeze1 = builder.AddNode(
        "Unsqueeze1", "Unsqueeze", {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {1, 192, 1024}}});
    auto net_output = builder.AddNode("NetOutput", "NetOutput", 1, 0, {192, 1024, 128}, Format::FORMAT_ND, ge::DT_INT4);

    builder.AddDataEdge(x, 0, dynamicQuantV2, 0);
    builder.AddDataEdge(dynamicQuantV2, 0, reshape0, 0);
    builder.AddDataEdge(const0, 0, reshape0, 1);
    builder.AddDataEdge(reshape0, 0, bitcast0, 0);
    builder.AddDataEdge(bitcast0, 0, reshape1, 0);
    builder.AddDataEdge(const1, 0, reshape1, 1);
    builder.AddDataEdge(input0, 0, scatter0, 0);
    builder.AddDataEdge(indices, 0, scatter0, 1);
    builder.AddDataEdge(reshape1, 0, scatter0, 2);
    builder.AddDataEdge(scatter0, 0, reshape2, 0);
    builder.AddDataEdge(const2, 0, reshape2, 1);
    builder.AddDataEdge(reshape2, 0, bitcast1, 0);
    builder.AddDataEdge(bitcast1, 0, reshape3, 0);
    builder.AddDataEdge(const3, 0, reshape3, 1);
    builder.AddDataEdge(reshape3, 0, net_output, 0);
    builder.AddDataEdge(input1, 0, scatter1, 0);
    builder.AddDataEdge(indices, 0, scatter1, 1);
    builder.AddDataEdge(dynamicQuantV2, 1, scatter1, 2);
    builder.AddDataEdge(scatter1, 0, unsqueeze0, 0);
    builder.AddDataEdge(dynamicQuantV2, 2, neg, 0);
    builder.AddDataEdge(input2, 0, scatter2, 0);
    builder.AddDataEdge(indices, 0, scatter2, 1);
    builder.AddDataEdge(neg, 0, scatter2, 2);
    builder.AddDataEdge(scatter2, 0, unsqueeze1, 0);
    builder.AddControlEdge(dynamicQuantV2, const0);
    builder.AddControlEdge(dynamicQuantV2, reshape1);
    builder.AddControlEdge(reshape2, const3);
    auto graph = builder.GetGraph();

    GraphUtils::DumpGEGraphToOnnx(*graph, "0");
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass(
        "DynamicQuantUpdateScatterV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);
    EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(dynamic_quant_update_scatter_v2_test, test_4)
{
    ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
    auto x = builder.AddNode("X", "Data", 0, 1, {192, 1, 20480}, ge::FORMAT_ND, ge::DT_FLOAT16);
    auto indices = builder.AddNode("Indices", "Data", 0, 1, {192}, ge::FORMAT_ND, ge::DT_INT32);
    auto input0 = builder.AddNode("Input0", "Data", 0, 1, {192, 1024, 1, 2560}, ge::FORMAT_ND, ge::DT_INT32);
    auto input1 = builder.AddNode("Input1", "Data", 0, 1, {192, 1024}, ge::FORMAT_ND, ge::DT_FLOAT);
    auto input2 = builder.AddNode("Input2", "Data", 0, 1, {192, 1024}, ge::FORMAT_ND, ge::DT_FLOAT);
    auto dynamicQuantV2 = builder.AddNode(
        "DynamicQuantV2", "DynamicQuantV2", {{Format::FORMAT_ND, ge::DT_FLOAT16, {192, 1, 20480}}},
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1, 20480}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}});
    auto const0 = builder.AddNode("Const0", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT32, {4}}});
    auto const1 = builder.AddNode("Const1", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT64, {4}}});
    auto const2 = builder.AddNode("Const2", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT64, {3}}});
    auto const3 = builder.AddNode("Const3", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT64, {3}}});
    auto reshape0 = builder.AddNode(
        "Reshape0", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1, 20480}}, {Format::FORMAT_ND, ge::DT_INT32, {4}}},
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1, 2560, 8}}});
    auto bitcast0 = builder.AddNode(
        "Bitcast0", "Bitcast", {{Format::FORMAT_ND, ge::DT_INT4, {192, 1, 2560, 8}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1, 2560}}});
    auto reshape1 = builder.AddNode(
        "Reshape1", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1, 2560}}, {Format::FORMAT_ND, ge::DT_INT64, {4}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1, 1, 2560}}});
    auto scatter0 = builder.AddNode(
        "Scatter0", "Scatter",
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 1, 2560}},
         {Format::FORMAT_ND, ge::DT_INT32, {192}},
         {Format::FORMAT_ND, ge::DT_INT32, {192, 1, 1, 2560}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 1, 2560}}});
    auto reshape2 = builder.AddNode(
        "Reshape2", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 1, 2560}}, {Format::FORMAT_ND, ge::DT_INT64, {3}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 2560}}});
    auto bitcast1 = builder.AddNode(
        "Bitcast1", "Bitcast", {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 2560}}},
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1024, 2560, 8}}});
    auto reshape3 = builder.AddNode(
        "Reshape3", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1024, 2560, 8}}, {Format::FORMAT_ND, ge::DT_INT64, {3}}},
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1024, 20480}}});
    auto scatter1 = builder.AddNode(
        "Scatter1", "Scatter",
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}},
         {Format::FORMAT_ND, ge::DT_INT32, {192}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}});
    auto unsqueeze0 = builder.AddNode(
        "Unsqueeze0", "Unsqueeze", {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {1, 192, 1024}}});
    auto neg = builder.AddNode(
        "Neg", "Neg", {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}}, {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}});
    auto scatter2 = builder.AddNode(
        "Scatter2", "Scatter",
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}},
         {Format::FORMAT_ND, ge::DT_INT32, {192}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}});
    auto unsqueeze1 = builder.AddNode(
        "Unsqueeze1", "Unsqueeze", {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {1, 192, 1024}}});
    auto net_output =
        builder.AddNode("NetOutput", "NetOutput", 1, 0, {192, 1024, 20480}, Format::FORMAT_ND, ge::DT_INT4);

    builder.AddDataEdge(x, 0, dynamicQuantV2, 0);
    builder.AddDataEdge(dynamicQuantV2, 0, reshape0, 0);
    builder.AddDataEdge(const0, 0, reshape0, 1);
    builder.AddDataEdge(reshape0, 0, bitcast0, 0);
    builder.AddDataEdge(bitcast0, 0, reshape1, 0);
    builder.AddDataEdge(const1, 0, reshape1, 1);
    builder.AddDataEdge(input0, 0, scatter0, 0);
    builder.AddDataEdge(indices, 0, scatter0, 1);
    builder.AddDataEdge(reshape1, 0, scatter0, 2);
    builder.AddDataEdge(scatter0, 0, reshape2, 0);
    builder.AddDataEdge(const2, 0, reshape2, 1);
    builder.AddDataEdge(reshape2, 0, bitcast1, 0);
    builder.AddDataEdge(bitcast1, 0, reshape3, 0);
    builder.AddDataEdge(const3, 0, reshape3, 1);
    builder.AddDataEdge(reshape3, 0, net_output, 0);
    builder.AddDataEdge(input1, 0, scatter1, 0);
    builder.AddDataEdge(indices, 0, scatter1, 1);
    builder.AddDataEdge(dynamicQuantV2, 1, scatter1, 2);
    builder.AddDataEdge(scatter1, 0, unsqueeze0, 0);
    builder.AddDataEdge(dynamicQuantV2, 2, neg, 0);
    builder.AddDataEdge(input2, 0, scatter2, 0);
    builder.AddDataEdge(indices, 0, scatter2, 1);
    builder.AddDataEdge(neg, 0, scatter2, 2);
    builder.AddDataEdge(scatter2, 0, unsqueeze1, 0);
    builder.AddControlEdge(dynamicQuantV2, const0);
    builder.AddControlEdge(dynamicQuantV2, reshape1);
    builder.AddControlEdge(reshape2, const3);
    auto graph = builder.GetGraph();

    GraphUtils::DumpGEGraphToOnnx(*graph, "0");
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass(
        "DynamicQuantUpdateScatterV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);
    EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(dynamic_quant_update_scatter_v2_test, test_5)
{
    ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
    auto x = builder.AddNode("X", "Data", 0, 1, {192, 1, 128}, ge::FORMAT_ND, ge::DT_FLOAT16);
    auto indices = builder.AddNode("Indices", "Data", 0, 1, {192}, ge::FORMAT_ND, ge::DT_INT32);
    auto input0 = builder.AddNode("Input0", "Data", 0, 1, {192, 1024, 1, 16}, ge::FORMAT_ND, ge::DT_INT32);
    auto input1 = builder.AddNode("Input1", "Data", 0, 1, {192, 1024}, ge::FORMAT_ND, ge::DT_FLOAT);
    auto input2 = builder.AddNode("Input2", "Data", 0, 1, {192, 1024}, ge::FORMAT_ND, ge::DT_FLOAT);
    auto dynamicQuantV2 = builder.AddNode(
        "DynamicQuantV2", "DynamicQuantV2", {{Format::FORMAT_ND, ge::DT_FLOAT16, {192, 1, 128}}},
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1, 128}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}});
    auto const0 = builder.AddNode("Const0", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT32, {4}}});
    auto const1 = builder.AddNode("Const1", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT64, {4}}});
    auto const2 = builder.AddNode("Const2", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT64, {3}}});
    auto const3 = builder.AddNode("Const3", "Const", {}, {{Format::FORMAT_ND, ge::DT_INT64, {3}}});
    auto reshape0 = builder.AddNode(
        "Reshape0", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1, 128}}, {Format::FORMAT_ND, ge::DT_INT32, {4}}},
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1, 16, 8}}});
    auto bitcast0 = builder.AddNode(
        "Bitcast0", "Bitcast", {{Format::FORMAT_ND, ge::DT_INT4, {192, 1, 16, 8}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1, 16}}});
    auto reshape1 = builder.AddNode(
        "Reshape1", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1, 16}}, {Format::FORMAT_ND, ge::DT_INT64, {4}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1, 1, 16}}});
    auto scatter0 = builder.AddNode(
        "Scatter0", "Scatter",
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1024, 1, 128}},
         {Format::FORMAT_ND, ge::DT_INT32, {192}},
         {Format::FORMAT_ND, ge::DT_INT32, {192, 1, 1, 16}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 1, 16}}});
    auto reshape2 = builder.AddNode(
        "Reshape2", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 1, 16}}, {Format::FORMAT_ND, ge::DT_INT64, {3}}},
        {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 16}}});
    auto bitcast1 = builder.AddNode(
        "Bitcast1", "Bitcast", {{Format::FORMAT_ND, ge::DT_INT32, {192, 1024, 16}}},
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1024, 16, 8}}});
    auto reshape3 = builder.AddNode(
        "Reshape3", "Reshape",
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1024, 16, 8}}, {Format::FORMAT_ND, ge::DT_INT64, {3}}},
        {{Format::FORMAT_ND, ge::DT_INT4, {192, 1024, 128}}});
    auto scatter1 = builder.AddNode(
        "Scatter1", "Scatter",
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}},
         {Format::FORMAT_ND, ge::DT_INT32, {192}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}});
    auto unsqueeze0 = builder.AddNode(
        "Unsqueeze0", "Unsqueeze", {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {1, 192, 1024}}});
    auto neg = builder.AddNode(
        "Neg", "Neg", {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}}, {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}});
    auto scatter2 = builder.AddNode(
        "Scatter2", "Scatter",
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}},
         {Format::FORMAT_ND, ge::DT_INT32, {192}},
         {Format::FORMAT_ND, ge::DT_FLOAT, {192, 1}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}});
    auto unsqueeze1 = builder.AddNode(
        "Unsqueeze1", "Unsqueeze", {{Format::FORMAT_ND, ge::DT_FLOAT, {192, 1024}}},
        {{Format::FORMAT_ND, ge::DT_FLOAT, {1, 192, 1024}}});
    auto net_output = builder.AddNode("NetOutput", "NetOutput", 1, 0, {192, 1024, 128}, Format::FORMAT_ND, ge::DT_INT4);

    builder.AddDataEdge(x, 0, dynamicQuantV2, 0);
    builder.AddDataEdge(dynamicQuantV2, 0, reshape0, 0);
    builder.AddDataEdge(const0, 0, reshape0, 1);
    builder.AddDataEdge(reshape0, 0, bitcast0, 0);
    builder.AddDataEdge(bitcast0, 0, reshape1, 0);
    builder.AddDataEdge(const1, 0, reshape1, 1);
    builder.AddDataEdge(input0, 0, scatter0, 0);
    builder.AddDataEdge(indices, 0, scatter0, 1);
    builder.AddDataEdge(reshape1, 0, scatter0, 2);
    builder.AddDataEdge(scatter0, 0, reshape2, 0);
    builder.AddDataEdge(const2, 0, reshape2, 1);
    builder.AddDataEdge(reshape2, 0, bitcast1, 0);
    builder.AddDataEdge(bitcast1, 0, reshape3, 0);
    builder.AddDataEdge(const3, 0, reshape3, 1);
    builder.AddDataEdge(reshape3, 0, net_output, 0);
    builder.AddDataEdge(input1, 0, scatter1, 0);
    builder.AddDataEdge(indices, 0, scatter1, 1);
    builder.AddDataEdge(dynamicQuantV2, 1, scatter1, 2);
    builder.AddDataEdge(scatter1, 0, unsqueeze0, 0);
    builder.AddDataEdge(dynamicQuantV2, 2, neg, 0);
    builder.AddDataEdge(input2, 0, scatter2, 0);
    builder.AddDataEdge(indices, 0, scatter2, 1);
    builder.AddDataEdge(neg, 0, scatter2, 2);
    builder.AddDataEdge(scatter2, 0, unsqueeze1, 0);
    builder.AddControlEdge(dynamicQuantV2, const0);
    builder.AddControlEdge(dynamicQuantV2, reshape1);
    builder.AddControlEdge(reshape2, const3);
    auto graph = builder.GetGraph();

    GraphUtils::DumpGEGraphToOnnx(*graph, "0");
    auto ret = fe::FusionPassTestUtils::RunGraphFusionPass(
        "DynamicQuantUpdateScatterV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *graph);
    EXPECT_EQ(ret, fe::NOT_CHANGED);
}
