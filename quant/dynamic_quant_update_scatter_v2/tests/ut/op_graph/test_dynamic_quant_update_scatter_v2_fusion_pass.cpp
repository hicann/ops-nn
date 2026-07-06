/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ut_op_util.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "../../../op_graph/fusion_pass/dynamic_quant_update_scatter_v2_fusion_pass.h"
#include "compliant_node_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;

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

static es::EsTensorHolder BuildReshapeNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x,
                                           const es::EsTensorHolder& shape)
{
    static int counter = 0;
    std::string name = "Reshape_" + std::to_string(counter++);
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
                     .IrDefAttrs({{"type", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                   es::CreateFrom(static_cast<int64_t>(dt))}})
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
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"smooth_scales", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                   {"group_index", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"scale", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                    {"offset", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    return {es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0)),
            es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1)),
            es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 2))};
}

static es::EsTensorHolder BuildScatterNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& var,
                                           const es::EsTensorHolder& indices, const es::EsTensorHolder& updates,
                                           const char* reduce)
{
    static int counter = 0;
    std::string name = "Scatter_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Scatter")
                     .Name(name.c_str())
                     .IrDefInputs({{"var", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"updates", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"var", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs({{"reduce", es::CompliantNodeBuilder::kEsAttrOptional, "String",
                                   es::CreateFrom(ge::AscendString(reduce))},
                                  {"axis", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                                   es::CreateFrom(static_cast<int64_t>(0))}})
                     .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *var.GetProducer(), var.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *indices.GetProducer(), indices.GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *updates.GetProducer(), updates.GetProducerOutIndex(), node, 2);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

class dynamic_quant_update_scatter_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "dynamic_quant_update_scatter_v2_test SetUp" << std::endl;
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 40;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void TearDownTestCase()
    {
        std::cout << "dynamic_quant_update_scatter_v2_test TearDown" << std::endl;
        fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    }
};

/**
 * Negative test: x dtype is DT_FLOAT (not FP16/BF16) -> fusion should NOT happen
 */
TEST_F(dynamic_quant_update_scatter_v2_test, test_wrong_x_dtype)
{
    std::cout << "DynamicQuantUpdateScatterV2FusionPass test_wrong_x_dtype" << std::endl;

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

    auto graph_builder = es::EsGraphBuilder("test_wrong_x_dtype");
    auto x = graph_builder.CreateInput(0, "x1", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto indices = graph_builder.CreateInput(1, "indices", DT_INT32, FORMAT_ND, shape_x.GetDims());
    auto input0 = graph_builder.CreateInput(2, "input0", DT_INT32, FORMAT_ND, shape_input0.GetDims());
    auto input1 = graph_builder.CreateInput(3, "input1", DT_FLOAT, FORMAT_ND, shape_input1.GetDims());
    auto input2 = graph_builder.CreateInput(4, "input2", DT_FLOAT, FORMAT_ND, shape_input2.GetDims());

    auto dynamic_quant_v2 = BuildDynamicQuantV2Node(graph_builder, x);
    auto reshape0 = BuildReshapeNode(graph_builder, dynamic_quant_v2.y, BuildConstNode(graph_builder));
    auto bitcast0 = BuildBitcastNode(graph_builder, reshape0, DT_INT32);
    auto reshape1 = BuildReshapeNode(graph_builder, bitcast0, BuildConstNode(graph_builder));
    auto reshape1b = BuildReshapeNode(graph_builder, reshape1, BuildConstNode(graph_builder));
    auto scatter0 = BuildScatterNode(graph_builder, input0, indices, reshape1b, "update");
    auto reshape2 = BuildReshapeNode(graph_builder, scatter0, BuildConstNode(graph_builder));
    auto bitcast1 = BuildBitcastNode(graph_builder, reshape2, DT_INT4);
    auto reshape3 = BuildReshapeNode(graph_builder, bitcast1, BuildConstNode(graph_builder));

    auto scatter1 = BuildScatterNode(graph_builder, input1, indices, dynamic_quant_v2.scale, "update");
    auto unsqueeze0 = BuildUnsqueezeNode(graph_builder, scatter1);

    auto neg = BuildNegNode(graph_builder, dynamic_quant_v2.offset);
    auto scatter2 = BuildScatterNode(graph_builder, input2, indices, neg, "update");
    auto unsqueeze1 = BuildUnsqueezeNode(graph_builder, scatter2);

    // x - use DT_FLOAT (wrong dtype, should be FP16 or BF16)
    TensorDesc dynamicquantv2_input_0_desc;
    dynamic_quant_v2.y.GetProducer()->GetInputDesc(0, dynamicquantv2_input_0_desc);
    dynamicquantv2_input_0_desc.SetDataType(DT_FLOAT);
    dynamicquantv2_input_0_desc.SetShape(shape_x);
    dynamic_quant_v2.y.GetProducer()->UpdateInputDesc(0, dynamicquantv2_input_0_desc);

    // DynamicQuantV2 outputs
    TensorDesc dqv2_out0;
    dqv2_out0.SetDataType(DT_INT4);
    dqv2_out0.SetShape(shape_x);
    dynamic_quant_v2.y.GetProducer()->UpdateOutputDesc(0, dqv2_out0);
    TensorDesc dqv2_out1;
    dqv2_out1.SetDataType(DT_FLOAT);
    dqv2_out1.SetShape(Shape({dims_x[0], 1}));
    dynamic_quant_v2.y.GetProducer()->UpdateOutputDesc(1, dqv2_out1);
    TensorDesc dqv2_out2;
    dqv2_out2.SetDataType(DT_FLOAT);
    dqv2_out2.SetShape(Shape({dims_x[0], 1}));
    dynamic_quant_v2.y.GetProducer()->UpdateOutputDesc(2, dqv2_out2);

    // reshape3 output (var output after Bitcast INT4)
    TensorDesc reshape3_out;
    reshape3_out.SetDataType(DT_INT4);
    reshape3_out.SetShape(Shape({dims_x[0], 1024, dims_x[2]}));
    reshape3.GetProducer()->UpdateOutputDesc(0, reshape3_out);

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
    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset(graph_outputs);
    CustomPassContext pass_context;

    ops::DynamicQuantUpdateScatterV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);

    bool findDynamicQuantUpdateScatterV2 = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "DynamicQuantUpdateScatterV2") {
            findDynamicQuantUpdateScatterV2 = true;
        }
    }
    EXPECT_EQ(findDynamicQuantUpdateScatterV2, false);
}

/**
 * Negative test: h dimension exceeds 14960 -> fusion should NOT happen
 */
TEST_F(dynamic_quant_update_scatter_v2_test, test_h_dim_too_large)
{
    std::cout << "DynamicQuantUpdateScatterV2FusionPass test_h_dim_too_large" << std::endl;

    std::vector<int64_t> dims_x{192, 1, 14968}; // h=14968 > 14960, not aligned to 8
    std::vector<int64_t> dims_indices{192};
    std::vector<int64_t> dims_input0{192, 1024, 1, 1871};
    std::vector<int64_t> dims_input1{192, 1024};
    std::vector<int64_t> dims_input2{192, 1024};
    Shape shape_x(dims_x);
    Shape shape_indices(dims_indices);
    Shape shape_input0(dims_input0);
    Shape shape_input1(dims_input1);
    Shape shape_input2(dims_input2);

    auto graph_builder = es::EsGraphBuilder("test_h_dim_too_large");
    auto x = graph_builder.CreateInput(0, "x1", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto indices = graph_builder.CreateInput(1, "indices", DT_INT32, FORMAT_ND, shape_x.GetDims());
    auto input0 = graph_builder.CreateInput(2, "input0", DT_INT32, FORMAT_ND, shape_input0.GetDims());
    auto input1 = graph_builder.CreateInput(3, "input1", DT_FLOAT, FORMAT_ND, shape_input1.GetDims());
    auto input2 = graph_builder.CreateInput(4, "input2", DT_FLOAT, FORMAT_ND, shape_input2.GetDims());

    auto dynamic_quant_v2 = BuildDynamicQuantV2Node(graph_builder, x);
    auto reshape0 = BuildReshapeNode(graph_builder, dynamic_quant_v2.y, BuildConstNode(graph_builder));
    auto bitcast0 = BuildBitcastNode(graph_builder, reshape0, DT_INT32);
    auto reshape1 = BuildReshapeNode(graph_builder, bitcast0, BuildConstNode(graph_builder));
    auto reshape1b = BuildReshapeNode(graph_builder, reshape1, BuildConstNode(graph_builder));
    auto scatter0 = BuildScatterNode(graph_builder, input0, indices, reshape1b, "update");
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

    // DynamicQuantV2 outputs
    TensorDesc dqv2_out0;
    dqv2_out0.SetDataType(DT_INT4);
    dqv2_out0.SetShape(shape_x);
    dynamic_quant_v2.y.GetProducer()->UpdateOutputDesc(0, dqv2_out0);
    TensorDesc dqv2_out1;
    dqv2_out1.SetDataType(DT_FLOAT);
    dqv2_out1.SetShape(Shape({dims_x[0], 1}));
    dynamic_quant_v2.y.GetProducer()->UpdateOutputDesc(1, dqv2_out1);
    TensorDesc dqv2_out2;
    dqv2_out2.SetDataType(DT_FLOAT);
    dqv2_out2.SetShape(Shape({dims_x[0], 1}));
    dynamic_quant_v2.y.GetProducer()->UpdateOutputDesc(2, dqv2_out2);

    // reshape3 output (var output after Bitcast INT4)
    TensorDesc reshape3_out;
    reshape3_out.SetDataType(DT_INT4);
    reshape3_out.SetShape(Shape({dims_x[0], 1024, dims_x[2]}));
    reshape3.GetProducer()->UpdateOutputDesc(0, reshape3_out);

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
    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset(graph_outputs);
    CustomPassContext pass_context;

    ops::DynamicQuantUpdateScatterV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);

    bool findDynamicQuantUpdateScatterV2 = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "DynamicQuantUpdateScatterV2") {
            findDynamicQuantUpdateScatterV2 = true;
        }
    }
    EXPECT_EQ(findDynamicQuantUpdateScatterV2, false);
}
