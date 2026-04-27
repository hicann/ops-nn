/*
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "../../../op_graph/fusion_pass/softmax_grad_ext_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace ops;

// Helper function to build TwoInputOneOutput node using CompliantNodeBuilder
es::EsTensorHolder BuildTwoInOneOutNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& input0,
    const es::EsTensorHolder& input1, const char *op_type)
{
    ge::Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType(op_type)
        .Name(op_type)
        .IrDefInputs({
            {"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .Build();
    
    // Connect input to two input one output node using AddEdgeAndUpdatePeerDesc
    es::AddEdgeAndUpdatePeerDesc(*graph, *input0.GetProducer(), input0.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *input1.GetProducer(), input1.GetProducerOutIndex(), node, 1);
    
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

// Helper function to build ReduceSum node in test
// Note: EsReduceSum not available in library, must use CompliantNodeBuilder
es::EsTensorHolder BuildReduceSumForTest(es::EsGraphBuilder& builder, const es::EsTensorHolder& input, int64_t axis, bool keep_dims)
{
    ge::Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    
    GNode reduce_sum_node = es::CompliantNodeBuilder(graph)
        .OpType("ReduceSum")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .IrDefAttrs({
            {"axes", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt", es::CreateFrom(std::vector<int64_t>{axis})},
            {"keep_dims", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(keep_dims)}
        })
        .Build();
    
    // Connect input to ReduceSum node using AddEdgeAndUpdatePeerDesc
    GNode* input_node = input.GetProducer();
    int32_t input_out_idx = input.GetProducerOutIndex();
    es::AddEdgeAndUpdatePeerDesc(*graph, *input_node, input_out_idx, reduce_sum_node, 0);
    
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(reduce_sum_node, 0));
}

class SoftmaxGradExtFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void InferShapeForTest(
        DataType dtype, Shape& shape,
        es::EsTensorHolder& input0, es::EsTensorHolder& input1, es::EsTensorHolder& input2, es::EsTensorHolder& op1,
        es::EsTensorHolder& op2, es::EsTensorHolder& op3, es::EsTensorHolder& op4, es::EsTensorHolder& op5)
    {
        //input0
        TensorDesc x1_output_desc;
        input0.GetProducer()->GetOutputDesc(0, x1_output_desc);
        x1_output_desc.SetDataType(dtype);
        x1_output_desc.SetShape(shape);
        input0.GetProducer()->UpdateOutputDesc(0, x1_output_desc);
        //input1
        TensorDesc x2_output_desc;
        input1.GetProducer()->GetOutputDesc(0, x2_output_desc);
        x2_output_desc.SetDataType(dtype);
        x2_output_desc.SetShape(shape);
        input1.GetProducer()->UpdateOutputDesc(0, x2_output_desc);
        //input2
        TensorDesc x3_output_desc;
        input2.GetProducer()->GetOutputDesc(0, x3_output_desc);
        x3_output_desc.SetDataType(dtype);
        x3_output_desc.SetShape(shape);
        input2.GetProducer()->UpdateOutputDesc(0, x3_output_desc);

        //op1
        TensorDesc op1_input_0_desc;
        op1.GetProducer()->GetInputDesc(0, op1_input_0_desc);
        op1_input_0_desc.SetDataType(dtype);
        op1_input_0_desc.SetShape(shape);
        TensorDesc op1_output_desc;
        op1.GetProducer()->GetOutputDesc(0, op1_output_desc);
        op1_output_desc.SetDataType(dtype);
        op1_output_desc.SetShape(shape);

        //
        op1.GetProducer()->UpdateInputDesc(0, op1_input_0_desc);
        op1.GetProducer()->UpdateInputDesc(1, op1_input_0_desc);
        op1.GetProducer()->UpdateOutputDesc(0, op1_output_desc);
        op3.GetProducer()->UpdateInputDesc(0, op1_input_0_desc);
        op3.GetProducer()->UpdateInputDesc(1, op1_input_0_desc);
        op3.GetProducer()->UpdateOutputDesc(0, op1_output_desc);
        op4.GetProducer()->UpdateInputDesc(0, op1_input_0_desc);
        op4.GetProducer()->UpdateInputDesc(1, op1_input_0_desc);
        op4.GetProducer()->UpdateOutputDesc(0, op1_output_desc);
        op5.GetProducer()->UpdateInputDesc(0, op1_input_0_desc);
        op5.GetProducer()->UpdateInputDesc(1, op1_input_0_desc);
        op5.GetProducer()->UpdateOutputDesc(0, op1_output_desc);

        //op2(sum)
        TensorDesc op2_input_0_desc;
        op2.GetProducer()->GetInputDesc(0, op2_input_0_desc);
        op2_input_0_desc.SetDataType(dtype);
        op2_input_0_desc.SetShape(shape);
        op2.GetProducer()->UpdateInputDesc(0, op2_input_0_desc);

        std::vector<int64_t> dims1{1};
        Shape shape1(dims1);
        TensorDesc op2_input_1_desc;
        op2.GetProducer()->GetInputDesc(1, op2_input_1_desc);
        op2_input_1_desc.SetDataType(DT_INT64);
        op2_input_1_desc.SetShape(shape1);
        op2.GetProducer()->UpdateInputDesc(1, op2_input_1_desc);

        TensorDesc op2_output_0_desc;
        op2.GetProducer()->GetOutputDesc(0, op2_output_0_desc);
        op2_output_0_desc.SetDataType(dtype);
        op2_output_0_desc.SetShape(shape);
        op2.GetProducer()->UpdateOutputDesc(0, op2_output_0_desc);
    }

    bool IsSoftmaxGradExtInputRight(GNode& node, Shape& shape, DataType dtype)
    {
        TensorDesc input0_desc;
        TensorDesc input1_desc;
        TensorDesc input2_desc;

        node.GetInputDesc(0, input0_desc);
        node.GetInputDesc(1, input1_desc);
        node.GetInputDesc(2, input2_desc);

        if (input0_desc.GetDataType() != dtype ||
            input1_desc.GetDataType() != dtype ||
            input2_desc.GetDataType() != dtype) {
            return false;
        }

        if (input0_desc.GetShape().GetShapeSize() != shape.GetShapeSize() ||
            input1_desc.GetShape().GetShapeSize() != shape.GetShapeSize() ||
            input2_desc.GetShape().GetShapeSize() != shape.GetShapeSize()) {
            return false;
        }
        return true;
    }
};

// Test SoftmaxGradExtFusionPass
TEST_F(SoftmaxGradExtFusionPassTest, softmax_grad_ext_fusion_fp16_OK)
{
    std::vector<int64_t> dims{2, 32, 128};
    Shape shape(dims);

    auto graph_builder = es::EsGraphBuilder("softmax_grad_ext_fusion_test");
    auto input0 = graph_builder.CreateInput(0, "input0", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input1 = graph_builder.CreateInput(1, "input1", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input2 = graph_builder.CreateInput(2, "input2", DT_FLOAT16, FORMAT_ND, shape.GetDims());

    // Use operator overloads for Mul/Sub (internally uses EsMul, EsSub)
    auto mul = BuildTwoInOneOutNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildReduceSumForTest(graph_builder, mul, -1, true);
    auto sub = BuildTwoInOneOutNode(graph_builder, input0, sum, "Sub");
    auto mul1 = BuildTwoInOneOutNode(graph_builder, input2, input1, "Mul");
    auto mulGrad = BuildTwoInOneOutNode(graph_builder, mul1, sub, "Mul");

    InferShapeForTest(DT_FLOAT16, shape, input0, input1, input2, mul, sum, sub, mul1, mulGrad);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({mulGrad});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_softmax_grad_ext_test1");
    CustomPassContext pass_contex;
    ops::SoftmaxGradExtFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_softmax_grad_ext_test1");

    bool findSoftmaxGradExt = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "SoftmaxGradExt" && IsSoftmaxGradExtInputRight(node, shape, DT_FLOAT16)) {
            findSoftmaxGradExt = true;
        }
    }
    EXPECT_EQ(findSoftmaxGradExt, true);
    EXPECT_EQ(node_count, 5); // 3 inputs + 1 SoftmaxGradExt
}

TEST_F(SoftmaxGradExtFusionPassTest, softmax_grad_ext_fusion_fp32_OK)
{
    std::vector<int64_t> dims{1, 64, 256};
    Shape shape(dims);

    auto graph_builder = es::EsGraphBuilder("softmax_grad_ext_fusion_test");
    auto input0 = graph_builder.CreateInput(0, "input0", DT_FLOAT, FORMAT_ND, shape.GetDims());
    auto input1 = graph_builder.CreateInput(1, "input1", DT_FLOAT, FORMAT_ND, shape.GetDims());
    auto input2 = graph_builder.CreateInput(2, "input2", DT_FLOAT, FORMAT_ND, shape.GetDims());

    auto mul = BuildTwoInOneOutNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildReduceSumForTest(graph_builder, mul, -1, true);
    auto sub = BuildTwoInOneOutNode(graph_builder, input0, sum, "Sub");
    auto mul1 = BuildTwoInOneOutNode(graph_builder, input2, input1, "Mul");
    auto mulGrad = BuildTwoInOneOutNode(graph_builder, mul1, sub, "Mul");

    InferShapeForTest(DT_FLOAT, shape, input0, input1, input2, mul, sum, sub, mul1, mulGrad);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({mulGrad});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_softmax_grad_ext_test2");
    CustomPassContext pass_contex;
    ops::SoftmaxGradExtFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_softmax_grad_ext_test2");

    bool findSoftmaxGradExt = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "SoftmaxGradExt" && IsSoftmaxGradExtInputRight(node, shape, DT_FLOAT)) {
            findSoftmaxGradExt = true;
        }
    }
    EXPECT_EQ(findSoftmaxGradExt, true);
    EXPECT_EQ(node_count, 5);
}

TEST_F(SoftmaxGradExtFusionPassTest, softmax_grad_ext_fusion_unknown_shape_Fail)
{
    std::vector<int64_t> dims{-1, 32, 128};
    Shape shape(dims);

    auto graph_builder = es::EsGraphBuilder("softmax_grad_ext_fusion_test");
    auto input0 = graph_builder.CreateInput(0, "input0", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input1 = graph_builder.CreateInput(1, "input1", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input2 = graph_builder.CreateInput(2, "input2", DT_FLOAT16, FORMAT_ND, shape.GetDims());

    auto mul = BuildTwoInOneOutNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildReduceSumForTest(graph_builder, mul, -1, true);
    auto sub = BuildTwoInOneOutNode(graph_builder, input0, sum, "Sub");
    auto mul1 = BuildTwoInOneOutNode(graph_builder, input2, input1, "Mul");
    auto mulGrad = BuildTwoInOneOutNode(graph_builder, mul1, sub, "Mul");

    InferShapeForTest(DT_FLOAT, shape, input0, input1, input2, mul, sum, sub, mul1, mulGrad);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({mulGrad});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_softmax_grad_ext_test3");
    CustomPassContext pass_contex;
    ops::SoftmaxGradExtFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test SoftmaxGradExtV2FusionPass
TEST_F(SoftmaxGradExtFusionPassTest, softmax_grad_ext_v2_fusion_variant0_OK)
{
    std::vector<int64_t> dims{2, 32, 128};
    Shape shape(dims);

    auto graph_builder = es::EsGraphBuilder("softmax_grad_ext_v2_fusion_test");
    auto input0 = graph_builder.CreateInput(0, "input0", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input1 = graph_builder.CreateInput(1, "input1", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input2 = graph_builder.CreateInput(2, "input2", DT_FLOAT16, FORMAT_ND, shape.GetDims());

    auto mul = BuildTwoInOneOutNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildReduceSumForTest(graph_builder, mul, -1, true);
    auto sub = BuildTwoInOneOutNode(graph_builder, input0, sum, "Sub");
    auto mul1 = BuildTwoInOneOutNode(graph_builder, input1, sub, "Mul");
    auto mulGrad = BuildTwoInOneOutNode(graph_builder, mul1, input2, "Mul");

    InferShapeForTest(DT_FLOAT16, shape, input0, input1, input2, mul, sum, sub, mul1, mulGrad);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({mulGrad});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_softmax_grad_ext_v2_test1");
    CustomPassContext pass_contex;
    ops::SoftmaxGradExtV2FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_softmax_grad_ext_v2_test1");

    bool findSoftmaxGradExt = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "SoftmaxGradExt" && IsSoftmaxGradExtInputRight(node, shape, DT_FLOAT16)) {
            findSoftmaxGradExt = true;
        }
    }
    EXPECT_EQ(findSoftmaxGradExt, true);
    EXPECT_EQ(node_count, 5);
}

TEST_F(SoftmaxGradExtFusionPassTest, softmax_grad_ext_v2_fusion_variant1_OK)
{
    std::vector<int64_t> dims{1, 64, 256};
    Shape shape(dims);

    auto graph_builder = es::EsGraphBuilder("softmax_grad_ext_v2_fusion_test");
    auto input0 = graph_builder.CreateInput(0, "input0", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input1 = graph_builder.CreateInput(1, "input1", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input2 = graph_builder.CreateInput(2, "input2", DT_FLOAT16, FORMAT_ND, shape.GetDims());

    auto mul = BuildTwoInOneOutNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildReduceSumForTest(graph_builder, mul, -1, true);
    auto sub = BuildTwoInOneOutNode(graph_builder, input0, sum, "Sub");
    auto mul1 = BuildTwoInOneOutNode(graph_builder, sub, input1, "Mul");
    auto mulGrad = BuildTwoInOneOutNode(graph_builder, mul1, input2, "Mul");

    InferShapeForTest(DT_FLOAT16, shape, input0, input1, input2, mul, sum, sub, mul1, mulGrad);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({mulGrad});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_softmax_grad_ext_v2_test2");
    CustomPassContext pass_contex;
    ops::SoftmaxGradExtV2FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_softmax_grad_ext_v2_test2");

    bool findSoftmaxGradExt = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "SoftmaxGradExt" && IsSoftmaxGradExtInputRight(node, shape, DT_FLOAT16)) {
            findSoftmaxGradExt = true;
        }
    }
    EXPECT_EQ(findSoftmaxGradExt, true);
    EXPECT_EQ(node_count, 5);
}

TEST_F(SoftmaxGradExtFusionPassTest, softmax_grad_ext_v2_fusion_variant2_OK)
{
    std::vector<int64_t> dims{2, 32, 128};
    Shape shape(dims);

    auto graph_builder = es::EsGraphBuilder("softmax_grad_ext_v2_fusion_test");
    auto input0 = graph_builder.CreateInput(0, "input0", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input1 = graph_builder.CreateInput(1, "input1", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input2 = graph_builder.CreateInput(2, "input2", DT_FLOAT16, FORMAT_ND, shape.GetDims());

    auto mul = BuildTwoInOneOutNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildReduceSumForTest(graph_builder, mul, -1, true);
    auto sub = BuildTwoInOneOutNode(graph_builder, input0, sum, "Sub");
    auto mul1 = BuildTwoInOneOutNode(graph_builder, input1, sub, "Mul");
    auto mulGrad = BuildTwoInOneOutNode(graph_builder, input2, mul1, "Mul");

    InferShapeForTest(DT_FLOAT16, shape, input0, input1, input2, mul, sum, sub, mul1, mulGrad);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({mulGrad});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_softmax_grad_ext_v2_test3");
    CustomPassContext pass_contex;
    ops::SoftmaxGradExtV2FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_softmax_grad_ext_v2_test3");

    bool findSoftmaxGradExt = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "SoftmaxGradExt" && IsSoftmaxGradExtInputRight(node, shape, DT_FLOAT16)) {
            findSoftmaxGradExt = true;
        }
    }
    EXPECT_EQ(findSoftmaxGradExt, true);
    EXPECT_EQ(node_count, 5);
}

TEST_F(SoftmaxGradExtFusionPassTest, softmax_grad_ext_v2_fusion_variant3_OK)
{
    std::vector<int64_t> dims{1, 64, 256};
    Shape shape(dims);

    auto graph_builder = es::EsGraphBuilder("softmax_grad_ext_v2_fusion_test");
    auto input0 = graph_builder.CreateInput(0, "input0", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input1 = graph_builder.CreateInput(1, "input1", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input2 = graph_builder.CreateInput(2, "input2", DT_FLOAT16, FORMAT_ND, shape.GetDims());

    auto mul = BuildTwoInOneOutNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildReduceSumForTest(graph_builder, mul, -1, true);
    auto sub = BuildTwoInOneOutNode(graph_builder, input0, sum, "Sub");
    auto mul1 = BuildTwoInOneOutNode(graph_builder, input1, sub, "Mul");
    auto mulGrad = BuildTwoInOneOutNode(graph_builder, input2, mul1, "Mul");

    InferShapeForTest(DT_FLOAT16, shape, input0, input1, input2, mul, sum, sub, mul1, mulGrad);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({mulGrad});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_softmax_grad_ext_v2_test4");
    CustomPassContext pass_contex;
    ops::SoftmaxGradExtV2FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_softmax_grad_ext_v2_test4");

    bool findSoftmaxGradExt = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "SoftmaxGradExt" && IsSoftmaxGradExtInputRight(node, shape, DT_FLOAT16)) {
            findSoftmaxGradExt = true;
        }
    }
    EXPECT_EQ(findSoftmaxGradExt, true);
    EXPECT_EQ(node_count, 5);
}

TEST_F(SoftmaxGradExtFusionPassTest, softmax_grad_ext_v2_fusion_unknown_shape_Fail)
{
    std::vector<int64_t> dims{-1, 32, 128};
    Shape shape(dims);

    auto graph_builder = es::EsGraphBuilder("softmax_grad_ext_v2_fusion_test");
    auto input0 = graph_builder.CreateInput(0, "input0", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input1 = graph_builder.CreateInput(1, "input1", DT_FLOAT16, FORMAT_ND, shape.GetDims());
    auto input2 = graph_builder.CreateInput(2, "input2", DT_FLOAT16, FORMAT_ND, shape.GetDims());

    auto mul = BuildTwoInOneOutNode(graph_builder, input0, input1, "Mul");
    auto sum = BuildReduceSumForTest(graph_builder, mul, -1, true);
    auto sub = BuildTwoInOneOutNode(graph_builder, input0, sum, "Sub");
    auto mul1 = BuildTwoInOneOutNode(graph_builder, input1, sub, "Mul");
    auto mulGrad = BuildTwoInOneOutNode(graph_builder, mul1, input2, "Mul");

    InferShapeForTest(DT_FLOAT16, shape, input0, input1, input2, mul, sum, sub, mul1, mulGrad);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({mulGrad});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_softmax_grad_ext_v2_test5");
    CustomPassContext pass_contex;
    ops::SoftmaxGradExtV2FusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}