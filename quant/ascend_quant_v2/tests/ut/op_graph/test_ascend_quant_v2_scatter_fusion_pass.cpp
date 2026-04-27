/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_nn_ops.h"
#include "compliant_node_builder.h"
#include "../../../op_graph/fusion_pass/ascend_quant_v2_scatter_fusion_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace es;
using namespace ops;

class AscendQuantV2ScatterFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void InferShapeForTest(
        DataType dtype1, Shape& shape1, DataType dtype2, Shape& shape2, DataType dtype3, Shape& shape3,
        es::EsTensorHolder& input0, es::EsTensorHolder& input1, es::EsTensorHolder& input2, es::EsTensorHolder& input3,
        es::EsTensorHolder& op1, es::EsTensorHolder& op2)
    {
        //input0
        TensorDesc x1_output_desc;
        input0.GetProducer()->GetOutputDesc(0, x1_output_desc);
        x1_output_desc.SetDataType(dtype1);
        x1_output_desc.SetShape(shape1);
        input0.GetProducer()->UpdateOutputDesc(0, x1_output_desc);
        //input1
        TensorDesc x2_output_desc;
        input1.GetProducer()->GetOutputDesc(0, x2_output_desc);
        x2_output_desc.SetDataType(dtype1);
        x2_output_desc.SetShape(shape2);
        input1.GetProducer()->UpdateOutputDesc(0, x2_output_desc);
        //input2
        TensorDesc x3_output_desc;
        input2.GetProducer()->GetOutputDesc(0, x3_output_desc);
        x3_output_desc.SetDataType(dtype2);
        x3_output_desc.SetShape(shape3);
        input2.GetProducer()->UpdateOutputDesc(0, x3_output_desc);
        //input3
        TensorDesc x4_output_desc;
        input3.GetProducer()->GetOutputDesc(0, x4_output_desc);
        x4_output_desc.SetDataType(dtype3);
        x4_output_desc.SetShape(shape2);
        input3.GetProducer()->UpdateOutputDesc(0, x4_output_desc);

        // update desc
        TensorDesc update_desc;
        op1.GetProducer()->GetInputDesc(0, update_desc);
        update_desc.SetDataType(dtype2);
        update_desc.SetShape(shape1);

        // ascendquantv2
        op1.GetProducer()->UpdateInputDesc(0, x1_output_desc);
        op1.GetProducer()->UpdateInputDesc(1, x2_output_desc);
        op1.GetProducer()->UpdateOutputDesc(0, update_desc);
        bool sqrt_mode = false;
        ge::AscendString round_mode = "round";
        int dst_type = 2;
        int quant_axis = -1;
        op1.GetProducer()->SetAttr("sqrt_mode", sqrt_mode);
        op1.GetProducer()->SetAttr("round_mode", round_mode);
        op1.GetProducer()->SetAttr("dst_type", dst_type);
        op1.GetProducer()->SetAttr("axis", quant_axis);

        // scatter
        op2.GetProducer()->UpdateInputDesc(0, x3_output_desc);
        op2.GetProducer()->UpdateInputDesc(1, x4_output_desc);
        op2.GetProducer()->UpdateInputDesc(2, update_desc);
        op2.GetProducer()->UpdateOutputDesc(0, x3_output_desc);
        ge::AscendString reduce = "update";
        int scatter_axis = -2;
        op2.GetProducer()->SetAttr("reduce", reduce);
        op2.GetProducer()->SetAttr("axis", scatter_axis);
    }

    es::EsTensorHolder BuildAscendQuantV2(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x,
                                      const es::EsTensorHolder& scale)
    {
        auto graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();

        // Build AscendQuantV2 node
        auto quantV2Node = es::CompliantNodeBuilder(graphPtr)
            .OpType("AscendQuantV2")
            .Name("AscendQuantV2")
            .IrDefInputs({
                {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"scale", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"offset", es::CompliantNodeBuilder::kEsIrInputOptional, ""}
            })
            .IrDefOutputs({
                {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
            })
            .Build();

        // Connect inputs to AscendQuantV2
        es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), x.GetProducerOutIndex(), quantV2Node, 0);
        es::AddEdgeAndUpdatePeerDesc(*graphPtr, *scale.GetProducer(), scale.GetProducerOutIndex(), quantV2Node, 1);

        return es::EsTensorHolder(
            graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(quantV2Node, 0));
    }

    es::EsTensorHolder BuildScatter(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& var,
                                        const es::EsTensorHolder& indices, const es::EsTensorHolder& ascend_quant_v2)
    {
        auto graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();
        // Build Scatter node (updates from AscendQuantV2 output)
        auto scatterNode = es::CompliantNodeBuilder(graphPtr)
            .OpType("Scatter")
            .Name("Scatter")
            .IrDefInputs({
                {"var", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"updates", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
            })
            .IrDefOutputs({
                {"var", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
            })
            .IrDefAttrs({
                {"reduce", es::CompliantNodeBuilder::kEsAttrOptional, "String", es::CreateFrom(ge::AscendString("update"))},
                {"axis", es::CompliantNodeBuilder::kEsAttrOptional, "Int", es::CreateFrom(static_cast<int64_t>(0))}
            })
            .Build();

        // Connect inputs to Scatter
        es::AddEdgeAndUpdatePeerDesc(*graphPtr, *var.GetProducer(), var.GetProducerOutIndex(), scatterNode, 0);
        es::AddEdgeAndUpdatePeerDesc(*graphPtr, *indices.GetProducer(), indices.GetProducerOutIndex(), scatterNode, 1);
        es::AddEdgeAndUpdatePeerDesc(*graphPtr, *ascend_quant_v2.GetProducer(), ascend_quant_v2.GetProducerOutIndex(), scatterNode, 2);

        // Build graph with scatter output
        auto scatterOutput = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(scatterNode, 0));
        return scatterOutput;
    }
};

// Test 1: Pattern creation test
TEST_F(AscendQuantV2ScatterFusionPassTest, pattern_test) {
    ops::AscendQuantV2ScatterFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

// Test 2: Pattern structure test - verify pattern has correct structure
TEST_F(AscendQuantV2ScatterFusionPassTest, pattern_structure_test) {
    ops::AscendQuantV2ScatterFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_EQ(patterns.size(), 1);  // Should have one pattern for AscendQuantV2 + Scatter
}

TEST_F(AscendQuantV2ScatterFusionPassTest, ascend_quant_v2_scatter_fusion_OK)
{
    std::vector<int64_t> dims1{16, 1, 16};
    std::vector<int64_t> dims2{16};
    std::vector<int64_t> dims3{16, 16, 16};
    Shape shape1(dims1);
    Shape shape2(dims2);
    Shape shape3(dims3);

    auto graph_builder = es::EsGraphBuilder("ascend_quant_v2_scatter_fusion_fusion_test");
    auto x = graph_builder.CreateInput(0, "input0", DT_BF16, FORMAT_ND, shape1.GetDims());
    auto scale = graph_builder.CreateInput(1, "input1", DT_BF16, FORMAT_ND, shape2.GetDims());
    auto var = graph_builder.CreateInput(2, "input2", DT_INT8, FORMAT_ND, shape3.GetDims());
    auto indices = graph_builder.CreateInput(3, "input3", DT_INT64, FORMAT_ND, shape2.GetDims());

    auto ascend_quant_v2 = BuildAscendQuantV2(graph_builder, x, scale);
    auto scatter = BuildScatter(graph_builder, var, indices, ascend_quant_v2);

    InferShapeForTest(
        DT_BF16, shape1, DT_INT8, shape2, DT_INT64, shape3, x, scale, var, indices, ascend_quant_v2, scatter);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({scatter});

    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_ascend_quant_v2_scatter_test1");
    CustomPassContext pass_contex;
    ops::AscendQuantV2ScatterFusionPass pass;
    Status status = pass.Run(graph, pass_contex);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_ascend_quant_v2_scatter_test1");

    bool findQuantUpdateScatter = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "QuantUpdateScatter") {
            findQuantUpdateScatter = true;
        }
    }
    EXPECT_EQ(findQuantUpdateScatter, true);
    EXPECT_EQ(node_count, 6);
}