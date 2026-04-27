/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with the License.
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
#include "es_nn_ops.h"
#include "compliant_node_builder.h"
#include "../../../op_graph/fusion_pass/prelu_grad_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace es;
using namespace ops;

class PReluGradFusionPassTest : public testing::Test {
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

    std::shared_ptr<Graph> BuildPReluGradGraph(DataType dtype) {
        std::vector<int64_t> dims_dy{2, 3, 4};
        std::vector<int64_t> dims_x{2, 3, 4};
        std::vector<int64_t> dims_weight{4};
        Shape shape_dy(dims_dy);
        Shape shape_x(dims_x);
        Shape shape_weight(dims_weight);

        EsGraphBuilder graph_builder("prelu_grad_fusion_test");
        auto dy = graph_builder.CreateInput(0, "dy", dtype, FORMAT_ND, shape_dy.GetDims());
        auto x = graph_builder.CreateInput(1, "x", dtype, FORMAT_ND, shape_x.GetDims());
        auto weight = graph_builder.CreateInput(2, "weight", dtype, FORMAT_ND, shape_weight.GetDims());

        auto *graph = graph_builder.GetCGraphBuilder()->GetGraph();
        auto prelu_grad = CompliantNodeBuilder(graph)
            .OpType("PReluGrad")
            .Name("PReluGrad")
            .IrDefInputs({
                {"dy", CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"weight", CompliantNodeBuilder::kEsIrInputRequired, ""},
            })
            .IrDefOutputs({
                {"dx", CompliantNodeBuilder::kEsIrOutputRequired, ""},
                {"da", CompliantNodeBuilder::kEsIrOutputRequired, ""},
            })
            .Build();

        TensorDesc dy_output_desc;
        dy.GetProducer()->GetOutputDesc(0, dy_output_desc);
        dy_output_desc.SetDataType(dtype);
        dy_output_desc.SetShape(shape_dy);
        dy.GetProducer()->UpdateOutputDesc(0, dy_output_desc);

        TensorDesc x_output_desc;
        x.GetProducer()->GetOutputDesc(0, x_output_desc);
        x_output_desc.SetDataType(dtype);
        x_output_desc.SetShape(shape_x);
        x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

        TensorDesc weight_output_desc;
        weight.GetProducer()->GetOutputDesc(0, weight_output_desc);
        weight_output_desc.SetDataType(dtype);
        weight_output_desc.SetShape(shape_weight);
        weight.GetProducer()->UpdateOutputDesc(0, weight_output_desc);

        es::AddEdgeAndUpdatePeerDesc(*graph, *dy.GetProducer(), dy.GetProducerOutIndex(), prelu_grad, 0);
        es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), prelu_grad, 1);
        es::AddEdgeAndUpdatePeerDesc(*graph, *weight.GetProducer(), weight.GetProducerOutIndex(), prelu_grad, 2);
        
        auto *dx_holder = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(prelu_grad, 0);
        auto *da_holder = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(prelu_grad, 1);
        return graph_builder.BuildAndReset({es::EsTensorHolder(dx_holder), es::EsTensorHolder(da_holder)});
    }
};

TEST_F(PReluGradFusionPassTest, prelu_grad_fusion_pattern_test) {
    ops::PReluGradFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(PReluGradFusionPassTest, prelu_grad_fusion_float_success) {
    std::shared_ptr<Graph> graph = BuildPReluGradGraph(DT_FLOAT);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_prelu_grad_test1");
    CustomPassContext pass_context;
    ops::PReluGradFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_prelu_grad_test1");

    bool findPReluGradUpdate = false;
    bool findPReluGradReduce = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "PReluGradUpdate") {
            findPReluGradUpdate = true;
        }
        if (type == "PReluGradReduce") {
            findPReluGradReduce = true;
        }
    }
    EXPECT_EQ(findPReluGradUpdate, true);
    EXPECT_EQ(findPReluGradReduce, true);
}

TEST_F(PReluGradFusionPassTest, prelu_grad_fusion_float16_success) {
    std::shared_ptr<Graph> graph = BuildPReluGradGraph(DT_FLOAT16);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_prelu_grad_test2");
    CustomPassContext pass_context;
    ops::PReluGradFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_prelu_grad_test2");

    bool findPReluGradUpdate = false;
    bool findPReluGradReduce = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "PReluGradUpdate") {
            findPReluGradUpdate = true;
        }
        if (type == "PReluGradReduce") {
            findPReluGradReduce = true;
        }
    }
    EXPECT_EQ(findPReluGradUpdate, true);
    EXPECT_EQ(findPReluGradReduce, true);
}

TEST_F(PReluGradFusionPassTest, prelu_grad_fusion_950_success) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::shared_ptr<Graph> graph = BuildPReluGradGraph(DT_FLOAT);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_prelu_grad_test3");
    CustomPassContext pass_context;
    ops::PReluGradFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_prelu_grad_test3");

    bool findPReluGradUpdate = false;
    bool findPReluGradReduce = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "PReluGradUpdate") {
            findPReluGradUpdate = true;
        }
        if (type == "PReluGradReduce") {
            findPReluGradReduce = true;
        }
    }
    EXPECT_EQ(findPReluGradUpdate, true);
    EXPECT_EQ(findPReluGradReduce, true);
}

TEST_F(PReluGradFusionPassTest, prelu_grad_fusion_unsupported_platform_fail) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_93";
    optiCompilationInfo.soc_version = "Ascend910_93";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::shared_ptr<Graph> graph = BuildPReluGradGraph(DT_FLOAT);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_prelu_grad_test4");
    CustomPassContext pass_context;
    ops::PReluGradFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_prelu_grad_test4");
}
