/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_nn_ops.h"
#include "compliant_node_builder.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/tensor_scatter_add_fusion_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace OPS::NN;

// ---------------------------------------------------------------------------
// 辅助函数：构建包含指定 op_type 节点的测试图
// TensorScatterAdd / ScatterNonAliasingAdd 无 ES API，使用 CompliantNodeBuilder
// ---------------------------------------------------------------------------
static std::shared_ptr<Graph> BuildTestGraph(
    const std::string& op_type,
    DataType x_dtype,
    const std::vector<int64_t>& x_dims,
    const std::vector<int64_t>& indices_dims,
    const std::vector<int64_t>& updates_dims)
{
    auto graph_builder = es::EsGraphBuilder("test_graph");
    auto x = graph_builder.CreateInput(0, "x", x_dtype, FORMAT_ND, x_dims);
    auto indices = graph_builder.CreateInput(1, "indices", DT_INT32, FORMAT_ND, indices_dims);
    auto updates = graph_builder.CreateInput(2, "updates", x_dtype, FORMAT_ND, updates_dims);

    ge::Graph* graph_ptr = graph_builder.GetCGraphBuilder()->GetGraph();

    GNode op_node = es::CompliantNodeBuilder(graph_ptr)
        .OpType(op_type.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                       {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                       {"updates", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();

    GNode x_node = *x.GetProducer();
    GNode indices_node = *indices.GetProducer();
    GNode updates_node = *updates.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graph_ptr, x_node, 0, op_node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph_ptr, indices_node, 0, op_node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph_ptr, updates_node, 0, op_node, 2);

    // 设置输入节点的 OutputDesc（参考 where 示例）
    TensorDesc x_out_desc;
    x_node.GetOutputDesc(0, x_out_desc);
    x_out_desc.SetDataType(x_dtype);
    x_out_desc.SetShape(Shape(x_dims));
    x_node.UpdateOutputDesc(0, x_out_desc);

    TensorDesc indices_out_desc;
    indices_node.GetOutputDesc(0, indices_out_desc);
    indices_out_desc.SetDataType(DT_INT32);
    indices_out_desc.SetShape(Shape(indices_dims));
    indices_node.UpdateOutputDesc(0, indices_out_desc);

    TensorDesc updates_out_desc;
    updates_node.GetOutputDesc(0, updates_out_desc);
    updates_out_desc.SetDataType(x_dtype);
    updates_out_desc.SetShape(Shape(updates_dims));
    updates_node.UpdateOutputDesc(0, updates_out_desc);

    // 设置算子节点的输入 TensorDesc
    TensorDesc op_in0_desc;
    op_node.GetInputDesc(0, op_in0_desc);
    op_in0_desc.SetDataType(x_dtype);
    op_in0_desc.SetShape(Shape(x_dims));
    op_in0_desc.SetFormat(FORMAT_ND);
    op_node.UpdateInputDesc(0, op_in0_desc);

    TensorDesc op_in1_desc;
    op_node.GetInputDesc(1, op_in1_desc);
    op_in1_desc.SetDataType(DT_INT32);
    op_in1_desc.SetShape(Shape(indices_dims));
    op_in1_desc.SetFormat(FORMAT_ND);
    op_node.UpdateInputDesc(1, op_in1_desc);

    TensorDesc op_in2_desc;
    op_node.GetInputDesc(2, op_in2_desc);
    op_in2_desc.SetDataType(x_dtype);
    op_in2_desc.SetShape(Shape(updates_dims));
    op_in2_desc.SetFormat(FORMAT_ND);
    op_node.UpdateInputDesc(2, op_in2_desc);

    // 设置算子节点的输出 TensorDesc
    TensorDesc op_out_desc;
    op_node.GetOutputDesc(0, op_out_desc);
    op_out_desc.SetDataType(x_dtype);
    op_out_desc.SetShape(Shape(x_dims));
    op_out_desc.SetFormat(FORMAT_ND);
    op_node.UpdateOutputDesc(0, op_out_desc);

    es::EsTensorHolder output(graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(op_node, 0));
    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({output});
    return graph;
}

// ---------------------------------------------------------------------------
// 辅助函数：在图中查找指定类型的节点
// ---------------------------------------------------------------------------
static bool FindNodeByType(const std::shared_ptr<Graph>& graph, const std::string& type_name)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == type_name.c_str()) {
            return true;
        }
    }
    return false;
}

// ===========================================================================
// 测试类
// ===========================================================================
class TensorScatterAddFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

// ===========================================================================
// 测试用例
// ===========================================================================

TEST_F(TensorScatterAddFusionPassTest, pattern_test)
{
    OPS::NN::TensorScatterAddFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_EQ(patterns.size(), 2);
}

TEST_F(TensorScatterAddFusionPassTest, tensor_scatter_add_float_success)
{
    auto graph = BuildTestGraph("TensorScatterAdd", DT_FLOAT, {4, 4, 4}, {2, 1}, {2, 4, 4});
    CustomPassContext pass_context;
    OPS::NN::TensorScatterAddFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterNdAdd"));
    EXPECT_FALSE(FindNodeByType(graph, "TensorScatterAdd"));
}

TEST_F(TensorScatterAddFusionPassTest, tensor_scatter_add_float16_success)
{
    auto graph = BuildTestGraph("TensorScatterAdd", DT_FLOAT16, {8, 16}, {4, 1}, {4, 16});
    CustomPassContext pass_context;
    OPS::NN::TensorScatterAddFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterNdAdd"));
    EXPECT_FALSE(FindNodeByType(graph, "TensorScatterAdd"));
}

TEST_F(TensorScatterAddFusionPassTest, scatter_non_aliasing_add_success)
{
    auto graph = BuildTestGraph("ScatterNonAliasingAdd", DT_FLOAT, {4, 4, 4}, {2, 1}, {2, 4, 4});
    CustomPassContext pass_context;
    OPS::NN::TensorScatterAddFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterNdAdd"));
    EXPECT_FALSE(FindNodeByType(graph, "ScatterNonAliasingAdd"));
}

TEST_F(TensorScatterAddFusionPassTest, tensor_scatter_add_ascend950_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto graph = BuildTestGraph("TensorScatterAdd", DT_FLOAT, {4, 4, 4}, {2, 1}, {2, 4, 4});
    CustomPassContext pass_context;
    OPS::NN::TensorScatterAddFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterNdAdd"));
}

TEST_F(TensorScatterAddFusionPassTest, unsupported_platform_fail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto graph = BuildTestGraph("TensorScatterAdd", DT_FLOAT, {4, 4, 4}, {2, 1}, {2, 4, 4});
    CustomPassContext pass_context;
    OPS::NN::TensorScatterAddFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(TensorScatterAddFusionPassTest, tensor_scatter_add_1d_success)
{
    auto graph = BuildTestGraph("TensorScatterAdd", DT_FLOAT, {8}, {3, 1}, {3});
    CustomPassContext pass_context;
    OPS::NN::TensorScatterAddFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterNdAdd"));
}

TEST_F(TensorScatterAddFusionPassTest, tensor_scatter_add_high_dim_success)
{
    auto graph = BuildTestGraph("TensorScatterAdd", DT_FLOAT, {2, 3, 4, 5}, {2, 2}, {2, 4, 5});
    CustomPassContext pass_context;
    OPS::NN::TensorScatterAddFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterNdAdd"));
}

TEST_F(TensorScatterAddFusionPassTest, tensor_scatter_add_int32_success)
{
    auto graph = BuildTestGraph("TensorScatterAdd", DT_INT32, {4, 4}, {2, 1}, {2, 4});
    CustomPassContext pass_context;
    OPS::NN::TensorScatterAddFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterNdAdd"));
}
