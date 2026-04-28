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
#include "../../../op_graph/fusion_pass/gather_fusion_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace OPS::NN;

// ---------------------------------------------------------------------------
// 辅助函数：构建包含 Gather 节点的测试图
// Gather 无 ES API，使用 CompliantNodeBuilder
// ---------------------------------------------------------------------------
static std::shared_ptr<Graph> BuildGatherTestGraph(
    DataType x_dtype,
    const std::vector<int64_t>& x_dims,
    const std::vector<int64_t>& indices_dims,
    int64_t batch_dims = 0,
    bool is_preprocessed = false,
    bool negative_index_support = false)
{
    auto graph_builder = es::EsGraphBuilder("test_graph");
    auto x = graph_builder.CreateInput(0, "x", x_dtype, FORMAT_ND, x_dims);
    auto indices = graph_builder.CreateInput(1, "indices", DT_INT32, FORMAT_ND, indices_dims);

    ge::Graph* graph_ptr = graph_builder.GetCGraphBuilder()->GetGraph();

    GNode op_node = es::CompliantNodeBuilder(graph_ptr)
        .OpType("Gather")
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                       {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"batch_dims", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
             es::CreateFrom(batch_dims)},
            {"is_preprocessed", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
             es::CreateFrom(is_preprocessed)},
            {"negative_index_support", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
             es::CreateFrom(negative_index_support)}
        })
        .Build();

    GNode x_node = *x.GetProducer();
    GNode indices_node = *indices.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graph_ptr, x_node, 0, op_node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph_ptr, indices_node, 0, op_node, 1);

    // 设置输入节点的 OutputDesc
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

    // 设置算子节点的输出 TensorDesc
    // Gather 输出 shape: x.shape[:axis] + indices.shape[batch_dims:] + x.shape[axis+1:]
    // 因为 axis=0（Gather 默认 axis=0），输出 shape = indices.shape + x.shape[1:]
    std::vector<int64_t> output_dims;
    for (const auto& d : indices_dims) {
        output_dims.push_back(d);
    }
    for (size_t i = 1; i < x_dims.size(); ++i) {
        output_dims.push_back(x_dims[i]);
    }

    TensorDesc op_out_desc;
    op_node.GetOutputDesc(0, op_out_desc);
    op_out_desc.SetDataType(x_dtype);
    op_out_desc.SetShape(Shape(output_dims));
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
class GatherToGatherV2FusionPassTest : public testing::Test {
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

// 验证 Pattern 能正确创建
TEST_F(GatherToGatherV2FusionPassTest, pattern_test)
{
    OPS::NN::GatherToGatherV2FusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_EQ(patterns.size(), 1);
}

// 正常融合：float 类型
TEST_F(GatherToGatherV2FusionPassTest, gather_float_fusion_success)
{
    auto graph = BuildGatherTestGraph(DT_FLOAT, {8, 16}, {4});
    CustomPassContext pass_context;
    OPS::NN::GatherToGatherV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "GatherV2"));
    EXPECT_FALSE(FindNodeByType(graph, "Gather"));
}

// 正常融合：float16 类型
TEST_F(GatherToGatherV2FusionPassTest, gather_float16_fusion_success)
{
    auto graph = BuildGatherTestGraph(DT_FLOAT16, {4, 8, 16}, {2, 3});
    CustomPassContext pass_context;
    OPS::NN::GatherToGatherV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "GatherV2"));
    EXPECT_FALSE(FindNodeByType(graph, "Gather"));
}

// 正常融合：int32 类型
TEST_F(GatherToGatherV2FusionPassTest, gather_int32_fusion_success)
{
    auto graph = BuildGatherTestGraph(DT_INT32, {16}, {4});
    CustomPassContext pass_context;
    OPS::NN::GatherToGatherV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "GatherV2"));
    EXPECT_FALSE(FindNodeByType(graph, "Gather"));
}

// Ascend950 平台融合成功
TEST_F(GatherToGatherV2FusionPassTest, gather_ascend950_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto graph = BuildGatherTestGraph(DT_FLOAT, {8, 16}, {4});
    CustomPassContext pass_context;
    OPS::NN::GatherToGatherV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "GatherV2"));
    EXPECT_FALSE(FindNodeByType(graph, "Gather"));
}

// 不支持的平台（Ascend910B）
TEST_F(GatherToGatherV2FusionPassTest, unsupported_platform_fail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto graph = BuildGatherTestGraph(DT_FLOAT, {8, 16}, {4});
    CustomPassContext pass_context;
    OPS::NN::GatherToGatherV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// 高维输入融合成功
TEST_F(GatherToGatherV2FusionPassTest, gather_high_dim_success)
{
    auto graph = BuildGatherTestGraph(DT_FLOAT, {2, 3, 4, 5}, {6});
    CustomPassContext pass_context;
    OPS::NN::GatherToGatherV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "GatherV2"));
    EXPECT_FALSE(FindNodeByType(graph, "Gather"));
}

// 属性传递：batch_dims 非零
TEST_F(GatherToGatherV2FusionPassTest, gather_with_batch_dims_success)
{
    auto graph = BuildGatherTestGraph(DT_FLOAT, {4, 8, 16}, {4, 2}, 1);
    CustomPassContext pass_context;
    OPS::NN::GatherToGatherV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "GatherV2"));
    EXPECT_FALSE(FindNodeByType(graph, "Gather"));
}

// 1D 输入融合成功
TEST_F(GatherToGatherV2FusionPassTest, gather_1d_success)
{
    auto graph = BuildGatherTestGraph(DT_FLOAT, {100}, {10});
    CustomPassContext pass_context;
    OPS::NN::GatherToGatherV2FusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "GatherV2"));
    EXPECT_FALSE(FindNodeByType(graph, "Gather"));
}
