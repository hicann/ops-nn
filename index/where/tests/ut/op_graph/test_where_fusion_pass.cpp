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
#include "es_nn_ops.h"
#include "../../../op_graph/fusion_pass/where_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

class WhereFusionPassTest : public testing::Test {
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

TEST_F(WhereFusionPassTest, where_fusion_float_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test1");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test1");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_float16_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT16);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test2");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test2");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_bf16_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_BF16, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_BF16);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test3");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test3");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_int8_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_INT8, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_INT8);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test4");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test4");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_int16_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_INT16, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_INT16);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test5");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test5");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_int32_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_INT32, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_INT32);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test6");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test6");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_int64_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_INT64, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_INT64);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test7");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test7");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_uint8_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_UINT8, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_UINT8);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test8");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test8");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_uint16_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_UINT16, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_UINT16);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test9");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test9");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_uint32_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_UINT32, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_UINT32);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test10");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test10");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_uint64_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_UINT64, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_UINT64);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test11");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test11");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_bool_success)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_BOOL, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_BOOL);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test12");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test12");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_950_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test13");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test13");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_unsupported_dtype_fail)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_DOUBLE, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_DOUBLE);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test14");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test14");
}

TEST_F(WhereFusionPassTest, where_fusion_complex64_fail)
{
    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_COMPLEX64, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_COMPLEX64);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test15");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test15");
}

TEST_F(WhereFusionPassTest, where_fusion_unsupported_platform_fail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dims_x{2, 3, 4};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test16");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test16");
}

TEST_F(WhereFusionPassTest, where_fusion_pattern_test)
{
    ops::WhereFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(WhereFusionPassTest, where_fusion_scalar_success)
{
    std::vector<int64_t> dims_x;
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test17");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test17");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_1d_success)
{
    std::vector<int64_t> dims_x{8};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test18");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test18");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}

TEST_F(WhereFusionPassTest, where_fusion_high_dim_success)
{
    std::vector<int64_t> dims_x{2, 3, 4, 5, 6};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);

    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset({where_output});
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_where_test19");
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_where_test19");

    bool findNonZero = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            findNonZero = true;
        }
    }
    EXPECT_EQ(findNonZero, true);
}
