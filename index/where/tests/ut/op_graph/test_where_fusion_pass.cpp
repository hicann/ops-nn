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
#include "platform/soc_spec.h"
#include "ge/es_graph_builder.h"
#include "es_nn_ops.h"
#include "../../../op_graph/fusion_pass/where_fusion_pass.h"
#include "register/register_custom_pass.h"
#include "runtime/runtime/base.h"

namespace {

thread_local NpuArch g_npuArch_mock = NpuArch::DAV_3510;

class NpuArchManagerMock {
public:
    explicit NpuArchManagerMock(NpuArch newArch) : originalArch_(g_npuArch_mock) { g_npuArch_mock = newArch; }
    ~NpuArchManagerMock() { g_npuArch_mock = originalArch_; }

private:
    NpuArch originalArch_;
    NpuArchManagerMock(const NpuArchManagerMock&) = delete;
    NpuArchManagerMock& operator=(const NpuArchManagerMock&) = delete;
};

void SetPlatform(const std::string& soc)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = soc;
    optiCompilationInfo.soc_version = soc;
    fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

bool CheckNonZeroNode(const std::shared_ptr<Graph>& graph)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "NonZero") {
            return true;
        }
    }
    return false;
}

std::shared_ptr<Graph> BuildWhereTestGraph(DataType dtype, const std::vector<int64_t>& dims)
{
    Shape shape_x(dims);
    auto graph_builder = es::EsGraphBuilder("where_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", dtype, FORMAT_ND, shape_x.GetDims());
    auto where_output = es::Where(x);

    GNode x_node = *x.GetProducer();
    GNode where_node = *where_output.GetProducer();

    TensorDesc x_out_desc;
    x_node.GetOutputDesc(0, x_out_desc);
    x_out_desc.SetDataType(dtype);
    x_out_desc.SetShape(shape_x);
    x_node.UpdateOutputDesc(0, x_out_desc);

    TensorDesc where_in_desc;
    where_node.GetInputDesc(0, where_in_desc);
    where_in_desc.SetDataType(dtype);
    where_in_desc.SetShape(shape_x);
    where_in_desc.SetFormat(FORMAT_ND);
    where_node.UpdateInputDesc(0, where_in_desc);

    return graph_builder.BuildAndReset({where_output});
}

} // namespace

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

class WhereFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatform("Ascend950"); }

    void SetUp() override { SetPlatform("Ascend950"); }

    void TearDown() override {}
};

TEST_F(WhereFusionPassTest, where_fusion_float_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_FLOAT, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_float16_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_FLOAT16, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_bf16_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_BF16, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_int8_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_INT8, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_int16_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_INT16, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_int32_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_INT32, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_int64_fail)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_INT64, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CheckNonZeroNode(graph), false);
}

TEST_F(WhereFusionPassTest, where_fusion_uint8_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_UINT8, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_uint16_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_UINT16, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_uint32_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_UINT32, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_uint64_fail)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_UINT64, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CheckNonZeroNode(graph), false);
}

TEST_F(WhereFusionPassTest, where_fusion_bool_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_BOOL, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_950_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    SetPlatform("Ascend950");
    auto graph = BuildWhereTestGraph(DT_FLOAT, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_unsupported_dtype_fail)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_DOUBLE, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CheckNonZeroNode(graph), false);
}

TEST_F(WhereFusionPassTest, where_fusion_complex64_fail)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_COMPLEX64, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CheckNonZeroNode(graph), false);
}

TEST_F(WhereFusionPassTest, where_fusion_unsupported_platform_fail)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_2201);
    SetPlatform("Ascend910B");
    auto graph = BuildWhereTestGraph(DT_FLOAT, {2, 3, 4});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CheckNonZeroNode(graph), false);
}

TEST_F(WhereFusionPassTest, where_fusion_pattern_test)
{
    ops::WhereFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(WhereFusionPassTest, where_fusion_scalar_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_FLOAT, {});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_1d_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_FLOAT, {8});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}

TEST_F(WhereFusionPassTest, where_fusion_high_dim_success)
{
    NpuArchManagerMock archMgr(NpuArch::DAV_3510);
    auto graph = BuildWhereTestGraph(DT_FLOAT, {2, 3, 4, 5, 6});
    CustomPassContext pass_context;
    ops::WhereFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CheckNonZeroNode(graph), true);
}