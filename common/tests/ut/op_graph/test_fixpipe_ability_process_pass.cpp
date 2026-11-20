/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fusion_pass/fixpipe_ability_process_pass.h"
#include "test_fixpipe_pass_framework.h"

// ============================================================================
// FixPipeAbilityProcessPass: set the ability attrs on the cube nodes
// ============================================================================
class FixPipeAbilityProcessPassTest : public FixpipePassTestBase {
protected:
    static void RunAbilityPass(std::shared_ptr<Graph>& graph, ge::Status& res)
    {
        FixPipeAbilityProcessPass pass;
        ge::CustomPassContext context;
        context.SetPassName("FixPipeAbilityProcessPass");
        res = pass.Run(graph, context);
    }

    static bool FindCubeNode(const std::shared_ptr<Graph>& graph, GNode& cubeNode)
    {
        return FindNodeByName(graph, "cube", cubeNode);
    }
};

TEST_F(FixPipeAbilityProcessPassTest, NullGraphFailed)
{
    std::shared_ptr<Graph> graph = nullptr;
    ge::Status res = ge::GRAPH_SUCCESS;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_FAILED);
}

TEST_F(FixPipeAbilityProcessPassTest, NoCubeNodeNotChanged)
{
    FixpipeTestGraph builder("no_cube_graph");
    auto xInput = builder.CreateInput("x", DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    builder.AddNode("relu", "Relu", {{"x", true}});
    builder.ConnectInput(xInput, "relu", kReluXPort);
    builder.SetInDesc("relu", kReluXPort, DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    builder.SetOutDesc("relu", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    auto graph = builder.Build({"relu"});

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
}

TEST_F(FixPipeAbilityProcessPassTest, Conv2dMultipleOutputAbilitySet)
{
    auto graph = BuildCubeGraph("Conv2D", {1, 3, 8, 8});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    // the Conv2D cube is in the multiple output support set and the platform allows it
    EXPECT_TRUE(GetAbilityAttr(cubeNode) & kAbilityMultipleOutput);
    EXPECT_FALSE(GetAbilityAttr(cubeNode) & kAbilityCantAccess);
}

TEST_F(FixPipeAbilityProcessPassTest, Conv2dTransposeMultipleOutputAbilitySet)
{
    auto graph = BuildCubeGraph("Conv2DTranspose", {1, 3, 8, 8});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    EXPECT_TRUE(GetAbilityAttr(cubeNode) & kAbilityMultipleOutput);
}

TEST_F(FixPipeAbilityProcessPassTest, MultipleOutputOffNoAbility)
{
    ApplyPlatform(false, true);
    auto graph = BuildCubeGraph("Conv2D", {1, 3, 8, 8});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    // the pass still counts the processed cube node as an effect
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    EXPECT_FALSE(GetAbilityAttr(cubeNode) & kAbilityMultipleOutput);
    EXPECT_FALSE(GetAbilityAttr(cubeNode) & kAbilityCantAccess);
}

TEST_F(FixPipeAbilityProcessPassTest, UnknownShapeCantAccess)
{
    // any negative output dim marks the cube node not accessible by the fixpipe fusion
    auto graph = BuildCubeGraph("Conv2D", {-1, 3, 8, 8});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    EXPECT_TRUE(GetAbilityAttr(cubeNode) & kAbilityCantAccess);
    EXPECT_FALSE(GetAbilityAttr(cubeNode) & kAbilityMultipleOutput);
}

TEST_F(FixPipeAbilityProcessPassTest, MatmulNchwOutputCantAccess)
{
    // the matmul itself used the fixpipe, the 4d output format cannot enter the fixpipe fusion
    auto graph = BuildCubeGraph("MatMulV2", {1, 3, 8, 8}, DT_FLOAT16, FORMAT_NCHW);
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    EXPECT_TRUE(GetAbilityAttr(cubeNode) & kAbilityCantAccess);
}

TEST_F(FixPipeAbilityProcessPassTest, MatmulFp32NzC016CantAccess)
{
    // the fp32 fractal nz output with c0 = 16 does not support the fixpipe fusion
    auto graph = BuildCubeGraph("MatMulV2", {2, 3, 16}, DT_FLOAT, FORMAT_FRACTAL_NZ);
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    EXPECT_TRUE(GetAbilityAttr(cubeNode) & kAbilityCantAccess);
}

TEST_F(FixPipeAbilityProcessPassTest, MatmulInt32ChainCantAccess)
{
    // the int32 in int32 out matmul chain does not support the fixpipe fusion
    FixpipeTestGraph builder("matmul_int32_chain");
    auto xInput = builder.CreateInput("x", DT_INT32, FORMAT_FRACTAL_NZ, {2, 3, 16});
    builder.AddNode("cube", "MatMulV2", {{"x1", true}, {"x2", true}});
    builder.ConnectInput(xInput, "cube", kCubeFmapPort);
    builder.SetInDesc("cube", kCubeFmapPort, DT_INT32, FORMAT_FRACTAL_NZ, {2, 3, 16});
    builder.SetInDesc("cube", kCubeFilterPort, DT_INT32, FORMAT_FRACTAL_NZ, {2, 3, 16});
    builder.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_FRACTAL_NZ, {2, 3, 16});
    auto graph = builder.Build({"cube"});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    EXPECT_TRUE(GetAbilityAttr(cubeNode) & kAbilityCantAccess);
}

TEST_F(FixPipeAbilityProcessPassTest, MatmulNdInputCantAccess)
{
    // the nd format matmul input does not support the fixpipe fusion
    auto graph = BuildCubeGraph("MatMulV2", {2, 3, 16}, DT_FLOAT16, FORMAT_FRACTAL_NZ, DT_FLOAT16, FORMAT_ND);
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    EXPECT_TRUE(GetAbilityAttr(cubeNode) & kAbilityCantAccess);
}

TEST_F(FixPipeAbilityProcessPassTest, Conv3dFp32InputCantAccess)
{
    // the conv3d fp32 input cannot enter the fixpipe fusion
    auto graph = BuildCubeGraph("Conv3D", {1, 3, 8, 8, 8}, DT_FLOAT16, FORMAT_NCDHW, DT_FLOAT, FORMAT_NCDHW,
                                {1, 16, 16, 16, 16});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    EXPECT_TRUE(GetAbilityAttr(cubeNode) & kAbilityCantAccess);
}

TEST_F(FixPipeAbilityProcessPassTest, Conv3dFp16NotInUtilWhiteListNotChanged)
{
    // the dn2nz intrinsic restricts the util func support to Conv2D/Conv2DTranspose, the
    // valid Conv3D node is skipped without any ability attr
    auto graph = BuildCubeGraph("Conv3D", {1, 3, 8, 8, 8}, DT_FLOAT16, FORMAT_NCDHW, DT_FLOAT16, FORMAT_NCDHW,
                                {1, 16, 16, 16, 16});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_SUCCESS;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    EXPECT_EQ(GetAbilityAttr(cubeNode), 0);
}

TEST_F(FixPipeAbilityProcessPassTest, BackpropFilterGmAtomicAddDn2nzAbsent)
{
    // without the dn2nz intrinsic every cube type passes the util func white list and the
    // conv backprop filter cube gets the gm atomic add ability
    SetDn2nzEnabled(false);
    auto graph = BuildCubeGraph("Conv2DBackpropFilterD", {16, 4, 3, 3});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunAbilityPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    GNode cubeNode;
    ASSERT_TRUE(FindCubeNode(graph, cubeNode));
    EXPECT_TRUE(GetAbilityAttr(cubeNode) & kAbilityGmAtomicAdd);
    EXPECT_FALSE(GetAbilityAttr(cubeNode) & kAbilityMultipleOutput);
    EXPECT_FALSE(GetAbilityAttr(cubeNode) & kAbilityCantAccess);
}
