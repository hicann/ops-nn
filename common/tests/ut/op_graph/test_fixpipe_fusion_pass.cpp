/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fusion_pass/fixpipe_pass.h"
#include "test_fixpipe_pass_framework.h"

// ============================================================================
// FIXPIPEFUSIONPASS: fold the post op chain of the cube node into the FixPipe unit
// ============================================================================
class FixpipeFusionPassTest : public FixpipePassTestBase {
protected:
    static void RunFusionPass(std::shared_ptr<Graph>& graph, ge::Status& res)
    {
        FIXPIPEFUSIONPASS pass;
        ge::CustomPassContext context;
        context.SetPassName("FIXPIPEFUSIONPASS");
        res = pass.Run(graph, context);
    }

    // x -> Conv2D -> AscendDequant -> (graph output), the deq_scale comes from a graph input
    static std::shared_ptr<Graph> BuildConvDequantGraph(FixpipeTestGraph& builder)
    {
        auto xInput = builder.CreateInput("x", DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
        auto filterInput = builder.CreateInput("filter", DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
        auto scaleInput = builder.CreateInput("deq_scale", DT_UINT64, FORMAT_ND, {1});
        builder.AddNode("cube", "Conv2D", {{"x", true}, {"filter", true}, {"bias", false}, {"offset_w", false}});
        GNode& dequant = builder.AddNode("dequant", "AscendDequant", {{"x", true}, {"deq_scale", true}});
        bool sqrtMode = false;
        (void)dequant.SetAttr(ge::AscendString("sqrt_mode"), sqrtMode);
        bool reluFlag = false;
        (void)dequant.SetAttr(ge::AscendString("relu_flag"), reluFlag);
        int64_t dequantDtype = static_cast<int64_t>(DT_FLOAT16);
        (void)dequant.SetAttr(ge::AscendString("dtype"), dequantDtype);

        builder.ConnectInput(xInput, "cube", kCubeFmapPort);
        builder.ConnectInput(filterInput, "cube", kCubeFilterPort);
        builder.ConnectInput(scaleInput, "dequant", kDequantScalePort);
        builder.Connect("cube", kOutPort, "dequant", kDequantXPort);

        builder.SetInDesc("cube", kCubeFmapPort, DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
        builder.SetInDesc("cube", kCubeFilterPort, DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
        builder.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
        builder.SetInDesc("dequant", kDequantXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
        builder.SetInDesc("dequant", kDequantScalePort, DT_UINT64, FORMAT_ND, {1});
        builder.SetOutDesc("dequant", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
        return builder.Build({"dequant"});
    }
};

TEST_F(FixpipeFusionPassTest, NullGraphFailed)
{
    std::shared_ptr<Graph> graph = nullptr;
    ge::Status res = ge::GRAPH_SUCCESS;
    RunFusionPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_FAILED);
}

TEST_F(FixpipeFusionPassTest, NoCubeNodeNotChanged)
{
    FixpipeTestGraph builder("no_cube_graph");
    auto xInput = builder.CreateInput("x", DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    builder.AddNode("relu", "Relu", {{"x", true}});
    builder.ConnectInput(xInput, "relu", kReluXPort);
    builder.SetInDesc("relu", kReluXPort, DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    builder.SetOutDesc("relu", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    auto graph = builder.Build({"relu"});

    ge::Status res = ge::GRAPH_SUCCESS;
    RunFusionPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
}

TEST_F(FixpipeFusionPassTest, CubeWithoutPostOpNotChanged)
{
    // the cube output has no postcube successor, no FixPipe node is created
    auto graph = BuildCubeGraph("Conv2D", {1, 3, 8, 8}, DT_FLOAT16);
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_SUCCESS;
    RunFusionPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeName(graph, "cube"));
    EXPECT_FALSE(HasNodeType(graph, "FixPipe"));
}

TEST_F(FixpipeFusionPassTest, ConvDequantFoldSuccess)
{
    FixpipeTestGraph builder("conv_dequant_graph");
    auto graph = BuildConvDequantGraph(builder);
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunFusionPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    // the dequant chain is folded into the FixPipe hardware unit
    EXPECT_TRUE(HasNodeType(graph, "FixPipe"));
    EXPECT_FALSE(HasNodeName(graph, "dequant"));
    // the cube node itself stays and feeds the FixPipe node
    EXPECT_TRUE(HasNodeName(graph, "cube"));

    GNode fixpipeNode;
    bool fixpipeFound = false;
    for (auto node : graph->GetAllNodes()) {
        ge::AscendString type;
        if (node.GetType(type) == ge::GRAPH_SUCCESS && type.GetString() != nullptr &&
            std::string(type.GetString()) == "FixPipe") {
            fixpipeNode = node;
            fixpipeFound = true;
            break;
        }
    }
    ASSERT_TRUE(fixpipeFound);
    EXPECT_EQ(GetInputProducerType(fixpipeNode, kOutPort), "Conv2D");
    // the deq_scale input is packed by the SetM1Dequant host op
    EXPECT_TRUE(HasNodeType(graph, "SetM1Dequant"));
    // the graph output boundary takes over the dequant output position
    for (auto node : graph->GetAllNodes()) {
        ge::AscendString type;
        if (node.GetType(type) == ge::GRAPH_SUCCESS && type.GetString() != nullptr &&
            std::string(type.GetString()) == "NetOutput") {
            EXPECT_EQ(GetInputProducerType(node, kOutPort), "FixPipe");
        }
    }
}

TEST_F(FixpipeFusionPassTest, ConvCastNotFoldNotChanged)
{
    // the Cast successor is not in any fixpipe unit func list, no pass is matched
    FixpipeTestGraph builder("conv_cast_graph");
    auto xInput = builder.CreateInput("x", DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
    auto filterInput = builder.CreateInput("filter", DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
    builder.AddNode("cube", "Conv2D", {{"x", true}, {"filter", true}, {"bias", false}, {"offset_w", false}});
    builder.AddNode("cast", "Cast", {{"x", true}});
    builder.ConnectInput(xInput, "cube", kCubeFmapPort);
    builder.ConnectInput(filterInput, "cube", kCubeFilterPort);
    builder.Connect("cube", kOutPort, "cast", kOutPort);
    builder.SetInDesc("cube", kCubeFmapPort, DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
    builder.SetInDesc("cube", kCubeFilterPort, DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
    builder.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetInDesc("cast", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetOutDesc("cast", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    auto graph = builder.Build({"cast"});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_SUCCESS;
    RunFusionPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasNodeType(graph, "FixPipe"));
    EXPECT_TRUE(HasNodeName(graph, "cast"));
}
