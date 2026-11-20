/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fusion_pass/fixpipe_relu_quant_pass.h"
#include "test_fixpipe_pass_framework.h"

// ============================================================================
// FIXPIPEARELUQUANTFUSIONPASS: swap the relu and the tail AscendQuant of the cube
// ============================================================================
class FixpipeReluQuantPassTest : public FixpipePassTestBase {
protected:
    void SetUp() override
    {
        FixpipePassTestBase::SetUp();
        RegisterCloneOpProtos();
    }

    static void RunReluQuantPass(std::shared_ptr<Graph>& graph, ge::Status& res)
    {
        FIXPIPEARELUQUANTFUSIONPASS pass;
        ge::CustomPassContext context;
        context.SetPassName("FIXPIPEARELUQUANTFUSIONPASS");
        res = pass.Run(graph, context);
    }

    // x -> Conv2D -> Relu -> AscendQuant -> (graph output)
    static std::shared_ptr<Graph> BuildConvReluQuantGraph(FixpipeTestGraph& builder, const std::string& quantName)
    {
        auto xInput = builder.CreateInput("x", DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
        auto filterInput = builder.CreateInput("filter", DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
        builder.AddNode("cube", "Conv2D", {{"x", true}, {"filter", true}, {"bias", false}, {"offset_w", false}});
        builder.AddNode("relu", "Relu", {{"x", true}});
        GNode& quant = builder.AddNode(quantName, "AscendQuant", {{"x", true}});
        bool sqrtMode = false;
        (void)quant.SetAttr(ge::AscendString("sqrt_mode"), sqrtMode);
        float quantScale = 1.0f;
        (void)quant.SetAttr(ge::AscendString("scale"), quantScale);
        float quantOffset = 0.5f;
        (void)quant.SetAttr(ge::AscendString("offset"), quantOffset);

        builder.ConnectInput(xInput, "cube", kCubeFmapPort);
        builder.ConnectInput(filterInput, "cube", kCubeFilterPort);
        builder.Connect("cube", kOutPort, "relu", kReluXPort);
        builder.Connect("relu", kOutPort, quantName, kQuantXPort);

        builder.SetInDesc("cube", kCubeFmapPort, DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
        builder.SetInDesc("cube", kCubeFilterPort, DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
        builder.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
        builder.SetInDesc("relu", kReluXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
        builder.SetOutDesc("relu", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
        builder.SetInDesc(quantName, kQuantXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
        builder.SetOutDesc(quantName, kOutPort, DT_INT8, FORMAT_NCHW, {1, 3, 8, 8});
        return builder.Build({quantName});
    }
};

TEST_F(FixpipeReluQuantPassTest, NullGraphFailed)
{
    std::shared_ptr<Graph> graph = nullptr;
    ge::Status res = ge::GRAPH_SUCCESS;
    RunReluQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_FAILED);
}

TEST_F(FixpipeReluQuantPassTest, NoCubeNodeNotChanged)
{
    FixpipeTestGraph builder("no_cube_graph");
    auto xInput = builder.CreateInput("x", DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    builder.AddNode("relu", "Relu", {{"x", true}});
    builder.ConnectInput(xInput, "relu", kReluXPort);
    builder.SetInDesc("relu", kReluXPort, DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    builder.SetOutDesc("relu", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    auto graph = builder.Build({"relu"});

    ge::Status res = ge::GRAPH_SUCCESS;
    RunReluQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
}

TEST_F(FixpipeReluQuantPassTest, ReluQuantSwapSuccess)
{
    FixpipeTestGraph builder("relu_quant_graph");
    auto graph = BuildConvReluQuantGraph(builder, "quant");
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunReluQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    // the stale relu is removed and the quant directly follows the cube, the cloned relu
    // (named with the _fixpipe_ suffix) consumes the quant output
    EXPECT_FALSE(HasNodeName(graph, "relu"));
    GNode quantNode;
    ASSERT_TRUE(FindNodeByName(graph, "quant", quantNode));
    EXPECT_EQ(GetInputProducerType(quantNode, kQuantXPort), "Conv2D");

    bool hasClonedRelu = false;
    GNode clonedRelu;
    for (auto node : graph->GetAllNodes()) {
        ge::AscendString name;
        if (node.GetName(name) != ge::GRAPH_SUCCESS || name.GetString() == nullptr) {
            continue;
        }
        if (std::string(name.GetString()).find("_fixpipe_") == std::string::npos) {
            continue;
        }
        ge::AscendString type;
        ASSERT_EQ(node.GetType(type), ge::GRAPH_SUCCESS);
        ASSERT_NE(type.GetString(), nullptr);
        EXPECT_EQ(std::string(type.GetString()), "Relu");
        // the cloned relu takes the quant output and keeps the quant output dtype
        EXPECT_EQ(GetInputProducerName(node, kReluXPort), "quant");
        TensorDesc clonedOutDesc;
        ASSERT_EQ(node.GetOutputDesc(kOutPort, clonedOutDesc), ge::GRAPH_SUCCESS);
        EXPECT_EQ(clonedOutDesc.GetDataType(), DT_INT8);
        hasClonedRelu = true;
    }
    EXPECT_TRUE(hasClonedRelu);
}

TEST_F(FixpipeReluQuantPassTest, ReluWithoutQuantNotChanged)
{
    // the relu without any quant successor is not matched by the pattern
    FixpipeTestGraph builder("relu_no_quant_graph");
    auto xInput = builder.CreateInput("x", DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
    auto filterInput = builder.CreateInput("filter", DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
    builder.AddNode("cube", "Conv2D", {{"x", true}, {"filter", true}, {"bias", false}, {"offset_w", false}});
    builder.AddNode("relu", "Relu", {{"x", true}});
    builder.ConnectInput(xInput, "cube", kCubeFmapPort);
    builder.ConnectInput(filterInput, "cube", kCubeFilterPort);
    builder.Connect("cube", kOutPort, "relu", kReluXPort);
    builder.SetInDesc("cube", kCubeFmapPort, DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
    builder.SetInDesc("cube", kCubeFilterPort, DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
    builder.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetInDesc("relu", kReluXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetOutDesc("relu", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    auto graph = builder.Build({"relu"});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_SUCCESS;
    RunReluQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeName(graph, "relu"));
}

TEST_F(FixpipeReluQuantPassTest, ReluWithTwoQuantsNotChanged)
{
    // the relu with two quant successors does not support the multiple quant output
    FixpipeTestGraph builder("relu_two_quants_graph");
    auto xInput = builder.CreateInput("x", DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
    auto filterInput = builder.CreateInput("filter", DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
    builder.AddNode("cube", "Conv2D", {{"x", true}, {"filter", true}, {"bias", false}, {"offset_w", false}});
    builder.AddNode("relu", "Relu", {{"x", true}});
    GNode& quant1 = builder.AddNode("quant1", "AscendQuant", {{"x", true}});
    bool sqrtMode = false;
    (void)quant1.SetAttr(ge::AscendString("sqrt_mode"), sqrtMode);
    float quant1Scale = 1.0f;
    (void)quant1.SetAttr(ge::AscendString("scale"), quant1Scale);
    float quant1Offset = 0.5f;
    (void)quant1.SetAttr(ge::AscendString("offset"), quant1Offset);
    GNode& quant2 = builder.AddNode("quant2", "AscendQuant", {{"x", true}});
    (void)quant2.SetAttr(ge::AscendString("sqrt_mode"), sqrtMode);
    float quant2Scale = 2.0f;
    (void)quant2.SetAttr(ge::AscendString("scale"), quant2Scale);
    float quant2Offset = 0.25f;
    (void)quant2.SetAttr(ge::AscendString("offset"), quant2Offset);

    builder.ConnectInput(xInput, "cube", kCubeFmapPort);
    builder.ConnectInput(filterInput, "cube", kCubeFilterPort);
    builder.Connect("cube", kOutPort, "relu", kReluXPort);
    builder.Connect("relu", kOutPort, "quant1", kQuantXPort);
    builder.Connect("relu", kOutPort, "quant2", kQuantXPort);

    builder.SetInDesc("cube", kCubeFmapPort, DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
    builder.SetInDesc("cube", kCubeFilterPort, DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
    builder.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetInDesc("relu", kReluXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetOutDesc("relu", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetInDesc("quant1", kQuantXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetOutDesc("quant1", kOutPort, DT_INT8, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetInDesc("quant2", kQuantXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetOutDesc("quant2", kOutPort, DT_INT8, FORMAT_NCHW, {1, 3, 8, 8});
    auto graph = builder.Build({"quant1", "quant2"});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_SUCCESS;
    RunReluQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeName(graph, "relu"));
    EXPECT_TRUE(HasNodeName(graph, "quant1"));
    EXPECT_TRUE(HasNodeName(graph, "quant2"));
}
