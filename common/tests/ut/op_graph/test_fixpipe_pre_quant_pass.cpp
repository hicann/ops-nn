/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fusion_pass/fixpipe_pre_quant_pass.h"
#include "test_fixpipe_pass_framework.h"

// ============================================================================
// FIXPIPEAPREQUANTFUSIONPASS: absorb the tail AscendQuant of the dequant chain
// ============================================================================
class FixpipePreQuantPassTest : public FixpipePassTestBase {
protected:
    void SetUp() override
    {
        FixpipePassTestBase::SetUp();
        RegisterCloneOpProtos();
    }

    static void RunPreQuantPass(std::shared_ptr<Graph>& graph, ge::Status& res)
    {
        FIXPIPEAPREQUANTFUSIONPASS pass;
        ge::CustomPassContext context;
        context.SetPassName("FIXPIPEAPREQUANTFUSIONPASS");
        res = pass.Run(graph, context);
    }

    // x -> Conv2D -> AscendDequant -> AscendQuant -> (graph output)
    static std::shared_ptr<Graph> BuildConvDequantQuantGraph(FixpipeTestGraph& builder)
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
        GNode& quant = builder.AddNode("quant", "AscendQuant", {{"x", true}});
        (void)quant.SetAttr(ge::AscendString("sqrt_mode"), sqrtMode);
        float quantScale = 1.0f;
        (void)quant.SetAttr(ge::AscendString("scale"), quantScale);
        float quantOffset = 0.5f;
        (void)quant.SetAttr(ge::AscendString("offset"), quantOffset);

        builder.ConnectInput(xInput, "cube", kCubeFmapPort);
        builder.ConnectInput(filterInput, "cube", kCubeFilterPort);
        builder.ConnectInput(scaleInput, "dequant", kDequantScalePort);
        builder.Connect("cube", kOutPort, "dequant", kDequantXPort);
        builder.Connect("dequant", kOutPort, "quant", kQuantXPort);

        builder.SetInDesc("cube", kCubeFmapPort, DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
        builder.SetInDesc("cube", kCubeFilterPort, DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
        builder.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
        builder.SetInDesc("dequant", kDequantXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
        builder.SetInDesc("dequant", kDequantScalePort, DT_UINT64, FORMAT_ND, {1});
        builder.SetOutDesc("dequant", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
        builder.SetInDesc("quant", kQuantXPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
        builder.SetOutDesc("quant", kOutPort, DT_INT8, FORMAT_NCHW, {1, 3, 8, 8});
        return builder.Build({"quant"});
    }
};

TEST_F(FixpipePreQuantPassTest, NullGraphFailed)
{
    std::shared_ptr<Graph> graph = nullptr;
    ge::Status res = ge::GRAPH_SUCCESS;
    RunPreQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_FAILED);
}

TEST_F(FixpipePreQuantPassTest, NoPreUnitConfigNotChanged)
{
    // without the pre_conv/pre_act units the ReadConfig of the pass fails, no fusion
    ApplyPlatform(true, false);
    FixpipeTestGraph builder("no_unit_graph");
    auto graph = BuildConvDequantQuantGraph(builder);
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_SUCCESS;
    RunPreQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeName(graph, "dequant"));
    EXPECT_TRUE(HasNodeName(graph, "quant"));
}

TEST_F(FixpipePreQuantPassTest, NoCubeNodeNotChanged)
{
    FixpipeTestGraph builder("no_cube_graph");
    auto xInput = builder.CreateInput("x", DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    builder.AddNode("relu", "Relu", {{"x", true}});
    builder.ConnectInput(xInput, "relu", kReluXPort);
    builder.SetInDesc("relu", kReluXPort, DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    builder.SetOutDesc("relu", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16});
    auto graph = builder.Build({"relu"});

    ge::Status res = ge::GRAPH_SUCCESS;
    RunPreQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
}

TEST_F(FixpipePreQuantPassTest, CubeWithoutDequantNotChanged)
{
    // the cube output has no AscendDequant successor, nothing is absorbed
    auto graph = BuildCubeGraph("Conv2D", {1, 3, 8, 8}, DT_FLOAT16);
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_SUCCESS;
    RunPreQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
}

TEST_F(FixpipePreQuantPassTest, DequantQuantAbsorbSuccess)
{
    FixpipeTestGraph builder("dequant_quant_graph");
    auto graph = BuildConvDequantQuantGraph(builder);
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_NOT_CHANGED;
    RunPreQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);

    // the tail quant is absorbed: the cloned dequant outputs the quant dtype and the scale
    // input is pre-multiplied by the VectorMulScalar host op
    EXPECT_FALSE(HasNodeName(graph, "dequant"));
    EXPECT_FALSE(HasNodeName(graph, "quant"));
    EXPECT_TRUE(HasNodeName(graph, "dequant_requant_0"));
    EXPECT_EQ(GetNodeTypeOfName(graph, "dequant_requant_0"), "AscendDequant");
    EXPECT_TRUE(HasNodeType(graph, "VectorMulScalar"));
    // the cube node stays and feeds the enhanced dequant directly
    EXPECT_TRUE(HasNodeName(graph, "cube"));

    GNode enhancedDequant;
    ASSERT_TRUE(FindNodeByName(graph, "dequant_requant_0", enhancedDequant));
    EXPECT_EQ(GetInputProducerType(enhancedDequant, kDequantXPort), "Conv2D");
    EXPECT_EQ(GetInputProducerType(enhancedDequant, kDequantScalePort), "VectorMulScalar");
    // the requant scale/offset attrs are folded onto the enhanced dequant
    float scaleValue = 0.0f;
    EXPECT_EQ(enhancedDequant.GetAttr(ge::AscendString("scale"), scaleValue), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scaleValue, 1.0f);
    float offsetValue = 0.0f;
    EXPECT_EQ(enhancedDequant.GetAttr(ge::AscendString("offset"), offsetValue), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(offsetValue, 0.5f);
    TensorDesc enhancedOutDesc;
    ASSERT_EQ(enhancedDequant.GetOutputDesc(kOutPort, enhancedOutDesc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(enhancedOutDesc.GetDataType(), DT_INT8);
}

TEST_F(FixpipePreQuantPassTest, DequantMultiBranchNotChanged)
{
    // the dequant output feeds both a quant and a relu, the multi branch without the
    // multiple output ability is not absorbed
    FixpipeTestGraph builder("dequant_multi_branch_graph");
    auto xInput = builder.CreateInput("x", DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
    auto filterInput = builder.CreateInput("filter", DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
    auto scaleInput = builder.CreateInput("deq_scale", DT_UINT64, FORMAT_ND, {1});
    builder.AddNode("cube", "Conv2D", {{"x", true}, {"filter", true}, {"bias", false}, {"offset_w", false}});
    GNode& dequant = builder.AddNode("dequant", "AscendDequant", {{"x", true}, {"deq_scale", true}});
    bool sqrtMode = false;
    (void)dequant.SetAttr(ge::AscendString("sqrt_mode"), sqrtMode);
    bool reluFlag = false;
    (void)dequant.SetAttr(ge::AscendString("relu_flag"), reluFlag);
    GNode& quant = builder.AddNode("quant", "AscendQuant", {{"x", true}});
    (void)quant.SetAttr(ge::AscendString("sqrt_mode"), sqrtMode);
    float quantScale = 1.0f;
    (void)quant.SetAttr(ge::AscendString("scale"), quantScale);
    float quantOffset = 0.5f;
    (void)quant.SetAttr(ge::AscendString("offset"), quantOffset);
    builder.AddNode("relu", "Relu", {{"x", true}});

    builder.ConnectInput(xInput, "cube", kCubeFmapPort);
    builder.ConnectInput(filterInput, "cube", kCubeFilterPort);
    builder.ConnectInput(scaleInput, "dequant", kDequantScalePort);
    builder.Connect("cube", kOutPort, "dequant", kDequantXPort);
    builder.Connect("dequant", kOutPort, "quant", kQuantXPort);
    builder.Connect("dequant", kOutPort, "relu", kReluXPort);

    builder.SetInDesc("cube", kCubeFmapPort, DT_INT8, FORMAT_NCHW, {1, 16, 16, 16});
    builder.SetInDesc("cube", kCubeFilterPort, DT_INT8, FORMAT_NCHW, {16, 32, 3, 3});
    builder.SetOutDesc("cube", kOutPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetInDesc("dequant", kDequantXPort, DT_INT32, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetInDesc("dequant", kDequantScalePort, DT_UINT64, FORMAT_ND, {1});
    builder.SetOutDesc("dequant", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetInDesc("quant", kQuantXPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetOutDesc("quant", kOutPort, DT_INT8, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetInDesc("relu", kReluXPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    builder.SetOutDesc("relu", kOutPort, DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8});
    auto graph = builder.Build({"quant", "relu"});
    ASSERT_NE(graph, nullptr);

    ge::Status res = ge::GRAPH_SUCCESS;
    RunPreQuantPass(graph, res);
    EXPECT_EQ(res, ge::GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeName(graph, "dequant"));
    EXPECT_TRUE(HasNodeName(graph, "quant"));
    EXPECT_FALSE(HasNodeType(graph, "VectorMulScalar"));
}
