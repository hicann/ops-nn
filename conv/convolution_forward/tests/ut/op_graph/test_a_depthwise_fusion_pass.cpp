/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <functional>
#include <vector>

#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "../../../op_graph/fusion_pass/a_depthwise_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace ADepthwiseFusion;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

namespace {
const Status kConvNotChanged = static_cast<Status>(CONV_NOT_CHANGED);

DepthwiseConv2DConfig MakeDepthwiseConfig(const char *nodeName, DataType dtype, Format fmapFormat,
    const std::vector<int64_t> &fmapShape, const std::vector<int64_t> &filterShape, Format filterOriginFormat,
    const std::vector<int64_t> &outputShape)
{
    TensorDesc fmapDesc = BuildTensorDesc(dtype, fmapFormat, fmapShape, fmapFormat, fmapShape);
    TensorDesc filterDesc = BuildTensorDesc(dtype, filterOriginFormat, filterShape, filterOriginFormat, filterShape);
    TensorDesc outputDesc = BuildTensorDesc(dtype, fmapFormat, outputShape, fmapFormat, outputShape);

    DepthwiseConv2DConfig config;
    config.SetName(nodeName)
        .AddInput(TensorInfo(dtype, fmapFormat, fmapShape, std::string(nodeName) + "_x").SetDesc(fmapDesc))
        .AddInput(TensorInfo(dtype, filterOriginFormat, filterShape, std::string(nodeName) + "_filter")
            .SetDesc(filterDesc))
        .AddOutput(TensorInfo(dtype, fmapFormat, outputShape, "y").SetDesc(outputDesc))
        .SetAttr("data_format", fmapFormat == FORMAT_NHWC ? "NHWC" : "NCHW");
    return config;
}

DepthwiseConv2DConfig MakeHwcnConfig(const std::string &nodeName, const std::vector<int64_t> &filterShape,
    DataType dtype = DT_FLOAT)
{
    return MakeDepthwiseConfig(nodeName.c_str(), dtype, FORMAT_NCHW, {1, 16, 256, 256}, filterShape, FORMAT_HWCN,
        {1, 4, 256, 256});
}

DepthwiseConv2DConfig MakeNchwConfig(const std::string &nodeName, const std::vector<int64_t> &filterShape,
    DataType dtype = DT_FLOAT)
{
    return MakeDepthwiseConfig(nodeName.c_str(), dtype, FORMAT_NCHW, {1, 16, 256, 256}, filterShape, FORMAT_NCHW,
        {1, 4, 256, 256});
}

DepthwiseConv2DConfig MakeNhwcConfig(const std::string &nodeName, const std::vector<int64_t> &filterShape)
{
    return MakeDepthwiseConfig(nodeName.c_str(), DT_FLOAT, FORMAT_NHWC, {1, 256, 256, 16}, filterShape, FORMAT_NHWC,
        {1, 256, 256, 4});
}

GraphPtr BuildDepthwiseGraph(const char *graphName, const DepthwiseConv2DConfig &cfg)
{
    TestGraph builder(graphName);
    builder.SetSocAscend950();
    return builder.AddDepthwiseConv2D(cfg).SetOutput(cfg.name).Build();
}

GraphPtr BuildDepthwiseWithFilterBranch(const char *graphName, const DepthwiseConv2DConfig &cfg, bool withRelu = true)
{
    TestGraph builder(graphName);
    builder.SetSocAscend950();
    return builder.AddDepthwiseConv2DWithFilterBranch(cfg, withRelu).SetOutput(cfg.name).Build();
}

void ExpectReshapeFusion(GraphPtr &graph, const std::vector<int64_t> &expIn, const std::vector<int64_t> &expOut,
    const std::vector<int64_t> &expAttr = {})
{
    GNode reshapeNode;
    ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "/Reshape_1", reshapeNode));
    std::vector<int64_t> inShape;
    std::vector<int64_t> outShape;
    ASSERT_TRUE(GraphChecker::GetOriginShape(reshapeNode, 0, false, inShape));
    ASSERT_TRUE(GraphChecker::GetOriginShape(reshapeNode, 0, true, outShape));
    EXPECT_EQ(inShape, expIn);
    EXPECT_EQ(outShape, expOut);
    if (!expAttr.empty()) {
        std::vector<int64_t> shapeAttr;
        ASSERT_TRUE(GraphChecker::GetListIntAttr(reshapeNode, "shape", shapeAttr));
        EXPECT_EQ(shapeAttr, expAttr);
    }
}

void ExpectDepthwiseFilterShape(GraphPtr &graph, const std::vector<int64_t> &expFilterShape)
{
    GNode depthwise;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "DepthwiseConv2D", depthwise));
    std::vector<int64_t> filterShape;
    ASSERT_TRUE(GraphChecker::GetOriginShape(depthwise, 1, false, filterShape));
    EXPECT_EQ(filterShape, expFilterShape);
}
} // namespace

class ADepthwiseFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ADepthwiseFusionPassTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ADepthwiseFusionPassTest TearDown" << std::endl; }

    void TestTotalPass(const std::string &passName, GraphPtr &graph, Status expectRes,
        int32_t depthwiseCountBefore = 1)
    {
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        ADepthwiseFusionPass pass;

        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
        if (res == SUCCESS) {
            EXPECT_GE(GraphChecker::CountNodes(graph, "Reshape"), 1);
            EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), depthwiseCountBefore);
        } else if (res == kConvNotChanged) {
            EXPECT_EQ(GraphChecker::CountNodes(graph, "Reshape"), 0);
            EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), depthwiseCountBefore);
        }
    }
};

TEST_F(ADepthwiseFusionPassTest, hwcn_reshape_success)
{
    struct Point {
        const char *pointName;
        std::vector<int64_t> filterShape;
        std::vector<int64_t> expIn;
        std::vector<int64_t> expOut;
        bool useDynamicFmap;
    } const points[] = {
        {"static", {1, 1, 16, 4}, {1, 1, 16, 4}, {1, 1, 1, 64}, false},
        {"dynamic", {-1, -1, 16, -1}, {-1, -1, 16, -1}, {-1, -1, 1, -16}, true},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        std::string name = std::string("hwcn_reshape_success_") + p.pointName;
        GraphPtr graph;
        if (p.useDynamicFmap) {
            auto cfg = MakeDepthwiseConfig(name.c_str(), DT_FLOAT, FORMAT_NCHW, {-1, -1, -1, -1}, p.filterShape,
                FORMAT_HWCN, {-1, -1, -1, -1});
            graph = BuildDepthwiseWithFilterBranch(name.c_str(), cfg);
        } else {
            auto cfg = MakeHwcnConfig(name, p.filterShape);
            graph = BuildDepthwiseGraph(name.c_str(), cfg);
        }
        TestTotalPass(name, graph, SUCCESS);
        ExpectReshapeFusion(graph, p.expIn, p.expOut, p.expOut);
        ExpectDepthwiseFilterShape(graph, p.expOut);
    }
}

TEST_F(ADepthwiseFusionPassTest, nchw_reshape_success)
{
    struct Point {
        const char *pointName;
        std::vector<int64_t> filterShape;
        std::vector<int64_t> expIn;
        std::vector<int64_t> expOut;
    } const points[] = {
        {"static", {4, 16, 1, 1}, {4, 16, 1, 1}, {64, 1, 1, 1}},
        {"dynamic", {-1, 16, -1, -1}, {-1, 16, -1, -1}, {-16, 1, -1, -1}},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        std::string name = std::string("nchw_reshape_success_") + p.pointName;
        auto cfg = MakeNchwConfig(name, p.filterShape);
        auto graph = BuildDepthwiseGraph(name.c_str(), cfg);
        TestTotalPass(name, graph, SUCCESS);
        ExpectReshapeFusion(graph, p.expIn, p.expOut, p.expOut);
        ExpectDepthwiseFilterShape(graph, p.expOut);
    }
}

TEST_F(ADepthwiseFusionPassTest, a_depthwise_fusion_no_change)
{
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
        bool skipDepthwiseCheck = false;
    } const points[] = {
        {"nhwc_format", []() {
             auto cfg = MakeDepthwiseConfig("depthwise_conv2d", DT_FLOAT, FORMAT_NCHW, {1, 16, 256, 256},
                 {4, 1, 1, 16}, FORMAT_NHWC, {1, 4, 256, 256});
             return BuildDepthwiseWithFilterBranch("a_depthwise_fusion_no_change_nhwc", cfg);
         }},
        {"nchw_c_already_one", []() {
             auto cfg = MakeDepthwiseConfig("depthwise_conv2d", DT_FLOAT, FORMAT_NCHW, {1, 1, 256, 256},
                 {4, 1, 1, 16}, FORMAT_NCHW, {1, 4, 256, 256});
             return BuildDepthwiseWithFilterBranch("a_depthwise_fusion_no_change_nchw_c1", cfg);
         }},
        {"hwcn_c_already_one", []() {
             auto cfg = MakeDepthwiseConfig("depthwise_conv2d", DT_FLOAT, FORMAT_NCHW, {1, 1, 256, 256},
                 {4, 1, 1, 16}, FORMAT_HWCN, {1, 4, 256, 256});
             return BuildDepthwiseWithFilterBranch("a_depthwise_fusion_no_change_hwcn_c1", cfg);
         }},
        {"filter_not_4d_skipped", []() {
             auto cfg = MakeHwcnConfig("depthwise_conv2d", {1, 1, 16, 4});
             cfg.inputs[1].shape = {1, 1, 16, 4, 1};
             cfg.inputs[1].tensorDesc =
                 BuildTensorDesc(DT_FLOAT, FORMAT_HWCN, {1, 1, 16, 4, 1}, FORMAT_HWCN, {1, 1, 16, 4, 1});
             return BuildDepthwiseGraph("filter_not_4d_skipped", cfg);
         }},
        {"filter_format_nd", []() {
             auto cfg = MakeHwcnConfig("depthwise_conv2d", {1, 1, 16, 4});
             cfg.inputs[1].format = FORMAT_ND;
             cfg.inputs[1].tensorDesc =
                 BuildTensorDesc(DT_FLOAT, FORMAT_ND, {1, 1, 16, 4}, FORMAT_ND, {1, 1, 16, 4});
             return BuildDepthwiseGraph("filter_format_nd", cfg);
         }},
        {"empty_graph_no_match", []() {
             return TestGraph("empty_graph_no_match").SetSocAscend950()
                 .AddRelu(ReluConfig::Basic("relu", DT_FLOAT16), true)
                 .SetOutput("relu")
                 .Build();
         }, true},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        std::string passName = std::string("a_depthwise_fusion_no_change_") + p.pointName;
        if (p.skipDepthwiseCheck) {
            CustomPassContext passContext;
            passContext.SetPassName(passName.c_str());
            ADepthwiseFusionPass pass;
            EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
            EXPECT_EQ(GraphChecker::CountNodes(graph, "Reshape"), 0);
            continue;
        }
        TestTotalPass(passName, graph, kConvNotChanged);
    }
}

TEST_F(ADepthwiseFusionPassTest, ascend_weight_quant_hwcn)
{
    auto graph = TestGraph("ascend_weight_quant_hwcn")
        .SetSocAscend950()
        .AddNode(AscendWeightQuantConfig::Basic("ascend_weight_quant", DT_FLOAT, FORMAT_HWCN, {1, 1, 16, 4}), {0})
        .AddDepthwiseConv2D(MakeHwcnConfig("depthwise_conv2d", {1, 1, 16, 4}), true, false)
        .Connect("ascend_weight_quant", 0, "depthwise_conv2d", 1)
        .SetOutput("depthwise_conv2d")
        .Build();

    TestTotalPass("ascend_weight_quant_hwcn", graph, SUCCESS);
    const std::vector<int64_t> kExpShape = {1, 1, 1, 64};
    GNode quantNode;
    GNode depthwiseNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "AscendWeightQuant", quantNode));
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "DepthwiseConv2D", depthwiseNode));
    std::vector<int64_t> quantOutShape;
    std::vector<int64_t> depthwiseFilterShape;
    ASSERT_TRUE(GraphChecker::GetOriginShape(quantNode, 0, true, quantOutShape));
    ASSERT_TRUE(GraphChecker::GetOriginShape(depthwiseNode, 1, false, depthwiseFilterShape));
    EXPECT_EQ(quantOutShape, kExpShape);
    EXPECT_EQ(depthwiseFilterShape, kExpShape);
    std::string quantToDepthwiseProducer;
    ASSERT_TRUE(GraphChecker::GetInputProducerType(depthwiseNode, 1, quantToDepthwiseProducer));
    EXPECT_EQ(quantToDepthwiseProducer, "AscendWeightQuant");
    ExpectReshapeFusion(graph, (std::vector<int64_t>{1, 1, 16, 4}), kExpShape, kExpShape);
}

TEST_F(ADepthwiseFusionPassTest, ascend_weight_quant_nchw)
{
    auto graph = TestGraph("ascend_weight_quant_nchw")
        .SetSocAscend950()
        .AddNode(AscendWeightQuantConfig::Basic("ascend_weight_quant", DT_FLOAT, FORMAT_NCHW, {4, 16, 1, 1}), {0})
        .AddDepthwiseConv2D(MakeNchwConfig("depthwise_conv2d", {4, 16, 1, 1}), true, false)
        .Connect("ascend_weight_quant", 0, "depthwise_conv2d", 1)
        .SetOutput("depthwise_conv2d")
        .Build();

    TestTotalPass("ascend_weight_quant_nchw", graph, SUCCESS);
    GNode reshapeNode;
    ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "/Reshape_1", reshapeNode));
    std::vector<int64_t> outShape;
    ASSERT_TRUE(GraphChecker::GetOriginShape(reshapeNode, 0, true, outShape));
    EXPECT_EQ(outShape, (std::vector<int64_t>{64, 1, 1, 1}));
}

TEST_F(ADepthwiseFusionPassTest, downstream_quant_bias_optimization)
{
    TestGraph builder("downstream_quant_bias_optimization");
    builder.SetSocAscend950();
    TensorInfo filterInfo(DT_FLOAT, FORMAT_HWCN, {1, 1, 16, 4}, "filter_src");
    filterInfo.SetDesc(BuildTensorDesc(DT_FLOAT, FORMAT_HWCN, {1, 1, 16, 4}, FORMAT_HWCN, {1, 1, 16, 4}));
    auto filterHolder = builder.CreateGraphInput(filterInfo);
    builder.AddDepthwiseConv2D(MakeHwcnConfig("depthwise_conv2d", {1, 1, 16, 4}), true, false);
    builder.AddNode(QuantBiasOptimizationConfig::Basic("quant_bias_opt"), {0});
    builder.SetOutput("depthwise_conv2d").SetOutput("quant_bias_opt");
    auto graph = builder.Build();
    builder.ConnectInput(filterHolder, 0, "depthwise_conv2d", 1);
    builder.ConnectInput(filterHolder, 0, "quant_bias_opt", 1);

    TestTotalPass("downstream_quant_bias_optimization", graph, SUCCESS);
    EXPECT_GE(GraphChecker::CountNodes(graph, "Reshape"), 1);
    GNode quantBiasNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "QuantBiasOptimization", quantBiasNode));
    std::vector<int64_t> quantBiasFilterShape;
    ASSERT_TRUE(GraphChecker::GetOriginShape(quantBiasNode, 1, false, quantBiasFilterShape));
    EXPECT_EQ(quantBiasFilterShape, (std::vector<int64_t>{1, 1, 1, 64}));
}

TEST_F(ADepthwiseFusionPassTest, a_depthwise_fusion_reshape_topology)
{
    struct Point {
        const char *pointName;
        std::function<void()> run;
    } const points[] = {
        {"filter_to_reshape_to_depthwise", []() {
             auto cfg = MakeHwcnConfig("depthwise_conv2d", {1, 1, 16, 4});
             auto graph = BuildDepthwiseGraph("filter_to_reshape_to_depthwise", cfg);
             CustomPassContext passContext;
             passContext.SetPassName("filter_to_reshape_to_depthwise");
             ADepthwiseFusionPass pass;
             ASSERT_EQ(pass.Run(graph, passContext), SUCCESS);
             GNode depthwise;
             GNode reshape;
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "DepthwiseConv2D", depthwise));
             ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "/Reshape_1", reshape));
             std::string producerType;
             ASSERT_TRUE(GraphChecker::GetInputProducerType(depthwise, 1, producerType));
             EXPECT_EQ(producerType, "Reshape");
             EXPECT_TRUE(GraphChecker::HasOutEdgeTo(reshape, 0, depthwise, 1));
         }},
        {"non_target_consumer_no_reshape", []() {
             auto cfg = MakeHwcnConfig("depthwise_conv2d", {1, 1, 16, 4});
             TensorInfo filterInfo(DT_FLOAT, FORMAT_HWCN, {1, 1, 16, 4}, "filter_src");
             filterInfo.SetDesc(BuildTensorDesc(DT_FLOAT, FORMAT_HWCN, {1, 1, 16, 4}, FORMAT_HWCN, {1, 1, 16, 4}));
             TestGraph builder("non_target_consumer_no_reshape");
             builder.SetSocAscend950();
             auto filterHolder = builder.CreateGraphInput(filterInfo);
             builder.AddDepthwiseConv2D(cfg, true, false);
             builder.AddRelu(ReluConfig::Basic("filter_relu", DT_FLOAT), false);
             builder.SetOutput("depthwise_conv2d").SetOutput("filter_relu").Build();
             auto graph = builder.Build();
             builder.ConnectInput(filterHolder, 0, "depthwise_conv2d", 1);
             builder.ConnectInput(filterHolder, 0, "filter_relu", 0);
             CustomPassContext passContext;
             passContext.SetPassName("non_target_consumer_no_reshape");
             ADepthwiseFusionPass pass;
             ASSERT_EQ(pass.Run(graph, passContext), SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Reshape"), 1);
             GNode filterRelu;
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Relu", filterRelu));
             std::string reluProducerType;
             ASSERT_TRUE(GraphChecker::GetInputProducerType(filterRelu, 0, reluProducerType));
             EXPECT_NE(reluProducerType, "Reshape");
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        p.run();
    }
}

TEST_F(ADepthwiseFusionPassTest, anchor_depthwise_unchanged)
{
    auto cfg = MakeHwcnConfig("depthwise_conv2d", {1, 1, 16, 4});
    cfg.SetAttr("strides", std::vector<int64_t>{1, 1, 1, 1});
    cfg.SetAttr("pads", std::vector<int64_t>{0, 0, 0, 0});
    cfg.SetAttr("dilations", std::vector<int64_t>{1, 1, 1, 1});
    auto graph = BuildDepthwiseGraph("anchor_depthwise_unchanged", cfg);
    TestTotalPass("anchor_depthwise_unchanged", graph, SUCCESS);

    GNode depthwise;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "DepthwiseConv2D", depthwise));
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    ASSERT_EQ(depthwise.GetAttr(STRIDES, strides), GRAPH_SUCCESS);
    ASSERT_EQ(depthwise.GetAttr(PADS, pads), GRAPH_SUCCESS);
    ASSERT_EQ(depthwise.GetAttr(DILATIONS, dilations), GRAPH_SUCCESS);
    EXPECT_EQ(strides, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(pads, (std::vector<int64_t>{0, 0, 0, 0}));
    EXPECT_EQ(dilations, (std::vector<int64_t>{1, 1, 1, 1}));
    std::vector<int64_t> filterShape;
    ASSERT_TRUE(GraphChecker::GetOriginShape(depthwise, 1, false, filterShape));
    EXPECT_EQ(filterShape, (std::vector<int64_t>{1, 1, 1, 64}));
}

TEST_F(ADepthwiseFusionPassTest, multi_depthwise_same_graph)
{
    auto graph = TestGraph("multi_depthwise_same_graph")
        .SetSocAscend950()
        .AddDepthwiseConv2D(MakeHwcnConfig("dw1", {1, 1, 16, 4}))
        .AddDepthwiseConv2D(MakeNchwConfig("dw2", {4, 16, 1, 1}))
        .SetOutput("dw1")
        .SetOutput("dw2")
        .Build();

    TestTotalPass("multi_depthwise_same_graph", graph, SUCCESS, 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Reshape"), 2);
}

TEST_F(ADepthwiseFusionPassTest, multi_soc_no_filter)
{
    struct Point {
        const char *pointName;
        SocConfig socConfig;
    } const points[] = {
        {"ascend950", SocConfig::Ascend950()},
        {"mc62", SocConfig::MC62()},
        {"non_nd", SocConfig("Ascend910B", "Ascend910B1")},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        std::string name = std::string("multi_soc_") + p.pointName;
        auto cfg = MakeHwcnConfig("depthwise_conv2d", {1, 1, 16, 4});
        auto graph = TestGraph(name.c_str()).SetSoc(p.socConfig).AddDepthwiseConv2D(cfg).SetOutput(cfg.name).Build();
        TestTotalPass(name, graph, SUCCESS);
        EXPECT_EQ(GraphChecker::CountNodes(graph, "Reshape"), 1);
    }
}

TEST_F(ADepthwiseFusionPassTest, hwcn_single_reshape_with_multi_consumer_filter)
{
    auto cfg = MakeHwcnConfig("depthwise_conv2d", {1, 1, 16, 16});
    auto graph = BuildDepthwiseWithFilterBranch("hwcn_single_reshape_with_multi_consumer_filter", cfg);
    TestTotalPass("hwcn_single_reshape_with_multi_consumer_filter", graph, SUCCESS);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Reshape"), 1);
}

TEST_F(ADepthwiseFusionPassTest, reentrant_run)
{
    auto cfg = MakeHwcnConfig("depthwise_conv2d", {1, 1, 16, 4});
    auto graph = BuildDepthwiseGraph("reentrant_run", cfg);

    TestTotalPass("reentrant_run_1", graph, SUCCESS);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Reshape"), 1);

    CustomPassContext passContext;
    passContext.SetPassName("reentrant_run_2");
    ADepthwiseFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), kConvNotChanged);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Reshape"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), 1);
}

#endif
