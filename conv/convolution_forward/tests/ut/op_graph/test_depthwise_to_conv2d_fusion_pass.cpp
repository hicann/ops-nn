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
#include "../../../op_graph/fusion_pass/depthwise_to_conv2d_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace DepthwiseToConv2dFusion;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

namespace {
GraphPtr BuildDepthwiseGraph(const char *graphName, const DepthwiseConv2DConfig &depthwiseCfg,
    const SocConfig *socConfig = nullptr) {
    TestGraph builder(graphName);
    if (socConfig != nullptr) {
        builder.SetSoc(*socConfig);
    } else {
        builder.SetSocAscend950();
    }
    return builder.AddDepthwiseConv2D(depthwiseCfg).SetOutput(depthwiseCfg.name).Build();
}

const SocConfig kMc62SocConfig = SocConfig::MC62();
const SocConfig kNonNdSocConfig("Ascend910B", "Ascend910B1");
} // namespace

class DepthwiseToConv2dFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "DepthwiseToConv2dFusionPassTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "DepthwiseToConv2dFusionPassTest TearDown" << std::endl;
    }

    void TestTotalPass(const std::string &passName, GraphPtr &graph, Status expectRes,
        int32_t depthwiseCountBefore = 1) {
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        DepthwiseToConv2dFusionPass pass({DEPTHWISE_CONV2D});

        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
        if (res == SUCCESS) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
            EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), 0);
        } else if (res == GRAPH_NOT_CHANGED) {
            EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), depthwiseCountBefore);
        }
    }

    GNode GetFirstConv2dNode(GraphPtr &graph) {
        GNode fused;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", fused));
        return fused;
    }
};

TEST_F(DepthwiseToConv2dFusionPassTest, depthwise_to_conv2d_fusion_success) {
    struct Point {
        const char *pointName;
        const SocConfig *socConfig;
        DepthwiseConv2DConfig cfg;
    } const points[] = {
        {"dynamic_nchw_dav3510", nullptr,
            DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT, FORMAT_NCHW, {-1, 16, 256, 256},
                {1, 1, 1, 64}, {-1, 4, 256, 256})},
        {"static_nchw_dav3510", nullptr,
            DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT, FORMAT_NCHW, {1, 16, 256, 256},
                {1, 1, 1, 64}, {1, 4, 256, 256})},
        {"dynamic_nchw_dav5102", &kMc62SocConfig,
            DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT16, FORMAT_NCHW, {-1, 16, 256, 256},
                {1, 1, 1, 64}, {-1, 4, 256, 256})},
        {"dynamic_nhwc_dav3510", nullptr,
            DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT, FORMAT_NHWC, {-1, 256, 256, 16},
                {1, 1, 1, 64}, {-1, 256, 256, 4})},
        {"dynamic_nchw_with_bias", nullptr, DepthwiseConv2DConfig::Basic("depthwise_conv2d").WithBias()},
        {"static_nchw_mc62", &kMc62SocConfig,
            DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT16, FORMAT_NCHW, {1, 16, 256, 256},
                {1, 1, 1, 64}, {1, 4, 256, 256})},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        std::string name = std::string("depthwise_to_conv2d_fusion_success_") + p.pointName;
        auto graph = BuildDepthwiseGraph(name.c_str(), p.cfg, p.socConfig);
        EXPECT_TRUE(GraphChecker::HasNode(graph, "DepthwiseConv2D"));
        TestTotalPass(name, graph, SUCCESS);

        auto convNode = GetFirstConv2dNode(graph);
        int64_t groups = 0;
        EXPECT_EQ(convNode.GetAttr(GROUPS, groups), GRAPH_SUCCESS);
        EXPECT_EQ(groups, 16);
    }
}

TEST_F(DepthwiseToConv2dFusionPassTest, depthwise_to_conv2d_no_fusion) {
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
        Status expectRes;
        int32_t depthwiseCountBefore;
    } const points[] = {
        {"wrong_op_type", []() {
             return TestGraph("depthwise_to_conv2d_no_fusion_wrong_op_type")
                 .SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("conv2d", DT_FLOAT, DT_FLOAT, FORMAT_NCHW, {1, 16, 256, 256},
                     {1, 1, 1, 64}, {1, 4, 256, 256}))
                 .SetOutput("conv2d")
                 .Build();
         }, GRAPH_NOT_CHANGED, 0},
        {"static_shape_non_nd_soc", []() {
             return BuildDepthwiseGraph("depthwise_to_conv2d_no_fusion_static_non_nd",
                 DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT, FORMAT_NCHW, {1, 16, 256, 256},
                     {1, 1, 1, 64}, {1, 4, 256, 256}),
                 &kNonNdSocConfig);
         }, GRAPH_NOT_CHANGED, 1},
        {"unsupported_format", []() {
             DepthwiseConv2DConfig cfg = DepthwiseConv2DConfig::Basic("depthwise_conv2d");
             cfg.inputs[0].format = FORMAT_NC1HWC0;
             cfg.inputs[0].tensorDesc = BuildTensorDesc(DT_FLOAT, FORMAT_NC1HWC0, {-1, 16, 256, 256},
                 FORMAT_NC1HWC0, {-1, 16, 256, 256});
             return BuildDepthwiseGraph("depthwise_to_conv2d_no_fusion_unsupported_format", cfg);
         }, GRAPH_NOT_CHANGED, 1},
        {"unknown_fmap_channel", []() {
             return BuildDepthwiseGraph("depthwise_to_conv2d_no_fusion_unknown_channel",
                 DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT, FORMAT_NCHW, {-1, -1, 256, 256}));
         }, GRAPH_NOT_CHANGED, 1},
        {"fmap_not_4d", []() {
             DepthwiseConv2DConfig cfg = DepthwiseConv2DConfig::Basic("depthwise_conv2d");
             cfg.inputs[0].shape = {-1, 16, 256};
             cfg.inputs[0].tensorDesc =
                 BuildTensorDesc(DT_FLOAT, FORMAT_NCHW, {-1, 16, 256}, FORMAT_NCHW, {-1, 16, 256});
             return BuildDepthwiseGraph("depthwise_to_conv2d_no_fusion_fmap_not_4d", cfg);
         }, GRAPH_NOT_CHANGED, 1},
        {"empty_graph_no_match", []() {
             return TestGraph("depthwise_to_conv2d_no_fusion_empty_graph")
                 .SetSocAscend950()
                 .AddRelu(ReluConfig::Basic("relu", DT_FLOAT))
                 .SetOutput("relu")
                 .Build();
         }, GRAPH_NOT_CHANGED, 0},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        std::string name = std::string("depthwise_to_conv2d_no_fusion_") + p.pointName;
        TestTotalPass(name, graph, p.expectRes, p.depthwiseCountBefore);
    }
}

TEST_F(DepthwiseToConv2dFusionPassTest, depthwise_to_conv2d_fusion_dynamic_non_nd_soc) {
    auto graph = BuildDepthwiseGraph("depthwise_to_conv2d_fusion_dynamic_non_nd",
        DepthwiseConv2DConfig::Basic("depthwise_conv2d"), &kNonNdSocConfig);
    TestTotalPass("depthwise_to_conv2d_fusion_dynamic_non_nd", graph, SUCCESS);
}

TEST_F(DepthwiseToConv2dFusionPassTest, depthwise_to_conv2d_fusion_desc_and_structure) {
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
        std::function<void(GraphPtr &)> verify;
    } const points[] = {
        {"desc_naming_and_origin_shape", []() {
             return BuildDepthwiseGraph("depthwise_to_conv2d_desc_naming",
                 DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT16, FORMAT_NCHW, {1, 16, 240, 352},
                     {1, 1, 1, 64}, {1, 4, 240, 352}).WithBias(DT_FLOAT));
         }, [](GraphPtr &graph) {
             GNode fused;
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", fused));
             std::string nodeName;
             ASSERT_TRUE(GraphChecker::GetNodeName(fused, nodeName));
             EXPECT_NE(nodeName.find("_To_Conv2D"), std::string::npos);
             EXPECT_EQ(fused.GetInputsSize(), size_t{3});
             TensorDesc fmapDesc;
             TensorDesc filterDesc;
             TensorDesc biasDesc;
             TensorDesc outDesc;
             ASSERT_EQ(fused.GetInputDesc(INPUT_FMAP_INDEX, fmapDesc), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetInputDesc(INPUT_FILTER_INDEX, filterDesc), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetInputDesc(INPUT_BIAS_INDEX, biasDesc), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetOutputDesc(OUTPUT_INDEX, outDesc), GRAPH_SUCCESS);
             EXPECT_EQ(fmapDesc.GetOriginShape().GetDims(), (std::vector<int64_t>{1, 16, 240, 352}));
             EXPECT_EQ(filterDesc.GetOriginShape().GetDims(), (std::vector<int64_t>{1, 1, 1, 64}));
             EXPECT_EQ(biasDesc.GetOriginShape().GetDims(), (std::vector<int64_t>{64}));
             EXPECT_EQ(outDesc.GetOriginShape().GetDims(), (std::vector<int64_t>{1, 4, 240, 352}));
             EXPECT_EQ(fmapDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
         }},
        {"replacement_subgraph_structure", []() {
             return TestGraph("depthwise_to_conv2d_replacement_structure")
                 .SetSocAscend950()
                 .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("dw"))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT))
                 .Connect("dw", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         }, [](GraphPtr &graph) {
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), 0);
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
             GNode relu;
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Relu", relu));
             std::string reluProducerType;
             ASSERT_TRUE(GraphChecker::GetInputProducerType(relu, 0, reluProducerType));
             EXPECT_EQ(reluProducerType, "Conv2D");
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("depthwise_to_conv2d_fusion_desc_and_structure_") + p.pointName, graph, SUCCESS);
        if (p.verify) {
            p.verify(graph);
        }
    }
}

TEST_F(DepthwiseToConv2dFusionPassTest, depthwise_to_conv2d_replacement_attr_and_desc) {
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
        std::function<void(GNode &)> verify;
    } const points[] = {
        {"attr_strides_pads_dilations_passthrough", []() {
             DepthwiseConv2DConfig cfg = DepthwiseConv2DConfig::Basic("depthwise_conv2d");
             cfg.SetAttr("strides", std::vector<int64_t>{2, 2, 2, 2});
             cfg.SetAttr("pads", std::vector<int64_t>{2, 2, 2, 2});
             cfg.SetAttr("dilations", std::vector<int64_t>{2, 2, 2, 2});
             return BuildDepthwiseGraph("depthwise_to_conv2d_attr_strides_pads_dilations", cfg);
         }, [](GNode &fused) {
             std::vector<int64_t> strides;
             std::vector<int64_t> pads;
             std::vector<int64_t> dilations;
             ASSERT_EQ(fused.GetAttr(STRIDES, strides), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetAttr(PADS, pads), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetAttr(DILATIONS, dilations), GRAPH_SUCCESS);
             EXPECT_EQ(strides, (std::vector<int64_t>{2, 2, 2, 2}));
             EXPECT_EQ(pads, (std::vector<int64_t>{2, 2, 2, 2}));
             EXPECT_EQ(dilations, (std::vector<int64_t>{2, 2, 2, 2}));
         }},
        {"attr_groups_nchw", []() {
             return BuildDepthwiseGraph("depthwise_to_conv2d_attr_groups_nchw",
                 DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT, FORMAT_NCHW, {-1, 24, 256, 256}));
         }, [](GNode &fused) {
             int64_t groups = 0;
             ASSERT_EQ(fused.GetAttr(GROUPS, groups), GRAPH_SUCCESS);
             EXPECT_EQ(groups, int64_t{24});
         }},
        {"attr_groups_nhwc", []() {
             return BuildDepthwiseGraph("depthwise_to_conv2d_attr_groups_nhwc",
                 DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT, FORMAT_NHWC, {-1, 256, 256, 32},
                     {1, 1, 1, 64}, {-1, 256, 256, 8}));
         }, [](GNode &fused) {
             int64_t groups = 0;
             ASSERT_EQ(fused.GetAttr(GROUPS, groups), GRAPH_SUCCESS);
             EXPECT_EQ(groups, int64_t{32});
         }},
        {"attr_data_format_nhwc", []() {
             DepthwiseConv2DConfig cfg = DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT, FORMAT_NHWC,
                 {-1, 256, 256, 16});
             cfg.SetAttr("data_format", std::string("NHWC"));
             return BuildDepthwiseGraph("depthwise_to_conv2d_attr_data_format_nhwc", cfg);
         }, [](GNode &fused) {
             AscendString dataFormat;
             ASSERT_EQ(fused.GetAttr(DATA_FORMAT, dataFormat), GRAPH_SUCCESS);
             EXPECT_STREQ(dataFormat.GetString(), "NHWC");
         }},
        {"attr_padding_passthrough", []() {
             DepthwiseConv2DConfig cfg = DepthwiseConv2DConfig::Basic("depthwise_conv2d");
             cfg.SetAttr("padding", std::string("SAME"));
             return BuildDepthwiseGraph("depthwise_to_conv2d_attr_padding_passthrough", cfg);
         }, [](GNode &fused) {
             AscendString paddingVal;
             ASSERT_EQ(fused.GetAttr(PADDING, paddingVal), GRAPH_SUCCESS);
             EXPECT_STREQ(paddingVal.GetString(), "SAME");
         }},
        {"offset_x_on_nd_soc", []() {
             DepthwiseConv2DConfig cfg = DepthwiseConv2DConfig::Basic("depthwise_conv2d");
             cfg.SetAttr("offset_x", int64_t{2});
             return BuildDepthwiseGraph("depthwise_to_conv2d_offset_x_on_nd_soc", cfg);
         }, [](GNode &fused) {
             int64_t offsetX = 0;
             ASSERT_EQ(fused.GetAttr(OFFSET_X, offsetX), GRAPH_SUCCESS);
             EXPECT_EQ(offsetX, int64_t{2});
         }},
        {"offset_x_not_set_on_non_nd_soc", []() {
             DepthwiseConv2DConfig cfg = DepthwiseConv2DConfig::Basic("depthwise_conv2d");
             cfg.SetAttr("offset_x", int64_t{2});
             return BuildDepthwiseGraph("depthwise_to_conv2d_offset_x_not_set_on_non_nd_soc", cfg, &kNonNdSocConfig);
         }, [](GNode &fused) {
             int64_t offsetX = 0;
             EXPECT_NE(fused.GetAttr(OFFSET_X, offsetX), GRAPH_SUCCESS);
         }},
        {"op_impl_mode_enum_passthrough", []() {
             DepthwiseConv2DConfig cfg = DepthwiseConv2DConfig::Basic("depthwise_conv2d");
             cfg.SetAttr("_op_impl_mode_enum", int64_t{0x40});
             return BuildDepthwiseGraph("depthwise_to_conv2d_op_impl_mode_enum", cfg);
         }, [](GNode &fused) {
             int64_t implMode = 0;
             ASSERT_EQ(fused.GetAttr(OP_IMPL_MODE_ENUM, implMode), GRAPH_SUCCESS);
             EXPECT_EQ(implMode, int64_t{0x40});
         }},
        {"desc_input_output_preserved", []() {
             return BuildDepthwiseGraph("depthwise_to_conv2d_desc_input_output_preserved",
                 DepthwiseConv2DConfig::Basic("depthwise_conv2d", DT_FLOAT16, FORMAT_NCHW, {1, 16, 240, 352},
                     {1, 1, 1, 64}, {1, 4, 240, 352}));
         }, [](GNode &fused) {
             TensorDesc fmapDesc;
             TensorDesc filterDesc;
             TensorDesc outDesc;
             ASSERT_EQ(fused.GetInputDesc(INPUT_FMAP_INDEX, fmapDesc), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetInputDesc(INPUT_FILTER_INDEX, filterDesc), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetOutputDesc(OUTPUT_INDEX, outDesc), GRAPH_SUCCESS);
             EXPECT_EQ(fmapDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(filterDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(fmapDesc.GetFormat(), FORMAT_NCHW);
             EXPECT_EQ(outDesc.GetFormat(), FORMAT_NCHW);
         }},
        {"bias_input_absent", []() {
             return BuildDepthwiseGraph("depthwise_to_conv2d_bias_input_absent",
                 DepthwiseConv2DConfig::Basic("depthwise_conv2d"));
         }, [](GNode &fused) {
             EXPECT_EQ(fused.GetInputsSize(), size_t{2});
         }},
        {"bias_input_present", []() {
             return BuildDepthwiseGraph("depthwise_to_conv2d_bias_input_present",
                 DepthwiseConv2DConfig::Basic("depthwise_conv2d").WithBias(DT_FLOAT));
         }, [](GNode &fused) {
             TensorDesc biasDesc;
             EXPECT_EQ(fused.GetInputsSize(), size_t{3});
             ASSERT_EQ(fused.GetInputDesc(INPUT_BIAS_INDEX, biasDesc), GRAPH_SUCCESS);
             EXPECT_EQ(biasDesc.GetDataType(), DT_FLOAT);
             EXPECT_EQ(biasDesc.GetFormat(), FORMAT_ND);
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("depthwise_to_conv2d_replacement_attr_and_desc_") + p.pointName, graph, SUCCESS);
        if (p.verify) {
            GNode fused = GetFirstConv2dNode(graph);
            p.verify(fused);
        }
    }
}

TEST_F(DepthwiseToConv2dFusionPassTest, depthwise_to_conv2d_graph_topology) {
    struct Point {
        const char *pointName;
        std::function<void(DepthwiseToConv2dFusionPassTest &)> run;
    } const points[] = {
        {"multi_depthwise_same_graph", [](DepthwiseToConv2dFusionPassTest &self) {
             auto graph = TestGraph("depthwise_to_conv2d_multi_depthwise_same_graph")
                 .SetSocAscend950()
                 .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("dw1"))
                 .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("dw2"))
                 .SetOutput("dw1")
                 .SetOutput("dw2")
                 .Build();
             self.TestTotalPass("depthwise_to_conv2d_multi_depthwise_same_graph", graph, SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 2);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), 0);
         }},
        {"multi_depthwise_serial", [](DepthwiseToConv2dFusionPassTest &self) {
             auto graph = TestGraph("depthwise_to_conv2d_multi_depthwise_serial")
                 .SetSocAscend950()
                 .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("dw1"))
                 .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("dw2", DT_FLOAT, FORMAT_NCHW,
                     {1, 4, 256, 256}, {1, 1, 1, 64}, {1, 2, 256, 256}))
                 .Connect("dw1", 0, "dw2", 0)
                 .SetOutput("dw2")
                 .Build();
             self.TestTotalPass("depthwise_to_conv2d_multi_depthwise_serial", graph, SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 2);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), 0);
         }},
        {"multi_depthwise_mixed_eligibility", [](DepthwiseToConv2dFusionPassTest &) {
             auto graph = TestGraph("depthwise_to_conv2d_multi_depthwise_mixed_eligibility")
                 .SetSocAscend950()
                 .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("dw_ok"))
                 .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("dw_bad", DT_FLOAT, FORMAT_NCHW,
                     {-1, -1, 256, 256}))
                 .SetOutput("dw_ok")
                 .SetOutput("dw_bad")
                 .Build();
             CustomPassContext passContext;
             passContext.SetPassName("depthwise_to_conv2d_multi_depthwise_mixed_eligibility");
             DepthwiseToConv2dFusionPass pass({DEPTHWISE_CONV2D});
             EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), 1);
         }},
        {"downstream_consumer_preserved", [](DepthwiseToConv2dFusionPassTest &self) {
             auto graph = TestGraph("depthwise_to_conv2d_downstream_consumer_preserved")
                 .SetSocAscend950()
                 .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("dw"))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT))
                 .Connect("dw", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
             self.TestTotalPass("depthwise_to_conv2d_downstream_consumer_preserved", graph, SUCCESS);
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        p.run(*this);
    }
}

TEST_F(DepthwiseToConv2dFusionPassTest, depthwise_to_conv2d_reentrant) {
    auto graph = BuildDepthwiseGraph("depthwise_to_conv2d_reentrant",
        DepthwiseConv2DConfig::Basic("depthwise_conv2d"));

    TestTotalPass("depthwise_to_conv2d_reentrant_1", graph, SUCCESS);

    CustomPassContext passContext;
    passContext.SetPassName("depthwise_to_conv2d_reentrant_2");
    DepthwiseToConv2dFusionPass pass({DEPTHWISE_CONV2D});
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "DepthwiseConv2D"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
}

#endif
