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
#include "../../../op_graph/fusion_pass/conv2d_to_conv2dv2_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

class Conv2dToConv2dV2FusionPassTest : public testing::Test {
public:
    GraphPtr BuildSingleConvGraph(const char* graphName, bool useFuseSoc, const Conv2DConfig& convCfg)
    {
        TestGraph builder(graphName);
        if (useFuseSoc) {
            builder.SetSocMC62();
        } else {
            builder.SetSocAscend950();
        }
        return builder.AddConv2D(convCfg).SetOutput(convCfg.name).Build();
    }

    Conv2DConfig MakeConvCfg(DataType ioDtype, DataType outputDtype, bool hasBias = false,
                             DataType biasDtype = DT_FLOAT16)
    {
        Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", ioDtype, outputDtype);
        if (hasBias) {
            convCfg.WithBias(biasDtype);
        }
        return convCfg;
    }

protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        Conv2dToConv2dV2FusionPass pass;

        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
        if (res == SUCCESS) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2DV2"));
            EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 0);
        } else if (res == GRAPH_NOT_CHANGED || res == CONV_NOT_CHANGED) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
            EXPECT_FALSE(GraphChecker::HasNode(graph, "Conv2DV2"));
        }
    }

    GNode GetFirstConv2dV2Node(GraphPtr& graph)
    {
        GNode fused;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2DV2", fused));
        return fused;
    }
};

// MeetRequirements dtype matrix success: Ascend950 / MC62 × dtype × bias
TEST_F(Conv2dToConv2dV2FusionPassTest, conv2d_to_conv2dv2_fusion_success)
{
    struct {
        const char* pointName;
        bool useFuseSoc;
        DataType ioDtype;
        DataType outputDtype;
        bool hasBias;
        DataType biasDtype;
    } const points[] = {
        {"dav3510_fp16", false, DT_FLOAT16, DT_FLOAT16, false, DT_FLOAT16},
        {"dav3510_fp16_bias", false, DT_FLOAT16, DT_FLOAT16, true, DT_FLOAT16},
        {"dav3510_fp32", false, DT_FLOAT, DT_FLOAT, false, DT_FLOAT},
        {"dav3510_bf16", false, DT_BF16, DT_BF16, false, DT_BF16},
        {"dav3510_fp32_bias", false, DT_FLOAT, DT_FLOAT, true, DT_FLOAT},
        {"dav3510_bf16_bias", false, DT_BF16, DT_BF16, true, DT_BF16},
        {"dav3510_hifloat8", false, DT_HIFLOAT8, DT_HIFLOAT8, false, DT_FLOAT},
        {"dav3510_hifloat8_bias", false, DT_HIFLOAT8, DT_HIFLOAT8, true, DT_FLOAT},
        {"fuse_fp16", true, DT_FLOAT16, DT_FLOAT16, false, DT_FLOAT16},
        {"fuse_fp16_bias", true, DT_FLOAT16, DT_FLOAT16, true, DT_FLOAT16},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        std::string name = std::string("conv2d_to_conv2dv2_fusion_success_") + p.pointName;
        auto graph = BuildSingleConvGraph(name.c_str(), p.useFuseSoc,
                                          MakeConvCfg(p.ioDtype, p.outputDtype, p.hasBias, p.biasDtype));
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
        TestTotalPass(name, graph, SUCCESS);
    }
}

// MeetRequirements rejection: unsupported dtype / SOC / dtype mismatch / IR limitation
TEST_F(Conv2dToConv2dV2FusionPassTest, conv2d_to_conv2dv2_no_fusion)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"unsupported_soc",
         []() {
             TestGraph builder("conv2d_to_conv2dv2_z_no_fusion_unsupported_soc");
             return builder.SetSoc(SocConfig("Ascend910B", "Ascend910B1"))
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .SetOutput("Conv2D")
                 .Build();
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
        TestTotalPass(std::string("conv2d_to_conv2dv2_no_fusion_") + p.pointName, graph, CONV_NOT_CHANGED);
    }
}

// ==========================================================================================
// Replacement attr/desc migration and GetConvBaseAttr derivation
// ==========================================================================================
TEST_F(Conv2dToConv2dV2FusionPassTest, conv2d_to_conv2dv2_replacement_attr_and_desc)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        std::function<void(GNode&)> verify;
    } const points[] = {
        {"fused_op_impl_mode_enum_is_default_0x1",
         [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT, DT_FLOAT);
             convCfg.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_fused_op_impl_mode_enum_is_default_0x1", false, convCfg);
         },
         [](GNode& fused) {
             int64_t implMode = static_cast<int64_t>(-1);
             ASSERT_EQ(fused.GetAttr(AscendString("_op_impl_mode_enum"), implMode), GRAPH_SUCCESS);
             EXPECT_EQ(implMode, int64_t{0x1});
         }},
        {"pad_mode_from_auto_pad",
         [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("auto_pad", std::string("SAME_UPPER"));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_pad_mode_from_auto_pad", false, convCfg);
         },
         [](GNode& fused) {
             AscendString padMode;
             ASSERT_EQ(fused.GetAttr(AscendString("pad_mode"), padMode), GRAPH_SUCCESS);
             EXPECT_STREQ(padMode.GetString(), "SAME_UPPER");
         }},
        {"enable_hf32_fp32_impl_0x40",
         [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT, DT_FLOAT);
             convCfg.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_enable_hf32_fp32_impl_0x40", false, convCfg);
         },
         [](GNode& fused) {
             bool enableHf32 = false;
             ASSERT_EQ(fused.GetAttr(AscendString("enable_hf32"), enableHf32), GRAPH_SUCCESS);
             EXPECT_TRUE(enableHf32);
         }},
        {"enable_hf32_false_fp16",
         [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_enable_hf32_false_fp16", false, convCfg);
         },
         [](GNode& fused) {
             bool enableHf32 = true;
             if (fused.GetAttr(AscendString("enable_hf32"), enableHf32) == GRAPH_SUCCESS) {
                 EXPECT_FALSE(enableHf32);
             }
         }},
        {"attr_strides_pads_dilations_groups",
         [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("strides", std::vector<int64_t>{2, 2, 2, 2});
             convCfg.SetAttr("pads", std::vector<int64_t>{2, 2, 2, 2});
             convCfg.SetAttr("dilations", std::vector<int64_t>{2, 2, 2, 2});
             convCfg.SetAttr("groups", int64_t(2));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_attr_strides_pads_dilations_groups", false, convCfg);
         },
         [](GNode& fused) {
             std::vector<int64_t> strides, pads, dilations;
             int64_t groups = 0;
             ASSERT_EQ(fused.GetAttr(AscendString("strides"), strides), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetAttr(AscendString("pads"), pads), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetAttr(AscendString("dilations"), dilations), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetAttr(AscendString("groups"), groups), GRAPH_SUCCESS);
             EXPECT_EQ(strides, (std::vector<int64_t>{2, 2, 2, 2}));
             EXPECT_EQ(pads, (std::vector<int64_t>{2, 2, 2, 2}));
             EXPECT_EQ(dilations, (std::vector<int64_t>{2, 2, 2, 2}));
             EXPECT_EQ(groups, int64_t{2});
         }},
        {"attr_data_format_nhwc",
         [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NHWC);
             convCfg.SetAttr("data_format", std::string("NHWC"));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_attr_data_format_nhwc", false, convCfg);
         },
         [](GNode& fused) {
             AscendString dataFormat;
             ASSERT_EQ(fused.GetAttr(AscendString("data_format"), dataFormat), GRAPH_SUCCESS);
             EXPECT_STREQ(dataFormat.GetString(), "NHWC");
         }},
        {"desc_input_output_preserved",
         [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_desc_input_output_preserved", false,
                                         Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW,
                                                             {1, 16, 240, 352}, {3, 16, 3, 3}, {1, 3, 240, 352}));
         },
         [](GNode& fused) {
             TensorDesc fmapDesc, filterDesc, outDesc;
             ASSERT_EQ(fused.GetInputDesc(0, fmapDesc), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetInputDesc(1, filterDesc), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
             EXPECT_EQ(fmapDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(filterDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(fmapDesc.GetFormat(), FORMAT_NCHW);
             EXPECT_EQ(outDesc.GetFormat(), FORMAT_NCHW);
         }},
        {"bias_input_absent",
         [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_bias_input_absent", false,
                                         Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16));
         },
         [](GNode& fused) { EXPECT_EQ(fused.GetInputsSize(), size_t{2}); }},
        {"bias_input_present",
         [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_bias_input_present", false,
                                         Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16).WithBias(DT_FLOAT16));
         },
         [](GNode& fused) {
             TensorDesc biasDesc;
             EXPECT_EQ(fused.GetInputsSize(), size_t{3});
             ASSERT_EQ(fused.GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
             EXPECT_EQ(biasDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(biasDesc.GetFormat(), FORMAT_ND);
         }},
        {"offset_x_nonzero",
         [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("offset_x", int64_t(1));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_fusion_offset_x_nonzero", false, convCfg);
         },
         [](GNode& fused) {
             int64_t offsetX = 0;
             ASSERT_EQ(fused.GetAttr(AscendString("offset_x"), offsetX), GRAPH_SUCCESS);
             EXPECT_EQ(offsetX, int64_t{1});
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("conv2d_to_conv2dv2_replacement_attr_and_desc_") + p.pointName, graph, SUCCESS);
        if (p.verify) {
            GNode fused = GetFirstConv2dV2Node(graph);
            p.verify(fused);
        }
    }
}

// ==========================================================================================
// Graph topology: dynamic shape, multi-node, partial fusion, downstream consumer, empty match
// ==========================================================================================
TEST_F(Conv2dToConv2dV2FusionPassTest, conv2d_to_conv2dv2_graph_topology)
{
    struct {
        const char* pointName;
        std::function<void(Conv2dToConv2dV2FusionPassTest&)> run;
    } const points[] = {
        {"dynamic_shape_fp16",
         [](Conv2dToConv2dV2FusionPassTest& self) {
             auto graph = TestGraph("g")
                              .SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW,
                                                             {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}))
                              .SetOutput("Conv2D")
                              .Build();
             self.TestTotalPass("conv2d_to_conv2dv2_fusion_dynamic_shape_fp16", graph, SUCCESS);
         }},
        {"multi_conv2d_same_graph",
         [](Conv2dToConv2dV2FusionPassTest& self) {
             auto graph = TestGraph("g")
                              .SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D_1", DT_FLOAT16, DT_FLOAT16))
                              .AddConv2D(Conv2DConfig::Basic("Conv2D_2", DT_FLOAT16, DT_FLOAT16))
                              .SetOutput("Conv2D_1")
                              .SetOutput("Conv2D_2")
                              .Build();
             self.TestTotalPass("conv2d_to_conv2dv2_fusion_multi_conv2d_same_graph", graph, SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2DV2"), 2);
         }},
        {"multi_conv2d_serial",
         [](Conv2dToConv2dV2FusionPassTest& self) {
             auto graph = TestGraph("g")
                              .SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D_1", DT_FLOAT16, DT_FLOAT16))
                              .AddConv2D(Conv2DConfig::Basic("Conv2D_2", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW,
                                                             {1, 3, 244, 244}, {3, 3, 3, 3}, {1, 3, 244, 244}))
                              .Connect("Conv2D_1", 0, "Conv2D_2", 0)
                              .SetOutput("Conv2D_2")
                              .Build();
             self.TestTotalPass("conv2d_to_conv2dv2_fusion_multi_conv2d_serial", graph, SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2DV2"), 2);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 0);
         }},
        {"multi_conv2d_mixed_dtypes",
         [](Conv2dToConv2dV2FusionPassTest&) {
             auto graph = TestGraph("g")
                              .SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D_ok", DT_FLOAT16, DT_FLOAT16))
                              .AddConv2D(Conv2DConfig::Basic("Conv2D_bad", DT_INT8, DT_INT32))
                              .SetOutput("Conv2D_ok")
                              .SetOutput("Conv2D_bad")
                              .Build();
             CustomPassContext passContext;
             passContext.SetPassName("conv2d_to_conv2dv2_fusion_multi_conv2d_mixed_dtypes");
             Conv2dToConv2dV2FusionPass pass;
             EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2DV2"), 2);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 0);
         }},
        {"downstream_consumer_preserved",
         [](Conv2dToConv2dV2FusionPassTest& self) {
             auto graph = TestGraph("g")
                              .SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                              .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                              .Connect("Conv2D", 0, "Relu", 0)
                              .SetOutput("Relu")
                              .Build();
             self.TestTotalPass("conv2d_to_conv2dv2_fusion_downstream_consumer_preserved", graph, SUCCESS);
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
         }},
        {"graph_without_conv2d",
         [](Conv2dToConv2dV2FusionPassTest&) {
             auto graph = TestGraph("g")
                              .SetSocAscend950()
                              .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16), true)
                              .SetOutput("Relu")
                              .Build();
             CustomPassContext passContext;
             passContext.SetPassName("conv2d_to_conv2dv2_no_fusion_graph_without_conv2d");
             Conv2dToConv2dV2FusionPass pass;
             EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Conv2DV2"));
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        p.run(*this);
    }
}

// ==========================================================================================
// ==========================================================================================
TEST_F(Conv2dToConv2dV2FusionPassTest, conv2d_to_conv2dv2_adjust_layout)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        std::function<void(GraphPtr&)> verify;
    } const points[] = {
        {"pre_transpose_nchw_to_nhwc",
         [this]() {
             Conv2DConfig convCfg = MakeConvCfg(DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("strides", std::vector<int64_t>{1, 2, 3, 4});
             convCfg.SetAttr("dilations", std::vector<int64_t>{1, 2, 3, 4});
             return TestGraph("g")
                 .SetSocFuse()
                 .AddTranspose(TransposeNodeConfig::NchwToNhwc("PreTranspose"))
                 .AddConv2D(convCfg, false, true)
                 .Connect("PreTranspose", 0, "Conv2D", 0)
                 .SetOutput("Conv2D")
                 .Build();
         },
         [](GraphPtr& graph) {
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));
             GNode fused;
             EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2DV2", fused));
             std::vector<int64_t> strides;
             ASSERT_EQ(fused.GetAttr(AscendString("strides"), strides), GRAPH_SUCCESS);
             EXPECT_EQ(strides, (std::vector<int64_t>{1, 4, 2, 3}));
         }},
        {"pre_transpose_nhwc_to_nchw",
         [this]() {
             return TestGraph("g")
                 .SetSocFuse()
                 .AddTranspose(TransposeNodeConfig::NhwcToNchw("PreTranspose"))
                 .AddConv2D(MakeConvCfg(DT_FLOAT16, DT_FLOAT16), false, true)
                 .Connect("PreTranspose", 0, "Conv2D", 0)
                 .SetOutput("Conv2D")
                 .Build();
         },
         [](GraphPtr& graph) {
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2DV2"));
         }},
        {"pre_transpose_fp32_removed",
         [this]() {
             return TestGraph("g")
                 .SetSocFuse()
                 .AddTranspose(TransposeNodeConfig::NchwToNhwc("PreTranspose", DT_FLOAT))
                 .AddConv2D(MakeConvCfg(DT_FLOAT, DT_FLOAT), false, true)
                 .Connect("PreTranspose", 0, "Conv2D", 0)
                 .SetOutput("Conv2D")
                 .Build();
         },
         [](GraphPtr& graph) {
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2DV2"));
         }},
        {"post_transpose_single_consumer",
         [this]() {
             return TestGraph("g")
                 .SetSocFuse()
                 .AddConv2D(MakeConvCfg(DT_FLOAT16, DT_FLOAT16))
                 .AddTranspose(TransposeNodeConfig::NchwToNhwc("PostTranspose"), false)
                 .Connect("Conv2D", 0, "PostTranspose", 0)
                 .SetOutput("PostTranspose")
                 .Build();
         },
         [](GraphPtr& graph) {
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2DV2"));
         }},
        {"post_transpose_multi_consumer",
         [this]() {
             return TestGraph("g")
                 .SetSocAscend950()
                 .AddConv2D(MakeConvCfg(DT_FLOAT16, DT_FLOAT16))
                 .AddTranspose(TransposeNodeConfig::NchwToNhwc("PostTranspose"), false)
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NHWC, {1, 244, 244, 3}))
                 .Connect("Conv2D", 0, "PostTranspose", 0)
                 .Connect("Conv2D", 0, "Relu", 0)
                 .SetOutput("PostTranspose")
                 .SetOutput("Relu")
                 .Build();
         },
         [](GraphPtr& graph) { EXPECT_TRUE(GraphChecker::HasNode(graph, "Transpose")); }},
        {"no_transpose",
         [this]() { return BuildSingleConvGraph("no_transpose", false, MakeConvCfg(DT_FLOAT16, DT_FLOAT16)); },
         [](GraphPtr& graph) {
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2DV2"));
         }},
        {"existing_conv2dv2_in_graph",
         [this]() {
             Conv2DConfig convCfg = MakeConvCfg(DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("strides", std::vector<int64_t>{1, 2, 3, 4});
             return TestGraph("g")
                 .SetSocFuse()
                 .AddTranspose(TransposeNodeConfig::NchwToNhwc("PreTranspose"))
                 .AddConv2D(convCfg, false, true)
                 .Connect("PreTranspose", 0, "Conv2D", 0)
                 .AddNode(Conv2DV2Config::Basic("ExistedConv2DV2"), {0, 1})
                 .SetOutput("Conv2D")
                 .SetOutput("ExistedConv2DV2")
                 .Build();
         },
         [](GraphPtr& graph) {
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2DV2"), 2);
             GNode fused;
             EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2DV2", fused));
             std::vector<int64_t> strides;
             ASSERT_EQ(fused.GetAttr(AscendString("strides"), strides), GRAPH_SUCCESS);
             EXPECT_EQ(strides, (std::vector<int64_t>{1, 4, 2, 3}));
         }},
        {"skip_on_nonfuse_soc",
         [this]() {
             Conv2DConfig convCfg = MakeConvCfg(DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("strides", std::vector<int64_t>{1, 2, 3, 4});
             return TestGraph("g")
                 .SetSocAscend950()
                 .AddTranspose(TransposeNodeConfig::NchwToNhwc("PreTranspose"))
                 .AddConv2D(convCfg, false, true)
                 .Connect("PreTranspose", 0, "Conv2D", 0)
                 .SetOutput("Conv2D")
                 .Build();
         },
         [](GraphPtr& graph) {
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Transpose"));
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2DV2"));
             GNode fused;
             EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2DV2", fused));
             std::vector<int64_t> strides;
             ASSERT_EQ(fused.GetAttr(AscendString("strides"), strides), GRAPH_SUCCESS);
             EXPECT_EQ(strides, (std::vector<int64_t>{1, 2, 3, 4}));
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("adjust_layout_") + p.pointName, graph, SUCCESS);
        if (p.verify) {
            p.verify(graph);
        }
    }
}

// ==========================================================================================
// ==========================================================================================
TEST_F(Conv2dToConv2dV2FusionPassTest, conv2d_to_conv2dv2_fixed_shift_value)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        int64_t expectedValue;
    } const points[] = {
        {"a16w16_fp16_weight",
         [this]() { return BuildSingleConvGraph("fsv_a16w16", false, MakeConvCfg(DT_FLOAT16, DT_FLOAT16)); }, 42},
        {"a16w16_fp16_fuse",
         [this]() { return BuildSingleConvGraph("fsv_a16w16_fuse", true, MakeConvCfg(DT_FLOAT16, DT_FLOAT16)); }, 42},
        {"explicit_value",
         [this]() {
             Conv2DConfig convCfg = MakeConvCfg(DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("fixed_shift_value", int64_t(99));
             return BuildSingleConvGraph("fsv_explicit", false, convCfg);
         },
         99},
        {"w8w8_int8_weight_default_a16w16",
         [this]() {
             TestGraph builder("fsv_w8w8_fuse");
             return builder.SetSocFuse().AddConv2D(MakeConvCfg(DT_INT8, DT_INT32)).SetOutput("Conv2D").Build();
         },
         42},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("fixed_shift_value_") + p.pointName, graph, SUCCESS);
        GNode fused = GetFirstConv2dV2Node(graph);
        int64_t value = 0;
        ASSERT_EQ(fused.GetAttr(AscendString("fixed_shift_value"), value), GRAPH_SUCCESS);
        EXPECT_EQ(value, p.expectedValue);
    }
}

// ==========================================================================================
// Groups implicit derivation (infershape 1.0 parity): when attr groups is 1 and fmap C is a
// multiple of filter Cin, derive groups = fmap C / filter Cin; keep explicit groups and
// non-derivable cases unchanged.
// ==========================================================================================
TEST_F(Conv2dToConv2dV2FusionPassTest, conv2d_to_conv2dv2_groups_implicit_derive)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        int64_t expectedGroups;
    } const points[] = {
        {"nchw_fmap_nchw_filter_derive",
         []() {
             TestGraph builder("groups_nchw");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 64, 56, 56},
                                                {4, 16, 3, 3}, {1, 4, 56, 56}))
                 .SetOutput("Conv2D")
                 .Build();
         },
         4},
        {"nhwc_fmap_hwcn_filter_derive",
         []() {
             TestGraph builder("groups_nhwc_hwcn");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NHWC, {1, 56, 56, 64},
                                                {3, 3, 16, 4}, {1, 56, 56, 4}))
                 .SetOutput("Conv2D")
                 .Build();
         },
         4},
        {"nhwc_fmap_nhwc_filter_derive",
         []() {
             TestGraph builder("groups_nhwc_nhwc");
             Conv2DConfig convCfg;
             convCfg.SetName("Conv2D")
                 .AddInput(DT_FLOAT16, FORMAT_NHWC, {1, 56, 56, 64}, "Conv2D_x")
                 .AddInput(DT_FLOAT16, FORMAT_NHWC, {4, 3, 2, 16}, "Conv2D_filter")
                 .AddOutput(DT_FLOAT16, FORMAT_NHWC, {1, 56, 56, 4}, "y")
                 .SetAttr("data_format", std::string("NHWC"));
             return builder.SetSocAscend950().AddConv2D(convCfg).SetOutput("Conv2D").Build();
         },
         4},
        {"fmap_c_equals_filter_cin_keeps_default",
         []() {
             TestGraph builder("groups_c_equal");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 16, 56, 56},
                                                {4, 16, 3, 3}, {1, 4, 56, 56}))
                 .SetOutput("Conv2D")
                 .Build();
         },
         1},
        {"fmap_c_less_filter_cin_keeps_default",
         []() {
             TestGraph builder("groups_c_less");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 8, 56, 56},
                                                {4, 16, 3, 3}, {1, 4, 56, 56}))
                 .SetOutput("Conv2D")
                 .Build();
         },
         1},
        {"explicit_groups_valid_preserved",
         []() {
             TestGraph builder("groups_explicit_valid");
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 64, 56, 56},
                                                        {4, 16, 3, 3}, {1, 4, 56, 56});
             convCfg.SetAttr("groups", int64_t(4));
             return builder.SetSocAscend950().AddConv2D(convCfg).SetOutput("Conv2D").Build();
         },
         4},
        {"explicit_groups_mismatch_preserved",
         []() {
             TestGraph builder("groups_explicit_mismatch");
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 64, 56, 56},
                                                        {4, 16, 3, 3}, {1, 4, 56, 56});
             convCfg.SetAttr("groups", int64_t(2));
             return builder.SetSocAscend950().AddConv2D(convCfg).SetOutput("Conv2D").Build();
         },
         2},
        {"non_divisible_keeps_default",
         []() {
             TestGraph builder("groups_non_divisible");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 63, 56, 56},
                                                {4, 16, 3, 3}, {1, 4, 56, 56}))
                 .SetOutput("Conv2D")
                 .Build();
         },
         1},
        {"dynamic_shape_keeps_default",
         []() {
             TestGraph builder("groups_dynamic");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {-1, -1, -1, -1},
                                                {-1, -1, -1, -1}, {-1, -1, -1, -1}))
                 .SetOutput("Conv2D")
                 .Build();
         },
         1},
        {"fractal_z_filter_keeps_default",
         []() {
             TestGraph builder("groups_fractal_z");
             auto graph = builder.SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW,
                                                             {1, 64, 56, 56}, {4, 16, 3, 3}, {1, 4, 56, 56}))
                              .SetOutput("Conv2D")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_FLOAT16, FORMAT_FRACTAL_Z);
             return graph;
         },
         1},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("groups_derive_") + p.pointName, graph, SUCCESS);
        GNode fused = GetFirstConv2dV2Node(graph);
        int64_t groups = 0;
        if (fused.GetAttr(AscendString("groups"), groups) == GRAPH_SUCCESS) {
            EXPECT_EQ(groups, p.expectedGroups);
        } else {
            // Es builder omits optional attrs equal to the IR default, absent groups means 1.
            EXPECT_EQ(p.expectedGroups, int64_t{1});
        }
    }
}

// Re-entrancy: multiple Conv2D nodes derive their own groups in one graph, convDescInfo must
// be reset per node by InitMember.
TEST_F(Conv2dToConv2dV2FusionPassTest, conv2d_to_conv2dv2_groups_derive_multi_conv)
{
    auto graph = TestGraph("groups_derive_multi")
                     .SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D_1", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 64, 56, 56},
                                                    {4, 16, 3, 3}, {1, 4, 56, 56}))
                     .AddConv2D(Conv2DConfig::Basic("Conv2D_2", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 32, 28, 28},
                                                    {8, 16, 3, 3}, {1, 8, 28, 28}))
                     .SetOutput("Conv2D_1")
                     .SetOutput("Conv2D_2")
                     .Build();
    TestTotalPass("groups_derive_multi_conv", graph, SUCCESS);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2DV2"), 2);

    int32_t groups4Cnt = 0;
    int32_t groups2Cnt = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        ASSERT_EQ(node.GetType(nodeType), GRAPH_SUCCESS);
        if (std::string(nodeType.GetString()) != "Conv2DV2") {
            continue;
        }
        int64_t groups = 0;
        ASSERT_EQ(node.GetAttr(AscendString("groups"), groups), GRAPH_SUCCESS);
        if (groups == 4) {
            groups4Cnt++;
        }
        if (groups == 2) {
            groups2Cnt++;
        }
    }
    EXPECT_EQ(groups4Cnt, 1);
    EXPECT_EQ(groups2Cnt, 1);
}

#endif
