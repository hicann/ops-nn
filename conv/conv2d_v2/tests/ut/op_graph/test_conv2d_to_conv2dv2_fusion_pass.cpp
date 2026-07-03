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
    GraphPtr BuildSingleConvGraph(const char* graphName, bool useDav5102, const Conv2DConfig& convCfg)
    {
        TestGraph builder(graphName);
        if (useDav5102) {
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
    static void SetUpTestCase()
    {
        std::cout << "Conv2dToConv2dV2FusionPassTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Conv2dToConv2dV2FusionPassTest TearDown" << std::endl;
    }

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        Conv2dToConv2dV2FusionPass pass({AscendString("Conv2D")});

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
        } else if (res == GRAPH_NOT_CHANGED) {
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

// ==========================================================================================
// MeetRequirements dtype matrix success: DAV_3510 / DAV_5102 × dtype × bias
// ==========================================================================================
TEST_F(Conv2dToConv2dV2FusionPassTest, conv2d_to_conv2dv2_fusion_success)
{
    struct {
        const char* pointName;
        bool useDav5102;
        DataType ioDtype;
        DataType outputDtype;
        bool hasBias;
        DataType biasDtype;
    } const points[] = {
        {"dav3510_fp16",          false, DT_FLOAT16,  DT_FLOAT16,  false, DT_FLOAT16},
        {"dav3510_fp16_bias",     false, DT_FLOAT16,  DT_FLOAT16,  true,  DT_FLOAT16},
        {"dav3510_fp32",          false, DT_FLOAT,    DT_FLOAT,    false, DT_FLOAT},
        {"dav3510_bf16",          false, DT_BF16,     DT_BF16,     false, DT_BF16},
        {"dav3510_fp32_bias",     false, DT_FLOAT,    DT_FLOAT,    true,  DT_FLOAT},
        {"dav3510_bf16_bias",     false, DT_BF16,     DT_BF16,     true,  DT_BF16},
        {"dav3510_hifloat8",      false, DT_HIFLOAT8, DT_HIFLOAT8, false, DT_FLOAT},
        {"dav3510_hifloat8_bias", false, DT_HIFLOAT8, DT_HIFLOAT8, true,  DT_FLOAT},
        {"dav5102_fp16",          true,  DT_FLOAT16,  DT_FLOAT16,  false, DT_FLOAT16},
        {"dav5102_fp16_bias",     true,  DT_FLOAT16,  DT_FLOAT16,  true,  DT_FLOAT16},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        std::string name = std::string("conv2d_to_conv2dv2_fusion_success_") + p.pointName;
        auto graph = BuildSingleConvGraph(name.c_str(), p.useDav5102,
                                          MakeConvCfg(p.ioDtype, p.outputDtype, p.hasBias, p.biasDtype));
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
        TestTotalPass(name, graph, SUCCESS);
    }
}

// ==========================================================================================
// MeetRequirements rejection: unsupported dtype / SOC / dtype mismatch / IR limitation
// ==========================================================================================
TEST_F(Conv2dToConv2dV2FusionPassTest, conv2d_to_conv2dv2_no_fusion)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"unsupported_dtype", [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_no_fusion_unsupported_dtype", false,
                                         Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32));
         }},
        {"bad_bias_dtype", [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_no_fusion_bad_bias_dtype", false,
                                         Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16).WithBias(DT_INT32));
         }},
        {"dav5102_int8_proto_unsupported", [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_no_fusion_dav5102_int8_proto_unsupported", true,
                                         Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32));
         }},
        {"filter_dtype_mismatch", [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16);
             convCfg.inputs[1].SetDtype(DT_FLOAT);
             convCfg.inputs[1].tensorDesc.SetDataType(DT_FLOAT);
             return BuildSingleConvGraph("conv2d_to_conv2dv2_no_fusion_filter_dtype_mismatch", false, convCfg);
         }},
        {"output_dtype_mismatch", [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_no_fusion_output_dtype_mismatch", false,
                                         Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT));
         }},
        {"hifloat8_bad_output", [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_no_fusion_hifloat8_bad_output", false,
                                         Conv2DConfig::Basic("Conv2D", DT_HIFLOAT8, DT_FLOAT));
         }},
        {"unsupported_soc", []() {
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
        TestTotalPass(std::string("conv2d_to_conv2dv2_no_fusion_") + p.pointName, graph, GRAPH_NOT_CHANGED);
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
        {"fused_op_impl_mode_enum_is_default_0x1", [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT, DT_FLOAT);
             convCfg.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_fused_op_impl_mode_enum_is_default_0x1", false, convCfg);
         }, [](GNode& fused) {
             int64_t implMode = static_cast<int64_t>(-1);
             ASSERT_EQ(fused.GetAttr(AscendString("_op_impl_mode_enum"), implMode), GRAPH_SUCCESS);
             EXPECT_EQ(implMode, int64_t{0x1});
         }},
        {"pad_mode_from_auto_pad", [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("auto_pad", std::string("SAME_UPPER"));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_pad_mode_from_auto_pad", false, convCfg);
         }, [](GNode& fused) {
             AscendString padMode;
             ASSERT_EQ(fused.GetAttr(AscendString("pad_mode"), padMode), GRAPH_SUCCESS);
             EXPECT_STREQ(padMode.GetString(), "SAME_UPPER");
         }},
        {"enable_hf32_fp32_impl_0x40", [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT, DT_FLOAT);
             convCfg.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_enable_hf32_fp32_impl_0x40", false, convCfg);
         }, [](GNode& fused) {
             bool enableHf32 = false;
             ASSERT_EQ(fused.GetAttr(AscendString("enable_hf32"), enableHf32), GRAPH_SUCCESS);
             EXPECT_TRUE(enableHf32);
         }},
        {"enable_hf32_false_fp16", [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_enable_hf32_false_fp16", false, convCfg);
         }, nullptr},
        {"attr_strides_pads_dilations_groups", [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("strides", std::vector<int64_t>{2, 2, 2, 2});
             convCfg.SetAttr("pads", std::vector<int64_t>{2, 2, 2, 2});
             convCfg.SetAttr("dilations", std::vector<int64_t>{2, 2, 2, 2});
             convCfg.SetAttr("groups", int64_t(2));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_attr_strides_pads_dilations_groups", false, convCfg);
         }, [](GNode& fused) {
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
        {"attr_data_format_nhwc", [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NHWC);
             convCfg.SetAttr("data_format", std::string("NHWC"));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_attr_data_format_nhwc", false, convCfg);
         }, [](GNode& fused) {
             AscendString dataFormat;
             ASSERT_EQ(fused.GetAttr(AscendString("data_format"), dataFormat), GRAPH_SUCCESS);
             EXPECT_STREQ(dataFormat.GetString(), "NHWC");
         }},
        {"desc_input_output_preserved", [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_desc_input_output_preserved", false,
                 Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW,
                     {1, 16, 240, 352}, {3, 16, 3, 3}, {1, 3, 240, 352}));
         }, [](GNode& fused) {
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
        {"bias_input_absent", [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_bias_input_absent", false,
                 Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16));
         }, [](GNode& fused) { EXPECT_EQ(fused.GetInputsSize(), size_t{2}); }},
        {"bias_input_present", [this]() {
             return BuildSingleConvGraph("conv2d_to_conv2dv2_bias_input_present", false,
                 Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16).WithBias(DT_FLOAT16));
         }, [](GNode& fused) {
             TensorDesc biasDesc;
             EXPECT_EQ(fused.GetInputsSize(), size_t{3});
             ASSERT_EQ(fused.GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
             EXPECT_EQ(biasDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(biasDesc.GetFormat(), FORMAT_ND);
         }},
        {"offset_x_nonzero", [this]() {
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16);
             convCfg.SetAttr("offset_x", int64_t(1));
             return BuildSingleConvGraph("conv2d_to_conv2dv2_fusion_offset_x_nonzero", false, convCfg);
         }, [](GNode& fused) {
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
        {"dynamic_shape_fp16", [](Conv2dToConv2dV2FusionPassTest& self) {
             auto graph = TestGraph("g").
                 SetSocAscend950().
                 AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW,
                     {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1})).
                 SetOutput("Conv2D").
                 Build();
             self.TestTotalPass("conv2d_to_conv2dv2_fusion_dynamic_shape_fp16", graph, SUCCESS);
         }},
        {"multi_conv2d_same_graph", [](Conv2dToConv2dV2FusionPassTest& self) {
             auto graph = TestGraph("g").
                 SetSocAscend950().
                 AddConv2D(Conv2DConfig::Basic("Conv2D_1", DT_FLOAT16, DT_FLOAT16)).
                 AddConv2D(Conv2DConfig::Basic("Conv2D_2", DT_FLOAT16, DT_FLOAT16)).
                 SetOutput("Conv2D_1").SetOutput("Conv2D_2").
                 Build();
             self.TestTotalPass("conv2d_to_conv2dv2_fusion_multi_conv2d_same_graph", graph, SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2DV2"), 2);
         }},
        {"multi_conv2d_serial", [](Conv2dToConv2dV2FusionPassTest& self) {
             auto graph = TestGraph("g").
                 SetSocAscend950().
                 AddConv2D(Conv2DConfig::Basic("Conv2D_1", DT_FLOAT16, DT_FLOAT16)).
                 AddConv2D(Conv2DConfig::Basic("Conv2D_2", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW,
                     {1, 3, 244, 244}, {3, 3, 3, 3}, {1, 3, 244, 244})).
                 Connect("Conv2D_1", 0, "Conv2D_2", 0).
                 SetOutput("Conv2D_2").
                 Build();
             self.TestTotalPass("conv2d_to_conv2dv2_fusion_multi_conv2d_serial", graph, SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2DV2"), 2);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 0);
         }},
        {"multi_conv2d_mixed_eligibility", [](Conv2dToConv2dV2FusionPassTest&) {
             auto graph = TestGraph("g").
                 SetSocAscend950().
                 AddConv2D(Conv2DConfig::Basic("Conv2D_ok", DT_FLOAT16, DT_FLOAT16)).
                 AddConv2D(Conv2DConfig::Basic("Conv2D_bad", DT_INT8, DT_INT32)).
                 SetOutput("Conv2D_ok").SetOutput("Conv2D_bad").
                 Build();
             CustomPassContext passContext;
             passContext.SetPassName("conv2d_to_conv2dv2_fusion_multi_conv2d_mixed_eligibility");
             Conv2dToConv2dV2FusionPass pass({AscendString("Conv2D")});
             EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2DV2"), 1);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
         }},
        {"downstream_consumer_preserved", [](Conv2dToConv2dV2FusionPassTest& self) {
             auto graph = TestGraph("g").
                 SetSocAscend950().
                 AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16)).
                 AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16)).
                 Connect("Conv2D", 0, "Relu", 0).
                 SetOutput("Relu").
                 Build();
             self.TestTotalPass("conv2d_to_conv2dv2_fusion_downstream_consumer_preserved", graph, SUCCESS);
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
         }},
        {"graph_without_conv2d", [](Conv2dToConv2dV2FusionPassTest&) {
             auto graph = TestGraph("g").
                 SetSocAscend950().
                 AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16), true).
                 SetOutput("Relu").
                 Build();
             CustomPassContext passContext;
             passContext.SetPassName("conv2d_to_conv2dv2_no_fusion_graph_without_conv2d");
             Conv2dToConv2dV2FusionPass pass({AscendString("Conv2D")});
             EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Conv2DV2"));
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        p.run(*this);
    }
}

#endif