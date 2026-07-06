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
#include "../../../op_graph/fusion_pass/conv2d_postcube_to_extendconv2d_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace ops;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

class Conv2DPostCubeToExtendConv2DFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Conv2DPostCubeToExtendConv2DFusionPassTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Conv2DPostCubeToExtendConv2DFusionPassTest TearDown" << std::endl; }

    void InitPassConvNode(Conv2DPostCubeToExtendConv2DFusionPass& pass, TestGraph& testGraphBuilder,
                          const std::string& convNodeName = "Conv2D")
    {
        GNode convNode = testGraphBuilder.GetNode(convNodeName);
        pass.InitMember();
        ConvFusionUtilsPass::GetConvDescInfo(convNode, pass.convDescInfo);
        ConvFusionUtilsPass::CheckSocList(Conv2DPostCubeToExtendConv2DFusion::SUPPORT_SOC_LIST, pass.npuArch);
    }

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        CustomPassContext passContext;
        Conv2DPostCubeToExtendConv2DFusionPass pass;
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
        if (expectRes == SUCCESS) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "ExtendConv2D"));
        } else if (expectRes == CONV_NOT_CHANGED) {
            EXPECT_FALSE(GraphChecker::HasNode(graph, "ExtendConv2D"));
        }
    }

    void TestConvFusionPreImpl(TestGraph& testGraphBuilder, const std::string& passName, GraphPtr& graph,
                               const std::string& convNodeName = "Conv2D")
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        CustomPassContext passContext;
        Conv2DPostCubeToExtendConv2DFusionPass pass;
        InitPassConvNode(pass, testGraphBuilder, convNodeName);
        GNode convNode = testGraphBuilder.GetNode(convNodeName);
        auto res = pass.ConvFusionPreImpl(graph, convNode, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, SUCCESS);
        EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));
    }

    void TestConvFusionReplaceImpl(TestGraph& testGraphBuilder, const std::string& passName, GraphPtr& graph,
                                   bool expectSuccess = true, const std::string& convNodeName = "Conv2D")
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        Conv2DPostCubeToExtendConv2DFusionPass pass;
        InitPassConvNode(pass, testGraphBuilder, convNodeName);
        GNode convNode = testGraphBuilder.GetNode(convNodeName);
        auto res = pass.ConvFusionReplaceImpl(graph, convNode);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectSuccess);
        if (expectSuccess) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "ExtendConv2D"));
        }
    }

    GNode GetFirstExtendConv2DNode(GraphPtr& graph)
    {
        GNode fused;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConv2D", fused));
        return fused;
    }
};

// ==========================================================================================
// Single-path fusion success: Conv2D + {Dequant, Relu, LeakyRelu, Requant, Quant} ... + {Relu}
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_to_extendconv2d_fusion_success)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        std::function<void(GNode&)> verify;
    } const points[] = {
        {"dequant_basic",
         [this]() {
             TestGraph builder("conv2d_dequant_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         nullptr},
        {"dequant_nhwc_bias",
         [this]() {
             TestGraph builder("conv2d_dequant_fusion_success_nhwc");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NHWC).WithBias())
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NHWC))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         nullptr},
        {"relu",
         [this]() {
             TestGraph builder("conv2d_relu_to_extendconv2d_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                 .Connect("Conv2D", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         },
         nullptr},
        {"leakyrelu",
         [this]() {
             TestGraph builder("conv2d_leakyrelu_to_conv2d_postcube_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddLeakyRelu(LeakyReluConfig::Basic("LeakyRelu", DT_FLOAT16))
                 .Connect("Conv2D", 0, "LeakyRelu", 0)
                 .SetOutput("LeakyRelu")
                 .Build();
         },
         nullptr},
        {"requant",
         [this]() {
             TestGraph builder("conv2d_requant_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                 .Connect("Conv2D", 0, "AscendRequant", 0)
                 .SetOutput("AscendRequant")
                 .Build();
         },
         nullptr},
        {"requant_nhwc_bias",
         [this]() {
             TestGraph builder("conv2d_bias_requant_fusion_success_nhwc");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NHWC).WithBias())
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant", FORMAT_NHWC))
                 .Connect("Conv2D", 0, "AscendRequant", 0)
                 .SetOutput("AscendRequant")
                 .Build();
         },
         nullptr},
        {"quant",
         [this]() {
             TestGraph builder("conv2d_quant_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16))
                 .Connect("Conv2D", 0, "AscendQuant", 0)
                 .SetOutput("AscendQuant")
                 .Build();
         },
         nullptr},
        {"dequant_relu",
         [this]() {
             TestGraph builder("conv2d_dequant_relu_to_extendconv2d_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .Connect("AscendDequant", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         },
         nullptr},
        {"requant_relu",
         [this]() {
             TestGraph builder("conv2d_requant_relu_to_extendconv2d_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                 .AddRelu(ReluConfig::Basic("Relu", DT_INT8))
                 .Connect("Conv2D", 0, "AscendRequant", 0)
                 .Connect("AscendRequant", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         },
         nullptr},
        {"quant_relu",
         [this]() {
             TestGraph builder("conv2d_quant_relu_to_extendconv2d_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant"))
                 .AddRelu(ReluConfig::Basic("Relu", DT_INT8))
                 .Connect("Conv2D", 0, "AscendQuant", 0)
                 .Connect("AscendQuant", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         },
         nullptr},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("conv2d_postcube_to_extendconv2d_fusion_success_") + p.pointName, graph, SUCCESS);
        if (p.verify) {
            GNode fused = GetFirstExtendConv2DNode(graph);
            p.verify(fused);
        }
    }
}

// ==========================================================================================
// Multi-output fusion success: Conv2D + multi output patterns
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_to_extendconv2d_multi_output_fusion_success)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"dequant_requant",
         [this]() {
             TestGraph builder("conv2d_dequant_requant_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .Connect("Conv2D", 0, "AscendRequant", 0)
                 .SetOutput("AscendDequant")
                 .SetOutput("AscendRequant")
                 .Build();
         }},
        {"dequant_requant_nhwc",
         [this]() {
             TestGraph builder("conv2d_dequant_requant_fusion_success_nhwc");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NHWC))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NHWC))
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant", FORMAT_NHWC))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .Connect("Conv2D", 0, "AscendRequant", 0)
                 .SetOutput("AscendDequant")
                 .SetOutput("AscendRequant")
                 .Build();
         }},
        {"requant_requant",
         [this]() {
             TestGraph builder("conv2d_requant_requant_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant1"))
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant2"))
                 .Connect("Conv2D", 0, "AscendRequant1", 0)
                 .Connect("Conv2D", 0, "AscendRequant2", 0)
                 .SetOutput("AscendRequant1")
                 .SetOutput("AscendRequant2")
                 .Build();
         }},
        {"relu_quant_relu",
         [this]() {
             TestGraph builder("conv2d_relu_out1_quant_relu_out2_to_extendconv2d_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddRelu(ReluConfig::Basic("Relu1", DT_FLOAT16))
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16))
                 .AddRelu(ReluConfig::Basic("Relu2", DT_FLOAT16))
                 .Connect("Conv2D", 0, "Relu1", 0)
                 .Connect("Conv2D", 0, "AscendQuant", 0)
                 .Connect("AscendQuant", 0, "Relu2", 0)
                 .SetOutput("Relu")
                 .SetOutput("Relu2")
                 .Build();
         }},
        {"out_quant_relu",
         [this]() {
             TestGraph builder("conv2d_out1_quant_relu_out2_to_extendconv2d_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant"))
                 .Connect("Conv2D", 0, "AscendQuant", 0)
                 .Connect("AscendQuant", 0, "Relu", 0)
                 .SetOutput("Conv2D")
                 .SetOutput("Relu")
                 .Build();
         }},
        {"dequant_out1_requant_relu",
         [this]() {
             TestGraph builder("conv2d_dequant_out1_requant_relu_out2_to_extendconv2d_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                 .AddRelu(ReluConfig::Basic("Relu", DT_INT8))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .Connect("Conv2D", 0, "AscendRequant", 0)
                 .Connect("AscendRequant", 0, "Relu", 0)
                 .SetOutput("AscendDequant")
                 .SetOutput("Relu")
                 .Build();
         }},
        {"dequant_relu_out1_requant_relu",
         [this]() {
             TestGraph builder("conv2d_dequant_relu_out1_requant_relu_out2_to_extendconv2d_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .AddRelu(ReluConfig::Basic("Relu1", DT_FLOAT16))
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                 .AddRelu(ReluConfig::Basic("Relu2", DT_INT8))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .Connect("Conv2D", 0, "AscendRequant", 0)
                 .Connect("AscendDequant", 0, "Relu1", 0)
                 .Connect("AscendRequant", 0, "Relu2", 0)
                 .SetOutput("Relu1")
                 .SetOutput("Relu2")
                 .Build();
         }},
        {"out_relu",
         [this]() {
             TestGraph builder("conv2d_out1_relu_out2_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                 .Connect("Conv2D", 0, "Relu", 0)
                 .SetOutput("Conv2D")
                 .SetOutput("Relu")
                 .Build();
         }},
        {"out_quant",
         [this]() {
             TestGraph builder("conv2d_out1_quant_out2_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant"))
                 .Connect("Conv2D", 0, "AscendQuant", 0)
                 .SetOutput("Conv2D")
                 .SetOutput("AscendQuant")
                 .Build();
         }},
        {"out_dequant",
         [this]() {
             TestGraph builder("conv2d_out1_dequant_out2_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("Conv2D")
                 .SetOutput("AscendDequant")
                 .Build();
         }},
        {"dequant_out1_relu",
         [this]() {
             TestGraph builder("conv2d_dequant_out1_relu_out2_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .Connect("AscendDequant", 0, "Relu", 0)
                 .SetOutput("AscendDequant")
                 .SetOutput("Relu")
                 .Build();
         }},
        {"quant_quant",
         [this]() {
             TestGraph builder("conv2d_quant_quant_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant1"))
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant2"))
                 .Connect("Conv2D", 0, "AscendQuant1", 0)
                 .Connect("Conv2D", 0, "AscendQuant2", 0)
                 .SetOutput("AscendQuant1")
                 .SetOutput("AscendQuant2")
                 .Build();
         }},
        {"relu_quant_quant",
         [this]() {
             TestGraph builder("conv2d_relu_quant_quant_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant1"))
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant2"))
                 .AddRelu(ReluConfig::Basic("Relu1", DT_FLOAT16))
                 .AddRelu(ReluConfig::Basic("Relu2", DT_FLOAT16))
                 .Connect("Conv2D", 0, "AscendQuant1", 0)
                 .Connect("Conv2D", 0, "AscendQuant2", 0)
                 .Connect("AscendQuant1", 0, "Relu1", 0)
                 .Connect("AscendQuant2", 0, "Relu2", 0)
                 .SetOutput("Relu1")
                 .SetOutput("Relu2")
                 .Build();
         }},
        {"requant_relu_relu",
         [this]() {
             TestGraph builder("conv2d_requant_relu_relu_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                 .AddRelu(ReluConfig::Basic("Relu1", DT_INT8))
                 .AddRelu(ReluConfig::Basic("Relu2", DT_INT8))
                 .Connect("Conv2D", 0, "AscendRequant", 0)
                 .Connect("AscendRequant", 0, "Relu1", 0)
                 .Connect("AscendRequant", 0, "Relu2", 0)
                 .SetOutput("Relu1")
                 .SetOutput("Relu2")
                 .Build();
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("conv2d_postcube_to_extendconv2d_multi_output_") + p.pointName, graph, SUCCESS);
    }
}

// ==========================================================================================
// No fusion: unsupported patterns / format mismatches / failed fusions
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_to_extendconv2d_no_fusion)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        Status expectRes;
    } const points[] = {
        {"dequant_leakyrelu",
         [this]() {
             TestGraph builder("conv2d_dequant_leakyrelu_to_extendconv2d_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .AddLeakyRelu(LeakyReluConfig::Basic("LeakyRelu", DT_FLOAT16))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .Connect("AscendDequant", 0, "LeakyRelu", 0)
                 .SetOutput("LeakyRelu")
                 .Build();
         },
         CONV_NOT_CHANGED},
        {"format_x_nhwc_filter_nchw",
         [this]() {
             TestGraph builder("conv2d_not_support_fusion_format_test1");
             auto graph = builder.SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                              .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                              .Connect("Conv2D", 0, "AscendDequant", 0)
                              .SetOutput("AscendDequant")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 0, DT_INT8, FORMAT_NHWC);
             builder.UpdateNodeOutputDesc("Conv2D", 0, DT_INT32, FORMAT_NCHW);
             return graph;
         },
         CONV_NOT_CHANGED},
        {"format_x_nchw_filter_nhwc",
         [this]() {
             TestGraph builder("conv2d_not_support_fusion_format_test2");
             auto graph = builder.SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                              .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                              .Connect("Conv2D", 0, "AscendDequant", 0)
                              .SetOutput("AscendDequant")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_INT8, FORMAT_NHWC);
             return graph;
         },
         CONV_NOT_CHANGED},
        {"output_nhwc",
         [this]() {
             TestGraph builder("conv2d_not_support_fusion_format_test3");
             auto graph = builder.SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                              .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                              .Connect("Conv2D", 0, "AscendDequant", 0)
                              .SetOutput("AscendDequant")
                              .Build();
             builder.UpdateNodeOutputDesc("Conv2D", 0, DT_INT32, FORMAT_NHWC);
             return graph;
         },
         CONV_NOT_CHANGED},
        {"filter_hwcn",
         [this]() {
             TestGraph builder("conv2d_not_support_fusion_format_test5");
             auto graph = builder.SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                              .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                              .Connect("Conv2D", 0, "AscendDequant", 0)
                              .SetOutput("AscendDequant")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_INT8, FORMAT_HWCN);
             return graph;
         },
         CONV_NOT_CHANGED},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("conv2d_postcube_to_extendconv2d_no_fusion_") + p.pointName, graph, p.expectRes);
    }
}

// ==========================================================================================
// No fusion via ReplaceImpl: failed postcube conditions (relu order, fp32 scale/in/out)
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_failed_replaceimpl)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr(TestGraph&)> build;
        bool expectSuccess;
    } const points[] = {
        {"relu_postcube",
         [](TestGraph& builder) {
             PostCubeConfig postCubeConfig = PostCubeConfig::Basic("FixPipe", DT_INT8).WithScale0();
             postCubeConfig.SetAttr("fusion_op_list", std::vector<std::string>{"AscendDequant", "Relu"});
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddPostCube(postCubeConfig)
                 .Connect("Conv2D", 0, "FixPipe", 0)
                 .SetOutput("FixPipe")
                 .Build();
         },
         true},
        {"postcube_scale_fp32",
         [](TestGraph& builder) {
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddPostCube(PostCubeConfig::Basic("FixPipe").WithScale0(DT_FLOAT))
                 .Connect("Conv2D", 0, "FixPipe", 0)
                 .SetOutput("FixPipe")
                 .Build();
         },
         false},
        {"postcube_fp32_in",
         [](TestGraph& builder) {
             auto graph = builder.SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                              .AddPostCube(PostCubeConfig::Basic("FixPipe").WithScale0())
                              .Connect("Conv2D", 0, "FixPipe", 0)
                              .SetOutput("FixPipe")
                              .Build();
             builder.UpdateNodeInputDesc("FixPipe", 0, DT_FLOAT, FORMAT_NCHW);
             return graph;
         },
         false},
        {"postcube_fp32_out",
         [](TestGraph& builder) {
             auto graph = builder.SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                              .AddPostCube(PostCubeConfig::Basic("FixPipe").WithScale0())
                              .Connect("Conv2D", 0, "FixPipe", 0)
                              .SetOutput("FixPipe")
                              .Build();
             builder.UpdateNodeOutputDesc("FixPipe", 0, DT_FLOAT, FORMAT_NCHW);
             return graph;
         },
         false},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        TestGraph builder(p.pointName);
        auto graph = p.build(builder);
        TestConvFusionReplaceImpl(builder, std::string("conv2d_postcube_failed_replaceimpl_") + p.pointName, graph,
                                  p.expectSuccess);
    }
}

// ==========================================================================================
// PostCube multi-other fusion: Conv2D + PostCube + Relu + Dequant patterns
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_multi_other)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr(TestGraph&)> build;
    } const points[] = {
        {"first_int8",
         [](TestGraph& builder) {
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddRelu(ReluConfig::Basic("Relu"))
                 .AddPostCube(PostCubeConfig::Basic("FixPipe", DT_INT8).WithScale0())
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv2D", 0, "FixPipe", 0)
                 .Connect("Conv2D", 0, "Relu", 0)
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("FixPipe")
                 .SetOutput("Relu")
                 .SetOutput("AscendDequant")
                 .Build();
         }},
        {"second_int8",
         [](TestGraph& builder) {
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddRelu(ReluConfig::Basic("Relu"))
                 .AddPostCube(PostCubeConfig::Basic("FixPipe", DT_INT8).WithScale0())
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv2D", 0, "Relu", 0)
                 .Connect("Conv2D", 0, "FixPipe", 0)
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("Relu")
                 .SetOutput("FixPipe")
                 .SetOutput("AscendDequant")
                 .Build();
         }},
        {"first_fp16",
         [](TestGraph& builder) {
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddRelu(ReluConfig::Basic("Relu"))
                 .AddPostCube(PostCubeConfig::Basic("FixPipe").WithScale0())
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv2D", 0, "FixPipe", 0)
                 .Connect("Conv2D", 0, "Relu", 0)
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("FixPipe")
                 .SetOutput("Relu")
                 .SetOutput("AscendDequant")
                 .Build();
         }},
        {"second_fp16",
         [](TestGraph& builder) {
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddRelu(ReluConfig::Basic("Relu"))
                 .AddPostCube(PostCubeConfig::Basic("FixPipe").WithScale0())
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv2D", 0, "Relu", 0)
                 .Connect("Conv2D", 0, "FixPipe", 0)
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("Relu")
                 .SetOutput("FixPipe")
                 .SetOutput("AscendDequant")
                 .Build();
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        TestGraph builder(p.pointName);
        auto graph = p.build(builder);
        TestConvFusionReplaceImpl(builder, std::string("conv2d_postcube_multi_other_") + p.pointName, graph);
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
        EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    }
}

// ==========================================================================================
// impl_mode reset: ExtendConv2D resets _op_impl_mode_enum to 0x1
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_to_extendconv2d_attr_and_desc)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        std::function<void(GNode&)> verify;
    } const points[] = {
        {"extendconv2d_fused_op_impl_mode_enum_is_default_0x1",
         [this]() {
             TestGraph builder("extendconv2d_fused_op_impl_mode_enum_is_default_0x1");
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D");
             convCfg.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return builder.SetSocAscend950()
                 .AddConv2D(convCfg)
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         [](GNode& fused) {
             int64_t implMode = static_cast<int64_t>(-1);
             ASSERT_EQ(fused.GetAttr(AscendString("_op_impl_mode_enum"), implMode), GRAPH_SUCCESS);
             EXPECT_EQ(implMode, int64_t{0x1});
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("conv2d_postcube_to_extendconv2d_attr_and_desc_") + p.pointName, graph, SUCCESS);
        if (p.verify) {
            GNode fused = GetFirstExtendConv2DNode(graph);
            p.verify(fused);
        }
    }
}

// ==========================================================================================
// Multi-3-output: Conv2D + Dequant + Requant + Relu -> PreImpl (FixPipe insertion only)
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_multi_3_output_to_postcube_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_multi_3_output_to_postcube_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                     .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                     .AddRelu(ReluConfig::Basic("Relu"))
                     .Connect("Conv2D", 0, "AscendDequant", 0)
                     .Connect("Conv2D", 0, "AscendRequant", 0)
                     .Connect("Conv2D", 0, "Relu", 0)
                     .SetOutput("AscendDequant")
                     .SetOutput("AscendRequant")
                     .SetOutput("Relu")
                     .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));

    TestConvFusionPreImpl(testGraphBuilder, "conv2d_multi_3_output_to_postcube_fusion_success", graph);
}

// ==========================================================================================
// Print graph structure debug test
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, print_graph_structure_test)
{
    TestGraph testGraphBuilder("print_graph_structure_test");

    auto graph = testGraphBuilder.SetSocMC62()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                     .AddRelu(ReluConfig::Basic("Relu"))
                     .Connect("Conv2D", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));

    if (CONV_DEBUG) {
        graph->DumpToFile(Graph::DumpFormat::kOnnx, "print_graph_structure_test_before");
    }
    CustomPassContext passContext;
    Conv2DPostCubeToExtendConv2DFusionPass pass;
    InitPassConvNode(pass, testGraphBuilder, "Conv2D");
    GNode convNode = testGraphBuilder.GetNode("Conv2D");
    auto res = pass.ConvFusionPreImpl(graph, convNode, passContext);
    if (CONV_DEBUG) {
        graph->DumpToFile(Graph::DumpFormat::kOnnx, "print_graph_structure_test_after");
    }
    EXPECT_EQ(res, SUCCESS);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));

    pass.outputCase = Conv2DPostCubeToExtendConv2DFusion::OutputCase::SINGLE;
    pass.postCubeFusionOps.push_back({"Relu"});
    pass.PrintGraphStructure();

    pass.outputCase = Conv2DPostCubeToExtendConv2DFusion::OutputCase::OTHER_POST_CUBE;
    pass.PrintGraphStructure();

    pass.outputCase = Conv2DPostCubeToExtendConv2DFusion::OutputCase::POST_CUBE_OTHER;
    pass.PrintGraphStructure();

    pass.outputCase = Conv2DPostCubeToExtendConv2DFusion::OutputCase::DUAL_POST_CUBE;
    pass.postCubeFusionOps.push_back({"Relu", "AscendQuant"});
    pass.PrintGraphStructure();
}

#endif