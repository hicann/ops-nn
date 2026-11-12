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
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void InitPassConvNode(Conv2DPostCubeToExtendConv2DFusionPass& pass, TestGraph& testGraphBuilder,
                          const std::string& convNodeName = "Conv2D")
    {
        GNode convNode = testGraphBuilder.GetNode(convNodeName);
        pass.InitMember();
        ConvFusionUtilsPass::GetConvDescInfo(convNode, pass.convDescInfo);
        ConvFusionUtilsPass::CheckSocList(Conv2DPostCubeToExtendConv2DFusion::SUPPORT_SOC_LIST, pass.npuArch, true);
    }

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
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
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        auto res = pass.ConvFusionReplaceImpl(graph, convNode, passContext);
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

// Single-path fusion success: Conv2D + {Dequant, Relu, LeakyRelu, Requant, Quant} ... + {Relu}
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
             auto graph = builder.SetSocFuse()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW))
                              .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                              .Connect("Conv2D", 0, "Relu", 0)
                              .SetOutput("Relu")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_FLOAT16, FORMAT_FRACTAL_Z);
             return graph;
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
        {"dequant_leakyrelu",
         [this]() {
             TestGraph builder("conv2d_dequant_leakyrelu_fusion_success");
             LeakyReluConfig leakyCfg = LeakyReluConfig::Basic("LeakyRelu", DT_FLOAT16);
             leakyCfg.SetAttr("negative_slope", 0.1f);
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .AddLeakyRelu(leakyCfg)
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .Connect("AscendDequant", 0, "LeakyRelu", 0)
                 .SetOutput("LeakyRelu")
                 .Build();
         },
         [](GNode& fused) {
             auto producer = fused
                                 .GetInDataNodesAndPortIndexs(
                                     Conv2DPostCubeToExtendConv2DFusion::EXTENDCONV2D_RELU_WEIGHT_0_INDEX)
                                 .first;
             ASSERT_NE(producer, nullptr);
             AscendString producerType;
             ASSERT_EQ(producer->GetType(producerType), GRAPH_SUCCESS);
             EXPECT_EQ(std::string(producerType.GetString()), "VectorMulScalar");
             float slope = 0.0f;
             ASSERT_EQ(producer->GetAttr(ge::AscendString("scalarattr"), slope), GRAPH_SUCCESS);
             EXPECT_FLOAT_EQ(slope, 0.1f);
         }},
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

// Multi-output fusion success: Conv2D + multi output patterns
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
             auto graph = builder.SetSocFuse()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW))
                              .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                              .Connect("Conv2D", 0, "Relu", 0)
                              .SetOutput("Conv2D")
                              .SetOutput("Relu")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_FLOAT16, FORMAT_FRACTAL_Z);
             return graph;
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

// No fusion: unsupported patterns / format mismatches / failed fusions
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_to_extendconv2d_no_fusion)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        Status expectRes;
    } const points[] = {
        {"relu_cut_on_dn2nz_soc",
         [this]() {
             TestGraph builder("conv2d_no_fusion_relu_dn2nz_cut");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                 .Connect("Conv2D", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
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
         true},
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
         true},
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
         true},
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
// impl_mode: ExtendConv2D does not carry _op_impl_mode_enum (canndev aligned)
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_to_extendconv2d_attr_and_desc)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        std::function<void(GNode&)> verify;
    } const points[] = {
        {"extendconv2d_fused_op_impl_mode_enum_not_set",
         [this]() {
             TestGraph builder("extendconv2d_fused_op_impl_mode_enum_not_set");
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
             EXPECT_NE(fused.GetAttr(AscendString("_op_impl_mode_enum"), implMode), GRAPH_SUCCESS);
             bool enableHf32 = false;
             ASSERT_EQ(fused.GetAttr(AscendString("enable_hf32"), enableHf32), GRAPH_SUCCESS);
             EXPECT_TRUE(enableHf32);
         }},
        {"extendconv2d_fused_enable_hf32_default_off",
         [this]() {
             TestGraph builder("extendconv2d_fused_enable_hf32_default_off");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         [](GNode& fused) {
             bool enableHf32 = true;
             if (fused.GetAttr(AscendString("enable_hf32"), enableHf32) == GRAPH_SUCCESS) {
                 EXPECT_FALSE(enableHf32);
             }
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

    auto graph = testGraphBuilder.SetSocFuse()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                     .Connect("Conv2D", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();
    testGraphBuilder.UpdateNodeInputDesc("Conv2D", 1, DT_FLOAT16, FORMAT_FRACTAL_Z);

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

// ==========================================================================================
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_adjust_layout)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        std::function<void(GraphPtr&)> verify;
    } const points[] = {
        {"single_output_post_transpose",
         []() {
             TestGraph builder("adjust_layout_single");
             auto graph = builder.SetSocFuse()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW))
                              .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW))
                              .AddTranspose(TransposeNodeConfig::NchwToNhwc("PostTranspose"), false)
                              .Connect("Conv2D", 0, "AscendDequant", 0)
                              .Connect("AscendDequant", 0, "PostTranspose", 0)
                              .SetOutput("PostTranspose")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_INT8, FORMAT_FRACTAL_Z);
             return graph;
         },
         [](GraphPtr& graph) {
             EXPECT_TRUE(GraphChecker::HasNode(graph, "ExtendConv2D"));
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));
         }},
        {"fractal_z_with_subformat",
         []() {
             TestGraph builder("adjust_layout_fz_subformat");
             auto graph = builder.SetSocFuse()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW))
                              .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW))
                              .AddTranspose(TransposeNodeConfig::NchwToNhwc("PostTranspose"), false)
                              .Connect("Conv2D", 0, "AscendDequant", 0)
                              .Connect("AscendDequant", 0, "PostTranspose", 0)
                              .SetOutput("PostTranspose")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_INT8,
                                         static_cast<Format>(static_cast<int32_t>(FORMAT_FRACTAL_Z) | 0x0100));
             return graph;
         },
         [](GraphPtr& graph) {
             EXPECT_TRUE(GraphChecker::HasNode(graph, "ExtendConv2D"));
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));
         }},
        {"dual_output_skip_transpose",
         []() {
             TestGraph builder("adjust_layout_dual");
             auto graph = builder.SetSocAscend950()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW))
                              .AddAscendDequant(AscendDequantConfig::Basic("Dequant0", DT_FLOAT16, FORMAT_NCHW))
                              .AddAscendRequant(AscendRequantConfig::Basic("Requant0", FORMAT_NCHW))
                              .AddTranspose(TransposeNodeConfig::NchwToNhwc("PostTranspose"), false)
                              .Connect("Conv2D", 0, "Dequant0", 0)
                              .Connect("Conv2D", 0, "Requant0", 0)
                              .Connect("Requant0", 0, "PostTranspose", 0)
                              .SetOutput("Dequant0")
                              .SetOutput("PostTranspose")
                              .Build();
             return graph;
         },
         [](GraphPtr& graph) {
             EXPECT_TRUE(GraphChecker::HasNode(graph, "ExtendConv2D"));
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Transpose"));
         }},
        {"single_output_post_transpose_with_existing_extendconv2d",
         []() {
             TestGraph builder("adjust_layout_existing");
             auto graph = builder.SetSocFuse()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW))
                              .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW))
                              .AddTranspose(TransposeNodeConfig::NchwToNhwc("PostTranspose"), false)
                              .AddNode(ExtendConv2DConfig::Basic("ExistedExtendConv2D"), {0, 1})
                              .Connect("Conv2D", 0, "AscendDequant", 0)
                              .Connect("AscendDequant", 0, "PostTranspose", 0)
                              .SetOutput("PostTranspose")
                              .SetOutput("ExistedExtendConv2D")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_INT8, FORMAT_FRACTAL_Z);
             return graph;
         },
         [](GraphPtr& graph) {
             EXPECT_EQ(GraphChecker::CountNodes(graph, "ExtendConv2D"), 2);
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));
         }},
        {"skip_on_nonfuse_soc",
         []() {
             TestGraph builder("adjust_layout_nonfuse");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW))
                 .AddTranspose(TransposeNodeConfig::NchwToNhwc("PostTranspose"), false)
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .Connect("AscendDequant", 0, "PostTranspose", 0)
                 .SetOutput("PostTranspose")
                 .Build();
         },
         [](GraphPtr& graph) {
             EXPECT_TRUE(GraphChecker::HasNode(graph, "ExtendConv2D"));
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Transpose"));
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
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_a16w8_weight_rearrange)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        bool expectRearrange;
    } const points[] = {
        {"non_a16w8_int8_filter",
         []() {
             TestGraph builder("non_a16w8");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         false},
        {"a16w8_insert_with_desc",
         []() {
             TestGraph builder("a16w8_desc");
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_INT32, FORMAT_NCHW);
             convCfg.inputs[1].SetDtype(DT_INT8);
             convCfg.inputs[1].tensorDesc.SetDataType(DT_INT8);
             auto graph = builder.SetSocFuse()
                              .AddConv2D(convCfg)
                              .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW))
                              .Connect("Conv2D", 0, "AscendDequant", 0)
                              .SetOutput("AscendDequant")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_INT8, FORMAT_FRACTAL_Z);
             return graph;
         },
         true},
        {"a16w8_non_fz_with_reformat",
         []() {
             TestGraph builder("a16w8_reformat");
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_INT32, FORMAT_NCHW);
             convCfg.inputs[1].SetDtype(DT_INT8);
             convCfg.inputs[1].tensorDesc.SetDataType(DT_INT8);
             return builder.SetSocFuse()
                 .AddConv2D(convCfg)
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         true},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("a16w8_") + p.pointName, graph, SUCCESS);
        if (p.expectRearrange) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "RearrangeA16W8WeightFzOp"));
            GNode rearrange;
            EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "RearrangeA16W8WeightFzOp", rearrange));
            TensorDesc inDesc;
            ASSERT_EQ(rearrange.GetInputDesc(0, inDesc), GRAPH_SUCCESS);
            EXPECT_EQ(inDesc.GetFormat(), FORMAT_FRACTAL_Z);
            EXPECT_EQ(inDesc.GetDataType(), DT_INT8);
            TensorDesc outDesc;
            ASSERT_EQ(rearrange.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
            EXPECT_EQ(outDesc.GetFormat(), FORMAT_FRACTAL_Z);
            EXPECT_EQ(outDesc.GetDataType(), DT_INT8);
        } else {
            EXPECT_FALSE(GraphChecker::HasNode(graph, "RearrangeA16W8WeightFzOp"));
        }
        if (p.pointName == std::string("a16w8_non_fz_with_reformat")) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "ReFormat"));
        }
    }
}

// ==========================================================================================
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_fixed_shift_value)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
        int64_t expectedValue;
    } const points[] = {
        {"a16w16_fp16_weight",
         [this]() {
             TestGraph builder("fsv_a16w16");
             auto graph = builder.SetSocFuse()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW))
                              .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW))
                              .Connect("Conv2D", 0, "Relu", 0)
                              .SetOutput("Relu")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_FLOAT16, FORMAT_FRACTAL_Z);
             return graph;
         },
         42},
        {"explicit_value",
         [this]() {
             TestGraph builder("fsv_explicit");
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW);
             convCfg.SetAttr("fixed_shift_value", int64_t(99));
             auto graph = builder.SetSocFuse()
                              .AddConv2D(convCfg)
                              .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW))
                              .Connect("Conv2D", 0, "Relu", 0)
                              .SetOutput("Relu")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_FLOAT16, FORMAT_FRACTAL_Z);
             return graph;
         },
         99},
        {"a16w8_int8_weight",
         [this]() {
             TestGraph builder("fsv_a16w8");
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_INT32, FORMAT_NCHW);
             convCfg.inputs[1].SetDtype(DT_INT8);
             convCfg.inputs[1].tensorDesc.SetDataType(DT_INT8);
             auto graph = builder.SetSocFuse()
                              .AddConv2D(convCfg)
                              .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW))
                              .Connect("Conv2D", 0, "AscendDequant", 0)
                              .SetOutput("AscendDequant")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_INT8, FORMAT_FRACTAL_Z);
             return graph;
         },
         13},
        {"w8w8_no_default_shift",
         [this]() {
             TestGraph builder("fsv_w8w8");
             auto graph = builder.SetSocFuse()
                              .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW))
                              .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW))
                              .Connect("Conv2D", 0, "AscendDequant", 0)
                              .SetOutput("AscendDequant")
                              .Build();
             builder.UpdateNodeInputDesc("Conv2D", 1, DT_INT8, FORMAT_FRACTAL_Z);
             return graph;
         },
         0},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("fixed_shift_value_") + p.pointName, graph, SUCCESS);
        GNode fused;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConv2D", fused));
        int64_t value = 0;
        ASSERT_EQ(fused.GetAttr(AscendString("fixed_shift_value"), value), GRAPH_SUCCESS);
        EXPECT_EQ(value, p.expectedValue);
    }
}

#endif

TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_quant_leakyrelu_inputs)
{
    TestGraph builder("conv2d_quant_leakyrelu_inputs");
    PostCubeConfig postCubeConfig = PostCubeConfig::Basic("FixPipe", DT_FLOAT16);
    postCubeConfig.WithScale0().WithRelu0();
    postCubeConfig.SetAttr("fusion_op_list", std::vector<std::string>{"AscendDequant", "LeakyRelu"});
    Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_INT32, FORMAT_NCHW);
    convCfg.inputs[1].SetDtype(DT_INT8);
    convCfg.inputs[1].tensorDesc.SetDataType(DT_INT8);
    auto graph = builder.SetSocFuse()
                     .AddConv2D(convCfg)
                     .AddPostCube(postCubeConfig)
                     .Connect("Conv2D", 0, "FixPipe", 0)
                     .SetOutput("FixPipe")
                     .Build();

    TestConvFusionReplaceImpl(builder, "conv2d_postcube_quant_leakyrelu_inputs", graph, true);
    GNode fused = GetFirstExtendConv2DNode(graph);
    TensorDesc scale0Desc;
    EXPECT_EQ(fused.GetInputDesc(Conv2DPostCubeToExtendConv2DFusion::EXTENDCONV2D_QUANT_SCALE_0_INDEX, scale0Desc),
              GRAPH_SUCCESS);
    TensorDesc relu0Desc;
    EXPECT_EQ(fused.GetInputDesc(Conv2DPostCubeToExtendConv2DFusion::EXTENDCONV2D_RELU_WEIGHT_0_INDEX, relu0Desc),
              GRAPH_SUCCESS);
    EXPECT_EQ(relu0Desc.GetDataType(), DT_FLOAT);
}

TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_w8w8_fp16out_origin_preserved)
{
    TestGraph builder("conv2d_w8w8_fp16out_origin");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_FLOAT16, FORMAT_NHWC))
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NHWC))
                     .Connect("Conv2D", 0, "AscendDequant", 0)
                     .SetOutput("AscendDequant")
                     .Build();
    builder.UpdateNodeInputDesc("AscendDequant", 0, DT_FLOAT16, FORMAT_NHWC);

    TensorDesc transDataDesc;
    transDataDesc.SetDataType(DT_INT8);
    transDataDesc.SetFormat(FORMAT_NHWC);
    transDataDesc.SetShape(Shape({1, 16, 48, 48}));
    transDataDesc.SetOriginFormat(FORMAT_NCHW);
    transDataDesc.SetOriginShape(Shape({1, 16, 48, 48}));
    builder.UpdateNodeInputDescEx("Conv2D", 0, transDataDesc);

    TestTotalPass("conv2d_postcube_w8w8_fp16out_origin_preserved", graph, SUCCESS);
    GNode fused;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConv2D", fused));
    TensorDesc fusedFmapDesc;
    EXPECT_EQ(fused.GetInputDesc(0, fusedFmapDesc), GRAPH_SUCCESS);
    EXPECT_EQ(fusedFmapDesc.GetOriginFormat(), FORMAT_NCHW);
    EXPECT_EQ(fusedFmapDesc.GetFormat(), FORMAT_NHWC);
}

TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_w8w8_scale_bypass_setm1)
{
    TestGraph builder("conv2d_w8w8_setm1_direct");
    PostCubeConfig postCubeConfig = PostCubeConfig::Basic("FixPipe", DT_FLOAT16);
    postCubeConfig.WithScale0(DT_UINT64, {4});
    postCubeConfig.SetAttr("fusion_op_list", std::vector<std::string>{"AscendDequant"});

    NodeConfig setM1Config("SetM1_scale");
    setM1Config.opType = "SetM1Dequant";
    setM1Config.inputDefs = {{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}};
    setM1Config.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    setM1Config.AddInput(DT_UINT64, FORMAT_ND, {4}, "ori_scale");
    setM1Config.AddOutput(DT_UINT64, FORMAT_ND, {4}, "y");

    auto graph = builder.SetSocFuse()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D"))
                     .AddNode(setM1Config, {0})
                     .AddPostCube(postCubeConfig, false)
                     .Connect("Conv2D", 0, "FixPipe", 0)
                     .Connect("SetM1_scale", 0, "FixPipe", 2)
                     .SetOutput("FixPipe")
                     .Build();

    TestConvFusionReplaceImpl(builder, "conv2d_postcube_w8w8_scale_direct_setm1", graph, true);
    GNode fused = GetFirstExtendConv2DNode(graph);
    TensorDesc scale0Desc;
    EXPECT_EQ(fused.GetInputDesc(Conv2DPostCubeToExtendConv2DFusion::EXTENDCONV2D_QUANT_SCALE_0_INDEX, scale0Desc),
              GRAPH_SUCCESS);
    EXPECT_EQ(scale0Desc.GetDataType(), DT_UINT64);
    auto producer = fused
                        .GetInDataNodesAndPortIndexs(
                            Conv2DPostCubeToExtendConv2DFusion::EXTENDCONV2D_QUANT_SCALE_0_INDEX)
                        .first;
    ASSERT_NE(producer, nullptr);
    AscendString producerType;
    ASSERT_EQ(producer->GetType(producerType), GRAPH_SUCCESS);
    EXPECT_STREQ(std::string(producerType.GetString()).c_str(), "SetM1Dequant");
}

// ==========================================================================================
// Groups implicit derivation (infershape 1.0 parity): when attr groups is 1 and fmap C is a
// multiple of filter Cin, derive groups = fmap C / filter Cin; keep explicit groups and
// non-derivable cases unchanged.
// ==========================================================================================
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_groups_implicit_derive)
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
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, {1, 64, 56, 56},
                                                {4, 16, 3, 3}, {1, 4, 56, 56}))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, {1, 4, 56, 56}))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         4},
        {"nhwc_fmap_hwcn_filter_derive",
         []() {
             TestGraph builder("groups_nhwc_hwcn");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NHWC, {1, 56, 56, 64},
                                                {3, 3, 16, 4}, {1, 56, 56, 4}))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NHWC, {1, 56, 56, 4}))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         4},
        {"nhwc_fmap_nhwc_filter_derive",
         []() {
             TestGraph builder("groups_nhwc_nhwc");
             Conv2DConfig convCfg;
             convCfg.SetName("Conv2D")
                 .AddInput(DT_INT8, FORMAT_NHWC, {1, 56, 56, 64}, "Conv2D_x")
                 .AddInput(DT_INT8, FORMAT_NHWC, {4, 3, 2, 16}, "Conv2D_filter")
                 .AddOutput(DT_INT32, FORMAT_NHWC, {1, 56, 56, 4}, "y")
                 .SetAttr("data_format", std::string("NHWC"));
             return builder.SetSocAscend950()
                 .AddConv2D(convCfg)
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NHWC, {1, 56, 56, 4}))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         4},
        {"fmap_c_equals_filter_cin_keeps_default",
         []() {
             TestGraph builder("groups_c_equal");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, {1, 16, 56, 56},
                                                {4, 16, 3, 3}, {1, 4, 56, 56}))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, {1, 4, 56, 56}))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         1},
        {"fmap_c_less_filter_cin_keeps_default",
         []() {
             TestGraph builder("groups_c_less");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, {1, 8, 56, 56}, {4, 16, 3, 3},
                                                {1, 4, 56, 56}))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, {1, 4, 56, 56}))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         1},
        {"explicit_groups_valid_preserved",
         []() {
             TestGraph builder("groups_explicit_valid");
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, {1, 64, 56, 56},
                                                        {4, 16, 3, 3}, {1, 4, 56, 56});
             convCfg.SetAttr("groups", int64_t(4));
             return builder.SetSocAscend950()
                 .AddConv2D(convCfg)
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, {1, 4, 56, 56}))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         4},
        {"explicit_groups_mismatch_preserved",
         []() {
             TestGraph builder("groups_explicit_mismatch");
             Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, {1, 64, 56, 56},
                                                        {4, 16, 3, 3}, {1, 4, 56, 56});
             convCfg.SetAttr("groups", int64_t(2));
             return builder.SetSocAscend950()
                 .AddConv2D(convCfg)
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, {1, 4, 56, 56}))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         2},
        {"non_divisible_keeps_default",
         []() {
             TestGraph builder("groups_non_divisible");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, {1, 63, 56, 56},
                                                {4, 16, 3, 3}, {1, 4, 56, 56}))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, {1, 4, 56, 56}))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         1},
        {"dynamic_shape_keeps_default",
         []() {
             TestGraph builder("groups_dynamic");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, {-1, -1, -1, -1},
                                                {-1, -1, -1, -1}, {-1, -1, -1, -1}))
                 .AddAscendDequant(
                     AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, {-1, -1, -1, -1}))
                 .Connect("Conv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         },
         1},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("groups_derive_") + p.pointName, graph, SUCCESS);
        GNode fused = GetFirstExtendConv2DNode(graph);
        int64_t groups = 0;
        if (fused.GetAttr(AscendString("groups"), groups) == GRAPH_SUCCESS) {
            EXPECT_EQ(groups, p.expectedGroups);
        } else {
            // Es builder omits optional attrs equal to the IR default, absent groups means 1.
            EXPECT_EQ(p.expectedGroups, int64_t{1});
        }
    }
}

// A16W8 chain consistency: the rearrange node reads groups from the derived ExtendConv2D node,
// so the derived groups must propagate to RearrangeA16W8WeightFzOp.
TEST_F(Conv2DPostCubeToExtendConv2DFusionPassTest, conv2d_postcube_groups_derive_a16w8_rearrange)
{
    TestGraph builder("groups_derive_a16w8");
    Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_INT32, FORMAT_NCHW, {1, 64, 56, 56},
                                               {4, 16, 3, 3}, {1, 4, 56, 56});
    convCfg.inputs[1].SetDtype(DT_INT8);
    convCfg.inputs[1].tensorDesc.SetDataType(DT_INT8);
    auto graph = builder.SetSocFuse()
                     .AddConv2D(convCfg)
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, {1, 4, 56, 56}))
                     .Connect("Conv2D", 0, "AscendDequant", 0)
                     .SetOutput("AscendDequant")
                     .Build();

    TestTotalPass("groups_derive_a16w8_rearrange", graph, SUCCESS);
    GNode fused = GetFirstExtendConv2DNode(graph);
    int64_t groups = 0;
    ASSERT_EQ(fused.GetAttr(AscendString("groups"), groups), GRAPH_SUCCESS);
    EXPECT_EQ(groups, int64_t{4});

    GNode rearrange;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "RearrangeA16W8WeightFzOp", rearrange));
    int64_t rearrangeGroups = 0;
    ASSERT_EQ(rearrange.GetAttr(AscendString("groups"), rearrangeGroups), GRAPH_SUCCESS);
    EXPECT_EQ(rearrangeGroups, int64_t{4});
}
