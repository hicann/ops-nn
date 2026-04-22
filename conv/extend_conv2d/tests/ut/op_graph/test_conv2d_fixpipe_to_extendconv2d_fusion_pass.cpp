/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "../../../op_graph/fusion_pass/conv2d_fixpipe_to_extendconv2d_fusion_pass.h"

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
class Conv2DFixPipeToExtendConv2DFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Conv2DFixPipeToExtendConv2DFusionPassTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Conv2DFixPipeToExtendConv2DFusionPassTest TearDown" << std::endl;
    }

    void TestTotalPass(std::string passName, GraphPtr &graph, Status expcetRes = SUCCESS) {
        if (CONV_DEBUG) {
            std::string dumpName = passName + "_before";
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString(dumpName.c_str()));
        }
        CustomPassContext passContex;
        passContex.SetPassName(passName.c_str());
        Conv2DFixPipeToExtendConv2DFusionPass pass;
        auto res = pass.Run(graph, passContex);
        if (CONV_DEBUG) {
            std::string dumpName = passName + "_after";
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString(dumpName.c_str()));
        }
        EXPECT_EQ(res, expcetRes);
        if (expcetRes == SUCCESS) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "ExtendConv2D"));
        }
    }

    void TestConvFixpipeFusion(TestGraph &testGraphBuilder, std::string passName, GraphPtr &graph,
        Status expcetRes = SUCCESS) {
        if (CONV_DEBUG) {
            std::string dumpName = passName + "_before";
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString(dumpName.c_str()));
        }
        CustomPassContext passContex;
        passContex.SetPassName(passName.c_str());
        Conv2DFixPipeToExtendConv2DFusionPass pass;
        GNode convNode = testGraphBuilder.GetNode("Conv2D");
        ConvFusionUtilsPass::GetConvDescInfo(convNode, pass.convDescInfo);
        auto boundary = pass.ConstructBoundary(convNode);
        EXPECT_EQ(boundary == nullptr, expcetRes != SUCCESS);
        if (expcetRes == SUCCESS) {
            auto replacement = pass.Replacement(convNode);
            auto res = ge::fusion::SubgraphRewriter::Replace(*boundary, *replacement);
            if (CONV_DEBUG) {
                std::string dumpName = passName + "_after";
                graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString(dumpName.c_str()));
            }
            EXPECT_EQ(res, SUCCESS);
            EXPECT_TRUE(GraphChecker::HasNode(graph, "ExtendConv2D"));
        }
    }

    void TestFixpipeTrans(TestGraph &testGraphBuilder, std::string passName, GraphPtr &graph) {
        if (CONV_DEBUG) {
            std::string dumpName = passName + "_before";
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString(dumpName.c_str()));
        }
        CustomPassContext passContex;
        passContex.SetPassName(passName.c_str());
        Conv2DFixPipeToExtendConv2DFusionPass pass;
        GNode convNode = testGraphBuilder.GetNode("Conv2D");
        ConvFusionUtilsPass::GetConvDescInfo(convNode, pass.convDescInfo);
        auto res = pass.FixpipeFusionImpl(graph, convNode, passContex);
        if (CONV_DEBUG) {
            std::string dumpName = passName + "_after";
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString(dumpName.c_str()));
        }
        EXPECT_EQ(res, true);
        EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));
    }
};

// Conv2D + Dequant -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_dequant_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_dequant_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv2d_dequant_fusion_success", graph, SUCCESS);
}

// Conv2D(NHWC/has bias) + Dequant -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_dequant_fusion_success_nhwc)
{
    TestGraph testGraphBuilder("conv2d_dequant_fusion_success_nhwc");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NHWC).WithBias())
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NHWC))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv2d_dequant_fusion_success_nhwc", graph, SUCCESS);
}

// Conv2D + Relu -> extendconv2d
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_relu_to_extendconv2d_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_relu_to_extendconv2d_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
        .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
        .Connect("Conv2D", 0, "Relu", 0)
        .SetOutput("Relu")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));

    TestTotalPass("conv2d_relu_to_extendconv2d_fusion_success", graph, SUCCESS);
}

// Conv2D + LeakyRelu -> ExtendConv2D(failed)
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_leakyrelu_to_conv2d_fixpipe_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_leakyrelu_to_conv2d_fixpipe_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
        .AddLeakyRelu(LeakyReluConfig::Basic("LeakyRelu", DT_FLOAT16))
        .Connect("Conv2D", 0, "LeakyRelu", 0)
        .SetOutput("LeakyRelu")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "LeakyRelu"));

    TestTotalPass("conv2d_leakyrelu_to_conv2d_fixpipe_fusion_success", graph, SUCCESS);
}

// Conv2D + Dequant + LeakyRelu -> ExtendConv2D(failed) + LeakyRelu
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_dequant_leakyrelu_to_extendconv2d_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_dequant_leakyrelu_to_extendconv2d_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .AddLeakyRelu(LeakyReluConfig::Basic("LeakyRelu", DT_FLOAT16))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .Connect("AscendDequant", 0, "LeakyRelu", 0)
        .SetOutput("LeakyRelu")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "LeakyRelu"));

    TestTotalPass("conv2d_dequant_leakyrelu_to_extendconv2d_fusion_success", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "LeakyRelu"));
}

// Conv2D + Requant -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_requant_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_requant_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
        .Connect("Conv2D", 0, "AscendRequant", 0)
        .SetOutput("AscendRequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));

    TestTotalPass("conv2d_requant_fusion_success", graph, SUCCESS);
}

// Conv2D(NHWC/has bias) + Requant -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_bias_requant_fusion_success_nhwc)
{
    TestGraph testGraphBuilder("conv2d_bias_requant_fusion_success_nhwc");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NHWC).WithBias())
        .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant", FORMAT_NHWC))
        .Connect("Conv2D", 0, "AscendRequant", 0)
        .SetOutput("AscendRequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));

    TestTotalPass("conv2d_bias_requant_fusion_success_nhwc", graph, SUCCESS);
}

// Conv2D + Dequant(1) -- Requant(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_dequant_requant_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_dequant_requant_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .Connect("Conv2D", 0, "AscendRequant", 0)
        .SetOutput("AscendDequant")
        .SetOutput("AscendRequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));

    TestTotalPass("conv2d_dequant_requant_fusion_success", graph, SUCCESS);
}

// Conv2D + Requant(1) + Requant(2) -> Conv2D + FixPipe
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_requant_out1_requant_out2_to_extendconv2d_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_requant_out1_requant_out2_to_extendconv2d_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant1"))
        .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant2"))
        .Connect("Conv2D", 0, "AscendRequant1", 0)
        .Connect("Conv2D", 0, "AscendRequant2", 0)
        .SetOutput("AscendRequant1")
        .SetOutput("AscendRequant2")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::CountNodes(graph, "AscendRequant") == 2);

    TestTotalPass("conv2d_requant_out1_requant_out2_to_extendconv2d_fusion_success", graph, SUCCESS);
}

// Conv2D(NHWC) + Dequant(1) -- Requant(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_dequant_requant_fusion_success_nhwc)
{
    TestGraph testGraphBuilder("conv2d_dequant_requant_fusion_success_nhwc");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NHWC))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NHWC))
        .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant", FORMAT_NHWC))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .Connect("Conv2D", 0, "AscendRequant", 0)
        .SetOutput("AscendDequant")
        .SetOutput("AscendRequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));

    TestTotalPass("conv2d_dequant_requant_fusion_success_nhwc", graph, SUCCESS);
}

// Conv2D + Relu(1) -- Quant + Relu(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_relu_out1_quant_relu_out2_to_extendconv2d_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_relu_out1_quant_relu_out2_to_extendconv2d_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
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

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendQuant"));

    TestTotalPass("conv2d_relu_out1_quant_relu_out2_to_extendconv2d_fusion_success", graph, SUCCESS);
}

// Conv2D + Dequant(1) -- Requant(2) -- Relu(3) -> Conv2D + FixPipe x 3, not happen in ATC
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_multi_3_output_to_fixpipe_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_multi_3_output_to_fixpipe_fusion_success");

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

    TestFixpipeTrans(testGraphBuilder, "conv2d_multi_3_output_to_fixpipe_fusion_success", graph);
}

// Conv2D + Dequant + Relu -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_dequant_relu_to_extendconv2d_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_dequant_relu_to_extendconv2d_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .Connect("AscendDequant", 0, "Relu", 0)
        .SetOutput("Relu")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));

    TestTotalPass("conv2d_dequant_relu_to_extendconv2d_fusion_success", graph, SUCCESS);
}

// Conv2D + Requant + Relu -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_requant_relu_to_extendconv2d_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_requant_relu_to_extendconv2d_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
        .AddRelu(ReluConfig::Basic("Relu", DT_INT8))
        .Connect("Conv2D", 0, "AscendRequant", 0)
        .Connect("AscendRequant", 0, "Relu", 0)
        .SetOutput("Relu")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));

    TestTotalPass("conv2d_requant_relu_to_extendconv2d_fusion_success", graph, SUCCESS);
}

// Conv2D + Quant + Relu -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_quant_relu_to_extendconv2d_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_quant_relu_to_extendconv2d_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
        .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant"))
        .AddRelu(ReluConfig::Basic("Relu", DT_INT8))
        .Connect("Conv2D", 0, "AscendQuant", 0)
        .Connect("AscendQuant", 0, "Relu", 0)
        .SetOutput("Relu")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendQuant"));

    TestTotalPass("conv2d_quant_relu_to_extendconv2d_fusion_success", graph, SUCCESS);
}

// Conv2D(1) + Quant + Relu(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_out1_quant_relu_out2_to_extendconv2d_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_out1_quant_relu_out2_to_extendconv2d_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
        .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
        .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant"))
        .Connect("Conv2D", 0, "AscendQuant", 0)
        .Connect("AscendQuant", 0, "Relu", 0)
        .SetOutput("Conv2D")
        .SetOutput("Relu")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendQuant"));

    TestTotalPass("conv2d_out1_quant_relu_out2_to_extendconv2d_fusion_success", graph, SUCCESS);
}

// Conv2D + Dequant(1) -- Requant + Relu(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_dequant_out1_requant_relu_out2_to_extendconv2d_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_dequant_out1_requant_relu_out2_to_extendconv2d_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
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

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));

    TestTotalPass("conv2d_dequant_out1_requant_relu_out2_to_extendconv2d_fusion_success", graph, SUCCESS);
}

// Conv2D + Dequant + Relu(1) -- Requant + Relu(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_dequant_relu_out1_requant_relu_out2_to_extendconv2d_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_dequant_relu_out1_requant_relu_out2_to_extendconv2d_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
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

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));

    TestTotalPass("conv2d_dequant_relu_out1_requant_relu_out2_to_extendconv2d_fusion_success", graph, SUCCESS);
}

// Conv2D + Quant -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_quant_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_quant_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
        .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16))
        .Connect("Conv2D", 0, "AscendQuant", 0)
        .SetOutput("AscendQuant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendQuant"));

    TestTotalPass("conv2d_quant_fusion_success", graph, SUCCESS);
}

// Conv2D(1) + Relu(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_out1_relu_out2_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_out1_relu_out2_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
        .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
        .Connect("Conv2D", 0, "Relu", 0)
        .SetOutput("Conv2D")
        .SetOutput("Relu")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));

    TestTotalPass("conv2d_out1_relu_out2_fusion_success", graph, SUCCESS);
}

// Conv2D(1) + Quant(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_out1_quant_out2_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_out1_quant_out2_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
        .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant"))
        .Connect("Conv2D", 0, "AscendQuant", 0)
        .SetOutput("Conv2D")
        .SetOutput("AscendQuant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendQuant"));

    TestTotalPass("conv2d_out1_quant_out2_fusion_success", graph, SUCCESS);
}

// Conv2D(1) + Dequant(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_out1_dequant_out2_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_out1_dequant_out2_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("Conv2D")
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv2d_out1_dequant_out2_fusion_success", graph, SUCCESS);
}

// Conv2D + Dequant(1) + Relu(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_dequant_out1_relu_out2_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_dequant_out1_relu_out2_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .Connect("AscendDequant", 0, "Relu", 0)
        .SetOutput("AscendDequant")
        .SetOutput("Relu")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));

    TestTotalPass("conv2d_dequant_out1_relu_out2_fusion_success", graph, SUCCESS);
}

// Conv2D + Quant(1)(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_quant_quant_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_quant_quant_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
        .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant1"))
        .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant2"))
        .Connect("Conv2D", 0, "AscendQuant1", 0)
        .Connect("Conv2D", 0, "AscendQuant2", 0)
        .SetOutput("AscendQuant1")
        .SetOutput("AscendQuant2")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendQuant"));

    TestTotalPass("conv2d_quant_quant_fusion_success", graph, SUCCESS);
}

// Conv2D + Quant + Relu(1) -- Quant + Relu(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_relu_quant_quant_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_relu_quant_quant_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
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

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendQuant"));

    TestTotalPass("conv2d_relu_quant_quant_fusion_success", graph, SUCCESS);
}

// Conv2D + Requant(1)(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_requant_requant_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_requant_requant_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant1"))
        .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant2"))
        .Connect("Conv2D", 0, "AscendRequant1", 0)
        .Connect("Conv2D", 0, "AscendRequant2", 0)
        .SetOutput("AscendRequant1")
        .SetOutput("AscendRequant2")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));

    TestTotalPass("conv2d_requant_requant_fusion_success", graph, SUCCESS);
}

// Conv2D + Requant + Relu(1)(2) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_requant_relu_relu_fusion_success)
{
    TestGraph testGraphBuilder("conv2d_requant_relu_relu_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
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

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));

    TestTotalPass("conv2d_requant_relu_relu_fusion_success", graph, SUCCESS);
}

// Conv2D + FixPipe(1) -- Relu(2) -- Dequant(3) -> ExtendConv2D(1) + (Relu -- Dequant)
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_fipipe_multi_other_fusion_success_first_int8)
{
    TestGraph testGraphBuilder("conv2d_fipipe_multi_other_fusion_success_first");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddRelu(ReluConfig::Basic("Relu"))
        .AddFixPipe(FixPipeConfig::Basic("FixPipe", DT_INT8).WithScale0())
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv2D", 0, "FixPipe", 0)
        .Connect("Conv2D", 0, "Relu", 0)
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("FixPipe")
        .SetOutput("Relu")
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestConvFixpipeFusion(testGraphBuilder, "conv2d_fipipe_multi_other_fusion_success_first", graph);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
}

// Conv2D + Relu(1) -- FixPipe(2) -- Dequant(3) -> ExtendConv2D(2) + (Relu -- Dequant)
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_fipipe_multi_other_fusion_success_second_int8)
{
    TestGraph testGraphBuilder("conv2d_fipipe_multi_other_fusion_success_second");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddRelu(ReluConfig::Basic("Relu"))
        .AddFixPipe(FixPipeConfig::Basic("FixPipe", DT_INT8).WithScale0())
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv2D", 0, "Relu", 0)
        .Connect("Conv2D", 0, "FixPipe", 0)
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("Relu")
        .SetOutput("FixPipe")
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestConvFixpipeFusion(testGraphBuilder, "conv2d_fipipe_multi_other_fusion_success_second", graph);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
}

// Conv2D + FixPipe(fp16_1) -- Relu(2) -- Dequant(3) -> ExtendConv2D(1) + (Relu -- Dequant)
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_fipipe_multi_other_fusion_success_first_fp16)
{
    TestGraph testGraphBuilder("conv2d_fipipe_multi_other_fusion_success_first");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddRelu(ReluConfig::Basic("Relu"))
        .AddFixPipe(FixPipeConfig::Basic("FixPipe").WithScale0())
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv2D", 0, "FixPipe", 0)
        .Connect("Conv2D", 0, "Relu", 0)
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("FixPipe")
        .SetOutput("Relu")
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestConvFixpipeFusion(testGraphBuilder, "conv2d_fipipe_multi_other_fusion_success_first", graph);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
}

// Conv2D + Relu(1) -- FixPipe(fp16_2) -- Dequant(3) -> ExtendConv2D(2) + (Relu -- Dequant)
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_fipipe_multi_other_fusion_success_second_fp16)
{
    TestGraph testGraphBuilder("conv2d_fipipe_multi_other_fusion_success_second");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddRelu(ReluConfig::Basic("Relu"))
        .AddFixPipe(FixPipeConfig::Basic("FixPipe").WithScale0())
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv2D", 0, "Relu", 0)
        .Connect("Conv2D", 0, "FixPipe", 0)
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("Relu")
        .SetOutput("FixPipe")
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestConvFixpipeFusion(testGraphBuilder, "conv2d_fipipe_multi_other_fusion_success_second", graph);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
}

TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, print_graph_structure_test)
{
    TestGraph testGraphBuilder("print_graph_structure_test");

    auto graph = testGraphBuilder.SetSocMC62CM12A()
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
    CustomPassContext passContex;
    passContex.SetPassName("print_graph_structure_test_before");
    Conv2DFixPipeToExtendConv2DFusionPass pass;
    GNode convNode = testGraphBuilder.GetNode("Conv2D");
    ConvFusionUtilsPass::GetConvDescInfo(convNode, pass.convDescInfo);
    auto res = pass.FixpipeFusionImpl(graph, convNode, passContex);
    if (CONV_DEBUG) {
        graph->DumpToFile(Graph::DumpFormat::kOnnx, "print_graph_structure_test_after");
    }
    EXPECT_EQ(res, true);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));

    pass.outputCase = Conv2DFixpipeToExtendConv2DFusion::OutputCase::SINGLE;
    pass.fixpipeFusionOps.push_back({"Relu"});
    pass.PrintGraphStructure();

    pass.outputCase = Conv2DFixpipeToExtendConv2DFusion::OutputCase::OTHER_FIXPIPE;
    pass.PrintGraphStructure();

    pass.outputCase = Conv2DFixpipeToExtendConv2DFusion::OutputCase::FIXPIPE_OTHER;
    pass.PrintGraphStructure();

    pass.outputCase = Conv2DFixpipeToExtendConv2DFusion::OutputCase::DUAL_FIXPIPE,
    pass.fixpipeFusionOps.push_back({"Relu", "AscendQuant"});
    pass.PrintGraphStructure();
}

// Conv2D + Dequant (x: NHWC, filter: NCHW) - 不支持的格式
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_not_support_fusion_format_test1)
{
    TestGraph testGraphBuilder("conv2d_not_support_fusion_format_test1");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    testGraphBuilder.UpdateNodeInputDesc("Conv2D", 0, DT_INT8, FORMAT_NHWC);
    testGraphBuilder.UpdateNodeOutputDesc("Conv2D", 0, DT_INT32, FORMAT_NCHW);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv2d_not_support_fusion_format_test1", graph, CONV_NOT_CHANGED);
}

// Conv2D + Dequant (x: NCHW, filter: NHWC) - 不支持的格式
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_not_support_fusion_format_test2)
{
    TestGraph testGraphBuilder("conv2d_not_support_fusion_format_test2");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    testGraphBuilder.UpdateNodeInputDesc("Conv2D", 1, DT_INT8, FORMAT_NHWC);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv2d_not_support_fusion_format_test2", graph, CONV_NOT_CHANGED);
}

// Conv2D + Dequant (conv2d输出: NHWC) - 不支持的格式
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_not_support_fusion_format_test3)
{
    TestGraph testGraphBuilder("conv2d_not_support_fusion_format_test3");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    testGraphBuilder.UpdateNodeOutputDesc("Conv2D", 0, DT_INT32, FORMAT_NHWC);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv2d_not_support_fusion_format_test3", graph, CONV_NOT_CHANGED);
}

// Conv2D + Dequant (filter: HWCN) - 不支持的格式
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, conv2d_not_support_fusion_format_test5)
{
    TestGraph testGraphBuilder("conv2d_not_support_fusion_format_test5");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv2D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    testGraphBuilder.UpdateNodeInputDesc("Conv2D", 1, DT_INT8, FORMAT_HWCN);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv2d_not_support_fusion_format_test5", graph, CONV_NOT_CHANGED);
}

// Conv2D + FixPipe(包含Relu + AscendDequant) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, fusion_failed_relu_fixpipe)
{
    TestGraph testGraphBuilder("fusion_failed_relu_fixpipe");

    FixPipeConfig fixpipeConfig = FixPipeConfig::Basic("FixPipe", DT_INT8).WithScale0();
    fixpipeConfig.SetAttr("fusion_op_list", std::vector<std::string>{"AscendDequant", "Relu"});

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddFixPipe(fixpipeConfig)
        .Connect("Conv2D", 0, "FixPipe", 0)
        .SetOutput("FixPipe")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));

    TestConvFixpipeFusion(testGraphBuilder, "fusion_failed_relu_fixpipe", graph);
}

// Conv2D + FixPipe(scale fp32) -> ExtendConv2D
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, fusion_failed_fixpipe_scale_fp32)
{
    TestGraph testGraphBuilder("fusion_failed_fixpipe_scale_fp32");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddFixPipe(FixPipeConfig::Basic("FixPipe").WithScale0(DT_FLOAT))
        .Connect("Conv2D", 0, "FixPipe", 0)
        .SetOutput("FixPipe")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));

    TestConvFixpipeFusion(testGraphBuilder, "fusion_failed_fixpipe_scale_fp32", graph, FAILED);
}

// Conv2D + FixPipe(输入FP32) - 不支持的输入类型
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, fusion_failed_fixpipe_fp32_in)
{
    TestGraph testGraphBuilder("fusion_failed_fixpipe_fp32_in");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddFixPipe(FixPipeConfig::Basic("FixPipe").WithScale0())
        .Connect("Conv2D", 0, "FixPipe", 0)
        .SetOutput("FixPipe")
        .Build();

    auto fixpipeNode = testGraphBuilder.GetNode("FixPipe");
    testGraphBuilder.UpdateNodeInputDesc("FixPipe", 0, DT_FLOAT, FORMAT_NCHW);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));

    TestConvFixpipeFusion(testGraphBuilder, "fusion_failed_fixpipe_fp32_in", graph, FAILED);
}

// Conv2D + FixPipe(输出FP32) - 不支持的输出类型
TEST_F(Conv2DFixPipeToExtendConv2DFusionPassTest, fusion_failed_fixpipe_fp32_out)
{
    TestGraph testGraphBuilder("fusion_failed_fixpipe_fp32_out");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv2D(Conv2DConfig::Basic("Conv2D"))
        .AddFixPipe(FixPipeConfig::Basic("FixPipe").WithScale0())
        .Connect("Conv2D", 0, "FixPipe", 0)
        .SetOutput("FixPipe")
        .Build();

    auto fixpipeNode = testGraphBuilder.GetNode("FixPipe");
    testGraphBuilder.UpdateNodeOutputDesc("FixPipe", 0, DT_FLOAT, FORMAT_NCHW);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "FixPipe"));

    TestConvFixpipeFusion(testGraphBuilder, "fusion_failed_fixpipe_fp32_out", graph, FAILED);
}