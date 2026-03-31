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
#include "../../../op_graph/fusion_pass/conv3d_dequant_to_quantconv3d_fusion_pass.h"

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

class Conv3DDequantToQuantConv3DFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Conv3DDequantToQuantConv3DFusionPassTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Conv3DDequantToQuantConv3DFusionPassTest TearDown" << std::endl;
    }

    void TestTotalPass(std::string passName, GraphPtr &graph, Status expcetRes) {
        CustomPassContext passContex;
        passContex.SetPassName(passName.c_str());
        Conv3DDequantToQuantConv3DFusionPass pass;

        if (CONV_DEBUG) {
            std::string dumpName = passName + "_before";
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString(dumpName.c_str()));
        }
        auto res = pass.Run(graph, passContex);
        if (CONV_DEBUG) {
            std::string dumpName = passName + "_after";
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString(dumpName.c_str()));
        }
        EXPECT_EQ(res, expcetRes);
        if (expcetRes == SUCCESS) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantConv3D"));
        }
    }
};

// Conv3D + Dequant -> QuantConv3D
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_dequant_fusion_success)
{
    TestGraph testGraphBuilder("conv3d_dequant_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv3D(Conv3DConfig::Basic("Conv3D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv3D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv3d_dequant_fusion_success", graph, SUCCESS);
}

// Conv3D(NDHWC) + Dequant -> QuantConv3D
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_dequant_fusion_success_nhwc)
{
    TestGraph testGraphBuilder("conv3d_dequant_fusion_success_nhwc");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NDHWC))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NDHWC))
        .Connect("Conv3D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv3d_dequant_fusion_success_nhwc", graph, SUCCESS);
}

// Conv3D(has bias) + Dequant -> QuantConv3D
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_bias_dequant_fusion_success)
{
    TestGraph testGraphBuilder("conv3d_bias_dequant_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv3D(Conv3DConfig::Basic("Conv3D").WithBias())
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .Connect("Conv3D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv3d_bias_dequant_fusion_success", graph, SUCCESS);
}

// Conv3D + Dequant(sqrt_mode=true) -> no fusion
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_dequant_no_fusion_sqrt_mode_true)
{
    TestGraph testGraphBuilder("conv3d_dequant_no_fusion_sqrt_mode_true");

    AscendDequantConfig dequantConvfig = AscendDequantConfig::Basic("AscendDequant");
    dequantConvfig.SetAttr("sqrt_mode", true);

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv3D(Conv3DConfig::Basic("Conv3D"))
        .AddAscendDequant(dequantConvfig)
        .Connect("Conv3D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv3d_dequant_no_fusion_sqrt_mode_true", graph, CONV_NOT_CHANGED);
}

// Conv3D + Dequant(relu_flag=true) -> no fusion
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_dequant_no_fusion_relu_flag_true)
{
    TestGraph testGraphBuilder("conv3d_dequant_no_fusion_relu_flag_true");

    AscendDequantConfig dequantConvfig = AscendDequantConfig::Basic("AscendDequant");
    dequantConvfig.SetAttr("relu_flag", true);

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv3D(Conv3DConfig::Basic("Conv3D"))
        .AddAscendDequant(dequantConvfig)
        .Connect("Conv3D", 0, "AscendDequant", 0)
        .SetOutput("AscendDequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("conv3d_dequant_no_fusion_relu_flag_true", graph, CONV_NOT_CHANGED);
}

// Conv3D + Dequant(1) + Requant(2) -> no fusion
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_dequant_requant_no_fusion_multi_output)
{
    TestGraph testGraphBuilder("conv3d_dequant_requant_no_fusion_multi_output");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv3D(Conv3DConfig::Basic("Conv3D"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
        .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
        .Connect("Conv3D", 0, "AscendDequant", 0)
        .Connect("Conv3D", 0, "AscendRequant", 0)
        .SetOutput("AscendDequant")
        .SetOutput("AscendRequant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendRequant"));

    TestTotalPass("conv3d_dequant_requant_no_fusion_multi_output", graph, CONV_NOT_CHANGED);
}

// Conv3D + Dequant + Relu + Conv3D + Dequant + Quant -> QuantConv3D + Relu + QuantConv3D + Quant
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_complexity_fusion_success)
{
    TestGraph testGraphBuilder("conv3d_complexity_fusion_success");

    auto graph = testGraphBuilder.SetSocAscend950()
        .AddConv3D(Conv3DConfig::Basic("Conv3D1"))
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant1"))
        .AddRelu(ReluConfig::Basic("Relu"))
        .AddConv3D(Conv3DConfig::Basic("Conv3D2"), false)
        .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant2"))
        .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant"))
        .Connect("Conv3D1", 0, "AscendDequant1", 0)
        .Connect("AscendDequant1", 0, "Relu", 0)
        .Connect("Relu", 0, "Conv3D2", 0)
        .Connect("Conv3D2", 0, "AscendDequant2", 0)
        .Connect("AscendDequant2", 0, "AscendQuant", 0)
        .SetOutput("AscendQuant")
        .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendQuant"));

    TestTotalPass("conv3d_complexity_fusion_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::CountNodes(graph, "Relu") == 1);
    EXPECT_TRUE(GraphChecker::CountNodes(graph, "QuantConv3D") == 2);
    EXPECT_TRUE(GraphChecker::CountNodes(graph, "AscendQuant") == 1);
}