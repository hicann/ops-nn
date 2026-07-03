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
#include "../../../op_graph/fusion_pass/conv3d_dequant_to_quantconv3d_fusion_pass.h"

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

class Conv3DDequantToQuantConv3DFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Conv3DDequantToQuantConv3DFusionPassTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Conv3DDequantToQuantConv3DFusionPassTest TearDown" << std::endl;
    }

    void TestTotalPass(const std::string &passName, GraphPtr &graph, Status expectRes)
    {
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        Conv3DDequantToQuantConv3DFusionPass pass;

        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
        if (expectRes == SUCCESS) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantConv3D"));
        } else if (expectRes == CONV_NOT_CHANGED) {
            EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantConv3D"));
        }
    }

    GNode GetFirstQuantConv3DNode(GraphPtr &graph)
    {
        GNode fused;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "QuantConv3D", fused));
        return fused;
    }
};

// ==========================================================================================
// Fusion success: Conv3D + Dequant -> QuantConv3D (basic, NDHWC, bias)
// ==========================================================================================
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_dequant_to_quantconv3d_fusion_success)
{
    struct {
        const char *pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"basic", [this]() {
             TestGraph builder("conv3d_dequant_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv3D(Conv3DConfig::Basic("Conv3D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv3D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         }},
        {"ndhwc", [this]() {
             TestGraph builder("conv3d_dequant_fusion_success_nhwc");
             return builder.SetSocAscend950()
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NDHWC))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NDHWC))
                 .Connect("Conv3D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         }},
        {"bias", [this]() {
             TestGraph builder("conv3d_bias_dequant_fusion_success");
             return builder.SetSocAscend950()
                 .AddConv3D(Conv3DConfig::Basic("Conv3D").WithBias())
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv3D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
        EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));
        TestTotalPass(std::string("conv3d_dequant_to_quantconv3d_fusion_success_") + p.pointName, graph, SUCCESS);
    }
}

// ==========================================================================================
// No fusion: sqrt_mode, relu_flag, multi-output rejection
// ==========================================================================================
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_dequant_to_quantconv3d_no_fusion)
{
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"sqrt_mode_true", [this]() {
             TestGraph builder("conv3d_dequant_no_fusion_sqrt_mode_true");
             AscendDequantConfig dequantCfg = AscendDequantConfig::Basic("AscendDequant");
             dequantCfg.SetAttr("sqrt_mode", true);
             return builder.SetSocAscend950()
                 .AddConv3D(Conv3DConfig::Basic("Conv3D"))
                 .AddAscendDequant(dequantCfg)
                 .Connect("Conv3D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         }},
        {"relu_flag_true", [this]() {
             TestGraph builder("conv3d_dequant_no_fusion_relu_flag_true");
             AscendDequantConfig dequantCfg = AscendDequantConfig::Basic("AscendDequant");
             dequantCfg.SetAttr("relu_flag", true);
             return builder.SetSocAscend950()
                 .AddConv3D(Conv3DConfig::Basic("Conv3D"))
                 .AddAscendDequant(dequantCfg)
                 .Connect("Conv3D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         }},
        {"multi_output", [this]() {
             TestGraph builder("conv3d_dequant_requant_no_fusion_multi_output");
             return builder.SetSocAscend950()
                 .AddConv3D(Conv3DConfig::Basic("Conv3D"))
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                 .Connect("Conv3D", 0, "AscendDequant", 0)
                 .Connect("Conv3D", 0, "AscendRequant", 0)
                 .SetOutput("AscendDequant")
                 .SetOutput("AscendRequant")
                 .Build();
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
        TestTotalPass(std::string("conv3d_dequant_to_quantconv3d_no_fusion_") + p.pointName, graph, CONV_NOT_CHANGED);
    }
}

// ==========================================================================================
// Graph topology: complex chain with multiple Conv3D + Dequant
// ==========================================================================================
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_dequant_to_quantconv3d_graph_topology)
{
    struct {
        const char *pointName;
        std::function<void(Conv3DDequantToQuantConv3DFusionPassTest &)> run;
    } const points[] = {
        {"complex_chain", [](Conv3DDequantToQuantConv3DFusionPassTest &self) {
             TestGraph builder("conv3d_complexity_fusion_success");
             auto graph = builder.SetSocAscend950()
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
             self.TestTotalPass("conv3d_complexity_fusion_success", graph, SUCCESS);
             EXPECT_TRUE(GraphChecker::CountNodes(graph, "Relu") == 1);
             EXPECT_TRUE(GraphChecker::CountNodes(graph, "QuantConv3D") == 2);
             EXPECT_TRUE(GraphChecker::CountNodes(graph, "AscendQuant") == 1);
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        p.run(*this);
    }
}

// ==========================================================================================
// impl_mode reset: QuantConv3D resets _op_impl_mode_enum to 0x1
// ==========================================================================================
TEST_F(Conv3DDequantToQuantConv3DFusionPassTest, conv3d_dequant_to_quantconv3d_attr_and_desc)
{
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
        std::function<void(GNode &)> verify;
    } const points[] = {
        {"quantconv3d_fused_op_impl_mode_enum_is_default_0x1", [this]() {
             TestGraph builder("quantconv3d_fused_op_impl_mode_enum_is_default_0x1");
             Conv3DConfig convCfg = Conv3DConfig::Basic("Conv3D");
             convCfg.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return builder.SetSocAscend950()
                 .AddConv3D(convCfg)
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                 .Connect("Conv3D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         }, [](GNode &fused) {
             int64_t implMode = static_cast<int64_t>(-1);
             ASSERT_EQ(fused.GetAttr(AscendString("_op_impl_mode_enum"), implMode), GRAPH_SUCCESS);
             EXPECT_EQ(implMode, int64_t{0x1});
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("conv3d_dequant_to_quantconv3d_attr_and_desc_") + p.pointName, graph, SUCCESS);
        if (p.verify) {
            GNode fused = GetFirstQuantConv3DNode(graph);
            p.verify(fused);
        }
    }
}

#endif