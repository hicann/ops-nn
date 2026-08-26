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
#include "../../../op_graph/fusion_pass/quant_unsqueeze_conv2d_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace Ops::NN::Conv;
using namespace ConvFusionUtils;
using namespace QuantUnsqueezeConv2dFusion;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

namespace {
constexpr int64_t DIM_3D = 3;
constexpr int64_t DIM_4D = 4;

const std::vector<int64_t> SHAPE_3D = {1, 2, 48};
const std::vector<int64_t> SHAPE_4D = {1, 2, 1, 48};
const std::vector<int64_t> FILTER_SHAPE = {4, 2, 1, 3};
const std::vector<int64_t> CONV_OUT_4D = {1, 4, 1, 48};
const std::vector<int64_t> SQUEEZE_OUT_3D = {1, 4, 48};

NodeConfig MakeUnsqueeze(const std::string& name, const std::vector<int64_t>& inShape,
                         const std::vector<int64_t>& outShape)
{
    NodeConfig cfg(name);
    cfg.opType = "Unsqueeze";
    cfg.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
    cfg.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    TensorInfo inInfo(DT_INT8, FORMAT_NCHW, inShape, name + "_x");
    inInfo.SetDesc(BuildTensorDesc(DT_INT8, FORMAT_NCHW, inShape));
    TensorInfo outInfo(DT_INT8, FORMAT_NCHW, outShape, "y");
    outInfo.SetDesc(BuildTensorDesc(DT_INT8, FORMAT_NCHW, outShape));
    cfg.AddInput(inInfo).AddOutput(outInfo).SetAttr("axes", std::vector<int64_t>{2});
    return cfg;
}

NodeConfig MakeSqueeze(const std::string& name, const std::vector<int64_t>& inShape,
                       const std::vector<int64_t>& outShape)
{
    NodeConfig cfg(name);
    cfg.opType = "Squeeze";
    cfg.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
    cfg.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    TensorInfo inInfo(DT_INT32, FORMAT_NCHW, inShape, name + "_x");
    inInfo.SetDesc(BuildTensorDesc(DT_INT32, FORMAT_NCHW, inShape));
    TensorInfo outInfo(DT_INT32, FORMAT_NCHW, outShape, "y");
    outInfo.SetDesc(BuildTensorDesc(DT_INT32, FORMAT_NCHW, outShape));
    cfg.AddInput(inInfo).AddOutput(outInfo).SetAttr("axis", std::vector<int64_t>{2});
    return cfg;
}

void SetupSoc(bool supportUb2Ub, bool supportDn2Nz = false)
{
    PlatformInfo platformInfo;
    OptionalInfo optiInfo;
    optiInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    if (supportUb2Ub) {
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_conv_ub_to_ub"] = {"s8s8"};
    }
    if (supportDn2Nz) {
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_dn2nz"] = {"s8"};
    }
    PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiInfo);
}
} // namespace

class QuantUnsqueezeConv2DFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        CustomPassContext passContext;
        passContext.SetPassName(FUSION_NAME.c_str());
        QuantUnsqueezeConv2DFusionPass pass;
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
    }

    // Data -> AscendQuant -> Unsqueeze -> Conv2D -> Squeeze -> AscendDequant -> [Broadcast]
    GraphPtr BuildChain(const std::string& tag, bool withBroadcast, int64_t unsqueezeDim = DIM_3D,
                        bool withQuant = true, bool withSqueeze = true, bool withDequant = true,
                        bool useReluBroadcast = false, bool dequantMultiConsumer = false)
    {
        TestGraph builder(tag);
        const auto& unsqueezeInShape = unsqueezeDim == DIM_3D ? SHAPE_3D : SHAPE_4D;

        if (withQuant) {
            builder.AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, SHAPE_3D), true);
        } else {
            builder.AddData(DataConfig::Basic("QuantInput", DT_FLOAT16, FORMAT_NCHW, SHAPE_3D), true);
        }

        builder.AddNode(MakeUnsqueeze("Unsqueeze", unsqueezeInShape, SHAPE_4D), {});
        if (withQuant) {
            builder.Connect("AscendQuant", 0, "Unsqueeze", 0);
        } else {
            builder.Connect("QuantInput", 0, "Unsqueeze", 0);
        }

        builder.AddConv2D(
            Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, SHAPE_4D, FILTER_SHAPE, CONV_OUT_4D), false,
            true, false);
        builder.Connect("Unsqueeze", 0, "Conv2D", 0);

        if (withSqueeze) {
            builder.AddNode(MakeSqueeze("Squeeze", CONV_OUT_4D, SQUEEZE_OUT_3D), {});
            builder.Connect("Conv2D", 0, "Squeeze", 0);
            if (withDequant) {
                builder.AddAscendDequant(
                    AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D), false, true);
                builder.Connect("Squeeze", 0, "AscendDequant", 0);
            } else {
                builder.AddRelu(ReluConfig::Basic("FakeDequant", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D), false);
                builder.Connect("Squeeze", 0, "FakeDequant", 0);
                builder.SetOutput("FakeDequant");
                return builder.Build();
            }
        } else if (withDequant) {
            builder.AddAscendDequant(
                AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D), false, true);
            builder.Connect("Conv2D", 0, "AscendDequant", 0);
        }

        if (withBroadcast) {
            if (useReluBroadcast) {
                builder.AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D), false);
                builder.Connect("AscendDequant", 0, "Relu", 0);
                builder.SetOutput("Relu");
            } else {
                builder.AddLeakyRelu(LeakyReluConfig::Basic("LeakyRelu", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D),
                                     false);
                builder.Connect("AscendDequant", 0, "LeakyRelu", 0);
                builder.SetOutput("LeakyRelu");
            }
        } else if (dequantMultiConsumer) {
            builder.AddRelu(ReluConfig::Basic("Consumer0", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D), false);
            builder.AddRelu(ReluConfig::Basic("Consumer1", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D), false);
            builder.Connect("AscendDequant", 0, "Consumer0", 0);
            builder.Connect("AscendDequant", 0, "Consumer1", 0);
            builder.SetOutput("Consumer0");
            builder.SetOutput("Consumer1");
        } else {
            builder.SetOutput("AscendDequant");
        }

        return builder.Build();
    }

    bool VerifyProducerType(GraphPtr& graph, const std::string& nodeType, int32_t inputIdx,
                            const std::string& expectProducerType)
    {
        GNode node;
        if (!GraphChecker::FindFirstNodeByOpType(graph, nodeType, node)) {
            return false;
        }
        std::string producerType;
        return GraphChecker::GetInputProducerType(node, inputIdx, producerType) && producerType == expectProducerType;
    }

    bool VerifyNodeDtype(GraphPtr& graph, const std::string& nodeType, int32_t outputIdx, DataType expectDtype)
    {
        GNode node;
        if (!GraphChecker::FindFirstNodeByOpType(graph, nodeType, node)) {
            return false;
        }
        TensorDesc desc;
        if (node.GetOutputDesc(outputIdx, desc) != GRAPH_SUCCESS) {
            return false;
        }
        return desc.GetDataType() == expectDtype;
    }

    bool VerifyNodeShape(GraphPtr& graph, const std::string& nodeType, int32_t index, bool isOutput,
                         const std::vector<int64_t>& expectDims)
    {
        GNode node;
        if (!GraphChecker::FindFirstNodeByOpType(graph, nodeType, node)) {
            return false;
        }
        TensorDesc desc;
        if (isOutput) {
            if (node.GetOutputDesc(index, desc) != GRAPH_SUCCESS) {
                return false;
            }
        } else if (node.GetInputDesc(index, desc) != GRAPH_SUCCESS) {
            return false;
        }
        return desc.GetShape().GetDims() == expectDims && desc.GetOriginShape().GetDims() == expectDims;
    }
};

// ==========================================================================================
// Success: basic chain and broadcast chain
// ==========================================================================================
TEST_F(QuantUnsqueezeConv2DFusionPassTest, fusion_success)
{
    struct Point {
        const char* pointName;
        bool withBroadcast;
    } const points[] = {
        {"no_broadcast", false},
        {"with_broadcast", true},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        SetupSoc(true);
        auto graph = BuildChain(std::string("fusion_success_") + p.pointName, p.withBroadcast);
        TestTotalPass(std::string("fusion_success_") + p.pointName, graph, SUCCESS);

        if (CONV_DEBUG) {
            GraphChecker::Print(graph);
        }

        EXPECT_TRUE(VerifyProducerType(graph, "AscendQuant", 0, "Unsqueeze"));
        EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 0, "AscendQuant"));
        EXPECT_TRUE(VerifyProducerType(graph, "AscendDequant", 0, "Conv2D"));
        EXPECT_TRUE(VerifyProducerType(graph, "Squeeze", 0, p.withBroadcast ? "LeakyRelu" : "AscendDequant"));
        EXPECT_TRUE(VerifyNodeDtype(graph, "Unsqueeze", 0, DT_FLOAT16));
        EXPECT_TRUE(VerifyNodeDtype(graph, "Squeeze", 0, DT_FLOAT16));
    }
}

// ==========================================================================================
// No fusion: unsupported capability, invalid dim, missing chain nodes
// ==========================================================================================
TEST_F(QuantUnsqueezeConv2DFusionPassTest, no_fusion)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"no_capability",
         [this]() {
             SetupSoc(false);
             return BuildChain("no_capability", false);
         }},
        {"invalid_dim",
         [this]() {
             SetupSoc(true);
             return BuildChain("invalid_dim", false, DIM_4D);
         }},
        {"missing_squeeze",
         [this]() {
             SetupSoc(true);
             return BuildChain("missing_squeeze", false, DIM_3D, true, false);
         }},
        {"missing_quant",
         [this]() {
             SetupSoc(true);
             return BuildChain("missing_quant", false, DIM_3D, false, true);
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("no_fusion_") + p.pointName, graph, CONV_NOT_CHANGED);
        EXPECT_FALSE(VerifyProducerType(graph, "AscendQuant", 0, "Unsqueeze"));
    }
}

// ==========================================================================================
// Reentrant: two independent chains, InitMember clears state between nodes
// ==========================================================================================
TEST_F(QuantUnsqueezeConv2DFusionPassTest, reentrant_two_chains)
{
    SetupSoc(true);
    TestGraph builder("reentrant_two_chains");

    // Chain 1: AscendQuant1 -> Unsqueeze1 -> Conv2D1 -> Squeeze1 -> AscendDequant1
    builder.AddAscendQuant(AscendQuantConfig::Basic("AscendQuant1", DT_FLOAT16, FORMAT_NCHW, SHAPE_3D), true);
    builder.AddNode(MakeUnsqueeze("Unsqueeze1", SHAPE_3D, SHAPE_4D), {});
    builder.Connect("AscendQuant1", 0, "Unsqueeze1", 0);
    builder.AddConv2D(
        Conv2DConfig::Basic("Conv2D1", DT_INT8, DT_INT32, FORMAT_NCHW, SHAPE_4D, FILTER_SHAPE, CONV_OUT_4D), false,
        true, false);
    builder.Connect("Unsqueeze1", 0, "Conv2D1", 0);
    builder.AddNode(MakeSqueeze("Squeeze1", CONV_OUT_4D, SQUEEZE_OUT_3D), {});
    builder.Connect("Conv2D1", 0, "Squeeze1", 0);
    builder.AddAscendDequant(AscendDequantConfig::Basic("AscendDequant1", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D),
                             false, true);
    builder.Connect("Squeeze1", 0, "AscendDequant1", 0);
    builder.SetOutput("AscendDequant1");

    // Chain 2: AscendQuant2 -> Unsqueeze2 -> Conv2D2 -> Squeeze2 -> AscendDequant2
    builder.AddAscendQuant(AscendQuantConfig::Basic("AscendQuant2", DT_FLOAT16, FORMAT_NCHW, SHAPE_3D), true);
    builder.AddNode(MakeUnsqueeze("Unsqueeze2", SHAPE_3D, SHAPE_4D), {});
    builder.Connect("AscendQuant2", 0, "Unsqueeze2", 0);
    builder.AddConv2D(
        Conv2DConfig::Basic("Conv2D2", DT_INT8, DT_INT32, FORMAT_NCHW, SHAPE_4D, FILTER_SHAPE, CONV_OUT_4D), false,
        true, false);
    builder.Connect("Unsqueeze2", 0, "Conv2D2", 0);
    builder.AddNode(MakeSqueeze("Squeeze2", CONV_OUT_4D, SQUEEZE_OUT_3D), {});
    builder.Connect("Conv2D2", 0, "Squeeze2", 0);
    builder.AddAscendDequant(AscendDequantConfig::Basic("AscendDequant2", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D),
                             false, true);
    builder.Connect("Squeeze2", 0, "AscendDequant2", 0);
    builder.SetOutput("AscendDequant2");

    auto graph = builder.Build();
    TestTotalPass("reentrant_two_chains", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 2);

    int swappedCount = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "AscendQuant") {
            std::string producerType;
            if (GraphChecker::GetInputProducerType(node, 0, producerType) && producerType == "Unsqueeze") {
                swappedCount++;
            }
        }
    }
    EXPECT_EQ(swappedCount, 2);
}

TEST_F(QuantUnsqueezeConv2DFusionPassTest, soc_dn2nz_only_success)
{
    SetupSoc(false, true);
    auto graph = BuildChain("soc_dn2nz_only_success", false);
    TestTotalPass("soc_dn2nz_only_success", graph, SUCCESS);
    EXPECT_TRUE(VerifyProducerType(graph, "AscendQuant", 0, "Unsqueeze"));
}

TEST_F(QuantUnsqueezeConv2DFusionPassTest, broadcast_relu_success)
{
    SetupSoc(true);
    auto graph = BuildChain("broadcast_relu_success", true, DIM_3D, true, true, true, true);
    TestTotalPass("broadcast_relu_success", graph, SUCCESS);
    EXPECT_TRUE(VerifyProducerType(graph, "Squeeze", 0, "Relu"));
}

TEST_F(QuantUnsqueezeConv2DFusionPassTest, dequant_multi_consumer_no_broadcast)
{
    SetupSoc(true);
    auto graph = BuildChain("dequant_multi_consumer", false, DIM_3D, true, true, true, false, true);
    TestTotalPass("dequant_multi_consumer_no_broadcast", graph, SUCCESS);
    EXPECT_TRUE(VerifyProducerType(graph, "Squeeze", 0, "AscendDequant"));
    EXPECT_TRUE(VerifyProducerType(graph, "AscendQuant", 0, "Unsqueeze"));
}

TEST_F(QuantUnsqueezeConv2DFusionPassTest, post_fusion_shape_update)
{
    SetupSoc(true);
    auto graph = BuildChain("post_fusion_shape_update", false);
    TestTotalPass("post_fusion_shape_update", graph, SUCCESS);
    EXPECT_TRUE(VerifyNodeShape(graph, "AscendQuant", 0, false, SHAPE_4D));
    EXPECT_TRUE(VerifyNodeShape(graph, "AscendQuant", 0, true, SHAPE_4D));
    EXPECT_TRUE(VerifyNodeShape(graph, "AscendDequant", 0, false, CONV_OUT_4D));
    EXPECT_TRUE(VerifyNodeShape(graph, "AscendDequant", 0, true, CONV_OUT_4D));
    EXPECT_TRUE(VerifyNodeDtype(graph, "Unsqueeze", 0, DT_FLOAT16));
    EXPECT_TRUE(VerifyNodeDtype(graph, "Squeeze", 0, DT_FLOAT16));
}

TEST_F(QuantUnsqueezeConv2DFusionPassTest, control_edge_on_quant_reject)
{
    SetupSoc(true);
    TestGraph builder("control_edge_on_quant");
    builder.AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, SHAPE_3D), true);
    builder.AddNode(MakeUnsqueeze("Unsqueeze", SHAPE_3D, SHAPE_4D), {});
    builder.Connect("AscendQuant", 0, "Unsqueeze", 0);
    builder.AddConv2D(
        Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, SHAPE_4D, FILTER_SHAPE, CONV_OUT_4D), false, true,
        false);
    builder.Connect("Unsqueeze", 0, "Conv2D", 0);
    builder.AddNode(MakeSqueeze("Squeeze", CONV_OUT_4D, SQUEEZE_OUT_3D), {});
    builder.Connect("Conv2D", 0, "Squeeze", 0);
    builder.AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D),
                             false, true);
    builder.Connect("Squeeze", 0, "AscendDequant", 0);
    builder.SetOutput("AscendDequant");
    auto graph = builder.Build();
    GNode quant = builder.GetNode("AscendQuant");
    GNode unsqueeze = builder.GetNode("Unsqueeze");
    ASSERT_EQ(graph->AddControlEdge(unsqueeze, quant), GRAPH_SUCCESS);
    TestTotalPass("control_edge_on_quant_reject", graph, CONV_NOT_CHANGED);
    EXPECT_FALSE(VerifyProducerType(graph, "AscendQuant", 0, "Unsqueeze"));
}

TEST_F(QuantUnsqueezeConv2DFusionPassTest, control_edge_on_conv_reject)
{
    SetupSoc(true);
    TestGraph builder("control_edge_on_conv");
    builder.AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, SHAPE_3D), true);
    builder.AddNode(MakeUnsqueeze("Unsqueeze", SHAPE_3D, SHAPE_4D), {});
    builder.Connect("AscendQuant", 0, "Unsqueeze", 0);
    builder.AddConv2D(
        Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, SHAPE_4D, FILTER_SHAPE, CONV_OUT_4D), false, true,
        false);
    builder.Connect("Unsqueeze", 0, "Conv2D", 0);
    builder.AddNode(MakeSqueeze("Squeeze", CONV_OUT_4D, SQUEEZE_OUT_3D), {});
    builder.Connect("Conv2D", 0, "Squeeze", 0);
    builder.AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, SQUEEZE_OUT_3D),
                             false, true);
    builder.Connect("Squeeze", 0, "AscendDequant", 0);
    builder.SetOutput("AscendDequant");
    auto graph = builder.Build();
    GNode conv = builder.GetNode("Conv2D");
    GNode squeeze = builder.GetNode("Squeeze");
    ASSERT_EQ(graph->AddControlEdge(squeeze, conv), GRAPH_SUCCESS);
    TestTotalPass("control_edge_on_conv_reject", graph, CONV_NOT_CHANGED);
    EXPECT_FALSE(VerifyProducerType(graph, "AscendQuant", 0, "Unsqueeze"));
}

TEST_F(QuantUnsqueezeConv2DFusionPassTest, missing_dequant_reject)
{
    SetupSoc(true);
    auto graph = BuildChain("missing_dequant", false, DIM_3D, true, true, false);
    TestTotalPass("missing_dequant_reject", graph, CONV_NOT_CHANGED);
    EXPECT_FALSE(VerifyProducerType(graph, "AscendQuant", 0, "Unsqueeze"));
}

#endif
