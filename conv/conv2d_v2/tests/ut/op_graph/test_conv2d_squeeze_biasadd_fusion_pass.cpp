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
#include "../../../op_graph/fusion_pass/conv2d_squeeze_biasadd_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace Conv2DSqueezeBiasaddFusion;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

struct SqueezeConfig : public NodeConfig {
    SqueezeConfig()
    {
        name = "Squeeze";
        opType = "Squeeze";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("axis", std::vector<int64_t>{0});
    }

    static SqueezeConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                               const std::vector<int64_t>& shape = {1, 3, 244, 244},
                               const std::vector<int64_t>& outputShape = {3, 244, 244})
    {
        SqueezeConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddOutput(dataType, format, outputShape, nodeName + "_y");
        return config;
    }

    SqueezeConfig& WithAxis(const std::vector<int64_t>& axis)
    {
        SetAttr("axis", axis);
        return *this;
    }
};

struct BiasAddConfig : public NodeConfig {
    BiasAddConfig()
    {
        name = "BiasAdd";
        opType = "BiasAdd";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"bias", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("data_format", std::string("NCHW"));
    }

    static BiasAddConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                               const std::vector<int64_t>& shape = {3, 244, 244}, DataType biasDtype = DT_FLOAT16,
                               const std::vector<int64_t>& biasShape = {3})
    {
        BiasAddConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddInput(biasDtype, FORMAT_ND, biasShape, nodeName + "_bias")
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }

    BiasAddConfig& WithDataFormat(const std::string& dataFormat)
    {
        SetAttr("data_format", dataFormat);
        return *this;
    }
};

struct VariableConfig : public NodeConfig {
    VariableConfig()
    {
        name = "Variable";
        opType = "Variable";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    }

    static VariableConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT16,
                                const std::vector<int64_t>& shape = {3})
    {
        VariableConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, FORMAT_ND, shape, nodeName + "_x")
            .AddOutput(dataType, FORMAT_ND, shape, nodeName + "_y");
        return config;
    }
};

struct SquareConfig : public NodeConfig {
    SquareConfig()
    {
        name = "Square";
        opType = "Square";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    }

    static SquareConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                              const std::vector<int64_t>& shape = {3, 244, 244})
    {
        SquareConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }
};

struct AvgPoolConfig : public NodeConfig {
    AvgPoolConfig()
    {
        name = "AvgPool";
        opType = "AvgPool";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("kernel_size", std::vector<int64_t>{1, 1});
        SetAttr("strides", std::vector<int64_t>{1, 1});
        SetAttr("pads", std::vector<int64_t>{0, 0, 0, 0});
        SetAttr("data_format", std::string("NCHW"));
    }

    static AvgPoolConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT16, Format format = FORMAT_NCHW,
                               const std::vector<int64_t>& shape = {1, 3, 244, 244})
    {
        AvgPoolConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }
};

struct AvgPoolV2Config : public NodeConfig {
    AvgPoolV2Config()
    {
        name = "AvgPoolV2";
        opType = "AvgPoolV2";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("kernel_size", std::vector<int64_t>{1, 1});
        SetAttr("strides", std::vector<int64_t>{1, 1});
        SetAttr("pads", std::vector<int64_t>{0, 0, 0, 0});
        SetAttr("data_format", std::string("NCHW"));
    }

    static AvgPoolV2Config Basic(const std::string& nodeName, DataType dataType = DT_FLOAT16,
                                 Format format = FORMAT_NCHW, const std::vector<int64_t>& shape = {1, 3, 244, 244})
    {
        AvgPoolV2Config config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }
};

struct FullyConnectedConfig : public NodeConfig {
    FullyConnectedConfig()
    {
        name = "FullyConnected";
        opType = "FullyConnected";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"w", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("data_format", std::string("NCHW"));
    }

    static FullyConnectedConfig Basic(const std::string& nodeName, DataType dataType = DT_FLOAT16,
                                      Format format = FORMAT_NCHW, const std::vector<int64_t>& shape = {1, 3, 244, 244})
    {
        FullyConnectedConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddInput(dataType, FORMAT_ND, {3, 3}, nodeName + "_w")
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }
};

class Conv2DSqueezeBiasaddFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Conv2DSqueezeBiasaddFusionPassTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Conv2DSqueezeBiasaddFusionPassTest TearDown" << std::endl; }

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        Conv2DSqueezeBiasaddFusionPass pass;
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
    }

    void VerifyReorderedTopology(GraphPtr& graph)
    {
        GNode squeezeNode;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Squeeze", squeezeNode));

        std::string squeezeInputProducerType;
        EXPECT_TRUE(GraphChecker::GetInputProducerType(squeezeNode, 0, squeezeInputProducerType));
        EXPECT_EQ(squeezeInputProducerType, "BiasAdd");

        GNode biasaddNode;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "BiasAdd", biasaddNode));

        std::string biasaddInputProducerType;
        EXPECT_TRUE(GraphChecker::GetInputProducerType(biasaddNode, 0, biasaddInputProducerType));
        EXPECT_EQ(biasaddInputProducerType, "Conv2D");
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

    GraphPtr BuildConvSqueezeBiasaddGraph(const std::string& graphName, DataType convDtype = DT_FLOAT16,
                                          Format convFormat = FORMAT_NCHW, DataType biasDtype = DT_FLOAT16,
                                          const std::vector<int64_t>& biasShape = {3},
                                          const std::vector<int64_t>& squeezeAxis = {0})
    {
        TestGraph builder(graphName);
        std::vector<int64_t> convInputShape = {1, 16, 244, 244};
        std::vector<int64_t> convFilterShape = {3, 16, 3, 3};
        std::vector<int64_t> convOutputShape = {1, 3, 244, 244};
        std::vector<int64_t> squeezeOutputShape = {3, 244, 244};

        Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", convDtype, convDtype, convFormat, convInputShape,
                                                   convFilterShape, convOutputShape);
        SqueezeConfig squeezeCfg = SqueezeConfig::Basic("Squeeze", convDtype, convFormat, convOutputShape,
                                                        squeezeOutputShape)
                                       .WithAxis(squeezeAxis);
        BiasAddConfig biasaddCfg = BiasAddConfig::Basic("BiasAdd", convDtype, convFormat, squeezeOutputShape, biasDtype,
                                                        biasShape);

        return builder.SetSocAscend950()
            .AddConv2D(convCfg)
            .AddNode(squeezeCfg)
            .AddNode(biasaddCfg, {1})
            .Connect("Conv2D", 0, "Squeeze", 0)
            .Connect("Squeeze", 0, "BiasAdd", 0)
            .SetOutput("BiasAdd")
            .Build();
    }
};

// ==========================================================================================
// Success: Conv2D -> Squeeze -> BiasAdd reordered to Conv2D -> BiasAdd -> Squeeze
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success)
{
    struct Point {
        const char* pointName;
        DataType convDtype;
        Format convFormat;
        DataType biasDtype;
    } const points[] = {
        {"fp16_nchw", DT_FLOAT16, FORMAT_NCHW, DT_FLOAT16}, {"fp32_nchw", DT_FLOAT, FORMAT_NCHW, DT_FLOAT},
        {"bf16_nchw", DT_BF16, FORMAT_NCHW, DT_BF16},       {"fp16_nhwc", DT_FLOAT16, FORMAT_NHWC, DT_FLOAT16},
        {"fp32_nhwc", DT_FLOAT, FORMAT_NHWC, DT_FLOAT},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        std::string name = std::string("fusion_success_") + p.pointName;
        auto graph = BuildConvSqueezeBiasaddGraph(name, p.convDtype, p.convFormat, p.biasDtype);
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Squeeze"));
        TestTotalPass(name, graph, SUCCESS);
        VerifyReorderedTopology(graph);
    }
}

// ==========================================================================================
// Success: Conv2D with bias -> Squeeze -> BiasAdd
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_conv_with_bias)
{
    TestGraph builder("fusion_success_conv_with_bias");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16).WithBias())
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("fusion_success_conv_with_bias", graph, SUCCESS);
    VerifyReorderedTopology(graph);
}

// ==========================================================================================
// Reentrant: two independent Conv->Squeeze->BiasAdd chains
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, reentrant_two_chains)
{
    TestGraph builder("reentrant_two_chains");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D_0", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(
                         SqueezeConfig::Basic("Squeeze_0", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd_0", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                     .Connect("Conv2D_0", 0, "Squeeze_0", 0)
                     .Connect("Squeeze_0", 0, "BiasAdd_0", 0)
                     .AddConv2D(Conv2DConfig::Basic("Conv2D_1", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(
                         SqueezeConfig::Basic("Squeeze_1", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd_1", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                     .Connect("Conv2D_1", 0, "Squeeze_1", 0)
                     .Connect("Squeeze_1", 0, "BiasAdd_1", 0)
                     .SetOutput("BiasAdd_0")
                     .SetOutput("BiasAdd_1")
                     .Build();

    TestTotalPass("reentrant_two_chains", graph, SUCCESS);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "BiasAdd"), 2);
}

// ==========================================================================================
// No fusion: bias dim != 1
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, no_fusion_bias_not_1d)
{
    TestGraph builder("no_fusion_bias_not_1d");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(
                         BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}, DT_FLOAT16, {1, 3}),
                         {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("no_fusion_bias_not_1d", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "BiasAdd"), 1);
}

// ==========================================================================================
// No fusion: training mode (bias producer is Variable)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, no_fusion_training_mode)
{
    TestGraph builder("no_fusion_training_mode");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(VariableConfig::Basic("Variable_bias", DT_FLOAT16, {3}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}))
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .Connect("Variable_bias", 0, "BiasAdd", 1)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("no_fusion_training_mode", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 1);
}

// ==========================================================================================
// No fusion: unknown shape on biasadd input
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, no_fusion_unknown_shape)
{
    TestGraph builder("no_fusion_unknown_shape");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    builder.UpdateNodeInputDesc("BiasAdd", 0, DT_FLOAT16, FORMAT_NCHW);
    TensorDesc unknownDesc;
    unknownDesc.SetDataType(DT_FLOAT16);
    unknownDesc.SetFormat(FORMAT_NCHW);
    unknownDesc.SetShape(Shape({-1, 3, 244, 244}));
    unknownDesc.SetOriginShape(Shape({-1, 3, 244, 244}));
    builder.UpdateNodeInputDescEx("BiasAdd", 0, unknownDesc);

    TestTotalPass("no_fusion_unknown_shape", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 1);
}

// ==========================================================================================
// No fusion: wrong topology - Squeeze input is not Conv2D
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, no_fusion_wrong_topology)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"squeeze_input_not_conv",
         [this]() {
             TestGraph builder("no_fusion_squeeze_input_not_conv");
             return builder.SetSocAscend950()
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                 .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                 .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                 .Connect("Relu", 0, "Squeeze", 0)
                 .Connect("Squeeze", 0, "BiasAdd", 0)
                 .SetOutput("BiasAdd")
                 .Build();
         }},
        {"squeeze_output_not_biasadd",
         [this]() {
             TestGraph builder("no_fusion_squeeze_output_not_biasadd");
             return builder.SetSocAscend950()
                 .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                 .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}))
                 .Connect("Conv2D", 0, "Squeeze", 0)
                 .Connect("Squeeze", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("no_fusion_wrong_topology_") + p.pointName, graph, CONV_NOT_CHANGED);
        EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 1);
    }
}

// ==========================================================================================
// Attr and desc preserved after fusion
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, attr_and_desc_preserved)
{
    auto graph = BuildConvSqueezeBiasaddGraph("attr_and_desc_preserved", DT_FLOAT16, FORMAT_NCHW, DT_FLOAT16, {3}, {0});
    TestTotalPass("attr_and_desc_preserved", graph, SUCCESS);
    VerifyReorderedTopology(graph);

    GNode squeezeNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Squeeze", squeezeNode));
    std::vector<int64_t> axis;
    EXPECT_TRUE(GraphChecker::GetListIntAttr(squeezeNode, "axis", axis));
    EXPECT_EQ(axis.size(), 1U);
    EXPECT_EQ(axis[0], 0);

    GNode biasaddNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "BiasAdd", biasaddNode));
    std::string dataFormat;
    EXPECT_TRUE(GraphChecker::GetNodeStringAttr(biasaddNode, "data_format", dataFormat));
    EXPECT_EQ(dataFormat, "NCHW");
}

// ==========================================================================================
// No fusion: bias shape contains -1 (unknown) — canndev test_4 migration
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, no_fusion_bias_unknown_shape)
{
    TestGraph builder("no_fusion_bias_unknown_shape");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}, DT_FLOAT16, {-1}),
                              {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("no_fusion_bias_unknown_shape", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 1);
}

// ==========================================================================================
// Success: BiasAdd has multiple consumers (Square + Relu), both migrate to Squeeze
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_biasadd_multiple_consumers)
{
    TestGraph builder("fusion_success_biasadd_multiple_consumers");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                     .AddNode(SquareConfig::Basic("Square", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}))
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .Connect("BiasAdd", 0, "Square", 0)
                     .Connect("BiasAdd", 0, "Relu", 0)
                     .SetOutput("Square")
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("fusion_success_biasadd_multiple_consumers", graph, SUCCESS);
    VerifyReorderedTopology(graph);

    GNode squeezeNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Squeeze", squeezeNode));
    EXPECT_EQ(squeezeNode.GetOutDataNodesAndPortIndexs(0).size(), 2U);
}

// ==========================================================================================
// Success: Squeeze has another consumer besides BiasAdd (stays connected to Squeeze)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_squeeze_other_consumer)
{
    TestGraph builder("fusion_success_squeeze_other_consumer");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}))
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .Connect("Squeeze", 0, "Relu", 0)
                     .SetOutput("BiasAdd")
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("fusion_success_squeeze_other_consumer", graph, SUCCESS);
    VerifyReorderedTopology(graph);

    GNode squeezeNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Squeeze", squeezeNode));
    EXPECT_EQ(squeezeNode.GetOutDataNodesAndPortIndexs(0).size(), 2U);
}

// ==========================================================================================
// Success: verify desc shapes after fusion (BiasAdd.x/y=4D, Squeeze.x=4D/y=3D)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_shape_verified)
{
    std::vector<int64_t> convOutputShape = {1, 3, 244, 244};
    std::vector<int64_t> squeezeOutputShape = {3, 244, 244};
    auto graph = BuildConvSqueezeBiasaddGraph("fusion_success_shape_verified");
    TestTotalPass("fusion_success_shape_verified", graph, SUCCESS);
    VerifyReorderedTopology(graph);

    EXPECT_TRUE(VerifyNodeShape(graph, "BiasAdd", 0, false, convOutputShape));
    EXPECT_TRUE(VerifyNodeShape(graph, "BiasAdd", 0, true, convOutputShape));
    EXPECT_TRUE(VerifyNodeShape(graph, "Squeeze", 0, false, convOutputShape));
    EXPECT_TRUE(VerifyNodeShape(graph, "Squeeze", 0, true, squeezeOutputShape));
}

// ==========================================================================================
// Success: BiasAdd data_format="NHWC" preserved
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_nhwc_biasadd_format)
{
    TestGraph builder("fusion_success_nhwc_biasadd_format");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NHWC))
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NHWC, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(
                         BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NHWC, {3, 244, 244}).WithDataFormat("NHWC"),
                         {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("fusion_success_nhwc_biasadd_format", graph, SUCCESS);
    VerifyReorderedTopology(graph);

    GNode biasaddNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "BiasAdd", biasaddNode));
    std::string dataFormat;
    EXPECT_TRUE(GraphChecker::GetNodeStringAttr(biasaddNode, "data_format", dataFormat));
    EXPECT_EQ(dataFormat, "NHWC");
}

// ==========================================================================================
// No fusion: BiasAdd input index swapped — Squeeze→BiasAdd.input1 instead of input0
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, no_fusion_biasadd_input_index_swapped)
{
    TestGraph builder("no_fusion_biasadd_input_index_swapped");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddData(DataConfig::Basic("Data_x", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}))
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Data_x", 0, "BiasAdd", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 1)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("no_fusion_biasadd_input_index_swapped", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 1);
}

// ==========================================================================================
// Success: BiasAdd data is 2D (Squeeze removes 2 dims of size 1)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_biasadd_2d_data)
{
    std::vector<int64_t> convOutputShape = {1, 1, 8, 8};
    std::vector<int64_t> squeezeOutputShape = {8, 8};
    TestGraph builder("fusion_success_biasadd_2d_data");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 16, 8, 8},
                                                    {1, 16, 3, 3}, convOutputShape))
                     .AddNode(
                         SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, convOutputShape, squeezeOutputShape)
                             .WithAxis({0, 1}))
                     .AddNode(
                         BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, squeezeOutputShape, DT_FLOAT16, {1}),
                         {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("fusion_success_biasadd_2d_data", graph, SUCCESS);
    VerifyReorderedTopology(graph);
    EXPECT_TRUE(VerifyNodeShape(graph, "BiasAdd", 0, false, convOutputShape));
    EXPECT_TRUE(VerifyNodeShape(graph, "BiasAdd", 0, true, convOutputShape));
    EXPECT_TRUE(VerifyNodeShape(graph, "Squeeze", 0, false, convOutputShape));
    EXPECT_TRUE(VerifyNodeShape(graph, "Squeeze", 0, true, squeezeOutputShape));
}

// ==========================================================================================
// Success: BiasAdd data is 4D (Squeeze output remains 4D, axis points to non-1 dim)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_biasadd_4d_data)
{
    std::vector<int64_t> convOutputShape = {1, 3, 8, 8};
    std::vector<int64_t> squeezeOutputShape = {1, 3, 8, 8};
    TestGraph builder("fusion_success_biasadd_4d_data");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 16, 8, 8},
                                                    {3, 16, 3, 3}, convOutputShape))
                     .AddNode(
                         SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, convOutputShape, squeezeOutputShape)
                             .WithAxis({-1}))
                     .AddNode(
                         BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, squeezeOutputShape, DT_FLOAT16, {3}),
                         {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("fusion_success_biasadd_4d_data", graph, SUCCESS);
    VerifyReorderedTopology(graph);
    EXPECT_TRUE(VerifyNodeShape(graph, "BiasAdd", 0, false, convOutputShape));
    EXPECT_TRUE(VerifyNodeShape(graph, "BiasAdd", 0, true, convOutputShape));
}

// ==========================================================================================
// Success: Conv2D input from Variable (bias from Const) — Pass only checks bias producer
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_conv_input_from_variable)
{
    TestGraph builder("fusion_success_conv_input_from_variable");
    auto graph = builder.SetSocAscend950()
                     .AddNode(VariableConfig::Basic("Variable_x", DT_FLOAT16, {1, 16, 244, 244}), {0})
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 16, 244, 244},
                                                    {3, 16, 3, 3}, {1, 3, 244, 244}),
                                false, true, true)
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                     .Connect("Variable_x", 0, "Conv2D", 0)
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("fusion_success_conv_input_from_variable", graph, SUCCESS);
    VerifyReorderedTopology(graph);
}

// ==========================================================================================
// Success: Squeeze without axis attr (axis not configured)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_squeeze_no_axis_attr)
{
    SqueezeConfig squeezeCfg = SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244},
                                                    {3, 244, 244});
    squeezeCfg.listIntAttrs.clear();

    TestGraph builder("fusion_success_squeeze_no_axis_attr");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(squeezeCfg)
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("fusion_success_squeeze_no_axis_attr", graph, SUCCESS);
    VerifyReorderedTopology(graph);

    GNode squeezeNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Squeeze", squeezeNode));
    std::vector<int64_t> axis;
    EXPECT_FALSE(GraphChecker::GetListIntAttr(squeezeNode, "axis", axis));
}

// ==========================================================================================
// Success: Squeeze with both axis and squeeze_dims attrs simultaneously
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_squeeze_axis_and_squeeze_dims)
{
    TestGraph builder("fusion_success_squeeze_axis_and_squeeze_dims");
    SqueezeConfig squeezeCfg = SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244})
                                   .WithAxis({0});
    squeezeCfg.SetAttr("squeeze_dims", std::vector<int64_t>{0});

    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16))
                     .AddNode(squeezeCfg)
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("fusion_success_squeeze_axis_and_squeeze_dims", graph, SUCCESS);
    VerifyReorderedTopology(graph);

    GNode squeezeNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Squeeze", squeezeNode));
    std::vector<int64_t> axis;
    EXPECT_TRUE(GraphChecker::GetListIntAttr(squeezeNode, "axis", axis));
    EXPECT_EQ(axis.size(), 1U);
    EXPECT_EQ(axis[0], 0);
    std::vector<int64_t> squeezeDims;
    EXPECT_TRUE(GraphChecker::GetListIntAttr(squeezeNode, "squeeze_dims", squeezeDims));
    EXPECT_EQ(squeezeDims.size(), 1U);
    EXPECT_EQ(squeezeDims[0], 0);
}

// ==========================================================================================
// Reentrant: two chains with heterogeneous dtypes (fp16 + int8/int32)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, reentrant_heterogeneous_dtypes)
{
    TestGraph builder("reentrant_heterogeneous_dtypes");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D_0", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 16, 8, 8},
                                                    {3, 16, 3, 3}, {1, 3, 8, 8}))
                     .AddNode(SqueezeConfig::Basic("Squeeze_0", DT_FLOAT16, FORMAT_NCHW, {1, 3, 8, 8}, {3, 8, 8}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd_0", DT_FLOAT16, FORMAT_NCHW, {3, 8, 8}), {1})
                     .Connect("Conv2D_0", 0, "Squeeze_0", 0)
                     .Connect("Squeeze_0", 0, "BiasAdd_0", 0)
                     .AddConv2D(Conv2DConfig::Basic("Conv2D_1", DT_INT8, DT_INT32, FORMAT_NCHW, {1, 16, 8, 8},
                                                    {3, 16, 3, 3}, {1, 3, 8, 8}))
                     .AddNode(SqueezeConfig::Basic("Squeeze_1", DT_INT32, FORMAT_NCHW, {1, 3, 8, 8}, {3, 8, 8}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd_1", DT_INT32, FORMAT_NCHW, {3, 8, 8}, DT_INT32, {3}), {1})
                     .Connect("Conv2D_1", 0, "Squeeze_1", 0)
                     .Connect("Squeeze_1", 0, "BiasAdd_1", 0)
                     .SetOutput("BiasAdd_0")
                     .SetOutput("BiasAdd_1")
                     .Build();

    TestTotalPass("reentrant_heterogeneous_dtypes", graph, SUCCESS);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "BiasAdd"), 2);
}

// ==========================================================================================
// No fusion: Squeeze input is DepthwiseConv2D / AvgPool / FullyConnected (not Conv2D)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, no_fusion_squeeze_input_not_conv2d_types)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"depthwise_conv2d",
         [this]() {
             TestGraph builder("no_fusion_depthwise_conv2d");
             return builder.SetSocAscend950()
                 .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_FLOAT16, FORMAT_NCHW,
                                                                  {1, 3, 244, 244}, {3, 1, 1, 1}, {1, 3, 244, 244}))
                 .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                 .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                 .Connect("DepthwiseConv2D", 0, "Squeeze", 0)
                 .Connect("Squeeze", 0, "BiasAdd", 0)
                 .SetOutput("BiasAdd")
                 .Build();
         }},
        {"avgpool",
         [this]() {
             TestGraph builder("no_fusion_avgpool");
             return builder.SetSocAscend950()
                 .AddNode(AvgPoolConfig::Basic("AvgPool", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}), {0})
                 .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                 .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                 .Connect("AvgPool", 0, "Squeeze", 0)
                 .Connect("Squeeze", 0, "BiasAdd", 0)
                 .SetOutput("BiasAdd")
                 .Build();
         }},
        {"fully_connected",
         [this]() {
             TestGraph builder("no_fusion_fully_connected");
             return builder.SetSocAscend950()
                 .AddNode(FullyConnectedConfig::Basic("FullyConnected", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}),
                          {0, 1})
                 .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                 .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                 .Connect("FullyConnected", 0, "Squeeze", 0)
                 .Connect("Squeeze", 0, "BiasAdd", 0)
                 .SetOutput("BiasAdd")
                 .Build();
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("no_fusion_squeeze_input_") + p.pointName, graph, CONV_NOT_CHANGED);
        EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 1);
    }
}

// ==========================================================================================
// Success: int8 Conv2D chain (input int8, output int32)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_int8_dtype)
{
    std::vector<int64_t> convOutputShape = {1, 3, 244, 244};
    std::vector<int64_t> squeezeOutputShape = {3, 244, 244};
    TestGraph builder("fusion_success_int8_dtype");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D", DT_INT8, DT_INT32, FORMAT_NCHW, {1, 16, 244, 244},
                                                    {3, 16, 3, 3}, convOutputShape))
                     .AddNode(
                         SqueezeConfig::Basic("Squeeze", DT_INT32, FORMAT_NCHW, convOutputShape, squeezeOutputShape))
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_INT32, FORMAT_NCHW, squeezeOutputShape, DT_INT32, {3}),
                              {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("fusion_success_int8_dtype", graph, SUCCESS);
    VerifyReorderedTopology(graph);
    EXPECT_TRUE(VerifyNodeDtype(graph, "BiasAdd", 0, DT_INT32));
    EXPECT_TRUE(VerifyNodeDtype(graph, "Squeeze", 0, DT_INT32));
}

// ==========================================================================================
// Success: group conv (groups=4, cin=16, cout=4, Cin/Cout factors, Cin!=Cout)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_group_conv)
{
    int64_t groups = 4;
    std::vector<int64_t> filterShape = {4, 4, 3, 3};
    std::vector<int64_t> convOutputShape = {1, 4, 244, 244};
    std::vector<int64_t> squeezeOutputShape = {4, 244, 244};
    Conv2DConfig convCfg = Conv2DConfig::Basic("Conv2D", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 16, 244, 244},
                                               filterShape, convOutputShape);
    convCfg.SetAttr("groups", groups);
    TestGraph builder("fusion_success_group_conv");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(convCfg)
                     .AddNode(
                         SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, convOutputShape, squeezeOutputShape))
                     .AddNode(
                         BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, squeezeOutputShape, DT_FLOAT16, {4}),
                         {1})
                     .Connect("Conv2D", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("fusion_success_group_conv", graph, SUCCESS);
    VerifyReorderedTopology(graph);
}

// ==========================================================================================
// Success: mixed precision cascade (Conv0 fp16 -> ... -> Conv1 int8 -> ...)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_mixed_precision_cascade)
{
    std::vector<int64_t> conv0OutShape = {1, 3, 8, 8};
    std::vector<int64_t> squeeze0OutShape = {1, 3, 8, 8};
    std::vector<int64_t> conv1OutShape = {1, 3, 8, 8};
    std::vector<int64_t> squeeze1OutShape = {3, 8, 8};
    TestGraph builder("fusion_success_mixed_precision_cascade");
    auto graph = builder.SetSocAscend950()
                     .AddConv2D(Conv2DConfig::Basic("Conv2D_0", DT_FLOAT16, DT_FLOAT16, FORMAT_NCHW, {1, 16, 8, 8},
                                                    {3, 16, 3, 3}, conv0OutShape))
                     .AddNode(
                         SqueezeConfig::Basic("Squeeze_0", DT_FLOAT16, FORMAT_NCHW, conv0OutShape, squeeze0OutShape)
                             .WithAxis({-1}))
                     .AddNode(
                         BiasAddConfig::Basic("BiasAdd_0", DT_FLOAT16, FORMAT_NCHW, squeeze0OutShape, DT_FLOAT16, {3}),
                         {1})
                     .AddAscendQuant(AscendQuantConfig::Basic("Quant", DT_FLOAT16, FORMAT_NCHW, squeeze0OutShape),
                                     false)
                     .AddConv2D(Conv2DConfig::Basic("Conv2D_1", DT_INT8, DT_INT32, FORMAT_NCHW, squeeze0OutShape,
                                                    {3, 3, 3, 3}, conv1OutShape),
                                false, true, true)
                     .AddNode(SqueezeConfig::Basic("Squeeze_1", DT_INT32, FORMAT_NCHW, conv1OutShape, squeeze1OutShape)
                                  .WithAxis({0}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd_1", DT_INT32, FORMAT_NCHW, squeeze1OutShape, DT_INT32, {3}),
                              {1})
                     .Connect("Conv2D_0", 0, "Squeeze_0", 0)
                     .Connect("Squeeze_0", 0, "BiasAdd_0", 0)
                     .Connect("BiasAdd_0", 0, "Quant", 0)
                     .Connect("Quant", 0, "Conv2D_1", 0)
                     .Connect("Conv2D_1", 0, "Squeeze_1", 0)
                     .Connect("Squeeze_1", 0, "BiasAdd_1", 0)
                     .SetOutput("BiasAdd_1")
                     .Build();

    TestTotalPass("fusion_success_mixed_precision_cascade", graph, SUCCESS);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "BiasAdd"), 2);
}

// ==========================================================================================
// Success: additional dtype combinations (uint8, int8->int8, fp16->fp32)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, fusion_success_dtype_combinations)
{
    struct Point {
        const char* pointName;
        DataType convDtype;
    } const points[] = {
        {"uint8", DT_UINT8},
        {"int8_to_int8", DT_INT8},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        std::string name = std::string("fusion_success_dtype_") + p.pointName;
        auto graph = BuildConvSqueezeBiasaddGraph(name, p.convDtype, FORMAT_NCHW, p.convDtype);
        TestTotalPass(name, graph, SUCCESS);
        VerifyReorderedTopology(graph);
        EXPECT_TRUE(VerifyNodeDtype(graph, "BiasAdd", 0, p.convDtype));
    }
}

// ==========================================================================================
// No fusion: Squeeze input is AvgPoolV2 (not Conv2D)
// ==========================================================================================
TEST_F(Conv2DSqueezeBiasaddFusionPassTest, no_fusion_squeeze_input_avgpool_v2)
{
    TestGraph builder("no_fusion_squeeze_input_avgpool_v2");
    auto graph = builder.SetSocAscend950()
                     .AddNode(AvgPoolV2Config::Basic("AvgPoolV2", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}), {0})
                     .AddNode(SqueezeConfig::Basic("Squeeze", DT_FLOAT16, FORMAT_NCHW, {1, 3, 244, 244}, {3, 244, 244}))
                     .AddNode(BiasAddConfig::Basic("BiasAdd", DT_FLOAT16, FORMAT_NCHW, {3, 244, 244}), {1})
                     .Connect("AvgPoolV2", 0, "Squeeze", 0)
                     .Connect("Squeeze", 0, "BiasAdd", 0)
                     .SetOutput("BiasAdd")
                     .Build();

    TestTotalPass("no_fusion_squeeze_input_avgpool_v2", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 1);
}

#endif // GE_COMPILER_VERSION_NUM >= 90000000U
