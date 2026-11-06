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
#include <set>
#include <string>
#include <vector>

#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "../../../op_graph/fusion_pass/conv3d_quant_process_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

namespace {
const std::vector<int64_t> FMAP_SHAPE_ROLLBACK = {1, 8, 8, 32, 32};
const std::vector<int64_t> FILTER_SHAPE_ROLLBACK = {3, 8, 3, 3, 3};
const std::vector<int64_t> OUT_SHAPE_ROLLBACK = {1, 3, 8, 32, 32};
const std::vector<int64_t> FMAP_SHAPE_BIAS_OPT = {1, 32, 8, 32, 32};
const std::vector<int64_t> FILTER_SHAPE_BIAS_OPT_4D = {32, 8, 3, 3}; // NCHW 4D，N 轴 = co
const std::vector<int64_t> OUT_SHAPE_BIAS_OPT = {1, 3, 8, 32, 32};

struct PadConfig : public NodeConfig {
    PadConfig()
    {
        name = "Pad";
        opType = "Pad";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    }

    static PadConfig Basic(const std::string& nodeName, DataType dataType, Format format,
                           const std::vector<int64_t>& shape)
    {
        PadConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, shape, nodeName + "_x")
            .AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }
};

struct ConstConfig : public NodeConfig {
    ConstConfig()
    {
        name = "Const";
        opType = "Const";
        inputDefs = {};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    }

    static ConstConfig Basic(const std::string& nodeName, DataType dataType, Format format,
                             const std::vector<int64_t>& shape)
    {
        ConstConfig config;
        config.SetName(nodeName).AddOutput(dataType, format, shape, nodeName + "_y");
        return config;
    }
};
} // namespace

static void NormalizeDeqScaleSource(GraphPtr& graph)
{
    // 先收集节点，避免在遍历图节点容器时增删节点。
    std::vector<GNode> dequantNodes;
    std::set<std::string> nodeNames;
    for (auto& node : graph->GetDirectNode()) {
        AscendString nodeName;
        if (node.GetName(nodeName) == GRAPH_SUCCESS) {
            nodeNames.insert(nodeName.GetString());
        }
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type == "AscendDequant") {
            dequantNodes.emplace_back(node);
        }
    }
    size_t scaleIndex = 0;
    for (auto& dequantNode : dequantNodes) {
        auto scalePair = dequantNode.GetInDataNodesAndPortIndexs(1);
        if (scalePair.first != nullptr) {
            AscendString srcType;
            if (scalePair.first->GetType(srcType) != GRAPH_SUCCESS || (srcType != "Data" && srcType != "RefData")) {
                continue; // 已是 Const 等合法源，继续处理其他 Dequant
            }
            FUSION_PASS_CHECK(graph->RemoveEdge(*scalePair.first, scalePair.second, dequantNode, 1) != GRAPH_SUCCESS,
                              OP_LOGW("NormalizeDeqScaleSource", "Remove auto scale edge failed."), return);
            if (scalePair.first->GetOutDataNodesAndPortIndexs(0).empty()) {
                FUSION_PASS_CHECK(graph->RemoveNode(*scalePair.first) != GRAPH_SUCCESS,
                                  OP_LOGW("NormalizeDeqScaleSource", "Remove auto scale node failed."), return);
            }
        }
        std::string constName;
        do {
            constName = "deq_scale_const_" + std::to_string(scaleIndex++);
        } while (nodeNames.count(constName) != 0U);
        nodeNames.insert(constName);
        ge::TensorDesc desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
        ge::Tensor tensor(desc);
        uint64_t deqVal = 1;
        tensor.SetData(reinterpret_cast<const uint8_t*>(&deqVal), sizeof(deqVal));
        ge::AttrValue attrValue;
        (void)attrValue.SetAttrValue(tensor);
        ge::GNode constNode = es::CompliantNodeBuilder(graph.get())
                                  .OpType("Const")
                                  .Name(constName.c_str())
                                  .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                  .IrDefAttrs(
                                      {{"value", es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", attrValue}})
                                  .InstanceOutputShape("y", std::vector<int64_t>{1})
                                  .InstanceOutputDataType("y", ge::DT_UINT64)
                                  .InstanceOutputFormat("y", ge::FORMAT_ND)
                                  .Build();
        FUSION_PASS_CHECK(graph->AddDataEdge(constNode, 0, dequantNode, 1) != GRAPH_SUCCESS,
                          OP_LOGW("NormalizeDeqScaleSource", "Add deq_scale const edge failed."), return);
    }
}

class Conv3DQuantProcessFusionPassForUT : public Conv3DQuantProcessFusionPass {
protected:
    bool CheckOpSupported(const ge::GNode&) const override { return true; }
};

class Conv3DQuantProcessFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        Conv3DQuantProcessFusionPassForUT pass;

        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        NormalizeDeqScaleSource(graph);
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
    }

    GNode GetFirstNodeByType(GraphPtr& graph, const std::string& opType)
    {
        GNode node;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, opType, node));
        return node;
    }
};

// ==========================================================================================
// Bias optimize: AscendQuant -> Conv3D -> AscendDequant, C > 16, 无 bias 输入
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_bias_optimize_no_bias_success)
{
    TestGraph builder("conv3d_quant_bias_optimize_no_bias_success");
    Conv3DConfig convConfig;
    convConfig.SetName("Conv3D")
        .AddInput(DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT, "Conv3D_x")
        .AddInput(DT_INT8, FORMAT_NCHW, FILTER_SHAPE_BIAS_OPT_4D, "Conv3D_filter")
        .AddOutput(DT_INT32, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT, "y");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT), true)
                     .AddConv3D(convConfig, false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW,
                                                             FILTER_SHAPE_BIAS_OPT_4D),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_BIAS_OPT_4D), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .Connect("AscendQuant", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("conv3d_quant_bias_optimize_no_bias_success", graph, SUCCESS);

    // bias optimize 保留量化链，插入 QuantBiasOptimization，无 bias 时补零 Const bias（co = filter N 轴 = 32）
    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 1);
    GNode biasConst;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
    TensorDesc biasDesc;
    ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetDataType(), DT_INT32);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), 32);
    GNode fused = GetFirstNodeByType(graph, "QuantBiasOptimization");
    AscendString cubeOpType;
    EXPECT_EQ(fused.GetAttr(AscendString("cube_op_type"), cubeOpType), GRAPH_SUCCESS);
    EXPECT_EQ(std::string(cubeOpType.GetString()), "Conv3D");

    GNode convNode = GetFirstNodeByType(graph, "Conv3D");
    std::string biasProducer;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(convNode, 2, biasProducer));
    EXPECT_EQ(biasProducer, "QuantBiasOptimization");
}

// ==========================================================================================
// Bias optimize: 有 bias 输入，不触达 GetCoValue
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_bias_optimize_with_bias_success)
{
    TestGraph builder("conv3d_quant_bias_optimize_with_bias_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT), true)
                     .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT,
                                                    FILTER_SHAPE_ROLLBACK, OUT_SHAPE_BIAS_OPT)
                                    .WithBias(),
                                false, false, true)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .Connect("AscendQuant", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("conv3d_quant_bias_optimize_with_bias_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    GNode biasConstUnused;
    EXPECT_FALSE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConstUnused));
}

// ==========================================================================================
// Quant rollback: C <= 16, 无 bias，quant/dequant 摘除、dtype 回退
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_rollback_no_bias_success)
{
    TestGraph builder("conv3d_quant_rollback_no_bias_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                    FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK),
                                false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .Connect("AscendQuant", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("conv3d_quant_rollback_no_bias_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantBiasRollBack"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);
    // 孤立节点清理：rollback 后 AscendWeightQuant 被删除
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendWeightQuant"), 0);
    GNode convNode = GetFirstNodeByType(graph, "Conv3D");
    // quant 摘除后 fmap 直连，filter 口由 QuantWeightRollBack 接管
    std::string filterProducer;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(convNode, 1, filterProducer));
    EXPECT_EQ(filterProducer, "QuantWeightRollBack");
    TensorDesc inDesc;
    ASSERT_EQ(convNode.GetInputDesc(0, inDesc), GRAPH_SUCCESS);
    EXPECT_EQ(inDesc.GetDataType(), DT_FLOAT16);
    TensorDesc outDesc;
    ASSERT_EQ(convNode.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
}

// ==========================================================================================
// Quant rollback: 有 bias，插入 QuantBiasRollBack
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_rollback_with_bias_success)
{
    TestGraph builder("conv3d_quant_rollback_with_bias_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                    FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK)
                                    .WithBias(),
                                false, false, true)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .Connect("AscendQuant", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("conv3d_quant_rollback_with_bias_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasRollBack"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendWeightQuant"), 0);
}

// ==========================================================================================
// Quant rollback with Pad: AscendQuant -> Pad -> Conv3D -> AscendDequant，Pad dtype 刷 FP16
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_rollback_with_pad_success)
{
    TestGraph builder("conv3d_quant_rollback_with_pad_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddNode(PadConfig::Basic("Pad", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), {})
                     .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                    FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK),
                                false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .Connect("AscendQuant", 0, "Pad", 0)
                     .Connect("Pad", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("conv3d_quant_rollback_with_pad_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "Pad"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    GNode padNode = GetFirstNodeByType(graph, "Pad");
    TensorDesc padOutDesc;
    ASSERT_EQ(padNode.GetOutputDesc(0, padOutDesc), GRAPH_SUCCESS);
    EXPECT_EQ(padOutDesc.GetDataType(), DT_FLOAT16);
}

// ==========================================================================================
// Bias optimize with Pad: pattern1（Quant -> Pad -> Conv3D -> Dequant）命中 bias optimize 分支
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_bias_optimize_with_pad_success)
{
    TestGraph builder("conv3d_quant_bias_optimize_with_pad_success");
    Conv3DConfig convConfig;
    convConfig.SetName("Conv3D")
        .AddInput(DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT, "Conv3D_x")
        .AddInput(DT_INT8, FORMAT_NCHW, FILTER_SHAPE_BIAS_OPT_4D, "Conv3D_filter")
        .AddOutput(DT_INT32, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT, "y");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT), true)
                     .AddNode(PadConfig::Basic("Pad", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT), {})
                     .AddConv3D(convConfig, false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW,
                                                             FILTER_SHAPE_BIAS_OPT_4D),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_BIAS_OPT_4D), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .Connect("AscendQuant", 0, "Pad", 0)
                     .Connect("Pad", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("conv3d_quant_bias_optimize_with_pad_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 1);
    GNode biasConst;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
    TensorDesc biasDesc;
    ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), 32);
    // bias optimize 不回退 dtype：Pad 保持 INT8 原样
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Pad"));
    GNode padNode = GetFirstNodeByType(graph, "Pad");
    TensorDesc padOutDesc;
    ASSERT_EQ(padNode.GetOutputDesc(0, padOutDesc), GRAPH_SUCCESS);
    EXPECT_EQ(padOutDesc.GetDataType(), DT_INT8);
}

// ==========================================================================================
// 5D filter + 有 bias: 不触达 GetCoValue，正常融合
// fmap 保持 NCHW（mode 判定可解析），仅 filter 为 NCDHW 5D
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_bias_optimize_5d_filter_with_bias_success)
{
    TestGraph builder("conv3d_quant_bias_optimize_5d_filter_with_bias_success");
    Conv3DConfig convConfig;
    convConfig.SetName("Conv3D")
        .AddInput(DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT, "Conv3D_x")
        .AddInput(DT_INT8, FORMAT_NCDHW, FILTER_SHAPE_ROLLBACK, "Conv3D_filter")
        .AddOutput(DT_INT32, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT, "y");
    convConfig.WithBias();
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT), true)
                     .AddConv3D(convConfig, false, false, true)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCDHW,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCDHW, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .Connect("AscendQuant", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("conv3d_quant_bias_optimize_5d_filter_with_bias_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    GNode biasConstUnused;
    EXPECT_FALSE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConstUnused));
}

// ==========================================================================================
// ND filter + 无 bias: GetCoValue 的 ND 分支直接返回，co 取调用方默认 1，bias shape{1}
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_bias_optimize_nd_filter_no_bias_success)
{
    TestGraph builder("conv3d_quant_bias_optimize_nd_filter_no_bias_success");
    Conv3DConfig convConfig;
    convConfig.SetName("Conv3D")
        .AddInput(DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT, "Conv3D_x")
        .AddInput(DT_INT8, FORMAT_ND, FILTER_SHAPE_ROLLBACK, "Conv3D_filter")
        .AddOutput(DT_INT32, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT, "y");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT), true)
                     .AddConv3D(convConfig, false, false, false)
                     .AddNode(
                         AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_ND, FILTER_SHAPE_ROLLBACK),
                         {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_ND, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .Connect("AscendQuant", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("conv3d_quant_bias_optimize_nd_filter_no_bias_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    GNode biasConst;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
    TensorDesc biasDesc;
    ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), 1);
}

// ==========================================================================================
// C0 边界: C=16 走 quant rollback，C=17 走 bias optimize（mode 判定分叉点）
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_c0_boundary)
{
    // C=16：channel > FP16_C0_VAL(16) 不成立 -> quant rollback
    {
        std::vector<int64_t> fmapShape = {1, 16, 8, 32, 32};
        std::vector<int64_t> filterShape4D = {16, 8, 3, 3};
        TestGraph builder("conv3d_quant_rollback_c0_eq16");
        Conv3DConfig convConfig;
        convConfig.SetName("Conv3D")
            .AddInput(DT_INT8, FORMAT_NCHW, fmapShape, "Conv3D_x")
            .AddInput(DT_INT8, FORMAT_NCHW, filterShape4D, "Conv3D_filter")
            .AddOutput(DT_INT32, FORMAT_NCHW, fmapShape, "y");
        auto graph = builder.SetSocAscend950()
                         .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, fmapShape),
                                         true)
                         .AddConv3D(convConfig, false, false, false)
                         .AddNode(
                             AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW, filterShape4D),
                             {})
                         .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, filterShape4D), {})
                         .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                         .AddAscendDequant(
                             AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, fmapShape))
                         .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, fmapShape))
                         .Connect("AscendQuant", 0, "Conv3D", 0)
                         .Connect("weight_const", 0, "AscendWeightQuant", 0)
                         .Connect("offset_const", 0, "AscendWeightQuant", 1)
                         .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                         .Connect("Conv3D", 0, "AscendDequant", 0)
                         .Connect("AscendDequant", 0, "Relu", 0)
                         .SetOutput("Relu")
                         .Build();
        TestTotalPass("conv3d_quant_rollback_c0_eq16", graph, SUCCESS);
        EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
        EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    }
    // C=17：channel > 16 成立 -> bias optimize（无 bias，filter N 轴 co = 17）
    {
        std::vector<int64_t> fmapShape = {1, 17, 8, 32, 32};
        std::vector<int64_t> filterShape4D = {17, 8, 3, 3};
        TestGraph builder("conv3d_quant_biasopt_c0_gt16");
        Conv3DConfig convConfig;
        convConfig.SetName("Conv3D")
            .AddInput(DT_INT8, FORMAT_NCHW, fmapShape, "Conv3D_x")
            .AddInput(DT_INT8, FORMAT_NCHW, filterShape4D, "Conv3D_filter")
            .AddOutput(DT_INT32, FORMAT_NCHW, fmapShape, "y");
        auto graph = builder.SetSocAscend950()
                         .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, fmapShape),
                                         true)
                         .AddConv3D(convConfig, false, false, false)
                         .AddNode(
                             AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW, filterShape4D),
                             {})
                         .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, filterShape4D), {})
                         .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                         .AddAscendDequant(
                             AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, fmapShape))
                         .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, fmapShape))
                         .Connect("AscendQuant", 0, "Conv3D", 0)
                         .Connect("weight_const", 0, "AscendWeightQuant", 0)
                         .Connect("offset_const", 0, "AscendWeightQuant", 1)
                         .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                         .Connect("Conv3D", 0, "AscendDequant", 0)
                         .Connect("AscendDequant", 0, "Relu", 0)
                         .SetOutput("Relu")
                         .Build();
        TestTotalPass("conv3d_quant_biasopt_c0_gt16", graph, SUCCESS);
        EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
        EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
        GNode biasConst;
        EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
        TensorDesc biasDesc;
        ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
        EXPECT_EQ(biasDesc.GetShape().GetDim(0), 17);
    }
}

// ==========================================================================================
// 5D filter 格式(NCDHW)已加入 AXIS map，bias optimize 且无 bias 时 GetCoValue 按 N 轴取
// dim[0] 成功，补零 Const bias 后融合成功（与 5d_filter_with_bias 用例互补覆盖无 bias 分支）
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_bias_optimize_5d_filter_no_bias_success)
{
    TestGraph builder("conv3d_quant_bias_optimize_5d_filter_no_bias_success");
    Conv3DConfig convConfig;
    convConfig.SetName("Conv3D")
        .AddInput(DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT, "Conv3D_x")
        .AddInput(DT_INT8, FORMAT_NCDHW, FILTER_SHAPE_ROLLBACK, "Conv3D_filter")
        .AddOutput(DT_INT32, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT, "y");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPT), true)
                     .AddConv3D(convConfig, false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCDHW,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCDHW, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_BIAS_OPT))
                     .Connect("AscendQuant", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("conv3d_quant_bias_optimize_5d_filter_no_bias_success", graph, SUCCESS);

    // 5D NCDHW filter {3,8,3,3,3}：Conv3D::GetCoValue 按 N 轴取 dim[0]=3，无 bias 时补零 Const bias 长度为 3
    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    GNode biasConst;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
    TensorDesc biasDesc;
    ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetDataType(), DT_INT32);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), 3);
}

// ==========================================================================================
// No fusion: 非 AscendQuant 前驱 / 非 AscendDequant 消费者 / 未知 shape / Pad 上游非 Quant
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_no_fusion)
{
    struct {
        const char* pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"no_quant_producer",
         [this]() {
             TestGraph builder("conv3d_quant_no_fusion_no_quant_producer");
             return builder.SetSocAscend950()
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK),
                            false, false, false)
                 .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                 .AddAscendDequant(
                     AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                 .Connect("weight_const", 0, "Conv3D", 0)
                 .Connect("Conv3D", 0, "AscendDequant", 0)
                 .Connect("AscendDequant", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         }},
        {"no_dequant_consumer",
         [this]() {
             TestGraph builder("conv3d_quant_no_fusion_no_dequant_consumer");
             return builder.SetSocAscend950()
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK),
                                 true)
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK),
                            false, false, false)
                 .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                 .AddRelu(ReluConfig::Basic("Relu", DT_INT32, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                 .Connect("AscendQuant", 0, "Conv3D", 0)
                 .Connect("weight_const", 0, "Conv3D", 1)
                 .Connect("Conv3D", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         }},
        {"unknown_shape",
         [this]() {
             TestGraph builder("conv3d_quant_no_fusion_unknown_shape");
             return builder.SetSocAscend950()
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, {1, -1, 8, 32, 32}),
                                 true)
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK),
                            false, false, false)
                 .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                 .AddAscendDequant(
                     AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                 .Connect("AscendQuant", 0, "Conv3D", 0)
                 .Connect("weight_const", 0, "Conv3D", 1)
                 .Connect("Conv3D", 0, "AscendDequant", 0)
                 .Connect("AscendDequant", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         }},
        {"pad_not_quant",
         [this]() {
             TestGraph builder("conv3d_quant_no_fusion_pad_not_quant");
             return builder.SetSocAscend950()
                 .AddNode(ConstConfig::Basic("input_const", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), {})
                 .AddNode(PadConfig::Basic("Pad", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), {})
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK),
                            false, false, false)
                 .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                 .AddAscendDequant(
                     AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                 .Connect("input_const", 0, "Pad", 0)
                 .Connect("Pad", 0, "Conv3D", 0)
                 .Connect("weight_const", 0, "Conv3D", 1)
                 .Connect("Conv3D", 0, "AscendDequant", 0)
                 .Connect("AscendDequant", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("conv3d_quant_no_fusion_") + p.pointName, graph, ge::GRAPH_NOT_CHANGED);
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
    }
}

// ==========================================================================================
// Multi consumer quant: quant 另有消费者时仅摘 cube 通路，quant 保留
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_rollback_multi_consumer_quant)
{
    TestGraph builder("conv3d_quant_rollback_multi_consumer_quant");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                    FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK),
                                false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("ReluAfterDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .Connect("AscendQuant", 0, "Conv3D", 0)
                     .Connect("AscendQuant", 0, "Relu", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "ReluAfterDequant", 0)
                     .SetOutput("Relu")
                     .SetOutput("ReluAfterDequant")
                     .Build();

    TestTotalPass("conv3d_quant_rollback_multi_consumer_quant", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);
    // 旁路保护：quant -> Relu 边原样保留
    GNode quantNode = GetFirstNodeByType(graph, "AscendQuant");
    GNode reluNode;
    ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "Relu", reluNode));
    EXPECT_TRUE(GraphChecker::HasOutEdgeTo(quantNode, 0, reluNode, 0));
}

// ==========================================================================================
// Multi consumer dequant: dequant 多消费边全量重连到 Conv 输出
// （与 quant 的"仅摘 cube 通路"不对称：dequant 类型无 cube 名过滤，所有消费者均重连）
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_rollback_multi_consumer_dequant)
{
    TestGraph builder("conv3d_quant_rollback_multi_consumer_dequant");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                    FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK),
                                false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu1", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu2", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .Connect("AscendQuant", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu1", 0)
                     .Connect("AscendDequant", 0, "Relu2", 0)
                     .SetOutput("Relu1")
                     .SetOutput("Relu2")
                     .Build();

    TestTotalPass("conv3d_quant_rollback_multi_consumer_dequant", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);
    // dequant 摘除后，两个消费者均直连 Conv3D 输出
    GNode convNode = GetFirstNodeByType(graph, "Conv3D");
    GNode relu1;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "Relu1", relu1));
    GNode relu2;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "Relu2", relu2));
    EXPECT_TRUE(GraphChecker::HasOutEdgeTo(convNode, 0, relu1, 0));
    EXPECT_TRUE(GraphChecker::HasOutEdgeTo(convNode, 0, relu2, 0));
}

// ==========================================================================================
// NHWC: Quant 输入 C 轴 = 3 映射（AXIS map NHWC 行），C=20 > 16 走 bias optimize
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_bias_optimize_nhwc_success)
{
    std::vector<int64_t> fmapShape = {1, 8, 8, 20};
    std::vector<int64_t> filterShape4D = {20, 8, 3, 3};
    TestGraph builder("conv3d_quant_bias_optimize_nhwc_success");
    Conv3DConfig convConfig;
    convConfig.SetName("Conv3D")
        .AddInput(DT_INT8, FORMAT_NHWC, fmapShape, "Conv3D_x")
        .AddInput(DT_INT8, FORMAT_NCHW, filterShape4D, "Conv3D_filter")
        .AddOutput(DT_INT32, FORMAT_NHWC, fmapShape, "y");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NHWC, fmapShape), true)
                     .AddConv3D(convConfig, false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCHW, filterShape4D),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCHW, filterShape4D), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NHWC, fmapShape))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NHWC, fmapShape))
                     .Connect("AscendQuant", 0, "Conv3D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "Conv3D", 1)
                     .Connect("Conv3D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("conv3d_quant_bias_optimize_nhwc_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    GNode biasConst;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
    TensorDesc biasDesc;
    ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), 20);
}

// ==========================================================================================
// 可重入: 两条量化链级联，验证 InitMember 复位后第二链正常融合
// ==========================================================================================
TEST_F(Conv3DQuantProcessFusionPassTest, conv3d_quant_rollback_reentrant)
{
    TestGraph builder("conv3d_quant_rollback_reentrant");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant1", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddConv3D(Conv3DConfig::Basic("Conv3D1", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                    FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK),
                                false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant1", DT_INT8, FORMAT_NCHW,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const_1", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const_1", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant1", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant2", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), false)
                     .AddConv3D(Conv3DConfig::Basic("Conv3D2", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                    FILTER_SHAPE_ROLLBACK, OUT_SHAPE_ROLLBACK),
                                false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant2", DT_INT8, FORMAT_NCHW,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const_2", DT_INT8, FORMAT_NCHW, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const_2", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant2", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, OUT_SHAPE_ROLLBACK))
                     .Connect("AscendQuant1", 0, "Conv3D1", 0)
                     .Connect("weight_const_1", 0, "AscendWeightQuant1", 0)
                     .Connect("offset_const_1", 0, "AscendWeightQuant1", 1)
                     .Connect("AscendWeightQuant1", 0, "Conv3D1", 1)
                     .Connect("Conv3D1", 0, "AscendDequant1", 0)
                     .Connect("AscendDequant1", 0, "AscendQuant2", 0)
                     .Connect("AscendQuant2", 0, "Conv3D2", 0)
                     .Connect("weight_const_2", 0, "AscendWeightQuant2", 0)
                     .Connect("offset_const_2", 0, "AscendWeightQuant2", 1)
                     .Connect("AscendWeightQuant2", 0, "Conv3D2", 1)
                     .Connect("Conv3D2", 0, "AscendDequant2", 0)
                     .Connect("AscendDequant2", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    // 两条链都必须拥有带数据的 Const scale；TestTotalPass 中再次规范化也应安全。
    NormalizeDeqScaleSource(graph);
    size_t normalizedDequantCount = 0;
    for (auto& node : graph->GetDirectNode()) {
        AscendString type;
        if (node.GetType(type) != GRAPH_SUCCESS || type != "AscendDequant") {
            continue;
        }
        Tensor scale;
        ASSERT_EQ(node.GetInputConstData(1, scale), GRAPH_SUCCESS);
        ASSERT_NE(scale.GetData(), nullptr);
        ++normalizedDequantCount;
    }
    ASSERT_EQ(normalizedDequantCount, 2U);

    TestTotalPass("conv3d_quant_rollback_reentrant", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "QuantWeightRollBack"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);
}

#endif
