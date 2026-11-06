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
#include "../../../op_graph/fusion_pass/dwconv2d_quant_process_fusion_pass.h"

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
const Status kConvNotChanged = ge::GRAPH_NOT_CHANGED;
const std::vector<int64_t> FMAP_SHAPE_BIAS_OPTIMIZE = {1, 20, 8, 8};
const std::vector<int64_t> FMAP_SHAPE_ROLLBACK = {1, 16, 8, 8};
const std::vector<int64_t> FILTER_SHAPE_BIAS_OPTIMIZE = {1, 1, 8, 20};
const std::vector<int64_t> FILTER_SHAPE_ROLLBACK = {1, 1, 8, 16};
const std::vector<int64_t> FILTER_SHAPE_5D = {3, 1, 8, 16, 16};

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
                continue;
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

class DWConv2DQuantProcessFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetSocIntrinsic(bool hasDn2Nz)
    {
        for (auto& socIter : PlatformInfoManager::Instance().platform_info_map_) {
            auto& intrinsicMap = socIter.second.ai_core_intrinsic_dtype_map;
            if (hasDn2Nz) {
                intrinsicMap["Intrinsic_data_move_out2l1_dn2nz"] = {"f16,f32"};
            } else {
                intrinsicMap.erase("Intrinsic_data_move_out2l1_dn2nz");
            }
        }
    }

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        DWConv2DQuantProcessFusionPass pass;
        SetSocIntrinsic(hasDn2NzSoc_);

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

    GNode GetDepthwiseNode(GraphPtr& graph)
    {
        GNode dwNode;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "DepthwiseConv2D", dwNode));
        return dwNode;
    }

    bool hasDn2NzSoc_ = false;
};

// ==========================================================================================
// Bias optimize: AscendQuant -> DepthwiseConv2D -> AscendDequant, C > 16, 无 bias 输入
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_bias_optimize_no_bias_success)
{
    TestGraph builder("dwconv2d_quant_bias_optimize_no_bias_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE),
                         true)
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE,
                                                      FILTER_SHAPE_BIAS_OPTIMIZE, FMAP_SHAPE_BIAS_OPTIMIZE),
                         false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_BIAS_OPTIMIZE),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_BIAS_OPTIMIZE), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    EXPECT_TRUE(GraphChecker::HasNode(graph, "DepthwiseConv2D"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendQuant"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "AscendDequant"));

    TestTotalPass("dwconv2d_quant_bias_optimize_no_bias_success", graph, SUCCESS);

    // bias optimize 保留量化链，插入 QuantBiasOptimization，无 bias 时补零 Const bias（co = HWCN N 轴 = 20）
    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 1);
    GNode biasConst;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
    TensorDesc biasDesc;
    ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetDataType(), DT_INT32);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), 20);
    GNode fused = GetFirstNodeByType(graph, "QuantBiasOptimization");
    int64_t dstType = -1;
    EXPECT_EQ(fused.GetAttr(AscendString("dst_type"), dstType), GRAPH_SUCCESS);
    EXPECT_EQ(dstType, static_cast<int64_t>(DT_INT8));
    int64_t groups = 0;
    EXPECT_EQ(fused.GetAttr(AscendString("groups"), groups), GRAPH_SUCCESS);
    EXPECT_EQ(groups, 1);
    AscendString cubeOpType;
    EXPECT_EQ(fused.GetAttr(AscendString("cube_op_type"), cubeOpType), GRAPH_SUCCESS);
    EXPECT_EQ(std::string(cubeOpType.GetString()), "DepthwiseConv2D");
    bool comeFromConst = false;
    EXPECT_TRUE(GraphChecker::GetNodeBoolAttr(fused, "_is_come_from_const_op", comeFromConst));
    EXPECT_TRUE(comeFromConst);
    float offsetAttr = -1.0F;
    EXPECT_EQ(fused.GetAttr(AscendString("offset"), offsetAttr), GRAPH_SUCCESS);
    EXPECT_EQ(offsetAttr, 0.0F);
    AscendString socVersion;
    EXPECT_EQ(fused.GetAttr(AscendString("soc_version"), socVersion), GRAPH_SUCCESS);
    // quant offset 写入 conv offset_x
    GNode dwNode = GetDepthwiseNode(graph);
    int64_t offsetX = -1;
    EXPECT_EQ(dwNode.GetAttr(AscendString("offset_x"), offsetX), GRAPH_SUCCESS);
    EXPECT_EQ(offsetX, 0);
}

// ==========================================================================================
// A5/950 家族（dn2nz intrinsic）：无 legacy depthwise TBE 实现，C > 16 仍走 quant rollback
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_rollback_on_dn2nz_soc_success)
{
    TestGraph builder("dwconv2d_quant_rollback_on_dn2nz_soc_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE),
                         true)
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE,
                                                      FILTER_SHAPE_BIAS_OPTIMIZE, FMAP_SHAPE_BIAS_OPTIMIZE),
                         false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_BIAS_OPTIMIZE),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_BIAS_OPTIMIZE), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();
    hasDn2NzSoc_ = true;

    TestTotalPass("dwconv2d_quant_rollback_on_dn2nz_soc_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantBiasRollBack"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendWeightQuant"), 0);

    GNode dwNode = GetDepthwiseNode(graph);
    TensorDesc inDesc;
    ASSERT_EQ(dwNode.GetInputDesc(0, inDesc), GRAPH_SUCCESS);
    EXPECT_EQ(inDesc.GetDataType(), DT_FLOAT16);
    ASSERT_EQ(dwNode.GetInputDesc(1, inDesc), GRAPH_SUCCESS);
    EXPECT_EQ(inDesc.GetDataType(), DT_FLOAT16);
    TensorDesc outDesc;
    ASSERT_EQ(dwNode.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
    std::string filterProducer;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(dwNode, 1, filterProducer));
    EXPECT_EQ(filterProducer, "QuantWeightRollBack");
}

// ==========================================================================================
// Bias optimize: dw 已有 bias 输入，不创建 _quant_bias 常量
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_bias_optimize_with_bias_success)
{
    TestGraph builder("dwconv2d_quant_bias_optimize_with_bias_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE),
                         true)
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE,
                                                      FILTER_SHAPE_BIAS_OPTIMIZE, FMAP_SHAPE_BIAS_OPTIMIZE)
                             .WithBias(DT_INT32, {20}),
                         false, false, true)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_BIAS_OPTIMIZE),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_BIAS_OPTIMIZE), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .SetOutput("AscendDequant")
                     .Build();

    TestTotalPass("dwconv2d_quant_bias_optimize_with_bias_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    GNode biasConst;
    EXPECT_FALSE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));

    GNode dwNode = GetDepthwiseNode(graph);
    std::string biasProducer;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(dwNode, 2, biasProducer));
    EXPECT_EQ(biasProducer, "QuantBiasOptimization");
}

// ==========================================================================================
// Quant rollback: C <= 16, 无 bias 输入
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_rollback_no_bias_success)
{
    TestGraph builder("dwconv2d_quant_rollback_no_bias_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                      FILTER_SHAPE_ROLLBACK, FMAP_SHAPE_ROLLBACK),
                         false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("dwconv2d_quant_rollback_no_bias_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantBiasRollBack"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendWeightQuant"), 0);

    GNode dwNode = GetDepthwiseNode(graph);
    std::string fmapProducer;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(dwNode, 0, fmapProducer));
    EXPECT_EQ(fmapProducer, "Data");
    std::string filterProducer;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(dwNode, 1, filterProducer));
    EXPECT_EQ(filterProducer, "QuantWeightRollBack");
    TensorDesc inDesc;
    ASSERT_EQ(dwNode.GetInputDesc(0, inDesc), GRAPH_SUCCESS);
    EXPECT_EQ(inDesc.GetDataType(), DT_FLOAT16);
    ASSERT_EQ(dwNode.GetInputDesc(1, inDesc), GRAPH_SUCCESS);
    EXPECT_EQ(inDesc.GetDataType(), DT_FLOAT16);
    TensorDesc outDesc;
    ASSERT_EQ(dwNode.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);

    GNode rollback = GetFirstNodeByType(graph, "QuantWeightRollBack");
    float scaleAttr = -1.0F;
    EXPECT_EQ(rollback.GetAttr(AscendString("scale"), scaleAttr), GRAPH_SUCCESS);
    EXPECT_EQ(scaleAttr, 1.0F);
    EXPECT_TRUE(GraphChecker::HasOutEdgeTo(rollback, 0, dwNode, 1));
}

// ==========================================================================================
// Quant rollback: 有 bias 输入，额外插入 QuantBiasRollBack
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_rollback_with_bias_success)
{
    TestGraph builder("dwconv2d_quant_rollback_with_bias_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                      FILTER_SHAPE_ROLLBACK, FMAP_SHAPE_ROLLBACK)
                             .WithBias(DT_INT32, {16}),
                         false, false, true)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("dwconv2d_quant_rollback_with_bias_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasRollBack"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);

    GNode dwNode = GetDepthwiseNode(graph);
    std::string biasProducer;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(dwNode, 2, biasProducer));
    EXPECT_EQ(biasProducer, "QuantBiasRollBack");
    GNode biasRollback = GetFirstNodeByType(graph, "QuantBiasRollBack");
    float scaleAttr = -1.0F;
    EXPECT_EQ(biasRollback.GetAttr(AscendString("scale"), scaleAttr), GRAPH_SUCCESS);
    EXPECT_EQ(scaleAttr, 1.0F);
}

// ==========================================================================================
// Quant rollback: AscendQuant -> Pad -> DepthwiseConv2D，Pad dtype 刷新为 FP16
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_rollback_with_pad_success)
{
    TestGraph builder("dwconv2d_quant_rollback_with_pad_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddNode(PadConfig::Basic("Pad", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), {})
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                      FILTER_SHAPE_ROLLBACK, FMAP_SHAPE_ROLLBACK),
                         false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .Connect("AscendQuant", 0, "Pad", 0)
                     .Connect("Pad", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("dwconv2d_quant_rollback_with_pad_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Pad"));

    GNode pad = GetFirstNodeByType(graph, "Pad");
    TensorDesc padInDesc;
    ASSERT_EQ(pad.GetInputDesc(0, padInDesc), GRAPH_SUCCESS);
    EXPECT_EQ(padInDesc.GetDataType(), DT_FLOAT16);
    TensorDesc padOutDesc;
    ASSERT_EQ(pad.GetOutputDesc(0, padOutDesc), GRAPH_SUCCESS);
    EXPECT_EQ(padOutDesc.GetDataType(), DT_FLOAT16);
}

// ==========================================================================================
// Bias optimize with Pad: pattern1（Quant -> Pad -> DepthwiseConv2D -> Dequant）命中 bias optimize 分支
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_bias_optimize_with_pad_success)
{
    TestGraph builder("dwconv2d_quant_bias_optimize_with_pad_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE),
                         true)
                     .AddNode(PadConfig::Basic("Pad", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE), {})
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE,
                                                      FILTER_SHAPE_BIAS_OPTIMIZE, FMAP_SHAPE_BIAS_OPTIMIZE),
                         false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_BIAS_OPTIMIZE),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_BIAS_OPTIMIZE), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .Connect("AscendQuant", 0, "Pad", 0)
                     .Connect("Pad", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("dwconv2d_quant_bias_optimize_with_pad_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 1);
    GNode biasConst;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
    TensorDesc biasDesc;
    ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), 20);
    // bias optimize 不回退 dtype：Pad 保持 INT8 原样
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Pad"));
    GNode padNode = GetFirstNodeByType(graph, "Pad");
    TensorDesc padOutDesc;
    ASSERT_EQ(padNode.GetOutputDesc(0, padOutDesc), GRAPH_SUCCESS);
    EXPECT_EQ(padOutDesc.GetDataType(), DT_INT8);
}

// ==========================================================================================
// Bias optimize: NHWC 覆盖（filter HWCN 的 N 轴取值，co = 1 × groups）
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_bias_optimize_nhwc_success)
{
    std::vector<int64_t> fmapShape = {1, 8, 8, 20};
    std::vector<int64_t> filterShape = {8, 8, 20, 1};
    TestGraph builder("dwconv2d_quant_bias_optimize_nhwc_success");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NHWC, fmapShape), true)
                     .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NHWC,
                                                                      fmapShape, filterShape, fmapShape),
                                         false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN, filterShape),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, filterShape), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NHWC, fmapShape))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NHWC, fmapShape))
                     .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("dwconv2d_quant_bias_optimize_nhwc_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    GNode biasConst;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
    TensorDesc biasDesc;
    ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), 1);
}

// ==========================================================================================
// Bias optimize: ND filter，GetCoValue 的 ND 分支直接返回，co 取调用方默认 1
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_bias_optimize_nd_filter_no_bias_success)
{
    TestGraph builder("dwconv2d_quant_bias_optimize_nd_filter_no_bias_success");
    DepthwiseConv2DConfig dwConfig;
    dwConfig.SetName("DepthwiseConv2D")
        .AddInput(DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE, "DepthwiseConv2D_x")
        .AddInput(DT_INT8, FORMAT_ND, FILTER_SHAPE_ROLLBACK, "DepthwiseConv2D_filter")
        .AddOutput(DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE, "y");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE),
                         true)
                     .AddDepthwiseConv2D(dwConfig, false, false, false)
                     .AddNode(
                         AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_ND, FILTER_SHAPE_ROLLBACK),
                         {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_ND, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("dwconv2d_quant_bias_optimize_nd_filter_no_bias_success", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    GNode biasConst;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
    TensorDesc biasDesc;
    ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), 1);
}

// ==========================================================================================
// Bias optimize: 5D filter(NCDHW) 格式已加入 kAxisIndexOfFormat，无 bias 时 GetCoValue 按
// N 轴取 co（PadShapeTo4Dim 截取前 4 维后 dim[0]），补零 Const bias 后融合成功
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_bias_optimize_5d_filter_no_bias_success)
{
    TestGraph builder("dwconv2d_quant_bias_optimize_5d_filter_no_bias_success");
    DepthwiseConv2DConfig dwConfig;
    dwConfig.SetName("DepthwiseConv2D")
        .AddInput(DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE, "DepthwiseConv2D_x")
        .AddInput(DT_INT8, FORMAT_NCDHW, FILTER_SHAPE_5D, "DepthwiseConv2D_filter")
        .AddOutput(DT_INT8, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE, "y");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE),
                         true)
                     .AddDepthwiseConv2D(dwConfig, false, false, false)
                     .AddNode(
                         AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_NCDHW, FILTER_SHAPE_5D),
                         {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_NCDHW, FILTER_SHAPE_5D), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_BIAS_OPTIMIZE))
                     .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();

    TestTotalPass("dwconv2d_quant_bias_optimize_5d_filter_no_bias_success", graph, SUCCESS);

    // 5D NCDHW filter：DWConv2D::GetCoValue 按 N 轴取 dim[0]=3，无 bias 时补零 Const bias 长度为 3
    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 1);
    GNode biasConst;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "_quant_bias", biasConst));
    TensorDesc biasDesc;
    ASSERT_EQ(biasConst.GetOutputDesc(0, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetDataType(), DT_INT32);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), 3);
}

// ==========================================================================================
// No fusion: C 维未知 shape / 结构不匹配 / Pad 上游非 Quant
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_no_fusion)
{
    struct Point {
        const char* pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"unknown_shape",
         [this]() {
             std::vector<int64_t> unknownFmap = {1, -1, 8, 8};
             TestGraph builder("dwconv2d_quant_no_fusion_unknown_shape");
             return builder.SetSocAscend950()
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, unknownFmap), true)
                 .AddDepthwiseConv2D(DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, unknownFmap,
                                                                  FILTER_SHAPE_ROLLBACK, unknownFmap),
                                     false, false, false)
                 .AddNode(
                     AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK),
                     {})
                 .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                 .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, unknownFmap))
                 .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                 .Connect("weight_const", 0, "AscendWeightQuant", 0)
                 .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                 .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         }},
        {"no_dequant_consumer",
         [this]() {
             TestGraph builder("dwconv2d_quant_no_fusion_no_dequant_consumer");
             return builder.SetSocAscend950()
                 .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK),
                                 true)
                 .AddDepthwiseConv2D(
                     DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                  FILTER_SHAPE_ROLLBACK, FMAP_SHAPE_ROLLBACK),
                     false, false, false)
                 .AddNode(
                     AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK),
                     {})
                 .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                 .AddRelu(ReluConfig::Basic("Relu", DT_INT32, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                 .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                 .Connect("weight_const", 0, "AscendWeightQuant", 0)
                 .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                 .Connect("DepthwiseConv2D", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         }},
        {"input_not_quant",
         [this]() {
             TestGraph builder("dwconv2d_quant_no_fusion_input_not_quant");
             return builder.SetSocAscend950()
                 .AddData(DataConfig::Basic("Data", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                 .AddDepthwiseConv2D(
                     DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                  FILTER_SHAPE_ROLLBACK, FMAP_SHAPE_ROLLBACK),
                     false, false, false)
                 .AddNode(
                     AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK),
                     {})
                 .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                 .AddAscendDequant(
                     AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                 .Connect("Data", 0, "DepthwiseConv2D", 0)
                 .Connect("weight_const", 0, "AscendWeightQuant", 0)
                 .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                 .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                 .SetOutput("AscendDequant")
                 .Build();
         }},
        {"pad_not_quant",
         [this]() {
             TestGraph builder("dwconv2d_quant_no_fusion_pad_not_quant");
             return builder.SetSocAscend950()
                 .AddNode(ConstConfig::Basic("input_const", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), {})
                 .AddNode(PadConfig::Basic("Pad", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), {})
                 .AddDepthwiseConv2D(
                     DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                  FILTER_SHAPE_ROLLBACK, FMAP_SHAPE_ROLLBACK),
                     false, false, false)
                 .AddNode(
                     AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK),
                     {})
                 .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                 .AddAscendDequant(
                     AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                 .Connect("input_const", 0, "Pad", 0)
                 .Connect("Pad", 0, "DepthwiseConv2D", 0)
                 .Connect("weight_const", 0, "AscendWeightQuant", 0)
                 .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                 .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                 .Connect("AscendDequant", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
         }},
    };

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        EXPECT_TRUE(GraphChecker::HasNode(graph, "DepthwiseConv2D"));
        TestTotalPass(std::string("dwconv2d_quant_no_fusion_") + p.pointName, graph, kConvNotChanged);
        EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
        EXPECT_FALSE(GraphChecker::HasNode(graph, "QuantBiasOptimization"));
    }
}

// ==========================================================================================
// Quant 多消费者：仅断开 quant->dw 边，quant 保留供其他消费者使用
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_rollback_multi_consumer_quant)
{
    TestGraph builder("dwconv2d_quant_rollback_multi_consumer_quant");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                      FILTER_SHAPE_ROLLBACK, FMAP_SHAPE_ROLLBACK),
                         false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu1", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu2", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                     .Connect("AscendQuant", 0, "Relu1", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu2", 0)
                     .SetOutput("Relu1")
                     .SetOutput("Relu2")
                     .Build();

    TestTotalPass("dwconv2d_quant_rollback_multi_consumer_quant", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);

    GNode dwNode = GetDepthwiseNode(graph);
    std::string fmapProducer;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(dwNode, 0, fmapProducer));
    EXPECT_EQ(fmapProducer, "Data");
    GNode quant = GetFirstNodeByType(graph, "AscendQuant");
    GNode relu1;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "Relu1", relu1));
    EXPECT_TRUE(GraphChecker::HasOutEdgeTo(quant, 0, relu1, 0));
}

// ==========================================================================================
// Dequant 多消费者：dequant 摘除后多消费边全量重连到 dw 输出
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_rollback_multi_consumer_dequant)
{
    TestGraph builder("dwconv2d_quant_rollback_multi_consumer_dequant");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                      FILTER_SHAPE_ROLLBACK, FMAP_SHAPE_ROLLBACK),
                         false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu1", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu2", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .Connect("AscendQuant", 0, "DepthwiseConv2D", 0)
                     .Connect("weight_const", 0, "AscendWeightQuant", 0)
                     .Connect("offset_const", 0, "AscendWeightQuant", 1)
                     .Connect("AscendWeightQuant", 0, "DepthwiseConv2D", 1)
                     .Connect("DepthwiseConv2D", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu1", 0)
                     .Connect("AscendDequant", 0, "Relu2", 0)
                     .SetOutput("Relu1")
                     .SetOutput("Relu2")
                     .Build();

    TestTotalPass("dwconv2d_quant_rollback_multi_consumer_dequant", graph, SUCCESS);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "QuantWeightRollBack"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);
    // dequant 摘除后，两个消费者均直连 DepthwiseConv2D 输出
    GNode dwNode = GetDepthwiseNode(graph);
    GNode relu1;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "Relu1", relu1));
    GNode relu2;
    EXPECT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "Relu2", relu2));
    EXPECT_TRUE(GraphChecker::HasOutEdgeTo(dwNode, 0, relu1, 0));
    EXPECT_TRUE(GraphChecker::HasOutEdgeTo(dwNode, 0, relu2, 0));
}

// ==========================================================================================
// 可重入：同图两条链均被处理
// ==========================================================================================
TEST_F(DWConv2DQuantProcessFusionPassTest, dwconv2d_quant_rollback_reentrant)
{
    TestGraph builder("dwconv2d_quant_rollback_reentrant");
    auto graph = builder.SetSocAscend950()
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant1", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D1", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                      FILTER_SHAPE_ROLLBACK, FMAP_SHAPE_ROLLBACK),
                         false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant1", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const_1", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const_1", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant1", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddDepthwiseConv2D(
                         DepthwiseConv2DConfig::Basic("DepthwiseConv2D2", DT_INT8, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK,
                                                      FILTER_SHAPE_ROLLBACK, FMAP_SHAPE_ROLLBACK),
                         false, false, false)
                     .AddNode(AscendWeightQuantConfig::Basic("AscendWeightQuant2", DT_INT8, FORMAT_HWCN,
                                                             FILTER_SHAPE_ROLLBACK),
                              {})
                     .AddNode(ConstConfig::Basic("weight_const_2", DT_INT8, FORMAT_HWCN, FILTER_SHAPE_ROLLBACK), {})
                     .AddNode(ConstConfig::Basic("offset_const_2", DT_INT8, FORMAT_ND, {1}), {})
                     .AddAscendQuant(
                         AscendQuantConfig::Basic("AscendQuant2", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK), true)
                     .AddAscendDequant(
                         AscendDequantConfig::Basic("AscendDequant2", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu1", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .AddRelu(ReluConfig::Basic("Relu2", DT_FLOAT16, FORMAT_NCHW, FMAP_SHAPE_ROLLBACK))
                     .Connect("AscendQuant1", 0, "DepthwiseConv2D1", 0)
                     .Connect("weight_const_1", 0, "AscendWeightQuant1", 0)
                     .Connect("offset_const_1", 0, "AscendWeightQuant1", 1)
                     .Connect("AscendWeightQuant1", 0, "DepthwiseConv2D1", 1)
                     .Connect("DepthwiseConv2D1", 0, "AscendDequant1", 0)
                     .Connect("AscendDequant1", 0, "Relu1", 0)
                     .Connect("AscendQuant2", 0, "DepthwiseConv2D2", 0)
                     .Connect("weight_const_2", 0, "AscendWeightQuant2", 0)
                     .Connect("offset_const_2", 0, "AscendWeightQuant2", 1)
                     .Connect("AscendWeightQuant2", 0, "DepthwiseConv2D2", 1)
                     .Connect("DepthwiseConv2D2", 0, "AscendDequant2", 0)
                     .Connect("AscendDequant2", 0, "Relu2", 0)
                     .SetOutput("Relu1")
                     .SetOutput("Relu2")
                     .Build();

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

    TestTotalPass("dwconv2d_quant_rollback_reentrant", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "QuantWeightRollBack"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendQuant"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "AscendDequant"), 0);
}

#endif
