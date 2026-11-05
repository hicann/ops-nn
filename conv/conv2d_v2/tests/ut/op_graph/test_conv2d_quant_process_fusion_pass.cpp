/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../op_graph/fusion_pass/conv2d_quant_process_fusion_pass.h"
#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "ge/compliant_node_builder.h"
#include "platform/platform_info.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace Conv2dQuantProcessFusion;

namespace {
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

constexpr int64_t C_SMALL = 4;  // C <= 16 走 QUANT_ROLLBACK
constexpr int64_t C_LARGE = 32; // C > 16 走 BIAS_OPTIMIZE
constexpr int64_t CO_OUT = 16;
const std::string NODE_QUANT = "AscendQuant";
const std::string NODE_PAD = "Pad";
const std::string NODE_CONV = "Conv2D";
const std::string NODE_DEQUANT = "AscendDequant";
const std::string NODE_WEIGHT_QUANT = "WeightQuant";

void SetSoc(const std::string& socVersion, bool tinyL1 = false)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiInfo;
    optiInfo.soc_version = socVersion.c_str();
    platformInfo.str_info.short_soc_version = socVersion.c_str();
    if (tinyL1) {
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_set_l1"] = {"f16"};
    } else {
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_set_l1"] = {"f16", "f32"};
    }
    PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiInfo);
}

void SetSocAscend950() { SetSoc("Ascend950"); }

NodeConfig MakeWeightQuantNode(const std::string& name, const std::vector<int64_t>& filterShape)
{
    NodeConfig cfg(name);
    cfg.opType = "AscendWeightQuant";
    cfg.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"offset", CompliantNodeBuilder::kEsIrInputRequired, ""}};
    cfg.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    cfg.AddInput(DT_FLOAT, FORMAT_NCHW, filterShape, name + "_w")
        .AddInput(DT_FLOAT, FORMAT_ND, {1}, name + "_offset")
        .AddOutput(DT_INT8, FORMAT_NCHW, filterShape, "y")
        .SetAttr("dst_type", int64_t(DT_INT8));
    return cfg;
}

NodeConfig MakePadNode(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape)
{
    NodeConfig cfg(NODE_PAD);
    cfg.opType = "Pad";
    cfg.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}};
    cfg.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
    cfg.AddInput(DT_INT8, FORMAT_NCHW, inShape, "pad_x").AddOutput(DT_INT8, FORMAT_NCHW, outShape, "y");
    return cfg;
}

// 基类 JudgeDeqscaleShape 要求 deq_scale 为 Const（真实图中来自 Initializer），框架 autoScale 造的是
// Data 节点，此处 Build 后补 Const 并连边
GraphPtr AttachDeqScaleConst(const GraphPtr& graph, const std::vector<int64_t>& dims)
{
    const size_t elemNum = static_cast<size_t>(std::abs(dims[0]));
    std::vector<ge::float32_t> scaleData(elemNum, 1.0F);
    ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor tensor(desc);
    tensor.SetData(reinterpret_cast<const uint8_t*>(scaleData.data()), scaleData.size() * sizeof(ge::float32_t));
    ge::AttrValue attrValue;
    (void)attrValue.SetAttrValue(tensor);
    auto constNode = ge::es::CompliantNodeBuilder(graph.get())
                         .OpType("Const")
                         .Name("deq_scale_const")
                         .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs({{"value", ge::es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", attrValue}})
                         .InstanceOutputShape("y", dims)
                         .InstanceOutputDataType("y", ge::DT_FLOAT)
                         .InstanceOutputFormat("y", ge::FORMAT_ND)
                         .Build();
    auto dequantNode = graph->FindNodeByName(ge::AscendString(NODE_DEQUANT.c_str()));
    graph->AddDataEdge(constNode, 0, *dequantNode, 1);
    return graph;
}

// Data(fmap) -> AscendQuant -> [Pad] -> Conv2D <- WeightQuant <- Data(weight/offset)
//                                          ^-- Data(bias)?        Conv2D -> AscendDequant <- Const(deq_scale)
// caffeWeight=true 时 conv.filter 直连 Data(weight)（无 AscendWeightQuant）
GraphPtr BuildQuantChain(const std::string& tag, int64_t cDim, bool withPad, bool withBias, bool caffeWeight = false)
{
    TestGraph builder(tag);
    builder.SetSocAscend950();
    const std::vector<int64_t> fmapShape = {1, cDim, 8, 8};
    const std::vector<int64_t> padOutShape = {1, cDim, 10, 10};
    const std::vector<int64_t> convInShape = withPad ? padOutShape : fmapShape;
    const std::vector<int64_t> filterShape = {CO_OUT, cDim, 1, 1};
    const std::vector<int64_t> outShape = {1, CO_OUT, 8, 8};

    auto quantCfg = AscendQuantConfig::Basic(NODE_QUANT, DT_FLOAT, FORMAT_NCHW, fmapShape);
    quantCfg.SetAttr("offset", 2.0F);
    builder.AddAscendQuant(quantCfg, true);
    std::string quantOut = NODE_QUANT;
    if (withPad) {
        builder.AddNode(MakePadNode(fmapShape, padOutShape), {});
        builder.Connect(NODE_QUANT, 0, NODE_PAD, 0);
        quantOut = NODE_PAD;
    }
    auto convCfg = Conv2DConfig::Basic(NODE_CONV, DT_INT8, DT_FLOAT, FORMAT_NCHW, convInShape, filterShape, outShape);
    if (withBias) {
        convCfg.WithBias(DT_INT32, {CO_OUT});
    }
    builder.AddConv2D(convCfg, false, false, withBias);
    builder.Connect(quantOut, 0, NODE_CONV, 0);

    if (caffeWeight) {
        builder.AddData(DataConfig::Basic("caffe_weight", DT_INT8, FORMAT_NCHW, filterShape), false);
        builder.Connect("caffe_weight", 0, NODE_CONV, 1);
    } else {
        builder.AddData(DataConfig::Basic("weight_data", DT_FLOAT, FORMAT_NCHW, filterShape), false);
        builder.AddData(DataConfig::Basic("offset_data", DT_FLOAT, FORMAT_ND, {1}), false);
        builder.AddNode(MakeWeightQuantNode(NODE_WEIGHT_QUANT, filterShape), {});
        builder.Connect("weight_data", 0, NODE_WEIGHT_QUANT, 0);
        builder.Connect("offset_data", 0, NODE_WEIGHT_QUANT, 1);
        builder.Connect(NODE_WEIGHT_QUANT, 0, NODE_CONV, 1);
    }

    builder.AddAscendDequant(AscendDequantConfig::Basic(NODE_DEQUANT, DT_FLOAT, FORMAT_NCHW, outShape, {CO_OUT}), false,
                             false);
    builder.Connect(NODE_CONV, 0, NODE_DEQUANT, 0);
    builder.SetOutput(NODE_DEQUANT);
    return AttachDeqScaleConst(builder.Build(), {CO_OUT});
}

ge::GNodePtr FindNodeByType(const GraphPtr& graph, const std::string& type)
{
    auto allNodes = graph->GetAllNodes();
    for (auto& node : allNodes) {
        ge::AscendString nodeType;
        if (node.GetType(nodeType) == ge::GRAPH_SUCCESS && nodeType.GetString() == type) {
            return std::make_shared<ge::GNode>(node);
        }
    }
    return nullptr;
}
} // namespace

// CheckOpSupported 走 GeUtils::CheckNodeSupportOnAicore（需 GELib+ops-info 初始化），UT 环境不可用，
// 桩化之；isCubeSupported=false 模拟能力不支持场景。
class TestableConv2DQuantProcessFusionPass : public Conv2DQuantProcessFusionPass {
public:
    static inline bool isCubeSupported = true;

protected:
    bool CheckOpSupported(const ge::GNode& node) const override
    {
        (void)node;
        return isCubeSupported;
    }
};

class Conv2DQuantProcessFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        SetSocAscend950();
        TestableConv2DQuantProcessFusionPass::isCubeSupported = true;
    }
    static void TearDownTestCase() {}

    void RunPass(GraphPtr& graph, Status expectRes)
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString("conv2d_quant_before"));
        }
        CustomPassContext passContext;
        passContext.SetPassName(FUSION_NAME.c_str());
        TestableConv2DQuantProcessFusionPass pass;
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString("conv2d_quant_after"));
        }
        EXPECT_EQ(res, expectRes);
    }
};

// 老 Conv2DQuantProcessFusionPassTest1 语义：rollback 主路径（有 bias）
TEST_F(Conv2DQuantProcessFusionPassTest, RollbackMainPath)
{
    auto graph = BuildQuantChain("rollback_main", C_SMALL, false, true);
    RunPass(graph, ge::SUCCESS);
    EXPECT_EQ(FindNodeByType(graph, NODE_QUANT), nullptr);
    EXPECT_EQ(FindNodeByType(graph, NODE_DEQUANT), nullptr);
    EXPECT_NE(FindNodeByType(graph, "QuantWeightRollBack"), nullptr);
    EXPECT_EQ(FindNodeByType(graph, "QuantBiasOptimization"), nullptr);
    auto conv = FindNodeByType(graph, NODE_CONV);
    ASSERT_NE(conv, nullptr);
    TensorDesc outDesc;
    EXPECT_EQ(conv->GetOutputDesc(0, outDesc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), ge::DT_FLOAT16);
    TensorDesc inDesc;
    EXPECT_EQ(conv->GetInputDesc(0, inDesc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inDesc.GetDataType(), ge::DT_FLOAT16);
}

// 老 Conv2DQuantProcessFusionPassTest4 语义：rollback 时 Pad dtype 同步修正
TEST_F(Conv2DQuantProcessFusionPassTest, RollbackWithPadDtypeFix)
{
    auto graph = BuildQuantChain("rollback_pad", C_SMALL, true, false);
    RunPass(graph, ge::SUCCESS);
    EXPECT_EQ(FindNodeByType(graph, NODE_QUANT), nullptr);
    EXPECT_EQ(FindNodeByType(graph, NODE_DEQUANT), nullptr);
    auto pad = FindNodeByType(graph, NODE_PAD);
    ASSERT_NE(pad, nullptr);
    TensorDesc padInDesc;
    TensorDesc padOutDesc;
    EXPECT_EQ(pad->GetInputDesc(0, padInDesc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pad->GetOutputDesc(0, padOutDesc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(padInDesc.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(padOutDesc.GetDataType(), ge::DT_FLOAT16);
}

// 老 GetCoValueTest 语义（经融合级覆盖）：无 bias rollback 不造零 bias（bias 只在 BIAS_OPTIMIZE 造）
TEST_F(Conv2DQuantProcessFusionPassTest, RollbackNoBias)
{
    auto graph = BuildQuantChain("rollback_nobias", C_SMALL, false, false);
    RunPass(graph, ge::SUCCESS);
    EXPECT_EQ(FindNodeByType(graph, NODE_QUANT), nullptr);
    EXPECT_NE(FindNodeByType(graph, NODE_CONV), nullptr);
}

// 大通道（C > 16）走 bias 优化：quant/dequant 保留 + QuantBiasOptimization + 按 Co 造零 bias
TEST_F(Conv2DQuantProcessFusionPassTest, BiasOptimizeMainPath)
{
    auto graph = BuildQuantChain("bias_optimize", C_LARGE, false, false);
    RunPass(graph, ge::SUCCESS);
    EXPECT_NE(FindNodeByType(graph, "QuantBiasOptimization"), nullptr);
    EXPECT_NE(FindNodeByType(graph, NODE_QUANT), nullptr);
    EXPECT_NE(FindNodeByType(graph, NODE_DEQUANT), nullptr);
    auto conv = FindNodeByType(graph, NODE_CONV);
    ASSERT_NE(conv, nullptr);
    TensorDesc biasDesc;
    ASSERT_EQ(conv->GetInputDesc(2, biasDesc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(biasDesc.GetShape().GetDim(0), CO_OUT);
}

// 拒绝：C 维 unknown 跳过融合
TEST_F(Conv2DQuantProcessFusionPassTest, RejectUnknownShapeC)
{
    TestGraph builder("unknown_c");
    builder.SetSocAscend950();
    const std::vector<int64_t> unknownShape = {1, -1, 8, 8};
    const std::vector<int64_t> filterShape = {CO_OUT, C_SMALL, 1, 1};
    builder.AddAscendQuant(AscendQuantConfig::Basic(NODE_QUANT, DT_FLOAT, FORMAT_NCHW, unknownShape), true);
    auto convCfg = Conv2DConfig::Basic(NODE_CONV, DT_INT8, DT_FLOAT, FORMAT_NCHW, unknownShape, filterShape,
                                       {1, CO_OUT, 8, 8});
    builder.AddConv2D(convCfg, false, true, false);
    builder.Connect(NODE_QUANT, 0, NODE_CONV, 0);
    builder.AddAscendDequant(
        AscendDequantConfig::Basic(NODE_DEQUANT, DT_FLOAT, FORMAT_NCHW, {1, CO_OUT, 8, 8}, {CO_OUT}), false, false);
    builder.Connect(NODE_CONV, 0, NODE_DEQUANT, 0);
    builder.SetOutput(NODE_DEQUANT);
    auto graph = AttachDeqScaleConst(builder.Build(), {CO_OUT});
    RunPass(graph, static_cast<Status>(CONV_NOT_CHANGED));
    EXPECT_NE(FindNodeByType(graph, NODE_QUANT), nullptr);
    EXPECT_NE(FindNodeByType(graph, NODE_DEQUANT), nullptr);
}

// 可重入：级联两个子图验证 InitMember
TEST_F(Conv2DQuantProcessFusionPassTest, ReentrantTwoChains)
{
    auto graph1 = BuildQuantChain("chain1", C_SMALL, false, true);
    auto graph2 = BuildQuantChain("chain2", C_SMALL, false, true);
    RunPass(graph1, ge::SUCCESS);
    RunPass(graph2, ge::SUCCESS);
    EXPECT_EQ(FindNodeByType(graph1, NODE_QUANT), nullptr);
    EXPECT_EQ(FindNodeByType(graph2, NODE_QUANT), nullptr);
    EXPECT_EQ(FindNodeByType(graph1, NODE_DEQUANT), nullptr);
    EXPECT_EQ(FindNodeByType(graph2, NODE_DEQUANT), nullptr);
}

// caffe 权重链（weight 直连 conv，无 AscendWeightQuant）：rollback 正常、WeightQuant 不存在
TEST_F(Conv2DQuantProcessFusionPassTest, RollbackCaffeWeightChain)
{
    auto graph = BuildQuantChain("caffe", C_SMALL, false, true, true);
    RunPass(graph, ge::SUCCESS);
    EXPECT_EQ(FindNodeByType(graph, NODE_QUANT), nullptr);
    EXPECT_EQ(FindNodeByType(graph, NODE_DEQUANT), nullptr);
    EXPECT_EQ(FindNodeByType(graph, "AscendWeightQuant"), nullptr);
    EXPECT_NE(FindNodeByType(graph, "QuantWeightRollBack"), nullptr);
}

// quant 多消费者：仅断本链、quant 节点保留
TEST_F(Conv2DQuantProcessFusionPassTest, RollbackQuantMultiConsumer)
{
    TestGraph builder("quant_multi");
    builder.SetSocAscend950();
    const std::vector<int64_t> fmapShape = {1, C_SMALL, 8, 8};
    const std::vector<int64_t> filterShape = {CO_OUT, C_SMALL, 1, 1};
    builder.AddAscendQuant(AscendQuantConfig::Basic(NODE_QUANT, DT_FLOAT, FORMAT_NCHW, fmapShape), true);
    auto convCfg = Conv2DConfig::Basic(NODE_CONV, DT_INT8, DT_FLOAT, FORMAT_NCHW, fmapShape, filterShape,
                                       {1, CO_OUT, 8, 8});
    builder.AddConv2D(convCfg, false, true, false);
    builder.Connect(NODE_QUANT, 0, NODE_CONV, 0);
    builder.AddNode(MakeWeightQuantNode(NODE_WEIGHT_QUANT, filterShape), {});
    builder.Connect(NODE_WEIGHT_QUANT, 0, NODE_CONV, 1);
    builder.AddRelu(ReluConfig::Basic("QuantOtherConsumer", DT_INT8, FORMAT_NCHW, fmapShape), false);
    builder.Connect(NODE_QUANT, 0, "QuantOtherConsumer", 0);
    builder.AddAscendDequant(
        AscendDequantConfig::Basic(NODE_DEQUANT, DT_FLOAT, FORMAT_NCHW, {1, CO_OUT, 8, 8}, {CO_OUT}), false, false);
    builder.Connect(NODE_CONV, 0, NODE_DEQUANT, 0);
    builder.SetOutput(NODE_DEQUANT);
    builder.SetOutput("QuantOtherConsumer");
    auto graph = AttachDeqScaleConst(builder.Build(), {CO_OUT});
    RunPass(graph, ge::SUCCESS);
    EXPECT_NE(FindNodeByType(graph, NODE_QUANT), nullptr);
    EXPECT_EQ(FindNodeByType(graph, NODE_DEQUANT), nullptr);
    auto conv = FindNodeByType(graph, NODE_CONV);
    ASSERT_NE(conv, nullptr);
    ge::TensorDesc inDesc;
    EXPECT_EQ(conv->GetInputDesc(0, inDesc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inDesc.GetDataType(), ge::DT_FLOAT16);
}

// dequant 多消费者：dequant 摘除、全部消费者重定向至 conv
TEST_F(Conv2DQuantProcessFusionPassTest, RollbackDequantMultiConsumer)
{
    TestGraph builder("dequant_multi");
    builder.SetSocAscend950();
    const std::vector<int64_t> fmapShape = {1, C_SMALL, 8, 8};
    const std::vector<int64_t> filterShape = {CO_OUT, C_SMALL, 1, 1};
    builder.AddAscendQuant(AscendQuantConfig::Basic(NODE_QUANT, DT_FLOAT, FORMAT_NCHW, fmapShape), true);
    auto convCfg = Conv2DConfig::Basic(NODE_CONV, DT_INT8, DT_FLOAT, FORMAT_NCHW, fmapShape, filterShape,
                                       {1, CO_OUT, 8, 8});
    builder.AddConv2D(convCfg, false, true, false);
    builder.Connect(NODE_QUANT, 0, NODE_CONV, 0);
    builder.AddNode(MakeWeightQuantNode(NODE_WEIGHT_QUANT, filterShape), {});
    builder.Connect(NODE_WEIGHT_QUANT, 0, NODE_CONV, 1);
    builder.AddAscendDequant(
        AscendDequantConfig::Basic(NODE_DEQUANT, DT_FLOAT, FORMAT_NCHW, {1, CO_OUT, 8, 8}, {CO_OUT}), false, false);
    builder.Connect(NODE_CONV, 0, NODE_DEQUANT, 0);
    builder.AddRelu(ReluConfig::Basic("Sink0", DT_FLOAT, FORMAT_NCHW, {1, CO_OUT, 8, 8}), false);
    builder.AddRelu(ReluConfig::Basic("Sink1", DT_FLOAT, FORMAT_NCHW, {1, CO_OUT, 8, 8}), false);
    builder.Connect(NODE_DEQUANT, 0, "Sink0", 0);
    builder.Connect(NODE_DEQUANT, 0, "Sink1", 0);
    builder.SetOutput("Sink0");
    builder.SetOutput("Sink1");
    auto graph = AttachDeqScaleConst(builder.Build(), {CO_OUT});
    RunPass(graph, ge::SUCCESS);
    EXPECT_EQ(FindNodeByType(graph, NODE_QUANT), nullptr);
    EXPECT_EQ(FindNodeByType(graph, NODE_DEQUANT), nullptr);
    EXPECT_NE(FindNodeByType(graph, "Relu"), nullptr);
}

// 能力不支持（CheckOpSupported false）仍走 rollback（老语义）
TEST_F(Conv2DQuantProcessFusionPassTest, RollbackNonSupportSoc)
{
    TestableConv2DQuantProcessFusionPass::isCubeSupported = false;
    auto graph = BuildQuantChain("soc_notsupport", C_SMALL, false, true);
    RunPass(graph, ge::SUCCESS);
    EXPECT_EQ(FindNodeByType(graph, NODE_QUANT), nullptr);
    EXPECT_EQ(FindNodeByType(graph, NODE_DEQUANT), nullptr);
    TestableConv2DQuantProcessFusionPass::isCubeSupported = true;
}

// 拒绝：deq_scale 非 1-D（bias optimize 路径）
TEST_F(Conv2DQuantProcessFusionPassTest, RejectDeqScaleNot1D)
{
    TestGraph builder("deq_scale_2d");
    builder.SetSocAscend950();
    const std::vector<int64_t> fmapShape = {1, C_LARGE, 8, 8};
    const std::vector<int64_t> filterShape = {CO_OUT, C_LARGE, 1, 1};
    builder.AddAscendQuant(AscendQuantConfig::Basic(NODE_QUANT, DT_FLOAT, FORMAT_NCHW, fmapShape), true);
    auto convCfg = Conv2DConfig::Basic(NODE_CONV, DT_INT8, DT_FLOAT, FORMAT_NCHW, fmapShape, filterShape,
                                       {1, CO_OUT, 8, 8});
    builder.AddConv2D(convCfg, false, true, false);
    builder.Connect(NODE_QUANT, 0, NODE_CONV, 0);
    builder.AddNode(MakeWeightQuantNode(NODE_WEIGHT_QUANT, filterShape), {});
    builder.Connect(NODE_WEIGHT_QUANT, 0, NODE_CONV, 1);
    builder.AddAscendDequant(
        AscendDequantConfig::Basic(NODE_DEQUANT, DT_FLOAT, FORMAT_NCHW, {1, CO_OUT, 8, 8}, {1, CO_OUT}), false, false);
    builder.Connect(NODE_CONV, 0, NODE_DEQUANT, 0);
    builder.SetOutput(NODE_DEQUANT);
    auto graph = AttachDeqScaleConst(builder.Build(), {1, CO_OUT});
    RunPass(graph, ge::FAILED);
    EXPECT_NE(FindNodeByType(graph, NODE_QUANT), nullptr);
}

// 属性断言：groups 透传至 AscendWeightQuant；host op 的 offset/dst_type/groups
TEST_F(Conv2DQuantProcessFusionPassTest, BiasOptimizeAttrVerify)
{
    auto graph = BuildQuantChain("attr_verify", C_LARGE, false, false);
    RunPass(graph, ge::SUCCESS);
    auto weightQuant = FindNodeByType(graph, "AscendWeightQuant");
    ASSERT_NE(weightQuant, nullptr);
    int64_t weightQuantGroups = -1;
    EXPECT_EQ(weightQuant->GetAttr(ge::AscendString("groups"), weightQuantGroups), ge::GRAPH_SUCCESS);
    EXPECT_EQ(weightQuantGroups, 1);
    auto hostOp = FindNodeByType(graph, "QuantBiasOptimization");
    ASSERT_NE(hostOp, nullptr);
    ge::float32_t hostOffset = -1.0F;
    int64_t hostDstType = -1;
    int64_t hostGroups = -1;
    EXPECT_EQ(hostOp->GetAttr(ge::AscendString("offset"), hostOffset), ge::GRAPH_SUCCESS);
    EXPECT_EQ(hostOffset, 2.0F);
    EXPECT_EQ(hostOp->GetAttr(ge::AscendString("dst_type"), hostDstType), ge::GRAPH_SUCCESS);
    EXPECT_EQ(hostDstType, static_cast<int64_t>(ge::DT_INT8));
    EXPECT_EQ(hostOp->GetAttr(ge::AscendString("groups"), hostGroups), ge::GRAPH_SUCCESS);
    EXPECT_EQ(hostGroups, 1);
    auto conv = graph->FindNodeByName(ge::AscendString(NODE_CONV.c_str()));
    ASSERT_NE(conv, nullptr);
    int64_t offsetX = -1;
    EXPECT_EQ(conv->GetAttr(ge::AscendString("offset_x"), offsetX), ge::GRAPH_SUCCESS);
    EXPECT_EQ(offsetX, 2);
}

// Tiny 产品（Intrinsic_set_l1 含 f16 无 f32）：C<=16 也强制 BIAS_OPTIMIZE（老 isTiny 语义）
TEST_F(Conv2DQuantProcessFusionPassTest, TinyProductForcesBiasOptimize)
{
    auto graph = BuildQuantChain("tiny", C_SMALL, false, true);
    SetSoc("Ascend950", true);
    RunPass(graph, ge::SUCCESS);
    EXPECT_NE(FindNodeByType(graph, "QuantBiasOptimization"), nullptr);
    EXPECT_NE(FindNodeByType(graph, NODE_QUANT), nullptr);
    EXPECT_NE(FindNodeByType(graph, NODE_DEQUANT), nullptr);
    SetSocAscend950();
}

// 属性断言：rollback 时 host op 透传 quant 的 scale（offset 移植仅 BIAS_OPTIMIZE 场景）
TEST_F(Conv2DQuantProcessFusionPassTest, RollbackHostOpScaleVerify)
{
    auto graph = BuildQuantChain("host_scale", C_SMALL, false, true);
    RunPass(graph, ge::SUCCESS);
    auto hostOp = FindNodeByType(graph, "QuantWeightRollBack");
    ASSERT_NE(hostOp, nullptr);
    ge::float32_t hostScale = -1.0F;
    EXPECT_EQ(hostOp->GetAttr(ge::AscendString("scale"), hostScale), ge::GRAPH_SUCCESS);
    EXPECT_EQ(hostScale, 1.0F);
}
} // namespace Ops
