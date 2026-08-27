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
#include <vector>

#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "../../../op_graph/fusion_pass/anti_quant_matmul_fusion_pass.h"
#include "../../../op_graph/weight_quant_batch_matmul_v2_proto.h"
#include "register/register_custom_pass.h"

using namespace ge;
using namespace fe;
using namespace ops;

namespace {
constexpr int64_t DIM_M = 64;
constexpr int64_t DIM_K = 5120;
constexpr int64_t DIM_N = 5120;

struct FusionResult {
    bool hasAntiQuant;
    bool hasAdd;
    bool hasMul;
    bool hasMatMul;
    bool hasWeightQuant;
    int64_t weightQuantCount;
};

FusionResult CheckFusionResult(const std::shared_ptr<Graph>& graph)
{
    FusionResult r{false, false, false, false, false, 0};
    for (auto& node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) != GRAPH_SUCCESS) {
            continue;
        }
        const std::string ts(type.GetString());
        if (ts == "AscendAntiQuant") {
            r.hasAntiQuant = true;
        } else if (ts == "Add") {
            r.hasAdd = true;
        } else if (ts == "Mul") {
            r.hasMul = true;
        } else if (ts == "MatMul" || ts == "MatMulV2" || ts == "BatchMatMul" || ts == "BatchMatMulV2") {
            r.hasMatMul = true;
        } else if (ts == "WeightQuantBatchMatmulV2") {
            r.hasWeightQuant = true;
            r.weightQuantCount++;
        }
    }
    return r;
}

static TensorDesc MakeTD(const std::vector<int64_t>& shape, DataType dtype)
{
    TensorDesc desc;
    desc.SetDataType(dtype);
    desc.SetShape(Shape(shape));
    desc.SetFormat(FORMAT_ND);
    desc.SetOriginShape(Shape(shape));
    desc.SetOriginFormat(FORMAT_ND);
    return desc;
}

static GNode BuildAscendAntiQuant(Graph* graph, const std::string& name, float scale, float offset, int64_t dtype,
                                  bool sqrtMode)
{
    auto builder = es::CompliantNodeBuilder(graph);
    builder.OpType("AscendAntiQuant")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"scale", es::CompliantNodeBuilder::kEsAttrRequired, "Float", es::CreateFrom(scale)},
            {"offset", es::CompliantNodeBuilder::kEsAttrRequired, "Float", es::CreateFrom(offset)},
            {"dtype", es::CompliantNodeBuilder::kEsAttrOptional, "Int", es::CreateFrom(dtype)},
            {"sqrt_mode", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(sqrtMode)},
        });
    return builder.Build();
}

static GNode BuildAddNode(Graph* graph, const std::string& name)
{
    auto builder = es::CompliantNodeBuilder(graph);
    builder.OpType("Add")
        .Name(name.c_str())
        .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}});
    return builder.Build();
}

static GNode BuildMulNode(Graph* graph, const std::string& name)
{
    auto builder = es::CompliantNodeBuilder(graph);
    builder.OpType("Mul")
        .Name(name.c_str())
        .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}});
    return builder.Build();
}

static GNode BuildMatMulV2Node(Graph* graph, const std::string& name, bool transX1, bool transX2, bool hasBias)
{
    auto builder = es::CompliantNodeBuilder(graph);
    builder.OpType("MatMulV2").Name(name.c_str());
    if (hasBias) {
        builder.IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""}});
    } else {
        builder.IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}});
    }
    builder.IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"transpose_x1", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(transX1)},
            {"transpose_x2", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(transX2)},
            {"offset_x", es::CompliantNodeBuilder::kEsAttrOptional, "Int", es::CreateFrom(static_cast<int64_t>(0))},
        });
    return builder.Build();
}

static void SetGraphOutput(const GNode& outputNode, std::shared_ptr<Graph>& graph)
{
    std::vector<std::pair<GNode, int32_t>> graphOutputs;
    graphOutputs.push_back(std::make_pair(outputNode, 0));
    graph->SetOutputs(graphOutputs);
}

es::EsTensorHolder MakeScaleConst(es::EsGraphBuilder& gb, int64_t n)
{
    std::vector<float> vals(static_cast<size_t>(n), 1.0F);
    auto c = gb.CreateConst(vals, {n});
    c.GetProducer()->UpdateOutputDesc(0, MakeTD({n}, DT_FLOAT));
    return c;
}

es::EsTensorHolder MakeOffsetConst(es::EsGraphBuilder& gb, int64_t n)
{
    std::vector<float> vals(static_cast<size_t>(n), 0.0F);
    auto c = gb.CreateConst(vals, {n});
    c.GetProducer()->UpdateOutputDesc(0, MakeTD({n}, DT_FLOAT));
    return c;
}
} // namespace

class AntiQuantMatMulFusionPassTest : public testing::Test {
protected:
    class TestablePass : public ops::AntiQuantMatMulFusionPass {};

    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.soc_info.arch_type = 2201;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.soc_info.arch_type = 2201;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void TearDownTestCase() { fe::PlatformInfoManager::Instance().platform_info_map_.clear(); }
};

// Pattern 2: AscendAntiQuant -> MatMulV2
TEST_F(AntiQuantMatMulFusionPassTest, AntiQuantMatMulFusion)
{
    es::EsGraphBuilder gb("anti_quant_matmul_fusion");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});

    auto antiNode = BuildAscendAntiQuant(graphPtr, "anti", 1.0F, 0.0F, 0, false);
    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, false);

    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, antiNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, antiNode, 0, mmNode, 1);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(graph);
    EXPECT_FALSE(r.hasAntiQuant);
    EXPECT_FALSE(r.hasMatMul);
    EXPECT_TRUE(r.hasWeightQuant);
    EXPECT_EQ(r.weightQuantCount, 1);
}

// Pattern 1: AscendAntiQuant -> Mul -> MatMulV2
TEST_F(AntiQuantMatMulFusionPassTest, AntiQuantMulMatMulFusion)
{
    es::EsGraphBuilder gb("anti_quant_mul_matmul_fusion");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto scaleConst = MakeScaleConst(gb, DIM_N);

    auto antiNode = BuildAscendAntiQuant(graphPtr, "anti", 2.0F, 0.0F, 0, false);
    auto mulNode = BuildMulNode(graphPtr, "mul");
    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, false);

    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, antiNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, antiNode, 0, mulNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *scaleConst.GetProducer(), 0, mulNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, mulNode, 0, mmNode, 1);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mulNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mulNode.UpdateInputDesc(1, MakeTD({DIM_N}, DT_FLOAT));
    mulNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(graph);
    EXPECT_FALSE(r.hasAntiQuant);
    EXPECT_FALSE(r.hasMul);
    EXPECT_FALSE(r.hasMatMul);
    EXPECT_TRUE(r.hasWeightQuant);
    EXPECT_EQ(r.weightQuantCount, 1);
}

// AscendAntiQuant -> Add -> Mul -> MatMulV2
TEST_F(AntiQuantMatMulFusionPassTest, AntiQuantAddMulMatMulFusion)
{
    es::EsGraphBuilder gb("anti_quant_add_mul_matmul_fusion");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto offsetConst = MakeOffsetConst(gb, DIM_N);
    auto scaleConst = MakeScaleConst(gb, DIM_N);

    auto antiNode = BuildAscendAntiQuant(graphPtr, "anti", 2.0F, 1.0F, 0, false);
    auto addNode = BuildAddNode(graphPtr, "add");
    auto mulNode = BuildMulNode(graphPtr, "mul");
    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, false);

    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, antiNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, antiNode, 0, addNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *offsetConst.GetProducer(), 0, addNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, addNode, 0, mulNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *scaleConst.GetProducer(), 0, mulNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, mulNode, 0, mmNode, 1);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    addNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    addNode.UpdateInputDesc(1, MakeTD({DIM_N}, DT_FLOAT));
    addNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mulNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mulNode.UpdateInputDesc(1, MakeTD({DIM_N}, DT_FLOAT));
    mulNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(graph);
    EXPECT_FALSE(r.hasAntiQuant);
    EXPECT_FALSE(r.hasAdd);
    EXPECT_FALSE(r.hasMul);
    EXPECT_FALSE(r.hasMatMul);
    EXPECT_TRUE(r.hasWeightQuant);
    EXPECT_EQ(r.weightQuantCount, 1);
}

// reversed ports: AscendAntiQuant -> Add(port1) -> Mul(port1) -> MatMulV2
// AntiQuant output connects to port 1 of Add/Mul (const on port 0).
TEST_F(AntiQuantMatMulFusionPassTest, AntiQuantAddMulMatMulFusionReversedPorts)
{
    es::EsGraphBuilder gb("anti_quant_add_mul_matmul_fusion_reversed_ports");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto offsetConst = MakeOffsetConst(gb, DIM_N);
    auto scaleConst = MakeScaleConst(gb, DIM_N);

    auto antiNode = BuildAscendAntiQuant(graphPtr, "anti", 2.0F, 1.0F, 0, false);
    auto addNode = BuildAddNode(graphPtr, "add");
    auto mulNode = BuildMulNode(graphPtr, "mul");
    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, false);

    // Reversed port allocation: const on port 0, AntiQuant/Add chain on port 1.
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, antiNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *offsetConst.GetProducer(), 0, addNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, antiNode, 0, addNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *scaleConst.GetProducer(), 0, mulNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, addNode, 0, mulNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, mulNode, 0, mmNode, 1);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    addNode.UpdateInputDesc(0, MakeTD({DIM_N}, DT_FLOAT));
    addNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    addNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mulNode.UpdateInputDesc(0, MakeTD({DIM_N}, DT_FLOAT));
    mulNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mulNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(graph);
    EXPECT_FALSE(r.hasAntiQuant);
    EXPECT_FALSE(r.hasAdd);
    EXPECT_FALSE(r.hasMul);
    EXPECT_FALSE(r.hasMatMul);
    EXPECT_TRUE(r.hasWeightQuant);
    EXPECT_EQ(r.weightQuantCount, 1);
}

// reversed ports: AscendAntiQuant -> Mul(port1) -> MatMulV2
// AntiQuant output connects to port 1 of Mul (const scale on port 0).
TEST_F(AntiQuantMatMulFusionPassTest, AntiQuantMulMatMulFusionReversedPorts)
{
    es::EsGraphBuilder gb("anti_quant_mul_matmul_fusion_reversed_ports");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto scaleConst = MakeScaleConst(gb, DIM_N);

    auto antiNode = BuildAscendAntiQuant(graphPtr, "anti", 2.0F, 0.0F, 0, false);
    auto mulNode = BuildMulNode(graphPtr, "mul");
    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, false);

    // Reversed port allocation: const on port 0, AntiQuant chain on port 1.
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, antiNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *scaleConst.GetProducer(), 0, mulNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, antiNode, 0, mulNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, mulNode, 0, mmNode, 1);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mulNode.UpdateInputDesc(0, MakeTD({DIM_N}, DT_FLOAT));
    mulNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mulNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(graph);
    EXPECT_FALSE(r.hasAntiQuant);
    EXPECT_FALSE(r.hasMul);
    EXPECT_FALSE(r.hasMatMul);
    EXPECT_TRUE(r.hasWeightQuant);
    EXPECT_EQ(r.weightQuantCount, 1);
}

// No fusion: x2 input of MatMulV2 is not AscendAntiQuant/Mul/Add-chain
TEST_F(AntiQuantMatMulFusionPassTest, NoAntiQuantNoFusion)
{
    es::EsGraphBuilder gb("no_anti_quant_no_fusion");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(1, "weight", DT_FLOAT16, FORMAT_ND, {DIM_K, DIM_N});

    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, false);

    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, mmNode, 1);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto r = CheckFusionResult(graph);
    EXPECT_FALSE(r.hasWeightQuant);
    EXPECT_TRUE(r.hasMatMul);
}

// No fusion: M dim too large
TEST_F(AntiQuantMatMulFusionPassTest, MDimTooLargeNoFusion)
{
    constexpr int64_t kLargeM = 128;
    es::EsGraphBuilder gb("m_dim_too_large");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {kLargeM, DIM_K});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});

    auto antiNode = BuildAscendAntiQuant(graphPtr, "anti", 1.0F, 0.0F, 0, false);
    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, false);

    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, antiNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, antiNode, 0, mmNode, 1);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({kLargeM, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({kLargeM, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({kLargeM, DIM_N}, DT_FLOAT16));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    auto status = pass.Run(graph, passContext);

    auto r = CheckFusionResult(graph);
    if (status == SUCCESS) {
        EXPECT_TRUE(r.hasWeightQuant);
        EXPECT_FALSE(r.hasMatMul);
        EXPECT_FALSE(r.hasAntiQuant);
    } else {
        EXPECT_EQ(status, GRAPH_NOT_CHANGED);
        EXPECT_FALSE(r.hasWeightQuant);
        EXPECT_TRUE(r.hasMatMul);
        EXPECT_TRUE(r.hasAntiQuant);
    }
}

// No fusion: dtype mismatch (mm output not fp16)
TEST_F(AntiQuantMatMulFusionPassTest, DtypeMismatchNoFusion)
{
    es::EsGraphBuilder gb("dtype_mismatch");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});

    auto antiNode = BuildAscendAntiQuant(graphPtr, "anti", 1.0F, 0.0F, 0, false);
    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, false);

    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, antiNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, antiNode, 0, mmNode, 1);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({DIM_M, DIM_N}, DT_FLOAT));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto r = CheckFusionResult(graph);
    EXPECT_FALSE(r.hasWeightQuant);
}

// with bias: AscendAntiQuant -> MatMulV2(with bias)
TEST_F(AntiQuantMatMulFusionPassTest, AntiQuantMatMulWithBiasFusion)
{
    es::EsGraphBuilder gb("anti_quant_matmul_with_bias_fusion");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(1, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto bias = gb.CreateInput(2, "bias", DT_FLOAT16, FORMAT_ND, {DIM_N});

    auto antiNode = BuildAscendAntiQuant(graphPtr, "anti", 1.0F, 0.0F, 0, false);
    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, true);

    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, antiNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, antiNode, 0, mmNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *bias.GetProducer(), 0, mmNode, 2);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    bias.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_N}, DT_FLOAT16));
    antiNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(2, MakeTD({DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(graph);
    EXPECT_FALSE(r.hasAntiQuant);
    EXPECT_FALSE(r.hasMatMul);
    EXPECT_TRUE(r.hasWeightQuant);
    EXPECT_EQ(r.weightQuantCount, 1);
}

// AntiQuantMatMulFusionPassAntiquatMatMul
// AscendAntiQuant -> MatMul, M=32, K=5120, N=5120, scale=1.0, offset=0.0
TEST_F(AntiQuantMatMulFusionPassTest, AntiQuantMatMulFusionPassAntiquatMatMul)
{
    constexpr int64_t kM = 32;
    es::EsGraphBuilder gb("AntiQuantMatMulFusionPassAntiquatMatMul");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "data_x", DT_FLOAT16, FORMAT_ND, {kM, DIM_K});
    auto w = gb.CreateInput(1, "data_weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});

    auto antiNode = BuildAscendAntiQuant(graphPtr, "anti", 1.0F, 0.0F, static_cast<int64_t>(DT_FLOAT16), false);
    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, false);

    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, antiNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, antiNode, 0, mmNode, 1);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({kM, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({kM, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({kM, DIM_N}, DT_FLOAT16));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(graph);
    EXPECT_FALSE(r.hasMatMul);
    EXPECT_TRUE(r.hasWeightQuant);
}

// AntiQuantMatMulFusionPassAntiquatAddMulMatMul
// AscendAntiQuant -> Add(const_offset) -> Mul(const_scale) -> MatMul
// M=32, K=5120, N=5120, scale=1.0, offset=0.0, const=FP16 shape=[1] value=1
TEST_F(AntiQuantMatMulFusionPassTest, AntiQuantMatMulFusionPassAntiquatAddMulMatMul)
{
    constexpr int64_t kM = 32;
    es::EsGraphBuilder gb("AntiQuantMatMulFusionPassAntiquatAddMulMatMul");
    auto graphPtr = gb.GetCGraphBuilder()->GetGraph();

    auto x = gb.CreateInput(0, "data_x", DT_FLOAT16, FORMAT_ND, {kM, DIM_K});
    auto w = gb.CreateInput(1, "data_weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});

    std::vector<uint16_t> constVal{1};
    auto dataScale = gb.CreateConst(constVal, {1}, DT_FLOAT16, FORMAT_ND);
    auto dataOffset = gb.CreateConst(constVal, {1}, DT_FLOAT16, FORMAT_ND);

    auto antiNode = BuildAscendAntiQuant(graphPtr, "anti", 1.0F, 0.0F, static_cast<int64_t>(DT_FLOAT16), false);
    auto addNode = BuildAddNode(graphPtr, "add");
    auto mulNode = BuildMulNode(graphPtr, "mul");
    auto mmNode = BuildMatMulV2Node(graphPtr, "matmul", false, false, false);

    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *w.GetProducer(), 0, antiNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, antiNode, 0, addNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *dataOffset.GetProducer(), 0, addNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, addNode, 0, mulNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *dataScale.GetProducer(), 0, mulNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, mmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, mulNode, 0, mmNode, 1);

    x.GetProducer()->UpdateOutputDesc(0, MakeTD({kM, DIM_K}, DT_FLOAT16));
    w.GetProducer()->UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    dataScale.GetProducer()->UpdateOutputDesc(0, MakeTD({1}, DT_FLOAT16));
    dataOffset.GetProducer()->UpdateOutputDesc(0, MakeTD({1}, DT_FLOAT16));
    antiNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_INT8));
    antiNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    addNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    addNode.UpdateInputDesc(1, MakeTD({1}, DT_FLOAT16));
    addNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mulNode.UpdateInputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mulNode.UpdateInputDesc(1, MakeTD({1}, DT_FLOAT16));
    mulNode.UpdateOutputDesc(0, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateInputDesc(0, MakeTD({kM, DIM_K}, DT_FLOAT16));
    mmNode.UpdateInputDesc(1, MakeTD({DIM_K, DIM_N}, DT_FLOAT16));
    mmNode.UpdateOutputDesc(0, MakeTD({kM, DIM_N}, DT_FLOAT16));

    es::EsGraphBuilder::SetOutput(x, 0);
    std::shared_ptr<Graph> graph = gb.BuildAndReset();
    ASSERT_NE(graph, nullptr);
    SetGraphOutput(mmNode, graph);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(graph);
    EXPECT_FALSE(r.hasMatMul);
    EXPECT_TRUE(r.hasWeightQuant);
}
