/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/batch_matmul_to_mul_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace {

constexpr char kPassName[] = "BatchMatMul2MulFusionPass";

void SetPlatformInfo910B()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.ai_core_spec.ub_size = 192 * 1024;
    platformInfo.ai_core_spec.l0_a_size = 64 * 1024;
    platformInfo.ai_core_spec.l0_b_size = 64 * 1024;
    platformInfo.ai_core_spec.l0_c_size = 256 * 1024;
    platformInfo.ai_core_spec.l1_size = 512 * 1024;
    optionalInfo.soc_version = "Ascend910B1";
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_nd2nz"] = {"float16"};
    platformInfo.str_info.short_soc_version = "Ascend910B";
    PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

void SetPlatformInfo91095()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.ai_core_spec.ub_size = 192 * 1024;
    platformInfo.ai_core_spec.l0_a_size = 64 * 1024;
    platformInfo.ai_core_spec.l0_b_size = 64 * 1024;
    platformInfo.ai_core_spec.l0_c_size = 1024 * 1024;
    platformInfo.ai_core_spec.l1_size = 1024 * 1024;
    optionalInfo.soc_version = "Ascend910_95";
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_nd2nz"] = {"float16"};
    platformInfo.str_info.short_soc_version = "Ascend950";
    PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

TensorDesc MakeTensorDesc(const std::vector<int64_t>& dims, DataType dtype, Format format = FORMAT_ND)
{
    TensorDesc desc(ge::Shape(dims), format, dtype);
    desc.SetOriginFormat(format);
    desc.SetOriginShape(ge::Shape(dims));
    return desc;
}

int CountNodes(const std::shared_ptr<Graph>& graph, const char* nodeType)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == nodeType) {
            count++;
        }
    }
    return count;
}

std::shared_ptr<Graph> BuildMatMulLikeGraph(const std::string& name, const char* opType,
                                            const std::vector<int64_t>& aDims, const std::vector<int64_t>& bDims,
                                            const std::vector<int64_t>& outDims, DataType dtype, bool transX1,
                                            bool transX2)
{
    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc(aDims, dtype);
    auto x2Desc = MakeTensorDesc(bDims, dtype);
    auto outDesc = MakeTensorDesc(outDims, dtype);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", dtype, FORMAT_ND, aDims);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", dtype, FORMAT_ND, bDims);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    bool isBatch = (strcmp(opType, "BatchMatMul") == 0 || strcmp(opType, "BatchMatMulV2") == 0);
    const char* transAttr1 = isBatch ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = isBatch ? "adj_x2" : "transpose_x2";

    // IR input definitions: x1, x2 required; bias optional (all matmul types have optional bias in IR)
    std::vector<CompliantNodeBuilder::IrInputDef> irInputs = {
        {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
    };
    bool irHasOffsetW = (strcmp(opType, "MatMulV2") == 0 || strcmp(opType, "BatchMatMulV2") == 0);
    if (irHasOffsetW) {
        irInputs.push_back({"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""});
    }

    std::vector<CompliantNodeBuilder::IrAttrDef> irAttrs = {
        {transAttr1, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
        {transAttr2, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
    };
    if (irHasOffsetW) {
        irAttrs.push_back({"offset_x", CompliantNodeBuilder::kEsAttrOptional, "Int", AttrValue()});
    }

    auto node = CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(name.c_str())
                    .IrDefInputs(irInputs)
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs(irAttrs)
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), node, 1);
    node.UpdateInputDesc(0, x1Desc);
    node.UpdateInputDesc(1, x2Desc);
    node.UpdateOutputDesc(0, outDesc);
    bool transX1Val = transX1;
    bool transX2Val = transX2;
    node.SetAttr(transAttr1, transX1Val);
    node.SetAttr(transAttr2, transX2Val);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    return graphBuilder.BuildAndReset({output});
}

std::shared_ptr<Graph> BuildBatchMatMulReduceGraph(const std::string& name, const char* opType,
                                                   const std::vector<int64_t>& aDims, const std::vector<int64_t>& bDims,
                                                   const std::vector<int64_t>& outDims,
                                                   const std::vector<int64_t>& reduceOutDims, DataType dtype,
                                                   bool transX1, bool transX2, bool withCast = false)
{
    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc(aDims, dtype);
    auto x2Desc = MakeTensorDesc(bDims, dtype);
    auto outDesc = MakeTensorDesc(outDims, dtype);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", dtype, FORMAT_ND, aDims);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", dtype, FORMAT_ND, bDims);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    const char* transAttr1 = "adj_x1";
    const char* transAttr2 = "adj_x2";

    auto bmmNode = CompliantNodeBuilder(graph)
                       .OpType(opType)
                       .Name("bmm")
                       .IrDefInputs({
                           {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
                           {"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""},
                       })
                       .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .IrDefAttrs({
                           {transAttr1, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                           {transAttr2, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                           {"offset_x", CompliantNodeBuilder::kEsAttrOptional, "Int", AttrValue()},
                       })
                       .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), bmmNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), bmmNode, 1);
    bmmNode.UpdateInputDesc(0, x1Desc);
    bmmNode.UpdateInputDesc(1, x2Desc);
    bmmNode.UpdateOutputDesc(0, outDesc);
    bool transX1Val = transX1;
    bool transX2Val = transX2;
    bmmNode.SetAttr(transAttr1, transX1Val);
    bmmNode.SetAttr(transAttr2, transX2Val);

    EsTensorHolder nextInput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bmmNode, 0));
    if (withCast) {
        auto castOutDesc = MakeTensorDesc(outDims, DT_FLOAT);
        auto castNode = CompliantNodeBuilder(graph)
                            .OpType("Cast")
                            .Name("cast")
                            .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                            .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .IrDefAttrs({{"dst_type", CompliantNodeBuilder::kEsAttrRequired, "Int",
                                          CreateFrom(static_cast<int64_t>(DT_FLOAT))}})
                            .Build();
        AddEdgeAndUpdatePeerDesc(*graph, bmmNode, 0, castNode, 0);
        castNode.UpdateInputDesc(0, outDesc);
        castNode.UpdateOutputDesc(0, castOutDesc);
        nextInput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(castNode, 0));
    }

    auto reduceOutDesc = MakeTensorDesc(reduceOutDims, withCast ? DT_FLOAT : dtype);
    auto reduceNode = CompliantNodeBuilder(graph)
                          .OpType("ReduceSumD")
                          .Name("reduce")
                          .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs({{"axes", CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                        CreateFrom(std::vector<int64_t>{0})},
                                       {"keep_dims", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)}})
                          .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *nextInput.GetProducer(), nextInput.GetProducerOutIndex(), reduceNode, 0);
    TensorDesc nextOutDesc;
    nextInput.GetProducer()->GetOutputDesc(nextInput.GetProducerOutIndex(), nextOutDesc);
    reduceNode.UpdateInputDesc(0, nextOutDesc);
    reduceNode.UpdateOutputDesc(0, reduceOutDesc);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reduceNode, 0));
    return graphBuilder.BuildAndReset({output});
}

} // namespace

class BatchMatMul2MulFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatformInfo910B(); }

    static void TearDownTestCase() {}

    void SetUp() override { SetPlatformInfo910B(); }

    void TearDown() override {}
};

// ===================== Pattern test =====================

TEST_F(BatchMatMul2MulFusionPassTest, patternTest)
{
    BatchMatMul2MulFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

// ===================== BatchMatMul fusion success on 910B =====================

// adj_x1=true, adj_x2=false -> 1 reshape (x1)
TEST_F(BatchMatMul2MulFusionPassTest, bmmToMulFusionMatched1Success)
{
    auto graph = BuildMatMulLikeGraph("bmmToMulMatched1", "BatchMatMul", {4800, 1, 300}, {4800, 1, 256},
                                      {4800, 300, 256}, DT_FLOAT, true, false);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 1);
    EXPECT_EQ(CountNodes(graph, "BatchMatMul"), 0);
}

// adj_x1=false, adj_x2=true -> 1 reshape (x2)
TEST_F(BatchMatMul2MulFusionPassTest, bmmToMulFusionMatched2Success)
{
    auto graph = BuildMatMulLikeGraph("bmmToMulMatched2", "BatchMatMul", {4800, 300, 1}, {4800, 256, 1},
                                      {4800, 300, 256}, DT_FLOAT, false, true);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 1);
}

// adj_x1=true, adj_x2=true -> 2 reshapes (x1, x2)
TEST_F(BatchMatMul2MulFusionPassTest, bmmToMulFusionMatched3Success)
{
    auto graph = BuildMatMulLikeGraph("bmmToMulMatched3", "BatchMatMul", {4800, 1, 300}, {4800, 256, 1},
                                      {4800, 300, 256}, DT_FLOAT, true, true);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 2);
    EXPECT_EQ(CountNodes(graph, "Mul"), 1);
}

// adj_x1=false, adj_x2=false -> 0 reshapes
TEST_F(BatchMatMul2MulFusionPassTest, bmmToMulFusionMatched4Success)
{
    auto graph = BuildMatMulLikeGraph("bmmToMulMatched4", "BatchMatMul", {4800, 300, 1}, {4800, 1, 256},
                                      {4800, 300, 256}, DT_FLOAT, false, false);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 0);
    EXPECT_EQ(CountNodes(graph, "Mul"), 1);
}

// ===================== MatMul fusion success on 910B =====================

// adj_x1=false, adj_x2=true -> 1 reshape (x2)
TEST_F(BatchMatMul2MulFusionPassTest, mmToMulFusionMatched1Success)
{
    auto graph = BuildMatMulLikeGraph("mmToMulMatched1", "MatMul", {9600, 1}, {448, 1}, {9600, 448}, DT_FLOAT, false,
                                      true);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 1);
    EXPECT_EQ(CountNodes(graph, "MatMul"), 0);
}

// adj_x1=false, adj_x2=false -> 0 reshapes
TEST_F(BatchMatMul2MulFusionPassTest, mmToMulFusionMatched2Success)
{
    auto graph = BuildMatMulLikeGraph("mmToMulMatched2", "MatMul", {9600, 1}, {1, 448}, {9600, 448}, DT_FLOAT, false,
                                      false);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 0);
    EXPECT_EQ(CountNodes(graph, "Mul"), 1);
}

// adj_x1=true, adj_x2=true -> 2 reshapes
TEST_F(BatchMatMul2MulFusionPassTest, mmToMulFusionMatched3Success)
{
    auto graph = BuildMatMulLikeGraph("mmToMulMatched3", "MatMul", {1, 9600}, {448, 1}, {9600, 448}, DT_FLOAT, true,
                                      true);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 2);
    EXPECT_EQ(CountNodes(graph, "Mul"), 1);
}

// ===================== Not matched cases =====================

// k != 1, no fusion
TEST_F(BatchMatMul2MulFusionPassTest, bmmToMulFusionKNot1NotChangedFail)
{
    auto graph = BuildMatMulLikeGraph("bmmToMulKNot1", "BatchMatMul", {16, 16, 16}, {16, 16, 16}, {16, 16, 16},
                                      DT_FLOAT, false, false);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Mul"), 0);
}

// ===================== 91095 platform =====================

TEST_F(BatchMatMul2MulFusionPassTest, bmmToMulFusion91095NotChangedFail)
{
    SetPlatformInfo91095();
    auto graph = BuildMatMulLikeGraph("bmmToMul91095", "BatchMatMul", {4800, 300, 1}, {4800, 1, 256}, {4800, 300, 256},
                                      DT_FLOAT, false, false);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Mul"), 0);
}

// ===================== BatchMatMulV2 + ReduceSumD scenarios (not changed) =====================

// BatchMatMulV2 --> ReduceSumD --> Output; k==1, should be blocked by BatchMatMulReduceFusionCheck
TEST_F(BatchMatMul2MulFusionPassTest, bmmV2ToReduceSumDNotChangedFail)
{
    auto graph = BuildBatchMatMulReduceGraph("bmmV2ToReduceSumD", "BatchMatMulV2", {5120, 512, 1}, {5120, 1, 1024},
                                             {5120, 512, 1024}, {512, 1024}, DT_FLOAT16, false, false);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// BatchMatMulV2 --> ReduceSumD --> Output; k!=1
TEST_F(BatchMatMul2MulFusionPassTest, bmmV2ToReduceSumDKNot1NotChangedFail)
{
    auto graph = BuildBatchMatMulReduceGraph("bmmV2ToReduceSumDKNot1", "BatchMatMulV2", {5120, 35, 29}, {5120, 29, 64},
                                             {5120, 35, 64}, {35, 64}, DT_FLOAT16, false, false);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output
TEST_F(BatchMatMul2MulFusionPassTest, bmmV2ToCastReduceSumDNotChangedFail)
{
    auto graph = BuildBatchMatMulReduceGraph("bmmV2ToCastReduceSumD", "BatchMatMulV2", {5120, 512, 1}, {5120, 1, 1024},
                                             {5120, 512, 1024}, {512, 1024}, DT_FLOAT16, false, false, true);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// ===================== fp16 dtype success =====================

TEST_F(BatchMatMul2MulFusionPassTest, bmmToMulFp16Success)
{
    auto graph = BuildMatMulLikeGraph("bmmToMulFp16", "BatchMatMul", {4800, 300, 1}, {4800, 1, 256}, {4800, 300, 256},
                                      DT_FLOAT16, false, false);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Mul"), 1);
}

// ===================== 4-dim BatchMatMul success =====================

TEST_F(BatchMatMul2MulFusionPassTest, bmmToMul4DimSuccess)
{
    auto graph = BuildMatMulLikeGraph("bmmToMul4Dim", "BatchMatMul", {7200, 2, 1, 1}, {7200, 2, 1, 1}, {7200, 2, 1, 1},
                                      DT_FLOAT, false, false);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMul2MulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Reshape"), 0);
    EXPECT_EQ(CountNodes(graph, "Mul"), 1);
}
