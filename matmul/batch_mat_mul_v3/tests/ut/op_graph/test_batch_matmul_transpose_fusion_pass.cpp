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
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/batch_matmul_transpose_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace fe;
using namespace ops;

namespace {

constexpr int64_t DIM_4 = 4;
constexpr int64_t DIM_16 = 16;
constexpr int64_t DIM_32 = 32;
constexpr int64_t DIM_64 = 64;
constexpr int64_t DIM_128 = 128;
constexpr int64_t DIM_256 = 256;

void SetPlatformInfo950()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.ai_core_spec.l1_size = 512 * 1024;
    platformInfo.soc_info.l2_size = 192 * 1024 * 1024;
    optionalInfo.soc_version = "Ascend950";
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_nd2nz"] = {"float16"};
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_l12bt"] = {"bf16"};
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_post_eltwise_func_list"] = {"float16"};
    platformInfo.str_info.short_soc_version = "Ascend950";
    PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
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

bool FindNodeByType(const std::shared_ptr<Graph>& graph, const char* opType, GNode& outNode)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == opType) {
            outNode = node;
            return true;
        }
    }
    return false;
}

std::vector<int64_t> ComputeTransposedDims(const std::vector<int64_t>& dims, const std::vector<int32_t>& perm)
{
    std::vector<int64_t> result;
    result.reserve(perm.size());
    for (auto p : perm) {
        result.push_back(dims[p]);
    }
    return result;
}

EsTensorHolder CreateTransposeNode(EsGraphBuilder& builder, const EsTensorHolder& input,
                                   const std::vector<int32_t>& perm)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    std::vector<int64_t> permInt64(perm.begin(), perm.end());
    std::vector<int64_t> permShape{static_cast<int64_t>(perm.size())};
    auto permConst = builder.CreateConst(permInt64, permShape, DT_INT64, FORMAT_ND);

    auto transNode = CompliantNodeBuilder(graph)
                         .OpType("Transpose")
                         .Name("transpose_custom")
                         .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"perm", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), transNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *permConst.GetProducer(), permConst.GetProducerOutIndex(), transNode, 1);
    return EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
}

struct GraphConfig {
    const char* opType;
    std::vector<int64_t> x1Dims;
    std::vector<int64_t> x2Dims;
    std::vector<int64_t> outDims;
    DataType dtype;
    bool hasTransposeX1 = false;
    bool hasTransposeX2 = false;
    bool useTransposeD = false;
    std::vector<int32_t> permX1 = {1, 0};
    std::vector<int32_t> permX2 = {1, 0};
};

std::shared_ptr<Graph> BuildGraph(const std::string& name, const GraphConfig& cfg)
{
    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc(cfg.x1Dims, cfg.dtype);
    auto x2Desc = MakeTensorDesc(cfg.x2Dims, cfg.dtype);
    auto outDesc = MakeTensorDesc(cfg.outDims, cfg.dtype);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", cfg.dtype, FORMAT_ND, cfg.x1Dims);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", cfg.dtype, FORMAT_ND, cfg.x2Dims);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    EsTensorHolder matmulInput1 = dataX1;
    EsTensorHolder matmulInput2 = dataX2;

    if (cfg.hasTransposeX1) {
        if (cfg.useTransposeD) {
            std::vector<int64_t> permInt64(cfg.permX1.begin(), cfg.permX1.end());
            auto transNode = CompliantNodeBuilder(graph)
                                 .OpType("TransposeD")
                                 .Name((name + "_transD_x1").c_str())
                                 .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                 .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                 .IrDefAttrs({{"perm", CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                               CreateFrom(permInt64)}})
                                 .Build();
            AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), transNode, 0);
            transNode.UpdateInputDesc(0, x1Desc);
            auto transDims = ComputeTransposedDims(cfg.x1Dims, cfg.permX1);
            transNode.UpdateOutputDesc(0, MakeTensorDesc(transDims, cfg.dtype));
            matmulInput1 = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
        } else {
            matmulInput1 = CreateTransposeNode(graphBuilder, dataX1, cfg.permX1);
            auto* transProducer = matmulInput1.GetProducer();
            transProducer->UpdateInputDesc(0, x1Desc);
            TensorDesc permDesc(ge::Shape({static_cast<int64_t>(cfg.permX1.size())}), FORMAT_ND, DT_INT64);
            permDesc.SetOriginFormat(FORMAT_ND);
            permDesc.SetOriginShape(ge::Shape({static_cast<int64_t>(cfg.permX1.size())}));
            transProducer->UpdateInputDesc(1, permDesc);
            auto transDims = ComputeTransposedDims(cfg.x1Dims, cfg.permX1);
            transProducer->UpdateOutputDesc(0, MakeTensorDesc(transDims, cfg.dtype));
        }
    }

    if (cfg.hasTransposeX2) {
        if (cfg.useTransposeD) {
            std::vector<int64_t> permInt64(cfg.permX2.begin(), cfg.permX2.end());
            auto transNode = CompliantNodeBuilder(graph)
                                 .OpType("TransposeD")
                                 .Name((name + "_transD_x2").c_str())
                                 .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                 .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                 .IrDefAttrs({{"perm", CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                               CreateFrom(permInt64)}})
                                 .Build();
            AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), transNode, 0);
            transNode.UpdateInputDesc(0, x2Desc);
            auto transDims = ComputeTransposedDims(cfg.x2Dims, cfg.permX2);
            transNode.UpdateOutputDesc(0, MakeTensorDesc(transDims, cfg.dtype));
            matmulInput2 = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
        } else {
            matmulInput2 = CreateTransposeNode(graphBuilder, dataX2, cfg.permX2);
            auto* transProducer = matmulInput2.GetProducer();
            transProducer->UpdateInputDesc(0, x2Desc);
            TensorDesc permDesc(ge::Shape({static_cast<int64_t>(cfg.permX2.size())}), FORMAT_ND, DT_INT64);
            permDesc.SetOriginFormat(FORMAT_ND);
            permDesc.SetOriginShape(ge::Shape({static_cast<int64_t>(cfg.permX2.size())}));
            transProducer->UpdateInputDesc(1, permDesc);
            auto transDims = ComputeTransposedDims(cfg.x2Dims, cfg.permX2);
            transProducer->UpdateOutputDesc(0, MakeTensorDesc(transDims, cfg.dtype));
        }
    }

    bool isBatch = (strcmp(cfg.opType, "BatchMatMul") == 0 || strcmp(cfg.opType, "BatchMatMulV2") == 0);
    const char* transAttr1 = isBatch ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = isBatch ? "adj_x2" : "transpose_x2";

    bool irHasOffsetW = (strcmp(cfg.opType, "MatMulV2") == 0 || strcmp(cfg.opType, "BatchMatMulV2") == 0);
    std::vector<CompliantNodeBuilder::IrInputDef> irInputs = {
        {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
    };
    if (irHasOffsetW) {
        irInputs.push_back({"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""});
    }

    std::vector<CompliantNodeBuilder::IrAttrDef> irAttrs = {
        {transAttr1, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
        {transAttr2, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
    };

    auto node = CompliantNodeBuilder(graph)
                    .OpType(cfg.opType)
                    .Name(name.c_str())
                    .IrDefInputs(irInputs)
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs(irAttrs)
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *matmulInput1.GetProducer(), matmulInput1.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *matmulInput2.GetProducer(), matmulInput2.GetProducerOutIndex(), node, 1);

    TensorDesc matmulInputDesc0 = x1Desc;
    if (cfg.hasTransposeX1) {
        matmulInputDesc0 = MakeTensorDesc(ComputeTransposedDims(cfg.x1Dims, cfg.permX1), cfg.dtype);
    }
    TensorDesc matmulInputDesc1 = x2Desc;
    if (cfg.hasTransposeX2) {
        matmulInputDesc1 = MakeTensorDesc(ComputeTransposedDims(cfg.x2Dims, cfg.permX2), cfg.dtype);
    }
    node.UpdateInputDesc(0, matmulInputDesc0);
    node.UpdateInputDesc(1, matmulInputDesc1);
    node.UpdateOutputDesc(0, outDesc);

    return graphBuilder.BuildAndReset(
        {EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0))});
}

} // namespace

class BatchMatMulTransposeFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatformInfo950(); }
    class TestablePass : public ops::BatchMatMulTransposeFusionPass {};
};

TEST_F(BatchMatMulTransposeFusionPassTest, BasicTransposeFusionInput0)
{
    auto graph = BuildGraph("bmm_trans_x1", {"BatchMatMul",
                                             {DIM_128, DIM_256},
                                             {DIM_64, DIM_256},
                                             {DIM_128, DIM_64},
                                             DT_FLOAT16,
                                             .hasTransposeX1 = true,
                                             .permX1 = {1, 0}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 0);

    GNode bmmNode;
    ASSERT_TRUE(FindNodeByType(graph, "BatchMatMul", bmmNode));
    bool adjX1 = false;
    EXPECT_EQ(bmmNode.GetAttr("adj_x1", adjX1), GRAPH_SUCCESS);
    EXPECT_TRUE(adjX1);
}

TEST_F(BatchMatMulTransposeFusionPassTest, BasicTransposeFusionInput1)
{
    auto graph = BuildGraph("bmm_trans_x2", {"BatchMatMul",
                                             {DIM_128, DIM_256},
                                             {DIM_64, DIM_256},
                                             {DIM_128, DIM_64},
                                             DT_FLOAT16,
                                             .hasTransposeX2 = true,
                                             .permX2 = {1, 0}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 0);

    GNode bmmNode;
    ASSERT_TRUE(FindNodeByType(graph, "BatchMatMul", bmmNode));
    bool adjX2 = false;
    EXPECT_EQ(bmmNode.GetAttr("adj_x2", adjX2), GRAPH_SUCCESS);
    EXPECT_TRUE(adjX2);
}

TEST_F(BatchMatMulTransposeFusionPassTest, BothInputsTranspose)
{
    auto graph = BuildGraph("bmm_trans_both", {"BatchMatMul",
                                               {DIM_128, DIM_256},
                                               {DIM_64, DIM_256},
                                               {DIM_128, DIM_64},
                                               DT_FLOAT16,
                                               .hasTransposeX1 = true,
                                               .hasTransposeX2 = true,
                                               .permX1 = {1, 0},
                                               .permX2 = {1, 0}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 0);

    GNode bmmNode;
    ASSERT_TRUE(FindNodeByType(graph, "BatchMatMul", bmmNode));
    bool adjX1 = false, adjX2 = false;
    EXPECT_EQ(bmmNode.GetAttr("adj_x1", adjX1), GRAPH_SUCCESS);
    EXPECT_EQ(bmmNode.GetAttr("adj_x2", adjX2), GRAPH_SUCCESS);
    EXPECT_TRUE(adjX1);
    EXPECT_TRUE(adjX2);
}

TEST_F(BatchMatMulTransposeFusionPassTest, BothInputsTransposeD)
{
    auto graph = BuildGraph("bmm_transd_both", {"BatchMatMul",
                                                {DIM_128, DIM_256},
                                                {DIM_64, DIM_256},
                                                {DIM_128, DIM_64},
                                                DT_FLOAT16,
                                                .hasTransposeX1 = true,
                                                .hasTransposeX2 = true,
                                                .useTransposeD = true,
                                                .permX1 = {1, 0},
                                                .permX2 = {1, 0}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 0);

    GNode bmmNode;
    ASSERT_TRUE(FindNodeByType(graph, "BatchMatMul", bmmNode));
    bool adjX1 = false, adjX2 = false;
    EXPECT_EQ(bmmNode.GetAttr("adj_x1", adjX1), GRAPH_SUCCESS);
    EXPECT_EQ(bmmNode.GetAttr("adj_x2", adjX2), GRAPH_SUCCESS);
    EXPECT_TRUE(adjX1);
    EXPECT_TRUE(adjX2);
}

TEST_F(BatchMatMulTransposeFusionPassTest, MatMulTypeFusion)
{
    auto graph = BuildGraph("mm_trans_x1", {"MatMul",
                                            {DIM_128, DIM_256},
                                            {DIM_256, DIM_64},
                                            {DIM_128, DIM_64},
                                            DT_FLOAT,
                                            .hasTransposeX1 = true,
                                            .permX1 = {1, 0}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 0);

    GNode mmNode;
    ASSERT_TRUE(FindNodeByType(graph, "MatMul", mmNode));
    bool transX1 = false;
    EXPECT_EQ(mmNode.GetAttr("transpose_x1", transX1), GRAPH_SUCCESS);
    EXPECT_TRUE(transX1);
}

TEST_F(BatchMatMulTransposeFusionPassTest, MatMulTypeTransposeDFusion)
{
    auto graph = BuildGraph("mm_transd_x1", {"MatMul",
                                             {DIM_128, DIM_256},
                                             {DIM_256, DIM_64},
                                             {DIM_128, DIM_64},
                                             DT_FLOAT,
                                             .hasTransposeX1 = true,
                                             .useTransposeD = true,
                                             .permX1 = {1, 0}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 0);

    GNode mmNode;
    ASSERT_TRUE(FindNodeByType(graph, "MatMul", mmNode));
    bool transX1 = false;
    EXPECT_EQ(mmNode.GetAttr("transpose_x1", transX1), GRAPH_SUCCESS);
    EXPECT_TRUE(transX1);
}

TEST_F(BatchMatMulTransposeFusionPassTest, BatchMatMulV2TypeFusion)
{
    auto graph = BuildGraph("bmmv2_trans_x2", {"BatchMatMulV2",
                                               {DIM_128, DIM_256},
                                               {DIM_64, DIM_256},
                                               {DIM_128, DIM_64},
                                               DT_BF16,
                                               .hasTransposeX2 = true,
                                               .permX2 = {1, 0}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 0);

    GNode bmmNode;
    ASSERT_TRUE(FindNodeByType(graph, "BatchMatMulV2", bmmNode));
    bool adjX2 = false;
    EXPECT_EQ(bmmNode.GetAttr("adj_x2", adjX2), GRAPH_SUCCESS);
    EXPECT_TRUE(adjX2);
}

TEST_F(BatchMatMulTransposeFusionPassTest, BatchMatMulV2TypeTransposeDFusion)
{
    auto graph = BuildGraph("bmmv2_transd_x2", {"BatchMatMulV2",
                                                {DIM_128, DIM_256},
                                                {DIM_64, DIM_256},
                                                {DIM_128, DIM_64},
                                                DT_BF16,
                                                .hasTransposeX2 = true,
                                                .useTransposeD = true,
                                                .permX2 = {1, 0}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 0);

    GNode bmmNode;
    ASSERT_TRUE(FindNodeByType(graph, "BatchMatMulV2", bmmNode));
    bool adjX2 = false;
    EXPECT_EQ(bmmNode.GetAttr("adj_x2", adjX2), GRAPH_SUCCESS);
    EXPECT_TRUE(adjX2);
}

TEST_F(BatchMatMulTransposeFusionPassTest, ThreeDShapeFusion)
{
    auto graph = BuildGraph("bmm_3d_trans", {"BatchMatMul",
                                             {DIM_4, DIM_128, DIM_256},
                                             {DIM_4, DIM_64, DIM_256},
                                             {DIM_4, DIM_128, DIM_64},
                                             DT_FLOAT16,
                                             .hasTransposeX1 = true,
                                             .permX1 = {0, 2, 1}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 0);

    GNode bmmNode;
    ASSERT_TRUE(FindNodeByType(graph, "BatchMatMul", bmmNode));
    bool adjX1 = false;
    EXPECT_EQ(bmmNode.GetAttr("adj_x1", adjX1), GRAPH_SUCCESS);
    EXPECT_TRUE(adjX1);
}

TEST_F(BatchMatMulTransposeFusionPassTest, UnsupportedDtypeShouldNotFuse)
{
    auto graph = BuildGraph("bmm_int32_trans", {"BatchMatMul",
                                                {DIM_128, DIM_256},
                                                {DIM_256, DIM_64},
                                                {DIM_128, DIM_64},
                                                DT_INT32,
                                                .hasTransposeX1 = true,
                                                .permX1 = {1, 0}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 1);
}

TEST_F(BatchMatMulTransposeFusionPassTest, NoTransposeShouldNotFuse)
{
    auto graph = BuildGraph("bmm_no_trans",
                            {"BatchMatMul", {DIM_128, DIM_256}, {DIM_256, DIM_64}, {DIM_128, DIM_64}, DT_FLOAT16});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
}

TEST_F(BatchMatMulTransposeFusionPassTest, InvalidPermShouldNotFuse)
{
    auto graph = BuildGraph("bmm_invalid_perm", {"BatchMatMul",
                                                 {DIM_128, DIM_256},
                                                 {DIM_256, DIM_64},
                                                 {DIM_128, DIM_64},
                                                 DT_FLOAT16,
                                                 .hasTransposeX1 = true,
                                                 .permX1 = {0, 1}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 1);
}

TEST_F(BatchMatMulTransposeFusionPassTest, InvalidPermBatchTransposedShouldNotFuse)
{
    auto graph = BuildGraph("bmm_batch_transposed", {"BatchMatMul",
                                                     {DIM_4, DIM_128, DIM_256},
                                                     {DIM_4, DIM_256, DIM_64},
                                                     {DIM_4, DIM_128, DIM_64},
                                                     DT_FLOAT16,
                                                     .hasTransposeX1 = true,
                                                     .permX1 = {1, 0, 2}});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, "Transpose") + CountNodes(graph, "TransposeD"), 1);
}
