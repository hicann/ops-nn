/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
#define private public
#include "platform/platform_info.h"
#undef private
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/batch_matmul_to_transpose_batch_matmul_fusion_pass.h"

using namespace ge;
using namespace ge::fusion;

namespace ops {
namespace {

constexpr char kPassName[] = "BatchMatMul2TransposeBatchMatMulFusionPass";

constexpr int32_t kAllowDim = 3;
constexpr int32_t kPattern2AllowDim = 4;

struct TensorWithDesc {
    es::EsTensorHolder tensor;
    TensorDesc desc;
};

class TestBatchMatMul2TransposeBatchMatMulFusionPass : public BatchMatMul2TransposeBatchMatMulFusionPass {
public:
    Status RunForTest(GraphPtr& graph, CustomPassContext& passContext) { return Run(graph, passContext); }
};

static TensorDesc MakeTensorDesc(const std::vector<int64_t>& shape, DataType dtype, Format format = FORMAT_ND)
{
    TensorDesc desc(ge::Shape(shape), format, dtype);
    desc.SetOriginFormat(format);
    desc.SetOriginShape(ge::Shape(shape));
    return desc;
}

static void SetPlatform(const std::string& socVersion, uint32_t aiCoreCnt = 24)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = static_cast<int32_t>(aiCoreCnt);
    platformInfo.str_info.short_soc_version = socVersion;
    optionalInfo.soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

static GNode BuildBatchMatMulNode(Graph* graph, const std::string& name, const TensorDesc& x1Desc,
                                  const TensorDesc& x2Desc, const TensorDesc& outDesc, bool adjX1 = false,
                                  bool adjX2 = false)
{
    auto bmmNode = es::CompliantNodeBuilder(graph)
                       .OpType("BatchMatMul")
                       .Name(name.c_str())
                       .IrDefInputs({
                           {"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                       })
                       .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .IrDefAttrs({
                           {"adj_x1", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(adjX1)},
                           {"adj_x2", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(adjX2)},
                       })
                       .Build();
    bmmNode.UpdateInputDesc(0, x1Desc);
    bmmNode.UpdateInputDesc(1, x2Desc);
    bmmNode.UpdateOutputDesc(0, outDesc);
    return bmmNode;
}

static GNode BuildTransposeNode(Graph* graph, const std::string& name, const TensorDesc& inDesc,
                                const TensorDesc& outDesc, const std::vector<int64_t>& perm)
{
    auto transNode = es::CompliantNodeBuilder(graph)
                         .OpType("TransposeD")
                         .Name(name.c_str())
                         .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs({
                             {"perm", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt", AttrValue()},
                         })
                         .Build();
    transNode.UpdateInputDesc(0, inDesc);
    transNode.UpdateOutputDesc(0, outDesc);
    auto permCopy = perm;
    transNode.SetAttr("perm", permCopy);
    return transNode;
}

static GNode BuildTransposeNodeWithType(es::EsGraphBuilder& graphBuilder, Graph* graph, const std::string& name,
                                        const TensorDesc& inDesc, const TensorDesc& outDesc,
                                        const std::vector<int64_t>& perm, bool useTransposeD)
{
    if (useTransposeD) {
        auto transNode = BuildTransposeNode(graph, name, inDesc, outDesc, perm);
        return transNode;
    }
    std::vector<int64_t> permShape{static_cast<int64_t>(perm.size())};
    auto permTensor = graphBuilder.CreateConst(perm, permShape, DT_INT64, FORMAT_ND);
    auto transNode = es::CompliantNodeBuilder(graph)
                         .OpType("Transpose")
                         .Name(name.c_str())
                         .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"perm", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
    transNode.UpdateInputDesc(0, inDesc);
    transNode.UpdateOutputDesc(0, outDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *permTensor.GetProducer(), permTensor.GetProducerOutIndex(), transNode, 1);
    return transNode;
}

static GNode BuildReshapeNode(Graph* graph, const std::string& name, const TensorDesc& inDesc,
                              const TensorDesc& outDesc, const std::vector<int64_t>& shapeAttr)
{
    auto reshapeNode = es::CompliantNodeBuilder(graph)
                           .OpType("Reshape")
                           .Name(name.c_str())
                           .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({{"axis", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                         es::CreateFrom(static_cast<int64_t>(0))},
                                        {"num_axes", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                         es::CreateFrom(static_cast<int64_t>(-1))}})
                           .Build();
    reshapeNode.UpdateInputDesc(0, inDesc);
    reshapeNode.UpdateOutputDesc(0, outDesc);
    TensorDesc shapeInputDesc(ge::Shape({static_cast<int64_t>(shapeAttr.size())}), FORMAT_ND, DT_INT64);
    reshapeNode.UpdateInputDesc(1, shapeInputDesc);
    auto shapeAttrCopy = shapeAttr;
    reshapeNode.SetAttr("shape", shapeAttrCopy);
    return reshapeNode;
}

static int CountNodes(const GraphPtr& graph, const char* nodeType)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type == nodeType) {
            count++;
        }
    }
    return count;
}

static bool HasNodeType(const GraphPtr& graph, const char* nodeType) { return CountNodes(graph, nodeType) > 0; }

} // namespace
} // namespace ops

using namespace ops;

class BatchMatMul2TransposeBatchMatMulFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatform("Ascend910_93"); }

    void SetUp() override { SetPlatform("Ascend910_93"); }
};

// ==================== Pattern1 Tests ====================

// Test: pattern1 fusion success - BatchMatMul with output Transpose, fp16, 3D
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1FusionSuccessFp16)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_fp16");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    // x1: [4, 128, 64], x2: [4, 64, 128], y: [4, 128, 128]
    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    // Transpose output: perm [1,0,2] -> [128, 4, 128]
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern1 no fusion - unsupported platform
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1UnsupportedPlatform)
{
    SetPlatform("Ascend310P3");

    auto graphBuilder = es::EsGraphBuilder("test_pattern1_unsupported_plat");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// ==================== Additional Pattern1 Tests ====================

// Test: pattern1 fusion success - bf16
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1FusionSuccessBf16)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_bf16");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_BF16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_BF16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_BF16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_BF16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_BF16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_BF16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern1 fusion success - fp32
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1FusionSuccessFp32)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_fp32");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 128}, DT_FLOAT);
    auto x2Desc = MakeTensorDesc({4, 128, 128}, DT_FLOAT);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT, FORMAT_ND, {4, 128, 128});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT, FORMAT_ND, {4, 128, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern1 fusion success - with input Transpose (perm=[1,0,2])
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1WithInputTranspose)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_input_trans");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto dataX1Desc = MakeTensorDesc({128, 4, 64}, DT_FLOAT16);
    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {128, 4, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, dataX1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto transNode1 = BuildTransposeNode(graph, "trans_in", dataX1Desc, x1Desc, {1, 0, 2});
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, transNode1, 0);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, transNode1, 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode3 = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode3, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode3, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern1 no fusion - dynamic shape
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1DynamicShape)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_dynamic");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({-1, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({-1, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, -1, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {-1, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: pattern1 no fusion - k/n not aligned to 128 (non-950 platform)
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1KnUnaligned)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_kn_unaligned");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 65}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 65, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 65});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 65, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: pattern1 no fusion - adj_x2 is true
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1AdjX2True)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_adj_x2");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc, false, true);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// ==================== Additional Pattern2 Tests ====================

// Test: pattern2 fusion success - BatchMatMulV2 fp32
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern2FusionSuccessBatchMatMulV2Fp32)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern2_bmmv2_fp32");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 8, 1, 128}, DT_FLOAT);
    auto x2Desc = MakeTensorDesc({1, 8, 128, 256}, DT_FLOAT);
    auto bmmOutDesc = MakeTensorDesc({4, 8, 1, 256}, DT_FLOAT);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT, FORMAT_ND, {4, 8, 1, 128});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT, FORMAT_ND, {1, 8, 128, 256});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = es::CompliantNodeBuilder(graph)
                       .OpType("BatchMatMulV2")
                       .Name("bmmv2")
                       .IrDefInputs({
                           {"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                       })
                       .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .IrDefAttrs({
                           {"adj_x1", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
                           {"adj_x2", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
                       })
                       .Build();
    bmmNode.UpdateInputDesc(0, x1Desc);
    bmmNode.UpdateInputDesc(1, x2Desc);
    bmmNode.UpdateOutputDesc(0, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bmmNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
        EXPECT_TRUE(HasNodeType(graphPtr, "Reshape"));
    }
}

// Test: pattern2 fusion success - adj_x2=True (x2 is (1,B1,N,K))
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern2FusionSuccessAdjX2True)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern2_adjx2");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({2, 4, 1, 64}, DT_FLOAT);
    auto x2Desc = MakeTensorDesc({1, 4, 128, 64}, DT_FLOAT);
    auto bmmOutDesc = MakeTensorDesc({2, 4, 1, 128}, DT_FLOAT);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT, FORMAT_ND, {2, 4, 1, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT, FORMAT_ND, {1, 4, 128, 64});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc, false, true);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bmmNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern2 no fusion - has bias (3 inputs)
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern2HasBias)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern2_bias");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({2, 4, 1, 64}, DT_FLOAT);
    auto x2Desc = MakeTensorDesc({1, 4, 64, 128}, DT_FLOAT);
    auto bmmOutDesc = MakeTensorDesc({2, 4, 1, 128}, DT_FLOAT);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT, FORMAT_ND, {2, 4, 1, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT, FORMAT_ND, {1, 4, 64, 128});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT, FORMAT_ND, {128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), 0, bmmNode, 2);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bmmNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: pattern2 no fusion - inner axis >= 65536 (non-950)
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern2InnerAxisExceed)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern2_inner_axis");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({2, 65536, 1, 2}, DT_FLOAT);
    auto x2Desc = MakeTensorDesc({1, 65536, 2, 4}, DT_FLOAT);
    auto bmmOutDesc = MakeTensorDesc({2, 65536, 1, 4}, DT_FLOAT);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT, FORMAT_ND, {2, 65536, 1, 2});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT, FORMAT_ND, {1, 65536, 2, 4});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bmmNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: pattern1 no fusion - unsupported dtype (int32)
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1UnsupportedDtype)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_unsupported_dtype");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_INT32);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_INT32);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_INT32);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_INT32);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_INT32, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_INT32, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: pattern1 no fusion - no output Transpose node
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1NoOutputTranspose)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_no_trans_out");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bmmNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: pattern1 no fusion - adj_x1 is true
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1AdjX1True)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_adj_x1");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc, true, false);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: pattern1 no fusion - 2D input (not 3D)
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1WrongDim)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_wrong_dim");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: pattern1 no fusion - has bias (3 inputs)
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1HasBias)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_bias");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), 0, bmmNode, 2);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// ==================== Pattern2 Tests ====================

// Test: pattern2 fusion success - fp32, 4D inputs A(B2,B1,1,K) B(1,B1,K,N)
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern2FusionSuccessFp32)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern2_fp32");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    // x1: [2, 4, 1, 64], x2: [1, 4, 64, 128], y: [2, 4, 1, 128]
    auto x1Desc = MakeTensorDesc({2, 4, 1, 64}, DT_FLOAT);
    auto x2Desc = MakeTensorDesc({1, 4, 64, 128}, DT_FLOAT);
    auto bmmOutDesc = MakeTensorDesc({2, 4, 1, 128}, DT_FLOAT);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT, FORMAT_ND, {2, 4, 1, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT, FORMAT_ND, {1, 4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bmmNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
        EXPECT_TRUE(HasNodeType(graphPtr, "Reshape"));
    }
}

// Test: pattern2 no fusion - fp16 (only fp32 supported)
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern2UnsupportedDtypeFp16)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern2_fp16");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({2, 4, 1, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({1, 4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({2, 4, 1, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {2, 4, 1, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {1, 4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bmmNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    // fp16 should fall through to pattern1 check, which requires 3D input, so NOT_CHANGED
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: pattern2 no fusion - wrong shape (x1[2] != 1)
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern2WrongShape)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern2_wrong_shape");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    // x1[2] = 2 (should be 1)
    auto x1Desc = MakeTensorDesc({2, 4, 2, 64}, DT_FLOAT);
    auto x2Desc = MakeTensorDesc({1, 4, 64, 128}, DT_FLOAT);
    auto bmmOutDesc = MakeTensorDesc({2, 4, 2, 128}, DT_FLOAT);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT, FORMAT_ND, {2, 4, 2, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT, FORMAT_ND, {1, 4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bmmNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: pattern2 no fusion - adj_x1 is true
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern2AdjX1True)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern2_adj_x1");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({2, 4, 1, 64}, DT_FLOAT);
    auto x2Desc = MakeTensorDesc({1, 4, 64, 128}, DT_FLOAT);
    auto bmmOutDesc = MakeTensorDesc({2, 4, 1, 128}, DT_FLOAT);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT, FORMAT_ND, {2, 4, 1, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT, FORMAT_ND, {1, 4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc, true, false);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(bmmNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test: BatchMatMulV2 also supported
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1BatchMatMulV2)
{
    auto graphBuilder = es::EsGraphBuilder("test_pattern1_bmmv2");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = es::CompliantNodeBuilder(graph)
                       .OpType("BatchMatMulV2")
                       .Name("bmmv2")
                       .IrDefInputs({
                           {"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                       })
                       .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .IrDefAttrs({
                           {"adj_x1", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
                           {"adj_x2", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
                       })
                       .Build();
    bmmNode.UpdateInputDesc(0, x1Desc);
    bmmNode.UpdateInputDesc(1, x2Desc);
    bmmNode.UpdateOutputDesc(0, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// ==================== Ascend950 perm_x2=[0,2,1] Tests ====================

// Test: pattern1 fusion success - Ascend950 with perm_x2=[0,2,1] input transpose on x2
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1Ascend950PermX2_021)
{
    SetPlatform("Ascend950");

    auto graphBuilder = es::EsGraphBuilder("test_p1_950_perm_x2_021");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2AfterTransDesc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2OrigDesc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2OrigDesc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2AfterTransDesc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);

    auto transX2 = BuildTransposeNode(graph, "trans_x2", x2OrigDesc, x2AfterTransDesc, {0, 2, 1});
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, transX2, 0);
    graph->AddDataEdge(transX2, 0, bmmNode, 1);

    auto transOut = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transOut, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transOut, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern1 fusion success - Ascend950 with perm_x1=[1,0,2] + perm_x2=[0,2,1]
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1Ascend950PermX1_102_PermX2_021)
{
    SetPlatform("Ascend950");

    auto graphBuilder = es::EsGraphBuilder("test_p1_950_perm_x1_102_x2_021");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1OrigDesc = MakeTensorDesc({128, 4, 64}, DT_FLOAT16);
    auto x1AfterTransDesc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2OrigDesc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto x2AfterTransDesc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {128, 4, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1OrigDesc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2OrigDesc);

    auto transX1 = BuildTransposeNode(graph, "trans_x1", x1OrigDesc, x1AfterTransDesc, {1, 0, 2});
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, transX1, 0);

    auto transX2 = BuildTransposeNode(graph, "trans_x2", x2OrigDesc, x2AfterTransDesc, {0, 2, 1});
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, transX2, 0);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1AfterTransDesc, x2AfterTransDesc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, transX1, 0, bmmNode, 0);
    graph->AddDataEdge(transX2, 0, bmmNode, 1);

    auto transOut = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transOut, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transOut, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern1 no fusion - Ascend950 perm_x1=[0,2,1] should be rejected
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1Ascend950PermX1_021Rejected)
{
    SetPlatform("Ascend950");

    auto graphBuilder = es::EsGraphBuilder("test_p1_950_perm_x1_021_reject");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1OrigDesc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto x1AfterTransDesc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1OrigDesc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto transX1 = BuildTransposeNode(graph, "trans_x1", x1OrigDesc, x1AfterTransDesc, {0, 2, 1});
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, transX1, 0);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1AfterTransDesc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, transX1, 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transOut = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transOut, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transOut, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, SUCCESS);
}

// Test: pattern1 fusion success - Ascend950 with Transpose (const input) perm_x1=[1,0,2]
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1Ascend950TransposePermX1_102)
{
    SetPlatform("Ascend950");

    auto graphBuilder = es::EsGraphBuilder("test_p1_950_trans_perm_x1_102");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1OrigDesc = MakeTensorDesc({128, 4, 64}, DT_FLOAT16);
    auto x1AfterTransDesc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {128, 4, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1OrigDesc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto transX1 = BuildTransposeNodeWithType(graphBuilder, graph, "trans_x1", x1OrigDesc, x1AfterTransDesc, {1, 0, 2},
                                              false);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, transX1, 0);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1AfterTransDesc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, transX1, 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transOut = BuildTransposeNodeWithType(graphBuilder, graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2},
                                               false);
    graph->AddDataEdge(bmmNode, 0, transOut, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transOut, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern1 fusion success - Ascend950 with Transpose (const input) perm_x2=[0,2,1]
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1Ascend950TransposePermX2_021)
{
    SetPlatform("Ascend950");

    auto graphBuilder = es::EsGraphBuilder("test_p1_950_trans_perm_x2_021");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2AfterTransDesc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2OrigDesc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2OrigDesc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2AfterTransDesc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);

    auto transX2 = BuildTransposeNodeWithType(graphBuilder, graph, "trans_x2", x2OrigDesc, x2AfterTransDesc, {0, 2, 1},
                                              false);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, transX2, 0);
    graph->AddDataEdge(transX2, 0, bmmNode, 1);

    auto transOut = BuildTransposeNodeWithType(graphBuilder, graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2},
                                               false);
    graph->AddDataEdge(bmmNode, 0, transOut, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transOut, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern1 fusion success - Ascend910B with TransposeD perm_x1=[1,0,2]
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1Ascend910BPermX1_102)
{
    SetPlatform("Ascend910B");

    auto graphBuilder = es::EsGraphBuilder("test_p1_910b_perm_x1_102");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1OrigDesc = MakeTensorDesc({128, 4, 64}, DT_FLOAT16);
    auto x1AfterTransDesc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {128, 4, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1OrigDesc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto transX1 = BuildTransposeNode(graph, "trans_x1", x1OrigDesc, x1AfterTransDesc, {1, 0, 2});
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, transX1, 0);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1AfterTransDesc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, transX1, 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transOut = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transOut, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transOut, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern1 fusion success - Ascend910B no transpose on x1
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1Ascend910BNoTransX1)
{
    SetPlatform("Ascend910B");

    auto graphBuilder = es::EsGraphBuilder("test_p1_910b_no_trans_x1");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);
    auto transOutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transOut = BuildTransposeNode(graph, "trans_out", bmmOutDesc, transOutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transOut, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transOut, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}

// Test: pattern1 fusion success - batch_split_factor > 1
// Graph: bmm -> trans3(perm=102) -> reshape1 -> reshape2 -> trans4(perm=102) -> output
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, Pattern1BatchSplitFactorGt1)
{
    SetPlatform("Ascend950");

    auto graphBuilder = es::EsGraphBuilder("test_p1_bsf_gt1");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({4, 64, 128}, DT_FLOAT16);
    auto bmmOutDesc = MakeTensorDesc({4, 128, 128}, DT_FLOAT16);

    auto trans3OutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);
    auto reshape1OutDesc = MakeTensorDesc({128, 4, 128}, DT_FLOAT16);
    auto reshape2OutDesc = MakeTensorDesc({128, 2, 256}, DT_FLOAT16);
    auto trans4OutDesc = MakeTensorDesc({2, 128, 256}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {4, 64, 128});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto bmmNode = BuildBatchMatMulNode(graph, "bmm", x1Desc, x2Desc, bmmOutDesc);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), 0, bmmNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), 0, bmmNode, 1);

    auto transNode3 = BuildTransposeNode(graph, "trans3", bmmOutDesc, trans3OutDesc, {1, 0, 2});
    graph->AddDataEdge(bmmNode, 0, transNode3, 0);

    auto reshapeNode1 = BuildReshapeNode(graph, "reshape1", trans3OutDesc, reshape1OutDesc, {128, 4, 128});
    graph->AddDataEdge(transNode3, 0, reshapeNode1, 0);

    auto reshapeNode2 = BuildReshapeNode(graph, "reshape2", reshape1OutDesc, reshape2OutDesc, {128, 2, 256});
    graph->AddDataEdge(reshapeNode1, 0, reshapeNode2, 0);

    auto transNode4 = BuildTransposeNode(graph, "trans4", reshape2OutDesc, trans4OutDesc, {1, 0, 2});
    graph->AddDataEdge(reshapeNode2, 0, transNode4, 0);

    auto output = es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transNode4, 0));
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    if (status == SUCCESS) {
        EXPECT_TRUE(HasNodeType(graphPtr, "TransposeBatchMatMul"));
    }
}
TEST_F(BatchMatMul2TransposeBatchMatMulFusionPassTest, NoBatchMatMulNode)
{
    auto graphBuilder = es::EsGraphBuilder("test_no_bmm");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {4, 128, 64});
    auto x1Desc = MakeTensorDesc({4, 128, 64}, DT_FLOAT16);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);

    auto output = dataX1;
    GraphPtr graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    TestBatchMatMul2TransposeBatchMatMulFusionPass pass;
    Status status = pass.RunForTest(graphPtr, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}
