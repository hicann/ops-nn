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
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/matmul_biasadd_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace fe;
using namespace ops;

namespace {

constexpr char kPassName[] = "MatMulBiasAddFusionPass";

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
    platformInfo.str_info.short_soc_version = "Ascend950";
    PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

TensorDesc MakeTensorDesc(const std::vector<int64_t>& dims, DataType dtype, Format format = FORMAT_ND)
{
    TensorDesc desc(Shape(dims), format, dtype);
    desc.SetOriginFormat(format);
    desc.SetOriginShape(Shape(dims));
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

bool FindFirstNodeByOpType(const std::shared_ptr<Graph>& graph, const char* opType, GNode& outNode)
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

std::vector<CompliantNodeBuilder::IrInputDef> BuildMatMulIrInputs(const char* opType)
{
    std::vector<CompliantNodeBuilder::IrInputDef> irInputs = {
        {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
    };
    bool irHasBias = (strcmp(opType, "BatchMatMul") != 0);
    if (irHasBias) {
        irInputs.push_back({"bias", CompliantNodeBuilder::kEsIrInputOptional, ""});
    }
    bool irHasOffsetW = (strcmp(opType, "MatMulV2") == 0 || strcmp(opType, "BatchMatMulV2") == 0);
    if (irHasOffsetW) {
        irInputs.push_back({"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""});
    }
    return irInputs;
}

std::vector<CompliantNodeBuilder::IrAttrDef> BuildMatMulIrAttrs(const char* opType)
{
    bool isBatch = (strcmp(opType, "BatchMatMul") == 0 || strcmp(opType, "BatchMatMulV2") == 0);
    const char* transAttr1 = isBatch ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = isBatch ? "adj_x2" : "transpose_x2";
    std::vector<CompliantNodeBuilder::IrAttrDef> irAttrs = {
        {transAttr1, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
        {transAttr2, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
    };
    bool irHasOffsetW = (strcmp(opType, "MatMulV2") == 0 || strcmp(opType, "BatchMatMulV2") == 0);
    if (irHasOffsetW) {
        irAttrs.push_back({"offset_x", CompliantNodeBuilder::kEsAttrOptional, "Int", AttrValue()});
    }
    return irAttrs;
}

std::shared_ptr<Graph> BuildMatMulBiasAddGraph(const std::string& name, const char* matmulOpType,
                                               const char* biasAddOpType, bool matmulIsFirstInput,
                                               const std::vector<int64_t>& aDims, const std::vector<int64_t>& bDims,
                                               const std::vector<int64_t>& outDims,
                                               const std::vector<int64_t>& biasDims, DataType dtype,
                                               DataType biasDtype = DT_UNDEFINED)
{
    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc(aDims, dtype);
    auto x2Desc = MakeTensorDesc(bDims, dtype);
    auto outDesc = MakeTensorDesc(outDims, dtype);
    DataType actualBiasDtype = (biasDtype == DT_UNDEFINED) ? dtype : biasDtype;
    auto biasDesc = MakeTensorDesc(biasDims, actualBiasDtype);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", dtype, FORMAT_ND, aDims);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", dtype, FORMAT_ND, bDims);
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", actualBiasDtype, FORMAT_ND, biasDims);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);

    auto matmulNode = CompliantNodeBuilder(graph)
                          .OpType(matmulOpType)
                          .Name("matmul")
                          .IrDefInputs(BuildMatMulIrInputs(matmulOpType))
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs(BuildMatMulIrAttrs(matmulOpType))
                          .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), matmulNode, 1);
    matmulNode.UpdateInputDesc(0, x1Desc);
    matmulNode.UpdateInputDesc(1, x2Desc);
    matmulNode.UpdateOutputDesc(0, outDesc);
    bool transX1 = false;
    bool transX2 = false;
    bool isBatch = (strcmp(matmulOpType, "BatchMatMul") == 0 || strcmp(matmulOpType, "BatchMatMulV2") == 0);
    const char* transAttr1 = isBatch ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = isBatch ? "adj_x2" : "transpose_x2";
    matmulNode.SetAttr(transAttr1, transX1);
    matmulNode.SetAttr(transAttr2, transX2);

    GNode biasAddNode;
    if (strcmp(biasAddOpType, "BiasAdd") == 0) {
        biasAddNode = CompliantNodeBuilder(graph)
                          .OpType("BiasAdd")
                          .Name("biasadd")
                          .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"bias", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs({{"data_format", CompliantNodeBuilder::kEsAttrOptional, "String",
                                        CreateFrom(AscendString("NHWC"))}})
                          .Build();
        AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, biasAddNode, 0);
        AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), biasAddNode, 1);
        biasAddNode.UpdateInputDesc(0, outDesc);
        biasAddNode.UpdateInputDesc(1, biasDesc);
    } else {
        biasAddNode = CompliantNodeBuilder(graph)
                          .OpType("Add")
                          .Name("add")
                          .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();
        int64_t matmulIdx = matmulIsFirstInput ? 0 : 1;
        int64_t biasIdx = matmulIsFirstInput ? 1 : 0;
        AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, biasAddNode, matmulIdx);
        AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), biasAddNode, biasIdx);
        biasAddNode.UpdateInputDesc(matmulIdx, outDesc);
        biasAddNode.UpdateInputDesc(biasIdx, biasDesc);
    }
    biasAddNode.UpdateOutputDesc(0, outDesc);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(biasAddNode, 0));
    return graphBuilder.BuildAndReset({output});
}

void CheckHasBiasAttr(const std::shared_ptr<Graph>& graph, const char* matmulOpType, bool expected)
{
    GNode node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, matmulOpType, node));
    bool hasBias = false;
    node.GetAttr("has_bias", hasBias);
    EXPECT_EQ(hasBias, expected);
}

void CheckInputCount(const std::shared_ptr<Graph>& graph, const char* matmulOpType, size_t expected)
{
    GNode node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, matmulOpType, node));
    EXPECT_EQ(node.GetInputsSize(), expected);
}

void CheckBiasInputShape(const std::shared_ptr<Graph>& graph, const char* matmulOpType,
                         const std::vector<int64_t>& expectedDims)
{
    GNode node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, matmulOpType, node));
    TensorDesc biasDesc;
    ASSERT_EQ(node.GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
    auto actualDims = biasDesc.GetShape().GetDims();
    EXPECT_EQ(actualDims.size(), expectedDims.size());
    for (size_t i = 0; i < actualDims.size() && i < expectedDims.size(); i++) {
        EXPECT_EQ(actualDims[i], expectedDims[i]);
    }
}

} // namespace

class MatMulBiasAddFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatformInfo950(); }

    static void TearDownTestCase() {}

    void SetUp() override { SetPlatformInfo950(); }

    void TearDown() override {}
};

TEST_F(MatMulBiasAddFusionPassTest, batchMatMulBiasAddFp16FusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("batchMatMulBiasAddFp16", "BatchMatMul", "BiasAdd", true, {2, 2, 16, 16},
                                         {2, 2, 16, 1024}, {2, 2, 16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    CheckHasBiasAttr(graph, "BatchMatMulV2", true);
    CheckInputCount(graph, "BatchMatMulV2", 3U);
}

TEST_F(MatMulBiasAddFusionPassTest, batchMatMulAddMmFirstFp16FusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("batchMatMulAddMmFirstFp16", "BatchMatMul", "Add", true, {2, 2, 16, 16},
                                         {2, 2, 16, 1024}, {2, 2, 16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    CheckHasBiasAttr(graph, "BatchMatMulV2", true);
    CheckInputCount(graph, "BatchMatMulV2", 3U);
}

TEST_F(MatMulBiasAddFusionPassTest, batchMatMulAddMmSecondFp16FusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("batchMatMulAddMmSecondFp16", "BatchMatMul", "Add", false, {2, 2, 16, 16},
                                         {2, 2, 16, 1024}, {2, 2, 16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    CheckHasBiasAttr(graph, "BatchMatMulV2", true);
    CheckInputCount(graph, "BatchMatMulV2", 3U);
}

TEST_F(MatMulBiasAddFusionPassTest, batchMatMulAddNot1DBiasFail)
{
    auto graph = BuildMatMulBiasAddGraph("batchMatMulAddNot1DBias", "BatchMatMul", "Add", true, {2, 2, 16, 16},
                                         {2, 2, 16, 1024}, {2, 2, 16, 1024}, {2, 2, 16, 1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulAddMmFirstFp16FusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("matMulAddMmFirstFp16", "MatMul", "Add", true, {16, 16}, {16, 1024},
                                         {16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    CheckHasBiasAttr(graph, "MatMul", true);
    CheckInputCount(graph, "MatMul", 3U);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulAddMmSecondFp16FusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("matMulAddMmSecondFp16", "MatMul", "Add", false, {16, 16}, {16, 1024},
                                         {16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    CheckHasBiasAttr(graph, "MatMul", true);
    CheckInputCount(graph, "MatMul", 3U);
    CheckBiasInputShape(graph, "MatMul", {1024});
}

TEST_F(MatMulBiasAddFusionPassTest, matMulBiasAddFp16FusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("matMulBiasAddFp16", "MatMul", "BiasAdd", true, {16, 16}, {16, 1024},
                                         {16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    CheckHasBiasAttr(graph, "MatMul", true);
    CheckInputCount(graph, "MatMul", 3U);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulAddBf16FusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("matMulAddBf16", "MatMul", "Add", false, {16, 16}, {16, 1024}, {16, 1024},
                                         {1024}, DT_BF16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    CheckHasBiasAttr(graph, "MatMul", true);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulAddDynamicShapeFusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("matMulAddDynamicShape", "MatMul", "Add", false, {-1, -1}, {16, 1024},
                                         {-1, 1024}, {-1}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    CheckHasBiasAttr(graph, "MatMul", true);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulV2AddFp16FusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("matMulV2AddFp16", "MatMulV2", "Add", true, {16, 16}, {16, 1024}, {16, 1024},
                                         {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    CheckHasBiasAttr(graph, "MatMulV2", true);
    CheckInputCount(graph, "MatMulV2", 3U);
}

TEST_F(MatMulBiasAddFusionPassTest, batchMatMulV2AddFp16FusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("batchMatMulV2AddFp16", "BatchMatMulV2", "Add", false, {2, 3, 32, 64},
                                         {2, 3, 64, 128}, {2, 3, 32, 128}, {128}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Add"), 0);
    CheckHasBiasAttr(graph, "BatchMatMulV2", true);
    CheckInputCount(graph, "BatchMatMulV2", 3U);
}

TEST_F(MatMulBiasAddFusionPassTest, batchMatMulV2BiasAddBf16FusionSuccess)
{
    auto graph = BuildMatMulBiasAddGraph("batchMatMulV2BiasAddBf16", "BatchMatMulV2", "BiasAdd", true, {2, 3, 32, 64},
                                         {2, 3, 64, 128}, {2, 3, 32, 128}, {128}, DT_BF16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BiasAdd"), 0);
    CheckHasBiasAttr(graph, "BatchMatMulV2", true);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulAddBiasDimMismatchFail)
{
    auto graph = BuildMatMulBiasAddGraph("matMulAddBiasDimMismatch", "MatMul", "Add", true, {16, 16}, {16, 1024},
                                         {16, 1024}, {512}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulBiasAddFusionPassTest, batchMatMulBiasAddWithReluFp16FusionSuccess)
{
    auto graphBuilder = EsGraphBuilder("batchMatMulBiasAddReluFp16");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({2, 2, 16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({2, 2, 16, 1024}, DT_FLOAT16);
    auto outDesc = MakeTensorDesc({2, 2, 16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {2, 2, 16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {2, 2, 16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);

    auto matmulNode = CompliantNodeBuilder(graph)
                          .OpType("BatchMatMul")
                          .Name("batchmatmul")
                          .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs({{"adj_x1", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                                       {"adj_x2", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)}})
                          .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), matmulNode, 1);
    matmulNode.UpdateInputDesc(0, x1Desc);
    matmulNode.UpdateInputDesc(1, x2Desc);
    matmulNode.UpdateOutputDesc(0, outDesc);
    bool adjX1 = false;
    bool adjX2 = false;
    matmulNode.SetAttr("adj_x1", adjX1);
    matmulNode.SetAttr("adj_x2", adjX2);

    auto biasAddNode = CompliantNodeBuilder(graph)
                           .OpType("BiasAdd")
                           .Name("biasadd")
                           .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"bias", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({{"data_format", CompliantNodeBuilder::kEsAttrOptional, "String",
                                         CreateFrom(AscendString("NHWC"))}})
                           .Build();
    AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, biasAddNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), biasAddNode, 1);
    biasAddNode.UpdateInputDesc(0, outDesc);
    biasAddNode.UpdateInputDesc(1, biasDesc);
    biasAddNode.UpdateOutputDesc(0, outDesc);

    auto reluNode = CompliantNodeBuilder(graph)
                        .OpType("Relu")
                        .Name("relu")
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    AddEdgeAndUpdatePeerDesc(*graph, biasAddNode, 0, reluNode, 0);
    reluNode.UpdateInputDesc(0, outDesc);
    reluNode.UpdateOutputDesc(0, outDesc);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reluNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graphPtr, "BiasAdd"), 0);
    EXPECT_EQ(CountNodes(graphPtr, "Relu"), 1);
    CheckHasBiasAttr(graphPtr, "BatchMatMulV2", true);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulAlreadyHasBiasFail)
{
    auto graphBuilder = EsGraphBuilder("matMulAlreadyHasBias");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto outDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);

    auto matmulNode = CompliantNodeBuilder(graph)
                          .OpType("MatMul")
                          .Name("matmul_with_bias")
                          .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""}})
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs(
                              {{"transpose_x1", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                               {"transpose_x2", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)}})
                          .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), matmulNode, 1);
    AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), matmulNode, 2);
    matmulNode.UpdateInputDesc(0, x1Desc);
    matmulNode.UpdateInputDesc(1, x2Desc);
    matmulNode.UpdateInputDesc(2, biasDesc);
    matmulNode.UpdateOutputDesc(0, outDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto addNode = CompliantNodeBuilder(graph)
                       .OpType("Add")
                       .Name("add")
                       .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                       .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();
    AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, addNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), addNode, 1);
    addNode.UpdateInputDesc(0, outDesc);
    addNode.UpdateInputDesc(1, biasDesc);
    addNode.UpdateOutputDesc(0, outDesc);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulOutputToMultipleNodesFail)
{
    auto graphBuilder = EsGraphBuilder("matMulMultiOutput");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto outDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);

    auto matmulNode = CompliantNodeBuilder(graph)
                          .OpType("MatMul")
                          .Name("matmul")
                          .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs(
                              {{"transpose_x1", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                               {"transpose_x2", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)}})
                          .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), matmulNode, 1);
    matmulNode.UpdateInputDesc(0, x1Desc);
    matmulNode.UpdateInputDesc(1, x2Desc);
    matmulNode.UpdateOutputDesc(0, outDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto addNode = CompliantNodeBuilder(graph)
                       .OpType("Add")
                       .Name("add")
                       .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                       .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();
    AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, addNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), addNode, 1);
    addNode.UpdateInputDesc(0, outDesc);
    addNode.UpdateInputDesc(1, biasDesc);
    addNode.UpdateOutputDesc(0, outDesc);

    auto reluNode = CompliantNodeBuilder(graph)
                        .OpType("Relu")
                        .Name("relu")
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, reluNode, 0);
    reluNode.UpdateInputDesc(0, outDesc);
    reluNode.UpdateOutputDesc(0, outDesc);

    auto addOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addNode, 0));
    auto reluOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reluNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({addOutput, reluOutput});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulBiasAddFusionPassTest, addNon1DBiasMmFirstFail)
{
    auto graph = BuildMatMulBiasAddGraph("addNon1DBiasMmFirst", "MatMul", "Add", true, {16, 16}, {16, 1024}, {16, 1024},
                                         {16, 1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulBiasAddFusionPassTest, addNon1DBiasMmSecondFail)
{
    auto graph = BuildMatMulBiasAddGraph("addNon1DBiasMmSecond", "MatMul", "Add", false, {16, 16}, {16, 1024},
                                         {16, 1024}, {16, 1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulAddMmFirstNon2DOutputFail)
{
    auto graph = BuildMatMulBiasAddGraph("matMulAddMmFirstNon2DOutput", "MatMul", "Add", true, {16, 16, 16},
                                         {16, 16, 1024}, {16, 16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulAddMmSecondNon2DOutputFail)
{
    auto graph = BuildMatMulBiasAddGraph("matMulAddMmSecondNon2DOutput", "MatMul", "Add", false, {16, 16, 16},
                                         {16, 16, 1024}, {16, 16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulBiasAddFusionPassTest, biasAddOutputToMultipleNodesFusionSuccess)
{
    auto graphBuilder = EsGraphBuilder("biasAddMultiDownstream");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto outDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);

    auto matmulNode = CompliantNodeBuilder(graph)
                          .OpType("MatMul")
                          .Name("matmul")
                          .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""}})
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs(
                              {{"transpose_x1", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                               {"transpose_x2", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)}})
                          .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), matmulNode, 1);
    matmulNode.UpdateInputDesc(0, x1Desc);
    matmulNode.UpdateInputDesc(1, x2Desc);
    matmulNode.UpdateOutputDesc(0, outDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto biasAddNode = CompliantNodeBuilder(graph)
                           .OpType("Add")
                           .Name("add")
                           .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .Build();
    AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, biasAddNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), biasAddNode, 1);
    biasAddNode.UpdateInputDesc(0, outDesc);
    biasAddNode.UpdateInputDesc(1, biasDesc);
    biasAddNode.UpdateOutputDesc(0, outDesc);

    auto reluNode = CompliantNodeBuilder(graph)
                        .OpType("Relu")
                        .Name("relu")
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    AddEdgeAndUpdatePeerDesc(*graph, biasAddNode, 0, reluNode, 0);
    reluNode.UpdateInputDesc(0, outDesc);
    reluNode.UpdateOutputDesc(0, outDesc);

    auto negNode = CompliantNodeBuilder(graph)
                       .OpType("Neg")
                       .Name("neg")
                       .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                       .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();
    AddEdgeAndUpdatePeerDesc(*graph, biasAddNode, 0, negNode, 0);
    negNode.UpdateInputDesc(0, outDesc);
    negNode.UpdateOutputDesc(0, outDesc);

    auto reluOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reluNode, 0));
    auto negOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(negNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({reluOutput, negOutput});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graphPtr, "Add"), 0);
    EXPECT_EQ(CountNodes(graphPtr, "Relu"), 1);
    EXPECT_EQ(CountNodes(graphPtr, "Neg"), 1);
    CheckHasBiasAttr(graphPtr, "MatMul", true);
}

TEST_F(MatMulBiasAddFusionPassTest, matMulBiasAddWithCtrlEdgesFusionSuccess)
{
    auto graphBuilder = EsGraphBuilder("matMulBiasAddCtrlEdges");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto outDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);

    auto matmulNode = CompliantNodeBuilder(graph)
                          .OpType("MatMul")
                          .Name("matmul")
                          .IrDefInputs(BuildMatMulIrInputs("MatMul"))
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs(BuildMatMulIrAttrs("MatMul"))
                          .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), matmulNode, 1);
    matmulNode.UpdateInputDesc(0, x1Desc);
    matmulNode.UpdateInputDesc(1, x2Desc);
    matmulNode.UpdateOutputDesc(0, outDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto biasAddNode = CompliantNodeBuilder(graph)
                           .OpType("BiasAdd")
                           .Name("biasadd")
                           .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"bias", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({{"data_format", CompliantNodeBuilder::kEsAttrOptional, "String",
                                         CreateFrom(AscendString("NHWC"))}})
                           .Build();
    AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, biasAddNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), biasAddNode, 1);
    biasAddNode.UpdateInputDesc(0, outDesc);
    biasAddNode.UpdateInputDesc(1, biasDesc);
    biasAddNode.UpdateOutputDesc(0, outDesc);

    auto reluNode = CompliantNodeBuilder(graph)
                        .OpType("Relu")
                        .Name("relu")
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    AddEdgeAndUpdatePeerDesc(*graph, biasAddNode, 0, reluNode, 0);
    reluNode.UpdateInputDesc(0, outDesc);
    reluNode.UpdateOutputDesc(0, outDesc);

    // IN 控制边: dataX2 --ctrl--> biasAdd
    graph->AddControlEdge(*dataX2.GetProducer(), biasAddNode);
    // OUT 控制边: biasAdd --ctrl--> relu
    graph->AddControlEdge(biasAddNode, reluNode);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reluNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graphPtr, "BiasAdd"), 0);
    CheckHasBiasAttr(graphPtr, "MatMul", true);

    // 验证 OUT 控制边转移到 matmul: matmul -> relu
    GNode matmulAfterFuse;
    ASSERT_TRUE(FindFirstNodeByOpType(graphPtr, "MatMul", matmulAfterFuse));
    auto outCtrlNodes = matmulAfterFuse.GetOutControlNodes();
    bool hasReluCtrl = false;
    for (auto& ctrlNode : outCtrlNodes) {
        if (ctrlNode != nullptr) {
            AscendString ctrlType;
            ctrlNode->GetType(ctrlType);
            if (ctrlType == "Relu") {
                hasReluCtrl = true;
            }
        }
    }
    EXPECT_TRUE(hasReluCtrl);

    // 验证 IN 控制边转移到 matmul: dataX2 -> matmul
    auto inCtrlNodes = matmulAfterFuse.GetInControlNodes();
    bool hasDataX2Ctrl = false;
    for (auto& ctrlNode : inCtrlNodes) {
        if (ctrlNode != nullptr) {
            AscendString ctrlName;
            ctrlNode->GetName(ctrlName);
            if (std::string(ctrlName.GetString()) == "dataX2") {
                hasDataX2Ctrl = true;
            }
        }
    }
    EXPECT_TRUE(hasDataX2Ctrl);
}
