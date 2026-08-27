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
#include "../../../op_graph/fusion_pass/matmul_reshape_biasadd_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace fe;
using namespace ops;

namespace {

constexpr char kPassName[] = "MatMulReshapeBiasAddFusionPass";
constexpr char kOpTypeReshape[] = "Reshape";

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

GNode BuildReshapeNode(Graph* graph, GNode& matmulNode, GNode& shapeDataNode, const TensorDesc& matmulOutDesc,
                       const TensorDesc& reshapeOutDesc, const TensorDesc& shapeDesc)
{
    auto reshapeNode = CompliantNodeBuilder(graph)
                           .OpType(kOpTypeReshape)
                           .Name("reshape")
                           .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"shape", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({{"axis", CompliantNodeBuilder::kEsAttrRequired, "Int",
                                         CreateFrom(static_cast<int64_t>(0))},
                                        {"num_axes", CompliantNodeBuilder::kEsAttrRequired, "Int",
                                         CreateFrom(static_cast<int64_t>(-1))}})
                           .Build();
    AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, reshapeNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, shapeDataNode, 0, reshapeNode, 1);
    reshapeNode.UpdateInputDesc(0, matmulOutDesc);
    reshapeNode.UpdateInputDesc(1, shapeDesc);
    reshapeNode.UpdateOutputDesc(0, reshapeOutDesc);
    return reshapeNode;
}

GNode BuildBiasAddOrAddNode(Graph* graph, const char* biasAddOpType, GNode& reshapeNode, GNode& biasDataNode,
                            bool reshapeIsFirstInput, const TensorDesc& reshapeOutDesc, const TensorDesc& biasDesc)
{
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
        AddEdgeAndUpdatePeerDesc(*graph, reshapeNode, 0, biasAddNode, 0);
        AddEdgeAndUpdatePeerDesc(*graph, biasDataNode, 0, biasAddNode, 1);
        biasAddNode.UpdateInputDesc(0, reshapeOutDesc);
        biasAddNode.UpdateInputDesc(1, biasDesc);
    } else {
        biasAddNode = CompliantNodeBuilder(graph)
                          .OpType("Add")
                          .Name("add")
                          .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();
        int32_t reshapeIdx = reshapeIsFirstInput ? 0 : 1;
        int32_t biasIdx = reshapeIsFirstInput ? 1 : 0;
        AddEdgeAndUpdatePeerDesc(*graph, reshapeNode, 0, biasAddNode, reshapeIdx);
        AddEdgeAndUpdatePeerDesc(*graph, biasDataNode, 0, biasAddNode, biasIdx);
        biasAddNode.UpdateInputDesc(reshapeIdx, reshapeOutDesc);
        biasAddNode.UpdateInputDesc(biasIdx, biasDesc);
    }
    biasAddNode.UpdateOutputDesc(0, reshapeOutDesc);
    return biasAddNode;
}

std::shared_ptr<Graph> BuildMatMulReshapeBiasAddGraph(
    const std::string& name, const char* matmulOpType, const char* biasAddOpType, bool reshapeIsFirstInput,
    const std::vector<int64_t>& aDims, const std::vector<int64_t>& bDims, const std::vector<int64_t>& matmulOutDims,
    const std::vector<int64_t>& reshapeOutDims, const std::vector<int64_t>& biasDims, DataType dtype,
    DataType biasDtype = DT_UNDEFINED)
{
    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc(aDims, dtype);
    auto x2Desc = MakeTensorDesc(bDims, dtype);
    auto matmulOutDesc = MakeTensorDesc(matmulOutDims, dtype);
    auto reshapeOutDesc = MakeTensorDesc(reshapeOutDims, dtype);
    DataType actualBiasDtype = (biasDtype == DT_UNDEFINED) ? dtype : biasDtype;
    auto biasDesc = MakeTensorDesc(biasDims, actualBiasDtype);
    auto shapeDesc = MakeTensorDesc({static_cast<int64_t>(reshapeOutDims.size())}, DT_INT64);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", dtype, FORMAT_ND, aDims);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", dtype, FORMAT_ND, bDims);
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", actualBiasDtype, FORMAT_ND, biasDims);
    auto dataShape = graphBuilder.CreateInput(3, "dataShape", DT_INT64, FORMAT_ND,
                                              {static_cast<int64_t>(reshapeOutDims.size())});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    dataShape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

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
    matmulNode.UpdateOutputDesc(0, matmulOutDesc);
    bool transX1 = false;
    bool transX2 = false;
    bool isBatch = (strcmp(matmulOpType, "BatchMatMul") == 0 || strcmp(matmulOpType, "BatchMatMulV2") == 0);
    const char* transAttr1 = isBatch ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = isBatch ? "adj_x2" : "transpose_x2";
    matmulNode.SetAttr(transAttr1, transX1);
    matmulNode.SetAttr(transAttr2, transX2);

    auto reshapeNode = BuildReshapeNode(graph, matmulNode, *dataShape.GetProducer(), matmulOutDesc, reshapeOutDesc,
                                        shapeDesc);

    auto biasAddNode = BuildBiasAddOrAddNode(graph, biasAddOpType, reshapeNode, *dataBias.GetProducer(),
                                             reshapeIsFirstInput, reshapeOutDesc, biasDesc);

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

void CheckNodeRemoved(const std::shared_ptr<Graph>& graph, const char* nodeType)
{
    EXPECT_EQ(CountNodes(graph, nodeType), 0);
}

} // namespace

class MatMulReshapeBiasAddFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatformInfo950(); }

    static void TearDownTestCase() {}

    void SetUp() override { SetPlatformInfo950(); }

    void TearDown() override {}
};

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddFp16FusionSuccess)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeAddFp16", "MatMul", "Add", true, {16, 16}, {16, 1024},
                                                {16, 1024}, {1, 16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graph, "Add");
    CheckHasBiasAttr(graph, "MatMul", true);
    CheckInputCount(graph, "MatMul", 3U);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeBiasAddFp16FusionSuccess)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeBiasAddFp16", "MatMul", "BiasAdd", true, {16, 16},
                                                {16, 1024}, {16, 1024}, {1, 16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graph, "BiasAdd");
    CheckHasBiasAttr(graph, "MatMul", true);
    CheckInputCount(graph, "MatMul", 3U);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddReshapeSecondFp16FusionSuccess)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeAddReshapeSecondFp16", "MatMul", "Add", false, {16, 16},
                                                {16, 1024}, {16, 1024}, {1, 16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graph, "Add");
    CheckHasBiasAttr(graph, "MatMul", true);
    CheckInputCount(graph, "MatMul", 3U);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeBiasAddBf16FusionSuccess)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeBiasAddBf16", "MatMul", "BiasAdd", true, {4, 4}, {4, 4},
                                                {4, 4}, {1, 4, 4}, {4}, DT_BF16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graph, "BiasAdd");
    CheckHasBiasAttr(graph, "MatMul", true);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeBiasAddFp32FusionSuccess)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeBiasAddFp32", "MatMul", "BiasAdd", true, {8192, 1024},
                                                {1024, 1024}, {8192, 1024}, {16, 512, 1024}, {1024}, DT_FLOAT);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graph, "BiasAdd");
    CheckHasBiasAttr(graph, "MatMul", true);
    CheckInputCount(graph, "MatMul", 3U);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulV2ReshapeAddFp16FusionSuccess)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulV2ReshapeAddFp16", "MatMulV2", "Add", true, {16, 16}, {16, 1024},
                                                {16, 1024}, {1, 16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graph, "Add");
    CheckHasBiasAttr(graph, "MatMulV2", true);
    CheckInputCount(graph, "MatMulV2", 3U);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddWithDownstreamFusionSuccess)
{
    auto graphBuilder = EsGraphBuilder("matMulReshapeAddWithDownstream");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto matmulOutDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto reshapeOutDesc = MakeTensorDesc({1, 16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);
    auto shapeDesc = MakeTensorDesc({3}, DT_INT64);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    auto dataShape = graphBuilder.CreateInput(3, "dataShape", DT_INT64, FORMAT_ND, {3});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    dataShape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

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
    matmulNode.UpdateOutputDesc(0, matmulOutDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto reshapeNode = BuildReshapeNode(graph, matmulNode, *dataShape.GetProducer(), matmulOutDesc, reshapeOutDesc,
                                        shapeDesc);

    auto biasAddNode = BuildBiasAddOrAddNode(graph, "BiasAdd", reshapeNode, *dataBias.GetProducer(), true,
                                             reshapeOutDesc, biasDesc);

    auto reluNode = CompliantNodeBuilder(graph)
                        .OpType("Relu")
                        .Name("relu")
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    AddEdgeAndUpdatePeerDesc(*graph, biasAddNode, 0, reluNode, 0);
    reluNode.UpdateInputDesc(0, reshapeOutDesc);
    reluNode.UpdateOutputDesc(0, reshapeOutDesc);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reluNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graphPtr, "BiasAdd");
    EXPECT_EQ(CountNodes(graphPtr, "Relu"), 1);
    CheckHasBiasAttr(graphPtr, "MatMul", true);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddNon2DOutputFail)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeAddNon2DOutput", "MatMul", "Add", true, {16, 16, 16},
                                                {16, 16, 1024}, {16, 16, 1024}, {16, 16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddSplitLastDimFail)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeAddSplitLastDim", "MatMul", "Add", true, {16, 16},
                                                {16, 1024}, {16, 1024}, {16, 512, 2}, {2}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddBiasDimMismatchFail)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeAddBiasDimMismatch", "MatMul", "Add", true, {16, 16},
                                                {16, 1024}, {16, 1024}, {1, 16, 1024}, {512}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddBiasBroadcastFail)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeAddBiasBroadcastFail", "MatMul", "Add", true, {16, 16},
                                                {16, 1024}, {16, 1024}, {1, 16, 1024}, {16, 64}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddReshapeMultiOutputFail)
{
    auto graphBuilder = EsGraphBuilder("matMulReshapeAddReshapeMultiOutput");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto matmulOutDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto reshapeOutDesc = MakeTensorDesc({1, 16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);
    auto shapeDesc = MakeTensorDesc({3}, DT_INT64);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    auto dataShape = graphBuilder.CreateInput(3, "dataShape", DT_INT64, FORMAT_ND, {3});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    dataShape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

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
    matmulNode.UpdateOutputDesc(0, matmulOutDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto reshapeNode = BuildReshapeNode(graph, matmulNode, *dataShape.GetProducer(), matmulOutDesc, reshapeOutDesc,
                                        shapeDesc);

    auto biasAddNode = BuildBiasAddOrAddNode(graph, "BiasAdd", reshapeNode, *dataBias.GetProducer(), true,
                                             reshapeOutDesc, biasDesc);

    auto reluNode = CompliantNodeBuilder(graph)
                        .OpType("Relu")
                        .Name("relu")
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    AddEdgeAndUpdatePeerDesc(*graph, reshapeNode, 0, reluNode, 0);
    reluNode.UpdateInputDesc(0, reshapeOutDesc);
    reluNode.UpdateOutputDesc(0, reshapeOutDesc);

    auto biasAddOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(biasAddNode, 0));
    auto reluOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reluNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({biasAddOutput, reluOutput});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddUnknownShapeFail)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeAddUnknownShape", "MatMul", "Add", true, {-1, -1},
                                                {16, 1024}, {-1, 1024}, {1, -1, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddDiffBiasDtypeFusionSuccess)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeAddDiffBiasDtype", "MatMul", "Add", true, {16, 16},
                                                {16, 1024}, {16, 1024}, {1, 16, 1024}, {1024}, DT_FLOAT16, DT_FLOAT);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graph, "Add");
    CheckHasBiasAttr(graph, "MatMul", true);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddMatMulAlreadyHasBiasFail)
{
    auto graphBuilder = EsGraphBuilder("matMulReshapeAddMatMulAlreadyHasBias");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto matmulOutDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto reshapeOutDesc = MakeTensorDesc({1, 16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);
    auto shapeDesc = MakeTensorDesc({3}, DT_INT64);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    auto dataShape = graphBuilder.CreateInput(3, "dataShape", DT_INT64, FORMAT_ND, {3});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    dataShape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

    auto matmulNode = CompliantNodeBuilder(graph)
                          .OpType("MatMul")
                          .Name("matmul_with_bias")
                          .IrDefInputs(BuildMatMulIrInputs("MatMul"))
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs(BuildMatMulIrAttrs("MatMul"))
                          .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), matmulNode, 1);
    AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), matmulNode, 2);
    matmulNode.UpdateInputDesc(0, x1Desc);
    matmulNode.UpdateInputDesc(1, x2Desc);
    matmulNode.UpdateInputDesc(2, biasDesc);
    matmulNode.UpdateOutputDesc(0, matmulOutDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto reshapeNode = BuildReshapeNode(graph, matmulNode, *dataShape.GetProducer(), matmulOutDesc, reshapeOutDesc,
                                        shapeDesc);

    auto biasAddNode = BuildBiasAddOrAddNode(graph, "Add", reshapeNode, *dataBias.GetProducer(), true, reshapeOutDesc,
                                             biasDesc);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(biasAddNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeBiasAddBiasDimMismatchFail)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeBiasAddBiasDimMismatch", "MatMul", "BiasAdd", true,
                                                {16, 16}, {16, 1024}, {16, 1024}, {1, 16, 1024}, {512}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddNonMatMulTypeFail)
{
    auto graphBuilder = EsGraphBuilder("matMulReshapeAddNonMatMulType");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto mulOutDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto reshapeOutDesc = MakeTensorDesc({1, 16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);
    auto shapeDesc = MakeTensorDesc({3}, DT_INT64);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    auto dataShape = graphBuilder.CreateInput(3, "dataShape", DT_INT64, FORMAT_ND, {3});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    dataShape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

    auto mulNode = CompliantNodeBuilder(graph)
                       .OpType("Mul")
                       .Name("mul")
                       .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                       .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), mulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), mulNode, 1);
    mulNode.UpdateInputDesc(0, x1Desc);
    mulNode.UpdateInputDesc(1, x2Desc);
    mulNode.UpdateOutputDesc(0, mulOutDesc);

    auto reshapeNode = BuildReshapeNode(graph, mulNode, *dataShape.GetProducer(), mulOutDesc, reshapeOutDesc,
                                        shapeDesc);

    auto biasAddNode = BuildBiasAddOrAddNode(graph, "Add", reshapeNode, *dataBias.GetProducer(), true, reshapeOutDesc,
                                             biasDesc);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(biasAddNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddMatMulMultiOutputFail)
{
    auto graphBuilder = EsGraphBuilder("matMulReshapeAddMatMulMultiOutput");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto matmulOutDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto reshapeOutDesc = MakeTensorDesc({1, 16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);
    auto shapeDesc = MakeTensorDesc({3}, DT_INT64);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    auto dataShape = graphBuilder.CreateInput(3, "dataShape", DT_INT64, FORMAT_ND, {3});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    dataShape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

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
    matmulNode.UpdateOutputDesc(0, matmulOutDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto reshapeNode = BuildReshapeNode(graph, matmulNode, *dataShape.GetProducer(), matmulOutDesc, reshapeOutDesc,
                                        shapeDesc);

    auto biasAddNode = BuildBiasAddOrAddNode(graph, "BiasAdd", reshapeNode, *dataBias.GetProducer(), true,
                                             reshapeOutDesc, biasDesc);

    auto reluNode = CompliantNodeBuilder(graph)
                        .OpType("Relu")
                        .Name("relu")
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, reluNode, 0);
    reluNode.UpdateInputDesc(0, matmulOutDesc);
    reluNode.UpdateOutputDesc(0, matmulOutDesc);

    auto biasAddOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(biasAddNode, 0));
    auto reluOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reluNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({biasAddOutput, reluOutput});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddIdentityShapeFusionSuccess)
{
    auto graph = BuildMatMulReshapeBiasAddGraph("matMulReshapeAddIdentityShape", "MatMul", "Add", true, {16, 1024},
                                                {1024, 1024}, {16, 1024}, {16, 1024}, {1024}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graph, "Add");
    CheckHasBiasAttr(graph, "MatMul", true);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddReshapeOutputToOtherOpFail)
{
    auto graphBuilder = EsGraphBuilder("matMulReshapeAddReshapeOutputToOtherOp");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto matmulOutDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto reshapeOutDesc = MakeTensorDesc({1, 16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);
    auto shapeDesc = MakeTensorDesc({3}, DT_INT64);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    auto dataShape = graphBuilder.CreateInput(3, "dataShape", DT_INT64, FORMAT_ND, {3});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    dataShape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

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
    matmulNode.UpdateOutputDesc(0, matmulOutDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto reshapeNode = BuildReshapeNode(graph, matmulNode, *dataShape.GetProducer(), matmulOutDesc, reshapeOutDesc,
                                        shapeDesc);

    auto addNode = BuildBiasAddOrAddNode(graph, "Add", reshapeNode, *dataBias.GetProducer(), true, reshapeOutDesc,
                                         biasDesc);

    auto mulNode = CompliantNodeBuilder(graph)
                       .OpType("Mul")
                       .Name("mul")
                       .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                       .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();
    AddEdgeAndUpdatePeerDesc(*graph, reshapeNode, 0, mulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, reshapeNode, 0, mulNode, 1);
    mulNode.UpdateInputDesc(0, reshapeOutDesc);
    mulNode.UpdateInputDesc(1, reshapeOutDesc);
    mulNode.UpdateOutputDesc(0, reshapeOutDesc);

    auto addOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addNode, 0));
    auto mulOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({addOutput, mulOutput});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeAddBiasOutputToMultipleNodesFusionSuccess)
{
    auto graphBuilder = EsGraphBuilder("matMulReshapeAddBiasMultiConsumer");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto matmulOutDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto reshapeOutDesc = MakeTensorDesc({1, 16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);
    auto shapeDesc = MakeTensorDesc({3}, DT_INT64);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    auto dataShape = graphBuilder.CreateInput(3, "dataShape", DT_INT64, FORMAT_ND, {3});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    dataShape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

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
    matmulNode.UpdateOutputDesc(0, matmulOutDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto reshapeNode = BuildReshapeNode(graph, matmulNode, *dataShape.GetProducer(), matmulOutDesc, reshapeOutDesc,
                                        shapeDesc);

    auto addNode = BuildBiasAddOrAddNode(graph, "Add", reshapeNode, *dataBias.GetProducer(), true, reshapeOutDesc,
                                         biasDesc);

    auto mulNode = CompliantNodeBuilder(graph)
                       .OpType("Mul")
                       .Name("mul")
                       .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                       .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), mulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), mulNode, 1);
    mulNode.UpdateInputDesc(0, biasDesc);
    mulNode.UpdateInputDesc(1, biasDesc);
    mulNode.UpdateOutputDesc(0, biasDesc);

    auto addOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(addNode, 0));
    auto mulOutput = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(mulNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({addOutput, mulOutput});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graphPtr, "Add");
    CheckHasBiasAttr(graphPtr, "MatMul", true);
    CheckInputCount(graphPtr, "MatMul", 3U);
    EXPECT_EQ(CountNodes(graphPtr, "Mul"), 1);
}

TEST_F(MatMulReshapeBiasAddFusionPassTest, matMulReshapeBiasAddWithCtrlEdgesFusionSuccess)
{
    auto graphBuilder = EsGraphBuilder("matMulReshapeBiasAddCtrlEdges");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc({16, 16}, DT_FLOAT16);
    auto x2Desc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto matmulOutDesc = MakeTensorDesc({16, 1024}, DT_FLOAT16);
    auto reshapeOutDesc = MakeTensorDesc({1, 16, 1024}, DT_FLOAT16);
    auto biasDesc = MakeTensorDesc({1024}, DT_FLOAT16);
    auto shapeDesc = MakeTensorDesc({3}, DT_INT64);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT16, FORMAT_ND, {16, 16});
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT16, FORMAT_ND, {16, 1024});
    auto dataBias = graphBuilder.CreateInput(2, "dataBias", DT_FLOAT16, FORMAT_ND, {1024});
    auto dataShape = graphBuilder.CreateInput(3, "dataShape", DT_INT64, FORMAT_ND, {3});
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    dataShape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

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
    matmulNode.UpdateOutputDesc(0, matmulOutDesc);
    bool transX1 = false;
    bool transX2 = false;
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);

    auto reshapeNode = BuildReshapeNode(graph, matmulNode, *dataShape.GetProducer(), matmulOutDesc, reshapeOutDesc,
                                        shapeDesc);

    auto biasAddNode = BuildBiasAddOrAddNode(graph, "BiasAdd", reshapeNode, *dataBias.GetProducer(), true,
                                             reshapeOutDesc, biasDesc);

    auto reluNode = CompliantNodeBuilder(graph)
                        .OpType("Relu")
                        .Name("relu")
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    AddEdgeAndUpdatePeerDesc(*graph, biasAddNode, 0, reluNode, 0);
    reluNode.UpdateInputDesc(0, reshapeOutDesc);
    reluNode.UpdateOutputDesc(0, reshapeOutDesc);

    graph->AddControlEdge(*dataX2.GetProducer(), biasAddNode);
    graph->AddControlEdge(biasAddNode, reluNode);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reluNode, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulReshapeBiasAddFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    CheckNodeRemoved(graphPtr, "BiasAdd");
    CheckHasBiasAttr(graphPtr, "MatMul", true);

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
