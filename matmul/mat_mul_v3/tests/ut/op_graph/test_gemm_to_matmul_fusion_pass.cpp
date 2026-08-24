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
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/gemm_to_matmul_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace {

constexpr char kPassName[] = "GemmToMatmulFusionPass";

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
    platformInfo.str_info.short_soc_version = "Ascend950";
    PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

void SetPlatformInfo910B1()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_nd2nz"] = {"float16"};
    optionalInfo.soc_version = "Ascend910B1";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

void SetPlatformInfo310P()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 8;
    optionalInfo.soc_version = "Ascend310P1";
    platformInfo.str_info.short_soc_version = "Ascend310P";
    PlatformInfoManager::Instance().platform_info_map_["Ascend310P1"] = platformInfo;
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

std::shared_ptr<Graph> BuildGemmGraph(const std::string& name, const std::vector<int64_t>& aDims,
                                      const std::vector<int64_t>& bDims, const std::vector<int64_t>& cDims,
                                      const std::vector<int64_t>& outDims, DataType aDtype, DataType bDtype,
                                      DataType cDtype, DataType outDtype, bool transA = false, bool transB = false)
{
    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto aDesc = MakeTensorDesc(aDims, aDtype);
    auto bDesc = MakeTensorDesc(bDims, bDtype);
    auto cDesc = MakeTensorDesc(cDims, cDtype);
    auto alphaDesc = MakeTensorDesc({1}, outDtype);
    auto betaDesc = MakeTensorDesc({1}, outDtype);
    auto outDesc = MakeTensorDesc(outDims, outDtype);

    auto dataA = graphBuilder.CreateInput(0, "dataA", aDtype, FORMAT_ND, aDims);
    auto dataB = graphBuilder.CreateInput(1, "dataB", bDtype, FORMAT_ND, bDims);
    auto dataC = graphBuilder.CreateInput(2, "dataC", cDtype, FORMAT_ND, cDims);
    auto dataAlpha = graphBuilder.CreateInput(3, "dataAlpha", outDtype, FORMAT_ND, {1});
    auto dataBeta = graphBuilder.CreateInput(4, "dataBeta", outDtype, FORMAT_ND, {1});
    dataA.GetProducer()->UpdateOutputDesc(0, aDesc);
    dataB.GetProducer()->UpdateOutputDesc(0, bDesc);
    dataC.GetProducer()->UpdateOutputDesc(0, cDesc);
    dataAlpha.GetProducer()->UpdateOutputDesc(0, alphaDesc);
    dataBeta.GetProducer()->UpdateOutputDesc(0, betaDesc);

    auto gemmNode = CompliantNodeBuilder(graph)
                        .OpType("GEMM")
                        .Name(name.c_str())
                        .IrDefInputs({
                            {"a", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"b", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"c", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"alpha", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"beta", CompliantNodeBuilder::kEsIrInputRequired, ""},
                        })
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .IrDefAttrs({
                            {"transpose_a", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                            {"transpose_b", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                        })
                        .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *dataA.GetProducer(), dataA.GetProducerOutIndex(), gemmNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataB.GetProducer(), dataB.GetProducerOutIndex(), gemmNode, 1);
    AddEdgeAndUpdatePeerDesc(*graph, *dataC.GetProducer(), dataC.GetProducerOutIndex(), gemmNode, 2);
    AddEdgeAndUpdatePeerDesc(*graph, *dataAlpha.GetProducer(), dataAlpha.GetProducerOutIndex(), gemmNode, 3);
    AddEdgeAndUpdatePeerDesc(*graph, *dataBeta.GetProducer(), dataBeta.GetProducerOutIndex(), gemmNode, 4);
    gemmNode.UpdateInputDesc(0, aDesc);
    gemmNode.UpdateInputDesc(1, bDesc);
    gemmNode.UpdateInputDesc(2, cDesc);
    gemmNode.UpdateInputDesc(3, alphaDesc);
    gemmNode.UpdateInputDesc(4, betaDesc);
    gemmNode.UpdateOutputDesc(0, outDesc);
    gemmNode.SetAttr("transpose_a", transA);
    gemmNode.SetAttr("transpose_b", transB);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(gemmNode, 0));
    return graphBuilder.BuildAndReset({output});
}

void CheckMatMulV2Attrs(const std::shared_ptr<Graph>& graph, bool expectedTransX1, bool expectedTransX2)
{
    GNode matmulNode;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "MatMulV2", matmulNode));
    bool transX1 = false;
    bool transX2 = false;
    int64_t offsetX = -1;
    matmulNode.GetAttr("transpose_x1", transX1);
    matmulNode.GetAttr("transpose_x2", transX2);
    matmulNode.GetAttr("offset_x", offsetX);
    EXPECT_EQ(transX1, expectedTransX1);
    EXPECT_EQ(transX2, expectedTransX2);
    EXPECT_EQ(offsetX, 0);
}

} // namespace

class GemmToMatmulFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatformInfo950(); }

    static void TearDownTestCase() {}

    void SetUp() override { SetPlatformInfo950(); }

    void TearDown() override {}
};

TEST_F(GemmToMatmulFusionPassTest, patternTest)
{
    GemmToMatmulFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(GemmToMatmulFusionPassTest, unsupportedPlatform310PFail)
{
    SetPlatformInfo310P();

    auto graph = BuildGemmGraph("unsupportedPlatform310P", {16, 32}, {32, 16}, {16, 16}, {16, 16}, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(GemmToMatmulFusionPassTest, gemm910BFp16FusionSuccess)
{
    SetPlatformInfo910B1();

    auto graph = BuildGemmGraph("gemm910BFp16FusionSuccess", {16, 32}, {32, 16}, {16, 16}, {16, 16}, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    CheckMatMulV2Attrs(graph, false, false);
}

TEST_F(GemmToMatmulFusionPassTest, gemmFp16FusionSuccess)
{
    auto graph = BuildGemmGraph("gemmFp16FusionSuccess", {16, 32}, {32, 16}, {16, 16}, {16, 16}, DT_FLOAT16, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    CheckMatMulV2Attrs(graph, false, false);
}

TEST_F(GemmToMatmulFusionPassTest, gemmFp32FusionSuccess)
{
    auto graph = BuildGemmGraph("gemmFp32FusionSuccess", {16, 32}, {32, 16}, {16, 16}, {16, 16}, DT_FLOAT, DT_FLOAT,
                                DT_FLOAT, DT_FLOAT);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    CheckMatMulV2Attrs(graph, false, false);
}

TEST_F(GemmToMatmulFusionPassTest, gemmTransposeFp16FusionSuccess)
{
    auto graph = BuildGemmGraph("gemmTransposeFp16", {32, 16}, {16, 32}, {16, 16}, {16, 16}, DT_FLOAT16, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16, true, true);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    CheckMatMulV2Attrs(graph, true, true);
}

TEST_F(GemmToMatmulFusionPassTest, gemmInt8CastFp16FusionSuccess)
{
    auto graph = BuildGemmGraph("gemmInt8CastFp16", {16, 32}, {32, 16}, {16, 16}, {16, 16}, DT_INT8, DT_INT8, DT_FLOAT,
                                DT_FLOAT);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Cast"), 2);

    GNode matmulNode;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "MatMulV2", matmulNode));
    TensorDesc x1Desc;
    ASSERT_EQ(matmulNode.GetInputDesc(0, x1Desc), GRAPH_SUCCESS);
    EXPECT_EQ(x1Desc.GetDataType(), DT_FLOAT16);
    TensorDesc x2Desc;
    ASSERT_EQ(matmulNode.GetInputDesc(1, x2Desc), GRAPH_SUCCESS);
    EXPECT_EQ(x2Desc.GetDataType(), DT_FLOAT16);

    CheckMatMulV2Attrs(graph, false, false);
}

TEST_F(GemmToMatmulFusionPassTest, gemmDifferentShapeFp16FusionSuccess)
{
    auto graph = BuildGemmGraph("gemmDifferentShapeFp16", {128, 256}, {256, 512}, {128, 512}, {128, 512}, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    GNode addNode;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "Add", addNode));
    TensorDesc outDesc;
    ASSERT_EQ(addNode.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
}

TEST_F(GemmToMatmulFusionPassTest, gemmTransposeAFp16FusionSuccess)
{
    auto graph = BuildGemmGraph("gemmTransposeAFp16", {32, 16}, {32, 16}, {16, 16}, {16, 16}, DT_FLOAT16, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16, true, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    CheckMatMulV2Attrs(graph, true, false);
}

TEST_F(GemmToMatmulFusionPassTest, gemmTransposeBFp16FusionSuccess)
{
    auto graph = BuildGemmGraph("gemmTransposeBFp16", {16, 32}, {16, 32}, {16, 16}, {16, 16}, DT_FLOAT16, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16, false, true);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    CheckMatMulV2Attrs(graph, false, true);
}

TEST_F(GemmToMatmulFusionPassTest, gemmInt8Fp16NoCastFusionSuccess)
{
    auto graph = BuildGemmGraph("gemmInt8Fp16NoCast", {16, 32}, {32, 16}, {16, 16}, {16, 16}, DT_INT8, DT_INT8,
                                DT_FLOAT16, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Cast"), 0);
}

TEST_F(GemmToMatmulFusionPassTest, gemm910BFp32FusionSuccess)
{
    SetPlatformInfo910B1();

    auto graph = BuildGemmGraph("gemm910BFp32", {16, 32}, {32, 16}, {16, 16}, {16, 16}, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                                DT_FLOAT);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);

    CheckMatMulV2Attrs(graph, false, false);

    SetPlatformInfo950();
}

TEST_F(GemmToMatmulFusionPassTest, gemm910BInt8CastFp16FusionSuccess)
{
    SetPlatformInfo910B1();

    auto graph = BuildGemmGraph("gemm910BInt8CastFp16", {16, 32}, {32, 16}, {16, 16}, {16, 16}, DT_INT8, DT_INT8,
                                DT_FLOAT, DT_FLOAT);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Cast"), 2);

    GNode matmulNode;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "MatMulV2", matmulNode));
    TensorDesc x1Desc;
    ASSERT_EQ(matmulNode.GetInputDesc(0, x1Desc), GRAPH_SUCCESS);
    EXPECT_EQ(x1Desc.GetDataType(), DT_FLOAT16);
    TensorDesc x2Desc;
    ASSERT_EQ(matmulNode.GetInputDesc(1, x2Desc), GRAPH_SUCCESS);
    EXPECT_EQ(x2Desc.GetDataType(), DT_FLOAT16);

    SetPlatformInfo950();
}

TEST_F(GemmToMatmulFusionPassTest, gemmSmallShapeFp16FusionSuccess)
{
    auto graph = BuildGemmGraph("gemmSmallShapeFp16", {4, 4}, {4, 4}, {4, 4}, {4, 4}, DT_FLOAT16, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
}

TEST_F(GemmToMatmulFusionPassTest, gemmInt32FusionSuccess)
{
    auto graph = BuildGemmGraph("gemmInt32", {16, 32}, {32, 16}, {16, 16}, {16, 16}, DT_INT32, DT_INT32, DT_INT32,
                                DT_INT32);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_EQ(CountNodes(graph, "Mul"), 2);
    EXPECT_EQ(CountNodes(graph, "Add"), 1);
    EXPECT_EQ(CountNodes(graph, "Cast"), 0);
}

TEST_F(GemmToMatmulFusionPassTest, gemmInvalidDimASkipFusion)
{
    auto graph = BuildGemmGraph("gemmInvalidDimA", {16}, {32, 16}, {16, 16}, {16, 16}, DT_FLOAT16, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 1);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 0);
    EXPECT_EQ(CountNodes(graph, "Mul"), 0);
}

TEST_F(GemmToMatmulFusionPassTest, gemmInvalidDimBSkipFusion)
{
    auto graph = BuildGemmGraph("gemmInvalidDimB", {16, 32}, {16, 32, 8}, {16, 16}, {16, 16}, DT_FLOAT16, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 1);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 0);
    EXPECT_EQ(CountNodes(graph, "Mul"), 0);
}

TEST_F(GemmToMatmulFusionPassTest, gemmInvalidDimCSkipFusion)
{
    auto graph = BuildGemmGraph("gemmInvalidDimC", {16, 32}, {32, 16}, {48}, {16, 16}, DT_FLOAT16, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 1);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 0);
    EXPECT_EQ(CountNodes(graph, "Mul"), 0);
}

TEST_F(GemmToMatmulFusionPassTest, gemmInt8OutputSkipFusion)
{
    auto graph = BuildGemmGraph("gemmInt8OutputSkip", {16, 32}, {32, 16}, {16, 16}, {16, 16}, DT_FLOAT16, DT_FLOAT16,
                                DT_FLOAT16, DT_INT8);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    GemmToMatmulFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "GEMM"), 1);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 0);
    EXPECT_EQ(CountNodes(graph, "Mul"), 0);
}
