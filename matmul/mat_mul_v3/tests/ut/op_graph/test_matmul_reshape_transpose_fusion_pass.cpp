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
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/matmul_reshape_transpose_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace {

constexpr char kPassName[] = "MatmulReshapeTransposeFusionPass";

void SetPlatformInfo910B()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    optionalInfo.soc_version = "Ascend910B1";
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
    platformInfo.str_info.short_soc_version = "Ascend910B";
    PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

void SetPlatformInfo910BNoL0c2Out()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    optionalInfo.soc_version = "Ascend910B1";
    // no Intrinsic_fix_pipe_l0c2out
    platformInfo.str_info.short_soc_version = "Ascend910B";
    PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platformInfo;
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

GNode BuildTransposeDNode(Graph* graph, const std::string& name, const std::vector<int64_t>& perm)
{
    auto node = CompliantNodeBuilder(graph)
                    .OpType("TransposeD")
                    .Name(name.c_str())
                    .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({
                        {"perm", CompliantNodeBuilder::kEsAttrRequired, "ListInt", CreateFrom(perm)},
                    })
                    .Build();
    return node;
}

GNode BuildReshapeNode(Graph* graph, const std::string& name)
{
    auto node = CompliantNodeBuilder(graph)
                    .OpType("Reshape")
                    .Name(name.c_str())
                    .IrDefInputs({
                        {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                        {"shape", CompliantNodeBuilder::kEsIrInputRequired, ""},
                    })
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();
    return node;
}

GNode BuildMatMulV2Node(Graph* graph, const std::string& name)
{
    auto node = CompliantNodeBuilder(graph)
                    .OpType("MatMulV2")
                    .Name(name.c_str())
                    .IrDefInputs({
                        {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
                    })
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({
                        {"transpose_x1", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                        {"transpose_x2", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                    })
                    .Build();
    return node;
}

// 构建完整的融合前图：TransposeD -> Reshape -> MatMulV2 -> Reshape -> TransposeD
// dataX1 [M, K] 是 matmul 的另一个输入
// dataX2 [N, K, 1] -> TransposeD(perm=[1,0,2]) -> [K, N, 1] -> Reshape -> [K, N] -> MatMulV2(x2)
// MatMulV2(x1=[M,K], x2=[K,N]) -> [M, N] -> Reshape -> [M, N, 1] -> TransposeD(perm=[1,0,2]) -> [N, M, 1]
std::shared_ptr<Graph> BuildFusionGraph(const std::string& name, const std::vector<int64_t>& x1Dims,
                                        const std::vector<int64_t>& x2Dims, DataType dtype, bool transX1 = false,
                                        bool transX2 = false)
{
    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc(x1Dims, dtype);
    auto x2Desc = MakeTensorDesc(x2Dims, dtype);

    // 计算各节点 shape
    // x2: [N, K, 1] -> TransposeD(perm=[1,0,2]) -> [K, N, 1]
    std::vector<int64_t> trans1OutDims = {x2Dims[1], x2Dims[0], x2Dims[2]};
    // -> Reshape -> [K, N]
    std::vector<int64_t> reshape1OutDims = {trans1OutDims[0], trans1OutDims[1]};
    // MatMulV2(x1=[M,K], x2=[K,N]) -> [M, N]
    std::vector<int64_t> mmOutDims = {x1Dims[0], reshape1OutDims[1]};
    // -> Reshape -> [M, N, 1]
    std::vector<int64_t> reshape2OutDims = {mmOutDims[0], mmOutDims[1], 1};
    // -> TransposeD(perm=[1,0,2]) -> [N, M, 1]
    std::vector<int64_t> trans2OutDims = {reshape2OutDims[1], reshape2OutDims[0], reshape2OutDims[2]};

    // 创建输入节点
    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", dtype, FORMAT_ND, x1Dims);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", dtype, FORMAT_ND, x2Dims);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    // 创建前置 TransposeD
    auto preTranspose = BuildTransposeDNode(graph, name + "_pre_transpose", {1, 0, 2});
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), preTranspose, 0);
    preTranspose.UpdateInputDesc(0, x2Desc);
    preTranspose.UpdateOutputDesc(0, MakeTensorDesc(trans1OutDims, dtype));

    // 创建前置 Reshape（需要 const shape 输入）
    auto preReshape = BuildReshapeNode(graph, name + "_pre_reshape");
    AddEdgeAndUpdatePeerDesc(*graph, preTranspose, 0, preReshape, 0);
    preReshape.UpdateInputDesc(0, MakeTensorDesc(trans1OutDims, dtype));
    preReshape.UpdateOutputDesc(0, MakeTensorDesc(reshape1OutDims, dtype));
    // 创建 const shape 输入
    auto preShapeConst = graphBuilder.CreateConst(reshape1OutDims, {static_cast<int64_t>(reshape1OutDims.size())});
    preShapeConst.GetProducer()->UpdateOutputDesc(
        0, MakeTensorDesc({static_cast<int64_t>(reshape1OutDims.size())}, DT_INT64));
    AddEdgeAndUpdatePeerDesc(*graph, *preShapeConst.GetProducer(), preShapeConst.GetProducerOutIndex(), preReshape, 1);
    TensorDesc preShapeConstDesc = MakeTensorDesc({static_cast<int64_t>(reshape1OutDims.size())}, DT_INT64);
    preReshape.UpdateInputDesc(1, preShapeConstDesc);

    // 创建 MatMulV2
    auto matmulV2 = BuildMatMulV2Node(graph, name + "_matmul");
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulV2, 0);
    AddEdgeAndUpdatePeerDesc(*graph, preReshape, 0, matmulV2, 1);
    matmulV2.UpdateInputDesc(0, x1Desc);
    matmulV2.UpdateInputDesc(1, MakeTensorDesc(reshape1OutDims, dtype));
    matmulV2.UpdateOutputDesc(0, MakeTensorDesc(mmOutDims, dtype));
    bool transX1Val = transX1;
    bool transX2Val = transX2;
    matmulV2.SetAttr("transpose_x1", transX1Val);
    matmulV2.SetAttr("transpose_x2", transX2Val);

    // 创建后置 Reshape
    auto suffReshape = BuildReshapeNode(graph, name + "_suff_reshape");
    AddEdgeAndUpdatePeerDesc(*graph, matmulV2, 0, suffReshape, 0);
    suffReshape.UpdateInputDesc(0, MakeTensorDesc(mmOutDims, dtype));
    suffReshape.UpdateOutputDesc(0, MakeTensorDesc(reshape2OutDims, dtype));
    auto suffShapeConst = graphBuilder.CreateConst(reshape2OutDims, {static_cast<int64_t>(reshape2OutDims.size())});
    suffShapeConst.GetProducer()->UpdateOutputDesc(
        0, MakeTensorDesc({static_cast<int64_t>(reshape2OutDims.size())}, DT_INT64));
    AddEdgeAndUpdatePeerDesc(*graph, *suffShapeConst.GetProducer(), suffShapeConst.GetProducerOutIndex(), suffReshape,
                             1);
    suffReshape.UpdateInputDesc(1, MakeTensorDesc({static_cast<int64_t>(reshape2OutDims.size())}, DT_INT64));

    // 创建后置 TransposeD
    auto suffTranspose = BuildTransposeDNode(graph, name + "_suff_transpose", {1, 0, 2});
    AddEdgeAndUpdatePeerDesc(*graph, suffReshape, 0, suffTranspose, 0);
    suffTranspose.UpdateInputDesc(0, MakeTensorDesc(reshape2OutDims, dtype));
    suffTranspose.UpdateOutputDesc(0, MakeTensorDesc(trans2OutDims, dtype));

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(suffTranspose, 0));
    return graphBuilder.BuildAndReset({output});
}

} // namespace

class MatmulReshapeTransposeFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatformInfo910B(); }
    static void TearDownTestCase() {}
    void SetUp() override { SetPlatformInfo910B(); }
    void TearDown() override {}
};

// L0-001: pattern 测试（验证 pass 能正确加载）
TEST_F(MatmulReshapeTransposeFusionPassTest, fusionSuccessFp32)
{
    auto graph = BuildFusionGraph("fusionSuccessFp32", {32, 1523}, {100, 1523, 1}, DT_FLOAT);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatmulReshapeTransposeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);

    // 验证：TransposeD 被删除
    EXPECT_EQ(CountNodes(graph, "TransposeD"), 0);
    // 验证：Reshape 保留
    EXPECT_EQ(CountNodes(graph, "Reshape"), 2);
    // 验证：MatMulV2 保留
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);

    // 验证：MatMulV2 的 transpose_x2 = true
    GNode matmulNode;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "MatMulV2", matmulNode));
    bool transX2 = false;
    ASSERT_EQ(matmulNode.GetAttr("transpose_x2", transX2), GRAPH_SUCCESS);
    EXPECT_TRUE(transX2);
}

// L0-002: 不支持的平台（无 Intrinsic_fix_pipe_l0c2out）
TEST_F(MatmulReshapeTransposeFusionPassTest, unsupportedPlatformFail)
{
    SetPlatformInfo910BNoL0c2Out();
    auto graph = BuildFusionGraph("unsupportedPlatform", {32, 1523}, {100, 1523, 1}, DT_FLOAT);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatmulReshapeTransposeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// L0-003: dtype 不是 float（不融合）
TEST_F(MatmulReshapeTransposeFusionPassTest, unsupportedDtypeFp16Fail)
{
    auto graph = BuildFusionGraph("unsupportedDtypeFp16", {32, 1523}, {100, 1523, 1}, DT_FLOAT16);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatmulReshapeTransposeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// L0-004: transpose_x1 = true（不融合）
TEST_F(MatmulReshapeTransposeFusionPassTest, transX1TrueFail)
{
    auto graph = BuildFusionGraph("transX1True", {32, 1523}, {100, 1523, 1}, DT_FLOAT, true, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatmulReshapeTransposeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// L0-005: transpose_x2 = true（不融合）
TEST_F(MatmulReshapeTransposeFusionPassTest, transX2TrueFail)
{
    auto graph = BuildFusionGraph("transX2True", {32, 1523}, {100, 1523, 1}, DT_FLOAT, false, true);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatmulReshapeTransposeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// L1-001: 不同 shape 的融合成功
TEST_F(MatmulReshapeTransposeFusionPassTest, fusionSuccessDifferentShape)
{
    auto graph = BuildFusionGraph("fusionSuccessDiffShape", {64, 256}, {128, 256, 1}, DT_FLOAT);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatmulReshapeTransposeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "TransposeD"), 0);
    EXPECT_EQ(CountNodes(graph, "MatMulV2"), 1);

    GNode matmulNode;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "MatMulV2", matmulNode));
    bool transX2 = false;
    matmulNode.GetAttr("transpose_x2", transX2);
    EXPECT_TRUE(transX2);
}

// L1-002: MatMul 类型（非 MatMulV2）也能融合
TEST_F(MatmulReshapeTransposeFusionPassTest, fusionSuccessMatMulType)
{
    auto graphBuilder = EsGraphBuilder("fusionSuccessMatMul");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    std::vector<int64_t> x1Dims = {32, 1523};
    std::vector<int64_t> x2Dims = {100, 1523, 1};
    auto x1Desc = MakeTensorDesc(x1Dims, DT_FLOAT);
    auto x2Desc = MakeTensorDesc(x2Dims, DT_FLOAT);

    std::vector<int64_t> trans1Out = {x2Dims[1], x2Dims[0], 1};
    std::vector<int64_t> reshape1Out = {trans1Out[0], trans1Out[1]};
    std::vector<int64_t> mmOut = {x1Dims[0], reshape1Out[1]};
    std::vector<int64_t> reshape2Out = {mmOut[0], mmOut[1], 1};
    std::vector<int64_t> trans2Out = {reshape2Out[1], reshape2Out[0], 1};

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", DT_FLOAT, FORMAT_ND, x1Dims);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", DT_FLOAT, FORMAT_ND, x2Dims);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);

    auto preTranspose = BuildTransposeDNode(graph, "pre_trans", {1, 0, 2});
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), preTranspose, 0);
    preTranspose.UpdateInputDesc(0, x2Desc);
    preTranspose.UpdateOutputDesc(0, MakeTensorDesc(trans1Out, DT_FLOAT));

    auto preReshape = BuildReshapeNode(graph, "pre_reshape");
    AddEdgeAndUpdatePeerDesc(*graph, preTranspose, 0, preReshape, 0);
    preReshape.UpdateInputDesc(0, MakeTensorDesc(trans1Out, DT_FLOAT));
    preReshape.UpdateOutputDesc(0, MakeTensorDesc(reshape1Out, DT_FLOAT));
    auto preShapeConst = graphBuilder.CreateConst(reshape1Out, {static_cast<int64_t>(reshape1Out.size())});
    preShapeConst.GetProducer()->UpdateOutputDesc(0, MakeTensorDesc({2}, DT_INT64));
    AddEdgeAndUpdatePeerDesc(*graph, *preShapeConst.GetProducer(), preShapeConst.GetProducerOutIndex(), preReshape, 1);
    preReshape.UpdateInputDesc(1, MakeTensorDesc({2}, DT_INT64));

    // 用 MatMul 而不是 MatMulV2
    auto matmulNode = CompliantNodeBuilder(graph)
                          .OpType("MatMul")
                          .Name("matmul")
                          .IrDefInputs({
                              {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
                          })
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs({
                              {"transpose_x1", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                              {"transpose_x2", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                          })
                          .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, preReshape, 0, matmulNode, 1);
    matmulNode.UpdateInputDesc(0, x1Desc);
    matmulNode.UpdateInputDesc(1, MakeTensorDesc(reshape1Out, DT_FLOAT));
    matmulNode.UpdateOutputDesc(0, MakeTensorDesc(mmOut, DT_FLOAT));

    auto suffReshape = BuildReshapeNode(graph, "suff_reshape");
    AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, suffReshape, 0);
    suffReshape.UpdateInputDesc(0, MakeTensorDesc(mmOut, DT_FLOAT));
    suffReshape.UpdateOutputDesc(0, MakeTensorDesc(reshape2Out, DT_FLOAT));
    auto suffShapeConst = graphBuilder.CreateConst(reshape2Out, {static_cast<int64_t>(reshape2Out.size())});
    suffShapeConst.GetProducer()->UpdateOutputDesc(0, MakeTensorDesc({3}, DT_INT64));
    AddEdgeAndUpdatePeerDesc(*graph, *suffShapeConst.GetProducer(), suffShapeConst.GetProducerOutIndex(), suffReshape,
                             1);
    suffReshape.UpdateInputDesc(1, MakeTensorDesc({3}, DT_INT64));

    auto suffTranspose = BuildTransposeDNode(graph, "suff_trans", {1, 0, 2});
    AddEdgeAndUpdatePeerDesc(*graph, suffReshape, 0, suffTranspose, 0);
    suffTranspose.UpdateInputDesc(0, MakeTensorDesc(reshape2Out, DT_FLOAT));
    suffTranspose.UpdateOutputDesc(0, MakeTensorDesc(trans2Out, DT_FLOAT));

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(suffTranspose, 0));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatmulReshapeTransposeFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graphPtr, "TransposeD"), 0);
    EXPECT_EQ(CountNodes(graphPtr, "MatMul"), 1);

    GNode mmNode;
    ASSERT_TRUE(FindFirstNodeByOpType(graphPtr, "MatMul", mmNode));
    bool transX2 = false;
    mmNode.GetAttr("transpose_x2", transX2);
    EXPECT_TRUE(transX2);
}
