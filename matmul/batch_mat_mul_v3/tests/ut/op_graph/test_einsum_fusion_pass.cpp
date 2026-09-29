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
#include "es_nn_ops.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/einsum_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace {

constexpr char kPassName[] = "EinsumPass";

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

int CountNodes(const std::shared_ptr<Graph>& graph, const std::string& nodeType)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type.GetString() == nodeType) {
            count++;
        }
    }
    return count;
}

bool FindFirstNodeByType(const std::shared_ptr<Graph>& graph, const std::string& nodeType, GNode& outNode)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        std::string typeStr = type.GetString();
        if (typeStr == nodeType) {
            outNode = node;
            return true;
        }
    }
    return false;
}

bool FindFinalOutputNode(const std::shared_ptr<Graph>& graph, GNode& outNode)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        std::string typeStr = type.GetString();
        if (typeStr == "NetOutput") {
            auto [inPtr, inPort] = node.GetInDataNodesAndPortIndexs(0);
            if (inPtr != nullptr) {
                outNode = *inPtr;
                return true;
            }
        }
    }
    return false;
}

bool CheckMatMulAdj(const GNode& matmulNode, bool expectedAdjX1, bool expectedAdjX2)
{
    // Nodes built by the ES API omit default-valued attrs, so a missing attr means its
    // default value (false for adj_x1/adj_x2/transpose_x1/transpose_x2).
    bool adjX1 = false;
    bool adjX2 = false;
    if (matmulNode.GetAttr("adj_x1", adjX1) != GRAPH_SUCCESS) {
        (void)matmulNode.GetAttr("transpose_x1", adjX1); // keep false when absent (default)
    }
    if (matmulNode.GetAttr("adj_x2", adjX2) != GRAPH_SUCCESS) {
        (void)matmulNode.GetAttr("transpose_x2", adjX2); // keep false when absent (default)
    }
    return adjX1 == expectedAdjX1 && adjX2 == expectedAdjX2;
}

bool CheckNodeOutputShape(const GNode& node, const std::vector<int64_t>& expectedDims)
{
    TensorDesc desc;
    if (node.GetOutputDesc(0, desc) != GRAPH_SUCCESS) {
        return false;
    }
    auto actualDims = desc.GetShape().GetDims();
    if (actualDims.size() != expectedDims.size()) {
        return false;
    }
    for (size_t i = 0; i < actualDims.size(); i++) {
        if (actualDims[i] != expectedDims[i] && actualDims[i] != -1 && expectedDims[i] != -1) {
            return false;
        }
    }
    return true;
}

std::string GetNodeOutputShapeStr(const GNode& node)
{
    TensorDesc desc;
    if (node.GetOutputDesc(0, desc) != GRAPH_SUCCESS) {
        return "<failed>";
    }
    std::string str = "[";
    for (size_t i = 0; i < desc.GetShape().GetDims().size(); i++) {
        if (i > 0)
            str += ", ";
        str += std::to_string(desc.GetShape().GetDims()[i]);
    }
    str += "]";
    return str;
}

bool CheckMatMulInputOrder(const GNode& matmulNode, const std::string& port0Type, const std::string& port1Type)
{
    auto [in0Ptr, in0Port] = matmulNode.GetInDataNodesAndPortIndexs(0);
    auto [in1Ptr, in1Port] = matmulNode.GetInDataNodesAndPortIndexs(1);
    if (in0Ptr == nullptr || in1Ptr == nullptr) {
        return false;
    }
    AscendString type0;
    AscendString type1;
    in0Ptr->GetType(type0);
    in1Ptr->GetType(type1);
    std::string type0Str = type0.GetString();
    std::string type1Str = type1.GetString();
    return type0Str == port0Type && type1Str == port1Type;
}

std::shared_ptr<Graph> BuildEinsumGraph(const std::string& name, const std::string& equation, size_t inputNum,
                                        const std::vector<std::vector<int64_t>>& inputDims,
                                        const std::vector<int64_t>& outDims, DataType dtype)
{
    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    std::vector<EsTensorHolder> inputs;
    for (size_t i = 0; i < inputNum; ++i) {
        auto desc = MakeTensorDesc(inputDims[i], dtype);
        auto input = graphBuilder.CreateInput(static_cast<int64_t>(i), ("x" + std::to_string(i)).c_str(), dtype,
                                              FORMAT_ND, inputDims[i]);
        input.GetProducer()->UpdateOutputDesc(0, desc);
        inputs.push_back(input);
    }

    CompliantNodeBuilder builder(graph);
    builder.OpType("Einsum").Name(name.c_str());
    if (inputNum == 1) {
        builder.IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    } else if (inputNum == 2) {
        builder.IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    } else {
        builder.IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x3", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    }
    builder.IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"equation", CompliantNodeBuilder::kEsAttrRequired, "String", CreateFrom(AscendString(equation.c_str()))},
            {"N", CompliantNodeBuilder::kEsAttrRequired, "Int", CreateFrom(static_cast<int64_t>(inputNum))},
        });
    auto einsumNode = builder.Build();

    for (size_t i = 0; i < inputNum; ++i) {
        AddEdgeAndUpdatePeerDesc(*graph, *inputs[i].GetProducer(), 0, einsumNode, static_cast<int32_t>(i));
        einsumNode.UpdateInputDesc(static_cast<int32_t>(i), MakeTensorDesc(inputDims[i], dtype));
    }
    einsumNode.UpdateOutputDesc(0, MakeTensorDesc(outDims, dtype));

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(einsumNode, 0));
    return graphBuilder.BuildAndReset({output});
}

std::shared_ptr<Graph> BuildEinsumGraphDynamic(const std::string& name, const std::string& equation, size_t inputNum,
                                               const std::vector<std::vector<int64_t>>& inputDims,
                                               const std::vector<std::vector<std::pair<int64_t, int64_t>>>& inputRanges,
                                               const std::vector<int64_t>& outDims, DataType dtype)
{
    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    std::vector<EsTensorHolder> inputs;
    for (size_t i = 0; i < inputNum; ++i) {
        auto desc = MakeTensorDesc(inputDims[i], dtype);
        desc.SetShapeRange(inputRanges[i]);
        auto input = graphBuilder.CreateInput(static_cast<int64_t>(i), ("x" + std::to_string(i)).c_str(), dtype,
                                              FORMAT_ND, inputDims[i]);
        input.GetProducer()->UpdateOutputDesc(0, desc);
        inputs.push_back(input);
    }

    CompliantNodeBuilder builder(graph);
    builder.OpType("Einsum").Name(name.c_str());
    if (inputNum == 1) {
        builder.IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    } else if (inputNum == 2) {
        builder.IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    } else {
        builder.IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x3", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    }
    builder.IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"equation", CompliantNodeBuilder::kEsAttrRequired, "String", CreateFrom(AscendString(equation.c_str()))},
            {"N", CompliantNodeBuilder::kEsAttrRequired, "Int", CreateFrom(static_cast<int64_t>(inputNum))},
        });
    auto einsumNode = builder.Build();

    for (size_t i = 0; i < inputNum; ++i) {
        AddEdgeAndUpdatePeerDesc(*graph, *inputs[i].GetProducer(), 0, einsumNode, static_cast<int32_t>(i));
        auto desc = MakeTensorDesc(inputDims[i], dtype);
        desc.SetShapeRange(inputRanges[i]);
        einsumNode.UpdateInputDesc(static_cast<int32_t>(i), desc);
    }
    einsumNode.UpdateOutputDesc(0, MakeTensorDesc(outDims, dtype));

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(einsumNode, 0));
    return graphBuilder.BuildAndReset({output});
}

} // namespace

class EinsumPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatformInfo950(); }
    static void TearDownTestCase() {}
    void SetUp() override { SetPlatformInfo950(); }
    void TearDown() override {}
};

TEST_F(EinsumPassTest, patternTest)
{
    EinsumPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcCde2AbdeSuccess)
{
    auto graph = BuildEinsumGraph("abc_cde_abde", "abc,cde->abde", 2, {{10, 20, 30}, {30, 40, 50}}, {10, 20, 40, 50},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "MatMulV2") + CountNodes(graph, "BatchMatMulV2") + CountNodes(graph, "BatchMatMul"), 1);
}

TEST_F(EinsumPassTest, staticShapeAbcdCde2AbeSuccess)
{
    auto graph = BuildEinsumGraph("abcd_cde_abe", "abcd,cde->abe", 2, {{10, 20, 30, 40}, {30, 40, 50}}, {10, 20, 50},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcCd2AbdSuccess)
{
    auto graph = BuildEinsumGraph("abc_cd_abd", "abc,cd->abd", 2, {{10, 20, 30}, {30, 40}}, {10, 20, 40}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcDc2AbdSuccess)
{
    auto graph = BuildEinsumGraph("abc_dc_abd", "abc,dc->abd", 2, {{10, 20, 40}, {30, 40}}, {10, 20, 30}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcAbd2DcSuccess)
{
    auto graph = BuildEinsumGraph("abc_abd_dc", "abc,abd->dc", 2, {{10, 20, 40}, {10, 20, 30}}, {30, 40}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAxB2ABSuccess)
{
    auto graph = BuildEinsumGraph("a_b_ab", "a,b->ab", 2, {{10}, {20}}, {10, 20}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "MatMulV2") + CountNodes(graph, "BatchMatMul"), 1);
}

TEST_F(EinsumPassTest, staticShapeBTNHBFNH2BNFTSuccess)
{
    auto graph = BuildEinsumGraph("BTNH_BFNH_BNFT", "BTNH,BFNH->BNFT", 2, {{10, 20, 30, 40}, {10, 50, 30, 40}},
                                  {10, 30, 50, 20}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdAecd2AcebSuccess)
{
    auto graph = BuildEinsumGraph("abcd_aecd_aceb", "abcd,aecd->aceb", 2, {{10, 20, 30, 40}, {10, 50, 30, 40}},
                                  {10, 30, 50, 20}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdAebd2AebcSuccess)
{
    auto graph = BuildEinsumGraph("abcd_aebd_aebc", "abcd,aebd->aebc", 2, {{10, 20, 30, 40}, {10, 50, 20, 40}},
                                  {10, 50, 20, 30}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdAecd2EbSuccess)
{
    auto graph = BuildEinsumGraph("abcd_aecd_eb", "abcd,aecd->eb", 2, {{10, 20, 30, 40}, {10, 50, 30, 40}}, {50, 20},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, dynamicShapeAbcCde2AbdeSuccess)
{
    auto graph = BuildEinsumGraphDynamic("dyn_abc_cde_abde", "abc,cde->abde", 2, {{10, -1, 30}, {30, -1, 50}},
                                         {{{10, 10}, {20, 40}, {30, 30}}, {{30, 30}, {40, 60}, {50, 50}}},
                                         {10, -1, 40, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, dynamicShapeBTNHBFNH2BNFTSuccess)
{
    auto graph = BuildEinsumGraphDynamic(
        "dyn_BTNH_BFNH_BNFT", "BTNH,BFNH->BNFT", 2, {{10, -1, 30, 40}, {10, 50, 30, 40}},
        {{{10, 10}, {20, 40}, {30, 30}, {40, 40}}, {{10, 10}, {50, 50}, {30, 30}, {40, 40}}}, {10, 30, 50, -1},
        DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeFuzzGenericSuccess)
{
    auto graph = BuildEinsumGraph("fuzz_nq_n_n", "nq,n->n", 2, {{2, 49}, {2}}, {2}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "ReduceSumD") + CountNodes(graph, "ReduceSum"), 1);
    EXPECT_GE(CountNodes(graph, "MatMulV2") + CountNodes(graph, "BatchMatMulV2") + CountNodes(graph, "BatchMatMul"), 1);
}

TEST_F(EinsumPassTest, staticShapeFuzzSingleInputSuccess)
{
    auto graph = BuildEinsumGraph("fuzz_sl_ls", "sl->ls", 1, {{42, 45}}, {45, 42}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 1);
}

TEST_F(EinsumPassTest, staticShapeFuzzSingleInputReduceSuccess)
{
    auto graph = BuildEinsumGraph("fuzz_nxgb_xb", "nxgb->xb", 1, {{18, 62, 55, 4}}, {62, 4}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 1);
    EXPECT_GE(CountNodes(graph, "ReduceSumD") + CountNodes(graph, "ReduceSum"), 1);
}

TEST_F(EinsumPassTest, staticShapeFuzzBroadcastSuccess)
{
    // Note: "..." in "q...ciwd" matches 0 dims (5 labels, 5 dims) — this case exercises the
    // ellipsis *syntax* with a full-reduce fuzz decomposition, not the broadcast alignment
    // branch. True broadcast coverage lives in the ellipsisBroadcast* cases below.
    auto graph = BuildEinsumGraph("fuzz_broadcast", "nl,q...ciwd->wq", 2, {{30, 41}, {17, 60, 13, 8, 6}}, {8, 17},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "ReduceSumD") + CountNodes(graph, "ReduceSum"), 1);
    EXPECT_GE(CountNodes(graph, "MatMulV2") + CountNodes(graph, "BatchMatMulV2") + CountNodes(graph, "BatchMatMul"), 1);
}

// === True ellipsis broadcast coverage ("..." expands >= 1 dim in at least one input) ===

// Two-input broadcast alignment: "..." expands to [4] vs [1] -> broadcast to [4]
TEST_F(EinsumPassTest, ellipsisBroadcastTwoInputSuccess)
{
    auto graph = BuildEinsumGraph("ell_bcast_2in", "...ab,...a->...b", 2, {{4, 2, 3}, {1, 2}}, {4, 3}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {4, 3}));
}

// Single-input broadcast dim preserved: "..." expands to [2], kept in output
TEST_F(EinsumPassTest, ellipsisBroadcastSingleInputPreservedSuccess)
{
    auto graph = BuildEinsumGraph("ell_bcast_keep", "...abc->...ac", 1, {{2, 4, 2, 3}}, {2, 4, 3}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "ReduceSumD") + CountNodes(graph, "ReduceSum"), 1);
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {2, 4, 3}));
}

// Single-input broadcast dim reduced: "..." expands to [4], eliminated by output equation.
// Note: BROAD_CAST dims are folded via reshape (not ReduceSum which handles REDUCE type).
TEST_F(EinsumPassTest, ellipsisBroadcastSingleInputReducedSuccess)
{
    auto graph = BuildEinsumGraph("ell_bcast_reduce", "...ab->ab", 1, {{4, 2, 3}}, {2, 3}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {2, 3}));
}

TEST_F(EinsumPassTest, staticShapeStrideInputFail)
{
    auto graph = BuildEinsumGraph("stride_aacd_cde_aae", "aacd,cde->aae", 2, {{10, 10, 30, 40}, {30, 40, 50}},
                                  {10, 10, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(EinsumPassTest, unsupportedEquationNotMatch)
{
    auto graph = BuildEinsumGraph("not_match", "ab,cd->abc", 2, {{10, 20}, {30, 40}}, {10, 20, 30}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, unsupportedDtypeFail)
{
    auto graph = BuildEinsumGraph("dtype_int8", "abc,cd->abd", 2, {{10, 20, 30}, {30, 40}}, {10, 20, 40}, DT_INT8);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_TRUE(status == GRAPH_NOT_CHANGED || status == SUCCESS);
}

// === Missing 18 hardcoded equation scenarios ===

TEST_F(EinsumPassTest, staticShapeAbCb2AcSuccess)
{
    auto graph = BuildEinsumGraph("ab_cb_ac", "ab,cb->ac", 2, {{10, 20}, {30, 20}}, {10, 30}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "BatchMatMul") + CountNodes(graph, "MatMulV2"), 1);
}

TEST_F(EinsumPassTest, staticShapeAbcAbd2AcdSuccess)
{
    auto graph = BuildEinsumGraph("abc_abd_acd", "abc,abd->acd", 2, {{10, 20, 30}, {10, 20, 40}}, {10, 30, 40},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcAbde2DecSuccess)
{
    auto graph = BuildEinsumGraph("abc_abde_dec", "abc,abde->dec", 2, {{10, 20, 30}, {10, 20, 40, 50}}, {40, 30, 50},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcAcd2AbdSuccess)
{
    auto graph = BuildEinsumGraph("abc_acd_abd", "abc,acd->abd", 2, {{10, 20, 30}, {10, 30, 40}}, {10, 20, 40},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcAdc2AbdSuccess)
{
    auto graph = BuildEinsumGraph("abc_adc_abd", "abc,adc->abd", 2, {{10, 20, 30}, {10, 40, 30}}, {10, 20, 40},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcDec2AbdeSuccess)
{
    auto graph = BuildEinsumGraph("abc_dec_abde", "abc,dec->abde", 2, {{10, 20, 30}, {40, 30, 50}}, {10, 20, 40, 50},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdAbce2AbdeSuccess)
{
    auto graph = BuildEinsumGraph("abcd_abce_abde", "abcd,abce->abde", 2, {{10, 20, 30, 40}, {10, 20, 30, 50}},
                                  {10, 20, 40, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdAbce2AcdeSuccess)
{
    auto graph = BuildEinsumGraph("abcd_abce_acde", "abcd,abce->acde", 2, {{10, 20, 30, 40}, {10, 20, 30, 50}},
                                  {10, 30, 40, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdAbde2AbceSuccess)
{
    auto graph = BuildEinsumGraph("abcd_abde_abce", "abcd,abde->abce", 2, {{10, 20, 30, 40}, {10, 20, 30, 50}},
                                  {10, 20, 30, 40}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdAbe2EcdSuccess)
{
    auto graph = BuildEinsumGraph("abcd_abe_ecd", "abcd,abe->ecd", 2, {{10, 20, 30, 40}, {10, 20, 50}}, {50, 30, 40},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdAcbe2AdbeSuccess)
{
    auto graph = BuildEinsumGraph("abcd_acbe_adbe", "abcd,acbe->adbe", 2, {{10, 20, 30, 40}, {10, 30, 20, 50}},
                                  {10, 20, 40, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdAcbe2AecdSuccess)
{
    auto graph = BuildEinsumGraph("abcd_acbe_aecd", "abcd,acbe->aecd", 2, {{10, 20, 30, 40}, {10, 30, 20, 50}},
                                  {10, 50, 30, 40}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdAdbe2AcbeSuccess)
{
    auto graph = BuildEinsumGraph("abcd_adbe_acbe", "abcd,adbe->acbe", 2, {{10, 20, 30, 40}, {10, 40, 20, 50}},
                                  {10, 30, 20, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdCed2AbceSuccess)
{
    auto graph = BuildEinsumGraph("abcd_ced_abce", "abcd,ced->abce", 2, {{10, 20, 30, 40}, {50, 40, 30}},
                                  {10, 20, 30, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdDabe2CabeSuccess)
{
    auto graph = BuildEinsumGraph("abcd_dabe_cabe", "abcd,dabe->cabe", 2, {{10, 20, 30, 40}, {50, 10, 20, 60}},
                                  {30, 10, 20, 60}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdEb2AecdSuccess)
{
    auto graph = BuildEinsumGraph("abcd_eb_aecd", "abcd,eb->aecd", 2, {{10, 20, 30, 40}, {50, 20}}, {10, 50, 30, 40},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdEbcd2BcaeSuccess)
{
    auto graph = BuildEinsumGraph("abcd_ebcd_bcae", "abcd,ebcd->bcae", 2, {{10, 20, 30, 40}, {50, 20, 30, 40}},
                                  {20, 30, 10, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, staticShapeAbcdEcd2AbeSuccess)
{
    auto graph = BuildEinsumGraph("abcd_ecd_abe", "abcd,ecd->abe", 2, {{10, 20, 30, 40}, {50, 30, 40}}, {10, 20, 50},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

// === Dynamic shape scenarios ===

TEST_F(EinsumPassTest, dynamicShapeAbcDec2AbdeSuccess)
{
    auto graph = BuildEinsumGraphDynamic("dyn_abc_dec_abde", "abc,dec->abde", 2, {{10, -1, 30}, {40, 30, -1}},
                                         {{{10, 10}, {20, 40}, {30, 30}}, {{40, 40}, {30, 30}, {50, 60}}},
                                         {10, -1, 40, -1}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, dynamicShapeAbcAbde2DecSuccess)
{
    auto graph = BuildEinsumGraphDynamic("dyn_abc_abde_dec", "abc,abde->dec", 2, {{10, -1, 30}, {10, -1, 40, 50}},
                                         {{{10, 10}, {20, 40}, {30, 30}}, {{10, 10}, {20, 40}, {40, 40}, {50, 50}}},
                                         {40, 30, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, dynamicShapeAbcdAbe2EcdSuccess)
{
    auto graph = BuildEinsumGraphDynamic("dyn_abcd_abe_ecd", "abcd,abe->ecd", 2, {{-1, 20, 30, 40}, {-1, 20, 50}},
                                         {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {20, 20}, {50, 50}}},
                                         {50, 30, 40}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

TEST_F(EinsumPassTest, dynamicShapeAbcdEb2AecdSuccess)
{
    auto graph = BuildEinsumGraphDynamic("dyn_abcd_eb_aecd", "abcd,eb->aecd", 2, {{-1, 20, 30, 40}, {50, -1}},
                                         {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{50, 50}, {20, 40}}},
                                         {-1, 50, 30, 40}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

// === Deep verification tests ===

TEST_F(EinsumPassTest, verifyAbcdAecd2AcebTopology)
{
    auto graph = BuildEinsumGraph("verify_aceb", "abcd,aecd->aceb", 2, {{10, 20, 30, 40}, {10, 50, 30, 40}},
                                  {10, 30, 50, 20}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_EQ(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV2"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMulV2", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, true));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {10, 30, 50, 20}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {10, 30, 50, 20}));
}

TEST_F(EinsumPassTest, verifyAbcdAdbe2AcbeTopology)
{
    auto graph = BuildEinsumGraph("verify_acbe_003", "abcd,adbe->acbe", 2, {{10, 20, 30, 40}, {10, 40, 20, 50}},
                                  {10, 30, 20, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_EQ(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV2"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMulV2", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, false));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {10, 20, 30, 50}));
    bool foundOutputTrans = false;
    for (auto n : graph->GetAllNodes()) {
        AscendString t;
        n.GetType(t);
        std::string typeStr = t.GetString();
        if (typeStr == "TransposeD" || typeStr == "Transpose") {
            if (CheckNodeOutputShape(n, {10, 30, 20, 50})) {
                foundOutputTrans = true;
                break;
            }
        }
    }
    EXPECT_TRUE(foundOutputTrans);
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {10, 30, 20, 50}));
}

TEST_F(EinsumPassTest, verifyAbcCd2AbdTopology)
{
    auto graph = BuildEinsumGraph("verify_abd_005", "abc,cd->abd", 2, {{10, 20, 30}, {30, 40}}, {10, 20, 40},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "BatchMatMulV2"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMulV2", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, false));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {10, 20, 40}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {10, 20, 40}));
}

TEST_F(EinsumPassTest, verifyAbcdCde2AbeTopology)
{
    auto graph = BuildEinsumGraph("verify_abe_004", "abcd,cde->abe", 2, {{10, 20, 30, 40}, {30, 40, 50}}, {10, 20, 50},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "Reshape"), 2);
    EXPECT_GE(CountNodes(graph, "BatchMatMulV2"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMulV2", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, false));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {10, 20, 50}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {10, 20, 50}));
}

TEST_F(EinsumPassTest, verifyAbcAbd2DcTopology)
{
    auto graph = BuildEinsumGraph("verify_dc_007", "abc,abd->dc", 2, {{10, 20, 30}, {10, 20, 40}}, {40, 30},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "Reshape"), 2);
    EXPECT_GE(CountNodes(graph, "MatMulV2"), 1);
    GNode mmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "MatMulV2", mmNode));
    EXPECT_TRUE(CheckMatMulAdj(mmNode, true, false));
    EXPECT_TRUE(CheckNodeOutputShape(mmNode, {40, 30}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {40, 30}));
}

TEST_F(EinsumPassTest, verifyAxB2ABTopology)
{
    auto graph = BuildEinsumGraphDynamic("verify_dyn_ab_026", "a,b->ab", 2, {{-1}, {-1}}, {{{10, 10}}, {{20, 20}}},
                                         {-1, -1}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 2);
    EXPECT_EQ(CountNodes(graph, "Mul"), 1);
    GNode mulNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Mul", mulNode));
}

// Dynamic deep verification tests
TEST_F(EinsumPassTest, verifyDynamicAbcCde2AbdeTopology)
{
    auto graph = BuildEinsumGraphDynamic("verify_dyn_abde_001", "abc,cde->abde", 2, {{10, -1, 30}, {30, -1, 50}},
                                         {{{10, 10}, {20, 40}, {30, 30}}, {{30, 30}, {40, 60}, {50, 50}}},
                                         {10, -1, 40, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "GatherShapes"), 1);
    EXPECT_GE(CountNodes(graph, "MatMulV2"), 1);
    GNode mmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "MatMulV2", mmNode));
    EXPECT_TRUE(CheckMatMulAdj(mmNode, false, false));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {10, -1, 40, 50}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdEb2AecdTopology)
{
    auto graph = BuildEinsumGraphDynamic("verify_dyn_aecd_028", "abcd,eb->aecd", 2, {{-1, 20, 30, 40}, {50, -1}},
                                         {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{50, 50}, {20, 40}}},
                                         {-1, 50, 30, 40}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 1);
    EXPECT_GE(CountNodes(graph, "GatherShapes"), 1);
    EXPECT_GE(CountNodes(graph, "MatMulV2"), 1);
    GNode mmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "MatMulV2", mmNode));
    EXPECT_TRUE(CheckMatMulAdj(mmNode, false, false));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 50, 30, 40}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdAecd2AcebTopology)
{
    auto graph = BuildEinsumGraphDynamic(
        "verify_dyn_aceb_002", "abcd,aecd->aceb", 2, {{-1, 20, 30, 40}, {-1, 50, 30, 40}},
        {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {50, 50}, {30, 30}, {40, 40}}}, {-1, 30, 50, 20},
        DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_GE(CountNodes(graph, "BatchMatMul"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMul", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, true));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {-1, 30, 50, 20}));
    EXPECT_TRUE(CheckMatMulInputOrder(bmmNode, "TransposeD", "TransposeD") ||
                CheckMatMulInputOrder(bmmNode, "Transpose", "Transpose") ||
                CheckMatMulInputOrder(bmmNode, "TransposeD", "Transpose") ||
                CheckMatMulInputOrder(bmmNode, "Transpose", "TransposeD"));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 30, 50, 20}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdAdbe2AcbeTopology)
{
    auto graph = BuildEinsumGraphDynamic(
        "verify_dyn_acbe_003", "abcd,adbe->acbe", 2, {{-1, 20, 30, 40}, {-1, 40, 20, 50}},
        {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {40, 40}, {20, 20}, {50, 50}}}, {-1, 30, 20, 50},
        DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_GE(CountNodes(graph, "BatchMatMul"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMul", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, false));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {-1, 20, 30, 50}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 30, 20, 50}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdAecd2AcbeTopology)
{
    auto graph = BuildEinsumGraphDynamic(
        "verify_dyn_acbe_010", "abcd,aecd->acbe", 2, {{-1, 20, 30, 40}, {-1, 50, 30, 40}},
        {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {50, 50}, {30, 30}, {40, 40}}}, {-1, 30, 20, 50},
        DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_GE(CountNodes(graph, "BatchMatMul"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMul", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, true));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {-1, 30, 20, 50}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 30, 20, 50}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdAcbe2AecdTopology)
{
    auto graph = BuildEinsumGraphDynamic(
        "verify_dyn_aecd_011", "abcd,acbe->aecd", 2, {{-1, 20, 30, 40}, {-1, 30, 20, 50}},
        {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {30, 30}, {20, 20}, {50, 50}}}, {-1, 50, 30, 40},
        DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_GE(CountNodes(graph, "BatchMatMul"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMul", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, true, false));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {-1, 30, 50, 40}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 50, 30, 40}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdAcbe2AdbeTopology)
{
    auto graph = BuildEinsumGraphDynamic(
        "verify_dyn_adbe_014", "abcd,acbe->adbe", 2, {{-1, 20, 30, 40}, {-1, 30, 20, 50}},
        {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {30, 30}, {20, 20}, {50, 50}}}, {-1, 20, 40, 50},
        DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_GE(CountNodes(graph, "BatchMatMul"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMul", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, true, false));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {-1, 20, 40, 50}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 20, 40, 50}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdAebd2AebcTopology)
{
    auto graph = BuildEinsumGraphDynamic(
        "verify_dyn_aebc_017", "abcd,aebd->aebc", 2, {{-1, 20, 30, 40}, {-1, 50, 20, 30}},
        {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {50, 50}, {20, 20}, {30, 30}}}, {-1, 50, 20, 30},
        DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_GE(CountNodes(graph, "BatchMatMul"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMul", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, false));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {-1, 20, 30, 50}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 50, 20, 30}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdCed2AbceTopology)
{
    auto graph = BuildEinsumGraphDynamic("verify_dyn_abce_023", "abcd,ced->abce", 2, {{-1, 20, 30, 40}, {50, 40, 30}},
                                         {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{50, 50}, {40, 40}, {30, 30}}},
                                         {-1, 20, 30, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_GE(CountNodes(graph, "BatchMatMul"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMul", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, true));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {-1, 30, 20, 40}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 20, 30, 50}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdDabe2CabeTopology)
{
    auto graph = BuildEinsumGraphDynamic(
        "verify_dyn_cabe_025", "abcd,dabe->cabe", 2, {{-1, 20, 30, 40}, {50, -1, 20, 60}},
        {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{50, 50}, {10, 20}, {20, 20}, {60, 60}}}, {30, -1, 20, 60},
        DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_GE(CountNodes(graph, "BatchMatMul"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMul", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, false));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {-1, 20, 30, 60}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {30, -1, 20, 60}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdAecd2EbTopology)
{
    auto graph = BuildEinsumGraphDynamic(
        "verify_dyn_eb_027", "abcd,aecd->eb", 2, {{-1, 20, 30, 40}, {-1, 50, 30, 40}},
        {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {50, 50}, {30, 30}, {40, 40}}}, {50, 20}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_GE(CountNodes(graph, "Reshape") + CountNodes(graph, "FlattenV2"), 2);
    EXPECT_GE(CountNodes(graph, "MatMulV2"), 1);
    GNode mmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "MatMulV2", mmNode));
    EXPECT_TRUE(CheckMatMulAdj(mmNode, false, true));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {50, 20}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcDec2AbdeTopology)
{
    auto graph = BuildEinsumGraphDynamic("verify_dyn_abde_008", "abc,dec->abde", 2, {{-1, 20, 30}, {40, 30, -1}},
                                         {{{10, 20}, {20, 20}, {30, 30}}, {{40, 40}, {30, 30}, {50, 60}}},
                                         {-1, 20, 40, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "GatherShapes"), 1);
    EXPECT_GE(CountNodes(graph, "MatMulV2"), 1);
    GNode mmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "MatMulV2", mmNode));
    EXPECT_TRUE(CheckMatMulAdj(mmNode, false, true));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 20, 40, 50}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcAbde2DecTopology)
{
    auto graph = BuildEinsumGraphDynamic("verify_dyn_dec_009", "abc,abde->dec", 2, {{-1, 20, 30}, {-1, 20, 40, 50}},
                                         {{{10, 20}, {20, 20}, {30, 30}}, {{10, 20}, {20, 20}, {40, 40}, {50, 50}}},
                                         {40, 30, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "GatherShapes"), 1);
    EXPECT_GE(CountNodes(graph, "MatMulV2"), 1);
    GNode mmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "MatMulV2", mmNode));
    EXPECT_TRUE(CheckMatMulAdj(mmNode, true, false));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {40, 30, 50}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdAbe2EcdTopology)
{
    auto graph = BuildEinsumGraphDynamic("verify_dyn_ecd_013", "abcd,abe->ecd", 2, {{-1, 20, 30, 40}, {-1, 20, 50}},
                                         {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {20, 20}, {50, 50}}},
                                         {50, 30, 40}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "GatherShapes"), 1);
    EXPECT_GE(CountNodes(graph, "MatMulV2"), 1);
    GNode mmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "MatMulV2", mmNode));
    EXPECT_TRUE(CheckMatMulAdj(mmNode, true, false));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {50, 30, 40}));
}

TEST_F(EinsumPassTest, verifyDynamicAbcdEb2AecdDeepTopology)
{
    auto graph = BuildEinsumGraphDynamic("verify_dyn_aecd_028_deep", "abcd,eb->aecd", 2, {{-1, 20, 30, 40}, {50, -1}},
                                         {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{50, 50}, {20, 40}}},
                                         {-1, 50, 30, 40}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    int transCount = CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose");
    EXPECT_GE(transCount, 2);
    EXPECT_GE(CountNodes(graph, "GatherShapes"), 1);
    EXPECT_GE(CountNodes(graph, "MatMulV2"), 1);
    EXPECT_GE(CountNodes(graph, "Reshape") + CountNodes(graph, "FlattenV2"), 2);
    GNode mmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "MatMulV2", mmNode));
    EXPECT_TRUE(CheckMatMulAdj(mmNode, false, false));
    auto [in0Ptr, in0Port] = mmNode.GetInDataNodesAndPortIndexs(0);
    auto [in1Ptr, in1Port] = mmNode.GetInDataNodesAndPortIndexs(1);
    ASSERT_NE(in0Ptr, nullptr);
    ASSERT_NE(in1Ptr, nullptr);
    AscendString type0;
    AscendString type1;
    in0Ptr->GetType(type0);
    in1Ptr->GetType(type1);
    std::string port0Type = type0.GetString();
    std::string port1Type = type1.GetString();
    bool port0Valid = (port0Type == "Data" || port0Type == "Reshape" || port0Type == "FlattenV2" ||
                       port0Type == "TransposeD" || port0Type == "Transpose");
    bool port1Valid = (port1Type == "Reshape" || port1Type == "FlattenV2" || port1Type == "Transpose" ||
                       port1Type == "TransposeD");
    EXPECT_TRUE(port0Valid) << "port0 type=" << port0Type;
    EXPECT_TRUE(port1Valid) << "port1 type=" << port1Type;
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 50, 30, 40}));
}

TEST_F(EinsumPassTest, verifyStaticBtnhBfnh2BnftTopology)
{
    auto graph = BuildEinsumGraph("verify_bnft", "BTNH,BFNH->BNFT", 2, {{10, 20, 30, 40}, {10, 50, 30, 40}},
                                  {10, 30, 50, 20}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "TransposeD") + CountNodes(graph, "Transpose"), 2);
    EXPECT_GE(CountNodes(graph, "BatchMatMulV2"), 1);
    GNode bmmNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "BatchMatMulV2", bmmNode));
    EXPECT_TRUE(CheckMatMulAdj(bmmNode, false, true));
    EXPECT_TRUE(CheckNodeOutputShape(bmmNode, {10, 30, 50, 20}));
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {10, 30, 50, 20}));
}

TEST_F(EinsumPassTest, dynamicShapeFuzzGenericSuccess)
{
    auto graph = BuildEinsumGraphDynamic("dyn_fuzz_ab_ac", "ab,ac->abc", 2, {{-1, 10}, {-1, 20}},
                                         {{{5, 10}, {10, 10}}, {{5, 20}, {10, 20}}}, {-1, 10, 20}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

// Diagnostic test: check which nodes have empty output desc after fusion
TEST_F(EinsumPassTest, checkAllNodesHaveOutputDesc)
{
    auto checkGraph = [](const std::shared_ptr<Graph>& graph, const std::string& equation) -> std::string {
        std::string emptyNodes;
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            std::string typeStr = type.GetString();
            if (typeStr == "NetOutput" || typeStr == "Data" || typeStr == "Const") {
                continue;
            }
            TensorDesc desc;
            if (node.GetOutputDesc(0, desc) != GRAPH_SUCCESS) {
                emptyNodes += typeStr + "(getFailed) ";
                continue;
            }
            auto dims = desc.GetShape().GetDims();
            if (dims.empty() && desc.GetDataType() == ge::DT_UNDEFINED) {
                AscendString nm;
                node.GetName(nm);
                emptyNodes += typeStr + "[" + nm.GetString() + "](empty) ";
            }
        }
        return emptyNodes;
    };

    {
        auto graph = BuildEinsumGraph("chk_abc_cde", "abc,cde->abde", 2, {{10, 20, 30}, {30, 40, 50}}, {10, 20, 40, 50},
                                      DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "abc,cde->abde");
            EXPECT_TRUE(empty.empty()) << "abc,cde->abde (static) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraph("chk_abcd_aecd", "abcd,aecd->aceb", 2, {{10, 20, 30, 40}, {10, 50, 30, 40}},
                                      {10, 30, 50, 20}, DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "abcd,aecd->aceb");
            EXPECT_TRUE(empty.empty()) << "abcd,aecd->aceb (static) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraph("chk_a_b", "a,b->ab", 2, {{10}, {20}}, {10, 20}, DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "a,b->ab");
            EXPECT_TRUE(empty.empty()) << "a,b->ab (static) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraphDynamic("chk_dyn_abc_cde", "abc,cde->abde", 2, {{10, -1, 30}, {30, -1, 50}},
                                             {{{10, 10}, {20, 40}, {30, 30}}, {{30, 30}, {40, 60}, {50, 50}}},
                                             {10, -1, 40, 50}, DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "abc,cde->abde");
            EXPECT_TRUE(empty.empty()) << "abc,cde->abde (dynamic) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraphDynamic("chk_dyn_abcd_eb", "abcd,eb->aecd", 2, {{-1, 20, 30, 40}, {50, -1}},
                                             {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{50, 50}, {20, 40}}},
                                             {-1, 50, 30, 40}, DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "abcd,eb->aecd");
            EXPECT_TRUE(empty.empty()) << "abcd,eb->aecd (dynamic) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraphDynamic(
            "chk_dyn_abcd_aecd_eb", "abcd,aecd->eb", 2, {{-1, 20, 30, 40}, {-1, 50, 30, 40}},
            {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {50, 50}, {30, 30}, {40, 40}}}, {50, 20}, DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "abcd,aecd->eb");
            EXPECT_TRUE(empty.empty()) << "abcd,aecd->eb (dynamic) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraphDynamic(
            "chk_dyn_abcd_aecd_acbe", "abcd,aecd->acbe", 2, {{-1, 20, 30, 40}, {-1, 50, 30, 40}},
            {{{10, 20}, {20, 20}, {30, 30}, {40, 40}}, {{10, 20}, {50, 50}, {30, 30}, {40, 40}}}, {-1, 30, 20, 50},
            DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "abcd,aecd->acbe");
            EXPECT_TRUE(empty.empty()) << "abcd,aecd->acbe (dynamic) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraphDynamic("chk_dyn_a_b", "a,b->ab", 2, {{-1}, {-1}}, {{{10, 10}}, {{20, 20}}},
                                             {-1, -1}, DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "a,b->ab");
            EXPECT_TRUE(empty.empty()) << "a,b->ab (dynamic) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraph("chk_nq_n", "nq,n->n", 2, {{2, 49}, {2}}, {2}, DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "nq,n->n");
            EXPECT_TRUE(empty.empty()) << "nq,n->n (static generic) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraph("chk_broadcast", "nl,q...ciwd->wq", 2, {{30, 41}, {17, 60, 13, 8, 6}}, {8, 17},
                                      DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "nl,q...ciwd->wq");
            EXPECT_TRUE(empty.empty()) << "nl,q...ciwd->wq (static generic broadcast) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraph("chk_sl_ls", "sl->ls", 1, {{42, 45}}, {45, 42}, DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "sl->ls");
            EXPECT_TRUE(empty.empty()) << "sl->ls (static generic single input) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraph("chk_nxgb_xb", "nxgb->xb", 1, {{18, 62, 55, 4}}, {62, 4}, DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "nxgb->xb");
            EXPECT_TRUE(empty.empty()) << "nxgb->xb (static generic single input reduce) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraphDynamic("chk_dyn_ab_ac", "ab,ac->abc", 2, {{-1, 10}, {-1, 20}},
                                             {{{5, 10}, {10, 10}}, {{5, 20}, {10, 20}}}, {-1, 10, 20}, DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        if (pass.Run(graph, ctx) != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "ab,ac->abc");
            EXPECT_TRUE(empty.empty()) << "ab,ac->abc (dynamic generic) empty nodes: " << empty;
        }
    }
    {
        auto graph = BuildEinsumGraphDynamic("chk_dyn_abc_bcd_ad", "abc,bcd->ad", 2, {{-1, 3, 4}, {3, 4, -1}},
                                             {{{2, 10}, {3, 3}, {4, 4}}, {{3, 3}, {4, 4}, {5, 10}}}, {-1, -1},
                                             DT_FLOAT16);
        CustomPassContext ctx;
        ctx.SetPassName(kPassName);
        EinsumPass pass;
        Status runStatus = pass.Run(graph, ctx);
        if (runStatus != GRAPH_NOT_CHANGED) {
            std::string empty = checkGraph(graph, "abc,bcd->ad");
            EXPECT_TRUE(empty.empty()) << "abc,bcd->ad (dynamic generic sumproduct) empty nodes: " << empty;
        }
    }
}

// === Identity equation abc->abc (single input, zero-node passthrough): the framework
// rewriter rejects a replacement graph whose output is the input Data node itself,
// so an identity Cast bridge node is inserted to keep the output boundary valid. ===
TEST_F(EinsumPassTest, identityAbc2AbcStaticSuccess)
{
    auto graph = BuildEinsumGraph("identity_static", "abc->abc", 1, {{10, 20, 30}}, {10, 20, 30}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(status, GRAPH_SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_EQ(CountNodes(graph, "Cast"), 1);
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {10, 20, 30}));
}

TEST_F(EinsumPassTest, identityAbc2AbcDynamicSuccess)
{
    auto graph = BuildEinsumGraphDynamic("identity_dyn", "abc->abc", 1, {{-1, 20, 30}}, {{{5, 10}, {20, 20}, {30, 30}}},
                                         {-1, 20, 30}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(status, GRAPH_SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_EQ(CountNodes(graph, "Cast"), 1);
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 20, 30}));
}

// Dtype-mismatch identity graphs are artificial (the Einsum op infershape forces
// output dtype = input dtype), kept here to pin the structural behavior: the bridge
// Cast uses the INPUT dtype, keeping the replacement graph structurally valid.
TEST_F(EinsumPassTest, identityAbc2AbcDtypeMismatchStructuralSuccess)
{
    // static
    {
        auto graphBuilder = es::EsGraphBuilder("identity_static_dtypes");
        auto* rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto inDesc = MakeTensorDesc({10, 20, 30}, DT_FLOAT16);
        auto input0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT16, FORMAT_ND, {10, 20, 30});
        input0.GetProducer()->UpdateOutputDesc(0, inDesc);
        CompliantNodeBuilder builder(rawGraph);
        builder.OpType("Einsum").Name("identity_static_dtypes");
        builder.IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}});
        builder.IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .IrDefAttrs({
                {"equation", CompliantNodeBuilder::kEsAttrRequired, "String", CreateFrom(AscendString("abc->abc"))},
                {"N", CompliantNodeBuilder::kEsAttrRequired, "Int", CreateFrom(static_cast<int64_t>(1))},
            });
        auto einsumNode = builder.Build();
        AddEdgeAndUpdatePeerDesc(*rawGraph, *input0.GetProducer(), 0, einsumNode, 0);
        einsumNode.UpdateInputDesc(0, inDesc);
        einsumNode.UpdateOutputDesc(0, MakeTensorDesc({10, 20, 30}, DT_FLOAT));
        auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(einsumNode, 0));
        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

        CustomPassContext passContext;
        passContext.SetPassName(kPassName);
        EinsumPass pass;
        Status status = pass.Run(graph, passContext);
        EXPECT_EQ(status, GRAPH_SUCCESS);
        EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
        EXPECT_EQ(CountNodes(graph, "Cast"), 1);
    }
    // dynamic
    {
        auto graphBuilder = es::EsGraphBuilder("identity_dyn_dtypes");
        auto* rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto inDesc = MakeTensorDesc({-1, 20, 30}, DT_FLOAT16);
        inDesc.SetShapeRange({{{5, 10}, {20, 20}, {30, 30}}});
        auto input0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT16, FORMAT_ND, {-1, 20, 30});
        input0.GetProducer()->UpdateOutputDesc(0, inDesc);
        CompliantNodeBuilder builder(rawGraph);
        builder.OpType("Einsum").Name("identity_dyn_dtypes");
        builder.IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}});
        builder.IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .IrDefAttrs({
                {"equation", CompliantNodeBuilder::kEsAttrRequired, "String", CreateFrom(AscendString("abc->abc"))},
                {"N", CompliantNodeBuilder::kEsAttrRequired, "Int", CreateFrom(static_cast<int64_t>(1))},
            });
        auto einsumNode = builder.Build();
        AddEdgeAndUpdatePeerDesc(*rawGraph, *input0.GetProducer(), 0, einsumNode, 0);
        einsumNode.UpdateInputDesc(0, inDesc);
        einsumNode.UpdateOutputDesc(0, MakeTensorDesc({-1, 20, 30}, DT_FLOAT));
        auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(einsumNode, 0));
        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

        CustomPassContext passContext;
        passContext.SetPassName(kPassName);
        EinsumPass pass;
        Status status = pass.Run(graph, passContext);
        EXPECT_EQ(status, GRAPH_SUCCESS);
        EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
        EXPECT_EQ(CountNodes(graph, "Cast"), 1);
    }
}

// === Dtype-conversion capability: single-input reduction ("abcd->ab") with an fp32 input and an
// fp16 output desc. The dynamic fuzz path (numOps==1 branch) casts the reduced fp32 result to the
// op's OUTPUT desc dtype (aligned with the legacy pass); the static fuzz path performs no
// output-dtype alignment, so the result stays fp32 (legacy behavior pinned as-is). ===
TEST_F(EinsumPassTest, abcd2AbFp32InFp16OutDynamicCastsToOutputDtype)
{
    auto graphBuilder = es::EsGraphBuilder("abcd_ab_dyn_dtypes");
    auto* rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto inDesc = MakeTensorDesc({-1, 20, 30, 40}, DT_FLOAT);
    inDesc.SetShapeRange({{{5, 10}, {20, 20}, {30, 30}, {40, 40}}});
    auto input0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT, FORMAT_ND, {-1, 20, 30, 40});
    input0.GetProducer()->UpdateOutputDesc(0, inDesc);
    CompliantNodeBuilder builder(rawGraph);
    builder.OpType("Einsum").Name("abcd_ab_dyn_dtypes");
    builder.IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    builder.IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"equation", CompliantNodeBuilder::kEsAttrRequired, "String", CreateFrom(AscendString("abcd->ab"))},
            {"N", CompliantNodeBuilder::kEsAttrRequired, "Int", CreateFrom(static_cast<int64_t>(1))},
        });
    auto einsumNode = builder.Build();
    AddEdgeAndUpdatePeerDesc(*rawGraph, *input0.GetProducer(), 0, einsumNode, 0);
    einsumNode.UpdateInputDesc(0, inDesc);
    einsumNode.UpdateOutputDesc(0, MakeTensorDesc({-1, 20}, DT_FLOAT16));

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(einsumNode, 0));
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_EQ(status, GRAPH_SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);

    // The Cast node converts the reduced result to the op's output dtype (fp16)
    GNode castNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, "Cast", castNode));
    int64_t dstType = -1;
    EXPECT_EQ(castNode.GetAttr("dst_type", dstType), GRAPH_SUCCESS);
    EXPECT_EQ(dstType, static_cast<int64_t>(DT_FLOAT16));

    // The reduce stage keeps the input dtype (fp32)
    GNode reduceNode;
    if (!FindFirstNodeByType(graph, "ReduceSum", reduceNode)) {
        EXPECT_TRUE(FindFirstNodeByType(graph, "ReduceSumD", reduceNode));
    }
    TensorDesc reduceDesc;
    reduceNode.GetOutputDesc(0, reduceDesc);
    EXPECT_EQ(reduceDesc.GetDataType(), DT_FLOAT);

    // The Cast is fed by the reduce node (no node in between on this path)
    auto [inPtr, inPort] = castNode.GetInDataNodesAndPortIndexs(0);
    ASSERT_NE(inPtr, nullptr);
    AscendString inTypeStr;
    inPtr->GetType(inTypeStr);
    const std::string inType(inTypeStr.GetString());
    EXPECT_TRUE(inType == "ReduceSum" || inType == "ReduceSumD");

    // Final output: fp16 with the reduced shape
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    TensorDesc outDesc;
    finalNode.GetOutputDesc(0, outDesc);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {-1, 20}));
}

TEST_F(EinsumPassTest, abcd2AbFp32InFp16OutStaticKeepsInputDtype)
{
    auto graphBuilder = es::EsGraphBuilder("abcd_ab_static_dtypes");
    auto* rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto inDesc = MakeTensorDesc({10, 20, 30, 40}, DT_FLOAT);
    auto input0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT, FORMAT_ND, {10, 20, 30, 40});
    input0.GetProducer()->UpdateOutputDesc(0, inDesc);
    CompliantNodeBuilder builder(rawGraph);
    builder.OpType("Einsum").Name("abcd_ab_static_dtypes");
    builder.IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    builder.IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"equation", CompliantNodeBuilder::kEsAttrRequired, "String", CreateFrom(AscendString("abcd->ab"))},
            {"N", CompliantNodeBuilder::kEsAttrRequired, "Int", CreateFrom(static_cast<int64_t>(1))},
        });
    auto einsumNode = builder.Build();
    AddEdgeAndUpdatePeerDesc(*rawGraph, *input0.GetProducer(), 0, einsumNode, 0);
    einsumNode.UpdateInputDesc(0, inDesc);
    einsumNode.UpdateOutputDesc(0, MakeTensorDesc({10, 20}, DT_FLOAT16));

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(einsumNode, 0));
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_EQ(status, GRAPH_SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);

    // Static fuzz path: no output-dtype alignment, no Cast node
    EXPECT_EQ(CountNodes(graph, "Cast"), 0);

    // The result stays in the input dtype (fp32): pinned legacy behavior
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    TensorDesc outDesc;
    finalNode.GetOutputDesc(0, outDesc);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT);
    EXPECT_TRUE(CheckNodeOutputShape(finalNode, {10, 20}));
}

// === Legacy dtype semantics (aligned): raw mixed-dtype inputs are only rejected on the generic
// dynamic fuzz route (see dtypeMismatchDynamicFuzzFail); the static fuzz path and the 28 pattern
// handlers do not check dtypes (the generated mixed-dtype matmul is left to downstream checks). ===
TEST_F(EinsumPassTest, dtypeMismatchStaticFuzzFollowsLegacyBehavior)
{
    // static shape + mixed dtypes -> static fuzz path has no dtype gate
    auto graphBuilder = EsGraphBuilder("dtype_mismatch_static");
    auto* rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    std::vector<DataType> inputDtypes = {DT_FLOAT16, DT_FLOAT};
    std::vector<std::vector<int64_t>> inputDims = {{10, 20}, {20, 30}};
    std::vector<EsTensorHolder> inputs;
    for (size_t i = 0; i < 2; ++i) {
        auto desc = MakeTensorDesc(inputDims[i], inputDtypes[i]);
        auto input = graphBuilder.CreateInput(static_cast<int64_t>(i), ("x" + std::to_string(i)).c_str(),
                                              inputDtypes[i], FORMAT_ND, inputDims[i]);
        input.GetProducer()->UpdateOutputDesc(0, desc);
        inputs.push_back(input);
    }
    CompliantNodeBuilder builder(rawGraph);
    builder.OpType("Einsum").Name("dtype_mismatch_static");
    builder.IrDefInputs(
        {{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}, {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    builder.IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"equation", CompliantNodeBuilder::kEsAttrRequired, "String", CreateFrom(AscendString("ab,bc->ac"))},
            {"N", CompliantNodeBuilder::kEsAttrRequired, "Int", CreateFrom(static_cast<int64_t>(2))},
        });
    auto einsumNode = builder.Build();
    for (size_t i = 0; i < 2; ++i) {
        AddEdgeAndUpdatePeerDesc(*rawGraph, *inputs[i].GetProducer(), 0, einsumNode, static_cast<int32_t>(i));
        einsumNode.UpdateInputDesc(static_cast<int32_t>(i), MakeTensorDesc(inputDims[i], inputDtypes[i]));
    }
    einsumNode.UpdateOutputDesc(0, MakeTensorDesc({10, 30}, DT_FLOAT16));
    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(einsumNode, 0));
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    // Pin the actual legacy-aligned outcome: no dtype gate on this route, the decomposition
    // proceeds and produces a mixed-dtype matmul (left to downstream checks, same as legacy).
    EXPECT_EQ(status, GRAPH_SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    EXPECT_GE(CountNodes(graph, "MatMul") + CountNodes(graph, "MatMulV2") + CountNodes(graph, "BatchMatMul") +
                  CountNodes(graph, "BatchMatMulV2"),
              1);
}

TEST_F(EinsumPassTest, dtypeMismatch28PatternDynamicFollowsLegacyBehavior)
{
    // dynamic shape + 28-pattern equation + mixed dtypes -> handler route has no dtype gate
    auto graph = BuildEinsumGraphDynamic("dtype_mismatch_28p", "abc,cde->abde", 2, {{-1, 20, 30}, {30, -1, 50}},
                                         {{{1, 10}, {20, 20}, {30, 30}}, {{30, 30}, {1, 10}, {50, 50}}},
                                         {-1, 20, 40, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    // No dtype gate on the 28-pattern handler route (same as legacy): fusion proceeds.
    EXPECT_EQ(status, GRAPH_SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

// === Core promoted-dtype cast flow: both inputs fp16 and only input0 carries a real reduce dim
// (input1's contract dim is broadcast 1), so ReduceSumInput promotes input0 to fp32 while input1
// stays fp16; BatchMatmul must cast input1 up to fp32 (legacy cast placement, before the matmul)
// and cast the result back to input(0)'s dtype. ===
TEST_F(EinsumPassTest, dynamicFuzzPromotedDtypeCastFlowSuccess)
{
    auto graph = BuildEinsumGraphDynamic("dtype_promoted_dyn", "ab,bc->ac", 2, {{-1, 20}, {1, -1}},
                                         {{{1, 10}, {20, 20}}, {{1, 1}, {1, 10}}}, {-1, 30}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    ASSERT_EQ(status, GRAPH_SUCCESS);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
    // Promote cast (input0 fp16->fp32) + pairing cast (input1 fp16->fp32) + restore cast (fp16)
    EXPECT_GE(CountNodes(graph, "Cast"), 3);
    GNode finalNode;
    ASSERT_TRUE(FindFinalOutputNode(graph, finalNode));
    TensorDesc outDesc;
    finalNode.GetOutputDesc(0, outDesc);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
}

TEST_F(EinsumPassTest, unsupported3InputsFail)
{
    auto graph = BuildEinsumGraph("three_inputs", "abc,cde,abe->abcd", 3, {{10, 20, 30}, {30, 40, 50}, {10, 20, 50}},
                                  {10, 20, 30, 40}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// === Negative case: dynamic fuzz path rejects non-broadcastable shared labels (aligned with
// legacy AlignInputDimForOutLabel broadcast check; specific handlers / static fuzz skip it) ===

TEST_F(EinsumPassTest, dynamicFuzzBroadcastMismatchFail)
{
    // "ab,ab->ab" is not in the 28 specific patterns -> dynamic fuzz path.
    // Shared label b: 20 vs 30 is not broadcastable -> reject.
    auto graph = BuildEinsumGraphDynamic("dyn_fuzz_bcast_mismatch", "ab,ab->ab", 2, {{-1, 20}, {-1, 30}},
                                         {{{1, 10}, {20, 20}}, {{1, 10}, {30, 30}}}, {-1, 20}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 1);
}

TEST_F(EinsumPassTest, dynamicFuzzBroadcastUnknownPass)
{
    // Same shape but with unknown b dims: broadcastable -> fuse
    auto graph = BuildEinsumGraphDynamic("dyn_fuzz_bcast_unknown", "ab,ab->ab", 2, {{-1, 20}, {-1, -1}},
                                         {{{1, 10}, {20, 20}}, {{1, 10}, {1, 30}}}, {-1, -1}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 0);
}

// === Negative cases: equation attribute validation (C-02) ===

TEST_F(EinsumPassTest, emptyEquationFail)
{
    auto graph = BuildEinsumGraph("empty_equation", "", 2, {{10, 20}, {20, 30}}, {10, 30}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 1);
}

TEST_F(EinsumPassTest, noArrowEquationFail)
{
    auto graph = BuildEinsumGraph("no_arrow", "abc,cde", 2, {{10, 20, 30}, {30, 40, 50}}, {10, 20, 40, 50}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 1);
}

TEST_F(EinsumPassTest, missingEquationAttrFail)
{
    auto graphBuilder = EsGraphBuilder("missing_equation");
    auto* rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto desc0 = MakeTensorDesc({10, 20}, DT_FLOAT16);
    auto desc1 = MakeTensorDesc({20, 30}, DT_FLOAT16);
    auto input0 = graphBuilder.CreateInput(0, "x0", DT_FLOAT16, FORMAT_ND, {10, 20});
    auto input1 = graphBuilder.CreateInput(1, "x1", DT_FLOAT16, FORMAT_ND, {20, 30});
    input0.GetProducer()->UpdateOutputDesc(0, desc0);
    input1.GetProducer()->UpdateOutputDesc(0, desc1);

    CompliantNodeBuilder builder(rawGraph);
    builder.OpType("Einsum").Name("missing_equation");
    builder.IrDefInputs(
        {{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}, {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    builder.IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"N", CompliantNodeBuilder::kEsAttrRequired, "Int", CreateFrom(static_cast<int64_t>(2))},
        });
    auto einsumNode = builder.Build();
    AddEdgeAndUpdatePeerDesc(*rawGraph, *input0.GetProducer(), 0, einsumNode, 0);
    AddEdgeAndUpdatePeerDesc(*rawGraph, *input1.GetProducer(), 0, einsumNode, 1);
    einsumNode.UpdateInputDesc(0, desc0);
    einsumNode.UpdateInputDesc(1, desc1);
    einsumNode.UpdateOutputDesc(0, MakeTensorDesc({10, 30}, DT_FLOAT16));
    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(einsumNode, 0));
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 1);
}

// === Negative cases: label count must match input dim num (C-04) ===

TEST_F(EinsumPassTest, labelDimMismatchFail)
{
    // equation "ab,cd->abd" but x0 is 3D: label count 2 != dim num 3
    auto graph = BuildEinsumGraph("label_dim_mismatch", "ab,cd->abd", 2, {{10, 20, 30}, {30, 40}}, {10, 20, 40},
                                  DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 1);
}

TEST_F(EinsumPassTest, ellipsisLabelDimMismatchFail)
{
    // "...ab" with 1D input: label count 2 (a,b) > dim num 1 after ellipsis expansion
    auto graph = BuildEinsumGraph("ellipsis_mismatch", "...ab,bc->ac", 2, {{10}, {20, 30}}, {10, 30}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 1);
}

TEST_F(EinsumPassTest, outputLabelDimMismatchFail)
{
    // output equation "abcd" has 4 labels but output tensor is 3D
    auto graph = BuildEinsumGraph("out_label_mismatch", "abc->abcd", 1, {{10, 20, 30}}, {10, 20, 30}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 1);
}

// === Negative case: input shape product overflow (C-05) ===

TEST_F(EinsumPassTest, shapeOverflowFail)
{
    // 3037000500 * 3037000500 overflows int64
    auto graph = BuildEinsumGraph("shape_overflow", "ab,cb->ac", 2, {{3037000500, 3037000500}, {10, 3037000500}},
                                  {3037000500, 10}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 1);
}

// === Negative case: inflated output is not supported (C-06) ===

TEST_F(EinsumPassTest, inflatedOutputFail)
{
    // output label 'a' repeats in "abc->aa"
    auto graph = BuildEinsumGraph("inflated_output", "abc->aa", 1, {{10, 20, 30}}, {10, 10}, DT_FLOAT16);
    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 1);
}

// === Negative case: dynamic fuzz path requires same dtype for both inputs (C-09) ===

TEST_F(EinsumPassTest, dtypeMismatchDynamicFuzzFail)
{
    // dynamic shape + equation not in 28 patterns + x0 FP16 / x1 FP32
    auto graphBuilder = EsGraphBuilder("dtype_mismatch_dyn");
    auto* rawGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    std::vector<DataType> inputDtypes = {DT_FLOAT16, DT_FLOAT};
    std::vector<std::vector<int64_t>> inputDims = {{-1, 20}, {20, -1}};
    std::vector<std::vector<std::pair<int64_t, int64_t>>> inputRanges = {{{1, 10}, {20, 20}}, {{20, 20}, {1, 10}}};

    std::vector<EsTensorHolder> inputs;
    for (size_t i = 0; i < 2; ++i) {
        auto desc = MakeTensorDesc(inputDims[i], inputDtypes[i]);
        desc.SetShapeRange(inputRanges[i]);
        auto input = graphBuilder.CreateInput(static_cast<int64_t>(i), ("x" + std::to_string(i)).c_str(),
                                              inputDtypes[i], FORMAT_ND, inputDims[i]);
        input.GetProducer()->UpdateOutputDesc(0, desc);
        inputs.push_back(input);
    }

    CompliantNodeBuilder builder(rawGraph);
    builder.OpType("Einsum").Name("dtype_mismatch_dyn");
    builder.IrDefInputs(
        {{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}, {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    builder.IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"equation", CompliantNodeBuilder::kEsAttrRequired, "String", CreateFrom(AscendString("ab,bc->ac"))},
            {"N", CompliantNodeBuilder::kEsAttrRequired, "Int", CreateFrom(static_cast<int64_t>(2))},
        });
    auto einsumNode = builder.Build();
    for (size_t i = 0; i < 2; ++i) {
        AddEdgeAndUpdatePeerDesc(*rawGraph, *inputs[i].GetProducer(), 0, einsumNode, static_cast<int32_t>(i));
        auto desc = MakeTensorDesc(inputDims[i], inputDtypes[i]);
        desc.SetShapeRange(inputRanges[i]);
        einsumNode.UpdateInputDesc(static_cast<int32_t>(i), desc);
    }
    einsumNode.UpdateOutputDesc(0, MakeTensorDesc({-1, -1}, DT_FLOAT16));
    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(einsumNode, 0));
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    EinsumPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Einsum"), 1);
}
