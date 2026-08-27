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
#include "../../../op_graph/fusion_pass/matmul_to_gemm_op_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace fe;
using namespace ops;

namespace {

constexpr char kPassName[] = "MatmulToGemmOpFusionPass";

// ---- Platform helpers ----

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

void SetPlatformInfo910B()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.ai_core_spec.l1_size = 512 * 1024;
    platformInfo.soc_info.l2_size = 64 * 1024 * 1024;
    optionalInfo.soc_version = "Ascend910B";
    // Milan: l0c2out + out2l1nd2nz present, l0c2ub + fixpipeL0c2ub absent, l12bt absent
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_nd2nz"] = {"float16"};
    // Deliberately NOT setting l0c2ub, fixpipeL0c2ub, l12bt
    platformInfo.str_info.short_soc_version = "Ascend910B";
    PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

void SetPlatformInfo310P()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 8;
    optionalInfo.soc_version = "Ascend310P";
    platformInfo.str_info.short_soc_version = "Ascend310P";
    // No cube intrinsics at all
    PlatformInfoManager::Instance().platform_info_map_["Ascend310P"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

// ---- Utility helpers ----

TensorDesc MakeDesc(const std::vector<int64_t>& dims, DataType dtype, Format fmt = FORMAT_ND)
{
    TensorDesc desc(ge::Shape(dims), fmt, dtype);
    desc.SetOriginFormat(fmt);
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

bool FindNode(const std::shared_ptr<Graph>& graph, const char* opType, GNode& outNode)
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

std::vector<CompliantNodeBuilder::IrInputDef> BuildMmInputs(const char* opType)
{
    std::vector<CompliantNodeBuilder::IrInputDef> inputs = {
        {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
    };
    if (strcmp(opType, "MatMulV2") == 0) {
        inputs.push_back({"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""});
    }
    return inputs;
}

std::vector<CompliantNodeBuilder::IrAttrDef> BuildMmAttrs(const char* opType)
{
    std::vector<CompliantNodeBuilder::IrAttrDef> attrs = {
        {"transpose_x1", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
        {"transpose_x2", CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
    };
    if (strcmp(opType, "MatMulV2") == 0) {
        attrs.push_back({"offset_x", CompliantNodeBuilder::kEsAttrOptional, "Int", AttrValue()});
    }
    return attrs;
}

/// Build a test graph: dataX1, dataX2 → MatMul → [Cast] → AssignAdd ← dataInput0
/// @param withCast  if true, insert Cast between MatMul and AssignAdd
/// @param mmDtype   MatMul input/output dtype (fp16/bf16/fp32)
/// @param mmOutDtype MatMul output dtype (defaults to mmDtype)
std::shared_ptr<Graph> BuildGraph(const std::string& name, const char* matmulOpType, bool withCast,
                                  const std::vector<int64_t>& aDims, const std::vector<int64_t>& bDims,
                                  const std::vector<int64_t>& outDims, DataType mmDtype,
                                  DataType mmOutDtype = DT_UNDEFINED, bool transX1 = false, bool transX2 = false,
                                  int64_t opImplMode = 0x01)
{
    if (mmOutDtype == DT_UNDEFINED) {
        mmOutDtype = mmDtype;
    }
    DataType assignAddDtype = withCast ? DT_FLOAT : mmOutDtype; // AssignAdd input1 dtype

    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    // Input data nodes
    auto x1Desc = MakeDesc(aDims, mmDtype);
    auto x2Desc = MakeDesc(bDims, mmDtype);
    auto outDesc = MakeDesc(outDims, mmOutDtype);
    auto input0Desc = MakeDesc(outDims, DT_FLOAT); // AssignAdd ref is always fp32
    auto valueDesc = MakeDesc(outDims, assignAddDtype);

    auto dataX1 = graphBuilder.CreateInput(0, "dataX1", mmDtype, FORMAT_ND, aDims);
    auto dataX2 = graphBuilder.CreateInput(1, "dataX2", mmDtype, FORMAT_ND, bDims);
    auto dataInput0 = graphBuilder.CreateInput(2, "dataInput0", DT_FLOAT, FORMAT_ND, outDims);
    dataX1.GetProducer()->UpdateOutputDesc(0, x1Desc);
    dataX2.GetProducer()->UpdateOutputDesc(0, x2Desc);
    dataInput0.GetProducer()->UpdateOutputDesc(0, input0Desc);

    // MatMul node
    auto matmulNode = CompliantNodeBuilder(graph)
                          .OpType(matmulOpType)
                          .Name("matmul")
                          .IrDefInputs(BuildMmInputs(matmulOpType))
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs(BuildMmAttrs(matmulOpType))
                          .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), matmulNode, 1);
    matmulNode.UpdateInputDesc(0, x1Desc);
    matmulNode.UpdateInputDesc(1, x2Desc);
    matmulNode.UpdateOutputDesc(0, outDesc);
    matmulNode.SetAttr("transpose_x1", transX1);
    matmulNode.SetAttr("transpose_x2", transX2);
    matmulNode.SetAttr("_op_impl_mode_enum", opImplMode);

    // AssignAdd node
    auto assignAddNode = CompliantNodeBuilder(graph)
                             .OpType("AssignAdd")
                             .Name("assignadd")
                             .IrDefInputs({{"ref", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                           {"value", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"ref", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .IrDefAttrs(
                                 {{"use_locking", CompliantNodeBuilder::kEsAttrOptional, "Bool", CreateFrom(false)}})
                             .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *dataInput0.GetProducer(), dataInput0.GetProducerOutIndex(), assignAddNode, 0);

    if (withCast) {
        // Cast node: mmDtype → fp32
        auto castNode = CompliantNodeBuilder(graph)
                            .OpType("Cast")
                            .Name("cast")
                            .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                            .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .IrDefAttrs({{"dst_type", CompliantNodeBuilder::kEsAttrRequired, "Int",
                                          CreateFrom(static_cast<int64_t>(DT_FLOAT))}})
                            .Build();
        AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, castNode, 0);
        castNode.UpdateInputDesc(0, outDesc);
        castNode.UpdateOutputDesc(0, MakeDesc(outDims, DT_FLOAT));
        AddEdgeAndUpdatePeerDesc(*graph, castNode, 0, assignAddNode, 1);
        assignAddNode.UpdateInputDesc(1, MakeDesc(outDims, DT_FLOAT));
    } else {
        AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, assignAddNode, 1);
        assignAddNode.UpdateInputDesc(1, valueDesc);
    }
    assignAddNode.UpdateInputDesc(0, input0Desc);
    assignAddNode.UpdateOutputDesc(0, input0Desc);

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(assignAddNode, 0));
    return graphBuilder.BuildAndReset({output});
}

} // namespace

// ---- Test fixture ----

class MatmulToGemmOpFusionPassTest : public testing::Test {
protected:
    static void SetUpTestSuite() { SetPlatformInfo950(); }
    static void TearDownTestSuite() {}
    void SetUp() override {}
    void TearDown() override {}
};

// =========================================================================
// L0: Pattern test
// =========================================================================

TEST_F(MatmulToGemmOpFusionPassTest, patternTest)
{
    MatmulToGemmOpFusionPass pass;
    auto patterns = pass.Patterns();
    EXPECT_EQ(patterns.size(), 6U); // 3 op types × 2 scenarios
}

// =========================================================================
// L0: GemmV3 path (David/950) — fusion success
// =========================================================================

TEST_F(MatmulToGemmOpFusionPassTest, gemmV3Fp16WithCastFusionSuccess)
{
    // David path: supportL12btBf16=true → GemmV3
    // Shape: m=512, n=512, k=512 (passes CheckShapeValid)
    auto graph = BuildGraph("v3_fp16_cast", "MatMul", true, {512, 512}, {512, 512}, {512, 512}, DT_FLOAT16);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "GemmV3"), 1);
    EXPECT_EQ(CountNodes(graph, "MatMul"), 0);
    EXPECT_EQ(CountNodes(graph, "Cast"), 0);
    EXPECT_EQ(CountNodes(graph, "AssignAdd"), 0);
}

TEST_F(MatmulToGemmOpFusionPassTest, gemmV3Bf16WithCastFusionSuccess)
{
    auto graph = BuildGraph("v3_bf16_cast", "MatMul", true, {512, 512}, {512, 512}, {512, 512}, DT_BF16);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "GemmV3"), 1);
}

TEST_F(MatmulToGemmOpFusionPassTest, gemmV3Fp32NoCastFusionSuccess)
{
    // fp32 input only on David (supportL12btBf16=true), no Cast (Pattern1)
    // fp32 output feeds AssignAdd directly
    auto graph = BuildGraph("v3_fp32_nocast", "MatMul", false, {512, 512}, {512, 512}, {512, 512}, DT_FLOAT, DT_FLOAT);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "GemmV3"), 1);
}

TEST_F(MatmulToGemmOpFusionPassTest, gemmV3Fp16NoCastFp32OutputFusionSuccess)
{
    // Pattern1 (no Cast): MatMul output is fp32, directly feeds AssignAdd
    auto graph = BuildGraph("v3_fp16_nocast", "MatMul", false, {512, 512}, {512, 512}, {512, 512}, DT_FLOAT16,
                            DT_FLOAT);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "GemmV3"), 1);
    EXPECT_EQ(CountNodes(graph, "MatMul"), 0);
    EXPECT_EQ(CountNodes(graph, "AssignAdd"), 0);
}

TEST_F(MatmulToGemmOpFusionPassTest, gemmV3Hf32FusionSuccess)
{
    // enable hf32 via _op_impl_mode_enum = 0x41
    auto graph = BuildGraph("v3_hf32", "MatMul", true, {512, 512}, {512, 512}, {512, 512}, DT_FLOAT16, DT_UNDEFINED,
                            false, false, 0x41);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    GNode gemmNode;
    ASSERT_TRUE(FindNode(graph, "GemmV3", gemmNode));
    bool enableHf32 = false;
    EXPECT_EQ(gemmNode.GetAttr("enable_hf32", enableHf32), GRAPH_SUCCESS);
    EXPECT_TRUE(enableHf32);
}

TEST_F(MatmulToGemmOpFusionPassTest, gemmV3TransposeFusionSuccess)
{
    // transA=true, transB=true: aShape=[K,M], bShape=[N,K]
    auto graph = BuildGraph("v3_trans", "MatMul", true, {512, 1024}, {1024, 512}, {1024, 512}, DT_FLOAT16, DT_UNDEFINED,
                            true, true);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    GNode gemmNode;
    ASSERT_TRUE(FindNode(graph, "GemmV3", gemmNode));
    bool transA = false;
    bool transB = false;
    gemmNode.GetAttr("transpose_a", transA);
    gemmNode.GetAttr("transpose_b", transB);
    EXPECT_TRUE(transA);
    EXPECT_TRUE(transB);
}

// =========================================================================
// L1: Different MatMul types (David path)
// =========================================================================

TEST_F(MatmulToGemmOpFusionPassTest, gemmV3MatMulV2WithCastFusionSuccess)
{
    auto graph = BuildGraph("v3_mmv2", "MatMulV2", true, {512, 512}, {512, 512}, {512, 512}, DT_FLOAT16);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "GemmV3"), 1);
}

TEST_F(MatmulToGemmOpFusionPassTest, gemmV3MatMulV3WithCastFusionSuccess)
{
    auto graph = BuildGraph("v3_mmv3", "MatMulV3", true, {512, 512}, {512, 512}, {512, 512}, DT_FLOAT16);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "GemmV3"), 1);
}

// =========================================================================
// L0: GemmV2 path (Milan/910B) — fusion success with white-list shape
// =========================================================================

class MatmulToGemmOpMilanTest : public testing::Test {
protected:
    static void SetUpTestSuite() { SetPlatformInfo910B(); }
    static void TearDownTestSuite() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MatmulToGemmOpMilanTest, gemmV2Fp16WithCastFusionSuccess)
{
    // White-list shape: "4096_12288_4096_6144_1_0"
    // aShape=[4096,12288], bShape=[4096,6144], transA=true, transB=false
    // output = [12288, 6144]
    auto graph = BuildGraph("v2_fp16_cast", "MatMul", true, {4096, 12288}, {4096, 6144}, {12288, 6144}, DT_FLOAT16,
                            DT_UNDEFINED, true, false);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "GemmV2"), 1);
    EXPECT_EQ(CountNodes(graph, "MatMul"), 0);
    EXPECT_EQ(CountNodes(graph, "Cast"), 0);
    EXPECT_EQ(CountNodes(graph, "AssignAdd"), 0);
}

TEST_F(MatmulToGemmOpMilanTest, gemmV2Bf16WithCastFusionSuccess)
{
    auto graph = BuildGraph("v2_bf16_cast", "MatMul", true, {4096, 12288}, {4096, 6144}, {12288, 6144}, DT_BF16,
                            DT_UNDEFINED, true, false);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "GemmV2"), 1);
}

TEST_F(MatmulToGemmOpMilanTest, gemmV2Fp16NoCastFp32OutputFusionSuccess)
{
    // Pattern1 (no Cast): MatMul fp16 input, fp32 output
    auto graph = BuildGraph("v2_fp16_nocast", "MatMul", false, {4096, 12288}, {4096, 6144}, {12288, 6144}, DT_FLOAT16,
                            DT_FLOAT, true, false);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, "GemmV2"), 1);
}

// =========================================================================
// L2: Failure cases — David path
// =========================================================================

TEST_F(MatmulToGemmOpFusionPassTest, unsupportedDtypeInt8Fail)
{
    // int8 not supported
    auto graph = BuildGraph("dtype_int8", "MatMul", true, {512, 512}, {512, 512}, {512, 512}, DT_INT8);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatmulToGemmOpFusionPassTest, shapeTooSmallFail)
{
    // David: m=256 < 512, fails CheckShapeValid
    auto graph = BuildGraph("small_shape", "MatMul", true, {256, 256}, {256, 256}, {256, 256}, DT_FLOAT16);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatmulToGemmOpFusionPassTest, kTooSmallFail)
{
    // David: k=128 <= 256, fails CheckShapeValid
    auto graph = BuildGraph("small_k", "MatMul", true, {512, 128}, {128, 512}, {512, 512}, DT_FLOAT16);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// =========================================================================
// L2: Failure cases — Milan path
// =========================================================================

TEST_F(MatmulToGemmOpMilanTest, shapeNotInWhiteListFail)
{
    // Shape not in white list
    auto graph = BuildGraph("not_whitelist", "MatMul", true, {256, 256}, {256, 256}, {256, 256}, DT_FLOAT16);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatmulToGemmOpMilanTest, unsupportedDtypeFp32Fail)
{
    // Milan: fp32 not supported (supportL12btBf16=false)
    auto graph = BuildGraph("dtype_fp32_milan", "MatMul", true, {4096, 12288}, {4096, 6144}, {12288, 6144}, DT_FLOAT,
                            DT_UNDEFINED, true, false);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// =========================================================================
// L2: Failure cases — unsupported platform
// =========================================================================

class MatmulToGemmOp310PTest : public testing::Test {
protected:
    static void SetUpTestSuite() { SetPlatformInfo310P(); }
    static void TearDownTestSuite() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MatmulToGemmOp310PTest, unsupportedPlatformFail)
{
    // 310P: no intrinsics, supportL12btBf16=false, platform check fails
    auto graph = BuildGraph("platform_310p", "MatMul", true, {512, 512}, {512, 512}, {512, 512}, DT_FLOAT16);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// =========================================================================
// L2: GemmV3 attribute verification
// =========================================================================

TEST_F(MatmulToGemmOpFusionPassTest, gemmV3AttrsVerification)
{
    auto graph = BuildGraph("v3_attrs", "MatMul", true, {512, 512}, {512, 512}, {512, 512}, DT_FLOAT16, DT_UNDEFINED,
                            false, false, 0x01);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    GNode gemmNode;
    ASSERT_TRUE(FindNode(graph, "GemmV3", gemmNode));

    // Check transpose attrs
    bool transA = true;
    bool transB = true;
    gemmNode.GetAttr("transpose_a", transA);
    gemmNode.GetAttr("transpose_b", transB);
    EXPECT_FALSE(transA);
    EXPECT_FALSE(transB);

    // Check alpha/beta (should be 1.0)
    float alpha = 0.0f;
    float beta = 0.0f;
    gemmNode.GetAttr("alpha", alpha);
    gemmNode.GetAttr("beta", beta);
    EXPECT_FLOAT_EQ(alpha, 1.0f);
    EXPECT_FLOAT_EQ(beta, 1.0f);

    // Check enable_hf32 (opImplMode=0x01, bit 0x40 not set → false)
    bool enableHf32 = true;
    gemmNode.GetAttr("enable_hf32", enableHf32);
    EXPECT_FALSE(enableHf32);
}

// =========================================================================
// L2: GemmV2 alpha/beta Const verification
// =========================================================================

TEST_F(MatmulToGemmOpMilanTest, gemmV2ConstNodesVerification)
{
    auto graph = BuildGraph("v2_const", "MatMul", true, {4096, 12288}, {4096, 6144}, {12288, 6144}, DT_FLOAT16,
                            DT_UNDEFINED, true, false);
    ASSERT_NE(graph, nullptr);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    // GemmV2 should have 5 inputs: a, b, alpha, beta, c
    GNode gemmNode;
    ASSERT_TRUE(FindNode(graph, "GemmV2", gemmNode));
    EXPECT_EQ(gemmNode.GetInputsSize(), 5U);

    // Check Const nodes exist (alpha and beta)
    EXPECT_EQ(CountNodes(graph, "Const"), 2);
}

// =========================================================================
// L2: No matching node (empty graph)
// =========================================================================

TEST_F(MatmulToGemmOpFusionPassTest, noMatchingNodeFail)
{
    auto graphBuilder = EsGraphBuilder("empty");
    auto data = graphBuilder.CreateInput(0, "data", DT_FLOAT16, FORMAT_ND, {512, 512});
    data.GetProducer()->UpdateOutputDesc(0, MakeDesc({512, 512}, DT_FLOAT16));
    auto uniqueGraph = graphBuilder.BuildAndReset({data});
    ASSERT_NE(uniqueGraph, nullptr);
    GraphPtr graph = std::move(uniqueGraph);

    MatmulToGemmOpFusionPass pass;
    CustomPassContext passContext;
    auto status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}
