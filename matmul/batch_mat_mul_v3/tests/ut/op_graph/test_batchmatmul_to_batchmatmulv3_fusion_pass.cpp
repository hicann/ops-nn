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
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/batchmatmul_to_batchmatmulv3_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace {

constexpr char kPassName[] = "BatchMatMulToBatchMatmulV3FusionPass";
constexpr int64_t kOpImplModeHf32 = 0x40;

static const char* g_npuArch = "3510";

void SetStubNpuArch(const char* arch)
{ g_npuArch = arch; }

} // namespace

extern "C" {

__attribute__((weak)) int32_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen)
{
    if (label == nullptr || key == nullptr || val == nullptr) {
        return 1;
    }
    if (strcmp(label, "version") == 0 && strcmp(key, "NpuArch") == 0) {
        const char* arch = g_npuArch;
        if (strlen(arch) >= maxLen) {
            return 1;
        }
        strncpy(val, arch, maxLen);
    }
    return 0;
}
}

namespace {

void SetPlatformInfo910B1()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    optionalInfo.soc_version = "Ascend910B1";
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_nd2nz"] = {"float16"};
    platformInfo.str_info.short_soc_version = "Ascend910B";
    PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

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

std::shared_ptr<Graph> BuildMatMulLikeGraph(const std::string& name, const char* opType,
                                            const std::vector<int64_t>& aDims, const std::vector<int64_t>& bDims,
                                            const std::vector<int64_t>& outDims, DataType dtype, bool transX1,
                                            bool transX2, const std::vector<int64_t>& biasDims = {},
                                            int64_t opImplModeEnum = -1)
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

    bool hasBias = !biasDims.empty();
    EsTensorHolder dataBias = nullptr;
    if (hasBias) {
        auto biasDesc = MakeTensorDesc(biasDims, dtype);
        dataBias = graphBuilder.CreateInput(2, "dataBias", dtype, FORMAT_ND, biasDims);
        dataBias.GetProducer()->UpdateOutputDesc(0, biasDesc);
    }

    bool isBatch = (strcmp(opType, "BatchMatMul") == 0 || strcmp(opType, "BatchMatMulV2") == 0);
    const char* transAttr1 = isBatch ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = isBatch ? "adj_x2" : "transpose_x2";

    std::vector<CompliantNodeBuilder::IrInputDef> irInputs = {
        {"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
        {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""},
    };
    if (hasBias) {
        irInputs.push_back({"bias", CompliantNodeBuilder::kEsIrInputOptional, ""});
    }

    auto node = CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(name.c_str())
                    .IrDefInputs(irInputs)
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({
                        {transAttr1, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                        {transAttr2, CompliantNodeBuilder::kEsAttrRequired, "Bool", CreateFrom(false)},
                    })
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), node, 1);
    node.UpdateInputDesc(0, x1Desc);
    node.UpdateInputDesc(1, x2Desc);
    if (hasBias) {
        AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), node, 2);
        TensorDesc biasDesc = MakeTensorDesc(biasDims, dtype);
        node.UpdateInputDesc(2, biasDesc);
        bool hasBiasAttr = true;
        node.SetAttr("has_bias", hasBiasAttr);
    }
    node.UpdateOutputDesc(0, outDesc);
    node.SetAttr(transAttr1, transX1);
    node.SetAttr(transAttr2, transX2);
    if (opImplModeEnum >= 0) {
        node.SetAttr("_op_impl_mode_enum", opImplModeEnum);
    }

    auto output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    return graphBuilder.BuildAndReset({output});
}

struct FusionAttrExpect {
    bool adjX1 = false;
    bool adjX2 = false;
    int64_t offsetX = 0;
    bool enableHf32 = false;
};

void CheckFusedV3Node(const FusionAttrExpect& expect, const std::shared_ptr<Graph>& graph)
{
    GNode v3Node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "BatchMatMulV3", v3Node));

    bool adjX1 = false;
    bool adjX2 = false;
    int64_t offsetX = 0;
    bool enableHf32 = false;
    v3Node.GetAttr("adj_x1", adjX1);
    v3Node.GetAttr("adj_x2", adjX2);
    v3Node.GetAttr("offset_x", offsetX);
    v3Node.GetAttr("enable_hf32", enableHf32);
    EXPECT_EQ(adjX1, expect.adjX1);
    EXPECT_EQ(adjX2, expect.adjX2);
    EXPECT_EQ(offsetX, expect.offsetX);
    EXPECT_EQ(enableHf32, expect.enableHf32);
}

void CheckInputDtype(const std::shared_ptr<Graph>& graph, size_t inputIdx, DataType expectedDtype)
{
    GNode v3Node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "BatchMatMulV3", v3Node));
    TensorDesc desc;
    ASSERT_EQ(v3Node.GetInputDesc(inputIdx, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetDataType(), expectedDtype);
}

void CheckOutputDtype(const std::shared_ptr<Graph>& graph, DataType expectedDtype)
{
    GNode v3Node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "BatchMatMulV3", v3Node));
    TensorDesc desc;
    ASSERT_EQ(v3Node.GetOutputDesc(0, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetDataType(), expectedDtype);
}

} // namespace

class BatchMatMulToBatchMatmulV3FusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    { SetPlatformInfo950(); }

    static void TearDownTestCase() {}

    void SetUp() override
    {
        SetPlatformInfo950();
        SetStubNpuArch("3510");
    }

    void TearDown() override {}
};

// ===================== Pattern test =====================

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, patternTest)
{
    BatchMatMulToBatchMatmulV3FusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

// ===================== Non-950 platform: empty graph fusion (GRAPH_NOT_CHANGED) =====================

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, non950PlatformFp16NotChangedFail)
{
    SetPlatformInfo910B1();
    SetStubNpuArch("910B");

    auto graph = BuildMatMulLikeGraph("non950Fp16", "BatchMatMulV2", {2, 3, 32, 64}, {2, 3, 64, 128}, {2, 3, 32, 128}, DT_FLOAT16,
                                         false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 0);
}

// ===================== 950: BatchMatMul -> BatchMatMulV3 =====================

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950BatchMatMulFp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950BmmFp16", "BatchMatMul", {2, 3, 32, 64}, {2, 3, 64, 128}, {2, 3, 32, 128}, DT_FLOAT16,
                                       false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);
    EXPECT_EQ(CountNodes(graph, "BatchMatMul"), 0);

    CheckInputDtype(graph, 0, DT_FLOAT16);
    CheckOutputDtype(graph, DT_FLOAT16);
    CheckFusedV3Node({false, false, 0, false}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950BatchMatMulTransX1Fp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950BmmTransX1Fp16", "BatchMatMul", {2, 3, 64, 32}, {2, 3, 64, 128}, {2, 3, 32, 128},
                                       DT_FLOAT16, true, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    CheckFusedV3Node({true, false, 0, false}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950BatchMatMulTransX2Fp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950BmmTransX2Fp16", "BatchMatMul", {2, 3, 32, 64}, {2, 3, 128, 64}, {2, 3, 32, 128},
                                       DT_FLOAT16, false, true);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    CheckFusedV3Node({false, true, 0, false}, graph);
}

// ===================== 950: BatchMatMulV2 -> BatchMatMulV3, different dtype =====================

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950V2Fp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950V2Fp16", "BatchMatMulV2", {2, 3, 32, 64}, {2, 3, 64, 128}, {2, 3, 32, 128},
                                         DT_FLOAT16, false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV2"), 0);

    CheckInputDtype(graph, 0, DT_FLOAT16);
    CheckInputDtype(graph, 1, DT_FLOAT16);
    CheckOutputDtype(graph, DT_FLOAT16);
    CheckFusedV3Node({false, false, 0, false}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950V2Bf16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950V2Bf16", "BatchMatMulV2", {2, 3, 32, 64}, {2, 3, 64, 128}, {2, 3, 32, 128}, DT_BF16,
                                         false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    CheckInputDtype(graph, 0, DT_BF16);
    CheckOutputDtype(graph, DT_BF16);
    CheckFusedV3Node({false, false, 0, false}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950V2Fp32FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950V2Fp32", "BatchMatMulV2", {2, 3, 32, 64}, {2, 3, 64, 128}, {2, 3, 32, 128}, DT_FLOAT,
                                         false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    CheckInputDtype(graph, 0, DT_FLOAT);
    CheckOutputDtype(graph, DT_FLOAT);
    CheckFusedV3Node({false, false, 0, false}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950V2Int8FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950V2Int8", "BatchMatMulV2", {2, 3, 32, 64}, {2, 3, 64, 128}, {2, 3, 32, 128}, DT_INT8,
                                         false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    CheckInputDtype(graph, 0, DT_INT8);
    CheckOutputDtype(graph, DT_INT8);
    CheckFusedV3Node({false, false, 0, false}, graph);
}

// ===================== 950: enable_hf32 attribute (op_impl_mode_enum=0x40) =====================

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950V2Fp32EnableHf32FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950V2Fp32Hf32", "BatchMatMulV2", {2, 3, 32, 64}, {2, 3, 64, 128}, {2, 3, 32, 128},
                                         DT_FLOAT, false, false, {}, kOpImplModeHf32);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    CheckFusedV3Node({false, false, 0, true}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950V2TransX1TransX2Fp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950V2TransX1TransX2Fp16", "BatchMatMulV2", {2, 3, 64, 32}, {2, 3, 128, 64},
                                         {2, 3, 32, 128}, DT_FLOAT16, true, true);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    CheckFusedV3Node({true, true, 0, false}, graph);
}

// ===================== 950: unsupported dtype INT32 fail =====================

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950V2UnsupportedDtypeInt32Fail)
{
    auto graph = BuildMatMulLikeGraph("ascend950V2Int32", "BatchMatMulV2", {2, 3, 32, 64}, {2, 3, 64, 128}, {2, 3, 32, 128}, DT_INT32,
                                         false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// ===================== 950: BatchMatMul with bias (post-biasadd state) -> BatchMatMulV3 =====================

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950BatchMatMulWithBiasFp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950BmmWithBiasFp16", "BatchMatMul", {2, 3, 32, 64}, {2, 3, 64, 128},
                                      {2, 3, 32, 128}, DT_FLOAT16, false, false, {128});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMul"), 0);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    GNode v3Node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "BatchMatMulV3", v3Node));
    ASSERT_EQ(v3Node.GetInputsSize(), 3U);
    TensorDesc biasDesc;
    ASSERT_EQ(v3Node.GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetDataType(), DT_FLOAT16);

    CheckFusedV3Node({false, false, 0, false}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950BatchMatMulWithBiasTransX1Fp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950BmmWithBiasTransX1Fp16", "BatchMatMul", {2, 3, 64, 32}, {2, 3, 64, 128},
                                      {2, 3, 32, 128}, DT_FLOAT16, true, false, {128});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMul"), 0);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    GNode v3Node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "BatchMatMulV3", v3Node));
    ASSERT_EQ(v3Node.GetInputsSize(), 3U);

    CheckFusedV3Node({true, false, 0, false}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950BatchMatMulWithBiasBf16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950BmmWithBiasBf16", "BatchMatMul", {2, 3, 32, 64}, {2, 3, 64, 128},
                                      {2, 3, 32, 128}, DT_BF16, false, false, {128});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMul"), 0);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    GNode v3Node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "BatchMatMulV3", v3Node));
    ASSERT_EQ(v3Node.GetInputsSize(), 3U);
    TensorDesc biasDesc;
    ASSERT_EQ(v3Node.GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetDataType(), DT_BF16);

    CheckFusedV3Node({false, false, 0, false}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950BatchMatMulWithBiasUserShapeFp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950BmmWithBiasUserShapeFp16", "BatchMatMul", {3000, 1, 60},
                                      {3000, 60, 64}, {3000, 1, 64}, DT_FLOAT16, false, false, {64});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMul"), 0);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    GNode v3Node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "BatchMatMulV3", v3Node));
    ASSERT_EQ(v3Node.GetInputsSize(), 3U);

    CheckFusedV3Node({false, false, 0, false}, graph);
}

// ===================== 950: BatchMatMulV2 with bias (post-biasadd state) -> BatchMatMulV3 =====================

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950BatchMatMulV2WithBiasFp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950BmmV2WithBiasFp16", "BatchMatMulV2", {2, 3, 32, 64}, {2, 3, 64, 128},
                                      {2, 3, 32, 128}, DT_FLOAT16, false, false, {128});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV2"), 0);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    GNode v3Node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "BatchMatMulV3", v3Node));
    ASSERT_EQ(v3Node.GetInputsSize(), 3U);
    TensorDesc biasDesc;
    ASSERT_EQ(v3Node.GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetDataType(), DT_FLOAT16);

    CheckFusedV3Node({false, false, 0, false}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950BatchMatMulV2WithBiasBf16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950BmmV2WithBiasBf16", "BatchMatMulV2", {2, 3, 32, 64}, {2, 3, 64, 128},
                                      {2, 3, 32, 128}, DT_BF16, false, false, {128});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV2"), 0);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    GNode v3Node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "BatchMatMulV3", v3Node));
    ASSERT_EQ(v3Node.GetInputsSize(), 3U);
    TensorDesc biasDesc;
    ASSERT_EQ(v3Node.GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
    EXPECT_EQ(biasDesc.GetDataType(), DT_BF16);

    CheckFusedV3Node({false, false, 0, false}, graph);
}

TEST_F(BatchMatMulToBatchMatmulV3FusionPassTest, ascend950BatchMatMulV2WithBiasFp32EnableHf32FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("ascend950BmmV2WithBiasFp32Hf32", "BatchMatMulV2", {2, 3, 32, 64},
                                      {2, 3, 64, 128}, {2, 3, 32, 128}, DT_FLOAT, false, false, {128}, kOpImplModeHf32);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    BatchMatMulToBatchMatmulV3FusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV2"), 0);
    EXPECT_EQ(CountNodes(graph, "BatchMatMulV3"), 1);

    GNode v3Node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "BatchMatMulV3", v3Node));
    ASSERT_EQ(v3Node.GetInputsSize(), 3U);

    CheckFusedV3Node({false, false, 0, true}, graph);
}
