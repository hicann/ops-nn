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
#include "../../../op_graph/fusion_pass/matmul_unsqueeze_squeeze_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace {

constexpr char kPassName[] = "MatMulUnsqueezeSqueezeFusionPass";

void SetPlatformInfo(const std::string& socVersion, const std::string& shortSocVersion)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    optionalInfo.soc_version = socVersion;
    platformInfo.str_info.short_soc_version = shortSocVersion;
    PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

void SetPlatformInfoWithL0c2out(const std::string& socVersion, const std::string& shortSocVersion)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    optionalInfo.soc_version = socVersion;
    platformInfo.str_info.short_soc_version = shortSocVersion;
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"float16"};
    PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
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

std::vector<int64_t> GetNodeOutputDims(const GNode& node, size_t idx = 0)
{
    TensorDesc desc;
    if (node.GetOutputDesc(idx, desc) != GRAPH_SUCCESS) {
        return {};
    }
    return desc.GetShape().GetDims();
}

std::vector<int64_t> GetNodeInputDims(const GNode& node, size_t idx)
{
    TensorDesc desc;
    if (node.GetInputDesc(idx, desc) != GRAPH_SUCCESS) {
        return {};
    }
    return desc.GetShape().GetDims();
}

void CheckNodeOutputDims(const std::shared_ptr<Graph>& graph, const char* opType,
                         const std::vector<int64_t>& expectedDims)
{
    GNode node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, opType, node));
    EXPECT_EQ(GetNodeOutputDims(node), expectedDims);
}

struct DequantConfig {
    bool hasDequant = false;
    DataType deqScaleDtype = DT_FLOAT16;
    std::vector<int64_t> deqScaleDims = {1};
};

std::shared_ptr<Graph> BuildMatMulLikeGraph(const std::string& name, const char* opType,
                                            const std::vector<int64_t>& aDims, const std::vector<int64_t>& bDims,
                                            const std::vector<int64_t>& outDims, DataType dtype, bool transX1,
                                            bool transX2, const std::vector<int64_t>& biasDims = {},
                                            const std::vector<int64_t>& offsetWDims = {},
                                            const DequantConfig& dequantCfg = {})
{
    auto graphBuilder = EsGraphBuilder(name.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x1Desc = MakeTensorDesc(aDims, dtype);
    auto x2Desc = MakeTensorDesc(bDims, dtype);
    auto outDesc = MakeTensorDesc(outDims, dequantCfg.hasDequant ? DT_FLOAT16 : dtype);

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

    bool hasOffsetW = !offsetWDims.empty();
    EsTensorHolder dataOffsetW = nullptr;
    int64_t offsetWInputIdx = hasBias ? 3 : 2;
    if (hasOffsetW) {
        auto offsetWDesc = MakeTensorDesc(offsetWDims, DT_INT8);
        dataOffsetW = graphBuilder.CreateInput(offsetWInputIdx, "dataOffsetW", DT_INT8, FORMAT_ND, offsetWDims);
        dataOffsetW.GetProducer()->UpdateOutputDesc(0, offsetWDesc);
    }

    EsTensorHolder deqScale = nullptr;
    if (dequantCfg.hasDequant) {
        int64_t deqInputIdx = 2;
        if (hasBias) {
            deqInputIdx = hasOffsetW ? 4 : 3;
        } else {
            deqInputIdx = hasOffsetW ? 3 : 2;
        }
        auto deqScaleDesc = MakeTensorDesc(dequantCfg.deqScaleDims, dequantCfg.deqScaleDtype);
        deqScale = graphBuilder.CreateInput(deqInputIdx, "deqScale", dequantCfg.deqScaleDtype, FORMAT_ND,
                                            dequantCfg.deqScaleDims);
        deqScale.GetProducer()->UpdateOutputDesc(0, deqScaleDesc);
    }

    bool isBatch = (strcmp(opType, "BatchMatMul") == 0 || strcmp(opType, "BatchMatMulV2") == 0);
    const char* transAttr1 = isBatch ? "adj_x1" : "transpose_x1";
    const char* transAttr2 = isBatch ? "adj_x2" : "transpose_x2";

    bool irHasOffsetW = (strcmp(opType, "MatMulV2") == 0 || strcmp(opType, "BatchMatMulV2") == 0);
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
    if (irHasOffsetW) {
        irAttrs.push_back({"offset_x", CompliantNodeBuilder::kEsAttrOptional, "Int", CreateFrom(int64_t(0))});
    }

    auto matmulNode = CompliantNodeBuilder(graph)
                          .OpType(opType)
                          .Name(name.c_str())
                          .IrDefInputs(irInputs)
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs(irAttrs)
                          .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *dataX1.GetProducer(), dataX1.GetProducerOutIndex(), matmulNode, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *dataX2.GetProducer(), dataX2.GetProducerOutIndex(), matmulNode, 1);
    matmulNode.UpdateInputDesc(0, x1Desc);
    matmulNode.UpdateInputDesc(1, x2Desc);
    if (hasBias) {
        AddEdgeAndUpdatePeerDesc(*graph, *dataBias.GetProducer(), dataBias.GetProducerOutIndex(), matmulNode, 2);
        matmulNode.UpdateInputDesc(2, MakeTensorDesc(biasDims, dtype));
    }
    if (hasOffsetW) {
        AddEdgeAndUpdatePeerDesc(*graph, *dataOffsetW.GetProducer(), dataOffsetW.GetProducerOutIndex(), matmulNode, 3);
        matmulNode.UpdateInputDesc(3, MakeTensorDesc(offsetWDims, DT_INT8));
    }
    auto matmulOutDesc = MakeTensorDesc(outDims, dtype);
    matmulNode.UpdateOutputDesc(0, matmulOutDesc);
    matmulNode.SetAttr(transAttr1, transX1);
    matmulNode.SetAttr(transAttr2, transX2);

    EsTensorHolder output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(matmulNode, 0));

    if (dequantCfg.hasDequant) {
        auto deqNode = CompliantNodeBuilder(graph)
                           .OpType("AscendDequant")
                           .Name("dequant")
                           .IrDefInputs({
                               {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                               {"deq_scale", CompliantNodeBuilder::kEsIrInputRequired, ""},
                           })
                           .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({
                               {"sqrt_mode", CompliantNodeBuilder::kEsAttrOptional, "Bool", CreateFrom(false)},
                               {"relu_flag", CompliantNodeBuilder::kEsAttrOptional, "Bool", CreateFrom(false)},
                               {"dtype", CompliantNodeBuilder::kEsAttrOptional, "Int",
                                CreateFrom(static_cast<int64_t>(DT_FLOAT16))},
                           })
                           .Build();
        AddEdgeAndUpdatePeerDesc(*graph, matmulNode, 0, deqNode, 0);
        AddEdgeAndUpdatePeerDesc(*graph, *deqScale.GetProducer(), deqScale.GetProducerOutIndex(), deqNode, 1);
        deqNode.UpdateOutputDesc(0, outDesc);
        output = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(deqNode, 0));
    }

    return graphBuilder.BuildAndReset({output});
}

void CheckUnsqueezeAxis(const std::shared_ptr<Graph>& graph, int64_t expectedAxis)
{
    GNode unsqueezeNode;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "Unsqueeze", unsqueezeNode));
    std::vector<int64_t> axes;
    ASSERT_EQ(unsqueezeNode.GetAttr("axes", axes), GRAPH_SUCCESS);
    ASSERT_EQ(axes.size(), 1U);
    EXPECT_EQ(axes[0], expectedAxis);
}

void CheckSqueezeAxis(const std::shared_ptr<Graph>& graph, int64_t expectedAxis)
{
    GNode squeezeNode;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "Squeeze", squeezeNode));
    std::vector<int64_t> axis;
    ASSERT_EQ(squeezeNode.GetAttr("axis", axis), GRAPH_SUCCESS);
    ASSERT_EQ(axis.size(), 1U);
    EXPECT_EQ(axis[0], expectedAxis);
}

void CheckNodeOutputDtype(const std::shared_ptr<Graph>& graph, const char* opType, DataType expectedDtype)
{
    GNode node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, opType, node));
    TensorDesc desc;
    ASSERT_EQ(node.GetOutputDesc(0, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetDataType(), expectedDtype);
}

void CheckNodeInputDtype(const std::shared_ptr<Graph>& graph, const char* opType, size_t inputIdx,
                         DataType expectedDtype)
{
    GNode node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, opType, node));
    TensorDesc desc;
    ASSERT_EQ(node.GetInputDesc(inputIdx, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetDataType(), expectedDtype);
}

void CheckNodeOutputShapeAndDtype(const std::shared_ptr<Graph>& graph, const char* opType,
                                  const std::vector<int64_t>& expectedDims, DataType expectedDtype)
{
    GNode node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, opType, node));
    TensorDesc desc;
    ASSERT_EQ(node.GetOutputDesc(0, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetShape().GetDims(), expectedDims) << "Output shape mismatch for " << opType;
    EXPECT_EQ(desc.GetDataType(), expectedDtype) << "Output dtype mismatch for " << opType;
}

void CheckNodeInputShapeAndDtype(const std::shared_ptr<Graph>& graph, const char* opType, size_t inputIdx,
                                 const std::vector<int64_t>& expectedDims, DataType expectedDtype)
{
    GNode node;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, opType, node));
    TensorDesc desc;
    ASSERT_EQ(node.GetInputDesc(inputIdx, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetShape().GetDims(), expectedDims) << "Input " << inputIdx << " shape mismatch for " << opType;
    EXPECT_EQ(desc.GetDataType(), expectedDtype) << "Input " << inputIdx << " dtype mismatch for " << opType;
}

} // namespace

class MatMulUnsqueezeSqueezeFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatformInfo("Ascend910B1", "Ascend910B"); }

    static void TearDownTestCase() {}

    void SetUp() override { SetPlatformInfo("Ascend910B1", "Ascend910B"); }

    void TearDown() override {}
};

// ===================== Pattern =====================

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, patternTest)
{
    MatMulUnsqueezeSqueezeFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

// ===================== 1dim x Ndim =====================

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMul1dX1Fp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("matMul1dX1Fp16", "MatMul", {16}, {16, 32}, {32}, DT_FLOAT16, false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);

    CheckUnsqueezeAxis(graph, 0);
    CheckSqueezeAxis(graph, 0);

    // Unsqueeze: input {16} fp16 -> output {1, 16} fp16
    CheckNodeInputShapeAndDtype(graph, "Unsqueeze", 0, {16}, DT_FLOAT16);
    CheckNodeOutputShapeAndDtype(graph, "Unsqueeze", {1, 16}, DT_FLOAT16);

    // MatMul: input0 {1,16} fp16, input1 {16,32} fp16 -> output {1,32} fp16
    CheckNodeInputShapeAndDtype(graph, "MatMul", 0, {1, 16}, DT_FLOAT16);
    CheckNodeInputShapeAndDtype(graph, "MatMul", 1, {16, 32}, DT_FLOAT16);
    CheckNodeOutputShapeAndDtype(graph, "MatMul", {1, 32}, DT_FLOAT16);

    // Squeeze: input {1,32} fp16 -> output {32} fp16
    CheckNodeInputShapeAndDtype(graph, "Squeeze", 0, {1, 32}, DT_FLOAT16);
    CheckNodeOutputShapeAndDtype(graph, "Squeeze", {32}, DT_FLOAT16);
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMul1dX1DynamicShapeFusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("matMul1dX1Dyn", "MatMul", {-1}, {-1, -1}, {-1}, DT_FLOAT16, false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);

    CheckUnsqueezeAxis(graph, 0);
    CheckSqueezeAxis(graph, 0);
}

// ===================== Ndim x 1dim =====================

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMul1dX2Fp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("matMul1dX2Fp16", "MatMul", {32, 16}, {16}, {32}, DT_FLOAT16, false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);

    CheckUnsqueezeAxis(graph, 1);
    CheckSqueezeAxis(graph, 1);

    // Unsqueeze: input {16} fp16 -> output {16, 1} fp16
    CheckNodeInputShapeAndDtype(graph, "Unsqueeze", 0, {16}, DT_FLOAT16);
    CheckNodeOutputShapeAndDtype(graph, "Unsqueeze", {16, 1}, DT_FLOAT16);

    // MatMul: input0 {32,16} fp16, input1 {16,1} fp16 -> output {32,1} fp16
    CheckNodeInputShapeAndDtype(graph, "MatMul", 0, {32, 16}, DT_FLOAT16);
    CheckNodeInputShapeAndDtype(graph, "MatMul", 1, {16, 1}, DT_FLOAT16);
    CheckNodeOutputShapeAndDtype(graph, "MatMul", {32, 1}, DT_FLOAT16);

    // Squeeze: input {32,1} fp16 -> output {32} fp16
    CheckNodeInputShapeAndDtype(graph, "Squeeze", 0, {32, 1}, DT_FLOAT16);
    CheckNodeOutputShapeAndDtype(graph, "Squeeze", {32}, DT_FLOAT16);
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMul1dX2DynamicShapeFusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("matMul1dX2Dyn", "MatMul", {-1, -1}, {-1}, {-1}, DT_FLOAT16, false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);

    CheckUnsqueezeAxis(graph, 1);
    CheckSqueezeAxis(graph, 1);
}

// ===================== BatchMatMul 1dim =====================

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, batchMatMul1dX1Fp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("batchMatMul1dX1Fp16", "BatchMatMul", {16}, {8, 16, 32}, {8, 32}, DT_FLOAT16,
                                      false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);

    CheckUnsqueezeAxis(graph, 0);
    CheckSqueezeAxis(graph, 1);

    // Unsqueeze: input {16} fp16 -> output {1, 16} fp16
    CheckNodeInputShapeAndDtype(graph, "Unsqueeze", 0, {16}, DT_FLOAT16);
    CheckNodeOutputShapeAndDtype(graph, "Unsqueeze", {1, 16}, DT_FLOAT16);

    // BatchMatMul: input0 {1,16} fp16, input1 {8,16,32} fp16 -> output {8,1,32} fp16
    CheckNodeInputShapeAndDtype(graph, "BatchMatMul", 0, {1, 16}, DT_FLOAT16);
    CheckNodeInputShapeAndDtype(graph, "BatchMatMul", 1, {8, 16, 32}, DT_FLOAT16);
    CheckNodeOutputShapeAndDtype(graph, "BatchMatMul", {8, 1, 32}, DT_FLOAT16);

    // Squeeze: input {8,1,32} fp16 -> output {8,32} fp16 (axis=1)
    CheckNodeInputShapeAndDtype(graph, "Squeeze", 0, {8, 1, 32}, DT_FLOAT16);
    CheckNodeOutputShapeAndDtype(graph, "Squeeze", {8, 32}, DT_FLOAT16);
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, batchMatMul1dX1DynamicShapeFusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("batchMatMul1dX1Dyn", "BatchMatMul", {-1}, {-1, -1, -1}, {-1, -1}, DT_FLOAT16,
                                      false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);

    CheckUnsqueezeAxis(graph, 0);
    CheckSqueezeAxis(graph, 1);
}

// ===================== 1dim x 1dim =====================

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulBoth1dFp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("matMulBoth1dFp16", "MatMul", {16}, {16}, {16}, DT_FLOAT16, false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 2);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 0);

    // MatMul output should be {1, 1} fp16 (both inputs unsqueezed to 2d)
    CheckNodeOutputShapeAndDtype(graph, "MatMul", {1, 1}, DT_FLOAT16);
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulBoth1dDequantL0c2outFusionSuccess)
{
    SetPlatformInfoWithL0c2out("Ascend950", "Ascend950");

    auto graph = BuildMatMulLikeGraph("matMulBoth1dDequant", "BatchMatMul", {16}, {16}, {16}, DT_INT8, false, false, {},
                                      {}, {true, DT_FLOAT16, {1}});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 2);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 0);
    EXPECT_EQ(CountNodes(graph, "AscendDequant"), 1);

    SetPlatformInfo("Ascend910B1", "Ascend910B");
}

// ===================== Not changed =====================

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMul2dBothFp16NotChangedFail)
{
    auto graph = BuildMatMulLikeGraph("matMul2dBothFp16", "MatMul", {16, 16}, {16, 16}, {16, 16}, DT_FLOAT16, false,
                                      false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulOutput1dOnlyNotChangedFail)
{
    auto graph = BuildMatMulLikeGraph("matMulOutput1dOnly", "MatMul", {16, 32}, {32, 16}, {16}, DT_FLOAT16, false,
                                      false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// ===================== Dequant scenarios =====================

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulWithDequantNoL0c2outFusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("matMulWithDequantNoL0c2out", "BatchMatMul", {1536}, {1536, 1000}, {1000},
                                      DT_INT8, false, false, {}, {}, {true, DT_FLOAT16, {1}});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);
    EXPECT_EQ(CountNodes(graph, "AscendDequant"), 1);

    CheckUnsqueezeAxis(graph, 0);
    CheckSqueezeAxis(graph, 0);
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulWithDequantL0c2outFusionSuccess)
{
    SetPlatformInfoWithL0c2out("Ascend950", "Ascend950");

    auto graph = BuildMatMulLikeGraph("matMulWithDequantL0c2out", "BatchMatMul", {1536}, {1536, 1000}, {1000}, DT_INT8,
                                      false, false, {}, {}, {true, DT_FLOAT16, {1}});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);
    EXPECT_EQ(CountNodes(graph, "AscendDequant"), 1);

    CheckUnsqueezeAxis(graph, 0);
    CheckSqueezeAxis(graph, 0);

    SetPlatformInfo("Ascend910B1", "Ascend910B");
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulBoth1dWithDequantL0c2outFusionSuccess)
{
    SetPlatformInfoWithL0c2out("Ascend950", "Ascend950");

    auto graph = BuildMatMulLikeGraph("matMulBoth1dDequantL0c2out", "BatchMatMul", {-1}, {-1}, {-1}, DT_INT8, false,
                                      false, {}, {}, {true, DT_FLOAT16, {1}});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 2);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 0);
    EXPECT_EQ(CountNodes(graph, "AscendDequant"), 1);

    SetPlatformInfo("Ascend910B1", "Ascend910B");
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulV2WithBiasDequantL0c2outFusionSuccess)
{
    SetPlatformInfoWithL0c2out("Ascend950", "Ascend950");

    auto graph = BuildMatMulLikeGraph("matMulV2BiasDequant", "MatMulV2", {1536}, {1536, 1000}, {1000}, DT_INT8, false,
                                      false, {1000}, {}, {true, DT_FLOAT16, {1}});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);
    EXPECT_EQ(CountNodes(graph, "AscendDequant"), 1);

    CheckUnsqueezeAxis(graph, 0);
    CheckSqueezeAxis(graph, 0);

    SetPlatformInfo("Ascend910B1", "Ascend910B");
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulV2WithOffsetWDequantL0c2outFusionSuccess)
{
    SetPlatformInfoWithL0c2out("Ascend950", "Ascend950");

    auto graph = BuildMatMulLikeGraph("matMulV2OffsetWDequant", "MatMulV2", {1536}, {1536, 1000}, {1000}, DT_INT8,
                                      false, false, {}, {1}, {true, DT_FLOAT16, {1}});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);
    EXPECT_EQ(CountNodes(graph, "AscendDequant"), 1);

    CheckUnsqueezeAxis(graph, 0);
    CheckSqueezeAxis(graph, 0);

    SetPlatformInfo("Ascend910B1", "Ascend910B");
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulX2OneDimDequantL0c2outFusionSuccess)
{
    SetPlatformInfoWithL0c2out("Ascend950", "Ascend950");

    auto graph = BuildMatMulLikeGraph("matMulX2OneDimDequant", "MatMul", {1000, 1536}, {1536}, {1000}, DT_INT8, false,
                                      false, {}, {}, {true, DT_FLOAT16, {1}});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);
    EXPECT_EQ(CountNodes(graph, "AscendDequant"), 1);

    CheckUnsqueezeAxis(graph, 1);
    CheckSqueezeAxis(graph, 1);

    SetPlatformInfo("Ascend910B1", "Ascend910B");
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, batchMatMulX2OneDimDequantL0c2outFusionSuccess)
{
    SetPlatformInfoWithL0c2out("Ascend950", "Ascend950");

    auto graph = BuildMatMulLikeGraph("batchMatMulX2OneDimDequant", "BatchMatMul", {8, 1000, 1536}, {1536}, {8, 1000},
                                      DT_INT8, false, false, {}, {}, {true, DT_FLOAT16, {1}});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);
    EXPECT_EQ(CountNodes(graph, "AscendDequant"), 1);

    CheckUnsqueezeAxis(graph, 1);
    CheckSqueezeAxis(graph, 2);

    SetPlatformInfo("Ascend910B1", "Ascend910B");
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, batchMatMulV2X2OneDimFp16FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("batchMatMulV2X2Fp16", "BatchMatMulV2", {2, 32, 16}, {16}, {2, 32}, DT_FLOAT16,
                                      false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);

    CheckUnsqueezeAxis(graph, 1);
    CheckSqueezeAxis(graph, 2);
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulV2X2OneDimDynamicShapeFusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("matMulV2X2Dyn", "MatMulV2", {-1, -1}, {-1}, {-1}, DT_FLOAT16, false, false);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);

    CheckUnsqueezeAxis(graph, 1);
    CheckSqueezeAxis(graph, 1);
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulTransposeAttrTransferSuccess)
{
    auto graph = BuildMatMulLikeGraph("matMulTransAttr", "MatMul", {16}, {16, 32}, {32}, DT_FLOAT16, true, true);

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);

    GNode matmulNode;
    ASSERT_TRUE(FindFirstNodeByOpType(graph, "MatMul", matmulNode));
    bool transX1 = false;
    bool transX2 = false;
    ASSERT_EQ(matmulNode.GetAttr("transpose_x1", transX1), GRAPH_SUCCESS);
    ASSERT_EQ(matmulNode.GetAttr("transpose_x2", transX2), GRAPH_SUCCESS);
    EXPECT_TRUE(transX1);
    EXPECT_TRUE(transX2);
}

TEST_F(MatMulUnsqueezeSqueezeFusionPassTest, matMulBf16BiasTransX2FusionSuccess)
{
    auto graph = BuildMatMulLikeGraph("matMulBf16BiasTransX2", "MatMul", {2032}, {1310, 2032}, {1310}, DT_BF16, false,
                                      true, {1310});

    CustomPassContext passContext;
    passContext.SetPassName(kPassName);
    MatMulUnsqueezeSqueezeFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, "Unsqueeze"), 1);
    EXPECT_EQ(CountNodes(graph, "Squeeze"), 1);
}
