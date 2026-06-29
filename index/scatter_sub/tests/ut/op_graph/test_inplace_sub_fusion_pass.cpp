/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_nn_ops.h"
#include "compliant_node_builder.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/inplace_sub_fusion_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace OPS::NN;

// ---------------------------------------------------------------------------
// 辅助函数:构建包含 InplaceSub 节点的测试图
// InplaceSub 无 ES API,使用 CompliantNodeBuilder
// ---------------------------------------------------------------------------
static std::shared_ptr<Graph> BuildTestGraph(
    DataType xDtype,
    const std::vector<int64_t>& xDims,
    const std::vector<int64_t>& indicesDims,
    const std::vector<int64_t>& vDims,
    DataType indicesDtype = DT_INT32)
{
    auto graphBuilder = es::EsGraphBuilder("test_graph");
    auto x = graphBuilder.CreateInput(0, "x", xDtype, FORMAT_ND, xDims);
    auto indices = graphBuilder.CreateInput(1, "indices", indicesDtype, FORMAT_ND, indicesDims);
    auto v = graphBuilder.CreateInput(2, "v", xDtype, FORMAT_ND, vDims);

    ge::Graph* graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();

    GNode opNode = es::CompliantNodeBuilder(graphPtr)
        .OpType("InplaceSub")
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                       {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                       {"v", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();

    GNode xNode = *x.GetProducer();
    GNode indicesNode = *indices.GetProducer();
    GNode vNode = *v.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, xNode, 0, opNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, indicesNode, 0, opNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, vNode, 0, opNode, 2);

    // 设置输入节点的 OutputDesc
    TensorDesc xOutDesc;
    xNode.GetOutputDesc(0, xOutDesc);
    xOutDesc.SetDataType(xDtype);
    xOutDesc.SetShape(Shape(xDims));
    xNode.UpdateOutputDesc(0, xOutDesc);

    TensorDesc indicesOutDesc;
    indicesNode.GetOutputDesc(0, indicesOutDesc);
    indicesOutDesc.SetDataType(indicesDtype);
    indicesOutDesc.SetShape(Shape(indicesDims));
    indicesNode.UpdateOutputDesc(0, indicesOutDesc);

    TensorDesc vOutDesc;
    vNode.GetOutputDesc(0, vOutDesc);
    vOutDesc.SetDataType(xDtype);
    vOutDesc.SetShape(Shape(vDims));
    vNode.UpdateOutputDesc(0, vOutDesc);

    // 设置算子节点的输入 TensorDesc
    TensorDesc opIn0Desc;
    opNode.GetInputDesc(0, opIn0Desc);
    opIn0Desc.SetDataType(xDtype);
    opIn0Desc.SetShape(Shape(xDims));
    opIn0Desc.SetFormat(FORMAT_ND);
    opNode.UpdateInputDesc(0, opIn0Desc);

    TensorDesc opIn1Desc;
    opNode.GetInputDesc(1, opIn1Desc);
    opIn1Desc.SetDataType(indicesDtype);
    opIn1Desc.SetShape(Shape(indicesDims));
    opIn1Desc.SetFormat(FORMAT_ND);
    opNode.UpdateInputDesc(1, opIn1Desc);

    TensorDesc opIn2Desc;
    opNode.GetInputDesc(2, opIn2Desc);
    opIn2Desc.SetDataType(xDtype);
    opIn2Desc.SetShape(Shape(vDims));
    opIn2Desc.SetFormat(FORMAT_ND);
    opNode.UpdateInputDesc(2, opIn2Desc);

    // 设置算子节点的输出 TensorDesc
    TensorDesc opOutDesc;
    opNode.GetOutputDesc(0, opOutDesc);
    opOutDesc.SetDataType(xDtype);
    opOutDesc.SetShape(Shape(xDims));
    opOutDesc.SetFormat(FORMAT_ND);
    opNode.UpdateOutputDesc(0, opOutDesc);

    es::EsTensorHolder output(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(opNode, 0));
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});
    return graph;
}

// ---------------------------------------------------------------------------
// 辅助函数:在图中查找指定类型的节点
// ---------------------------------------------------------------------------
static bool FindNodeByType(const std::shared_ptr<Graph>& graph, const std::string& typeName)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == typeName.c_str()) {
            return true;
        }
    }
    return false;
}

static void SetPlatform(const std::string& socVersion)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = socVersion;
    optiCompilationInfo.soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

// ===========================================================================
// 测试类
// ===========================================================================
class AInplaceSubFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        SetPlatform("Ascend950");
    }

    void SetUp() override
    {
        SetPlatform("Ascend950");
    }
};

// ===========================================================================
// 测试用例
// ===========================================================================

TEST_F(AInplaceSubFusionPassTest, patternTest)
{
    OPS::NN::AInplaceSubFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_EQ(patterns.size(), 1);
}

TEST_F(AInplaceSubFusionPassTest, inplaceSubFloatSuccess)
{
    auto graph = BuildTestGraph(DT_FLOAT, {33, 5}, {600}, {600, 5});
    CustomPassContext passContext;
    OPS::NN::AInplaceSubFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterSub"));
    EXPECT_FALSE(FindNodeByType(graph, "InplaceSub"));
}

TEST_F(AInplaceSubFusionPassTest, inplaceSubFloat16Success)
{
    auto graph = BuildTestGraph(DT_FLOAT16, {8, 16}, {4}, {4, 16});
    CustomPassContext passContext;
    OPS::NN::AInplaceSubFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterSub"));
    EXPECT_FALSE(FindNodeByType(graph, "InplaceSub"));
}

TEST_F(AInplaceSubFusionPassTest, inplaceSubInt32Success)
{
    auto graph = BuildTestGraph(DT_INT32, {4, 4}, {2}, {2, 4});
    CustomPassContext passContext;
    OPS::NN::AInplaceSubFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterSub"));
}

TEST_F(AInplaceSubFusionPassTest, inplaceSub1dSuccess)
{
    auto graph = BuildTestGraph(DT_FLOAT, {8}, {3}, {3});
    CustomPassContext passContext;
    OPS::NN::AInplaceSubFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterSub"));
}

TEST_F(AInplaceSubFusionPassTest, inplaceSubHighDimSuccess)
{
    auto graph = BuildTestGraph(DT_FLOAT, {6, 3, 4, 5}, {2}, {2, 3, 4, 5});
    CustomPassContext passContext;
    OPS::NN::AInplaceSubFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterSub"));
}

TEST_F(AInplaceSubFusionPassTest, inplaceSubAscend910_93Success)
{
    SetPlatform("Ascend910_93");
    auto graph = BuildTestGraph(DT_FLOAT, {33, 5}, {600}, {600, 5});
    CustomPassContext passContext;
    OPS::NN::AInplaceSubFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(FindNodeByType(graph, "TensorMove"));
    EXPECT_TRUE(FindNodeByType(graph, "ScatterSub"));
}

// 不支持平台:Ascend310 不融合,InplaceSub 保留
TEST_F(AInplaceSubFusionPassTest, unsupportedPlatformFail)
{
    SetPlatform("Ascend310");
    auto graph = BuildTestGraph(DT_FLOAT, {33, 5}, {600}, {600, 5});
    CustomPassContext passContext;
    OPS::NN::AInplaceSubFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(FindNodeByType(graph, "TensorMove"));
    EXPECT_FALSE(FindNodeByType(graph, "ScatterSub"));
    EXPECT_TRUE(FindNodeByType(graph, "InplaceSub"));
}

// 不支持的 data dtype(如 DT_DOUBLE):ScatterSub 不支持,不融合
TEST_F(AInplaceSubFusionPassTest, unsupportedDtypeFail)
{
    SetPlatform("Ascend950");
    auto graph = BuildTestGraph(DT_DOUBLE, {4, 8}, {2}, {2, 8});
    CustomPassContext passContext;
    OPS::NN::AInplaceSubFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(FindNodeByType(graph, "TensorMove"));
    EXPECT_FALSE(FindNodeByType(graph, "ScatterSub"));
    EXPECT_TRUE(FindNodeByType(graph, "InplaceSub"));
}

// 不支持的 indices dtype(非 int32/int64,如 DT_INT16):不融合
TEST_F(AInplaceSubFusionPassTest, unsupportedIndicesDtypeFail)
{
    SetPlatform("Ascend950");
    auto graph = BuildTestGraph(DT_FLOAT, {4, 8}, {2}, {2, 8}, DT_INT16);
    CustomPassContext passContext;
    OPS::NN::AInplaceSubFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(FindNodeByType(graph, "TensorMove"));
    EXPECT_FALSE(FindNodeByType(graph, "ScatterSub"));
    EXPECT_TRUE(FindNodeByType(graph, "InplaceSub"));
}