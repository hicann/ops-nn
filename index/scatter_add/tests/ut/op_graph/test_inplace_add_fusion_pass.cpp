/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "es_nn_ops.h"
#include "../../../op_graph/fusion_pass/inplace_add_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {
const char* kInplaceAddType = "InplaceAdd";
const char* kTensorMoveType = "TensorMove";
const char* kScatterAddType = "ScatterAdd";

// Build a single InplaceAdd(x, indices, v) -> y graph for testing.
// InplaceAdd has no ES API, so the node is built with CompliantNodeBuilder.
// After connecting edges the input descs are mirrored onto the node so that the
// dtype guard in the fusion pass reads the real dtype (avoid false-green UT).
std::shared_ptr<Graph> BuildInplaceAddGraph(const std::vector<int64_t>& xDims, DataType xDtype,
                                            const std::vector<int64_t>& indicesDims, DataType indicesDtype,
                                            const std::vector<int64_t>& vDims, DataType vDtype)
{
    auto graphBuilder = es::EsGraphBuilder("inplace_add_test");
    auto x = graphBuilder.CreateInput(0, "x", xDtype, FORMAT_ND, xDims);
    auto indices = graphBuilder.CreateInput(1, "indices", indicesDtype, FORMAT_ND, indicesDims);
    auto v = graphBuilder.CreateInput(2, "v", vDtype, FORMAT_ND, vDims);

    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto inplaceAdd = es::CompliantNodeBuilder(graph)
                          .OpType(kInplaceAddType)
                          .Name("InplaceAdd")
                          .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"v", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();

    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), inplaceAdd, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *indices.GetProducer(), indices.GetProducerOutIndex(), inplaceAdd, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *v.GetProducer(), v.GetProducerOutIndex(), inplaceAdd, 2);

    // Mirror real input/output descs onto the InplaceAdd node so dtype guard is exercised.
    TensorDesc xDesc(Shape(xDims), FORMAT_ND, xDtype);
    TensorDesc indicesDesc(Shape(indicesDims), FORMAT_ND, indicesDtype);
    TensorDesc vDesc(Shape(vDims), FORMAT_ND, vDtype);
    inplaceAdd.UpdateInputDesc(0, xDesc);
    inplaceAdd.UpdateInputDesc(1, indicesDesc);
    inplaceAdd.UpdateInputDesc(2, vDesc);
    inplaceAdd.UpdateOutputDesc(0, xDesc);

    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(inplaceAdd, 0);
    auto y = es::EsTensorHolder(yHolder);
    return graphBuilder.BuildAndReset({y});
}

int CountNodeByType(const std::shared_ptr<Graph>& graph, const char* opType)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == opType) {
            count++;
        }
    }
    return count;
}
} // namespace

class InplaceAddFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};
TEST_F(InplaceAddFusionPassTest, inplaceAddFusionPatternTest)
{
    ops::AInplaceAddFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(InplaceAddFusionPassTest, inplaceAddFusionFloatSuccess)
{
    auto graph = BuildInplaceAddGraph({4, 8}, DT_FLOAT, {2}, DT_INT32, {2, 8}, DT_FLOAT);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 0);
    EXPECT_EQ(CountNodeByType(graph, kTensorMoveType), 1);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 1);

    // The fused ScatterAdd must carry the inplace fusion option.
    bool foundInplaceOption = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == kScatterAddType) {
            AscendString option;
            if (node.GetAttr("fusion_op_build_options", option) == GRAPH_SUCCESS) {
                foundInplaceOption = std::string(option.GetString()) == "{\"is_inplace\",True}";
            }
        }
    }
    EXPECT_TRUE(foundInplaceOption);
}

TEST_F(InplaceAddFusionPassTest, inplaceAddFusionFloat16Success)
{
    auto graph = BuildInplaceAddGraph({4, 8}, DT_FLOAT16, {2}, DT_INT32, {2, 8}, DT_FLOAT16);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 0);
    EXPECT_EQ(CountNodeByType(graph, kTensorMoveType), 1);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 1);
}

TEST_F(InplaceAddFusionPassTest, inplaceAddFusionBf16Success)
{
    auto graph = BuildInplaceAddGraph({4, 8}, DT_BF16, {2}, DT_INT64, {2, 8}, DT_BF16);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 1);
}

TEST_F(InplaceAddFusionPassTest, inplaceAddFusionInt32Success)
{
    auto graph = BuildInplaceAddGraph({4, 8}, DT_INT32, {2}, DT_INT64, {2, 8}, DT_INT32);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 1);
}

TEST_F(InplaceAddFusionPassTest, inplaceAddFusionInt64IndicesSuccess)
{
    auto graph = BuildInplaceAddGraph({6, 4}, DT_FLOAT, {3}, DT_INT64, {3, 4}, DT_FLOAT);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 0);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 1);
}

TEST_F(InplaceAddFusionPassTest, inplaceAddFusionHighDimSuccess)
{
    auto graph = BuildInplaceAddGraph({4, 3, 5, 6}, DT_FLOAT, {2}, DT_INT32, {2, 3, 5, 6}, DT_FLOAT);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 1);
}

// Negative: data dtype not supported by ScatterAdd -> not fused.
TEST_F(InplaceAddFusionPassTest, inplaceAddFusionUnsupportedDtypeFail)
{
    auto graph = BuildInplaceAddGraph({4, 8}, DT_DOUBLE, {2}, DT_INT32, {2, 8}, DT_DOUBLE);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 1);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 0);
}

// Negative: indices dtype not in {int32, int64} -> not fused.
TEST_F(InplaceAddFusionPassTest, inplaceAddFusionUnsupportedIndicesDtypeFail)
{
    auto graph = BuildInplaceAddGraph({4, 8}, DT_FLOAT, {2}, DT_INT16, {2, 8}, DT_FLOAT);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 1);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 0);
}

// Negative: Ascend310 is the only excluded platform in the original pass.
TEST_F(InplaceAddFusionPassTest, inplaceAddFusionUnsupportedPlatformFail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend310";
    optiCompilationInfo.soc_version = "Ascend310";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend310"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto graph = BuildInplaceAddGraph({4, 8}, DT_FLOAT, {2}, DT_INT32, {2, 8}, DT_FLOAT);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 1);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 0);
}

// Positive: Ascend950 (a non-310 platform) still fuses.
TEST_F(InplaceAddFusionPassTest, inplaceAddFusion950Success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto graph = BuildInplaceAddGraph({4, 8}, DT_FLOAT, {2}, DT_INT32, {2, 8}, DT_FLOAT);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 1);
}