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

void SetPlatform(const std::string& socVersion, const std::string& shortSocVersion = "")
{
    fe::PlatformInfo platformInfo{};
    fe::OptionalInfo optiCompilationInfo{};
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = shortSocVersion.empty() ? socVersion : shortSocVersion;
    optiCompilationInfo.soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}
} // namespace

class InplaceAddFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatform("Ascend910_93"); }

    void SetUp() override { SetPlatform("Ascend910_93"); }
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

TEST_F(InplaceAddFusionPassTest, inplaceAddFusionInt8AndUint8UnsupportedOnAscend910_93)
{
    const std::vector<DataType> dataDtypes = {DT_INT8, DT_UINT8};
    for (const auto dtype : dataDtypes) {
        SCOPED_TRACE(static_cast<int32_t>(dtype));
        auto graph = BuildInplaceAddGraph({4, 8}, dtype, {2}, DT_INT32, {2, 8}, dtype);
        CustomPassContext passContext;
        ops::AInplaceAddFusionPass pass;
        Status status = pass.Run(graph, passContext);

        EXPECT_EQ(status, GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 1);
        EXPECT_EQ(CountNodeByType(graph, kTensorMoveType), 0);
        EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 0);
    }
}

TEST_F(InplaceAddFusionPassTest, inplaceAddFusionBf16UnsupportedOnAscend910_93)
{
    auto graph = BuildInplaceAddGraph({4, 8}, DT_BF16, {2}, DT_INT64, {2, 8}, DT_BF16);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 1);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 0);
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

TEST_F(InplaceAddFusionPassTest, inplaceAddFusionAscend910DtypeMatrix)
{
    SetPlatform("Ascend910A", "Ascend910");
    struct DtypeScenario {
        DataType dtype;
        bool shouldFuse;
    };
    const std::vector<DtypeScenario> scenarios = {
        {DT_FLOAT, true}, {DT_INT32, true}, {DT_FLOAT16, false}, {DT_INT8, false}, {DT_UINT8, false}, {DT_BF16, false},
    };

    for (const auto& scenario : scenarios) {
        SCOPED_TRACE(static_cast<int32_t>(scenario.dtype));
        auto graph = BuildInplaceAddGraph({4, 8}, scenario.dtype, {2}, DT_INT32, {2, 8}, scenario.dtype);
        CustomPassContext passContext;
        ops::AInplaceAddFusionPass pass;
        Status status = pass.Run(graph, passContext);

        EXPECT_EQ(status, scenario.shouldFuse ? SUCCESS : GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), scenario.shouldFuse ? 0 : 1);
        EXPECT_EQ(CountNodeByType(graph, kTensorMoveType), scenario.shouldFuse ? 1 : 0);
        EXPECT_EQ(CountNodeByType(graph, kScatterAddType), scenario.shouldFuse ? 1 : 0);
    }
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

// Negative: Ascend310 does not use the legacy lowering.
TEST_F(InplaceAddFusionPassTest, inplaceAddFusionUnsupportedPlatformFail)
{
    SetPlatform("Ascend310");

    auto graph = BuildInplaceAddGraph({4, 8}, DT_FLOAT, {2}, DT_INT32, {2, 8}, DT_FLOAT);
    CustomPassContext passContext;
    ops::AInplaceAddFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 1);
    EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 0);
}

// Ascend950 keeps the original graph node for every dtype supported by the native InplaceAdd implementation.
TEST_F(InplaceAddFusionPassTest, inplaceAddFusion950SkipsAllNativeDataDtypes)
{
    SetPlatform("Ascend950");

    const std::vector<DataType> nativeDataDtypes = {
        DT_FLOAT16, DT_FLOAT,  DT_BF16,   DT_INT8,   DT_INT16,     DT_INT32,     DT_INT64,
        DT_UINT8,   DT_UINT16, DT_UINT32, DT_UINT64, DT_COMPLEX32, DT_COMPLEX64,
    };
    for (const auto dtype : nativeDataDtypes) {
        SCOPED_TRACE(static_cast<int32_t>(dtype));
        auto graph = BuildInplaceAddGraph({4, 8}, dtype, {2}, DT_INT32, {2, 8}, dtype);
        CustomPassContext passContext;
        ops::AInplaceAddFusionPass pass;
        Status status = pass.Run(graph, passContext);

        EXPECT_EQ(status, GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 1);
        EXPECT_EQ(CountNodeByType(graph, kTensorMoveType), 0);
        EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 0);
    }
}

TEST_F(InplaceAddFusionPassTest, inplaceAddFusion950SkipsNativeShapeBoundaries)
{
    SetPlatform("Ascend950");

    struct ShapeScenario {
        std::vector<int64_t> x;
        std::vector<int64_t> indices;
        std::vector<int64_t> v;
    };
    const std::vector<ShapeScenario> nativeShapeBoundaries = {
        {{4}, {2}, {2}},
        {{4, 2, 1, 1, 1, 1, 1, 1}, {2}, {2, 2, 1, 1, 1, 1, 1, 1}},
        {{4, 8}, {0}, {0, 8}},
        {{0, 8}, {0}, {0, 8}},
        {{4, 0, 3}, {2}, {2, 0, 3}},
    };

    for (const auto& shapes : nativeShapeBoundaries) {
        SCOPED_TRACE(testing::PrintToString(shapes.x));
        auto graph = BuildInplaceAddGraph(shapes.x, DT_FLOAT, shapes.indices, DT_INT32, shapes.v, DT_FLOAT);
        CustomPassContext passContext;
        ops::AInplaceAddFusionPass pass;
        Status status = pass.Run(graph, passContext);

        EXPECT_EQ(status, GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeByType(graph, kInplaceAddType), 1);
        EXPECT_EQ(CountNodeByType(graph, kTensorMoveType), 0);
        EXPECT_EQ(CountNodeByType(graph, kScatterAddType), 0);
    }
}
