/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "../../../op_graph/fusion_pass/sorted_sparse_segment_mean_grad_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {
const char* kSourceOpType = "SparseSegmentMeanGrad";
const char* kSortOpType = "Sort";
const char* kTargetOpType = "SortedSparseSegmentMeanGrad";

void SetPlatform(const std::string& soc)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = soc;
    optionalInfo.soc_version = soc;
    fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

// Build a SparseSegmentMeanGrad node (no ES API) via CompliantNodeBuilder, wiring its 4 inputs.
es::EsTensorHolder CreateSparseSegmentMeanGradNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& grad,
                                                   const es::EsTensorHolder& indices,
                                                   const es::EsTensorHolder& segmentIds,
                                                   const es::EsTensorHolder& outputDim0, DataType gradDtype)
{
    auto CheckGraphSuccess = [](graphStatus status, const char* expr) {
        if (status != GRAPH_SUCCESS) {
            ADD_FAILURE() << expr << " failed, status=" << status;
            return false;
        }
        return true;
    };

    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto node = es::CompliantNodeBuilder(graph)
                    .OpType(kSourceOpType)
                    .Name(kSourceOpType)
                    .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"indices", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"segment_ids", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"output_dim0", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", gradDtype)
                    .InstanceOutputShape("y", {8, 10})
                    .InstanceOutputFormat("y", FORMAT_ND)
                    .Build();

    const es::EsTensorHolder* inputs[] = {&grad, &indices, &segmentIds, &outputDim0};
    for (int32_t i = 0; i < 4; ++i) {
        auto& producer = *inputs[i]->GetProducer();
        int32_t producerOutIdx = inputs[i]->GetProducerOutIndex();
        if (!CheckGraphSuccess(es::AddEdgeAndUpdatePeerDesc(*graph, producer, producerOutIdx, node, i),
                               "AddEdgeAndUpdatePeerDesc")) {
            return {};
        }
        // CompliantNodeBuilder cannot set input dtypes, and AddEdgeAndUpdatePeerDesc does not propagate
        // the producer dtype onto this node's input desc. Mirror it explicitly so the fusion pass sees the
        // real grad/indices dtypes (otherwise the unsupported-dtype guard cannot be exercised).
        TensorDesc producerOutDesc;
        if (!CheckGraphSuccess(producer.GetOutputDesc(producerOutIdx, producerOutDesc), "GetOutputDesc")) {
            return {};
        }
        if (!CheckGraphSuccess(node.UpdateInputDesc(i, producerOutDesc), "UpdateInputDesc")) {
            return {};
        }
    }

    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0);
    return es::EsTensorHolder(yHolder);
}

bool FindNodeType(const std::shared_ptr<Graph>& graph, const char* type)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (nodeType == type) {
            return true;
        }
    }
    return false;
}

// Build the full test graph: 4 Data inputs -> SparseSegmentMeanGrad.
std::shared_ptr<Graph> BuildGraph(const std::string& name, DataType gradDtype, DataType indicesDtype)
{
    auto graphBuilder = es::EsGraphBuilder(name.c_str());
    auto grad = graphBuilder.CreateInput(0, "x", gradDtype, FORMAT_ND, {8, 10});
    auto indices = graphBuilder.CreateInput(1, "indices", indicesDtype, FORMAT_ND, {8});
    auto segmentIds = graphBuilder.CreateInput(2, "segment_ids", indicesDtype, FORMAT_ND, {8});
    auto outputDim0 = graphBuilder.CreateInput(3, "output_dim0", DT_INT32, FORMAT_ND, {});
    auto y = CreateSparseSegmentMeanGradNode(graphBuilder, grad, indices, segmentIds, outputDim0, gradDtype);
    return graphBuilder.BuildAndReset({y});
}
} // namespace

class SortedSparseSegmentMeanGradFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }
};

// test1: fp32 grad + int32 indices -> fusion success
TEST_F(SortedSparseSegmentMeanGradFusionPassTest, sortedSparseSegmentMeanGradFusionFp32Int32Success)
{
    auto graph = BuildGraph("fusion_fp32_int32", DT_FLOAT, DT_INT32);

    CustomPassContext passContext;
    SortedSparseSegmentMeanGradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_TRUE(FindNodeType(graph, kSortOpType));
    EXPECT_TRUE(FindNodeType(graph, kTargetOpType));
    EXPECT_FALSE(FindNodeType(graph, kSourceOpType));
}

// test2: fp16 grad + int64 indices -> fusion success
TEST_F(SortedSparseSegmentMeanGradFusionPassTest, sortedSparseSegmentMeanGradFusionFp16Int64Success)
{
    auto graph = BuildGraph("fusion_fp16_int64", DT_FLOAT16, DT_INT64);

    CustomPassContext passContext;
    SortedSparseSegmentMeanGradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_TRUE(FindNodeType(graph, kSortOpType));
    EXPECT_TRUE(FindNodeType(graph, kTargetOpType));
    EXPECT_FALSE(FindNodeType(graph, kSourceOpType));
}

// test3: bf16 grad + int32 indices -> fusion success
TEST_F(SortedSparseSegmentMeanGradFusionPassTest, sortedSparseSegmentMeanGradFusionBf16Int32Success)
{
    auto graph = BuildGraph("fusion_bf16_int32", DT_BF16, DT_INT32);

    CustomPassContext passContext;
    SortedSparseSegmentMeanGradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_TRUE(FindNodeType(graph, kSortOpType));
    EXPECT_TRUE(FindNodeType(graph, kTargetOpType));
    EXPECT_FALSE(FindNodeType(graph, kSourceOpType));
}

// test4: non-regbase platform Ascend910_93 -> not changed (op only ships arch35/Ascend950 impl)
TEST_F(SortedSparseSegmentMeanGradFusionPassTest, sortedSparseSegmentMeanGrad910_93NotFused)
{
    SetPlatform("Ascend910_93");
    auto graph = BuildGraph("non_regbase_910_93", DT_FLOAT, DT_INT32);

    CustomPassContext passContext;
    SortedSparseSegmentMeanGradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    EXPECT_TRUE(FindNodeType(graph, kSourceOpType));
    EXPECT_FALSE(FindNodeType(graph, kTargetOpType));
}

// test5: unsupported platform Ascend310 -> not changed
TEST_F(SortedSparseSegmentMeanGradFusionPassTest, sortedSparseSegmentMeanGradUnsupportedPlatformFail)
{
    SetPlatform("Ascend310");
    auto graph = BuildGraph("unsupported_platform", DT_FLOAT, DT_INT32);

    CustomPassContext passContext;
    SortedSparseSegmentMeanGradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    EXPECT_TRUE(FindNodeType(graph, kSourceOpType));
    EXPECT_FALSE(FindNodeType(graph, kTargetOpType));
}

// test6: unsupported grad dtype DT_INT32 -> not changed
TEST_F(SortedSparseSegmentMeanGradFusionPassTest, sortedSparseSegmentMeanGradUnsupportedDtypeFail)
{
    auto graph = BuildGraph("unsupported_dtype", DT_INT32, DT_INT32);

    CustomPassContext passContext;
    SortedSparseSegmentMeanGradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    EXPECT_TRUE(FindNodeType(graph, kSourceOpType));
    EXPECT_FALSE(FindNodeType(graph, kTargetOpType));
}

// test7: the other regbase platform MC62CM12A -> fusion success
TEST_F(SortedSparseSegmentMeanGradFusionPassTest, sortedSparseSegmentMeanGradFusionMc62Success)
{
    SetPlatform("MC62CM12A");
    auto graph = BuildGraph("fusion_mc62", DT_FLOAT, DT_INT32);

    CustomPassContext passContext;
    SortedSparseSegmentMeanGradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    EXPECT_TRUE(FindNodeType(graph, kSortOpType));
    EXPECT_TRUE(FindNodeType(graph, kTargetOpType));
    EXPECT_FALSE(FindNodeType(graph, kSourceOpType));
}