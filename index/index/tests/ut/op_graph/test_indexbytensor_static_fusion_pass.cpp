/**
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
#include "../../../op_graph/fusion_pass/indexbytensor_static_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

class IndexByTensorStaticFusionPassTest : public testing::Test {
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

static GraphPtr BuildIndexByTensorGraph(const std::string& graphName,
                                        const std::vector<int64_t>& xDims, DataType xDtype, Format xFormat,
                                        const std::vector<std::vector<int64_t>>& indicesDims,
                                        const std::vector<int64_t>& indicesMask,
                                        const std::vector<int64_t>& yDims)
{
    es::EsGraphBuilder graphBuilder(graphName.c_str());
    auto graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto x = graphBuilder.CreateInput(0, "x", xDtype, xFormat, xDims);

    std::vector<es::EsTensorHolder> rIndices;
    for (size_t i = 0; i < indicesDims.size(); i++) {
        std::string idxName = "indices" + std::to_string(i);
        auto rIdx = graphBuilder.CreateInput(static_cast<int64_t>(i + 1), idxName.c_str(),
                                             DT_INT64, FORMAT_ND, indicesDims[i]);
        rIndices.push_back(rIdx);
    }

    auto ibtBuilder = es::CompliantNodeBuilder(graphPtr);
    ibtBuilder.OpType("IndexByTensor")
        .Name("IndexByTensor")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"indices", es::CompliantNodeBuilder::kEsIrInputDynamic, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .IrDefAttrs({
            {"indices_mask", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt",
             es::CreateFrom(indicesMask)}
        })
        .InstanceDynamicInputNum("indices", static_cast<int32_t>(indicesDims.size()));

    auto ibtNode = ibtBuilder.Build();

    auto xDataNode = x.GetProducer();
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *xDataNode, 0, ibtNode, 0);

    for (size_t i = 0; i < rIndices.size(); i++) {
        auto idxDataNode = rIndices[i].GetProducer();
        es::AddEdgeAndUpdatePeerDesc(*graphPtr, *idxDataNode, 0, ibtNode,
                                     static_cast<int32_t>(1 + i));
    }

    TensorDesc xDesc(Shape(xDims), xFormat, xDtype);
    ibtNode.UpdateInputDesc(0, xDesc);

    for (size_t i = 0; i < indicesDims.size(); i++) {
        TensorDesc idxDesc(Shape(indicesDims[i]), FORMAT_ND, DT_INT64);
        ibtNode.UpdateInputDesc(static_cast<uint32_t>(1 + i), idxDesc);
    }

    TensorDesc yDesc(Shape(yDims), xFormat, xDtype);
    ibtNode.UpdateOutputDesc(0, yDesc);

    es::EsGraphBuilder::SetOutput(x, 0);
    std::unique_ptr<Graph> graphUnique = graphBuilder.BuildAndReset();

    std::vector<std::pair<GNode, int32_t>> graphOutputs;
    graphOutputs.push_back(std::make_pair(ibtNode, 0));
    graphUnique->SetOutputs(graphOutputs);

    return GraphPtr(std::move(graphUnique));
}

TEST_F(IndexByTensorStaticFusionPassTest, indexbytensorFusionTwoIndicesSuccess)
{
    std::vector<int64_t> dimsX{2, 3, 4, 5};
    std::vector<std::vector<int64_t>> indicesDims{{3, 2}, {3, 2}};
    std::vector<int64_t> indicesMask{0, 1, 1};
    std::vector<int64_t> dimsY{2, 3, 2, 5};

    GraphPtr graph = BuildIndexByTensorGraph("indexbytensor_fusion_test1", dimsX, DT_FLOAT, FORMAT_ND,
                                             indicesDims, indicesMask, dimsY);

    CustomPassContext passContext;
    ops::IndexByTensorStaticFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool findIndex = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "Index") {
            findIndex = true;
        }
    }
    EXPECT_EQ(findIndex, true);
}

TEST_F(IndexByTensorStaticFusionPassTest, indexbytensorFusionOneIndexSuccess)
{
    std::vector<int64_t> dimsX{2048, 128};
    std::vector<std::vector<int64_t>> indicesDims{{8, 1024}};
    std::vector<int64_t> indicesMask{1};
    std::vector<int64_t> dimsY{8, 1024, 128};

    GraphPtr graph = BuildIndexByTensorGraph("indexbytensor_fusion_test2", dimsX, DT_FLOAT, FORMAT_ND,
                                             indicesDims, indicesMask, dimsY);

    CustomPassContext passContext;
    ops::IndexByTensorStaticFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool findIndex = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "Index") {
            findIndex = true;
        }
    }
    EXPECT_EQ(findIndex, true);
}

TEST_F(IndexByTensorStaticFusionPassTest, indexbytensorFusionDynamicShapeFail)
{
    std::vector<int64_t> dimsX{2, -1, 4, 5};
    std::vector<std::vector<int64_t>> indicesDims{{3, 2}};
    std::vector<int64_t> indicesMask{0, 1};
    std::vector<int64_t> dimsY{2, 3, 2, 5};

    GraphPtr graph = BuildIndexByTensorGraph("indexbytensor_dynamic_test", dimsX, DT_FLOAT, FORMAT_ND,
                                             indicesDims, indicesMask, dimsY);

    CustomPassContext passContext;
    ops::IndexByTensorStaticFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(IndexByTensorStaticFusionPassTest, indexbytensorFusionFloat16Success)
{
    std::vector<int64_t> dimsX{2, 3, 4, 5};
    std::vector<std::vector<int64_t>> indicesDims{{3, 2}, {3, 2}};
    std::vector<int64_t> indicesMask{0, 1, 1};
    std::vector<int64_t> dimsY{2, 3, 2, 5};

    GraphPtr graph = BuildIndexByTensorGraph("indexbytensor_fp16_test", dimsX, DT_FLOAT16, FORMAT_ND,
                                             indicesDims, indicesMask, dimsY);

    CustomPassContext passContext;
    ops::IndexByTensorStaticFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool findIndex = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "Index") {
            findIndex = true;
        }
    }
    EXPECT_EQ(findIndex, true);
}
