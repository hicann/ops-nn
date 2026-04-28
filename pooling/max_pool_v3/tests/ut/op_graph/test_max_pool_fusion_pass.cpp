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
#include "../../../op_graph/fusion_pass/max_pool_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {
void SetPlatform(const std::string &soc)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = soc;
    optionalInfo.soc_version = soc;
    fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

es::EsTensorHolder CreatePoolNode(es::EsGraphBuilder &graphBuilder, const char *opType,
    const es::EsTensorHolder &x, const std::vector<int64_t> &ksize,
    const std::vector<int64_t> &strides, const std::string &padding,
    const std::string &dataFormat)
{
    auto CheckGraphSuccess = [](graphStatus status, const char *expr)
    {
        if (status != GRAPH_SUCCESS) {
            ADD_FAILURE() << expr << " failed, status=" << status;
            return false;
        }
        return true;
    };

    auto *graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto node = es::CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(opType)
                    .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({
                        {"ksize", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                        {"strides", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                        {"padding", es::CompliantNodeBuilder::kEsAttrOptional, "VT_STRING", AttrValue()},
                        {"data_format", es::CompliantNodeBuilder::kEsAttrOptional, "VT_STRING", AttrValue()}})
                    .InstanceOutputDataType("y", DT_FLOAT)
                    .InstanceOutputShape("y", {1, 4, 4, 3})
                    .InstanceOutputFormat("y", FORMAT_NHWC)
                    .Build();

    if (!CheckGraphSuccess(
            es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0),
            "AddEdgeAndUpdatePeerDesc x")) {
        return {};
    }

    if (!ksize.empty()) {
        std::vector<int64_t> ksizeAttr = ksize;
        if (!CheckGraphSuccess(node.SetAttr("ksize", ksizeAttr), "node.SetAttr(\"ksize\", ksizeAttr)")) {
            return {};
        }
    }
    if (!strides.empty()) {
        std::vector<int64_t> stridesAttr = strides;
        if (!CheckGraphSuccess(node.SetAttr("strides", stridesAttr), "node.SetAttr(\"strides\", stridesAttr)")) {
            return {};
        }
    }
    if (!padding.empty()) {
        AscendString paddingAttr = padding.c_str();
        if (!CheckGraphSuccess(node.SetAttr("padding", paddingAttr), "node.SetAttr(\"padding\", paddingAttr)")) {
            return {};
        }
    }
    if (!dataFormat.empty()) {
        AscendString dataFormatAttr = dataFormat.c_str();
        if (!CheckGraphSuccess(node.SetAttr("data_format", dataFormatAttr), "node.SetAttr(\"data_format\", dataFormatAttr)")) {
            return {};
        }
    }

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0);
    return es::EsTensorHolder(yHolder);
}

GNode FindNodeByType(const std::shared_ptr<Graph> &graph, const char *type)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (nodeType == type) {
            return node;
        }
    }
    return GNode();
}

void ExpectNodeTypeNotFound(const std::shared_ptr<Graph> &graph, const char *type)
{
    GNode node = FindNodeByType(graph, type);
    AscendString nodeType;
    EXPECT_NE(node.GetType(nodeType), GRAPH_SUCCESS);
}

void CheckFusedNodeExists(const std::shared_ptr<Graph> &graph, bool expectFound)
{
    bool found = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (nodeType == "MaxPoolV3") {
            found = true;
            break;
        }
    }
    EXPECT_EQ(found, expectFound);
}
} // namespace

class MaxPoolFusionPassTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetPlatform("Ascend910_93");
    }
};

// ========== 旧版7个测试用例 ==========

// test1: MaxPool NHWC SAME -> 融合成功
TEST_F(MaxPoolFusionPassTest, max_pool_fusion_pass_test1)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_fusion_pass_test1");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto y = CreatePoolNode(graphBuilder, "MaxPool", x,
        {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    CheckFusedNodeExists(graph, true);
}

// test2: MaxPool missing ksize -> 融合失败
TEST_F(MaxPoolFusionPassTest, max_pool_fusion_pass_missing_ksize)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_fusion_pass_missing_ksize");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto y = CreatePoolNode(graphBuilder, "MaxPool", x,
        {}, {1, 2, 2, 1}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

// test3: MaxPool missing strides -> 融合失败
TEST_F(MaxPoolFusionPassTest, max_pool_fusion_pass_missing_strides)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_fusion_pass_missing_strides");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto y = CreatePoolNode(graphBuilder, "MaxPool", x,
        {1, 2, 2, 1}, {}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

// test4: MaxPool missing padding -> 融合失败
TEST_F(MaxPoolFusionPassTest, max_pool_fusion_pass_missing_padding)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_fusion_pass_missing_padding");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto y = CreatePoolNode(graphBuilder, "MaxPool", x,
        {1, 2, 2, 1}, {1, 2, 2, 1}, "", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

// test5: MaxPool NCHW SAME -> 融合成功
TEST_F(MaxPoolFusionPassTest, max_pool_fusion_pass_test_nchw)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_fusion_pass_test_nchw");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8});
    auto y = CreatePoolNode(graphBuilder, "MaxPool", x,
        {1, 1, 2, 2}, {1, 1, 2, 2}, "SAME", "NCHW");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    CheckFusedNodeExists(graph, true);
}

// test6: unsupported soc Ascend310 -> 融合失败
TEST_F(MaxPoolFusionPassTest, max_pool_fusion_pass_unsupported_soc)
{
    SetPlatform("Ascend310");
    auto graphBuilder = es::EsGraphBuilder("max_pool_fusion_pass_unsupported_soc");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto y = CreatePoolNode(graphBuilder, "MaxPool", x,
        {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

// test7: unsupported dtype DT_BOOL -> 融合失败
TEST_F(MaxPoolFusionPassTest, max_pool_fusion_pass_unsupported_dtype)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_fusion_pass_unsupported_dtype");
    auto x = graphBuilder.CreateInput(0, "x", DT_BOOL, FORMAT_NHWC, {1, 8, 8, 3});
    auto y = CreatePoolNode(graphBuilder, "MaxPool", x,
        {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

// test8: invalid ksize size=3 -> 融合失败
TEST_F(MaxPoolFusionPassTest, max_pool_fusion_pass_invalid_ksize)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_fusion_pass_invalid_ksize");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto y = CreatePoolNode(graphBuilder, "MaxPool", x,
        {1, 2, 2}, {1, 2, 2, 1}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}
