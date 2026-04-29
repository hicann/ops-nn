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
#include "../../../op_graph/fusion_pass/max_pool_with_argmax_v3_fusion_pass.h"

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

std::pair<es::EsTensorHolder, es::EsTensorHolder> CreatePoolNode(es::EsGraphBuilder &graphBuilder, const char *opType,
    const es::EsTensorHolder &x, const std::vector<int64_t> &ksize, const std::vector<int64_t> &strides,
    const std::vector<int64_t> &pads, const std::vector<int64_t> &dilation, const bool setInvalidDilation = false,
    const char *v0Padding = "VALID")
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
                    .IrDefOutputs({
                        {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                        {"argmax", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({
                        {"ksize", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                        {"strides", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                        {"padding", es::CompliantNodeBuilder::kEsAttrOptional, "VT_STRING", AttrValue()},
                        {"pads", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                        {"dilation", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                        {"ceil_mode", es::CompliantNodeBuilder::kEsAttrOptional, "VT_BOOL", AttrValue()},
                        {"dtype", es::CompliantNodeBuilder::kEsAttrOptional, "VT_INT", AttrValue()}})
                    .InstanceOutputDataType("y", DT_FLOAT)
                    .InstanceOutputShape("y", {64, 32, 16, 16})
                    .InstanceOutputFormat("y", FORMAT_NCHW)
                    .InstanceOutputDataType("argmax", DT_UINT16)
                    .InstanceOutputShape("argmax", {64, 32, 16, 16})
                    .InstanceOutputFormat("argmax", FORMAT_NCHW)
                    .Build();

    if (!CheckGraphSuccess(es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0),
            "AddEdgeAndUpdatePeerDesc")) {
        return {};
    }

    std::vector<int64_t> ksizeAttr = ksize;
    std::vector<int64_t> stridesAttr = strides;
    std::vector<int64_t> padsAttr = pads;
    std::vector<int64_t> dilationAttr = setInvalidDilation ? std::vector<int64_t>{1, 1, 1} : dilation;
    bool ceilMode = false;
    int64_t dtype = 3;
    if (std::string(opType) == "MaxPoolWithArgmax") {
        AscendString padding = v0Padding;
        if (!CheckGraphSuccess(node.SetAttr("padding", padding), "node.SetAttr(\"padding\", padding)")) {
            return {};
        }
    } else {
        if (!CheckGraphSuccess(node.SetAttr("pads", padsAttr), "node.SetAttr(\"pads\", padsAttr)")) {
            return {};
        }
        if (!CheckGraphSuccess(node.SetAttr("dtype", dtype), "node.SetAttr(\"dtype\", dtype)")) {
            return {};
        }
        if (!CheckGraphSuccess(node.SetAttr("ceil_mode", ceilMode), "node.SetAttr(\"ceil_mode\", ceilMode)")) {
            return {};
        }
        if (!CheckGraphSuccess(node.SetAttr("dilation", dilationAttr), "node.SetAttr(\"dilation\", dilationAttr)")) {
            return {};
        }
    }
    if (!CheckGraphSuccess(node.SetAttr("ksize", ksizeAttr), "node.SetAttr(\"ksize\", ksizeAttr)")) {
        return {};
    }
    if (!CheckGraphSuccess(node.SetAttr("strides", stridesAttr), "node.SetAttr(\"strides\", stridesAttr)")) {
        return {};
    }

    auto *yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0);
    auto *argmaxHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1);
    return {es::EsTensorHolder(yHolder), es::EsTensorHolder(argmaxHolder)};
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
        if (nodeType == "MaxPoolWithArgmaxV3") {
            found = true;
            break;
        }
    }
    EXPECT_EQ(found, expectFound);
}
} // namespace

class MaxPoolWithArgmaxV3FusionPassTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetPlatform("Ascend950");
    }
};

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v1_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v1_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {64, 32, 32, 32});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV1", x, {1, 2, 2, 1}, {1, 2, 2, 1},
        {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v2_nhwc_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v2_nhwc_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NHWC, {64, 32, 32, 32});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV2", x, {1, 2, 2, 1}, {1, 2, 2, 1},
        {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v2_nchw_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v2_nchw_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {64, 32, 32, 32});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV2", x, {1, 1, 2, 3}, {1, 1, 2, 3},
        {0, 2, 1, 0}, {1, 1, 2, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v2_invalid_dilation_not_fused)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v2_invalid_dilation_not_fused");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {64, 32, 32, 32});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV2", x, {1, 2, 2, 1}, {1, 2, 2, 1},
        {1, 1, 1, 1}, {1, 1, 1, 1}, true);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    ExpectNodeTypeNotFound(graph, "MaxPoolWithArgmaxV3");
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_unsupported_soc_not_fused)
{
    SetPlatform("Ascend910_95");
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_unsupported_soc_not_fused");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {64, 32, 32, 32});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV1", x, {1, 2, 2, 1}, {1, 2, 2, 1},
        {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    ExpectNodeTypeNotFound(graph, "MaxPoolWithArgmaxV3");
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_unsupported_dtype_not_fused)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_unsupported_dtype_not_fused");
    auto x = graphBuilder.CreateInput(0, "x", DT_INT32, FORMAT_NCHW, {64, 32, 32, 32});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV1", x, {1, 2, 2, 1}, {1, 2, 2, 1},
        {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    ExpectNodeTypeNotFound(graph, "MaxPoolWithArgmaxV3");
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v1_bf16_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v1_bf16_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_BF16, FORMAT_NCHW, {32, 16, 28, 28});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV1", x, {1, 3, 4, 1}, {1, 2, 5, 1},
        {0, 2, 1, 0}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v2_float_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v2_float_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, {8, 56, 56, 32});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV2", x, {1, 5, 3, 1}, {1, 4, 2, 1},
        {0, 3, 2, 0}, {1, 1, 2, 3});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v2_bf16_nhwc_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v2_bf16_nhwc_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_BF16, FORMAT_NHWC, {4, 20, 20, 64});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV2", x, {1, 4, 2, 1}, {1, 3, 2, 1},
        {0, 0, 1, 0}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v1_zero_pad_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v1_zero_pad_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {16, 64, 14, 14});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV1", x, {1, 2, 3, 1}, {1, 2, 3, 1},
        {0, 0, 0, 0}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v2_nchw_identity_dilation_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v2_nchw_identity_dilation_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 8, 18, 18});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV2", x, {1, 1, 3, 3}, {1, 1, 2, 2},
        {0, 1, 1, 0}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v2_invalid_format_not_fused)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v2_invalid_format_not_fused");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {64, 32, 32, 32});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV2", x, {1, 2, 2, 1}, {1, 2, 2, 1},
        {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    ExpectNodeTypeNotFound(graph, "MaxPoolWithArgmaxV3");
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v1_large_window_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v1_large_window_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {8, 8, 64, 64});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV1", x, {1, 7, 5, 1}, {1, 3, 2, 1},
        {0, 4, 2, 0}, {1, 1, 3, 2});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolWithArgmaxV3FusionPassTest, maxpoolwithargmax_v2_unsupported_soc_950_fused)
{
    SetPlatform("Ascend950");
    auto graphBuilder = es::EsGraphBuilder("maxpoolwithargmax_v2_unsupported_soc_950_fused");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NHWC, {2, 24, 24, 16});
    auto outputs = CreatePoolNode(graphBuilder, "MaxPoolWithArgmaxV2", x, {1, 3, 3, 1}, {1, 2, 2, 1},
        {0, 1, 1, 0}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outputs.first, outputs.second});

    CustomPassContext passContext;
    MaxPoolWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}
