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
#include "../../../op_graph/fusion_pass/max_pool_grad_with_argmax_v3_fusion_pass.h"

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

es::EsTensorHolder CreatePoolGradNode(es::EsGraphBuilder &graphBuilder, const char *opType,
    const es::EsTensorHolder &x, const es::EsTensorHolder &grad, const es::EsTensorHolder &argmax,
    const std::vector<int64_t> &ksize, const std::vector<int64_t> &strides,
    const std::vector<int64_t> &pads, const std::vector<int64_t> &dilation,
    const bool setInvalidDilation = false, const char *v0Padding = "VALID")
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
                    .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                        {"grad", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                        {"argmax", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({
                        {"ksize", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                        {"strides", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                        {"padding", es::CompliantNodeBuilder::kEsAttrOptional, "VT_STRING", AttrValue()},
                        {"pads", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                        {"dilation", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                        {"ceil_mode", es::CompliantNodeBuilder::kEsAttrOptional, "VT_BOOL", AttrValue()},
                        {"dtype", es::CompliantNodeBuilder::kEsAttrOptional, "VT_INT", AttrValue()}})
                    .InstanceOutputDataType("y", DT_FLOAT)
                    .InstanceOutputShape("y", {64, 32, 32, 32})
                    .InstanceOutputFormat("y", FORMAT_NCHW)
                    .Build();

    if (!CheckGraphSuccess(
            es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0),
            "AddEdgeAndUpdatePeerDesc x")) {
        return {};
    }
    if (!CheckGraphSuccess(
            es::AddEdgeAndUpdatePeerDesc(*graph, *grad.GetProducer(), grad.GetProducerOutIndex(), node, 1),
            "AddEdgeAndUpdatePeerDesc grad")) {
        return {};
    }
    if (!CheckGraphSuccess(
            es::AddEdgeAndUpdatePeerDesc(*graph, *argmax.GetProducer(), argmax.GetProducerOutIndex(), node, 2),
            "AddEdgeAndUpdatePeerDesc argmax")) {
        return {};
    }

    std::vector<int64_t> ksizeAttr = ksize;
    std::vector<int64_t> stridesAttr = strides;
    std::vector<int64_t> padsAttr = pads;
    std::vector<int64_t> dilationAttr = setInvalidDilation ? std::vector<int64_t>{1, 1, 1} : dilation;
    bool ceilMode = false;
    int64_t dtype = 3;

    if (std::string(opType) == "MaxPoolGradWithArgmax") {
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
        if (nodeType == "MaxPoolGradWithArgmaxV3") {
            found = true;
            break;
        }
    }
    EXPECT_EQ(found, expectFound);
}
} // namespace

class MaxPoolGradWithArgmaxV3FusionPassTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetPlatform("Ascend950");
    }
};

// ========== 旧版4个测试用例 ==========

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmaxv3_fusion_pass_test_2)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmaxv3_fusion_pass_test_2");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT, FORMAT_NCHW, {1, 16, 16, 16});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NCHW, {1, 16, 16, 16});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV1", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmaxv3_fusion_pass_test_3)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmaxv3_fusion_pass_test_3");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT, FORMAT_NCHW, {1, 16, 16, 16});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NCHW, {1, 16, 16, 16});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV1", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, true);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmaxv3_fusion_pass_test_4)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmaxv3_fusion_pass_test_4");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT, FORMAT_NHWC, {1, 16, 16, 16});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NHWC, {1, 16, 16, 16});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV1", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    CheckFusedNodeExists(graph, true);
}

// ========== 扩展测试用例 ==========

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v1_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v1_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT, FORMAT_NCHW, {64, 32, 16, 16});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NCHW, {64, 32, 16, 16});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV1", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v1_invalid_dilation_not_fused)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v1_invalid_dilation_not_fused");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT, FORMAT_NCHW, {64, 32, 16, 16});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NCHW, {64, 32, 16, 16});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV1", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, true);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    ExpectNodeTypeNotFound(graph, "MaxPoolGradWithArgmaxV3");
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v1_nhwc_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v1_nhwc_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NHWC, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT, FORMAT_NHWC, {64, 16, 16, 32});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NHWC, {64, 16, 16, 32});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV1", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v2_nchw_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v2_nchw_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT, FORMAT_NCHW, {64, 32, 16, 16});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NCHW, {64, 32, 16, 16});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV2", x, grad, argmax,
        {1, 1, 2, 3}, {1, 1, 2, 3}, {0, 2, 1, 0}, {1, 1, 2, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v2_nhwc_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v2_nhwc_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NHWC, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT16, FORMAT_NHWC, {64, 16, 16, 32});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NHWC, {64, 16, 16, 32});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV2", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_unsupported_soc_not_fused)
{
    SetPlatform("Ascend910_95");
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_unsupported_soc_not_fused");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT, FORMAT_NCHW, {64, 32, 16, 16});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NCHW, {64, 32, 16, 16});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV1", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    ExpectNodeTypeNotFound(graph, "MaxPoolGradWithArgmaxV3");
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_unsupported_dtype_not_fused)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_unsupported_dtype_not_fused");
    auto x = graphBuilder.CreateInput(0, "x", DT_INT32, FORMAT_NCHW, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_INT32, FORMAT_NCHW, {64, 32, 16, 16});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NCHW, {64, 32, 16, 16});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV1", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    ExpectNodeTypeNotFound(graph, "MaxPoolGradWithArgmaxV3");
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v1_bf16_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v1_bf16_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_BF16, FORMAT_NCHW, {32, 16, 28, 28});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_BF16, FORMAT_NCHW, {32, 16, 14, 14});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NCHW, {32, 16, 14, 14});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV1", x, grad, argmax,
        {1, 3, 4, 1}, {1, 2, 5, 1}, {0, 2, 1, 0}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v2_bf16_nhwc_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v2_bf16_nhwc_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_BF16, FORMAT_NHWC, {4, 20, 20, 64});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_BF16, FORMAT_NHWC, {4, 10, 10, 64});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NHWC, {4, 10, 10, 64});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV2", x, grad, argmax,
        {1, 4, 2, 1}, {1, 3, 2, 1}, {0, 0, 1, 0}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v1_zero_pad_to_v3_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v1_zero_pad_to_v3_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {16, 64, 14, 14});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT16, FORMAT_NCHW, {16, 64, 7, 7});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NCHW, {16, 64, 7, 7});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV1", x, grad, argmax,
        {1, 2, 3, 1}, {1, 2, 3, 1}, {0, 0, 0, 0}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v2_nchw_identity_dilation_success)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v2_nchw_identity_dilation_success");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 8, 18, 18});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT16, FORMAT_NCHW, {2, 8, 9, 9});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_NCHW, {2, 8, 9, 9});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV2", x, grad, argmax,
        {1, 1, 3, 3}, {1, 1, 2, 2}, {0, 1, 1, 0}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    CheckFusedNodeExists(graph, true);
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v0_invalid_format_not_fused)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v0_invalid_format_not_fused");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT, FORMAT_ND, {64, 32, 16, 16});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_ND, {64, 32, 16, 16});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmax", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {0, 0, 0, 0}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    ExpectNodeTypeNotFound(graph, "MaxPoolGradWithArgmaxV3");
}

TEST_F(MaxPoolGradWithArgmaxV3FusionPassTest, maxpoolgradwithargmax_v2_invalid_format_not_fused)
{
    auto graphBuilder = es::EsGraphBuilder("maxpoolgradwithargmax_v2_invalid_format_not_fused");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, {64, 32, 32, 32});
    auto grad = graphBuilder.CreateInput(1, "grad", DT_FLOAT16, FORMAT_ND, {64, 32, 16, 16});
    auto argmax = graphBuilder.CreateInput(2, "argmax", DT_INT64, FORMAT_ND, {64, 32, 16, 16});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGradWithArgmaxV2", x, grad, argmax,
        {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolGradWithArgmaxV3FusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    ExpectNodeTypeNotFound(graph, "MaxPoolGradWithArgmaxV3");
}
