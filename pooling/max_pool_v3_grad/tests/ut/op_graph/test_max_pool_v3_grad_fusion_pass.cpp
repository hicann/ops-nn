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
#include "../../../op_graph/fusion_pass/max_pool_v3_grad_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {
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

es::EsTensorHolder CreatePoolGradNode(es::EsGraphBuilder& graphBuilder, const char* opType,
                                      const es::EsTensorHolder& x1, const es::EsTensorHolder& x2,
                                      const es::EsTensorHolder& grad, const std::vector<int64_t>& ksize,
                                      const std::vector<int64_t>& strides, const std::string& padding,
                                      const std::string& dataFormat)
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
                    .OpType(opType)
                    .Name(opType)
                    .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"grad", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({{"ksize", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                                 {"strides", es::CompliantNodeBuilder::kEsAttrOptional, "VT_LIST_INT", AttrValue()},
                                 {"padding", es::CompliantNodeBuilder::kEsAttrOptional, "VT_STRING", AttrValue()},
                                 {"data_format", es::CompliantNodeBuilder::kEsAttrOptional, "VT_STRING", AttrValue()}})
                    .InstanceOutputDataType("y", DT_FLOAT)
                    .InstanceOutputShape("y", {1, 8, 8, 3})
                    .InstanceOutputFormat("y", FORMAT_NHWC)
                    .Build();

    if (!CheckGraphSuccess(es::AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), node, 0),
                           "AddEdgeAndUpdatePeerDesc x1")) {
        return {};
    }
    if (!CheckGraphSuccess(es::AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), node, 1),
                           "AddEdgeAndUpdatePeerDesc x2")) {
        return {};
    }
    if (!CheckGraphSuccess(
            es::AddEdgeAndUpdatePeerDesc(*graph, *grad.GetProducer(), grad.GetProducerOutIndex(), node, 2),
            "AddEdgeAndUpdatePeerDesc grad")) {
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
        if (!CheckGraphSuccess(node.SetAttr("data_format", dataFormatAttr),
                               "node.SetAttr(\"data_format\", dataFormatAttr)")) {
            return {};
        }
    }

    auto* yHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0);
    return es::EsTensorHolder(yHolder);
}

GNode FindNodeByType(const std::shared_ptr<Graph>& graph, const char* type)
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

void CheckFusedNodeExists(const std::shared_ptr<Graph>& graph, bool expectFound)
{
    bool found = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (nodeType == "MaxPoolV3Grad") {
            found = true;
            break;
        }
    }
    EXPECT_EQ(found, expectFound);
}
} // namespace

class MaxPoolV3GradFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }
};

// test1: MaxPoolGrad NHWC SAME -> 融合成功
TEST_F(MaxPoolV3GradFusionPassTest, max_pool_v3_grad_fusion_pass_test_nhwc_same)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_v3_grad_fusion_pass_test_nhwc_same");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto grad = graphBuilder.CreateInput(2, "grad", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGrad", x1, x2, grad, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolV3GradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    CheckFusedNodeExists(graph, true);
}

// test2: MaxPoolGrad NCHW VALID -> 融合成功
TEST_F(MaxPoolV3GradFusionPassTest, max_pool_v3_grad_fusion_pass_test_nchw_valid)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_v3_grad_fusion_pass_test_nchw_valid");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_NCHW, {1, 3, 4, 4});
    auto grad = graphBuilder.CreateInput(2, "grad", DT_FLOAT, FORMAT_NCHW, {1, 3, 4, 4});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGrad", x1, x2, grad, {1, 1, 2, 2}, {1, 1, 2, 2}, "VALID", "NCHW");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolV3GradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
    CheckFusedNodeExists(graph, true);
}

// test3: MaxPoolGrad missing ksize -> 融合失败
TEST_F(MaxPoolV3GradFusionPassTest, max_pool_v3_grad_fusion_pass_missing_ksize)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_v3_grad_fusion_pass_missing_ksize");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto grad = graphBuilder.CreateInput(2, "grad", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGrad", x1, x2, grad, {}, {1, 2, 2, 1}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolV3GradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

// test4: MaxPoolGrad missing strides -> 融合失败
TEST_F(MaxPoolV3GradFusionPassTest, max_pool_v3_grad_fusion_pass_missing_strides)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_v3_grad_fusion_pass_missing_strides");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto grad = graphBuilder.CreateInput(2, "grad", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGrad", x1, x2, grad, {1, 2, 2, 1}, {}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolV3GradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

// test5: MaxPoolGrad missing padding -> 融合失败
TEST_F(MaxPoolV3GradFusionPassTest, max_pool_v3_grad_fusion_pass_missing_padding)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_v3_grad_fusion_pass_missing_padding");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto grad = graphBuilder.CreateInput(2, "grad", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGrad", x1, x2, grad, {1, 2, 2, 1}, {1, 2, 2, 1}, "", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolV3GradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

// test6: unsupported soc Ascend310 -> 融合失败
TEST_F(MaxPoolV3GradFusionPassTest, max_pool_v3_grad_fusion_pass_unsupported_soc)
{
    SetPlatform("Ascend310");
    auto graphBuilder = es::EsGraphBuilder("max_pool_v3_grad_fusion_pass_unsupported_soc");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto grad = graphBuilder.CreateInput(2, "grad", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGrad", x1, x2, grad, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolV3GradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

// test7: unsupported dtype DT_BOOL -> 融合失败
TEST_F(MaxPoolV3GradFusionPassTest, max_pool_v3_grad_fusion_pass_unsupported_dtype)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_v3_grad_fusion_pass_unsupported_dtype");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_BOOL, FORMAT_NHWC, {1, 8, 8, 3});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_BOOL, FORMAT_NHWC, {1, 4, 4, 3});
    auto grad = graphBuilder.CreateInput(2, "grad", DT_BOOL, FORMAT_NHWC, {1, 4, 4, 3});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGrad", x1, x2, grad, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolV3GradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}

// test8: invalid ksize size=3 -> 融合失败
TEST_F(MaxPoolV3GradFusionPassTest, max_pool_v3_grad_fusion_pass_invalid_ksize)
{
    auto graphBuilder = es::EsGraphBuilder("max_pool_v3_grad_fusion_pass_invalid_ksize");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto grad = graphBuilder.CreateInput(2, "grad", DT_FLOAT, FORMAT_NHWC, {1, 4, 4, 3});
    auto y = CreatePoolGradNode(graphBuilder, "MaxPoolGrad", x1, x2, grad, {1, 2, 2}, {1, 2, 2, 1}, "SAME", "NHWC");
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({y});

    CustomPassContext passContext;
    MaxPoolV3GradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    CheckFusedNodeExists(graph, false);
}
