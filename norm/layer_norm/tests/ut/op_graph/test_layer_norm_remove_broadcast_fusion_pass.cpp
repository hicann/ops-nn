/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/layer_norm_remove_broadcast_fusion_pass.h"

using namespace fe;
using namespace ge;
using namespace ops;

namespace {
constexpr float kEpsilon = 0.00001f;
constexpr int64_t kXInputIdx = 0;
constexpr int64_t kGammaInputIdx = 1;
constexpr int64_t kBetaInputIdx = 2;
constexpr int64_t kShapeInputIdx = 3;

class LayerNormRemoveBroadcastFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }

    static void SetPlatform(const std::string& soc)
    {
        PlatformInfo platform_info;
        OptionalInfo optional_info;
        platform_info.soc_info.ai_core_cnt = 64;
        platform_info.str_info.short_soc_version = soc;
        optional_info.soc_version = soc;
        PlatformInfoManager::Instance().platform_info_map_[soc] = platform_info;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optional_info);
    }

    static std::shared_ptr<Graph> BuildGraph(const std::vector<int64_t>& x_shape,
                                             const std::vector<int64_t>& gamma_shape,
                                             const std::vector<int64_t>& beta_shape, int64_t begin_norm_axis = -1,
                                             DataType x_dtype = DT_FLOAT16, DataType params_dtype = DT_FLOAT)
    {
        auto graph_builder = es::EsGraphBuilder("layer_norm_remove_broadcast_fusion_test");
        auto x = graph_builder.CreateInput(kXInputIdx, "x", x_dtype, FORMAT_ND, x_shape);
        auto gamma = graph_builder.CreateInput(kGammaInputIdx, "gamma", params_dtype, FORMAT_ND, gamma_shape);
        auto beta = graph_builder.CreateInput(kBetaInputIdx, "beta", params_dtype, FORMAT_ND, beta_shape);
        auto shape = graph_builder.CreateInput(kShapeInputIdx, "shape", DT_INT64, FORMAT_ND,
                                               {static_cast<int64_t>(x_shape.size())});
        auto gamma_brc = es::BroadcastTo(gamma, shape);
        auto beta_brc = es::BroadcastTo(beta, shape);
        auto layer_norm = es::LayerNorm(x, gamma_brc, beta_brc, begin_norm_axis, 0, kEpsilon);
        UpdateInputDesc(gamma_brc, 0, params_dtype, gamma_shape);
        UpdateInputDesc(gamma_brc, 1, DT_INT64, {static_cast<int64_t>(x_shape.size())});
        UpdateOutputDesc(gamma_brc, 0, params_dtype, x_shape);
        UpdateInputDesc(beta_brc, 0, params_dtype, beta_shape);
        UpdateInputDesc(beta_brc, 1, DT_INT64, {static_cast<int64_t>(x_shape.size())});
        UpdateOutputDesc(beta_brc, 0, params_dtype, x_shape);
        UpdateInputDesc(layer_norm.y, 0, x_dtype, x_shape);
        UpdateInputDesc(layer_norm.y, 1, params_dtype, x_shape);
        UpdateInputDesc(layer_norm.y, 2, params_dtype, x_shape);
        UpdateOutputDesc(layer_norm.y, 0, x_dtype, x_shape);
        UpdateOutputDesc(layer_norm.mean, 1, params_dtype, x_shape);
        UpdateOutputDesc(layer_norm.variance, 2, params_dtype, x_shape);
        return graph_builder.BuildAndReset({layer_norm.y, layer_norm.mean, layer_norm.variance});
    }

    static void UpdateInputDesc(const es::EsTensorHolder& tensor, int32_t index, DataType dtype,
                                const std::vector<int64_t>& shape)
    {
        TensorDesc desc;
        tensor.GetProducer()->GetInputDesc(index, desc);
        desc.SetDataType(dtype);
        desc.SetFormat(FORMAT_ND);
        desc.SetShape(Shape(shape));
        tensor.GetProducer()->UpdateInputDesc(index, desc);
    }

    static void UpdateOutputDesc(const es::EsTensorHolder& tensor, int32_t index, DataType dtype,
                                 const std::vector<int64_t>& shape)
    {
        TensorDesc desc;
        tensor.GetProducer()->GetOutputDesc(index, desc);
        desc.SetDataType(dtype);
        desc.SetFormat(FORMAT_ND);
        desc.SetShape(Shape(shape));
        tensor.GetProducer()->UpdateOutputDesc(index, desc);
    }

    static Status RunPass(std::shared_ptr<Graph>& graph)
    {
        CustomPassContext pass_context;
        LayerNormRemoveBroadcastFusionPass pass;
        return pass.Run(graph, pass_context);
    }

    static int CountOpType(const std::shared_ptr<Graph>& graph, const std::string& op_type)
    {
        int count = 0;
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == op_type.c_str()) {
                ++count;
            }
        }
        return count;
    }

    static bool FindLayerNormNode(const std::shared_ptr<Graph>& graph, GNode& layer_norm_node)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == "LayerNorm") {
                layer_norm_node = node;
                return true;
            }
        }
        return false;
    }

    static bool IsLayerNormInputFrom(const GNode& layer_norm_node, int32_t input_idx, const std::string& op_type)
    {
        const auto in_node = layer_norm_node.GetInDataNodesAndPortIndexs(input_idx);
        if (in_node.first == nullptr) {
            return false;
        }
        AscendString type;
        in_node.first->GetType(type);
        return type == op_type.c_str();
    }
};
} // namespace

TEST_F(LayerNormRemoveBroadcastFusionPassTest, remove_broadcast_success)
{
    auto graph = BuildGraph({2, 4, 8}, {8}, {8});

    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, "BroadcastTo"), 0);

    GNode layer_norm_node;
    ASSERT_TRUE(FindLayerNormNode(graph, layer_norm_node));
    EXPECT_TRUE(IsLayerNormInputFrom(layer_norm_node, 1, "Data"));
    EXPECT_TRUE(IsLayerNormInputFrom(layer_norm_node, 2, "Data"));

    int64_t begin_norm_axis = 0;
    int64_t begin_params_axis = 0;
    EXPECT_EQ(layer_norm_node.GetAttr("begin_norm_axis", begin_norm_axis), GRAPH_SUCCESS);
    EXPECT_EQ(layer_norm_node.GetAttr("begin_params_axis", begin_params_axis), GRAPH_SUCCESS);
    EXPECT_EQ(begin_norm_axis, 2);
    EXPECT_EQ(begin_params_axis, 2);
}

TEST_F(LayerNormRemoveBroadcastFusionPassTest, positive_begin_norm_axis_remove_broadcast_success)
{
    auto graph = BuildGraph({2, 4, 8}, {8}, {8}, 2);

    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, "BroadcastTo"), 0);

    GNode layer_norm_node;
    ASSERT_TRUE(FindLayerNormNode(graph, layer_norm_node));
    EXPECT_TRUE(IsLayerNormInputFrom(layer_norm_node, 1, "Data"));
    EXPECT_TRUE(IsLayerNormInputFrom(layer_norm_node, 2, "Data"));
}

TEST_F(LayerNormRemoveBroadcastFusionPassTest, dynamic_shape_not_remove_broadcast)
{
    auto graph = BuildGraph({-1, 4, 8}, {8}, {8});

    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, "BroadcastTo"), 2);
}

TEST_F(LayerNormRemoveBroadcastFusionPassTest, scalar_gamma_beta_not_remove_broadcast)
{
    auto graph = BuildGraph({2, 4, 8}, {}, {});

    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, "BroadcastTo"), 2);
}

TEST_F(LayerNormRemoveBroadcastFusionPassTest, different_gamma_beta_shape_not_remove_broadcast)
{
    auto graph = BuildGraph({2, 4, 8}, {8}, {4});

    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, "BroadcastTo"), 2);
}

TEST_F(LayerNormRemoveBroadcastFusionPassTest, unsupported_platform_not_remove_broadcast)
{
    SetPlatform("Ascend910B");
    auto graph = BuildGraph({2, 4, 8}, {8}, {8});

    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, "BroadcastTo"), 2);
}

TEST_F(LayerNormRemoveBroadcastFusionPassTest, gamma_rank_greater_than_x_rank_not_remove_broadcast)
{
    auto graph = BuildGraph({4, 8}, {1, 4, 8}, {1, 4, 8});

    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, "BroadcastTo"), 2);
}

TEST_F(LayerNormRemoveBroadcastFusionPassTest, unmatched_begin_norm_axis_not_remove_broadcast)
{
    auto graph = BuildGraph({2, 4, 8}, {8}, {8}, 1);

    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, "BroadcastTo"), 2);
}
