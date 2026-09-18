/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/layer_norm_fusion_pass.h"

using namespace ge;
using namespace ops;

namespace {
constexpr float kEpsilon = 0.00001f;

struct GraphOption {
    bool form_b = false;         // false = 形态 A（mul0=Mul(sub0,rsqrt0)）
    bool swap_squared = false;   // SquaredDifference 操作数交换
    bool swap_add0 = false;      // true 时 Add(mean1, eps)，即 eps 落在输入 1
    bool swap_mul1 = false;      // mul1 操作数交换
    bool keep_dims = true;       // mean0 的 keep_dims
    bool keep_dims_mean1 = true; // mean1 的 keep_dims（单独控制，用于锁住 A2 修复）
    std::vector<int64_t> axes1 = {-1};
    bool extra_consumer_on_sub = false; // 破坏自包含
    bool shared_axes = false; // 两个 ReduceMean 共用同一个 axes 常量（GE 常量去重后的常态）
};

class LayerNormFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950", kArch950); }

    static constexpr int32_t kArch950 = 3510;

    static void SetPlatform(const std::string& soc, int32_t npu_arch)
    {
        fe::PlatformInfo platform_info;
        fe::OptionalInfo optional_info;
        platform_info.soc_info.ai_core_cnt = 64;
        platform_info.str_info.short_soc_version = soc;
        optional_info.soc_version = soc;
        fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platform_info;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optional_info);

        fe::PlatFormInfos platform_infos;
        (void)platform_infos.Init();
        std::map<std::string, std::string> version_res;
        version_res["NpuArch"] = std::to_string(npu_arch);
        version_res["Short_SoC_version"] = soc;
        platform_infos.SetPlatformRes("version", version_res);
        fe::OptionalInfos optional_infos;
        (void)optional_infos.Init();
        optional_infos.SetSocVersion(soc);
        fe::PlatformInfoManager::Instance().platform_infos_map_[soc] = platform_infos;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optional_infos);
    }

    static void SetDesc(const es::EsTensorHolder& tensor, int32_t index, bool is_input, DataType dtype,
                        const std::vector<int64_t>& shape)
    {
        TensorDesc desc;
        if (is_input) {
            tensor.GetProducer()->GetInputDesc(index, desc);
        } else {
            tensor.GetProducer()->GetOutputDesc(index, desc);
        }
        desc.SetDataType(dtype);
        desc.SetFormat(FORMAT_ND);
        desc.SetShape(Shape(shape));
        desc.SetOriginShape(Shape(shape));
        if (is_input) {
            tensor.GetProducer()->UpdateInputDesc(index, desc);
        } else {
            tensor.GetProducer()->UpdateOutputDesc(index, desc);
        }
    }

    static std::shared_ptr<Graph> BuildGraph(const std::vector<int64_t>& x_shape, const GraphOption& opt)
    {
        const int64_t last_dim = x_shape.back();
        std::vector<int64_t> mean_shape = x_shape;
        mean_shape.back() = 1;
        const std::vector<int64_t> axes_shape = {1};
        const std::vector<int64_t> param_shape = {last_dim};

        auto graph_builder = es::EsGraphBuilder("layer_norm_fusion_test");
        auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, x_shape);
        auto gamma = graph_builder.CreateInput(1, "gamma", DT_FLOAT, FORMAT_ND, param_shape);
        auto axes0_const = graph_builder.CreateConst(std::vector<int64_t>{-1}, axes_shape);
        auto axes1_const = opt.shared_axes ? axes0_const : graph_builder.CreateConst(opt.axes1, axes_shape);
        auto eps_const = graph_builder.CreateConst(std::vector<float>{kEpsilon}, axes_shape);
        auto beta_const = graph_builder.CreateConst(std::vector<float>(last_dim, 0.0f), param_shape);

        auto mean0 = es::ReduceMean(x, axes0_const, opt.keep_dims);
        auto squared0 = opt.swap_squared ? es::SquaredDifference(mean0, x) : es::SquaredDifference(x, mean0);
        auto sub0 = es::Sub(x, mean0);
        auto mean1 = es::ReduceMean(squared0, axes1_const, opt.keep_dims_mean1);
        auto add0 = opt.swap_add0 ? es::Add(mean1, eps_const) : es::Add(eps_const, mean1);
        auto rsqrt0 = es::Rsqrt(add0);

        es::EsTensorHolder mul0;
        es::EsTensorHolder mul1;
        if (opt.form_b) {
            mul0 = es::Mul(rsqrt0, gamma);
            mul1 = opt.swap_mul1 ? es::Mul(sub0, mul0) : es::Mul(mul0, sub0);
        } else {
            mul0 = es::Mul(sub0, rsqrt0);
            mul1 = opt.swap_mul1 ? es::Mul(gamma, mul0) : es::Mul(mul0, gamma);
        }
        auto add1 = es::Add(beta_const, mul1);

        SetDesc(mean0, 0, true, DT_FLOAT, x_shape);
        SetDesc(mean0, 1, true, DT_INT64, axes_shape);
        SetDesc(mean0, 0, false, DT_FLOAT, mean_shape);
        SetDesc(squared0, 0, true, DT_FLOAT, opt.swap_squared ? mean_shape : x_shape);
        SetDesc(squared0, 1, true, DT_FLOAT, opt.swap_squared ? x_shape : mean_shape);
        SetDesc(squared0, 0, false, DT_FLOAT, x_shape);
        SetDesc(sub0, 0, true, DT_FLOAT, x_shape);
        SetDesc(sub0, 1, true, DT_FLOAT, mean_shape);
        SetDesc(sub0, 0, false, DT_FLOAT, x_shape);
        SetDesc(mean1, 0, true, DT_FLOAT, x_shape);
        SetDesc(mean1, 1, true, DT_INT64, axes_shape);
        SetDesc(mean1, 0, false, DT_FLOAT, mean_shape);
        SetDesc(add0, 0, true, DT_FLOAT, opt.swap_add0 ? mean_shape : axes_shape);
        SetDesc(add0, 1, true, DT_FLOAT, opt.swap_add0 ? axes_shape : mean_shape);
        SetDesc(add0, 0, false, DT_FLOAT, mean_shape);
        SetDesc(rsqrt0, 0, true, DT_FLOAT, mean_shape);
        SetDesc(rsqrt0, 0, false, DT_FLOAT, mean_shape);
        SetDesc(mul0, 0, true, DT_FLOAT, x_shape);
        SetDesc(mul0, 1, true, DT_FLOAT, mean_shape);
        SetDesc(mul0, 0, false, DT_FLOAT, x_shape);
        SetDesc(mul1, 0, true, DT_FLOAT, x_shape);
        SetDesc(mul1, 1, true, DT_FLOAT, param_shape);
        SetDesc(mul1, 0, false, DT_FLOAT, x_shape);
        SetDesc(add1, 0, true, DT_FLOAT, param_shape);
        SetDesc(add1, 1, true, DT_FLOAT, x_shape);
        SetDesc(add1, 0, false, DT_FLOAT, x_shape);

        std::vector<es::EsTensorHolder> outputs = {add1};
        if (opt.extra_consumer_on_sub) {
            auto extra = es::Rsqrt(sub0);
            SetDesc(extra, 0, true, DT_FLOAT, x_shape);
            SetDesc(extra, 0, false, DT_FLOAT, x_shape);
            outputs.emplace_back(extra);
        }
        return graph_builder.BuildAndReset(outputs);
    }

    static Status RunPass(std::shared_ptr<Graph>& graph)
    {
        CustomPassContext pass_context;
        LayerNormFusionPass pass;
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
};
} // namespace

// 形态 A：mul0 = Mul(sub0, rsqrt0)、mul1 = Mul(mul0, gamma)，融合并校验 LayerNorm 三个属性
TEST_F(LayerNormFusionPassTest, fuse_success_form_a)
{
    GraphOption opt;
    auto graph = BuildGraph({2, 4, 8}, opt);

    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 1);
    EXPECT_EQ(CountOpType(graph, "ReduceMean"), 0);
    EXPECT_EQ(CountOpType(graph, "SquaredDifference"), 0);
    EXPECT_EQ(CountOpType(graph, "Rsqrt"), 0);
    EXPECT_EQ(CountOpType(graph, "Sub"), 0);
    EXPECT_EQ(CountOpType(graph, "Mul"), 0);

    GNode layer_norm_node;
    ASSERT_TRUE(FindLayerNormNode(graph, layer_norm_node));
    int64_t begin_norm_axis = 0;
    int64_t begin_params_axis = 0;
    float32_t epsilon = 0.0f;
    EXPECT_EQ(layer_norm_node.GetAttr("begin_norm_axis", begin_norm_axis), GRAPH_SUCCESS);
    EXPECT_EQ(layer_norm_node.GetAttr("begin_params_axis", begin_params_axis), GRAPH_SUCCESS);
    EXPECT_EQ(layer_norm_node.GetAttr("epsilon", epsilon), GRAPH_SUCCESS);
    EXPECT_EQ(begin_norm_axis, -1);
    EXPECT_EQ(begin_params_axis, -1);
    EXPECT_NEAR(epsilon, kEpsilon, 1e-9f);
}

// 形态 B：mul0 = Mul(rsqrt0, gamma)，mul1 = Mul(mul0, sub0)
TEST_F(LayerNormFusionPassTest, fuse_success_form_b)
{
    GraphOption opt;
    opt.form_b = true;
    auto graph = BuildGraph({2, 4, 8}, opt);

    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 1);
    EXPECT_EQ(CountOpType(graph, "ReduceMean"), 0);
}

// 交换律：SquaredDifference 与 mul1 的操作数换序后仍应命中
TEST_F(LayerNormFusionPassTest, fuse_success_commutative_operands)
{
    GraphOption opt;
    opt.swap_squared = true;
    opt.swap_mul1 = true;
    auto graph = BuildGraph({2, 4, 8}, opt);

    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 1);
}

// keep_dims = false
TEST_F(LayerNormFusionPassTest, not_changed_when_keep_dims_false)
{
    GraphOption opt;
    opt.keep_dims = false;
    auto graph = BuildGraph({2, 4, 8}, opt);

    RunPass(graph);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 0);
    EXPECT_EQ(CountOpType(graph, "ReduceMean"), 2);
}

// 两个 ReduceMean 的 axes 不一致 → 拒绝融合
TEST_F(LayerNormFusionPassTest, not_changed_when_axes_mismatch)
{
    GraphOption opt;
    opt.axes1 = {0};
    auto graph = BuildGraph({2, 4, 8}, opt);

    RunPass(graph);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 0);
    EXPECT_EQ(CountOpType(graph, "ReduceMean"), 2);
}

// eps 落在 add0 输入 1
TEST_F(LayerNormFusionPassTest, not_changed_when_eps_on_second_input)
{
    GraphOption opt;
    opt.swap_add0 = true;
    auto graph = BuildGraph({2, 4, 8}, opt);

    RunPass(graph);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 0);
}

// 中间节点被片段外消费 → 破坏 pattern 自包含 → 静默不命中
TEST_F(LayerNormFusionPassTest, not_changed_when_intermediate_node_has_outer_consumer)
{
    GraphOption opt;
    opt.extra_consumer_on_sub = true;
    auto graph = BuildGraph({2, 4, 8}, opt);

    RunPass(graph);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 0);
    EXPECT_EQ(CountOpType(graph, "Sub"), 1);
}

// A2 修复锁定：mean0.keep_dims=true
TEST_F(LayerNormFusionPassTest, not_changed_when_mean1_keep_dims_false)
{
    GraphOption opt;
    opt.keep_dims_mean1 = false;
    auto graph = BuildGraph({2, 4, 8}, opt);

    RunPass(graph);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 0);
    EXPECT_EQ(CountOpType(graph, "ReduceMean"), 2);
}

// 两个 ReduceMean 共用同一个 axes 常量（GE CSE 后的常态）→ 仍应融合
TEST_F(LayerNormFusionPassTest, fuse_success_shared_axes_const)
{
    GraphOption opt;
    opt.shared_axes = true;
    auto graph = BuildGraph({2, 4, 8}, opt);

    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 1);
    EXPECT_EQ(CountOpType(graph, "ReduceMean"), 0);
}
