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
#include "../../../op_graph/fusion_pass/layernorm_inference_fusion_pass.h"

using namespace ge;
using namespace ops;

namespace {
constexpr float kEpsilon = 0.00001f;

struct GraphOption {
    bool swap_squared = false; // SquaredDifference 操作数交换
    bool swap_mul1 = false;    // Mul(rsqrt, gamma) 操作数交换
    bool swap_add2 = false;    // Add(mul3, sub) 操作数交换
    bool keep_dims = true;
    std::vector<int64_t> axes2 = {-1};
    bool training_flow = false; // 给 rsqrt 再挂一个消费者，模拟训练图（反向复用 rsqrt）
    bool shared_axes = false;   // 两个 ReduceMean 共用同一个 axes 常量
};

class LayerNormInferenceFusionPassTest : public testing::Test {
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

        auto graph_builder = es::EsGraphBuilder("layernorm_inference_fusion_test");
        auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, x_shape);
        auto gamma = graph_builder.CreateInput(1, "gamma", DT_FLOAT, FORMAT_ND, param_shape);
        auto beta = graph_builder.CreateInput(2, "beta", DT_FLOAT, FORMAT_ND, param_shape);
        auto axes1_const = graph_builder.CreateConst(std::vector<int64_t>{-1}, axes_shape);
        auto axes2_const = opt.shared_axes ? axes1_const : graph_builder.CreateConst(opt.axes2, axes_shape);
        auto eps_const = graph_builder.CreateConst(std::vector<float>{kEpsilon}, axes_shape);

        auto mean1 = es::ReduceMean(x, axes1_const, opt.keep_dims);
        auto squared = opt.swap_squared ? es::SquaredDifference(mean1, x) : es::SquaredDifference(x, mean1);
        auto mean2 = es::ReduceMean(squared, axes2_const, opt.keep_dims);
        auto add1 = es::Add(mean2, eps_const);
        auto rsqrt = es::Rsqrt(add1);
        auto mul1 = opt.swap_mul1 ? es::Mul(gamma, rsqrt) : es::Mul(rsqrt, gamma);
        auto mul2 = es::Mul(mean1, mul1);
        auto mul3 = es::Mul(mul1, x);
        auto sub = es::Sub(beta, mul2);
        auto add2 = opt.swap_add2 ? es::Add(sub, mul3) : es::Add(mul3, sub);

        SetDesc(mean1, 0, true, DT_FLOAT, x_shape);
        SetDesc(mean1, 1, true, DT_INT64, axes_shape);
        SetDesc(mean1, 0, false, DT_FLOAT, mean_shape);
        SetDesc(squared, 0, true, DT_FLOAT, opt.swap_squared ? mean_shape : x_shape);
        SetDesc(squared, 1, true, DT_FLOAT, opt.swap_squared ? x_shape : mean_shape);
        SetDesc(squared, 0, false, DT_FLOAT, x_shape);
        SetDesc(mean2, 0, true, DT_FLOAT, x_shape);
        SetDesc(mean2, 1, true, DT_INT64, axes_shape);
        SetDesc(mean2, 0, false, DT_FLOAT, mean_shape);
        SetDesc(add1, 0, true, DT_FLOAT, mean_shape);
        SetDesc(add1, 1, true, DT_FLOAT, axes_shape);
        SetDesc(add1, 0, false, DT_FLOAT, mean_shape);
        SetDesc(rsqrt, 0, true, DT_FLOAT, mean_shape);
        SetDesc(rsqrt, 0, false, DT_FLOAT, mean_shape);
        SetDesc(mul1, 0, true, DT_FLOAT, opt.swap_mul1 ? param_shape : mean_shape);
        SetDesc(mul1, 1, true, DT_FLOAT, opt.swap_mul1 ? mean_shape : param_shape);
        SetDesc(mul1, 0, false, DT_FLOAT, x_shape);
        SetDesc(mul2, 0, true, DT_FLOAT, mean_shape);
        SetDesc(mul2, 1, true, DT_FLOAT, x_shape);
        SetDesc(mul2, 0, false, DT_FLOAT, x_shape);
        SetDesc(mul3, 0, true, DT_FLOAT, x_shape);
        SetDesc(mul3, 1, true, DT_FLOAT, x_shape);
        SetDesc(mul3, 0, false, DT_FLOAT, x_shape);
        SetDesc(sub, 0, true, DT_FLOAT, param_shape);
        SetDesc(sub, 1, true, DT_FLOAT, x_shape);
        SetDesc(sub, 0, false, DT_FLOAT, x_shape);
        SetDesc(add2, 0, true, DT_FLOAT, x_shape);
        SetDesc(add2, 1, true, DT_FLOAT, x_shape);
        SetDesc(add2, 0, false, DT_FLOAT, x_shape);

        std::vector<es::EsTensorHolder> outputs = {add2};
        if (opt.training_flow) {
            // 训练图特征：反向分支额外复用 rsqrt，rsqrt 出边数 > 1
            auto extra = es::Mul(rsqrt, rsqrt);
            SetDesc(extra, 0, true, DT_FLOAT, mean_shape);
            SetDesc(extra, 1, true, DT_FLOAT, mean_shape);
            SetDesc(extra, 0, false, DT_FLOAT, mean_shape);
            outputs.emplace_back(extra);
        }
        return graph_builder.BuildAndReset(outputs);
    }

    static Status RunPass(std::shared_ptr<Graph>& graph)
    {
        CustomPassContext pass_context;
        LayerNormInferenceFusionPass pass;
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

// 推理形态基线：融合并校验 LayerNorm 三个属性
TEST_F(LayerNormInferenceFusionPassTest, fuse_success)
{
    GraphOption opt;
    auto graph = BuildGraph({2, 4, 8}, opt);

    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 1);
    EXPECT_EQ(CountOpType(graph, "ReduceMean"), 0);
    EXPECT_EQ(CountOpType(graph, "SquaredDifference"), 0);
    EXPECT_EQ(CountOpType(graph, "Rsqrt"), 0);
    EXPECT_EQ(CountOpType(graph, "Mul"), 0);
    EXPECT_EQ(CountOpType(graph, "Sub"), 0);

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

// 交换律：三处可交换位换序后仍应命中
TEST_F(LayerNormInferenceFusionPassTest, fuse_success_commutative_operands)
{
    GraphOption opt;
    opt.swap_squared = true;
    opt.swap_mul1 = true;
    opt.swap_add2 = true;
    auto graph = BuildGraph({2, 4, 8}, opt);

    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 1);
}

// keep_dims=false → 拒绝融合
TEST_F(LayerNormInferenceFusionPassTest, not_changed_when_keep_dims_false)
{
    GraphOption opt;
    opt.keep_dims = false;
    auto graph = BuildGraph({2, 4, 8}, opt);

    RunPass(graph);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 0);
    EXPECT_EQ(CountOpType(graph, "ReduceMean"), 2);
}

// 两个 ReduceMean 的 axes 不一致 → 拒绝融合
TEST_F(LayerNormInferenceFusionPassTest, not_changed_when_axes_mismatch)
{
    GraphOption opt;
    opt.axes2 = {0};
    auto graph = BuildGraph({2, 4, 8}, opt);

    RunPass(graph);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 0);
    EXPECT_EQ(CountOpType(graph, "ReduceMean"), 2);
}

// 训练图：rsqrt 被反向分支复用，破坏 pattern 自包含 → 不融合
TEST_F(LayerNormInferenceFusionPassTest, not_changed_for_training_flow)
{
    GraphOption opt;
    opt.training_flow = true;
    auto graph = BuildGraph({2, 4, 8}, opt);

    RunPass(graph);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 0);
    EXPECT_EQ(CountOpType(graph, "Rsqrt"), 1);
}

// 两个 ReduceMean 共用同一个 axes 常量（GE CSE 后的常态）→ 仍应融合
TEST_F(LayerNormInferenceFusionPassTest, fuse_success_shared_axes_const)
{
    GraphOption opt;
    opt.shared_axes = true;
    auto graph = BuildGraph({2, 4, 8}, opt);

    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, "LayerNorm"), 1);
    EXPECT_EQ(CountOpType(graph, "ReduceMean"), 0);
}
