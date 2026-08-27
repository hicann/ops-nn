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

#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/inplace_add_layer_norm_proto.h"
#include "../../../op_graph/fusion_pass/inplace_add_layer_norm_fusion_pass.h"

using namespace fe;
using namespace ge;
using namespace ops;

namespace {
constexpr float kEpsilon = 0.00001f;
constexpr int64_t kX1Idx = 0;
constexpr int64_t kX2Idx = 1;
constexpr int64_t kGammaIdx = 2;
constexpr int64_t kBetaIdx = 3;
constexpr int64_t kBiasIdx = 4;

// 守卫 G6 要求 l2/8 <= x1_bytes <= 2*l2。
constexpr int64_t kL2Size = 134217728L;
constexpr int64_t kRowsInBand = 16384L;    // fp16 [16384,1024] = 32MB，落在 [16MB,256MB]
constexpr int64_t kRowsTooSmall = 256L;    // 0.5MB，低于下界
constexpr int64_t kRowsTooLarge = 262144L; // 512MB，高于上界
constexpr int64_t kCols = 1024L;
constexpr int64_t kDynamicReduceAxis = 5120L; // 动态 shape 下的经验归约轴

const char* const kAddLayerNorm = "AddLayerNorm";
const char* const kInplaceAddLayerNorm = "InplaceAddLayerNorm";

class ZInplaceAddLayerNormFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }

    static void SetPlatform(const std::string& soc, int64_t l2Size = kL2Size)
    {
        PlatformInfo platformInfo;
        OptionalInfo optionalInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.soc_info.l2_size = static_cast<uint64_t>(l2Size);
        platformInfo.str_info.short_soc_version = soc;
        optionalInfo.soc_version = soc;
        PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
    }

    // 构造单个 AddLayerNorm 节点的图。
    static std::shared_ptr<Graph> BuildGraph(bool withBias, int64_t rows = kRowsInBand, int64_t cols = kCols,
                                             DataType xDtype = DT_FLOAT16, DataType gammaDtype = DT_FLOAT16,
                                             bool extraConsumerOnX1 = false, bool extraConsumerOnY = false)
    {
        const std::vector<int64_t> xShape = {rows, cols};
        const std::vector<int64_t> pShape = {cols};

        auto builder = es::EsGraphBuilder("inplace_add_layer_norm_fusion_test");
        auto x1 = builder.CreateInput(kX1Idx, "x1", xDtype, FORMAT_ND, xShape);
        auto x2 = builder.CreateInput(kX2Idx, "x2", xDtype, FORMAT_ND, xShape);
        auto gamma = builder.CreateInput(kGammaIdx, "gamma", gammaDtype, FORMAT_ND, pShape);
        auto beta = builder.CreateInput(kBetaIdx, "beta", gammaDtype, FORMAT_ND, pShape);

        es::AddLayerNormOutput out;
        if (withBias) {
            auto bias = builder.CreateInput(kBiasIdx, "bias", gammaDtype, FORMAT_ND, pShape);
            out = es::AddLayerNorm(x1, x2, gamma, beta, bias, kEpsilon, false);
        } else {
            out = es::AddLayerNorm(x1, x2, gamma, beta, nullptr, kEpsilon, false);
        }

        UpdateInputDesc(out.y, 0, xDtype, xShape);
        UpdateInputDesc(out.y, 1, xDtype, xShape);
        UpdateInputDesc(out.y, 2, gammaDtype, pShape);
        UpdateInputDesc(out.y, 3, gammaDtype, pShape);
        if (withBias) {
            UpdateInputDesc(out.y, 4, gammaDtype, pShape);
        }
        UpdateOutputDesc(out.y, 0, xDtype, xShape);
        UpdateOutputDesc(out.mean, 1, DT_FLOAT, {rows, 1});
        UpdateOutputDesc(out.rstd, 2, DT_FLOAT, {rows, 1});
        UpdateOutputDesc(out.x, 3, xDtype, xShape);

        std::vector<es::EsTensorHolder> outputs = {out.y, out.mean, out.rstd, out.x};
        if (extraConsumerOnX1) {
            outputs.emplace_back(es::Relu(x1));
        }
        if (extraConsumerOnY) {
            outputs.emplace_back(es::Relu(out.y));
        }
        return builder.BuildAndReset(outputs);
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
        CustomPassContext passContext;
        ZInplaceAddLayerNormFusionPass pass;
        return pass.Run(graph, passContext);
    }

    static int CountOpType(const std::shared_ptr<Graph>& graph, const char* opType)
    {
        int count = 0;
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == AscendString(opType)) {
                ++count;
            }
        }
        return count;
    }

    static bool FindNodeByType(const std::shared_ptr<Graph>& graph, const char* opType, GNode& found)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == AscendString(opType)) {
                found = node;
                return true;
            }
        }
        return false;
    }
};

TEST_F(ZInplaceAddLayerNormFusionPassTest, control_edges_transferred_to_new_node)
{
    auto graph = BuildGraph(false, kRowsInBand, kCols, DT_FLOAT16, DT_FLOAT16, false, true);

    GNode addLn;
    ASSERT_TRUE(FindNodeByType(graph, kAddLayerNorm, addLn));
    GNode relu;
    ASSERT_TRUE(FindNodeByType(graph, "Relu", relu));

    // Relu --ctrl--> AddLayerNorm --ctrl--> Relu 都挂上，覆盖入/出两个方向
    ASSERT_EQ(graph->AddControlEdge(relu, addLn), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddControlEdge(addLn, relu), GRAPH_SUCCESS);
    ASSERT_EQ(addLn.GetInControlNodes().size(), 1U);
    ASSERT_EQ(addLn.GetOutControlNodes().size(), 1U);

    ASSERT_EQ(RunPass(graph), SUCCESS);

    GNode inplaceNode;
    ASSERT_TRUE(FindNodeByType(graph, kInplaceAddLayerNorm, inplaceNode));
    EXPECT_EQ(inplaceNode.GetInControlNodes().size(), 1U);
    EXPECT_EQ(inplaceNode.GetOutControlNodes().size(), 1U);
}

// ---------- 正向：两种 bias 形态 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, fusion_success_without_bias)
{
    auto graph = BuildGraph(false);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 0);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, fusion_success_with_bias)
{
    auto graph = BuildGraph(true);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 0);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, fusion_success_bf16)
{
    auto graph = BuildGraph(false, kRowsInBand, kCols, DT_BF16, DT_BF16);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 1);
}

// ---------- graph 路线特有：接线与 attr 必须逐位保住 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, inputs_and_outputs_rewired)
{
    auto graph = BuildGraph(true);
    ASSERT_EQ(RunPass(graph), SUCCESS);

    GNode newNode;
    ASSERT_TRUE(FindNodeByType(graph, kInplaceAddLayerNorm, newNode));
    EXPECT_EQ(newNode.GetOutputsSize(), 4U);
    for (int32_t i = 0; i < 5; ++i) {
        EXPECT_NE(newNode.GetInDataNodesAndPortIndexs(i).first, nullptr) << "input " << i << " 未接线";
    }
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, downstream_consumer_rewired_to_new_node)
{
    // y 除了作为图输出，还被一个 Relu 消费；替换后该 Relu 的生产者必须是新节点
    auto graph = BuildGraph(false, kRowsInBand, kCols, DT_FLOAT16, DT_FLOAT16, false, true);
    ASSERT_EQ(RunPass(graph), SUCCESS);

    GNode relu;
    ASSERT_TRUE(FindNodeByType(graph, "Relu", relu));
    auto producer = relu.GetInDataNodesAndPortIndexs(0).first;
    ASSERT_NE(producer, nullptr);
    AscendString producerType;
    producer->GetType(producerType);
    EXPECT_EQ(producerType, AscendString(kInplaceAddLayerNorm));
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, attrs_transferred_to_new_node)
{
    auto graph = BuildGraph(false);
    ASSERT_EQ(RunPass(graph), SUCCESS);

    GNode newNode;
    ASSERT_TRUE(FindNodeByType(graph, kInplaceAddLayerNorm, newNode));
    float32_t epsilon = 0.0F;
    EXPECT_EQ(newNode.GetAttr(AscendString("epsilon"), epsilon), GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(epsilon, kEpsilon);
    bool additionalOutput = true;
    EXPECT_EQ(newNode.GetAttr(AscendString("additional_output"), additionalOutput), GRAPH_SUCCESS);
    EXPECT_FALSE(additionalOutput);
}

// ---------- 守卫 G2：平台校验 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_unsupported_platform)
{
    SetPlatform("Ascend310P");
    auto graph = BuildGraph(false);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 0);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, platform_ascend910b_supported)
{
    SetPlatform("Ascend910B");
    auto graph = BuildGraph(false);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 1);
}

// ---------- 守卫 G4：x1 被其它节点消费时不可原地 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_input_shared_by_other_consumers)
{
    auto graph = BuildGraph(false, kRowsInBand, kCols, DT_FLOAT16, DT_FLOAT16, true);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 0);
}

// ---------- 守卫 G6：shape 必须落在 L2 区间 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_shape_below_l2_band)
{
    auto graph = BuildGraph(false, kRowsTooSmall);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_shape_above_l2_band)
{
    auto graph = BuildGraph(false, kRowsTooLarge);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
}

// ---------- 守卫 G6：动态 shape 走末维判定 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, fusion_success_dynamic_shape_on_reduce_axis)
{
    auto graph = BuildGraph(false, -1L, kDynamicReduceAxis);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 1);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_dynamic_shape_other_axis)
{
    auto graph = BuildGraph(false, -1L, kCols);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
}

// ---------- 守卫 G7：x1/x2/gamma dtype 必须一致 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_mixed_input_dtype)
{
    auto graph = BuildGraph(false, kRowsInBand, kCols, DT_FLOAT16, DT_FLOAT);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 0);
}
} // namespace
