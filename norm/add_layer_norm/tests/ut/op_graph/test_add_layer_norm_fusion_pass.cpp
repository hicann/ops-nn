/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "../../../op_graph/fusion_pass/add_layer_norm_fusion_pass.h"

using namespace ge;

namespace {
using Holder = es::EsTensorHolder;
const std::vector<int64_t> kXShape = {2, 2, 64};
const std::vector<int64_t> kScaleShape = {64};
const std::vector<int64_t> kStatsShape = {2, 2, 1};

enum class Scenario { V4_DIRECT, V4_BIAS, V3_DOUBLE_ADD, V3_INPUT_CAST, V3_NON_LAST_AXIS };

TensorDesc Desc(DataType dtype, const std::vector<int64_t>& dims)
{
    TensorDesc desc(Shape(dims), FORMAT_ND, dtype);
    desc.SetOriginShape(Shape(dims));
    desc.SetOriginFormat(FORMAT_ND);
    desc.SetRealDimCnt(static_cast<int64_t>(dims.size()));
    return desc;
}

void SetOutput(const Holder& holder, const TensorDesc& desc)
{
    ASSERT_NE(holder.GetProducer(), nullptr);
    ASSERT_EQ(holder.GetProducer()->UpdateOutputDesc(holder.GetProducerOutIndex(), desc), GRAPH_SUCCESS);
}

void SetInput(const Holder& holder, int32_t index, const TensorDesc& desc)
{
    ASSERT_NE(holder.GetProducer(), nullptr);
    ASSERT_EQ(holder.GetProducer()->UpdateInputDesc(index, desc), GRAPH_SUCCESS);
}

Holder Data(es::EsGraphBuilder& builder, int64_t index, const char* name, const TensorDesc& desc)
{
    auto data = builder.CreateInput(index, name, desc.GetDataType(), desc.GetFormat(), desc.GetShape().GetDims());
    SetInput(data, 0, desc);
    SetOutput(data, desc);
    return data;
}

Holder Add(const Holder& x1, const Holder& x2, const TensorDesc& desc1, const TensorDesc& desc2)
{
    auto add = es::Add(x1, x2);
    SetInput(add, 0, desc1);
    SetInput(add, 1, desc2);
    SetOutput(add, desc1);
    return add;
}

Holder Cast(const Holder& x, const TensorDesc& input, const TensorDesc& output)
{
    auto cast = es::Cast(x, output.GetDataType());
    SetInput(cast, 0, input);
    SetOutput(cast, output);
    return cast;
}

GraphPtr BuildGraph(Scenario scenario)
{
    const bool v4 = scenario == Scenario::V4_DIRECT || scenario == Scenario::V4_BIAS;
    const bool third = scenario == Scenario::V3_DOUBLE_ADD;
    const bool inputCast = scenario == Scenario::V3_INPUT_CAST;
    const bool withBias = scenario == Scenario::V4_BIAS;
    const bool nonLast = scenario == Scenario::V3_NON_LAST_AXIS;
    const DataType addType = third ? DT_BF16 : (inputCast ? DT_FLOAT : DT_FLOAT16);
    const TensorDesc xDesc = Desc(addType, kXShape);
    const TensorDesc rawDesc = Desc(inputCast ? DT_FLOAT16 : addType, kXShape);
    const TensorDesc normDesc = Desc(third ? DT_FLOAT : addType, kXShape);
    const TensorDesc scaleDesc = Desc(normDesc.GetDataType(), nonLast ? std::vector<int64_t>{2, 64} : kScaleShape);
    const TensorDesc statsDesc = Desc(v4 ? DT_FLOAT : normDesc.GetDataType(),
                                      nonLast ? std::vector<int64_t>{2, 1, 1} : kStatsShape);
    es::EsGraphBuilder builder("add_layer_norm_retained_case");
    int64_t index = 0;
    auto x1 = Data(builder, index++, "x1", rawDesc);
    auto x2 = Data(builder, index++, "x2", xDesc);
    Holder bias;
    if (third || withBias) {
        bias = Data(builder, index++, "bias", Desc(addType, kScaleShape));
    }
    auto gamma = Data(builder, index++, "gamma", scaleDesc);
    auto beta = Data(builder, index++, "beta", scaleDesc);
    Holder left = inputCast ? Cast(x1, rawDesc, xDesc) : x1;
    Holder right = x2;
    if (third || withBias) {
        right = Add(x2, bias, xDesc, Desc(addType, kScaleShape));
    }
    auto add = Add(left, right, xDesc, xDesc);
    Holder normX = third ? Cast(add, xDesc, normDesc) : add;
    Holder y;
    Holder mean;
    Holder rstd;
    if (v4) {
        auto normalizedShape = Data(builder, index++, "normalized_shape", Desc(DT_INT64, {1}));
        auto norm = es::LayerNormV4(normX, normalizedShape, gamma, beta, 1.0e-5F);
        y = norm.y;
        mean = norm.mean;
        rstd = norm.rstd;
        SetInput(y, 1, Desc(DT_INT64, {1}));
        SetInput(y, 2, scaleDesc);
        SetInput(y, 3, scaleDesc);
    } else {
        const int64_t axis = nonLast ? 1 : -1;
        auto norm = es::LayerNormV3(normX, gamma, beta, axis, axis, 1.0e-5F);
        y = norm.y;
        mean = norm.mean;
        rstd = norm.rstd;
        SetInput(y, 1, scaleDesc);
        SetInput(y, 2, scaleDesc);
    }
    SetInput(y, 0, normDesc);
    SetOutput(y, normDesc);
    SetOutput(mean, statsDesc);
    SetOutput(rstd, statsDesc);
    if (third) {
        y = Cast(y, normDesc, xDesc);
    }
    return inputCast ? builder.BuildAndReset({y, add}) : builder.BuildAndReset({y});
}

std::string Type(const GNode& node)
{
    AscendString type;
    return node.GetType(type) == GRAPH_SUCCESS && type.GetString() != nullptr ? type.GetString() : "";
}

size_t Count(const GraphPtr& graph, const std::string& type)
{
    size_t count = 0;
    for (const auto& node : graph->GetAllNodes()) {
        count += Type(node) == type;
    }
    return count;
}

GNode Find(const GraphPtr& graph, const std::string& type)
{
    for (const auto& node : graph->GetAllNodes()) {
        if (Type(node) == type) {
            return node;
        }
    }
    return GNode();
}

void CheckFused(const GraphPtr& graph, DataType dtype, bool hasBias)
{
    ASSERT_EQ(Count(graph, "AddLayerNorm"), 1U);
    EXPECT_EQ(Count(graph, "Add"), 0U);
    EXPECT_EQ(Count(graph, "LayerNormV3"), 0U);
    EXPECT_EQ(Count(graph, "LayerNormV4"), 0U);
    EXPECT_EQ(Count(graph, "Cast"), 0U);
    auto fused = Find(graph, "AddLayerNorm");
    for (int32_t port = 0; port < 4; ++port) {
        TensorDesc desc;
        ASSERT_EQ(fused.GetOutputDesc(port, desc), GRAPH_SUCCESS);
        EXPECT_EQ(desc.GetDataType(), (port == 1 || port == 2) ? DT_FLOAT : dtype);
        EXPECT_EQ(desc.GetShape().GetDims(), (port == 1 || port == 2) ? kStatsShape : kXShape);
    }
    EXPECT_EQ(fused.GetInDataNodesAndPortIndexs(4).first != nullptr, hasBias);
}

class AddLayerNormFusionPassTest : public testing::Test {
protected:
    void SetUp() override
    {
        fe::PlatFormInfos platformInfos;
        fe::OptionalInfos optionalInfos;
        platformInfos.Init();
        optionalInfos.Init();
        std::map<std::string, std::string> version = {{"NpuArch", "3510"}};
        platformInfos.SetPlatformRes("version", version);
        optionalInfos.SetSocVersion("3510");
        fe::PlatformInfoManager::Instance().platform_infos_map_["3510"] = platformInfos;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfos);
    }

    static Status Run(GraphPtr& graph)
    {
        CustomPassContext context;
        ge::fusion::AddLayerNormFusionPass pass;
        return pass.Run(graph, context);
    }
};

TEST_F(AddLayerNormFusionPassTest, p1_lnv4_fp16)
{
    auto graph = BuildGraph(Scenario::V4_DIRECT);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(Run(graph), SUCCESS);
    CheckFused(graph, DT_FLOAT16, false);
    EXPECT_EQ(Count(graph, "Reshape"), 0U);
}

TEST_F(AddLayerNormFusionPassTest, legacy_p1_v4_fp16_bias)
{
    auto graph = BuildGraph(Scenario::V4_BIAS);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(Run(graph), SUCCESS);
    CheckFused(graph, DT_FLOAT16, true);
    EXPECT_EQ(Count(graph, "Reshape"), 0U);
    const auto source = Find(graph, "AddLayerNorm").GetInDataNodesAndPortIndexs(4);
    ASSERT_NE(source.first, nullptr);
    EXPECT_EQ(Type(*source.first), "Data");
}

TEST_F(AddLayerNormFusionPassTest, p3_lnv3_bf16)
{
    auto graph = BuildGraph(Scenario::V3_DOUBLE_ADD);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(Run(graph), SUCCESS);
    CheckFused(graph, DT_BF16, true);
    auto fused = Find(graph, "AddLayerNorm");
    for (int32_t port : {2, 3}) {
        TensorDesc desc;
        ASSERT_EQ(fused.GetInputDesc(port, desc), GRAPH_SUCCESS);
        EXPECT_EQ(desc.GetDataType(), DT_FLOAT);
    }
}

TEST_F(AddLayerNormFusionPassTest, input_cast_before_add)
{
    auto graph = BuildGraph(Scenario::V3_INPUT_CAST);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(Run(graph), SUCCESS);
    CheckFused(graph, DT_FLOAT, false);
    auto fused = Find(graph, "AddLayerNorm");
    TensorDesc input;
    ASSERT_EQ(fused.GetInputDesc(0, input), GRAPH_SUCCESS);
    EXPECT_EQ(input.GetDataType(), DT_FLOAT16);
    bool additionalOutput = false;
    ASSERT_EQ(fused.GetAttr("additional_output", additionalOutput), GRAPH_SUCCESS);
    EXPECT_TRUE(additionalOutput);
    auto output = Find(graph, "NetOutput");
    const auto residual = output.GetInDataNodesAndPortIndexs(1);
    ASSERT_NE(residual.first, nullptr);
    EXPECT_EQ(Type(*residual.first), "AddLayerNorm");
    EXPECT_EQ(residual.second, 3);
}

TEST_F(AddLayerNormFusionPassTest, invalid_axis_reject)
{
    auto graph = BuildGraph(Scenario::V3_NON_LAST_AXIS);
    ASSERT_NE(graph, nullptr);
    const auto count = graph->GetAllNodes().size();
    EXPECT_EQ(Run(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(graph->GetAllNodes().size(), count);
    EXPECT_EQ(Count(graph, "AddLayerNorm"), 0U);
    EXPECT_EQ(Count(graph, "Add"), 1U);
    EXPECT_EQ(Count(graph, "LayerNormV3"), 1U);
    const auto source = Find(graph, "LayerNormV3").GetInDataNodesAndPortIndexs(0);
    ASSERT_NE(source.first, nullptr);
    EXPECT_EQ(Type(*source.first), "Add");
}
} // namespace
