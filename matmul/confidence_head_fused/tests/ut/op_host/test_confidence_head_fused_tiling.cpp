/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_infos_def.h"
#include "../../../../../tests/ut/common/op_impl_registry.h"
#include "tiling_context_faker.h"
#include "../../../op_host/op_tiling/arch35/confidence_head_fused_compile_info.h"
#include "../../../op_kernel/arch35/confidence_head_fused_tiling_data.h"

namespace {
constexpr float CLAMP_LIMIT = static_cast<float>(std::numeric_limits<int32_t>::max());

class ConfidenceHeadFusedTiling : public testing::Test {
protected:
    void SetUp() override
    {
        compileInfo.socVersion = platform_ascendc::SocVersion::ASCEND950;
        compileInfo.aicNum = 32;
        compileInfo.aivNum = 64;
        compileInfo.l1Size = 524288;
        compileInfo.l0aSize = 65536;
        compileInfo.l0bSize = 65536;
        compileInfo.l0cSize = 262144;
        compileInfo.ubSize = 262144;
    }

    void CheckTiling(ge::graphStatus expectedStatus, uint32_t expectedCores = 0)
    {
        const auto impl = Ops::NN::OpImplRegistry::GetInstance().GetOpImpl("ConfidenceHeadFused");
        ASSERT_NE(impl, nullptr);
        ASSERT_NE(impl->tiling, nullptr);

        std::vector<gert::StorageShape*> inputs{&hidden, &weight};
        std::vector<size_t> inputIndices{0, 1};
        std::vector<uint32_t> instances{1, 1};
        const std::vector<gert::StorageShape*> optionalInputs{
            hasMarkov ? &markov : nullptr, hasTemperature ? &temperature : nullptr, hasBias ? &bias : nullptr};
        for (size_t i = 0; i < optionalInputs.size(); ++i) {
            instances.push_back(optionalInputs[i] != nullptr ? 1 : 0);
            if (optionalInputs[i] != nullptr) {
                inputs.push_back(optionalInputs[i]);
                inputIndices.push_back(i + 2);
            }
        }

        auto raw = gert::TilingData::CreateCap(4096);
        auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
        ASSERT_NE(raw, nullptr);
        ASSERT_NE(workspaceHolder, nullptr);
        auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

        // Keep platform resources consistent with each case's compile info.
        fe::PlatFormInfos platformInfo;
        ASSERT_TRUE(platformInfo.Init());
        const std::string socVersion = compileInfo.socVersion == platform_ascendc::SocVersion::ASCEND350 ? "Ascend350" :
                                                                                                           "Ascend950";
        std::map<std::string, std::string> versionInfo = {{"Short_SoC_version", socVersion}, {"NpuArch", "3510"}};
        std::map<std::string, std::string> socInfos = {{"ai_core_cnt", std::to_string(compileInfo.aicNum)},
                                                       {"cube_core_cnt", std::to_string(compileInfo.aicNum)},
                                                       {"vector_core_cnt", std::to_string(compileInfo.aivNum)},
                                                       {"core_type_list", "CubeCore,VectorCore"}};
        std::map<std::string, std::string> aicoreSpec = {{"l1_size", std::to_string(compileInfo.l1Size)},
                                                         {"l0_a_size", std::to_string(compileInfo.l0aSize)},
                                                         {"l0_b_size", std::to_string(compileInfo.l0bSize)},
                                                         {"l0_c_size", std::to_string(compileInfo.l0cSize)},
                                                         {"ub_size", std::to_string(compileInfo.ubSize)}};
        platformInfo.SetPlatformRes("version", versionInfo);
        platformInfo.SetPlatformRes("SoCInfo", socInfos);
        platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
        platformInfo.SetCoreNumByCoreType("AICore");

        gert::TilingContextFaker faker;
        faker.SetOpType("ConfidenceHeadFused")
            .NodeIoNum(5, 1)
            .IrInstanceNum(instances, {1})
            .InputShapes(inputs)
            .OutputShapes({&out})
            .CompileInfo(&compileInfo)
            .PlatformInfo(&platformInfo)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, outputFormat)
            .NodeAttrs({{"clamp_low", Ops::NN::AnyValue::CreateFrom<float>(clampLow)},
                        {"clamp_high", Ops::NN::AnyValue::CreateFrom<float>(clampHigh)}})
            .TilingData(raw.get())
            .Workspace(workspace);
        for (size_t i = 0; i < inputIndices.size(); ++i) {
            const auto index = inputIndices[i];
            faker.NodeInputTd(static_cast<int32_t>(i), inputDtypes[index], ge::FORMAT_ND, inputFormat);
        }
        auto holder = faker.Build();
        auto context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(impl->tiling(context), expectedStatus);
        if (expectedStatus != ge::GRAPH_SUCCESS) {
            return;
        }

        const auto tilingData = context->GetRawTilingData();
        ASSERT_NE(tilingData, nullptr);
        const auto data = static_cast<const CHF::ConfidenceHeadFusedTilingData*>(tilingData->GetData());
        ASSERT_NE(data, nullptr);
        EXPECT_EQ(tilingData->GetDataSize(), sizeof(CHF::ConfidenceHeadFusedTilingData));
        EXPECT_EQ(data->params.n, expectedRows);
        EXPECT_EQ(data->params.kEmbed, expectedKEmbed);
        EXPECT_EQ(data->params.kMarkov, expectedKMarkov);
        EXPECT_EQ(data->params.hasMarkov, hasMarkov);
        EXPECT_EQ(data->params.hasBias, hasBias);
        const auto& hiddenShape = hidden.GetOriginShape();
        EXPECT_EQ(data->params.biasN, static_cast<uint32_t>(hiddenShape.GetDim(hiddenShape.GetDimNum() - 2)));
        EXPECT_EQ(data->params.hasLowLimit, clampLow != -CLAMP_LIMIT);
        EXPECT_EQ(data->params.hasHighLimit, clampHigh != CLAMP_LIMIT);
        EXPECT_FLOAT_EQ(data->params.lowLimit, clampLow);
        EXPECT_FLOAT_EQ(data->params.highLimit, clampHigh);
        EXPECT_EQ(data->params.usedCoreNum, expectedCores);
        EXPECT_EQ(context->GetBlockDim(), expectedCores);
        EXPECT_EQ(context->GetTilingKey(), hasMarkov ? 1UL : 0UL);
        EXPECT_EQ(data->cubeTiling.baseN, 128U);
        EXPECT_EQ(data->cubeTiling.baseK, 128U);
        EXPECT_EQ(data->cubeTiling.kL1, 256U);
        EXPECT_EQ(data->cubeTiling.l1BufferNum, 4U);
        const auto workspaceSizes = context->GetWorkspaceSizes(1);
        ASSERT_NE(workspaceSizes, nullptr);
        EXPECT_EQ(workspaceSizes[0], 0U);
    }

    optiling::confidence_head_fused::ConfidenceHeadFusedCompileInfo compileInfo;
    gert::StorageShape hidden = {{257, 64}, {257, 64}};
    gert::StorageShape weight = {{64}, {64}};
    gert::StorageShape markov = {{257, 16}, {257, 16}};
    gert::StorageShape temperature = {{1}, {1}};
    gert::StorageShape bias = {{257}, {257}};
    gert::StorageShape out = {{257}, {257}};
    std::array<ge::DataType, 5> inputDtypes{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT};
    ge::Format inputFormat = ge::FORMAT_ND;
    ge::Format outputFormat = ge::FORMAT_ND;
    bool hasMarkov = false;
    bool hasTemperature = false;
    bool hasBias = false;
    float clampLow = -CLAMP_LIMIT;
    float clampHigh = CLAMP_LIMIT;
    uint32_t expectedRows = 257;
    uint32_t expectedKEmbed = 64;
    uint32_t expectedKMarkov = 0;
};

TEST_F(ConfidenceHeadFusedTiling, fp16_2d_without_optional_inputs) { CheckTiling(ge::GRAPH_SUCCESS, 3); }

TEST_F(ConfidenceHeadFusedTiling, bf16_3d_with_optional_inputs_ascend350)
{
    compileInfo.socVersion = platform_ascendc::SocVersion::ASCEND350;
    inputDtypes = {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT};
    hidden = {{2, 129, 31}, {2, 129, 31}};
    weight = {{48}, {48}};
    markov = {{2, 129, 17}, {2, 129, 17}};
    bias = {{129}, {129}};
    out = {{2, 129}, {2, 129}};
    hasMarkov = true;
    hasTemperature = true;
    hasBias = true;
    clampLow = -60.0F;
    clampHigh = 60.0F;
    expectedRows = 258;
    expectedKEmbed = 31;
    expectedKMarkov = 17;
    // Each batch has two tiles; bias broadcasting prevents a tile from crossing batches.
    CheckTiling(ge::GRAPH_SUCCESS, 4);
}

TEST_F(ConfidenceHeadFusedTiling, empty_reduction)
{
    hidden = {{257, 0}, {257, 0}};
    weight = {{0}, {0}};
    expectedKEmbed = 0;
    CheckTiling(ge::GRAPH_SUCCESS, 3);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_weight_length)
{
    weight = {{65}, {65}};
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_bias_length)
{
    hasBias = true;
    bias = {{256}, {256}};
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_temperature_shape)
{
    hasTemperature = true;
    temperature = {{2}, {2}};
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, insufficient_ub)
{
    compileInfo.ubSize = 3071;
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, bias_buffers_exceed_l1)
{
    compileInfo.l1Size = 263296;
    hasBias = true;
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, l1_without_bias_has_no_bias_buffers)
{
    compileInfo.l1Size = 263296;
    CheckTiling(ge::GRAPH_SUCCESS, 3);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_aic_aiv_ratio)
{
    compileInfo.aivNum = 32;
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, flattened_rows_overflow)
{
    hidden = {{65536, 65536, 64}, {65536, 65536, 64}};
    out = {{65536, 65536}, {65536, 65536}};
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, empty_n_skips_cube_memory_check)
{
    hidden = {{0, 64}, {0, 64}};
    out = {{0}, {0}};
    expectedRows = 0;
    compileInfo.l1Size = 1;
    CheckTiling(ge::GRAPH_SUCCESS, 1);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_weight_dtype)
{
    inputDtypes[1] = ge::DT_BF16;
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_markov_dtype)
{
    hasMarkov = true;
    inputDtypes[2] = ge::DT_FLOAT;
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_hidden_rank)
{
    hidden = {{64}, {64}};
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_weight_rank)
{
    weight = {{1, 64}, {1, 64}};
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, negative_hidden_dimension)
{
    hidden = {{257, -1}, {257, -1}};
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_output_rank)
{
    out = {{1, 257}, {1, 257}};
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_output_leading_dimension)
{
    out = {{256}, {256}};
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_markov_leading_dimension)
{
    hasMarkov = true;
    markov = {{256, 16}, {256, 16}};
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_hidden_format)
{
    inputFormat = ge::FORMAT_NCHW;
    CheckTiling(ge::GRAPH_FAILED);
}

TEST_F(ConfidenceHeadFusedTiling, invalid_output_format)
{
    outputFormat = ge::FORMAT_NCHW;
    CheckTiling(ge::GRAPH_FAILED);
}
} // namespace
