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
#include "tiling_case_executor.h"
#include "../../../op_host/single_layer_lstm_grad_tiling_arch35.h"
#include "../../../op_kernel/arch35/single_layer_lstm_grad_regbase_tiling_data.h"
#include "../../../op_kernel/arch35/single_layer_lstm_grad_wide_workspace.h"
#include "../../../op_host/single_layer_lstm_grad_tiling.h"
#include <limits>
#include <cstring>

TEST(SingleLayerLstmGradRegbaseLayout, BiasCompensationSurvivesOtherPhases)
{
    for (int64_t dtypeSize : {2, 4}) {
        for (int64_t hidden : {17, 32, 48, 97, 512}) {
            LstmGradRegbase::LstmGradRegbaseSmallUbLayout layout;
            layout.Fill(3, 5, hidden, 64, 15, dtypeSize);
            const int64_t biasBytes = LstmGradRegbase::AlignUpI64(4 * layout.hAlignF * 4, 64);
            EXPECT_EQ(layout.dbCompOff, layout.dbStageOff + biasBytes);
            EXPECT_EQ(layout.smallStageOff, layout.dbCompOff + biasBytes);
            EXPECT_EQ(layout.dbCompOff % 64, 0);
            EXPECT_LT(layout.outStageOff, layout.dbStageOff);
            EXPECT_GT(layout.totalBytes, layout.smallStageOff);
        }
    }
}

// Exercise the production small-path planner without invoking the legacy path.
// A declined plan is exposed as failure only by this test registration.
namespace optiling {
static ge::graphStatus GradRegbaseUnitTiling(gert::TilingContext* context)
{
    bool handled = false;
    const auto status = TilingSingleLayerLstmGrad4RegbaseSmall(context, handled);
    return status == ge::GRAPH_SUCCESS && handled ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
IMPL_OP_OPTILING(SingleLayerLstmGradRegbaseUT).Tiling(GradRegbaseUnitTiling);
} // namespace optiling

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;
struct CompileInfoStub {
    uint64_t workspace = 0;
} compileInfo;
constexpr int64_t T = 3;
constexpr int64_t B = 5;
constexpr int64_t I = 33;
constexpr int64_t H = 17;

gert::TilingContextPara MakePara(ge::DataType dtype, int badIndex = -1, ge::DataType badDtype = ge::DT_FLOAT16,
                                 bool withSequence = false, bool wrongWeight = false, int64_t inputSize = I,
                                 int64_t biasComponents = 1)
{
    const TensorDesc sequence{{{T, B, H}, {T, B, H}}, dtype, ge::FORMAT_ND};
    const TensorDesc state{{{1, B, H}, {1, B, H}}, dtype, ge::FORMAT_ND};
    const TensorDesc narrowState{{{1, B, H}, {1, B, H}}, dtype, ge::FORMAT_ND};
    const int64_t cols = inputSize + H + (wrongWeight ? 1 : 0);
    const TensorDesc weight{{{4 * H, cols}, {4 * H, cols}}, dtype, ge::FORMAT_ND};
    const TensorDesc bias{{{4 * H}, {4 * H}}, dtype, ge::FORMAT_ND};
    std::vector<TensorDesc> inputs = {
        {{{T, B, inputSize}, {T, B, inputSize}}, dtype, ge::FORMAT_ND},
        weight,
        bias,
        {{{}, {}}, dtype, ge::FORMAT_ND}, // unused y input
        state,
        state,
        sequence,
        sequence,
        sequence,
        narrowState,
        narrowState,
        sequence,
        sequence,
        sequence,
        sequence,
        sequence,
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
    };
    if (badIndex >= 0) {
        // TensorDescription stores shapes, dtype and format; retain each shape.
        const bool init = badIndex == 4 || badIndex == 5 || badIndex == 9 || badIndex == 10;
        if (badIndex == 0) {
            inputs[badIndex] = {{{T, B, inputSize}, {T, B, inputSize}}, badDtype, ge::FORMAT_ND};
        } else if (init) {
            inputs[badIndex] = {{{1, B, H}, {1, B, H}}, badDtype, ge::FORMAT_ND};
        } else {
            inputs[badIndex] = {{{T, B, H}, {T, B, H}}, badDtype, ge::FORMAT_ND};
        }
    }
    if (withSequence) {
        inputs[16] = {{{B}, {B}}, ge::DT_INT64, ge::FORMAT_ND};
    }
    std::vector<TensorDesc> outputs = {weight, bias, inputs[0], narrowState, narrowState};
    inputs[2] = {{{biasComponents * 4 * H}, {biasComponents * 4 * H}}, dtype, ge::FORMAT_ND};
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom<std::string>("UNIDIRECTIONAL")},
        {"gate_order", Ops::NN::AnyValue::CreateFrom<std::string>("ifjo")},
    };
    return gert::TilingContextPara("SingleLayerLstmGradRegbaseUT", inputs, outputs, attrs, &compileInfo, 72, 248 * 1024,
                                   4096);
}
} // namespace

TEST(SingleLayerLstmGradRegbase, fp32_workspace_includes_framework_prefix)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara(ge::DT_FLOAT), info));
    EXPECT_EQ(info.tilingKey, 20000);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_GT(info.workspaceSizes[0], 0);
}

class SingleLayerLstmGradRegbaseNarrow : public testing::TestWithParam<ge::DataType> {};
TEST_P(SingleLayerLstmGradRegbaseNarrow, accepts_same_dtype_and_sizes_accumulator)
{
    TilingInfo fp32, narrow;
    ASSERT_TRUE(ExecuteTiling(MakePara(ge::DT_FLOAT), fp32));
    ASSERT_TRUE(ExecuteTiling(MakePara(GetParam()), narrow));
    EXPECT_EQ(narrow.tilingKey, 20000);
    EXPECT_GT(narrow.blockNum, 0U);
    ASSERT_EQ(narrow.workspaceSizes.size(), 1U);
    ASSERT_EQ(fp32.workspaceSizes.size(), 1U);
    const auto* plan = reinterpret_cast<const LstmGradRegbaseSmallTilingData*>(narrow.tilingData.get());
    const int64_t replayFloats = 7 * plan->usedCores * T * plan->bBlock * H;
    EXPECT_EQ(narrow.workspaceSizes[0] - fp32.workspaceSizes[0], (4 * H * (I + H + 1) + replayFloats) * 4);
    EXPECT_EQ(plan->biasComponents, 1);
}
TEST_P(SingleLayerLstmGradRegbaseNarrow, preserves_two_bias_components_for_replay)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara(GetParam(), -1, ge::DT_FLOAT, false, false, I, 2), info));
    const auto* plan = reinterpret_cast<const LstmGradRegbaseSmallTilingData*>(info.tilingData.get());
    EXPECT_EQ(plan->biasComponents, 2);
    EXPECT_FALSE(ExecuteTiling(MakePara(GetParam(), -1, ge::DT_FLOAT, false, false, I, 3), info));
    EXPECT_FALSE(ExecuteTiling(MakePara(ge::DT_FLOAT, -1, ge::DT_FLOAT, false, false, I, 2), info));
}
TEST_P(SingleLayerLstmGradRegbaseNarrow, declines_each_mixed_internal_input)
{
    for (int index : {0, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15}) {
        SCOPED_TRACE(index);
        TilingInfo info;
        EXPECT_FALSE(ExecuteTiling(MakePara(GetParam(), index, ge::DT_FLOAT), info));
    }
}
TEST_P(SingleLayerLstmGradRegbaseNarrow, declines_wrong_seed_width)
{
    for (int index : {9, 10}) {
        SCOPED_TRACE(index);
        TilingInfo info;
        EXPECT_FALSE(ExecuteTiling(MakePara(GetParam(), index, ge::DT_FLOAT), info));
    }
}
TEST_P(SingleLayerLstmGradRegbaseNarrow, declines_sequence_length)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakePara(GetParam(), -1, ge::DT_FLOAT, true), info));
}
TEST_P(SingleLayerLstmGradRegbaseNarrow, declines_wrong_weight_shape)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakePara(GetParam(), -1, ge::DT_FLOAT, false, true), info));
}
INSTANTIATE_TEST_SUITE_P(MixedWidth, SingleLayerLstmGradRegbaseNarrow, testing::Values(ge::DT_FLOAT16, ge::DT_BF16));

TEST(SingleLayerLstmGradWide, private_regions_are_aligned_and_disjoint)
{
    for (int64_t hidden : {17, 2048, 4096, 8192}) {
        for (int64_t parts : {0, 1, 2}) {
            LstmGradWide::Workspace plan;
            ASSERT_TRUE(plan.Fill(3, 5, 33, hidden, parts));
            const uint64_t offsets[] = {plan.x,      plan.w,      plan.bias,   plan.initH, plan.initC, plan.dy,
                                        plan.dh,     plan.dc,     plan.cache,  plan.dx,    plan.dw,    plan.db,
                                        plan.dhPrev, plan.dcPrev, plan.legacy, plan.bytes};
            const uint64_t counts[] = {
                plan.inputElements, plan.weightElements, uint64_t{4} * hidden * parts,
                plan.stateElements, plan.stateElements,  plan.planeElements,
                plan.stateElements, plan.stateElements,  7 * plan.planeElements,
                plan.inputElements, plan.weightElements, uint64_t{4} * hidden,
                plan.stateElements, plan.stateElements,  uint64_t{15} * 4 * hidden + uint64_t{20} * (33 + hidden)};
            for (size_t n = 0; n < 15; ++n) {
                EXPECT_EQ(offsets[n] % LstmGradWide::Workspace::ALIGN, 0U);
                EXPECT_GE(offsets[n + 1] - offsets[n], counts[n] * sizeof(float));
            }
        }
    }
}

TEST(SingleLayerLstmGradWide, rejects_invalid_or_overflowing_private_shapes)
{
    LstmGradWide::Workspace plan;
    EXPECT_FALSE(plan.Fill(0, 2, 8192, 8192, 2));
    EXPECT_FALSE(plan.Fill(1, 2, -1, 8192, 2));
    EXPECT_FALSE(plan.Fill(1, 2, 8192, 8192, 3));
    EXPECT_FALSE(plan.Fill(std::numeric_limits<int64_t>::max(), 2, 8192, 8192, 2));
    EXPECT_FALSE(plan.Fill(1, 2, std::numeric_limits<int64_t>::max(), 8192, 2));
    EXPECT_FALSE(plan.Fill(1, 2, 8192, std::numeric_limits<int64_t>::max(), 2));
}

TEST(SingleLayerLstmGradWide, large_narrow_plan_uses_private_fp32_workspace)
{
    for (int64_t hidden : {4096, 8192}) {
        optiling::SingleLayerLstmGradPlanRequest request;
        request.timeStep = 1;
        request.batch = 2;
        request.inputSize = 8192;
        request.hiddenSize = hidden;
        request.isBias = 1;
        request.biasComponents = 2;
        request.elemBytes = 2;
        request.aicCoreNum = 36;
        request.ubSizePlatForm = 248 * 1024;
        request.sysWorkspaceSize = 4096;
        request.isRegbase = true;
        alignas(8) uint8_t data[4096]{};
        optiling::SingleLayerLstmGradPlanResult result;
        ASSERT_EQ(optiling::PlanSingleLayerLstmGrad(request, data, sizeof(data), result), ge::GRAPH_SUCCESS);
        LstmGradWide::Workspace layout;
        ASSERT_TRUE(layout.Fill(1, 2, 8192, hidden, 2));
        EXPECT_EQ(result.workspaceSize, layout.bytes + request.sysWorkspaceSize);
        EXPECT_EQ(result.blockDim, 8);
        EXPECT_NE(result.tilingKey, 20000);
        request.isSeqLength = 1;
        EXPECT_NE(optiling::PlanSingleLayerLstmGrad(request, data, sizeof(data), result), ge::GRAPH_SUCCESS);
    }
}

TEST(SingleLayerLstmGradWide, framework_serialization_preserves_replay_bias_components)
{
    constexpr int64_t hidden = 4096;
    constexpr int64_t input = 1;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t parts : {1, 2}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            SCOPED_TRACE(parts);
            const TensorDesc state{{{1, 1, hidden}, {1, 1, hidden}}, dtype, ge::FORMAT_ND};
            const TensorDesc x{{{1, 1, input}, {1, 1, input}}, dtype, ge::FORMAT_ND};
            const TensorDesc weight{{{4 * hidden, input + hidden}, {4 * hidden, input + hidden}}, dtype, ge::FORMAT_ND};
            const TensorDesc bias{{{4 * hidden}, {4 * hidden}}, dtype, ge::FORMAT_ND};
            std::vector<TensorDesc> inputs = {x,
                                              weight,
                                              {{{parts * 4 * hidden}, {parts * 4 * hidden}}, dtype, ge::FORMAT_ND},
                                              {{{}, {}}, dtype, ge::FORMAT_ND},
                                              state,
                                              state,
                                              state,
                                              state,
                                              state,
                                              state,
                                              state,
                                              state,
                                              state,
                                              state,
                                              state,
                                              state,
                                              {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}};
            std::vector<TensorDesc> outputs = {weight, bias, x, state, state};
            std::vector<OpAttr> attrs = {{"direction", Ops::NN::AnyValue::CreateFrom<std::string>("UNIDIRECTIONAL")},
                                         {"gate_order", Ops::NN::AnyValue::CreateFrom<std::string>("ifjo")}};
            gert::TilingContextPara para("SingleLayerLstmGrad", inputs, outputs, attrs, &compileInfo, 72, 248 * 1024,
                                         4096);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(para, info));
            ASSERT_NE(info.tilingKey, 20000);
            optiling::SingleLayerLstmGradTilingData schema;
            ASSERT_EQ(info.tilingDataSize, schema.GetDataSize());
            // privateBiasComponents is the final int64 field of the serialized layout.
            int64_t serializedParts = -1;
            std::memcpy(&serializedParts, info.tilingData.get() + info.tilingDataSize - sizeof(int64_t),
                        sizeof(serializedParts));
            EXPECT_EQ(serializedParts, parts);
        }
    }
}

class SingleLayerLstmGradZeroFeatures : public testing::TestWithParam<ge::DataType> {};
TEST_P(SingleLayerLstmGradZeroFeatures, recurrent_only_plan)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara(GetParam(), -1, ge::DT_FLOAT, false, false, 0), info));
    EXPECT_EQ(info.tilingKey, 20000);
    EXPECT_EQ(info.blockNum, 1U);
    ASSERT_GE(info.tilingDataSize, sizeof(LstmGradRegbaseSmallTilingData));
    const auto* plan = reinterpret_cast<const LstmGradRegbaseSmallTilingData*>(info.tilingData.get());
    EXPECT_EQ(plan->inputSize, 0);
    EXPECT_EQ(plan->numIChunks, 0);
    EXPECT_EQ(plan->usedCores, 1);
    EXPECT_EQ(plan->hiddenSize, H);
}
INSTANTIATE_TEST_SUITE_P(AllWidths, SingleLayerLstmGradZeroFeatures,
                         testing::Values(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16));
