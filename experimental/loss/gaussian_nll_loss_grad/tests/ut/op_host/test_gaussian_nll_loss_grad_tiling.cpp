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
#include "tiling_context_faker.h"
#include "../../../op_kernel/gaussian_nll_loss_grad_tiling_data.h"

namespace GaussianNllLossGradUT {
constexpr uint64_t RAW_BUFFER_COUNT = 5;
constexpr uint64_t FLOAT_BUFFER_COUNT = 6;
constexpr uint64_t RAW_BUFFER_EXTRA_BYTES = 32;
constexpr uint64_t ALIGN_ELEMS = 8;
constexpr uint64_t PLATFORM_UB_RESERVED_BYTES = 256;

static uint64_t WorstCaseKernelUbBytes(uint64_t tileDataNum)
{
    return (RAW_BUFFER_COUNT + FLOAT_BUFFER_COUNT) * sizeof(float) * tileDataNum +
           RAW_BUFFER_COUNT * RAW_BUFFER_EXTRA_BYTES;
}

static uint64_t AvailableUbBytes(uint64_t platformUbBytes)
{
    return platformUbBytes > PLATFORM_UB_RESERVED_BYTES ? platformUbBytes - PLATFORM_UB_RESERVED_BYTES : 0;
}

struct CompileInfo {
} compileInfo;

struct Case {
    std::initializer_list<int64_t> gradOutput;
    std::initializer_list<int64_t> input;
    std::initializer_list<int64_t> target;
    std::initializer_list<int64_t> var;
    std::initializer_list<int64_t> gradInput;
    std::initializer_list<int64_t> gradVar;
    ge::DataType firstDtype = ge::DT_FLOAT;
    ge::DataType otherDtype = ge::DT_FLOAT;
    bool full = false;
    float eps = 1e-6f;
    std::string reduction = "mean";
    uint64_t cores = 8;
    uint64_t ub = 196608;
};

static bool ExecuteCase(const Case& param, TilingInfo& info)
{
    gert::StorageShape gradOutputShape = {param.gradOutput, param.gradOutput};
    gert::StorageShape inputShape = {param.input, param.input};
    gert::StorageShape targetShape = {param.target, param.target};
    gert::StorageShape varShape = {param.var, param.var};
    gert::StorageShape gradInputShape = {param.gradInput, param.gradInput};
    gert::StorageShape gradVarShape = {param.gradVar, param.gradVar};
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {gradOutputShape, param.firstDtype, ge::FORMAT_ND},
        {inputShape, param.otherDtype, ge::FORMAT_ND},
        {targetShape, param.otherDtype, ge::FORMAT_ND},
        {varShape, param.otherDtype, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {gradInputShape, param.otherDtype, ge::FORMAT_ND},
        {gradVarShape, param.otherDtype, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"full", Ops::NN::AnyValue::CreateFrom<bool>(param.full)},
        {"eps", Ops::NN::AnyValue::CreateFrom<float>(param.eps)},
        {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(param.reduction)},
    };
    gert::TilingContextPara context("GaussianNllLossGrad", inputs, outputs, attrs, &compileInfo, param.cores, param.ub,
                                    4096);
    return ExecuteTiling(context, info);
}

static const GaussianNllLossGradTilingData* Data(const TilingInfo& info)
{
    return reinterpret_cast<const GaussianNllLossGradTilingData*>(info.tilingData.get());
}

TEST(GaussianNllLossGradTiling, ClassifiesAllBroadcastModes)
{
    struct Expected {
        Case param;
        uint32_t targetMode;
        uint32_t targetAxis;
        uint32_t targetInner;
        uint32_t varMode;
        uint32_t varReduce;
    };
    const Expected cases[] = {
        {{{2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT, false, 1e-6f, "none"},
         0,
         1,
         1,
         0,
         1},
        {{{2, 3}, {2, 3}, {2, 1}, {2, 1}, {2, 3}, {2, 1}, ge::DT_FLOAT16, ge::DT_FLOAT16, true, 1e-4f, "none"},
         1,
         3,
         1,
         1,
         3},
        {{{1}, {2, 3}, {1, 3}, {2}, {2, 3}, {2}, ge::DT_BF16, ge::DT_BF16, false, 1e-6f, "sum"}, 1, 2, 3, 2, 3},
        {{{1}, {2, 3}, {2, 3}, {1}, {2, 3}, {1}, ge::DT_FLOAT, ge::DT_FLOAT, false, 1e-6f, "mean"}, 0, 1, 1, 3, 6},
    };
    for (const auto& item : cases) {
        TilingInfo info;
        ASSERT_TRUE(ExecuteCase(item.param, info));
        ASSERT_EQ(info.tilingDataSize, sizeof(GaussianNllLossGradTilingData));
        ASSERT_EQ(info.workspaceSizes, std::vector<int64_t>({0}));
        EXPECT_EQ(info.tilingKey, 0);
        const auto* data = Data(info);
        EXPECT_EQ(data->targetBroadcastMode, item.targetMode);
        EXPECT_EQ(data->targetBroadcastAxisSize, item.targetAxis);
        EXPECT_EQ(data->targetInnerStride, item.targetInner);
        EXPECT_EQ(data->varBroadcastMode, item.varMode);
        EXPECT_EQ(data->varReduceSize, item.varReduce);
        EXPECT_GT(data->tileDataNum, 0U);
        EXPECT_LE(WorstCaseKernelUbBytes(data->tileDataNum), AvailableUbBytes(item.param.ub));
    }
}

TEST(GaussianNllLossGradTiling, AccountsForEveryKernelBuffer)
{
    Case param{{1}, {128}, {128}, {128}, {128}, {128}};
    param.cores = 1;
    param.ub = 1280;
    TilingInfo info;
    ASSERT_TRUE(ExecuteCase(param, info));
    const auto* data = Data(info);
    EXPECT_EQ(data->tileDataNum, 16U);
    EXPECT_LE(WorstCaseKernelUbBytes(data->tileDataNum), AvailableUbBytes(param.ub));
    EXPECT_GT(WorstCaseKernelUbBytes(data->tileDataNum + ALIGN_ELEMS), AvailableUbBytes(param.ub));

    param.ub = PLATFORM_UB_RESERVED_BYTES + WorstCaseKernelUbBytes(ALIGN_ELEMS) - 1;
    EXPECT_FALSE(ExecuteCase(param, info));
    param.ub = PLATFORM_UB_RESERVED_BYTES + WorstCaseKernelUbBytes(ALIGN_ELEMS);
    ASSERT_TRUE(ExecuteCase(param, info));
    EXPECT_EQ(Data(info)->tileDataNum, ALIGN_ELEMS);
}

TEST(GaussianNllLossGradTiling, ConservesLogicalElementsAndSupportsMultipleTiles)
{
    Case param{{1}, {10000}, {10000}, {1}, {10000}, {1}};
    param.cores = 1;
    param.reduction = "mean";
    TilingInfo info;
    ASSERT_TRUE(ExecuteCase(param, info));
    const auto* data = Data(info);
    EXPECT_GT(data->finalBigTileNum, 1U);
    const uint64_t conserved = static_cast<uint64_t>(data->tailBlockNum) * data->bigCoreDataNum +
                               (info.blockNum - data->tailBlockNum) * static_cast<uint64_t>(data->smallCoreDataNum);
    EXPECT_EQ(conserved, data->totalDataNum);
}

TEST(GaussianNllLossGradTiling, HandlesEmptyTensor)
{
    Case param{{1}, {0, 3}, {0, 3}, {1}, {0, 3}, {1}};
    TilingInfo info;
    ASSERT_TRUE(ExecuteCase(param, info));
    const auto* data = Data(info);
    EXPECT_EQ(data->totalDataNum, 0U);
    EXPECT_EQ(data->meanScale, 0.0f);
    EXPECT_EQ(info.blockNum, 1U);
}

TEST(GaussianNllLossGradTiling, RejectsInvalidContracts)
{
    std::vector<Case> cases;
    cases.push_back({{2, 3}, {2, 3}, {1, 1}, {2, 3}, {2, 3}, {2, 3}});
    cases.push_back({{2, 3}, {2, 3}, {2, 3}, {3}, {2, 3}, {3}});
    cases.push_back({{1}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT, false, 1e-6f, "none"});
    cases.push_back({{1}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT16, ge::DT_FLOAT});
    Case badEps{{1}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}};
    badEps.eps = 0.0f;
    cases.push_back(badEps);
    Case badReduction = badEps;
    badReduction.eps = 1e-6f;
    badReduction.reduction = "batchmean";
    cases.push_back(badReduction);
    for (size_t i = 0; i < cases.size(); ++i) {
        TilingInfo info;
        EXPECT_FALSE(ExecuteCase(cases[i], info)) << "invalid case index " << i;
    }
}
} // namespace GaussianNllLossGradUT
