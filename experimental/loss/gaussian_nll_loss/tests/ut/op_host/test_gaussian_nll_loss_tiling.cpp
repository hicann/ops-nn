/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "tiling/math/log_tiling.h"
#include "../../../op_kernel/gaussian_nll_loss_tiling_data.h"
#include "../../../op_kernel/gaussian_nll_loss_tiling_key.h"

namespace {
struct CompileInfo {
} g_compileInfo;

gert::StorageShape MakeShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(0);
    shape.MutableStorageShape().SetDimNum(0);
    for (const int64_t dimension : dimensions) {
        shape.MutableOriginShape().AppendDim(dimension);
        shape.MutableStorageShape().AppendDim(dimension);
    }
    return shape;
}

struct TestParam {
    const char* name;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> targetShape;
    std::vector<int64_t> varShape;
    std::vector<int64_t> lossShape;
    ge::DataType inputDtype;
    ge::DataType targetDtype;
    ge::DataType varDtype;
    ge::DataType lossDtype;
    bool full;
    float eps;
    std::string reduction;
    uint64_t cores;
    uint64_t ubBytes;
    bool success;
};

const TestParam kCases[] = {
    {"same_float_none",
     {2, 3},
     {2, 3},
     {2, 3},
     {2, 3},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     false,
     1e-6f,
     "none",
     4,
     262144,
     true},
    {"target_axis_var_scalar_sum",
     {2, 3, 4},
     {2, 1, 4},
     {},
     {1},
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     ge::DT_FLOAT16,
     true,
     1e-5f,
     "sum",
     8,
     262144,
     true},
    {"var_last_one_mean",
     {2, 3, 4},
     {2, 3, 4},
     {2, 3, 1},
     {1},
     ge::DT_BF16,
     ge::DT_BF16,
     ge::DT_BF16,
     ge::DT_BF16,
     false,
     1e-6f,
     "mean",
     8,
     262144,
     true},
    {"var_missing_last_large",
     {2, 100000},
     {2, 100000},
     {2},
     {1},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     false,
     1e-6f,
     "sum",
     8,
     262144,
     true},
    {"empty_none",
     {0, 3},
     {0, 3},
     {},
     {0, 3},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     false,
     1e-6f,
     "none",
     64,
     262144,
     true},
    {"empty_mean",
     {0, 3},
     {0, 3},
     {},
     {1},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     false,
     1e-6f,
     "mean",
     64,
     262144,
     true},
    {"bad_target_two_axes",
     {2, 3, 4},
     {1, 1, 4},
     {},
     {2, 3, 4},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     false,
     1e-6f,
     "none",
     8,
     262144,
     false},
    {"bad_var_shape",
     {2, 3, 4},
     {2, 3, 4},
     {2, 4},
     {2, 3, 4},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     false,
     1e-6f,
     "none",
     8,
     262144,
     false},
    {"dtype_mismatch",
     {8},
     {8},
     {8},
     {8},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT16,
     ge::DT_FLOAT,
     false,
     1e-6f,
     "none",
     8,
     262144,
     false},
    {"bad_output",
     {8},
     {8},
     {8},
     {8},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     false,
     1e-6f,
     "sum",
     8,
     262144,
     false},
    {"bad_reduction",
     {8},
     {8},
     {8},
     {8},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     false,
     1e-6f,
     "bad",
     8,
     262144,
     false},
    {"zero_eps",
     {8},
     {8},
     {8},
     {8},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     false,
     0.0f,
     "none",
     8,
     262144,
     false},
    {"non_finite_eps",
     {8},
     {8},
     {8},
     {8},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     false,
     std::numeric_limits<float>::infinity(),
     "none",
     8,
     262144,
     false},
    {"small_ub",
     {8},
     {8},
     {8},
     {8},
     ge::DT_BF16,
     ge::DT_BF16,
     ge::DT_BF16,
     ge::DT_BF16,
     false,
     1e-6f,
     "none",
     8,
     16,
     false},
};

gert::TilingContextPara BuildContext(const TestParam& param)
{
    const gert::StorageShape input = MakeShape(param.inputShape);
    const gert::StorageShape target = MakeShape(param.targetShape);
    const gert::StorageShape var = MakeShape(param.varShape);
    const gert::StorageShape loss = MakeShape(param.lossShape);
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {{input, param.inputDtype, ge::FORMAT_ND},
                                                                      {target, param.targetDtype, ge::FORMAT_ND},
                                                                      {var, param.varDtype, ge::FORMAT_ND}};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {{loss, param.lossDtype, ge::FORMAT_ND}};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"full", Ops::NN::AnyValue::CreateFrom<bool>(param.full)},
        {"eps", Ops::NN::AnyValue::CreateFrom<float>(param.eps)},
        {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(param.reduction)}};
    return gert::TilingContextPara("GaussianNllLoss", inputs, outputs, attrs, &g_compileInfo, param.cores,
                                   param.ubBytes, 4096);
}

uint64_t ShapeSize(const std::vector<int64_t>& shape)
{
    uint64_t size = 1;
    for (const int64_t dimension : shape) {
        size *= static_cast<uint64_t>(dimension);
    }
    return size;
}

uint64_t ExpectedTilingKey(const std::string& reduction)
{
    if (reduction == "sum") {
        return GET_TPL_TILING_KEY(GAUSSIAN_NLL_LOSS_REDUCTION_SUM);
    }
    if (reduction == "mean") {
        return GET_TPL_TILING_KEY(GAUSSIAN_NLL_LOSS_REDUCTION_MEAN);
    }
    return GET_TPL_TILING_KEY(GAUSSIAN_NLL_LOSS_REDUCTION_NONE);
}

class GaussianNllLossTilingTest : public testing::TestWithParam<TestParam> {};

TEST_P(GaussianNllLossTilingTest, Contract)
{
    const TestParam& param = GetParam();
    auto context = BuildContext(param);
    TilingInfo info;
    const bool ok = ExecuteTiling(context, info);
    if (!param.success) {
        EXPECT_FALSE(ok);
        return;
    }
    ASSERT_TRUE(ok);
    EXPECT_EQ(info.tilingKey, ExpectedTilingKey(param.reduction));
    ASSERT_EQ(info.tilingDataSize, sizeof(GaussianNllLossTilingData));
    const auto* data = reinterpret_cast<const GaussianNllLossTilingData*>(info.tilingData.get());
    const uint64_t total = ShapeSize(param.inputShape);
    const uint64_t blocks = total == 0 ? 1 : std::min(total, param.cores);
    EXPECT_EQ(info.blockNum, blocks);
    EXPECT_EQ(data->blockNum, blocks);
    const uint32_t elementSizeBytes = param.inputDtype == ge::DT_FLOAT ? 4 : 2;
    EXPECT_EQ(static_cast<uint64_t>(data->tileDataNum) * elementSizeBytes % 32, 0);
    uint32_t logMaxLiveNodeCount = 0;
    uint32_t logExtraBufferBytes = 0;
    AscendC::GetLogTmpBufferFactorSize(sizeof(float), logMaxLiveNodeCount, logExtraBufferBytes);
    const uint64_t queuedBytesPerElement = 4 * 2 * elementSizeBytes;
    const uint64_t explicitFloatBufferCount = 2 + (param.inputDtype == ge::DT_FLOAT ? 0 : 3);
    const uint64_t bytesPerElement = queuedBytesPerElement +
                                     (explicitFloatBufferCount + logMaxLiveNodeCount) * sizeof(float);
    const uint64_t reductionScratchBytes = param.reduction == "none" ? 0 : (1 + param.cores) * 8 * sizeof(float);
    EXPECT_LE(static_cast<uint64_t>(data->tileDataNum) * bytesPerElement + reductionScratchBytes + logExtraBufferBytes,
              param.ubBytes);
    EXPECT_FLOAT_EQ(data->eps, param.eps);
    EXPECT_FLOAT_EQ(data->fullConstant, param.full ? 0.91893853320467274178f : 0.0f);
    if (param.reduction == "mean") {
        if (total == 0) {
            EXPECT_TRUE(std::isnan(data->meanScale));
        } else {
            EXPECT_FLOAT_EQ(data->meanScale, 1.0f / static_cast<float>(total));
        }
    }
    EXPECT_EQ(static_cast<uint64_t>(data->tailBlockNum) * data->bigCoreDataNum +
                  (blocks - data->tailBlockNum) * data->smallCoreDataNum,
              total);
    if (param.targetShape != param.inputShape) {
        EXPECT_EQ(data->targetBroadcastMode, 1);
    }
    if (param.varShape.empty()) {
        EXPECT_EQ(data->varBroadcastMode, 2);
    } else if (param.varShape != param.inputShape) {
        EXPECT_EQ(data->varBroadcastMode, 1);
    }
    if (param.reduction == "none" || blocks == 1) {
        ASSERT_EQ(info.workspaceSizes, std::vector<int64_t>({0}));
    } else {
        ASSERT_EQ(info.workspaceSizes.size(), 1);
        EXPECT_GT(info.workspaceSizes[0], 16 * 1024 * 1024);
    }
}

INSTANTIATE_TEST_SUITE_P(GaussianNllLossCases, GaussianNllLossTilingTest, testing::ValuesIn(kCases),
                         [](const testing::TestParamInfo<TestParam>& info) { return info.param.name; });
} // namespace
