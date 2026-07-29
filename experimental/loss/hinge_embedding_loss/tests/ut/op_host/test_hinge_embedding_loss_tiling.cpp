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
#include <limits>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_kernel/hinge_embedding_loss_tiling_data.h"
#include "../../../op_kernel/hinge_embedding_loss_tiling_key.h"

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
    std::vector<int64_t> outputShape;
    ge::DataType inputDtype;
    ge::DataType targetDtype;
    ge::DataType outputDtype;
    float margin;
    std::string reduction;
    uint64_t cores;
    uint64_t ub;
    bool success;
};

const TestParam kCases[] = {
    {"none_float", {10}, {10}, {10}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 1.0f, "none", 4, 262144, true},
    {"sum_half", {64}, {64}, {1}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, 0.5f, "sum", 8, 262144, true},
    {"mean_bfloat16_large",
     {100000},
     {100000},
     {1},
     ge::DT_BF16,
     ge::DT_BF16,
     ge::DT_BF16,
     2.0f,
     "mean",
     8,
     262144,
     true},
    {"empty_none", {0}, {0}, {0}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 1.0f, "none", 64, 262144, true},
    {"shape_mismatch",
     {2, 3},
     {2, 4},
     {2, 3},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     1.0f,
     "none",
     64,
     262144,
     false},
    {"dtype_mismatch", {8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, 1.0f, "none", 64, 262144, false},
    {"bad_output", {8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 1.0f, "sum", 64, 262144, false},
    {"bad_reduction", {8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 1.0f, "bad", 64, 262144, false},
    {"non_finite_margin",
     {8},
     {8},
     {8},
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     ge::DT_FLOAT,
     std::numeric_limits<float>::infinity(),
     "none",
     64,
     262144,
     false},
    {"small_ub", {8}, {8}, {8}, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, 1.0f, "none", 64, 16, false},
};

gert::TilingContextPara BuildContext(const TestParam& param)
{
    const gert::StorageShape input = MakeShape(param.inputShape);
    const gert::StorageShape target = MakeShape(param.targetShape);
    const gert::StorageShape output = MakeShape(param.outputShape);
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {{input, param.inputDtype, ge::FORMAT_ND},
                                                                      {target, param.targetDtype, ge::FORMAT_ND}};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {{output, param.outputDtype, ge::FORMAT_ND}};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"margin", Ops::NN::AnyValue::CreateFrom<float>(param.margin)},
        {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(param.reduction)}};
    return gert::TilingContextPara("HingeEmbeddingLoss", inputs, outputs, attrs, &g_compileInfo, param.cores, param.ub,
                                   4096);
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
        return GET_TPL_TILING_KEY(HINGE_EMBEDDING_LOSS_REDUCTION_SUM);
    }
    if (reduction == "mean") {
        return GET_TPL_TILING_KEY(HINGE_EMBEDDING_LOSS_REDUCTION_MEAN);
    }
    return GET_TPL_TILING_KEY(HINGE_EMBEDDING_LOSS_REDUCTION_NONE);
}

class HingeEmbeddingLossTilingTest : public testing::TestWithParam<TestParam> {};

TEST_P(HingeEmbeddingLossTilingTest, Contract)
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
    ASSERT_EQ(info.tilingDataSize, sizeof(HingeEmbeddingLossTilingData));
    const auto* data = reinterpret_cast<const HingeEmbeddingLossTilingData*>(info.tilingData.get());
    const uint64_t total = ShapeSize(param.inputShape);
    const uint64_t blocks = total == 0 ? 1 : std::min(total, param.cores);
    EXPECT_EQ(info.blockNum, blocks);
    EXPECT_EQ(data->blockNum, blocks);
    const uint32_t elementSizeBytes = param.inputDtype == ge::DT_FLOAT ? 4 : 2;
    EXPECT_EQ(static_cast<uint64_t>(data->tileDataNum) * elementSizeBytes % 32, 0);
    EXPECT_FLOAT_EQ(data->margin, param.margin);
    EXPECT_EQ(static_cast<uint64_t>(data->tailBlockNum) * data->bigCoreDataNum +
                  (blocks - data->tailBlockNum) * data->smallCoreDataNum,
              total);
    if (param.reduction == "none" || blocks == 1) {
        ASSERT_EQ(info.workspaceSizes, std::vector<int64_t>({0}));
    } else {
        ASSERT_EQ(info.workspaceSizes.size(), 1);
        EXPECT_GT(info.workspaceSizes[0], 16 * 1024 * 1024);
    }
}

INSTANTIATE_TEST_SUITE_P(HingeEmbeddingLossCases, HingeEmbeddingLossTilingTest, testing::ValuesIn(kCases),
                         [](const testing::TestParamInfo<TestParam>& info) { return info.param.name; });
} // namespace
