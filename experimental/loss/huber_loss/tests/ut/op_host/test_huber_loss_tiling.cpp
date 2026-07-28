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
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_kernel/huber_loss_tiling_data.h"

namespace {
constexpr uint64_t kDefaultCoreNum = 64;
constexpr uint64_t kDefaultUbSize = 262144;

struct HuberLossCompileInfo {
} g_compileInfo;

struct HuberLossTestParam {
    const char* caseName;
    std::vector<int64_t> predictionShape;
    std::vector<int64_t> targetShape;
    std::vector<int64_t> outputShape;
    ge::DataType predictionDtype;
    ge::DataType targetDtype;
    ge::DataType outputDtype;
    float delta;
    uint64_t coreNum;
    uint64_t ubSize;
    bool success;
};

const HuberLossTestParam kTestCases[] = {
    {"float32_single", {1}, {1}, {1}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 1.0f, 64, 262144, true},
    {"float16_uneven", {10}, {10}, {10}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, 0.5f, 4, 262144, true},
    {"bfloat16_large", {100000}, {100000}, {100000}, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, 2.0f, 8, 262144, true},
    {"float32_empty", {0}, {0}, {0}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 1.0f, 64, 262144, true},
    {"shape_mismatch", {2, 3}, {2, 4}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 1.0f, 64, 262144, false},
    {"output_shape_mismatch", {2, 3}, {2, 3}, {6}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 1.0f, 64, 262144, false},
    {"input_dtype_mismatch", {8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, 1.0f, 64, 262144, false},
    {"output_dtype_mismatch", {8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, 1.0f, 64, 262144, false},
    {"unsupported_dtype", {8}, {8}, {8}, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, 1.0f, 64, 262144, false},
    {"zero_delta", {8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 64, 262144, false},
    {"negative_delta", {8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, -1.0f, 64, 262144, false},
    {"insufficient_ub", {8}, {8}, {8}, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, 1.0f, 64, 1024, false},
};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dimensions)
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

gert::TilingContextPara BuildContext(const HuberLossTestParam& param)
{
    gert::StorageShape predictionShape = MakeStorageShape(param.predictionShape);
    gert::StorageShape targetShape = MakeStorageShape(param.targetShape);
    gert::StorageShape outputShape = MakeStorageShape(param.outputShape);
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {predictionShape, param.predictionDtype, ge::FORMAT_ND}, {targetShape, param.targetDtype, ge::FORMAT_ND}};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {{outputShape, param.outputDtype, ge::FORMAT_ND}};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {{"delta", Ops::NN::AnyValue::CreateFrom<float>(param.delta)}};
    return gert::TilingContextPara("HuberLoss", inputs, outputs, attrs, &g_compileInfo, param.coreNum, param.ubSize,
                                   4096);
}

uint64_t ShapeSize(const std::vector<int64_t>& shape)
{
    uint64_t size = 1;
    for (const int64_t dim : shape) {
        size *= static_cast<uint64_t>(dim);
    }
    return size;
}

class HuberLossTilingTest : public testing::TestWithParam<HuberLossTestParam> {};

TEST_P(HuberLossTilingTest, ProducesLogicalTiling)
{
    const HuberLossTestParam& param = GetParam();
    auto context = BuildContext(param);
    TilingInfo info;
    const bool ok = ExecuteTiling(context, info);
    if (!param.success) {
        EXPECT_FALSE(ok);
        return;
    }

    ASSERT_TRUE(ok);
    EXPECT_EQ(info.tilingKey, 0);
    ASSERT_EQ(info.workspaceSizes, std::vector<int64_t>({0}));
    ASSERT_EQ(info.tilingDataSize, sizeof(HuberLossTilingData));
    const auto* data = reinterpret_cast<const HuberLossTilingData*>(info.tilingData.get());
    const uint64_t total = ShapeSize(param.predictionShape);
    const uint64_t expectedBlocks = total == 0 ? 1 : std::min(total, param.coreNum);

    EXPECT_EQ(info.blockNum, expectedBlocks);
    EXPECT_GT(data->tileDataNum, 0U);
    EXPECT_EQ(data->tileDataNum % 64U, 0U);
    EXPECT_FLOAT_EQ(data->delta, param.delta);
    EXPECT_EQ(static_cast<uint64_t>(data->tailBlockNum) * data->bigCoreDataNum +
                  (expectedBlocks - data->tailBlockNum) * data->smallCoreDataNum,
              total);
    EXPECT_EQ(data->finalSmallTileNum,
              data->smallCoreDataNum == 0 ? 0U : (data->smallCoreDataNum + data->tileDataNum - 1) / data->tileDataNum);
    EXPECT_EQ(data->finalBigTileNum,
              data->bigCoreDataNum == 0 ? 0U : (data->bigCoreDataNum + data->tileDataNum - 1) / data->tileDataNum);
    EXPECT_EQ(data->smallTailDataNum, data->smallCoreDataNum == 0 ?
                                          0U :
                                          data->smallCoreDataNum - (data->finalSmallTileNum - 1) * data->tileDataNum);
    EXPECT_EQ(data->bigTailDataNum,
              data->bigCoreDataNum == 0 ? 0U : data->bigCoreDataNum - (data->finalBigTileNum - 1) * data->tileDataNum);
}

TEST(HuberLossTilingTest, AlignsTileToVectorWidth)
{
    constexpr uint64_t kFloatBytesPerElement = 41;
    constexpr uint64_t kRawTileElements = 96;
    const HuberLossTestParam param = {"vector_width_alignment",
                                      {128},
                                      {128},
                                      {128},
                                      ge::DT_FLOAT,
                                      ge::DT_FLOAT,
                                      ge::DT_FLOAT,
                                      1.0f,
                                      1,
                                      kFloatBytesPerElement * kRawTileElements,
                                      true};
    auto context = BuildContext(param);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(HuberLossTilingData));
    const auto* data = reinterpret_cast<const HuberLossTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->tileDataNum, 64U);
}

INSTANTIATE_TEST_SUITE_P(HuberLossCases, HuberLossTilingTest, testing::ValuesIn(kTestCases),
                         [](const testing::TestParamInfo<HuberLossTestParam>& info) { return info.param.caseName; });
} // namespace
