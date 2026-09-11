/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "index/kth_value/op_host/arch35/kth_value_tiling_common.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/kth_value_tiling_data.h"

using namespace std;
using namespace ge;

class KthValueTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

struct KthValueCompileInfo {
    int32_t coreNum = 64;
};

namespace {
constexpr size_t WORK_SPACE_SIZE = 16777216;
KthValueCompileInfo g_compileInfo = {64};

gert::TilingContextPara MakeKthValueTilingContext(const gert::StorageShape& xShape,
                                                  const gert::StorageShape& valuesShape,
                                                  const gert::StorageShape& indicesShape, ge::DataType xDtype,
                                                  int64_t k, int64_t dim = -1)
{
    return gert::TilingContextPara(
        "KthValue",
        {
            {xShape, xDtype, ge::FORMAT_ND},
        },
        {
            {valuesShape, xDtype, ge::FORMAT_ND},
            {indicesShape, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("k", Ops::NN::AnyValue::CreateFrom<int64_t>(k)),
            gert::TilingContextPara::OpAttr("dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)),
        },
        &g_compileInfo);
}
} // namespace

TEST_F(KthValueTilingTest, test_kthvalue_merge_sort_fp32_2x1024)
{
    auto tilingContextPara = MakeKthValueTilingContext({{2, 1024}, {2, 1024}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}},
                                                       ge::DT_FLOAT, 5);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 256);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_axis_one_copy_fp32_100x1)
{
    auto tilingContextPara = MakeKthValueTilingContext({{100, 1}, {100, 1}}, {{100, 1}, {100, 1}}, {{100, 1}, {100, 1}},
                                                       ge::DT_FLOAT, 1, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 263);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_merge_sort_bf16_8x512)
{
    auto tilingContextPara = MakeKthValueTilingContext({{8, 512}, {8, 512}}, {{8, 1}, {8, 1}}, {{8, 1}, {8, 1}},
                                                       ge::DT_BF16, 100);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 256);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_merge_sort_fp32_3d_8x16x1024)
{
    auto tilingContextPara = MakeKthValueTilingContext({{8, 16, 1024}, {8, 16, 1024}}, {{8, 16, 1}, {8, 16, 1}},
                                                       {{8, 16, 1}, {8, 16, 1}}, ge::DT_FLOAT, 500, 2);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 256);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_axis_one_copy_int32_64x1)
{
    auto tilingContextPara = MakeKthValueTilingContext({{64, 1}, {64, 1}}, {{64, 1}, {64, 1}}, {{64, 1}, {64, 1}},
                                                       ge::DT_INT32, 1, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 263);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_axis_one_copy_allows_unsorted_dim_over_uint32)
{
    int64_t largeBatch = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    auto tilingContextPara = MakeKthValueTilingContext({{largeBatch, 1}, {largeBatch, 1}},
                                                       {{largeBatch, 1}, {largeBatch, 1}},
                                                       {{largeBatch, 1}, {largeBatch, 1}}, ge::DT_FLOAT, 1, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 263);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(KthValueTilingData));
    const auto* tilingData = reinterpret_cast<const KthValueTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->unsortedDimNum, largeBatch);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_merge_intra_core_allows_unsorted_dim_over_uint32)
{
    int64_t largeBatch = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    auto tilingContextPara = MakeKthValueTilingContext({{largeBatch, 4097}, {largeBatch, 4097}},
                                                       {{largeBatch, 1}, {largeBatch, 1}},
                                                       {{largeBatch, 1}, {largeBatch, 1}}, ge::DT_FLOAT, 64);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 260);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(KthValueTilingData));
    const auto* tilingData = reinterpret_cast<const KthValueTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->unsortedDimNum, largeBatch);
    EXPECT_GT(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_merge_sort_fp32_1d_512)
{
    auto tilingContextPara = MakeKthValueTilingContext({{512}, {512}}, {{1}, {1}}, {{1}, {1}}, ge::DT_FLOAT, 256);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 256);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_radix_select_int32_large_axis_mid_many_rows)
{
    auto tilingContextPara = MakeKthValueTilingContext({{16, 65536}, {16, 65536}}, {{16, 1}, {16, 1}},
                                                       {{16, 1}, {16, 1}}, ge::DT_INT32, 32768);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 267);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(KthValueTilingData));
    const auto* tilingData = reinterpret_cast<const KthValueTilingData*>(tilingInfo.tilingData.get());
    EXPECT_GT(tilingData->numTileDataSize, 0U);
    EXPECT_EQ(tilingData->unsortedDimParallel, 16U);
    EXPECT_GT(tilingData->lastDimNeedCore, 1U);
    EXPECT_EQ(tilingInfo.blockNum, tilingData->unsortedDimParallel * tilingData->lastDimNeedCore);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    size_t histogramWorkspace = static_cast<size_t>(tilingInfo.blockNum) * 256UL * sizeof(uint64_t);
    size_t groupStateWorkspace = static_cast<size_t>(tilingData->unsortedDimParallel) * 8UL * sizeof(uint64_t);
    EXPECT_GE(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE + histogramWorkspace + groupStateWorkspace);
    EXPECT_LT(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE + 2UL * histogramWorkspace);
}

TEST_F(KthValueTilingTest, test_kthvalue_radix_select_int64_small_axis_many_rows_targets)
{
    std::vector<gert::StorageShape> xShapes = {
        {{33, 1, 1, 1, 1, 1, 1, 7154}, {33, 1, 1, 1, 1, 1, 1, 7154}},
        {{45, 8670}, {45, 8670}},
        {{59, 6931}, {59, 6931}},
        {{42, 1, 1, 1, 7496}, {42, 1, 1, 1, 7496}},
        {{56, 1, 9479}, {56, 1, 9479}},
    };
    std::vector<gert::StorageShape> outputShapes = {
        {{33, 1, 1, 1, 1, 1, 1, 1}, {33, 1, 1, 1, 1, 1, 1, 1}},
        {{45, 1}, {45, 1}},
        {{59, 1}, {59, 1}},
        {{42, 1, 1, 1, 1}, {42, 1, 1, 1, 1}},
        {{56, 1, 1}, {56, 1, 1}},
    };
    std::vector<int64_t> kValues = {3264, 5033, 4147, 1663, 2441};

    for (size_t i = 0; i < xShapes.size(); ++i) {
        auto tilingContextPara = MakeKthValueTilingContext(xShapes[i], outputShapes[i], outputShapes[i], ge::DT_INT64,
                                                           kValues[i]);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        EXPECT_EQ(tilingInfo.tilingKey, 267);
    }
}

TEST_F(KthValueTilingTest, test_kthvalue_int64_small_axis_short_rank_select_16936x23)
{
    auto tilingContextPara = MakeKthValueTilingContext({{16936, 23}, {16936, 23}}, {{16936, 1}, {16936, 1}},
                                                       {{16936, 1}, {16936, 1}}, ge::DT_INT64, 18);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 268);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(KthValueTilingData));
    const auto* tilingData = reinterpret_cast<const KthValueTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->numTileDataSize, 23U);
    EXPECT_EQ(tilingData->unsortedDimParallel, 64U);
    EXPECT_GT(tilingData->keyParams0, 0U);
    EXPECT_GT(tilingData->keyParams1, 0U);
    EXPECT_EQ(tilingData->keyParams2, 6U);
    EXPECT_EQ(tilingData->kthIndex, 17);
    EXPECT_EQ(tilingData->unsortedDimNum, 16936);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST_F(KthValueTilingTest, test_kthvalue_twostage_stays_on_original_route)
{
    auto tilingContextPara = MakeKthValueTilingContext({{4096, 9}, {4096, 9}}, {{4096, 1}, {4096, 1}},
                                                       {{4096, 1}, {4096, 1}}, ge::DT_FLOAT16, 5, 1);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 262);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(KthValueTilingData));
    const auto* tilingData = reinterpret_cast<const KthValueTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->numTileDataSize, 9);
    EXPECT_EQ(tilingData->unsortedDimParallel, 64);
    EXPECT_EQ(tilingData->sortLoopTimes, 64);
    EXPECT_EQ(tilingData->keyParams0, 64);
    EXPECT_EQ(tilingData->keyParams1, 64);
    EXPECT_EQ(tilingData->keyParams2, 1);
    EXPECT_EQ(tilingData->keyParams4, 0);
    EXPECT_EQ(tilingData->innerChunk, 0);
}

TEST_F(KthValueTilingTest, test_kthvalue_radix_counter_range_boundary)
{
    constexpr int64_t radixUint32ValueMax = 0x3fffffff;

    EXPECT_TRUE(optiling::IsRadixUint32CounterRange(radixUint32ValueMax));
    EXPECT_FALSE(optiling::IsRadixUint32CounterRange(radixUint32ValueMax + 1));
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_invalid_dtype)
{
    auto tilingContextPara = MakeKthValueTilingContext({{2, 100}, {2, 100}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}},
                                                       ge::DT_BOOL, 5);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_radix_select_rejects_invalid_dtype)
{
    auto tilingContextPara = MakeKthValueTilingContext({{2, 100000}, {2, 100000}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}},
                                                       ge::DT_BOOL, 5);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_empty_shape)
{
    auto tilingContextPara = MakeKthValueTilingContext({{0, 100}, {0, 100}}, {{0, 1}, {0, 1}}, {{0, 1}, {0, 1}},
                                                       ge::DT_FLOAT, 5);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_k_zero)
{
    auto tilingContextPara = MakeKthValueTilingContext({{2, 100}, {2, 100}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}},
                                                       ge::DT_FLOAT, 0);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_k_exceeds_axis)
{
    auto tilingContextPara = MakeKthValueTilingContext({{2, 10}, {2, 10}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}},
                                                       ge::DT_FLOAT, 11);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_dim_out_of_range)
{
    auto tilingContextPara = MakeKthValueTilingContext({{2, 100}, {2, 100}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}},
                                                       ge::DT_FLOAT, 5, 3);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_negative_dim_out_of_range)
{
    auto tilingContextPara = MakeKthValueTilingContext({{2, 100}, {2, 100}}, {{2, 1}, {2, 1}}, {{2, 1}, {2, 1}},
                                                       ge::DT_FLOAT, 5, -3);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_values_dtype_mismatch)
{
    gert::TilingContextPara tilingContextPara(
        "KthValue",
        {
            gert::TilingContextPara::TensorDescription({{2, 100}, {2, 100}}, ge::DT_FLOAT, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::TensorDescription({{2, 1}, {2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND),
            gert::TilingContextPara::TensorDescription({{2, 1}, {2, 1}}, ge::DT_INT64, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::OpAttr("k", Ops::NN::AnyValue::CreateFrom<int64_t>(5)),
            gert::TilingContextPara::OpAttr("dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)),
        },
        &g_compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(KthValueTilingTest, test_kthvalue_rejects_indices_dtype_not_int64)
{
    gert::TilingContextPara tilingContextPara(
        "KthValue",
        {
            gert::TilingContextPara::TensorDescription({{2, 100}, {2, 100}}, ge::DT_FLOAT, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::TensorDescription({{2, 1}, {2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND),
            gert::TilingContextPara::TensorDescription({{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND),
        },
        {
            gert::TilingContextPara::OpAttr("k", Ops::NN::AnyValue::CreateFrom<int64_t>(5)),
            gert::TilingContextPara::OpAttr("dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)),
        },
        &g_compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
