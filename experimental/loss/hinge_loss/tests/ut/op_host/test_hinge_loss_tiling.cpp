/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include <vector>
#include "tiling_case_executor.h"
#include "../../../op_kernel/hinge_loss_tiling_data.h"

namespace {
constexpr int64_t TILE_DATA_NUM = 4360;
constexpr uint64_t CORE_NUM = 64;
constexpr uint64_t UB_SIZE = 262144;

TilingInfo RunFloat32Tiling(std::initializer_list<int64_t> shape)
{
    struct HingeLossCompileInfo {
    } info;
    gert::TilingContextPara context(
        "HingeLoss", {{{shape, shape}, ge::DT_FLOAT, ge::FORMAT_ND}, {{shape, shape}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{shape, shape}, ge::DT_FLOAT, ge::FORMAT_ND}}, {}, &info, CORE_NUM, UB_SIZE, 4096);
    TilingInfo result;
    EXPECT_TRUE(ExecuteTiling(context, result));
    return result;
}
} // namespace

TEST(HingeLossTiling, float32_sets_zero_workspace)
{
    TilingInfo result = RunFloat32Tiling({256, 32});
    ASSERT_EQ(result.workspaceSizes, std::vector<int64_t>({0}));
    ASSERT_EQ(result.tilingDataSize, sizeof(HingeLossTilingData));
}

TEST(HingeLossTiling, uneven_multicore_split_uses_real_element_count)
{
    constexpr int64_t elementNum = CORE_NUM * TILE_DATA_NUM + 7;
    TilingInfo result = RunFloat32Tiling({elementNum});
    ASSERT_EQ(result.blockNum, CORE_NUM);
    ASSERT_EQ(result.tilingDataSize, sizeof(HingeLossTilingData));

    const auto* data = reinterpret_cast<const HingeLossTilingData*>(result.tilingData.get());
    EXPECT_EQ(data->tileDataNum, TILE_DATA_NUM);
    EXPECT_EQ(data->smallCoreDataNum, TILE_DATA_NUM);
    EXPECT_EQ(data->bigCoreDataNum, TILE_DATA_NUM + 1);
    EXPECT_EQ(data->tailBlockNum, 7);
    EXPECT_EQ(data->finalSmallTileNum, 1);
    EXPECT_EQ(data->smallTailDataNum, TILE_DATA_NUM);
    EXPECT_EQ(data->finalBigTileNum, 2);
    EXPECT_EQ(data->bigTailDataNum, 1);
    EXPECT_EQ(static_cast<int64_t>(data->tailBlockNum) * data->bigCoreDataNum +
                  (static_cast<int64_t>(result.blockNum) - data->tailBlockNum) * data->smallCoreDataNum,
              elementNum);
}

TEST(HingeLossTiling, large_tensor_assigns_multiple_tiles_per_core)
{
    constexpr int64_t elementNum = CORE_NUM * TILE_DATA_NUM * 3 + 17;
    TilingInfo result = RunFloat32Tiling({elementNum});
    ASSERT_EQ(result.blockNum, CORE_NUM);

    const auto* data = reinterpret_cast<const HingeLossTilingData*>(result.tilingData.get());
    EXPECT_EQ(data->smallCoreDataNum, TILE_DATA_NUM * 3);
    EXPECT_EQ(data->bigCoreDataNum, TILE_DATA_NUM * 3 + 1);
    EXPECT_EQ(data->tailBlockNum, 17);
    EXPECT_EQ(data->finalSmallTileNum, 3);
    EXPECT_EQ(data->smallTailDataNum, TILE_DATA_NUM);
    EXPECT_EQ(data->finalBigTileNum, 4);
    EXPECT_EQ(data->bigTailDataNum, 1);
    EXPECT_EQ(static_cast<int64_t>(data->tailBlockNum) * data->bigCoreDataNum +
                  (static_cast<int64_t>(result.blockNum) - data->tailBlockNum) * data->smallCoreDataNum,
              elementNum);
}
