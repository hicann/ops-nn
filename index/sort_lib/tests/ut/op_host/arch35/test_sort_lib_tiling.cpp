/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sort_lib_tiling.cpp
 * \brief sortlib tiling 对外接口 UT（看护 SortTilingCompute / IsInt32Safe 行为）
 */

#include <gtest/gtest.h>
#include "graph/types.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../../../op_host/arch35/sort_lib_tiling.h"

namespace {
// AscendC::Sort 接口仅在 DAV_3510(950/350) 可用；其它 arch 未测试，不校验结果，仅运行被测代码。
bool IsSortApiSupportedArch()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return false;
    }
    NpuArch arch = platform->GetCurNpuArch();
    return arch == NpuArch::DAV_3510;
}
} // namespace

// 空输入：直接返回固定配置，errCode 正常，单核语义，workspace 为 0
TEST(SortLibTilingTest, EmptyInput)
{
    SortLib::SortTilingResult r = SortLib::SortTilingCompute(0, 64, 253952, 4, 4, true, ge::DT_INT32);
    if (!IsSortApiSupportedArch()) {
        return;
    }
    EXPECT_EQ(r.errCode, SortLib::SORT_TILING_OK);
    EXPECT_EQ(r.totalElements, 0);
    EXPECT_EQ(r.isSingleCore, 1u);
    EXPECT_EQ(r.numTileData, 1u);
    EXPECT_EQ(r.tileCount, 1u);
    EXPECT_EQ(r.activeCores, 1u);
    EXPECT_EQ(r.coreNumNeed, 1u);
    EXPECT_EQ(r.tmpUbSize, 4096u);
    EXPECT_EQ(r.workspaceBytes, 0);
}

// UB 连 DCACHE 都不够：返回错误码，其余字段保持默认 0
TEST(SortLibTilingTest, UbLessThanDcache)
{
    SortLib::SortTilingResult r = SortLib::SortTilingCompute(1000, 64, 16 * 1024, 4, 4, true, ge::DT_INT32);
    if (!IsSortApiSupportedArch()) {
        return;
    }
    EXPECT_EQ(r.errCode, SortLib::SORT_TILING_ERR_UB_LESS_THAN_DCACHE);
    EXPECT_EQ(r.totalElements, 0);
    EXPECT_EQ(r.isSingleCore, 0u);
    EXPECT_EQ(r.numTileData, 0u);
    EXPECT_EQ(r.tileCount, 0u);
    EXPECT_EQ(r.activeCores, 0u);
    EXPECT_EQ(r.coreNumNeed, 0u);
    EXPECT_EQ(r.tmpUbSize, 0u);
    EXPECT_EQ(r.workspaceBytes, 0);
}

// UB 不足以容纳最小 tile（usableUb < 直方图固定开销）：返回错误码
TEST(SortLibTilingTest, UbInsufficient)
{
    SortLib::SortTilingResult r = SortLib::SortTilingCompute(1000000, 64, 33 * 1024, 4, 4, true, ge::DT_INT32);
    if (!IsSortApiSupportedArch()) {
        return;
    }
    EXPECT_EQ(r.errCode, SortLib::SORT_TILING_ERR_UB_INSUFFICIENT);
    EXPECT_EQ(r.totalElements, 0);
    EXPECT_EQ(r.isSingleCore, 0u);
    EXPECT_EQ(r.numTileData, 0u);
    EXPECT_EQ(r.tileCount, 0u);
    EXPECT_EQ(r.activeCores, 0u);
    EXPECT_EQ(r.coreNumNeed, 0u);
    EXPECT_EQ(r.tmpUbSize, 0u);
    EXPECT_EQ(r.workspaceBytes, 0);
}

// 小 N：命中单核快路径，numTileData 直接取输入 n（非对齐值），workspace 为 0
TEST(SortLibTilingTest, SmallNSingleCore)
{
    SortLib::SortTilingResult r = SortLib::SortTilingCompute(1000, 64, 253952, 4, 4, true, ge::DT_INT32);
    if (!IsSortApiSupportedArch()) {
        return;
    }
    EXPECT_EQ(r.errCode, SortLib::SORT_TILING_OK);
    EXPECT_EQ(r.totalElements, 1000);
    EXPECT_EQ(r.isSingleCore, 1u);
    EXPECT_EQ(r.numTileData, 1000u);
    EXPECT_EQ(r.tileCount, 1u);
    EXPECT_EQ(r.activeCores, 1u);
    EXPECT_EQ(r.coreNumNeed, 1u);
    EXPECT_EQ(r.workspaceBytes, 0);
}

// 大 N：多核路径，numTileData 对齐到 256，activeCores 在 (0, coreCount] 内
TEST(SortLibTilingTest, LargeNMultiCore)
{
    SortLib::SortTilingResult r = SortLib::SortTilingCompute(1000000, 64, 253952, 4, 4, true, ge::DT_INT32);
    if (!IsSortApiSupportedArch()) {
        return;
    }
    EXPECT_EQ(r.errCode, SortLib::SORT_TILING_OK);
    EXPECT_EQ(r.totalElements, 1000000);
    EXPECT_EQ(r.isSingleCore, 0u);
    EXPECT_GT(r.numTileData, 0u);
    EXPECT_EQ(r.numTileData % 256, 0u);
    EXPECT_GT(r.tileCount, 1u);
    EXPECT_GT(r.activeCores, 0u);
    EXPECT_LE(r.activeCores, 64u);
    EXPECT_EQ(r.coreNumNeed, r.activeCores);
    EXPECT_GT(r.workspaceBytes, 0);
}

// IsInt32Safe 边界：<= 2^30 用 32 位计数，> 2^30 用 64 位计数
TEST(SortLibTilingTest, IsInt32SafeBoundary)
{
    EXPECT_TRUE(SortLib::IsInt32Safe(1LL << 30));
    EXPECT_FALSE(SortLib::IsInt32Safe((1LL << 30) + 1));
}

// int64 索引（indexSize=8）：单核快路径，索引缓冲翻倍
TEST(SortLibTilingTest, Int64IndexSingleCore)
{
    SortLib::SortTilingResult r = SortLib::SortTilingCompute(1000, 64, 253952, 4, 8, true, ge::DT_INT32);
    if (!IsSortApiSupportedArch()) {
        return;
    }
    EXPECT_EQ(r.errCode, SortLib::SORT_TILING_OK);
    EXPECT_EQ(r.totalElements, 1000);
    EXPECT_EQ(r.isSingleCore, 1u);
    EXPECT_EQ(r.numTileData, 1000u);
    EXPECT_EQ(r.tileCount, 1u);
    EXPECT_EQ(r.activeCores, 1u);
    EXPECT_EQ(r.coreNumNeed, 1u);
    EXPECT_EQ(r.workspaceBytes, 0);
}

// int16 键（dtypeSize=2）：单核快路径
TEST(SortLibTilingTest, Int16KeySingleCore)
{
    SortLib::SortTilingResult r = SortLib::SortTilingCompute(1000, 64, 253952, 2, 4, true, ge::DT_INT16);
    if (!IsSortApiSupportedArch()) {
        return;
    }
    EXPECT_EQ(r.errCode, SortLib::SORT_TILING_OK);
    EXPECT_EQ(r.totalElements, 1000);
    EXPECT_EQ(r.isSingleCore, 1u);
    EXPECT_EQ(r.numTileData, 1000u);
    EXPECT_EQ(r.tileCount, 1u);
    EXPECT_EQ(r.activeCores, 1u);
    EXPECT_EQ(r.coreNumNeed, 1u);
    EXPECT_EQ(r.workspaceBytes, 0);
}

// int64 键（dtypeSize=8）：多核路径，numTileData 对齐到 256
TEST(SortLibTilingTest, Int64KeyMultiCore)
{
    SortLib::SortTilingResult r = SortLib::SortTilingCompute(1000000, 64, 253952, 8, 4, true, ge::DT_INT64);
    if (!IsSortApiSupportedArch()) {
        return;
    }
    EXPECT_EQ(r.errCode, SortLib::SORT_TILING_OK);
    EXPECT_EQ(r.totalElements, 1000000);
    EXPECT_EQ(r.isSingleCore, 0u);
    EXPECT_GT(r.numTileData, 0u);
    EXPECT_EQ(r.numTileData % 256, 0u);
    EXPECT_GT(r.tileCount, 1u);
    EXPECT_GT(r.activeCores, 0u);
    EXPECT_LE(r.activeCores, 64u);
    EXPECT_EQ(r.coreNumNeed, r.activeCores);
    EXPECT_GT(r.workspaceBytes, 0);
}

// 64 位计数（isInt32Safe=false → counterSize=8）：多核路径
TEST(SortLibTilingTest, LargeN64BitCounter)
{
    SortLib::SortTilingResult r = SortLib::SortTilingCompute(1000000, 64, 253952, 4, 4, false, ge::DT_INT32);
    if (!IsSortApiSupportedArch()) {
        return;
    }
    EXPECT_EQ(r.errCode, SortLib::SORT_TILING_OK);
    EXPECT_EQ(r.totalElements, 1000000);
    EXPECT_EQ(r.isSingleCore, 0u);
    EXPECT_GT(r.numTileData, 0u);
    EXPECT_EQ(r.numTileData % 256, 0u);
    EXPECT_GT(r.tileCount, 1u);
    EXPECT_GT(r.activeCores, 0u);
    EXPECT_LE(r.activeCores, 64u);
    EXPECT_EQ(r.coreNumNeed, r.activeCores);
    EXPECT_GT(r.workspaceBytes, 0);
}
