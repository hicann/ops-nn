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

#include "../../../op_kernel/fused_matmul_silu_tiling_data.h"

namespace {

struct FusedMatmulSiluCompileInfo {};

gert::TilingContextPara MakeTilingCase(int64_t m, int64_t k, int64_t n)
{
    static FusedMatmulSiluCompileInfo compileInfo;
    return gert::TilingContextPara("FusedMatmulSilu",
                                   {
                                       {{{m, k}, {m, k}}, ge::DT_BF16, ge::FORMAT_ND},
                                       {{{n, k}, {n, k}}, ge::DT_BF16, ge::FORMAT_ND},
                                       {{{n}, {n}}, ge::DT_BF16, ge::FORMAT_ND},
                                   },
                                   {
                                       {{{m, n}, {m, n}}, ge::DT_BF16, ge::FORMAT_ND},
                                   },
                                   {}, &compileInfo, 32, 262144, 4096);
}

gert::TilingContextPara MakeDynamicMTilingCase(int64_t m, int64_t k, int64_t n)
{
    static FusedMatmulSiluCompileInfo compileInfo;
    return gert::TilingContextPara("FusedMatmulSilu",
                                   {
                                       {{{-1, k}, {m, k}}, ge::DT_BF16, ge::FORMAT_ND},
                                       {{{n, k}, {n, k}}, ge::DT_BF16, ge::FORMAT_ND},
                                       {{{n}, {n}}, ge::DT_BF16, ge::FORMAT_ND},
                                   },
                                   {
                                       {{{-1, n}, {m, n}}, ge::DT_BF16, ge::FORMAT_ND},
                                   },
                                   {}, &compileInfo, 32, 262144, 4096);
}

} // namespace

class FusedMatmulSiluTilingTest : public testing::Test {};

TEST_F(FusedMatmulSiluTilingTest, tiling_success_reference_shape)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeTilingCase(2, 256, 4096), tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(FusedMatmulSiluTilingData));

    auto* tiling = reinterpret_cast<FusedMatmulSiluTilingData*>(tilingInfo.tilingData.get());
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->params.m, 2);
    EXPECT_EQ(tiling->params.n, 4096);
    EXPECT_EQ(tiling->params.k, 256);
    EXPECT_EQ(tiling->params.usedCoreNum, tiling->matmulTiling.usedCoreNum);
    EXPECT_GT(tiling->params.usedCoreNum, 1);
    EXPECT_EQ(tilingInfo.blockNum, tiling->params.usedCoreNum);
    // The test platform has 256 KiB UB. The post-process should use available UB rather
    // than retain the previous 1024-element cap.
    EXPECT_GT(tiling->params.vectorTileElems, 1024);
}

TEST_F(FusedMatmulSiluTilingTest, tiling_success_dynamic_m_origin_shape)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(MakeDynamicMTilingCase(2, 256, 4096), tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(FusedMatmulSiluTilingData));

    auto* tiling = reinterpret_cast<FusedMatmulSiluTilingData*>(tilingInfo.tilingData.get());
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->params.m, 2);
    EXPECT_EQ(tiling->params.n, 4096);
    EXPECT_EQ(tiling->params.k, 256);
}

TEST_F(FusedMatmulSiluTilingTest, tiling_failed_unaligned_k)
{
    ExecuteTestCase(MakeTilingCase(2, 255, 4096), ge::GRAPH_FAILED);
}

TEST_F(FusedMatmulSiluTilingTest, tiling_failed_unsupported_k)
{
    ExecuteTestCase(MakeTilingCase(2, 8192, 4096), ge::GRAPH_FAILED);
}
