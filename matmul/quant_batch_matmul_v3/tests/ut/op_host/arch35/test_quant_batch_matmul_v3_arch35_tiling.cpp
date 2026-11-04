/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../test_quant_batch_matmul_v3_tiling.h"
#include "../../../../op_host/mx_scale_nz_shape.h"
#include "../../../../op_host/op_tiling/quant_batch_matmul_v3_tiling_cache_key.h"

static const std::vector<QuantBatchMatmulV3TilingTestParam> kCasesParamsDav3510 = GetParams("Dav3510");

INSTANTIATE_TEST_CASE_P(QUANTMM_DAV_3510, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(kCasesParamsDav3510));

TEST_F(TestQuantBatchMatmulV3Tiling, multiThreadDav3510)
{
    TestMultiThread(kCasesParamsDav3510.data(), kCasesParamsDav3510.size(), 3);
}

TEST(QuantBatchMatmulScaleNzShape, CanonicalAxesAndPadding)
{
    const gert::Shape logical({3, 47, 2}); // K=130 -> Kg=3, N has a 16-element tail.
    const gert::Shape packed({3, 3, 16, 2});
    EXPECT_TRUE(QuantBatchMatmul::CheckMxScaleNzShape(logical, packed, 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({47, 3, 2}), packed, 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(logical, gert::Shape({3, 3, 16, 1}), 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(logical, gert::Shape({2, 3, 16, 2}), 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({2, 47, 2}), packed, 47, 130));
}

TEST(QuantBatchMatmulScaleNzShape, BatchRankAndEqualAxes)
{
    EXPECT_TRUE(
        QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({2, 3, 47, 2}), gert::Shape({2, 3, 3, 16, 2}), 47, 130));
    EXPECT_FALSE(
        QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({2, 3, 47, 2}), gert::Shape({1, 3, 3, 16, 2}), 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({3, 47}), gert::Shape({3, 3, 16}), 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({3, 47, 2}), gert::Shape({3, 3, 16, 2}), 0, 130));
    // Shape checks cannot infer the meaning of equal axes or detect an earlier wrong data permutation.
    EXPECT_TRUE(QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({3, 3, 2}), gert::Shape({1, 3, 16, 2}), 3, 130));
}

TEST(QuantBatchMatmulScaleNzShape, ScaleFormatSeparatesTilingCacheEntries)
{
    optiling::QuantBatchMatmulInfo params;
    params.Reset();
    Ops::NN::Optiling::AiCoreParams cores{};
    params.scaleFormat = ge::FORMAT_ND;
    const optiling::QuantBatchMatmulV3HashInput nd(params, cores);
    params.scaleFormat = ge::FORMAT_FRACTAL_NZ;
    const optiling::QuantBatchMatmulV3HashInput nz(params, cores);
    EXPECT_FALSE(nd == nz);
}
