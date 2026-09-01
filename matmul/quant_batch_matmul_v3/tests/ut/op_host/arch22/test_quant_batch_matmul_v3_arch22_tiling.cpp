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

static const std::vector<QuantBatchMatmulV3TilingTestParam> kCasesParams910B2 = GetParams("Ascend910B2");
static const std::vector<QuantBatchMatmulV3TilingTestParam> kCasesParams910B4 = GetParams("Ascend910B4");
static const std::vector<QuantBatchMatmulV3TilingTestParam> kCasesParams310P3 = GetParams("Ascend310P3");

INSTANTIATE_TEST_CASE_P(QUANTMM910B, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(kCasesParams910B2));
INSTANTIATE_TEST_CASE_P(QUANTMM910B4, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(kCasesParams910B4));
INSTANTIATE_TEST_CASE_P(QUANTMM310P, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(kCasesParams310P3));

TEST_F(TestQuantBatchMatmulV3Tiling, multiThread310P3)
{
    TestMultiThread(kCasesParams310P3.data(), kCasesParams310P3.size(), 3);
}
