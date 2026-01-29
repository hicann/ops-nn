/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>

#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include <iostream>
#include <sstream>
#include <string>

#include "data_utils.h"
#include "string.h"
#include "tikicpulib.h"
#endif

#include <cstdint>

using namespace std;

template <
    int SOC_VERSION_TYPE, int SUB_SOC_VERSION_TYPE, int TEMPLATE_CUSTOM, int LEVEL1_QUANT_TYPE, int LEVEL0_QUANT_TYPE,
    bool TRANS_A, bool TRANS_B, bool HAS_BIAS, bool IS_WEIGHT_NZ>
__global__ __aicore__ void dual_level_quant_batch_matmul(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Level0Scale, GM_ADDR x1Level1Scale, GM_ADDR x2Level0Scale, GM_ADDR x2Level1Scale,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class TestDualLevelQuantBatchMatmul : public testing::Test {};

TEST_F(TestDualLevelQuantBatchMatmul, kernelTest)
{
    auto wrapper = [](GM_ADDR x1, GM_ADDR x2, GM_ADDR x1_level0_scale, GM_ADDR x1_level1_scale, GM_ADDR x2_level0_scale,
                      GM_ADDR x2_level1_scale, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::dual_level_quant_batch_matmul<
            DLQBMM_SOC_RESERVERD, DLQBMM_DEFAULT, DLQBMM_TEMPLATE_CUBEBOUND, DLQBMM_QUANT_TYPE_MX,
            DLQBMM_QUANT_TYPE_PER_GROUP, false, true, true, true>(
            x1, x2, x1_level0_scale, x1_level1_scale, x2_level0_scale, x2_level1_scale, bias, y, workspace, tiling);
    };

    ICPU_RUN_KF(wrapper, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}