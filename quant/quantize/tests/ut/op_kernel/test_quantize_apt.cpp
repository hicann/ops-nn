/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quantize.cpp
 * \brief quantize kernel unit test
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/quantize_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void quantize(GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y,
                                               GM_ADDR workspace, GM_ADDR tiling);

class quantize_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "quantize_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "quantize_test TearDown\n" << endl; }
};

TEST_F(quantize_test, test_case_per_tensor_no_offset_fp16_to_int8)
{
    size_t inputXSize = 128 * 256 * sizeof(half);
    size_t outputYSize = 128 * 256 * sizeof(int8_t);
    size_t scaleSize = sizeof(half);
    size_t tiling_data_size = sizeof(QuantizeTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* zero_points = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    QuantizeTilingData* tilingData = reinterpret_cast<QuantizeTilingData*>(tiling);

    tilingData->numCore = 2;
    tilingData->blockAxis = 0;
    tilingData->ubAxis = 1;
    tilingData->dim0 = 128;
    tilingData->dim1 = 256;
    tilingData->dim2 = 1;
    tilingData->blockUnion = 1;
    tilingData->blockFactor = 64;
    tilingData->blockTailFactor = 64;
    tilingData->baseN = 16;
    tilingData->baseLen = 256;
    tilingData->hasZeroPoint = 0;
    tilingData->axis = 0;
    tilingData->roundMode = 0;
    tilingData->sqrtMode = 0;

    ICPU_SET_TILING_KEY(0);

    auto quantize_kernel = [](GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workSpace,
                              GM_ADDR tiling) {
        ::quantize<TPL_PER_TENSOR, TPL_NONE, TPL_DIV_MODE_DIV, TPL_ROUND_MODE_NONE, TPL_NO_SQRT_MODE>(
            x, scales, zero_points, y, workSpace, tiling);
    };
    ICPU_RUN_KF(quantize_kernel, blockDim, x, scales, zero_points, y, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(scales);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingData);
    free(path_);
}
