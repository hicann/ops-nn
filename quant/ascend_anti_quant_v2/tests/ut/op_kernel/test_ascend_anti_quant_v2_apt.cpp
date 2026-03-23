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
 * \file test_ascend_anti_quant_v2.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/ascend_anti_quant_v2_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void ascend_anti_quant_v2(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class ascend_anti_quant_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ascend_anti_quant_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ascend_anti_quant_v2_test TearDown\n" << endl;
    }
};

TEST_F(ascend_anti_quant_v2_test, test_case_100)
{
    size_t inputXSize = 128 * 512 * sizeof(hifloat8_t);
    size_t inputScaleSize = 512 * sizeof(float);
    size_t inputOffsetSize = 512 * sizeof(float);
    size_t outputYSize = 128 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(AscendAntiQuantV2TilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(inputScaleSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(inputOffsetSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    AscendAntiQuantV2TilingData* tilingDatafromBin = reinterpret_cast<AscendAntiQuantV2TilingData*>(tiling);

    tilingDatafromBin->numCore = 43;
    tilingDatafromBin->blockAxis = 0;
    tilingDatafromBin->dim0 = 128;
    tilingDatafromBin->dim1 = 512;
    tilingDatafromBin->dim2 = 1;
    tilingDatafromBin->blockFactor = 3;
    tilingDatafromBin->blockTailFactor = 2;
    tilingDatafromBin->baseN = 3;
    tilingDatafromBin->baseLen = 512;
    tilingDatafromBin->hasOffset = 1;
    tilingDatafromBin->sqrtMode = 1;

    ICPU_SET_TILING_KEY(13);

    auto ascend_anti_quant_v2_kernel = [](GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, 
                                            GM_ADDR workSpace, GM_ADDR tiling) {
        ::ascend_anti_quant_v2<1, 1, 1>(x, scale, offset, y, workSpace, tiling);
    };
    ICPU_RUN_KF(
        ascend_anti_quant_v2_kernel, blockDim, x, scale, offset, y, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}