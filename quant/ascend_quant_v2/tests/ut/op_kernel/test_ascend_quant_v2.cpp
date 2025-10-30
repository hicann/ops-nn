/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "ascend_quant_v2_tiling_def.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void ascend_quant_v2(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class ascend_quant_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ascend_quant_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ascend_quant_v2_test TearDown\n" << endl;
    }
};

// test float16
TEST_F(ascend_quant_v2_test, test_case_1)
{
    size_t xByteSize = 10240 * 1023 * sizeof(half);
    size_t scaleByteSize = 1023 * sizeof(half);
    size_t offsetByteSize = 1023 * sizeof(half);
    size_t yByteSize = 10240 * 1023 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(AscendQuantV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendQuantV2TilingData* tilingDatafromBin = reinterpret_cast<AscendQuantV2TilingData*>(tiling);

    tilingDatafromBin->numCore = 48;
    tilingDatafromBin->blockAxis = 0;
    tilingDatafromBin->ubAxis = 0;
    tilingDatafromBin->dim0 = 10240;
    tilingDatafromBin->dim1 = 1023;
    tilingDatafromBin->dim2 = 1;
    tilingDatafromBin->blockUnion = 1;
    tilingDatafromBin->blockFactor = 214;
    tilingDatafromBin->blockTailFactor = 182;
    tilingDatafromBin->baseN = 61;
    tilingDatafromBin->baseLen = 1024;
    tilingDatafromBin->hasOffset = 1;
    tilingDatafromBin->sqrtMode = 0;
    tilingDatafromBin->roundMode = 0;
    tilingDatafromBin->reserve1 = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(ascend_quant_v2, blockDim, x, scale, offset, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// test float
TEST_F(ascend_quant_v2_test, test_case_2)
{
    size_t xByteSize = 10240 * 1023 * sizeof(float);
    size_t scaleByteSize = 1023 * sizeof(float);
    size_t offsetByteSize = 1023 * sizeof(float);
    size_t yByteSize = 10240 * 1023 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(AscendQuantV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendQuantV2TilingData* tilingDatafromBin = reinterpret_cast<AscendQuantV2TilingData*>(tiling);

    tilingDatafromBin->numCore = 48;
    tilingDatafromBin->blockAxis = 0;
    tilingDatafromBin->ubAxis = 0;
    tilingDatafromBin->dim0 = 10240;
    tilingDatafromBin->dim1 = 1023;
    tilingDatafromBin->dim2 = 1;
    tilingDatafromBin->blockUnion = 1;
    tilingDatafromBin->blockFactor = 214;
    tilingDatafromBin->blockTailFactor = 182;
    tilingDatafromBin->baseN = 35;
    tilingDatafromBin->baseLen = 1024;
    tilingDatafromBin->hasOffset = 1;
    tilingDatafromBin->sqrtMode = 0;
    tilingDatafromBin->roundMode = 0;
    tilingDatafromBin->reserve1 = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(ascend_quant_v2, blockDim, x, scale, offset, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}