/**
 * Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_mx_quant.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/arch35/dynamic_mx_quant_tilingdata.h"
#include "../../../op_kernel/arch35/dynamic_mx_quant_struct.h"
#include <cstdint>

using namespace std;

#include "../../../op_kernel/dynamic_mx_quant.cpp"

class dynamic_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "dynamic_mx_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "dynamic_mx_quant_test TearDown\n" << endl;
    }
};

TEST_F(dynamic_mx_quant_test, test_case_tail_axis_fp8_e5m2)
{
    size_t inputXSize = 128 * 256 * sizeof(float);
    size_t outputY = 128 * 256 * sizeof(int8_t);
    size_t outputScale = 128 * 8 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(DynamicMxQuantTailAxisTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputY);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(outputScale);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicMxQuantTailAxisTilingData* tilingDatafromBin =
        reinterpret_cast<DynamicMxQuantTailAxisTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 33;
    tilingDatafromBin->ubSize = 2048;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->blockSize = 32;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->usedCoreNum = 2;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 2;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 256;
    tilingDatafromBin->colNormalBlockNum = 1;
    tilingDatafromBin->colTailLen = 256;
    tilingDatafromBin->rowNormalBlockNum = 64;
    tilingDatafromBin->rowTailLen = 64;
    tilingDatafromBin->maxUbBlockNum = 8;
    tilingDatafromBin->dstTypeMax = 0.0f;
    tilingDatafromBin->invDstTypeMax = 1.0f / 6.0f;

    ICPU_SET_TILING_KEY(33);

    auto dynamic_mx_quant_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workSpace, GM_ADDR tiling) {
        ::dynamic_mx_quant<TPL_TAIL_AXIS_QUANT_OPTI, TPL_SCALE_ALG_0, TPL_NOT_CARE_ROUND_MODE, TPL_NOT_CARE_SCALE>(x, y, mxScale, workSpace, tiling);
    };
    ICPU_RUN_KF(dynamic_mx_quant_kernel, blockDim, x, y, mxScale, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}

TEST_F(dynamic_mx_quant_test, test_case_tail_axis_scale_alg_one_fp8_e4m3fn)
{
    size_t inputXSize = 128 * 256 * sizeof(float);
    size_t outputY = 128 * 256 * sizeof(int8_t);
    size_t outputScale = 128 * 8 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(DynamicMxQuantTailAxisTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputY);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(outputScale);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicMxQuantTailAxisTilingData* tilingDatafromBin =
        reinterpret_cast<DynamicMxQuantTailAxisTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 65;
    tilingDatafromBin->ubSize = 2048;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->blockSize = 32;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->usedCoreNum = 2;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 2;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 256;
    tilingDatafromBin->colNormalBlockNum = 1;
    tilingDatafromBin->colTailLen = 256;
    tilingDatafromBin->rowNormalBlockNum = 64;
    tilingDatafromBin->rowTailLen = 64;
    tilingDatafromBin->maxUbBlockNum = 8;
    tilingDatafromBin->dstTypeMax = 0.0f;
    tilingDatafromBin->invDstTypeMax = 1.0f / 6.0f;

    ICPU_SET_TILING_KEY(65);

    auto dynamic_mx_quant_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workSpace, GM_ADDR tiling) {
        ::dynamic_mx_quant<TPL_TAIL_AXIS_QUANT_OPTI, TPL_SCALE_ALG_1, TPL_NOT_CARE_ROUND_MODE, TPL_NOT_CARE_SCALE>(x, y, mxScale, workSpace, tiling);
    };
    ICPU_RUN_KF(dynamic_mx_quant_kernel, blockDim, x, y, mxScale, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}

TEST_F(dynamic_mx_quant_test, test_case_tail_axis_scale_alg_two)
{
    size_t inputXSize = 128 * 256 * sizeof(float);
    size_t outputY = 128 * 256 * sizeof(int8_t);
    size_t outputScale = 128 * 8 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(DynamicMxQuantTailAxisTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputY);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(outputScale);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicMxQuantTailAxisTilingData* tilingDatafromBin =
        reinterpret_cast<DynamicMxQuantTailAxisTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 97;
    tilingDatafromBin->ubSize = 2048;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->blockSize = 32;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->usedCoreNum = 2;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 2;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 256;
    tilingDatafromBin->colNormalBlockNum = 1;
    tilingDatafromBin->colTailLen = 256;
    tilingDatafromBin->rowNormalBlockNum = 64;
    tilingDatafromBin->rowTailLen = 64;
    tilingDatafromBin->maxUbBlockNum = 8;
    tilingDatafromBin->dstTypeMax = 6.0f;
    tilingDatafromBin->invDstTypeMax = 1.0f / 6.0f;

    ICPU_SET_TILING_KEY(97);

    auto dynamic_mx_quant_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workSpace, GM_ADDR tiling) {
        ::dynamic_mx_quant<TPL_TAIL_AXIS_QUANT_OPTI, TPL_SCALE_ALG_2_SPEC_DST_MAX_VALUE, TPL_NOT_CARE_ROUND_MODE, TPL_NOT_CARE_SCALE>(x, y, mxScale, workSpace, tiling);
    };
    ICPU_RUN_KF(dynamic_mx_quant_kernel, blockDim, x, y, mxScale, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}
