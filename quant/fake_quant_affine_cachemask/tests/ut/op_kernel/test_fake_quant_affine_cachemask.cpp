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
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void fake_quant_affine_cachemask(
    GM_ADDR x, GM_ADDR scale, GM_ADDR zero_point, GM_ADDR y, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling);
class fake_quant_affine_achemask_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "fake_quant_affine_achemask_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "fake_quant_affine_achemask_test TearDown\n" << endl;
    }
};

TEST_F(fake_quant_affine_achemask_test, test_case_fp32)
{
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(float);
    size_t scale_size = sizeof(float);
    size_t zero_point_size = sizeof(int32_t);
    size_t y_size = totalLength * sizeof(float);
    size_t mask_size = totalLength * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(FakeQuantAffineCachemaskTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scale_size);
    uint8_t* zero_point = (uint8_t*)AscendC::GmAlloc(zero_point_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    system(
        "cp -r "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/fake_quant_affine_cachemask/"
        "fake_quant_affine_cachemask_data ./");
    system("chmod -R 755 ./fake_quant_affine_cachemask_data/");
    system("cd ./fake_quant_affine_cachemask_data/ && rm -rf ./*bin");
    system("cd ./fake_quant_affine_cachemask_data/ && python3 gen_data.py 32 float32");
    system("cd ./fake_quant_affine_cachemask_data/ && python3 gen_tiling.py 32 float32 1 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/fake_quant_affine_cachemask_data/data.bin", x_size, x, x_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/scale.bin", scale_size, scale, scale_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/zero_point.bin", zero_point_size, zero_point, zero_point_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    FakeQuantAffineCachemaskTilingData* tilingDatafromBin =
        reinterpret_cast<FakeQuantAffineCachemaskTilingData*>(tiling);
    tilingDatafromBin->quantMin = -200;
    tilingDatafromBin->quantMax = 200;
    tilingDatafromBin->loopNum = 1;
    tilingDatafromBin->remainNum = 0;
    tilingDatafromBin->calcLength = 32;
    tilingDatafromBin->headNum = 1;
    tilingDatafromBin->dataPerRepeat = 64;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 2752;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        fake_quant_affine_cachemask, blockDim, x, scale, zero_point, y, mask, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(fake_quant_affine_achemask_test, test_case_fp16)
{
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(half);
    size_t scale_size = sizeof(half);
    size_t zero_point_size = sizeof(int32_t);
    size_t y_size = totalLength * sizeof(half);
    size_t mask_size = totalLength * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(FakeQuantAffineCachemaskTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scale_size);
    uint8_t* zero_point = (uint8_t*)AscendC::GmAlloc(zero_point_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    system(
        "cp -r "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/fake_quant_affine_cachemask/"
        "fake_quant_affine_cachemask_data ./");
    system("chmod -R 755 ./fake_quant_affine_cachemask_data/");
    system("cd ./fake_quant_affine_cachemask_data/ && rm -rf ./*bin");
    system("cd ./fake_quant_affine_cachemask_data/ && python3 gen_data.py 32 float16");
    system("cd ./fake_quant_affine_cachemask_data/ && python3 gen_tiling.py 32 float16 1 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/fake_quant_affine_cachemask_data/data.bin", x_size, x, x_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/scale.bin", scale_size, scale, scale_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/zero_point.bin", zero_point_size, zero_point, zero_point_size);
    // ReadFile(path + "/fake_quant_affine_cachemask_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    FakeQuantAffineCachemaskTilingData* tilingDatafromBin =
        reinterpret_cast<FakeQuantAffineCachemaskTilingData*>(tiling);
    tilingDatafromBin->quantMin = -200;
    tilingDatafromBin->quantMax = 200;
    tilingDatafromBin->loopNum = 1;
    tilingDatafromBin->remainNum = 0;
    tilingDatafromBin->calcLength = 32;
    tilingDatafromBin->headNum = 1;
    tilingDatafromBin->dataPerRepeat = 128;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 3968;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        fake_quant_affine_cachemask, blockDim, x, scale, zero_point, y, mask, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
