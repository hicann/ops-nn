/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>
#include <cmath>
#include <limits>

#define DTYPE_ZERO_POINT DT_FLOAT

using namespace std;

extern "C" __global__ __aicore__ void fake_quant_affine_cachemask(GM_ADDR x, GM_ADDR scale, GM_ADDR zero_point,
                                                                  GM_ADDR y, GM_ADDR mask, GM_ADDR workspace,
                                                                  GM_ADDR tiling);
class fake_quant_affine_achemask_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "fake_quant_affine_achemask_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "fake_quant_affine_achemask_test TearDown\n" << endl; }
};

TEST_F(fake_quant_affine_achemask_test, test_case_fp32)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t totalLength = 1;

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
    uint32_t blockDim = 1;

    system("cp -r "
           "../../../../quant/fake_quant_affine_cachemask/tests/ut/op_kernel/fake_quant_affine_cachemask_data ./");
    system("chmod -R 755 ./fake_quant_affine_cachemask_data/");
    system("cd ./fake_quant_affine_cachemask_data/ && rm -rf ./*bin");
    system("cd ./fake_quant_affine_cachemask_data/ && python3 gen_data.py 1 float32");
    system("cd ./fake_quant_affine_cachemask_data/ && python3 gen_tiling.py 1 float32 1 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/fake_quant_affine_cachemask_data/data.bin", x_size, x, x_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/scale.bin", scale_size, scale, scale_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/zero_point.bin", zero_point_size, zero_point, zero_point_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ((float*)x)[0] = 45.5f;
    ((float*)scale)[0] = 7.0f;
    *(float*)zero_point = 0.0f;

    FakeQuantAffineCachemaskTilingData* tilingDatafromBin = reinterpret_cast<FakeQuantAffineCachemaskTilingData*>(
        tiling);
    tilingDatafromBin->quantMin = -200;
    tilingDatafromBin->quantMax = 200;
    tilingDatafromBin->loopNum = 1;
    tilingDatafromBin->remainNum = 0;
    tilingDatafromBin->calcLength = 1;
    tilingDatafromBin->headNum = 1;
    tilingDatafromBin->dataPerRepeat = 64;
    tilingDatafromBin->totalLengthAligned = 8;
    tilingDatafromBin->tileLength = 2688;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(fake_quant_affine_cachemask, blockDim, x, scale, zero_point, y, mask, workspace,
                (uint8_t*)(tilingDatafromBin));

    EXPECT_FLOAT_EQ(((float*)y)[0], 42.0f);
    EXPECT_EQ(mask[0], 1);

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
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t totalLength = 1;

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
    uint32_t blockDim = 1;

    system("cp -r "
           "../../../../quant/fake_quant_affine_cachemask/tests/ut/op_kernel/fake_quant_affine_cachemask_data ./");
    system("chmod -R 755 ./fake_quant_affine_cachemask_data/");
    system("cd ./fake_quant_affine_cachemask_data/ && rm -rf ./*bin");
    system("cd ./fake_quant_affine_cachemask_data/ && python3 gen_data.py 1 float16");
    system("cd ./fake_quant_affine_cachemask_data/ && python3 gen_tiling.py 1 float16 1 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/fake_quant_affine_cachemask_data/data.bin", x_size, x, x_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/scale.bin", scale_size, scale, scale_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/zero_point.bin", zero_point_size, zero_point, zero_point_size);
    ReadFile(path + "/fake_quant_affine_cachemask_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ((half*)x)[0] = (half)45.5f;
    ((half*)scale)[0] = (half)7.0f;
    *(float*)zero_point = 0.0f;

    FakeQuantAffineCachemaskTilingData* tilingDatafromBin = reinterpret_cast<FakeQuantAffineCachemaskTilingData*>(
        tiling);
    tilingDatafromBin->quantMin = -200;
    tilingDatafromBin->quantMax = 200;
    tilingDatafromBin->loopNum = 1;
    tilingDatafromBin->remainNum = 0;
    tilingDatafromBin->calcLength = 1;
    tilingDatafromBin->headNum = 1;
    tilingDatafromBin->dataPerRepeat = 128;
    tilingDatafromBin->totalLengthAligned = 16;
    tilingDatafromBin->tileLength = 3968;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(fake_quant_affine_cachemask, blockDim, x, scale, zero_point, y, mask, workspace,
                (uint8_t*)(tilingDatafromBin));

    EXPECT_FLOAT_EQ(static_cast<float>(((half*)y)[0]), 42.0f);
    EXPECT_EQ(mask[0], 1);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(fake_quant_affine_achemask_test, test_case_fp16_mask_range_beyond_half_precision)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    const uint32_t totalLength = 4;

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
    uint32_t blockDim = 1;

    const float xHost[4] = {327.6f, 327.7f, 327.7f, 328.0f};
    for (uint32_t i = 0; i < totalLength; i++) {
        ((half*)x)[i] = (half)xHost[i];
    }
    ((half*)scale)[0] = (half)0.01f;
    ((int32_t*)zero_point)[0] = 0;
    for (uint32_t i = 0; i < totalLength; i++) {
        mask[i] = 0xFF;
    }

    FakeQuantAffineCachemaskTilingData* tilingData = reinterpret_cast<FakeQuantAffineCachemaskTilingData*>(tiling);
    tilingData->quantMin = 0;
    tilingData->quantMax = 32767;
    tilingData->loopNum = 1;
    tilingData->remainNum = 0;
    tilingData->calcLength = totalLength;
    tilingData->headNum = 1;
    tilingData->dataPerRepeat = 128;
    tilingData->totalLengthAligned = 16;
    tilingData->tileLength = 3968;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(fake_quant_affine_cachemask, blockDim, x, scale, zero_point, y, mask, workspace,
                (uint8_t*)(tilingData));

    EXPECT_EQ(mask[0], 1);
    EXPECT_EQ(mask[1], 0);
    EXPECT_EQ(mask[2], 0);
    EXPECT_EQ(mask[3], 0);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(fake_quant_affine_achemask_test, test_case_fp16_nan_inf)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // 260 elements = one full 256-element tile (2 repeats) + a 4-element tail.
    // Allocate by the 32B-aligned length to match totalLengthAligned.
    const uint32_t totalLength = 260;
    const uint32_t alignedLength = 272;
    size_t x_size = alignedLength * sizeof(half);
    size_t scale_size = sizeof(half);
    size_t zero_point_size = sizeof(float);
    size_t y_size = alignedLength * sizeof(half);
    size_t mask_size = alignedLength * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(FakeQuantAffineCachemaskTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scale_size);
    uint8_t* zero_point = (uint8_t*)AscendC::GmAlloc(zero_point_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    for (uint32_t i = 0; i < alignedLength; ++i) {
        ((half*)x)[i] = (half)1.0f;
    }
    // Use explicit half bit patterns: casting float NaN to half on this toolchain yields +Inf.
    // Head of the first tile.
    ((uint16_t*)x)[0] = 0x3C00U; // 1.0
    ((uint16_t*)x)[1] = 0x7E00U; // NaN
    ((uint16_t*)x)[2] = 0x7C00U; // +Inf
    ((uint16_t*)x)[3] = 0xFC00U; // -Inf
    // Tail tile: offset 256, length 4.
    ((uint16_t*)x)[256] = 0x3C00U; // 1.0
    ((uint16_t*)x)[257] = 0x7E00U; // NaN
    ((uint16_t*)x)[258] = 0x7C00U; // +Inf
    ((uint16_t*)x)[259] = 0xFC00U; // -Inf
    ((half*)scale)[0] = (half)0.05f;
    *(float*)zero_point = 120.0f;
    for (uint32_t i = 0; i < alignedLength; ++i) {
        mask[i] = 0xFF;
    }

    auto* td = reinterpret_cast<FakeQuantAffineCachemaskTilingData*>(tiling);
    td->quantMin = 0;
    td->quantMax = 255;
    td->loopNum = 1;
    td->remainNum = 0;
    td->calcLength = totalLength;
    td->headNum = 1;
    td->dataPerRepeat = 128;
    td->totalLengthAligned = alignedLength;
    td->tileLength = 256;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(fake_quant_affine_cachemask, 1, x, scale, zero_point, y, mask, workspace, (uint8_t*)td);

    half* yh = (half*)y;
    EXPECT_NEAR((float)yh[0], 1.0f, 1e-3f);
    // PyTorch semantics: NaN -> quant_min, i.e. (0 - 120) * 0.05 = -6.0.
    EXPECT_NEAR((float)yh[1], -6.0f, 1e-3f);
    EXPECT_NEAR((float)yh[2], 6.75f, 1e-3f);
    EXPECT_NEAR((float)yh[3], -6.0f, 1e-3f);
    EXPECT_EQ(mask[0], 1);
    EXPECT_EQ(mask[1], 0);
    EXPECT_EQ(mask[2], 0);
    EXPECT_EQ(mask[3], 0);

    // Tail tile must use the same NaN/Inf semantics.
    EXPECT_NEAR((float)yh[256], 1.0f, 1e-3f);
    EXPECT_NEAR((float)yh[257], -6.0f, 1e-3f);
    EXPECT_NEAR((float)yh[258], 6.75f, 1e-3f);
    EXPECT_NEAR((float)yh[259], -6.0f, 1e-3f);
    EXPECT_EQ(mask[256], 1);
    EXPECT_EQ(mask[257], 0);
    EXPECT_EQ(mask[258], 0);
    EXPECT_EQ(mask[259], 0);

    // Negative quant_min distinguishes NaN -> quant_min from NaN -> 0.
    *(float*)zero_point = 0.0f;
    td->quantMin = -128;
    td->quantMax = 127;
    for (uint32_t i = 0; i < alignedLength; ++i) {
        mask[i] = 0xFF;
    }
    ICPU_RUN_KF(fake_quant_affine_cachemask, 1, x, scale, zero_point, y, mask, workspace, (uint8_t*)td);

    EXPECT_NEAR((float)yh[0], 1.0f, 1e-3f);
    EXPECT_NEAR((float)yh[1], -6.4f, 1e-2f); // NaN -> quant_min
    EXPECT_NEAR((float)yh[2], 6.35f, 1e-2f); // +Inf -> quant_max
    EXPECT_NEAR((float)yh[3], -6.4f, 1e-2f); // -Inf -> quant_min
    EXPECT_EQ(mask[0], 1);
    EXPECT_EQ(mask[1], 0);
    EXPECT_EQ(mask[2], 0);
    EXPECT_EQ(mask[3], 0);
    EXPECT_NEAR((float)yh[256], 1.0f, 1e-3f);
    EXPECT_NEAR((float)yh[257], -6.4f, 1e-2f);
    EXPECT_NEAR((float)yh[258], 6.35f, 1e-2f);
    EXPECT_NEAR((float)yh[259], -6.4f, 1e-2f);
    EXPECT_EQ(mask[256], 1);
    EXPECT_EQ(mask[257], 0);
    EXPECT_EQ(mask[258], 0);
    EXPECT_EQ(mask[259], 0);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(fake_quant_affine_achemask_test, test_case_fp32_nan_inf)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // 260 elements = one full 256-element tile (4 repeats) + a 4-element tail.
    // Allocate by the 32B-aligned length to match totalLengthAligned.
    const uint32_t totalLength = 260;
    const uint32_t alignedLength = 264;
    size_t x_size = alignedLength * sizeof(float);
    size_t scale_size = sizeof(float);
    size_t zero_point_size = sizeof(float);
    size_t y_size = alignedLength * sizeof(float);
    size_t mask_size = alignedLength * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(FakeQuantAffineCachemaskTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scale_size);
    uint8_t* zero_point = (uint8_t*)AscendC::GmAlloc(zero_point_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    for (uint32_t i = 0; i < alignedLength; ++i) {
        ((float*)x)[i] = 1.0f;
    }
    // Head of the first tile.
    ((float*)x)[0] = 1.0f;
    ((float*)x)[1] = std::numeric_limits<float>::quiet_NaN();
    ((float*)x)[2] = std::numeric_limits<float>::infinity();
    ((float*)x)[3] = -std::numeric_limits<float>::infinity();
    // Tail tile: offset 256, length 4.
    ((float*)x)[256] = 1.0f;
    ((float*)x)[257] = std::numeric_limits<float>::quiet_NaN();
    ((float*)x)[258] = std::numeric_limits<float>::infinity();
    ((float*)x)[259] = -std::numeric_limits<float>::infinity();
    ((float*)scale)[0] = 0.05f;
    *(float*)zero_point = 120.0f;
    for (uint32_t i = 0; i < alignedLength; ++i) {
        mask[i] = 0xFF;
    }

    auto* td = reinterpret_cast<FakeQuantAffineCachemaskTilingData*>(tiling);
    td->quantMin = 0;
    td->quantMax = 255;
    td->loopNum = 1;
    td->remainNum = 0;
    td->calcLength = totalLength;
    td->headNum = 1;
    td->dataPerRepeat = 64;
    td->totalLengthAligned = alignedLength;
    td->tileLength = 256;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(fake_quant_affine_cachemask, 1, x, scale, zero_point, y, mask, workspace, (uint8_t*)td);

    float* yf = (float*)y;
    EXPECT_NEAR(yf[0], 1.0f, 1e-5f);
    // PyTorch semantics: NaN -> quant_min, i.e. (0 - 120) * 0.05 = -6.0.
    EXPECT_NEAR(yf[1], -6.0f, 1e-5f);
    EXPECT_NEAR(yf[2], 6.75f, 1e-5f);
    EXPECT_NEAR(yf[3], -6.0f, 1e-5f);
    EXPECT_EQ(mask[0], 1);
    EXPECT_EQ(mask[1], 0);
    EXPECT_EQ(mask[2], 0);
    EXPECT_EQ(mask[3], 0);

    // Tail tile must use the same NaN/Inf semantics.
    EXPECT_NEAR(yf[256], 1.0f, 1e-5f);
    EXPECT_NEAR(yf[257], -6.0f, 1e-5f);
    EXPECT_NEAR(yf[258], 6.75f, 1e-5f);
    EXPECT_NEAR(yf[259], -6.0f, 1e-5f);
    EXPECT_EQ(mask[256], 1);
    EXPECT_EQ(mask[257], 0);
    EXPECT_EQ(mask[258], 0);
    EXPECT_EQ(mask[259], 0);

    // Negative quant_min distinguishes NaN -> quant_min from NaN -> 0.
    *(float*)zero_point = 0.0f;
    td->quantMin = -128;
    td->quantMax = 127;
    for (uint32_t i = 0; i < alignedLength; ++i) {
        mask[i] = 0xFF;
    }
    ICPU_RUN_KF(fake_quant_affine_cachemask, 1, x, scale, zero_point, y, mask, workspace, (uint8_t*)td);

    EXPECT_NEAR(yf[0], 1.0f, 1e-5f);
    EXPECT_NEAR(yf[1], -6.4f, 1e-5f); // NaN -> quant_min
    EXPECT_NEAR(yf[2], 6.35f, 1e-5f); // +Inf -> quant_max
    EXPECT_NEAR(yf[3], -6.4f, 1e-5f); // -Inf -> quant_min
    EXPECT_EQ(mask[0], 1);
    EXPECT_EQ(mask[1], 0);
    EXPECT_EQ(mask[2], 0);
    EXPECT_EQ(mask[3], 0);
    EXPECT_NEAR(yf[256], 1.0f, 1e-5f);
    EXPECT_NEAR(yf[257], -6.4f, 1e-5f);
    EXPECT_NEAR(yf[258], 6.35f, 1e-5f);
    EXPECT_NEAR(yf[259], -6.4f, 1e-5f);
    EXPECT_EQ(mask[256], 1);
    EXPECT_EQ(mask[257], 0);
    EXPECT_EQ(mask[258], 0);
    EXPECT_EQ(mask[259], 0);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
