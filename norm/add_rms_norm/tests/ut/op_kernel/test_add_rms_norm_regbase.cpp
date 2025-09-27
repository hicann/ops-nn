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
#include "add_rms_norm_regbase_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void add_rms_norm(
    uint8_t* x1, uint8_t* x2, uint8_t* gamma, uint8_t* y, uint8_t* rstd, uint8_t* x, uint8_t* workspace,
    uint8_t* tiling);

class add_rms_norm_regbase_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "add_rms_norm_regbase_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_rms_norm_regbase_test TearDown\n" << endl;
    }
};

TEST_F(add_rms_norm_regbase_test, test_case_key1)
{
    int M = 4;
    int N = 64;
    size_t xByteSize = M * N * sizeof(int16_t);
    size_t gammaByteSize = N * sizeof(int16_t);
    size_t rstdByteSize = M * sizeof(float);
    size_t yByteSize = M * N * sizeof(int16_t);
    size_t tilingDataSize = sizeof(AddRMSNormRegbaseTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 256);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 4;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormRegbaseTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormRegbaseTilingData*>(tiling);

    tilingDatafromBin->numRow = M;
    tilingDatafromBin->numCol = N;
    tilingDatafromBin->numColAlign = N;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->rowFactor = 64;
    tilingDatafromBin->ubFactor = 32;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->avgFactor = (1.0 / N);
    tilingDatafromBin->ubLoop = 0;
    tilingDatafromBin->colBuferLength = 64;
    tilingDatafromBin->multiNNum = 1;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_regbase_test, test_case_key2)
{
    int M = 4;
    int N = 64;
    size_t xByteSize = M * N * sizeof(float);
    size_t gammaByteSize = N * sizeof(float);
    size_t rstdByteSize = M * sizeof(float);
    size_t yByteSize = M * N * sizeof(float);
    size_t tilingDataSize = sizeof(AddRMSNormRegbaseTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 256);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 4;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormRegbaseTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormRegbaseTilingData*>(tiling);

    tilingDatafromBin->numRow = M;
    tilingDatafromBin->numCol = N;
    tilingDatafromBin->numColAlign = N;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->rowFactor = 64;
    tilingDatafromBin->ubFactor = 32;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->avgFactor = (1.0 / N);
    tilingDatafromBin->ubLoop = 0;
    tilingDatafromBin->colBuferLength = 64;
    tilingDatafromBin->multiNNum = 1;

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_regbase_test, test_case_key3)
{
    int M = 4;
    int N = 64;
    size_t xByteSize = M * N * sizeof(int16_t);
    size_t gammaByteSize = N * sizeof(int16_t);
    size_t rstdByteSize = M * sizeof(float);
    size_t yByteSize = M * N * sizeof(int16_t);
    size_t tilingDataSize = sizeof(AddRMSNormRegbaseTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 256);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 4;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormRegbaseTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormRegbaseTilingData*>(tiling);

    tilingDatafromBin->numRow = M;
    tilingDatafromBin->numCol = N;
    tilingDatafromBin->numColAlign = N;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->rowFactor = 64;
    tilingDatafromBin->ubFactor = 32;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->avgFactor = (1.0 / N);
    tilingDatafromBin->ubLoop = 0;
    tilingDatafromBin->colBuferLength = 64;
    tilingDatafromBin->multiNNum = 1;

    ICPU_SET_TILING_KEY(3);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_regbase_test, test_case_key1001)
{
    int M = 4;
    int N = 64;
    size_t xByteSize = M * N * sizeof(int16_t);
    size_t gammaByteSize = N * sizeof(int16_t);
    size_t rstdByteSize = M * sizeof(float);
    size_t yByteSize = M * N * sizeof(int16_t);
    size_t tilingDataSize = sizeof(AddRMSNormRegbaseTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 256);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 4;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormRegbaseTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormRegbaseTilingData*>(tiling);

    tilingDatafromBin->numRow = M;
    tilingDatafromBin->numCol = N;
    tilingDatafromBin->numColAlign = N;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->rowFactor = 64;
    tilingDatafromBin->ubFactor = 32;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->avgFactor = (1.0 / N);
    tilingDatafromBin->ubLoop = 2;
    tilingDatafromBin->colBuferLength = 32;
    tilingDatafromBin->multiNNum = 1;

    ICPU_SET_TILING_KEY(1001);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_regbase_test, test_case_key1002)
{
    int M = 4;
    int N = 64;
    size_t xByteSize = M * N * sizeof(float);
    size_t gammaByteSize = N * sizeof(float);
    size_t rstdByteSize = M * sizeof(float);
    size_t yByteSize = M * N * sizeof(float);
    size_t tilingDataSize = sizeof(AddRMSNormRegbaseTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 256);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 4;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormRegbaseTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormRegbaseTilingData*>(tiling);

    tilingDatafromBin->numRow = M;
    tilingDatafromBin->numCol = N;
    tilingDatafromBin->numColAlign = N;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->rowFactor = 64;
    tilingDatafromBin->ubFactor = 32;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->avgFactor = (1.0 / N);
    tilingDatafromBin->ubLoop = 2;
    tilingDatafromBin->colBuferLength = 32;
    tilingDatafromBin->multiNNum = 1;

    ICPU_SET_TILING_KEY(1002);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
