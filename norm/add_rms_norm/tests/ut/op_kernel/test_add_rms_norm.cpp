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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "add_rms_norm_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void add_rms_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling);

class add_rms_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "add_rms_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_rms_norm_test TearDown\n" << endl;
    }
};

TEST_F(add_rms_norm_test, test_case_10)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    ICPU_SET_TILING_KEY(10);
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

TEST_F(add_rms_norm_test, test_case_20)
{
    size_t inputByteSize = 2 * 256 * sizeof(float);
    size_t gammaByteSize = 256 * sizeof(float);
    size_t outputByteSize = 2 * 256 * sizeof(float);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 10240;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    ICPU_SET_TILING_KEY(20);
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

TEST_F(add_rms_norm_test, test_case_30)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    ICPU_SET_TILING_KEY(30);
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

TEST_F(add_rms_norm_test, test_case_11)
{
    size_t inputByteSize = 2 * 25600 * sizeof(int16_t);
    size_t gammaByteSize = 25600 * sizeof(int16_t);
    size_t outputByteSize = 2 * 25600 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 25600;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 8544;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    ICPU_SET_TILING_KEY(11);
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

TEST_F(add_rms_norm_test, test_case_21)
{
    size_t inputByteSize = 2 * 25600 * sizeof(float);
    size_t gammaByteSize = 25600 * sizeof(float);
    size_t outputByteSize = 2 * 25600 * sizeof(float);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 25600;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 8544;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    ICPU_SET_TILING_KEY(21);
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

TEST_F(add_rms_norm_test, test_case_14)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 8;
    tilingDatafromBin->ub_factor = 2048;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    ICPU_SET_TILING_KEY(14);
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
