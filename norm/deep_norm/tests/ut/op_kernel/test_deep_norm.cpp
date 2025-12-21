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
#include "deep_norm_tiling_def.h"
#include "data_utils.h"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void deep_norm(
    GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling);

class deep_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "deep_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "deep_norm_test TearDown\n" << endl;
    }
};

TEST_F(deep_norm_test, test_case_1)
{   // fp16 < 4096  tiling key 1
    // size config
    size_t N = 24;
    size_t D = 2560;
    size_t byteNum = sizeof(int16_t);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 1;
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 0;
    tilingDatafromBin->updated_last_times = 0;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_0)
{   // bf16 < 4096  tiling key 0
    // size config
    size_t N = 24;
    size_t D = 2560;
    size_t byteNum = sizeof(int16_t);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 0;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 0;
    tilingDatafromBin->updated_last_times = 0;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_2)
{   // fp32 < 4096  tiling key 2
    // size config
    size_t N = 24;
    size_t D = 2560;
    size_t byteNum = sizeof(float);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 2;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 0;
    tilingDatafromBin->updated_last_times = 0;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_5)
{   // fp16 4096 ~ 15360  tiling key 5
    // size config
    size_t N = 24;
    size_t D = 5000;
    size_t byteNum = sizeof(int16_t);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 5;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 2512;
    tilingDatafromBin->updated_last_times = 2;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_4)
{   // bf16 4096 ~ 15360  tiling key 4
    // size config
    size_t N = 24;
    size_t D = 5000;
    size_t byteNum = sizeof(int16_t);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 4;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 2512;
    tilingDatafromBin->updated_last_times = 2;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_6)
{   // fp32 4096 ~ 8192  tiling key 6
    // size config
    size_t N = 24;
    size_t D = 5000;
    size_t byteNum = sizeof(float);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 6;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 2512;
    tilingDatafromBin->updated_last_times = 2;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_13)
{   // fp16 > 15360  tiling key 13
    // size config
    size_t N = 24;
    size_t D = 16000;
    size_t byteNum = sizeof(int16_t);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 13;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 4000;
    tilingDatafromBin->updated_last_times = 4;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_12)
{   // bf16  > 15360  tiling key 12
    // size config
    size_t N = 24;
    size_t D = 16000;
    size_t byteNum = sizeof(int16_t);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 12;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 4000;
    tilingDatafromBin->updated_last_times = 4;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_14)
{   // fp32 > 8192  tiling key 14
    // size config
    size_t N = 24;
    size_t D = 9000;
    size_t byteNum = sizeof(float);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 14;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 3008;
    tilingDatafromBin->updated_last_times = 3;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_17)
{   // fp16 < 100  tiling key 17
    // size config
    size_t N = 24;
    size_t D = 256;
    size_t byteNum = sizeof(int16_t);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 17;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 0;
    tilingDatafromBin->updated_last_times = 0;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_16)
{   // bf16 < 100  tiling key 16
    // size config
    size_t N = 24;
    size_t D = 256;
    size_t byteNum = sizeof(int16_t);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 16;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 0;
    tilingDatafromBin->updated_last_times = 0;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_test, test_case_18)
{   // fp32 < 100  tiling key 18
    // size config
    size_t N = 24;
    size_t D = 256;
    size_t byteNum = sizeof(float);
    size_t floatNum = sizeof(float);
    uint32_t blockDim = 24;
    uint8_t tiling_key = 18;
    //
    size_t inputByteSize = N * D * byteNum;
    size_t gammaByteSize = D * byteNum;
    size_t outputByteSize = N * D * byteNum;
    size_t rstdByteSize = N * floatNum;
    size_t tiling_data_size = sizeof(DeepNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormTilingData* tilingDatafromBin = reinterpret_cast<DeepNormTilingData*>(tiling);

    // need calc
    tilingDatafromBin->num_core = blockDim;
    tilingDatafromBin->num_last_dim = D;
    tilingDatafromBin->num_first_dim = N;
    tilingDatafromBin->nl_firstdim_per_core = 1;
    tilingDatafromBin->l_firstdim_per_core = 1;
    tilingDatafromBin->first_dim_per_times = 1;
    tilingDatafromBin->updated_last_dim = 0;
    tilingDatafromBin->updated_last_times = 0;
    // const vars
    tilingDatafromBin->eps_str = 0.000001;
    tilingDatafromBin->ave_str = 1 / D;
    tilingDatafromBin->alpha_str = 0.3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
