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
#include "rms_norm_grad_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void rms_norm_grad(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling);

class rms_norm_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rms_norm_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rms_norm_grad_test TearDown\n" << endl;
    }
};

TEST_F(rms_norm_grad_test, test_case_221)
{
    size_t inputxByteSize = 4 * 32 * sizeof(int16_t);
    size_t inputdyByteSize = 4 * 32 * sizeof(int16_t);
    size_t inputrstdByteSize = 4 * sizeof(float);
    size_t inputgammaByteSize = 32 * sizeof(int16_t);
    size_t outputByteSize = 4 * 32 * sizeof(int16_t);
    size_t outputdgammaByteSize = 32 * sizeof(float);
    size_t tiling_data_size = sizeof(RmsNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputdyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputxByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(inputrstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputgammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(outputdgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(50 * 32 + 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;

    char* path_ = get_current_dir_name();
    string path(path_);

    RmsNormGradTilingData* tilingDatafromBin = reinterpret_cast<RmsNormGradTilingData*>(tiling);

    tilingDatafromBin->row = 4;
    tilingDatafromBin->col = 32;
    tilingDatafromBin->avg_factor = 0.03125;
    tilingDatafromBin->data_type = 1;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->ub_split_dim = 0;
    tilingDatafromBin->ub_factor = 1;
    tilingDatafromBin->core_calc_num = 1;
    tilingDatafromBin->core_calc_tail = 0;
    tilingDatafromBin->block_dim = 4;
    tilingDatafromBin->ub_calc_num = 1;
    tilingDatafromBin->ub_calc_tail = 0;
    tilingDatafromBin->ub_calc_loop = 1;
    tilingDatafromBin->ub_calc_tail_num = 1;
    tilingDatafromBin->ub_calc_tail_tail = 0;
    tilingDatafromBin->ub_calc_tail_loop = 1;
    tilingDatafromBin->fixed_output = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(rms_norm_grad, blockDim, dy, x, rstd, gamma, dx, dgamma, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rms_norm_grad_test, test_case_221_normal)
{
    size_t inputxByteSize = 4 * 1024 * sizeof(int16_t);
    size_t inputdyByteSize = 4 * 1024 * sizeof(int16_t);
    size_t inputrstdByteSize = 4 * sizeof(float);
    size_t inputgammaByteSize = 1024 * sizeof(int16_t);
    size_t outputByteSize = 4 * 1024 * sizeof(int16_t);
    size_t outputdgammaByteSize = 1024 * sizeof(float);
    size_t tiling_data_size = sizeof(RmsNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputdyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputxByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(inputrstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputgammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(outputdgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(50 * 1024 + 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;

    char* path_ = get_current_dir_name();
    string path(path_);

    RmsNormGradTilingData* tilingDatafromBin = reinterpret_cast<RmsNormGradTilingData*>(tiling);

    tilingDatafromBin->row = 4;
    tilingDatafromBin->col = 1024;
    tilingDatafromBin->avg_factor = 0.03125;
    tilingDatafromBin->data_type = 1;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->ub_split_dim = 0;
    tilingDatafromBin->ub_factor = 1;
    tilingDatafromBin->core_calc_num = 1;
    tilingDatafromBin->core_calc_tail = 0;
    tilingDatafromBin->block_dim = 4;
    tilingDatafromBin->ub_calc_num = 1;
    tilingDatafromBin->ub_calc_tail = 0;
    tilingDatafromBin->ub_calc_loop = 1;
    tilingDatafromBin->ub_calc_tail_num = 1;
    tilingDatafromBin->ub_calc_tail_tail = 0;
    tilingDatafromBin->ub_calc_tail_loop = 1;
    tilingDatafromBin->fixed_output = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(rms_norm_grad, blockDim, dy, x, rstd, gamma, dx, dgamma, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rms_norm_grad_test, test_case_222)
{
    size_t inputxByteSize = 4 * 128 * sizeof(int16_t);
    size_t inputdyByteSize = 4 * 128 * sizeof(int16_t);
    size_t inputrstdByteSize = 4 * sizeof(float);
    size_t inputgammaByteSize = 128 * sizeof(int16_t);
    size_t outputByteSize = 4 * 128 * sizeof(int16_t);
    size_t outputdgammaByteSize = 128 * sizeof(float);
    size_t tiling_data_size = sizeof(RmsNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputdyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputxByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(inputrstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputgammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(outputdgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(50 * 32 + 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;

    char* path_ = get_current_dir_name();
    string path(path_);

    RmsNormGradTilingData* tilingDatafromBin = reinterpret_cast<RmsNormGradTilingData*>(tiling);

    tilingDatafromBin->row = 4;
    tilingDatafromBin->col = 128;
    tilingDatafromBin->avg_factor = 0.03125;
    tilingDatafromBin->data_type = 1;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->ub_split_dim = 1;
    tilingDatafromBin->ub_factor = 64;
    tilingDatafromBin->core_calc_num = 1;
    tilingDatafromBin->core_calc_tail = 0;
    tilingDatafromBin->block_dim = 4;
    tilingDatafromBin->ub_calc_num = 64;
    tilingDatafromBin->ub_calc_tail = 0;
    tilingDatafromBin->ub_calc_loop = 2;
    tilingDatafromBin->ub_calc_tail_num = 64;
    tilingDatafromBin->ub_calc_tail_tail = 0;
    tilingDatafromBin->ub_calc_tail_loop = 2;
    tilingDatafromBin->fixed_output = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(rms_norm_grad, blockDim, dy, x, rstd, gamma, dx, dgamma, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
