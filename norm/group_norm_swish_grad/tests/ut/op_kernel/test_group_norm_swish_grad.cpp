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
#include "group_norm_swish_grad_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void group_norm_swish_grad(
    GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR dx, GM_ADDR dgamma,
    GM_ADDR dbeta, GM_ADDR workspace, GM_ADDR tiling);

class group_norm_swish_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "group_norm_swish_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "group_norm_swish_grad_test TearDown\n" << endl;
    }
};

TEST_F(group_norm_swish_grad_test, test_case_mode0_align)
{
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t HXW = 24 * 24;
    uint32_t G = 8;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(float);
    size_t mean_size = NXG * sizeof(float);
    size_t rstd_size = NXG * sizeof(float);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(float);
    size_t beta_size = C * sizeof(float);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormSwishGradTilingData);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(beta_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    system(
        "cp -r "
        "../../../../norm/group_norm_swish_grad/tests/ut/op_kernel/group_norm_swish_grad_data ./");
    system("chmod -R 755 ./group_norm_swish_grad_data/");
    system("cd ./group_norm_swish_grad_data/ && rm -rf ./*bin");
    system("cd ./group_norm_swish_grad_data/ && python3 gen_data.py 1 32 576 8 float");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/group_norm_swish_grad_data/input_dy.bin", dy_size, dy, dy_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_mean.bin", mean_size, mean, mean_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_rstd.bin", rstd_size, rstd, rstd_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_gamma.bin", gamma_size, gamma, gamma_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_beta.bin", beta_size, beta, beta_size);
    GroupNormSwishGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 1;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 32;                       // 2
    tilingDatafromBin->HXW = 576;                    // 3
    tilingDatafromBin->G = 8;                        // 4
    tilingDatafromBin->NXG = 8;                      // 5 
    tilingDatafromBin->C_G = 4;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 8;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 16;      // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 1;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 4;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 0;            // 16
    tilingDatafromBin->stage2CoreUsed = 0;           // 17
    tilingDatafromBin->castEleNum = 1;               // 18
    tilingDatafromBin->tailCastNum = 0;              // 19
    tilingDatafromBin->coreBatchParts = 1;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dgamma_is_require = 1;        // 23
    tilingDatafromBin->dbeta_is_require = 1;         // 24
    tilingDatafromBin->swish_scale = 1.0;            // 25
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        group_norm_swish_grad, blockDim, dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(group_norm_swish_grad_test, test_case_mode0_not_align)
{
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t HXW = 24 * 24 - 10;
    uint32_t G = 8;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(float);
    size_t mean_size = NXG * sizeof(float);
    size_t rstd_size = NXG * sizeof(float);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(float);
    size_t beta_size = C * sizeof(float);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormSwishGradTilingData);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(beta_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    system(
        "cp -r "
        "../../../../norm/group_norm_swish_grad/tests/ut/op_kernel/group_norm_swish_grad_data ./");
    system("chmod -R 755 ./group_norm_swish_grad_data/");
    system("cd ./group_norm_swish_grad_data/ && rm -rf ./*bin");
    system("cd ./group_norm_swish_grad_data/ && python3 gen_data.py 1 32 566 8 float");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/group_norm_swish_grad_data/input_dy.bin", dy_size, dy, dy_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_mean.bin", mean_size, mean, mean_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_rstd.bin", rstd_size, rstd, rstd_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_gamma.bin", gamma_size, gamma, gamma_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_beta.bin", beta_size, beta, beta_size);
    GroupNormSwishGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 2;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 32;                       // 2
    tilingDatafromBin->HXW = 566;                    // 3
    tilingDatafromBin->G = 8;                        // 4
    tilingDatafromBin->NXG = 8;                      // 5
    tilingDatafromBin->C_G = 4;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 8;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 17;      // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 1;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 4;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 0;            // 16
    tilingDatafromBin->stage2CoreUsed = 0;           // 17
    tilingDatafromBin->castEleNum = 1;               // 18
    tilingDatafromBin->tailCastNum = 0;              // 19
    tilingDatafromBin->coreBatchParts = 1;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dgamma_is_require = 1;        // 23
    tilingDatafromBin->dbeta_is_require = 1;         // 24
    tilingDatafromBin->swish_scale = 1.0;            // 25
    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(
        group_norm_swish_grad, blockDim, dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(group_norm_swish_grad_test, test_case_mode1)
{
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t HXW = 64 * 128;
    uint32_t G = 8;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(float);
    size_t mean_size = NXG * sizeof(float);
    size_t rstd_size = NXG * sizeof(float);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(float);
    size_t beta_size = C * sizeof(float);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormSwishGradTilingData);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(beta_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    system(
        "cp -r "
        "../../../../norm/group_norm_swish_grad/tests/ut/op_kernel/group_norm_swish_grad_data ./");
    system("chmod -R 755 ./group_norm_swish_grad_data/");
    system("cd ./group_norm_swish_grad_data/ && rm -rf ./*bin");
    system("cd ./group_norm_swish_grad_data/ && python3 gen_data.py 1 32 8192 8 float");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/group_norm_swish_grad_data/input_dy.bin", dy_size, dy, dy_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_mean.bin", mean_size, mean, mean_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_rstd.bin", rstd_size, rstd, rstd_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_gamma.bin", gamma_size, gamma, gamma_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_beta.bin", beta_size, beta, beta_size);
    GroupNormSwishGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 1;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 32;                       // 2
    tilingDatafromBin->HXW = 8192;                   // 3
    tilingDatafromBin->G = 8;                        // 4
    tilingDatafromBin->NXG = 8;                      // 5
    tilingDatafromBin->C_G = 4;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 8;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 1;       // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 4;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 1;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 0;            // 16
    tilingDatafromBin->stage2CoreUsed = 0;           // 17
    tilingDatafromBin->castEleNum = 1;               // 18
    tilingDatafromBin->tailCastNum = 0;              // 19
    tilingDatafromBin->coreBatchParts = 1;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dgamma_is_require = 1;        // 23
    tilingDatafromBin->dbeta_is_require = 1;         // 24
    tilingDatafromBin->swish_scale = 1.0;            // 25
    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(
        group_norm_swish_grad, blockDim, dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(group_norm_swish_grad_test, test_case_mode3)
{
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t HXW = 128 * 128;
    uint32_t G = 8;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(float);
    size_t mean_size = NXG * sizeof(float);
    size_t rstd_size = NXG * sizeof(float);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(float);
    size_t beta_size = C * sizeof(float);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormSwishGradTilingData);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(beta_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    system(
        "cp -r "
        "../../../../norm/group_norm_swish_grad/tests/ut/op_kernel/group_norm_swish_grad_data ./");
    system("chmod -R 755 ./group_norm_swish_grad_data/");
    system("cd ./group_norm_swish_grad_data/ && rm -rf ./*bin");
    system("cd ./group_norm_swish_grad_data/ && python3 gen_data.py 1 32 16384 8 float");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/group_norm_swish_grad_data/input_dy.bin", dy_size, dy, dy_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_mean.bin", mean_size, mean, mean_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_rstd.bin", rstd_size, rstd, rstd_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_gamma.bin", gamma_size, gamma, gamma_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_beta.bin", beta_size, beta, beta_size);
    GroupNormSwishGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 2;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 32;                       // 2
    tilingDatafromBin->HXW = 16384;                  // 3
    tilingDatafromBin->G = 8;                        // 4
    tilingDatafromBin->NXG = 8;                      // 5
    tilingDatafromBin->C_G = 4;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 8;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 0;       // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 0;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 0;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 9728; // 13
    tilingDatafromBin->mode2_ub_iteration_num = 2;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 6656;     // 15
    tilingDatafromBin->workSpaceSize = 0;            // 16
    tilingDatafromBin->stage2CoreUsed = 0;           // 17
    tilingDatafromBin->castEleNum = 1;               // 18
    tilingDatafromBin->tailCastNum = 0;              // 19
    tilingDatafromBin->coreBatchParts = 1;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dgamma_is_require = 1;        // 23
    tilingDatafromBin->dbeta_is_require = 1;         // 24
    tilingDatafromBin->swish_scale = 1.0;            // 25
    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(
        group_norm_swish_grad, blockDim, dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(group_norm_swish_grad_test, test_case_mode0_bf16)
{
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t HXW = 24 * 24;
    uint32_t G = 8;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(float);
    size_t mean_size = NXG * sizeof(float);
    size_t rstd_size = NXG * sizeof(float);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(float);
    size_t beta_size = C * sizeof(float);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormSwishGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(beta_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    system(
        "cp -r "
        "../../../../norm/group_norm_swish_grad/tests/ut/op_kernel/group_norm_swish_grad_data ./");
    system("chmod -R 755 ./group_norm_swish_grad_data/");
    system("cd ./group_norm_swish_grad_data/ && rm -rf ./*bin");
    system("cd ./group_norm_swish_grad_data/ && python3 gen_data.py 1 32 576 8 float16");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/group_norm_swish_grad_data/input_dy.bin", dy_size, dy, dy_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_mean.bin", mean_size, mean, mean_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_rstd.bin", rstd_size, rstd, rstd_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_gamma.bin", gamma_size, gamma, gamma_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_beta.bin", beta_size, beta, beta_size);
    GroupNormSwishGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 2;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 32;                       // 2
    tilingDatafromBin->HXW = 576;                    // 3
    tilingDatafromBin->G = 8;                        // 4
    tilingDatafromBin->NXG = 8;                      // 5
    tilingDatafromBin->C_G = 4;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 8;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 11;      // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 1;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 4;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 32;           // 16
    tilingDatafromBin->stage2CoreUsed = 1;           // 17
    tilingDatafromBin->castEleNum = 64;              // 18
    tilingDatafromBin->tailCastNum = 32;             // 19
    tilingDatafromBin->coreBatchParts = 1;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dgamma_is_require = 1;        // 23
    tilingDatafromBin->dbeta_is_require = 1;         // 24
    tilingDatafromBin->swish_scale = 1.0;            // 25
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        group_norm_swish_grad, blockDim, dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(group_norm_swish_grad_test, test_case_mode1_bf16)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t HXW = 64 * 128;
    uint32_t G = 8;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(float);
    size_t mean_size = NXG * sizeof(float);
    size_t rstd_size = NXG * sizeof(float);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(float);
    size_t beta_size = C * sizeof(float);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormSwishGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(beta_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    system(
        "cp -r "
        "../../../../norm/group_norm_swish_grad/tests/ut/op_kernel/group_norm_swish_grad_data ./");
    system("chmod -R 755 ./group_norm_swish_grad_data/");
    system("cd ./group_norm_swish_grad_data/ && rm -rf ./*bin");
    system("cd ./group_norm_swish_grad_data/ && python3 gen_data.py 1 32 8192 8 float16");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/group_norm_swish_grad_data/input_dy.bin", dy_size, dy, dy_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_mean.bin", mean_size, mean, mean_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_rstd.bin", rstd_size, rstd, rstd_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_gamma.bin", gamma_size, gamma, gamma_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_beta.bin", beta_size, beta, beta_size);
    GroupNormSwishGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 2;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 32;                       // 2
    tilingDatafromBin->HXW = 4096;                   // 3
    tilingDatafromBin->G = 8;                        // 4
    tilingDatafromBin->NXG = 8;                      // 5
    tilingDatafromBin->C_G = 4;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 8;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 1;       // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 4;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 1;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 32;           // 16
    tilingDatafromBin->stage2CoreUsed = 1;           // 17
    tilingDatafromBin->castEleNum = 64;              // 18
    tilingDatafromBin->tailCastNum = 32;             // 19
    tilingDatafromBin->coreBatchParts = 1;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dgamma_is_require = 1;        // 23
    tilingDatafromBin->dbeta_is_require = 1;         // 24
    tilingDatafromBin->swish_scale = 1.0;            // 25
    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(
        group_norm_swish_grad, blockDim, dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(group_norm_swish_grad_test, test_case_mode3_bf16)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t HXW = 128 * 128;
    uint32_t G = 8;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(float);
    size_t mean_size = NXG * sizeof(float);
    size_t rstd_size = NXG * sizeof(float);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(float);
    size_t beta_size = C * sizeof(float);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormSwishGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(beta_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    system(
        "cp -r "
        "../../../../norm/group_norm_swish_grad/tests/ut/op_kernel/group_norm_swish_grad_data ./");
    system("chmod -R 755 ./group_norm_swish_grad_data/");
    system("cd ./group_norm_swish_grad_data/ && rm -rf ./*bin");
    system("cd ./group_norm_swish_grad_data/ && python3 gen_data.py 1 32 8192 8 float16");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/group_norm_swish_grad_data/input_dy.bin", dy_size, dy, dy_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_mean.bin", mean_size, mean, mean_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_rstd.bin", rstd_size, rstd, rstd_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_gamma.bin", gamma_size, gamma, gamma_size);
    ReadFile(path + "/group_norm_swish_grad_data/input_beta.bin", beta_size, beta, beta_size);
    GroupNormSwishGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 2;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 32;                       // 2
    tilingDatafromBin->HXW = 16384;                  // 3
    tilingDatafromBin->G = 8;                        // 4
    tilingDatafromBin->NXG = 8;                      // 5
    tilingDatafromBin->C_G = 4;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 8;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 0;       // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 0;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 0;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 6400; // 13
    tilingDatafromBin->mode2_ub_iteration_num = 3;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 3584;     // 15
    tilingDatafromBin->workSpaceSize = 32;           // 16
    tilingDatafromBin->stage2CoreUsed = 1;           // 17
    tilingDatafromBin->castEleNum = 64;              // 18
    tilingDatafromBin->tailCastNum = 32;             // 19
    tilingDatafromBin->coreBatchParts = 1;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dgamma_is_require = 1;        // 23
    tilingDatafromBin->dbeta_is_require = 1;         // 24
    tilingDatafromBin->swish_scale = 1.0;            // 25
    ICPU_SET_TILING_KEY(12);
    ICPU_RUN_KF(
        group_norm_swish_grad, blockDim, dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}
