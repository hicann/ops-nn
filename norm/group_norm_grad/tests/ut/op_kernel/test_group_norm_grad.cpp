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
#include "group_norm_grad_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;
// using namespace AscendC;

extern "C" __global__ __aicore__ void group_norm_grad(
    GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
    GM_ADDR workspace, GM_ADDR tilingdata);
class group_norm_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "group_norm_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "group_norm_grad_test TearDown\n" << endl;
    }
};

TEST_F(group_norm_grad_test, test_case_mode0_align)
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

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    char* path_ = get_current_dir_name();
    string path(path_);

    GroupNormGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 0;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 32;                       // 2
    tilingDatafromBin->HXW = 576;                    // 3
    tilingDatafromBin->G = 8;                        // 4
    tilingDatafromBin->NXG = 8;                      // 5
    tilingDatafromBin->C_G = 4;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 8;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 28;      // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 1;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 4;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 0;            // 16
    tilingDatafromBin->stage2CoreUsed = 0;           // 17
    tilingDatafromBin->castEleNum = 0;               // 18
    tilingDatafromBin->tailCastNum = 0;              // 19
    tilingDatafromBin->coreBatchParts = 0;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dx_is_require = 1;            // 23
    tilingDatafromBin->dgamma_is_require = 1;        // 24
    tilingDatafromBin->dbeta_is_require = 1;         // 25
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        group_norm_grad, blockDim, dy, mean, rstd, x, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(group_norm_grad_test, test_case_mode0_not_align)
{
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t HXW = 512;
    uint32_t G = 8;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(float);
    size_t mean_size = NXG * sizeof(float);
    size_t rstd_size = NXG * sizeof(float);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(float);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    char* path_ = get_current_dir_name();
    string path(path_);

    GroupNormGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 0;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 32;                       // 2
    tilingDatafromBin->HXW = 512;                    // 3
    tilingDatafromBin->G = 8;                        // 4
    tilingDatafromBin->NXG = 8;                      // 5
    tilingDatafromBin->C_G = 4;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 8;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 28;      // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 1;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 4;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 0;            // 16
    tilingDatafromBin->stage2CoreUsed = 0;           // 17
    tilingDatafromBin->castEleNum = 0;               // 18
    tilingDatafromBin->tailCastNum = 0;              // 19
    tilingDatafromBin->coreBatchParts = 0;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dx_is_require = 1;            // 23
    tilingDatafromBin->dgamma_is_require = 1;        // 24
    tilingDatafromBin->dbeta_is_require = 1;         // 25
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        group_norm_grad, blockDim, dy, mean, rstd, x, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(group_norm_grad_test, test_case_mode1)
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

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    char* path_ = get_current_dir_name();
    string path(path_);

    GroupNormGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormGradTilingData*>(tiling);
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
    tilingDatafromBin->castEleNum = 0;               // 18
    tilingDatafromBin->tailCastNum = 0;              // 19
    tilingDatafromBin->coreBatchParts = 0;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dx_is_require = 1;            // 23
    tilingDatafromBin->dgamma_is_require = 1;        // 24
    tilingDatafromBin->dbeta_is_require = 1;         // 25
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        group_norm_grad, blockDim, dy, mean, rstd, x, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(group_norm_grad_test, test_case_mode2)
{
    uint32_t N = 1;
    uint32_t C = 40;
    uint32_t HXW = 64 * 64;
    uint32_t G = 8;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(float);
    size_t mean_size = NXG * sizeof(float);
    size_t rstd_size = NXG * sizeof(float);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(float);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    char* path_ = get_current_dir_name();
    string path(path_);

    GroupNormGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 2;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 40;                       // 2
    tilingDatafromBin->HXW = 4096;                   // 3
    tilingDatafromBin->G = 8;                        // 4
    tilingDatafromBin->NXG = 8;                      // 5
    tilingDatafromBin->C_G = 5;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 8;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 3;       // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 2;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 2;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 0;            // 16
    tilingDatafromBin->stage2CoreUsed = 0;           // 17
    tilingDatafromBin->castEleNum = 0;               // 18
    tilingDatafromBin->tailCastNum = 0;              // 19
    tilingDatafromBin->coreBatchParts = 0;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dx_is_require = 1;            // 23
    tilingDatafromBin->dgamma_is_require = 1;        // 24
    tilingDatafromBin->dbeta_is_require = 1;         // 25
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        group_norm_grad, blockDim, dy, mean, rstd, x, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(group_norm_grad_test, test_case_mode3)
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

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    char* path_ = get_current_dir_name();
    string path(path_);

    GroupNormGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 3;                // 0
    tilingDatafromBin->N = 1;                         // 1
    tilingDatafromBin->C = 32;                        // 2
    tilingDatafromBin->HXW = 16354;                   // 3
    tilingDatafromBin->G = 8;                         // 4
    tilingDatafromBin->NXG = 8;                       // 5
    tilingDatafromBin->C_G = 4;                       // 6
    tilingDatafromBin->task_num_per_core = 1;         // 7
    tilingDatafromBin->task_num_per_tail_core = 1;    // 8
    tilingDatafromBin->tail_core = 8;                 // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 0;        // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 0;       // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 0;       // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 16264; // 13
    tilingDatafromBin->mode2_ub_iteration_num = 2;    // 14
    tilingDatafromBin->mode2_ub_tail_num = 120;       // 15
    tilingDatafromBin->workSpaceSize = 0;             // 16
    tilingDatafromBin->stage2CoreUsed = 0;            // 17
    tilingDatafromBin->castEleNum = 0;                // 18
    tilingDatafromBin->tailCastNum = 0;               // 19
    tilingDatafromBin->coreBatchParts = 0;            // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0;  // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;         // 22
    tilingDatafromBin->dx_is_require = 1;             // 23
    tilingDatafromBin->dgamma_is_require = 1;         // 24
    tilingDatafromBin->dbeta_is_require = 1;          // 25
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        group_norm_grad, blockDim, dy, mean, rstd, x, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(group_norm_grad_test, test_case_mode5_float32_deterministic)
{
    uint32_t N = 32;
    uint32_t C = 64;
    uint32_t HXW = 1;
    uint32_t G = 64;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(float);
    size_t mean_size = NXG * sizeof(float);
    size_t rstd_size = NXG * sizeof(float);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(float);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(float);
    size_t dgamma_size = C * sizeof(float);
    size_t dbeta_size = C * sizeof(float);
    size_t tiling_data_size = sizeof(GroupNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16 + 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;

    char* path_ = get_current_dir_name();
    string path(path_);

    GroupNormGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 5;               // 0
    tilingDatafromBin->N = 32;                       // 1
    tilingDatafromBin->C = 64;                       // 2
    tilingDatafromBin->HXW = 1;                      // 3
    tilingDatafromBin->G = 64;                       // 4
    tilingDatafromBin->NXG = 2048;                   // 5
    tilingDatafromBin->C_G = 1;                      // 6
    tilingDatafromBin->task_num_per_core = 43;       // 7
    tilingDatafromBin->task_num_per_tail_core = 42;  // 8
    tilingDatafromBin->tail_core = 32;               // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 2033;    // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 1;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 1;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 1024;         // 16
    tilingDatafromBin->stage2CoreUsed = 1;           // 17
    tilingDatafromBin->castEleNum = 64;              // 18
    tilingDatafromBin->tailCastNum = 64;             // 19
    tilingDatafromBin->coreBatchParts = 8;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 8; // 21
    tilingDatafromBin->repeatTime4Stage2 = 1;        // 22
    tilingDatafromBin->dx_is_require = 1;            // 23
    tilingDatafromBin->dgamma_is_require = 1;        // 24
    tilingDatafromBin->dbeta_is_require = 1;         // 25
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(
        group_norm_grad, blockDim, dy, mean, rstd, x, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(group_norm_grad_test, test_case_mode0_bf16)
{
    uint32_t N = 1;
    uint32_t C = 128;
    uint32_t HXW = 64;
    uint32_t G = 4;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(half);
    size_t mean_size = NXG * sizeof(half);
    size_t rstd_size = NXG * sizeof(half);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(half);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(half);
    size_t dgamma_size = C * sizeof(half);
    size_t dbeta_size = C * sizeof(half);
    size_t tiling_data_size = sizeof(GroupNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;

    char* path_ = get_current_dir_name();
    string path(path_);

    GroupNormGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 0;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 128;                       // 2
    tilingDatafromBin->HXW = 64;                    // 3
    tilingDatafromBin->G = 4;                        // 4
    tilingDatafromBin->NXG = 4;                      // 5
    tilingDatafromBin->C_G = 32;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 4;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 169;      // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 1;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 32;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 128;            // 16
    tilingDatafromBin->stage2CoreUsed = 4;           // 17
    tilingDatafromBin->castEleNum = 64;               // 18
    tilingDatafromBin->tailCastNum = 64;              // 19
    tilingDatafromBin->coreBatchParts = 0;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dx_is_require = 1;            // 23
    tilingDatafromBin->dgamma_is_require = 1;        // 24
    tilingDatafromBin->dbeta_is_require = 1;         // 25
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        group_norm_grad, blockDim, dy, mean, rstd, x, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(group_norm_grad_test, test_case_mode1_bf16)
{
    uint32_t N = 1;
    uint32_t C = 128;
    uint32_t HXW = 64;
    uint32_t G = 4;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(half);
    size_t mean_size = NXG * sizeof(half);
    size_t rstd_size = NXG * sizeof(half);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(half);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(half);
    size_t dgamma_size = C * sizeof(half);
    size_t dbeta_size = C * sizeof(half);
    size_t tiling_data_size = sizeof(GroupNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;

    char* path_ = get_current_dir_name();
    string path(path_);

    GroupNormGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 1;               // 0
    tilingDatafromBin->N = 1;                        // 1
    tilingDatafromBin->C = 128;                       // 2
    tilingDatafromBin->HXW = 64;                   // 3
    tilingDatafromBin->G = 4;                        // 4
    tilingDatafromBin->NXG = 4;                      // 5
    tilingDatafromBin->C_G = 32;                      // 6
    tilingDatafromBin->task_num_per_core = 1;        // 7
    tilingDatafromBin->task_num_per_tail_core = 1;   // 8
    tilingDatafromBin->tail_core = 4;                // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 169;       // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 1;      // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 32;      // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 0;    // 13
    tilingDatafromBin->mode2_ub_iteration_num = 0;   // 14
    tilingDatafromBin->mode2_ub_tail_num = 0;        // 15
    tilingDatafromBin->workSpaceSize = 128;            // 16
    tilingDatafromBin->stage2CoreUsed = 4;           // 17
    tilingDatafromBin->castEleNum = 64;               // 18
    tilingDatafromBin->tailCastNum = 64;              // 19
    tilingDatafromBin->coreBatchParts = 0;           // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0; // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;        // 22
    tilingDatafromBin->dx_is_require = 1;            // 23
    tilingDatafromBin->dgamma_is_require = 1;        // 24
    tilingDatafromBin->dbeta_is_require = 1;         // 25
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        group_norm_grad, blockDim, dy, mean, rstd, x, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(group_norm_grad_test, test_case_mode3_bf16)
{
    uint32_t N = 1;
    uint32_t C = 64;
    uint32_t HXW = 128 * 128;
    uint32_t G = 2;
    uint32_t NXG = N * G;
    uint32_t C_G = C / G;

    size_t dy_size = N * C * HXW * sizeof(half);
    size_t mean_size = NXG * sizeof(half);
    size_t rstd_size = NXG * sizeof(half);
    size_t x_size = dy_size;
    size_t gamma_size = C * sizeof(half);

    // outputs
    size_t dx_size = N * C * HXW * sizeof(half);
    size_t dgamma_size = C * sizeof(half);
    size_t dbeta_size = C * sizeof(half);
    size_t tiling_data_size = sizeof(GroupNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dy_size);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(mean_size);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstd_size);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgamma_size);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbeta_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    GroupNormGradTilingData* tilingDatafromBin = reinterpret_cast<GroupNormGradTilingData*>(tiling);
    tilingDatafromBin->Tiling_key = 3;                // 0
    tilingDatafromBin->N = 1;                         // 1
    tilingDatafromBin->C = 64;                        // 2
    tilingDatafromBin->HXW = 16384;                   // 3
    tilingDatafromBin->G = 2;                         // 4
    tilingDatafromBin->NXG = 2;                       // 5
    tilingDatafromBin->C_G = 32;                       // 6
    tilingDatafromBin->task_num_per_core = 1;         // 7
    tilingDatafromBin->task_num_per_tail_core = 1;    // 8
    tilingDatafromBin->tail_core = 2;                 // 9
    tilingDatafromBin->mode1_ub_cap_C_num = 0;        // 10
    tilingDatafromBin->mode1_ub_iter_C_num = 0;       // 11
    tilingDatafromBin->mode1_ub_tail_C_num = 0;       // 12
    tilingDatafromBin->mode2_ub_capacity_ele = 10752; // 13
    tilingDatafromBin->mode2_ub_iteration_num = 2;    // 14
    tilingDatafromBin->mode2_ub_tail_num = 5632;      // 15
    tilingDatafromBin->workSpaceSize = 64;             // 16
    tilingDatafromBin->stage2CoreUsed = 1;            // 17
    tilingDatafromBin->castEleNum = 64;                // 18
    tilingDatafromBin->tailCastNum = 64;               // 19
    tilingDatafromBin->coreBatchParts = 0;            // 20
    tilingDatafromBin->coreBatchPartsTailRepeat = 0;  // 21
    tilingDatafromBin->repeatTime4Stage2 = 0;         // 22
    tilingDatafromBin->dx_is_require = 1;             // 23
    tilingDatafromBin->dgamma_is_require = 1;         // 24
    tilingDatafromBin->dbeta_is_require = 1;          // 25
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        group_norm_grad, blockDim, dy, mean, rstd, x, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}
