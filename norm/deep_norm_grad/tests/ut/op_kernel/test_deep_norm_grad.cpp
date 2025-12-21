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
#include "deep_norm_grad_tiling_def.h"
#include "data_utils.h"

using namespace std;

class deep_norm_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "deep_norm_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "deep_norm_grad_test TearDown\n" << endl;
    }
};

extern "C" __global__ __aicore__ void deep_norm_grad(
    uint8_t* dy, uint8_t* x, uint8_t* gx, uint8_t* gamma, uint8_t* mean, uint8_t* rstd, uint8_t* dx, uint8_t* dgx,
    uint8_t* dbeta, uint8_t* dgamma, uint8_t* workspace, uint8_t* tiling);

TEST_F(deep_norm_grad_test, test_case_0)
{ // tiling key 0
    size_t N = 2;
    size_t D = 1024;
    size_t dyByteSize = N * D * sizeof(float);
    size_t xByteSize = N * D * sizeof(float);
    size_t gxByteSize = N * D * sizeof(float);
    size_t gammaByteSize = D * sizeof(float);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t dxByteSize = N * D * sizeof(float);
    size_t dgxByteSize = N * D * sizeof(float);
    size_t dbetaByteSize = D * sizeof(float);
    size_t dgammaByteSize = D * sizeof(float);
    size_t tiling_data_size = sizeof(DeepNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(gxByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgx = (uint8_t*)AscendC::GmAlloc(dgxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(50 * 32 + 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormGradTilingData* tilingDatafromBin = reinterpret_cast<DeepNormGradTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 2;
    tilingDatafromBin->nDimNum = N;
    tilingDatafromBin->dDimNum = D;
    tilingDatafromBin->nDealPerCore = 1;
    tilingDatafromBin->nDealLastCore = 1;
    tilingDatafromBin->mergeNCount = 8;
    tilingDatafromBin->cutDTime = 1;
    tilingDatafromBin->cutDPerTime = 6984;
    tilingDatafromBin->cutDLastTime = 1024;
    tilingDatafromBin->alpha = 0.3;
    tilingDatafromBin->fixedOutputFlag = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        deep_norm_grad, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_grad_test, test_case_1)
{ // tiling key 1
    size_t N = 2;
    size_t D = 8192;
    size_t dyByteSize = N * D * sizeof(float);
    size_t xByteSize = N * D * sizeof(float);
    size_t gxByteSize = N * D * sizeof(float);
    size_t gammaByteSize = D * sizeof(float);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t dxByteSize = N * D * sizeof(float);
    size_t dgxByteSize = N * D * sizeof(float);
    size_t dbetaByteSize = D * sizeof(float);
    size_t dgammaByteSize = D * sizeof(float);
    size_t tiling_data_size = sizeof(DeepNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(gxByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgx = (uint8_t*)AscendC::GmAlloc(dgxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(36 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormGradTilingData* tilingDatafromBin = reinterpret_cast<DeepNormGradTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 2;
    tilingDatafromBin->nDimNum = N;
    tilingDatafromBin->dDimNum = D;
    tilingDatafromBin->nDealPerCore = 1;
    tilingDatafromBin->nDealLastCore = 1;
    tilingDatafromBin->mergeNCount = 1;
    tilingDatafromBin->cutDTime = 2;
    tilingDatafromBin->cutDPerTime = 6104;
    tilingDatafromBin->cutDLastTime = 2088;
    tilingDatafromBin->alpha = 0;
    tilingDatafromBin->fixedOutputFlag = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        deep_norm_grad, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_grad_test, test_case_10)
{ // tiling key 10
    size_t N = 2;
    size_t D = 1024;
    size_t dyByteSize = N * D * sizeof(int16_t);
    size_t xByteSize = N * D * sizeof(int16_t);
    size_t gxByteSize = N * D * sizeof(int16_t);
    size_t gammaByteSize = D * sizeof(int16_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t dxByteSize = N * D * sizeof(int16_t);
    size_t dgxByteSize = N * D * sizeof(int16_t);
    size_t dbetaByteSize = D * sizeof(float);
    size_t dgammaByteSize = D * sizeof(float);
    size_t tiling_data_size = sizeof(DeepNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(gxByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgx = (uint8_t*)AscendC::GmAlloc(dgxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(50 * 32 + 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormGradTilingData* tilingDatafromBin = reinterpret_cast<DeepNormGradTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 2;
    tilingDatafromBin->nDimNum = N;
    tilingDatafromBin->dDimNum = D;
    tilingDatafromBin->nDealPerCore = 1;
    tilingDatafromBin->nDealLastCore = 1;
    tilingDatafromBin->mergeNCount = 8;
    tilingDatafromBin->cutDTime = 1;
    tilingDatafromBin->cutDPerTime = 1024;
    tilingDatafromBin->cutDLastTime = 1024;
    tilingDatafromBin->alpha = 0.3;
    tilingDatafromBin->fixedOutputFlag = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(
        deep_norm_grad, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_grad_test, test_case_11)
{ // tiling key 11
    size_t N = 40;
    size_t D = 8192;
    size_t dyByteSize = N * D * sizeof(int16_t);
    size_t xByteSize = N * D * sizeof(int16_t);
    size_t gxByteSize = N * D * sizeof(int16_t);
    size_t gammaByteSize = D * sizeof(int16_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t dxByteSize = N * D * sizeof(int16_t);
    size_t dgxByteSize = N * D * sizeof(int16_t);
    size_t dbetaByteSize = D * sizeof(float);
    size_t dgammaByteSize = D * sizeof(float);
    size_t tiling_data_size = sizeof(DeepNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(gxByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgx = (uint8_t*)AscendC::GmAlloc(dgxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(36 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormGradTilingData* tilingDatafromBin = reinterpret_cast<DeepNormGradTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 40;
    tilingDatafromBin->nDimNum = N;
    tilingDatafromBin->dDimNum = D;
    tilingDatafromBin->nDealPerCore = 1;
    tilingDatafromBin->nDealLastCore = 1;
    tilingDatafromBin->mergeNCount = 1;
    tilingDatafromBin->cutDTime = 2;
    tilingDatafromBin->cutDPerTime = 4432;
    tilingDatafromBin->cutDLastTime = 3760;
    tilingDatafromBin->alpha = 0;
    tilingDatafromBin->fixedOutputFlag = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(
        deep_norm_grad, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_grad_test, test_case_22)
{ // tiling key 22
    size_t N = 40;
    size_t D = 133;
    size_t dyByteSize = N * D * sizeof(int16_t);
    size_t xByteSize = N * D * sizeof(int16_t);
    size_t gxByteSize = N * D * sizeof(int16_t);
    size_t gammaByteSize = D * sizeof(int16_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t dxByteSize = N * D * sizeof(int16_t);
    size_t dgxByteSize = N * D * sizeof(int16_t);
    size_t dbetaByteSize = D * sizeof(float);
    size_t dgammaByteSize = D * sizeof(float);
    size_t tiling_data_size = sizeof(DeepNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(gxByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgx = (uint8_t*)AscendC::GmAlloc(dgxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(36 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormGradTilingData* tilingDatafromBin = reinterpret_cast<DeepNormGradTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 40;
    tilingDatafromBin->nDimNum = N;
    tilingDatafromBin->dDimNum = D;
    tilingDatafromBin->nDealPerCore = 1;
    tilingDatafromBin->nDealLastCore = 1;
    tilingDatafromBin->mergeNCount = 25;
    tilingDatafromBin->cutDTime = 1;
    tilingDatafromBin->cutDPerTime = 133;
    tilingDatafromBin->cutDLastTime = 133;
    tilingDatafromBin->alpha = 0;
    tilingDatafromBin->fixedOutputFlag = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(22);
    ICPU_RUN_KF(
        deep_norm_grad, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_grad_test, test_case_20)
{ // tiling key 20
    size_t N = 2;
    size_t D = 1024;
    size_t dyByteSize = N * D * sizeof(int16_t);
    size_t xByteSize = N * D * sizeof(int16_t);
    size_t gxByteSize = N * D * sizeof(int16_t);
    size_t gammaByteSize = D * sizeof(int16_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t dxByteSize = N * D * sizeof(int16_t);
    size_t dgxByteSize = N * D * sizeof(int16_t);
    size_t dbetaByteSize = D * sizeof(float);
    size_t dgammaByteSize = D * sizeof(float);
    size_t tiling_data_size = sizeof(DeepNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(gxByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgx = (uint8_t*)AscendC::GmAlloc(dgxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(50 * 32 + 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormGradTilingData* tilingDatafromBin = reinterpret_cast<DeepNormGradTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 2;
    tilingDatafromBin->nDimNum = N;
    tilingDatafromBin->dDimNum = D;
    tilingDatafromBin->nDealPerCore = 1;
    tilingDatafromBin->nDealLastCore = 1;
    tilingDatafromBin->mergeNCount = 15;
    tilingDatafromBin->cutDTime = 1;
    tilingDatafromBin->cutDPerTime = 1024;
    tilingDatafromBin->cutDLastTime = 1024;
    tilingDatafromBin->alpha = 0.3;
    tilingDatafromBin->fixedOutputFlag = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(20);
    ICPU_RUN_KF(
        deep_norm_grad, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_grad_test, test_case_21)
{ // tiling key 21
    size_t N = 2;
    size_t D = 8192;
    size_t dyByteSize = N * D * sizeof(int16_t);
    size_t xByteSize = N * D * sizeof(int16_t);
    size_t gxByteSize = N * D * sizeof(int16_t);
    size_t gammaByteSize = D * sizeof(int16_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t dxByteSize = N * D * sizeof(int16_t);
    size_t dgxByteSize = N * D * sizeof(int16_t);
    size_t dbetaByteSize = D * sizeof(float);
    size_t dgammaByteSize = D * sizeof(float);
    size_t tiling_data_size = sizeof(DeepNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(gxByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgx = (uint8_t*)AscendC::GmAlloc(dgxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(50 * 32 + 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormGradTilingData* tilingDatafromBin = reinterpret_cast<DeepNormGradTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 2;
    tilingDatafromBin->nDimNum = N;
    tilingDatafromBin->dDimNum = D;
    tilingDatafromBin->nDealPerCore = 1;
    tilingDatafromBin->nDealLastCore = 1;
    tilingDatafromBin->mergeNCount = 1;
    tilingDatafromBin->cutDTime = 2;
    tilingDatafromBin->cutDPerTime = 4432;
    tilingDatafromBin->cutDLastTime = 3760;
    tilingDatafromBin->alpha = 0.3;
    tilingDatafromBin->fixedOutputFlag = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(21);
    ICPU_RUN_KF(
        deep_norm_grad, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_grad_test, test_case_2)
{ // tiling key 2
    size_t N = 10;
    size_t D = 9;
    size_t dyByteSize = N * D * sizeof(float);
    size_t xByteSize = N * D * sizeof(float);
    size_t gxByteSize = N * D * sizeof(float);
    size_t gammaByteSize = D * sizeof(float);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t dxByteSize = N * D * sizeof(float);
    size_t dgxByteSize = N * D * sizeof(float);
    size_t dbetaByteSize = D * sizeof(float);
    size_t dgammaByteSize = D * sizeof(float);
    size_t tiling_data_size = sizeof(DeepNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(gxByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgx = (uint8_t*)AscendC::GmAlloc(dgxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(50 * 32 + 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 10;

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormGradTilingData* tilingDatafromBin = reinterpret_cast<DeepNormGradTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 10;
    tilingDatafromBin->nDimNum = N;
    tilingDatafromBin->dDimNum = D;
    tilingDatafromBin->nDealPerCore = 1;
    tilingDatafromBin->nDealLastCore = 1;
    tilingDatafromBin->mergeNCount = 337;
    tilingDatafromBin->cutDTime = 1;
    tilingDatafromBin->cutDPerTime = 9;
    tilingDatafromBin->cutDLastTime = 9;
    tilingDatafromBin->alpha = 0;
    tilingDatafromBin->fixedOutputFlag = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        deep_norm_grad, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(deep_norm_grad_test, test_case_12)
{ // tiling key 12
    size_t N = 10;
    size_t D = 9;
    size_t dyByteSize = N * D * sizeof(int16_t);
    size_t xByteSize = N * D * sizeof(int16_t);
    size_t gxByteSize = N * D * sizeof(int16_t);
    size_t gammaByteSize = D * sizeof(int16_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t dxByteSize = N * D * sizeof(int16_t);
    size_t dgxByteSize = N * D * sizeof(int16_t);
    size_t dbetaByteSize = D * sizeof(float);
    size_t dgammaByteSize = D * sizeof(float);
    size_t tiling_data_size = sizeof(DeepNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gx = (uint8_t*)AscendC::GmAlloc(gxByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgx = (uint8_t*)AscendC::GmAlloc(dgxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(50 * 32 + 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    DeepNormGradTilingData* tilingDatafromBin = reinterpret_cast<DeepNormGradTilingData*>(tiling);

    tilingDatafromBin->useCoreNum = 10;
    tilingDatafromBin->nDimNum = N;
    tilingDatafromBin->dDimNum = D;
    tilingDatafromBin->nDealPerCore = 1;
    tilingDatafromBin->nDealLastCore = 1;
    tilingDatafromBin->mergeNCount = 209;
    tilingDatafromBin->cutDTime = 1;
    tilingDatafromBin->cutDPerTime = 9;
    tilingDatafromBin->cutDLastTime = 9;
    tilingDatafromBin->alpha = 0.3;
    tilingDatafromBin->fixedOutputFlag = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(12);
    ICPU_RUN_KF(
        deep_norm_grad, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}