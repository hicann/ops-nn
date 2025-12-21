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
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "data_utils.h"
#include "layer_norm_grad_v3_tiling_def.h"
#include "tikicpulib.h"

using namespace std;

extern "C" void layer_norm_grad_v3(
    uint8_t* dy, uint8_t* x, uint8_t* rstd, uint8_t* mean, uint8_t* gamma, uint8_t* dx, uint8_t* dgamma, uint8_t* dbeta,
    uint8_t* workspace, uint8_t* tiling);

class layer_norm_grad_v3_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "layer_norm_grad_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "layer_norm_grad_v3_test TearDown\n" << endl;
    }
};

TEST_F(layer_norm_grad_v3_test, test_case_1)
{
    size_t row = 1;
    size_t col = 1024;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t rstdByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormGradV3TilingDataSingleRead);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(row * col * sizeof(float) * 2 + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormGradV3TilingDataSingleRead* tilingDatafromBin =
        reinterpret_cast<LayerNormGradV3TilingDataSingleRead*>(tiling);

    tilingDatafromBin->row = 1;
    tilingDatafromBin->col = 1024;
    tilingDatafromBin->colAlignM = 1024;
    tilingDatafromBin->colAlignV = 1024;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->blockFormer = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->ubFormer = 1;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 1;
    tilingDatafromBin->ubTailOfTailBlock = 1;
    tilingDatafromBin->bufferElemNums = 1024;

    uint32_t blockDim = tilingDatafromBin->blockNum;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(
        layer_norm_grad_v3, blockDim, dy, x, rstd, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(102);
    ICPU_RUN_KF(
        layer_norm_grad_v3, blockDim, dy, x, rstd, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(103);
    ICPU_RUN_KF(
        layer_norm_grad_v3, blockDim, dy, x, rstd, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(104);
    ICPU_RUN_KF(
        layer_norm_grad_v3, blockDim, dy, x, rstd, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(105);
    ICPU_RUN_KF(
        layer_norm_grad_v3, blockDim, dy, x, rstd, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_grad_v3_test, test_case_2)
{
    size_t row = 1;
    size_t col = 1024;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t rstdByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormGradV3TilingDataSingleRead);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(row * col * sizeof(float) * 2 + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormGradV3TilingDataSingleRead* tilingDatafromBin =
        reinterpret_cast<LayerNormGradV3TilingDataSingleRead*>(tiling);

    tilingDatafromBin->row = 1;
    tilingDatafromBin->col = 1024;
    tilingDatafromBin->colAlignM = 1024;
    tilingDatafromBin->colAlignV = 1024;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->blockFormer = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->ubFormer = 1;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 1;
    tilingDatafromBin->ubTailOfTailBlock = 1;
    tilingDatafromBin->bufferElemNums = 1024;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t blockDim = tilingDatafromBin->blockNum;

    ICPU_SET_TILING_KEY(102);
    ICPU_RUN_KF(
        layer_norm_grad_v3, blockDim, dy, x, rstd, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_grad_v3_test, test_case_workspace_201)
{
    size_t row = 18;
    size_t col = 20;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t rstdByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormGradV3TilingDataWorkspace);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(18 * col * 4 * 4 + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormGradV3TilingDataWorkspace* tilingDatafromBin =
        reinterpret_cast<LayerNormGradV3TilingDataWorkspace*>(tiling);

    tilingDatafromBin->row = 18;
    tilingDatafromBin->col = 20;
    tilingDatafromBin->blockNum = 18;
    tilingDatafromBin->blockFormer = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->ubLoop = 1;
    tilingDatafromBin->ubFormer = 1;
    tilingDatafromBin->ubTail = 1;
    tilingDatafromBin->colAlignM = 1;
    tilingDatafromBin->colAlignV = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t blockDim = tilingDatafromBin->blockNum;

    ICPU_SET_TILING_KEY(201);
    ICPU_RUN_KF(
        layer_norm_grad_v3, blockDim, dy, x, rstd, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_grad_v3_test, test_case_workspace_211)
{
    size_t row = 18;
    size_t col = 20;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t rstdByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormGradV3TilingDataWorkspace);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(18 * col * 4 * 4 + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormGradV3TilingDataWorkspace* tilingDatafromBin =
        reinterpret_cast<LayerNormGradV3TilingDataWorkspace*>(tiling);

    tilingDatafromBin->row = 18;
    tilingDatafromBin->col = 20;
    tilingDatafromBin->blockNum = 18;
    tilingDatafromBin->blockFormer = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->ubLoop = 2;
    tilingDatafromBin->ubFormer = 1;
    tilingDatafromBin->ubTail = 1;
    tilingDatafromBin->colAlignM = 1;
    tilingDatafromBin->colAlignV = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t blockDim = tilingDatafromBin->blockNum;

    ICPU_SET_TILING_KEY(211);
    ICPU_RUN_KF(
        layer_norm_grad_v3, blockDim, dy, x, rstd, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_grad_v3_test, test_case_common_401)
{
    size_t row = 1;
    size_t col = 512;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t rstdByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormGradV3TilingDataCommon);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(row * col * sizeof(float) * 2 + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormGradV3TilingDataCommon* tilingDatafromBin = reinterpret_cast<LayerNormGradV3TilingDataCommon*>(tiling);

    tilingDatafromBin->row = 1;
    tilingDatafromBin->col = 512;
    tilingDatafromBin->colAlignM = 512;
    tilingDatafromBin->colAlignV = 512;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->blockFormer = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->ubFormer = 1;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 1;
    tilingDatafromBin->ubTailOfTailBlock = 1;
    tilingDatafromBin->wholeBufferBytes = 512 * 4;
    tilingDatafromBin->lastRBufferBytes = 512 * 4;
    tilingDatafromBin->nlastRBufferBytes = 512 * 4;
    tilingDatafromBin->lastBrcbBufferBytes = 512 * 4;
    tilingDatafromBin->wholeBufferElemNums = 512 * 4;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t blockDim = tilingDatafromBin->blockNum;

    ICPU_SET_TILING_KEY(401);
    ICPU_RUN_KF(
        layer_norm_grad_v3, blockDim, dy, x, rstd, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_grad_v3_test, test_case_common_311)
{
    size_t row = 1;
    size_t col = 60;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t rstdByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormGradV3TilingDataCommon);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(row * col * sizeof(float) * 2 + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormGradV3TilingDataCommon* tilingDatafromBin = reinterpret_cast<LayerNormGradV3TilingDataCommon*>(tiling);

    tilingDatafromBin->row = 1;
    tilingDatafromBin->col = 60;
    tilingDatafromBin->colAlignM = 64;
    tilingDatafromBin->colAlignV = 64;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->blockFormer = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->ubFormer = 1;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 1;
    tilingDatafromBin->ubTailOfTailBlock = 1;
    tilingDatafromBin->wholeBufferBytes = 512 * 4;
    tilingDatafromBin->lastRBufferBytes = 512 * 4;
    tilingDatafromBin->nlastRBufferBytes = 512 * 4;
    tilingDatafromBin->lastBrcbBufferBytes = 512 * 4;
    tilingDatafromBin->wholeBufferElemNums = 512 * 4;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t blockDim = tilingDatafromBin->blockNum;

    ICPU_SET_TILING_KEY(311);
    ICPU_RUN_KF(
        layer_norm_grad_v3, blockDim, dy, x, rstd, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
