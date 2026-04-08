/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "ada_layer_norm_grad_tiling_def.h"
#include "tikicpulib.h"

using namespace std;

extern "C" void ada_layer_norm_grad(
    uint8_t* dy, uint8_t* x, uint8_t* rstd, uint8_t* mean, uint8_t* scale, uint8_t* gamma, uint8_t* beta, uint8_t* dx, uint8_t* dscale, 
    uint8_t* dshift, uint8_t* dgamma, uint8_t* dbeta, uint8_t* workspace, uint8_t* tiling);

class ada_layer_norm_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ada_layer_norm_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ada_layer_norm_grad_test TearDown\n" << endl;
    }
};

TEST_F(ada_layer_norm_grad_test, test_case_workspace_201)
{
    size_t batch = 2;
    size_t seq = 9;
    size_t row = 18;
    size_t col = 20;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t rstdByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t scaleByteSize = batch * col * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t betaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dscaleByteSize = batch * col * sizeof(float);
    size_t dshiftByteSize = batch * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(AdaLayerNormGradTilingDataWorkspace);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dscale = (uint8_t*)AscendC::GmAlloc(dscaleByteSize);
    uint8_t* dshift = (uint8_t*)AscendC::GmAlloc(dshiftByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(2 * col * 4 * (2 + 4) + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AdaLayerNormGradTilingDataWorkspace* tilingDatafromBin =
        reinterpret_cast<AdaLayerNormGradTilingDataWorkspace*>(tiling);

    tilingDatafromBin->batch = 2;
    tilingDatafromBin->seq = 9;
    tilingDatafromBin->row = 18;
    tilingDatafromBin->col = 20;
    tilingDatafromBin->blockNum = 2;
    tilingDatafromBin->blockFormer = 9;
    tilingDatafromBin->blockTail = 9;
    tilingDatafromBin->ubLoop = 1;
    tilingDatafromBin->ubFormer = 1;
    tilingDatafromBin->ubTail = 1;
    tilingDatafromBin->colAlignM = 1;
    tilingDatafromBin->colAlignV = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t blockDim = tilingDatafromBin->blockNum;

    ICPU_SET_TILING_KEY(301);
    ICPU_RUN_KF(
        ada_layer_norm_grad, blockDim, dy, x, rstd, mean, scale, gamma, beta, dx, dscale, dshift, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(scale);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dscale);
    AscendC::GmFree(dshift);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(ada_layer_norm_grad_test, test_case_common_401)
{
    size_t batch = 2;
    size_t seq = 2;
    size_t row = 4;
    size_t col = 512;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t rstdByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t scaleByteSize = batch * col * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t betaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dscaleByteSize = batch * col * sizeof(float);
    size_t dshiftByteSize = batch * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(AdaLayerNormGradTilingDataCommon);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dscale = (uint8_t*)AscendC::GmAlloc(dscaleByteSize);
    uint8_t* dshift = (uint8_t*)AscendC::GmAlloc(dshiftByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(row * col * sizeof(float) * 4 + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AdaLayerNormGradTilingDataCommon* tilingDatafromBin = reinterpret_cast<AdaLayerNormGradTilingDataCommon*>(tiling);

    tilingDatafromBin->batch = 2;
    tilingDatafromBin->seq = 2;
    tilingDatafromBin->row = 4;
    tilingDatafromBin->col = 512;
    tilingDatafromBin->colAlignM = 512;
    tilingDatafromBin->colAlignV = 512;
    tilingDatafromBin->blockNum = 2;
    tilingDatafromBin->blockFormer = 2;
    tilingDatafromBin->blockTail = 2;
    tilingDatafromBin->ubFormer = 2;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 1;
    tilingDatafromBin->ubTailOfTailBlock = 1;
    tilingDatafromBin->wholeBufferBytes = 512 * 4 * 2;
    tilingDatafromBin->lastRBufferBytes = 512 * 4;
    tilingDatafromBin->blockFormerScaleBufferBytes = 512 * 4 * 2;
    tilingDatafromBin->nlastRBufferBytes = 512 * 4 * 2;
    tilingDatafromBin->lastBrcbBufferBytes = 512 * 4;
    tilingDatafromBin->wholeBufferElemNums = 512 * 2;
    tilingDatafromBin->blockFormerScaleBufferElemNums = 512 * 2;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t blockDim = tilingDatafromBin->blockNum;

    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(
        ada_layer_norm_grad, blockDim, dy, x, rstd, mean, scale, gamma, beta, dx, dscale, dshift, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(scale);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dscale);
    AscendC::GmFree(dshift);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}