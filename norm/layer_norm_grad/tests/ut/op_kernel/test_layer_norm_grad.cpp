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
#include "layer_norm_grad_tiling_def.h"
#include "tikicpulib.h"

using namespace std;

extern "C" void layer_norm_grad(
    uint8_t* dy, uint8_t* x, uint8_t* variance, uint8_t* mean, uint8_t* gamma, uint8_t* dx, uint8_t* dgamma, uint8_t* dbeta,
    uint8_t* workspace, uint8_t* tiling);

class layer_norm_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "layer_norm_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "layer_norm_grad_test TearDown\n" << endl;
    }
};

TEST_F(layer_norm_grad_test, test_case_float32_recompute)
{
    size_t row = 1;
    size_t col = 1024;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t varianceByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormGradTilingDataRecompute);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(varianceByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(row * col * sizeof(float) * 2 + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormGradTilingDataRecompute* tilingDatafromBin =
        reinterpret_cast<LayerNormGradTilingDataRecompute*>(tiling);

    tilingDatafromBin->row = 1;
    tilingDatafromBin->col = 1024;
    tilingDatafromBin->gammaBetaMainBlockFactor = 64;
    tilingDatafromBin->gammaBetaNloopMainBlock = 1;
    tilingDatafromBin->gammaBetaNtailMainBlock = 64;
    tilingDatafromBin->gammaBetaNloopTailBlock = 1;
    tilingDatafromBin->gammaBetaNtailTailBlock = 64;
    tilingDatafromBin->gammaBetaMtail = 1;
    tilingDatafromBin->gammaBetaBasicBlockLoop = 0;
    tilingDatafromBin->gammaBetaMainFoldCount = 0;
    tilingDatafromBin->backwardMainBlockFactor = 1;
    tilingDatafromBin->backwardMainBlockCount = 1;
    tilingDatafromBin->backwardTailBlockCount = 0;
    tilingDatafromBin->backwardTailBlockFactor = 0;
    tilingDatafromBin->backwardMLoopMain = 1;
    tilingDatafromBin->backwardMLoopTail = 0;
    tilingDatafromBin->backwardMLoopTailTail = 0;
    tilingDatafromBin->backwardMTailTail = 0;
    tilingDatafromBin->backwardNLoopMain = 1;
    tilingDatafromBin->backwardNTotalLoopMain = 1;
    tilingDatafromBin->backwardNLoopTail = 0;
    tilingDatafromBin->backwardBasicBlockLoop = 0;
    tilingDatafromBin->backwardMainFoldCount = 1;
    tilingDatafromBin->backwardNfactorBlockAligned = 1024;
    tilingDatafromBin->backwardMfactorBlockAligned = 8;
    tilingDatafromBin->backwardCeilVLCount = 16;
    tilingDatafromBin->backwardFoldPoint = 8;
    tilingDatafromBin->backwardFoldSize = 8;
    tilingDatafromBin->gammaBetaBlockDim = 16;
    tilingDatafromBin->gammaBetaCacheBufferCount = 1;
    tilingDatafromBin->gammaBetaResultCacheID = 0;
    tilingDatafromBin->gammaBetaNfactor = 64;
    tilingDatafromBin->gammaBetaMfactor = 128;
    tilingDatafromBin->backwardBlockDim = 1;
    tilingDatafromBin->backwardMfactor = 1;
    tilingDatafromBin->backwardNfactor = 1024;
    tilingDatafromBin->backwardCacheBufferCountMain = 1;
    tilingDatafromBin->backwardResultCacheIDMain = 0;

    tilingDatafromBin->pdxIsRequire = 1;
    tilingDatafromBin->pdgammaIsRequire = 1;
    tilingDatafromBin->pdbetaIsRequire = 1;
    tilingDatafromBin->epsilon = 1e-12;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t blockDim = tilingDatafromBin->backwardBlockDim;

    ICPU_SET_TILING_KEY(500);
    ICPU_RUN_KF(
        layer_norm_grad, blockDim, dy, x, variance, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(variance);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_grad_test, test_case_float32_big_m)
{
    size_t row = 10000;
    size_t col = 256;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t varianceByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormGradTilingDataGroupedReduceBigM);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(varianceByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(row * col * sizeof(float) * 2 + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormGradTilingDataGroupedReduceBigM* tilingDatafromBin =
        reinterpret_cast<LayerNormGradTilingDataGroupedReduceBigM*>(tiling);

    tilingDatafromBin->row = 10000;
    tilingDatafromBin->col = 256;

    tilingDatafromBin->gammaBetaUsableBlocks = 64;
    tilingDatafromBin->gammaBetaMPerBlock = 156;
    tilingDatafromBin->gammaBetaMReminder = 16;
    tilingDatafromBin->gammaBetaNloop = 4;
    tilingDatafromBin->gammaBetaNtail = 0;
    tilingDatafromBin->gammaBetaMfactorBlockAligned = 64;
    tilingDatafromBin->gammaBetaNfactorBlockAligned = 64;

    tilingDatafromBin->gammabetaMToProcessMainBlock = 157;
    tilingDatafromBin->gammabetaMLoopMainBlock = 2;
    tilingDatafromBin->gammabetaMTotalLoopMainBlock = 3;
    tilingDatafromBin->gammabetaMTailMainBlock = 29;
    tilingDatafromBin->gammabetaBasicBlockLoopMainBlock = 2;
    tilingDatafromBin->gammabetaMainFoldCountMainBlock = 0;
    tilingDatafromBin->gammabetaCacheBufferCountMainBlock = 2;
    tilingDatafromBin->gammabetaResultCacheIDMainBlock = 1;

    tilingDatafromBin->gammabetaMToProcessTailBlock = 156;
    tilingDatafromBin->gammabetaMLoopTailBlock = 2;
    tilingDatafromBin->gammabetaMTotalLoopTailBlock = 3;
    tilingDatafromBin->gammabetaMTailTailBlock = 28;
    tilingDatafromBin->gammabetaBasicBlockLoopTailBlock = 2;
    tilingDatafromBin->gammabetaMainFoldCountTailBlock = 0;
    tilingDatafromBin->gammabetaCacheBufferCountTailBlock = 2;
    tilingDatafromBin->gammabetaResultCacheIDTailBlock = 1;

    tilingDatafromBin->gammaBetaMTailStg2 = 0;
    tilingDatafromBin->gammaBetaMBasicBlockLoopStg2 = 0;
    tilingDatafromBin->gammaBetaMMainFoldCountStg2 = 1;
    tilingDatafromBin->gammaBetaMResultCacheIDStg2 = 0;

    tilingDatafromBin->pdxIsRequire = 1;
    tilingDatafromBin->pdgammaIsRequire = 1;
    tilingDatafromBin->pdbetaIsRequire = 1;
    tilingDatafromBin->epsilon = 1e-12;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t blockDim = 64;

    ICPU_SET_TILING_KEY(600);
    ICPU_RUN_KF(
        layer_norm_grad, blockDim, dy, x, variance, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(variance);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_grad_test, test_case_float32_big_n)
{
    size_t row = 256;
    size_t col = 10000;
    size_t dyByteSize = row * col * sizeof(float);
    size_t xByteSize = row * col * sizeof(float);
    size_t varianceByteSize = row * sizeof(float);
    size_t meanByteSize = row * sizeof(float);
    size_t gammaByteSize = col * sizeof(float);
    size_t dxByteSize = row * col * sizeof(float);
    size_t dgammaByteSize = col * sizeof(float);
    size_t dbetaByteSize = col * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormGradTilingDataGroupedReduceBigN);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(varianceByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(row * col * sizeof(float) * 2 + 16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormGradTilingDataGroupedReduceBigN* tilingDatafromBin =
        reinterpret_cast<LayerNormGradTilingDataGroupedReduceBigN*>(tiling);

    tilingDatafromBin->row = 256;
    tilingDatafromBin->col = 10000;
    // pdgamma, pdbeta
    tilingDatafromBin->gammaBetaMainBlockFactor = 157;
    tilingDatafromBin->gammaBetaBlockDim = 64;
    tilingDatafromBin->gammaBetaNloopMainBlock = 1;
    tilingDatafromBin->gammaBetaNtailMainBlock = 157;
    tilingDatafromBin->gammaBetaNloopTailBlock = 1;
    tilingDatafromBin->gammaBetaNtailTailBlock = 109;
    tilingDatafromBin->gammaBetaMtail = 0;
    tilingDatafromBin->gammaBetaBasicBlockLoop = 2;
    tilingDatafromBin->gammaBetaMainFoldCount = 2;
    tilingDatafromBin->gammaBetaCacheBufferCount = 2;
    tilingDatafromBin->gammaBetaResultCacheID = 1;
    tilingDatafromBin->gammaBetaNfactor = 160;
    tilingDatafromBin->gammaBetaMfactor = 64;
    
    tilingDatafromBin->backwardBlockDim = 64;
    tilingDatafromBin->backwardNPerBlock = 156;
    tilingDatafromBin->backwardNRem = 16;
    tilingDatafromBin->nToProcessMain = 157;
    tilingDatafromBin->nToProcessTail = 156;
    tilingDatafromBin->backwardMTotalLoop = 3;
    tilingDatafromBin->backwardMtail = 48;
    tilingDatafromBin->backwardNloopMain = 2;
    tilingDatafromBin->backwardNtailMain = 29;
    tilingDatafromBin->backwardBasicBlockLoopMain = 2;
    tilingDatafromBin->backwardMainFoldCountMain = 0;
    tilingDatafromBin->backwardNfactorBlockAligned = 64;
    tilingDatafromBin->backwardMfactor = 104;
    tilingDatafromBin->backwardMfactorBlockAligned = 104;
    tilingDatafromBin->backwardCacheBufferCountMain = 2;
    tilingDatafromBin->backwardResultCacheIDMain = 1;
    tilingDatafromBin->backwardNloopTail = 2;
    tilingDatafromBin->backwardNtailTail = 28;
    tilingDatafromBin->backwardBasicBlockLoopTail = 2;
    tilingDatafromBin->backwardMainFoldCountTail = 0;
    tilingDatafromBin->backwardCacheBufferCountTail = 2;
    tilingDatafromBin->backwardResultCacheIDTail = 1;

    tilingDatafromBin->pdxIsRequire = 1;
    tilingDatafromBin->pdgammaIsRequire = 1;
    tilingDatafromBin->pdbetaIsRequire = 1;
    tilingDatafromBin->epsilon = 1e-12;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t blockDim = tilingDatafromBin->backwardBlockDim;

    ICPU_SET_TILING_KEY(700);
    ICPU_RUN_KF(
        layer_norm_grad, blockDim, dy, x, variance, mean, gamma, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(variance);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
