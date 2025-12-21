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
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void modulate_grad(
    GM_ADDR grad_Output, GM_ADDR Input, GM_ADDR Scale,
    GM_ADDR Shift, GM_ADDR grad_input, GM_ADDR grad_scale,
    GM_ADDR grad_shift, GM_ADDR workspace, GM_ADDR tiling);

class modulate_grad_test : public testing::Test {
   protected:
    static void SetUpTestCase() { cout << "modulate_grad_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "modulate_grad_test TearDown\n" << endl; }
};

TEST_F(modulate_grad_test, modulate_grad_kernel_test_TilingD_8_4_512) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_gradoutput = 8 * 4 * 512 * sizeof(float);
    size_t shape_input = 8 * 4 * 512 * sizeof(float);
    size_t shape_scale = 8 * 512 * sizeof(float);
    size_t shape_shift = 8 * 512 * sizeof(float);
    size_t shape_gradinput =  8 * 4 * 512 * sizeof(float);
    size_t shape_gradscale =  8 * 512 * sizeof(float);
    size_t shape_gradshift =  8 * 512 * sizeof(float);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ModulateGradTiling);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *gradoutputGM = (uint8_t *)AscendC::GmAlloc(shape_gradoutput);
    uint8_t *inputGM = (uint8_t *)AscendC::GmAlloc(shape_input);
    uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
    uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
    uint8_t *gradinputGM = (uint8_t *)AscendC::GmAlloc(shape_gradinput);
    uint8_t *gradscaleGM = (uint8_t *)AscendC::GmAlloc(shape_gradscale);
    uint8_t *gradshiftGM = (uint8_t *)AscendC::GmAlloc(shape_gradshift);
    
    memset(gradoutputGM, 0, shape_gradoutput);
    memset(inputGM, 0, shape_input);
    memset(scaleGM, 0, shape_scale);
    memset(shiftGM, 0, shape_shift);
    memset(gradinputGM, 0, shape_gradinput);
    memset(gradscaleGM, 0, shape_gradscale);
    memset(gradshiftGM, 0, shape_gradshift);

    ModulateGradTiling *tilingData = reinterpret_cast<ModulateGradTiling*>(tiling);
    tilingData->B = 8;
    tilingData->L = 4;
    tilingData->D = 512;
    tilingData->dataType = 2;
    tilingData->block_dim = 48;
    tilingData->dataTypeSize = 4;
    tilingData->splitB = 1;
    tilingData->coresPerB = 6;
    tilingData->usedCores = 48;
    tilingData->formerNum = 2;
    tilingData->formerLength= 86;
    tilingData->tailNum = 4;
    tilingData->tailLength= 85;
    tilingData->has_scale = 1;
    tilingData->has_shift = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(modulate_grad, 48, gradoutputGM, inputGM, scaleGM, shiftGM, gradinputGM, gradscaleGM, gradshiftGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gradoutputGM);
    AscendC::GmFree((void*)inputGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)shiftGM);
    AscendC::GmFree((void*)gradinputGM);
    AscendC::GmFree((void*)gradscaleGM);
    AscendC::GmFree((void*)gradshiftGM);
}
TEST_F(modulate_grad_test, modulate_grad_kernel_test_TilingB_8_4_512) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_gradoutput = 48 * 10 * 20000 * sizeof(float);
    size_t shape_input = 48 * 10 * 20000 * sizeof(float);
    size_t shape_scale = 48 * 20000 * sizeof(float);
    size_t shape_shift = 48 * 20000 * sizeof(float);
    size_t shape_gradinput =  48 * 10 * 20000 * sizeof(float);
    size_t shape_gradscale =  48 * 20000 * sizeof(float);
    size_t shape_gradshift =  48 * 20000 * sizeof(float);

    size_t sysWorkspaceSize = 64 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ModulateGradTiling);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *gradoutputGM = (uint8_t *)AscendC::GmAlloc(shape_gradoutput);
    uint8_t *inputGM = (uint8_t *)AscendC::GmAlloc(shape_input);
    uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
    uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
    uint8_t *gradinputGM = (uint8_t *)AscendC::GmAlloc(shape_gradinput);
    uint8_t *gradscaleGM = (uint8_t *)AscendC::GmAlloc(shape_gradscale);
    uint8_t *gradshiftGM = (uint8_t *)AscendC::GmAlloc(shape_gradshift);
    
    memset(gradoutputGM, 0, shape_gradoutput);
    memset(inputGM, 0, shape_input);
    memset(scaleGM, 0, shape_scale);
    memset(shiftGM, 0, shape_shift);
    memset(gradinputGM, 0, shape_gradinput);
    memset(gradscaleGM, 0, shape_gradscale);
    memset(gradshiftGM, 0, shape_gradshift);

    ModulateGradTiling *tilingData = reinterpret_cast<ModulateGradTiling*>(tiling);
    tilingData->B = 48;
    tilingData->L = 10;
    tilingData->D = 20000;
    tilingData->dataType = 2;
    tilingData->block_dim = 48;
    tilingData->dataTypeSize = 4;
    tilingData->splitB = 0;
    tilingData->coresPerB = 1;
    tilingData->usedCores = 48;
    tilingData->formerNum = 0;
    tilingData->formerLength= 0;
    tilingData->tailNum = 48;
    tilingData->tailLength= 1;
    tilingData->has_scale = 1;
    tilingData->has_shift = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(modulate_grad, 48, gradoutputGM, inputGM, scaleGM, shiftGM, gradinputGM, gradscaleGM, gradshiftGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gradoutputGM);
    AscendC::GmFree((void*)inputGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)shiftGM);
    AscendC::GmFree((void*)gradinputGM);
    AscendC::GmFree((void*)gradscaleGM);
    AscendC::GmFree((void*)gradshiftGM);
}


// TEST_F(modulate_test, modulate_kernel_test_TilingB_8_4_512) {
//     AscendC::SetKernelMode(KernelMode::MIX_MODE);
//     size_t shape_gradoutput = 8 * 4 * 512 * sizeof(half);
//     size_t shape_input = 8 * 4 * 512 * sizeof(half);
//     size_t shape_scale = 8  * 512 * sizeof(half);
//     size_t shape_shift = 8  * 512 * sizeof(half);
//     size_t shape_gradinput =  8 * 4 * 512 * sizeof(half);
//     size_t shape_gradscale = 8  * 512 * sizeof(half);
//     size_t shape_gradshift = 8  * 512 * sizeof(half);

//     size_t sysWorkspaceSize = 16 * 1024 * 1024;
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
//     size_t tilingSize = sizeof(ModulateTiling);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

//     uint8_t *gradoutputGM = (uint8_t *)AscendC::GmAlloc(shape_gradoutput);
//     uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
//     uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
//     uint8_t *gradinputGM = (uint8_t *)AscendC::GmAlloc(shape_gradinput);
    
//     memset(gradoutputGM, 0, shape_gradoutput);
//     memset(scaleGM, 0, shape_scale);
//     memset(shiftGM, 0, shape_shift);
//     memset(gradinputGM, 0, shape_gradinput);

//     ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
//     tilingData->inputB = 8;
//     tilingData->inputL = 4;
//     tilingData->inputD = 512;
//     tilingData->ubLength = 19456;
//     tilingData->frontNum = 8;
//     tilingData->frontLength = 1;
//     tilingData->tailNum = 0;
//     tilingData->tailLength = 0;
//     tilingData->useDTiling = 0;
//     tilingData->parameterStatus = 0;

//     ICPU_SET_TILING_KEY(0);
//     ICPU_RUN_KF(modulate_grad, 2, gradoutputGM, scaleGM, shiftGM, gradinputGM, workspace, (uint8_t*)(tilingData));
//     AscendC::GmFree((void*)workspace);
//     AscendC::GmFree((void*)tiling);
//     AscendC::GmFree((void*)gradoutputGM);
//     AscendC::GmFree((void*)scaleGM);
//     AscendC::GmFree((void*)shiftGM);
//     AscendC::GmFree((void*)gradinputGM);
// }

// TEST_F(modulate_test, modulate_kernel_test_TilingB_64_4_10) {
//     AscendC::SetKernelMode(KernelMode::MIX_MODE);
//     size_t shape_gradoutput = 64 * 4 * 10 * sizeof(half);
//     size_t shape_scale = 64 * 1 * 10 * sizeof(half);
//     size_t shape_shift = 64 * 1 * 10 * sizeof(half);
//     size_t shape_gradinput =  64 * 4 * 10 * sizeof(half);

//     size_t sysWorkspaceSize = 16 * 1024 * 1024;
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
//     size_t tilingSize = sizeof(ModulateTilingData);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

//     uint8_t *gradoutputGM = (uint8_t *)AscendC::GmAlloc(shape_gradoutput);
//     uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
//     uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
//     uint8_t *gradinputGM = (uint8_t *)AscendC::GmAlloc(shape_gradinput);
    
//     memset(gradoutputGM, 0, shape_gradoutput);
//     memset(scaleGM, 0, shape_scale);
//     memset(shiftGM, 0, shape_shift);
//     memset(gradinputGM, 0, shape_gradinput);

//     ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
//     tilingData->inputB = 64;
//     tilingData->inputL = 4;
//     tilingData->inputD = 10;
//     tilingData->ubLength = 19456;
//     tilingData->frontNum = 16;
//     tilingData->frontLength = 2;
//     tilingData->tailNum = 32;
//     tilingData->tailLength = 2;
//     tilingData->useDTiling = 0;
//     tilingData->parameterStatus = 0;

//     ICPU_SET_TILING_KEY(0);
//     ICPU_RUN_KF(modulate_grad, 2, gradoutputGM, scaleGM, shiftGM, gradinputGM, workspace, (uint8_t*)(tilingData));
//     AscendC::GmFree((void*)workspace);
//     AscendC::GmFree((void*)tiling);
//     AscendC::GmFree((void*)gradoutputGM);
//     AscendC::GmFree((void*)scaleGM);
//     AscendC::GmFree((void*)shiftGM);
//     AscendC::GmFree((void*)gradinputGM);
// }

// TEST_F(modulate_test, modulate_kernel_test_TilingB_32_8_1024) {
//     AscendC::SetKernelMode(KernelMode::MIX_MODE);
//     size_t shape_gradoutput = 32 * 8 * 1024 * sizeof(half);
//     size_t shape_scale = 32 * 1 * 1024 * sizeof(half);
//     size_t shape_shift = 32 * 1 * 1024 * sizeof(half);
//     size_t shape_gradinput =  32 * 8 * 1024 * sizeof(half);

//     size_t sysWorkspaceSize = 16 * 1024 * 1024;
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
//     size_t tilingSize = sizeof(ModulateTilingData);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

//     uint8_t *gradoutputGM = (uint8_t *)AscendC::GmAlloc(shape_gradoutput);
//     uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
//     uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
//     uint8_t *gradinputGM = (uint8_t *)AscendC::GmAlloc(shape_gradinput);
    
//     memset(gradoutputGM, 0, shape_gradoutput);
//     memset(scaleGM, 0, shape_scale);
//     memset(shiftGM, 0, shape_shift);
//     memset(gradinputGM, 0, shape_gradinput);

//     ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
//     tilingData->inputB = 32;
//     tilingData->inputL = 8;
//     tilingData->inputD = 1024;
//     tilingData->ubLength = 19456;
//     tilingData->frontNum = 32;
//     tilingData->frontLength = 1;
//     tilingData->tailNum = 0;
//     tilingData->tailLength = 0;
//     tilingData->useDTiling = 0;
//     tilingData->parameterStatus = 0;

//     ICPU_SET_TILING_KEY(0);
//     ICPU_RUN_KF(modulate_grad, 2, gradoutputGM, scaleGM, shiftGM, gradinputGM, workspace, (uint8_t*)(tilingData));
//     AscendC::GmFree((void*)workspace);
//     AscendC::GmFree((void*)tiling);
//     AscendC::GmFree((void*)gradoutputGM);
//     AscendC::GmFree((void*)scaleGM);
//     AscendC::GmFree((void*)shiftGM);
//     AscendC::GmFree((void*)gradinputGM);
// }

// TEST_F(modulate_test, modulate_kernel_test_TilingL_32_8_5120) {
//     AscendC::SetKernelMode(KernelMode::MIX_MODE);
//     size_t shape_gradoutput = 32 * 8 * 5120 * sizeof(half);
//     size_t shape_scale = 32 * 1 * 5120 * sizeof(half);
//     size_t shape_shift = 32 * 1 * 5120 * sizeof(half);
//     size_t shape_gradinput =  32 * 8 * 5120 * sizeof(half);

//     size_t sysWorkspaceSize = 16 * 1024 * 1024;
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
//     size_t tilingSize = sizeof(ModulateTilingData);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

//     uint8_t *gradoutputGM = (uint8_t *)AscendC::GmAlloc(shape_gradoutput);
//     uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
//     uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
//     uint8_t *gradinputGM = (uint8_t *)AscendC::GmAlloc(shape_gradinput);
    
//     memset(gradoutputGM, 0, shape_gradoutput);
//     memset(scaleGM, 0, shape_scale);
//     memset(shiftGM, 0, shape_shift);
//     memset(gradinputGM, 0, shape_gradinput);

//     ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
//     tilingData->inputB = 32;
//     tilingData->inputL = 8;
//     tilingData->inputD = 5120;
//     tilingData->ubLength = 19456;
//     tilingData->frontNum = 16;
//     tilingData->frontLength = 6;
//     tilingData->tailNum = 32;
//     tilingData->tailLength = 5;
//     tilingData->useDTiling = 0;
//     tilingData->parameterStatus = 0;

//     ICPU_SET_TILING_KEY(0);
//     ICPU_RUN_KF(modulate_grad, 2, gradoutputGM, scaleGM, shiftGM, gradinputGM, workspace, (uint8_t*)(tilingData));
//     AscendC::GmFree((void*)workspace);
//     AscendC::GmFree((void*)tiling);
//     AscendC::GmFree((void*)gradoutputGM);
//     AscendC::GmFree((void*)scaleGM);
//     AscendC::GmFree((void*)shiftGM);
//     AscendC::GmFree((void*)gradinputGM);
// }

// TEST_F(modulate_test, modulate_kernel_test_TilingL_16_8_1024) {
//     AscendC::SetKernelMode(KernelMode::MIX_MODE);
//     size_t shape_gradoutput = 16 * 8 * 1024 * sizeof(half);
//     size_t shape_scale = 16 * 1 * 1024 * sizeof(half);
//     size_t shape_shift = 16 * 1 * 1024 * sizeof(half);
//     size_t shape_gradinput =  16 * 8 * 1024 * sizeof(half);

//     size_t sysWorkspaceSize = 16 * 1024 * 1024;
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
//     size_t tilingSize = sizeof(ModulateTilingData);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

//     uint8_t *gradoutputGM = (uint8_t *)AscendC::GmAlloc(shape_gradoutput);
//     uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
//     uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
//     uint8_t *gradinputGM = (uint8_t *)AscendC::GmAlloc(shape_gradinput);
    
//     memset(gradoutputGM, 0, shape_gradoutput);
//     memset(scaleGM, 0, shape_scale);
//     memset(shiftGM, 0, shape_shift);
//     memset(gradinputGM, 0, shape_gradinput);

//     ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
//     tilingData->inputB = 16;
//     tilingData->inputL = 8;
//     tilingData->inputD = 1024;
//     tilingData->ubLength = 19456;
//     tilingData->frontNum = 32;
//     tilingData->frontLength = 3;
//     tilingData->tailNum = 16;
//     tilingData->tailLength = 2;
//     tilingData->useDTiling = 0;
//     tilingData->parameterStatus = 0;

//     ICPU_SET_TILING_KEY(0);
//     ICPU_RUN_KF(modulate_grad, 2, gradoutputGM, scaleGM, shiftGM, gradinputGM, workspace, (uint8_t*)(tilingData));
//     AscendC::GmFree((void*)workspace);
//     AscendC::GmFree((void*)tiling);
//     AscendC::GmFree((void*)gradoutputGM);
//     AscendC::GmFree((void*)scaleGM);
//     AscendC::GmFree((void*)shiftGM);
//     AscendC::GmFree((void*)gradinputGM);
// }

// TEST_F(modulate_test, modulate_kernel_test_without_scale_16_8_1024) {
//     AscendC::SetKernelMode(KernelMode::MIX_MODE);
//     size_t shape_gradoutput = 16 * 8 * 1024 * sizeof(half);
//     size_t shape_shift = 16 * 1 * 1024 * sizeof(half);
//     size_t shape_gradinput =  16 * 8 * 1024 * sizeof(half);

//     size_t sysWorkspaceSize = 16 * 1024 * 1024;
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
//     size_t tilingSize = sizeof(ModulateTilingData);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

//     uint8_t *gradoutputGM = (uint8_t *)AscendC::GmAlloc(shape_gradoutput);
//     uint8_t *scaleGM = nullptr;
//     uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
//     uint8_t *gradinputGM = (uint8_t *)AscendC::GmAlloc(shape_gradinput);
    
//     memset(gradoutputGM, 0, shape_gradoutput);
//     memset(shiftGM, 0, shape_shift);
//     memset(gradinputGM, 0, shape_gradinput);

//     ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
//     tilingData->inputB = 16;
//     tilingData->inputL = 8;
//     tilingData->inputD = 1024;
//     tilingData->ubLength = 19456;
//     tilingData->frontNum = 32;
//     tilingData->frontLength = 3;
//     tilingData->tailNum = 16;
//     tilingData->tailLength = 2;
//     tilingData->useDTiling = 0;
//     tilingData->parameterStatus = 0;

//     ICPU_SET_TILING_KEY(1);
//     ICPU_RUN_KF(modulate_grad, 2, gradoutputGM, scaleGM, shiftGM, gradinputGM, workspace, (uint8_t*)(tilingData));
//     AscendC::GmFree((void*)workspace);
//     AscendC::GmFree((void*)tiling);
//     AscendC::GmFree((void*)gradoutputGM);
//     AscendC::GmFree((void*)shiftGM);
//     AscendC::GmFree((void*)gradinputGM);
// }

// TEST_F(modulate_test, modulate_kernel_test_without_shift_16_8_1024) {
//     AscendC::SetKernelMode(KernelMode::MIX_MODE);
//     size_t shape_gradoutput = 16 * 8 * 1024 * sizeof(half);
//     size_t shape_scale = 16 * 1 * 1024 * sizeof(half);
//     size_t shape_gradinput =  16 * 8 * 1024 * sizeof(half);

//     size_t sysWorkspaceSize = 16 * 1024 * 1024;
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
//     size_t tilingSize = sizeof(ModulateTilingData);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

//     uint8_t *gradoutputGM = (uint8_t *)AscendC::GmAlloc(shape_gradoutput);
//     uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
//     uint8_t *shiftGM = nullptr;
//     uint8_t *gradinputGM = (uint8_t *)AscendC::GmAlloc(shape_gradinput);
    
//     memset(gradoutputGM, 0, shape_gradoutput);
//     memset(scaleGM, 0, shape_scale);
//     memset(gradinputGM, 0, shape_gradinput);

//     ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
//     tilingData->inputB = 16;
//     tilingData->inputL = 8;
//     tilingData->inputD = 1024;
//     tilingData->ubLength = 19456;
//     tilingData->frontNum = 32;
//     tilingData->frontLength = 3;
//     tilingData->tailNum = 16;
//     tilingData->tailLength = 2;
//     tilingData->useDTiling = 0;
//     tilingData->parameterStatus = 0;

//     ICPU_SET_TILING_KEY(2);
//     ICPU_RUN_KF(modulate_grad, 2, gradoutputGM, scaleGM, shiftGM, gradinputGM, workspace, (uint8_t*)(tilingData));
//     AscendC::GmFree((void*)workspace);
//     AscendC::GmFree((void*)tiling);
//     AscendC::GmFree((void*)gradoutputGM);
//     AscendC::GmFree((void*)scaleGM);
//     AscendC::GmFree((void*)gradinputGM);
// }
