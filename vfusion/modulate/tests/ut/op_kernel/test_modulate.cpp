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

extern "C" __global__ __aicore__ void modulate(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, 
                                                GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class modulate_test : public testing::Test {
   protected:
    static void SetUpTestCase() { cout << "modulate_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "modulate_test TearDown\n" << endl; }
};

TEST_F(modulate_test, modulate_kernel_test_TilingD_8_4_20480) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = 8 * 4 * 20480 * sizeof(float);
    size_t shape_scale = 8 * 1 * 20480 * sizeof(float);
    size_t shape_shift = 8 * 1 * 20480 * sizeof(float);
    size_t shape_y =  8 * 4 * 20480 * sizeof(float);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ModulateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
    uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(shape_y);
    
    memset(xGM, 0, shape_x);
    memset(scaleGM, 0, shape_scale);
    memset(shiftGM, 0, shape_shift);
    memset(yGM, 0, shape_y);

    ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
    tilingData->inputB = 8;
    tilingData->inputL = 4;
    tilingData->inputD = 20480;
    tilingData->ubLength = 9728;
    tilingData->frontNum = 32;
    tilingData->frontLength = 427;
    tilingData->tailNum = 16;
    tilingData->tailLength = 426;
    tilingData->useDTiling = 1;
    tilingData->parameterStatus = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(modulate, 2, xGM, scaleGM, shiftGM, yGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)shiftGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(modulate_test, modulate_kernel_test_TilingB_8_4_20480) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = 8 * 4 * 20480 * sizeof(half);
    size_t shape_scale = 8 * 1 * 20480 * sizeof(half);
    size_t shape_shift = 8 * 1 * 20480 * sizeof(half);
    size_t shape_y =  8 * 4 * 20480 * sizeof(half);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ModulateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
    uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(shape_y);
    
    memset(xGM, 0, shape_x);
    memset(scaleGM, 0, shape_scale);
    memset(shiftGM, 0, shape_shift);
    memset(yGM, 0, shape_y);

    ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
    tilingData->inputB = 8;
    tilingData->inputL = 4;
    tilingData->inputD = 20480;
    tilingData->ubLength = 19456;
    tilingData->frontNum = 8;
    tilingData->frontLength = 1;
    tilingData->tailNum = 0;
    tilingData->tailLength = 0;
    tilingData->useDTiling = 0;
    tilingData->parameterStatus = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(modulate, 2, xGM, scaleGM, shiftGM, yGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)shiftGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(modulate_test, modulate_kernel_test_TilingB_64_4_10) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = 64 * 4 * 10 * sizeof(half);
    size_t shape_scale = 64 * 1 * 10 * sizeof(half);
    size_t shape_shift = 64 * 1 * 10 * sizeof(half);
    size_t shape_y =  64 * 4 * 10 * sizeof(half);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ModulateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
    uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(shape_y);
    
    memset(xGM, 0, shape_x);
    memset(scaleGM, 0, shape_scale);
    memset(shiftGM, 0, shape_shift);
    memset(yGM, 0, shape_y);

    ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
    tilingData->inputB = 64;
    tilingData->inputL = 4;
    tilingData->inputD = 10;
    tilingData->ubLength = 19456;
    tilingData->frontNum = 16;
    tilingData->frontLength = 2;
    tilingData->tailNum = 32;
    tilingData->tailLength = 2;
    tilingData->useDTiling = 0;
    tilingData->parameterStatus = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(modulate, 2, xGM, scaleGM, shiftGM, yGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)shiftGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(modulate_test, modulate_kernel_test_TilingB_32_8_1024) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = 32 * 8 * 1024 * sizeof(half);
    size_t shape_scale = 32 * 1 * 1024 * sizeof(half);
    size_t shape_shift = 32 * 1 * 1024 * sizeof(half);
    size_t shape_y =  32 * 8 * 1024 * sizeof(half);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ModulateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
    uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(shape_y);
    
    memset(xGM, 0, shape_x);
    memset(scaleGM, 0, shape_scale);
    memset(shiftGM, 0, shape_shift);
    memset(yGM, 0, shape_y);

    ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
    tilingData->inputB = 32;
    tilingData->inputL = 8;
    tilingData->inputD = 1024;
    tilingData->ubLength = 19456;
    tilingData->frontNum = 32;
    tilingData->frontLength = 1;
    tilingData->tailNum = 0;
    tilingData->tailLength = 0;
    tilingData->useDTiling = 0;
    tilingData->parameterStatus = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(modulate, 2, xGM, scaleGM, shiftGM, yGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)shiftGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(modulate_test, modulate_kernel_test_TilingL_32_8_5120) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = 32 * 8 * 5120 * sizeof(half);
    size_t shape_scale = 32 * 1 * 5120 * sizeof(half);
    size_t shape_shift = 32 * 1 * 5120 * sizeof(half);
    size_t shape_y =  32 * 8 * 5120 * sizeof(half);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ModulateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
    uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(shape_y);
    
    memset(xGM, 0, shape_x);
    memset(scaleGM, 0, shape_scale);
    memset(shiftGM, 0, shape_shift);
    memset(yGM, 0, shape_y);

    ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
    tilingData->inputB = 32;
    tilingData->inputL = 8;
    tilingData->inputD = 5120;
    tilingData->ubLength = 19456;
    tilingData->frontNum = 16;
    tilingData->frontLength = 6;
    tilingData->tailNum = 32;
    tilingData->tailLength = 5;
    tilingData->useDTiling = 0;
    tilingData->parameterStatus = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(modulate, 2, xGM, scaleGM, shiftGM, yGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)shiftGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(modulate_test, modulate_kernel_test_TilingL_16_8_1024) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = 16 * 8 * 1024 * sizeof(half);
    size_t shape_scale = 16 * 1 * 1024 * sizeof(half);
    size_t shape_shift = 16 * 1 * 1024 * sizeof(half);
    size_t shape_y =  16 * 8 * 1024 * sizeof(half);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ModulateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
    uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(shape_y);
    
    memset(xGM, 0, shape_x);
    memset(scaleGM, 0, shape_scale);
    memset(shiftGM, 0, shape_shift);
    memset(yGM, 0, shape_y);

    ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
    tilingData->inputB = 16;
    tilingData->inputL = 8;
    tilingData->inputD = 1024;
    tilingData->ubLength = 19456;
    tilingData->frontNum = 32;
    tilingData->frontLength = 3;
    tilingData->tailNum = 16;
    tilingData->tailLength = 2;
    tilingData->useDTiling = 0;
    tilingData->parameterStatus = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(modulate, 2, xGM, scaleGM, shiftGM, yGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)shiftGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(modulate_test, modulate_kernel_test_without_scale_16_8_1024) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = 16 * 8 * 1024 * sizeof(half);
    size_t shape_shift = 16 * 1 * 1024 * sizeof(half);
    size_t shape_y =  16 * 8 * 1024 * sizeof(half);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ModulateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *scaleGM = nullptr;
    uint8_t *shiftGM = (uint8_t *)AscendC::GmAlloc(shape_shift);
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(shape_y);
    
    memset(xGM, 0, shape_x);
    memset(shiftGM, 0, shape_shift);
    memset(yGM, 0, shape_y);

    ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
    tilingData->inputB = 16;
    tilingData->inputL = 8;
    tilingData->inputD = 1024;
    tilingData->ubLength = 19456;
    tilingData->frontNum = 32;
    tilingData->frontLength = 3;
    tilingData->tailNum = 16;
    tilingData->tailLength = 2;
    tilingData->useDTiling = 0;
    tilingData->parameterStatus = 0;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(modulate, 2, xGM, scaleGM, shiftGM, yGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)shiftGM);
    AscendC::GmFree((void*)yGM);
}

TEST_F(modulate_test, modulate_kernel_test_without_shift_16_8_1024) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shape_x = 16 * 8 * 1024 * sizeof(half);
    size_t shape_scale = 16 * 1 * 1024 * sizeof(half);
    size_t shape_y =  16 * 8 * 1024 * sizeof(half);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ModulateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *xGM = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *scaleGM = (uint8_t *)AscendC::GmAlloc(shape_scale);
    uint8_t *shiftGM = nullptr;
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(shape_y);
    
    memset(xGM, 0, shape_x);
    memset(scaleGM, 0, shape_scale);
    memset(yGM, 0, shape_y);

    ModulateTilingData *tilingData = reinterpret_cast<ModulateTilingData*>(tiling);
    tilingData->inputB = 16;
    tilingData->inputL = 8;
    tilingData->inputD = 1024;
    tilingData->ubLength = 19456;
    tilingData->frontNum = 32;
    tilingData->frontLength = 3;
    tilingData->tailNum = 16;
    tilingData->tailLength = 2;
    tilingData->useDTiling = 0;
    tilingData->parameterStatus = 0;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(modulate, 2, xGM, scaleGM, shiftGM, yGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)yGM);
}

