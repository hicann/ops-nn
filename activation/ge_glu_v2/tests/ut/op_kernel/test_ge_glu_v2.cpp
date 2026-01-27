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
#include "ge_glu_v2_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void ge_glu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, GM_ADDR tiling);

class ge_glu_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "ge_glu_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ge_glu_v2_test TearDown\n" << endl;
    }
};

// TEST_F(ge_glu_v2_test, test_case_101)
// {
//     size_t inputByteSize = 1 * 80 * 2560 * sizeof(int16_t);
//     size_t outputByteSize = 1 * 80 * 1280 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(6*1024*1024+40*32*4);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 1 80 2560 float16");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case0");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 4;
//     tilingDatafromBin->loopNum = 0;
//     tilingDatafromBin->tailLoopNum = 0;
//     tilingDatafromBin->nLastTailGroup = 2;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 1280;
//     tilingDatafromBin->realCoreNum = 40;
//     tilingDatafromBin->numPerCore = 2;
//     tilingDatafromBin->tilingKey = 101;
//     tilingDatafromBin->blockSize = 16;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->ny = 1280;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(101);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_201)
// {
//     size_t inputByteSize = 1 * 80 * 2560 * sizeof(int16_t);
//     size_t outputByteSize = 1 * 80 * 1280 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 1 80 2560 float16");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case1");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 4;
//     tilingDatafromBin->loopNum = 0;
//     tilingDatafromBin->tailLoopNum = 0;
//     tilingDatafromBin->nLastTailGroup = 2;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 1280;
//     tilingDatafromBin->realCoreNum = 40;
//     tilingDatafromBin->numPerCore = 2;
//     tilingDatafromBin->tilingKey = 201;
//     tilingDatafromBin->blockSize = 16;
//     tilingDatafromBin->activateLeft = 0;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(201);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_301)
// {
//     size_t inputByteSize = 1 * 80 * 2560 * sizeof(float);
//     size_t outputByteSize = 1 * 80 * 1280 * sizeof(float);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 1 80 2560 float32");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case1");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 4;
//     tilingDatafromBin->loopNum = 0;
//     tilingDatafromBin->tailLoopNum = 0;
//     tilingDatafromBin->nLastTailGroup = 2;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 1280;
//     tilingDatafromBin->realCoreNum = 40;
//     tilingDatafromBin->numPerCore = 2;
//     tilingDatafromBin->tilingKey = 301;
//     tilingDatafromBin->blockSize = 8;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->ny = 1280;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(301);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_103)
// {
//     size_t inputByteSize = 2 * 2 * 12352 * sizeof(int16_t);
//     size_t outputByteSize = 2 * 2 * 6176 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 2 2 12352 float16");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case0");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 2;
//     tilingDatafromBin->loopNum = 6;
//     tilingDatafromBin->tailLoopNum = 64;
//     tilingDatafromBin->nLastTailGroup = 0;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 6144;
//     tilingDatafromBin->realCoreNum = 40;
//     tilingDatafromBin->numPerCore = 0;
//     tilingDatafromBin->tilingKey = 103;
//     tilingDatafromBin->blockSize = 16;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->ny = 12352;
//     tilingDatafromBin->approximate = 1;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(103);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_203)
// {
//     size_t inputByteSize = 1 * 80 * 2560 * sizeof(int16_t);
//     size_t outputByteSize = 1 * 80 * 1280 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 1 80 2560 float16");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case1");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 2;
//     tilingDatafromBin->loopNum = 12;
//     tilingDatafromBin->tailLoopNum = 64;
//     tilingDatafromBin->nLastTailGroup = 0;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 6144;
//     tilingDatafromBin->realCoreNum = 48;
//     tilingDatafromBin->numPerCore = 0;
//     tilingDatafromBin->tilingKey = 203;
//     tilingDatafromBin->blockSize = 16;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->ny = 12352;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(203);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_303)
// {
//     size_t inputByteSize = 2 * 2 * 12352 * sizeof(float);
//     size_t outputByteSize = 2 * 2 * 6176 * sizeof(float);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 2 2 12352 float32");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case1");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 1;
//     tilingDatafromBin->loopNum = 8;
//     tilingDatafromBin->tailLoopNum = 32;
//     tilingDatafromBin->nLastTailGroup = 0;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 6144;
//     tilingDatafromBin->realCoreNum = 40;
//     tilingDatafromBin->numPerCore = 0;
//     tilingDatafromBin->tilingKey = 303;
//     tilingDatafromBin->blockSize = 8;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->ny = 6176;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(303);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_111)
// {
//     size_t inputByteSize = 1 * 80 * 2560 * sizeof(int16_t);
//     size_t outputByteSize = 1 * 80 * 1280 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 1 80 2560 float16");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case0");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 4;
//     tilingDatafromBin->loopNum = 0;
//     tilingDatafromBin->tailLoopNum = 0;
//     tilingDatafromBin->nLastTailGroup = 2;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 1280;
//     tilingDatafromBin->realCoreNum = 40;
//     tilingDatafromBin->numPerCore = 2;
//     tilingDatafromBin->tilingKey = 111;
//     tilingDatafromBin->blockSize = 16;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->approximate = 0;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(111);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_211)
// {
//     size_t inputByteSize = 1 * 80 * 2560 * sizeof(int16_t);
//     size_t outputByteSize = 1 * 80 * 1280 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 1 80 2560 float16");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case1");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 4;
//     tilingDatafromBin->loopNum = 0;
//     tilingDatafromBin->tailLoopNum = 0;
//     tilingDatafromBin->nLastTailGroup = 2;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 1280;
//     tilingDatafromBin->realCoreNum = 40;
//     tilingDatafromBin->numPerCore = 2;
//     tilingDatafromBin->tilingKey = 211;
//     tilingDatafromBin->blockSize = 16;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->approximate = 0;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(211);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_311)
// {
//     size_t inputByteSize = 1 * 80 * 2560 * sizeof(float);
//     size_t outputByteSize = 1 * 80 * 1280 * sizeof(float);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 1 80 2560 float32");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case1");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 4;
//     tilingDatafromBin->loopNum = 0;
//     tilingDatafromBin->tailLoopNum = 0;
//     tilingDatafromBin->nLastTailGroup = 2;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 1280;
//     tilingDatafromBin->realCoreNum = 40;
//     tilingDatafromBin->numPerCore = 2;
//     tilingDatafromBin->tilingKey = 311;
//     tilingDatafromBin->blockSize = 8;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->ny = 1280;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(311);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_113)
// {
//     size_t inputByteSize = 2 * 2 * 12352 * sizeof(int16_t);
//     size_t outputByteSize = 2 * 2 * 12352 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 2 2 12352 float16");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case0");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 1;
//     tilingDatafromBin->loopNum = 8;
//     tilingDatafromBin->tailLoopNum = 32;
//     tilingDatafromBin->nLastTailGroup = 0;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 6144;
//     tilingDatafromBin->realCoreNum = 40;
//     tilingDatafromBin->numPerCore = 0;
//     tilingDatafromBin->tilingKey = 113;
//     tilingDatafromBin->blockSize = 16;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->ny = 6176;
//     tilingDatafromBin->approximate = 0;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(113);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_213)
// {
//     size_t inputByteSize = 1 * 80 * 2560 * sizeof(int16_t);
//     size_t outputByteSize = 1 * 80 * 1280 * sizeof(int16_t);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 1 80 2560 float16");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case1");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 2;
//     tilingDatafromBin->loopNum = 12;
//     tilingDatafromBin->tailLoopNum = 64;
//     tilingDatafromBin->nLastTailGroup = 0;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 6144;
//     tilingDatafromBin->realCoreNum = 48;
//     tilingDatafromBin->numPerCore = 0;
//     tilingDatafromBin->tilingKey = 213;
//     tilingDatafromBin->blockSize = 16;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->ny = 12352;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(213);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(ge_glu_v2_test, test_case_313)
// {
//     size_t inputByteSize = 2 * 2 * 12352 * sizeof(float);
//     size_t outputByteSize = 2 * 2 * 6176 * sizeof(float);
//     size_t tiling_data_size = sizeof(GeGluV2TilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 40;
//     system("cp -r ../../../../activation/ge_glu_v2/tests/ut/op_kernel/ge_glu_v2_data ./");
//     system("chmod -R 755 ./ge_glu_v2_data/");
//     system("cd ./ge_glu_v2_data/ && rm -rf ./*bin");
//     system("cd ./ge_glu_v2_data/ && python3 gen_data.py 2 2 12352 float32");
//     system("cd ./ge_glu_v2_data/ && python3 gen_tiling.py case1");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     GeGluV2TilingData* tilingDatafromBin = reinterpret_cast<GeGluV2TilingData*>(tiling);

//     tilingDatafromBin->group = 1;
//     tilingDatafromBin->loopNum = 8;
//     tilingDatafromBin->tailLoopNum = 32;
//     tilingDatafromBin->nLastTailGroup = 0;
//     tilingDatafromBin->lastTailGroup = 0;
//     tilingDatafromBin->splitSize = 6144;
//     tilingDatafromBin->realCoreNum = 40;
//     tilingDatafromBin->numPerCore = 0;
//     tilingDatafromBin->tilingKey = 313;
//     tilingDatafromBin->blockSize = 8;
//     tilingDatafromBin->activateLeft = 0;
//     tilingDatafromBin->ny = 6176;
//     tilingDatafromBin->approximate = 0;

//     ReadFile(path + "/ge_glu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(313);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(ge_glu_v2, blockDim, x, y, gelu, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(y);
//     AscendC::GmFree(gelu);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }