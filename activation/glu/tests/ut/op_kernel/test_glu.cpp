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
#include "glu_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void glu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class glu_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "glu_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "glu_test TearDown\n" << endl;
    }
};

TEST_F(glu_test, test_case_1)
{
#undef DTYPE_INPUT
#define DTYPE_INPUT DT_FP32

    size_t inputByteSize = 24 * 2 * 1 * sizeof(float);
    size_t outputByteSize = 24 * 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(6*1024*1024+40*32*4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;
    system("cp -r ../../../../activation/glu/tests/ut/op_kernel/glu_data ./");
    system("chmod -R 755 ./glu_data/");
    system("cd ./glu_data/ && rm -rf ./*bin");
    system("cd ./glu_data/ && python3 gen_data.py 24 2 1 1 float32");

    char* path_ = get_current_dir_name();
    string path(path_);

    GluTilingData* tilingDatafromBin = reinterpret_cast<GluTilingData*>(tiling);

    tilingDatafromBin->group = 6144;
    tilingDatafromBin->loopNum = 0;
    tilingDatafromBin->tailLoopNum = 0;
    tilingDatafromBin->nLastTailGroup = 1;
    tilingDatafromBin->lastTailGroup = 0;
    tilingDatafromBin->splitSize = 1;
    tilingDatafromBin->realCoreNum = 24;
    tilingDatafromBin->numPerCore = 1;
    tilingDatafromBin->tilingKey = 1;
    tilingDatafromBin->blockSize = 8;
    tilingDatafromBin->ny = 1;

    ReadFile(path + "/glu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/glu_data/output.bin", y, outputByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    system("cd ./glu_data/ && python3 compare_data.py 'float32'");
    free(path_);
}

TEST_F(glu_test, test_case_2)
{
#undef DTYPE_INPUT
#define DTYPE_INPUT DT_FP32

    size_t inputByteSize = 1 * 80 * 2560 * sizeof(float);
    size_t outputByteSize = 1 * 80 * 1280 * sizeof(float);
    size_t tiling_data_size = sizeof(GluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(6*1024*1024+40*32*4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;
    system("cp -r ../../../../activation/glu/tests/ut/op_kernel/glu_data ./");
    system("chmod -R 755 ./glu_data/");
    system("cd ./glu_data/ && rm -rf ./*bin");
    system("cd ./glu_data/ && python3 gen_data.py 1 80 2560 2 float32");

    char* path_ = get_current_dir_name();
    string path(path_);

    GluTilingData* tilingDatafromBin = reinterpret_cast<GluTilingData*>(tiling);

    tilingDatafromBin->group = 6;
    tilingDatafromBin->loopNum = 0;
    tilingDatafromBin->tailLoopNum = 0;
    tilingDatafromBin->nLastTailGroup = 2;
    tilingDatafromBin->lastTailGroup = 0;
    tilingDatafromBin->splitSize = 1280;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->numPerCore = 2;
    tilingDatafromBin->tilingKey = 2;
    tilingDatafromBin->blockSize = 8;
    tilingDatafromBin->ny = 1280;

    ReadFile(path + "/glu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/glu_data/output.bin", y, outputByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    system("cd ./glu_data/ && python3 compare_data.py 'float32'");
    free(path_);
}

TEST_F(glu_test, test_case_3)
{
#undef DTYPE_INPUT
#define DTYPE_INPUT DT_FP32

    size_t inputByteSize = 1 * 8 * 2560 * sizeof(float);
    size_t outputByteSize = 1 * 4 * 2560 * sizeof(float);
    size_t tiling_data_size = sizeof(GluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(6*1024*1024+40*32*4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;
    system("cp -r ../../../../activation/glu/tests/ut/op_kernel/glu_data ./");
    system("chmod -R 755 ./glu_data/");
    system("cd ./glu_data/ && rm -rf ./*bin");
    system("cd ./glu_data/ && python3 gen_data.py 1 8 2560 1 float32");

    char* path_ = get_current_dir_name();
    string path(path_);

    GluTilingData* tilingDatafromBin = reinterpret_cast<GluTilingData*>(tiling);

    tilingDatafromBin->group = 1;
    tilingDatafromBin->loopNum = 2;
    tilingDatafromBin->tailLoopNum = 2048;
    tilingDatafromBin->nLastTailGroup = 0;
    tilingDatafromBin->lastTailGroup = 0;
    tilingDatafromBin->splitSize = 8192;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->numPerCore = 0;
    tilingDatafromBin->tilingKey = 3;
    tilingDatafromBin->blockSize = 8;
    tilingDatafromBin->ny = 10240;

    ReadFile(path + "/glu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(3);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/glu_data/output.bin", y, outputByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    system("cd ./glu_data/ && python3 compare_data.py 'float32'");
    free(path_);
}