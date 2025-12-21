/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_batch_norm_v3.cpp
 * \brief
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "batch_norm_v3_regbase_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

using namespace std;

extern "C" __global__ __aicore__ void batch_norm_v3(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR variance, GM_ADDR y, GM_ADDR mean_out,
    GM_ADDR variance_out, GM_ADDR batch_mean, GM_ADDR batch_rstd, GM_ADDR workspace, GM_ADDR tiling);

class batch_norm_v3_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "batch_norm_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "batch_norm_v3_test TearDown\n" << endl;
    }
};

TEST_F(batch_norm_v3_test, test_case_0001)
{
    size_t xByteSize = 1 * 4 * 2 * 8192 * sizeof(float);
    size_t gammaByteSize = 4 * sizeof(float);
    size_t betaByteSize = 4 * sizeof(float);
    size_t yByteSize = 1 * 4 * 2 * 8192 * sizeof(float);
    size_t meanByteSize = 1 * 4 * sizeof(float);
    size_t rstdByteSize = 1 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(BatchNormV3WelfordRegbaseTilingData);
    uint32_t blockDim = 4;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* batch_mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* batch_rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3WelfordRegbaseTilingData* tilingDatafromBin =
        reinterpret_cast<BatchNormV3WelfordRegbaseTilingData*>(tiling);

    tilingDatafromBin->r1 = 1;
    tilingDatafromBin->r0 = 15521;
    tilingDatafromBin->a0 = 4;
    tilingDatafromBin->loopR1outer = 1;
    tilingDatafromBin->r1Factor = 1;
    tilingDatafromBin->loopR0outer = 1;
    tilingDatafromBin->r0Factor = 10088;
    tilingDatafromBin->realCoreNum = 4;
    tilingDatafromBin->numLastCore = 0;
    tilingDatafromBin->aBlockFactor = 1;
    tilingDatafromBin->aGatherLimit = 1;
    tilingDatafromBin->parallelN = 10088;
    tilingDatafromBin->processSize = 10088;
    tilingDatafromBin->ubSize = 10088;
    tilingDatafromBin->elemNum = 15521;
    tilingDatafromBin->vlLenFp32 = 64;
    tilingDatafromBin->cutR1OrR0 = 0;
    tilingDatafromBin->binaryAddLastNum = 64;
    tilingDatafromBin->binaryAddQuotient = 8192;
    float epsilon = 0.00001;
    float momentum = 0.1;

    ICPU_SET_TILING_KEY(30000);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, gamma, gamma, mean, variance, y, mean_out, variance_out, batch_mean, batch_rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(batch_mean);
    AscendC::GmFree(batch_rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(batch_norm_v3_test, test_case_0002)
{
    size_t xByteSize = 8 * 4 * 2 * 1024 * sizeof(float);
    size_t gammaByteSize = 4 * sizeof(float);
    size_t betaByteSize = 4 * sizeof(float);
    size_t yByteSize = 8 * 4 * 2 * 1024 * sizeof(float);
    size_t meanByteSize = 1 * 4 * sizeof(float);
    size_t rstdByteSize = 1 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(BatchNormV3WelfordRegbaseTilingData);
    uint32_t blockDim = 4;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* batch_mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* batch_rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3WelfordRegbaseTilingData* tilingDatafromBin =
        reinterpret_cast<BatchNormV3WelfordRegbaseTilingData*>(tiling);

    tilingDatafromBin->r1 = 8;
    tilingDatafromBin->r0 = 2048;
    tilingDatafromBin->a0 = 4;
    tilingDatafromBin->loopR1outer = 2;
    tilingDatafromBin->r1Factor = 4;
    tilingDatafromBin->loopR0outer = 1;
    tilingDatafromBin->r0Factor = 2048;
    tilingDatafromBin->realCoreNum = 4;
    tilingDatafromBin->numLastCore = 0;
    tilingDatafromBin->aBlockFactor = 1;
    tilingDatafromBin->aGatherLimit = 1;
    tilingDatafromBin->parallelN = 8192;
    tilingDatafromBin->processSize = 8192;
    tilingDatafromBin->ubSize = 8192;
    tilingDatafromBin->elemNum = 16384;
    tilingDatafromBin->vlLenFp32 = 4;
    tilingDatafromBin->cutR1OrR0 = 1;
    tilingDatafromBin->binaryAddLastNum = 64;
    tilingDatafromBin->binaryAddQuotient = 8192;
    float epsilon = 0.00001;
    float momentum = 0.1;

    ICPU_SET_TILING_KEY(30000);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, gamma, gamma, mean, variance, y, mean_out, variance_out, batch_mean, batch_rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(batch_mean);
    AscendC::GmFree(batch_rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(batch_norm_v3_test, test_case_0003)
{
    size_t xByteSize = 1 * 4 * 4096 * 1 * sizeof(float);
    size_t gammaByteSize = 4 * sizeof(float);
    size_t betaByteSize = 4 * sizeof(float);
    size_t yByteSize = 1 * 4 * 4096 * 1 * sizeof(float);
    size_t meanByteSize = 1 * 4 * sizeof(float);
    size_t rstdByteSize = 1 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(BatchNormV3FullReduceRegbaseTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* batch_mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* batch_rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3FullReduceRegbaseTilingData* tilingDatafromBin =
        reinterpret_cast<BatchNormV3FullReduceRegbaseTilingData*>(tiling);

    tilingDatafromBin->r1 = 1;
    tilingDatafromBin->r0 = 4096;
    tilingDatafromBin->a = 4;
    tilingDatafromBin->aFactor = 3;
    tilingDatafromBin->aBlockFactor = 1;
    tilingDatafromBin->blockNum = 4;
    tilingDatafromBin->r1r0LoopCount = 64;
    tilingDatafromBin->binaryAddQuotient = 64;
    tilingDatafromBin->binaryAddK = 0;
    tilingDatafromBin->binaryAddLastNum = 32;
    tilingDatafromBin->powerOfTwoForR = 4096;
    tilingDatafromBin->epsilon = 0.00001;
    tilingDatafromBin->momentum = 0.1;

    ICPU_SET_TILING_KEY(200000);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, gamma, gamma, mean, variance, y, mean_out, variance_out, batch_mean, batch_rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(batch_mean);
    AscendC::GmFree(batch_rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// TEST_F(batch_norm_v3_test, test_case_infer_0001)
// {
//     size_t xByteSize = 1 * 24 * 1 * sizeof(half);
//     size_t gammaByteSize = 24 * sizeof(half);
//     size_t betaByteSize = 24 * sizeof(half);
//     size_t yByteSize = 1 * 24 * 1 * sizeof(half);
//     size_t meanByteSize = 24 * sizeof(half);
//     size_t varByteSize = 24 * sizeof(half);
//     size_t tiling_data_size = sizeof(BatchNormV3InferTilingData);
//     uint32_t blockDim = 43;

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
//     uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
//     uint8_t* var = (uint8_t*)AscendC::GmAlloc(varByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

//     BatchNormV3InferTilingData* tilingDatafromBin = reinterpret_cast<BatchNormV3InferTilingData*>(tiling);

//     tilingDatafromBin->totalTiles = 1;
//     tilingDatafromBin->tilesPerCore = 1;
//     tilingDatafromBin->usedCoreNums = 1;
//     tilingDatafromBin->totalB0Len = 1;
//     tilingDatafromBin->totalALen = 24;
//     tilingDatafromBin->totalB1Len = 1;
//     tilingDatafromBin->b0Outer = 1;
//     tilingDatafromBin->aOuter = 1;
//     tilingDatafromBin->b1Outer = 1;

//     tilingDatafromBin->tileBlockB0Len = 1;
//     tilingDatafromBin->tileBlockB0Tail = 1;
//     tilingDatafromBin->tileBlockALen = 24;
//     tilingDatafromBin->tileBlockATail = 24;
//     tilingDatafromBin->tileBlockB1Len = 128;
//     tilingDatafromBin->tileBlockB1Tail = 1;
//     tilingDatafromBin->tileBlockAPaddingNum = 0;
//     tilingDatafromBin->epsilon = 0.00001;

//     system("cp -r ../../../../norm/batch_norm_v3/tests/ut/op_kernel/batch_norm_v3_data ./");
//     system("chmod -R 755 ./batch_norm_v3_data/");

//     system("cd ./batch_norm_v3_data/ && rm -rf ./*bin");
//     system(
//         "cd ./batch_norm_v3_data/ && python3 gen_golden_data.py '[1, 24, 1, 1]' 'float16' 'NCHW' '[24]' '0.00001' "
//         "'float16' ");

//     char* path_ = get_current_dir_name();
//     std::string path(path_);
//     ReadFile(path + "/batch_norm_v3_data/x.bin", xByteSize, x, xByteSize);
//     ReadFile(path + "/batch_norm_v3_data/mean.bin", meanByteSize, mean, meanByteSize);
//     ReadFile(path + "/batch_norm_v3_data/var.bin", varByteSize, var, varByteSize);
//     ReadFile(path + "/batch_norm_v3_data/beta.bin", betaByteSize, beta, betaByteSize);
//     ReadFile(path + "/batch_norm_v3_data/gamma.bin", meanByteSize, gamma, meanByteSize);

//     ICPU_SET_TILING_KEY(910000);
//     ICPU_RUN_KF(
//         batch_norm_v3, blockDim, x, gamma, beta, mean, var, y, nullptr, nullptr, nullptr, nullptr, workspace,
//         (uint8_t*)(tilingDatafromBin));

//     WriteFile(path + "/batch_norm_v3_data/y.bin", y, yByteSize);

//     AscendC::GmFree(x);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(beta);
//     AscendC::GmFree(mean);
//     AscendC::GmFree(var);
//     AscendC::GmFree(y);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);

//     int res = system("cd ./batch_norm_v3_data/ && python3 compare_data.py 'float16'");
//     system("rm -rf batch_norm_v3_data");
//     free(path_);
//     ASSERT_EQ(res, 0);
// }

// TEST_F(batch_norm_v3_test, test_case_infer_last_channel_0001)
// {
//     size_t xByteSize = 1 * 128 * sizeof(half);
//     size_t gammaByteSize = 128 * sizeof(half);
//     size_t betaByteSize = 128 * sizeof(half);
//     size_t yByteSize = 1 * 128 * sizeof(half);
//     size_t meanByteSize = 128 * sizeof(half);
//     size_t varByteSize = 128 * sizeof(half);
//     size_t tiling_data_size = sizeof(BatchNormV3InferLastChannelTilingData);
//     uint32_t blockDim = 1;

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
//     uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
//     uint8_t* var = (uint8_t*)AscendC::GmAlloc(varByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

//     BatchNormV3InferLastChannelTilingData* tilingDatafromBin =
//         reinterpret_cast<BatchNormV3InferLastChannelTilingData*>(tiling);

//     tilingDatafromBin->totalTiles = 1;
//     tilingDatafromBin->tilesPerCore = 1;
//     tilingDatafromBin->usedCoreNums = 1;
//     tilingDatafromBin->totalALen = 128;
//     tilingDatafromBin->aOuter = 1;
//     tilingDatafromBin->bOuter = 1;
//     tilingDatafromBin->tileBlockALen = 128;
//     tilingDatafromBin->tileBlockATail = 128;
//     tilingDatafromBin->tileBlockAPaddingNum = 0;
//     tilingDatafromBin->tileBlockBLen = 1;
//     tilingDatafromBin->tileBlockBTail = 1;
//     tilingDatafromBin->epsilon = 0.0001;

//     system("cp -r ../../../../norm/batch_norm_v3/tests/ut/op_kernel/batch_norm_v3_data ./");
//     system("chmod -R 755 ./batch_norm_v3_data/");
//     auto ret = system("cd ./batch_norm_v3_data/ && rm -rf ./*bin");
//     ret = system(
//         "cd ./batch_norm_v3_data/ && python3 gen_golden_data.py '[1, 1, 1, 128]' 'float16' 'NHWC' '[128]' '0.0001' "
//         "'float16' ");

//     char* path_ = get_current_dir_name();
//     std::string path(path_);

//     ReadFile(path + "/batch_norm_v3_data/x.bin", xByteSize, x, xByteSize);
//     ReadFile(path + "/batch_norm_v3_data/mean.bin", meanByteSize, mean, meanByteSize);
//     ReadFile(path + "/batch_norm_v3_data/var.bin", varByteSize, var, varByteSize);
//     ReadFile(path + "/batch_norm_v3_data/beta.bin", betaByteSize, beta, betaByteSize);
//     ReadFile(path + "/batch_norm_v3_data/gamma.bin", meanByteSize, gamma, meanByteSize);

//     ICPU_SET_TILING_KEY(900000);
//     ICPU_RUN_KF(
//         batch_norm_v3, blockDim, x, gamma, beta, mean, var, y, nullptr, nullptr, nullptr, nullptr, workspace,
//         (uint8_t*)(tilingDatafromBin));

//     WriteFile(path + "/batch_norm_v3_data/y.bin", y, yByteSize);

//     AscendC::GmFree(x);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(beta);
//     AscendC::GmFree(mean);
//     AscendC::GmFree(var);
//     AscendC::GmFree(y);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);

//     int res = system("cd ./batch_norm_v3_data/ && python3 compare_data.py 'float16'");
//     system("rm -rf batch_norm_v3_data");
//     free(path_);
//     ASSERT_EQ(res, 0);
// }