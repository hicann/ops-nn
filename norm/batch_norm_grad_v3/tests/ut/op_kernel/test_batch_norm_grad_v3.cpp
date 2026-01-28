/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file test_batch_norm_grad_v3.cpp
 * \brief
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "batch_norm_grad_v3_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void batch_norm_grad_v3(GM_ADDR dy, GM_ADDR x, GM_ADDR weight, GM_ADDR running_mean,
                                                         GM_ADDR running_var, GM_ADDR save_mean, GM_ADDR save_rstd,
                                                         GM_ADDR dx, GM_ADDR dweight, GM_ADDR dbias,
                                                         GM_ADDR workspace, GM_ADDR tiling);

class batch_norm_grad_v3_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "batch_norm_grad_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "batch_norm_grad_v3_test TearDown\n" << endl;
    }
};

inline int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for(auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void ExcuteTestCase(const std::vector<int64_t> &xShape, const std::vector<int64_t> &wShape, const std::string &dtype,
                    const std::string &caseName, int64_t tilingKey, uint32_t blockNum, uint8_t *tiling, bool is_training = true)
{
    uint32_t typeSize = 4;
    uint32_t fp32TypeSize = 4;
    if (dtype != "float") {
      typeSize = 2;
    }
    // 每个block对应 1 aic + 2 aiv
    uint32_t realBlockNum = (blockNum + 1) / 2;
    size_t xFileSize = GetShapeSize(xShape) * typeSize;
    size_t weightFileSize = GetShapeSize(wShape) * typeSize;
    size_t meanFileSize = GetShapeSize(wShape) * fp32TypeSize;

    size_t workspaceFileSize = 16*1024*1024;

    uint8_t *dy = (uint8_t *)AscendC::GmAlloc((xFileSize + 31)/32*32);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc((xFileSize + 31)/32*32);
    uint8_t *weight = (uint8_t *)AscendC::GmAlloc((weightFileSize + 31)/32*32);
    uint8_t *runningMean = (uint8_t *)AscendC::GmAlloc((meanFileSize + 31)/32*32);
    uint8_t *runningVar = (uint8_t *)AscendC::GmAlloc((meanFileSize + 31)/32*32);
    uint8_t *saveMean = (uint8_t *)AscendC::GmAlloc((meanFileSize + 31)/32*32);
    uint8_t *saveRstd = (uint8_t *)AscendC::GmAlloc((meanFileSize + 31)/32*32);
    uint8_t *dx = (uint8_t *)AscendC::GmAlloc((xFileSize + 31)/32*32);
    uint8_t *dweight = (uint8_t *)AscendC::GmAlloc((weightFileSize + 31)/32*32);
    uint8_t *dbias = (uint8_t *)AscendC::GmAlloc((weightFileSize + 31)/32*32);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    // uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);



    // system("cp -r ../../../../norm/batch_norm_grad_v3/tests/ut/op_kernel/batch_norm_grad_v3_data ./");
    // system("chmod -R 755 ./batch_norm_grad_v3_data/");
    // system("cd ./batch_norm_grad_v3_data/ && rm -rf ./*bin");
    // std::string cmd = "cd ./batch_norm_grad_v3_data/ && python3 gen_data.py " + caseName + " " + dtype;
    // system(cmd.c_str());
    // cmd = "cd ./batch_norm_grad_v3_data/ && python3 gen_tiling.py " + caseName;
    // system(cmd.c_str());

    char *path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/batch_norm_grad_v3_data/input_dy.bin", xFileSize, dy, xFileSize);
    // ReadFile(path + "/batch_norm_grad_v3_data/input_x.bin", xFileSize, x, xFileSize);
    // ReadFile(path + "/batch_norm_grad_v3_data/input_weight.bin", weightFileSize, weight, weightFileSize);
    // ReadFile(path + "/batch_norm_grad_v3_data/input_running_mean.bin", meanFileSize, runningMean, meanFileSize);
    // ReadFile(path + "/batch_norm_grad_v3_data/input_running_var.bin", meanFileSize, runningVar, meanFileSize);
    // ReadFile(path + "/batch_norm_grad_v3_data/input_save_mean.bin", meanFileSize, saveMean, meanFileSize);
    // ReadFile(path + "/batch_norm_grad_v3_data/input_save_rstd.bin", meanFileSize, saveRstd, meanFileSize);

    // ReadFile(path + "/batch_norm_grad_v3_data/tiling.bin", tilingSize, tiling, tilingSize);


    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(batch_norm_grad_v3, realBlockNum, dy, x, weight, runningMean, runningVar, saveMean, saveRstd,
                dx, dweight, dbias, workspace, tiling);
    // WriteFile(path + "/batch_norm_grad_v3_data/cce_cpu_dx.bin", dx, xFileSize);
    // WriteFile(path + "/batch_norm_grad_v3_data/cce_cpu_dweight.bin", dweight, weightFileSize);
    // WriteFile(path + "/batch_norm_grad_v3_data/cce_cpu_dbias.bin", dbias, weightFileSize);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(runningMean);
    AscendC::GmFree(runningVar);
    AscendC::GmFree(saveMean);
    AscendC::GmFree(saveRstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dweight);
    AscendC::GmFree(dbias);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    // if (!is_training) {
    //     std::string cmpCmd = "cd ./batch_norm_grad_v3_data/ && python3 compare_data.py " + dtype;
    //     int res = system(cmpCmd.c_str());
    //     system("rm -rf batch_norm_grad_v3_data");
    //     free(path_);
    //     ASSERT_EQ(res, 0);
    //     return;
    // }

    free(path_);
}

TEST_F(batch_norm_grad_v3_test, test_split_r1_float32)
{
    std::vector<int64_t> xShape = {5, 2, 32, 128};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 31000001;
    std::string caseName = "case1";
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradV3RARRecomputeTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3RARRecomputeTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3RARRecomputeTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 5;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 4096;
    tilingDatafromBin->baseTilingData.rAlign = 8192;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 0;
    tilingDatafromBin->baseTilingData.formerBlockDim = 1;
    tilingDatafromBin->baseTilingData.tailBlockDim = 2;
    tilingDatafromBin->generalBinAddTilingData.binaryAddQuotient = 8192;
    tilingDatafromBin->generalBinAddTilingData.binaryAddk = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->tailBinAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->tailBinAddTilingData.binaryAddk = 0;
    tilingDatafromBin->tailBinAddTilingData.binaryAddLastNum = 32;
    tilingDatafromBin->ubRDimFactor = 8192;
    tilingDatafromBin->ubRDimFactorAlign = 8192;
    tilingDatafromBin->ubRDimLoopNum = 2;
    tilingDatafromBin->ubRDimTail = 4096;
    tilingDatafromBin->ubRDimTailFactor = 4096;
    tilingDatafromBin->ubRDimTailFactorAlign = 4096;
    tilingDatafromBin->ubRDimTailLoopNum = 1;
    tilingDatafromBin->ubRDimTailTail = 0;
    tilingDatafromBin->ubRDimTailTailFactor = 0;
    tilingDatafromBin->ubRDimTailTailFactorAlign = 0;
    tilingDatafromBin->ubRDimTailTailLoopNum = 0;
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_v3_test, test_split_r1_float16)
{
    std::vector<int64_t> xShape = {170, 2, 2, 125};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float16";
    uint64_t tilingKey = 31000002;
    std::string caseName = "case2";
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradV3RARRecomputeTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3RARRecomputeTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3RARRecomputeTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 170;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 250;
    tilingDatafromBin->baseTilingData.rAlign = 8000;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 0;
    tilingDatafromBin->baseTilingData.formerBlockDim = 1;
    tilingDatafromBin->baseTilingData.tailBlockDim = 2;
    tilingDatafromBin->generalBinAddTilingData.binaryAddQuotient = 8192;
    tilingDatafromBin->generalBinAddTilingData.binaryAddk = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->tailBinAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->tailBinAddTilingData.binaryAddk = 0;
    tilingDatafromBin->tailBinAddTilingData.binaryAddLastNum = 32;
    tilingDatafromBin->ubRDimFactor = 16000;
    tilingDatafromBin->ubRDimFactorAlign = 16000;
    tilingDatafromBin->ubRDimLoopNum = 2;
    tilingDatafromBin->ubRDimTail = 10500;
    tilingDatafromBin->ubRDimTailFactor = 8000;
    tilingDatafromBin->ubRDimTailFactorAlign = 8000;
    tilingDatafromBin->ubRDimTailLoopNum = 1;
    tilingDatafromBin->ubRDimTailTail = 0;
    tilingDatafromBin->ubRDimTailTailFactor = 0;
    tilingDatafromBin->ubRDimTailTailFactorAlign = 0;
    tilingDatafromBin->ubRDimTailTailLoopNum = 0;
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_v3_test, test_split_r1_bfloat16)
{
    std::vector<int64_t> xShape = {8, 2, 8, 16, 16};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "bfloat16";
    uint64_t tilingKey = 31000003;
    std::string caseName = "case3";
    uint32_t blockNum = 4;
    size_t tilingSize = sizeof(BatchNormGradV3RARRecomputeTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3RARRecomputeTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3RARRecomputeTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 8;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 2048;
    tilingDatafromBin->baseTilingData.rAlign = 0;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 2;
    tilingDatafromBin->baseTilingData.formerBlockDim = 0;
    tilingDatafromBin->baseTilingData.tailBlockDim = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddQuotient = 8192;
    tilingDatafromBin->generalBinAddTilingData.binaryAddk = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->tailBinAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->tailBinAddTilingData.binaryAddk = 0;
    tilingDatafromBin->tailBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->ubRDimFactor = 12288;
    tilingDatafromBin->ubRDimFactorAlign = 12288;
    tilingDatafromBin->ubRDimLoopNum = 1;
    tilingDatafromBin->ubRDimTail = 16384;
    tilingDatafromBin->ubRDimTailFactor = 6144;
    tilingDatafromBin->ubRDimTailFactorAlign = 6144;
    tilingDatafromBin->ubRDimTailLoopNum = 2;
    tilingDatafromBin->ubRDimTailTail = 4096;
    tilingDatafromBin->ubRDimTailTailFactor = 6144;
    tilingDatafromBin->ubRDimTailTailFactorAlign = 6144;
    tilingDatafromBin->ubRDimTailTailLoopNum = 2;
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_v3_test, test_full_load_float32)
{
    std::vector<int64_t> xShape = {32, 2, 13 ,16};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 10000001;
    std::string caseName = "case4";
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradV3RARFullLoadTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3RARFullLoadTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3RARFullLoadTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 32;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 208;
    tilingDatafromBin->baseTilingData.rAlign = 6656;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 0;
    tilingDatafromBin->baseTilingData.formerBlockDim = 1;
    tilingDatafromBin->baseTilingData.tailBlockDim = 2;
    tilingDatafromBin->binaryAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->binaryAddTilingData.binaryAddk = 0;
    tilingDatafromBin->binaryAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->formerUbDim = 2;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfTailBlock = 2;
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_v3_test, test_full_load_float16)
{
    std::vector<int64_t> xShape = {32, 2, 16 ,16};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float16";
    uint64_t tilingKey = 10000002;
    std::string caseName = "case5";
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradV3RARFullLoadTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3RARFullLoadTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3RARFullLoadTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 32;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 256;
    tilingDatafromBin->baseTilingData.rAlign = 8192;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 0;
    tilingDatafromBin->baseTilingData.formerBlockDim = 1;
    tilingDatafromBin->baseTilingData.tailBlockDim = 2;
    tilingDatafromBin->binaryAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->binaryAddTilingData.binaryAddk = 0;
    tilingDatafromBin->binaryAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->formerUbDim = 2;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfTailBlock = 2;
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_v3_test, test_full_load_bfloat16)
{
    std::vector<int64_t> xShape = {32, 2, 16, 16};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "bfloat16";
    uint64_t tilingKey = 10000003;
    std::string caseName = "case6";
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradV3RARFullLoadTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3RARFullLoadTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3RARFullLoadTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 32;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 256;
    tilingDatafromBin->baseTilingData.rAlign = 8192;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 0;
    tilingDatafromBin->baseTilingData.formerBlockDim = 1;
    tilingDatafromBin->baseTilingData.tailBlockDim = 2;
    tilingDatafromBin->binaryAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->binaryAddTilingData.binaryAddk = 0;
    tilingDatafromBin->binaryAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->formerUbDim = 2;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfTailBlock = 2;
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_v3_test, test_split_r0_float32)
{
    std::vector<int64_t> xShape = {1, 2, 448, 448};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 32000001;
    std::string caseName = "case7";
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradV3RARRecomputeTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3RARRecomputeTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3RARRecomputeTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 1;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 200704;
    tilingDatafromBin->baseTilingData.rAlign = 0;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 2;
    tilingDatafromBin->baseTilingData.formerBlockDim = 0;
    tilingDatafromBin->baseTilingData.tailBlockDim = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddQuotient = 8192;
    tilingDatafromBin->generalBinAddTilingData.binaryAddk = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->tailBinAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->tailBinAddTilingData.binaryAddk = 0;
    tilingDatafromBin->tailBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->ubRDimFactor = 14880;
    tilingDatafromBin->ubRDimFactorAlign = 14880;
    tilingDatafromBin->ubRDimLoopNum = 8;
    tilingDatafromBin->ubRDimTail = 81664;
    tilingDatafromBin->ubRDimTailFactor = 7440;
    tilingDatafromBin->ubRDimTailFactorAlign = 7440;
    tilingDatafromBin->ubRDimTailLoopNum = 10;
    tilingDatafromBin->ubRDimTailTail = 7264;
    tilingDatafromBin->ubRDimTailTailFactor = 7440;
    tilingDatafromBin->ubRDimTailTailFactorAlign = 7440;
    tilingDatafromBin->ubRDimTailTailLoopNum = 2;
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_v3_test, test_split_r0_float16)
{
    std::vector<int64_t> xShape = {10, 2, 192, 108};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float16";
    uint64_t tilingKey = 32000002;
    std::string caseName = "case8";
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradV3RARRecomputeTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3RARRecomputeTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3RARRecomputeTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 10;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 20736;
    tilingDatafromBin->baseTilingData.rAlign = 0;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 2;
    tilingDatafromBin->baseTilingData.formerBlockDim = 0;
    tilingDatafromBin->baseTilingData.tailBlockDim = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddQuotient = 8192;
    tilingDatafromBin->generalBinAddTilingData.binaryAddk = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->tailBinAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->tailBinAddTilingData.binaryAddk = 0;
    tilingDatafromBin->tailBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->ubRDimFactor = 14880;
    tilingDatafromBin->ubRDimFactorAlign = 14880;
    tilingDatafromBin->ubRDimLoopNum = 1;
    tilingDatafromBin->ubRDimTail = 5856;
    tilingDatafromBin->ubRDimTailFactor = 7440;
    tilingDatafromBin->ubRDimTailFactorAlign = 7440;
    tilingDatafromBin->ubRDimTailLoopNum = 0;
    tilingDatafromBin->ubRDimTailTail = 5856;
    tilingDatafromBin->ubRDimTailTailFactor = 7440;
    tilingDatafromBin->ubRDimTailTailFactorAlign = 7440;
    tilingDatafromBin->ubRDimTailTailLoopNum = 2;
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

// TEST_F(batch_norm_grad_v3_test, test_split_core_r1_float32)
// {
//     std::vector<int64_t> xShape = {1000, 2, 448, 448};
//     std::vector<int64_t> wShape = {2};
//     std::string dtype = "float";
//     uint64_t tilingKey = 1000;
//     std::string caseName = "test_split_core_r1_float32";
//     uint32_t blockNum = 30;
//     size_t tilingSize = sizeof(BatchNormGradV3RARSplitCoreR1TilingData);
//     uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
//     BatchNormGradV3RARSplitCoreR1TilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3RARSplitCoreR1TilingData*>(tiling);

//     tilingDatafromBin->baseTilingData.r1Dim = 10;
//     tilingDatafromBin->baseTilingData.aDim = 2;
//     tilingDatafromBin->baseTilingData.r0Dim = 20736;
//     tilingDatafromBin->baseTilingData.rAlign = 0;
//     tilingDatafromBin->baseTilingData.blockNum = 2;
//     tilingDatafromBin->baseTilingData.tailBlockNum = 2;
//     tilingDatafromBin->baseTilingData.formerBlockDim = 0;
//     tilingDatafromBin->baseTilingData.tailBlockDim = 1;
//     tilingDatafromBin->r1Dim = 0;
//     tilingDatafromBin->aDim = 0;
//     tilingDatafromBin->aDimAligned = 0;
//     tilingDatafromBin->r0Dim = 0;
//     tilingDatafromBin->usedCoreNums = 0;
//     tilingDatafromBin->r1Inner = 0;
//     tilingDatafromBin->r1Tail = 0;
//     tilingDatafromBin->r0InnerStg0 = 0;
//     tilingDatafromBin->r0OuterStg0 = 0;
//     tilingDatafromBin->r0TailStg0 = 0;
//     tilingDatafromBin->r1InnerInnerStg0 = 0;
//     tilingDatafromBin->r1InnerOuterStg0 = 0;
//     tilingDatafromBin->r1InnerTailStg0 = 0;
//     tilingDatafromBin->r1TailOuterStg0 = 0;
//     tilingDatafromBin->r1TailTailStg0 = 0;
//     tilingDatafromBin->aInnerStg0 = 0;
//     tilingDatafromBin->aInnerAlignedStg0 = 0;
//     tilingDatafromBin->aOuterStg0 = 0;
//     tilingDatafromBin->aTailStg0 = 0;
//     tilingDatafromBin->aInnerStg1 = 0;
//     tilingDatafromBin->aOuterStg1 = 0;
//     tilingDatafromBin->aTailStg1 = 0;
//     tilingDatafromBin->r0InnerStg2 = 0;
//     tilingDatafromBin->r0OuterStg2 = 0;
//     tilingDatafromBin->r0TailStg2 = 0;
//     tilingDatafromBin->r1InnerInnerStg2 = 0;
//     tilingDatafromBin->r1InnerOuterStg2 = 0;
//     tilingDatafromBin->r1InnerTailStg2 = 0;
//     tilingDatafromBin->r1TailOuterStg2 = 0;
//     tilingDatafromBin->r1TailTailStg2 = 0;
//     tilingDatafromBin->aInnerStg2 = 0;
//     tilingDatafromBin->aInnerAlignedStg2 = 0;
//     tilingDatafromBin->aOuterStg2 = 0;
//     tilingDatafromBin->aTailStg2 = 0;
//     ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
// }

TEST_F(batch_norm_grad_v3_test, test_infer_channel_last_float32)
{
    std::vector<int64_t> xShape = {10, 2, 3, 200};
    std::vector<int64_t> wShape = {200};
    std::string dtype = "float";
    uint64_t tilingKey = 900001;
    std::string caseName = "infer_channel_last_fp32";
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradV3InferChannelLastTilingData) + sizeof(float);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3InferChannelLastTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3InferChannelLastTilingData*>(tiling);
    tilingDatafromBin->totalTiles = 0;
    tilingDatafromBin->tilesPerCore = 0;
    tilingDatafromBin->usedCoreNums = 0;
    tilingDatafromBin->totalALen = 0;
    tilingDatafromBin->aOuter = 0;
    tilingDatafromBin->bOuter = 0;
    tilingDatafromBin->tileBlockALen = 0;
    tilingDatafromBin->tileBlockATail = 0;
    tilingDatafromBin->tileBlockAPaddingNum = 0;
    tilingDatafromBin->tileBlockBLen = 0;
    tilingDatafromBin->tileBlockBTail = 0;
    tilingDatafromBin->epsilon = 0;
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_v3_test, test_infer_float32)
{
    std::vector<int64_t> xShape = {1, 2, 1, 1};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 910001;
    std::string caseName = "infer_fp32";
    uint32_t blockNum = 1;
    size_t tilingSize = sizeof(BatchNormGradV3InferTilingData) + sizeof(float);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3InferTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3InferTilingData*>(tiling);
    tilingDatafromBin->totalTiles = 1;
    tilingDatafromBin->tilesPerCore = 1;
    tilingDatafromBin->usedCoreNums = 1;
    tilingDatafromBin->totalB0Len = 1;
    tilingDatafromBin->totalALen = 2;
    tilingDatafromBin->totalB1Len = 1;
    tilingDatafromBin->b0Outer = 1;
    tilingDatafromBin->aOuter = 1;
    tilingDatafromBin->b1Outer = 1;
    tilingDatafromBin->tileBlockB0Len = 1;
    tilingDatafromBin->tileBlockB0Tail = 1;
    tilingDatafromBin->tileBlockALen = 2;
    tilingDatafromBin->tileBlockATail = 2;
    tilingDatafromBin->tileBlockB1Len = 64;
    tilingDatafromBin->tileBlockB1Tail = 1;
    tilingDatafromBin->tileBlockAPaddingNum = 0;
    tilingDatafromBin->epsilon = 0;

    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_v3_test, test_infer_full_load_float32_910b)
{
    std::vector<int64_t> xShape = {1, 2, 1, 1};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 10001;
    std::string caseName = "infer_full_load_fp32";
    uint32_t blockNum = 1;
    size_t tilingSize = sizeof(BatchNormGradV3FullLoadTilingData) + sizeof(float);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    BatchNormGradV3FullLoadTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3FullLoadTilingData*>(tiling);

    tilingDatafromBin->b1Dim = 1;
    tilingDatafromBin->aDim = 2;
    tilingDatafromBin->b0Dim = 1;
    tilingDatafromBin->bAlign = 8;
    tilingDatafromBin->coreChannelNum = 1;
    tilingDatafromBin->coreChannelNumTail = 1;
    tilingDatafromBin->cUbBlock = 956;
    tilingDatafromBin->needCoreNum = 2;
    tilingDatafromBin->epsilon = 0;

    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin, false);
}

TEST_F(batch_norm_grad_v3_test, test_infer_split_load_cross_core_float32_910b)
{
    std::vector<int64_t> xShape = {10, 2, 100, 100};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 20011;
    std::string caseName = "infer_split_load_fp32";
    uint32_t blockNum = 40;
    size_t tilingSize = sizeof(BatchNormGradV3SplitLoadCrossCoreTilingData) + sizeof(float);
    // uint32_t tilingData[tilingSize] = {10000, 2, 100, 1000000, 1, 1, 6128, 1, 2, 3872, 2, 0};
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3SplitLoadCrossCoreTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3SplitLoadCrossCoreTilingData*>(tiling);

    tilingDatafromBin->b1Dim = 1000;
    tilingDatafromBin->aDim = 2;
    tilingDatafromBin->b0Dim = 10;
    tilingDatafromBin->bAlign = 100000;
    tilingDatafromBin->coreChannelNum = 0;
    tilingDatafromBin->coreChannelNumTail = 0;
    tilingDatafromBin->bUbBlock = 12272;
    tilingDatafromBin->b0UbBlock = 1;
    tilingDatafromBin->bUbLoop = 1;
    tilingDatafromBin->bUbTailLoop = 0;
    tilingDatafromBin->bUbBlockTail = 10000;
    tilingDatafromBin->b0UbBlockTail = 0;
    tilingDatafromBin->b0UbTailBlockTail = 0;
    tilingDatafromBin->needCoreNum = 20;
    tilingDatafromBin->groupCore = 10;
    tilingDatafromBin->groupBlockNum = 1;
    tilingDatafromBin->groupBlockNumTail = 1;
    tilingDatafromBin->morehalfChannel = 0;
    tilingDatafromBin->groupBlockNumHalf = 0;
    tilingDatafromBin->groupBlockNumHalfTail = 0;
    tilingDatafromBin->bUbLoopHalf = 0;
    tilingDatafromBin->b0UbBlockTailHalf = 0;
    tilingDatafromBin->bUbTailLoopHalf = 0;
    tilingDatafromBin->b0UbTailBlockTailHalf = 0;
    tilingDatafromBin->moreMultiChannel = 0;
    tilingDatafromBin->bUbLoopMulti = 1;
    tilingDatafromBin->b0UbBlockTailMulti = 10000;
    tilingDatafromBin->coreNum = 40;
    tilingDatafromBin->epsilon = 0;
    
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin, false);
}


TEST_F(batch_norm_grad_v3_test, test_infer_split_load_float32_910b)
{
    std::vector<int64_t> xShape = {10, 40, 100, 100};
    std::vector<int64_t> wShape = {40};
    std::string dtype = "float";
    uint64_t tilingKey = 20001;
    std::string caseName = "infer_split_load_fp32";
    uint32_t blockNum = 40;
    size_t tilingSize = sizeof(BatchNormGradV3SplitLoadTilingData) + sizeof(float);
    // uint32_t tilingData[tilingSize] = {10000, 2, 100, 1000000, 1, 1, 6128, 1, 2, 3872, 2, 0};
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3SplitLoadTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3SplitLoadTilingData*>(tiling);

    tilingDatafromBin->b1Dim = 1000;
    tilingDatafromBin->aDim = 40;
    tilingDatafromBin->b0Dim = 10;
    tilingDatafromBin->bAlign = 100000;
    tilingDatafromBin->coreChannelNum = 1;
    tilingDatafromBin->coreChannelNumTail = 1;
    tilingDatafromBin->bUbBlock = 12264;
    tilingDatafromBin->b0UbBlock = 1;
    tilingDatafromBin->bUbLoop = 1;
    tilingDatafromBin->bUbBlockTail = 10000;
    tilingDatafromBin->b0UbBlockTail = 0;
    tilingDatafromBin->needCoreNum = 40;
    tilingDatafromBin->epsilon = 0;

    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin, false);
}

TEST_F(batch_norm_grad_v3_test, test_train_full_load_float32_910b)
{
    std::vector<int64_t> xShape = {1, 2, 1, 1};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 30001;
    std::string caseName = "train_full_load_fp32";
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradV3FullLoadTilingData) + sizeof(float);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3FullLoadTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3FullLoadTilingData*>(tiling);

    tilingDatafromBin->b1Dim = 1;
    tilingDatafromBin->aDim = 2;
    tilingDatafromBin->b0Dim = 1;
    tilingDatafromBin->bAlign = 8;
    tilingDatafromBin->coreChannelNum = 1;
    tilingDatafromBin->coreChannelNumTail = 1;
    tilingDatafromBin->cUbBlock = 716;
    tilingDatafromBin->needCoreNum = 2;
    tilingDatafromBin->epsilon = 0;

    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin, false);
}

TEST_F(batch_norm_grad_v3_test, test_train_split_load_float32_910b)
{
    std::vector<int64_t> xShape = {10, 40, 100, 100};
    std::vector<int64_t> wShape = {40};
    std::string dtype = "float";
    uint64_t tilingKey = 40001;
    std::string caseName = "train_split_load_fp32";
    uint32_t blockNum = 40;
    size_t tilingSize = sizeof(BatchNormGradV3SplitLoadTilingData) + sizeof(float);
    // uint32_t tilingData[tilingSize] = {10000, 2, 100, 1000000, 1, 1, 6128, 1, 2, 3872, 2, 0};
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3SplitLoadTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3SplitLoadTilingData*>(tiling);

    tilingDatafromBin->b1Dim = 1000;
    tilingDatafromBin->aDim = 40;
    tilingDatafromBin->b0Dim = 10;
    tilingDatafromBin->bAlign = 100000;
    tilingDatafromBin->coreChannelNum = 1;
    tilingDatafromBin->coreChannelNumTail = 1;
    tilingDatafromBin->bUbBlock = 12264;
    tilingDatafromBin->b0UbBlock = 1;
    tilingDatafromBin->bUbLoop = 1;
    tilingDatafromBin->bUbBlockTail = 10000;
    tilingDatafromBin->b0UbBlockTail = 0;
    tilingDatafromBin->needCoreNum = 40;
    tilingDatafromBin->epsilon = 0;

    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin, false);
}

TEST_F(batch_norm_grad_v3_test, test_train_split_load_cross_core_float32_910b)
{
    std::vector<int64_t> xShape = {10, 2, 100, 100};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 40011;
    std::string caseName = "train_split_load_fp32";
    uint32_t blockNum = 40;
    size_t tilingSize = sizeof(BatchNormGradV3SplitLoadCrossCoreTilingData) + sizeof(float);
    // uint32_t tilingData[tilingSize] = {10000, 2, 100, 1000000, 1, 1, 6128, 1, 2, 3872, 2, 0};
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradV3SplitLoadCrossCoreTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradV3SplitLoadCrossCoreTilingData*>(tiling);

    tilingDatafromBin->b1Dim = 1000;
    tilingDatafromBin->aDim = 2;
    tilingDatafromBin->b0Dim = 10;
    tilingDatafromBin->bAlign = 100000;
    tilingDatafromBin->coreChannelNum = 0;
    tilingDatafromBin->coreChannelNumTail = 0;
    tilingDatafromBin->bUbBlock = 12272;
    tilingDatafromBin->b0UbBlock = 1;
    tilingDatafromBin->bUbLoop = 1;
    tilingDatafromBin->bUbTailLoop = 0;
    tilingDatafromBin->bUbBlockTail = 10000;
    tilingDatafromBin->b0UbBlockTail = 0;
    tilingDatafromBin->b0UbTailBlockTail = 0;
    tilingDatafromBin->needCoreNum = 20;
    tilingDatafromBin->groupCore = 10;
    tilingDatafromBin->groupBlockNum = 1;
    tilingDatafromBin->groupBlockNumTail = 1;
    tilingDatafromBin->morehalfChannel = 0;
    tilingDatafromBin->groupBlockNumHalf = 0;
    tilingDatafromBin->groupBlockNumHalfTail = 0;
    tilingDatafromBin->bUbLoopHalf = 0;
    tilingDatafromBin->b0UbBlockTailHalf = 0;
    tilingDatafromBin->bUbTailLoopHalf = 0;
    tilingDatafromBin->b0UbTailBlockTailHalf = 0;
    tilingDatafromBin->moreMultiChannel = 0;
    tilingDatafromBin->bUbLoopMulti = 1;
    tilingDatafromBin->b0UbBlockTailMulti = 10000;
    tilingDatafromBin->coreNum = 40;
    tilingDatafromBin->epsilon = 0;
    
    ExcuteTestCase(xShape, wShape, dtype, caseName, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin, false);
}
