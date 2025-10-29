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

#include <iostream>
#include "gtest/gtest.h"
#include "data_utils.h"
#include "tikicpulib.h"
#include "group_norm_swish_tiling.h"

extern "C" __global__ __aicore__ void group_norm_swish(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling);
class group_norm_swish_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "group_norm_swish_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "group_norm_swish_test TearDown\n" << std::endl;
    }
};

inline static int64_t CeilDiv(int64_t value, int64_t factor) {
    if (factor == 0) {
        return value;
    } else if (value % factor == 0) {
        return value / factor;
    } else {
        return value / factor + 1;
    }
}

inline static int64_t CeilInt(int64_t value, int64_t factor) {
    return CeilDiv(value, factor) * factor;
}

TEST_F(group_norm_swish_test, test_case_211) {
  int64_t N = 1024;
  int64_t C = 32;
  int64_t H = 1;
  int64_t W = 1;
  int64_t G = 32;
  size_t inputXSize = N * C * H * sizeof(float);
  size_t inputGammaSize = C * sizeof(float);
  size_t outputMeanSize = N * G * sizeof(float);
  size_t tiling_data_size = sizeof(GroupNormSwishTilingData);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputGammaSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputGammaSize);
  uint8_t* swish = (uint8_t*)AscendC::GmAlloc(inputXSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(outputMeanSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputMeanSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 32;
  system("cp -r ../../../../norm/group_norm_swish/tests/ut/op_kernel/group_norm_swish_data ./");
  system("chmod -R 755 ./group_norm_swish_data/");
  system("cd ./group_norm_swish_data/ && rm -rf ./*bin");
  system("cd ./group_norm_swish_data/ && python3 gen_data.py 1024 32 1 1 float32 float32");

  char* path_ = get_current_dir_name();
  std::string path(path_);

  GroupNormSwishTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishTilingData*>(tiling);

  tilingDatafromBin->numGroups = G;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSwish = false;
  tilingDatafromBin->swishScale = 1.0;
  tilingDatafromBin->hwNum = H * W;
  tilingDatafromBin->shapeC = C;
  tilingDatafromBin->shapeCAlign = (C + 15) / 16 * 16;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->numPerGroup = tilingDatafromBin->shapeD * tilingDatafromBin->hwNum;
  tilingDatafromBin->groupPerCore = CeilDiv(N * G, blockDim);
  tilingDatafromBin->groupLastCore = N * G - (blockDim - 1) * tilingDatafromBin->groupPerCore;
  tilingDatafromBin->groupPerCoreAlign = CeilInt(tilingDatafromBin->groupPerCore, 8);
  tilingDatafromBin->numPerLoop = 8192;
  tilingDatafromBin->loopTimes = CeilDiv(tilingDatafromBin->numPerGroup, 8192);
  tilingDatafromBin->loopTimesAlign = CeilInt(tilingDatafromBin->loopTimes, 8);
  tilingDatafromBin->numTailLoop = tilingDatafromBin->numPerGroup - (tilingDatafromBin->loopTimes - 1) * 8192;

  ReadFile(path + "/group_norm_swish_data/input_x.bin", inputXSize, x, inputXSize);
  ReadFile(path + "/group_norm_swish_data/input_gamma.bin", inputGammaSize, beta, inputGammaSize);
  ReadFile(path + "/group_norm_swish_data/input_beta.bin", inputGammaSize, gamma, inputGammaSize);
  ICPU_SET_TILING_KEY(211);
  ICPU_RUN_KF(group_norm_swish, blockDim / 2, x, gamma, beta, swish, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(swish);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(group_norm_swish_test, test_case_212) {
  int64_t N = 32;
  int64_t C = 64;
  int64_t H = 1024;
  int64_t W = 1;
  int64_t G = 32;
  size_t inputXSize = N * C * H * sizeof(float);
  size_t inputGammaSize = C * sizeof(float);
  size_t outputMeanSize = N * G * sizeof(float);
  size_t tiling_data_size = sizeof(GroupNormSwishTilingData);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputGammaSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputGammaSize);
  uint8_t* swish = (uint8_t*)AscendC::GmAlloc(inputXSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(outputMeanSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputMeanSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 32;
  system("cp -r ../../../../norm/group_norm_swish/tests/ut/op_kernel/group_norm_swish_data ./");
  system("chmod -R 755 ./group_norm_swish_data/");
  system("cd ./group_norm_swish_data/ && rm -rf ./*bin");
  system("cd ./group_norm_swish_data/ && python3 gen_data.py 32 64 1024 1 float32 float32");

  char* path_ = get_current_dir_name();
  std::string path(path_);

  GroupNormSwishTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishTilingData*>(tiling);

  tilingDatafromBin->numGroups = G;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSwish = false;
  tilingDatafromBin->swishScale = 1.0;
  tilingDatafromBin->hwNum = H * W;
  tilingDatafromBin->shapeC = C;
  tilingDatafromBin->shapeCAlign = (C + 15) / 16 * 16;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->numPerGroup = tilingDatafromBin->shapeD * tilingDatafromBin->hwNum;
  tilingDatafromBin->groupPerCore = CeilDiv(N * G, blockDim);
  tilingDatafromBin->groupLastCore = N * G - (blockDim - 1) * tilingDatafromBin->groupPerCore;
  tilingDatafromBin->groupPerCoreAlign = CeilInt(tilingDatafromBin->groupPerCore, 8);
  tilingDatafromBin->numPerLoop = 8192;
  tilingDatafromBin->loopTimes = CeilDiv(tilingDatafromBin->numPerGroup, 8192);
  tilingDatafromBin->loopTimesAlign = CeilInt(tilingDatafromBin->loopTimes, 8);
  tilingDatafromBin->numTailLoop = tilingDatafromBin->numPerGroup - (tilingDatafromBin->loopTimes - 1) * 8192;

  ReadFile(path + "/group_norm_swish_data/input_x.bin", inputXSize, x, inputXSize);
  ReadFile(path + "/group_norm_swish_data/input_gamma.bin", inputGammaSize, beta, inputGammaSize);
  ReadFile(path + "/group_norm_swish_data/input_beta.bin", inputGammaSize, gamma, inputGammaSize);
  ICPU_SET_TILING_KEY(212);
  ICPU_RUN_KF(group_norm_swish, blockDim / 2, x, gamma, beta, swish, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(swish);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}



TEST_F(group_norm_swish_test, test_case_213) {
  int64_t N = 32;
  int64_t C = 128;
  int64_t H = 8192;
  int64_t W = 1;
  int64_t G = 32;
  size_t inputXSize = N * C * H * sizeof(float);
  size_t inputGammaSize = C * sizeof(float);
  size_t outputMeanSize = N * G * sizeof(float);
  size_t tiling_data_size = sizeof(GroupNormSwishTilingData);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputGammaSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputGammaSize);
  uint8_t* swish = (uint8_t*)AscendC::GmAlloc(inputXSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(outputMeanSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputMeanSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 32;
  system("cp -r ../../../../norm/group_norm_swish/tests/ut/op_kernel/group_norm_swish_data ./");
  system("chmod -R 755 ./group_norm_swish_data/");
  system("cd ./group_norm_swish_data/ && rm -rf ./*bin");
  system("cd ./group_norm_swish_data/ && python3 gen_data.py 32 128 8192 1 float32 float32");

  char* path_ = get_current_dir_name();
  std::string path(path_);

  GroupNormSwishTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishTilingData*>(tiling);

  tilingDatafromBin->numGroups = G;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSwish = false;
  tilingDatafromBin->swishScale = 1.0;
  tilingDatafromBin->hwNum = H * W;
  tilingDatafromBin->shapeC = C;
  tilingDatafromBin->shapeCAlign = (C + 15) / 16 * 16;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->numPerGroup = tilingDatafromBin->shapeD * tilingDatafromBin->hwNum;
  tilingDatafromBin->groupPerCore = 1;
  tilingDatafromBin->groupLastCore = 1;
  tilingDatafromBin->groupPerCoreAlign = 8;
  tilingDatafromBin->numPerLoop = 8192;
  tilingDatafromBin->loopTimes = 1;
  tilingDatafromBin->loopTimesAlign = 8;
  tilingDatafromBin->numTailLoop = 1;

  ReadFile(path + "/group_norm_swish_data/input_x.bin", inputXSize, x, inputXSize);
  ReadFile(path + "/group_norm_swish_data/input_gamma.bin", inputGammaSize, beta, inputGammaSize);
  ReadFile(path + "/group_norm_swish_data/input_beta.bin", inputGammaSize, gamma, inputGammaSize);
  ICPU_SET_TILING_KEY(213);
  ICPU_RUN_KF(group_norm_swish, blockDim / 2, x, gamma, beta, swish, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(swish);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}



TEST_F(group_norm_swish_test, test_case_214) {
  int64_t N = 32;
  int64_t C = 8200;
  int64_t H = 32;
  int64_t W = 1;
  int64_t G = 1;
  size_t inputXSize = N * C * H * sizeof(float);
  size_t inputGammaSize = C * sizeof(float);
  size_t outputMeanSize = N * G * sizeof(float);
  size_t tiling_data_size = sizeof(GroupNormSwishTilingData);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputGammaSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputGammaSize);
  uint8_t* swish = (uint8_t*)AscendC::GmAlloc(inputXSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(outputMeanSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputMeanSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 32;
  system("cp -r ../../../../norm/group_norm_swish/tests/ut/op_kernel/group_norm_swish_data ./");
  system("chmod -R 755 ./group_norm_swish_data/");
  system("cd ./group_norm_swish_data/ && rm -rf ./*bin");
  system("cd ./group_norm_swish_data/ && python3 gen_data.py 32 8200 32 1 float32 float32");

  char* path_ = get_current_dir_name();
  std::string path(path_);

  GroupNormSwishTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSwishTilingData*>(tiling);

  tilingDatafromBin->numGroups = G;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSwish = false;
  tilingDatafromBin->swishScale = 1.0;
  tilingDatafromBin->hwNum = H * W;
  tilingDatafromBin->shapeC = C;
  tilingDatafromBin->shapeCAlign = (C + 15) / 16 * 16;
  tilingDatafromBin->shapeD = 1;
  tilingDatafromBin->numPerGroup = tilingDatafromBin->shapeD * tilingDatafromBin->hwNum;
  tilingDatafromBin->groupPerCore = 1;
  tilingDatafromBin->groupLastCore = 1;
  tilingDatafromBin->groupPerCoreAlign = 8;
  tilingDatafromBin->numPerLoop = 8192;
  tilingDatafromBin->loopTimes = 1;
  tilingDatafromBin->loopTimesAlign = 8;
  tilingDatafromBin->numTailLoop = 1;

  ReadFile(path + "/group_norm_swish_data/input_x.bin", inputXSize, x, inputXSize);
  ReadFile(path + "/group_norm_swish_data/input_gamma.bin", inputGammaSize, beta, inputGammaSize);
  ReadFile(path + "/group_norm_swish_data/input_beta.bin", inputGammaSize, gamma, inputGammaSize);
  ICPU_SET_TILING_KEY(214);
  ICPU_RUN_KF(group_norm_swish, blockDim / 2, x, gamma, beta, swish, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(swish);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}
