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
#include "group_norm_silu_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void group_norm_silu(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu,
                                                      GM_ADDR mean, GM_ADDR rstd,
                                                      GM_ADDR workspace, GM_ADDR tiling);

class group_norm_silu_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "group_norm_silu_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "group_norm_silu_test TearDown\n" << endl;
  }
};

TEST_F(group_norm_silu_test, test_case_102) {
  size_t inputByteSize = 64 * sizeof(float);
  size_t outputByteSize = 3 * 64 * 64 * 64 * sizeof(float);
  size_t meanByteSize = 3 * 32 * sizeof(float);
  size_t tiling_data_size = sizeof(GroupNormSiluTilingData);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* silu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 48;
  system("cp -r ../../../../norm/group_norm_silu/tests/ut/op_kernel/group_norm_silu_data ./");
  system("chmod -R 755 ./group_norm_silu_data/");
  system("cd ./group_norm_silu_data/ && rm -rf ./*bin");
  system("cd ./group_norm_silu_data/ && python3 gen_data.py 3 64 64 64 float32 float32");

  char* path_ = get_current_dir_name();
  string path(path_);

  GroupNormSiluTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSiluTilingData*>(tiling);

  tilingDatafromBin->numGroups = 32;
  tilingDatafromBin->hwNum = 64 * 64;
  tilingDatafromBin->shapeC = 64;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->elemNum = tilingDatafromBin->hwNum *  tilingDatafromBin->shapeD;
  tilingDatafromBin->realCoreNum = 48;
  tilingDatafromBin->numPerCore = 2;
  tilingDatafromBin->numLastCore = 2;
  tilingDatafromBin->processSize = 8192;
  tilingDatafromBin->loopNum = 1;
  tilingDatafromBin->loopTail = 8192;
  tilingDatafromBin->innerLoopNum = 1;
  tilingDatafromBin->innerLoopTail = 4096;
  tilingDatafromBin->tilingKey = 102;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSilu = true;

  ReadFile(path + "/group_norm_silu_data/input_x.bin", outputByteSize, x, outputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_gamma.bin", inputByteSize, beta, inputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_beta.bin", inputByteSize, gamma, inputByteSize);
  ICPU_SET_TILING_KEY(102);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(group_norm_silu, blockDim, x, gamma, beta, silu, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(silu);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(group_norm_silu_test, test_case_102_2) {
  size_t inputByteSize = 192 * sizeof(float);
  size_t outputByteSize = 1 * 192 * 64 * 64 * sizeof(float);
  size_t meanByteSize = 1 * 192 * sizeof(float);
  size_t tiling_data_size = sizeof(GroupNormSiluTilingData);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* silu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 32;
  system("cp -r ../../../../norm/group_norm_silu/tests/ut/op_kernel/group_norm_silu_data ./");
  system("chmod -R 755 ./group_norm_silu_data/");
  system("cd ./group_norm_silu_data/ && rm -rf ./*bin");
  system("cd ./group_norm_silu_data/ && python3 gen_data.py 1 192 64 64 float32 float32");

  char* path_ = get_current_dir_name();
  string path(path_);

  GroupNormSiluTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSiluTilingData*>(tiling);

  tilingDatafromBin->numGroups = 32;
  tilingDatafromBin->hwNum = 64 * 64;
  tilingDatafromBin->shapeC = 192;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->elemNum = tilingDatafromBin->hwNum *  tilingDatafromBin->shapeD;
  tilingDatafromBin->realCoreNum = 32;
  tilingDatafromBin->numPerCore = 1;
  tilingDatafromBin->numLastCore = 1;
  tilingDatafromBin->processSize = 8192;
  tilingDatafromBin->loopNum = 3;
  tilingDatafromBin->loopTail = 8192;
  tilingDatafromBin->innerLoopNum = 1;
  tilingDatafromBin->innerLoopTail = 4096;
  tilingDatafromBin->tilingKey = 102;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSilu = false;

  ReadFile(path + "/group_norm_silu_data/input_x.bin", outputByteSize, x, outputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_gamma.bin", inputByteSize, beta, inputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_beta.bin", inputByteSize, gamma, inputByteSize);
  ICPU_SET_TILING_KEY(102);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(group_norm_silu, blockDim, x, gamma, beta, silu, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(silu);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(group_norm_silu_test, test_case_104) {
  size_t inputByteSize = 64 * sizeof(float);
  size_t outputByteSize = 3 * 64 * 128 * 128 * sizeof(float);
  size_t meanByteSize = 3 * 32 * sizeof(float);
  size_t tiling_data_size = sizeof(GroupNormSiluTilingData);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* silu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 48;
  system("cp -r ../../../../norm/group_norm_silu/tests/ut/op_kernel/group_norm_silu_data ./");
  system("chmod -R 755 ./group_norm_silu_data/");
  system("cd ./group_norm_silu_data/ && rm -rf ./*bin");
  system("cd ./group_norm_silu_data/ && python3 gen_data.py 3 64 128 128 float32 float32");

  char* path_ = get_current_dir_name();
  string path(path_);

  GroupNormSiluTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSiluTilingData*>(tiling);

  tilingDatafromBin->numGroups = 32;
  tilingDatafromBin->hwNum = 128 * 128;
  tilingDatafromBin->shapeC = 64;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->elemNum = tilingDatafromBin->hwNum *  tilingDatafromBin->shapeD;
  tilingDatafromBin->realCoreNum = 48;
  tilingDatafromBin->numPerCore = 2;
  tilingDatafromBin->numLastCore = 2;
  tilingDatafromBin->processSize = 8192;
  tilingDatafromBin->loopNum = 4;
  tilingDatafromBin->loopTail = 8192;
  tilingDatafromBin->innerLoopNum = 2;
  tilingDatafromBin->innerLoopTail = 8192;
  tilingDatafromBin->tilingKey = 104;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSilu = true;

  ReadFile(path + "/group_norm_silu_data/input_x.bin", outputByteSize, x, outputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_gamma.bin", inputByteSize, beta, inputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_beta.bin", inputByteSize, gamma, inputByteSize);
  ICPU_SET_TILING_KEY(104);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(group_norm_silu, blockDim, x, gamma, beta, silu, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(silu);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(group_norm_silu_test, test_case_1061) {
  size_t inputByteSize = 3008 * sizeof(float);
  size_t outputByteSize = 1 * 3008 * 8 * 8 * sizeof(float);
  size_t meanByteSize = 1 * 752 * sizeof(float);
  size_t tiling_data_size = sizeof(GroupNormSiluTilingData);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* silu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 48;
  system("cp -r ../../../../norm/group_norm_silu/tests/ut/op_kernel/group_norm_silu_data ./");
  system("chmod -R 755 ./group_norm_silu_data/");
  system("cd ./group_norm_silu_data/ && rm -rf ./*bin");
  system("cd ./group_norm_silu_data/ && python3 gen_data.py 1 3008 8 8 float32 float32");

  char* path_ = get_current_dir_name();
  string path(path_);

  GroupNormSiluTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSiluTilingData*>(tiling);

  tilingDatafromBin->numGroups = 752;
  tilingDatafromBin->hwNum = 8 * 8;
  tilingDatafromBin->shapeC = 3008;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->elemNum = tilingDatafromBin->shapeD * tilingDatafromBin->hwNum;
  tilingDatafromBin->realCoreNum = 47;
  tilingDatafromBin->numPerCore = 16;
  tilingDatafromBin->numLastCore = 16;
  tilingDatafromBin->processSize = 8192;
  tilingDatafromBin->loopNum = 1;
  tilingDatafromBin->loopTail = 64;
  tilingDatafromBin->innerLoopNum = 1;
  tilingDatafromBin->innerLoopTail = 64;
  tilingDatafromBin->tilingKey = 106;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSilu = false;

  ReadFile(path + "/group_norm_silu_data/input_x.bin", outputByteSize, x, outputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_gamma.bin", inputByteSize, beta, inputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_beta.bin", inputByteSize, gamma, inputByteSize);
  ICPU_SET_TILING_KEY(106);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(group_norm_silu, blockDim, x, gamma, beta, silu, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(silu);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(group_norm_silu_test, test_case_1062) {
  size_t inputByteSize = 3008 * sizeof(float);
  size_t outputByteSize = 1 * 3008 * 8 * 8 * sizeof(float);
  size_t meanByteSize = 1 * 752 * sizeof(float);
  size_t tiling_data_size = sizeof(GroupNormSiluTilingData);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* silu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 48;
  system("cp -r ../../../../norm/group_norm_silu/tests/ut/op_kernel/group_norm_silu_data./");
  system("chmod -R 755 ./group_norm_silu_data/");
  system("cd ./group_norm_silu_data/ && rm -rf ./*bin");
  system("cd ./group_norm_silu_data/ && python3 gen_data.py 1 3008 8 8 float32 float32");

  char* path_ = get_current_dir_name();
  string path(path_);

  GroupNormSiluTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSiluTilingData*>(tiling);

  tilingDatafromBin->numGroups = 752;
  tilingDatafromBin->hwNum = 8 * 8;
  tilingDatafromBin->shapeC = 3008;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->elemNum = tilingDatafromBin->shapeD * tilingDatafromBin->hwNum;
  tilingDatafromBin->realCoreNum = 47;
  tilingDatafromBin->numPerCore = 16;
  tilingDatafromBin->numLastCore = 16;
  tilingDatafromBin->processSize = 8192;
  tilingDatafromBin->loopNum = 1;
  tilingDatafromBin->loopTail = 64;
  tilingDatafromBin->innerLoopNum = 1;
  tilingDatafromBin->innerLoopTail = 64;
  tilingDatafromBin->tilingKey = 106;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSilu = true;

  ReadFile(path + "/group_norm_silu_data/input_x.bin", outputByteSize, x, outputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_gamma.bin", inputByteSize, beta, inputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_beta.bin", inputByteSize, gamma, inputByteSize);
  ICPU_SET_TILING_KEY(106);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(group_norm_silu, blockDim, x, gamma, beta, silu, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(silu);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(group_norm_silu_test, test_case_1082) {
  size_t inputByteSize = 3008 * sizeof(float);
  size_t outputByteSize = 1 * 3008 * 1 * 1 * sizeof(float);
  size_t meanByteSize = 1 * 752 * sizeof(float);
  size_t tiling_data_size = sizeof(GroupNormSiluTilingData);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* silu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 48;
  system("cp -r ../../../../norm/group_norm_silu/tests/ut/op_kernel/group_norm_silu_data ./");
  system("chmod -R 755 ./group_norm_silu_data/");
  system("cd ./group_norm_silu_data/ && rm -rf ./*bin");
  system("cd ./group_norm_silu_data/ && python3 gen_data.py 1 3008 1 1 float32 float32");

  char* path_ = get_current_dir_name();
  string path(path_);

  GroupNormSiluTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSiluTilingData*>(tiling);

  tilingDatafromBin->numGroups = 752;
  tilingDatafromBin->hwNum = 1 * 1;
  tilingDatafromBin->shapeC = 3008;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->elemNum = tilingDatafromBin->shapeD * tilingDatafromBin->hwNum;
  tilingDatafromBin->realCoreNum = 47;
  tilingDatafromBin->numPerCore = 16;
  tilingDatafromBin->numLastCore = 16;
  tilingDatafromBin->processSize = 8192;
  tilingDatafromBin->loopNum = 1;
  tilingDatafromBin->loopTail = 1;
  tilingDatafromBin->innerLoopNum = 1;
  tilingDatafromBin->innerLoopTail = 1;
  tilingDatafromBin->tilingKey = 108;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSilu = true;

  ReadFile(path + "/group_norm_silu_data/input_x.bin", outputByteSize, x, outputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_gamma.bin", inputByteSize, beta, inputByteSize);
  ReadFile(path + "/group_norm_silu_data/input_beta.bin", inputByteSize, gamma, inputByteSize);
  ICPU_SET_TILING_KEY(108);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(group_norm_silu, blockDim, x, gamma, beta, silu, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(silu);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}