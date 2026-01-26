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
#include "group_norm_silu_quant_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void group_norm_silu_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR silu,
                                                      GM_ADDR mean, GM_ADDR rstd,
                                                      GM_ADDR workspace, GM_ADDR tiling);

class group_norm_silu_quant_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "group_norm_silu_quant_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "group_norm_silu_quant_test TearDown\n" << endl;
  }
};

TEST_F(group_norm_silu_quant_test, test_case_dtype16_perchannel) {
  size_t inputByteSize = 192 * sizeof(int16_t);
  size_t quantScaleByteSize = 192 * sizeof(int32_t);
  size_t outputByteSize = 1 * 192 * 64 * 64 * sizeof(int16_t);
  size_t meanByteSize = 1 * 192 * sizeof(int16_t);
  size_t tiling_data_size = sizeof(GroupNormSiluQuantTilingData);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* quantScale = (uint8_t*)AscendC::GmAlloc(quantScaleByteSize);
  uint8_t* silu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 32;

  char* path_ = get_current_dir_name();
  string path(path_);

  GroupNormSiluQuantTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSiluQuantTilingData*>(tiling);

  tilingDatafromBin->numGroups = 32;
  tilingDatafromBin->hwNum = 64 * 64;
  tilingDatafromBin->shapeC = 192;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->shapeQuantScale = 192;
  tilingDatafromBin->elemNum = 64 * 64 * 6;
  tilingDatafromBin->realCoreNum = 32;
  tilingDatafromBin->numPerCore = 1;
  tilingDatafromBin->numLastCore = 1;
  tilingDatafromBin->processSize = 8192;
  tilingDatafromBin->loopNum = 3;
  tilingDatafromBin->loopTail = 8192;
  tilingDatafromBin->innerLoopNum = 1;
  tilingDatafromBin->innerLoopTail = 4096;
  tilingDatafromBin->tilingKey = 1031;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSilu = true;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_SET_TILING_KEY(1031);
  ICPU_RUN_KF(group_norm_silu_quant, blockDim, x, gamma, beta, quantScale, silu, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(quantScale);
  AscendC::GmFree(silu);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(group_norm_silu_quant_test, test_case_dtype16_pertensor) {
  size_t inputByteSize = 192 * sizeof(int16_t);
  size_t quantScaleByteSize = sizeof(int32_t);
  size_t outputByteSize = 1 * 192 * 64 * 64 * sizeof(int16_t);
  size_t meanByteSize = 1 * 192 * sizeof(int16_t);
  size_t tiling_data_size = sizeof(GroupNormSiluQuantTilingData);
  uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputByteSize);
  uint8_t* quantScale = (uint8_t*)AscendC::GmAlloc(quantScaleByteSize);
  uint8_t* silu = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(meanByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 32;

  char* path_ = get_current_dir_name();
  string path(path_);

  GroupNormSiluQuantTilingData* tilingDatafromBin = reinterpret_cast<GroupNormSiluQuantTilingData*>(tiling);

  tilingDatafromBin->numGroups = 32;
  tilingDatafromBin->hwNum = 64 * 64;
  tilingDatafromBin->shapeC = 192;
  tilingDatafromBin->shapeD = tilingDatafromBin->shapeC / tilingDatafromBin->numGroups;
  tilingDatafromBin->shapeQuantScale = 1;
  tilingDatafromBin->elemNum = 64 * 64 * 6;
  tilingDatafromBin->realCoreNum = 32;
  tilingDatafromBin->numPerCore = 1;
  tilingDatafromBin->numLastCore = 1;
  tilingDatafromBin->processSize = 8192;
  tilingDatafromBin->loopNum = 3;
  tilingDatafromBin->loopTail = 8192;
  tilingDatafromBin->innerLoopNum = 1;
  tilingDatafromBin->innerLoopTail = 4096;
  tilingDatafromBin->tilingKey = 1031;
  tilingDatafromBin->epsilon = 0.00001;
  tilingDatafromBin->activateSilu = true;

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_SET_TILING_KEY(1031);
  ICPU_RUN_KF(group_norm_silu_quant, blockDim, x, gamma, beta, quantScale, silu, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(x);
  AscendC::GmFree(gamma);
  AscendC::GmFree(beta);
  AscendC::GmFree(quantScale);
  AscendC::GmFree(silu);
  AscendC::GmFree(mean);
  AscendC::GmFree(rstd);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}