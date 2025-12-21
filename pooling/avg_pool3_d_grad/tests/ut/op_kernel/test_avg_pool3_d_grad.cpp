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
#include "avg_pool3_d_grad_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void avg_pool3_d_grad(GM_ADDR orig_input_shape, GM_ADDR grads, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class avg_pool3_d_grad_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "avg_pool3_d_grad_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "avg_pool3_d_grad_test TearDown\n" << endl;
  }
};

TEST_F(avg_pool3_d_grad_test, test_case_float32) {
  size_t origInputShapeByteSize = 5 * sizeof(int32_t);
  size_t gradsByteSize = 4 * 4 * 4 * 1024 * sizeof(float);
  size_t outputByteSize = 8 * 8 * 8 * 1024 * sizeof(float);
  size_t tiling_data_size = sizeof(AvgPool3dGradTilingParam);
  uint8_t* origInputShape = (uint8_t*)AscendC::GmAlloc(origInputShapeByteSize);
  uint8_t* grads = (uint8_t*)AscendC::GmAlloc(gradsByteSize);
  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 48;

  AvgPool3dGradTilingParam* tilingData = reinterpret_cast<AvgPool3dGradTilingParam*>(tiling);

  tilingData->attrParam.outN = 1;
  tilingData->attrParam.outD = 4;
  tilingData->attrParam.outH = 4;
  tilingData->attrParam.outW = 4;
  tilingData->attrParam.inN = 1;
  tilingData->attrParam.inD = 8;
  tilingData->attrParam.inH = 8;
  tilingData->attrParam.inW = 8;
  tilingData->attrParam.kD = 2;
  tilingData->attrParam.kH = 2;
  tilingData->attrParam.kW = 2;
  tilingData->attrParam.dD = 2;
  tilingData->attrParam.dH = 2;
  tilingData->attrParam.dW = 2;
  tilingData->attrParam.padD = 0;
  tilingData->attrParam.padH = 0;
  tilingData->attrParam.padW = 0;
  tilingData->attrParam.countIncludePad = 1;
  tilingData->attrParam.divisorOverride = 0;
  tilingData->attrParam.isOverLap = 0;
  tilingData->attrParam.isDetermine = 0;

  tilingData->cParam.normalCoreCNum = 1;
  tilingData->cParam.lastCoreCNum = 17;
  tilingData->cParam.cTotal = 1024;
  tilingData->cParam.cCount = 1024;
  tilingData->cParam.cNum = 1;
  tilingData->cParam.cLine = 21;
  tilingData->cParam.cTail = 1024;
  tilingData->cParam.cAlign = 1024;

  ICPU_SET_TILING_KEY(2000);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(avg_pool3_d_grad, blockDim, origInputShape, grads, output, workspace, tiling);

  AscendC::GmFree(grads);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}

TEST_F(avg_pool3_d_grad_test, test_case_float32_normal_ncdhw) {
  size_t origInputShapeByteSize = 5 * sizeof(int32_t);
  size_t gradsByteSize = 4 * 4 * 4 * 1 * sizeof(float);
  size_t outputByteSize = 8 * 8 * 8 * 1 * sizeof(float);
  size_t tiling_data_size = sizeof(AvgPool3dGradTilingParam);
  uint8_t* origInputShape = (uint8_t*)AscendC::GmAlloc(origInputShapeByteSize);
  uint8_t* grads = (uint8_t*)AscendC::GmAlloc(gradsByteSize);
  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 48;

  AvgPool3dGradTilingParam* tilingData = reinterpret_cast<AvgPool3dGradTilingParam*>(tiling);

  tilingData->attrParam.outD = 4;
  tilingData->attrParam.outH = 4;
  tilingData->attrParam.outW = 4;
  tilingData->attrParam.inD = 8;
  tilingData->attrParam.inH = 8;
  tilingData->attrParam.inW = 8;
  tilingData->attrParam.kD = 2;
  tilingData->attrParam.kH = 2;
  tilingData->attrParam.kW = 2;
  tilingData->attrParam.dD = 2;
  tilingData->attrParam.dH = 2;
  tilingData->attrParam.dW = 2;
  tilingData->attrParam.padD = 0;
  tilingData->attrParam.padH = 0;
  tilingData->attrParam.padW = 0;
  tilingData->attrParam.countIncludePad = 1;
  tilingData->attrParam.divisorOverride = 0;
  tilingData->attrParam.isOverLap = 0;
  tilingData->attrParam.isDetermine = 0;

  tilingData->blockParam.singleCoreNc = 1536;
  tilingData->blockParam.singleCoreDo = 1;
  tilingData->blockParam.singleCoreHo = 1;
  tilingData->blockParam.singleCoreWo = 2;
  tilingData->blockParam.ncCnt = 1;
  tilingData->blockParam.doCnt = 4;
  tilingData->blockParam.hoCnt = 4;
  tilingData->blockParam.woCnt = 2;
  tilingData->blockParam.totalCnt = 32;
  tilingData->blockParam.ncTailData = 1;
  tilingData->blockParam.doTailData = 1;
  tilingData->blockParam.hoTailData = 1;
  tilingData->blockParam.woTailData = 2;
  tilingData->blockParam.baseDo = 1;
  tilingData->blockParam.baseHo = 1;
  tilingData->blockParam.baseWo = 2;
  tilingData->blockParam.baseDi = 1;
  tilingData->blockParam.baseHi = 1;
  tilingData->blockParam.baseWi = 4;
  tilingData->blockParam.ubSize = 175872;
  tilingData->blockParam.isScatter = 0;

  ICPU_SET_TILING_KEY(5000);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(avg_pool3_d_grad, blockDim, origInputShape, grads, output, workspace, tiling);

  AscendC::GmFree(grads);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}

