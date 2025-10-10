/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_cross_entropy_loss_grad.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "cross_entropy_loss_grad_tiling_def.h"
#include "data_utils.h"


#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void cross_entropy_loss_grad(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
                                                              GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
                                                              GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling);
class cross_entropy_loss_grad_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "cross_entropy_loss_grad_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "cross_entropy_loss_grad_test TearDown\n" << endl;
  }
};

TEST_F(cross_entropy_loss_grad_test, test_case_weight_not_none_fp16) {
  size_t gradLossByteSize = 1 * sizeof(half);
  size_t logProbByteSize = 30 * 1024 * sizeof(half);
  size_t targetByteSize = 30 * sizeof(int64_t);
  size_t weightByteSize = 1024 * sizeof(float);
  // output
  size_t xGradByteSize = 30 * 1024 * sizeof(half);
  size_t tilingDataSize = sizeof(CrossEntropyLossGradTilingData);

  uint8_t* grad_loss = (uint8_t*)AscendC::GmAlloc(gradLossByteSize + 32);
  uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbByteSize + 32);
  uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetByteSize + 32);
  uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize + 32);
  uint8_t* grad_zloss = (uint8_t*)AscendC::GmAlloc(0);
  uint8_t* lse_for_zloss = (uint8_t*)AscendC::GmAlloc(0);
  uint8_t* x_grad = (uint8_t*)AscendC::GmAlloc(xGradByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 224);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 30;

  CrossEntropyLossGradTilingData* tilingDatafromBin = reinterpret_cast<CrossEntropyLossGradTilingData*>(tiling);

  tilingDatafromBin->reduction = 1;
  tilingDatafromBin->ignoreIndex = -100;
  tilingDatafromBin->labelSmoothing = 0;
  tilingDatafromBin->rowVal = 30;
  tilingDatafromBin->colVal = 1024;
  tilingDatafromBin->frontCoreNum = 30;
  tilingDatafromBin->tailCoreNum = 0;
  tilingDatafromBin->usedCoreNum = 30;
  tilingDatafromBin->frontRowNum = 1;
  tilingDatafromBin->tailRowNum = 0;
  tilingDatafromBin->alignColLoopNum = 1024;
  tilingDatafromBin->colLoop = 1;
  tilingDatafromBin->colLoopNumTail = 0;
  tilingDatafromBin->targetSize = 32;
  tilingDatafromBin->targetCastSize = 32;
  tilingDatafromBin->gradLossSize = 0;
  tilingDatafromBin->gradLossFp32Size = 0;
  tilingDatafromBin->ignoreSize = 32;
  tilingDatafromBin->maskSize = 32;
  tilingDatafromBin->targetWeightSize = 32;
  tilingDatafromBin->tBuf2Size = 32;
  tilingDatafromBin->tBuf3Size = 192;

  ICPU_SET_TILING_KEY(21);
  ICPU_RUN_KF(cross_entropy_loss_grad, blockDim, grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss,
              x_grad, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(grad_loss);
  AscendC::GmFree(log_prob);
  AscendC::GmFree(target);
  AscendC::GmFree(weight);
  AscendC::GmFree(x_grad);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}

TEST_F(cross_entropy_loss_grad_test, test_case_weight_none_bf16) {
  size_t gradLossByteSize = 1 * sizeof(bfloat16_t);
  size_t logProbByteSize = 30 * 1024 * sizeof(bfloat16_t);
  size_t targetByteSize = 30 * sizeof(int64_t);
  size_t weightByteSize = 1024 * sizeof(float);
  // output
  size_t xGradByteSize = 30 * 1024 * sizeof(bfloat16_t);
  size_t tilingDataSize = sizeof(CrossEntropyLossGradTilingData);

  uint8_t* grad_loss = (uint8_t*)AscendC::GmAlloc(gradLossByteSize + 32);
  uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbByteSize + 32);
  uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetByteSize + 32);
  uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize + 32);
  uint8_t* grad_zloss = (uint8_t*)AscendC::GmAlloc(0);
  uint8_t* lse_for_zloss = (uint8_t*)AscendC::GmAlloc(0);
  uint8_t* x_grad = (uint8_t*)AscendC::GmAlloc(xGradByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 30;

  CrossEntropyLossGradTilingData* tilingDatafromBin = reinterpret_cast<CrossEntropyLossGradTilingData*>(tiling);

  tilingDatafromBin->reduction = 1;
  tilingDatafromBin->ignoreIndex = -100;
  tilingDatafromBin->labelSmoothing = 0;
  tilingDatafromBin->rowVal = 30;
  tilingDatafromBin->colVal = 1024;
  tilingDatafromBin->frontCoreNum = 30;
  tilingDatafromBin->tailCoreNum = 0;
  tilingDatafromBin->usedCoreNum = 30;
  tilingDatafromBin->frontRowNum = 1;
  tilingDatafromBin->tailRowNum = 0;
  tilingDatafromBin->alignColLoopNum = 1024;
  tilingDatafromBin->colLoop = 1;
  tilingDatafromBin->colLoopNumTail = 0;
  tilingDatafromBin->targetSize = 32;
  tilingDatafromBin->targetCastSize = 32;
  tilingDatafromBin->gradLossSize = 0;
  tilingDatafromBin->gradLossFp32Size = 0;
  tilingDatafromBin->ignoreSize = 32;
  tilingDatafromBin->maskSize = 32;
  tilingDatafromBin->targetWeightSize = 0;
  tilingDatafromBin->tBuf2Size = 32;
  tilingDatafromBin->tBuf3Size = 192;

  ICPU_SET_TILING_KEY(10);
  ICPU_RUN_KF(cross_entropy_loss_grad, blockDim, grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss,
              x_grad, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(grad_loss);
  AscendC::GmFree(log_prob);
  AscendC::GmFree(target);
  AscendC::GmFree(weight);
  AscendC::GmFree(x_grad);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}

TEST_F(cross_entropy_loss_grad_test, test_case_weight_none_fp32) {
  size_t gradLossByteSize = 1 * sizeof(float);
  size_t logProbByteSize = 30 * 1024 * sizeof(float);
  size_t targetByteSize = 30 * sizeof(int64_t);
  size_t weightByteSize = 1024 * sizeof(float);
  // output
  size_t xGradByteSize = 30 * 1024 * sizeof(float);
  size_t tilingDataSize = sizeof(CrossEntropyLossGradTilingData);

  uint8_t* grad_loss = (uint8_t*)AscendC::GmAlloc(gradLossByteSize + 32);
  uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbByteSize + 32);
  uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetByteSize + 32);
  uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize + 32);
  uint8_t* grad_zloss = (uint8_t*)AscendC::GmAlloc(0);
  uint8_t* lse_for_zloss = (uint8_t*)AscendC::GmAlloc(0);
  uint8_t* x_grad = (uint8_t*)AscendC::GmAlloc(xGradByteSize + 32);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
  uint32_t blockDim = 30;

  CrossEntropyLossGradTilingData* tilingDatafromBin = reinterpret_cast<CrossEntropyLossGradTilingData*>(tiling);

  tilingDatafromBin->reduction = 1;
  tilingDatafromBin->ignoreIndex = -100;
  tilingDatafromBin->labelSmoothing = 0;
  tilingDatafromBin->rowVal = 30;
  tilingDatafromBin->colVal = 1024;
  tilingDatafromBin->frontCoreNum = 30;
  tilingDatafromBin->tailCoreNum = 0;
  tilingDatafromBin->usedCoreNum = 30;
  tilingDatafromBin->frontRowNum = 1;
  tilingDatafromBin->tailRowNum = 0;
  tilingDatafromBin->alignColLoopNum = 1024;
  tilingDatafromBin->colLoop = 1;
  tilingDatafromBin->colLoopNumTail = 0;
  tilingDatafromBin->targetSize = 32;
  tilingDatafromBin->targetCastSize = 32;
  tilingDatafromBin->gradLossSize = 0;
  tilingDatafromBin->gradLossFp32Size = 0;
  tilingDatafromBin->ignoreSize = 32;
  tilingDatafromBin->maskSize = 32;
  tilingDatafromBin->targetWeightSize = 0;
  tilingDatafromBin->tBuf2Size = 32;
  tilingDatafromBin->tBuf3Size = 192;

  ICPU_SET_TILING_KEY(30);
  ICPU_RUN_KF(cross_entropy_loss_grad, blockDim, grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss,
              x_grad, workspace, (uint8_t*)(tilingDatafromBin));

  AscendC::GmFree(grad_loss);
  AscendC::GmFree(log_prob);
  AscendC::GmFree(target);
  AscendC::GmFree(weight);
  AscendC::GmFree(x_grad);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}
