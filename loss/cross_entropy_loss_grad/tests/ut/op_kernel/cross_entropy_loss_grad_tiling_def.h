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
 * \file cross_entropy_loss_grad_tiling.h
 * \brief
 */
#ifndef _CROSS_ENTROPY_LOSS_GRAD_TILING_H_
#define _CROSS_ENTROPY_LOSS_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"
#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct CrossEntropyLossGradTilingData {
  uint64_t reduction = 1;
  uint64_t ignoreIndex = -100;
  float labelSmoothing = 0.0;
  uint64_t rowVal = 1;
  uint64_t colVal = 1;
  uint64_t frontCoreNum = 0;
  uint64_t tailCoreNum = 0;
  uint64_t usedCoreNum = 0;
  uint64_t frontRowNum = 0;
  uint64_t tailRowNum = 0;
  uint64_t alignColLoopNum = 0;
  uint64_t colLoop = 0;
  uint64_t colLoopNumTail = 0;
  uint64_t targetSize = 0;
  uint64_t targetCastSize = 0;
  uint64_t gradLossSize = 0;
  uint64_t gradLossFp32Size = 0;
  uint64_t ignoreSize = 0;
  uint64_t maskSize = 0;
  uint64_t targetWeightSize = 0;
  uint64_t tBuf1Size = 0;
  uint64_t tBuf2Size = 0;
  uint64_t tBuf3Size = 0;
};

inline void InitCrossEntropyLossGradTilingData(uint8_t* tiling, CrossEntropyLossGradTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(CrossEntropyLossGradTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
  CrossEntropyLossGradTilingData tilingData;                           \
  InitCrossEntropyLossGradTilingData(tilingPointer, &tilingData)
#endif