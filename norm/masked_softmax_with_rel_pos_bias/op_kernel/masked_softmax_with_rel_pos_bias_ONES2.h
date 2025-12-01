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
 * \file masked_softmax_with_rel_pos_bias_ONES2.h
 * \brief
 */
#ifndef ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_ONES2_H
#define ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_ONES2_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MaskedSoftmaxWithRelPosBias {

using namespace AscendC;

template <typename T>
class MaskedSoftmaxWithRelPosBiasONES2 {
public:
  __aicore__ inline MaskedSoftmaxWithRelPosBiasONES2(
      const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ MaskedSoftmaxWithRelPosBiasONES2TilingData)
          : tilingData(MaskedSoftmaxWithRelPosBiasONES2TilingData){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR attenMask, GM_ADDR bias, GM_ADDR y) {
    if (GetBlockIdx() < tilingData->tailStartCoreIdx) {
      offset = GetBlockIdx() * tilingData->singleCoreSize;
      batchSize = tilingData->singleCoreSize;
    } else {
      offset = GetBlockIdx() * (tilingData->singleCoreSize - 1) + tilingData->tailStartCoreIdx;
      batchSize = tilingData->singleCoreSize - 1;
    }

    typeSize = sizeof(T);
    stackNum = tilingData->stackNum;  // 65535 is for DataCopyParams
    loopCount = batchSize / stackNum;
    loopTailSize = batchSize - batchSize / stackNum * stackNum;

    yGm.SetGlobalBuffer((__gm__ T*)y + offset, batchSize);
    pipe.InitBuffer(vecOutQueue, 1, (stackNum * typeSize / 32 + 1) * 32); // 32 is for aligned
  }

  __aicore__ inline void Process() {
    for (uint32_t i = 0; i < loopCount; i++) {
      Compute(i, stackNum);
      CopyOut(i, stackNum);
    }
    if (loopTailSize > 0) {
      Compute(loopCount, loopTailSize);
      CopyOut(loopCount, loopTailSize);
    }
  }

private:
  __aicore__ inline void Compute(int32_t i, uint32_t num) {
    LocalTensor<T> yLocal = vecOutQueue.AllocTensor<T>();

    T scalar = 1.0;
    Duplicate<T>(yLocal, scalar, num);

    vecOutQueue.EnQue<T>(yLocal);
  }

  __aicore__ inline void CopyOut(int32_t i, uint32_t num) {
    LocalTensor<T> yLocal = vecOutQueue.DeQue<T>();
    DataCopyParams copyParamsLast{(uint16_t)(1), (uint16_t)(num * typeSize), (uint16_t)(0), (uint16_t)(0)};
#if __CCE_AICORE__ == 220
    DataCopyPad(yGm[static_cast<int64_t>(i) * stackNum], yLocal, copyParamsLast);
#endif
    vecOutQueue.FreeTensor(yLocal);
  }

private:
  TPipe pipe;
  TQue<QuePosition::VECOUT, 1> vecOutQueue;
  uint32_t offset, batchSize;
  uint32_t stackNum, loopCount, loopTailSize, typeSize;
  GlobalTensor<T> yGm;
  const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ tilingData;
};

}
#endif
