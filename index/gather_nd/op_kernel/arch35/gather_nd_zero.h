/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file gather_nd_zero.h
* \brief
*/
#ifndef ASCENDC_GATHER_ND_GATHER_ND_ZERO_H_
#define ASCENDC_GATHER_ND_GATHER_ND_ZERO_H_

namespace GatherNd {

constexpr uint32_t MAX_INPUT_X_SHAPE = 8;
constexpr uint32_t MAX_INPUT_INDICES_SHAPE = 2;

template <typename T1>
class GatherNdZero {
 public:
  __aicore__ inline GatherNdZero(){};

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace,
                              const GatherNdTilingData* __restrict ordTilingData, TPipe* pipeIn);
  __aicore__ inline void CopyIn(LocalTensor<T1>& dstTensor, uint32_t offset, uint32_t gatherLength);
  __aicore__ inline void CopyUb(LocalTensor<T1>& dstTensor, LocalTensor<T1>& srcTensor, uint32_t gatherLength);
  __aicore__ inline void CopyOut(LocalTensor<T1>& srcTensor, uint32_t offset, uint32_t gatherLength);

  __aicore__ inline void Process();

 protected:
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> vecInQue;
  TQue<QuePosition::VECOUT, 1> vecOutQue;

  uint32_t coreNum;
  uint32_t cBlockIdx;
  const GatherNdTilingData* __restrict tilingData;

  // input
  GlobalTensor<T1> xGm;

  // output
  GlobalTensor<T1> yGm;

  // shape info
  uint64_t xShape[8];
  uint64_t indicesShape[2];

  // buffer info
  uint32_t xUbSize;
  uint32_t indicesUbSize;

  // split info
  uint64_t gatherSize;
  uint64_t indicesNum;
  uint64_t outputSize;
  uint64_t blockFactor;
  uint64_t ubFactor;
  uint32_t rank;
  uint32_t blockNum;
  uint64_t curSize;
  uint64_t curBegin;
  uint64_t indicesBegin;

  bool isBroadCast = false;
  constexpr static uint32_t T1_BLOCK_NUM = 32 / sizeof(T1);
};

template <typename T1>
__aicore__ inline void GatherNdZero<T1>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace,
                                              const GatherNdTilingData* __restrict ordTilingData, TPipe* pipeIn) {
  cBlockIdx = GetBlockIdx();
  tilingData = ordTilingData;

  // init pipe
  pipe = pipeIn;

  // init gm
  xGm.SetGlobalBuffer((__gm__ T1*)x);
  yGm.SetGlobalBuffer((__gm__ T1*)y);

  // tiling info
  for (uint32_t i = 0; i < MAX_INPUT_X_SHAPE; i++) {
    xShape[i] = tilingData->xShape[i];
  }

  for (uint32_t i = 0; i < MAX_INPUT_INDICES_SHAPE; i++) {
    indicesShape[i] = tilingData->indicesShape[i];
  }

  isBroadCast = (indicesShape[1] == 0);

  xUbSize = tilingData->xUbSize;
  gatherSize = tilingData->gatherSize;
  indicesNum = tilingData->indicesNum;
  outputSize = tilingData->outputSize;
  blockFactor = tilingData->blockFactor;
  ubFactor = tilingData->ubFactor;
  blockNum = tilingData->blockNum;

  pipe->InitBuffer(vecInQue, 1, xUbSize);
  pipe->InitBuffer(vecOutQue, 1, xUbSize);
}

template <typename T1>
__aicore__ inline void GatherNdZero<T1>::CopyIn(LocalTensor<T1>& dstTensor, uint32_t offset, uint32_t gatherLength) {
    DataCopyPad(dstTensor, xGm[offset], {static_cast<uint16_t>(1), static_cast<uint32_t>(gatherLength), 0, 0, 0},
                {false, 0, 0, 0});
}

template <typename T1>
__aicore__ inline void GatherNdZero<T1>::CopyOut(LocalTensor<T1>& srcTensor, uint32_t offset, uint32_t gatherLength) {
    DataCopyPad(yGm[offset], srcTensor, {static_cast<uint16_t>(1), static_cast<uint32_t>(gatherLength), 0, 0, 0});
}

template <typename T1>
__aicore__ inline void GatherNdZero<T1>::CopyUb(LocalTensor<T1>& dstTensor, LocalTensor<T1>& srcTensor,
                                                uint32_t gatherLength) {
  uint32_t gatherLengthAlign = (gatherLength + T1_BLOCK_NUM - 1) / T1_BLOCK_NUM * T1_BLOCK_NUM;
  DataCopy(dstTensor, srcTensor, gatherLengthAlign);
}

template <typename T1>
__aicore__ inline void GatherNdZero<T1>::Process() {
  if (isBroadCast) {
    uint64_t beginRowIdx = cBlockIdx * blockFactor;
    uint64_t endRowdx = (cBlockIdx + 1) * blockFactor < indicesNum ? (cBlockIdx + 1) * blockFactor : indicesNum;
    uint64_t ubLoop = (gatherSize + ubFactor - 1) / ubFactor;
    uint64_t ubTail = gatherSize % ubFactor;

    for (uint64_t b = beginRowIdx; b < endRowdx; b++) {
      for (uint64_t u = 0; u < ubLoop; u++) {
        uint64_t dmaSize = (u < (ubLoop - 1)) ? ubFactor : ubTail;
        LocalTensor<T1> xBuffer = vecInQue.AllocTensor<T1>();
        LocalTensor<T1> outputBuffer = vecOutQue.AllocTensor<T1>();

        uint64_t xBegin = u * ubFactor;
        CopyIn(xBuffer, u * ubFactor, dmaSize * sizeof(T1));
        vecInQue.EnQue(xBuffer);
        vecInQue.DeQue<T1>();
        CopyUb(outputBuffer, xBuffer, dmaSize);

        vecInQue.FreeTensor(xBuffer);
        vecOutQue.EnQue(outputBuffer);
        vecOutQue.DeQue<T1>();
        uint64_t gatherBegin = b * gatherSize + u * ubFactor;
        CopyOut(outputBuffer, gatherBegin, dmaSize * sizeof(T1));
        vecOutQue.FreeTensor(outputBuffer);
      }
    }
  }
}
}

#endif  // ASCENDC_GATHER_ND_GATHER_ND_ZERO_H_
 