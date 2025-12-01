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
 * \file masked_softmax_with_rel_pos_bias_S1_bias.h
 * \brief
 */
#ifndef ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_B_S1_BIAS_H
#define ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_B_S1_BIAS_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MaskedSoftmaxWithRelPosBias {

using namespace AscendC;
template <typename T, bool existMuls, bool isAligend>
class MaskedSoftmaxWithRelPosBiasBS1Bias {
 public:
  __aicore__ inline MaskedSoftmaxWithRelPosBiasBS1Bias(
      const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ maskedSoftmaxWithRelPosBiasBTilingData)
      : tilingData(maskedSoftmaxWithRelPosBiasBTilingData){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y) {
    const uint32_t blockId = GetBlockIdx();
    if (blockId < tilingData->tailStartCoreIdx) {
      s1StartOffset = blockId * tilingData->singleCoreSize;
      batchSize = tilingData->singleCoreSize;
    } else {
      s1StartOffset = blockId * (tilingData->singleCoreSize - 1) + tilingData->tailStartCoreIdx;
      batchSize = tilingData->singleCoreSize - 1;
    }
    if (batchSize == 0) {
      return;
    }
    s2_ = tilingData->s2;
    ns1 = tilingData->n * tilingData->s1;
    s2Aligned = tilingData->s2Aligned;
    s2DtypeSize = tilingData->s2DtypeSize;
    stackNum = tilingData->stackNum;
    uint32_t xOffset = s1StartOffset * s2_;
    // 因为每个核上处理的batchSize不一样，所以不能放到host侧计算
    uint64_t bufferSize = batchSize * s2_;
    xGm.SetGlobalBuffer((__gm__ T*)x + xOffset, bufferSize);
    yGm.SetGlobalBuffer((__gm__ T*)y + xOffset, bufferSize);
    biasGm.SetGlobalBuffer((__gm__ T*)bias, tilingData->ns1s2);

    // 内存需要32B对齐
    pipe.InitBuffer(vecInQueue, BUFFER_NUM, tilingData->inQueueLen);
    pipe.InitBuffer(vecOutQueue, BUFFER_NUM, tilingData->inQueueLen);
    pipe.InitBuffer(maskQueue, BUFFER_NUM, tilingData->maskQueueLen);
    pipe.InitBuffer(softmaxBuffer, tilingData->tmpXubSize);
    if constexpr (!(AscendC::IsSameType<T, float>::value)) {
      pipe.InitBuffer(castBuffer, tilingData->castTempSize);
    }
  }
  __aicore__ inline void Process() {
#if (__CCE_AICORE__ == 220)
    PRELOAD(2);  // 2 is for 16kb
#endif
    if (batchSize == 0) {
      return;
    }
    uint32_t elementNum = tilingData->xCopyEleNum;
    SoftMaxShapeInfo softmaxShapeInfo{stackNum, s2Aligned, stackNum, s2_};
    LocalTensor<uint8_t> softmaxSharedTmpBuffer = softmaxBuffer.Get<uint8_t>();
    int64_t offset = 0;
    uint32_t loopNum = tilingData->loopNum;
    for (uint32_t loopCount = 0; loopCount < loopNum; loopCount++) {
      CopyIn(offset, stackNum, elementNum);
      Compute(softmaxSharedTmpBuffer, softmaxShapeInfo, tilingData->softmaxTilingData, elementNum);
      CopyOut(offset, stackNum, elementNum);
      offset += stackNum;
    }

    uint32_t tailNum = batchSize - stackNum * loopNum;
    // tailNum 的softmax Tiling 是不一样的
    if (tailNum != 0) {
      if (stackNum != 0) {
        offset = stackNum * loopNum;
        elementNum = tailNum * s2Aligned;
        CopyIn(offset, tailNum, elementNum);
        softmaxShapeInfo.srcM = tailNum;
        softmaxShapeInfo.oriSrcM = tailNum;
        Compute(softmaxSharedTmpBuffer, softmaxShapeInfo, tilingData->tailSoftmaxTilingData, elementNum);
        CopyOut(offset, tailNum, elementNum);
      } else {
        elementNum = s2Aligned;
        softmaxShapeInfo.srcM = 1;
        softmaxShapeInfo.oriSrcM = 1;
        for (uint32_t loopCount = 0; loopCount < tailNum; loopCount++) {
          CopyIn(offset, 1, elementNum);
          Compute(softmaxSharedTmpBuffer, softmaxShapeInfo, tilingData->tailSoftmaxTilingData, elementNum);
          CopyOut(offset, 1, elementNum);
          offset++;
        }
      }
    }
  }

 protected:
  __aicore__ inline void CopyIn(int64_t offset, uint32_t stackNum, uint32_t copyEleNum) {
    LocalTensor<T> xLocal = vecInQueue.AllocTensor<T>();
    LocalTensor<T> biasLocal = maskQueue.AllocTensor<T>();
    int64_t ns1Offset = (s1StartOffset + offset) % ns1;
    int64_t tmpNS1 = ns1 - ns1Offset;
    if constexpr (isAligend) {
      DataCopy(xLocal, xGm[offset * s2_], copyEleNum);
      if (tmpNS1 >= stackNum) {
        DataCopy(biasLocal, biasGm[ns1Offset * s2_], copyEleNum);
      } else {
        DataCopy(biasLocal, biasGm[ns1Offset * s2_], tmpNS1 * s2_);
        DataCopy(biasLocal[tmpNS1 * s2_], biasGm, (stackNum - tmpNS1) * s2_);
      }
    } else {
#if (__CCE_AICORE__ > 200)
      DataCopyParams copyParamsLast{static_cast<uint16_t>(stackNum), static_cast<uint16_t>(s2DtypeSize), (uint16_t)(0),
                                    (uint16_t)(0)};
      DataCopyPadParams padParamsNormal{false, 0, 0, 0};
      DataCopyPad(xLocal, xGm[offset * s2_], copyParamsLast, padParamsNormal);
      if (tmpNS1 >= stackNum) {
        DataCopyPad(biasLocal, biasGm[ns1Offset * s2_], copyParamsLast, padParamsNormal);
      } else {
        DataCopyParams bais1copyParamsLast{static_cast<uint16_t>(tmpNS1), static_cast<uint16_t>(s2DtypeSize),
                                           (uint16_t)(0), (uint16_t)(0)};
        DataCopyPad(biasLocal, biasGm[ns1Offset * s2_], bais1copyParamsLast, padParamsNormal);
        DataCopyParams bais2copyParamsLast{static_cast<uint16_t>(stackNum - tmpNS1), static_cast<uint16_t>(s2DtypeSize),
                                           (uint16_t)(0), (uint16_t)(0)};
        DataCopyPad(biasLocal[tmpNS1 * s2Aligned], biasGm, bais2copyParamsLast, padParamsNormal);
      }
#endif
    }
    maskQueue.EnQue(biasLocal);
    vecInQueue.EnQue(xLocal);
  }

  __aicore__ inline void Compute(const LocalTensor<uint8_t>& softmaxSharedTmpBuffer,
                                 const SoftMaxShapeInfo& softmaxShapeInfo, const SoftMaxTiling& softmaxTilingData,
                                 uint32_t elementNum) {
    LocalTensor<T> xLocal = vecInQueue.DeQue<T>();
    LocalTensor<T> biasLocal = maskQueue.DeQue<T>();
    LocalTensor<T> yLocal = vecOutQueue.AllocTensor<T>();
    if constexpr (!(AscendC::IsSameType<T, float>::value)) {
      LocalTensor<float> xFloatLocal = castBuffer.Get<float>();
      LocalTensor<float> biasFloatLocal = xFloatLocal[elementNum];
      LocalTensor<float> yFloatLocal = biasFloatLocal[elementNum];
      // Set xFloatLocal size equal to yFloatLocal, avoid AscendC::SoftMax index out of range
      xFloatLocal.SetSize(yFloatLocal.GetSize());

      // cast
      Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, elementNum);
      Cast(biasFloatLocal, biasLocal, RoundMode::CAST_NONE, elementNum);
      PipeBarrier<PIPE_V>();
      if constexpr (existMuls) {
        Muls(yFloatLocal, xFloatLocal, tilingData->scaleValue, elementNum);
        PipeBarrier<PIPE_V>();
        Add(xFloatLocal, yFloatLocal, biasFloatLocal, elementNum);
        PipeBarrier<PIPE_V>();
        AscendC::SoftMax<float, true, false>(yFloatLocal, xFloatLocal, softmaxSharedTmpBuffer, softmaxTilingData,
                                              softmaxShapeInfo);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<T, half>::value) {
          Cast(yLocal, yFloatLocal, RoundMode::CAST_NONE, elementNum);
        } else {
          Cast(yLocal, yFloatLocal, RoundMode::CAST_RINT, elementNum);
        }
        PipeBarrier<PIPE_V>();
      } else {
        Add(yFloatLocal, xFloatLocal, biasFloatLocal, elementNum);
        PipeBarrier<PIPE_V>();
        AscendC::SoftMax<float, true, false>(xFloatLocal, yFloatLocal, softmaxSharedTmpBuffer, softmaxTilingData,
                                              softmaxShapeInfo);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<T, half>::value) {
          Cast(yLocal, xFloatLocal, RoundMode::CAST_NONE, elementNum);
        } else {
          Cast(yLocal, xFloatLocal, RoundMode::CAST_RINT, elementNum);
        }
        PipeBarrier<PIPE_V>();
      }
    } else {
      if constexpr (existMuls) {
        Muls(yLocal, xLocal, tilingData->scaleValue, elementNum);
        PipeBarrier<PIPE_V>();
        Add(xLocal, yLocal, biasLocal, elementNum);
        PipeBarrier<PIPE_V>();
        AscendC::SoftMax<T, true, false>(yLocal, xLocal, softmaxSharedTmpBuffer, softmaxTilingData, softmaxShapeInfo);
        PipeBarrier<PIPE_V>();
      } else {
        Add(xLocal, xLocal, biasLocal, elementNum);
        PipeBarrier<PIPE_V>();
        AscendC::SoftMax<T, true, false>(yLocal, xLocal, softmaxSharedTmpBuffer, softmaxTilingData, softmaxShapeInfo);
        PipeBarrier<PIPE_V>();
      }
    }
    vecInQueue.FreeTensor(xLocal);
    maskQueue.FreeTensor(biasLocal);
    vecOutQueue.EnQue<T>(yLocal);
  }

  __aicore__ inline void CopyOut(int64_t offset, uint32_t stackNum, uint32_t copyEleNum) {
    LocalTensor<T> yLocal = vecOutQueue.DeQue<T>();
    if constexpr (isAligend) {
      DataCopy(yGm[offset * s2_], yLocal, copyEleNum);
    } else {
#if (__CCE_AICORE__ > 200)
      DataCopyParams copyParamsLast{static_cast<uint16_t>(stackNum), static_cast<uint16_t>(s2DtypeSize), (uint16_t)(0),
                                    (uint16_t)(0)};
      DataCopyPad(yGm[offset * s2_], yLocal, copyParamsLast);
#endif
    }
    vecOutQueue.FreeTensor(yLocal);
  }

 protected:
  TPipe pipe;
  TQue<QuePosition::VECIN, 1> vecInQueue;
  TQue<QuePosition::VECIN, 1> maskQueue;
  TQue<QuePosition::VECOUT, 1> vecOutQueue;
  TBuf<QuePosition::VECCALC> castBuffer;
  TBuf<QuePosition::VECCALC> softmaxBuffer;
  uint32_t s2_{0};
  uint32_t ns1{0};
  uint32_t s2Aligned{0};
  uint32_t s2DtypeSize{0};
  uint32_t stackNum{0};
  uint32_t s1StartOffset{0};
  uint32_t batchSize{0};
  GlobalTensor<T> xGm, yGm, biasGm;
  const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ tilingData;
};
}  // namespace MaskedSoftmaxWithRelPosBias
#endif