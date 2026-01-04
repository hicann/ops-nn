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
 * \file masked_softmax_with_rel_pos_bias_B.h
 * \brief
 */
#ifndef ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_B_H
#define ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_B_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MaskedSoftmaxWithRelPosBias {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template <typename T, bool existAtten, bool existMuls, bool isAligend>
class MaskedSoftmaxWithRelPosBiasB {
 public:
  __aicore__ inline MaskedSoftmaxWithRelPosBiasB(
      const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ maskedSoftmaxWithRelPosBiasBTilingData)
      : tilingData(maskedSoftmaxWithRelPosBiasBTilingData){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR attenMask, GM_ADDR bias, GM_ADDR y) {
    int64_t offset = 0;
    if (GetBlockIdx() < tilingData->tailStartCoreIdx) {
      offset = GetBlockIdx() * tilingData->singleCoreSize;
      batchSize = tilingData->singleCoreSize;
    } else {
      offset = GetBlockIdx() * (tilingData->singleCoreSize - 1) + tilingData->tailStartCoreIdx;
      batchSize = tilingData->singleCoreSize - 1;
    }
    if (batchSize == 0) {
      return;
    }
    w_ = tilingData->w;
    n_ = tilingData->n;
    s2_ = tilingData->s2;
    wns1s2 = tilingData->wns1s2;
    wns1 = tilingData->wns1;
    ws1s2 = tilingData->ws1s2;
    ns1s2 = tilingData->ns1s2;
    s1s2 = tilingData->s1s2;
    wns1s2Aligned = tilingData->wns1s2Aligned;
    s1s2Aligned = tilingData->s1s2Aligned;
    ns1s2Aligned = tilingData->ns1s2Aligned;
    s2Aligned = tilingData->s2Aligned;
    s2DtypeSize = tilingData->s2DtypeSize;
    offset *= wns1s2;
    // 因为每个核上处理的batchSize不一样，所以不能放到host侧计算
    uint64_t bufferSize = batchSize * wns1s2;
    xGm.SetGlobalBuffer((__gm__ T*)x + offset, bufferSize);
    yGm.SetGlobalBuffer((__gm__ T*)y + offset, bufferSize);
    if constexpr (existAtten) {
      attenMaskGm.SetGlobalBuffer((__gm__ T*)attenMask, ws1s2);
    }
    biasGm.SetGlobalBuffer((__gm__ T*)bias, ns1s2);

    // 内存需要32B对齐
    pipe.InitBuffer(vecInQueue, BUFFER_NUM, tilingData->inQueueLen);
    pipe.InitBuffer(vecOutQueue, BUFFER_NUM, tilingData->inQueueLen);
    pipe.InitBuffer(maskQueue, 1, tilingData->maskQueueLen);
  }
  __aicore__ inline void Process() {
    if (batchSize == 0) {
      return;
    }
    if constexpr (existAtten) {
      CopyInAttenMaskAndBias();
    } else {
      CopyInBias();
    }

    LocalTensor<T> attenMaskLocal = maskQueue.DeQue<T>();

    if constexpr (existAtten) {
      BroadCastAttenMaskAndBias(attenMaskLocal);
    } else {
      BroadCastBias(attenMaskLocal);
    }

    int64_t offset = 0;
    uint32_t stackNum = tilingData->stackNum;
    uint32_t loopNum = 0;
    if (stackNum != 0) {
      loopNum = batchSize / stackNum;
    }
    // 存在loopNum，stackNum为零的情况
    uint32_t mulsNum = stackNum * wns1s2Aligned;
    SoftMaxShapeInfo softmaxShapeInfo{stackNum * wns1, s2Aligned, stackNum * wns1, s2_};
    for (int64_t loopCount = 0; loopCount < loopNum; loopCount++) {
      offset = loopCount * stackNum;
      CopyInX(offset, stackNum, tilingData->xCopyEleNum);
      if constexpr (existAtten) {
        ComputeAttenMaskAndBias(offset, attenMaskLocal, stackNum, mulsNum, softmaxShapeInfo,
                                tilingData->softmaxTilingData);
      } else {
        ComputeBias(offset, attenMaskLocal, stackNum, mulsNum, softmaxShapeInfo, tilingData->softmaxTilingData);
      }
      CopyOut(offset, stackNum, tilingData->xCopyEleNum);
    }
    offset += stackNum;
    uint32_t tailNum = batchSize - stackNum * loopNum;
    // tailNum 的softmax Tiling 是不一样的
    if (tailNum != 0) {
      mulsNum = tailNum * wns1s2Aligned;
      CopyInX(offset, tailNum, mulsNum);
      softmaxShapeInfo.srcM = tailNum * wns1;
      softmaxShapeInfo.oriSrcM = tailNum * wns1;
      if constexpr (existAtten) {
        ComputeAttenMaskAndBias(offset, attenMaskLocal, tailNum, mulsNum, softmaxShapeInfo,
                                tilingData->tailSoftmaxTilingData);
      } else {
        ComputeBias(offset, attenMaskLocal, tailNum, mulsNum, softmaxShapeInfo, tilingData->tailSoftmaxTilingData);
      }
      CopyOut(offset, tailNum, mulsNum);
    }
    maskQueue.FreeTensor(attenMaskLocal);
  }

 protected:
  // attention 和 bias只需要copy in一次
  __aicore__ inline void CopyInAttenMaskAndBias() {
    if constexpr (isAligend) {
      LocalTensor<T> maskLocal = maskQueue.AllocTensor<T>();
      auto tempAttenLocal = maskLocal[tilingData->attenBiasNum];
      DataCopy(tempAttenLocal, attenMaskGm, ws1s2);
      auto biasLocal = maskLocal[wns1s2];
      DataCopy(biasLocal, biasGm, ns1s2);
      maskQueue.EnQue(maskLocal);
    } else {
#if (__CCE_AICORE__ > 200)
      LocalTensor<T> maskLocal = maskQueue.AllocTensor<T>();
      // 从[w s1 s2] -> [w s1 s2AlignedSize]
      DataCopyParams attenMaskCopyParamsLast{(uint16_t)(tilingData->ws1), (uint16_t)(s2DtypeSize), (uint16_t)(0), (uint16_t)(0)};
      auto tempLocal = maskLocal[tilingData->attenBiasNum];
      DataCopyPadParams padParamsNormal{false, 0, 0, 0};
      DataCopyPad(tempLocal, attenMaskGm, attenMaskCopyParamsLast, padParamsNormal);

      auto biasLocal = maskLocal[wns1s2Aligned];
      DataCopyParams biasCopyParamsLast{(uint16_t)(tilingData->ns1), (uint16_t)(s2DtypeSize), (uint16_t)(0), (uint16_t)(0)};
      DataCopyPad(biasLocal, biasGm, biasCopyParamsLast, padParamsNormal);
      maskQueue.EnQue(maskLocal);
#endif
    }
  }

  __aicore__ inline void CopyInBias() {
    if constexpr (isAligend) {
      LocalTensor<T> biasLocal = maskQueue.AllocTensor<T>();
      DataCopy(biasLocal, biasGm, ns1s2);
      maskQueue.EnQue(biasLocal);
    } else {
#if (__CCE_AICORE__ > 200)
      LocalTensor<T> biasLocal = maskQueue.AllocTensor<T>();
      DataCopyParams biasCopyParamsLast{(uint16_t)(tilingData->ns1), (uint16_t)(s2DtypeSize), (uint16_t)(0), (uint16_t)(0)};
      DataCopyPadParams padParamsNormal{false, 0, 0, 0};
      DataCopyPad(biasLocal, biasGm, biasCopyParamsLast, padParamsNormal);
      maskQueue.EnQue(biasLocal);
#endif
    }
  }

  __aicore__ inline void BroadCastAttenMaskAndBias(LocalTensor<T>& attenMaskLocal) {
    auto tempAttenLocal = attenMaskLocal[tilingData->attenBiasNum];
    // 后续支持counter模式后，可以只要一个for循环
    //[w, s1, s2] -> [w, n, s1, s2]
    uint32_t attenOffset = 0;
    uint32_t tempOffset = 0;
    for (uint32_t wi = 0; wi < w_; wi++) {
      for (uint32_t ni = 0; ni < n_; ni++) {
        Adds(attenMaskLocal[attenOffset], tempAttenLocal[tempOffset], (T)0, s1s2Aligned);
        attenOffset += s1s2Aligned;
      }
      tempOffset += s1s2Aligned;
    }
    LocalTensor<T> biasLocal = attenMaskLocal[wns1s2Aligned];
    uint32_t biasOffset = ns1s2Aligned;
    for (uint32_t wi = 1; wi < w_; wi++) {
      Adds(biasLocal[biasOffset], biasLocal, (T)0, ns1s2Aligned);
      biasOffset += ns1s2Aligned;
    }
  }

  __aicore__ inline void BroadCastBias(LocalTensor<T>& biasLocal) {
    uint32_t biasOffset = ns1s2Aligned;
    for (uint32_t wi = 1; wi < w_; wi++) {
      Adds(biasLocal[biasOffset], biasLocal, (T)0, ns1s2Aligned);
      biasOffset += ns1s2Aligned;
    }
  }

  __aicore__ inline void CopyInX(int64_t offset, uint32_t stackNum, uint32_t copyEleNum) {
    if constexpr (isAligend) {
      LocalTensor<T> xLocal = vecInQueue.AllocTensor<T>();
      DataCopy(xLocal, xGm[offset * wns1s2Aligned], copyEleNum);
      vecInQueue.EnQue(xLocal);
    } else {
#if (__CCE_AICORE__ > 200)
      LocalTensor<T> xLocal = vecInQueue.AllocTensor<T>();
      DataCopyParams copyParamsLast{(uint16_t)(stackNum * tilingData->wns1), (uint16_t)(s2DtypeSize), (uint16_t)(0), (uint16_t)(0)};
      DataCopyPadParams padParamsNormal{false, 0, 0, 0};
      DataCopyPad(xLocal, xGm[offset * wns1s2], copyParamsLast, padParamsNormal);
      vecInQueue.EnQue(xLocal);
#endif
    }
  }

  __aicore__ inline void ComputeAttenMaskAndBias(int64_t offset, LocalTensor<T>& attenMaskLocal, uint32_t stackNum,
                                                 uint32_t mulsNum, const SoftMaxShapeInfo& softmaxShapeInfo,
                                                 const SoftMaxTiling& softmaxTilingData) {
    LocalTensor<T> xLocal = vecInQueue.DeQue<T>();
    LocalTensor<T> yLocal = vecOutQueue.AllocTensor<T>();

    uint32_t xOffset = 0;
    if constexpr (existMuls) {
      Muls(yLocal, xLocal, static_cast<T>(tilingData->scaleValue), mulsNum);
      PipeBarrier<PIPE_V>();
      for (uint32_t stackIdx = 0; stackIdx < stackNum; stackIdx++) {
        Add(yLocal[xOffset], yLocal[xOffset], attenMaskLocal, wns1s2Aligned);
        xOffset += wns1s2Aligned;
      }
    } else {
      for (uint32_t stackIdx = 0; stackIdx < stackNum; stackIdx++) {
        Add(yLocal[xOffset], xLocal[xOffset], attenMaskLocal, wns1s2Aligned);
        xOffset += wns1s2Aligned;
      }
    }
    PipeBarrier<PIPE_V>();
    xOffset = 0;
    LocalTensor<T> biasLocal = attenMaskLocal[wns1s2Aligned];
    for (uint32_t stackIdx = 0; stackIdx < stackNum; stackIdx++) {
      Add(yLocal[xOffset], yLocal[xOffset], biasLocal, wns1s2Aligned);
      xOffset += wns1s2Aligned;
    }
    PipeBarrier<PIPE_V>();
    AscendC::SoftMax<T, false, false>(yLocal, yLocal, softmaxTilingData, softmaxShapeInfo);
    vecOutQueue.EnQue<T>(yLocal);
    vecInQueue.FreeTensor(xLocal);
  }

  __aicore__ inline void ComputeBias(int64_t offset, LocalTensor<T>& biasLocal, uint32_t stackNum, uint32_t mulsNum,
                                     const SoftMaxShapeInfo& softmaxShapeInfo, const SoftMaxTiling& softmaxTilingData) {
    LocalTensor<T> xLocal = vecInQueue.DeQue<T>();
    LocalTensor<T> yLocal = vecOutQueue.AllocTensor<T>();
    uint32_t xOffset = 0;
    if constexpr (existMuls) {
      Muls(yLocal, xLocal, static_cast<T>(tilingData->scaleValue), mulsNum);
      PipeBarrier<PIPE_V>();
      for (uint32_t stackIdx = 0; stackIdx < stackNum; stackIdx++) {
        Add(yLocal[xOffset], yLocal[xOffset], biasLocal, wns1s2Aligned);
        xOffset += wns1s2Aligned;
      }
    } else {
      for (uint32_t stackIdx = 0; stackIdx < stackNum; stackIdx++) {
        Add(yLocal[xOffset], xLocal[xOffset], biasLocal, wns1s2Aligned);
        xOffset += wns1s2Aligned;
      }
    }
    PipeBarrier<PIPE_V>();
    AscendC::SoftMax<T, false, false>(yLocal, yLocal, softmaxTilingData, softmaxShapeInfo);
    vecOutQueue.EnQue<T>(yLocal);
    vecInQueue.FreeTensor(xLocal);
  }

  __aicore__ inline void CopyOut(int64_t offset, uint32_t stackNum, uint32_t copyEleNum) {
    if constexpr (isAligend) {
      LocalTensor<T> yLocal = vecOutQueue.DeQue<T>();
      DataCopy(yGm[offset * wns1s2], yLocal, copyEleNum);
      vecOutQueue.FreeTensor(yLocal);
    } else {
#if (__CCE_AICORE__ > 200)
      LocalTensor<T> yLocal = vecOutQueue.DeQue<T>();
      DataCopyParams copyParamsLast{(uint16_t)(stackNum * wns1), (uint16_t)(s2DtypeSize), (uint16_t)(0), (uint16_t)(0)};
      DataCopyPad(yGm[offset * wns1s2], yLocal, copyParamsLast);
      vecOutQueue.FreeTensor(yLocal);
#endif
    }
  }

 protected:
  TPipe pipe;
  TQue<QuePosition::VECIN, 1> vecInQueue;
  TQue<QuePosition::VECIN, 1> maskQueue;
  TQue<QuePosition::VECOUT, 1> vecOutQueue;
  uint32_t w_, n_, s2_;
  uint32_t wns1s2;
  uint32_t wns1;
  uint32_t ws1s2;
  uint32_t ns1s2;
  uint32_t s1s2;
  uint32_t wns1s2Aligned;
  uint32_t s1s2Aligned;
  uint32_t ns1s2Aligned;
  uint32_t s2Aligned;
  uint32_t s2DtypeSize;
  uint32_t batchSize{0};
  GlobalTensor<T> xGm, yGm, attenMaskGm, biasGm;
  const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ tilingData;
};

template <typename T, bool existAtten, bool existMuls, bool isAligend>
class MaskedSoftmaxWithRelPosBiasBBf16AndHalf
    : public MaskedSoftmaxWithRelPosBiasB<T, existAtten, existMuls, isAligend> {
 public:
  __aicore__ inline MaskedSoftmaxWithRelPosBiasBBf16AndHalf(
      const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ maskedSoftmaxWithRelPosBiasBTilingData)
      : MaskedSoftmaxWithRelPosBiasB<T, existAtten, existMuls, isAligend>(
            maskedSoftmaxWithRelPosBiasBTilingData){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR attenMask, GM_ADDR bias, GM_ADDR y) {
    MaskedSoftmaxWithRelPosBiasB<T, existAtten, existMuls, isAligend>::Init(x, attenMask, bias, y);
    if (this->batchSize == 0) {
      return;
    }
    this->pipe.InitBuffer(castBuffer, this->tilingData->castTempSize);
    ws1s2Alingned = this->w_ * this->s1s2Aligned;
  }
  __aicore__ inline void Process() {
    if (this->batchSize == 0) {
      return;
    }
    if constexpr (existAtten) {
      CopyInAttenMaskAndBias();
    } else {
      this->CopyInBias();
    }

    LocalTensor<T> attenMaskLocal = this->maskQueue.template DeQue<T>();
    LocalTensor<float> castfloatBuffer = castBuffer.Get<float>();
    LocalTensor<float> attenMaskCastTensor = castfloatBuffer;
    // 这个偏移用tiling
    uint32_t biasOffset = 0;
    if constexpr (existAtten) {
      biasOffset = this->wns1s2Aligned;
    }
    LocalTensor<float> biasCastTensor = castfloatBuffer[biasOffset];
    if constexpr (existAtten) {
      BroadCastAttenMaskAndBias(attenMaskLocal, attenMaskCastTensor, biasCastTensor);
    } else {
      BroadCastBias(attenMaskLocal, biasCastTensor);
    }

    int64_t offset = 0;
    uint32_t stackNum = this->tilingData->stackNum;
    uint32_t loopNum = 0;
    if (stackNum != 0) {
      loopNum = this->batchSize / stackNum;
    }

    // 存在loopNum，stackNum为零的情况
    uint32_t mulsNum = stackNum * this->wns1s2Aligned;
    SoftMaxShapeInfo softmaxShapeInfo{stackNum * this->wns1, this->s2Aligned, stackNum * this->wns1, this->s2_};
    for (int64_t loopCount = 0; loopCount < loopNum; loopCount++) {
      offset = loopCount * stackNum;
      this->CopyInX(offset, stackNum, this->tilingData->xCopyEleNum);
      if constexpr (existAtten) {
        LocalTensor<float> xCastTensor = castfloatBuffer[this->tilingData->attenBiasNum];
        ComputeAttenMaskAndBias(offset, xCastTensor, attenMaskCastTensor, biasCastTensor, stackNum, mulsNum,
                                softmaxShapeInfo, this->tilingData->softmaxTilingData);
      } else {
        LocalTensor<float> xCastTensor = castfloatBuffer[this->wns1s2Aligned];
        ComputeBias(offset, xCastTensor, attenMaskCastTensor, stackNum, mulsNum, softmaxShapeInfo,
                    this->tilingData->softmaxTilingData);
      }

      this->CopyOut(offset, stackNum, this->tilingData->xCopyEleNum);
    }
    uint32_t tailNum = this->batchSize - stackNum * loopNum;
    offset += stackNum;
    // tailNum 的softmax Tiling 是不一样的
    if (tailNum != 0) {
      mulsNum = tailNum * this->wns1s2Aligned;
      this->CopyInX(offset, tailNum, mulsNum);
      softmaxShapeInfo.srcM = tailNum * this->wns1;
      softmaxShapeInfo.oriSrcM = tailNum * this->wns1;
      if constexpr (existAtten) {
        LocalTensor<float> xCastTensor = castfloatBuffer[this->tilingData->attenBiasNum];
        ComputeAttenMaskAndBias(offset, xCastTensor, attenMaskCastTensor, biasCastTensor, tailNum, mulsNum,
                                softmaxShapeInfo, this->tilingData->softmaxTilingData);
      } else {
        LocalTensor<float> xCastTensor = castfloatBuffer[this->wns1s2Aligned];
        ComputeBias(offset, xCastTensor, attenMaskCastTensor, tailNum, mulsNum, softmaxShapeInfo,
                    this->tilingData->softmaxTilingData);
      }
      this->CopyOut(offset, tailNum, mulsNum);
    }
    this->maskQueue.FreeTensor(attenMaskLocal);
  }

 private:
  __aicore__ inline void CopyInAttenMaskAndBias() {
    if constexpr (isAligend) {
      LocalTensor<T> maskLocal = this->maskQueue.template AllocTensor<T>();
      DataCopy(maskLocal, this->attenMaskGm, this->ws1s2);
      auto biasLocal = maskLocal[this->ws1s2];
      DataCopy(biasLocal, this->biasGm, this->ns1s2);
      this->maskQueue.EnQue(maskLocal);
    } else {
      LocalTensor<T> maskLocal = this->maskQueue.template AllocTensor<T>();
      DataCopyParams attenMaskCopyParamsLast(this->tilingData->ws1, this->s2DtypeSize, (uint16_t)(0), (uint16_t)(0));
      DataCopyPadParams padParamsNormal{false, 0, 0, 0};
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
      DataCopyPad(maskLocal, this->attenMaskGm, attenMaskCopyParamsLast, padParamsNormal);
#endif
      auto biasLocal = maskLocal[ws1s2Alingned];
      DataCopyParams biasCopyParamsLast(this->tilingData->ns1, this->s2DtypeSize, (uint16_t)(0), (uint16_t)(0));
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
      DataCopyPad(biasLocal, this->biasGm, biasCopyParamsLast, padParamsNormal);
#endif
      this->maskQueue.EnQue(maskLocal);
    }
  }

  __aicore__ inline void BroadCastAttenMaskAndBias(LocalTensor<T>& attenMaskLocal,
                                                   LocalTensor<float>& attenMaskCastTensor,
                                                   LocalTensor<float>& biasCastTensor) {
    //[w, s1, s2] -> [w, n, s1, s2]
    uint32_t attenOffset = 0;
    uint32_t tempOffset = 0;
    for (uint32_t wi = 0; wi < this->w_; wi++) {
      for (uint32_t ni = 0; ni < this->n_; ni++) {
        Cast(attenMaskCastTensor[attenOffset], attenMaskLocal[tempOffset], RoundMode::CAST_NONE, this->s1s2Aligned);
        attenOffset += this->s1s2Aligned;
      }
      tempOffset += this->s1s2Aligned;
    }
    //[n, s1, s2] -> [w, n, s1, s2]
    LocalTensor<T> biasLocal = attenMaskLocal[this->ws1s2Alingned];
    uint32_t biasOffset = 0;
    for (uint32_t wi = 0; wi < this->w_; wi++) {
      Cast(biasCastTensor[biasOffset], biasLocal, RoundMode::CAST_NONE, this->ns1s2Aligned);
      biasOffset += this->ns1s2Aligned;
    }
  }

  __aicore__ inline void BroadCastBias(LocalTensor<T>& biasLocal, LocalTensor<float>& biasCastTensor) {
    //[n, s1, s2] -> [w, n, s1, s2]
    uint32_t biasOffset = 0;
    for (uint32_t wi = 0; wi < this->w_; wi++) {
      Cast(biasCastTensor[biasOffset], biasLocal, RoundMode::CAST_NONE, this->ns1s2Aligned);
      biasOffset += this->ns1s2Aligned;
    }
  }

  __aicore__ inline void ComputeAttenMaskAndBias(int64_t offset, LocalTensor<float>& xCastTensor,
                                                 LocalTensor<float>& attenMaskLocal, LocalTensor<float>& biasCastTensor,
                                                 uint32_t stackNum, uint32_t mulsNum,
                                                 const SoftMaxShapeInfo& softmaxShapeInfo,
                                                 const SoftMaxTiling& softmaxTilingData) {
    LocalTensor<T> xLocal = this->vecInQueue.template DeQue<T>();
    LocalTensor<T> yLocal = this->vecOutQueue.template AllocTensor<T>();
    Cast(xCastTensor, xLocal, RoundMode::CAST_NONE, mulsNum);
    PipeBarrier<PIPE_V>();
    uint32_t xOffset = 0;
    if constexpr (existMuls) {
      Muls(xCastTensor, xCastTensor, this->tilingData->scaleValue, mulsNum);
      PipeBarrier<PIPE_V>();
      for (uint32_t stackIdx = 0; stackIdx < stackNum; stackIdx++) {
        Add(xCastTensor[xOffset], xCastTensor[xOffset], attenMaskLocal, this->wns1s2Aligned);
        xOffset += this->wns1s2Aligned;
      }
    } else {
      for (uint32_t stackIdx = 0; stackIdx < stackNum; stackIdx++) {
        Add(xCastTensor[xOffset], xCastTensor[xOffset], attenMaskLocal, this->wns1s2Aligned);
        xOffset += this->wns1s2Aligned;
      }
    }
    PipeBarrier<PIPE_V>();
    xOffset = 0;
    for (uint32_t stackIdx = 0; stackIdx < stackNum; stackIdx++) {
      Add(xCastTensor[xOffset], xCastTensor[xOffset], biasCastTensor, this->wns1s2Aligned);
      xOffset += this->wns1s2Aligned;
    }
    PipeBarrier<PIPE_V>();
    AscendC::SoftMax<float, false, false>(xCastTensor, xCastTensor, softmaxTilingData, softmaxShapeInfo);
    PipeBarrier<PIPE_V>();
    // CAST_RINT 应该存在精度损失
    if constexpr (AscendC::IsSameType<T, half>::value) {
      Cast(yLocal, xCastTensor, RoundMode::CAST_NONE, mulsNum);
    } else {
      Cast(yLocal, xCastTensor, RoundMode::CAST_RINT, mulsNum);
    }
    this->vecOutQueue.template EnQue<T>(yLocal);
    this->vecInQueue.FreeTensor(xLocal);
  }

  __aicore__ inline void ComputeBias(int64_t offset, LocalTensor<float>& xCastTensor, LocalTensor<float>& biasLocal,
                                     uint32_t stackNum, uint32_t mulsNum, const SoftMaxShapeInfo& softmaxShapeInfo,
                                     const SoftMaxTiling& softmaxTilingData) {
    LocalTensor<T> xLocal = this->vecInQueue.template DeQue<T>();
    LocalTensor<T> yLocal = this->vecOutQueue.template AllocTensor<T>();
    Cast(xCastTensor, xLocal, RoundMode::CAST_NONE, mulsNum);
    PipeBarrier<PIPE_V>();
    uint32_t xOffset = 0;
    if constexpr (existMuls) {
      Muls(xCastTensor, xCastTensor, this->tilingData->scaleValue, mulsNum);
      PipeBarrier<PIPE_V>();
      for (uint32_t stackIdx = 0; stackIdx < stackNum; stackIdx++) {
        Add(xCastTensor[xOffset], xCastTensor[xOffset], biasLocal, this->wns1s2Aligned);
        xOffset += this->wns1s2Aligned;
      }
    } else {
      for (uint32_t stackIdx = 0; stackIdx < stackNum; stackIdx++) {
        Add(xCastTensor[xOffset], xCastTensor[xOffset], biasLocal, this->wns1s2Aligned);
        xOffset += this->wns1s2Aligned;
      }
    }
    PipeBarrier<PIPE_V>();
    AscendC::SoftMax<float, false, false>(xCastTensor, xCastTensor, softmaxTilingData, softmaxShapeInfo);
    PipeBarrier<PIPE_V>();
    if constexpr (AscendC::IsSameType<T, half>::value) {
      Cast(yLocal, xCastTensor, RoundMode::CAST_NONE, mulsNum);
    } else {
      Cast(yLocal, xCastTensor, RoundMode::CAST_RINT, mulsNum);
    }
    this->vecOutQueue.template EnQue<T>(yLocal);
    this->vecInQueue.FreeTensor(xLocal);
  }

 private:
  TBuf<QuePosition::VECCALC> castBuffer;
  uint32_t ws1s2Alingned{0};
};
}  // namespace MaskedSoftmaxWithRelPosBias
#endif