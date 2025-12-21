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
 * \file masked_softmax_with_rel_pos_bias_BWNS1.h
 * \brief
 */
#ifndef ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_BWNS1_H
#define ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_BWNS1_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MaskedSoftmaxWithRelPosBias {
constexpr int32_t DOUBLE_NUM = 2;

using namespace AscendC;
template <typename T, bool existAtten, bool existMuls>
class MaskedSoftmaxWithRelPosBiasBWNS1 {
public:
  __aicore__ inline MaskedSoftmaxWithRelPosBiasBWNS1(
      const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ maskedSoftmaxWithRelPosBiasBWNS1TilingData)
          : tilingData(maskedSoftmaxWithRelPosBiasBWNS1TilingData){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR attenMask, GM_ADDR bias, GM_ADDR y) {
    const uint32_t blockId = GetBlockIdx();
    if (blockId < tilingData->tailStartCoreIdx) {
      offset = blockId * tilingData->singleCoreSize;
      batchSize = tilingData->singleCoreSize;
    } else {
      offset = blockId * (tilingData->singleCoreSize - 1) + tilingData->tailStartCoreIdx;
      batchSize = tilingData->singleCoreSize - 1;
    }

    if (batchSize == 0) {
      return;
    }

    stackNum = tilingData->stackNum;
    loopCount = batchSize / stackNum;
    loopTailSize = batchSize - loopCount * stackNum;

    w_ = tilingData->w;
    n_ = tilingData->n;
    s1_ = tilingData->s1;
    s2_ = tilingData->s2;

    // 一些常量
    s2DtypeSize = tilingData->s2DtypeSize;
    s2AlignedSize = tilingData->s2Aligned;
    s2AlignedDtypeSize = s2AlignedSize * sizeof(T);
    mulNAndS1 = n_ * s1_;
    mulLoopStepSizeAndS2AlignedSize = stackNum * s2AlignedSize;
    mulLoopStepSizeAndS2_ = stackNum * s2_;
    mulLoopTailSizeAnds2AlignedSize = loopTailSize * s2AlignedSize;

    scaleValue = tilingData->scaleValue;

    xGm.SetGlobalBuffer((__gm__ T*)x + offset * s2_, batchSize * s2_);
    yGm.SetGlobalBuffer((__gm__ T*)y + offset * s2_, batchSize * s2_);
    attenMaskGm.SetGlobalBuffer((__gm__ T*)attenMask);
    biasGm.SetGlobalBuffer((__gm__ T*)bias);

    auto bufferSize = stackNum * s2AlignedDtypeSize;
    pipe.InitBuffer(vecInQueue, BUFFER_NUM, bufferSize);
    pipe.InitBuffer(vecOutQueue, BUFFER_NUM, bufferSize);
    pipe.InitBuffer(maskQueue, BUFFER_NUM, DOUBLE_NUM * bufferSize);

    // fp16 和 bf16 需要提升精度计算，需要额外空间
    if constexpr (!AscendC::IsSameType<T, float>::value) {
      pipe.InitBuffer(CastBuff_, 4 * mulLoopStepSizeAndS2AlignedSize * sizeof(float)); // 4 is for cast
    }
  }

  __aicore__ inline void Process() {
    if(batchSize == 0) {
      return;
    }
    if (s2AlignedSize == s2_) {
      ProcessAlignedSize();
    } else {
#if (__CCE_AICORE__ > 200)
      ProcessNotAlignedSize();
#endif
    }
  }

private:
  __aicore__ inline void ProcessAlignedSize() {
    SoftMaxShapeInfo softmaxShapeInfo{stackNum, s2AlignedSize, stackNum, s2_};
    int64_t loopOffset = 0;
    for (uint32_t i = 0; i < loopCount; i++) {
      CopyIn(loopOffset, stackNum, mulLoopStepSizeAndS2AlignedSize);
      Compute(softmaxShapeInfo, tilingData->softmaxTilingData, mulLoopStepSizeAndS2AlignedSize);
      CopyOut(loopOffset, mulLoopStepSizeAndS2AlignedSize);
      loopOffset += stackNum;
    }

    if (loopTailSize > 0) {
      loopOffset = stackNum * loopCount;
      CopyIn(loopOffset, loopTailSize, mulLoopTailSizeAnds2AlignedSize);
      softmaxShapeInfo.srcM = loopTailSize;
      softmaxShapeInfo.oriSrcM = loopTailSize;
      Compute(softmaxShapeInfo, tilingData->tailSoftmaxTilingData, mulLoopTailSizeAnds2AlignedSize);
      CopyOut(loopOffset, mulLoopTailSizeAnds2AlignedSize);
    }
  }

#if (__CCE_AICORE__ > 200)
  __aicore__ inline void ProcessNotAlignedSize() {
    SoftMaxShapeInfo softmaxShapeInfo{stackNum, s2AlignedSize, stackNum, s2_};
    int64_t loopOffset = 0;
    for (uint32_t i = 0; i < loopCount; i++) {
      CopyInPad(loopOffset, stackNum);
      Compute(softmaxShapeInfo, tilingData->softmaxTilingData, mulLoopStepSizeAndS2AlignedSize);
      CopyOutPad(loopOffset, stackNum);
      loopOffset += stackNum;
    }

    if (loopTailSize > 0) {
      loopOffset = stackNum * loopCount;
      CopyInPad(loopOffset, loopTailSize);
      softmaxShapeInfo.srcM = loopTailSize;
      softmaxShapeInfo.oriSrcM = loopTailSize;
      LocalTensor<T> yLocal = vecOutQueue.AllocTensor<T>();
      Compute(softmaxShapeInfo, tilingData->tailSoftmaxTilingData, mulLoopTailSizeAnds2AlignedSize);
      CopyOutPad(loopOffset, loopTailSize);
    }
  }
#endif

  __aicore__ inline void CopyIn(int64_t loopOffset, uint32_t num, uint32_t mulNumAndS2) {
    LocalTensor<T> xLocal = vecInQueue.AllocTensor<T>();
    LocalTensor<T> maskLocal = maskQueue.AllocTensor<T>();
    DataCopy(xLocal, xGm[loopOffset * s2_], mulNumAndS2);

    int64_t localOffset = offset + loopOffset;
    uint32_t tmpS1 = s1_ - static_cast<uint32_t>(localOffset % s1_);
    if (num <= tmpS1) {
      int64_t attenMaskGmOffset = localOffset / (mulNAndS1) % w_ * s1_ + localOffset % s1_;
      DataCopy(maskLocal[0], attenMaskGm[attenMaskGmOffset * s2_], mulNumAndS2);
      int64_t biasGmOffset = localOffset % (mulNAndS1);
      DataCopy(maskLocal[mulLoopStepSizeAndS2_], biasGm[biasGmOffset * s2_], mulNumAndS2);
    } else {
      uint32_t tailNum = num - tmpS1;

      int64_t attenMaskGmOffset = localOffset / (mulNAndS1) % w_ * s1_ + localOffset % s1_;
      DataCopy(maskLocal[0], attenMaskGm[attenMaskGmOffset * s2_], tmpS1 * s2_);

      attenMaskGmOffset = (localOffset + tmpS1) / (mulNAndS1) % w_ * s1_ + (localOffset + tmpS1) % s1_;
      DataCopy(maskLocal[tmpS1 * s2_], attenMaskGm[attenMaskGmOffset * s2_], tailNum * s2_);

      int64_t biasGmOffset = localOffset % (mulNAndS1);
      DataCopy(maskLocal[mulLoopStepSizeAndS2_], biasGm[biasGmOffset * s2_], tmpS1 * s2_);

      biasGmOffset = (localOffset + tmpS1) % (mulNAndS1);
      DataCopy(maskLocal[mulLoopStepSizeAndS2_ + tmpS1 * s2_], biasGm[biasGmOffset * s2_],
               tailNum * s2_);
    }
    vecInQueue.EnQue(xLocal);
    maskQueue.EnQue(maskLocal);
  }

  __aicore__ inline void Compute(const SoftMaxShapeInfo& softmaxShapeInfo, const SoftMaxTiling& softMaxTilingData,
                                 uint32_t mulNumAndS2) {
    LocalTensor<T> xLocal = vecInQueue.DeQue<T>();
    LocalTensor<T> attenMaskLocal = maskQueue.DeQue<T>();
    LocalTensor<T> biasLocal = attenMaskLocal[mulLoopStepSizeAndS2AlignedSize];
    LocalTensor<T> yLocal = vecOutQueue.AllocTensor<T>();
    if constexpr (!AscendC::IsSameType<T, float>::value) {
      LocalTensor<float> xFloatLocal = CastBuff_.Get<float>();
      LocalTensor<float> attenMaskFloatLocal = xFloatLocal[mulLoopStepSizeAndS2AlignedSize];
      LocalTensor<float> biasMaskFloatLocal = attenMaskFloatLocal[mulLoopStepSizeAndS2AlignedSize];
      LocalTensor<float> yFloatLocal = biasMaskFloatLocal[mulLoopStepSizeAndS2AlignedSize];
      // Set xFloatLocal size equal to yFloatLocal, avoid AscendC::SoftMax index out of range
      xFloatLocal.SetSize(yFloatLocal.GetSize());

      // cast
      Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, mulNumAndS2);
      Cast(attenMaskFloatLocal, attenMaskLocal, RoundMode::CAST_NONE, mulNumAndS2);
      Cast(biasMaskFloatLocal, biasLocal, RoundMode::CAST_NONE, mulNumAndS2);
      PipeBarrier<PIPE_V>();

      if constexpr (existMuls) {
        Muls(yFloatLocal, xFloatLocal, scaleValue, mulNumAndS2);
        PipeBarrier<PIPE_V>();
        Add(xFloatLocal, yFloatLocal, attenMaskFloatLocal, mulNumAndS2);
        PipeBarrier<PIPE_V>();
        Add(yFloatLocal, xFloatLocal, biasMaskFloatLocal, mulNumAndS2);
        PipeBarrier<PIPE_V>();
        AscendC::SoftMax<float, true, false>(xFloatLocal, yFloatLocal, softMaxTilingData, softmaxShapeInfo);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<T, half>::value) {
          Cast(yLocal, xFloatLocal, RoundMode::CAST_NONE, mulNumAndS2);
        } else {
          Cast(yLocal, xFloatLocal, RoundMode::CAST_RINT, mulNumAndS2);
        }
        PipeBarrier<PIPE_V>();
      } else {
        Add(yFloatLocal, xFloatLocal, attenMaskFloatLocal, mulNumAndS2);
        PipeBarrier<PIPE_V>();
        Add(xFloatLocal, yFloatLocal, biasMaskFloatLocal, mulNumAndS2);
        PipeBarrier<PIPE_V>();
        AscendC::SoftMax<float, true, false>(yFloatLocal, xFloatLocal, softMaxTilingData, softmaxShapeInfo);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<T, half>::value) {
          Cast(yLocal, yFloatLocal, RoundMode::CAST_NONE, mulNumAndS2);
        } else {
          Cast(yLocal, yFloatLocal, RoundMode::CAST_RINT, mulNumAndS2);
        }
        PipeBarrier<PIPE_V>();
      }
    } else {
      if constexpr (existMuls) {
        Muls(xLocal, xLocal, scaleValue, mulNumAndS2);
        PipeBarrier<PIPE_V>();
        Add(yLocal, xLocal, attenMaskLocal, mulNumAndS2);
        PipeBarrier<PIPE_V>();
        Add(xLocal, yLocal, biasLocal, mulNumAndS2);
        PipeBarrier<PIPE_V>();
        AscendC::SoftMax<T, true, false>(yLocal, xLocal, softMaxTilingData, softmaxShapeInfo);
        PipeBarrier<PIPE_V>();
      } else {
        Add(yLocal, xLocal, attenMaskLocal, mulNumAndS2);
        PipeBarrier<PIPE_V>();
        Add(xLocal, yLocal, biasLocal, mulNumAndS2);
        PipeBarrier<PIPE_V>();
        AscendC::SoftMax<T, true, false>(yLocal, xLocal, softMaxTilingData, softmaxShapeInfo);
        PipeBarrier<PIPE_V>();
      }
    }
    vecInQueue.FreeTensor(xLocal);
    maskQueue.FreeTensor(attenMaskLocal);
    vecOutQueue.EnQue<T>(yLocal);
  }

  __aicore__ inline void CopyOut(int64_t loopOffset, uint32_t mulNumAndS2) {
    LocalTensor<T> yLocal = vecOutQueue.DeQue<T>();
    DataCopy(yGm[loopOffset * s2_], yLocal, mulNumAndS2);
    vecOutQueue.FreeTensor(yLocal);
  }

#if (__CCE_AICORE__ > 200)
  __aicore__ inline void CopyInPad(int64_t loopOffset, uint32_t num) {
    LocalTensor<T> xLocal = vecInQueue.AllocTensor<T>();
    LocalTensor<T> maskLocal = maskQueue.AllocTensor<T>();

    // 数据copy的时候，需要考虑softmax的最后一个轴需要32B对齐
    DataCopyParams copyParamsLast{(uint16_t)(num), (uint16_t)(s2DtypeSize), (uint16_t)(0), (uint16_t)(0)};
    DataCopyPadParams padParamsNormal{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[loopOffset * s2_], copyParamsLast, padParamsNormal);

    int64_t localOffset = offset + loopOffset;
    uint32_t tmpS1 = s1_ - static_cast<uint32_t>(localOffset % s1_);
    if (num <= tmpS1) {
      DataCopyParams copyParamsMask{(uint16_t)(num), (uint16_t)(s2DtypeSize), (uint16_t)(0), (uint16_t)(0)};

      int64_t attenMaskGmOffset = localOffset / (mulNAndS1) % w_ * s1_ + localOffset % s1_;
      DataCopyPad(maskLocal[0], attenMaskGm[attenMaskGmOffset * s2_], copyParamsMask, padParamsNormal);
      int64_t biasGmOffset = localOffset % (mulNAndS1);
      DataCopyPad(maskLocal[mulLoopStepSizeAndS2AlignedSize], biasGm[biasGmOffset * s2_],
                  copyParamsMask, padParamsNormal);
    } else {
      uint32_t tailNum = num - tmpS1;
      DataCopyParams copyParamsMask{(uint16_t)(tmpS1), (uint16_t)(s2DtypeSize), (uint16_t)(0), (uint16_t)(0)};
      DataCopyParams tailCopyParamsMask{(uint16_t)(tailNum), (uint16_t)(s2DtypeSize), (uint16_t)(0), (uint16_t)(0)};

      int64_t attenMaskGmOffset = localOffset / (mulNAndS1) % w_ * s1_ + localOffset % s1_;
      DataCopyPad(maskLocal[0], attenMaskGm[attenMaskGmOffset * s2_], copyParamsMask, padParamsNormal);

      attenMaskGmOffset = (localOffset + tmpS1) / (mulNAndS1) % w_ * s1_ + (localOffset + tmpS1) % s1_;
      DataCopyPad(maskLocal[tmpS1 * s2AlignedSize], attenMaskGm[attenMaskGmOffset * s2_],
                  tailCopyParamsMask, padParamsNormal);

      int64_t biasGmOffset = localOffset % (mulNAndS1);
      DataCopyPad(maskLocal[mulLoopStepSizeAndS2AlignedSize], biasGm[biasGmOffset * s2_], copyParamsMask, padParamsNormal);

      biasGmOffset = (localOffset + tmpS1)% (mulNAndS1);
      DataCopyPad(maskLocal[mulLoopStepSizeAndS2AlignedSize + tmpS1 * s2AlignedSize], biasGm[biasGmOffset * s2_],
               tailCopyParamsMask, padParamsNormal);
    }

    vecInQueue.EnQue(xLocal);
    maskQueue.EnQue(maskLocal);
  }

  __aicore__ inline void CopyOutPad(int64_t loopOffset, uint32_t num) {
    LocalTensor<T> yLocal = vecOutQueue.DeQue<T>();
    DataCopyParams copyParamsLast{(uint16_t)(num), (uint16_t)(s2DtypeSize), (uint16_t)(0), (uint16_t)(0)};
    DataCopyPad(yGm[loopOffset * s2_], yLocal, copyParamsLast);
    vecOutQueue.FreeTensor(yLocal);
  }
#endif

private:
  TPipe pipe;
  TQue<QuePosition::VECIN, 1> vecInQueue;
  TQue<QuePosition::VECIN, 1> maskQueue;
  TQue<QuePosition::VECOUT, 1> vecOutQueue;
  TBuf<QuePosition::VECCALC> CastBuff_;
  uint32_t offset, batchSize, w_, n_, s1_, s2_, s2DtypeSize;
  uint32_t s2AlignedSize, s2AlignedDtypeSize;
  uint32_t stackNum, loopCount, loopTailSize;
  GlobalTensor<T> xGm, yGm, attenMaskGm, biasGm;
  // 一些常量
  uint32_t subS2AndS2Aligned, mulNAndS1, mulLoopStepSizeAndS2AlignedSize, mulLoopStepSizeAndS2_,
           mulLoopTailSizeAnds2AlignedSize;
  float scaleValue;
  const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ tilingData;
};

}
#endif
