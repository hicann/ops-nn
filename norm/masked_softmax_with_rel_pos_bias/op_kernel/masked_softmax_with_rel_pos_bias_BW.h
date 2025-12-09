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
 * \file masked_softmax_with_rel_pos_bias_BW.h
 * \brief
 */
#ifndef ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_BW_H
#define ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_BW_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MaskedSoftmaxWithRelPosBias {

using namespace AscendC;
template <typename T, bool existAtten, bool existMuls>
class MaskedSoftmaxWithRelPosBiasBW {
 public:
  __aicore__ inline MaskedSoftmaxWithRelPosBiasBW(
      const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ maskedSoftmaxWithRelPosBiasBWTilingData)
      : tilingData(maskedSoftmaxWithRelPosBiasBWTilingData){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR attenMask, GM_ADDR bias, GM_ADDR y) {
    int64_t offset = 0;
    if (GetBlockIdx() < tilingData->tailStartCoreIdx) {
      offset = GetBlockIdx() * tilingData->singleCoreSize;
      batchSize = tilingData->singleCoreSize;
    } else {
      offset = GetBlockIdx() * (tilingData->singleCoreSize - 1) + tilingData->tailStartCoreIdx;
      batchSize = tilingData->singleCoreSize - 1;
    }
    w_ = tilingData->w;
    n_ = tilingData->n;
    s1_ = tilingData->s1;
    s2_ = tilingData->s2;
    softMaxTiling = tilingData->softmaxTilingData;
    wns1 = tilingData->wns1;
    ws1s2 = tilingData->ws1s2;
    ns1s2 = tilingData->ns1s2;
    ns1 = tilingData->ns1;
    s1s2 = tilingData->s1s2;
    s1s2AlignedSize = tilingData->s1s2Aligned;
    totalSizeAlignedSize = tilingData->ns1s2Aligned * sizeof(T);
    s2DtypeSize = tilingData->s2DtypeSize;
    s2AlignedSize = tilingData->s2Aligned;
    scaleValue = tilingData->scaleValue;
    atten_mask_offset = offset % w_;
    offset *= ns1s2;
    isAligend = (s2_ == s2AlignedSize);
    xGm.SetGlobalBuffer((__gm__ T*)x + offset, batchSize * ns1s2);
    yGm.SetGlobalBuffer((__gm__ T*)y + offset, batchSize * ns1s2);
    attenMaskGm.SetGlobalBuffer((__gm__ T*)attenMask, ws1s2);
    biasGm.SetGlobalBuffer((__gm__ T*)bias, ns1s2);

    // ub空间为192 * 1024 Byte
    // 需要对齐的元素个数，对齐32B字节
    // 这些数据都可以在tiling阶段完成
    totalSizeAligned = tilingData->ns1s2Aligned;
    pipe.InitBuffer(vecInQueue, BUFFER_NUM, totalSizeAlignedSize);
    pipe.InitBuffer(vecOutQueue, BUFFER_NUM, totalSizeAlignedSize);
    // 需要在这里进行广播，所以直接分配广播后的buffer的大小
    pipe.InitBuffer(biasMaskQueue, BUFFER_NUM, totalSizeAlignedSize);

    if constexpr (existAtten) {
      pipe.InitBuffer(attenMaskQueue, BUFFER_NUM, totalSizeAlignedSize);
    }

    if constexpr (!AscendC::IsSameType<T, float>::value) {
      pipe.InitBuffer(CastBuff_, 3 * totalSizeAligned * sizeof(float)); // F16 + BF16新增3个
    }
  }
  __aicore__ inline void Process() {
    LocalTensor<T> biasLocal = biasMaskQueue.AllocTensor<T>();
    if (isAligend) {
      DataCopy(biasLocal[0], biasGm, totalSizeAligned);
    } else {
#if (__CCE_AICORE__ > 200)
      // bias 从[n, s1, s2] 广播为[w, n, s1, s2]
      DataCopyParams biasCopyParamsLast{ns1, s2DtypeSize, (uint16_t)(0),
                                        (uint16_t)(0)};
      DataCopyPadParams padParamsNormal{false, 0, 0, 0};
      DataCopyPad(biasLocal[0], biasGm, biasCopyParamsLast, padParamsNormal);
#endif
    }

    biasMaskQueue.EnQue(biasLocal);
    LocalTensor<T> biasMaskLocal = biasMaskQueue.DeQue<T>();
    for (uint32_t b = 0; b < batchSize; b++) {  // 需要同时载入多个batch, 考虑在compute里面进行for循环
      CopyIn(b);
      Compute(b, biasMaskLocal);
      CopyOut(b);
    }
    biasMaskQueue.FreeTensor(biasMaskLocal);
  }

 private:
  __aicore__ inline void CopyIn(int32_t b) {
    LocalTensor<T> xLocal = vecInQueue.AllocTensor<T>();
    if (isAligend) {
      DataCopy(xLocal, xGm[static_cast<int64_t>(b) * ns1s2], totalSizeAligned);
    } else {
#if (__CCE_AICORE__ > 200)
      // 数据copy的时候，需要考虑softmax的最后一个轴需要32B对齐
      DataCopyParams copyParamsLast{(uint16_t)(ns1), s2DtypeSize, (uint16_t)(0),
                                    (uint16_t)(0)};
      DataCopyPadParams padParamsNormal{false, 0, 0, 0};
      DataCopyPad(xLocal, xGm[static_cast<int64_t>(b) * ns1s2], copyParamsLast, padParamsNormal);
#endif
    }

    if constexpr (existAtten) {
      LocalTensor<T> attenMaskLocal = attenMaskQueue.AllocTensor<T>();
      // atten_mask 从[w,s1,s2] 广播为[w, n, s1, s2]
      // 使用DataCopy完成广播操作
      if (isAligend) {
        for (uint32_t ni = 0; ni < n_; ni++) {
          DataCopy(attenMaskLocal[ni * s1s2AlignedSize],
                    attenMaskGm[((atten_mask_offset + b) % w_) * s1s2],
                    s1s2AlignedSize);
        }
      } else {
#if (__CCE_AICORE__ > 200)
        DataCopyParams attenMaskCopyParamsLast{(uint16_t)(s1_), s2DtypeSize,
                                                (uint16_t)(0), (uint16_t)(0)};
        DataCopyPadParams padParamsNormal{false, 0, 0, 0};
        for (uint32_t ni = 0; ni < n_; ni++) {
          DataCopyPad(attenMaskLocal[ni * s1s2AlignedSize],
                      attenMaskGm[((atten_mask_offset + b) % w_) * s1s2],
                      attenMaskCopyParamsLast, padParamsNormal);
        }
#endif
      }
      attenMaskQueue.EnQue(attenMaskLocal);
    }
    vecInQueue.EnQue(xLocal);
  }

    __aicore__ inline void Compute(int32_t b, LocalTensor<T> &biasMaskLocal) {
      LocalTensor<T> xLocal = vecInQueue.DeQue<T>();
      LocalTensor<T> yLocal = vecOutQueue.AllocTensor<T>();

      if constexpr (!(AscendC::IsSameType<T, float>::value)) {
        LocalTensor<float> floatLocal = CastBuff_.Get<float>();
        LocalTensor<float> xFloatLocal = floatLocal[0];
        LocalTensor<float> biasMaskFloatLocal = floatLocal[totalSizeAligned * 2];

        // cast
        Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, totalSizeAligned);
        Cast(biasMaskFloatLocal, biasMaskLocal, RoundMode::CAST_NONE, totalSizeAligned);
        PipeBarrier<PIPE_V>();
        if (existMuls) {
          Muls(xFloatLocal, xFloatLocal, scaleValue, totalSizeAligned);
          PipeBarrier<PIPE_V>();
        }
        if constexpr (existAtten) {
          LocalTensor<float> attenMaskFloatLocal = floatLocal[totalSizeAligned];
          LocalTensor<T> attenMaskLocal = attenMaskQueue.DeQue<T>();
          Cast(attenMaskFloatLocal, attenMaskLocal, RoundMode::CAST_NONE, totalSizeAligned);
          PipeBarrier<PIPE_V>();
          Add(xFloatLocal, xFloatLocal, attenMaskFloatLocal, totalSizeAligned);
          PipeBarrier<PIPE_V>();
          attenMaskQueue.FreeTensor(attenMaskLocal);
        }

        Add(xFloatLocal, xFloatLocal, biasMaskFloatLocal, totalSizeAligned);
        PipeBarrier<PIPE_V>();

        SoftMaxShapeInfo softmaxShapeInfo{ns1, s2AlignedSize, ns1, s2_};
        AscendC::SoftMax<float, false, false>(xFloatLocal, xFloatLocal, softMaxTiling, softmaxShapeInfo);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<T, half>::value) {
          Cast(yLocal, xFloatLocal, RoundMode::CAST_NONE, totalSizeAligned);
        } else {
          Cast(yLocal, xFloatLocal, RoundMode::CAST_RINT, totalSizeAligned);
        }
      } else {
        if constexpr (existMuls) {
          Muls(yLocal, xLocal, static_cast<T>(scaleValue), totalSizeAligned);
          PipeBarrier<PIPE_V>();
        }

        if constexpr (existAtten) {
          LocalTensor<T> attenMaskLocal = attenMaskQueue.DeQue<T>();
          if (existMuls) {
            Add(yLocal, yLocal, attenMaskLocal, totalSizeAligned);
          } else {
            Add(yLocal, xLocal, attenMaskLocal, totalSizeAligned);
          }

          PipeBarrier<PIPE_V>();
          attenMaskQueue.FreeTensor(attenMaskLocal);
        }
        if constexpr (existMuls || existAtten) {
          Add(yLocal, yLocal, biasMaskLocal, totalSizeAligned);
        } else {
          Add(yLocal, xLocal, biasMaskLocal, totalSizeAligned);
        }
        PipeBarrier<PIPE_V>();

        SoftMaxShapeInfo softmaxShapeInfo{ns1, s2AlignedSize, ns1, s2_};
        AscendC::SoftMax<T, false, false>(yLocal, yLocal, softMaxTiling,
                                          softmaxShapeInfo);
      }

      vecOutQueue.EnQue<T>(yLocal);
      vecInQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t b) {
      LocalTensor<T> yLocal = vecOutQueue.DeQue<T>();
      if (isAligend) {
        DataCopy(yGm[static_cast<int64_t>(b) * ns1s2], yLocal, totalSizeAligned);
      } else {
#if (__CCE_AICORE__ > 200)
        DataCopyParams copyParamsLast{(uint16_t)(ns1), s2DtypeSize, (uint16_t)(0),
                                      (uint16_t)(0)};
        DataCopyPad(yGm[static_cast<int64_t>(b) * ns1s2], yLocal, copyParamsLast);
#endif
      }
      vecOutQueue.FreeTensor(yLocal);
    }

   private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECIN, 1> biasMaskQueue;
    TQue<QuePosition::VECIN, 1> attenMaskQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TBuf<QuePosition::VECCALC> CastBuff_;
    uint32_t batchSize, w_, n_, s1_, s2_;
    uint32_t s2AlignedSize;
    uint32_t totalSizeAligned;
    uint32_t totalSizeAlignedSize;
    uint32_t wns1;
    uint32_t ws1s2;
    uint32_t ns1s2;
    uint32_t s1s2;
    uint32_t s1s2AlignedSize;
    uint32_t ns1s2AlignedSize;
    uint16_t ns1;
    uint16_t s2DtypeSize;
    int64_t atten_mask_offset;
    float scaleValue;
    bool isAligend;
    GlobalTensor<T> xGm, yGm, attenMaskGm, biasGm;
    SoftMaxTiling softMaxTiling;
    const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ tilingData;
  };
}
#endif