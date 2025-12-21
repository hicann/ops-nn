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
 * \file masked_softmax_with_rel_pos_bias_BWN.h
 * \brief
 */
#ifndef ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_BWN_H
#define ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_BWN_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MaskedSoftmaxWithRelPosBias {
using namespace AscendC;
template <typename T, bool existAtten, bool existMuls>
class MaskedSoftmaxWithRelPosBiasBWN {
 public:
  __aicore__ inline MaskedSoftmaxWithRelPosBiasBWN(
      const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ maskedSoftmaxWithRelPosBiasBTilingData)
      : tilingData(maskedSoftmaxWithRelPosBiasBTilingData){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR attenMask, GM_ADDR bias, GM_ADDR y) {
    int64_t offset = 0;
    if (GetBlockIdx() < tilingData->tailStartCoreIdx) {
      offset = GetBlockIdx() * tilingData->singleCoreSize;
      batchSize = tilingData->singleCoreSize; // 记录当前核的处理batch的个数
    } else {
      offset = GetBlockIdx() * (tilingData->singleCoreSize - 1) + tilingData->tailStartCoreIdx;
      batchSize = tilingData->singleCoreSize - 1;
    }

    if (batchSize == 0) {
      return;
    }
    w_ = tilingData->w;
    n_ = tilingData->n;
    s1_ = tilingData->s1;
    s2_ = tilingData->s2;
    s1s2_ = s1_ * s2_;
    ws1s2_ = w_ * s1s2_;
    ns1s2_ = n_ * s1s2_;
    softMaxTiling = tilingData->softmaxTilingData;
    atten_mask_offset = offset;
    relative_pos_bias_offset = offset % n_;
    uint32_t offsets1s2_ = offset * s1s2_;
    xGm.SetGlobalBuffer((__gm__ T*)x + offsets1s2_, batchSize * s1s2_);
    yGm.SetGlobalBuffer((__gm__ T*)y + offsets1s2_, batchSize * s1s2_);
    if constexpr (existAtten) {
      attenMaskGm.SetGlobalBuffer((__gm__ T*)attenMask, ws1s2_);
    }
    biasGm.SetGlobalBuffer((__gm__ T*)bias, ns1s2_);

    stackSize = tilingData->stackNum;
    this->onceBatchSize = tilingData->stackNum;
    s2AlignedSize = tilingData->s2Aligned;
    s1s2AlignedSize_ = s1_ * s2AlignedSize;
    if constexpr (existAtten) {
      totalSizeAligned = this->onceBatchSize * s1s2AlignedSize_;
      pipe.InitBuffer(maskQueue, BUFFER_NUM, stackSize * 2 * s1s2AlignedSize_ * sizeof(T)); // 2 is for mask and stten
      if constexpr (!(AscendC::IsSameType<T, float>::value)) {
        pipe.InitBuffer(CastBuff_, 4 * stackSize * s1s2AlignedSize_ * sizeof(float)); // BF16新增4个
      }
    } else {
      totalSizeAligned = 0;
      pipe.InitBuffer(maskQueue, BUFFER_NUM, stackSize * s1s2AlignedSize_ * sizeof(T));
      if constexpr (!(AscendC::IsSameType<T, float>::value)) {
        pipe.InitBuffer(CastBuff_, 3 * stackSize * s1s2AlignedSize_ * sizeof(float)); // BF16新增3个，不带atten
      }
    }
    pipe.InitBuffer(vecInQueue, BUFFER_NUM, stackSize * s1s2AlignedSize_ * sizeof(T));
    pipe.InitBuffer(vecOutQueue, BUFFER_NUM, stackSize * s1s2AlignedSize_ * sizeof(T));
  }
  __aicore__ inline void Process() {
    if (stackSize == 0) {
      return;
    }
    uint32_t tmpn = 0;
    uint32_t round = batchSize / stackSize;
    uint32_t tail = batchSize % stackSize;
    SoftMaxShapeInfo softmaxShapeInfo{stackSize * s1_, s2AlignedSize, stackSize * s1_, s2_};
    if (s2AlignedSize == s2_) {
      for (uint32_t i = 0; i < round; i++) {
        CopyInAligned(i, stackSize, tmpn);
        Compute(i, stackSize, tmpn, softmaxShapeInfo, tilingData->softmaxTilingData);
        CopyOutAligned(i, stackSize);
      }
      if (tail != 0) {
        SoftMaxShapeInfo tailSoftmaxShapeInfo{stackSize * s1_, s2AlignedSize, stackSize * s1_, s2_};
        CopyInAligned(round, tail, tmpn);
        Compute(round, tail, tmpn, tailSoftmaxShapeInfo, tilingData->tailSoftmaxTilingData);
        CopyOutAligned(round, tail);
      }
    } else {
#if (__CCE_AICORE__ > 200)
      for (uint32_t i = 0; i < round; i++) {
        CopyInNonAligned(i, stackSize, tmpn);
        Compute(i, stackSize, tmpn, softmaxShapeInfo, tilingData->softmaxTilingData);
        CopyOutNonAligned(i, stackSize);
      }
      if (tail != 0) {
        SoftMaxShapeInfo tailSoftmaxShapeInfo{stackSize * s1_, s2AlignedSize, stackSize * s1_, s2_};
        CopyInNonAligned(round, tail, tmpn);
        Compute(round, tail, tmpn, tailSoftmaxShapeInfo, tilingData->tailSoftmaxTilingData);
        CopyOutNonAligned(round, tail);
      }
#endif
    }
  }

 private:
  __aicore__ inline void ComputeFp32(int32_t b, uint32_t stackSize1, uint32_t& tmpn, const SoftMaxShapeInfo& softmaxShapeInfo, const SoftMaxTiling& softmaxTilingData) {
    LocalTensor<T> xLocal = vecInQueue.DeQue<T>();
    LocalTensor<T> attenMaskLocal = maskQueue.DeQue<T>();
    if constexpr (existAtten) {
      if (stackSize <= tmpn) {
          uint32_t attenOffset = s1s2AlignedSize_;
          for (uint32_t j = 1; j < stackSize; j++) {
            Adds(attenMaskLocal[attenOffset], attenMaskLocal[0], (T)0, s1s2AlignedSize_);
            attenOffset += s1s2AlignedSize_;
          }
      } else {
          uint32_t attenOffset1 = s1s2AlignedSize_;
          for (uint32_t j = 1; j < tmpn; j++) {
            Adds(attenMaskLocal[attenOffset1], attenMaskLocal[0], (T)0, s1s2AlignedSize_);
            attenOffset1 += s1s2AlignedSize_;
          }
          uint32_t i2 = tmpn + 1;
          uint32_t attenOffset2 =  i2 * s1s2AlignedSize_;
          for (uint32_t j = i2; j < stackSize; j++) {
            Adds(attenMaskLocal[attenOffset2], attenMaskLocal[tmpn * s1s2AlignedSize_], (T)0, s1s2AlignedSize_);
            attenOffset2 += s1s2AlignedSize_;
          }
      }
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<T> biasLocal = attenMaskLocal[totalSizeAligned];
    LocalTensor<T> yLocal = vecOutQueue.AllocTensor<T>();
    uint32_t calNum = stackSize1 * s1s2AlignedSize_;
    if constexpr (existMuls && existAtten) {
      Muls(xLocal, xLocal, static_cast<T>(tilingData->scaleValue), calNum);
      PipeBarrier<PIPE_V>();
      Add(yLocal, xLocal, attenMaskLocal, calNum);
      PipeBarrier<PIPE_V>();
      Add(xLocal, yLocal, biasLocal, calNum);
      PipeBarrier<PIPE_V>();
      AscendC::SoftMax<T, true, false>(yLocal, xLocal, softmaxTilingData, softmaxShapeInfo);
    } else if constexpr (existMuls) {
      Muls(yLocal, xLocal, static_cast<T>(tilingData->scaleValue), calNum);
      PipeBarrier<PIPE_V>();
      Add(xLocal, yLocal, biasLocal, calNum);
      PipeBarrier<PIPE_V>();
      AscendC::SoftMax<T, true, false>(yLocal, xLocal, softmaxTilingData, softmaxShapeInfo);
    } else if constexpr (existAtten) {
      Add(yLocal, xLocal, attenMaskLocal, calNum);
      PipeBarrier<PIPE_V>();
      Add(xLocal, yLocal, biasLocal, calNum);
      PipeBarrier<PIPE_V>();
      AscendC::SoftMax<T, true, false>(yLocal, xLocal, softmaxTilingData, softmaxShapeInfo);
    } else {
      Add(xLocal, xLocal, biasLocal, calNum);
      PipeBarrier<PIPE_V>();
      AscendC::SoftMax<T, true, false>(yLocal, xLocal, softmaxTilingData, softmaxShapeInfo);
    }
    PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue<T>(yLocal);
    vecInQueue.FreeTensor(xLocal);
    maskQueue.FreeTensor(attenMaskLocal);
  }
  __aicore__ inline void CastAttenMaskForBf16AndFp16(LocalTensor<T>& attenMaskLocal, LocalTensor<float>& attenMaskFloatLocal, uint32_t& tmpn) {
    if constexpr (existAtten) {
      if (stackSize <= tmpn) {
          uint32_t i = 0;
          uint32_t attenOffset = 0;
          for (uint32_t j = i; j < stackSize; j++) {
            Cast(attenMaskFloatLocal[attenOffset], attenMaskLocal[0], RoundMode::CAST_NONE, s1s2AlignedSize_);
            attenOffset += s1s2AlignedSize_;
          }
      } else {
          uint32_t i = 0;
          uint32_t attenOffset1 = 0;
          for (uint32_t j = i; j < tmpn; j++) {
            Cast(attenMaskFloatLocal[attenOffset1], attenMaskLocal[0], RoundMode::CAST_NONE, s1s2AlignedSize_);
            attenOffset1 += s1s2AlignedSize_;
          }
          uint32_t i2 = tmpn;
          uint32_t attenOffset2 =  tmpn * s1s2AlignedSize_;
          for (uint32_t j = i2; j < stackSize; j++) {
            Cast(attenMaskFloatLocal[attenOffset2], attenMaskLocal[tmpn * s1s2AlignedSize_], RoundMode::CAST_NONE, s1s2AlignedSize_);
            attenOffset2 += s1s2AlignedSize_;
          }
      }
    }
  }
  __aicore__ inline void ComputeBf16AndFp16(int32_t b, uint32_t stackSize1, uint32_t& tmpn, const SoftMaxShapeInfo& softmaxShapeInfo,
                                            const SoftMaxTiling& softmaxTilingData) {
    LocalTensor<T> xLocal = vecInQueue.DeQue<T>();
    LocalTensor<T> attenMaskLocal = maskQueue.DeQue<T>();
    LocalTensor<T> biasLocal = attenMaskLocal[totalSizeAligned];
    LocalTensor<T> yLocal = vecOutQueue.AllocTensor<T>();

    uint32_t calNum = stackSize1 * s1s2AlignedSize_;
    LocalTensor<float> xFloatLocal = CastBuff_.Get<float>();
    LocalTensor<float> attenMaskFloatLocal = xFloatLocal[calNum]; // 如果existAtten为false，则此变量不使用。
    LocalTensor<float> biasMaskFloatLocal = attenMaskFloatLocal;  // 如果existAtten为false，此变量才会使用，否者会用新的biasMaskFloatLocal的值。
    if constexpr (existAtten) {
      biasMaskFloatLocal = attenMaskFloatLocal[calNum];
    }
    LocalTensor<float> yFloatLocal = biasMaskFloatLocal[calNum];
    // Set xFloatLocal size equal to yFloatLocal, avoid AscendC::SoftMax index out of range
    xFloatLocal.SetSize(yFloatLocal.GetSize());

    //cast
    Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, calNum);
    if constexpr (existAtten) {
      CastAttenMaskForBf16AndFp16(attenMaskLocal, attenMaskFloatLocal, tmpn);
    }
    Cast(biasMaskFloatLocal, biasLocal, RoundMode::CAST_NONE, calNum);
    PipeBarrier<PIPE_V>();
    // 这里保证了源地址和目的地址不是同一个，解决ub bank冲突
    if constexpr (existMuls && existAtten) {
      Muls(yFloatLocal, xFloatLocal, tilingData->scaleValue, calNum);
      PipeBarrier<PIPE_V>();
      Add(xFloatLocal, yFloatLocal, attenMaskFloatLocal, calNum);
      PipeBarrier<PIPE_V>();
      Add(yFloatLocal, xFloatLocal, biasMaskFloatLocal, calNum);
      PipeBarrier<PIPE_V>();
      AscendC::SoftMax<float, true, false>(xFloatLocal, yFloatLocal, softmaxTilingData, softmaxShapeInfo);
      PipeBarrier<PIPE_V>();
      if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(yLocal, xFloatLocal, RoundMode::CAST_NONE, calNum);
      } else {
        Cast(yLocal, xFloatLocal, RoundMode::CAST_RINT, calNum);
      }
      PipeBarrier<PIPE_V>();
    } else if constexpr (existMuls) {
      Muls(yFloatLocal, xFloatLocal, tilingData->scaleValue, calNum);
      PipeBarrier<PIPE_V>();
      Add(xFloatLocal, yFloatLocal, biasMaskFloatLocal, calNum);
      PipeBarrier<PIPE_V>();
      AscendC::SoftMax<float, true, false>(yFloatLocal, xFloatLocal, softmaxTilingData, softmaxShapeInfo);
      PipeBarrier<PIPE_V>();
      if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(yLocal, yFloatLocal, RoundMode::CAST_NONE, calNum);
      } else {
        Cast(yLocal, yFloatLocal, RoundMode::CAST_RINT, calNum);
      }
      PipeBarrier<PIPE_V>();
    } else if constexpr (existAtten) {
      Add(yFloatLocal, xFloatLocal, attenMaskFloatLocal, calNum);
      PipeBarrier<PIPE_V>();
      Add(xFloatLocal, yFloatLocal, biasMaskFloatLocal, calNum);
      PipeBarrier<PIPE_V>();
      AscendC::SoftMax<float, true, false>(yFloatLocal, xFloatLocal, softmaxTilingData, softmaxShapeInfo);
      PipeBarrier<PIPE_V>();
      if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(yLocal, yFloatLocal, RoundMode::CAST_NONE, calNum);
      } else {
        Cast(yLocal, yFloatLocal, RoundMode::CAST_RINT, calNum);
      }
      PipeBarrier<PIPE_V>();
    } else {
      Add(yFloatLocal, xFloatLocal, biasMaskFloatLocal, calNum);
      PipeBarrier<PIPE_V>();
      AscendC::SoftMax<float, true, false>(xFloatLocal, yFloatLocal, softmaxTilingData, softmaxShapeInfo);
      PipeBarrier<PIPE_V>();
      if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(yLocal, xFloatLocal, RoundMode::CAST_NONE, calNum);
      } else {
        Cast(yLocal, xFloatLocal, RoundMode::CAST_RINT, calNum);
      }
      PipeBarrier<PIPE_V>();
    }
    vecOutQueue.EnQue<T>(yLocal);
    vecInQueue.FreeTensor(xLocal);
    maskQueue.FreeTensor(attenMaskLocal);
  }

  __aicore__ inline void Compute(int32_t i, uint32_t stackSize, uint32_t& tmpn, const SoftMaxShapeInfo& softmaxShapeInfo,
                                 const SoftMaxTiling& softmaxTilingData) {
      if constexpr (!(AscendC::IsSameType<T, float>::value)) {
        ComputeBf16AndFp16(i, stackSize, tmpn, softmaxShapeInfo, softmaxTilingData);
      } else {
        ComputeFp32(i, stackSize, tmpn, softmaxShapeInfo, softmaxTilingData);
      }
  }

#if (__CCE_AICORE__ > 200)
  __aicore__ inline void CopyInNonAligned(int32_t process, uint32_t stackSize, uint32_t& tmpn) {
    LocalTensor<T> xLocal = vecInQueue.AllocTensor<T>();
    LocalTensor<T> maskLocal = maskQueue.AllocTensor<T>();

    DataCopyParams copyParamsLast{(uint16_t)(s1_ * stackSize), (uint16_t)(s2_ * sizeof(T)), (uint16_t)(0), (uint16_t)(0)};
    DataCopyPadParams padParamsNormal{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[(static_cast<int64_t>(process) * this->onceBatchSize) * s1s2_], copyParamsLast, padParamsNormal);
    auto biasLocal = maskLocal[totalSizeAligned];
    uint32_t localOffset = (process * this->onceBatchSize + relative_pos_bias_offset);
    tmpn = n_ - localOffset % n_;
    if (stackSize <= tmpn) {
      if constexpr (existAtten) {
        uint32_t i = 0;
        DataCopyParams attenMaskCopyParamsLast{(uint16_t)(s1_), (uint16_t)(s2_ * sizeof(T)), (uint16_t)(0), (uint16_t)(0)};
        DataCopyPad(maskLocal[i * s1s2AlignedSize_],
                    attenMaskGm[(static_cast<int64_t>(process) * this->onceBatchSize + i + atten_mask_offset) / n_ % w_ * s1s2_],
                    attenMaskCopyParamsLast, padParamsNormal);
      }
      DataCopyParams biasCopyParamsLast{(uint16_t)(s1_ * stackSize), (uint16_t)(s2_ * sizeof(T)), (uint16_t)(0), (uint16_t)(0)};
      DataCopyPad(biasLocal, biasGm[static_cast<int64_t>(localOffset % n_) * s1s2_], biasCopyParamsLast, padParamsNormal);
    } else {
      if constexpr (existAtten) {
          uint32_t i = 0;
          DataCopyParams attenMaskCopyParamsLast{(uint16_t)(s1_), (uint16_t)(s2_ * sizeof(T)), (uint16_t)(0), (uint16_t)(0)};
          DataCopyPad(maskLocal[i * s1s2AlignedSize_],
                      attenMaskGm[(static_cast<int64_t>(process) * this->onceBatchSize + i + atten_mask_offset) / n_ % w_ * s1s2_],
                      attenMaskCopyParamsLast, padParamsNormal);
          uint32_t i2 = tmpn;
          DataCopyPad(maskLocal[i2 * s1s2AlignedSize_],
                      attenMaskGm[(static_cast<int64_t>(process) * this->onceBatchSize + i2 + atten_mask_offset) / n_ % w_ * s1s2_],
                      attenMaskCopyParamsLast, padParamsNormal);
      }
      uint32_t tailNum = stackSize - tmpn;
      DataCopyParams biasCopyParamsLast{(uint16_t)(s1_ * tmpn), (uint16_t)(s2_ * sizeof(T)), (uint16_t)(0), (uint16_t)(0)};
      DataCopyPad(biasLocal, biasGm[static_cast<int64_t>(localOffset % n_) * s1s2_], biasCopyParamsLast, padParamsNormal);
      DataCopyParams biasCopyParamsLastTail{(uint16_t)(s1_ * tailNum), (uint16_t)(s2_ * sizeof(T)), (uint16_t)(0), (uint16_t)(0)};
      DataCopyPad(biasLocal[tmpn*s1s2AlignedSize_], biasGm, biasCopyParamsLastTail, padParamsNormal);
    }
    vecInQueue.EnQue(xLocal);
    maskQueue.EnQue(maskLocal);
  }

  __aicore__ inline void CopyOutNonAligned(int32_t process, uint32_t stackSize) {
    LocalTensor<T> yLocal = vecOutQueue.DeQue<T>();
    DataCopyParams copyParamsLast{(uint16_t)(stackSize * s1_), (uint16_t)(s2_ * sizeof(T)), (uint16_t)(0), (uint16_t)(0)};
    DataCopyPad(yGm[static_cast<int64_t>(process) * this->onceBatchSize * s1s2_], yLocal, copyParamsLast);
    vecOutQueue.FreeTensor(yLocal);
  }
#endif

  __aicore__ inline void CopyInAligned(int32_t process, uint32_t stackSize, uint32_t& tmpn) {
    LocalTensor<T> xLocal = vecInQueue.AllocTensor<T>();
    LocalTensor<T> maskLocal = maskQueue.AllocTensor<T>();
    DataCopy(xLocal, xGm[(static_cast<int64_t>(process) * this->onceBatchSize) * s1s2_], stackSize * s1s2AlignedSize_);
    auto biasLocal = maskLocal[totalSizeAligned];
    {
      uint32_t localOffset = (process * this->onceBatchSize + relative_pos_bias_offset);
      tmpn = n_ - localOffset % n_;
      if (stackSize <= tmpn) {
        if constexpr (existAtten) {
          uint32_t i = 0;
          DataCopy(maskLocal[i * s1s2AlignedSize_], attenMaskGm[(static_cast<int64_t>(process) * this->onceBatchSize + i + atten_mask_offset) / n_ % w_ * s1s2_],
                   s1s2AlignedSize_);
        }
        DataCopy(biasLocal, biasGm[static_cast<int64_t>(localOffset % n_) * s1s2_], stackSize * s1s2AlignedSize_);
      } else {
        if constexpr (existAtten) {
          uint32_t i = 0;
          DataCopy(maskLocal[i * s1s2AlignedSize_], attenMaskGm[(static_cast<int64_t>(process) * this->onceBatchSize + i + atten_mask_offset) / n_ % w_ * s1s2_],
                   s1s2AlignedSize_);
          uint32_t i2 = tmpn;
          DataCopy(maskLocal[i2 * s1s2AlignedSize_], attenMaskGm[(static_cast<int64_t>(process) * this->onceBatchSize + i2 + atten_mask_offset) / n_ % w_ * s1s2_],
                   s1s2AlignedSize_);
        }
        uint32_t tailNum = stackSize - tmpn;
        DataCopy(biasLocal, biasGm[static_cast<int64_t>(localOffset % n_) * s1s2_], tmpn * s1s2AlignedSize_);
        DataCopy(biasLocal[tmpn*s1s2AlignedSize_], biasGm, tailNum * s1s2AlignedSize_);
      }
    }
    vecInQueue.EnQue(xLocal);
    maskQueue.EnQue(maskLocal);
  }

  __aicore__ inline void CopyOutAligned(int32_t process, uint32_t stackSize) {
    LocalTensor<T> yLocal = vecOutQueue.DeQue<T>();
    DataCopy(yGm[static_cast<int64_t>(process) * this->onceBatchSize * s1s2_], yLocal, stackSize * s1s2AlignedSize_);
    vecOutQueue.FreeTensor(yLocal);
  }

 private:
  TPipe pipe;
  TQue<QuePosition::VECIN, 1> vecInQueue;
  TQue<QuePosition::VECIN, 1> maskQueue;
  TQue<QuePosition::VECOUT, 1> vecOutQueue;
  TBuf<QuePosition::VECCALC> CastBuff_;
  uint32_t batchSize, w_, n_, s1_, s2_;

  uint32_t totalSizeAligned;
  uint32_t alignNum;
  uint32_t onceBatchSize;
  int64_t atten_mask_offset;
  int64_t relative_pos_bias_offset;
  uint32_t stackSize; // Compute 循环一次 处理的onceBatchSize * s1*s2的个数
  GlobalTensor<T> xGm, yGm, attenMaskGm, biasGm;
  SoftMaxTiling softMaxTiling;
  uint32_t s1s2_;
  uint32_t s1s2AlignedSize_;
  uint32_t s2AlignedSize;
  uint32_t ws1s2_;
  uint32_t ns1s2_;
  const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ tilingData;
};
}
#endif