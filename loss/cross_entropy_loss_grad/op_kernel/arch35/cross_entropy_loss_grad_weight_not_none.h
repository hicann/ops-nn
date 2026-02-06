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
 * \file cross_entropy_loss_grad_weight_not_none.h
 * \brief
 */
#ifndef CROSS_ENTROPY_LOSS_GRAD_WEIGHT_NOT_NONE_H
#define CROSS_ENTROPY_LOSS_GRAD_WEIGHT_NOT_NONE_H

#include "cross_entropy_loss_grad_base.h"

namespace CrossEntropyLossGradRegbase {

template <typename T1, typename T2>
class CrossEntropyLossGradWeightNotNone: protected CrossEntropyLossGradBase<T1, T2> {
public:
  __aicore__ inline CrossEntropyLossGradWeightNotNone(){};
  __aicore__ inline void Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target, GM_ADDR weight, GM_ADDR grad_zloss,
                              GM_ADDR lse_for_zloss, GM_ADDR x_grad, GM_ADDR workspace,
                              const CrossEntropyLossGradRegbaseTilingData& tilingData);
  __aicore__ inline void Process();

protected:
  __aicore__ inline void ComputerEachBatch(uint64_t nLoopIdx, uint64_t peerLoopN, uint64_t nLoopNum);
  __aicore__ inline void ComputeLog(uint64_t nLoopIdx, uint64_t peerLoopN, uint64_t cLoopIdx, uint64_t calcLen, const float smoothGradScalar, const float smoothGradReduceSum);
  __aicore__ inline void LogSoftmaxWeightEachBatch(uint64_t nLoopNum);
  __aicore__ inline void LogSoftmaxWeightMainAddTailBlock(uint64_t nLoopNum);
  __aicore__ inline void LogSoftmaxWeightAlignBlock(uint64_t nLoopNum);
  __aicore__ inline void TargetWeightSumTailN(uint64_t targetNum, uint64_t peerLoopNIdx);
  __aicore__ inline void ComputeLogSmoothLossMainBlock(uint64_t nLoopNum, uint64_t cOnceNumTail, uint64_t cOnceNumAlign, LocalTensor<float> &weightLocal, LocalTensor<float> &tmpLocal);
  __aicore__ inline void ComputeLogSmoothLossTailBlock(uint64_t nLoopNum, uint64_t cOnceNumTail, uint64_t cOnceNumAlign, LocalTensor<float> &weightLocal, LocalTensor<float> &tmpLocal);
  __aicore__ inline void ComputeLogSmoothLoss(uint64_t nLoopNum, uint64_t cOnceNumTail, uint64_t cOnceNumAlign, LocalTensor<float> &weightLocal, LocalTensor<float> &tmpLocal);
  __aicore__ inline void SmoothLabel(uint64_t nLoopIdx, uint64_t calcLen, LocalTensor<T1> &logProbLocal, const float smoothGradScalar, const float smoothGradReduceSum);
  __aicore__ inline void ProcessPerCore(uint64_t peerLoopNIdx, uint64_t nLoopNum);
  __aicore__ inline void GradReductionMeanSumWeight(float factor, uint64_t targetNum);
  __aicore__ inline void GradReductionMeanSumSmoothWeight(float factor, uint64_t targetNum);
  __aicore__ inline void UpdateCache(LocalTensor<float> &ubDst, LocalTensor<float> &ubSrc, int64_t index, int64_t dimA);
private:
  TBuf<TPosition::VECCALC> gradLoss;
  TBuf<TPosition::VECCALC> targetWeightBuf;
  TBuf<TPosition::VECCALC> fp32Buf4;
  TBuf<TPosition::VECCALC> fp32Buf6;
  TBuf<TPosition::VECCALC> fp32Buf7;
  TBuf<TPosition::VECCALC> smoothSumBuf; // weight not null and smooth

  LocalTensor<T1> xGradLocal;
  LocalTensor<float> targetWeightLocal;
  LocalTensor<float> fp32Buf4Local;
  LocalTensor<float> smoothSumLocal; // weight not null and smooth
};

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::UpdateCache(LocalTensor<float> &ubDst, LocalTensor<float> &ubSrc, int64_t index, int64_t dimA)
 {
    const int64_t cacheID = ScalarGetCountOfValue<1>(index ^ (index + 1)) - 1;
    int64_t elementOneRepeat = platform::GetVRegSize() / sizeof(float);
    const int64_t stride = ops::CeilAlign(dimA, elementOneRepeat);

    // count A轴的大小 * vlSize
    const uint16_t outerLoopTimes =
        ops::CeilDiv(static_cast<int64_t>(dimA * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    const uint16_t innerLoopTimes = static_cast<uint16_t>(cacheID);
    const uint32_t outerLoopStride = platform::GetVRegSize() / sizeof(float);
    const uint32_t innerLoopStride = static_cast<uint32_t>(stride);  // cacahe的每一个idex的块的大小， A轴的大小
    LocalTensor<float> dstTensor = ubDst;
    LocalTensor<float> srcTensor = ubSrc;

    __VEC_SCOPE__
    {
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* cah = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheID * stride;
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();
        uint32_t sreg = static_cast<uint32_t>(dimA);
        AscendC::MicroAPI::RegTensor<float> aReg, bReg;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {  // outerLoopTimes是dimA的大小
            pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            DataCopy(aReg, (__local_mem__ float*)src + i * outerLoopStride);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                DataCopy(bReg, (__local_mem__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            DataCopy((__local_mem__ float*)cah + i * outerLoopStride, aReg, pMask);
        }
    }
 }

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
                                                           GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
                                                           GM_ADDR x_grad, GM_ADDR workspace,
                                                           const CrossEntropyLossGradRegbaseTilingData& tilingData) {

  CrossEntropyLossGradBase<T1, T2>::Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss,
                                    x_grad, workspace, tilingData);

  uint64_t bufferSize = this->alignColLoopNum > this->colLoopNumTail ?
                      this->alignColLoopNum : AlignUp(this->colLoopNumTail, MASK);
  bufferSize = bufferSize > 256 ? bufferSize : 256;
  this->pipe.InitBuffer(this->inQueWeight_, BUFFER_NUM, bufferSize * sizeof(float));
  this->pipe.InitBuffer(targetWeightBuf, this->targetWeightSize);
  this->pipe.InitBuffer(fp32Buf4, bufferSize * sizeof(float));
  targetWeightLocal = targetWeightBuf.Get<float>();
  fp32Buf4Local = fp32Buf4.Get<float>();

  if (this->labelSmoothing != 0) {
    this->pipe.InitBuffer(fp32Buf6, bufferSize * sizeof(float));
    uint64_t nNum = this->nLoopTimes == 0 ? this->nLoopTail : this->alignNLoopNum;
    this->pipe.InitBuffer(fp32Buf7, bufferSize * nNum * sizeof(float));
    this->pipe.InitBuffer(smoothSumBuf, (bufferSize + this->cacheStart) * sizeof(float));  //weight != null
    smoothSumLocal = smoothSumBuf.Get<float>();
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::TargetWeightSumTailN(uint64_t targetNum, uint64_t peerLoopNIdx) {
  if (this->ignoreIndex >= 0) {

    Mul(this->weightMaskLocal, this->ignoreSelect, targetWeightLocal, targetNum);
  } else {
    Muls(this->weightMaskLocal, targetWeightLocal, 1.0f, targetNum);
  }

  this->WeightReduceSumTailN(targetNum, peerLoopNIdx, this->weightMaskLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::SmoothLabel(uint64_t nLoopIdx, uint64_t calcLen, LocalTensor<T1>& logProbLocal, const float smoothGradScalar, const float smoothGradReduceSum) {
  LocalTensor<float> weightLocal = this->inQueWeight_.template DeQue<float>();
  uint32_t oneRepeat = platform::GetVRegSize() / sizeof(float);
  uint32_t totalLen = calcLen;
  uint32_t repeatTimes = CeilDivision(totalLen, oneRepeat);
  auto logProbAddr = (__ubuf__ T1 *)logProbLocal.GetPhyAddr();
  auto xGradTmpAddr = (__ubuf__ float *)fp32Buf4Local.GetPhyAddr();
  auto xGradAddr = (__ubuf__ T1 *)xGradLocal.GetPhyAddr();
  auto weightAddr = (__ubuf__ float *)weightLocal.GetPhyAddr();
  __VEC_SCOPE__ {
      MicroAPI::MaskReg pregUp;
      MicroAPI::MaskReg pregT;
      MicroAPI::RegTensor<float> regLogProb;
      MicroAPI::RegTensor<float> regXGrad;
      MicroAPI::RegTensor<float> regWeight;
      for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
          pregUp = MicroAPI::UpdateMask<float>(totalLen);
          this->LoadRegTensor(regLogProb, logProbAddr, pregUp, (int32_t)oneRepeat);
          MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regWeight, weightAddr, (int32_t)oneRepeat);
          MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regXGrad, xGradTmpAddr, (int32_t)oneRepeat);
          MicroAPI::Exp(regLogProb, regLogProb, pregUp);
          MicroAPI::Muls(regLogProb, regLogProb, smoothGradReduceSum, pregUp);
          MicroAPI::Muls(regWeight, regWeight, smoothGradScalar, pregUp);
          MicroAPI::Sub(regLogProb, regLogProb, regWeight, pregUp);
          MicroAPI::Add(regLogProb, regLogProb, regXGrad, pregUp);
          this->CopyOutRegTensor(xGradAddr, regLogProb, pregUp, (int32_t)oneRepeat);
      }
  }
  this->inQueWeight_.FreeTensor(weightLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::ComputeLog(uint64_t nLoopIdx, uint64_t peerLoopN,
                                                                             uint64_t cLoopIdx, uint64_t calcLen,
                                                                             const float smoothGradScalar,
                                                                             const float smoothGradReduceSum) {
    LocalTensor<T1> logProbLocal = this->inQueLogProb_.template DeQue<T1>();

    this->ComputeLogPre(logProbLocal, fp32Buf4Local, nLoopIdx, peerLoopN, cLoopIdx, calcLen);
    xGradLocal = this->outQueXGrad_.template AllocTensor<T1>();
    if (this->labelSmoothing != 0) {
        this->CopyInWeight(cLoopIdx, calcLen);
        SmoothLabel(nLoopIdx, calcLen, logProbLocal, smoothGradScalar, smoothGradReduceSum);
    } else {
        if constexpr (!IsSameType<T1, float>::value) {
          Cast(xGradLocal, fp32Buf4Local, RoundMode::CAST_RINT, calcLen);
        } else {
          uint32_t peerLoopNum = B8_MAX_NUM * MASK;
          uint32_t tailNum = calcLen % peerLoopNum;
          uint32_t repeatLoop = calcLen / peerLoopNum;
          for (int32_t i = 0; i < repeatLoop; i++) {
            Copy(xGradLocal[i * peerLoopNum], fp32Buf4Local[i * peerLoopNum], MASK, B8_MAX_NUM, {1, 1, 8, 8});
          }
          if (tailNum != 0) {
            Copy(xGradLocal[repeatLoop * peerLoopNum], fp32Buf4Local[repeatLoop * peerLoopNum], MASK, (tailNum + MASK -1) / MASK, {1, 1, 8, 8});
          }
        }
    }
    this->inQueLogProb_.template FreeTensor(logProbLocal);
    this->outQueXGrad_.template EnQue<T1>(xGradLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::ComputeLogSmoothLossMainBlock(uint64_t nLoopNum, uint64_t cOnceNumTail, uint64_t cOnceNumAlign, LocalTensor<float> &weightLocal, LocalTensor<float> &tmpLocal) {
  uint16_t nTimes = nLoopNum;
  uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
  uint16_t repeatTimes1 = cOnceNumTail / vfLen;
  uint16_t cOnceNum = this->cOnceNum;

  auto smGradAddr = (__ubuf__ float *)tmpLocal.GetPhyAddr();
  auto weightAddr = (__ubuf__ float *)weightLocal.GetPhyAddr();
  auto smoothAddr = (__ubuf__ float *)this->smoothLocal.GetPhyAddr();
  __VEC_SCOPE__
  {
    MicroAPI::MaskReg copyOutReg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> regWeight;
    MicroAPI::RegTensor<float> regSmoothGrad;
    for (uint16_t i = 0; i < nTimes; i++) {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(regSmoothGrad, smoothAddr + i);

        for (uint16_t loop = 0; loop < repeatTimes1; loop++) {
            MicroAPI::AddrReg srcOffset = MicroAPI::CreateAddrReg<float>(loop, vfLen);
            MicroAPI::AddrReg dstOffset = MicroAPI::CreateAddrReg<float>(i, cOnceNum, loop, vfLen);
            MicroAPI::DataCopy(regWeight, weightAddr, srcOffset);
            MicroAPI::Mul(regWeight, regWeight, regSmoothGrad, copyOutReg);
            MicroAPI::DataCopy(smGradAddr, regWeight, dstOffset, copyOutReg);
        }
    }
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::ComputeLogSmoothLossTailBlock(uint64_t nLoopNum, uint64_t cOnceNumTail, uint64_t cOnceNumAlign, LocalTensor<float> &weightLocal, LocalTensor<float> &tmpLocal) {
  uint16_t nTimes = nLoopNum;
  uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
  uint16_t repeatTimes1 = cOnceNumTail / vfLen;
  uint32_t tailNum = cOnceNumTail % vfLen;
  uint16_t cOnceNum = this->cOnceNum;
  uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;

  auto smoothAddr = (__ubuf__ float *)this->smoothLocal.GetPhyAddr();
  auto smGradTailAddr = (__ubuf__ float *)tmpLocal.GetPhyAddr() + repeatTimes1 * vfLen;
  auto weightAddr = (__ubuf__ float *)weightLocal.GetPhyAddr();
  auto weightTailAddr = (__ubuf__ float *)weightLocal.GetPhyAddr() + repeatTimes1 * vfLen;
  auto smGradAddr = (__ubuf__ float *)tmpLocal.GetPhyAddr();
  __VEC_SCOPE__
  {
    MicroAPI::RegTensor<float> regWeight;
    MicroAPI::RegTensor<float> regSmoothGrad;
    MicroAPI::RegTensor<float> regOut;
    MicroAPI::RegTensor<float> regOutAdd;
    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(tailNum);
    MicroAPI::MaskReg copyOutReg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < nTimes; i++) {
      MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(regSmoothGrad, smoothAddr + i);
      for (uint16_t loop = 0; loop < repeatTimes1; loop++) {
          MicroAPI::AddrReg srcOffset = MicroAPI::CreateAddrReg<float>(loop, vfLen);
          MicroAPI::AddrReg dstOffset = MicroAPI::CreateAddrReg<float>(i, cOnceNum, loop, vfLen);
          MicroAPI::DataCopy(regWeight, weightAddr, srcOffset);
          MicroAPI::Mul(regWeight, regWeight, regSmoothGrad, copyOutReg);
          MicroAPI::DataCopy(regOut, smGradAddr, dstOffset);
          MicroAPI::Add(regOutAdd, regWeight, regOut, copyOutReg);
          MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
          MicroAPI::DataCopy(smGradAddr, regOutAdd, dstOffset, copyOutReg);
      }
      for (uint16_t loopT = 0; loopT < tailLoopTimes; loopT++) {
          MicroAPI::DataCopy(regWeight, weightTailAddr);
          MicroAPI::Mul(regWeight, regWeight, regSmoothGrad, preg);
          MicroAPI::DataCopy(regOut, smGradTailAddr + i * cOnceNum);
          MicroAPI::Add(regOutAdd, regWeight, regOut, preg);
          MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
          MicroAPI::DataCopy(smGradTailAddr + i * cOnceNum, regOutAdd, preg);
      }
    }
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::ComputeLogSmoothLoss(uint64_t nLoopNum, uint64_t cOnceNumTail, uint64_t cOnceNumAlign, LocalTensor<float> &weightLocal, LocalTensor<float> &tmpLocal) {
  uint16_t nTimes = nLoopNum;
  uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
  uint16_t repeatTimes1 = cOnceNumTail / vfLen;
  uint32_t tailNum = cOnceNumTail % vfLen;
  uint32_t tailNumAlign = cOnceNumAlign - repeatTimes1 * vfLen;
  uint16_t cAlign = cOnceNumAlign;
  uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;

  auto smGradAddr = (__ubuf__ float *)tmpLocal.GetPhyAddr();
  auto weightAddr = (__ubuf__ float *)weightLocal.GetPhyAddr();
  auto smGradTailAddr = (__ubuf__ float *)tmpLocal.GetPhyAddr() + repeatTimes1 * vfLen;
  auto weightTailAddr = (__ubuf__ float *)weightLocal.GetPhyAddr() + repeatTimes1 * vfLen;
  auto smoothAddr = (__ubuf__ float *)this->smoothLocal.GetPhyAddr();
  __VEC_SCOPE__
  {
    MicroAPI::MaskReg copyOutReg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> regWeight;
    MicroAPI::RegTensor<float> regOut;
    MicroAPI::RegTensor<float> regSmoothGrad;
    MicroAPI::RegTensor<float> regOutAdd;
    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(tailNum);
    MicroAPI::MaskReg preg1 = MicroAPI::UpdateMask<float>(tailNumAlign);
    for (uint16_t i = 0; i < nTimes; i++) {
      MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(regSmoothGrad, smoothAddr + i);

      for(uint16_t loop = 0; loop < repeatTimes1; loop++) {
        MicroAPI::AddrReg srcOffset = MicroAPI::CreateAddrReg<float>(loop, vfLen);
        MicroAPI::AddrReg dstOffset = MicroAPI::CreateAddrReg<float>(i, cAlign, loop, vfLen);
        MicroAPI::DataCopy(regWeight, weightAddr, srcOffset);
        MicroAPI::Mul(regWeight, regWeight, regSmoothGrad, copyOutReg);
        MicroAPI::DataCopy(smGradAddr, regWeight, dstOffset, copyOutReg);
      }
      for(uint16_t loopT = 0; loopT < tailLoopTimes; loopT++) {
        MicroAPI::DataCopy(regWeight, weightTailAddr);
        MicroAPI::Mul(regWeight, regWeight, regSmoothGrad, preg);
        MicroAPI::DataCopy(smGradTailAddr + i * cAlign, regWeight, preg1);
      }
    }
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::LogSoftmaxWeightMainAddTailBlock(uint64_t nLoopNum) {
  LocalTensor<float> tmpLocal = fp32Buf7.Get<float>();
  for (uint64_t i = 0; i < this->kTimesTail; ++i) {
    uint64_t tailC = this->cOnceNum;
    uint64_t align = this->cOnceNum;
    if (i == this->kTimesTail - 1) {
      tailC = this->cOnceNumTail;
      align = this->cOnceNumTailAlign;
    }
    uint64_t offset = this->cOnceNum * i;

    LocalTensor<float> weightLocal = this->inQueWeight_.template AllocTensor<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = this->cOnceNum * sizeof(float);
    copyParams.srcStride = (this->colVal - this->cOnceNum) * sizeof(float);
    copyParams.dstStride = 0;

    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(weightLocal, this->weightGm[offset], copyParams, padParams);
    this->inQueWeight_.template EnQue(weightLocal);
    LocalTensor<float> weightUb = this->inQueWeight_.template DeQue<float>();
    ComputeLogSmoothLossMainBlock(nLoopNum, this->cOnceNum, this->cOnceNum, weightUb, tmpLocal);

    this->inQueWeight_.template FreeTensor(weightUb);

    uint64_t offsetTail = this->cOnceNum * (i + this->kTimes);
    LocalTensor<float> weightLocalTail = this->inQueWeight_.template AllocTensor<float>();
    DataCopyExtParams copyParamsT;
    copyParamsT.blockCount = 1;
    copyParamsT.blockLen = tailC * sizeof(float);
    copyParamsT.srcStride = (this->colVal - tailC) * sizeof(float);
    copyParamsT.dstStride = 0;

    DataCopyPad(weightLocalTail, this->weightGm[offsetTail], copyParamsT, padParams);
    this->inQueWeight_.template EnQue(weightLocalTail);
    LocalTensor<float> weightUbT = this->inQueWeight_.template DeQue<float>();
    ComputeLogSmoothLossTailBlock(nLoopNum, tailC, align, weightUbT, tmpLocal);

    this->inQueWeight_.template FreeTensor(weightUbT);

    uint32_t srcShape[2] = {static_cast<uint32_t>(nLoopNum), static_cast<uint32_t>(this->cOnceNum)};

    LocalTensor<float> sumUb = this->fp32Buf6.template Get<float>();
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumUb, tmpLocal, srcShape, false);
    UpdateCache(smoothSumLocal, sumUb, i, nLoopNum);
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::LogSoftmaxWeightAlignBlock(uint64_t nLoopNum) {
  LocalTensor<float> tmpLocal = fp32Buf7.Get<float>();
  for (uint64_t j = this->kTimesTail; j < this->kTimes; ++j) {
    LocalTensor<float> weightLocal = this->inQueWeight_.template AllocTensor<float>();
    uint64_t onceC = this->cOnceNum;
    uint64_t onceCAlign = this->cOnceNum;
    if ((j == this->kTimes - 1) && (this->kTimesTail == 0)) {
      onceC = this->cOnceNumTail;
      onceCAlign = this->cOnceNumTailAlign;
    }
    uint64_t offset = this->cOnceNum * j;

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = onceC * sizeof(float);
    copyParams.srcStride = (this->colVal - onceC) * sizeof(float);
    copyParams.dstStride = 0;

    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(weightLocal, this->weightGm[offset], copyParams, padParams);
    this->inQueWeight_.template EnQue(weightLocal);
    LocalTensor<float> weightUb = this->inQueWeight_.template DeQue<float>();

    ComputeLogSmoothLoss(nLoopNum, onceC, onceCAlign, weightUb, tmpLocal);
    this->inQueWeight_.template FreeTensor(weightUb);

    uint32_t srcShape[2] = {static_cast<uint32_t>(nLoopNum), static_cast<uint32_t>(onceCAlign)};
    LocalTensor<float> sumUb = this->fp32Buf6.template Get<float>();
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumUb, tmpLocal, srcShape, false);
    UpdateCache(smoothSumLocal, sumUb, j, nLoopNum);
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::LogSoftmaxWeightEachBatch(uint64_t nLoopNum) {
  LogSoftmaxWeightMainAddTailBlock(nLoopNum);
  LogSoftmaxWeightAlignBlock(nLoopNum);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::ComputerEachBatch(uint64_t nLoopIdx,
                                                                                    uint64_t peerLoopN,
                                                                                    uint64_t nLoopNum) {
    float smoothGradScalar = 0;
    float smoothGradReduceSum = 0;
    if (this->labelSmoothing != 0) {
        this->PipeV2S();
        smoothGradScalar = this->smoothLocal.GetValue(nLoopIdx);
        smoothGradReduceSum = smoothSumLocal.GetValue(nLoopIdx + this->cacheStart);
        this->PipeS2V();
    }
    // 一个核内的第n行
    for (uint64_t cLoopIdx = 0; cLoopIdx < this->colLoop; cLoopIdx++) {  // 循环一行内的每个loop
        this->CopyInLog(nLoopIdx + peerLoopN, cLoopIdx, this->alignColLoopNum);
        ComputeLog(nLoopIdx, peerLoopN, cLoopIdx, this->alignColLoopNum, smoothGradScalar, smoothGradReduceSum);
        this->CopyOutLog(nLoopIdx + peerLoopN, cLoopIdx, this->alignColLoopNum);
    }
    if (this->colLoopNumTail != 0) {
        this->CopyInLog(nLoopIdx + peerLoopN, this->colLoop, this->colLoopNumTail);
        ComputeLog(nLoopIdx, peerLoopN, this->colLoop, this->colLoopNumTail, smoothGradScalar, smoothGradReduceSum);
        this->CopyOutLog(nLoopIdx + peerLoopN, this->colLoop, this->colLoopNumTail);
    }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::Process() {
  if (this->reduction == REDUCTION_MEAN && this->nLoopTimes != 0) {
    this->TargetWeightReduceSum(targetWeightLocal);
  }
  for (uint64_t n = 0; n < this->nLoopTimes; n++) {
    ProcessPerCore(n * this->alignNLoopNum, this->alignNLoopNum);
  }
  if (this->nLoopTail != 0) {
    ProcessPerCore(this->nLoopTimes * this->alignNLoopNum, this->nLoopTail);
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::ProcessPerCore(uint64_t peerLoopNIdx, uint64_t nLoopNum) {
  this->TargetWeight(this->targetOffset + peerLoopNIdx, nLoopNum, targetWeightLocal);
  if (this->ignoreIndex >= 0) {
    this->ComputeIgnoreMask(nLoopNum, peerLoopNIdx);
  }
  if (this->reduction == REDUCTION_NONE) {
    if (this->labelSmoothing == 0) {
      this->GradReductionNone(nLoopNum, peerLoopNIdx);
      Mul(this->targetCast, targetWeightLocal, this->targetCast, nLoopNum);
    } else {
      this->GradReductionNoneSmooth(nLoopNum, peerLoopNIdx);
      Mul(this->targetCast, targetWeightLocal, this->targetCast, nLoopNum);
      LogSoftmaxWeightEachBatch(nLoopNum);
    }
  } else if (this->reduction == REDUCTION_MEAN) {
    if (this->nLoopTimes == 0) {
      TargetWeightSumTailN(nLoopNum, peerLoopNIdx);
      this->PipeV2S();
    }
    float reduceFactor = this->blockBuf2Local.GetValue(0);
    this->PipeS2V();
    if (this->labelSmoothing == 0) {
      GradReductionMeanSumWeight(reduceFactor, nLoopNum);
    } else {
      GradReductionMeanSumSmoothWeight(reduceFactor, nLoopNum);
      LogSoftmaxWeightEachBatch(nLoopNum);
    }
  } else if (this->reduction == REDUCTION_SUM) {
    if (this->labelSmoothing == 0) {
      GradReductionMeanSumWeight(1.0f, nLoopNum);
    } else {
      GradReductionMeanSumSmoothWeight(1.0f, nLoopNum);
      LogSoftmaxWeightEachBatch(nLoopNum);
    }
  }
  for (uint64_t nLoopIdx = 0; nLoopIdx < nLoopNum; nLoopIdx++) {
    ComputerEachBatch(nLoopIdx, peerLoopNIdx, nLoopNum);
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::GradReductionMeanSumWeight(float factor, uint64_t targetNum) {
  float meanSumOutGrad = 0.0f;
  if constexpr (std::is_same<T1, bfloat16_t>::value) {
    meanSumOutGrad = ToFloat(this->gradLossGm.GetValue(0)) * (1 - this->labelSmoothing);
  } else if constexpr (std::is_same<T1, half>::value) {
    meanSumOutGrad = static_cast<float>(this->gradLossGm.GetValue(0)) * (1 - this->labelSmoothing);
  } else if constexpr (std::is_same<T1, float>::value) {
    meanSumOutGrad = this->gradLossGm.GetValue(0) * (1 - this->labelSmoothing);
  }

  this->PipeS2V();
  uint32_t oneRepeat = platform::GetVRegSize() / sizeof(float);
  uint32_t totalLen = targetNum;
  uint32_t repeatTimes = CeilDivision(totalLen, oneRepeat);

  auto ignoreSelectAddr = (__ubuf__ float *)this->ignoreSelect.GetPhyAddr();
  auto targetCastAddr = (__ubuf__ float *)this->targetCast.GetPhyAddr();
  auto targetWeightAddr = (__ubuf__ float *)targetWeightLocal.GetPhyAddr();
  __VEC_SCOPE__
  {
    MicroAPI::MaskReg pregUp;
    MicroAPI::RegTensor<float> regGradLoss;
    MicroAPI::RegTensor<float> regIgnoreSelect;
    MicroAPI::RegTensor<float> regtargetWeight;
    MicroAPI::RegTensor<float> regFactor;
    MicroAPI::RegTensor<float> regSmoothGrad;
    for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
      pregUp = MicroAPI::UpdateMask<float>(totalLen);
      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regtargetWeight, targetWeightAddr, (int32_t)oneRepeat);
      MicroAPI::Duplicate(regGradLoss, meanSumOutGrad, pregUp);
      MicroAPI::Duplicate(regFactor, factor, pregUp);
      MicroAPI::Div(regGradLoss, regGradLoss, regFactor, pregUp);
      if (this->ignoreIndex >= 0) {
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regIgnoreSelect, ignoreSelectAddr, (int32_t)oneRepeat);

        MicroAPI::Mul(regGradLoss, regIgnoreSelect, regGradLoss, pregUp);
      }
        MicroAPI::Mul(regGradLoss, regGradLoss, regtargetWeight, pregUp);
      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(targetCastAddr, regGradLoss, (int32_t)oneRepeat, pregUp);
    }
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNotNone<T1, T2>::GradReductionMeanSumSmoothWeight(float factor, uint64_t targetNum) {
  float meanSumOutGrad = 0.0f;
  float meanSumSmoothGrad = 0.0f;
  if constexpr (std::is_same<T1, bfloat16_t>::value) {
    meanSumOutGrad = ToFloat(this->gradLossGm.GetValue(0)) * (1 - this->labelSmoothing);
    meanSumSmoothGrad = ToFloat(this->gradLossGm.GetValue(0));
  } else if constexpr (std::is_same<T1, half>::value) {
    meanSumOutGrad = static_cast<float>(this->gradLossGm.GetValue(0)) * (1 - this->labelSmoothing);
    meanSumSmoothGrad = static_cast<float>(this->gradLossGm.GetValue(0));
  } else if constexpr (std::is_same<T1, float>::value) {
    meanSumOutGrad = this->gradLossGm.GetValue(0) * (1 - this->labelSmoothing);
    meanSumSmoothGrad = this->gradLossGm.GetValue(0);
  }
  this->PipeS2V();
  uint32_t oneRepeat = platform::GetVRegSize() / sizeof(float);
  uint32_t totalLen = targetNum;
  float smFactor = this->smoothFactor;
  uint32_t repeatTimes = CeilDivision(totalLen, oneRepeat);

  auto ignoreSelectAddr = (__ubuf__ float *)this->ignoreSelect.GetPhyAddr();
  auto targetCastAddr = (__ubuf__ float *)this->targetCast.GetPhyAddr();
  auto targetWeightAddr = (__ubuf__ float *)targetWeightLocal.GetPhyAddr();
  auto smoothAddr = (__ubuf__ float *)this->smoothLocal.GetPhyAddr();
  __VEC_SCOPE__
  {
    MicroAPI::MaskReg pregUp;
    MicroAPI::RegTensor<float> regGradLoss;
    MicroAPI::RegTensor<float> regIgnoreSelect;
    MicroAPI::RegTensor<float> regtargetWeight;
    MicroAPI::RegTensor<float> regSmoothGrad;
    MicroAPI::RegTensor<float> regFactor;
    for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
      pregUp = MicroAPI::UpdateMask<float>(totalLen);
      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regtargetWeight, targetWeightAddr, (int32_t)oneRepeat);

      MicroAPI::Duplicate(regSmoothGrad, meanSumSmoothGrad, pregUp);
      MicroAPI::Duplicate(regFactor, factor, pregUp);
      MicroAPI::Duplicate(regGradLoss, meanSumOutGrad, pregUp);
      MicroAPI::Div(regGradLoss, regGradLoss, regFactor, pregUp);
      MicroAPI::Div(regSmoothGrad, regSmoothGrad, regFactor, pregUp);
      MicroAPI::Muls(regSmoothGrad, regSmoothGrad, smFactor, pregUp);
      if (this->ignoreIndex >= 0) {
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regIgnoreSelect, ignoreSelectAddr, (int32_t)oneRepeat);

        MicroAPI::Mul(regGradLoss, regIgnoreSelect, regGradLoss, pregUp);
        MicroAPI::Mul(regSmoothGrad, regIgnoreSelect, regSmoothGrad, pregUp);
      }
      MicroAPI::Mul(regGradLoss, regtargetWeight, regGradLoss, pregUp);
      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(targetCastAddr, regGradLoss, (int32_t)oneRepeat, pregUp);
      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(smoothAddr, regSmoothGrad, (int32_t)oneRepeat, pregUp);
    }
  }
}

} // namespace CrossEntropyLossGradRegbase
#endif  // CROSS_ENTROPY_LOSS_GRAD_WEIGHT_NOT_NONE_H