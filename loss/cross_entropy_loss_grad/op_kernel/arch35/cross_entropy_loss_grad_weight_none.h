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
 * \file cross_entropy_loss_grad_weight_none.h
 * \brief
 */
#ifndef CROSS_ENTROPY_LOSS_GRAD_WEIGHT_NONE_H
#define CROSS_ENTROPY_LOSS_GRAD_WEIGHT_NONE_H

#include "cross_entropy_loss_grad_base.h"

namespace CrossEntropyLossGradRegbase {

template <typename T1, typename T2>
class CrossEntropyLossGradWeightNone: protected CrossEntropyLossGradBase<T1, T2> {
public:
  __aicore__ inline CrossEntropyLossGradWeightNone(){};
  __aicore__ inline void Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target, GM_ADDR weight, GM_ADDR grad_zloss,
                              GM_ADDR lse_for_zloss, GM_ADDR x_grad, GM_ADDR workspace,
                              const CrossEntropyLossGradRegbaseTilingData& tilingData);
  __aicore__ inline void Process();

protected:
  __aicore__ inline void ComputerEachBatch(uint64_t nLoopIdx, uint64_t peerLoopN, uint64_t nLoopNum);
  __aicore__ inline void ComputeLog(uint64_t nLoopIdx, uint64_t peerLoopN, uint64_t cLoopIdx, uint64_t calcLen);
  __aicore__ inline void SmoothLabel(uint64_t nLoopIdx, uint64_t calcLen, LocalTensor<T1>& logProbLocal);
  __aicore__ inline void ProcessPerCore(uint64_t peerLoopNIdx, uint64_t nLoopNum);

 private:

  TBuf<TPosition::VECCALC> fp32Buf4;

  LocalTensor<T1> xGradLocal;
  LocalTensor<float> fp32Buf4Local;
};

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNone<T1, T2>::Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
                                                           GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
                                                           GM_ADDR x_grad, GM_ADDR workspace,
                                                           const CrossEntropyLossGradRegbaseTilingData& tilingData) {
  CrossEntropyLossGradBase<T1, T2>::Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss,
                                    x_grad, workspace, tilingData);
  uint64_t bufferSize = this->alignColLoopNum > this->colLoopNumTail ? this->alignColLoopNum : AlignUp(this->colLoopNumTail, MASK);
  bufferSize = bufferSize > 256 ? bufferSize : 256;
  this->pipe.InitBuffer(fp32Buf4, bufferSize * sizeof(float));
  fp32Buf4Local = fp32Buf4.Get<float>();
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNone<T1, T2>::SmoothLabel(uint64_t nLoopIdx, uint64_t calcLen, LocalTensor<T1>& logProbLocal) {
  this->PipeV2S();
  float smoothGradScalar = this->smoothLocal.GetValue(nLoopIdx);

  this->PipeS2V();
  float smoothGradreduceSum = smoothGradScalar * this->colVal;
  uint32_t oneRepeat = platform::GetVRegSize() / sizeof(float);
  uint32_t totalLen = calcLen;
  uint32_t repeatTimes = CeilDivision(totalLen, oneRepeat);
  auto logProbAddr = (__ubuf__ T1 *)logProbLocal.GetPhyAddr();
  auto xGradTmpAddr = (__ubuf__ float *)fp32Buf4Local.GetPhyAddr();
  auto xGradAddr = (__ubuf__ T1 *)xGradLocal.GetPhyAddr();
  __VEC_SCOPE__ {
      MicroAPI::MaskReg pregUp;
      MicroAPI::MaskReg pregT;
      MicroAPI::RegTensor<float> regLogProb;
      MicroAPI::RegTensor<float> regXGrad;
      MicroAPI::RegTensor<float> regSmooth;
      MicroAPI::RegTensor<float> regSmoothGrad;
      for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
          pregUp = MicroAPI::UpdateMask<float>(totalLen);
          this->LoadRegTensor(regLogProb, logProbAddr, pregUp, (int32_t)oneRepeat);
          MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regXGrad, xGradTmpAddr, (int32_t)oneRepeat);
          MicroAPI::Exp(regLogProb, regLogProb, pregUp);
          MicroAPI::Muls(regSmooth, regLogProb, smoothGradreduceSum, pregUp);
          MicroAPI::Duplicate(regSmoothGrad, smoothGradScalar, pregUp);
          MicroAPI::Sub(regSmooth, regSmooth, regSmoothGrad, pregUp);
          MicroAPI::Add(regSmooth, regSmooth, regXGrad, pregUp);
          this->CopyOutRegTensor(xGradAddr, regSmooth, pregUp, (int32_t)oneRepeat);
      }
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNone<T1, T2>::ComputeLog(uint64_t nLoopIdx, uint64_t peerLoopN,
                                                                          uint64_t cLoopIdx, uint64_t calcLen) {
    LocalTensor<T1> logProbLocal = this->inQueLogProb_.template DeQue<T1>();
    this->ComputeLogPre(logProbLocal, fp32Buf4Local, nLoopIdx, peerLoopN, cLoopIdx, calcLen);
    xGradLocal = this->outQueXGrad_.template AllocTensor<T1>();
    if (this->labelSmoothing == 0) {
        if constexpr (IsSameType<T1, float>::value) {
          uint32_t peerLoopNum = B8_MAX_NUM * MASK;
          uint32_t repeatLoop = calcLen / peerLoopNum;
          uint32_t tailNum = calcLen % peerLoopNum;
          for (int32_t i = 0; i < repeatLoop; i++) {
            Copy(xGradLocal[i * peerLoopNum], fp32Buf4Local[i * peerLoopNum], MASK, B8_MAX_NUM, {1, 1, 8, 8});
          }
          if (tailNum != 0) {
            Copy(xGradLocal[repeatLoop * peerLoopNum], fp32Buf4Local[repeatLoop * peerLoopNum], MASK, (tailNum + MASK -1) / MASK, {1, 1, 8, 8});
          }
        } else {
            Cast(xGradLocal, fp32Buf4Local, RoundMode::CAST_RINT, calcLen);
        }
    } else {
        SmoothLabel(nLoopIdx, calcLen, logProbLocal);
    }
    this->inQueLogProb_.template FreeTensor(logProbLocal);
    this->outQueXGrad_.template EnQue<T1>(xGradLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNone<T1, T2>::ComputerEachBatch(uint64_t nLoopIdx, uint64_t peerLoopN,
                                                                                 uint64_t nLoopNum) {
    for (uint64_t cLoopIdx = 0; cLoopIdx < this->colLoop; cLoopIdx++) {
        this->CopyInLog(nLoopIdx + peerLoopN, cLoopIdx, this->alignColLoopNum);
        ComputeLog(nLoopIdx, peerLoopN, cLoopIdx, this->alignColLoopNum);
        this->CopyOutLog(nLoopIdx + peerLoopN, cLoopIdx, this->alignColLoopNum);
    }
    if (this->colLoopNumTail != 0) {
        this->CopyInLog(nLoopIdx + peerLoopN, this->colLoop, this->colLoopNumTail);
        ComputeLog(nLoopIdx, peerLoopN, this->colLoop, this->colLoopNumTail);
        this->CopyOutLog(nLoopIdx + peerLoopN, this->colLoop, this->colLoopNumTail);
    }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNone<T1, T2>::Process() {
  if (this->reduction == REDUCTION_MEAN && this->ignoreIndex >= 0 && this->nLoopTimes != 0) {
    this->WeightReduceSum();
  }
  for (uint64_t n = 0; n < this->nLoopTimes; n++) {
    ProcessPerCore(n * this->alignNLoopNum, this->alignNLoopNum);
  }
  if (this->nLoopTail != 0) {
    ProcessPerCore(this->nLoopTimes * this->alignNLoopNum, this->nLoopTail);
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradWeightNone<T1, T2>::ProcessPerCore(uint64_t peerLoopNIdx, uint64_t nLoopNum) {
  if (this->ignoreIndex >= 0) {
    this->ComputeIgnoreMask(nLoopNum, peerLoopNIdx);
  }
  if (this->reduction == REDUCTION_NONE) {
    if (this->labelSmoothing == 0) {
      this->GradReductionNone(nLoopNum, peerLoopNIdx);
    } else {
      this->GradReductionNoneSmooth(nLoopNum, peerLoopNIdx);
    }
  }
  else if (this->reduction == REDUCTION_MEAN) {
    float reduceFactor = this->rowVal;
    if (this->ignoreIndex >= 0) {
      if (this->nLoopTimes == 0){
        this->WeightReduceSumTailN(nLoopNum, peerLoopNIdx, this->ignoreSelect);
        this->PipeV2S();
      }
      reduceFactor = this->blockBuf2Local.GetValue(0);
      this->PipeS2V();
    }
    if (this->labelSmoothing == 0) {
      this->GradReductionMeanSum(reduceFactor, nLoopNum);
    } else {
      this->GradReductionMeanSumSmooth(reduceFactor, nLoopNum);
    }
  }
  else {
    if (this->labelSmoothing == 0) {
      this->GradReductionMeanSum(1.0f, nLoopNum);
    } else {
      this->GradReductionMeanSumSmooth(1.0f, nLoopNum);
    }
  }
  for (uint64_t nLoopIdx = 0; nLoopIdx < nLoopNum; nLoopIdx++) {
    ComputerEachBatch(nLoopIdx, peerLoopNIdx, nLoopNum);
  }
}
} // namespace CrossEntropyLossGradRegbase
#endif  // CROSS_ENTROPY_LOSS_GRAD_WEIGHT_NONE_H