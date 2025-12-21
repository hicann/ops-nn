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
 * \file cross_entropy_loss_grad_base.h
 * \brief
 */

#ifndef CROSS_ENTROPY_LOSS_GRAD_BASE_H
#define CROSS_ENTROPY_LOSS_GRAD_BASE_H
#include "kernel_operator.h"
#include "../inc/platform.h"

namespace CrossEntropyLossGradRegbase {
using namespace AscendC;
using AscendC::MicroAPI::RegTensor;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BYTE_ONE_BLOCK = 32;
constexpr int32_t ELEMENT_ONE_BLOCK = 8;
constexpr int32_t MASK = 64;
constexpr int32_t B8_MAX_NUM = 255;
constexpr int32_t REDUCTION_NONE = 0;
constexpr int32_t REDUCTION_MEAN = 1;
constexpr int32_t REDUCTION_SUM = 2;

template <typename T1, typename T2>
class CrossEntropyLossGradBase {
public:
    __aicore__ inline CrossEntropyLossGradBase(){};
    __aicore__ inline void Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target, GM_ADDR weight, GM_ADDR grad_zloss,
                                GM_ADDR lse_for_zloss, GM_ADDR x_grad, GM_ADDR workspace,
                                const CrossEntropyLossGradRegbaseTilingData& tilingData);
    __aicore__ inline void InitData(const CrossEntropyLossGradRegbaseTilingData& tilingData);
    __aicore__ inline void InitUB();
    __aicore__ inline void CopyInLog(uint64_t nLoopIdx, uint64_t cLoopIdx, uint64_t calcLen);
    __aicore__ inline void CopyOutLog(uint64_t nLoopIdx, uint64_t cLoopIdx, uint64_t calcLen);
    __aicore__ inline void CopyInGradLoss(uint64_t targetNum, uint64_t peerLoopNIdx);
    __aicore__ inline void ComputeIgnoreMask(uint64_t targetNum, uint64_t peerLoopNIdx);
    __aicore__ inline void GradReductionNone(uint64_t targetNum, uint64_t peerLoopNIdx);
    __aicore__ inline void GradReductionNoneSmooth(uint64_t targetNum, uint64_t peerLoopNIdx);
    __aicore__ inline void WeightReduceSum();
    __aicore__ inline void WeightReduceSumTailN(uint64_t targetNum, uint64_t peerLoopNIdx, LocalTensor<float>& weightYn);
    __aicore__ inline void TargetWeightReduceSum(LocalTensor<float> &targetWeightLocal);
    __aicore__ inline void CopyOutWsAndReduce();
    __aicore__ inline void TargetWeight(uint64_t targetOffset, uint64_t targetNum, LocalTensor<float> &targetWeightLocal);
    __aicore__ inline void LoadRegTensor(MicroAPI::RegTensor<float> &dst, __local_mem__ T1 *&input, MicroAPI::MaskReg &pregUp, int32_t oneRepeat);
    __aicore__ inline void CopyOutRegTensor(__local_mem__ T1 *&output, MicroAPI::RegTensor<float> &src, MicroAPI::MaskReg &pregUp, int32_t oneRepeat);
    __aicore__ inline void GradReductionMeanSum(float factor, uint64_t targetNum);
    __aicore__ inline void GradReductionMeanSumSmooth(float factor, uint64_t targetNum);
    __aicore__ inline void CopyInWeight(uint64_t cLoopIdx, uint64_t calcLen);
    __aicore__ inline void ComputeLogPre(LocalTensor<T1> &logProbLocal, LocalTensor<float> &fp32Buf4Local,
                                         uint64_t nLoopIdx, uint64_t peerLoopN, uint64_t cLoopIdx, uint64_t calcLen);
    __aicore__ inline void PipeM2V();
    __aicore__ inline void PipeV2M();
    __aicore__ inline void PipeS2V();
    __aicore__ inline void PipeV2S();
    __aicore__ inline void PipeS2MTE3();

protected:
  TPipe pipe;
  GlobalTensor<T1> gradLossGm;
  GlobalTensor<T1> logProbGm;
  GlobalTensor<float> weightGm;
  GlobalTensor<T2> targetGm;
  GlobalTensor<T1> xGradGm;
  GlobalTensor<float> workspaceGm;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueGradLoss_;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueLogProb_;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueWeight_;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueTarget_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueXGrad_;

  TBuf<TPosition::VECCALC> ignoreSelectBuf_;
  TBuf<TPosition::VECCALC> targetCastBuf_;
  TBuf<TPosition::VECCALC> smoothBuf_;
  TBuf<TPosition::VECCALC> blockBuf2;
  TBuf<TPosition::VECCALC> blockBuf3;
  TBuf<TPosition::VECCALC> weightMask;

  LocalTensor<float> ignoreSelect;
  LocalTensor<float> targetCast;
  LocalTensor<float> smoothLocal;
  LocalTensor<float> blockBuf2Local;
  LocalTensor<float> blockBuf3Local;
  LocalTensor<float> weightMaskLocal;

  // tilingdata
  uint64_t rowVal;
  uint64_t colVal;
  uint64_t reduction;
  int64_t ignoreIndex;
  float labelSmoothing;
  uint64_t frontCoreNum;
  uint64_t usedCoreNum;
  uint64_t tailCoreNum;
  uint64_t tailRowNum;
  uint64_t frontRowNum;
  uint64_t alignColLoopNum;
  uint64_t alignNCNum;
  uint64_t colLoopNumTail;
  uint64_t colLoop;
  uint64_t targetCastSize;
  uint64_t targetSize;
  uint64_t smoothSize;
  uint64_t smoothSumSize;
  uint64_t gradLossSize;
  uint64_t ignoreSize;
  uint64_t reduceSize;
  uint64_t maskSize;
  uint64_t targetWeightSize;
  uint64_t smoothWeightSize;
  uint64_t tBuf2Size;
  uint64_t tBuf3Size;
  uint64_t alignNLoopNum;

  // init tmp data
  uint32_t coreIndex;
  uint64_t logOffset;
  uint64_t targetOffset;
  uint64_t nPeerCoreNum;
  float meanSumOutGrad;
  float meanSumSmoothGrad;
  float smoothFactor;
  float labelSmFactor;

  uint64_t nLoopTimes;
  uint64_t nLoopTail;
  // 二分
  uint64_t kTimes;
  uint64_t cOnceNum;
  uint64_t cOnceNumTail;
  uint64_t kTimesTail;
  uint64_t cOnceNumTailAlign;
  uint64_t cacheStart;

  constexpr static MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN};  // bf16/fp16 --float

  constexpr static MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT};  // float---bf16/fp16
};

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::InitData(const CrossEntropyLossGradRegbaseTilingData& tiling) {
  rowVal = tiling.rowVal;
  colVal = tiling.colVal;
  reduction = tiling.reduction;
  ignoreIndex = tiling.ignoreIndex;
  labelSmoothing = tiling.labelSmoothing;
  usedCoreNum = tiling.usedCoreNum;
  frontCoreNum = tiling.frontCoreNum;
  tailCoreNum = tiling.tailCoreNum;
  frontRowNum = tiling.frontRowNum;
  tailRowNum = tiling.tailRowNum;
  colLoop = tiling.colLoop;
  colLoopNumTail = tiling.colLoopNumTail;
  alignColLoopNum = tiling.alignColLoopNum;
  alignNCNum = tiling.alignNCNum;
  smoothSize = tiling.smoothSize;
  gradLossSize = tiling.gradLossSize;
  targetSize = tiling.targetSize;
  targetCastSize = tiling.targetCastSize;
  targetWeightSize = tiling.targetWeightSize;
  ignoreSize = tiling.ignoreSize;
  tBuf2Size = tiling.tBuf2Size;
  tBuf3Size = tiling.tBuf3Size;
  kTimes = tiling.kTimes;
  cOnceNum = tiling.cOnceNum;
  cOnceNumTail = tiling.cOnceNumTail;
  kTimesTail = tiling.kTimesTail;
  cOnceNumTailAlign = tiling.cOnceNumTailAlign;
  cacheStart = tiling.cacheStart;

  alignNLoopNum = platform::GetVRegSize() / sizeof(float);

  smoothFactor = static_cast<float>(labelSmoothing) / static_cast<float>(colVal);
  labelSmFactor = static_cast<float>(1.0) - this->labelSmoothing;
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
                                                         GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
                                                         GM_ADDR x_grad, GM_ADDR workspace,
                                                         const CrossEntropyLossGradRegbaseTilingData& tilingData) {
  InitData(tilingData);
  coreIndex = GetBlockIdx();
  InitUB();

  if (coreIndex < frontCoreNum) {
    logOffset = coreIndex * colVal * frontRowNum;
    targetOffset = coreIndex * frontRowNum;
    nPeerCoreNum = frontRowNum;
  } else {
    logOffset = frontCoreNum * colVal * frontRowNum + (coreIndex - frontCoreNum) * colVal * tailRowNum;
    targetOffset = frontCoreNum * frontRowNum + (coreIndex - frontCoreNum) * tailRowNum;
    nPeerCoreNum = tailRowNum;   // 确定该核需要处理多少行，target一次处理多少个数
  }


  nLoopTimes = this->nPeerCoreNum / this->alignNLoopNum;
  nLoopTail = this->nPeerCoreNum % this->alignNLoopNum;
  gradLossGm.SetGlobalBuffer((__gm__ T1*)grad_loss);
  logProbGm.SetGlobalBuffer((__gm__ T1*)log_prob + logOffset);
  targetGm.SetGlobalBuffer((__gm__ T2*)target);
  weightGm.SetGlobalBuffer((__gm__ float*)weight);
  xGradGm.SetGlobalBuffer((__gm__ T1*)x_grad + logOffset);
  workspaceGm.SetGlobalBuffer((__gm__ float*)workspace);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::InitUB() {
  uint32_t inOutQueInputSize = MASK;
  inOutQueInputSize = this->alignNCNum > this->colLoopNumTail ?
                      this->alignNCNum : AlignUp(this->colLoopNumTail, MASK);
  this->pipe.InitBuffer(inQueLogProb_, BUFFER_NUM, inOutQueInputSize * sizeof(T1)); // c
  this->pipe.InitBuffer(outQueXGrad_, BUFFER_NUM, inOutQueInputSize * sizeof(T1));  // c
  this->pipe.InitBuffer(inQueTarget_, BUFFER_NUM, this->targetSize);
  if (this->reduction == REDUCTION_NONE) {
    this->pipe.InitBuffer(inQueGradLoss_, BUFFER_NUM, this->gradLossSize);
  }
  this->pipe.InitBuffer(ignoreSelectBuf_, this->ignoreSize);
  this->pipe.InitBuffer(targetCastBuf_, this->targetCastSize);
  if (this->labelSmoothing != 0) {
    this->pipe.InitBuffer(smoothBuf_, this->smoothSize);
    smoothLocal = smoothBuf_.Get<float>();
  }
  if (this->reduction == REDUCTION_MEAN) {
    this->pipe.InitBuffer(blockBuf2, this->tBuf2Size);
    this->pipe.InitBuffer(blockBuf3, this->tBuf2Size);
    this->pipe.InitBuffer(weightMask, this->tBuf3Size);
    blockBuf2Local = blockBuf2.Get<float>();
    blockBuf3Local = blockBuf3.Get<float>();
    weightMaskLocal = weightMask.Get<float>();
  }
  ignoreSelect = ignoreSelectBuf_.Get<float>();
  targetCast = targetCastBuf_.Get<float>();
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::CopyInLog(uint64_t nLoopIdx, uint64_t cLoopIdx, uint64_t calcLen) {
  LocalTensor<T1> logProbLocal = inQueLogProb_.AllocTensor<T1>();
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(calcLen * sizeof(T1)), 0, 0, 0};
  DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
  DataCopyPad(logProbLocal, this->logProbGm[nLoopIdx * this->colVal + cLoopIdx * this->alignColLoopNum],
              copyParams, padParams);
  inQueLogProb_.EnQue(logProbLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::CopyOutLog(uint64_t nLoopIdx, uint64_t cLoopIdx, uint64_t calcLen) {
  LocalTensor<T1> xGradLocal = outQueXGrad_.DeQue<T1>();
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(calcLen * sizeof(T1)), 0, 0, 0};
  DataCopyPad(this->xGradGm[nLoopIdx * this->colVal + cLoopIdx * this->alignColLoopNum], xGradLocal, copyParams);
  outQueXGrad_.FreeTensor(xGradLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::CopyOutWsAndReduce() {
  DataCopyPad(this->workspaceGm[this->coreIndex], this->blockBuf2Local, {1, static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0});
  SyncAll();
  DataCopyPad(this->weightMaskLocal, this->workspaceGm, {1, static_cast<uint32_t>(this->usedCoreNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
  this->PipeM2V();
  uint32_t srcShapeWs[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(this->usedCoreNum)};
  ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf2Local, this->weightMaskLocal, srcShapeWs, false);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::TargetWeightReduceSum(LocalTensor<float>& targetWeightLocal) {
  Duplicate(this->blockBuf2Local, 0.0f, ELEMENT_ONE_BLOCK);
  for (uint64_t n = 0; n < this->nLoopTimes; n++) {
    uint64_t nIdx = n * this->alignNLoopNum;
    this->TargetWeight(this->targetOffset + nIdx, this->alignNLoopNum, targetWeightLocal);
    uint32_t srcShape[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(this->alignNLoopNum)};
    if (this->ignoreIndex >= 0) {
      this->ComputeIgnoreMask(this->alignNLoopNum, nIdx);
      Mul(this->weightMaskLocal, this->ignoreSelect, targetWeightLocal, this->alignNLoopNum);
      ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf3Local, this->weightMaskLocal, srcShape, false);
    } else {
      ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf3Local, targetWeightLocal, srcShape, false);
    }
      Add(this->blockBuf2Local[0], this->blockBuf2Local[0], this->blockBuf3Local[0], 1);
  }
  if (this->nLoopTail != 0) {
    uint64_t nIdx = this->nLoopTimes * this->alignNLoopNum;
    this->TargetWeight(this->targetOffset + nIdx, this->nLoopTail, targetWeightLocal);
    uint32_t srcShape[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(this->nLoopTail)};
    if (this->ignoreIndex >= 0) {
      this->ComputeIgnoreMask(this->nLoopTail, nIdx);
      Mul(this->weightMaskLocal, this->ignoreSelect, targetWeightLocal, this->nLoopTail);
      ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf3Local, this->weightMaskLocal, srcShape, false);
    } else {
      ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf3Local, targetWeightLocal, srcShape, false);
    }
      Add(this->blockBuf2Local[0], this->blockBuf2Local[0], this->blockBuf3Local[0], 1);
  }
  this->PipeV2M();
  CopyOutWsAndReduce();
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::WeightReduceSum() {
  Duplicate(this->blockBuf2Local, 0.0f, ELEMENT_ONE_BLOCK);
  for (uint64_t n = 0; n < this->nLoopTimes; n++) {
    uint64_t nIdx = n * this->alignNLoopNum;
    this->ComputeIgnoreMask(this->alignNLoopNum, nIdx);
    uint32_t srcShape[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(this->alignNLoopNum)};
    ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf3Local, this->ignoreSelect, srcShape, false);
    Add(this->blockBuf2Local[0], this->blockBuf2Local[0], this->blockBuf3Local[0], 1);
  }
  if (this->nLoopTail != 0) {
    uint64_t nIdx = this->nLoopTimes * this->alignNLoopNum;
    uint32_t srcShape[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(this->nLoopTail)};
    this->ComputeIgnoreMask(this->nLoopTail, nIdx);
    ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf3Local, this->ignoreSelect, srcShape, false);
    Add(this->blockBuf2Local[0], this->blockBuf2Local[0], this->blockBuf3Local[0], 1);
  }
  this->PipeV2M();
  CopyOutWsAndReduce();
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::TargetWeight(uint64_t targetOffset, uint64_t targetNum, LocalTensor<float>& targetWeightLocal) {
  for (uint64_t targetIdx = 0; targetIdx < targetNum; targetIdx++) {
    uint64_t targetValue = this->targetGm.GetValue(targetOffset + targetIdx);
    assert((0 <= targetValue && targetValue < this->colVal), "target data %lu out of range[%d %d)!\n", targetValue, 0, this->colVal);
    targetWeightLocal.SetValue(targetIdx, this->weightGm.GetValue(targetValue));
  }
  this->PipeS2V();
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::CopyInGradLoss(uint64_t targetNum, uint64_t peerLoopNIdx) {
  LocalTensor<T1> gradLossLocal = inQueGradLoss_.AllocTensor<T1>();
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(targetNum * sizeof(T1)), 0, 0, 0};
  DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
  DataCopyPad(gradLossLocal, this->gradLossGm[this->targetOffset + peerLoopNIdx], copyParams, padParams);
  inQueGradLoss_.EnQue(gradLossLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::ComputeIgnoreMask(uint64_t targetNum, uint64_t peerLoopNIdx) {
  LocalTensor<T2> targetLocal = inQueTarget_.AllocTensor<T2>();
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(targetNum * sizeof(T2)), 0, 0, 0};
  DataCopyPadExtParams<T2> padParams{false, 0, 0, 0};
  DataCopyPad(targetLocal, this->targetGm[this->targetOffset + peerLoopNIdx], copyParams, padParams);
  inQueTarget_.EnQue(targetLocal);
  targetLocal = inQueTarget_.DeQue<T2>();
  uint32_t batchNum = targetNum;
  uint32_t vfLen = platform::GetVRegSize() / sizeof(T2);
  uint16_t repeatTimes = CeilDivision(batchNum, vfLen);
  auto targetUbAddr = (__ubuf__ T2 *)targetLocal.GetPhyAddr();
  auto ignoreMaskUbAddr = (__ubuf__ float *)ignoreSelect.GetPhyAddr();
  T2 ignore = ignoreIndex;
  __VEC_SCOPE__
  {
      MicroAPI::MaskReg preg;
      MicroAPI::MaskReg preg1;
      MicroAPI::MaskReg dstMask;
      MicroAPI::RegTensor<T2> srcReg;
      MicroAPI::RegTensor<float> dstReg1;
      MicroAPI::RegTensor<float> srcReg0;
      MicroAPI::RegTensor<float> dstReg0;
      MicroAPI::RegTensor<float> dstReg;
      MicroAPI::MaskReg regAllFp32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
      MicroAPI::Duplicate(dstReg1, 1.0f, regAllFp32);
      MicroAPI::Duplicate(dstReg0, 0.0f, regAllFp32);
      for (uint16_t loop = 0; loop < repeatTimes; loop++) {
          preg = MicroAPI::UpdateMask<T2>(batchNum);
          MicroAPI::AddrReg srcOffset = MicroAPI::CreateAddrReg<T2>(loop, vfLen);
          MicroAPI::AddrReg addrReg = MicroAPI::CreateAddrReg<float>(loop, vfLen);
          MicroAPI::DataCopy(srcReg, targetUbAddr, srcOffset);
          MicroAPI::CompareScalar<T2, CMPMODE::EQ>(dstMask, srcReg, ignore, preg);
          if constexpr (sizeof(T2) == 8) {
            MicroAPI::MaskPack(preg1, dstMask);
            MicroAPI::Select(dstReg, dstReg0, dstReg1, preg1);
          } else {
            MicroAPI::Select(dstReg, dstReg0, dstReg1, dstMask);
          }
          MicroAPI::DataCopy(ignoreMaskUbAddr, dstReg, addrReg, regAllFp32);
      }
  }
  inQueTarget_.FreeTensor(targetLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::LoadRegTensor(MicroAPI::RegTensor<float> &dst, __local_mem__ T1 *&input, MicroAPI::MaskReg &pregUp, int32_t oneRepeat) {
  MicroAPI::RegTensor<T1> regTmp;
  MicroAPI::RegTensor<T1> regCopyIn;
  if constexpr (!IsSameType<T1, float>::value) {
    MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regCopyIn, input, (int32_t)oneRepeat);
    MicroAPI::UnPack((MicroAPI::RegTensor<int32_t> &)regTmp, (MicroAPI::RegTensor<int16_t> &)regCopyIn);
    MicroAPI::Cast<float, T1, castTrait0>(dst, regTmp, pregUp);
  } else {
    MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, input, (int32_t)oneRepeat);
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::CopyOutRegTensor(__local_mem__ T1 *&output, MicroAPI::RegTensor<float> &src, MicroAPI::MaskReg &pregUp, int32_t oneRepeat) {
  MicroAPI::RegTensor<T1> regTmp;
  MicroAPI::RegTensor<T1> regCopyOut;
  MicroAPI::MaskReg pregT;

  if constexpr (!IsSameType<T1, float>::value) {
      MicroAPI::Cast<T1, float, castTrait1>(regTmp, src, pregUp);
      MicroAPI::Pack((MicroAPI::RegTensor<uint16_t> &)regCopyOut, (MicroAPI::RegTensor<uint32_t> &)regTmp);
      MicroAPI::MaskPack(pregT, pregUp);
      MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(output, regCopyOut, (int32_t)oneRepeat, pregT);
  } else {
      MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(output, src, (int32_t)oneRepeat, pregUp);
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::WeightReduceSumTailN(uint64_t targetNum, uint64_t peerLoopNIdx, LocalTensor<float>& weightYn) {
  if (peerLoopNIdx == 0) {
    this->blockBuf2Local.SetValue(0, 0);
    this->PipeS2V();
  }
  if (targetNum != 0) {
    uint32_t srcShape[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(targetNum)};
    ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf2Local, weightYn, srcShape, false);
  }
  if (peerLoopNIdx != 0) {
    float wsValue = workspaceGm.GetValue(this->coreIndex);
    this->PipeS2V();
    Adds(this->blockBuf2Local, this->blockBuf2Local, wsValue, MASK);
  }
  this->PipeV2M();
  DataCopyPad(workspaceGm[this->coreIndex], this->blockBuf2Local, {1, (uint32_t)(1 * sizeof(float)), 0, 0, 0});
  #ifndef __CCE_KT_TEST__
    SyncAll();
  #endif
  DataCopyPad(this->weightMaskLocal, workspaceGm, {1, (uint32_t)(this->usedCoreNum * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
  this->PipeM2V();
  uint32_t srcShapeWs[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(this->usedCoreNum)};
  ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf2Local, this->weightMaskLocal, srcShapeWs, false);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::ComputeLogPre(LocalTensor<T1> &logProbLocal,
                                                                       LocalTensor<float> &fp32Buf4Local,
                                                                       uint64_t nLoopIdx, uint64_t peerLoopN,
                                                                       uint64_t cLoopIdx, uint64_t calcLen) {
    uint64_t cloopOffset = cLoopIdx * this->alignColLoopNum;
    uint64_t targetValue = this->targetGm.GetValue(this->targetOffset + nLoopIdx + peerLoopN);
    assert((0 <= targetValue && targetValue < this->colVal), "target data %lu out of range[%d %d)!\n", targetValue, 0,
           this->colVal);
    uint64_t posIdx = targetValue - cLoopIdx * this->alignColLoopNum;
    this->PipeV2S();
    float nllLossGradScalar = this->targetCast.GetValue(nLoopIdx);
    this->PipeS2V();
    uint32_t oneRepeat = platform::GetVRegSize() / sizeof(float);
    uint32_t totalLen = calcLen;
    uint32_t repeatTimes = CeilDivision(totalLen, oneRepeat);
    auto logProbAddr = (__ubuf__ T1 *)logProbLocal.GetPhyAddr();
    auto xGradTmpAddr = (__ubuf__ float *)fp32Buf4Local.GetPhyAddr();
    __VEC_SCOPE__ {
        MicroAPI::MaskReg pregUp;
        MicroAPI::MaskReg pregT;
        MicroAPI::RegTensor<float> regLogProb;
        MicroAPI::RegTensor<float> reglogGradLoss;
        MicroAPI::RegTensor<float> regNllLoss;
        for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
            pregUp = MicroAPI::UpdateMask<float>(totalLen);
            this->LoadRegTensor(regLogProb, logProbAddr, pregUp, (int32_t)oneRepeat);
            MicroAPI::Exp(regLogProb, regLogProb, pregUp);
            MicroAPI::Muls(regLogProb, regLogProb, nllLossGradScalar, pregUp);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(xGradTmpAddr, regLogProb,
                                                                               (int32_t)oneRepeat, pregUp);
        }
    }

    if (cloopOffset <= targetValue && targetValue <= cloopOffset + calcLen) {
        this->PipeV2S();
        fp32Buf4Local.SetValue(posIdx, fp32Buf4Local.GetValue(posIdx) - nllLossGradScalar);
        this->PipeS2V();
    }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::GradReductionNone(uint64_t targetNum, uint64_t peerLoopNIdx) {
  CopyInGradLoss(targetNum, peerLoopNIdx);
  LocalTensor<T1> gradLossLocal = inQueGradLoss_.DeQue<T1>();

  uint32_t totalLen = targetNum;
  uint32_t oneRepeat = platform::GetVRegSize() / sizeof(float);
  uint32_t repeatTimes = CeilDivision(totalLen, oneRepeat);

  auto gradLossAddr = (__ubuf__ T1 *)gradLossLocal.GetPhyAddr();
  auto ignoreSelectAddr = (__ubuf__ float *)ignoreSelect.GetPhyAddr();
  auto targetCastAddr = (__ubuf__ float *)targetCast.GetPhyAddr();
  __VEC_SCOPE__
  {
    MicroAPI::MaskReg pregUp;
    MicroAPI::RegTensor<float> regGradLoss;
    MicroAPI::RegTensor<float> regIgnoreSelect;
    for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
      pregUp = MicroAPI::UpdateMask<float>(totalLen);
      LoadRegTensor(regGradLoss, gradLossAddr, pregUp, (int32_t)oneRepeat);

      MicroAPI::Muls(regGradLoss, regGradLoss, 1 - this->labelSmoothing, pregUp);
      if (this->ignoreIndex >= 0) {
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regIgnoreSelect, ignoreSelectAddr, (int32_t)oneRepeat);
        MicroAPI::Mul(regGradLoss, regGradLoss, regIgnoreSelect, pregUp);
      }
      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(targetCastAddr, regGradLoss, (int32_t)oneRepeat, pregUp);
    }
  }
  inQueGradLoss_.FreeTensor(gradLossLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::GradReductionNoneSmooth(uint64_t targetNum, uint64_t peerLoopNIdx) {
  CopyInGradLoss(targetNum, peerLoopNIdx);
  LocalTensor<T1> gradLossLocal = inQueGradLoss_.DeQue<T1>();

  uint32_t oneRepeat = platform::GetVRegSize() / sizeof(float);
  uint32_t totalLen = targetNum;
  uint32_t repeatTimes = CeilDivision(totalLen, oneRepeat);

  auto smoothAddr = (__ubuf__ float *)smoothLocal.GetPhyAddr();
  auto gradLossAddr = (__ubuf__ T1 *)gradLossLocal.GetPhyAddr();
  auto ignoreSelectAddr = (__ubuf__ float *)ignoreSelect.GetPhyAddr();
  auto targetCastAddr = (__ubuf__ float *)targetCast.GetPhyAddr();
  __VEC_SCOPE__
  {
    MicroAPI::MaskReg pregUp;
    MicroAPI::RegTensor<float> regGradLoss;
    MicroAPI::RegTensor<float> regIgnoreSelect;
    MicroAPI::RegTensor<float> regSmoothGrad;
    for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
      pregUp = MicroAPI::UpdateMask<float>(totalLen);
      LoadRegTensor(regGradLoss, gradLossAddr, pregUp, (int32_t)oneRepeat);

      MicroAPI::Muls(regSmoothGrad, regGradLoss, this->smoothFactor, pregUp);
      MicroAPI::Muls(regGradLoss, regGradLoss, 1 - this->labelSmoothing, pregUp);
      if (this->ignoreIndex >= 0) {
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regIgnoreSelect, ignoreSelectAddr, (int32_t)oneRepeat);

        MicroAPI::Mul(regSmoothGrad, regSmoothGrad, regIgnoreSelect, pregUp);

        MicroAPI::Mul(regGradLoss, regGradLoss, regIgnoreSelect, pregUp);
      }
      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(targetCastAddr, regGradLoss, (int32_t)oneRepeat, pregUp);
      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(smoothAddr, regSmoothGrad, (int32_t)oneRepeat, pregUp);
    }
  }
  inQueGradLoss_.FreeTensor(gradLossLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::CopyInWeight(uint64_t cLoopIdx, uint64_t calcLen) {
  LocalTensor<float> weightLocal = this->inQueWeight_.template AllocTensor<float>();
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(calcLen * sizeof(float)), 0, 0, 0};
  DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
  DataCopyPad(weightLocal, this->weightGm[cLoopIdx * this->alignColLoopNum], copyParams, padParams);
  this->inQueWeight_.template EnQue(weightLocal);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::GradReductionMeanSum(float factor, uint64_t targetNum) {

  if constexpr (std::is_same<T1, bfloat16_t>::value) {
    meanSumOutGrad = ToFloat(this->gradLossGm.GetValue(0)) * (1 - labelSmoothing);
  } else if constexpr (std::is_same<T1, half>::value) {
    meanSumOutGrad = static_cast<float>(this->gradLossGm.GetValue(0)) * (1 - labelSmoothing);
  } else if constexpr (std::is_same<T1, float>::value) {
    meanSumOutGrad = this->gradLossGm.GetValue(0) * (1 - labelSmoothing);
  }

  this->PipeS2V();
  uint32_t oneRepeat = platform::GetVRegSize() / sizeof(float);
  uint32_t totalLen = targetNum;
  uint32_t repeatTimes = CeilDivision(totalLen, oneRepeat);

  auto ignoreSelectAddr = (__ubuf__ float *)ignoreSelect.GetPhyAddr();
  auto targetCastAddr = (__ubuf__ float *)targetCast.GetPhyAddr();
  __VEC_SCOPE__
  {
    MicroAPI::MaskReg pregUp;
    MicroAPI::RegTensor<float> regIgnoreSelect;
    MicroAPI::RegTensor<float> regSmoothGrad;
    MicroAPI::RegTensor<float> regFactor;
    MicroAPI::RegTensor<float> regMeanSumOutGrad;
    for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
      pregUp = MicroAPI::UpdateMask<float>(totalLen);
      MicroAPI::Duplicate(regFactor, factor, pregUp);
      MicroAPI::Duplicate(regMeanSumOutGrad, meanSumOutGrad, pregUp);
      MicroAPI::Div(regMeanSumOutGrad, regMeanSumOutGrad, regFactor, pregUp);
      if (this->ignoreIndex >= 0) {
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regIgnoreSelect, ignoreSelectAddr, (int32_t)oneRepeat);
        MicroAPI::Mul(regMeanSumOutGrad, regIgnoreSelect, regMeanSumOutGrad, pregUp);
      }
      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(targetCastAddr, regMeanSumOutGrad, (int32_t)oneRepeat, pregUp);
    }
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::GradReductionMeanSumSmooth(float factor, uint64_t targetNum) {

  if constexpr (std::is_same<T1, bfloat16_t>::value) {
    meanSumOutGrad = ToFloat(this->gradLossGm.GetValue(0)) * (1 - labelSmoothing);
    meanSumSmoothGrad = ToFloat(this->gradLossGm.GetValue(0));
  } else if constexpr (std::is_same<T1, half>::value) {
    meanSumOutGrad = static_cast<float>(this->gradLossGm.GetValue(0)) * (1 - labelSmoothing);
    meanSumSmoothGrad = static_cast<float>(this->gradLossGm.GetValue(0));
  } else if constexpr (std::is_same<T1, float>::value) {
    meanSumOutGrad = this->gradLossGm.GetValue(0) * (1 - labelSmoothing);
    meanSumSmoothGrad = this->gradLossGm.GetValue(0);
  }

  this->PipeS2V();
  uint32_t oneRepeat = platform::GetVRegSize() / sizeof(float);
  uint32_t totalLen = targetNum;
  float smFactor = this->smoothFactor;
  uint32_t repeatTimes = CeilDivision(totalLen, oneRepeat);

  auto ignoreSelectAddr = (__ubuf__ float *)ignoreSelect.GetPhyAddr();
  auto targetCastAddr = (__ubuf__ float *)targetCast.GetPhyAddr();
  auto smoothAddr = (__ubuf__ float *)smoothLocal.GetPhyAddr();
  __VEC_SCOPE__
  {
    MicroAPI::MaskReg pregUp;
    MicroAPI::RegTensor<float> regIgnoreSelect;
    MicroAPI::RegTensor<float> regSmoothGrad;
    MicroAPI::RegTensor<float> regFactor;
    MicroAPI::RegTensor<float> regMeanSumSmoothGrad;
    MicroAPI::RegTensor<float> regMeanSumOutGrad;
    for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
      pregUp = MicroAPI::UpdateMask<float>(totalLen);
      MicroAPI::Duplicate(regFactor, factor, pregUp);
      MicroAPI::Duplicate(regMeanSumOutGrad, meanSumOutGrad, pregUp);
      MicroAPI::Duplicate(regMeanSumSmoothGrad, meanSumSmoothGrad, pregUp);
      MicroAPI::Div(regMeanSumOutGrad, regMeanSumOutGrad, regFactor, pregUp);
      MicroAPI::Div(regMeanSumSmoothGrad, regMeanSumSmoothGrad, regFactor, pregUp);

      MicroAPI::Muls(regMeanSumSmoothGrad, regMeanSumSmoothGrad, smFactor, pregUp);
      if (this->ignoreIndex >= 0) {
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regIgnoreSelect, ignoreSelectAddr, (int32_t)oneRepeat);
        MicroAPI::Mul(regMeanSumOutGrad, regIgnoreSelect, regMeanSumOutGrad, pregUp);
        MicroAPI::Mul(regMeanSumSmoothGrad, regIgnoreSelect, regMeanSumSmoothGrad, pregUp);
      }

      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(targetCastAddr, regMeanSumOutGrad, (int32_t)oneRepeat, pregUp);
      MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(smoothAddr, regMeanSumSmoothGrad, (int32_t)oneRepeat, pregUp);
    }
  }
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::PipeM2V() {
    event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::PipeV2M() {
    event_t eventVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMTE3);
}


template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::PipeS2V() {
    event_t eventS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventS2V);
    WaitFlag<HardEvent::S_V>(eventS2V);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::PipeV2S() {
    event_t eventV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventV2S);
    WaitFlag<HardEvent::V_S>(eventV2S);
}

template <typename T1, typename T2>
__aicore__ inline void CrossEntropyLossGradBase<T1, T2>::PipeS2MTE3() {
  event_t eventSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
  SetFlag<HardEvent::S_MTE3>(eventSToMTE3);
  WaitFlag<HardEvent::S_MTE3>(eventSToMTE3);
}

} // namespace CrossEntropyLossGradRegbase
#endif  // CROSS_ENTROPY_LOSS_GRAD_BASE_H