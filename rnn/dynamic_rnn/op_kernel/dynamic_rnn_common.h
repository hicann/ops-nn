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
 * \file dynamic_rnn_common.h
 * \brief
 */
#ifndef _DYNAMIC_RNN_COMMON_H_
#define _DYNAMIC_RNN_COMMON_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

constexpr int64_t LSTM_GATE_SIZE = 4;
constexpr int64_t DEFAULT_QUEUE_BUFFE_SIZE = 2;

__aicore__ inline int Ceil(int a, int b) {
  if (b == 0) {
    return a;
  }
  return (a + b - 1) / b;
}


struct TRnnOffsets {
  int64_t AOffset;
  int64_t BOffset;
  int64_t COffset;
  int64_t BiasOffset;
};

struct CalcSize {
  int64_t oriBaseM;
  int64_t oriBaseN;
  int64_t tailBaseM;
  int64_t tailBaseN;
  int64_t mLoop;
  int64_t nLoop;
  int64_t hiddenMNSize;
  int64_t hiddenMKAllSize;  // mm2 左矩阵的大小
  int64_t oneBaseMTailN;
  int64_t oneLineBaseMBaseN;
  int64_t oneLineMN;
  int64_t allCellSize;
  int64_t hiddenMKSize;
  int64_t hiddenTailMKSize;
  int64_t oneTailMBaseN;
  int64_t oneTailMTailN;
  int64_t outSize;
};

// input GlobalTensors
template <typename T>
struct InputGm {
  AscendC::GlobalTensor<T> xGm;
  AscendC::GlobalTensor<T> weightGm;
  AscendC::GlobalTensor<T> biasGm;
  AscendC::GlobalTensor<T> seqLengthGm;
  AscendC::GlobalTensor<T> initHGm;
  AscendC::GlobalTensor<T> initCGm;
  AscendC::GlobalTensor<T> wciGm;
  AscendC::GlobalTensor<T> wcfGm;
  AscendC::GlobalTensor<T> wcoGm;
  AscendC::GlobalTensor<T> maskGm;
};

template <typename T>
struct OutputGm {
  __aicore__ inline OutputGm() = default;
  AscendC::GlobalTensor<T> outYGm;
  AscendC::GlobalTensor<T> outHGm;
  AscendC::GlobalTensor<T> outCGm;
  AscendC::GlobalTensor<T> outIGm;
  AscendC::GlobalTensor<T> outJGm;
  AscendC::GlobalTensor<T> outFGm;
  AscendC::GlobalTensor<T> outOGm;
  AscendC::GlobalTensor<T> outTanhCGm;
  AscendC::GlobalTensor<float> workspace;
};

struct LstmBean {
  GM_ADDR inputX;
  GM_ADDR weight;
  GM_ADDR bias;
  GM_ADDR seqLength;
  GM_ADDR initH;
  GM_ADDR initC;
  GM_ADDR wCi;
  GM_ADDR wCf;
  GM_ADDR wCo;
  GM_ADDR mask;
  GM_ADDR outputY;
  GM_ADDR outputH;
  GM_ADDR outputC;
  GM_ADDR outputI;
  GM_ADDR outputJ;
  GM_ADDR outputF;
  GM_ADDR outputO;
  GM_ADDR outputTanhC;
};

struct tailSize {
    int64_t tailSingleCoreN;
    int64_t tailSingleCoreM;
    int64_t notTailNCoreCount;
    int64_t notTailMCoreCount;
    int32_t nCoreLoop;
    int32_t mCoreLoop;
    int64_t nCoreIndx;
    int64_t mCoreIndx;
};

using namespace AscendC;
template <typename T>
class LstmMmSplitNDNDBase {
 public:
  __aicore__ inline LstmMmSplitNDNDBase() = default;
  __aicore__ inline void GetCoreIndex(TCubeTiling& param, int32_t& subKIndx, tailSize& mmTail, int32_t kSize);
  __aicore__ inline void CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail, int32_t kSize);
  __aicore__ inline void InitBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                        GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                        GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                        GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC, GM_ADDR workspace);
  __aicore__ inline void InitVars();
  __aicore__ inline void InitQue();
  __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                        GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                        GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                        GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                        const DynamicRNNTilingData* __restrict rnnTiling, GM_ADDR workspace);
  __aicore__ inline void InitBuffersV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias,
                        GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf,
                        GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                        GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC, GM_ADDR workspace);
  __aicore__ inline void InitV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias,
                        GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo,
                        GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                        GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                        const DynamicRNNTilingData* __restrict rnnTiling, GM_ADDR workspace);
  __aicore__ inline int64_t Ceil(int64_t x, int64_t y);
  AscendC::TPipe pipe;
  // output GlobalTensors
  struct OutputGm {
    __aicore__ inline OutputGm() = default;
    AscendC::GlobalTensor<T> outYGm;
    AscendC::GlobalTensor<T> outHGm;
    AscendC::GlobalTensor<T> outCGm;
    AscendC::GlobalTensor<T> outIGm;
    AscendC::GlobalTensor<T> outJGm;
    AscendC::GlobalTensor<T> outFGm;
    AscendC::GlobalTensor<T> outOGm;
    AscendC::GlobalTensor<T> outTanhCGm;
    AscendC::GlobalTensor<float> workspace;
  };

  // input GlobalTensors
  struct InputGm {
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> weightInputGm;
    AscendC::GlobalTensor<T> weightHiddenGm;
    AscendC::GlobalTensor<T> biasGm;
    AscendC::GlobalTensor<T> seqLengthGm;
    AscendC::GlobalTensor<T> initHGm;
    AscendC::GlobalTensor<T> initCGm;
    AscendC::GlobalTensor<T> wciGm;
    AscendC::GlobalTensor<T> wcfGm;
    AscendC::GlobalTensor<T> wcoGm;
    AscendC::GlobalTensor<T> maskGm;
  };

  // Queue
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidCIn;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidVecIn;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidVecIn2;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qidVecOut;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

  // LocalTensor
  AscendC::LocalTensor<float> ubLocal1, ubLocal2, ubLocal3, ubLocal4;

  OutputGm outputGm;
  InputGm inputGm;

  int64_t inputMKAllSize;
  int64_t iOffset;
  int64_t oOffset;
  int64_t jOffset;
  int64_t fOffset;
  int64_t tailSingleCoreN;
  int64_t tailSingleCoreM;
  int64_t notTailNCoreCount;
  int64_t notTailMCoreCount;
  int32_t nCoreLoop;
  int32_t mCoreLoop;
  TRnnOffsets inputOffsets;
  TRnnOffsets hiddenOffsets;

  int64_t allCellSize;
  int64_t oneCellSize;
  tailSize hiddenTail;
  tailSize inputTail;
  int32_t oriSingleCoreN;
  TRnnOffsets oriInputOffsets;
  TRnnOffsets oriHiddenOffsets;

  AscendC::GlobalTensor<int32_t> sync_gm;
  const DynamicRNNTilingData* __restrict tiling;
  TCubeTiling inputMMTiling;
  TCubeTiling hiddenMMTiling;
  AscendC::LocalTensor<int> sync_buf;

  int64_t blockSize;
  int64_t calBlockSize;
  int64_t vectorCoreM;
  int64_t vectorTailM;
  int64_t vectorCoreNum;
  int64_t vectorBaseM;
  int64_t vectorBaseTailM;
  int64_t vectorTailTailM;
  int64_t baseVector;
  int64_t calcSize;
  int64_t calcSizeAlign;
  int64_t blockIdx;
  int64_t vectorSplitM;
  int64_t vectorSplitN;
  int64_t vectorTailSplitM;
  int64_t vectorTailN;
  int64_t vectorBaseN;
  int64_t calcM;
  int64_t calcN;
  int64_t coreCalcM;
};

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::GetCoreIndex(TCubeTiling& param, int32_t& subKIndx, tailSize& mmTail,
                                                            int32_t kSize) {
  auto temp0 = this->Ceil(param.M, param.singleCoreM);
  auto temp1 = this->Ceil(param.N, param.singleCoreN);
  auto temp2 = this->Ceil(kSize, param.singleCoreK);  // 不切K, 应该=1
  if (temp0 == 0) {
    temp0 = 1;
  }
  if (temp2 == 0) {
    temp2 = 1;
  }
  auto divideKcoreNum = param.usedCoreNum / temp2;
  mmTail.mCoreIndx = (GetBlockIdx() % divideKcoreNum) % temp0;
  mmTail.nCoreIndx = (GetBlockIdx() % divideKcoreNum) / temp0;
  subKIndx = GetBlockIdx() / divideKcoreNum;  // 缺省为0
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail,
                                                            int32_t kSize) {
  int32_t subKIndx;
  this->GetCoreIndex(param, subKIndx, mmTail, kSize);
  offset.AOffset = mmTail.mCoreIndx * kSize * param.singleCoreM;
  offset.BOffset = mmTail.nCoreIndx * param.singleCoreN;
  offset.BiasOffset = mmTail.nCoreIndx * param.singleCoreN;

  mmTail.nCoreLoop = this->Ceil(param.N, param.singleCoreN);
  mmTail.tailSingleCoreN = param.N - (mmTail.nCoreLoop - 1) * param.singleCoreN;
  mmTail.notTailNCoreCount = mmTail.nCoreLoop - 1;
  mmTail.mCoreLoop = this->Ceil(param.M, param.singleCoreM);
  mmTail.tailSingleCoreM = param.M - (mmTail.mCoreLoop - 1) * param.singleCoreM;
  mmTail.notTailMCoreCount = mmTail.mCoreLoop - 1;
  offset.COffset = mmTail.mCoreIndx * param.N * param.singleCoreM + mmTail.nCoreIndx * param.singleCoreN;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias,
                                                           GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi,
                                                           GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY,
                                                           GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                                           GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                           GM_ADDR outputTanhC, GM_ADDR workspace) {
  this->CalcGMOffset(this->hiddenMMTiling, this->hiddenOffsets, this->hiddenTail, static_cast<int32_t>(this->tiling->hiddenSize));
  this->CalcGMOffset(this->inputMMTiling, this->inputOffsets, this->inputTail, static_cast<int32_t>(this->tiling->inputSize));
  this->oneCellSize = this->tiling->batch * this->tiling->hiddenSize;
  this->allCellSize = this->oneCellSize * LSTM_GATE_SIZE;
  this->oriHiddenOffsets = this->hiddenOffsets;
  this->oriInputOffsets = this->inputOffsets;

  this->inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX),
                              this->tiling->timeStep * this->tiling->batch * this->tiling->inputSize);
  this->inputGm.weightInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight),
                                   this->tiling->inputSize * LSTM_GATE_SIZE * this->tiling->hiddenSize);
  this->inputGm.weightHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(
                                         weight + this->tiling->inputSize * LSTM_GATE_SIZE * this->tiling->hiddenSize * sizeof(T)),
                                         this->tiling->hiddenSize * LSTM_GATE_SIZE * this->tiling->hiddenSize);

  this->inputGm.biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias), LSTM_GATE_SIZE * this->tiling->hiddenSize);

  if (this->tiling->isSeqLength != 0) {
    this->inputGm.seqLengthGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(seqLength),
                                this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
  }

  if (this->tiling->isInithc != 0) {
    this->inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH), this->tiling->batch * this->tiling->hiddenSize);
    this->inputGm.initCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC), this->tiling->batch * this->tiling->hiddenSize);
  }
  this->outputGm.outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputY),
                                  this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
  this->outputGm.outHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputH),
                                  this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
  this->outputGm.outCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputC),
                                  this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
  if (this->tiling->isTraining == 1) {
    this->outputGm.outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputI),
                                    this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    this->outputGm.outJGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputJ),
                                    this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    this->outputGm.outFGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputF),
                                    this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    this->outputGm.outOGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputO),
                                    this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    this->outputGm.outTanhCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputTanhC),
                                        this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
  }
  this->outputGm.workspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace),
                                     this->tiling->timeStep * this->tiling->batch * LSTM_GATE_SIZE * this->tiling->hiddenSize);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitVars() {
  int64_t ubSize = 21504; // after div the max num of exiting node at the same time,include 4 gate
  if constexpr (std::is_same<T, float>::value) {
    ubSize = 16384;
  }
  int64_t calcMaxSize = ubSize / sizeof(float);
  int64_t blockN = this->tiling->usedCoreNum;
  this->blockSize = 32 / sizeof(T);
  this->calBlockSize = 32 / sizeof(float);
  int64_t calcSize =  this->blockSize;
  if constexpr (std::is_same<T, float>::value) {
    calcSize = this->calBlockSize;
  }
  this->vectorCoreM = this->Ceil(this->tiling->batch, blockN);
  this->vectorCoreNum = this->Ceil(this->tiling->batch, this->vectorCoreM);
  this->vectorTailM = this->tiling->batch % this->vectorCoreM ? this->tiling->batch % this->vectorCoreM : this->vectorCoreM;

  this->vectorSplitN = this->Ceil(this->tiling->hiddenSize, calcMaxSize);
  if (this->vectorSplitN == 1) {
    this->vectorBaseN = this->tiling->hiddenSize;
    this->vectorTailN = 0;
  } else {
    this->vectorBaseN = this->Ceil(this->Ceil(this->tiling->hiddenSize, this->vectorSplitN), calcSize) * calcSize;
    this->vectorTailN = this->tiling->hiddenSize - this->vectorBaseN * (this->vectorSplitN - 1);
  }

  this->vectorBaseM = ((calcMaxSize / this->vectorBaseN) > this->vectorCoreM) ? this->vectorCoreM : (calcMaxSize / this->vectorBaseN);

  this->vectorBaseTailM = this->vectorCoreM % this->vectorBaseM;
  this->vectorTailTailM = this->vectorTailM % this->vectorBaseM;

  this->vectorSplitM = this->Ceil(this->vectorCoreM, this->vectorBaseM);
  this->vectorTailSplitM = this->Ceil(this->vectorTailM, this->vectorBaseM);

  this->baseVector = this->vectorBaseM * this->Ceil(this->vectorBaseN, calcSize) * calcSize;

  this->iOffset = 0;
  this->jOffset = this->tiling->gateOrder == 0 ? this->tiling->hiddenSize : 2 * this->tiling->hiddenSize;
  this->fOffset = this->tiling->gateOrder == 0 ? 2 * this->tiling->hiddenSize : this->tiling->hiddenSize;
  this->oOffset = 3 * this->tiling->hiddenSize;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitQue() {
  this->pipe.InitBuffer(this->qidVecIn, 1, this->baseVector * sizeof(float));
  this->pipe.InitBuffer(this->qidVecOut, 1, this->baseVector * sizeof(T));
  this->pipe.InitBuffer(this->calcBuf, 4 * this->baseVector * sizeof(float));
  if constexpr (!std::is_same<T, float>::value) {
    this->pipe.InitBuffer(this->qidCIn, 1, this->baseVector * sizeof(T));
    this->pipe.InitBuffer(this->qidVecIn2, 1, this->baseVector * sizeof(float));
  }
  // Init Local Tensors
  this->ubLocal1 = this->calcBuf.template Get<float>(4 * this->baseVector);
  this->ubLocal2 = this->ubLocal1[this->baseVector];
  this->ubLocal3 = this->ubLocal2[this->baseVector];
  this->ubLocal4 = this->ubLocal3[this->baseVector];
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::Init(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength,
                                                GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo,
                                                GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                                GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                GM_ADDR outputTanhC, const DynamicRNNTilingData* __restrict rnnTiling,
                                                GM_ADDR workspace) {
  this->tiling = rnnTiling;
  this->inputMMTiling = this->tiling->inputMMParam;
  this->hiddenMMTiling = this->tiling->hiddenMMParam;
  this->InitBuffers(inputX, weight, bias, seqLength, initH, initC, wCi, wCf, wCo, mask, outputY, outputH, outputC, outputI,
              outputJ, outputF, outputO, outputTanhC, workspace);
  this->InitVars();
  this->InitQue();
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitBuffersV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden,
                                                    GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC,
                                                    GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                                                    GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                                    GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                    GM_ADDR outputTanhC, GM_ADDR workspace) {
  this->CalcGMOffset(this->hiddenMMTiling, this->hiddenOffsets, this->hiddenTail, static_cast<int32_t>(this->tiling->hiddenSize));
  this->CalcGMOffset(this->inputMMTiling, this->inputOffsets, this->inputTail, static_cast<int32_t>(this->tiling->inputSize));
  this->oneCellSize = this->tiling->batch * this->tiling->hiddenSize;
  this->allCellSize = this->oneCellSize * LSTM_GATE_SIZE;
  this->oriHiddenOffsets = this->hiddenOffsets;
  this->oriInputOffsets = this->inputOffsets;

  this->inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX),
                              this->tiling->timeStep * this->tiling->batch * this->tiling->inputSize);
  this->inputGm.weightInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weightInput),
                                   this->tiling->inputSize * LSTM_GATE_SIZE * this->tiling->hiddenSize);
  this->inputGm.weightHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weightHidden),
                                   this->tiling->hiddenSize * LSTM_GATE_SIZE * this->tiling->hiddenSize);

  if (this->tiling->isBias == 1) {
    this->inputGm.biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias), LSTM_GATE_SIZE * this->tiling->hiddenSize);
  }
  if (this->tiling->isSeqLength != 0) {
    this->inputGm.seqLengthGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(seqLength),
                                this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
  }
  if (this->tiling->isInithc != 0) {
    this->inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH), this->tiling->batch * this->tiling->hiddenSize);
    this->inputGm.initCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC), this->tiling->batch * this->tiling->hiddenSize);
  }
  this->outputGm.outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputY),
                                  this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
  this->outputGm.outHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputH),
                                  this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
  this->outputGm.outCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputC),
                                  this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
  if (this->tiling->isTraining == 1) {
    this->outputGm.outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputI),
                                    this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    this->outputGm.outJGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputJ),
                                    this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    this->outputGm.outFGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputF),
                                    this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    this->outputGm.outOGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputO),
                                    this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    this->outputGm.outTanhCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputTanhC),
                                        this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
  }
  this->outputGm.workspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace),
                                     this->tiling->timeStep * this->tiling->batch * LSTM_GATE_SIZE * this->tiling->hiddenSize);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden,
                                                GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC,
                                                GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                                                GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                                GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF,
                                                GM_ADDR outputO, GM_ADDR outputTanhC,
                                                const DynamicRNNTilingData* __restrict rnnTiling, GM_ADDR workspace) {
  this->tiling = rnnTiling;
  this->inputMMTiling = this->tiling->inputMMParam;
  this->hiddenMMTiling = this->tiling->hiddenMMParam;
  this->InitBuffersV2(inputX, weightInput, weightHidden, bias, seqLength, initH, initC,
            wCi, wCf, wCo, mask, outputY, outputH, outputC, outputI, outputJ, outputF, outputO, outputTanhC, workspace);
  this->InitVars();
  this->InitQue();
}

template <typename T>
__aicore__ inline int64_t LstmMmSplitNDNDBase<T>::Ceil(int64_t x, int64_t y) {
  if (y == 0) {
    return x;
  }
  return (x + y - 1) / y;
}
#endif
