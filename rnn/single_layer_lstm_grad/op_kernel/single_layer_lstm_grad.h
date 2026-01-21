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
 * \file single_layer_lstm_grad.h
 * \brief
 */
#ifndef _SINGLE_LAYER_LSTM_GRAD_H_
#define _SINGLE_LAYER_LSTM_GRAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

constexpr int64_t LSTM_GATE_SIZE = 4;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t HALF_AND_CAST_BYTES = 6;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t UB_SIZE = 192 * 1024;
constexpr int64_t HALF_BYTES = 2;
constexpr int64_t DEFAULT_AIV_CORE_NUM = 48;
constexpr int64_t UB_RESERVED_SIZE = 1024;
constexpr int64_t VECTOR_ALIGN_SIZE = 16;
constexpr int64_t BYTES_PER_KILOBYTE = 1024;
constexpr int64_t ALIGN_8_BYTES = 8;
constexpr int64_t ALIGN_16_BYTES = 16;
constexpr int64_t DOUBLE_BLOCK_BYTES = 64;
constexpr int64_t BINARY_REDUCE_BASE = 2;
constexpr int64_t DEFAULT_BLOCK_COUNT = 1;
constexpr int64_t GATE_ORDER_IJFO = 0;
constexpr int64_t GATE_ORDER_IFJO = 1;
constexpr int64_t DIRECTION_FORWARD = 0;
constexpr int64_t DIRECTION_BACKWARD = 1;
constexpr int64_t SEQ_LENGTH_ENABLED = 1;
constexpr int64_t AIC_AIV_NUM = 2;
constexpr uint32_t SRC_STRIDE_OFFSET_DIVISOR = 8;
constexpr float FLOAT_ONE = 1.0f;
constexpr float FLOAT_NEG_ONE = -1.0f;
constexpr int64_t OFFSET_I = 0;
constexpr int64_t OFFSET_J = 1;
constexpr int64_t OFFSET_F = 2;
constexpr int64_t OFFSET_O = 3;
constexpr int64_t SEQ_FP32_OFFSET = 10;
constexpr int64_t DH_FP32_OFFSET = 14;
constexpr int64_t DOUBLE = 2;
constexpr int64_t MAX_COPY_LINES = 4095;

struct TRnnOffsets {
  int64_t AOffset{0};
  int64_t BOffset{0};
  int64_t COffset{0};
  int64_t BiasOffset{0};
};

struct tailSize {
  int64_t tailSingleCoreN{0};
  int64_t tailSingleCoreM{0};
  int64_t notTailNCoreCount{0};
  int64_t notTailMCoreCount{0};
  int32_t nCoreLoop{0};
  int32_t mCoreLoop{0};
  int64_t nCoreIndx{0};
  int64_t mCoreIndx{0};
};

struct blockParams {
  int64_t mShape{0};
  int64_t nShape{0};
  int64_t mCntIndex{0};
  int64_t nCntIndex{0};
  int64_t offset{0};
  int64_t gateOutOffset{0};
};

struct blockReduceParams {
  int64_t nReduceCntIndx{0};
  int64_t nReduceShape{0};
  int64_t nReduceShapeAlign{0};
  int64_t offset{0};
};

struct singCloreParams {
  int64_t mCntIndex{0};
  int64_t nCntIndex{0};
  int64_t mSingleCoreShape{0};
  int64_t nSingleCoreShape{0};
  int64_t hiddenOffset{0};
  int64_t gateOutOffset{0};

  int64_t reduceNCoreIndex{0};
  int64_t nSingleCoreReduceShape{0};
};

__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b) {
  if (b == 0) {
    return a;
  }
  return (a + b - 1) / b;
}

template <typename T, const MatmulConfig &MM_GATE_CFG, const MatmulConfig &MM_WEIGHT_CFG, bool DXH_HUGE = false,
          bool XH_HUGE = false>
class RNNGrad {
 public:
  __aicore__ inline RNNGrad(){};
  __aicore__ inline void Process() {
    if constexpr (!std::is_same<T, float>::value) {
      InitCastWFp32();
      SyncAll();
    }
    for (int64_t tIdx = tiling.timeStep - 1; tIdx >= 0; tIdx--) {
      bool firstLoop = tIdx == tiling.timeStep - 1;
      bool useInitCH = tIdx == 0;
      // Loop forward or backward
      int64_t actTIdx = tiling.direction == DIRECTION_BACKWARD ? tiling.timeStep - tIdx - 1 : tIdx;
      InitVectorBuf();
      ProcessDgates(actTIdx, firstLoop, useInitCH);  // Calcu dgates
      SyncAll();
      ProcessDgateMM(actTIdx); // Dgate mm
      SyncAll();
      InitSubDataBuf();
      if constexpr (DXH_HUGE) {
        ProcessDxhSplitLarge(actTIdx, firstLoop, useInitCH);
      } else if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
        ProcessDxhSplitLarge(actTIdx, firstLoop, useInitCH);
      } else {
        ProcessDxhSplit(actTIdx, useInitCH);
      } // Dxh split
      SyncAll();
    }

    if constexpr (XH_HUGE) {
      ProcessConcatXhLarge();
    } else {
      ProcessConcatXh();
    } // Concat xh for dw mm
    SyncAll();
    ProcessDWMM(); // Dw mm
    if (tiling.isBias) {
      SyncAll();
      ProcessReduce(); // Reduce and ProcessDWMM can be parallelized, but additional memory is required.
    }
    if constexpr (!std::is_same<T, float>::value) {
      SyncAll();
      CastDwBack();
    }
  }

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                              GM_ADDR initH, GM_ADDR initC, GM_ADDR h, GM_ADDR c, GM_ADDR dy,
                              GM_ADDR dh, GM_ADDR dc, GM_ADDR i, GM_ADDR j,
                              GM_ADDR f, GM_ADDR o, GM_ADDR tanhct, GM_ADDR seq_length,
                              GM_ADDR dw, GM_ADDR db, GM_ADDR dx,
                              GM_ADDR dhPrev, GM_ADDR dcPrev,
                              const SingleLayerLstmGradTilingData* __restrict rnnTiling,
                              GM_ADDR workspace, TPipe* inputPipe) {
    pipe = inputPipe;
    ascendc_assert(GetBlockNum() != 0 && "block dim can not be zero!");
    aivCoreNum = GetBlockNum() * AIC_AIV_NUM;
    tiling = *rnnTiling;
    dxhInputTiling = rnnTiling->dxhInputTiling;
    dxhHiddenTiling = rnnTiling->dxhHiddenTiling;
    xhInputTiling = rnnTiling->xhInputTiling;
    xhHiddenTiling = rnnTiling->xhHiddenTiling;
    dwMMTiling = tiling.dwMMParam;
    dgateMMTiling = tiling.dgateMMParam;
    InitGlobalBuffers(x, weight, bias, y, initH, initC, h, c, dy, dh, dc, i, j, f,
                o, tanhct, seq_length, dw, db, dx, dhPrev, dcPrev, workspace);
    InitVars();
  }
 public:
  TPipe *pipe;

  // Describe Matmul input/output dtype&format
  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, true>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                 MM_WEIGHT_CFG>
      dwMM;

  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                 MM_GATE_CFG>
      dgateMM;

 private:
  struct OutputGm {
    __aicore__ inline OutputGm() = default;
    AscendC::GlobalTensor<T> dwGm;
    AscendC::GlobalTensor<T> dbGm;
    AscendC::GlobalTensor<T> dxGm;
    AscendC::GlobalTensor<T> dhPrevGm;
    AscendC::GlobalTensor<T> dcPrevGm;
    AscendC::GlobalTensor<float> dgateGm; // workspace t * b * 4 * hiddenSize
    AscendC::GlobalTensor<float> xhGm; // workspace b * (hiddensize + inputSize)
    AscendC::GlobalTensor<float> dxhGm; // workspace b * (hiddensize + inputSize)
    AscendC::GlobalTensor<float> wFp32Gm; // workspace 4 * hiddensize * (hiddensize + inputSize)
    AscendC::GlobalTensor<float> dwFp32Gm; // workspace 4 * hiddensize * (hiddensize + inputSize)
    AscendC::GlobalTensor<float> dhPrevFp32Gm; // workspace batch * hiddensize
    AscendC::GlobalTensor<float> dcPrevFp32Gm; // workspace batch * hiddensize
  };

  // input GlobalTensors
  struct InputGm {
    __aicore__ inline InputGm() = default;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> weightGm;
    AscendC::GlobalTensor<T> biasGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<T> initHGm;
    AscendC::GlobalTensor<T> initCGm;
    AscendC::GlobalTensor<T> hGm;
    AscendC::GlobalTensor<T> cGm;
    AscendC::GlobalTensor<T> dyGm;
    AscendC::GlobalTensor<T> dhGm;
    AscendC::GlobalTensor<T> dcGm;
    AscendC::GlobalTensor<T> iGm;
    AscendC::GlobalTensor<T> jGm;
    AscendC::GlobalTensor<T> fGm;
    AscendC::GlobalTensor<T> oGm;
    AscendC::GlobalTensor<T> tanhctGm;
    AscendC::GlobalTensor<T> seqLengthGm;
  };

  TBuf<AscendC::TPosition::VECCALC> itBuf;
  TBuf<AscendC::TPosition::VECCALC> jtBuf;
  TBuf<AscendC::TPosition::VECCALC> ftBuf;
  TBuf<AscendC::TPosition::VECCALC> otBuf;
  TBuf<AscendC::TPosition::VECCALC> ctBuf;
  TBuf<AscendC::TPosition::VECCALC> ctPreBuf;
  TBuf<AscendC::TPosition::VECCALC> ditBuf;
  TBuf<AscendC::TPosition::VECCALC> djtBuf;
  TBuf<AscendC::TPosition::VECCALC> dftBuf;
  TBuf<AscendC::TPosition::VECCALC> dotBuf;
  TBuf<AscendC::TPosition::VECCALC> dcnextBuf;
  TBuf<AscendC::TPosition::VECCALC> dhnextBuf;
  TBuf<AscendC::TPosition::VECCALC> dhtBuf;
  TBuf<AscendC::TPosition::VECCALC> dctBuf;
  TBuf<AscendC::TPosition::VECCALC> dyBuf;
  TBuf<AscendC::TPosition::VECCALC> tmpJBuf;
  TBuf<AscendC::TPosition::VECCALC> tanhBuf;
  TBuf<AscendC::TPosition::VECCALC> tmpCBuf;
  TBuf<AscendC::TPosition::VECCALC> subDataBuf;
  TBuf<AscendC::TPosition::VECCALC> seqBuf;
  TBuf<AscendC::TPosition::VECCALC> wCastBuf;
  TBuf<AscendC::TPosition::VECCALC> wBuf;

  // dgates tensor
  LocalTensor<float> itTensor;
  LocalTensor<float> jtTensor;
  LocalTensor<float> ftTensor;
  LocalTensor<float> otTensor;
  LocalTensor<float> ctTensor;
  LocalTensor<float> ctPreTensor;
  LocalTensor<float> dyTensor;
  LocalTensor<float> ditTensor;
  LocalTensor<float> djtTensor;
  LocalTensor<float> dftTensor;
  LocalTensor<float> dotTensor;
  LocalTensor<float> dcnextTensor;
  LocalTensor<float> dhnextTensor;
  LocalTensor<float> dhtTensor;
  LocalTensor<float> dctTensor;
  LocalTensor<float> tmpJTensor;
  LocalTensor<float> tanhTensor;
  LocalTensor<float> tmpCTensor;
  LocalTensor<float> seqTensor;

  // ub buffer by offset
  LocalTensor<uint8_t> totalUb;
  // split dh out
  LocalTensor<T> dhPrevUb;
  LocalTensor<float> dhPrevFp32Ub;
  LocalTensor<float> seqFp32Ub;
  LocalTensor<T> seqUb;
  LocalTensor<float> dhNextFp32Ub;
  LocalTensor<T> dhNextUb;

  // split dx out
  LocalTensor<T> dxUb;
  LocalTensor<float> dxFp32Ub;
  LocalTensor<T> outDxUb;

  // concat xh
  LocalTensor<T> xUb;
  LocalTensor<T> hUb;
  LocalTensor<float> xFp32Ub;
  LocalTensor<float> hFp32Ub;
  LocalTensor<float> outXUb;
  LocalTensor<float> outHUb;

  blockParams block_;
  singCloreParams core_;
  blockReduceParams reduceBlock_;

  OutputGm outputGm;
  InputGm inputGm;

  int64_t calcSize = 0;
  int64_t calcSizeAlign = 0;
  TRnnOffsets dwOffsets;
  TRnnOffsets dgateOffsets;

  int64_t allCellSize = 0;
  int64_t oneCellSize = 0;
  tailSize dgateTail;
  tailSize dwTail;
  TRnnOffsets oriDwOffsets;
  TRnnOffsets oriDgateOffsets;

  CutBatchTiling dxhInputTiling;
  CutBatchTiling dxhHiddenTiling;
  CutBatchTiling xhInputTiling;
  CutBatchTiling xhHiddenTiling;
  SingleLayerLstmGradTilingData tiling;
  TCubeTiling dwMMTiling;
  TCubeTiling dgateMMTiling;

  int64_t blockSize = 0;
  int64_t calBlockSize = 0;
  int64_t aivCoreNum = DEFAULT_AIV_CORE_NUM;
  int64_t paraBytes = sizeof(T);
  DataCopyPadExtParams<float> padInFloatParams = {false, 0, 0, 0};
  DataCopyPadExtParams<T> padInTParams = {false, 0, 0, 0};

 private:
  __aicore__ inline void InitCastWFp32() {
    int64_t maxCalcNum = (tiling.ubSize - UB_RESERVED_SIZE) / (sizeof(T) + FLOAT_BYTES) / VECTOR_ALIGN_SIZE *
                         VECTOR_ALIGN_SIZE;
    pipe->InitBuffer(wBuf, HALF_BYTES * maxCalcNum);
    pipe->InitBuffer(wCastBuf, FLOAT_BYTES * maxCalcNum);
    LocalTensor<T> wTensor = wBuf.Get<T>();
    LocalTensor<float> wCastTensor = wCastBuf.Get<float>();
    int64_t totalNum = (tiling.hiddenSize + tiling.inputSize) * tiling.hiddenSize * LSTM_GATE_SIZE;
    int64_t totalLoops = CeilDiv(totalNum, maxCalcNum);
    int64_t calcTail = totalNum - (totalLoops - 1) * maxCalcNum;
    for (int64_t loopIndex = 0; loopIndex < totalLoops; loopIndex++) {
        if (GetBlockIdx() == loopIndex % GetBlockNum()) {
            int64_t calcNum = (loopIndex == totalLoops - 1) ? calcTail : maxCalcNum;
            DataCopyExtParams copyParamsIn = {1, static_cast<uint32_t>(calcNum * sizeof(T)), 0, 0, 0};
            DataCopyPad(wTensor, inputGm.weightGm[loopIndex * maxCalcNum], copyParamsIn, padInTParams);
            MTE2ToVSync();
            Cast(wCastTensor, wTensor, RoundMode::CAST_NONE, calcNum);
            VToMTE3Sync();
            VToMTE2Sync();
            DataCopyExtParams copyParamsOut = {1, static_cast<uint32_t>(calcNum * FLOAT_BYTES), 0, 0, 0};
            DataCopyPad(outputGm.wFp32Gm[loopIndex * maxCalcNum], wCastTensor, copyParamsOut);
            MTE3ToVSync();
        }
    }
  }

  __aicore__ inline void CastDwBack() {
    pipe->Reset();
    int64_t maxCalcNum = (tiling.ubSize - UB_RESERVED_SIZE) / (sizeof(T) + FLOAT_BYTES) / VECTOR_ALIGN_SIZE *
                         VECTOR_ALIGN_SIZE;
    pipe->InitBuffer(wBuf, HALF_BYTES * maxCalcNum);
    pipe->InitBuffer(wCastBuf, FLOAT_BYTES * maxCalcNum);
    LocalTensor<T> dwTensor = wBuf.Get<T>();
    LocalTensor<float> dwCastTensor = wCastBuf.Get<float>();
    int64_t totalNum = (tiling.hiddenSize + tiling.inputSize) * tiling.hiddenSize * LSTM_GATE_SIZE;
    int64_t totalLoops = CeilDiv(totalNum, maxCalcNum);
    int64_t calcTail = totalNum - (totalLoops - 1) * maxCalcNum;
    for (int64_t loopIndex = 0; loopIndex < totalLoops; loopIndex++) {
        if (GetBlockIdx() == loopIndex % GetBlockNum()) {
          int64_t calcNum = (loopIndex == totalLoops - 1) ? calcTail : maxCalcNum;
          DataCopyExtParams copyParamsIn = {1, static_cast<uint32_t>(calcNum * FLOAT_BYTES), 0, 0, 0};
          DataCopyPad(dwCastTensor, outputGm.dwFp32Gm[loopIndex * maxCalcNum], copyParamsIn, padInFloatParams);
          MTE2ToVSync();
          if constexpr (std::is_same<T, half>::value) {
            Cast(dwTensor, dwCastTensor, RoundMode::CAST_NONE, calcNum);
          } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(dwTensor, dwCastTensor, RoundMode::CAST_RINT, calcNum);
          }
          VToMTE3Sync();
          VToMTE2Sync();
          DataCopyExtParams copyParamsOut = {1, static_cast<uint32_t>(calcNum * HALF_BYTES), 0, 0, 0};
          DataCopyPad(outputGm.dwGm[loopIndex * maxCalcNum], dwTensor, copyParamsOut);
          MTE3ToVSync();
        }
    }
  }

  __aicore__ inline void InitVars() {
    blockSize = BLOCK_BYTES / paraBytes;
    calBlockSize = BLOCK_BYTES / paraBytes;
  }

  __aicore__ inline void InitVectorBuf() {
    // Apply by fp32
    pipe->Reset();
    pipe->InitBuffer(itBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN); // BaseN is aligned.
    pipe->InitBuffer(jtBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    // Init Local Tensors
    pipe->InitBuffer(ftBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(otBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(ctBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(ctPreBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);

    pipe->InitBuffer(ditBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(djtBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(dftBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(dotBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(dcnextBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(dhnextBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(dhtBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(dctBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(tmpJBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(tanhBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(tmpCBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(dyBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
      pipe->InitBuffer(seqBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    }
  }

  __aicore__ inline void InitSubDataBuf() {
    pipe->Reset();
    pipe->InitBuffer(subDataBuf, tiling.ubSize);
  }

  __aicore__ inline void InitGlobalBuffers(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR initH, GM_ADDR initC, GM_ADDR h, GM_ADDR c, GM_ADDR dy,
                                            GM_ADDR dh, GM_ADDR dc, GM_ADDR i, GM_ADDR j,
                                            GM_ADDR f, GM_ADDR o, GM_ADDR tanhct, GM_ADDR seqLength,
                                            GM_ADDR dw, GM_ADDR db, GM_ADDR dx,
                                            GM_ADDR dhPrev, GM_ADDR dcPrev, GM_ADDR workspace) {
    CalcGMOffset(dgateMMTiling, dgateOffsets, dgateTail, static_cast<int32_t>(tiling.hiddenSize * LSTM_GATE_SIZE));
    CalcGMDwOffset(dwMMTiling, dwOffsets, dwTail, static_cast<int32_t>(tiling.batch * tiling.timeStep));
    oneCellSize = tiling.batch * tiling.hiddenSize;
    allCellSize = oneCellSize * LSTM_GATE_SIZE;
    oriDgateOffsets = dgateOffsets;
    oriDwOffsets = dwOffsets;

    inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),
                                tiling.timeStep * tiling.batch * tiling.inputSize);
    inputGm.weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight),
                                    tiling.hiddenSize * LSTM_GATE_SIZE * (tiling.inputSize + tiling.hiddenSize));
    inputGm.yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),
                                tiling.timeStep * tiling.batch * tiling.hiddenSize);
    inputGm.biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias), LSTM_GATE_SIZE * tiling.hiddenSize);
    inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH), tiling.batch * tiling.hiddenSize);
    inputGm.initCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC), tiling.batch * tiling.hiddenSize);
    inputGm.hGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h),
                                  tiling.timeStep * tiling.batch * tiling.hiddenSize);
    inputGm.cGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(c),
                                  tiling.timeStep * tiling.batch * tiling.hiddenSize);
    inputGm.dyGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dy),
                                  tiling.timeStep * tiling.batch * tiling.hiddenSize);
    if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
      inputGm.seqLengthGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(seqLength), tiling.timeStep * tiling.batch *
                                          tiling.hiddenSize);
    }

    inputGm.dhGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dh), tiling.batch * tiling.hiddenSize);
    inputGm.dcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dc), tiling.batch * tiling.hiddenSize);

    inputGm.iGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(i),
                                    tiling.timeStep * tiling.batch * tiling.hiddenSize);
    inputGm.jGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(j),
                                          tiling.timeStep * tiling.batch * tiling.hiddenSize);
    inputGm.fGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(f), tiling.timeStep * tiling.batch * tiling.hiddenSize);
    inputGm.oGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(o), tiling.timeStep * tiling.batch * tiling.hiddenSize);
    inputGm.tanhctGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tanhct), tiling.timeStep * tiling.batch *
                                     tiling.hiddenSize);

    outputGm.dwGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dw),
                                    tiling.hiddenSize * LSTM_GATE_SIZE * (tiling.hiddenSize + tiling.inputSize));
    outputGm.dbGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(db),
                                    paraBytes * tiling.hiddenSize);
    outputGm.dxGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dx),
                                    tiling.timeStep * tiling.batch * tiling.inputSize);
    outputGm.dhPrevGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dhPrev),
                                    tiling.batch * tiling.hiddenSize);
    outputGm.dcPrevGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dcPrev),
                                    tiling.batch * tiling.hiddenSize);
    outputGm.dgateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), tiling.timeStep * tiling.batch *
                                     LSTM_GATE_SIZE * tiling.hiddenSize);
    int64_t dxhGmOffset = tiling.timeStep * tiling.batch * LSTM_GATE_SIZE * tiling.hiddenSize * LSTM_GATE_SIZE;
    outputGm.dxhGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + dxhGmOffset),
                                    tiling.batch * (tiling.hiddenSize + tiling.inputSize));
    int64_t xhGmOffset = dxhGmOffset + tiling.batch * (tiling.hiddenSize + tiling.inputSize) * FLOAT_BYTES;
    outputGm.xhGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + xhGmOffset),
                                  tiling.timeStep * tiling.batch * (tiling.hiddenSize + tiling.inputSize));
    if constexpr (!std::is_same<T, float>::value) {
      int64_t wFp32GmOffset = xhGmOffset + tiling.timeStep * tiling.batch * (tiling.hiddenSize + tiling.inputSize) *
                              FLOAT_BYTES;
      outputGm.wFp32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + wFp32GmOffset),
                                      LSTM_GATE_SIZE * tiling.hiddenSize * (tiling.hiddenSize + tiling.inputSize));
      int64_t dwFp32GmOffset = wFp32GmOffset +
        LSTM_GATE_SIZE * tiling.hiddenSize * (tiling.hiddenSize + tiling.inputSize) * FLOAT_BYTES;
      outputGm.dwFp32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + dwFp32GmOffset),
                                      LSTM_GATE_SIZE * tiling.hiddenSize * (tiling.hiddenSize + tiling.inputSize));
      int64_t dhPrevFp32GmOffset = dwFp32GmOffset +
        LSTM_GATE_SIZE * tiling.hiddenSize * (tiling.hiddenSize + tiling.inputSize) * FLOAT_BYTES;
      outputGm.dhPrevFp32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + dhPrevFp32GmOffset),
                                    tiling.batch * tiling.hiddenSize);
      int64_t dcPrevFp32GmOffset = dhPrevFp32GmOffset + tiling.batch * tiling.hiddenSize * FLOAT_BYTES;
      outputGm.dcPrevFp32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + dcPrevFp32GmOffset),
                                      tiling.batch * tiling.hiddenSize);
    } else {
      outputGm.dhPrevFp32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dhPrev), tiling.batch * tiling.hiddenSize);
      outputGm.dcPrevFp32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dcPrev), tiling.batch * tiling.hiddenSize);
      outputGm.wFp32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight),
                                      LSTM_GATE_SIZE * tiling.hiddenSize * (tiling.hiddenSize + tiling.inputSize));
      outputGm.dwFp32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dw),
                                      LSTM_GATE_SIZE * tiling.hiddenSize * (tiling.hiddenSize + tiling.inputSize));
    }
  }

  __aicore__ inline void GetCoreIndex(TCubeTiling& param, int32_t& subKIndx, tailSize& mmTail,
                                                              int32_t kSize) {
    auto temp0 = Ceil(param.M, param.singleCoreM);
    auto temp1 = Ceil(param.N, param.singleCoreN);
    auto temp2 = Ceil(kSize, param.singleCoreK);
    if (temp0 == 0) {
      temp0 = 1;
    }
    if (temp2 == 0) {
      temp2 = 1;
    }
    auto divideKcoreNum = param.usedCoreNum / temp2;
    mmTail.mCoreIndx = (GetBlockIdx() % divideKcoreNum) % temp0;
    mmTail.nCoreIndx = (GetBlockIdx() % divideKcoreNum) / temp0;
    subKIndx = GetBlockIdx() / divideKcoreNum;  // default 0
  }

  __aicore__ inline void CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail, int32_t kSize) {
    int32_t subKIndx;
    GetCoreIndex(param, subKIndx, mmTail, kSize);
    offset.AOffset = mmTail.mCoreIndx * kSize * param.singleCoreM;
    offset.BOffset = mmTail.nCoreIndx * param.singleCoreN;
    offset.BiasOffset = mmTail.nCoreIndx * param.singleCoreN;

    mmTail.nCoreLoop = Ceil(param.N, param.singleCoreN);
    mmTail.tailSingleCoreN = param.N - (mmTail.nCoreLoop - 1) * param.singleCoreN;
    mmTail.notTailNCoreCount = mmTail.nCoreLoop - 1;
    mmTail.mCoreLoop = Ceil(param.M, param.singleCoreM);
    mmTail.tailSingleCoreM = param.M - (mmTail.mCoreLoop - 1) * param.singleCoreM;
    mmTail.notTailMCoreCount = mmTail.mCoreLoop - 1;
    offset.COffset = mmTail.mCoreIndx * param.N * param.singleCoreM + mmTail.nCoreIndx * param.singleCoreN;
  }

  __aicore__ inline void CalcGMDwOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail, int32_t kSize) {
    int32_t subKIndx;
    GetCoreIndex(param, subKIndx, mmTail, kSize);
    offset.AOffset = mmTail.mCoreIndx * param.singleCoreM;
    offset.BOffset = mmTail.nCoreIndx * param.singleCoreN;
    offset.BiasOffset = mmTail.nCoreIndx * param.singleCoreN;

    mmTail.nCoreLoop = Ceil(param.N, param.singleCoreN);
    mmTail.tailSingleCoreN = param.N - (mmTail.nCoreLoop - 1) * param.singleCoreN;
    mmTail.notTailNCoreCount = mmTail.nCoreLoop - 1;
    mmTail.mCoreLoop = Ceil(param.M, param.singleCoreM);
    mmTail.tailSingleCoreM = param.M - (mmTail.mCoreLoop - 1) * param.singleCoreM;
    mmTail.notTailMCoreCount = mmTail.mCoreLoop - 1;
    offset.COffset = mmTail.mCoreIndx * param.N * param.singleCoreM + mmTail.nCoreIndx * param.singleCoreN;
  }

  __aicore__ inline void ProcessDWMM() {
    if (GetBlockIdx() < dwMMTiling.usedCoreNum) {
      dwMM.SetTensorA(outputGm.dgateGm[dwOffsets.AOffset], true);
      dwMM.SetTensorB(outputGm.xhGm[dwOffsets.BOffset]);
      if (dwTail.nCoreIndx == dwTail.notTailNCoreCount && dwTail.mCoreIndx == dwTail.notTailMCoreCount) {
        dwMM.SetTail(dwTail.tailSingleCoreM, dwTail.tailSingleCoreN);
      } else if (dwTail.nCoreIndx == dwTail.notTailNCoreCount) {
        dwMM.SetTail(dwMMTiling.singleCoreM, dwTail.tailSingleCoreN);
      } else if (dwTail.mCoreIndx == dwTail.notTailMCoreCount) {
        dwMM.SetTail(dwTail.tailSingleCoreM, dwMMTiling.singleCoreN);
      }
      dwMM.IterateAll(outputGm.dwFp32Gm[dwOffsets.COffset], false);
    }
  }

  __aicore__ inline void ProcessDgateMM(int64_t tIdx) {
    if (GetBlockIdx() < dgateMMTiling.usedCoreNum) {
      dgateOffsets.AOffset = oriDgateOffsets.AOffset + tIdx * tiling.batch * tiling.hiddenSize * LSTM_GATE_SIZE;
      dgateMM.SetTensorA(outputGm.dgateGm[dgateOffsets.AOffset]);
      dgateMM.SetTensorB(outputGm.wFp32Gm[dgateOffsets.BOffset]);
      if (dgateTail.nCoreIndx == dgateTail.notTailNCoreCount && dgateTail.mCoreIndx == dgateTail.notTailMCoreCount) {
          dgateMM.SetTail(dgateTail.tailSingleCoreM, dgateTail.tailSingleCoreN);
      } else if (dgateTail.nCoreIndx == dgateTail.notTailNCoreCount) {
          dgateMM.SetTail(dgateMMTiling.singleCoreM, dgateTail.tailSingleCoreN);
      } else if (dgateTail.mCoreIndx == dgateTail.notTailMCoreCount) {
          dgateMM.SetTail(dgateTail.tailSingleCoreM, dgateMMTiling.singleCoreN);
      }
      dgateMM.IterateAll(outputGm.dxhGm[dgateOffsets.COffset], false);
    }
  }

  __aicore__ inline void CopyOutputGate(GlobalTensor<float>& gm, LocalTensor<float>& ub, int64_t gateIdx,
                                        int64_t nShapeAligned) {
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = block_.mShape;
    dataCopyParams.blockLen = block_.nShape * FLOAT_BYTES;
    dataCopyParams.srcStride = (nShapeAligned - block_.nShape) / SRC_STRIDE_OFFSET_DIVISOR;
    dataCopyParams.dstStride = (tiling.hiddenSize * LSTM_GATE_SIZE - block_.nShape) * FLOAT_BYTES;
    int64_t offset = block_.gateOutOffset + gateIdx * tiling.hiddenSize;
    DataCopyPad(gm[offset], ub, dataCopyParams);
  }

  template <typename copyType>
  __aicore__ inline void CopyOutputDcPrev(const GlobalTensor<copyType>& gm, const LocalTensor<copyType>& ub,
                                          int64_t nShapeAligned) {
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = block_.mShape;
    dataCopyParams.blockLen = block_.nShape * sizeof(copyType);
    dataCopyParams.srcStride = (nShapeAligned - block_.nShape) / (BLOCK_BYTES / sizeof(copyType));
    dataCopyParams.dstStride = (tiling.hiddenSize - block_.nShape) * sizeof(copyType);
    int64_t offset = block_.offset;
    DataCopyPad(gm[offset], ub, dataCopyParams);
  }

  template <typename copyType>
  __aicore__ inline void CopyTInput(const LocalTensor<copyType>& ub, const GlobalTensor<copyType>& gm, int64_t tIdx) {
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = block_.mShape;
    dataCopyParams.blockLen = block_.nShape * sizeof(copyType);
    dataCopyParams.srcStride = (tiling.hiddenSize - block_.nShape) * sizeof(copyType);
    dataCopyParams.dstStride = 0;
    int64_t offset = tIdx * tiling.hiddenSize * tiling.batch + block_.offset;
    DataCopyPadExtParams<copyType> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    DataCopyPad(ub, gm[offset], dataCopyParams, padParams);
  }

    template <typename copyType>
  __aicore__ inline void CopyTInput16Aligned(const LocalTensor<copyType>& ub, const GlobalTensor<copyType>& gm,
                                             int64_t tIdx) {
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = block_.mShape;
    dataCopyParams.blockLen = block_.nShape * sizeof(copyType);
    dataCopyParams.srcStride = (tiling.hiddenSize - block_.nShape) * sizeof(copyType);
    dataCopyParams.dstStride = (CeilDiv(block_.nShape, ALIGN_16_BYTES) * ALIGN_16_BYTES - block_.nShape) /
                               (BLOCK_BYTES / sizeof(copyType));
    int64_t offset = tIdx * tiling.hiddenSize * tiling.batch + block_.offset;
    DataCopyPadExtParams<copyType> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    DataCopyPad(ub, gm[offset], dataCopyParams, padParams);
  }

  __aicore__ inline void VToMTE2Sync() {
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
  }

  __aicore__ inline void MTE2ToVSync() {
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
  }

  __aicore__ inline void MTE2ToMTE3Sync() {
    event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
  }

  __aicore__ inline void VToMTE3Sync() {
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
  }

  __aicore__ inline void MTE3ToVSync() {
    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
  }

  __aicore__ inline void MTE3ToMTE2Sync() {
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
  }

  __aicore__ inline void InitializeDgatesTensors() {
    itTensor = itBuf.Get<float>();
    jtTensor = jtBuf.Get<float>();
    ftTensor = ftBuf.Get<float>();
    otTensor = otBuf.Get<float>();
    ctTensor = ctBuf.Get<float>();
    ctPreTensor = ctPreBuf.Get<float>();

    dyTensor = dyBuf.Get<float>();
    ditTensor = ditBuf.Get<float>();
    djtTensor = djtBuf.Get<float>();
    dftTensor = dftBuf.Get<float>();
    dotTensor = dotBuf.Get<float>();
    dcnextTensor = dcnextBuf.Get<float>();
    dhnextTensor = dhnextBuf.Get<float>();
    dhtTensor = dhtBuf.Get<float>();
    dctTensor = dctBuf.Get<float>();

    tmpJTensor = tmpJBuf.Get<float>();
    tanhTensor = tanhBuf.Get<float>();
    tmpCTensor = tmpCBuf.Get<float>();
    if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
        seqTensor = seqBuf.Get<float>();
    }
  }

  __aicore__ inline void ProcessDgates(int64_t tIdx, bool firstLoop, bool useInitCH) {
    auto mCoreIndex = GetBlockIdx();
    int64_t offset = tIdx * allCellSize;
    auto mixGm = outputGm.dgateGm[offset];
    InitializeDgatesTensors();
    for (int64_t totalIndex = 0; totalIndex < tiling.mCnt * tiling.nCnt; totalIndex++) {
      if (mCoreIndex == totalIndex % aivCoreNum) {
        core_.mCntIndex = totalIndex / tiling.nCnt;
        core_.nCntIndex = totalIndex % tiling.nCnt;
        core_.mSingleCoreShape = core_.mCntIndex == (tiling.mCnt - 1) ? tiling.singleCoreMTail : tiling.singleCoreM;
        core_.nSingleCoreShape = core_.nCntIndex == (tiling.nCnt - 1) ? tiling.singleCoreNTail : tiling.singleCoreN;
        core_.hiddenOffset = core_.mCntIndex * tiling.singleCoreM * tiling.hiddenSize +
                             core_.nCntIndex * tiling.singleCoreN;
        core_.gateOutOffset = core_.mCntIndex * tiling.singleCoreM * tiling.hiddenSize * LSTM_GATE_SIZE +
                              core_.nCntIndex * tiling.singleCoreN;
        SubProcessDgates(tIdx, mixGm, firstLoop, useInitCH);
      }
    }
  }

  __aicore__ inline void SubProcessDgates(int64_t tIdx, GlobalTensor<float>& mixGm, bool firstLoop, bool useInitCH) {
    int64_t incoreMCnt = CeilDiv(core_.mSingleCoreShape, tiling.baseM);
    int64_t incoreNCnt = CeilDiv(core_.nSingleCoreShape, tiling.baseN);
    int64_t baseMTail = core_.mSingleCoreShape - (incoreMCnt - 1) * tiling.baseM;
    int64_t baseNTail = core_.nSingleCoreShape - (incoreNCnt - 1) * tiling.baseN;
    for (int64_t mCntIndex = 0; mCntIndex < incoreMCnt; mCntIndex++) {
      block_.mCntIndex = mCntIndex;
      block_.mShape = mCntIndex == (incoreMCnt - 1) ? baseMTail : tiling.baseM;
      for (int64_t nCntIndex = 0; nCntIndex < incoreNCnt; nCntIndex++) {
        block_.nCntIndex = nCntIndex;
        block_.nShape = nCntIndex == (incoreNCnt - 1) ? baseNTail : tiling.baseN;
        block_.offset = core_.hiddenOffset + block_.mCntIndex * tiling.baseM * tiling.hiddenSize +
                        block_.nCntIndex * tiling.baseN;
        block_.gateOutOffset = core_.gateOutOffset + 
          block_.mCntIndex * tiling.baseM * tiling.hiddenSize * LSTM_GATE_SIZE + block_.nCntIndex * tiling.baseN;
        ProcessDgatesOnce(tIdx, mixGm, firstLoop, useInitCH);
      }
    }
  }

  __aicore__ inline void LoadDgatesInputData(int64_t tIdx, bool firstLoop, bool useInitCH, int64_t calcSizeAlign) {
    if constexpr (std::is_same<T, float>::value) {
      if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
        CopyTInput(seqTensor, inputGm.seqLengthGm, tIdx);
      }
      CopyTInput<float>(dyTensor, inputGm.dyGm, tIdx);
      CopyTInput<float>(otTensor, inputGm.oGm, tIdx);
      CopyTInput<float>(itTensor, inputGm.iGm, tIdx);
      CopyTInput<float>(jtTensor, inputGm.jGm, tIdx);
      CopyTInput<float>(ftTensor, inputGm.fGm, tIdx);
      CopyTInput<float>(ctTensor, inputGm.cGm, tIdx);
      if (useInitCH) {
          CopyTInput<float>(ctPreTensor, inputGm.initCGm, 0);
      } else if (tiling.direction == DIRECTION_FORWARD) {
          CopyTInput<float>(ctPreTensor, inputGm.cGm, tIdx - 1);
      } else {
          CopyTInput<float>(ctPreTensor, inputGm.cGm, tIdx + 1);
      }
      CopyTInput<float>(tanhTensor, inputGm.tanhctGm, tIdx);
      if (firstLoop) {
          CopyTInput<float>(dhnextTensor, inputGm.dhGm, 0);
          CopyTInput<float>(dcnextTensor, inputGm.dcGm, 0);
      } else {
          CopyTInput<float>(dhnextTensor, outputGm.dhPrevGm, 0);
          CopyTInput<float>(dcnextTensor, outputGm.dcPrevGm, 0);
      }
      MTE2ToVSync();
    } else if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
      if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
          CopyTInput<T>(seqTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.seqLengthGm, tIdx);
      }
      CopyTInput<T>(dyTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.dyGm, tIdx);
      CopyTInput<T>(otTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.oGm, tIdx);
      CopyTInput<T>(itTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.iGm, tIdx);
      CopyTInput<T>(jtTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.jGm, tIdx);
      CopyTInput<T>(ftTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.fGm, tIdx);
      CopyTInput<T>(ctTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.cGm, tIdx);
      
      if (useInitCH) {
          CopyTInput<T>(ctPreTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.initCGm, 0);
      } else if (tiling.direction == DIRECTION_FORWARD) {
          CopyTInput<T>(ctPreTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.cGm, tIdx - 1);
      } else {
          CopyTInput<T>(ctPreTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.cGm, tIdx + 1);
      }

      CopyTInput<T>(tanhTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.tanhctGm, tIdx);

      if (firstLoop) {
        CopyTInput<T>(dhnextTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.dhGm, 0);
        CopyTInput<T>(dcnextTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.dcGm, 0);
        MTE2ToVSync();
        Cast(dhnextTensor, dhnextTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
        Cast(dcnextTensor, dcnextTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
        PipeBarrier<PIPE_V>();
      } else {
        CopyTInput16Aligned<float>(dhnextTensor, outputGm.dhPrevFp32Gm, 0);
        CopyTInput16Aligned<float>(dcnextTensor, outputGm.dcPrevFp32Gm, 0);
      }
      
      MTE2ToVSync();

      if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
        Cast(seqTensor, seqTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      }
      Cast(dyTensor, dyTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(otTensor, otTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(itTensor, itTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(jtTensor, jtTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(ftTensor, ftTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(ctTensor, ctTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(ctPreTensor, ctPreTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(tanhTensor, tanhTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void ComputeDgates(int64_t calcSizeAlign) {
    // dh(t)
    Add(dhtTensor, dyTensor, dhnextTensor, calcSizeAlign);
    Mul(dctTensor, tanhTensor, tanhTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(tmpCTensor, otTensor, dhtTensor, calcSizeAlign);
    Muls(dctTensor, dctTensor, FLOAT_NEG_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(dctTensor, dctTensor, FLOAT_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(dctTensor, dctTensor, tmpCTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Add(dctTensor, dctTensor, dcnextTensor, calcSizeAlign);

    // do(t)
    Muls(otTensor, otTensor, FLOAT_NEG_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(otTensor, otTensor, FLOAT_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(dotTensor, tmpCTensor, tanhTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(dotTensor, dotTensor, otTensor, calcSizeAlign);

    // dj(t)
    Mul(djtTensor, jtTensor, jtTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Muls(djtTensor, djtTensor, FLOAT_NEG_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(djtTensor, djtTensor, FLOAT_ONE, calcSizeAlign);
    Mul(tmpJTensor, dctTensor, itTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(djtTensor, tmpJTensor, djtTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    
    // di(t)
    Muls(ditTensor, itTensor, FLOAT_NEG_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(ditTensor, ditTensor, FLOAT_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(ditTensor, ditTensor, jtTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(ditTensor, ditTensor, tmpJTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();

    // df(t)
    Muls(dftTensor, ftTensor, FLOAT_NEG_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(dftTensor, dftTensor, FLOAT_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(dftTensor, dftTensor, ftTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(dftTensor, dftTensor, dctTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(dftTensor, dftTensor, ctPreTensor, calcSizeAlign);
    
    // dc_prev
    Mul(dctTensor, dctTensor, ftTensor, calcSizeAlign);

    // apply seqlength
    if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
      PipeBarrier<PIPE_V>();
      Mul(ditTensor, ditTensor, seqTensor, calcSizeAlign);
      Mul(djtTensor, djtTensor, seqTensor, calcSizeAlign);
      Mul(dftTensor, dftTensor, seqTensor, calcSizeAlign);
      Mul(dotTensor, dotTensor, seqTensor, calcSizeAlign);
      Mul(dctTensor, dctTensor, seqTensor, calcSizeAlign);  // mask * dc_prev
      Muls(seqTensor, seqTensor, -1.0f, calcSizeAlign);
      PipeBarrier<PIPE_V>();
      Adds(seqTensor, seqTensor, 1.0f, calcSizeAlign);
      PipeBarrier<PIPE_V>();
      // (1 - mask) * dc_t, dc_t is equal to dc_next when padding
      Mul(dcnextTensor, dcnextTensor, seqTensor, calcSizeAlign);
      PipeBarrier<PIPE_V>();
      Add(dctTensor, dctTensor, dcnextTensor, calcSizeAlign);
    }
    VToMTE3Sync();
  }

    __aicore__ inline void StoreDgatesResults(int64_t tIdx, GlobalTensor<float>& mixGm, bool useInitCH, int64_t nShapeAligned, int64_t calcSizeAlign) {
    if (tiling.gateOrder == GATE_ORDER_IJFO) { // ijfo
      CopyOutputGate(mixGm, ditTensor, OFFSET_I, nShapeAligned);
      CopyOutputGate(mixGm, djtTensor, OFFSET_J, nShapeAligned);
      CopyOutputGate(mixGm, dftTensor, OFFSET_F, nShapeAligned);
      CopyOutputGate(mixGm, dotTensor, OFFSET_O, nShapeAligned);
    } else { // ifjo
      CopyOutputGate(mixGm, ditTensor, OFFSET_I, nShapeAligned);
      CopyOutputGate(mixGm, dftTensor, OFFSET_J, nShapeAligned);
      CopyOutputGate(mixGm, djtTensor, OFFSET_F, nShapeAligned);
      CopyOutputGate(mixGm, dotTensor, OFFSET_O, nShapeAligned);
    }

    // copy dc_prev out
    if (useInitCH && std::is_same<T, half>::value) {
      PipeBarrier<PIPE_V>();
      Cast(dctTensor.template ReinterpretCast<T>(), dctTensor, RoundMode::CAST_NONE, calcSizeAlign);
      VToMTE3Sync();
      CopyOutputDcPrev(outputGm.dcPrevGm, dctTensor.template ReinterpretCast<T>(), nShapeAligned);
    } else if (useInitCH && std::is_same<T, bfloat16_t>::value) {
      PipeBarrier<PIPE_V>();
      Cast(dctTensor.template ReinterpretCast<T>(), dctTensor, RoundMode::CAST_RINT, calcSizeAlign);
      VToMTE3Sync();
      CopyOutputDcPrev(outputGm.dcPrevGm, dctTensor.template ReinterpretCast<T>(), nShapeAligned);
    } else {
      CopyOutputDcPrev(outputGm.dcPrevFp32Gm, dctTensor, nShapeAligned);
    }
    MTE3ToMTE2Sync();
  }

  __aicore__ inline void ProcessDgatesOnce(int64_t tIdx, GlobalTensor<float>& mixGm, bool firstLoop, bool useInitCH) {
    int64_t nShapeAligned = Ceil(block_.nShape, calBlockSize) * calBlockSize;
    calcSizeAlign = block_.mShape * nShapeAligned;

    LoadDgatesInputData(tIdx, firstLoop, useInitCH, calcSizeAlign);
    
    ComputeDgates(calcSizeAlign);
    
    StoreDgatesResults(tIdx, mixGm, useInitCH, nShapeAligned, calcSizeAlign);
  }

    __aicore__ inline void ProcessReduce() {
    auto mCoreIndex = GetBlockIdx();
    if (mCoreIndex < tiling.nReduceCnt) {
      core_.reduceNCoreIndex = mCoreIndex;
      core_.nSingleCoreReduceShape = core_.reduceNCoreIndex == (tiling.nReduceCnt - 1) ? tiling.singleCoreReduceNTail :
                                     tiling.singleCoreReduceN;
      int64_t incoreNCnt = CeilDiv(core_.nSingleCoreReduceShape, tiling.baseReduceN);
      int64_t baseReduceNTail = core_.nSingleCoreReduceShape - (incoreNCnt - 1) * tiling.baseReduceN;
      for (int64_t nCntIndex = 0; nCntIndex < incoreNCnt; nCntIndex++) {
        reduceBlock_.nReduceCntIndx = nCntIndex;
        reduceBlock_.nReduceShape = nCntIndex == (incoreNCnt - 1) ? baseReduceNTail : tiling.baseReduceN;
        reduceBlock_.nReduceShapeAlign = CeilDiv(reduceBlock_.nReduceShape, ALIGN_8_BYTES) * ALIGN_8_BYTES;
        reduceBlock_.offset = core_.reduceNCoreIndex * tiling.singleCoreReduceN + reduceBlock_.nReduceCntIndx *
                              tiling.baseReduceN;
        int64_t maxReduceNumOnce = tiling.maxReduceNumOnce;
        int64_t reduceNumLeft = tiling.reduceBlockSize;
        while(reduceNumLeft >= 1) {
          // The system can reallocate cores each time to optimize performance.
          int64_t reduceBlockSize = reduceNumLeft > maxReduceNumOnce ? maxReduceNumOnce : reduceNumLeft;
          int64_t reduceSubRepeatTimes = CeilDiv(reduceNumLeft, reduceBlockSize);
          int64_t reduceBlockSizeTail = reduceNumLeft - (reduceSubRepeatTimes - 1) * maxReduceNumOnce;
          reduceNumLeft = (reduceNumLeft > 1 && reduceSubRepeatTimes > 1) ? reduceSubRepeatTimes : 0;
          SubProcessReduce(outputGm.dgateGm, outputGm.dbGm, reduceNumLeft, reduceBlockSize, reduceBlockSizeTail,
                           reduceSubRepeatTimes);
        }
      }
    }
  }

  __aicore__ inline void BinarySum(LocalTensor<float>& subDataTensor, int64_t start, int64_t end) {
    int64_t n = end - start;
    if (n <= 1) {
      return;
    }

    int64_t totalLevels = 0;
    for (int64_t tmp = n; tmp > 1; tmp = (tmp + 1) / BINARY_REDUCE_BASE) {
        totalLevels++;
    }

    for (int64_t level = 0; level < totalLevels; level++) {
      int64_t step = 1 << level;
      int64_t mergeStep = BINARY_REDUCE_BASE * step;
      for (int64_t i = start; i < end; i += mergeStep) {
          int64_t mid = i + step;
          int64_t rightEnd = (i + mergeStep) > end ? end : (i + mergeStep);
          if (mid < rightEnd) {
            Add(subDataTensor[i * reduceBlock_.nReduceShapeAlign],
                subDataTensor[i * reduceBlock_.nReduceShapeAlign],
                subDataTensor[mid * reduceBlock_.nReduceShapeAlign],
                reduceBlock_.nReduceShape);
            PipeBarrier<PIPE_V>();
          }
      }
    }
  }

  __aicore__ inline void SubProcessReduce(GlobalTensor<float>& mixGm, GlobalTensor<T>& outReduceGm,
                                          int64_t reduceNumLeft, int64_t reduceBlockSize, int64_t reduceBlockSizeTail,
                                          int64_t reduceSubRepeatTimes) {
    LocalTensor<float> subDataTensor = subDataBuf.Get<float>();
    for (int64_t mRepeatIdx = 0; mRepeatIdx < reduceSubRepeatTimes; mRepeatIdx++) {
      int64_t start = mRepeatIdx * reduceBlockSize;
      int64_t mLines = mRepeatIdx == (reduceSubRepeatTimes - 1) ? reduceBlockSizeTail : reduceBlockSize;
      int64_t end = start + mLines;
      int64_t copyInLoop = CeilDiv(mLines, MAX_COPY_LINES);
      int64_t offset = start * tiling.hiddenSize * LSTM_GATE_SIZE + reduceBlock_.offset;
      for (int64_t copyLoopIdx = 0; copyLoopIdx < copyInLoop; copyLoopIdx++) {
        int64_t curLines = copyLoopIdx == copyInLoop - 1 ? mLines % MAX_COPY_LINES : MAX_COPY_LINES;
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(curLines), 
          static_cast<uint32_t>(reduceBlock_.nReduceShape * FLOAT_BYTES),
          static_cast<uint32_t>((tiling.hiddenSize * LSTM_GATE_SIZE - reduceBlock_.nReduceShape) * FLOAT_BYTES),
          0u, 0u};
        int64_t currentOffset = offset + copyLoopIdx * LSTM_GATE_SIZE * tiling.hiddenSize * MAX_COPY_LINES;
        DataCopyPad(subDataTensor, mixGm[currentOffset], dataCopyParams, padInFloatParams);
      }
      MTE2ToVSync();
      BinarySum(subDataTensor, start - mRepeatIdx * reduceBlockSize, end - mRepeatIdx * reduceBlockSize);
      DataCopyExtParams outDataCopyParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(reduceBlock_.nReduceShape * paraBytes), 0u, 0u, 0u};

      DataCopyExtParams outFp32DataCopyParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(reduceBlock_.nReduceShape * FLOAT_BYTES), 0u, 0u, 0u};
      if (reduceNumLeft <= 1) {
        // Get reduce out
        if constexpr (std::is_same<T, half>::value) {
          Cast(subDataTensor.template ReinterpretCast<T>(), subDataTensor, RoundMode::CAST_NONE,
               reduceBlock_.nReduceShape);
          VToMTE3Sync();
          DataCopyPad(outputGm.dbGm[reduceBlock_.offset], subDataTensor.template ReinterpretCast<T>(),
                      outDataCopyParams);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
          Cast(subDataTensor.template ReinterpretCast<T>(), subDataTensor, RoundMode::CAST_RINT,
               reduceBlock_.nReduceShape);
          VToMTE3Sync();
          DataCopyPad(outputGm.dbGm[reduceBlock_.offset], subDataTensor.template ReinterpretCast<T>(),
                      outDataCopyParams);
        } else {
          VToMTE3Sync();
          DataCopyPad(outputGm.dbGm[reduceBlock_.offset], subDataTensor, outFp32DataCopyParams);
        }
      } else {
        // Get temp out
        VToMTE3Sync();
        DataCopyPad(mixGm[mRepeatIdx * tiling.hiddenSize * LSTM_GATE_SIZE + reduceBlock_.offset], subDataTensor,
                    outFp32DataCopyParams);
      }
      MTE3ToMTE2Sync();
    }
  }

  __aicore__ inline void ProcessDxhSplit(int64_t tIdx, bool useInitCH) {
    auto curBlockIdx = GetBlockIdx();
    int64_t factor = sizeof(T) == FLOAT_BYTES ? FLOAT_BYTES : HALF_AND_CAST_BYTES;
    int64_t ubSize = (tiling.ubSize - UB_RESERVED_SIZE) / factor;
    LocalTensor<float> dxhUb = subDataBuf.Get<float>();
    int64_t curCoreTaskNum = curBlockIdx < dxhInputTiling.splitPreCore ? (dxhInputTiling.splitTaskPerCore + 1) :
                            dxhInputTiling.splitTaskPerCore;
    int64_t startTaskId = curBlockIdx < dxhInputTiling.splitPreCore ?
                          (dxhInputTiling.splitTaskPerCore + 1) * curBlockIdx :
                          (dxhInputTiling.splitTaskPerCore * curBlockIdx + dxhInputTiling.splitPreCore);
    for (int64_t taskLoopId = 0; taskLoopId < curCoreTaskNum; taskLoopId++) {
      int64_t taskIdx = startTaskId + taskLoopId;
      int64_t curMLines = (taskIdx == dxhInputTiling.taskNum - 1) ?
                          dxhInputTiling.copyMLinesTail : dxhInputTiling.copyMLines;
      DataCopyExtParams dataCopyInDxParams{static_cast<uint16_t>(curMLines),
        static_cast<uint32_t>(tiling.inputSize * FLOAT_BYTES),
        static_cast<uint32_t>(tiling.hiddenSize * FLOAT_BYTES),
        static_cast<uint32_t>((tiling.inputSizeAligned - tiling.inputSize) / SRC_STRIDE_OFFSET_DIVISOR), 0};
      DataCopyExtParams dataCopyInDhParams{static_cast<uint16_t>(curMLines),
        static_cast<uint32_t>(tiling.hiddenSize * FLOAT_BYTES),
        static_cast<uint32_t>(tiling.inputSize * FLOAT_BYTES),
        static_cast<uint32_t>((tiling.hiddenSizeAligned - tiling.hiddenSize) / SRC_STRIDE_OFFSET_DIVISOR), 0};
      DataCopyPad(dxhUb, outputGm.dxhGm[taskIdx * (tiling.hiddenSize + tiling.inputSize) * dxhInputTiling.copyMLines],
                  dataCopyInDxParams, padInFloatParams);
      DataCopyPad(dxhUb[curMLines * tiling.inputSizeAligned],
        outputGm.dxhGm[taskIdx * (tiling.hiddenSize + tiling.inputSize) * dxhInputTiling.copyMLines + tiling.inputSize],
        dataCopyInDhParams, padInFloatParams);
      if constexpr (std::is_same<T, half>::value) {
        MTE2ToVSync();
        Cast(dxhUb.template ReinterpretCast<T>(), dxhUb, RoundMode::CAST_NONE, tiling.inputSizeAligned * curMLines);
        VToMTE3Sync();
      } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        MTE2ToVSync();
        Cast(dxhUb.template ReinterpretCast<T>(), dxhUb, RoundMode::CAST_RINT, tiling.inputSizeAligned * curMLines);
        VToMTE3Sync();
      } else {
        MTE2ToMTE3Sync();
      }
      // dx copy out
      DataCopyExtParams dataCopyDxParams{static_cast<uint16_t>(curMLines),
        static_cast<uint32_t>(tiling.inputSize * paraBytes), 
        static_cast<uint32_t>((tiling.inputSizeAligned - tiling.inputSize) / calBlockSize), 0, 0};
      DataCopyPad(
        outputGm.dxGm[tIdx * tiling.batch * tiling.inputSize + taskIdx * tiling.inputSize * dxhInputTiling.copyMLines],
        dxhUb.template ReinterpretCast<T>(), dataCopyDxParams);
      MTE3ToMTE2Sync();
      // dh copy out
      CopyOutDh(useInitCH, taskIdx, curMLines, dxhUb);
    }
  }

  __aicore__ inline void CopyOutDh(bool useInitCH, int64_t taskIdx, int64_t curMLines,
                                   const LocalTensor<float>& dxhUb) {
    auto dhStartAddr = dxhUb[curMLines * tiling.inputSizeAligned];
    if (useInitCH && std::is_same<T, half>::value) {
        MTE2ToVSync();
        Cast(dhStartAddr.template ReinterpretCast<T>(),
            dhStartAddr,
            RoundMode::CAST_NONE, tiling.hiddenSizeAligned * curMLines);
        VToMTE3Sync();
        DataCopyExtParams dataCopyDhParams{static_cast<uint16_t>(curMLines),
            static_cast<uint32_t>(tiling.hiddenSize * paraBytes),
            static_cast<uint32_t>((tiling.hiddenSizeAligned - tiling.hiddenSize) / calBlockSize), 0, 0};
        DataCopyPad(outputGm.dhPrevGm[taskIdx * tiling.hiddenSize * dxhInputTiling.copyMLines],
            dhStartAddr.template ReinterpretCast<T>(), dataCopyDhParams);
    } else if (useInitCH && std::is_same<T, bfloat16_t>::value) {
      MTE2ToVSync();
      Cast(dhStartAddr.template ReinterpretCast<T>(),
          dhStartAddr, RoundMode::CAST_RINT, tiling.hiddenSizeAligned * curMLines);
      VToMTE3Sync();
      DataCopyExtParams dataCopyDhParams{static_cast<uint16_t>(curMLines),
          static_cast<uint32_t>(tiling.hiddenSize * paraBytes),
          static_cast<uint32_t>((tiling.hiddenSizeAligned - tiling.hiddenSize) / calBlockSize), 0, 0};
      DataCopyPad(outputGm.dhPrevGm[taskIdx * tiling.hiddenSize * dxhInputTiling.copyMLines],
          dhStartAddr.template ReinterpretCast<T>(), dataCopyDhParams);
    } else {
      DataCopyExtParams dataCopyDhParams{static_cast<uint16_t>(curMLines),
          static_cast<uint32_t>(tiling.hiddenSize * FLOAT_BYTES),
          static_cast<uint32_t>((tiling.hiddenSizeAligned - tiling.hiddenSize) / SRC_STRIDE_OFFSET_DIVISOR), 0, 0};
      DataCopyPad(outputGm.dhPrevFp32Gm[taskIdx * tiling.hiddenSize * dxhInputTiling.copyMLines],
          dhStartAddr, dataCopyDhParams);
    }
    MTE3ToMTE2Sync();
  }

  __aicore__ inline void ProcessDxSplitLarge(int64_t tIdx) {
    // Process dx when inputsize and hiddensize is large.
    auto curBlockIdx = GetBlockIdx();

    int64_t curCoreTaskNumC = curBlockIdx < dxhInputTiling.splitPreCore ?
        (dxhInputTiling.splitTaskPerCore + 1) : dxhInputTiling.splitTaskPerCore;
    int64_t startTaskIdC = curBlockIdx < dxhInputTiling.splitPreCore ?
        (curCoreTaskNumC * curBlockIdx) : (curCoreTaskNumC * curBlockIdx + dxhInputTiling.splitPreCore);

    totalUb = subDataBuf.Get<uint8_t>();
    dxUb = totalUb.ReinterpretCast<T>();
    if constexpr (!std::is_same<T, float>::value) {
      dxFp32Ub = totalUb[dxhInputTiling.copyMLines * dxhInputTiling.copyNLength * sizeof(T)].template ReinterpretCast<float>();
    } else {
      dxFp32Ub = dxUb;
    }

    for (int64_t taskLoopId = 0; taskLoopId < curCoreTaskNumC; taskLoopId++) {
      int64_t taskIdx = startTaskIdC + taskLoopId;
      int64_t curMLines = (taskIdx == dxhInputTiling.taskNum - 1) ?
                          dxhInputTiling.copyMLinesTail : dxhInputTiling.copyMLines;
        
      for (int32_t nLoopIdx = 0; nLoopIdx < dxhInputTiling.nLoop; nLoopIdx++) {
        int64_t curCopyNLength = (nLoopIdx == dxhInputTiling.nLoop - 1) ?
                                dxhInputTiling.copyNLengthTail : dxhInputTiling.copyNLength;
        int64_t curCopyNLengthAligned = CeilDiv(curCopyNLength, calBlockSize) * calBlockSize;
        DataCopyExtParams dataCopyInDxParams{
            static_cast<uint16_t>(curMLines),
            static_cast<uint32_t>(curCopyNLength * FLOAT_BYTES), 
            static_cast<uint32_t>((tiling.hiddenSize + tiling.inputSize - curCopyNLength) * FLOAT_BYTES),
            static_cast<uint32_t>((curCopyNLengthAligned - curCopyNLength) / SRC_STRIDE_OFFSET_DIVISOR), 0
        };
        DataCopyPad(dxFp32Ub,
          outputGm.dxhGm[taskIdx * (tiling.hiddenSize + tiling.inputSize) * dxhInputTiling.copyMLines +
                          dxhInputTiling.copyNLength * nLoopIdx],
          dataCopyInDxParams, padInFloatParams);

        if constexpr (std::is_same<T, half>::value) {
          MTE2ToVSync();
          Cast(dxUb, dxFp32Ub, RoundMode::CAST_NONE, curCopyNLengthAligned * curMLines);
          VToMTE3Sync();
          VToMTE2Sync();
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
          MTE2ToVSync();
          Cast(dxUb, dxFp32Ub, RoundMode::CAST_RINT, curCopyNLengthAligned * curMLines);
          VToMTE3Sync();
          VToMTE2Sync();
        } else {
          MTE2ToMTE3Sync();
        }

        DataCopyExtParams dataCopyDxParams{
          static_cast<uint16_t>(curMLines),
          static_cast<uint32_t>(curCopyNLength * paraBytes),
          static_cast<uint32_t>((curCopyNLengthAligned - curCopyNLength) / calBlockSize), 0, 0
        };
        DataCopyPad(outputGm.dxGm[tIdx * tiling.batch * tiling.inputSize +
                                  taskIdx * tiling.inputSize * dxhInputTiling.copyMLines +
                                  dxhInputTiling.copyNLength * nLoopIdx],
                    dxUb, dataCopyDxParams);
        
        if constexpr (std::is_same<T, float>::value) {
          MTE3ToMTE2Sync();
        } else {
          MTE3ToVSync();
        }
      }
    }
  }

  __aicore__ inline void ProcessDhSequenceLengthMask(
    bool firstLoop,
    int64_t curMLines,
    int64_t curCopyNLength,
    int64_t curCopyNLengthAligned,
    int64_t tIdx,
    int64_t taskIdx,
    int64_t nLoopIdx,
    int64_t calBlockSize) {
    DataCopyExtParams dataCopyInSeqParams{
        static_cast<uint16_t>(curMLines),
        static_cast<uint32_t>(curCopyNLength * sizeof(T)),
        static_cast<uint32_t>((tiling.hiddenSize - curCopyNLength) * FLOAT_BYTES),
        static_cast<uint32_t>((curCopyNLengthAligned - curCopyNLength) / calBlockSize), 
        0
    };
    //  copy in seqlength and dhnext
    DataCopyPad(seqUb,
        inputGm.seqLengthGm[tIdx * tiling.hiddenSize * tiling.batch + 
                           taskIdx * tiling.hiddenSize * dxhHiddenTiling.copyMLines + 
                           dxhHiddenTiling.copyNLength * nLoopIdx],
        dataCopyInSeqParams, padInTParams);

    if (firstLoop) {
        DataCopyPad(dhNextUb,
            inputGm.dhGm[taskIdx * tiling.hiddenSize * dxhHiddenTiling.copyMLines + 
                        dxhHiddenTiling.copyNLength * nLoopIdx],
            dataCopyInSeqParams, padInTParams);
    } else {
        DataCopyExtParams dataCopyInDhNextParams {
            static_cast<uint16_t>(curMLines),
            static_cast<uint32_t>(curCopyNLength * FLOAT_BYTES),
            static_cast<uint32_t>((tiling.hiddenSize - curCopyNLength) * FLOAT_BYTES),
            static_cast<uint32_t>((curCopyNLengthAligned - curCopyNLength) / 8), 
            0
        };
        
        DataCopyPad(dhNextFp32Ub,
            outputGm.dhPrevFp32Gm[taskIdx * tiling.hiddenSize * dxhHiddenTiling.copyMLines + 
                                 dxhHiddenTiling.copyNLength * nLoopIdx],
            dataCopyInDhNextParams, padInFloatParams);
    }
    
    MTE2ToVSync();
    
    // cast
    if constexpr (!std::is_same<T, float>::value) {
        Cast(seqFp32Ub, seqUb, RoundMode::CAST_NONE, curCopyNLengthAligned * curMLines);
        if (firstLoop) {
            Cast(dhNextFp32Ub, dhNextUb, RoundMode::CAST_NONE, curCopyNLengthAligned * curMLines);
        }
        PipeBarrier<PIPE_V>();
    }
    
    // 1 - mask
    Muls(seqFp32Ub, seqFp32Ub, -1.0f, curCopyNLengthAligned * curMLines);
    PipeBarrier<PIPE_V>();
    Adds(seqFp32Ub, seqFp32Ub, 1.0f, curCopyNLengthAligned * curMLines);
    PipeBarrier<PIPE_V>();
    
    // dhPrev + (1 - mask) * dhNext
    Mul(dhNextFp32Ub, dhNextFp32Ub, seqFp32Ub, curCopyNLengthAligned * curMLines);
    PipeBarrier<PIPE_V>();
    Add(dhPrevFp32Ub, dhPrevFp32Ub, dhNextFp32Ub, curCopyNLengthAligned * curMLines);
  }

  __aicore__ inline void ProcessDhSplitLarge(int64_t tIdx, bool firstLoop, bool useInitCH) {
      // Process dh_prev when inputsize and hiddensize is large.
    auto curBlockIdx = GetBlockIdx();
    
    int64_t curCoreTaskNumH = curBlockIdx < dxhHiddenTiling.splitPreCore ?
                              (dxhHiddenTiling.splitTaskPerCore + 1) : dxhHiddenTiling.splitTaskPerCore;
    int64_t startTaskIdH = curBlockIdx < dxhHiddenTiling.splitPreCore ? 
                          (curCoreTaskNumH * curBlockIdx) : (curCoreTaskNumH * curBlockIdx + dxhHiddenTiling.splitPreCore);

    totalUb = subDataBuf.Get<uint8_t>();
    dhPrevUb = totalUb.ReinterpretCast<T>();
    if constexpr (!std::is_same<T, float>::value) {
      dhPrevFp32Ub = totalUb[dxhHiddenTiling.copyMLines * dxhHiddenTiling.copyNLength * sizeof(T)].template ReinterpretCast<float>();
      if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
        seqUb = totalUb[dxhHiddenTiling.copyMLines * dxhHiddenTiling.copyNLength * HALF_AND_CAST_BYTES].template ReinterpretCast<T>();
        dhNextUb = totalUb[dxhHiddenTiling.copyMLines * dxhHiddenTiling.copyNLength * (HALF_AND_CAST_BYTES + sizeof(T))].template ReinterpretCast<T>();
        seqFp32Ub = totalUb[dxhHiddenTiling.copyMLines * dxhHiddenTiling.copyNLength * SEQ_FP32_OFFSET].template ReinterpretCast<float>();
        dhNextFp32Ub = totalUb[dxhHiddenTiling.copyMLines * dxhHiddenTiling.copyNLength * DH_FP32_OFFSET].template ReinterpretCast<float>();
      }
    } else if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
      dhPrevFp32Ub = dhPrevUb;
      seqUb = totalUb[dxhHiddenTiling.copyMLines * dxhHiddenTiling.copyNLength * sizeof(T)].template ReinterpretCast<T>();
      seqFp32Ub = seqUb;
      dhNextUb = totalUb[dxhHiddenTiling.copyMLines * dxhHiddenTiling.copyNLength * sizeof(T) * DOUBLE].template ReinterpretCast<T>();
      dhNextFp32Ub = dhNextUb;
    }

    for (int64_t taskLoopId = 0; taskLoopId < curCoreTaskNumH; taskLoopId++) {
        int64_t taskIdx = startTaskIdH + taskLoopId;
        int64_t curMLines = (taskIdx == dxhHiddenTiling.taskNum - 1) ? 
                            dxhHiddenTiling.copyMLinesTail : dxhHiddenTiling.copyMLines;
      for (int64_t nLoopIdx = 0; nLoopIdx < dxhHiddenTiling.nLoop; nLoopIdx++) {
        int64_t curCopyNLength = (nLoopIdx == dxhHiddenTiling.nLoop - 1) ? 
                                dxhHiddenTiling.copyNLengthTail : dxhHiddenTiling.copyNLength;
        int64_t curCopyNLengthAligned = CeilDiv(curCopyNLength, calBlockSize) * calBlockSize;
        DataCopyExtParams dataCopyInDxParams{
          static_cast<uint16_t>(curMLines),
          static_cast<uint32_t>(curCopyNLength * FLOAT_BYTES),
          static_cast<uint32_t>((tiling.hiddenSize + tiling.inputSize - curCopyNLength) * FLOAT_BYTES),
          static_cast<uint32_t>((curCopyNLengthAligned - curCopyNLength) / SRC_STRIDE_OFFSET_DIVISOR), 0
        };
        // copy in dhprev
        DataCopyPad(dhPrevFp32Ub,
            outputGm.dxhGm[taskIdx * (tiling.hiddenSize + tiling.inputSize) * dxhHiddenTiling.copyMLines +
                          tiling.inputSize + dxhHiddenTiling.copyNLength * nLoopIdx], 
            dataCopyInDxParams, padInFloatParams);
        // dhprev do mask
        if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
          ProcessDhSequenceLengthMask(firstLoop, curMLines, curCopyNLength, curCopyNLengthAligned, tIdx, taskIdx,
                                      nLoopIdx, calBlockSize);
        }
        // copy out
        CopyOutDhPrevLarge(useInitCH, curMLines, curCopyNLength, curCopyNLengthAligned, taskIdx, nLoopIdx);
      }
    }
  }

  __aicore__ inline void CopyOutDhPrevLarge(bool useInitCH, int64_t curMLines, int64_t curCopyNLength,
                                      int64_t curCopyNLengthAligned, int64_t taskIdx, int64_t nLoopIdx) {
    if (useInitCH && !std::is_same<T, float>::value) {
      auto roundMode = std::is_same<T, half>::value ? RoundMode::CAST_NONE : RoundMode::CAST_RINT;
      if (tiling.isSeqLength != SEQ_LENGTH_ENABLED) {
        MTE2ToVSync();
      } else {
        PipeBarrier<PIPE_V>();
      }
      Cast(dhPrevUb, dhPrevFp32Ub, roundMode, curCopyNLengthAligned * curMLines);
      VToMTE3Sync();
      VToMTE2Sync();
      DataCopyExtParams dataCopyDhParams{
        static_cast<uint16_t>(curMLines),
        static_cast<uint32_t>(curCopyNLength * paraBytes),
        static_cast<uint32_t>((curCopyNLengthAligned - curCopyNLength) / calBlockSize), 0, 0
      };
      DataCopyPad(outputGm.dhPrevGm[taskIdx * tiling.hiddenSize * dxhHiddenTiling.copyMLines +
                                    dxhHiddenTiling.copyNLength * nLoopIdx],
                  dhPrevUb, dataCopyDhParams);
      MTE3ToVSync();
  } else {
      if (tiling.isSeqLength != SEQ_LENGTH_ENABLED) {
        MTE2ToMTE3Sync();
      } else {
        VToMTE3Sync();
      }
      DataCopyExtParams dataCopyDhParams{
        static_cast<uint16_t>(curMLines),
        static_cast<uint32_t>(curCopyNLength * FLOAT_BYTES), 
        static_cast<uint32_t>((curCopyNLengthAligned - curCopyNLength) / SRC_STRIDE_OFFSET_DIVISOR), 0, 0
      };
      DataCopyPad(outputGm.dhPrevFp32Gm[taskIdx * tiling.hiddenSize * dxhHiddenTiling.copyMLines +
                                        dxhHiddenTiling.copyNLength * nLoopIdx], 
                  dhPrevFp32Ub, dataCopyDhParams);
      if (tiling.isSeqLength == SEQ_LENGTH_ENABLED) {
        MTE3ToVSync();
      } else {
        MTE3ToMTE2Sync();
      }
    }
  }

  __aicore__ inline void ProcessDxhSplitLarge(int64_t tIdx, bool firstLoop, bool useInitCH) {
    // Process dx and dh_prev when inputsize and hiddensize is large.
    ProcessDxSplitLarge(tIdx);
    ProcessDhSplitLarge(tIdx, firstLoop, useInitCH);
  }

  __aicore__ inline void ProcessConcatXh() {
    auto curBlockIdx = GetBlockIdx();
    int64_t curCoreTaskNum = curBlockIdx < xhInputTiling.splitPreCore ?
      (xhInputTiling.splitTaskPerCore + 1) : xhInputTiling.splitTaskPerCore;
    int64_t startTaskId = curBlockIdx < xhInputTiling.splitPreCore ?
      (curCoreTaskNum * curBlockIdx) : (curCoreTaskNum * curBlockIdx + xhInputTiling.splitPreCore);
    totalUb = subDataBuf.Get<uint8_t>();
    xUb = totalUb.ReinterpretCast<T>();
    hUb = totalUb[xhInputTiling.copyMLines * tiling.inputSizeAligned * sizeof(T)].ReinterpretCast<T>();
    if constexpr (!std::is_same<T, float>::value) {
      xFp32Ub = totalUb[xhInputTiling.copyMLines * (tiling.inputSizeAligned + tiling.hiddenSizeAligned) * sizeof(T)].ReinterpretCast<float>();
      hFp32Ub = totalUb[xhInputTiling.copyMLines * (tiling.inputSizeAligned + tiling.hiddenSizeAligned) * sizeof(T) +
                        xhInputTiling.copyMLines * tiling.inputSizeAligned * FLOAT_BYTES].ReinterpretCast<float>();
      outXUb = xFp32Ub;
      outHUb = hFp32Ub;
    } else {
      outXUb = xUb;
      outHUb = hUb;
    }

    for (int64_t tIdx = 0; tIdx < tiling.timeStep; tIdx++) {
      bool useInitCH = (tIdx == 0 && tiling.direction == DIRECTION_FORWARD) ||
                        (tIdx == tiling.timeStep - 1 && tiling.direction == DIRECTION_BACKWARD);
      for (int64_t taskLoopId = 0; taskLoopId < curCoreTaskNum; taskLoopId++) {
        int64_t taskIdx = startTaskId + taskLoopId;
        int64_t curMLines = (taskIdx == xhInputTiling.taskNum - 1) ?
                            xhInputTiling.copyMLinesTail : xhInputTiling.copyMLines;
        // Need to implement loop for uint16 overflow protection.
        DataCopyExtParams dataCopyXParams{static_cast<uint16_t>(curMLines),
          static_cast<uint32_t>(tiling.inputSize * paraBytes), 0, 0, 0};
        DataCopyExtParams dataCopyHParams{static_cast<uint16_t>(curMLines),
          static_cast<uint32_t>(tiling.hiddenSize * paraBytes), 0, 0, 0};

        GlobalTensor<T> srcHGm;
        if (useInitCH) {
          srcHGm = inputGm.initHGm[taskIdx * tiling.hiddenSize * xhInputTiling.copyMLines];
        } else if (tiling.direction == DIRECTION_FORWARD) {
          srcHGm = inputGm.hGm[((tIdx - 1) * tiling.batch + taskIdx * xhInputTiling.copyMLines) * tiling.hiddenSize];
        } else {
          srcHGm = inputGm.hGm[((tIdx + 1) * tiling.batch + taskIdx * xhInputTiling.copyMLines) * tiling.hiddenSize];
        }
        DataCopyPad(xUb, inputGm.xGm[(tIdx * tiling.batch + taskIdx * xhInputTiling.copyMLines)* tiling.inputSize],
                    dataCopyXParams, padInTParams);
        DataCopyPad(hUb, srcHGm, dataCopyHParams, padInTParams);

        if constexpr (!std::is_same<T, float>::value) {
          MTE2ToVSync();
          Cast(xFp32Ub, xUb, RoundMode::CAST_NONE,
                          curMLines * tiling.inputSizeAligned);
          Cast(hFp32Ub, hUb, RoundMode::CAST_NONE,
                          curMLines * tiling.hiddenSizeAligned);
          VToMTE3Sync();
          VToMTE2Sync();
        } else {
          MTE2ToMTE3Sync();
        }
        DataCopyExtParams dataCopyOutXParams{static_cast<uint16_t>(curMLines),
          static_cast<uint32_t>(tiling.inputSize * FLOAT_BYTES),
          static_cast<uint32_t>((tiling.inputSizeAligned - tiling.inputSize) / SRC_STRIDE_OFFSET_DIVISOR),
          static_cast<uint32_t>(tiling.hiddenSize * FLOAT_BYTES), 0};
        DataCopyExtParams dataCopyOutHParams{static_cast<uint16_t>(curMLines),
          static_cast<uint32_t>(tiling.hiddenSize * FLOAT_BYTES),
          static_cast<uint32_t>((tiling.hiddenSizeAligned - tiling.hiddenSize) / SRC_STRIDE_OFFSET_DIVISOR),
          static_cast<uint32_t>(tiling.inputSize * FLOAT_BYTES), 0};
        DataCopyPad(outputGm.xhGm[(tIdx * tiling.batch + taskIdx * xhInputTiling.copyMLines) *
                    (tiling.inputSize + tiling.hiddenSize)], outXUb, dataCopyOutXParams);
        MTE3ToMTE2Sync();
        DataCopyPad(outputGm.xhGm[(tIdx * tiling.batch + taskIdx * xhInputTiling.copyMLines) *
                    (tiling.inputSize + tiling.hiddenSize) + tiling.inputSize], outHUb, dataCopyOutHParams);
        MTE3ToMTE2Sync();
      }
    }
  }

  __aicore__ inline void ProcessConcatXhLarge() {
    ProcessCopyInputXLarge();
    ProcessCopyHiddenHLarge();
  }
  __aicore__ inline void ProcessCopyInputXLarge() {
    // Concat dx and dh_prev when inputsize and hiddensize is large.
    auto curBlockIdx = GetBlockIdx();
    int64_t curCoreTaskNumC = curBlockIdx < xhInputTiling.splitPreCore ?
        (xhInputTiling.splitTaskPerCore + 1) : xhInputTiling.splitTaskPerCore;
    int64_t startTaskIdC = curBlockIdx < xhInputTiling.splitPreCore ?
        (curCoreTaskNumC * curBlockIdx) : (curCoreTaskNumC * curBlockIdx + xhInputTiling.splitPreCore);
    totalUb = subDataBuf.Get<uint8_t>();
    xUb = totalUb.ReinterpretCast<T>();
    if constexpr (!std::is_same<T, float>::value) {
      xFp32Ub = xUb[xhInputTiling.copyNLength * xhInputTiling.copyMLines].template ReinterpretCast<float>();
      outXUb = xFp32Ub;
    } else {
      outXUb = xUb;
    }
    for (int64_t taskLoopId = 0; taskLoopId < curCoreTaskNumC; taskLoopId++) {
      int64_t taskIdx = startTaskIdC + taskLoopId;
      int64_t curMLines = (taskIdx == xhInputTiling.taskNum - 1) ?
          xhInputTiling.copyMLinesTail : xhInputTiling.copyMLines;
      
      for (int64_t nLoopIdx = 0; nLoopIdx < xhInputTiling.nLoop; nLoopIdx++) {
        int64_t curCopyNLength = (nLoopIdx == xhInputTiling.nLoop - 1) ?
            xhInputTiling.copyNLengthTail : xhInputTiling.copyNLength;
        int64_t curCopyNLengthAligned = CeilDiv(curCopyNLength, calBlockSize) * calBlockSize;
        DataCopyExtParams dataCopyXParams{
          static_cast<uint16_t>(curMLines),
          static_cast<uint32_t>(curCopyNLength * paraBytes), 0, 0, 0
        };
        DataCopyPad(xUb, 
          inputGm.xGm[(taskIdx * xhInputTiling.copyMLines) * tiling.inputSize +
                      xhInputTiling.copyNLength * nLoopIdx],
          dataCopyXParams, padInTParams);
        if constexpr (!std::is_same<T, float>::value) {
          MTE2ToVSync();
          Cast(xFp32Ub, xUb, RoundMode::CAST_NONE, curMLines * curCopyNLengthAligned);
          VToMTE3Sync();
          VToMTE2Sync();
        } else {
          MTE2ToMTE3Sync();
        }
        DataCopyExtParams dataCopyOutXParams{
          static_cast<uint16_t>(curMLines),
          static_cast<uint32_t>(curCopyNLength * FLOAT_BYTES),
          static_cast<uint32_t>((curCopyNLengthAligned - curCopyNLength) / SRC_STRIDE_OFFSET_DIVISOR),
          static_cast<uint32_t>((tiling.hiddenSize + tiling.inputSize - curCopyNLength) * FLOAT_BYTES), 0
        };
        DataCopyPad(
          outputGm.xhGm[(taskIdx * xhInputTiling.copyMLines) * (tiling.inputSize + tiling.hiddenSize) +
                        nLoopIdx * xhInputTiling.copyNLength],
          outXUb, dataCopyOutXParams);
        
        if constexpr (!std::is_same<T, float>::value) {
          MTE3ToVSync();
        } else {
          MTE3ToMTE2Sync();
        }
      }
    }
  }

  __aicore__ inline void ProcessCopyHiddenHLarge() {
    auto curBlockIdx = GetBlockIdx();
    int64_t curCoreTaskNumH = curBlockIdx < xhHiddenTiling.splitPreCore ?
        (xhHiddenTiling.splitTaskPerCore + 1) : xhHiddenTiling.splitTaskPerCore;
    int64_t startTaskIdH = curBlockIdx < xhHiddenTiling.splitPreCore ?
        (curCoreTaskNumH * curBlockIdx) : (curCoreTaskNumH * curBlockIdx + xhHiddenTiling.splitPreCore);

    totalUb = subDataBuf.Get<uint8_t>();
    hUb = totalUb.ReinterpretCast<T>();
    if constexpr (!std::is_same<T, float>::value) {
      hFp32Ub = hUb[xhHiddenTiling.copyNLength * xhHiddenTiling.copyMLines].template ReinterpretCast<float>();
      outHUb = hFp32Ub;
    } else {
      outHUb = hUb;
    }

    for (int64_t tIdx = 0; tIdx < tiling.timeStep; tIdx++) {
      bool useInitCH = (tIdx == 0 && tiling.direction == DIRECTION_FORWARD) ||
        (tIdx == tiling.timeStep - 1 && tiling.direction == DIRECTION_BACKWARD);
      
      for (int64_t taskLoopId = 0; taskLoopId < curCoreTaskNumH; taskLoopId++) {
        int64_t taskIdx = startTaskIdH + taskLoopId;
        int64_t curMLines = (taskIdx == xhHiddenTiling.taskNum - 1) ?
          xhHiddenTiling.copyMLinesTail : xhHiddenTiling.copyMLines;
        
        for (int64_t nLoopIdx = 0; nLoopIdx < xhHiddenTiling.nLoop; nLoopIdx++) {
          int64_t curCopyNLength = (nLoopIdx == xhHiddenTiling.nLoop - 1) ?
            xhHiddenTiling.copyNLengthTail : xhHiddenTiling.copyNLength;
          int64_t curCopyNLengthAligned = CeilDiv(curCopyNLength, calBlockSize) * calBlockSize;
          
          DataCopyExtParams dataCopyHParams{
            static_cast<uint16_t>(curMLines),
            static_cast<uint32_t>(curCopyNLength * paraBytes), 0, 0, 0
          };

          GlobalTensor<T> srcHGm;
          if (useInitCH) {
            srcHGm = inputGm.initHGm[taskIdx * tiling.hiddenSize * xhHiddenTiling.copyMLines +
                                    nLoopIdx * xhHiddenTiling.copyNLength];
          } else if (tiling.direction == DIRECTION_FORWARD) {
            srcHGm = inputGm.hGm[((tIdx - 1) * tiling.batch +
              taskIdx * xhHiddenTiling.copyMLines) * tiling.hiddenSize + 
              nLoopIdx * xhHiddenTiling.copyNLength];
          } else {
            srcHGm = inputGm.hGm[((tIdx + 1) * tiling.batch +
              taskIdx * xhHiddenTiling.copyMLines) * tiling.hiddenSize + 
              nLoopIdx * xhHiddenTiling.copyNLength];
          }
          
          DataCopyPad(hUb, srcHGm, dataCopyHParams, padInTParams);
          
          if constexpr (!std::is_same<T, float>::value) {
            MTE2ToVSync();
            Cast(hFp32Ub, hUb, RoundMode::CAST_NONE, curMLines * curCopyNLengthAligned);
            VToMTE3Sync();
            VToMTE2Sync();
          } else {
            MTE2ToMTE3Sync();
          }
          
          DataCopyExtParams dataCopyOutHParams{
            static_cast<uint16_t>(curMLines),
            static_cast<uint32_t>(curCopyNLength * FLOAT_BYTES), 
            static_cast<uint32_t>((curCopyNLengthAligned - curCopyNLength) / SRC_STRIDE_OFFSET_DIVISOR),
            static_cast<uint32_t>((tiling.hiddenSize + tiling.inputSize - curCopyNLength) * FLOAT_BYTES),
            0
          };
          DataCopyPad(outputGm.xhGm[(tIdx * tiling.batch +
            taskIdx * xhHiddenTiling.copyMLines) * (tiling.inputSize + tiling.hiddenSize) +
            tiling.inputSize + nLoopIdx * xhHiddenTiling.copyNLength],
            outHUb, dataCopyOutHParams);
          
          if constexpr (!std::is_same<T, float>::value) {
            MTE3ToVSync();
          } else {
            MTE3ToMTE2Sync();
          }
        }
      }
    }
  }
};

#endif