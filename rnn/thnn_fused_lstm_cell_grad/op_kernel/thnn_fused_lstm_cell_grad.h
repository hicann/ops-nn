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
 * \file thnn_fused_lstm_cell_grad.h
 * \brief
 */
#ifndef _THNN_FUSED_LSTM_CELL_GRAD_TILING_H_
#define _THNN_FUSED_LSTM_CELL_GRAD_TILING_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
using namespace AscendC;

constexpr int64_t LSTM_GATE_SIZE = 4;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t HALF_BYTES = 2;
constexpr int64_t DEFAULT_AIV_CORE_NUM = 48;
constexpr int64_t UB_RESERVED_SIZE = 1024;
constexpr int64_t VECTOR_ALIGN_SIZE = 16;
constexpr int64_t ALIGN_FLOAT_NUM = 8;
constexpr int64_t BINARY_REDUCE_BASE = 2;
constexpr float FLOAT_ONE = 1.0f;
constexpr float FLOAT_NEG_ONE = -1.0f;
constexpr int64_t OFFSET_I = 0;
constexpr int64_t OFFSET_F = 1;
constexpr int64_t OFFSET_J = 2;
constexpr int64_t OFFSET_O = 3;
constexpr int64_t MAX_COPY_LINES = 4095;

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

template <typename T>
class ThnnFusedLstmCellGrad {
 public:
  __aicore__ inline ThnnFusedLstmCellGrad(){};
  __aicore__ inline void Process() {
    InitVectorBuf();
    ProcessDgates();  // Calcu dgates
    if (tiling.isBias) {
      SyncAll();
      InitSubDataBuf();
      ProcessReduce();
      if constexpr (!std::is_same<T, float>::value) {
        SyncAll();
        CastDgatesBack();
      }
    }
  }

  __aicore__ inline void Init(GM_ADDR dhy, GM_ADDR dc, GM_ADDR cx, GM_ADDR cy, GM_ADDR storage, GM_ADDR dgates,
      GM_ADDR dc_prev, GM_ADDR db, const ThnnFusedLstmCellGradTilingData* __restrict rnnTiling, GM_ADDR workspace,
      TPipe* inputPipe) {
    pipe = inputPipe;
    ascendc_assert(GetBlockNum() != 0 && "block dim can not be zero!");
    aivCoreNum = GetBlockNum();
    tiling = *rnnTiling;
    InitGlobalBuffers(dhy, dc, cx, cy, storage, dgates, dc_prev, db, workspace);
    InitVars();
  }
 public:
  TPipe *pipe;

 private:
  struct OutputGm {
    __aicore__ inline OutputGm() = default;
    AscendC::GlobalTensor<T> dbGm;
    AscendC::GlobalTensor<T> dcPrevGm;
    AscendC::GlobalTensor<T> dgatesGm; 
    AscendC::GlobalTensor<float> dgatesTempGm; // workspace
    AscendC::GlobalTensor<float> reduceTempGm; // workspace
  };

  // input GlobalTensors
  struct InputGm {
    __aicore__ inline InputGm() = default;
    AscendC::GlobalTensor<T> cxGm;
    AscendC::GlobalTensor<T> cyGm;
    AscendC::GlobalTensor<T> dhyGm;
    AscendC::GlobalTensor<T> dcGm;
    AscendC::GlobalTensor<T> storageGm;
  };

  TBuf<AscendC::TPosition::VECCALC> iBuf;
  TBuf<AscendC::TPosition::VECCALC> jBuf;
  TBuf<AscendC::TPosition::VECCALC> fBuf;
  TBuf<AscendC::TPosition::VECCALC> oBuf;
  TBuf<AscendC::TPosition::VECCALC> cBuf;
  TBuf<AscendC::TPosition::VECCALC> cPreBuf;
  TBuf<AscendC::TPosition::VECCALC> diBuf;
  TBuf<AscendC::TPosition::VECCALC> djBuf;
  TBuf<AscendC::TPosition::VECCALC> dfBuf;
  TBuf<AscendC::TPosition::VECCALC> doBuf;
  TBuf<AscendC::TPosition::VECCALC> dcnextBuf;
  TBuf<AscendC::TPosition::VECCALC> dhyBuf;
  TBuf<AscendC::TPosition::VECCALC> dcBuf;
  TBuf<AscendC::TPosition::VECCALC> tanhBuf;
  TBuf<AscendC::TPosition::VECCALC> tmpDiBuf;
  TBuf<AscendC::TPosition::VECCALC> tmpDfBuf;
  TBuf<AscendC::TPosition::VECCALC> tmpDcBuf;
  TBuf<AscendC::TPosition::VECCALC> tmpDoBuf;
  TBuf<AscendC::TPosition::VECCALC> subDataBuf;
  TBuf<AscendC::TPosition::VECCALC> dgatesBuf;
  TBuf<AscendC::TPosition::VECCALC> dgatesCastBuf;

  // dgates tensor
  LocalTensor<float> iTensor;
  LocalTensor<float> jTensor;
  LocalTensor<float> fTensor;
  LocalTensor<float> oTensor;
  LocalTensor<float> cTensor; // cell out
  LocalTensor<float> cPreTensor; // cell in
  LocalTensor<float> dhyTensor;
  LocalTensor<float> diTensor;
  LocalTensor<float> djTensor;
  LocalTensor<float> dfTensor;
  LocalTensor<float> doTensor;
  LocalTensor<float> dcnextTensor;
  LocalTensor<float> dcTensor;
  LocalTensor<float> tanhTensor;
  LocalTensor<float> tmpCTensor;
  LocalTensor<float> tmpDiTensor;
  LocalTensor<float> tmpDfTensor;
  LocalTensor<float> tmpDcTensor;
  LocalTensor<float> tmpDoTensor;

  blockParams block_;
  singCloreParams core_;
  blockReduceParams reduceBlock_;

  OutputGm outputGm;
  InputGm inputGm;

  int64_t calcSize = 0;
  int64_t calcSizeAlign = 0;

  int64_t allCellSize = 0;
  int64_t oneCellSize = 0;

  ThnnFusedLstmCellGradTilingData tiling;

  int64_t blockSize = 0;
  int64_t calBlockSize = 0;
  int64_t aivCoreNum = DEFAULT_AIV_CORE_NUM;
  int64_t paraBytes = sizeof(T);
  DataCopyPadExtParams<float> padInFloatParams = {false, 0, 0, 0};
  DataCopyPadExtParams<T> padInTParams = {false, 0, 0, 0};

 private:
  __aicore__ inline void InitVars() {
    blockSize = BLOCK_BYTES / paraBytes;
    calBlockSize = BLOCK_BYTES / paraBytes;
  }

  __aicore__ inline void InitVectorBuf() {
    // Apply by fp32
    pipe->InitBuffer(iBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN); // BaseN is aligned.
    pipe->InitBuffer(jBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    // Init Local Tensors
    pipe->InitBuffer(fBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(oBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(cBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(cPreBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);

    pipe->InitBuffer(dcnextBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(dhyBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);

    pipe->InitBuffer(diBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(djBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(dfBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(doBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(dcBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);

    pipe->InitBuffer(tanhBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(tmpDiBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(tmpDfBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(tmpDcBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
    pipe->InitBuffer(tmpDoBuf, FLOAT_BYTES * tiling.baseM * tiling.baseN);
  }

  __aicore__ inline void InitSubDataBuf() {
    pipe->Reset();
    pipe->InitBuffer(subDataBuf, tiling.ubSize);
  }

  __aicore__ inline void CastDgatesBack() {
    pipe->Reset();
    int64_t maxCalcNum = (tiling.ubSize - UB_RESERVED_SIZE) / (sizeof(T) + FLOAT_BYTES) / VECTOR_ALIGN_SIZE *
                          VECTOR_ALIGN_SIZE;
    pipe->InitBuffer(dgatesBuf, HALF_BYTES * maxCalcNum);
    pipe->InitBuffer(dgatesCastBuf, FLOAT_BYTES * maxCalcNum);
    LocalTensor<T> dgatesTensor = dgatesBuf.Get<T>();
    LocalTensor<float> dgatesCastTensor = dgatesCastBuf.Get<float>();
    int64_t totalNum = tiling.batch * tiling.hiddenSize * LSTM_GATE_SIZE;
    int64_t totalLoops = CeilDiv(totalNum, maxCalcNum);
    int64_t calcTail = totalNum - (totalLoops - 1) * maxCalcNum;
    for (int64_t loopIndex = 0; loopIndex < totalLoops; loopIndex++) {
      if (GetBlockIdx() == loopIndex % GetBlockNum()) {
        int64_t calcNum = (loopIndex == totalLoops - 1) ? calcTail : maxCalcNum;
        DataCopyExtParams copyParamsIn = {1, static_cast<uint32_t>(calcNum * FLOAT_BYTES), 0, 0, 0};
        DataCopyPad(dgatesCastTensor, outputGm.dgatesTempGm[loopIndex * maxCalcNum], copyParamsIn, padInFloatParams);
        MTE2ToVSync();
        if constexpr (std::is_same<T, half>::value) {
          Cast(dgatesTensor, dgatesCastTensor, RoundMode::CAST_NONE, calcNum);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
          Cast(dgatesTensor, dgatesCastTensor, RoundMode::CAST_RINT, calcNum);
        }
        VToMTE3Sync();
        VToMTE2Sync();
        DataCopyExtParams copyParamsOut = {1, static_cast<uint32_t>(calcNum * HALF_BYTES), 0, 0, 0};
        DataCopyPad(outputGm.dgatesGm[loopIndex * maxCalcNum], dgatesTensor, copyParamsOut);
        MTE3ToVSync();
      }
    }
  }

  __aicore__ inline void InitGlobalBuffers(GM_ADDR dhy, GM_ADDR dc, GM_ADDR cx, GM_ADDR cy, GM_ADDR storage,
      GM_ADDR dgates, GM_ADDR dc_prev, GM_ADDR db, GM_ADDR workspace) {
    oneCellSize = tiling.batch * tiling.hiddenSize;
    allCellSize = oneCellSize * LSTM_GATE_SIZE;

    inputGm.cxGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cx), tiling.batch * tiling.hiddenSize);
    inputGm.cyGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cy),
                                tiling.batch * tiling.hiddenSize);
    inputGm.dhyGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dhy),
                                  tiling.batch * tiling.hiddenSize);
    inputGm.dcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dc), tiling.batch * tiling.hiddenSize);

    inputGm.storageGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(storage), tiling.batch * tiling.hiddenSize);

    if (tiling.isBias) {
      outputGm.dbGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(db), LSTM_GATE_SIZE * tiling.hiddenSize);
    }
    outputGm.dcPrevGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dc_prev),
                                    tiling.batch * tiling.hiddenSize);
    outputGm.dgatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dgates),
                                     tiling.batch * LSTM_GATE_SIZE * tiling.hiddenSize);
    int64_t reduceOffset = (tiling.isBias == 1 && !std::is_same<T, float>::value) ? tiling.batch * tiling.hiddenSize * LSTM_GATE_SIZE  * FLOAT_BYTES : 0;
    if (tiling.isBias == 1 && !std::is_same<T, float>::value) {
      outputGm.dgatesTempGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace),
                                           tiling.batch * tiling.hiddenSize * LSTM_GATE_SIZE);
    }
    if (tiling.batch > tiling.maxReduceNumOnce) {
      outputGm.reduceTempGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + reduceOffset),
                                           CeilDiv(tiling.batch, tiling.maxReduceNumOnce) * tiling.hiddenSize * LSTM_GATE_SIZE);
    }
  }

  template <typename copyType>
  __aicore__ inline void CopyOutputGate(const GlobalTensor<copyType>& gm, const LocalTensor<copyType>& ub, int64_t gateIdx,
                                        int64_t nShapeAligned) {
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = block_.mShape;
    dataCopyParams.blockLen = block_.nShape * sizeof(copyType);
    dataCopyParams.srcStride = (nShapeAligned - block_.nShape)  * sizeof(copyType) / BLOCK_BYTES;
    dataCopyParams.dstStride = (tiling.hiddenSize * LSTM_GATE_SIZE - block_.nShape) * sizeof(copyType);
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
  __aicore__ inline void CopyInput(const LocalTensor<copyType>& ub, const GlobalTensor<copyType>& gm) {
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = block_.mShape;
    dataCopyParams.blockLen = block_.nShape * sizeof(copyType);
    dataCopyParams.srcStride = (tiling.hiddenSize - block_.nShape) * sizeof(copyType);
    dataCopyParams.dstStride = 0;
    int64_t offset = block_.offset;
    DataCopyPadExtParams<copyType> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    DataCopyPad(ub, gm[offset], dataCopyParams, padParams);
  }

    template <typename copyType>
  __aicore__ inline void CopyGateInput(const LocalTensor<copyType>& ub, const GlobalTensor<copyType>& gm, int64_t gateIndex) {
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = block_.mShape;
    dataCopyParams.blockLen = block_.nShape * sizeof(copyType);
    dataCopyParams.srcStride = (LSTM_GATE_SIZE * tiling.hiddenSize - block_.nShape) * sizeof(copyType);
    dataCopyParams.dstStride = 0;
    int64_t offset = block_.gateOutOffset + gateIndex * tiling.hiddenSize;
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
    iTensor = iBuf.Get<float>();
    jTensor = jBuf.Get<float>();
    fTensor = fBuf.Get<float>();
    oTensor = oBuf.Get<float>();
    cTensor = cBuf.Get<float>();
    cPreTensor = cPreBuf.Get<float>();

    dhyTensor = dhyBuf.Get<float>();
    diTensor = diBuf.Get<float>();
    djTensor = djBuf.Get<float>();
    dfTensor = dfBuf.Get<float>();
    doTensor = doBuf.Get<float>();
    dcnextTensor = dcnextBuf.Get<float>();
    dcTensor = dcBuf.Get<float>();
    tanhTensor = tanhBuf.Get<float>();

    tmpDiTensor = tmpDiBuf.Get<float>();
    tmpDfTensor = tmpDfBuf.Get<float>();
    tmpDcTensor = tmpDcBuf.Get<float>();
    tmpDoTensor = tmpDoBuf.Get<float>();
  }

  __aicore__ inline void ProcessDgates() {
    auto mCoreIndex = GetBlockIdx();
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
        SubProcessDgates();
      }
    }
  }

  __aicore__ inline void SubProcessDgates() {
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
        ProcessDgatesOnce();
      }
    }
  }

  __aicore__ inline void LoadDgatesInputData(int64_t calcSizeAlign) {
    if constexpr (std::is_same<T, float>::value) {
      CopyInput<float>(dhyTensor, inputGm.dhyGm);
      CopyGateInput<float>(iTensor, inputGm.storageGm, OFFSET_I);
      CopyGateInput<float>(fTensor, inputGm.storageGm, OFFSET_F);
      CopyGateInput<float>(jTensor, inputGm.storageGm, OFFSET_J);
      CopyGateInput<float>(oTensor, inputGm.storageGm, OFFSET_O);
      CopyInput<float>(cTensor, inputGm.cyGm);
      CopyInput<float>(cPreTensor, inputGm.cxGm);
      CopyInput<float>(dcnextTensor, inputGm.dcGm);
      MTE2ToVSync();
    } else if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
      CopyInput<T>(dhyTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.dhyGm);
      CopyGateInput<T>(iTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.storageGm, OFFSET_I);
      CopyGateInput<T>(fTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.storageGm, OFFSET_F);
      CopyGateInput<T>(jTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.storageGm, OFFSET_J);
      CopyGateInput<T>(oTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.storageGm, OFFSET_O);
      CopyInput<T>(cTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.cyGm);
      CopyInput<T>(cPreTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.cxGm);
      CopyInput<T>(dcnextTensor.template ReinterpretCast<T>()[calcSizeAlign], inputGm.dcGm);
      MTE2ToVSync();
      Cast(dcnextTensor, dcnextTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(dhyTensor, dhyTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(oTensor, oTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(iTensor, iTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(jTensor, jTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(fTensor, fTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(cTensor, cTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      Cast(cPreTensor, cPreTensor.template ReinterpretCast<T>()[calcSizeAlign], RoundMode::CAST_NONE, calcSizeAlign);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void ComputeDgates(int64_t calcSizeAlign) {
    // tanh
    Tanh(tanhTensor, cTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(dcTensor, tanhTensor, tanhTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();

    // dcTemp
    Mul(tmpDcTensor, oTensor, dhyTensor, calcSizeAlign);
    Muls(dcTensor, dcTensor, FLOAT_NEG_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(dcTensor, dcTensor, FLOAT_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(dcTensor, dcTensor, tmpDcTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Add(dcTensor, dcTensor, dcnextTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();

    // temp
    Mul(tmpDiTensor, dcTensor, jTensor, calcSizeAlign);
    Mul(tmpDfTensor, dcTensor, cPreTensor, calcSizeAlign);
    Mul(tmpDcTensor, dcTensor, iTensor, calcSizeAlign);
    Mul(tmpDoTensor, dhyTensor, tanhTensor, calcSizeAlign);

    // dcPrev
    Mul(dcTensor, dcTensor, fTensor, calcSizeAlign);

    // di
    Muls(diTensor, iTensor, FLOAT_NEG_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(diTensor, diTensor, FLOAT_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(diTensor, tmpDiTensor, diTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(diTensor, diTensor, iTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();

    // df
    Muls(dfTensor, fTensor, FLOAT_NEG_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(dfTensor, dfTensor, FLOAT_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(dfTensor, tmpDfTensor, dfTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(dfTensor, dfTensor, fTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();

    // dg
    Mul(djTensor, jTensor, jTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Muls(djTensor, djTensor, FLOAT_NEG_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(djTensor, djTensor, FLOAT_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(djTensor, tmpDcTensor, djTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();

    // do
    Muls(tmpDiTensor, oTensor, FLOAT_NEG_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(tmpDiTensor, tmpDiTensor, FLOAT_ONE, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(doTensor, tmpDoTensor, tmpDiTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Mul(doTensor, doTensor, oTensor, calcSizeAlign);
  }

  __aicore__ inline void StoreDgatesResults(int64_t nShapeAligned, int64_t calcSizeAlign) {
    // ifjo
    if ((tiling.isBias == 1 && !std::is_same<T, float>::value)) {
      VToMTE3Sync();
      CopyOutputGate(outputGm.dgatesTempGm, diTensor, 0, nShapeAligned);
      CopyOutputGate(outputGm.dgatesTempGm, dfTensor, 1, nShapeAligned);
      CopyOutputGate(outputGm.dgatesTempGm, djTensor, 2, nShapeAligned);
      CopyOutputGate(outputGm.dgatesTempGm, doTensor, 3, nShapeAligned);
    } else if constexpr (std::is_same<T, float>::value) {
      VToMTE3Sync();
      CopyOutputGate(outputGm.dgatesGm, diTensor, 0, nShapeAligned);
      CopyOutputGate(outputGm.dgatesGm, dfTensor, 1, nShapeAligned);
      CopyOutputGate(outputGm.dgatesGm, djTensor, 2, nShapeAligned);
      CopyOutputGate(outputGm.dgatesGm, doTensor, 3, nShapeAligned);
    } else if (tiling.isBias != 1 && !std::is_same<T, float>::value) {
      auto roundMode = std::is_same<T, bfloat16_t>::value ? 
          RoundMode::CAST_RINT : RoundMode::CAST_NONE;
      PipeBarrier<PIPE_V>();
      Cast(diTensor.template ReinterpretCast<T>(), diTensor, roundMode, calcSizeAlign);
      Cast(dfTensor.template ReinterpretCast<T>(), dfTensor, roundMode, calcSizeAlign);
      Cast(djTensor.template ReinterpretCast<T>(), djTensor, roundMode, calcSizeAlign);
      Cast(doTensor.template ReinterpretCast<T>(), doTensor, roundMode, calcSizeAlign);
      VToMTE3Sync();
      CopyOutputGate<T>(outputGm.dgatesGm, diTensor.template ReinterpretCast<T>(), OFFSET_I, nShapeAligned);
      CopyOutputGate<T>(outputGm.dgatesGm, dfTensor.template ReinterpretCast<T>(), OFFSET_F, nShapeAligned);
      CopyOutputGate<T>(outputGm.dgatesGm, djTensor.template ReinterpretCast<T>(), OFFSET_J, nShapeAligned);
      CopyOutputGate<T>(outputGm.dgatesGm, doTensor.template ReinterpretCast<T>(), OFFSET_O, nShapeAligned);
    }

    // copy dc_prev out
    if constexpr (std::is_same<T, half>::value) {
      PipeBarrier<PIPE_V>();
      Cast(dcTensor.template ReinterpretCast<T>(), dcTensor, RoundMode::CAST_NONE, calcSizeAlign);
      VToMTE3Sync();
      CopyOutputDcPrev(outputGm.dcPrevGm, dcTensor.template ReinterpretCast<T>(), nShapeAligned);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
      PipeBarrier<PIPE_V>();
      Cast(dcTensor.template ReinterpretCast<T>(), dcTensor, RoundMode::CAST_RINT, calcSizeAlign);
      VToMTE3Sync();
      CopyOutputDcPrev(outputGm.dcPrevGm, dcTensor.template ReinterpretCast<T>(), nShapeAligned);
    } else {
      CopyOutputDcPrev(outputGm.dcPrevGm, dcTensor, nShapeAligned);
    }
    MTE3ToMTE2Sync();
  }

  __aicore__ inline void ProcessDgatesOnce() {
    int64_t nShapeAligned = CeilDiv(block_.nShape, calBlockSize) * calBlockSize;
    calcSizeAlign = block_.mShape * nShapeAligned;

    LoadDgatesInputData(calcSizeAlign);
    ComputeDgates(calcSizeAlign);
    StoreDgatesResults(nShapeAligned, calcSizeAlign);
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
        reduceBlock_.nReduceShapeAlign = CeilDiv(reduceBlock_.nReduceShape, ALIGN_FLOAT_NUM) * ALIGN_FLOAT_NUM;
        reduceBlock_.offset = core_.reduceNCoreIndex * tiling.singleCoreReduceN + reduceBlock_.nReduceCntIndx *
                              tiling.baseReduceN;
        int64_t maxReduceNumOnce = tiling.maxReduceNumOnce;
        int64_t reduceNumLeft = tiling.reduceBlockSize;
        int64_t loopTime = 0;
        GlobalTensor<T> inputReduceGm = outputGm.dgatesGm;
        GlobalTensor<T> outReduceGm = outputGm.dbGm;
        while (reduceNumLeft >= 1) {
          // The system can reallocate cores each time to optimize performance.
          int64_t reduceBlockSize = reduceNumLeft > maxReduceNumOnce ? maxReduceNumOnce : reduceNumLeft;
          int64_t reduceSubRepeatTimes = CeilDiv(reduceNumLeft, reduceBlockSize);
          int64_t reduceBlockSizeTail = reduceNumLeft - (reduceSubRepeatTimes - 1) * maxReduceNumOnce;
          reduceNumLeft = (reduceNumLeft > 1 && reduceSubRepeatTimes > 1) ? reduceSubRepeatTimes : 0;
          SubProcessReduce(reduceNumLeft, reduceBlockSize, reduceBlockSizeTail, reduceSubRepeatTimes, loopTime);
          loopTime++;
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

  __aicore__ inline void LoadDataForReduce(int64_t start, int64_t mLines, int64_t loopTime, 
                                          LocalTensor<float> subDataTensor) {
    int64_t offset = start * tiling.hiddenSize * LSTM_GATE_SIZE + reduceBlock_.offset;
    int64_t copyInLoop = CeilDiv(mLines, MAX_COPY_LINES);
    for (int64_t copyLoopIdx = 0; copyLoopIdx < copyInLoop; copyLoopIdx++) {
      int64_t curLines = copyLoopIdx == copyInLoop - 1 ? mLines % MAX_COPY_LINES : MAX_COPY_LINES;
      DataCopyExtParams dataCopyParams{static_cast<uint16_t>(curLines), 
        static_cast<uint32_t>(reduceBlock_.nReduceShape * FLOAT_BYTES),
        static_cast<uint32_t>((tiling.hiddenSize * LSTM_GATE_SIZE - reduceBlock_.nReduceShape) * FLOAT_BYTES),
        0u, 0u};
      int64_t currentOffset = offset + copyLoopIdx * LSTM_GATE_SIZE * tiling.hiddenSize * MAX_COPY_LINES;
      if (loopTime >= 1) {
        DataCopyPad(subDataTensor, outputGm.reduceTempGm[currentOffset], dataCopyParams, padInFloatParams);
      } else if constexpr (!std::is_same<T, float>::value) {
        DataCopyPad(subDataTensor, outputGm.dgatesTempGm[currentOffset], dataCopyParams, padInFloatParams);
      } else {
        DataCopyPad(subDataTensor, outputGm.dgatesGm[currentOffset], dataCopyParams, padInFloatParams);
      }
    }
  }

  __aicore__ inline void StoreReduceResult(int64_t mRepeatIdx, int64_t reduceNumLeft,
                                          LocalTensor<float> subDataTensor) {
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
      int64_t tempOffset = mRepeatIdx * tiling.hiddenSize * LSTM_GATE_SIZE + reduceBlock_.offset;
      VToMTE3Sync();
      DataCopyPad(outputGm.reduceTempGm[tempOffset], subDataTensor, outFp32DataCopyParams);
    }
  }

  __aicore__ inline void SubProcessReduce(int64_t reduceNumLeft, int64_t reduceBlockSize, 
                                          int64_t reduceBlockSizeTail, int64_t reduceSubRepeatTimes, 
                                          int64_t loopTime) {
    LocalTensor<float> subDataTensor = subDataBuf.Get<float>();
    for (int64_t mRepeatIdx = 0; mRepeatIdx < reduceSubRepeatTimes; mRepeatIdx++) {
      int64_t start = mRepeatIdx * reduceBlockSize;
      int64_t mLines = mRepeatIdx == (reduceSubRepeatTimes - 1) ? reduceBlockSizeTail : reduceBlockSize;
      int64_t end = start + mLines;
      LoadDataForReduce(start, mLines, loopTime, subDataTensor);
      MTE2ToVSync();
      BinarySum(subDataTensor, start - mRepeatIdx * reduceBlockSize, end - mRepeatIdx * reduceBlockSize);
      StoreReduceResult(mRepeatIdx, reduceNumLeft, subDataTensor);
      MTE3ToMTE2Sync();
    }
  }
};

#endif