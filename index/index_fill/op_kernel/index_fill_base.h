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
 * \file index_fill_base.h
 * \brief indices process and other public function
 */
#ifndef INDEX_FILL_BASE_H
#define INDEX_FILL_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace IndexFillNS {
using namespace AscendC;
constexpr uint32_t INDICES_UB_LENGTH = 2048; // 2048个U
constexpr uint32_t WS_UB_LENGTH = 8192;
constexpr uint32_t X_UB_LENGTH = 16384;
constexpr uint8_t SYS_FLAG = 133;
constexpr uint32_t INT32_ALIGNED_NUM = 8;
constexpr uint32_t INT8_ALIGNED_NUM = 32;
constexpr uint32_t COMPARE_ALIGNED = 128;
constexpr uint32_t HALF_ALIGNED_NUM = 32;

template <typename T, typename U>
class IndexFillBase {
public:
  __aicore__ inline IndexFillBase() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling,
                              const IndexFillTilingData* tilingData, TPipe* tPipe) {
    this->coreNum = tilingData->coreNum;
    this->N = tilingData->N;
    this->indicesNum = tilingData->indicesNum;
    this->indicesProcessMode = tilingData->indicesProcessMode;
    this->frontCoreNumTaskIndices = tilingData->frontCoreNumTaskIndices;
    this->tailCoreNumTaskIndices = tilingData->tailCoreNumTaskIndices;
    this->frontCoreDataTaskIndices = tilingData->frontCoreDataTaskIndices;
    this->tailCoreDataTaskIndices = tilingData->tailCoreDataTaskIndices;
    this->ubSize = tilingData->ubSize;
    this->P = tilingData->P;
    this->Q = tilingData->Q;

    this->pipe = tPipe;

    this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    this->indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(indices));
    this->valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(value));
    this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
    auto userWorkSpace = GetUserWorkspace(workspace);
    this->wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWorkSpace));// wsLocal, sysLocal
    this->sysGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(userWorkSpace) + this->N * sizeof(half));
    uint64_t preHalfNums = (((this->N * sizeof(half) + this->coreNum * sizeof(uint8_t) +
                              INT8_ALIGNED_NUM - 1) / INT8_ALIGNED_NUM) * INT8_ALIGNED_NUM) / sizeof(half);
    this->repeatGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWorkSpace) + preHalfNums);
    uint64_t preInt32Nums = (((this->N * sizeof(half) + this->coreNum * sizeof(uint8_t) +
                              INT8_ALIGNED_NUM - 1) / INT8_ALIGNED_NUM) * INT8_ALIGNED_NUM) / sizeof(int32_t);
    this->gatherGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkSpace) + preInt32Nums);

    this->pipe->InitBuffer(this->allUbBuf, this->ubSize * sizeof(uint8_t));
    this->allUbLocal = this->allUbBuf.template Get<uint8_t>();

    InitCoreSys();
  }

  __aicore__ inline void InitCoreSys() {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(uint8_t)), 0, 0, 0};
    this->allUbLocal.SetValue(0, 0);
    DataCopyPad(this->sysGm[GetBlockIdx()], this->allUbLocal, copyParams);
    PIPE_MTE3_S();
    if (this->indicesProcessMode == 0) {
        // 将wsGm置零, 另一种模式依赖核自己置零
        this->wsLocal = this->allUbLocal.template ReinterpretCast<half>();
        Duplicate(this->wsLocal, static_cast<half>(0), static_cast<int32_t>(this->N));
        PIPE_V_S();
        CopyOutWsGm(0, this->wsLocal, this->N);
        PIPE_MTE3_S();
    }
    SyncAll();
  }

  __aicore__ inline void ExecIndicesTask() {
    if (this->indicesProcessMode == 0) {
        IndicesTaskByIndicesNum();
    } else {
        IndicesTaskByN();
    }
  }

  __aicore__ inline void GetCoreTask(uint64_t& coreDataTaskIndices, uint64_t& coreIndicesGmStart, uint64_t& coreIndicesGmEnd) {
    if (GetBlockIdx() < this->frontCoreNumTaskIndices) {
        coreDataTaskIndices = this->frontCoreDataTaskIndices;
        coreIndicesGmStart = this->frontCoreDataTaskIndices * GetBlockIdx();
        coreIndicesGmEnd = coreIndicesGmStart + coreDataTaskIndices;
    } else {
        coreDataTaskIndices = this->tailCoreDataTaskIndices;
        coreIndicesGmStart = this->frontCoreDataTaskIndices * this->frontCoreNumTaskIndices +
                             (GetBlockIdx() - this->frontCoreNumTaskIndices) * this->tailCoreDataTaskIndices;
        coreIndicesGmEnd = coreIndicesGmStart + coreDataTaskIndices;
    }
  }

  __aicore__ inline void IndicesTaskByN() {
    // N比较大，使用wsLocal先把自己分到的wsGm置0，然后遍历所有Tensor，遇到范围内的索引就单点搬出
    // UB布局：先wsLocal[INDICES_UB_LENGTH * sizeof(half)]，
    //         后indicesLocal[INDICES_UB_LENGTH * sizeof(U)]，tilingLocal[8 * sizeof(int32_t)]
    uint64_t coreDataTaskIndices = 0;
    uint64_t coreIndicesGmStart = 0;
    uint64_t coreIndicesGmEnd = 0;
    GetCoreTask(coreDataTaskIndices, coreIndicesGmStart, coreIndicesGmEnd);

    if (coreDataTaskIndices > 0) {
        // wsGm置为0
        this->wsLocal = this->allUbLocal.template ReinterpretCast<half>();
        Duplicate(this->wsLocal, static_cast<half>(0), static_cast<int32_t>(INDICES_UB_LENGTH));
        PIPE_V_S();
        uint64_t NBlocks = coreDataTaskIndices / INDICES_UB_LENGTH;
        uint64_t NLeft = coreDataTaskIndices - NBlocks * INDICES_UB_LENGTH;
        for (uint64_t i = 0; i <= NBlocks; i++) {
            if (i == NBlocks && NLeft == 0) {
                break;
            }
            uint64_t NNums = (i == NBlocks ? NLeft : INDICES_UB_LENGTH);
            uint64_t gmStartHalf = (coreIndicesGmStart + i * INDICES_UB_LENGTH);
            uint64_t mteNumsHalf = NNums;
            CopyOutWsGm(gmStartHalf, this->wsLocal, mteNumsHalf);
            PIPE_MTE3_S();
        }
        // 迭代所有索引
        uint64_t indicesBlocks = this->indicesNum / INDICES_UB_LENGTH;
        uint64_t indicesLeft = this->indicesNum - indicesBlocks * INDICES_UB_LENGTH;
        auto oneLocal = this->allUbLocal[INDICES_UB_LENGTH * sizeof(U)].template ReinterpretCast<half>();
        oneLocal.SetValue(0, 1);
        for (uint64_t i = 0; i <= indicesBlocks; i++) {
            if (i == indicesBlocks && indicesLeft == 0) {
                break;
            }
            uint64_t indicesNums = (i == indicesBlocks ? indicesLeft : INDICES_UB_LENGTH);
            uint64_t gmStartInt8 = INDICES_UB_LENGTH * i * sizeof(U);
            uint64_t mteNumsInt8 = indicesNums * sizeof(U);
            CopyIn(this->indicesGm, gmStartInt8, 0, mteNumsInt8);
            PIPE_MTE2_S();
            this->indicesLocal = this->allUbLocal.template ReinterpretCast<U>();
            for (uint64_t j = 0; j < indicesNums; j++) {
                U indicesValue = this->indicesLocal.GetValue(j);
                indicesValue = (indicesValue < 0 ? indicesValue + this->N : indicesValue);
                if (indicesValue >= coreIndicesGmStart && indicesValue < coreIndicesGmEnd) {
                    // 防止索引冲突严重，引发性能下降
                    CopyOutWsGm(indicesValue, oneLocal, 1);
                }
            }
        }
        PIPE_MTE3_S();
    }
    WriteFlagToGm();
    WaitForIndicesTask();
  }

  __aicore__ inline void IndicesTaskByIndicesNum() {
    // N不大，直接加载全部ws进入ub，然后加载核分到的Indices进ub进行遍历
    // UB布局：wsLocal[N * sizeof(half)]，indicesLocal[INDICES_UB_LENGTH * sizeof(U)]，tilingLocal[8 * sizeof(int32_t)]
    uint64_t coreDataTaskIndices = 0;
    uint64_t coreIndicesGmStart = 0;
    if (GetBlockIdx() < this->frontCoreNumTaskIndices) {
        coreDataTaskIndices = this->frontCoreDataTaskIndices;
        coreIndicesGmStart = this->frontCoreDataTaskIndices * GetBlockIdx();
    } else {
        coreDataTaskIndices = this->tailCoreDataTaskIndices;
        coreIndicesGmStart = this->frontCoreDataTaskIndices * this->frontCoreNumTaskIndices +
                             (GetBlockIdx() - this->frontCoreNumTaskIndices) * this->tailCoreDataTaskIndices;
    }
    if (coreDataTaskIndices > 0) {
        this->wsLocal = this->allUbLocal.template ReinterpretCast<half>();
        Duplicate(this->wsLocal, static_cast<half>(0), static_cast<int32_t>(this->N));
        PIPE_V_S();
        uint64_t indicesBlocks = coreDataTaskIndices / INDICES_UB_LENGTH;
        uint64_t indicesLeft = coreDataTaskIndices - indicesBlocks * INDICES_UB_LENGTH;
        for (uint64_t i = 0; i <= indicesBlocks; i++) {
            if (i == indicesBlocks && indicesLeft == 0) {
                break;
            }
            uint64_t indicesNums = (i == indicesBlocks ? indicesLeft : INDICES_UB_LENGTH);
            uint64_t gmStartInt8 = (coreIndicesGmStart + INDICES_UB_LENGTH * i) * sizeof(U);
            uint64_t ubStartInt8 = ((this->N * sizeof(half) + INT8_ALIGNED_NUM - 1) / INT8_ALIGNED_NUM) *
                                   INT8_ALIGNED_NUM;
            uint64_t mteNumsInt8 = indicesNums * sizeof(U);
            CopyIn(this->indicesGm, gmStartInt8, ubStartInt8, mteNumsInt8);
            PIPE_MTE2_S();
            this->indicesLocal = this->allUbLocal[ubStartInt8].template ReinterpretCast<U>();
            for (uint64_t j = 0; j < indicesNums; j++) {
                U indicesValue = this->indicesLocal.GetValue(j);
                indicesValue = (indicesValue < 0 ? indicesValue + this->N : indicesValue);
                if (indicesValue >= 0 && indicesValue < this->N) {
                  this->wsLocal.SetValue(indicesValue, 1);
                }
            }
        }
        // 所有核的结果累加到GM上
        SetAtomicAdd<half>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->N * sizeof(half)), 0, 0, 0};
        DataCopyPad(this->wsGm, this->wsLocal, copyParams);
        SetAtomicNone();
        PIPE_MTE3_S();
    }
    WriteFlagToGm();
    WaitForIndicesTask();
  }

  __aicore__ inline void WaitForIndicesTask() {
    uint64_t ubStartInt8 = 0;
    LocalTensor<uint8_t> tmpLocal = this->allUbLocal[ubStartInt8];
    while(true) {
        CopyInSysGm(0, tmpLocal, this->coreNum * sizeof(uint8_t));
        PIPE_MTE2_S();
        bool coreTaskEnd = true;
        for (uint64_t i = 0; i < this->coreNum; i++) {
            if (tmpLocal.GetValue(i) != SYS_FLAG) {
                coreTaskEnd = false;
            }
        }
        if (coreTaskEnd) {
            break;
        }
    }
  }

  __aicore__ inline void WriteFlagToGm() {
    uint64_t ubStartInt8 = 0;
    LocalTensor<uint8_t> tmpLocal = this->allUbLocal[ubStartInt8];
    tmpLocal.SetValue(0, SYS_FLAG);

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad(this->sysGm[GetBlockIdx()], tmpLocal, copyParams);
    PIPE_MTE3_S();
  }

  __aicore__ inline void CopyInSysGm(uint64_t gmStartInt8, LocalTensor<uint8_t> tilingLocal_, uint64_t mteNumsInt8) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNumsInt8 * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 0};
    DataCopyPad(tilingLocal_, this->sysGm[gmStartInt8], copyParams, padParams);
  }

  __aicore__ inline void CopyInWsGm(uint64_t gmStartHalf, LocalTensor<half> wsLocal_, uint64_t mteNumsHalf) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNumsHalf * sizeof(half)), 0, 0, 0};
    DataCopyPadExtParams<half> padParams{true, 0, 0, 0};
    DataCopyPad(wsLocal_, this->wsGm[gmStartHalf], copyParams, padParams);
  }

  __aicore__ inline void CopyOutWsGm(uint64_t gmStartHalf, LocalTensor<half> wsLocal_, uint64_t mteNumsHalf) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNumsHalf * sizeof(half)), 0, 0, 0};
    DataCopyPad(this->wsGm[gmStartHalf], wsLocal_, copyParams);
  }

  __aicore__ inline void CopyIn(GlobalTensor<uint8_t> gmTensor, uint64_t gmStartInt8,
                                uint64_t ubStartInt8, uint64_t mteNumsInt8) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNumsInt8 * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 0};
    DataCopyPad(this->allUbLocal[ubStartInt8], gmTensor[gmStartInt8], copyParams, padParams);
  }

  __aicore__ inline void CopyOut(GlobalTensor<uint8_t> gmTensor, uint64_t gmStartInt8,
                                 uint64_t ubStartInt8, uint64_t mteNumsInt8) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNumsInt8 * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad(gmTensor[gmStartInt8], this->allUbLocal[ubStartInt8], copyParams);
  }

  __aicore__ inline void PIPE_V_S() {
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
  }

  __aicore__ inline void PIPE_MTE2_S() {
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
  }

  __aicore__ inline void PIPE_MTE3_S() {
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
  }

  __aicore__ inline bool WillWorkForTask(uint64_t taskNums) {
    uint64_t nextCorePtr = WrapAdd(this->coreNum, this->corePtr, taskNums);// nextCorePtr is not work
    bool willWork = false;

    if(nextCorePtr > this->corePtr) {
        if(this->coreId >= this->corePtr && this->coreId < nextCorePtr) {
            willWork = true;
        }
    } else if(nextCorePtr == this->corePtr) {
        willWork = true;
    } else {
        if(this->coreId >= this->corePtr || this->coreId < nextCorePtr) {
            willWork = true;
        }
    }
    this->corePtr = nextCorePtr;
    return willWork;
  }

  __aicore__ inline int64_t WrapAdd(int64_t M, int64_t N, int64_t K) {
    int64_t result = N + K;
    M = M == 0 ? 1 : M;
    if (result >= M) {
        result %= M;
    }
    return result;
  }

  __aicore__ inline void ReadValue() {
    this->CopyIn(this->valueGm, 0, 0, sizeof(T));
    this->PIPE_MTE2_S();
    this->valueLocal = this->allUbLocal.template ReinterpretCast<T>();
    this->value = this->valueLocal.GetValue(0);
  }

  __aicore__ inline void coreSplit(uint64_t coreNums, uint64_t dataNums,
                                   uint64_t& frontCoreNums, uint64_t& tailCoreNums,
                                   uint64_t& frontCoreDataNums, uint64_t& tailCoreDataNums) {
    coreNums = coreNums == 0 ? 1 : coreNums;
    tailCoreDataNums = dataNums / coreNums;
    if (tailCoreDataNums == 0) {
        frontCoreNums = dataNums;
        tailCoreNums = 0;
        frontCoreDataNums = 1;
        tailCoreDataNums = 0;  
    } else {
        if (dataNums % coreNums == 0) {
          frontCoreNums = coreNums;
          frontCoreDataNums = tailCoreDataNums;
          tailCoreNums = 0;
          tailCoreDataNums = 0;
        } else {
          frontCoreNums = dataNums - tailCoreDataNums * coreNums;
          frontCoreDataNums = tailCoreDataNums + 1;
          tailCoreNums = coreNums - frontCoreNums;
        }
    }
  }

  __aicore__ inline uint64_t AlignedToTarget(uint64_t value, uint64_t target) {
    if (target == 0) {
      return value;
    } else {
      return ((value + target - 1) / target) * target;
    }
  }

  __aicore__ inline void SliceP(uint64_t& corePStart, uint64_t& corePNums) {
    uint64_t frontCoreNums = 0;
    uint64_t tailCoreNums = 0;
    uint64_t frontCorePNums = 0;
    uint64_t tailCorePNums = 0;
    coreSplit(this->coreNum, this->P, frontCoreNums, tailCoreNums, frontCorePNums, tailCorePNums);
    
    if (GetBlockIdx() < frontCoreNums) {
        corePStart = GetBlockIdx() * frontCorePNums;
        corePNums = frontCorePNums;
    } else if(tailCoreNums > 0){
        corePStart = frontCoreNums * frontCorePNums + (GetBlockIdx() - frontCoreNums) * tailCorePNums;
        corePNums = tailCorePNums;
    }
  }

public:
  T value = 0;
  uint64_t coreNum = 0;
  uint64_t N = 0;
  uint64_t P = 0;
  uint64_t Q = 0;
  uint64_t indicesNum = 0;
  uint64_t indicesProcessMode = 0;
  uint64_t frontCoreNumTaskIndices = 0;
  uint64_t tailCoreNumTaskIndices = 0;
  uint64_t frontCoreDataTaskIndices = 0; 
  uint64_t tailCoreDataTaskIndices = 0;
  uint64_t ubSize = 0;

  uint64_t corePtr = 0;
  uint64_t coreId = GetBlockIdx();
  
  GlobalTensor<uint8_t> xGm;       // T
  GlobalTensor<uint8_t> indicesGm; // U
  GlobalTensor<uint8_t> valueGm;   // T
  GlobalTensor<uint8_t> yGm;       // T
  GlobalTensor<half> wsGm;      // half
  GlobalTensor<uint8_t> sysGm;
  GlobalTensor<half> repeatGm;
  GlobalTensor<int32_t> gatherGm;

  LocalTensor<uint8_t> allUbLocal;
  TPipe* pipe;
  TBuf<TPosition::VECCALC> allUbBuf;

  LocalTensor<T> xLocal;
  LocalTensor<U> indicesLocal;
  LocalTensor<T> valueLocal;
  LocalTensor<T> yLocal;
  LocalTensor<half> wsLocal;
};
}
#endif