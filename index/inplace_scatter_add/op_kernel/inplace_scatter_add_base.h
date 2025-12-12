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
 * \file inplace_scatter_add_base.h
 * \brief
 */

#ifndef INPLACE_SCATTER_ADD_BASE_H
#define INPLACE_SCATTER_ADD_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace InplaceScatterAddNS {
using namespace AscendC;
constexpr uint32_t ASCENDC_SORT_NUMS = 32;
constexpr uint32_t BIG_TAIL_OP_MAX_SORT_NUMS = 6976;
constexpr uint32_t SMALL_TAIL_OP_MAX_SORT_NUMS = 4064;
constexpr uint32_t MAX_BATCH_SIZE = 128;
constexpr uint32_t DOUBLE = 2;

template <typename VT, typename IT>
class BaseOp {
public:
  __aicore__ inline BaseOp() {}

  __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, const InplaceScatterAddTilingData* tilingData, TPipe* tPipe) {
    this->ubSize = tilingData->ubSize;
    this->M = tilingData->M;
    this->N = tilingData->N;
    this->K = tilingData->K;
    this->NAligned = tilingData->NAligned;
    this->frontCoreNum =  tilingData->frontCoreNum;
    this->tailCoreNum = tilingData->tailCoreNum;
    this->frontCoreIndicesNum = tilingData->frontCoreIndicesNum;
    this->tailCoreIndicesNum = tilingData->tailCoreIndicesNum;

    if (GetBlockIdx() < this->frontCoreNum) {
        this->coreIndicesNum = this->frontCoreIndicesNum;
        this->coreStartIndex = GetBlockIdx() * this->frontCoreIndicesNum;
    } else {
        this->coreIndicesNum = this->tailCoreIndicesNum;
        this->coreStartIndex = this->frontCoreNum * this->frontCoreIndicesNum + (GetBlockIdx() - this->frontCoreNum) * this->tailCoreIndicesNum;
    }

    this->varGm.SetGlobalBuffer(reinterpret_cast<__gm__ VT*>(var));
    this->indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
    this->updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ VT*>(updates));

    this->pipe = tPipe;
    this->pipe->InitBuffer(this->allUbBuf, this->ubSize * sizeof(float));
    this->allUbLocal = this->allUbBuf.template Get<VT>();
  }

  __aicore__ inline void SortIndices(uint32_t startIndex, uint32_t sortIndexNums) {
    auto sortScoreLocal = this->indicesLocal.template ReinterpretCast<float>();
    auto sortIndex = this->allUbLocal.template ReinterpretCast<int32_t>()[sortIndexNums];
    CreateVecIndex(sortIndex, static_cast<int32_t>(startIndex), sortIndexNums);
    PIPE_V_S();
    auto sortIndexLocal = sortIndex.template ReinterpretCast<uint32_t>();
    auto sortDstLocal = this->allUbLocal.template ReinterpretCast<float>()[sortIndexNums * DOUBLE];
    auto sortTempLocal = this->allUbLocal.template ReinterpretCast<float>()[sortIndexNums * DOUBLE * DOUBLE];
    uint32_t repeatTimes = sortIndexNums / ASCENDC_SORT_NUMS;
    Sort<float, true>(sortDstLocal, sortScoreLocal, sortIndexLocal, sortTempLocal, static_cast<int32_t>(repeatTimes));
    PIPE_V_S();
    Extract(sortScoreLocal, sortIndexLocal, sortDstLocal, static_cast<int32_t>(repeatTimes));
    PIPE_V_S();
  }

  __aicore__ inline void CopyInIndices(uint32_t mte2Start, uint32_t mteNum) {
    if (sizeof(IT) == sizeof(int32_t)) {
      this->indicesLocal = this->allUbLocal.template ReinterpretCast<int32_t>();
      DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNum * sizeof(int32_t)), 0, 0, 0};
      DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
      DataCopyPad(this->indicesLocal, this->indicesGm[mte2Start], copyParams, padParams);
    } else if (sizeof(IT) == sizeof(int64_t)) {
      this->indicesLocal = this->allUbLocal.template ReinterpretCast<int32_t>();
      DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNum * sizeof(IT)), 0, 0, 0};
      DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
      DataCopyPad(this->indicesLocal, this->indicesGm[mte2Start * DOUBLE], copyParams, padParams);
      PIPE_MTE2_S();
      auto indicesLocalInt64 = this->indicesLocal.template ReinterpretCast<int64_t>();
      Cast(this->indicesLocal, indicesLocalInt64, RoundMode::CAST_NONE, mteNum);
      PIPE_V_S();
    }
  }

  __aicore__ inline void CoreTiling() {
    this->sortMaxTimes = this->coreIndicesNum / this->maxSortNums;
    this->leftSortIndices = this->coreIndicesNum - this->sortMaxTimes * this->maxSortNums;
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

public:
  uint32_t ubSize = 0;
  uint32_t M = 0;
  uint32_t N = 0;
  uint32_t K = 0;
  uint32_t NAligned = 0;
  uint32_t frontCoreNum = 0;
  uint32_t tailCoreNum = 0;
  uint32_t frontCoreIndicesNum = 0;
  uint32_t tailCoreIndicesNum = 0;

  uint32_t maxSortNums = 0;
  uint32_t sortMaxTimes = 0;
  uint32_t leftSortIndices = 0;

  uint32_t coreIndicesNum = 0;
  uint32_t coreStartIndex = 0;

  GlobalTensor<VT> varGm;
  GlobalTensor<int32_t> indicesGm;
  GlobalTensor<VT> updatesGm;

  LocalTensor<int32_t> indicesLocal;

  LocalTensor<VT> allUbLocal;
  TPipe* pipe;
  TBuf<TPosition::VECCALC> allUbBuf;
};
}
#endif