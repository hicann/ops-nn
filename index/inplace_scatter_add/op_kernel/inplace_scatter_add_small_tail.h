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
 * \file inplace_scatter_add_small_tail.h
 * \brief
 */

#ifndef INPLACE_SCATTER_ADD_SMALL_TAIL_H
#define INPLACE_SCATTER_ADD_SMALL_TAIL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "inplace_scatter_add_base.h"

namespace InplaceScatterAddNS {
using namespace AscendC;

template <typename VT, typename IT>
class SmallTailOp : public BaseOp<VT, IT> {
public:
  __aicore__ inline void Process() {
    GetMaxSortNums();
    this->CoreTiling();
    for(uint32_t i = 0; i < this->sortMaxTimes; i++) {
        uint32_t startIndex = this->coreStartIndex + i * this->maxSortNums;
        SortedProcess(startIndex, this->maxSortNums, 0);
    }
    if (this->leftSortIndices > 0) {
        uint32_t startIndex = this->coreStartIndex + this->sortMaxTimes * this->maxSortNums;
        uint32_t sortIndexNums = (this->leftSortIndices / ASCENDC_SORT_NUMS) * ASCENDC_SORT_NUMS;
        uint32_t notSortIndexNums = this->leftSortIndices - sortIndexNums;
        SortedProcess(startIndex, sortIndexNums, 0);
        startIndex = startIndex + sortIndexNums;
        UnSortedProcess(startIndex, notSortIndexNums, 0);
    }
  }

  __aicore__ inline void UnSortedProcess(uint32_t startIndex, uint32_t indexNums, uint32_t startPosIdx) {
    if (indexNums) {
      this->CopyInIndices(startIndex, indexNums);
      this->PIPE_MTE2_S();
      uint32_t notSortIndexNumsAligned = ((indexNums + ASCENDC_SORT_NUMS - 1) / ASCENDC_SORT_NUMS) * ASCENDC_SORT_NUMS;
      auto posIdxLocalTensor = this->allUbLocal.template ReinterpretCast<int32_t>()[notSortIndexNumsAligned];
      CreateVecIndex(posIdxLocalTensor, static_cast<int32_t>(startPosIdx), indexNums);
      this->PIPE_V_S();
      ScatterProcess(startIndex, indexNums);
    }
  }

  __aicore__ inline void SortedProcess(uint32_t startIndex, uint32_t indexNums, uint32_t startPosIdx) {
    if (indexNums > 0) {
      this->CopyInIndices(startIndex, indexNums);
      this->PIPE_MTE2_S();
      this->SortIndices(startPosIdx, indexNums);
      this->PIPE_V_S();
      ScatterProcess(startIndex, indexNums);
    }
  }

  __aicore__ inline void ScatterProcess(uint32_t startIndex, uint32_t indexNums) {
    uint32_t indexNumsAligned = ((indexNums + ASCENDC_SORT_NUMS - 1) / ASCENDC_SORT_NUMS) * ASCENDC_SORT_NUMS;
    uint32_t indexUb = indexNumsAligned * DOUBLE;
    CopyInUpdates(startIndex, indexNums, indexUb);
    this->PIPE_MTE2_S();
    ComputeAndOutUpdates(indexNumsAligned, indexNums, indexUb);
    this->PIPE_MTE3_S();
  }

  __aicore__ inline void ComputeAndOutUpdates(uint32_t indexNums, uint32_t mte3Nums, uint32_t dst) {
    auto updatesLocal = this->allUbLocal[dst];
    auto posIdxLocal = this->allUbLocal.template ReinterpretCast<uint32_t>()[indexNums];
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->N * sizeof(VT)), 0, 0, 0};
    uint32_t pre = 0;
    uint32_t preIndexValue = this->indicesLocal.GetValue(pre);
    uint32_t prePosIdxValue = posIdxLocal.GetValue(pre);
    SetAtomicAdd<VT>();
    for (uint32_t i = 1; i < mte3Nums; i++) {
      uint32_t iIndexValue = this->indicesLocal.GetValue(i);
      uint32_t iPosIdxValue = posIdxLocal.GetValue(i);
      if (preIndexValue == iIndexValue) {
        Add(updatesLocal[prePosIdxValue * this->NAligned], updatesLocal[prePosIdxValue * this->NAligned], updatesLocal[iPosIdxValue * this->NAligned], this->NAligned);
        this->PIPE_V_S(); // this is high performance
      } else {
        DataCopyPad(this->varGm[preIndexValue * this->N], updatesLocal[prePosIdxValue * this->NAligned], copyParams);
        pre = i;
        preIndexValue = iIndexValue;
        prePosIdxValue = iPosIdxValue;
      }
    }

    uint32_t last = mte3Nums - 1;
    uint32_t lastIndexValue = this->indicesLocal.GetValue(last);
    uint32_t lastPosIdxValue = posIdxLocal.GetValue(last);
    if (lastIndexValue == preIndexValue) {
      DataCopyPad(this->varGm[preIndexValue * this->N], updatesLocal[prePosIdxValue * this->NAligned], copyParams);
    } else {
      DataCopyPad(this->varGm[lastIndexValue * this->N], updatesLocal[lastPosIdxValue * this->NAligned], copyParams);
    }
    SetAtomicNone();
  }

  __aicore__ inline void CopyInUpdates(uint32_t gmStartIndex, uint32_t mte2Nums, uint32_t dst) {
    if(this->N == this->NAligned) {
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(mte2Nums * this->N * sizeof(VT)), 0, 0, 0};
        DataCopyPadExtParams<VT> padParams{true, 0, 0, 0};
        auto updatesLocal = this->allUbLocal[dst];
        DataCopyPad(updatesLocal, this->updatesGm[gmStartIndex * this->N], copyParams, padParams);
    } else {
        DataCopyExtParams copyParams{static_cast<uint16_t>(mte2Nums), static_cast<uint32_t>(this->N * sizeof(VT)), 0, 0, 0};
        DataCopyPadExtParams<VT> padParams{true, 0, 0, 0};
        auto updatesLocal = this->allUbLocal[dst];
        DataCopyPad(updatesLocal, this->updatesGm[gmStartIndex * this->N], copyParams, padParams);
    }
  }

  __aicore__ inline void GetMaxSortNums() {
    uint32_t maxSortNumsTemp = this->ubSize / (this->NAligned + DOUBLE);
    maxSortNumsTemp = (maxSortNumsTemp / ASCENDC_SORT_NUMS) * ASCENDC_SORT_NUMS;
    this->maxSortNums = maxSortNumsTemp > SMALL_TAIL_OP_MAX_SORT_NUMS ? SMALL_TAIL_OP_MAX_SORT_NUMS : maxSortNumsTemp;
  }
};
}
#endif