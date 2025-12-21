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
 * \file inplace_scatter_add_big_tail.h
 * \brief
 */

#ifndef INPLACE_SCATTER_ADD_BIG_TAIL_H
#define INPLACE_SCATTER_ADD_BIG_TAIL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "inplace_scatter_add_base.h"

namespace InplaceScatterAddNS {
using namespace AscendC;

template <typename VT, typename IT>
class BigTailOp : public BaseOp<VT, IT> {
public:
  __aicore__ inline void Process() {
    GetMaxSortNums();
    this->CoreTiling();
    for(uint32_t i = 0; i < this->sortMaxTimes; i++) {
        uint32_t startIndex = this->coreStartIndex + i * this->maxSortNums;
        SortedProcess(startIndex, this->maxSortNums, startIndex);
    }
    if (this->leftSortIndices > 0) {
        uint32_t startIndex = this->coreStartIndex + this->sortMaxTimes * this->maxSortNums;
        uint32_t sortIndexNums = (this->leftSortIndices / ASCENDC_SORT_NUMS) * ASCENDC_SORT_NUMS;
        uint32_t notSortIndexNums = this->leftSortIndices - sortIndexNums;
        SortedProcess(startIndex, sortIndexNums, startIndex);
        startIndex = startIndex + sortIndexNums;
        UnSortedProcess(startIndex, notSortIndexNums, startIndex);
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

  __aicore__ inline void ScatterProcess(uint32_t startIndex, uint32_t indexNums) {
    // allUbLocal now：indices, posIdx at head，others for updatesLocal use。
    uint32_t indexNumsAligned = ((indexNums + ASCENDC_SORT_NUMS - 1) / ASCENDC_SORT_NUMS) * ASCENDC_SORT_NUMS;
    uint32_t updatesUbStart = indexNumsAligned * DOUBLE;
    uint32_t leftUbSize = this->ubSize - updatesUbStart;
    uint32_t batchSize = leftUbSize / (DOUBLE * this->NAligned);
    uint32_t maxBatchSize = MAX_BATCH_SIZE;
    batchSize = batchSize > maxBatchSize ? maxBatchSize : batchSize;
    if (batchSize > 0) {
      BatchProcess(updatesUbStart, leftUbSize, indexNums, indexNumsAligned, batchSize);
    } else {
      LargeTailProcess(updatesUbStart, leftUbSize, indexNums, indexNumsAligned);
    }
  }

  __aicore__ inline void BatchProcess(uint32_t updatesUbStart, uint32_t leftUbSize,
                                      uint32_t indexNums, uint32_t indexNumsAligned, uint32_t batchSize) {
    uint32_t batchNums = indexNums / batchSize;
    uint32_t leftIndices = indexNums - batchNums * batchSize;
    bool updatesLocalId = false;
    if (batchNums > 0) {
      uint32_t dst = updatesUbStart + static_cast<uint32_t>(updatesLocalId) * batchSize * this->NAligned;
      uint32_t batchStartIndex = 0;
      CopyInUpdates(indexNumsAligned, batchStartIndex, batchSize, dst);
    }
    for (uint32_t i = 0; i < batchNums; i++) {
      this->PIPE_MTE2_S();
      if (i < batchNums - 1) {
        uint32_t dst = updatesUbStart + static_cast<uint32_t>(!updatesLocalId) * batchSize * this->NAligned;
        uint32_t batchStartIndex = batchSize * (i + 1);
        CopyInUpdates(indexNumsAligned, batchStartIndex, batchSize, dst);
      }
      uint32_t dst = updatesUbStart + static_cast<uint32_t>(updatesLocalId) * batchSize * this->NAligned;
      uint32_t batchStartIndex = batchSize * i;
      ComputeAndOutUpdates(batchStartIndex, batchSize, dst);
      this->PIPE_MTE3_S();
      updatesLocalId = !updatesLocalId;
    }
    if (leftIndices > 0) {
      uint32_t dst = updatesUbStart + static_cast<uint32_t>(updatesLocalId) * batchSize * this->NAligned;
      uint32_t leftStartIndex = batchNums * batchSize;
      CopyInUpdates(indexNumsAligned, leftStartIndex, leftIndices, dst);
      this->PIPE_MTE2_S();
      ComputeAndOutUpdates(leftStartIndex, leftIndices, dst);
      this->PIPE_MTE3_S();
    }
  }

  __aicore__ inline void LargeTailProcess(uint32_t updatesUbStart, uint32_t leftUbSize,
                                          uint32_t indexNums, uint32_t indexNumsAligned) {
    auto posIdxLocalTensor = this->allUbLocal.template ReinterpretCast<uint32_t>()[indexNumsAligned];
    auto updatesLocal = this->allUbLocal[updatesUbStart];
    uint32_t leftUbSizeAligned = (leftUbSize / ASCENDC_SORT_NUMS) * ASCENDC_SORT_NUMS;
    uint32_t mteTimes = this->N / leftUbSizeAligned;
    uint32_t mteLeft = this->N - mteTimes * leftUbSizeAligned;
    for (uint32_t i = 0; i < indexNums; i++) {
      uint32_t indexValue = this->indicesLocal.GetValue(i);
      uint32_t posIdxValue = posIdxLocalTensor.GetValue(i);
      for (uint32_t j = 0; j < mteTimes; j++) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(leftUbSizeAligned * sizeof(VT)), 0, 0, 0};
        DataCopyPadExtParams<VT> padParams{true, 0, 0, 0};
        DataCopyPad(updatesLocal, this->updatesGm[posIdxValue * this->N + j * leftUbSizeAligned], copyParams, padParams);
        this->PIPE_MTE2_S();
        SetAtomicAdd<VT>();
        DataCopyPad(this->varGm[indexValue * this->N + j * leftUbSizeAligned], updatesLocal, copyParams);
        SetAtomicNone();
        this->PIPE_MTE3_S();
      }
      if (mteLeft > 0) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteLeft * sizeof(VT)), 0, 0, 0};
        DataCopyPadExtParams<VT> padParams{true, 0, 0, 0};
        DataCopyPad(updatesLocal, this->updatesGm[posIdxValue * this->N + mteTimes * leftUbSizeAligned], copyParams, padParams);
        this->PIPE_MTE2_S();

        SetAtomicAdd<VT>();
        DataCopyPad(this->varGm[indexValue * this->N + mteTimes * leftUbSizeAligned], updatesLocal, copyParams);
        SetAtomicNone();
        this->PIPE_MTE3_S();
      }
    }
  }

  __aicore__ inline void ComputeAndOutUpdates(uint32_t startIndex, uint32_t computeNums, uint32_t dst) {
    auto updatesLocal = this->allUbLocal[dst];
    uint32_t pre = 0;
    uint32_t preIndexValue = this->indicesLocal.GetValue(startIndex + pre);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->N * sizeof(VT)), 0, 0, 0};
    SetAtomicAdd<VT>();
    for (uint32_t i = 1; i < computeNums; i++) {
      uint32_t iIndexValue = this->indicesLocal.GetValue(startIndex + i);
      if (preIndexValue == iIndexValue) {
        continue;
      } else {
        if(pre < i - 1) {
          SameIndexAdd(pre, i - 1, dst);
          this->PIPE_V_S();
        }
        DataCopyPad(this->varGm[preIndexValue * this->N], updatesLocal[pre * this->NAligned], copyParams);
        pre = i;
        preIndexValue = iIndexValue;
      }
    }
    uint32_t last = computeNums - 1;
    uint32_t lastIndexValue = this->indicesLocal.GetValue(startIndex + last);
    if (lastIndexValue == preIndexValue) {
      SameIndexAdd(pre, last, dst);
      this->PIPE_V_S();
      DataCopyPad(this->varGm[preIndexValue * this->N], updatesLocal[pre * this->NAligned], copyParams);
    } else {
      DataCopyPad(this->varGm[lastIndexValue * this->N], updatesLocal[last * this->NAligned], copyParams);
    }
    SetAtomicNone();
  }

  __aicore__ inline void SameIndexAdd(uint32_t start, uint32_t end, uint32_t dst) {
    if (end > start) {
      auto updatesLocal = this->allUbLocal[dst];
      while(true) {
        if (end - start == 1) {
          PipeBarrier<PIPE_V>();
          Add(updatesLocal[start * this->NAligned], updatesLocal[start * this->NAligned], updatesLocal[end * this->NAligned], this->NAligned);
          break;
        } else {
          uint32_t mid = start + (end - start) / DOUBLE + 1;
          PipeBarrier<PIPE_V>();
          Add(updatesLocal[start * this->NAligned], updatesLocal[start * this->NAligned], updatesLocal[mid * this->NAligned], (end - mid + 1) * this->NAligned);
          end = mid - 1;
        }
      }
    }
  }

  __aicore__ inline void CopyInUpdates(uint32_t indexNums, uint32_t startIndex, uint32_t mte2Nums, uint32_t dst) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->N * sizeof(VT)), 0, 0, 0};
    DataCopyPadExtParams<VT> padParams{true, 0, 0, 0};
    auto posIdxLocalTensor = this->allUbLocal.template ReinterpretCast<uint32_t>()[indexNums];
    auto updatesLocal = this->allUbLocal[dst];
    for (uint32_t i = 0; i < mte2Nums; i++) {
      uint32_t posIdxValue = posIdxLocalTensor.GetValue(startIndex + i);
      DataCopyPad(updatesLocal[i * this->NAligned], this->updatesGm[posIdxValue * this->N], copyParams, padParams);
    }
  }

  __aicore__ inline void GetMaxSortNums() {
    this->maxSortNums = BIG_TAIL_OP_MAX_SORT_NUMS;
  }
};
}
#endif