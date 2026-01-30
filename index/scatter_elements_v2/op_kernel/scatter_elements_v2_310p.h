/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2_310p.h
 * \brief
 */

#ifndef SCATTER_ELEMENTS_V2_310P
#define SCATTER_ELEMENTS_V2_310P

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace ScatterElementsV2310PNS {
using namespace AscendC;
constexpr uint64_t ALIGNED_LENGTH = 32;
constexpr uint64_t INDICES_UB_NUMS = 4096;
constexpr uint64_t VAR_UB_NUMS = 48000;
constexpr uint64_t VAR_ROWS = 64;
constexpr uint64_t TENSOR_UB_LENGTH = 8192; // 310P上可放大
constexpr uint64_t INDICES_LIMIT = 1024;
constexpr uint64_t INT32_ALIGNED_NUM = 8;

template <typename MT, typename U, typename CT>
class BaseOp {
public:
  __aicore__ inline BaseOp() {}

  __aicore__ inline void BaseInit(const ScatterElementsV2TilingData* tilingData, TPipe* tPipe) {
    this->pipe = tPipe;
    this->M = tilingData->M;
    this->varN = tilingData->varN;
    this->indicesN = tilingData->indicesN;
    this->updatesN = tilingData->updatesN;
    this->computeMode = tilingData->computeMode;
    this->frontCoreNum = tilingData->frontCoreNum;
    this->tailCoreNum = tilingData->tailCoreNum;
    this->frontCoreData = tilingData->frontCoreData;
    this->tailCoreData = tilingData->tailCoreData;
    this->coreNum = tilingData->usedCoreNum;
    this->MLeft = tilingData->MLeft;
  }

  __aicore__ inline void CopyOut(GlobalTensor<int8_t> gmTensor, LocalTensor<int8_t> ubTensor,
                                uint64_t mte3Start, uint64_t mteNum) {
    DataCopy(gmTensor[mte3Start], ubTensor, mteNum);
  }

  __aicore__ inline void CopyIn(GlobalTensor<int8_t> gmTensor, LocalTensor<int8_t> ubTensor,
                                uint64_t mte2Start, uint64_t mteNum) {
    DataCopy(ubTensor, gmTensor[mte2Start], mteNum);
  }

  __aicore__ inline void SWaitV() {
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
  }

  __aicore__ inline void SWaitMTE2() {
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
  }

  __aicore__ inline void SWaitMTE3() {
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
  }
protected:
  uint64_t M = 0;
  uint64_t varN = 0;
  uint64_t indicesN = 0;
  uint64_t updatesN = 0;
  uint64_t computeMode = 0;
  uint32_t coreNum = 0;
  uint64_t MLeft = 0;
  uint64_t frontCoreNum = 0;
  uint64_t tailCoreNum = 0;
  uint64_t frontCoreData = 0;
  uint64_t tailCoreData = 0;
  uint64_t coreStartM = 0;
  uint64_t coreMNums = 0;
  uint64_t coreEndM = 0;

  GlobalTensor<int8_t> varGm;
  GlobalTensor<int8_t> indicesGm;
  GlobalTensor<int8_t> updatesGm;

  LocalTensor<int8_t> varLocal;
  LocalTensor<int8_t> indicesLocal;
  LocalTensor<int8_t> updatesLocal;

  TPipe* pipe;
  TBuf<TPosition::VECCALC> varBuf;
  TBuf<TPosition::VECCALC> indicesBuf;
  TBuf<TPosition::VECCALC> updatesBuf;
};

template <typename MT, typename U, typename CT>
class MultyCoreMOp : public BaseOp<MT, U, CT> {
public:
  __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                              const ScatterElementsV2TilingData* tilingData, TPipe* tPipe) {
    this->BaseInit(tilingData, tPipe);
    if (GetBlockIdx() < this->frontCoreNum) {
      this->coreStartM = GetBlockIdx() * this->frontCoreData;
      this->coreEndM = this->coreStartM + this->frontCoreData;
    } else {
      this->coreStartM = this->frontCoreNum * this->frontCoreData +
                         (GetBlockIdx() - this->frontCoreNum) * this->tailCoreData;
      this->coreEndM = this->coreStartM + this->tailCoreData;
    }

    this->varGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(var));
    this->indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(indices));
    this->updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(updates));
    this->pipe->InitBuffer(this->varBuf, VAR_UB_NUMS * sizeof(float));
    this->pipe->InitBuffer(this->indicesBuf, INDICES_UB_NUMS * sizeof(U));
    this->pipe->InitBuffer(this->updatesBuf, INDICES_UB_NUMS * sizeof(float));
    this->varLocal = this->varBuf.template Get<int8_t>();
    this->indicesLocal = this->indicesBuf.template Get<int8_t>();
    this->updatesLocal = this->updatesBuf.template Get<int8_t>();
  }

  __aicore__ inline uint64_t AlignedToTarget(uint64_t value, uint64_t target) {
    if (target == 0) {
        return value;
    } else {
        return ((value + target - 1) / target) * target;
    }
  }

  __aicore__ inline void ProcessIndicesLeft(uint64_t rowId, uint64_t indicesBlocks, uint64_t indicesLeft,
                                            uint64_t validIndicesStart, uint64_t validIndicesEnd, uint64_t varBak = 0) {
    uint64_t indicesLeftAligned = AlignedToTarget(indicesLeft, ALIGNED_LENGTH / sizeof(U));
    uint64_t alignedBakIndices = indicesLeftAligned - indicesLeft;
    uint64_t mte2Start = rowId * this->indicesN * sizeof(U) + indicesBlocks * INDICES_UB_NUMS * sizeof(U) - alignedBakIndices * sizeof(U);
    if (mte2Start < 0) { //防止后退越界
      alignedBakIndices = alignedBakIndices + mte2Start / sizeof(U);
      mte2Start = 0;
    }
    this->CopyIn(this->indicesGm, this->indicesLocal, mte2Start, indicesLeftAligned * sizeof(U));
    uint64_t updatesLengthAligned = AlignedToTarget(indicesLeft, ALIGNED_LENGTH / sizeof(MT));
    uint64_t alignedBakUpdates = updatesLengthAligned - indicesLeft;
    mte2Start = rowId * this->updatesN * sizeof(MT) + indicesBlocks * INDICES_UB_NUMS * sizeof(MT) - alignedBakUpdates * sizeof(MT);
    if (mte2Start < 0) {
      alignedBakUpdates = alignedBakUpdates + mte2Start / sizeof(MT);
      mte2Start = 0;
    }
    this->CopyIn(this->updatesGm, this->updatesLocal, mte2Start, updatesLengthAligned * sizeof(MT));
    this->SWaitMTE2();
    LocalTensor<MT> varLocalTemp = this->varLocal.template ReinterpretCast<MT>();
    LocalTensor<U> indicesLocalTemp = this->indicesLocal.template ReinterpretCast<U>();
    LocalTensor<MT> updatesLocalTemp = this->updatesLocal.template ReinterpretCast<MT>();
    if (this->computeMode == 0) {
      for (uint64_t i = 0; i < indicesLeft; i++) {
        U indicesValue = indicesLocalTemp.GetValue(i + alignedBakIndices);
        indicesValue = indicesValue >=0 ? indicesValue : indicesValue + this->varN;
        if (indicesValue < validIndicesStart || indicesValue >= validIndicesEnd) {
          continue;
        }
        MT updatesValue = updatesLocalTemp.GetValue(i + alignedBakUpdates);
        varLocalTemp.template SetValue(indicesValue - validIndicesStart + varBak, updatesValue);
      }
    } else if (this->computeMode == 1) {
      for (uint64_t i = 0; i < indicesLeft; i++) {
        U indicesValue = indicesLocalTemp.GetValue(i + alignedBakIndices);
        indicesValue = indicesValue >=0 ? indicesValue : indicesValue + this->varN;
        if (indicesValue < validIndicesStart || indicesValue >= validIndicesEnd) {
          continue;
        }
        MT updatesValue = updatesLocalTemp.GetValue(i + alignedBakUpdates);
        MT varValue = varLocalTemp.GetValue(indicesValue - validIndicesStart + varBak);
        CT sumValue = static_cast<CT>(updatesValue) + static_cast<CT>(varValue);
        varLocalTemp.template SetValue(indicesValue - validIndicesStart + varBak, static_cast<MT>(sumValue));
      }
    }
  }

  __aicore__ inline void ProcessIndicesBlock(uint64_t rowId, uint64_t blockId, uint64_t validIndicesStart,
                                             uint64_t validIndicesEnd, uint64_t varBak = 0) {
    // 搬入索引块，和updates块
    uint64_t mte2Start = rowId * this->indicesN * sizeof(U) + blockId * INDICES_UB_NUMS * sizeof(U);
    this->CopyIn(this->indicesGm, this->indicesLocal, mte2Start, INDICES_UB_NUMS * sizeof(U));
    mte2Start = rowId * this->updatesN * sizeof(MT) + blockId * INDICES_UB_NUMS * sizeof(MT);
    this->CopyIn(this->updatesGm, this->updatesLocal, mte2Start, INDICES_UB_NUMS * sizeof(MT));
    this->SWaitMTE2();
    // 迭代索引值
    LocalTensor<MT> varLocalTemp = this->varLocal.template ReinterpretCast<MT>();
    LocalTensor<U> indicesLocalTemp = this->indicesLocal.template ReinterpretCast<U>();
    LocalTensor<MT> updatesLocalTemp = this->updatesLocal.template ReinterpretCast<MT>();
    if (this->computeMode == 0) {
      for (uint64_t i = 0; i < INDICES_UB_NUMS; i++) {
        U indicesValue = indicesLocalTemp.GetValue(i);
        indicesValue = indicesValue >=0 ? indicesValue : indicesValue + this->varN;
        if (indicesValue < validIndicesStart || indicesValue >= validIndicesEnd) {
          continue;
        }
        MT updatesValue = updatesLocalTemp.GetValue(i);
        varLocalTemp.template SetValue(indicesValue - validIndicesStart + varBak, updatesValue);
      }
    } else if (this->computeMode == 1) {
      for (uint64_t i = 0; i < INDICES_UB_NUMS; i++) {
        U indicesValue = indicesLocalTemp.GetValue(i);
        indicesValue = indicesValue >=0 ? indicesValue : indicesValue + this->varN;
        if (indicesValue < validIndicesStart || indicesValue >= validIndicesEnd) {
          continue;
        }
        MT updatesValue = updatesLocalTemp.GetValue(i);
        MT varValue = varLocalTemp.GetValue(indicesValue - validIndicesStart + varBak);
        CT sumValue = static_cast<CT>(updatesValue) + static_cast<CT>(varValue);
        varLocalTemp.template SetValue(indicesValue - validIndicesStart + varBak, static_cast<MT>(sumValue));
      }
    }
  }

  __aicore__ inline void ProcessOneBlock(uint64_t rowId, uint64_t blockId) {
    // 1.搬入var
    uint64_t mte2Start = rowId * this->varN * sizeof(MT) + blockId * VAR_UB_NUMS * sizeof(MT);
    this->CopyIn(this->varGm, this->varLocal, mte2Start, VAR_UB_NUMS * sizeof(MT));
    this->SWaitMTE2();
    // 2.计算指向这段var的索引范围
    uint64_t validIndicesStart = blockId * VAR_UB_NUMS;
    uint64_t validIndicesEnd = validIndicesStart + VAR_UB_NUMS;
    // 3.遍历该行所有索引值
    // 3.1 切分indices与updates
    uint64_t indicesBlocks = this->indicesN / INDICES_UB_NUMS;
    uint64_t indicesLeft = this->indicesN - indicesBlocks * INDICES_UB_NUMS;
    // 3.2 处理整块索引
    for (uint64_t i = 0; i < indicesBlocks; i++) {
      ProcessIndicesBlock(rowId, i, validIndicesStart, validIndicesEnd);
    }
    // 3.3 处理尾块索引
    if (indicesLeft > 0) {
      ProcessIndicesLeft(rowId, indicesBlocks, indicesLeft, validIndicesStart, validIndicesEnd);
    }
    // 4.搬出var
    PipeBarrier<PIPE_ALL>();
    this->CopyOut(this->varGm, this->varLocal, mte2Start, VAR_UB_NUMS * sizeof(MT));
    PipeBarrier<PIPE_ALL>();
    this->SWaitMTE3();
  }

  __aicore__ inline void ProcessVarLeft(uint64_t rowId, uint64_t frontNums, uint64_t nums) {
    uint64_t numsAligned = AlignedToTarget(nums, ALIGNED_LENGTH / sizeof(MT));
    uint64_t alignedBak = numsAligned - nums;
    // 1.搬入var
    uint64_t mte2Start = rowId * this->varN * sizeof(MT) + frontNums * sizeof(MT) - alignedBak * sizeof(MT);
    if (mte2Start < 0) {
      alignedBak = alignedBak + mte2Start / sizeof(MT);
      mte2Start = 0;
    }
    this->CopyIn(this->varGm, this->varLocal, mte2Start, numsAligned * sizeof(MT));
    this->SWaitMTE2();
    // 2.计算指向这段var的索引范围
    uint64_t validIndicesStart = frontNums;
    uint64_t validIndicesEnd = validIndicesStart + nums;
    // 3.遍历该行所有索引值
    // 3.1 切分indices与updates
    uint64_t indicesBlocks = this->indicesN / INDICES_UB_NUMS;
    uint64_t indicesLeft = this->indicesN - indicesBlocks * INDICES_UB_NUMS;
    // 3.2 处理整块索引
    for (uint64_t i = 0; i < indicesBlocks; i++) {
      ProcessIndicesBlock(rowId, i, validIndicesStart, validIndicesEnd, alignedBak);
    }
    // 3.3 处理尾块索引
    if (indicesLeft > 0) {
      ProcessIndicesLeft(rowId, indicesBlocks, indicesLeft, validIndicesStart, validIndicesEnd, alignedBak);
    }
    // 4.搬出var
    PipeBarrier<PIPE_ALL>();
    this->CopyOut(this->varGm, this->varLocal, mte2Start, numsAligned * sizeof(MT));
    PipeBarrier<PIPE_ALL>();
    this->SWaitMTE3();
  }

  __aicore__ inline void ProcessOneRow(uint64_t rowId) {
    // 切分该行var
    uint64_t varBlocks = this->varN / VAR_UB_NUMS;
    uint64_t varLeft = this->varN - varBlocks * VAR_UB_NUMS;
    for (uint64_t i = 0; i < varBlocks; i++) {
      ProcessOneBlock(rowId, i);
    }
    if (varLeft > 0) {
      uint64_t alignedNum = varLeft / (ALIGNED_LENGTH / sizeof(MT)) * (ALIGNED_LENGTH / sizeof(MT));
      uint64_t unAlignedNum = varLeft - alignedNum;
      // 对齐部分直接搬入处理，非对齐部分回退搬入处理
      if (alignedNum != 0) {
        ProcessVarLeft(rowId, varBlocks * VAR_UB_NUMS, alignedNum);
      }
      if (unAlignedNum != 0) {
        ProcessVarLeft(rowId, varBlocks * VAR_UB_NUMS + alignedNum, unAlignedNum);
      }
    }
  }

  __aicore__ inline void Process() {
    for (uint64_t i = this->coreStartM; i < this->coreEndM; i++) {
      ProcessOneRow(i);
    }
  }
};

template <typename MT, typename U, typename CT>
class MultyMOp : public BaseOp<MT, U, CT> {
public:
  __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                              const ScatterElementsV2TilingData* tilingData, TPipe* tPipe) {
    this->BaseInit(tilingData, tPipe);
    if (GetBlockIdx() == this->coreNum - 1) {
        this->coreStartM = (this->frontCoreNum * this->frontCoreData +
                           this->tailCoreNum * this->tailCoreData) * VAR_ROWS;
        this->coreMNums = this->MLeft;
    } else if (GetBlockIdx() < this->frontCoreNum) {
      this->coreStartM = GetBlockIdx() * this->frontCoreData * VAR_ROWS;
      this->coreMNums = this->frontCoreData * VAR_ROWS;
    } else {
      this->coreStartM = (this->frontCoreNum * this->frontCoreData +
                         (GetBlockIdx() - this->frontCoreNum) * this->tailCoreData) * VAR_ROWS;
      this->coreMNums = this->tailCoreData * VAR_ROWS;
    }

    this->varGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(var));
    this->indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(indices));
    this->updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(updates));

    this->pipe->InitBuffer(this->varBuf, TENSOR_UB_LENGTH * sizeof(MT));
    this->pipe->InitBuffer(this->indicesBuf, TENSOR_UB_LENGTH * sizeof(U));
    this->pipe->InitBuffer(this->updatesBuf, TENSOR_UB_LENGTH * sizeof(MT));
    this->varLocal = this->varBuf.template Get<int8_t>();
    this->indicesLocal = this->indicesBuf.template Get<int8_t>();
    this->updatesLocal = this->updatesBuf.template Get<int8_t>();
  }

  __aicore__ inline uint64_t AlignedToTarget(uint64_t value, uint64_t target) {
    if (target == 0) {
        return value;
    } else {
        return ((value + target - 1) / target) * target;
    }
  }

  __aicore__ inline void CopyInIndicesAndUpdates(uint64_t startRow, uint64_t rows) {
    uint64_t mte2Start = startRow * this->indicesN * sizeof(U);
    uint64_t mteNum = AlignedToTarget(rows * this->indicesN * sizeof(U), ALIGNED_LENGTH);
    this->CopyIn(this->indicesGm, this->indicesLocal, mte2Start, mteNum);
    mte2Start = startRow * this->updatesN * sizeof(MT);;
    mteNum = AlignedToTarget(rows * this->updatesN * sizeof(MT), ALIGNED_LENGTH);
    this->CopyIn(this->updatesGm, this->updatesLocal, mte2Start, mteNum);
    this->SWaitMTE2();
  }

  __aicore__ inline void ProcessMultyIndices(uint64_t mteRows, uint64_t frontRows) {
    LocalTensor<MT> varLocalTemp = this->varLocal.template ReinterpretCast<MT>();
    LocalTensor<U> indicesLocalTemp = this->indicesLocal.template ReinterpretCast<U>();
    LocalTensor<MT> updatesLocalTemp = this->updatesLocal.template ReinterpretCast<MT>();
    // 按行处理索引和updates
    if (this->computeMode == 0) {
        for (uint64_t i = 0; i < mteRows; i++) {
            for (uint64_t j = 0; j < this->indicesN; j++) {
                U indicesValue = indicesLocalTemp.GetValue(i * this->indicesN + j);
                indicesValue = indicesValue >=0 ? indicesValue : indicesValue + this->varN;
                if (indicesValue >= this->varN)  {
                    continue;
                }
                MT updatesValue = updatesLocalTemp.GetValue(i * this->updatesN + j);
                varLocalTemp.SetValue(frontRows * this->varN + i * this->varN + indicesValue, updatesValue);
            }
        }
    } else if (this->computeMode == 1) {
        for (uint64_t i = 0; i < mteRows; i++) {
            for (uint64_t j = 0; j < this->indicesN; j++) {
                U indicesValue = indicesLocalTemp.GetValue(i * this->indicesN + j);
                indicesValue = indicesValue >=0 ? indicesValue : indicesValue + this->varN;
                if (indicesValue >= this->varN)  {
                    continue;
                }
                MT updatesValue = updatesLocalTemp.GetValue(i * this->updatesN + j);
                MT varValue = varLocalTemp.GetValue(frontRows * this->varN + i * this->varN + indicesValue);
                CT sumValue = static_cast<CT>(updatesValue) + static_cast<CT>(varValue);
                varLocalTemp.SetValue(frontRows * this->varN + i * this->varN + indicesValue,
                                      static_cast<MT>(sumValue));
            }
        }
    }
  }

  __aicore__ inline void ProcessBigIndices(uint64_t rows, uint64_t startRow) {
    // indices、updates按行搬入
    uint64_t indicesBlocks = this->indicesN / TENSOR_UB_LENGTH;
    uint64_t indicesLeft = this->indicesN - indicesBlocks * TENSOR_UB_LENGTH;
    for(uint64_t i = 0; i < rows; i++) {
        for(uint64_t j = 0; j <= indicesBlocks; j++) {
            if (j == indicesBlocks && indicesLeft == 0) {
                break;
            }
            uint64_t indicesLength = (j == indicesBlocks ? indicesLeft : TENSOR_UB_LENGTH);
            // 搬入indices
            uint64_t indicesLengthAligned = AlignedToTarget(indicesLength, ALIGNED_LENGTH / sizeof(U));
            uint64_t alignedBakIndices = indicesLengthAligned - indicesLength;
            uint64_t mte2Start = (startRow + i) * this->indicesN * sizeof(U) + j * TENSOR_UB_LENGTH * sizeof(U) - alignedBakIndices * sizeof(U);
            if (mte2Start < 0) {
                alignedBakIndices = alignedBakIndices + mte2Start / sizeof(U);
                mte2Start = 0;
            }
            this->CopyIn(this->indicesGm, this->indicesLocal, mte2Start, indicesLengthAligned * sizeof(U));
            uint64_t updatesLengthAligned = AlignedToTarget(indicesLength, ALIGNED_LENGTH / sizeof(MT));
            uint64_t alignedBakUpdates = updatesLengthAligned - indicesLength;
            mte2Start = (startRow + i) * this->updatesN * sizeof(MT) + j * TENSOR_UB_LENGTH * sizeof(MT) - alignedBakUpdates * sizeof(MT);
            if (mte2Start < 0) {
                alignedBakUpdates = alignedBakUpdates + mte2Start / sizeof(MT);
                mte2Start = 0;
            }
            this->CopyIn(this->updatesGm, this->updatesLocal, mte2Start, updatesLengthAligned * sizeof(MT));
            this->SWaitMTE2();

            LocalTensor<MT> varLocalTemp = this->varLocal.template ReinterpretCast<MT>();
            LocalTensor<U> indicesLocalTemp = this->indicesLocal.template ReinterpretCast<U>();
            LocalTensor<MT> updatesLocalTemp = this->updatesLocal.template ReinterpretCast<MT>();
            if (this->computeMode == 0) {
                for (uint64_t k = 0; k < indicesLength; k++) {
                    U indicesValue = indicesLocalTemp.GetValue(k + alignedBakIndices);
                    indicesValue = indicesValue >=0 ? indicesValue : indicesValue + this->varN;
                    if (indicesValue >= this->varN) {
                        continue;
                    }
                    MT updatesValue = updatesLocalTemp.GetValue(k + alignedBakUpdates);
                    varLocalTemp.template SetValue(i * this->varN + indicesValue, updatesValue);
                }
            } else if (this->computeMode == 1) {
                for (uint64_t k = 0; k < indicesLength; k++) {
                    U indicesValue = indicesLocalTemp.GetValue(k + alignedBakIndices);
                    indicesValue = indicesValue >=0 ? indicesValue : indicesValue + this->varN;
                    if (indicesValue >= this->varN) {
                        continue;
                    }
                    MT updatesValue = updatesLocalTemp.GetValue(k + alignedBakUpdates);
                    MT varValue = varLocalTemp.GetValue(i * this->varN + indicesValue);
                    CT sumValue = static_cast<CT>(updatesValue) + static_cast<CT>(varValue);
                    varLocalTemp.template SetValue(i * this->varN + indicesValue, static_cast<MT>(sumValue));
                }
            }
        }
    }
  }

  __aicore__ inline void ProcessRows(uint64_t rows, uint64_t startRow) {
    // 搬入rows行var
    uint64_t mte2Start = startRow * this->varN * sizeof(MT);
    uint64_t mteNum = AlignedToTarget(rows * this->varN * sizeof(MT), ALIGNED_LENGTH);
    this->CopyIn(this->varGm, this->varLocal, mte2Start, mteNum);
    this->SWaitMTE2();
    if (this->indicesN <= INDICES_LIMIT && this->updatesN <= INDICES_LIMIT) {
       uint64_t indicesRows = ((TENSOR_UB_LENGTH / this->indicesN) / INT32_ALIGNED_NUM) * INT32_ALIGNED_NUM;
       uint64_t times = rows / indicesRows;
       uint64_t left = rows - times * indicesRows;
       for (uint64_t i = 0; i <= times; i++) {
          if (i == times && left == 0) {
            break;
          }
          uint64_t mteRows = (i == times ? left : indicesRows);
          // 搬入mteRows行indices、updates
          CopyInIndicesAndUpdates(startRow + i * indicesRows, mteRows);
          ProcessMultyIndices(mteRows, i * indicesRows);
       }
    } else {
       ProcessBigIndices(rows, startRow);
    }

    PipeBarrier<PIPE_ALL>();
    this->CopyOut(this->varGm, this->varLocal, mte2Start, mteNum);
    PipeBarrier<PIPE_ALL>();
    this->SWaitMTE3();
  }

  __aicore__ inline void Process() {
    uint64_t mBlocks = this->coreMNums / VAR_ROWS;
    uint64_t mLeft = this->coreMNums - mBlocks * VAR_ROWS;
    for (uint64_t i = 0; i <= mBlocks; i++) {
        if (i == mBlocks && mLeft == 0) {
            break;
        }
        uint64_t rows = (i == mBlocks ? mLeft : VAR_ROWS);
        uint64_t startRow = this->coreStartM + i * VAR_ROWS;
        ProcessRows(rows, startRow);
    }
  }
};

template <typename T, typename U>
__aicore__ void scatter_elements_v2_310p_split_m(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                 const ScatterElementsV2TilingData* tilingData, TPipe* tPipe) {
    if constexpr (std::is_same<T, half>::value) {
        MultyCoreMOp<T, U, float> op;
        op.Init(var, indices, updates, tilingData, tPipe);
        op.Process();
    } else {
        MultyCoreMOp<T, U, T> op;
        op.Init(var, indices, updates, tilingData, tPipe);
        op.Process();
    }
}

template <typename T, typename U>
__aicore__ void scatter_elements_v2_310p_multy_m(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                 const ScatterElementsV2TilingData* tilingData, TPipe* tPipe) {
    if constexpr (std::is_same<T, half>::value) {
        MultyMOp<T, U, float> op;
        op.Init(var, indices, updates, tilingData, tPipe);
        op.Process();
    } else {
        MultyMOp<T, U, T> op;
        op.Init(var, indices, updates, tilingData, tPipe);
        op.Process();
    }
}
}
#endif