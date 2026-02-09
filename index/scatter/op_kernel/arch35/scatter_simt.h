/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_simt.h
 * \brief
 */

#ifndef ASCENDC_SCATTER_SIMT_H_
#define ASCENDC_SCATTER_SIMT_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace SCATTER {
using namespace AscendC;

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtOneCoreCompute0(__gm__ T1* updatesGm, __gm__ T2* indicesGm, 
                                                                         __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                         __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                         int64_t startOffset, int32_t batchNum) {
  for (uint32_t idx = Simt::GetThreadIdx(); idx < batchNum; idx += Simt::GetThreadNum()) {
    // Todo: change int32_t to int64_t
    int32_t addr = startOffset + idx;
    uint32_t i;
    uint32_t j;
    CalcIndex2<T3>(addr, i, j, factorArr, shiftArr, mArr);
    int32_t dstOffset = addr + i * coefArr[0] + j * coefArr[1] + (int32_t)(indicesGm[i]) * coefArr[7];
    outputGm[dstOffset] = updatesGm[startOffset+idx];
  }
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtOneCoreCompute1(__gm__ T1* updatesGm, __gm__ T2* indicesGm,
                                                                         __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                         __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                         int64_t startOffset, int32_t batchNum) {
  for (uint32_t idx = Simt::GetThreadIdx(); idx < batchNum; idx += Simt::GetThreadNum()) {
    // Todo: change int32_t to int64_t
    int32_t addr = startOffset + idx;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    CalcIndex3<T3>(addr, i, j, k, factorArr, shiftArr, mArr);

    int32_t dstOffset = addr + i * coefArr[4] + j * coefArr[5] + k * coefArr[6] + (int32_t)indicesGm[i];
    outputGm[dstOffset] = updatesGm[startOffset+idx];
  }
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtOneCoreCompute2(__gm__ T1* updatesGm, __gm__ T2* indicesGm,
                                                                         __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                         __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                         int64_t startOffset, int32_t batchNum) {
  for (uint32_t idx = Simt::GetThreadIdx(); idx < batchNum; idx += Simt::GetThreadNum()) {
    // Todo: change int32_t to int64_t
    int32_t addr = startOffset + idx;
    uint32_t i;
    uint32_t j;
    CalcIndex2<T3>(addr, i, j, factorArr, shiftArr, mArr);
    uint32_t dstOffset = addr + (uint32_t)indicesGm[i * 2] * coefArr[2] - i * coefArr[3] + j * coefArr[1] +
                          ((uint32_t)indicesGm[i * 2 + 1]) * coefArr[7];
    outputGm[dstOffset] = updatesGm[startOffset+idx];
  }
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtOneCoreCompute3(__gm__ T1* updatesGm, __gm__ T2* indicesGm,
                                                                         __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                         __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                         int64_t startOffset, int32_t batchNum) {
  for (uint32_t idx = Simt::GetThreadIdx(); idx < batchNum; idx += Simt::GetThreadNum()) {
    // Todo: change int32_t to int64_t
    int32_t addr = startOffset + idx;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    CalcIndex3<T3>(addr, i, j, k, factorArr, shiftArr, mArr);
    int32_t dstOffset = addr + (int32_t)indicesGm[i * 2] * coefArr[2]  - i * factorArr[0] + j * coefArr[5] +
                        k * coefArr[6] + (int32_t)indicesGm[i * 2 + 1];
    outputGm[dstOffset] = updatesGm[startOffset+idx];
  }
}

template<typename T1, typename T2, typename T3>
class ScatterSimt {
 public:
  __aicore__ inline ScatterSimt() {};

  inline __aicore__ void ScatterOneCoreComputeUint32(GlobalTensor<T1> updatesGm, GlobalTensor<T2> indicesGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                              __local_mem__ T3* shiftArr, __local_mem__ T3* mArr, int64_t startOffset, int32_t batchNum) {
    if (tiling->indicesDim == 1) {
      if (tiling->axis == SECOND_LAST_DIM) {
        Simt::VF_CALL<SimtOneCoreCompute0<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__gm__ T1*)(updatesGm.GetPhyAddr()), (__gm__ T2*)(indicesGm.GetPhyAddr()),
                                                (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, startOffset, batchNum);
      } else {
        Simt::VF_CALL<SimtOneCoreCompute1<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__gm__ T1*)(updatesGm.GetPhyAddr()), (__gm__ T2*)(indicesGm.GetPhyAddr()),
                                                (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, startOffset, batchNum);
      }
    } else {
      if (tiling->axis == SECOND_LAST_DIM) {
        Simt::VF_CALL<SimtOneCoreCompute2<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__gm__ T1*)(updatesGm.GetPhyAddr()), (__gm__ T2*)(indicesGm.GetPhyAddr()),
                                                (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, startOffset, batchNum);
      } else {
        Simt::VF_CALL<SimtOneCoreCompute3<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__gm__ T1*)(updatesGm.GetPhyAddr()), (__gm__ T2*)(indicesGm.GetPhyAddr()),
                                                (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, startOffset, batchNum);
      }
    }
  }

  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR indices,
                              GM_ADDR updates,
                              GM_ADDR y,
                              GM_ADDR workspace,
                              const ScatterTilingData* tilingData,
                              TPipe *pipeIn) {
    pipe = pipeIn;
    tiling = tilingData;

    pipe->InitBuffer(simtBuf_, SIMT_BUF_SIZE);
    int64_t inputCount = tiling->inputDim0 * tiling->inputDim1 * tiling->inputDim2 * tiling->inputDim3;
    inputGm.SetGlobalBuffer((__gm__ T1*)x, inputCount);
    // output reuse input
    outputGm.SetGlobalBuffer((__gm__ T1*)y, inputCount);

    // indicesCount = indicesDim[0] * indicesDim(1 or 2) = updatesDim[0] * indicesDim
    int64_t indicesCount = tiling->indicesDim * tiling->updatesDim0;
    indicesGm.SetGlobalBuffer((__gm__ T2*)indices, indicesCount);

    int64_t updatesCount = tiling->updatesDim0 * tiling->updatesDim1 * tiling->updatesDim2 * tiling->updatesDim3;
    updatesGm.SetGlobalBuffer((__gm__ T1*)updates, updatesCount);

    // coreNum = static_cast<uint32_t>(tiling->coreNum);
    usedCore = static_cast<uint32_t>(tiling->simtUsedCore);
    perCoreNum = static_cast<uint64_t>(tiling->simtPerCoreNum);
    tailCoreNum = static_cast<uint64_t>(tiling->simtTailCoreNum);

    blockIdx = GetBlockIdx();
    batchNum = (blockIdx == this->usedCore - 1) ? this->tailCoreNum : this->perCoreNum;

    factor0 = (T3)(tiling->updatesDim1 * tiling->updatesDim2 * tiling->updatesDim3);
    factor1 = (T3)(tiling->updatesDim2 * tiling->updatesDim3);
    factor4 = (T3)tiling->updatesDim3;

    coef0 = (T3)(tiling->inputDim2 - tiling->updatesDim2) * tiling->inputDim1 * tiling->inputDim3;
    coef1 = (T3)(tiling->inputDim2 - tiling->updatesDim2) * tiling->inputDim3;
    coef2 = (T3)(tiling->inputDim2 * tiling->inputDim1 * tiling->inputDim3);
    coef3 = (T3)(tiling->updatesDim2 * tiling->inputDim1 * tiling->inputDim3);
    coef4 = (T3)((tiling->inputDim3 - tiling->updatesDim3) * tiling->inputDim1 * tiling->inputDim2);
    coef5 = (T3)((tiling->inputDim3 - tiling->updatesDim3) * tiling->inputDim2);
    coef6 = (T3)(tiling->inputDim3 - tiling->updatesDim3);
    coef7 = (T3)(tiling->inputDim3);

    GetUintDivMagicAndShift(m0, shift0, factor0);
    GetUintDivMagicAndShift(m1, shift1, factor1);
    GetUintDivMagicAndShift(m4, shift4, factor4);
  }

  __aicore__ inline void Process() {
    using U = std::conditional_t<IsSameType<T3, uint32_t>::value, int32_t, int64_t>;
    // 整块核个数
    int32_t blockCoreNum = usedCore - 1;
    if (blockIdx < usedCore) {
      // 修改simt函数中的结构体入参为ub指针传入。
      LocalTensor<T3> factorArr = simtBuf_.Get<T3>(FACTOR_ARR_LEN);
      LocalTensor<T3> coefArr = simtBuf_.GetWithOffset<T3>(COEF_ARR_LEN, FACTOR_ARR_LEN * sizeof(T3));
      LocalTensor<T3> shiftArr = simtBuf_.GetWithOffset<T3>(SHIFT_ARR_LEN, (FACTOR_ARR_LEN + COEF_ARR_LEN) * sizeof(T3));
      LocalTensor<T3> mArr = simtBuf_.GetWithOffset<T3>(M_ARR_LEN, (FACTOR_ARR_LEN + COEF_ARR_LEN + SHIFT_ARR_LEN) * sizeof(T3));
      factorArr(INDEX_ZERO) = factor0;
      factorArr(INDEX_ONE) = factor1;
      factorArr(INDEX_TWO) = factor4;
      coefArr(INDEX_ZERO) = coef0;
      coefArr(INDEX_ONE) = coef1;
      coefArr(INDEX_TWO) = coef2;
      coefArr(INDEX_THREE) = coef3;
      coefArr(INDEX_FOUR) = coef4;
      coefArr(INDEX_FIVE) = coef5;
      coefArr(INDEX_SIX) = coef6;
      coefArr(INDEX_SEVEN) = coef7;
      shiftArr(INDEX_ZERO) = shift0;
      shiftArr(INDEX_ONE) = shift1;
      shiftArr(INDEX_TWO) = shift4;
      mArr(INDEX_ZERO) = m0;
      mArr(INDEX_ONE) = m1;
      mArr(INDEX_TWO) = m4;
      DataSyncBarrier<MemDsbT::UB>();
        
      U startOffset = perCoreNum * blockIdx;
      if(sizeof(T3) == sizeof(uint32_t)) {
      ScatterOneCoreComputeUint32(updatesGm, indicesGm, (__local_mem__ T3*)(factorArr.GetPhyAddr()), (__local_mem__ T3*)(coefArr.GetPhyAddr()), 
        (__local_mem__ T3*)(shiftArr.GetPhyAddr()), (__local_mem__ T3*)(mArr.GetPhyAddr()), startOffset, batchNum);
      }
    }
  }

 private:
  int64_t indicesUbSize;
  int64_t indicesBufferSize;
  int64_t updatesUbSize;
  int64_t updatesBufferSize;
  int64_t blockFactor;

  const ScatterTilingData *tiling;

  TPipe *pipe;
  TQue<QuePosition::VECIN, INDICES_BUFFER_NUM> indices;
  TQue<QuePosition::VECIN, UPDATES_BUFFER_NUM> updates;
  TBuf<TPosition::VECCALC> simtBuf_;

  GlobalTensor<T1> inputGm;
  GlobalTensor<T2> indicesGm;
  GlobalTensor<T1> updatesGm;
  GlobalTensor<T1> outputGm;

  int64_t usedCore;
  int64_t perCoreNum;
  int64_t tailCoreNum;

  uint64_t batchNum{ 0 };
  uint64_t blockIdx{ 0 };

  constexpr static uint32_t SIMT_BUF_SIZE = 1024;

  T3 factor0;
  T3 factor1;
  T3 factor4;
  T3 coef0;
  T3 coef1;
  T3 coef2;
  T3 coef3;
  T3 coef4;
  T3 coef5;
  T3 coef6;
  T3 coef7;

  T3 shift0;
  T3 shift1;
  T3 shift4;
  T3 m0;
  T3 m1;
  T3 m4;
};
}  // namespace Scatter

#endif