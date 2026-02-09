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
 * \file scatter.h
 * \brief
 */

#ifndef SCATTER_IMPL_H
#define SCATTER_IMPL_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace SCATTER {
using namespace AscendC;
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
#else
constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 1024;
#endif
constexpr uint32_t BLOCK_SIZE = 32;
constexpr int32_t INDICES_BUFFER_NUM = 2;
constexpr int32_t UPDATES_BUFFER_NUM = 2;
constexpr int32_t UB_DIVISION_COUNT = 2;
constexpr int32_t BIT64_SIZE = 8;
constexpr int32_t SECOND_LAST_DIM = -2;
constexpr uint64_t ONE = 1;
constexpr int32_t MIN_FACTOR = 1024;
constexpr int32_t BIT64 = 64;
constexpr int32_t BIT32 = 32;

constexpr int32_t SIMT_CALC_NUM = 3;
constexpr int32_t UINT32_SIZE_BYTE = 4;
constexpr int32_t UINT64_SIZE_BYTE = 8;
constexpr int32_t SIMT_OFFSET_NUM = 17;
constexpr int32_t SIMT_ARG_SIZE = 256; // actually use 17 * 8 + 3 * 8 + 3 * 4 = 172B
constexpr int8_t INDEX_ZERO = 0;
constexpr int8_t INDEX_ONE = 1;
constexpr int8_t INDEX_TWO = 2;
constexpr int8_t INDEX_THREE = 3;
constexpr int8_t INDEX_FOUR = 4;
constexpr int8_t INDEX_FIVE = 5;
constexpr int8_t INDEX_SIX = 6;
constexpr int8_t INDEX_SEVEN = 7;
constexpr int8_t FACTOR_ARR_LEN = 3;
constexpr int8_t COEF_ARR_LEN = 8;
constexpr int8_t SHIFT_ARR_LEN = 3;
constexpr int8_t M_ARR_LEN = 3;

template<typename T3>
__aicore__ inline void CalcIndex2(int32_t addr, uint32_t &i, uint32_t &j, __local_mem__ T3* factorArr, __local_mem__ T3* shiftArr, __local_mem__ T3* mArr) {
    // fast division, addr / factor0
    uint32_t t = Simt::MulHi(static_cast<uint32_t>(addr), static_cast<uint32_t>(mArr[INDEX_ZERO]));
    t = t + addr;
    i = t >> shiftArr[INDEX_ZERO];
    int32_t remain = addr - i * factorArr[INDEX_ZERO];

    t = Simt::MulHi(static_cast<uint32_t>(remain), static_cast<uint32_t>(mArr[INDEX_ONE]));
    t = t + remain;
    j = t >> shiftArr[INDEX_ONE];
}

template<typename T3>
__aicore__ inline void CalcIndex3(int32_t addr, uint32_t &i, uint32_t &j, uint32_t &k, __local_mem__ T3* factorArr, __local_mem__ T3* shiftArr, __local_mem__ T3* mArr) {
    // fast division, addr / factor0
    uint32_t t = Simt::MulHi(static_cast<uint32_t>(addr), static_cast<uint32_t>(mArr[INDEX_ZERO]));
    t = t + addr;
    i = t >> shiftArr[INDEX_ZERO];
    int32_t remain = addr - i * factorArr[INDEX_ZERO];

    t = Simt::MulHi(static_cast<uint32_t>(remain), static_cast<uint32_t>(mArr[INDEX_ONE]));
    t = t + remain;
    j = t >> shiftArr[INDEX_ONE];
    remain = remain - j * factorArr[INDEX_ONE];

    t = Simt::MulHi(static_cast<uint32_t>(remain), static_cast<uint32_t>(mArr[INDEX_TWO]));
    t = t + remain;
    k = t >> shiftArr[INDEX_TWO];
}

template<typename T3>
__aicore__ inline void CalcUint64Index2(int64_t addr, uint64_t &i, uint64_t &j, __local_mem__ T3* factorArr, __local_mem__ T3* shiftArr, __local_mem__ T3* mArr) {
    // uint64_t &i, uint64_t &j
    i = Simt::UintDiv(static_cast<uint64_t>(addr), static_cast<uint64_t>(mArr[INDEX_ZERO]), static_cast<uint64_t>(shiftArr[INDEX_ZERO]));
    int64_t remain = addr - i * factorArr[INDEX_ZERO];
    j = Simt::UintDiv(static_cast<uint64_t>(remain), static_cast<uint64_t>(mArr[INDEX_ONE]), static_cast<uint64_t>(shiftArr[INDEX_ONE]));
}

template<typename T3>
__aicore__ inline void CalcUint64Index3(int64_t addr, uint64_t &i, uint64_t &j, uint64_t &k, __local_mem__ T3* factorArr, __local_mem__ T3* shiftArr, __local_mem__ T3* mArr) {
  // uint64_t &i, uint64_t &j, uint64_t &k
    i = Simt::UintDiv(static_cast<uint64_t>(addr), static_cast<uint64_t>(mArr[INDEX_ZERO]), static_cast<uint64_t>(shiftArr[INDEX_ZERO]));
    int64_t remain = addr - i * factorArr[INDEX_ZERO];
    j = Simt::UintDiv(static_cast<uint64_t>(remain), static_cast<uint64_t>(mArr[INDEX_ONE]), static_cast<uint64_t>(shiftArr[INDEX_ONE]));
    remain = remain - j * factorArr[INDEX_ONE];
    k = Simt::UintDiv(static_cast<uint64_t>(remain), static_cast<uint64_t>(mArr[INDEX_TWO]), static_cast<uint64_t>(shiftArr[INDEX_TWO]));
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtCompute0(__local_mem__ T1* updatesUb, __local_mem__ T2* indicesUb, 
                                                                         __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                         __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                         int32_t indexStart, int32_t startOffset, int32_t ubCount) {
  for (uint32_t idx = Simt::GetThreadIdx(); idx < ubCount; idx += Simt::GetThreadNum()) {
    // Todo: change int32_t to int64_t
    int32_t addr = startOffset + idx;
    uint32_t i;
    uint32_t j;
    CalcIndex2<T3>(addr, i, j, factorArr, shiftArr, mArr);
    int32_t dstOffset = addr + i * coefArr[INDEX_ZERO] + j * coefArr[INDEX_ONE] + (int32_t)(indicesUb[i - indexStart]) * coefArr[INDEX_SEVEN];
    outputGm[dstOffset] = updatesUb[idx];
  }
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtCompute1(__local_mem__ T1* updatesUb, __local_mem__ T2* indicesUb,
                                                                         __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                         __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                         int32_t indexStart, int32_t startOffset, int32_t ubCount) {
  for (uint32_t idx = Simt::GetThreadIdx(); idx < ubCount; idx += Simt::GetThreadNum()) {
    // Todo: change int32_t to int64_t
    int32_t addr = startOffset + idx;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    CalcIndex3<T3>(addr, i, j, k, factorArr, shiftArr, mArr);

    int32_t dstOffset = addr + i * coefArr[INDEX_FOUR] + j * coefArr[INDEX_FIVE] + k * coefArr[INDEX_SIX] + (int32_t)indicesUb[i - indexStart];
    outputGm[dstOffset] = updatesUb[idx];
  }
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtCompute2(__local_mem__ T1* updatesUb, __local_mem__ T2* indicesUb,
                                                                         __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                         __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                         int32_t indexStart, int32_t startOffset, int32_t ubCount) {
  for (uint32_t idx = Simt::GetThreadIdx(); idx < ubCount; idx += Simt::GetThreadNum()) {
    // Todo: change int32_t to int64_t
    int32_t addr = startOffset + idx;
    uint32_t i;
    uint32_t j;
    CalcIndex2<T3>(addr, i, j, factorArr, shiftArr, mArr);
    uint32_t dstOffset = addr + (uint32_t)indicesUb[(i - indexStart) * 2] * coefArr[INDEX_TWO] - i * coefArr[INDEX_THREE] + j * coefArr[INDEX_ONE] +
                          ((uint32_t)indicesUb[(i - indexStart) * 2 + 1]) * coefArr[INDEX_SEVEN];
    outputGm[dstOffset] = updatesUb[idx];
  }
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtCompute3(__local_mem__ T1* updatesUb, __local_mem__ T2* indicesUb,
                                                                         __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                         __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                         int32_t indexStart, int32_t startOffset, int32_t ubCount) {
  for (uint32_t idx = Simt::GetThreadIdx(); idx < ubCount; idx += Simt::GetThreadNum()) {
    // Todo: change int32_t to int64_t
    int32_t addr = startOffset + idx;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    CalcIndex3<T3>(addr, i, j, k, factorArr, shiftArr, mArr);
    int32_t dstOffset = addr + (int32_t)indicesUb[(i - indexStart) * 2] * coefArr[INDEX_TWO] - i * factorArr[INDEX_ZERO] + j * coefArr[INDEX_FIVE] +
                        k * coefArr[INDEX_SIX] + (int32_t)indicesUb[(i - indexStart) * 2 + 1];
    outputGm[dstOffset] = updatesUb[idx];
  }
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtUint64Compute0(__local_mem__ T1* updatesUb, __local_mem__ T2* indicesUb,
                                                                               __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                               __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                               int64_t indexStart, int64_t startOffset, int64_t ubCount) {
  for (int64_t idx = Simt::GetThreadIdx(); idx < ubCount; idx += Simt::GetThreadNum()) {
    int64_t addr = startOffset + idx;
    uint64_t i;
    uint64_t j;
    CalcUint64Index2(addr, i, j, factorArr, shiftArr, mArr);
    int64_t dstOffset = addr + i * coefArr[INDEX_ZERO] + j * coefArr[INDEX_ONE] + (int64_t)(indicesUb[i - indexStart]) * coefArr[INDEX_SEVEN];
    outputGm[dstOffset] = updatesUb[idx];
  }
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtUint64Compute1(__local_mem__ T1* updatesUb, __local_mem__ T2* indicesUb,
                                                                               __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                               __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                               int64_t indexStart, int64_t startOffset, int64_t ubCount) {
  for (int64_t idx = Simt::GetThreadIdx(); idx < ubCount; idx += Simt::GetThreadNum()) {
    int64_t addr = startOffset + idx;
    uint64_t i;
    uint64_t j;
    uint64_t k;
    CalcUint64Index3(addr, i, j, k, factorArr, shiftArr, mArr);
    int64_t dstOffset = addr + i * coefArr[INDEX_FOUR] + j * coefArr[INDEX_FIVE] + k * coefArr[INDEX_SIX] + (int64_t)indicesUb[i - indexStart];
    outputGm[dstOffset] = updatesUb[idx];
  }
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtUint64Compute2(__local_mem__ T1* updatesUb, __local_mem__ T2* indicesUb,
                                                                               __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                               __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                               int64_t indexStart, int64_t startOffset, int64_t ubCount) {
  for (int64_t idx = Simt::GetThreadIdx(); idx < ubCount; idx += Simt::GetThreadNum()) {
    int64_t addr = startOffset + idx;
    uint64_t i;
    uint64_t j;
    CalcUint64Index2(addr, i, j, factorArr, shiftArr, mArr);
    uint64_t dstOffset = addr + (uint64_t)indicesUb[(i - indexStart) * 2] * coefArr[INDEX_TWO] - i * coefArr[INDEX_THREE] + j * coefArr[INDEX_ONE] +
                           ((uint64_t)indicesUb[(i - indexStart) * 2 + 1]) * coefArr[INDEX_SEVEN];
    outputGm[dstOffset] = updatesUb[idx];
  }
}

template<typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtUint64Compute3(__local_mem__ T1* updatesUb, __local_mem__ T2* indicesUb,
                                                                               __gm__ T1* outputGm, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                                                               __local_mem__ T3* shiftArr, __local_mem__ T3* mArr,
                                                                               int64_t indexStart, int64_t startOffset, int64_t ubCount) {
  for (int64_t idx = Simt::GetThreadIdx(); idx < ubCount; idx += Simt::GetThreadNum()) {
    int64_t addr = startOffset + idx;
    uint64_t i;
    uint64_t j;
    uint64_t k;
    CalcUint64Index3(addr, i, j, k, factorArr, shiftArr, mArr);
    int64_t dstOffset = addr + (int64_t)indicesUb[(i - indexStart) * 2] * coefArr[INDEX_TWO] - i * factorArr[INDEX_ZERO] + j * coefArr[INDEX_FIVE] +
                          k * coefArr[INDEX_SIX] + (int64_t)indicesUb[(i - indexStart) * 2 + 1];
    outputGm[dstOffset] = updatesUb[idx];
  }
}

template<typename T1, typename T2, typename T3>
class Scatter {
 public:
  __aicore__ inline Scatter() {};

  inline __aicore__ void ScatterComputeUint32(LocalTensor<T1> updatesUb, LocalTensor<T2> indicesUb, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                              __local_mem__ T3* shiftArr, __local_mem__ T3* mArr, int32_t indexStart, int32_t startOffset, int32_t ubCount) {
    if (tiling->indicesDim == 1) {
      if (tiling->axis == SECOND_LAST_DIM) {
        Simt::VF_CALL<SimtCompute0<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__local_mem__ T1*)(updatesUb.GetPhyAddr()), (__local_mem__ T2*)(indicesUb.GetPhyAddr()),
                                                (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, indexStart, startOffset, ubCount);
      } else {
        Simt::VF_CALL<SimtCompute1<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__local_mem__ T1*)(updatesUb.GetPhyAddr()), (__local_mem__ T2*)(indicesUb.GetPhyAddr()),
                                                (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, indexStart, startOffset, ubCount);
      }
    } else {
      if (tiling->axis == SECOND_LAST_DIM) {
        Simt::VF_CALL<SimtCompute2<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__local_mem__ T1*)(updatesUb.GetPhyAddr()), (__local_mem__ T2*)(indicesUb.GetPhyAddr()),
                                                (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, indexStart, startOffset, ubCount);
      } else {
        Simt::VF_CALL<SimtCompute3<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__local_mem__ T1*)(updatesUb.GetPhyAddr()), (__local_mem__ T2*)(indicesUb.GetPhyAddr()),
                                                (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, indexStart, startOffset, ubCount);
      }
    }
  }

  inline __aicore__ void ScatterComputeUint64(LocalTensor<T1> updatesUb, LocalTensor<T2> indicesUb, __local_mem__ T3* factorArr, __local_mem__ T3* coefArr, 
                                              __local_mem__ T3* shiftArr, __local_mem__ T3* mArr, int64_t indexStart, int64_t startOffset, int64_t ubCount) {
    if (tiling->indicesDim == 1) {
      if (tiling->axis == SECOND_LAST_DIM) {
        Simt::VF_CALL<SimtUint64Compute0<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__local_mem__ T1*)(updatesUb.GetPhyAddr()), (__local_mem__ T2*)(indicesUb.GetPhyAddr()),
                                                      (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, indexStart, startOffset, ubCount);
      } else {
        Simt::VF_CALL<SimtUint64Compute1<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__local_mem__ T1*)(updatesUb.GetPhyAddr()), (__local_mem__ T2*)(indicesUb.GetPhyAddr()),
                                                      (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, indexStart, startOffset, ubCount);
      }
    } else {
      if (tiling->axis == SECOND_LAST_DIM) {
        Simt::VF_CALL<SimtUint64Compute2<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__local_mem__ T1*)(updatesUb.GetPhyAddr()), (__local_mem__ T2*)(indicesUb.GetPhyAddr()),
                                                      (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, indexStart, startOffset, ubCount);
      } else {
        Simt::VF_CALL<SimtUint64Compute3<T1, T2, T3>>(Simt::Dim3(THREAD_NUM), (__local_mem__ T1*)(updatesUb.GetPhyAddr()), (__local_mem__ T2*)(indicesUb.GetPhyAddr()),
                                                      (__gm__ T1*)(outputGm.GetPhyAddr()), factorArr, coefArr, shiftArr, mArr, indexStart, startOffset, ubCount);
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
    auto blockIdx = GetBlockIdx();
    blockFactor = tiling->blockFactor;
    // 整块核个数
    int32_t blockCoreNum = tiling->aivCoreNum - tiling->tailCoreNum;
    int64_t totalUpdatesCount = tiling->updatesDim0 * tiling->updatesDim1 * tiling->updatesDim2 * tiling->updatesDim3;

    if (blockIdx >= blockCoreNum) {
      // 尾块核
      if (totalUpdatesCount > (tiling->aivCoreNum * MIN_FACTOR)) {
        blockFactor = blockFactor - 1;
      } else {
        blockFactor = (tiling->updatesDim0 * tiling->updatesDim1 * tiling->updatesDim2 * tiling->updatesDim3) % MIN_FACTOR;
      }
    }

    if (blockFactor > 0) {
      // Todo: change int32_t to int64_t
      U startOffset = blockIdx >= blockCoreNum ?
                            tiling->blockFactor * blockCoreNum + blockFactor * (blockIdx - blockCoreNum) :
                            tiling->blockFactor * blockIdx;
      U indexStart = startOffset / factor0;
      U indexEnd = (startOffset + blockFactor - 1) / factor0;
      U indicesSize = (indexEnd - indexStart + 1) * tiling->indicesDim * sizeof(T2);

      if (indicesSize <= tiling->ubSize / UB_DIVISION_COUNT) {
        indicesUbSize = (indicesSize + BLOCK_SIZE * INDICES_BUFFER_NUM - 1) / (BLOCK_SIZE * INDICES_BUFFER_NUM) *
                        (BLOCK_SIZE * INDICES_BUFFER_NUM);
        updatesUbSize = (tiling->ubSize - indicesUbSize) / (BLOCK_SIZE * UPDATES_BUFFER_NUM) *
                        (BLOCK_SIZE * UPDATES_BUFFER_NUM);
      } else {
        indicesUbSize = (tiling->ubSize / UB_DIVISION_COUNT + BLOCK_SIZE * INDICES_BUFFER_NUM - 1) /
                        (BLOCK_SIZE * INDICES_BUFFER_NUM) * (BLOCK_SIZE * INDICES_BUFFER_NUM);
        updatesUbSize = (tiling->ubSize - indicesUbSize) / (BLOCK_SIZE * UPDATES_BUFFER_NUM) *
                        (BLOCK_SIZE * UPDATES_BUFFER_NUM);
      }
      indicesBufferSize = indicesUbSize / INDICES_BUFFER_NUM;
      pipe->InitBuffer(indices, INDICES_BUFFER_NUM, indicesBufferSize);
      updatesBufferSize = updatesUbSize / UPDATES_BUFFER_NUM;
      pipe->InitBuffer(updates, UPDATES_BUFFER_NUM, updatesBufferSize);

      LocalTensor<T2> indicesUb = indices.template AllocTensor<T2>();
      if (indicesSize <= tiling->ubSize / UB_DIVISION_COUNT) {
        DataCopyExtParams copyParams{1, (uint32_t)indicesSize, 0, 0, 0};
        DataCopyPadExtParams<T2> padParams{false, 0, 0, 0};
        DataCopyPad(indicesUb, indicesGm[indexStart * tiling->indicesDim], copyParams, padParams);
        indices.EnQue(indicesUb);
        indices.DeQue<T2>();
      }

      U ubLoop = (blockFactor * sizeof(T1) + updatesBufferSize - 1) / updatesBufferSize;
      U ubTail = ((blockFactor * sizeof(T1)) % updatesBufferSize) / sizeof(T1);
      for (U loop = 0; loop < ubLoop; loop++) {
        if (indicesSize > tiling->ubSize / UB_DIVISION_COUNT) {
          indexStart = startOffset / factor0;
          indexEnd = (startOffset + blockFactor - 1) / factor0;
          indicesSize = (indexEnd - indexStart + 1) * tiling->indicesDim * sizeof(T2);

          DataCopyExtParams copyParams{1, (uint32_t)indicesSize, 0, 0, 0};
          DataCopyPadExtParams<T2> padParams{false, 0, 0, 0};
          DataCopyPad(indicesUb, indicesGm[indexStart * tiling->indicesDim], copyParams, padParams);
          indices.EnQue(indicesUb);
          indices.DeQue<T2>();
        }

        U ubCount = (loop == ubLoop - 1 ? ubTail : updatesBufferSize / sizeof(T1));
        LocalTensor<T1> updatesUb = updates.template AllocTensor<T1>();

        DataCopyExtParams copyParams{1, (uint32_t)(ubCount * sizeof(T1)), 0, 0, 0};
        DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
        DataCopyPad(updatesUb, updatesGm[startOffset], copyParams, padParams);
        updates.EnQue(updatesUb);
        updates.DeQue<T1>();

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
        if(sizeof(T3) == sizeof(uint32_t)) {
          ScatterComputeUint32(updatesUb, indicesUb, (__local_mem__ T3*)(factorArr.GetPhyAddr()), (__local_mem__ T3*)(coefArr.GetPhyAddr()), 
          (__local_mem__ T3*)(shiftArr.GetPhyAddr()), (__local_mem__ T3*)(mArr.GetPhyAddr()), indexStart, startOffset, ubCount);
        } else {
          ScatterComputeUint64(updatesUb, indicesUb, (__local_mem__ T3*)(factorArr.GetPhyAddr()), (__local_mem__ T3*)(coefArr.GetPhyAddr()), 
          (__local_mem__ T3*)(shiftArr.GetPhyAddr()), (__local_mem__ T3*)(mArr.GetPhyAddr()), indexStart, startOffset, ubCount);
        }
        startOffset = startOffset + updatesBufferSize / sizeof(T1);
        updates.FreeTensor<T1>(updatesUb);
        if (indicesSize > tiling->ubSize / UB_DIVISION_COUNT) {
          indices.FreeTensor<T2>(indicesUb);
        }
      }
      if (indicesSize <= tiling->ubSize / UB_DIVISION_COUNT) {
        indices.FreeTensor<T2>(indicesUb);
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

  constexpr static uint32_t SIMT_BUF_SIZE = 1024;

  // Todo: change int32_t to int64_t
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