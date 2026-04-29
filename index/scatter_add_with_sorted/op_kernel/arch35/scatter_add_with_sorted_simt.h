/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License")
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SCATTER_ADD_WITH_SORTED_SIMT_H
#define SCATTER_ADD_WITH_SORTED_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "scatter_add_with_sorted_struct.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace ScatterAddWithSorted {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 256;
#else
constexpr uint32_t THREAD_NUM = 1024;
#endif
constexpr uint32_t THREAD_MIN_INDICES_NUM = 10;
constexpr uint32_t TEN = 10;

template <typename T, typename U, typename ADDR_T, bool isUpdateScalar, bool withPos>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ScatterAddWithSortedSimtCompute(
    ADDR_T totalCol, ADDR_T indicesBlockOffset, ADDR_T indicesBlockNum, ADDR_T varFirstDimSize, ADDR_T magic,
    ADDR_T shift, T updateScalarValue, __gm__ T* var, __gm__ U* sortedIndices, __gm__ U* pos, __gm__ T* updates)
{
    ADDR_T startIndices = indicesBlockOffset;
    ADDR_T endIndices = startIndices + indicesBlockNum;
    ADDR_T updatesBlockOffset = startIndices * totalCol;
    ADDR_T updatesBlockNum = indicesBlockNum * totalCol;
    ADDR_T updatesBlockEnd = updatesBlockOffset + updatesBlockNum;
    for (ADDR_T i = updatesBlockOffset + threadIdx.x; i < updatesBlockEnd; i += blockDim.x) {
        ADDR_T indiceRow = Simt::UintDiv(i, magic, shift);
        U varRow = sortedIndices[indiceRow];
        if (varRow < 0 || varRow >= varFirstDimSize) {
            continue;
        }
        if (indiceRow != startIndices) {
            if (varRow == sortedIndices[indiceRow - 1]) { // 当前线程对应索引与前一个索引相同，由前一个线程处理
                continue;
            }
        }

        ADDR_T colIdx = i - indiceRow * totalCol;
        T sumRes = 0;
        for (ADDR_T j = indiceRow; j < endIndices; j++) {
            if (sortedIndices[j] != varRow) {
                break;
            }
            if constexpr (isUpdateScalar) {
                sumRes += updateScalarValue;
            } else {
                if constexpr (withPos) {
                    sumRes += updates[pos[j] * totalCol + colIdx];
                } else {
                    sumRes += updates[j * totalCol + colIdx];
                }
            }
        }

        ADDR_T varGmOffset = varRow * totalCol + colIdx;
        asc_atomic_add(var + varGmOffset, sumRes);
    }
}

template <typename T, typename U, typename ADDR_T, bool isUpdateScalar, bool withPos>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ScatterAddWithSortedSimtComputeCutIndices(
    ADDR_T threadIndicesNum, ADDR_T tailThreadRowIndicesNum, ADDR_T indicesBlockOffset, ADDR_T varFirstDimSize,
    T updateScalarValue, __gm__ T* var, __gm__ U* sortedIndices, __gm__ U* pos, __gm__ T* updates)
{
    uint32_t threadIdxRow = threadIdx.y;
    uint32_t threadNumRow = blockDim.y;
    uint32_t colIdx = threadIdx.x;
    uint32_t totalCol = blockDim.x;

    ADDR_T startIndices = indicesBlockOffset + threadIdxRow * threadIndicesNum;
    ADDR_T endIndices = startIndices + threadIndicesNum;
    if (threadIdxRow == threadNumRow - 1) {
        endIndices = startIndices + tailThreadRowIndicesNum;
    }

    T sumRes = 0;
    U preIndices = sortedIndices[startIndices];
    U varRow = preIndices;
    for (ADDR_T i = startIndices; i < endIndices; i += 1) {
        varRow = sortedIndices[i];

        if (preIndices != varRow && preIndices >= 0 && preIndices < varFirstDimSize) {
            ADDR_T varGmOffsetLast = preIndices * totalCol + colIdx;
            asc_atomic_add(var + varGmOffsetLast, sumRes);
            sumRes = 0;
            preIndices = varRow;
        }
        if constexpr (isUpdateScalar) {
            sumRes += updateScalarValue;
        } else {
            if constexpr (withPos) {
                sumRes += updates[pos[i] * totalCol + colIdx];
            } else {
                sumRes += updates[i * totalCol + colIdx];
            }
        }
    }
    if (varRow >= 0 && varRow < varFirstDimSize) {
        ADDR_T varGmOffset = varRow * totalCol + colIdx;
        asc_atomic_add(var + varGmOffset, sumRes);
    }
}

template<typename T, typename U, typename ADDR_T, bool isUpdateScalar, bool withPos>
class ScatterAddWithSortedSIMT {
public:
    __aicore__ inline ScatterAddWithSortedSIMT(const ScatterAddWithSortedSimtTilingData& tilingData)
        : tilingData_(tilingData){};

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR updates, GM_ADDR indices, GM_ADDR pos);
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<U> posGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    const ScatterAddWithSortedSimtTilingData& tilingData_;

    uint32_t blockIdx_ = 0;
    ADDR_T indicesBlockNum_ = 0;
};

template<typename T, typename U, typename ADDR_T, bool isUpdateScalar, bool withPos>
__aicore__ inline void ScatterAddWithSortedSIMT<T, U, ADDR_T, isUpdateScalar, withPos>::Init(
                        GM_ADDR var, GM_ADDR updates, GM_ADDR indices, GM_ADDR pos)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    
    varGm_.SetGlobalBuffer((__gm__ T *)(var));
    updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
    indicesGm_.SetGlobalBuffer((__gm__ U*)indices);
    posGm_.SetGlobalBuffer((__gm__ U*)pos);

    indicesBlockNum_ = tilingData_.normBlockIndices;
    if (blockIdx_ == tilingData_.usedCoreNum - 1) {
        indicesBlockNum_ = tilingData_.tailBlockIndices;
    }
}

template<typename T, typename U, typename ADDR_T, bool isUpdateScalar, bool withPos>
__aicore__ inline void ScatterAddWithSortedSIMT<T, U, ADDR_T, isUpdateScalar, withPos>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    ADDR_T totalCol = static_cast<uint32_t>(tilingData_.varShape[1]);
    ADDR_T varFirstDimSize = static_cast<ADDR_T>(tilingData_.varShape[0]);

    uint32_t blockIdx = blockIdx_;
    T updateScalarValue = ((__gm__ T*)(updatesGm_.GetPhyAddr()))[0];
    ADDR_T indicesBlockNum = indicesBlockNum_;
    ADDR_T indicesBlockOffset = blockIdx * tilingData_.normBlockIndices;
    ADDR_T updatesBlockNum = indicesBlockNum_ * totalCol;

    bool isNoCutIndices = tilingData_.indicesNum <= tilingData_.varShape[0] * TWO;
    isNoCutIndices = isNoCutIndices || (tilingData_.indicesNum <= tilingData_.varShape[0] * TEN && tilingData_.varShape[1] <= TWO);
    if (isNoCutIndices) {
        ADDR_T magic = 0;
        ADDR_T shift = 0;
        GetUintDivMagicAndShift(magic, shift, totalCol);
        asc_vf_call<ScatterAddWithSortedSimtCompute<T, U, ADDR_T, isUpdateScalar, withPos>>(dim3(THREAD_NUM), 
                totalCol, indicesBlockOffset, indicesBlockNum, varFirstDimSize, magic, shift, updateScalarValue,
                (__gm__ T*)(varGm_.GetPhyAddr()), (__gm__ U*)(indicesGm_.GetPhyAddr()), (__gm__ U*)(posGm_.GetPhyAddr()),
                (__gm__ T*)(updatesGm_.GetPhyAddr()));
    } else {
        uint32_t currentMaxThread = updatesBlockNum >= THREAD_NUM ? THREAD_NUM : updatesBlockNum;
        uint32_t threadNumRow = currentMaxThread / totalCol;
        threadNumRow = AscendC::Std::min(static_cast<ADDR_T>(threadNumRow),
                                         ops::CeilDiv(indicesBlockNum, static_cast<ADDR_T>(THREAD_MIN_INDICES_NUM)));
        ADDR_T threadIndicesNum = ops::CeilDiv(indicesBlockNum, static_cast<ADDR_T>(threadNumRow));
        threadNumRow = ops::CeilDiv(indicesBlockNum, threadIndicesNum);
        ADDR_T tailThreadRowIndicesNum = indicesBlockNum - threadIndicesNum * (threadNumRow - 1);
    
        asc_vf_call<ScatterAddWithSortedSimtComputeCutIndices<T, U, ADDR_T, isUpdateScalar, withPos>>(
                dim3({static_cast<uint32_t>(totalCol), threadNumRow}), threadIndicesNum, tailThreadRowIndicesNum,
                indicesBlockOffset, varFirstDimSize, updateScalarValue, (__gm__ T*)(varGm_.GetPhyAddr()),
                (__gm__ U*)(indicesGm_.GetPhyAddr()), (__gm__ U*)(posGm_.GetPhyAddr()), (__gm__ T*)(updatesGm_.GetPhyAddr()));
    }
}

}
#endif