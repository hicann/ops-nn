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
 * \file scatter_add_simd_support_atomicadd.h
 * \brief simd kernel of scatter_add
 */

#ifndef SCATTER_ADD_SIMD_SUPPORT_ATOMICADD_H
#define SCATTER_ADD_SIMD_SUPPORT_ATOMICADD_H

#include "scatter_add_common.h"

namespace ScatterAdd {
using namespace AscendC;
using namespace ScatterAddCommon;

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
class ScatterAddSIMDSupportAtomicAdd {
public:
    __aicore__ inline ScatterAddSIMDSupportAtomicAdd(const ScatterAddTilingData& tilingData, TPipe& pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace);
    __aicore__ inline void ProcessAtomicAdd();
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> varRefGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> updatesQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indicesQueue_;
    TPipe& pipe_;
    const ScatterAddTilingData& tilingData_;
};

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDSupportAtomicAdd<T, U, updatesIsScalar, scatterOp>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
    GM_ADDR varRef, GM_ADDR workspace)
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }

    pipe_.InitBuffer(updatesQueue_, DOUBLE_BUFFER, tilingData_.ubFactorRow * tilingData_.ubFactorCol * sizeof(T));
    pipe_.InitBuffer(indicesQueue_, DOUBLE_BUFFER, tilingData_.ubFactorRow * sizeof(U));
    varRefGm_.SetGlobalBuffer((__gm__ T *)(varRef));

    if (GetBlockIdx() < tilingData_.atomicAddCoreNum) {
        indicesGm_.SetGlobalBuffer((__gm__ U *)(indices));
        updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
    }
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDSupportAtomicAdd<T, U, updatesIsScalar, scatterOp>::ProcessAtomicAdd()
{
    if constexpr (updatesIsScalar) {
        LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();
        BroadcastUpdatesScalar<T, scatterOp>(updatesLocal, updatesGm_, static_cast<int32_t>(tilingData_.ubFactorCol));
        updatesQueue_.EnQue(updatesLocal);
    }

    uint64_t rowIdx = GetBlockIdx() / tilingData_.colTileNum;   // 当前block在第几个分块行
    uint64_t colIdx = GetBlockIdx() % tilingData_.colTileNum;   // 当前block在第几个分块列
    uint64_t curCoreRows = rowIdx != (tilingData_.rowTileNum - 1) ? tilingData_.normBlockRow : tilingData_.tailBlockRow;  // 当前分块行数
    uint64_t curCoreCols = colIdx != (tilingData_.colTileNum - 1) ? tilingData_.normBlockCol : tilingData_.tailBlockCol;  // 当前分块列数

    uint64_t blockOffsetUpdate = rowIdx * tilingData_.normBlockRow * tilingData_.varShape[1] + colIdx * tilingData_.normBlockCol;  // 当前updates块的偏移位置
    uint64_t blockOffsetindices = rowIdx * tilingData_.normBlockRow;   // 当前indices的偏移位置
    uint64_t colUbLoopNum = ops::CeilDiv(curCoreCols, tilingData_.ubFactorCol);   // 当前分块在列方向需要UB循环多少次
    uint64_t rowUbLoopNum = ops::CeilDiv(curCoreRows, tilingData_.ubFactorRow);   // 当前分块在行方向需要UB循环多少次

    for (uint64_t rowLoop = 0; rowLoop < rowUbLoopNum; rowLoop++) {
        uint64_t rows = (rowLoop == rowUbLoopNum - 1) ? (curCoreRows - rowLoop * tilingData_.ubFactorRow) : tilingData_.ubFactorRow;

        LocalTensor<U> indicesLocal = indicesQueue_.AllocTensor<U>();
        CopyIn(indicesLocal, indicesGm_, blockOffsetindices + rowLoop * tilingData_.ubFactorRow, 1, rows);
        indicesQueue_.EnQue<U>(indicesLocal);

        indicesLocal = indicesQueue_.DeQue<U>();
        for (uint64_t colLoop = 0; colLoop < colUbLoopNum; colLoop++) {
            uint64_t cols = (colLoop == colUbLoopNum - 1) ? (curCoreCols - colLoop * tilingData_.ubFactorCol) : tilingData_.ubFactorCol;  // 当前UB拿到的updates小块的列数
            uint64_t colsAlign = ops::Aligned(static_cast<uint64_t>(cols * sizeof(T)), static_cast<uint64_t>(ONE_BLOCK_SIZE)) / sizeof(T);

            if constexpr (updatesIsScalar) {
                LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();
                SetAtomicAdd<T>();
                for (uint64_t i = 0; i < rows; i++) {
                    uint64_t dstIdx = indicesLocal.GetValue(i);      // 找到当次循环对应的indices内的值，即var的索引
                    if (dstIdx < 0 || dstIdx >= tilingData_.varShape[0]) {
                        continue;
                    }
                    uint64_t dstOffset = dstIdx * tilingData_.varShape[1] + colIdx * tilingData_.normBlockCol + colLoop * tilingData_.ubFactorCol;
                    CopyOut(varRefGm_, updatesLocal, dstOffset, 1, cols);
                }
                SetAtomicNone();
                updatesQueue_.EnQue<T>(updatesLocal);
            } else {
                LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();
                uint64_t offset = blockOffsetUpdate + rowLoop * tilingData_.ubFactorRow * tilingData_.varShape[1] + colLoop * tilingData_.ubFactorCol;
                auto MTE2WaitMTE3EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(MTE2WaitMTE3EventID);
                WaitFlag<HardEvent::MTE3_MTE2>(MTE2WaitMTE3EventID);
                CopyIn(updatesLocal, updatesGm_, offset, rows, cols, tilingData_.varShape[1] - cols);
                updatesQueue_.EnQue<T>(updatesLocal);
    
                updatesLocal = updatesQueue_.DeQue<T>();
                event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);

                if constexpr (scatterOp == SUB) {
                    auto VWaitMTE2EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(VWaitMTE2EventID);
                    WaitFlag<HardEvent::MTE2_V>(VWaitMTE2EventID);
 	  	            NegateUpdate<T>(updatesLocal, static_cast<uint32_t>(rows * colsAlign));
                    auto MTE3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(MTE3WaitVEventID);
                    WaitFlag<HardEvent::V_MTE3>(MTE3WaitVEventID);
 	  	        }

                SetAtomicAdd<T>();
                for (uint64_t i = 0; i < rows; i++) {
                    uint64_t dstIdx = indicesLocal.GetValue(i);      // 找到当次循环对应的indices内的值，即var的索引
                    if (dstIdx < 0 || dstIdx >= tilingData_.varShape[0]) {
                        continue;
                    }
                    uint64_t dstOffset = dstIdx * tilingData_.varShape[1] + colIdx * tilingData_.normBlockCol + colLoop * tilingData_.ubFactorCol;
                    CopyOut(varRefGm_, updatesLocal[i * colsAlign], dstOffset, 1, cols);
                }
                SetAtomicNone();
                updatesQueue_.FreeTensor(updatesLocal);
            }
        }
        indicesQueue_.FreeTensor(indicesLocal);
    }

    if constexpr (updatesIsScalar) {
        LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();
        updatesQueue_.FreeTensor<T>(updatesLocal);
    }
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDSupportAtomicAdd<T, U, updatesIsScalar, scatterOp>::Process()
{
    if (GetBlockIdx() >= GetBlockNum() || GetBlockIdx() >= tilingData_.atomicAddCoreNum) {
        return;
    }

    ProcessAtomicAdd();
}
}
#endif  // SCATTER_ADD_SIMD_SUPPORT_ATOMICADD_H