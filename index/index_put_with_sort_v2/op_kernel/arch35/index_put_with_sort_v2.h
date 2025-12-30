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
 * \file index_put_with_sort_v2.h
 * \brief
 */

#ifndef INDEX_PUT_WITH_SORT_V2_H
#define INDEX_PUT_WITH_SORT_V2_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AscendC
{
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t TILING_NUM = 36;
constexpr size_t NON_INDEXED_DIM_SIZE_OFFSET = 2;
constexpr size_t NON_IDXED_STRIDE_OFFSET = 3;
constexpr size_t NON_IDXED_SELF_STRIDE_OFFSET = 11;
constexpr size_t NON_IDXED_VALUE_STRIDE_OFFSET = 19;
constexpr size_t IDXED_VALUE_STRIDE = 27;
constexpr int32_t THREAD_NUM_FULL = 1024;
constexpr int32_t THREAD_NUM_HALF = 512;

template<typename TX, typename TIDX, bool ACCUMULATE, bool ALL_INDEXED, bool INDEXED_BLOCK_MODE>
__aicore__ __attribute__((always_inline)) inline void SimtIndexPutV2(__gm__ TX* output, __gm__ TIDX* sortIndices,
    __gm__ int32_t* posIdx, __gm__ TX* values, __ubuf__ int64_t* tilingUb) {
    const auto& nonIndexedDimNum = tilingUb[0];
    const auto& indexedDimSize = tilingUb[1];
    const auto& nonIndexedDimSize = tilingUb[NON_INDEXED_DIM_SIZE_OFFSET];
    __ubuf__ int64_t* nonIdxedStride = tilingUb + NON_IDXED_STRIDE_OFFSET;
    __ubuf__ int64_t* nonIdxedSelfStride = tilingUb + NON_IDXED_SELF_STRIDE_OFFSET;
    __ubuf__ int64_t* nonIdxedValueStride = tilingUb + NON_IDXED_VALUE_STRIDE_OFFSET;
    const auto& idxedValueStride = tilingUb[IDXED_VALUE_STRIDE];
    // 动态计算外层循环起始位置和步长
    int64_t outerStart, outerStep;
    if constexpr (INDEXED_BLOCK_MODE) {
        outerStart = Simt::GetBlockIdx() * Simt::GetThreadNum<0>() + Simt::GetThreadIdx<0>();
        outerStep = Simt::GetBlockNum() * Simt::GetThreadNum<0>();
    } else {
        outerStart = Simt::GetThreadIdx<0>();
        outerStep = Simt::GetThreadNum<0>();
    }
for (int64_t idxedIdx = outerStart; idxedIdx < indexedDimSize; idxedIdx += outerStep) {
        int64_t curIdxedIdx = idxedIdx;
        int64_t idxedSelfIdx = sortIndices[curIdxedIdx];
        if (curIdxedIdx != 0) {
            int64_t idxedSelfIdxPre = sortIndices[curIdxedIdx - 1];
            if (idxedSelfIdxPre == idxedSelfIdx) {
                continue;
            }
        }

        do {  // 注意accumulate的区别，False为只执行第一次
            int64_t idxedValueIdx = posIdx[curIdxedIdx] * idxedValueStride;

            if constexpr (ALL_INDEXED) {
                if constexpr (ACCUMULATE) {
                    output[idxedSelfIdx] += values[idxedValueIdx];
                } else {
                    output[idxedSelfIdx] = values[idxedValueIdx];
                }
            } else {
                // 动态计算内层循环起始位置和步长
                int64_t innerStart, innerStep;
                if constexpr (INDEXED_BLOCK_MODE) {
                    innerStart = Simt::GetThreadIdx<1>();
                    innerStep = Simt::GetThreadNum<1>();
                } else {
                    innerStart = Simt::GetBlockIdx() * Simt::GetThreadNum<1>() + Simt::GetThreadIdx<1>();
                    innerStep = Simt::GetBlockNum() * Simt::GetThreadNum<1>();
                }
                for (int64_t k = innerStart; k < nonIndexedDimSize; k += innerStep) {
                    int64_t nonIdxedSelfIdx = 0;
                    int64_t nonIdxedValueIdx = 0;
                    int64_t remaining = k;

                    // 将一维索引k分解为多维索引
                    for (int64_t i = 0; i < nonIndexedDimNum; i++) {
                        const int64_t idxI = remaining / nonIdxedStride[i]; // 当前维度索引
                        remaining %= nonIdxedStride[i]; // 剩余索引
                        nonIdxedSelfIdx += idxI * nonIdxedSelfStride[i]; // 累加内存偏移
                        nonIdxedValueIdx += idxI * nonIdxedValueStride[i];
                    }
                    if constexpr (ACCUMULATE) {
                        output[idxedSelfIdx + nonIdxedSelfIdx] += values[idxedValueIdx + nonIdxedValueIdx];
                    } else {
                        output[idxedSelfIdx + nonIdxedSelfIdx] = values[idxedValueIdx + nonIdxedValueIdx];
                    }
                }
            }

            if constexpr (ACCUMULATE == false) {
                break;
            }
            curIdxedIdx++;
            if (curIdxedIdx == indexedDimSize) {
                break;
            }
        } while (sortIndices[curIdxedIdx] == idxedSelfIdx);
    }
}

template <typename TX, typename TIDX, bool ACCUMULATE, bool ALL_INDEXED, bool INDEXED_BLOCK_MODE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_FULL) inline void SimtIndexPutV2Normal(
    __gm__ TX* output, __gm__ TIDX* sortIndices, __gm__ int32_t* posIdx, __gm__ TX* values, __ubuf__ int64_t* tilingUb)
{
    SimtIndexPutV2<TX, TIDX, ACCUMULATE, ALL_INDEXED, INDEXED_BLOCK_MODE>(
        output, sortIndices, posIdx, values, tilingUb);
}

template <typename TX, typename TIDX, bool ACCUMULATE, bool ALL_INDEXED, bool INDEXED_BLOCK_MODE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_HALF) inline void SimtIndexPutV2Half(
    __gm__ TX* output, __gm__ TIDX* sortIndices, __gm__ int32_t* posIdx, __gm__ TX* values, __ubuf__ int64_t* tilingUb)
{
    SimtIndexPutV2<TX, TIDX, ACCUMULATE, ALL_INDEXED, INDEXED_BLOCK_MODE>(
        output, sortIndices, posIdx, values, tilingUb);
}

template<typename TX, typename TIDX, bool ACCUMULATE, bool ALL_INDEXED, bool INDEXED_BLOCK_MODE>
class IndexPutWithSortV2Kernel {
public:
    __aicore__ inline IndexPutWithSortV2Kernel(TPipe *pipe, const IndexPutWithSortV2TilingData *tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR values,
        GM_ADDR output)
    {
        outputGm_.SetGlobalBuffer((__gm__ TX *)output);
        indicesGm_.SetGlobalBuffer((__gm__ TIDX *)sortIndices);
        posIdxGm_.SetGlobalBuffer((__gm__ int32_t *)posIdx);
        valuesGm_.SetGlobalBuffer((__gm__ TX *)values);

        Ppipe_->InitBuffer(tilingBuf_, TILING_NUM * sizeof(int64_t));
    }

    __aicore__ inline void Process()
    {
        __gm__ TX *output = (__gm__ TX *)outputGm_.GetPhyAddr();
        __gm__ TX *values = (__gm__ TX *)valuesGm_.GetPhyAddr();
        __gm__ TIDX *sortIndices = (__gm__ TIDX *)indicesGm_.GetPhyAddr();
        __gm__ int32_t *posIdx = (__gm__ int32_t *)posIdxGm_.GetPhyAddr();

        LocalTensor<int64_t> tilingUb = tilingBuf_.Get<int64_t>();
        const int64_t* tilingP = reinterpret_cast<const int64_t*>(tiling_);
        for (size_t i = 0; i < TILING_NUM; i++) {
            tilingUb.SetValue(i, tilingP[i]);
        }
        __ubuf__ int64_t *tilingUbAddr = (__ubuf__ int64_t *)tilingUb.GetPhyAddr();
        if constexpr(ALL_INDEXED) {
            Simt::VF_CALL<SimtIndexPutV2Normal<TX, TIDX, ACCUMULATE, ALL_INDEXED, INDEXED_BLOCK_MODE>>(
                Simt::Dim3{tiling_->indexedThreadNum, tiling_->nonIndexedThreadNum, 1}, output, sortIndices, posIdx,
                values, tilingUbAddr);
        } else {
            Simt::VF_CALL<SimtIndexPutV2Half<TX, TIDX, ACCUMULATE, ALL_INDEXED, INDEXED_BLOCK_MODE>>(
                Simt::Dim3{tiling_->indexedThreadNum, tiling_->nonIndexedThreadNum, 1}, output, sortIndices, posIdx,
                values, tilingUbAddr);
        }
    }

private:
    TPipe *Ppipe_;
    const IndexPutWithSortV2TilingData *tiling_;
    GlobalTensor<TX> outputGm_;
    GlobalTensor<TIDX> indicesGm_;
    GlobalTensor<int32_t> posIdxGm_;
    GlobalTensor<TX> valuesGm_;

    TBuf<TPosition::VECCALC> tilingBuf_;
};
}  // namespace AscendC

#endif // INDEX_PUT_WITH_SORT_V2_H