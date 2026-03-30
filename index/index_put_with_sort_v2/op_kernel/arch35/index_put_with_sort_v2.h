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

#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace AscendC
{
static constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t TILING_NUM = 36;
constexpr size_t NON_INDEXED_DIM_SIZE_OFFSET = 2;
constexpr size_t NON_IDXED_STRIDE_OFFSET = 3;
constexpr size_t NON_IDXED_SELF_STRIDE_OFFSET = 11;
constexpr size_t NON_IDXED_VALUE_STRIDE_OFFSET = 19;
constexpr size_t IDXED_VALUE_STRIDE = 27;
constexpr int32_t THREAD_NUM_FULL = 512;
constexpr int32_t THREAD_NUM_HALF = 512;

template <typename T>
struct calcParams {
    T shift_[MAX_DIM_NUM] = {0};
    T m_[MAX_DIM_NUM] = {0};
};

template <typename T>
using AccumType = std::conditional_t<
    std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>, int32_t,
    std::conditional_t<std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>, float, T>>;

template <typename TX, typename AccumT, bool ACCUMULATE, typename IdxT>
__aicore__ inline void WriteBack(__gm__ TX* output, IdxT idx, AccumT value) {
    if constexpr (ACCUMULATE) {
        output[idx] = static_cast<TX>(value + static_cast<AccumT>(output[idx]));
    } else {
        output[idx] = static_cast<TX>(value);
    }
}

template <typename TX, typename TIDX, bool ACCUMULATE, bool ALL_INDEXED, bool INDEXED_BLOCK_MODE, uint16_t LAUNCH_BOUND_LIMIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(LAUNCH_BOUND_LIMIT) inline void SimtIndexPutV2(__gm__ TX* output, __gm__ TIDX* sortIndices,
    __gm__ int32_t* posIdx, __gm__ TX* values, __ubuf__ int64_t* tilingUb, __ubuf__ calcParams<uint64_t>* calcParamsPtr) {
    const auto& nonIndexedDimNum = tilingUb[0];
    const auto& indexedDimSize = tilingUb[1];
    const auto& nonIndexedDimSize = tilingUb[NON_INDEXED_DIM_SIZE_OFFSET];
    __ubuf__ int64_t* nonIdxedStride = tilingUb + NON_IDXED_STRIDE_OFFSET;
    __ubuf__ int64_t* nonIdxedSelfStride = tilingUb + NON_IDXED_SELF_STRIDE_OFFSET;
    __ubuf__ int64_t* nonIdxedValueStride = tilingUb + NON_IDXED_VALUE_STRIDE_OFFSET;
    const auto& idxedValueStride = tilingUb[IDXED_VALUE_STRIDE];
    // 动态计算外层循环起始位置和步长
    uint64_t outerStart, outerStep;
    if constexpr (INDEXED_BLOCK_MODE) {
        outerStart = Simt::GetBlockIdx() * Simt::GetThreadNum<0>() + Simt::GetThreadIdx<0>();
        outerStep = Simt::GetBlockNum() * Simt::GetThreadNum<0>();
    } else {
        outerStart = Simt::GetThreadIdx<0>();
        outerStep = Simt::GetThreadNum<0>();
    }
    #pragma unroll
    for (uint64_t idxedIdx = outerStart; idxedIdx < indexedDimSize; idxedIdx += outerStep) {
        uint64_t curIdxedIdx = idxedIdx;
        uint64_t idxedSelfIdx = sortIndices[curIdxedIdx];
        if (curIdxedIdx != 0) {
            uint64_t idxedSelfIdxPre = sortIndices[curIdxedIdx - 1];
            if (idxedSelfIdxPre == idxedSelfIdx) {
                continue;
            }
        }
        using AccumT = AccumType<TX>;
        AccumT res = 0;
        if constexpr (ALL_INDEXED) {
            do {
                uint64_t idxedValueIdx = posIdx[curIdxedIdx] * idxedValueStride;
                res += static_cast<AccumT>(values[idxedValueIdx]);
            } while (ACCUMULATE && ++curIdxedIdx < indexedDimSize && sortIndices[curIdxedIdx] == idxedSelfIdx);
            WriteBack<TX, AccumT, ACCUMULATE>(output, idxedSelfIdx, res);
        } else {
            do {
                uint64_t idxedValueIdx = posIdx[curIdxedIdx] * idxedValueStride;
                uint64_t innerStart, innerStep;
                if constexpr (INDEXED_BLOCK_MODE) {
                    innerStart = Simt::GetThreadIdx<1>();
                    innerStep = Simt::GetThreadNum<1>();
                } else {
                    innerStart = Simt::GetBlockIdx() * Simt::GetThreadNum<1>() + Simt::GetThreadIdx<1>();
                    innerStep = Simt::GetBlockNum() * Simt::GetThreadNum<1>();
                }
                for (uint64_t k = innerStart; k < nonIndexedDimSize; k += innerStep) {
                    uint64_t nonIdxedSelfIdx = 0;
                    uint64_t nonIdxedValueIdx = 0;
                    uint64_t remaining = k;

                    // 将一维索引k分解为多维索引
                    #pragma unroll
                    for (uint64_t i = 0; i < nonIndexedDimNum; i++) {
                        const uint64_t idxI = AscendC::Simt::UintDiv(remaining, calcParamsPtr->m_[i], calcParamsPtr->shift_[i]);
                        remaining = remaining - idxI * nonIdxedStride[i]; // 剩余索引
                        nonIdxedSelfIdx += idxI * nonIdxedSelfStride[i]; // 累加内存偏移
                        nonIdxedValueIdx += idxI * nonIdxedValueStride[i];
                    }
                    res = static_cast<AccumT>(values[idxedValueIdx + nonIdxedValueIdx]);
                    WriteBack<TX, AccumT, ACCUMULATE>(output, idxedSelfIdx + nonIdxedSelfIdx, res);
                }
            } while (ACCUMULATE && ++curIdxedIdx < indexedDimSize && sortIndices[curIdxedIdx] == idxedSelfIdx);
        }
    }
}

template <typename TX, typename TIDX, bool ACCUMULATE, bool ALL_INDEXED, bool INDEXED_BLOCK_MODE>
class IndexPutWithSortV2Kernel {
public:
    __aicore__ inline IndexPutWithSortV2Kernel(TPipe *pipe, const IndexPutWithSortV2TilingData *tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR values,
        GM_ADDR output) {
        outputGm_.SetGlobalBuffer((__gm__ TX *)output);
        indicesGm_.SetGlobalBuffer((__gm__ TIDX *)sortIndices);
        posIdxGm_.SetGlobalBuffer((__gm__ int32_t *)posIdx);
        valuesGm_.SetGlobalBuffer((__gm__ TX *)values);

        Ppipe_->InitBuffer(tilingBuf_, TILING_NUM * sizeof(int64_t));
        Ppipe_->InitBuffer(Buf_, sizeof(calcParams<uint64_t>));
    }

    __aicore__ inline void Process() {
        __gm__ TX *output = (__gm__ TX *)outputGm_.GetPhyAddr();
        __gm__ TX *values = (__gm__ TX *)valuesGm_.GetPhyAddr();
        __gm__ TIDX *sortIndices = (__gm__ TIDX *)indicesGm_.GetPhyAddr();
        __gm__ int32_t *posIdx = (__gm__ int32_t *)posIdxGm_.GetPhyAddr();

        LocalTensor<int64_t> tilingUb = tilingBuf_.Get<int64_t>();
        const int64_t* tilingP = reinterpret_cast<const int64_t*>(tiling_);
        #pragma unroll
        for (size_t i = 0; i < TILING_NUM; i++) {
            tilingUb.SetValue(i, tilingP[i]);
        }
        __ubuf__ int64_t *tilingUbAddr = (__ubuf__ int64_t *)tilingUb.GetPhyAddr();
        __ubuf__ int64_t* nonIdxedStride = tilingUbAddr + NON_IDXED_STRIDE_OFFSET;

        LocalTensor<uint64_t> calcParamsUb = Buf_.Get<uint64_t>();
        __ubuf__ calcParams<uint64_t>* calcParamsPtr = (__ubuf__ calcParams<uint64_t>*)calcParamsUb.GetPhyAddr();
        int64_t nonIndexedDimNum = tilingUbAddr[0];
        #pragma unroll
        for (int64_t i = 0; i < nonIndexedDimNum; i++) {
            uint64_t m{0}, shift{0};
            uint64_t divisor = static_cast<uint64_t>(nonIdxedStride[i]);
            AscendC::GetUintDivMagicAndShift(m, shift, divisor);
            calcParamsPtr->m_[i] = m, calcParamsPtr->shift_[i] = shift;
        }
        constexpr uint16_t THREAD_NUM_LIMIT = ALL_INDEXED ? THREAD_NUM_FULL : THREAD_NUM_HALF;
        Simt::VF_CALL<SimtIndexPutV2<TX, TIDX, ACCUMULATE, ALL_INDEXED, INDEXED_BLOCK_MODE, THREAD_NUM_LIMIT>>(
                        Simt::Dim3{tiling_->indexedThreadNum, tiling_->nonIndexedThreadNum, 1}, output, sortIndices, posIdx,
                        values, tilingUbAddr, calcParamsPtr);
    }

private:
    TPipe *Ppipe_;
    const IndexPutWithSortV2TilingData *tiling_;
    GlobalTensor<TX> outputGm_;
    GlobalTensor<TIDX> indicesGm_;
    GlobalTensor<int32_t> posIdxGm_;
    GlobalTensor<TX> valuesGm_;

    TBuf<TPosition::VECCALC> tilingBuf_;
    TBuf<TPosition::VECCALC> Buf_;
};
}  // namespace AscendC
#endif // INDEX_PUT_WITH_SORT_V2_H
