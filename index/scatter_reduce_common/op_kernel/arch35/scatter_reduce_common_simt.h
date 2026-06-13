/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_reduce_common_simt.h
 * \brief SIMT atomic scatter-reduce kernel for the non-nd ScatterMax/Min ops. Independent of scatter_add.
 *        For each index entry m: var[indices[m]] (slice of sliceSize elems) = reduce(var[...], updates[m]).
 */
#ifndef SCATTER_REDUCE_COMMON_SIMT_H
#define SCATTER_REDUCE_COMMON_SIMT_H

#include "scatter_reduce_common_struct.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace ScatterReduceCommon {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;  // threads per AIV core (one VF block); matches other SIMT ops

// Reduce two values in their accumulator (compute) type.
template <typename ACC, uint8_t Mode>
__simt_callee__ inline ACC ReduceAcc(ACC a, ACC b)
{
    if constexpr (Mode == MODE_MAX) {
        return a > b ? a : b;
    } else if constexpr (Mode == MODE_MIN) {
        return a < b ? a : b;
    } else if constexpr (Mode == MODE_MUL) {
        return a * b;
    } else {
        return a / b;  // MODE_DIV
    }
}

// 32-bit element (float / int32): direct CAS on __gm__ T* (asc_atomic_cas has float*/int32_t*
// overloads). The retry loop converges regardless of initial read freshness; mul/max/min over a
// row are order-independent, div matches TF's order-undefined semantics for duplicate indices.
template <typename T, uint8_t Mode>
__simt_callee__ inline void AtomicReduce32(__gm__ T* addr, T val)
{
    T cur = *addr;
    T prev;
    do {
        prev = cur;
        cur = asc_atomic_cas(addr, prev, static_cast<T>(ReduceAcc<T, Mode>(prev, val)));
    } while (cur != prev);
}

// Sub-32-bit element (fp16 / int8 / uint8): asc_atomic_cas has no overload for these, so CAS on the
// enclosing 32-bit word. The element is promoted to float (fp16) or int32 (int8/uint8) for the
// reduce and cast back — so A5 covers the same dtypes as the A2/A3 (TBE) implementation. The CAS
// preserves the other sub-elements of the word via keepMask. base must be 4-byte aligned (GM
// tensors are >=32-byte aligned, so the enclosing word is always within the allocation).
template <typename T, uint8_t Mode>
__simt_callee__ inline void AtomicReduceSubWord(__gm__ uint32_t* base, uint64_t elemIdx, T val)
{
    using ACC = typename Conditional<AscendC::IsSameType<T, half>::value, float, int32_t>::type;
    constexpr uint32_t SUBMASK = (sizeof(T) == 1) ? 0xFFU : 0xFFFFU;
    uint64_t byteOff = elemIdx * sizeof(T);
    __gm__ uint32_t* wordAddr = base + (byteOff >> 2);
    uint32_t shift = (static_cast<uint32_t>(byteOff) & 3U) * 8U;
    uint32_t keepMask = ~(SUBMASK << shift);
    ACC valAcc = static_cast<ACC>(val);
    uint32_t cur = *wordAddr;
    uint32_t prev;
    do {
        prev = cur;
        uint32_t subBits = (prev >> shift) & SUBMASK;
        T curVal;
        if constexpr (AscendC::IsSameType<T, half>::value) {
            uint16_t hb = static_cast<uint16_t>(subBits);
            curVal = *reinterpret_cast<half*>(&hb);
        } else if constexpr (AscendC::IsSameType<T, int8_t>::value) {
            curVal = static_cast<int8_t>(subBits);
        } else {
            curVal = static_cast<uint8_t>(subBits);
        }
        T newVal = static_cast<T>(ReduceAcc<ACC, Mode>(static_cast<ACC>(curVal), valAcc));
        uint32_t newBits;
        if constexpr (AscendC::IsSameType<T, half>::value) {
            newBits = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&newVal));
        } else {
            newBits = static_cast<uint32_t>(static_cast<uint8_t>(newVal));
        }
        uint32_t newWord = (prev & keepMask) | ((newBits & SUBMASK) << shift);
        cur = asc_atomic_cas(wordAddr, prev, newWord);
    } while (cur != prev);
}

// SIMT VF: each thread reduces this core's update elements into var atomically. updates and indices
// are read DIRECTLY from GM (no UB staging) so the per-core element count is unbounded — same shape
// as other SIMT ops (e.g. foreach_acos). [blockBegin, blockBegin+blockElems) is this core's range.
template <typename INDICES_T, typename PARAMS_T, typename ADDR_T, uint8_t Mode>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtComputeReduce(
    __gm__ INDICES_T* idxGmAddr, __gm__ PARAMS_T* xGmAddr, __gm__ PARAMS_T* outputGmAddr,
    const ADDR_T blockBegin, const ADDR_T blockElems, const ADDR_T sliceSize, const ADDR_T varFirstDim)
{
    for (ADDR_T index = static_cast<ADDR_T>(threadIdx.x); index < blockElems;
         index += static_cast<ADDR_T>(blockDim.x)) {
        ADDR_T globalIdx = blockBegin + index;         // global update-element position
        ADDR_T m = globalIdx / sliceSize;              // which index entry
        ADDR_T sliceOff = globalIdx - m * sliceSize;   // offset within the slice
        INDICES_T idxVal = idxGmAddr[m];
        if (idxVal < 0 || static_cast<ADDR_T>(idxVal) >= varFirstDim) {
            continue;                                  // out-of-bound index: skip
        }
        ADDR_T dstIndex = static_cast<ADDR_T>(idxVal) * sliceSize + sliceOff;
        if constexpr (sizeof(PARAMS_T) == 4) {
            AtomicReduce32<PARAMS_T, Mode>(outputGmAddr + dstIndex, xGmAddr[globalIdx]);
        } else {
            AtomicReduceSubWord<PARAMS_T, Mode>(
                reinterpret_cast<__gm__ uint32_t*>(outputGmAddr), static_cast<uint64_t>(dstIndex),
                xGmAddr[globalIdx]);
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
class ScatterReduceSimt {
public:
    __aicore__ inline ScatterReduceSimt(const ScatterReduceSimtTilingData& tilingData, TPipe& pipe)
        : tiling_(tilingData)
    {
        (void)pipe;  // GM-direct kernel: no UB staging, TPipe unused
    }
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates);
    __aicore__ inline void Process();

private:
    const ScatterReduceSimtTilingData& tiling_;
    GlobalTensor<INDICES_T> idxGm_;
    GlobalTensor<PARAMS_T> xGm_;       // updates
    GlobalTensor<PARAMS_T> outputGm_;  // var (in-place reduce target)
    uint32_t blockIdx_{0};
    ADDR_T xBlockOffSet_{0};
    ADDR_T currBlockTilingSize_{0};
};

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::Init(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates)
{
    if (tiling_.sliceSize == 0) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    xBlockOffSet_ = static_cast<ADDR_T>(tiling_.blockTilingSize) * blockIdx_;
    currBlockTilingSize_ = (blockIdx_ == tiling_.blockNum - 1) ? static_cast<ADDR_T>(tiling_.tailBlockTilingSize)
                                                               : static_cast<ADDR_T>(tiling_.blockTilingSize);
    idxGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    xGm_.SetGlobalBuffer((__gm__ PARAMS_T*)updates);
    outputGm_.SetGlobalBuffer((__gm__ PARAMS_T*)var);
}

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::Process()
{
    if (tiling_.sliceSize == 0 || blockIdx_ >= tiling_.blockNum) {
        return;
    }
    // One SIMT launch over this core's whole element range; the grid-stride loop and direct GM reads
    // handle any size without a UB tile, so there is no per-tile element cap.
    asc_vf_call<SimtComputeReduce<INDICES_T, PARAMS_T, ADDR_T, Mode>>(
        dim3(THREAD_NUM), (__gm__ INDICES_T*)idxGm_.GetPhyAddr(), (__gm__ PARAMS_T*)xGm_.GetPhyAddr(),
        (__gm__ PARAMS_T*)outputGm_.GetPhyAddr(), xBlockOffSet_, currBlockTilingSize_,
        static_cast<ADDR_T>(tiling_.sliceSize), static_cast<ADDR_T>(tiling_.varFirstDim));
}
}  // namespace ScatterReduceCommon

#endif  // SCATTER_REDUCE_COMMON_SIMT_H
