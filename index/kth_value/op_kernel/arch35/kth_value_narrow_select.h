/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_NARROW_SELECT_H
#define KTH_VALUE_NARROW_SELECT_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/small_axis_two_stage_base.h"

namespace KthValue {
constexpr uint32_t NARROW_SELECT_MODE = 1U;
constexpr uint32_t NARROW_SELECT_WARP_SIZE = 32U;
constexpr uint32_t NARROW_SELECT_CAPACITY_SMALL = 64U;
constexpr uint32_t NARROW_SELECT_CAPACITY_MEDIUM = 128U;
constexpr uint32_t NARROW_SELECT_CAPACITY_LARGE = 256U;
constexpr uint32_t NARROW_SELECT_CAPACITY_MAX = 512U;
constexpr uint32_t NARROW_SELECT_CAPACITY_BYTE = 384U;
constexpr uint16_t NARROW_SELECT_ABS_MASK = 0x7FFFU;
constexpr uint16_t NARROW_SELECT_SIGN_MASK = 0x8000U;
constexpr uint16_t NARROW_SELECT_KEY_MASK = 0xFFFFU;
constexpr uint16_t NARROW_SELECT_HALF_INF = 0x7C00U;
constexpr uint16_t NARROW_SELECT_BFLOAT_INF = 0x7F80U;
constexpr int32_t NARROW_SELECT_UPPER_SENTINEL = 65536;
constexpr int32_t NARROW_SELECT_BYTE_SENTINEL = 256;
constexpr int32_t NARROW_SELECT_LOWER_SENTINEL = -129;
constexpr int32_t NARROW_SELECT_BINARY_RADIX = 2;

// Ordering keys collapse both zeros and all NaN encodings. The final value is read
// from the selected raw input, so the output preserves its zero sign or NaN payload.
template <typename T>
__simt_callee__ __aicore__ inline int32_t MakeNarrowFloatKey(uint16_t raw)
{
    uint16_t absBits = raw & NARROW_SELECT_ABS_MASK;
    constexpr uint16_t infBits = IsSameType<T, half>::value ? NARROW_SELECT_HALF_INF : NARROW_SELECT_BFLOAT_INF;
    if (absBits == 0U) {
        raw = 0U;
    }
    if (absBits > infBits) {
        return NARROW_SELECT_KEY_MASK;
    }
    if ((raw & NARROW_SELECT_SIGN_MASK) != 0U) {
        return (~raw) & NARROW_SELECT_KEY_MASK;
    }
    return raw ^ NARROW_SELECT_SIGN_MASK;
}

template <typename T>
__simt_callee__ __aicore__ inline int32_t LoadNarrowKey(__ubuf__ T* input)
{
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        return MakeNarrowFloatKey<T>(*reinterpret_cast<__ubuf__ uint16_t*>(input));
    } else {
        return static_cast<int32_t>(*input);
    }
}

template <uint32_t Chunks>
__simt_callee__ __aicore__ inline int32_t SelectNarrowKey(const int32_t (&keys)[Chunks], int32_t low, int32_t high,
                                                          uint32_t kth)
{
    while (low < high) {
        int32_t mid = low + (high - low) / NARROW_SELECT_BINARY_RADIX;
        uint32_t count = 0;
#pragma unroll
        for (uint32_t c = 0; c < Chunks; ++c) {
            count += keys[c] <= mid ? 1U : 0U;
        }
        count = asc_reduce_add(count);
        if (count > kth) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

// Resolve ties in original column order: skip complete matching chunks, then select
// the rank-th matching lane with an inclusive prefix. Every lane participates in collectives.
template <uint32_t Chunks>
__simt_callee__ __aicore__ inline uint32_t FindNarrowIndex(const int32_t (&keys)[Chunks], int32_t key, uint32_t kth,
                                                           uint32_t lane, uint32_t cols)
{
    uint32_t less = 0;
#pragma unroll
    for (uint32_t c = 0; c < Chunks; ++c) {
        less += keys[c] < key ? 1U : 0U;
    }
    less = asc_reduce_add(less);
    uint32_t rank = kth - less;
    uint32_t selected = cols;
#pragma unroll
    for (uint32_t c = 0; c < Chunks; ++c) {
        uint32_t match = keys[c] == key ? 1U : 0U;
        uint32_t count = asc_reduce_add(match);
        if (rank < count) {
            uint32_t prefix = match;
#pragma unroll
            for (uint32_t shift = 1; shift < NARROW_SELECT_WARP_SIZE; shift *= NARROW_SELECT_BINARY_RADIX) {
                uint32_t before = asc_shfl_up(prefix, shift);
                if (lane >= shift) {
                    prefix += before;
                }
            }
            selected = match != 0 && prefix == rank + 1 ? c * NARROW_SELECT_WARP_SIZE + lane : cols;
            selected = asc_reduce_min(selected);
            break;
        }
        rank -= count;
    }
    return selected;
}

// One warp selects a narrow-key row without sorting; cached keys avoid repeated UB loads.
template <typename T, uint32_t MaxAxis>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::TWO_STAGE_THREAD_NUM) __aicore__
    void SimtSelectNarrowRows(uint32_t rows, uint32_t cols, uint32_t kth, uint64_t outputStart, __ubuf__ T* input,
                              __gm__ volatile T* values, __gm__ volatile int64_t* indices)
{
    constexpr bool isFloat = IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value;
    constexpr int32_t upperSentinel = isFloat ? NARROW_SELECT_UPPER_SENTINEL : NARROW_SELECT_BYTE_SENTINEL;
    constexpr uint32_t lanes = NARROW_SELECT_WARP_SIZE;
    constexpr uint32_t chunks = MaxAxis / lanes;
    // One warp owns one row; invalid tail lanes hold a key above every valid value.
    // Key selection finds the kth value first, then FindNarrowIndex recovers its stable index.
    uint32_t lane = static_cast<uint32_t>(threadIdx.x) % lanes;
    uint32_t warp = static_cast<uint32_t>(threadIdx.x) / lanes;
    for (uint32_t row = warp; row < rows; row += SmallAxisCommon::TWO_STAGE_THREAD_NUM / lanes) {
        int32_t keys[chunks];
        int32_t low = upperSentinel;
        int32_t high = NARROW_SELECT_LOWER_SENTINEL;
#pragma unroll
        for (uint32_t c = 0; c < chunks; ++c) {
            uint32_t col = c * lanes + lane;
            int32_t value = upperSentinel;
            if (col < cols) {
                value = LoadNarrowKey(input + row * cols + col);
            }
            keys[c] = value;
            if (col < cols) {
                low = value < low ? value : low;
                high = value > high ? value : high;
            }
        }
        low = asc_reduce_min(low);
        high = asc_reduce_max(high);
        uint32_t selected = kth;
        if (low != high) {
            low = SelectNarrowKey(keys, low, high, kth);
            selected = FindNarrowIndex(keys, low, kth, lane, cols);
        }
        if (lane == 0) {
            if constexpr (isFloat) {
                values[outputStart + row] = input[row * cols + selected];
            } else {
                values[outputStart + row] = static_cast<T>(low);
            }
            indices[outputStart + row] = selected;
        }
    }
}

} // namespace KthValue

#endif
