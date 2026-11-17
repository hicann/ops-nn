/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ELEMENTS_V2_AMAX_POLICY_H
#define SCATTER_ELEMENTS_V2_AMAX_POLICY_H

#include <algorithm>
#include <cstdint>

namespace ScatterElementsV2Amax {
// Buffer geometry, not performance thresholds. All allocations are rounded
// as TPipe::InitBuffer does on Ascend 910B (32 bytes).
constexpr uint64_t kBlockBytes = 32;
// CANN 9.1 dav-2201 Compare/Select use TMP_UB_OFFSET=184 KiB,
// TMP_UB_SIZE=8 KiB (basic_api/utils/kernel_utils_constants.h).
// Keep all TPipe buffers below that fixed SDK scratch region.
constexpr uint64_t kRuntimeUbBytes = 8 * 1024;
constexpr uint64_t kCompareElements = 64; // One FP32 vector repeat.
constexpr uint64_t kInputTile = 2048;
constexpr uint64_t kIndexTile = 1024;
constexpr uint64_t kExactFloatInteger = 1ULL << 24;

inline uint64_t CeilDiv(uint64_t n, uint64_t d) { return (n + d - 1) / d; }
inline uint64_t AlignUp(uint64_t n, uint64_t a) { return CeilDiv(n, a) * a; }
inline uint64_t BufferBytes(uint64_t bytes) { return AlignUp(bytes, kBlockBytes); }
inline uint64_t Gcd(uint64_t a, uint64_t b)
{
    while (b) {
        const uint64_t next = a % b;
        a = b;
        b = next;
    }
    return a;
}

enum class Algorithm { InputPartition, OutputPartition };
enum class Layout { Elements, Rows };
enum class Reason { AtomicFits, AtomicAlignment, AtomicCapacity, RowBufferCapacity, AtomicIndexRange };

struct Plan {
    Algorithm algorithm = Algorithm::OutputPartition;
    Layout layout = Layout::Elements;
    Reason reason = Reason::AtomicCapacity;
    uint32_t cores = 0;
    uint32_t chunk = 0; // Input elements, output elements, or buffered rows.
    uint32_t tile = kInputTile;
    uint64_t ubBytes = 0;
    bool valid = false;
};

inline uint64_t AtomicBytes(uint64_t output, uint64_t bytes)
{
    return BufferBytes(output * bytes) + 2 * BufferBytes(output * sizeof(int32_t)) +
           BufferBytes(kInputTile * sizeof(int64_t)) + BufferBytes(kInputTile * bytes);
}

inline uint64_t RowBytes(uint64_t rows, uint64_t columns, uint64_t bytes)
{
    const uint64_t stride = AlignUp(columns, kCompareElements);
    // Output, seen rows, one source row, two optional FP32 conversion rows,
    // three comparison masks; index, converted index, positions, selected,
    // and two index-selection masks. Every term matches a kernel TBuf.
    return BufferBytes(rows * stride * bytes) + BufferBytes(rows * sizeof(uint32_t)) + BufferBytes(stride * bytes) +
           (bytes == 4 ? 0 : 2 * BufferBytes(stride * 4)) + 3 * BufferBytes(stride / 8) + BufferBytes(kIndexTile * 8) +
           3 * BufferBytes(kIndexTile * 4) + 2 * BufferBytes(kIndexTile / 8);
}

inline uint64_t ElementBytes(uint64_t chunk, uint64_t bytes)
{
    return BufferBytes(chunk * bytes) + BufferBytes(chunk * 4) + BufferBytes(kInputTile * 8) +
           BufferBytes(kInputTile * bytes) + 3 * BufferBytes(kInputTile * 4) + BufferBytes(kInputTile / 4);
}

inline Plan Select(uint64_t n, uint64_t e, uint64_t c, uint64_t bytes, bool compactRows, uint64_t cores, uint64_t ub)
{
    Plan p;
    if (!n || !e || !c || !cores || (bytes != 2 && bytes != 4) || n > UINT32_MAX || c > UINT32_MAX || e > UINT32_MAX ||
        n > UINT32_MAX / c || e > UINT32_MAX / c)
        return p;
    if (ub <= kRuntimeUbBytes)
        return p;
    ub -= kRuntimeUbBytes;
    const uint64_t input = e * c, output = n * c;
    const uint64_t alignment = kBlockBytes / bytes;
    const bool aligned = output % alignment == 0;
    if (aligned && input <= uint64_t(INT32_MAX) / 2 && AtomicBytes(output, bytes) <= ub) {
        p.algorithm = Algorithm::InputPartition;
        p.reason = Reason::AtomicFits;
        p.cores = std::min(cores, CeilDiv(input, kInputTile));
        p.chunk = CeilDiv(input, p.cores);
        p.ubBytes = AtomicBytes(output, bytes);
        p.valid = true;
        return p;
    }
    p.reason = !aligned                        ? Reason::AtomicAlignment :
               input > uint64_t(INT32_MAX) / 2 ? Reason::AtomicIndexRange :
                                                 Reason::AtomicCapacity;
    if (compactRows && n <= kExactFloatInteger) {
        const uint64_t rowAlignment = kBlockBytes / Gcd(c * bytes, kBlockBytes);
        const uint64_t groups = CeilDiv(n, rowAlignment);
        const uint64_t desired = CeilDiv(groups, std::min(groups, cores)) * rowAlignment;
        // Monotonic resource fit, independent of input degree distribution.
        uint64_t low = 0, high = CeilDiv(desired, rowAlignment);
        while (low < high) {
            const uint64_t mid = (low + high + 1) / 2;
            if (RowBytes(std::min(n, mid * rowAlignment), c, bytes) <= ub)
                low = mid;
            else
                high = mid - 1;
        }
        if (low) {
            p.layout = Layout::Rows;
            p.chunk = std::min(n, low * rowAlignment);
            p.tile = kIndexTile;
            p.cores = std::min(cores, CeilDiv(n, p.chunk));
            p.ubBytes = RowBytes(p.chunk, c, bytes);
            p.valid = true;
            return p;
        }
        p.reason = Reason::RowBufferCapacity;
    }
    const uint64_t fixed = ElementBytes(0, bytes);
    if (ub <= fixed)
        return p;
    const uint64_t capacity = (ub - fixed) / (bytes + sizeof(uint32_t)) / alignment * alignment;
    if (!capacity)
        return p;
    p.cores = std::min(cores, CeilDiv(output, alignment));
    p.chunk = std::min(capacity, AlignUp(CeilDiv(output, p.cores), alignment));
    p.ubBytes = ElementBytes(p.chunk, bytes);
    p.valid = p.ubBytes <= ub;
    return p;
}
} // namespace ScatterElementsV2Amax

#endif
