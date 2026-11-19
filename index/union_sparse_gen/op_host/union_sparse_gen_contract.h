/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_CONTRACT_H
#define UNION_SPARSE_GEN_CONTRACT_H
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <cstdint>
#include <limits>

namespace UnionSparseGenContract {
// Shared GE attribute order; keep OpDef, L0, InferShape and tiling in sync.
constexpr size_t kBlockShape = 0, kUnionSize = 1, kMaskType = 2, kNumGroups = 3, kMaxLogical = 4, kMaxQ = 5, kMaxKv = 6,
                 kLayout = 7, kSorted = 8, kResidual = 9;
// Tensor positions and axes are protocol roles, not arbitrary array offsets.
constexpr size_t kSparseIdxInput = 0, kSparseCountInput = 1, kCuQInput = 2, kCuKvInput = 3;
constexpr size_t kUsedQInput = 4, kUsedKvInput = 5, kInputCount = 6;
constexpr size_t kFirstOptionalInput = kCuQInput, kFirstUsedLengthInput = kUsedQInput;
constexpr size_t kUnionIdxOutput = 0, kUnionCountOutput = 1, kExactMaskOutput = 2, kOutputCount = 3;
constexpr size_t kHeadAxis = 0, kQueryAxis = 1, kSparseEntryAxis = 2;
constexpr size_t kGroupAxis = 1, kUnionEntryAxis = 2, kMaskTokenAxis = 2, kLengthAxis = 0;
constexpr size_t kSparseIdxRank = 3, kSparseCountRank = 2, kLengthRank = 1;
constexpr size_t kUnionIdxRank = 3, kUnionCountRank = 2, kExactMaskRank = 3;
constexpr size_t kBlockXAxis = 0, kBlockYAxis = 1;
constexpr int64_t kIndexElementBytes = sizeof(int32_t);
constexpr int64_t kExplicitBlocks = 0, kUnspecifiedMaxSeqLen = -1;
constexpr int64_t kDefaultMaskType = 0;
constexpr int64_t kUnspecifiedGeometry = -1;
constexpr size_t kInputRanks[kInputCount] = {kSparseIdxRank, kSparseCountRank, kLengthRank,
                                             kLengthRank,    kLengthRank,      kLengthRank};
constexpr size_t kOutputRanks[kOutputCount] = {kUnionIdxRank, kUnionCountRank, kExactMaskRank};
constexpr bool IsPrefixInput(size_t index) { return index == kCuQInput || index == kCuKvInput; }
// Public specification and length element sizes, shared by all host entry points.
constexpr size_t kBlockShapeDimensions = 2;
constexpr int64_t kSupportedBlockX = 1, kSupportedUnionSize = 8;
constexpr int64_t kSupportedBlockY = 128;
constexpr int64_t kDefaultBlockY = kSupportedBlockY;
constexpr int64_t kSparsePatternLayout = 4;
constexpr int64_t kPrefixElementBytes = sizeof(int64_t), kUsedLengthElementBytes = sizeof(int32_t);
enum class MaskType : int64_t { NonCausal = 0, Causal = 1 };
inline bool SupportsMask(int64_t type)
{
    return type == static_cast<int64_t>(MaskType::NonCausal) || type == static_cast<int64_t>(MaskType::Causal);
}
inline bool SupportsBlockY(int64_t y) { return y == 4 || y == 8 || y == 16 || y == 128; }
inline bool SupportsResidual(int64_t residual) { return residual == 0 || residual == 1; }
inline bool SupportsBlockShape(const int64_t* data, size_t size)
{
    return size == kBlockShapeDimensions && data != nullptr && data[kBlockXAxis] == kSupportedBlockX &&
           SupportsBlockY(data[kBlockYAxis]);
}
inline bool SupportsSpec(int64_t x, int64_t y, int64_t w, int64_t mask, int64_t layout, bool /* sorted */,
                         int64_t residual)
{
    return x == kSupportedBlockX && SupportsBlockY(y) && w == kSupportedUnionSize && SupportsMask(mask) &&
           layout == kSparsePatternLayout && SupportsResidual(residual);
}
inline bool Attributes(int64_t layout) { return layout == kSparsePatternLayout; }
// A negative descriptor size denotes an absent optional input. Values stay on Device.
inline bool LengthGeometry(int64_t totalQ, int64_t cuQSize, int64_t usedQSize, int64_t cuKvSize, int64_t usedKvSize,
                           int64_t maxQ, int64_t maxKv, int64_t& batch)
{
    const bool fixedQ = cuQSize < 0 && usedQSize < 0;
    const bool fixedKv = cuKvSize < 0 && usedKvSize < 0;
    if (totalQ < 0 || (fixedQ && maxQ < 0) || (fixedKv && maxKv < 0)) {
        return false;
    }
    batch = -1;
    const int64_t sizes[] = {cuQSize, usedQSize, cuKvSize, usedKvSize};
    for (size_t i = 0; i < 4; ++i) {
        if (sizes[i] < 0) {
            continue;
        }
        const bool prefix = i % 2 == 0;
        const int64_t bytes = prefix ? kPrefixElementBytes : kUsedLengthElementBytes;
        const int64_t candidate = sizes[i] - (prefix ? 1 : 0);
        if (candidate < 0 || sizes[i] > std::numeric_limits<int64_t>::max() / bytes - 1 ||
            (batch >= 0 && candidate != batch)) {
            return false;
        }
        batch = candidate;
    }
    if (fixedQ) {
        // blockX=1: the final request may occupy less than a full maxQ slot.
        if (maxQ == 0) {
            batch = batch < 0 ? 0 : batch;
            return totalQ == 0;
        }
        const int64_t fixedBatch = totalQ / maxQ + (totalQ % maxQ != 0);
        if (batch >= 0 && batch != fixedBatch) {
            return false;
        }
        batch = fixedBatch;
        return true;
    }
    if (batch == 0) {
        return totalQ == 0;
    }
    if (cuQSize >= 0) {
        return true;
    }
    return totalQ % batch == 0;
}
// maxKv is consumed only without either KV tensor. Block IDs must remain INT32.
inline bool FixedKvFits(int64_t maxKv, int64_t blockY)
{
    return maxKv >= 0 && blockY > 0 && maxKv / blockY + (maxKv % blockY != 0) <= std::numeric_limits<int32_t>::max();
}
inline bool Capacity(int64_t k, int64_t width, int64_t logical, int64_t residual, int64_t& capacity)
{
    if (k <= 0 || width != kSupportedUnionSize || logical <= 0 || logical > std::numeric_limits<int32_t>::max() ||
        !SupportsResidual(residual) || k > std::numeric_limits<int64_t>::max() / width - residual) {
        return false;
    }
    capacity = std::min(width * (k + residual), logical);
    return true;
}
inline bool OutputFits(int64_t heads, int64_t groups, int64_t capacity, int64_t y)
{
    int64_t bytes = sizeof(int32_t);
    for (int64_t factor : {heads, groups, capacity, y}) {
        if (factor < 0 || (factor && bytes > std::numeric_limits<int64_t>::max() / factor)) {
            return false;
        }
        bytes *= factor;
    }
    return true;
}
inline bool InputFits(int64_t heads, int64_t totalQ, int64_t topK) { return OutputFits(heads, totalQ, topK, 1); }
} // namespace UnionSparseGenContract
#endif
