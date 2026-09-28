/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ELEMENTS_V2_LOW_MEMORY_POLICY_H
#define SCATTER_ELEMENTS_V2_LOW_MEMORY_POLICY_H

#include <cstdint>
#include <cstddef>

namespace ScatterElementsV2LowMemory {
// API and tiling use different SocVersion enum types. Keep the supported
// platforms identical at both gates of the shared arch22 implementation.
template <typename SocVersion>
constexpr bool IsSupportedSoc(SocVersion soc)
{
    return soc == SocVersion::ASCEND910B || soc == SocVersion::ASCEND910_93;
}

// The existing kernel uses at most five column partitions. Keep the candidate
// bounded below 448 MiB of tensor scratch, leaving room for runtime workspace.
constexpr uint64_t MAX_TENSOR_SCRATCH_BYTES = 448ULL * 1024 * 1024;
// Short rows do not amortize the repeated transpose/synchronization stages.
// Keep them on the original path; the lower bound is covered by A2 A/B tests.
constexpr int64_t MIN_REDUCTION_ROW = 256;
constexpr uint64_t MAX_REDUCTION_ROW = 20480;
constexpr uint64_t MIN_ELEMENTS = 10000000;

template <typename Shape>
constexpr bool IsBoundedFirstAxisShape(const Shape& data, const Shape& indices, const Shape& updates, int64_t axis,
                                       uint64_t coreNum)
{
    const auto rank = data.GetDimNum();
    if (coreNum == 0 || rank < 2 || rank > 8 || indices.GetDimNum() != rank || updates.GetDimNum() != rank ||
        (axis != 0 && axis != -static_cast<int64_t>(rank))) {
        return false;
    }
    const int64_t rows = data.GetDim(0);
    if (rows < MIN_REDUCTION_ROW || static_cast<uint64_t>(rows) > MAX_REDUCTION_ROW) {
        return false;
    }
    uint64_t elements = 1;
    // Also bounds byte offsets used by the existing transpose implementation.
    constexpr uint64_t MAX_ELEMENTS = MAX_TENSOR_SCRATCH_BYTES / 12 * 5;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = data.GetDim(i);
        if (dim <= 0 || dim != indices.GetDim(i) || dim != updates.GetDim(i) ||
            static_cast<uint64_t>(dim) > MAX_ELEMENTS / elements) {
            return false;
        }
        elements *= static_cast<uint64_t>(dim);
    }
    const uint64_t columns = elements / static_cast<uint64_t>(rows);
    // This is the pre-existing first-axis performance threshold. The caller
    // additionally checks dtype, architecture, layout and include_self.
    if (elements <= MIN_ELEMENTS || columns < 256) {
        return false;
    }
    uint64_t parts = columns / coreNum;
    parts = parts == 0 ? 1 : (parts > 5 ? 5 : parts);
    const uint64_t partColumns = columns / parts;
    const uint64_t tailColumns = columns % parts;
    const uint64_t maxColumns = partColumns > tailColumns ? partColumns : tailColumns;
    return maxColumns * static_cast<uint64_t>(rows) * 12 <= MAX_TENSOR_SCRATCH_BYTES;
}
} // namespace ScatterElementsV2LowMemory
#endif
