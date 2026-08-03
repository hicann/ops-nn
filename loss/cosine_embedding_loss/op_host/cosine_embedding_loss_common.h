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
 * \file cosine_embedding_loss_common.h
 * \brief Shape and attribute helpers shared by CosineEmbeddingLoss host implementations.
 */
#ifndef OPS_LOSS_COSINE_EMBEDDING_LOSS_COMMON_H_
#define OPS_LOSS_COSINE_EMBEDDING_LOSS_COMMON_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "exe_graph/runtime/shape.h"

namespace ops {
namespace cosine_embedding_loss {
constexpr size_t kMaxRank = 8;
constexpr size_t kFeatureAxis = 1;
constexpr uint32_t kReductionNone = 0;
constexpr uint32_t kReductionSum = 1;
constexpr uint32_t kReductionMean = 2;
constexpr const char* kDefaultReduction = "mean";

using Dims = std::vector<int64_t>;

inline bool ShapeToDims(const gert::Shape& shape, Dims& dims)
{
    const size_t rank = shape.GetDimNum();
    if (rank == 0 || rank > kMaxRank) {
        return false;
    }
    dims.clear();
    dims.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim == 0 || dim < -1) {
            return false;
        }
        dims.push_back(dim);
    }
    return true;
}

inline bool RuntimeShapeToDims(const gert::Shape& shape, Dims& dims)
{
    if (!ShapeToDims(shape, dims)) {
        return false;
    }
    return std::all_of(dims.begin(), dims.end(), [](int64_t dim) { return dim > 0; });
}

inline bool BroadcastDim(int64_t lhs, int64_t rhs, int64_t& output)
{
    if (lhs == rhs) {
        output = lhs;
        return true;
    }
    if (lhs == 1) {
        output = rhs;
        return true;
    }
    if (rhs == 1) {
        output = lhs;
        return true;
    }
    if (lhs == -1 || rhs == -1) {
        output = -1;
        return true;
    }
    return false;
}

inline bool BroadcastShapes(const Dims& lhs, const Dims& rhs, Dims& output)
{
    const size_t rank = std::max(lhs.size(), rhs.size());
    if (rank > kMaxRank) {
        return false;
    }
    output.assign(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t lhsDim = i < lhs.size() ? lhs[lhs.size() - 1 - i] : 1;
        const int64_t rhsDim = i < rhs.size() ? rhs[rhs.size() - 1 - i] : 1;
        if (!BroadcastDim(lhsDim, rhsDim, output[rank - 1 - i])) {
            return false;
        }
    }
    return true;
}

inline bool RemoveAxis(const Dims& input, size_t axis, Dims& output)
{
    if (axis >= input.size()) {
        return false;
    }
    output.clear();
    output.reserve(input.size() - 1);
    for (size_t i = 0; i < input.size(); ++i) {
        if (i != axis) {
            output.push_back(input[i]);
        }
    }
    return true;
}

inline bool CheckedMultiply(int64_t lhs, int64_t rhs, int64_t& product)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    product = lhs * rhs;
    return true;
}

inline bool ElementCount(const Dims& dims, int64_t& elements)
{
    elements = 1;
    for (const int64_t dim : dims) {
        if (dim <= 0 || !CheckedMultiply(elements, dim, elements)) {
            return false;
        }
    }
    return true;
}

inline void SetShape(gert::Shape& shape, const Dims& dims)
{
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
}

inline const char* ReductionOrDefault(const char* reduction)
{
    return reduction == nullptr ? kDefaultReduction : reduction;
}

inline bool ParseReduction(const char* reduction, uint32_t& reductionKey)
{
    const char* value = ReductionOrDefault(reduction);
    if (strcmp(value, "none") == 0) {
        reductionKey = kReductionNone;
        return true;
    }
    if (strcmp(value, "sum") == 0) {
        reductionKey = kReductionSum;
        return true;
    }
    if (strcmp(value, kDefaultReduction) == 0) {
        reductionKey = kReductionMean;
        return true;
    }
    return false;
}
} // namespace cosine_embedding_loss
} // namespace ops

#endif // OPS_LOSS_COSINE_EMBEDDING_LOSS_COMMON_H_
