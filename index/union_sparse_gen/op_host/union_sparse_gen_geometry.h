/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_GEOMETRY_H
#define UNION_SPARSE_GEN_GEOMETRY_H

#include "exe_graph/runtime/tensor.h"
#include "union_sparse_gen_contract.h"
#include "log/log.h"

namespace UnionSparseGenGeometry {
namespace Contract = UnionSparseGenContract;
constexpr const char* kOpName = "UnionSparseGen";
struct Result {
    int64_t groups = Contract::kUnspecifiedGeometry;
    int64_t logical = Contract::kUnspecifiedGeometry;
};

inline bool Invalid(const char* reason)
{
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "automatic geometry", reason);
    return false;
}

// Shape presence and value availability are different. Never dereference a
// Device address on Host, or replace a present seqUsed with prefix lengths.
inline bool Readable(const gert::Tensor* tensor, ge::DataType type, int64_t elements)
{
    if (tensor == nullptr || tensor->GetDataType() != type || elements < 0) {
        return false;
    }
    const auto placement = tensor->GetPlacement();
    const auto elementBytes = type == ge::DT_INT64 ? sizeof(int64_t) : sizeof(int32_t);
    return (placement == gert::kOnHost || placement == gert::kFollowing) && tensor->GetAddr() != nullptr &&
           static_cast<uint64_t>(elements) <= tensor->GetSize() / elementBytes;
}

template <typename Context>
bool ResolveFamily(const Context* context, int64_t batch, int64_t totalQ, int64_t maxQ, int64_t divisor, bool query,
                   int64_t& result)
{
    const auto prefixIndex = query ? Contract::kCuQInput : Contract::kCuKvInput;
    const auto usedIndex = query ? Contract::kUsedQInput : Contract::kUsedKvInput;
    const bool hasPrefix = context->GetOptionalInputShape(prefixIndex) != nullptr;
    const bool hasUsed = context->GetOptionalInputShape(usedIndex) != nullptr;
    if (!hasPrefix && !hasUsed) {
        return Invalid("missing effective length source");
    }
    const auto* prefix = hasPrefix ? context->GetOptionalInputTensor(prefixIndex) : nullptr;
    const auto* used = hasUsed ? context->GetOptionalInputTensor(usedIndex) : nullptr;
    if ((hasPrefix && !Readable(prefix, ge::DT_INT64, batch + 1)) ||
        (hasUsed && !Readable(used, ge::DT_INT32, batch))) {
        result = Contract::kUnspecifiedGeometry;
        return true; // InferShape may defer; runtime tiling must reject this.
    }
    const auto* offsets = hasPrefix ? prefix->template GetData<int64_t>() : nullptr;
    const auto* lengths = hasUsed ? used->template GetData<int32_t>() : nullptr;
    if (hasPrefix && (offsets[0] != 0 || (query && offsets[batch] != totalQ))) {
        return Invalid("prefix must start at zero and Q prefix must cover stored rows");
    }
    int64_t accumulated = query ? 0 : 1;
    for (int64_t b = 0; b < batch; ++b) {
        if (hasPrefix && (offsets[b] < 0 || offsets[b + 1] < offsets[b])) {
            return Invalid("prefix lengths must be nonnegative and nondecreasing");
        }
        const int64_t storage = hasPrefix ? offsets[b + 1] - offsets[b] : (query ? maxQ : 0);
        const int64_t effective = hasUsed ? lengths[b] : storage;
        if (effective < 0 || ((hasPrefix || query) && effective > storage) || (query && maxQ > 0 && effective > maxQ)) {
            return Invalid("effective lengths must fit their request storage and Q capacity");
        }
        const int64_t rounded = effective / divisor + (effective % divisor != 0);
        if (query) {
            if (rounded > std::numeric_limits<int64_t>::max() - accumulated) {
                return Invalid("group sum overflows int64");
            }
            accumulated += rounded;
        } else {
            accumulated = std::max(accumulated, rounded);
        }
    }
    if (!query && accumulated > std::numeric_limits<int32_t>::max()) {
        return Invalid("logical block count exceeds INT32_MAX");
    }
    result = accumulated;
    return true;
}

template <typename Context>
bool Resolve(const Context* context, int64_t batch, int64_t totalQ, int64_t maxQ, int64_t width, int64_t blockY,
             Result& result)
{
    if (result.groups < Contract::kUnspecifiedGeometry || result.logical < Contract::kUnspecifiedGeometry ||
        result.logical == 0 || result.logical > std::numeric_limits<int32_t>::max() || batch < 0 || width <= 0 ||
        blockY <= 0 || (batch == 0 && (totalQ != 0 || result.groups > 0))) {
        return Invalid("expected G>=0 or -1, Lmax in [1,INT32_MAX] or -1, and valid batch geometry");
    }
    // Explicit geometry never accesses length tensors. No cross-call cache.
    if (batch == 0) {
        result.groups = 0;
        result.logical = result.logical == Contract::kUnspecifiedGeometry ? 1 : result.logical;
        return true;
    }
    if (result.groups == Contract::kUnspecifiedGeometry &&
        !ResolveFamily(context, batch, totalQ, maxQ, width, true, result.groups)) {
        return false;
    }
    return result.logical != Contract::kUnspecifiedGeometry ||
           ResolveFamily(context, batch, totalQ, maxQ, blockY, false, result.logical);
}
} // namespace UnionSparseGenGeometry
#endif
