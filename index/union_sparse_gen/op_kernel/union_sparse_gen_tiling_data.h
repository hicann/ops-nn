/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_TILING_DATA_H
#define UNION_SPARSE_GEN_TILING_DATA_H

#include <cstdint>

// Physical storage units shared by host budgeting and both device backends.
namespace UnionSparseGenUnits {
constexpr int32_t kBitsPerByte = 8;
constexpr int32_t kWordBytes = sizeof(uint32_t), kHalfwordBytes = sizeof(uint16_t);
constexpr int32_t kPrefixBytes = sizeof(int64_t);
constexpr int32_t kBitsPerWord = kWordBytes * kBitsPerByte;
constexpr int32_t kBitsPerHalfword = kHalfwordBytes * kBitsPerByte;
constexpr int32_t kHalfwordsPerWord = kWordBytes / kHalfwordBytes;
constexpr int32_t kDmaBlockBytes = 32;
constexpr int32_t kDmaWords = kDmaBlockBytes / kWordBytes;
constexpr int32_t kKiB = 1024;
} // namespace UnionSparseGenUnits

// A cached window has 128 requests and 129 prefix elements. INT64 DMA pads
// that prefix to 132 elements; INT32 used lengths need exactly 128 elements.
constexpr int64_t kUnionSparseGenMetaWindow = 128;
constexpr int64_t kUnionSparseGenPrefixStride = 132;
constexpr int64_t kUnionSparseGenLegacyMetaBudget = 2176;
#if defined(__CCE_AICORE__)
__aicore__
#endif
    constexpr int64_t
    UnionSparseGenMetadataBytes(bool hasCuQ, bool hasUsedQ, bool hasUsedKv)
{
    return (hasCuQ ? kUnionSparseGenPrefixStride * sizeof(int64_t) : 0) +
           (hasUsedQ ? kUnionSparseGenMetaWindow * sizeof(int32_t) : 0) +
           (hasUsedKv ? kUnionSparseGenMetaWindow * sizeof(int32_t) : kUnionSparseGenPrefixStride * sizeof(int64_t));
}
// Preserve the previous algorithm resource ceiling when actual metadata shrinks.
// This floor is budget only: kernel InitBuffer allocates the consumed windows.
#if defined(__CCE_AICORE__)
__aicore__
#endif
    constexpr int64_t
    UnionSparseGenMetadataBudgetBytes(bool hasCuQ, bool hasUsedQ, bool hasUsedKv)
{
    const int64_t bytes = UnionSparseGenMetadataBytes(hasCuQ, hasUsedQ, hasUsedKv);
    return bytes > kUnionSparseGenLegacyMetaBudget ? bytes : kUnionSparseGenLegacyMetaBudget;
}

struct UnionSparseGenCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct UnionSparseGenTilingData {
    int64_t batchSize = 0;
    int64_t headCount = 0;
    int64_t totalQ = 0;
    int64_t topK = 0;
    int64_t numGroups = 0;
    int64_t maxLogicalBlocks = 0;
    int64_t unionCapacity = 0;
    int64_t unionSize = 0;
    int64_t blockY = 0;
    int64_t maskType = 0;
    int64_t usedCoreNum = 0;
    int64_t workPerCore = 0;
    int64_t workTail = 0;
    int64_t useInt32 = 0;
    int64_t hasSeqUsedQ = 0;
    int64_t hasSeqUsedKv = 0;
    // SIMD-only UB plan; zero for the SIMT backend.
    int64_t simdInputWindow = 0;
    int64_t simdTileBlocks = 0;
    int64_t simdAlgorithmBytes = 0;
    int64_t isInputIdxSorted = 1;
    int64_t residualBlockMode = 0;
    int64_t hasCuSqlenKv = 1;
};

#endif
