/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_ARCH35_TILING_DATA_H
#define UNION_SPARSE_GEN_ARCH35_TILING_DATA_H
#include "../union_sparse_gen_tiling_data.h"
#include <cstddef>

namespace UnionSparseGenA5Plan {
using UnionSparseGenUnits::kBitsPerByte;
using UnionSparseGenUnits::kBitsPerWord;
using UnionSparseGenUnits::kDmaBlockBytes;
using UnionSparseGenUnits::kDmaWords;
using UnionSparseGenUnits::kKiB;
using UnionSparseGenUnits::kPrefixBytes;
using UnionSparseGenUnits::kWordBytes;
constexpr int32_t kThreadNum = 512;
constexpr int32_t kWarpLanes = 32, kWarpShift = 5, kWarpLaneMask = kWarpLanes - 1;
constexpr int32_t kWordShift = 5;
constexpr int32_t kBinaryScanRadix = 2;
constexpr int32_t kMinGroupWidth = 8, kHalfWarpLanes = kWarpLanes / 2;
constexpr int32_t kMaxGroupWidth = 32;
constexpr int32_t kPrefixSlots = 32; // Preserve the existing state ABI and its unused prefix reserve.
constexpr int32_t kVectorWords = 4, kVectorShift = 2;
constexpr int32_t kGatherBlockY = 4, kNarrowBlockY = 8, kMediumBlockY = 16, kWideBlockY = 128;
// Each butterfly exchange joins two lane groups of the named size.
constexpr int32_t kPairShuffleDistance = 1, kQuadShuffleDistance = 2, kOctetShuffleDistance = 4;
constexpr int32_t kHalfWarpShuffleDistance = 8, kFullWarpShuffleDistance = 16;
constexpr uint32_t kAllQueryBits = 0xFFFFFFFFU, kInvalidSortKey = 0xFFFFFFFFU;
// Four radix planes (input key/member and sorted key/member); two compact planes.
constexpr int32_t kInputKeyPlane = 0, kInputMemberPlane = 1, kSortedKeyPlane = 2, kSortedMemberPlane = 3;
constexpr int32_t kRadixPlanes = 4, kCompactPlanes = 2;
constexpr int32_t kRadixRecordBytes = kRadixPlanes * kWordBytes;
constexpr int32_t kCompactRecordBytes = kCompactPlanes * kWordBytes;
constexpr int32_t kPresenceBitsPerBlock = 1;
constexpr int32_t kTableBlockAlignment = 256;
constexpr int32_t kMinTableBytes = 2 * kKiB;
constexpr int32_t kBudgetShrinkFactor = 2;
constexpr uint64_t kCompilerReserveBytes = 64U * kKiB;
} // namespace UnionSparseGenA5Plan

struct UnionSparseGenA5TilingData {
    UnionSparseGenTilingData semantic;
    int64_t radixCapacity = 0;
    int64_t radixTmpBytes = 0;
    int64_t radixScratchBytes = 0;
    int64_t tableBlockCapacity = 0;
    int64_t compactCapacity = 0;
    int64_t tableScratchBytes = 0;
};
struct UnionSparseGenA5GroupState {
    int64_t qStart, qLen, kvLen, local, rows, logical, head, outputBase;
    int64_t inputCount, recordCount, rank;
    int32_t prefix[UnionSparseGenA5Plan::kPrefixSlots];
    int32_t counts[UnionSparseGenA5Plan::kMaxGroupWidth];
    int64_t starts[UnionSparseGenA5Plan::kMaxGroupWidth + 1];
};
constexpr int64_t kUnionSparseGenA5StateBytes = (sizeof(UnionSparseGenA5GroupState) +
                                                 UnionSparseGenA5Plan::kDmaBlockBytes - 1) /
                                                UnionSparseGenA5Plan::kDmaBlockBytes *
                                                UnionSparseGenA5Plan::kDmaBlockBytes;
constexpr int32_t kUnionSparseGenA5InputCountWord = offsetof(UnionSparseGenA5GroupState, inputCount) /
                                                    UnionSparseGenA5Plan::kPrefixBytes;
constexpr int32_t kUnionSparseGenA5RecordCountWord = offsetof(UnionSparseGenA5GroupState, recordCount) /
                                                     UnionSparseGenA5Plan::kPrefixBytes;
#endif
