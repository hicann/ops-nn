/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_ARCH22_MEMBERSHIP_PLAN_H
#define UNION_SPARSE_GEN_ARCH22_MEMBERSHIP_PLAN_H
#include <cstdint>
#include "../union_sparse_gen_tiling_data.h"

namespace UnionSparseGenA2Plan {
// Hardware units and record layout; keep byte counts separate from element counts.
using UnionSparseGenUnits::kBitsPerByte;
using UnionSparseGenUnits::kBitsPerHalfword;
using UnionSparseGenUnits::kBitsPerWord;
using UnionSparseGenUnits::kDmaBlockBytes;
using UnionSparseGenUnits::kDmaWords;
using UnionSparseGenUnits::kHalfwordBytes;
using UnionSparseGenUnits::kHalfwordsPerWord;
using UnionSparseGenUnits::kKiB;
using UnionSparseGenUnits::kWordBytes;
constexpr int32_t kVectorRepeatBytes = 256;
constexpr int32_t kVectorRepeatBlocks = kVectorRepeatBytes / kDmaBlockBytes;
constexpr int32_t kCompareLanes = kVectorRepeatBytes / kWordBytes;
constexpr int32_t kPairWords = 2; // Encoded block key followed by query membership.
constexpr int32_t kSortBuffers = 2;
constexpr int32_t kInputPrefetchSlots = 2;
constexpr int32_t kResidentRecordWords = kPairWords * kSortBuffers;
constexpr int32_t kResidentRecordBytes = kResidentRecordWords * kWordBytes;
constexpr int32_t kOutputSlots = 2;
constexpr int32_t kPrefixWords = sizeof(int64_t) / sizeof(int32_t);
constexpr int32_t kBroadcastLanes = 8;
constexpr int32_t kGatherBlockY = 4;
constexpr int32_t kGatherKeyPattern = 1;
constexpr int32_t kGatherMemberPattern = 2;
constexpr int32_t kSort32Elements = 32;
constexpr int32_t kSort32MaxRepeats = 255;
constexpr int64_t kMaxSort32Records = kSort32Elements * kSort32MaxRepeats;
constexpr int32_t kMergeFanIn = 4;
// MrgSort's four source slots; the count of active inputs is compared with
// these zero-based positions before selecting a real source or slot zero.
enum MergeInputSlot : int32_t {
    kMergeFirstInput = 0,
    kMergeSecondInput = 1,
    kMergeThirdInput = 2,
    kMergeFourthInput = 3
};
constexpr int32_t kBinaryTreeArity = 2;
constexpr int32_t kMaxGroupWidth = 32;
constexpr int32_t kDefaultVectorLanes = 256;
constexpr int32_t kMaxResidentRuns = 128;
constexpr int64_t kMrgSortCountLimit = 65536; // The hardware merge length is uint16_t.
constexpr int32_t kTableAlignmentBlocks = 64;
constexpr int32_t kExtraPrefixTables = 3;
constexpr int32_t kScratchWordsPerLane = 16;
constexpr int32_t kFixedScratchBytes = 1024;
constexpr int64_t VectorScratchBytes(int64_t lanes)
{
    return lanes * (kScratchWordsPerLane * kWordBytes) + kFixedScratchBytes;
}
constexpr int64_t kMembershipSortScratchBytes = VectorScratchBytes(kDefaultVectorLanes) +
                                                kExtraPrefixTables * kDefaultVectorLanes * kWordBytes;

// Four-way cascade: 32 row leaves, 8 first-level nodes, 2 second-level nodes, root.
constexpr int32_t kStreamLevel1Begin = kMaxGroupWidth;
constexpr int32_t kStreamLevel1Count = kMaxGroupWidth / kMergeFanIn;
constexpr int32_t kStreamLevel2Begin = kStreamLevel1Begin + kStreamLevel1Count;
constexpr int32_t kStreamLevel2Count = kStreamLevel1Count / kMergeFanIn;
constexpr int32_t kStreamRoot = kStreamLevel2Begin + kStreamLevel2Count;
constexpr int32_t kStreamNodeCount = kStreamRoot + 1;
constexpr int32_t kNarrowCascadeMaxWidth = 16;
constexpr int32_t kStreamLevel2Depth = 2;
constexpr int32_t kStreamRootDepth = 3;
constexpr int32_t kNarrowCascadeLevels = kStreamLevel2Depth + 1;
constexpr int32_t kWideCascadeLevels = kStreamRootDepth + 1;

// Measured work coefficients, not hardware limits. Preserve the calibrated margin.
constexpr int64_t kSmallMembershipDomainBlocks = 128;
constexpr int64_t kMinBatchedEntries = 512;
constexpr int32_t kPreferredStreamWidth = 8;
constexpr int64_t kWorkComparisonScale = 10;
constexpr int64_t kPreferredWorkScale = 9;
constexpr int64_t kScalarEntryWork = 2;
constexpr int64_t kTableScanBlocksPerWork = 8;
constexpr int64_t kSmallTableRowWork = 8;
constexpr int64_t kLargeTableRowWork = 64;
constexpr int64_t kWriterBaseWork = 256;
constexpr int64_t kWriterWorkScale = 128;
constexpr int64_t kOutputDominanceRatio = 4;
constexpr int64_t kStreamingRunWork = 64;
constexpr int64_t kResidentRunWork = 32;
constexpr int64_t kVectorRowWork = 32;
constexpr int64_t kVectorTileWork = 6;
constexpr int64_t kStreamingWorkPenalty = 2;

// Resource policy and compaction layout. Keep equal-valued roles distinct.
constexpr int32_t kMinGroupWidth = 8;
constexpr int32_t kNarrowBlockY = 8;
constexpr int32_t kWideVectorLanes = 512;
constexpr int32_t kBudgetShrinkFactor = 2;
constexpr int32_t kSmallKMax = 128;
constexpr int32_t kCostInputCap = 512;
constexpr int32_t kDirectInputWindow = 128, kWideGroupInputWindow = 128;
constexpr int32_t kVectorInputWindow = 256;
constexpr int32_t kLargeOutputBytes = 32 * kKiB, kSmallOutputBytes = 16 * kKiB;
constexpr int32_t kCompilerReserveBytes = 16 * kKiB;
constexpr int32_t kCountBytes = kMaxGroupWidth * kWordBytes;
constexpr int32_t kMinOutputBytes = kKiB;
constexpr int32_t kMinMembershipBytes = 1280;
constexpr int32_t kTableScratchBytes = kKiB, kTableGuardBytes = 512;
constexpr int32_t kCompactionHalfVector = 2; // ID and int32 value vectors precede the half staging vector.
constexpr uint32_t kAllQueryBits = 0xFFFFFFFFU;

enum class Execution { Ordered, MembershipTable, MembershipAdaptive };

// Bound K before rounding/multiplying, including extreme legal storage strides.
#if defined(__CCE_AICORE__)
__aicore__
#endif
    constexpr bool
    MembershipSortFits(int64_t topK, int64_t width, int64_t residual, int64_t algorithmBytes)
{
    if (topK <= 0 || width <= 0 || residual < 0 || residual > 1 || topK > kMaxSort32Records - residual) {
        return false;
    }
    const int64_t stride = (topK + residual + kSort32Elements - 1) / kSort32Elements * kSort32Elements;
    return stride <= kMaxSort32Records && width <= kMaxSort32Records / stride &&
           algorithmBytes >= kMembershipSortScratchBytes &&
           stride <= (algorithmBytes - kMembershipSortScratchBytes) / width / kResidentRecordBytes;
}
#if defined(__CCE_AICORE__)
__aicore__
#endif
    constexpr int64_t
    MembershipSortStride(int64_t topK, int64_t residual)
{
    return (topK + residual + kSort32Elements - 1) / kSort32Elements * kSort32Elements;
}
} // namespace UnionSparseGenA2Plan
#endif
