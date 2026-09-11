/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_TILING_COMMON_H
#define KTH_VALUE_TILING_COMMON_H

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>

#include "graph/graph.h"
#include "graph/utils/type_utils.h"
#include "tiling/sort/sort_tiling_intf.h"
#include "util/math_util.h"

namespace gert {
class Shape;
class TilingContext;
} // namespace gert

namespace optiling {

// =============================================================================
// General constants
// =============================================================================
constexpr size_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BIN_NUM = 256;                      // 直方图一次处理256B
constexpr uint32_t SMALL_TILE_DATA_NUM = 1024;         // 测试数据得出一次至少处理1024，sort性能比较好
constexpr uint32_t SIMT_UB = 32768;                    // 预留了32k给simt使用
constexpr int64_t RADIX_UINT32_VALUE_MAX = 0x3fffffff; // uint32 radix counters reserve the top two bits for state
constexpr uint32_t SMALL_AXIS_THRESHOLD = 512;
constexpr int64_t NON_LAST_SMALL_AXIS_MIN_AXIS_LEN = 2;
constexpr int64_t NON_LAST_SMALL_AXIS_THRESHOLD = 2048;
constexpr int64_t ONE_CORE_DATA_SIZE = 2048;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t SORT32_SMALL_AXIS_THRESHOLD = 32;
constexpr uint32_t SMALL_AXIS_MAX_DATACOPY_BLOCK_COUNT = 4095; // DataCopy hardware limit for blockCount
constexpr uint32_t SORT_STRUCT_BYTES = 8;                      // fp32 sort struct size (index + value)

constexpr bool IsRadixUint32CounterRange(int64_t axisLen) { return axisLen <= RADIX_UINT32_VALUE_MAX; }
constexpr bool NeedsSignedZeroSourceOrder(ge::DataType dataType)
{
    return dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT;
}

// =============================================================================
// Safe arithmetic utilities
// =============================================================================
bool CeilAlignUint32(uint64_t rawSize, uint32_t alignSize, uint32_t& alignedSize);
bool CeilDivUint32(uint64_t value, uint64_t divisor, uint32_t& result);

uint64_t ComputeUbAfterSimtReserve(uint32_t ubSize);

// =============================================================================
// General utility functions
// =============================================================================
uint32_t GetPreferredInnerChunk(ge::DataType dataType, uint32_t index);

// =============================================================================
// Small-axis routing
// =============================================================================
constexpr uint32_t TWO_STAGE_RANK_INVERSE_MAX_N = 64;

enum class SmallAxisRouteKind : uint8_t { NONE = 0, INSERTION, TWO_STAGE };

struct SmallAxisRoutePlan {
    SmallAxisRouteKind kind = SmallAxisRouteKind::NONE;
    uint32_t batchSize = 0;
    uint32_t batchNum = 0;
    uint32_t blockDim = 0;
    uint32_t tmpUbSize = 0;
    bool useRankInverse = false;
    // A grouped batch contains only complete outer slices: batchSize == outerSlicesPerBatch * innerSize.
    bool groupOuterSlices = false;
};

struct SmallAxisRule {
    ge::DataType dtype;
    uint32_t insertionMaxN; // max sortAxisNum for INSERTION; 0 = unsupported
    uint32_t twoStageMaxN;  // max sortAxisNum for TWO_STAGE; 0 = unsupported
    // INSERTION concurrency tiers: {maxN, minSegs} pairs, terminated by {0, 0}.
    // A hit requires: N <= maxN && fullCoreSegs >= minSegs for some tier.
    uint32_t insertionTiers[4][2];
    // TWO_STAGE concurrency tiers: {maxN, minSegs} pairs, ascending maxN, terminated by {0, 0}.
    // Same lookup as INSERTION: find first tier where N <= maxN, then check fullCoreSegs >= minSegs.
    uint32_t twoStageTiers[4][2];
};

constexpr SmallAxisRule kSmallAxisRules[] = {
    // Small-axis thresholds are empirical performance breakpoints.
    // INSERTION is preferred only for very small N and enough full-core segments; otherwise its serial compare cost
    // grows quickly. TWO_STAGE covers wider N ranges, but each tier still requires enough segments to amortize
    // Sort API and SIMT scatter overhead. Each {maxN, minSegs} pair means this route is used only when
    // sortAxisNum <= maxN and each active core can receive at least minSegs segments.
    // dtype, insertionMaxN, twoStageMaxN, insertion tiers, two-stage tiers
    {ge::DT_INT64, 16, 512, {{8, 1}, {16, 4}, {0, 0}}, {{15, 8}, {128, 4}, {512, 8}, {0, 0}}},
    {ge::DT_UINT64, 16, 512, {{8, 1}, {16, 4}, {0, 0}}, {{15, 8}, {128, 4}, {512, 8}, {0, 0}}},
    {ge::DT_INT32, 11, 384, {{8, 2}, {11, 4}, {0, 0}}, {{11, 8}, {64, 4}, {384, 8}, {0, 0}}},
    {ge::DT_UINT32, 11, 384, {{8, 2}, {11, 4}, {0, 0}}, {{11, 8}, {64, 4}, {384, 8}, {0, 0}}},
    {ge::DT_INT16, 8, 192, {{4, 2}, {8, 4}, {0, 0}}, {{7, 8}, {64, 4}, {192, 12}, {0, 0}}},
    {ge::DT_UINT16, 8, 192, {{4, 2}, {8, 4}, {0, 0}}, {{7, 8}, {64, 4}, {192, 12}, {0, 0}}},
    {ge::DT_INT8, 8, 128, {{4, 2}, {8, 7}, {0, 0}}, {{3, 8}, {64, 7}, {128, 16}, {0, 0}}},
    {ge::DT_UINT8, 8, 128, {{4, 2}, {8, 7}, {0, 0}}, {{3, 8}, {64, 7}, {128, 16}, {0, 0}}},
    {ge::DT_FLOAT, 8, 24, {{4, 16}, {8, 48}, {0, 0}}, {{24, 64}, {0, 0}}},
    {ge::DT_BF16, 8, 54, {{4, 16}, {8, 48}, {0, 0}}, {{54, 64}, {0, 0}}},
    {ge::DT_FLOAT16, 8, 54, {{4, 16}, {8, 48}, {0, 0}}, {{54, 64}, {0, 0}}},
};
constexpr size_t kNumSmallAxisRules = sizeof(kSmallAxisRules) / sizeof(kSmallAxisRules[0]);

uint32_t LookupMinSegs(const uint32_t (*tiers)[2], uint32_t axisNum);
const SmallAxisRule* FindSmallAxisRule(ge::DataType dataType);
bool UseTwoStageRankInverse(uint32_t axisLen);
uint32_t ComputeInsertionBytesPerSeg(ge::DataType dataType, uint32_t axisLen, uint32_t dtypeSize,
                                     uint32_t indexDtypeSize, uint32_t blockUbSize);

// =============================================================================
// SortKthTileInfo — central tiling information struct
// =============================================================================
struct SortKthTileInfo {
    uint32_t coreNumNeed = 0;
    uint32_t lastDimTileNum = 0;
    uint32_t unsortedDimParallel = 1;
    uint32_t ubSize = 0;
    uint32_t blockUbSize = 0;
    uint32_t dtypeSize = 0;
    uint32_t y2DtypeSize = 0;
    // Optional kernel-side final-index width; zero keeps the output dtype width.
    uint32_t finalIdxUbElemBytes = 0;
    uint32_t maxCoreNum = 0;
    uint32_t numTileDataSize = 0;
    uint32_t sortLoopTimes = 0;
    uint32_t lastDimNeedCore = 0;
    uint32_t keyParams0 = 0;
    uint32_t keyParams1 = 0;
    uint32_t keyParams2 = 0;
    uint32_t keyParams3 = 0;
    uint32_t keyParams4 = 0;
    uint32_t keyParams5 = 0;
    uint32_t tmpUbSize = 0;
    bool isDescend = false;
    ge::DataType dataType = ge::DT_UINT8;
    uint32_t isInt32 = 0;
    int64_t rank = 0;
    int64_t sortAxis = 0;
    bool isNonLastAxis = false;
    int64_t lastAxis = 1;
    int64_t unsortedDim = 1;
    int64_t outerSize = 1;
    int64_t innerSize = 1;
    uint32_t innerLoopNum = 1;
    uint32_t innerChunk = 1;
    uint32_t inputRowBytes = 0;
    uint32_t valueAxisBytes = 0;
    uint32_t indexAxisBytes = 0;
    uint32_t outputIndexRowBytes = 0;
    uint32_t xUbSize = 0;
    uint32_t idxUbSize = 0;
    uint32_t outputRowsPerLoop = 0;
    uint64_t oneBufferQueSize = 0;
    uint32_t oneCoreTmpUbSize = 0;
    size_t workspaceSize = 0;
};

void ComputeAxisDimProducts(const gert::Shape& shape, int64_t axis, SortKthTileInfo& info);

// =============================================================================
// Non-last axis sort
// =============================================================================
constexpr uint32_t MERGE_SORT_MAX_AXIS_FP16 = 1024; // fp16 或者 bf16 走merge sort条件
constexpr uint32_t MERGE_SORT_MAX_AXIS_FP32 = 4096; // fp32 走merge sort条件

bool UseNonLastMergeSort(ge::DataType dataType, uint32_t axisLen);
ge::DataType GetNonLastSortDtype(ge::DataType dataType, bool useMergeSort);
uint32_t GetNonLastSortDtypeSize(uint32_t dtypeSize, bool useMergeSort, ge::DataType dataType);

uint32_t GetNonLastSortCount(ge::DataType dataType, uint32_t axisLen);
bool GetNonLastSortTmpSize(ge::DataType dataType, uint32_t sortCount, bool useMergeSort, bool isDescend,
                           uint32_t& tmpUbSize);
bool ComputeNonLastBatchNum(int64_t outerSize, int64_t innerSize, uint32_t innerChunk, uint32_t& batchNum);

struct NonLastSmallAxisCandidate {
    uint32_t innerChunk = 0;
    uint32_t innerLoopNum = 0;
    uint32_t activeCore = 0;
    uint64_t tileCount = 0;
    uint64_t peakUb = 0;
    uint32_t inputRowBytes = 0;
    uint32_t valueAxisBytes = 0;
    uint32_t indexAxisBytes = 0;
    uint32_t outputIndexRowBytes = 0;
};

// Search for the best innerChunk for non-last-axis small-axis sort.
// Non-last-axis sort processes a [outerSize, axisLen, innerSize] tensor by
// sorting axisLen elements at each (outer, inner) position independently.
// innerChunk controls how many adjacent inner positions are batched into one
// sort invocation, trading UB for fewer total tiles (tileCount).
// Candidate quality is evaluated with the following metrics.
//   tileCount = outerSize × ceil(innerSize / innerChunk)
//   activeCore = min(maxCoreNum, tileCount)
// Candidates are selected in the following priority order.
//   1. Maximise activeCore — better core utilisation
//   2. Tie-break: prefer larger innerChunk — fewer kernel invocations per core
// innerChunk candidates come from GetPreferredInnerChunk(), which returns
// dtype-dependent powers-of-2 in decreasing order (largest first).
bool SearchNonLastSmallAxisPlan(
    const SortKthTileInfo& info, uint64_t usableUb,
    std::function<bool(SortKthTileInfo&, uint32_t, uint64_t&, NonLastSmallAxisCandidate&)> estimateUb,
    NonLastSmallAxisCandidate& best, SortKthTileInfo* selectedInfo = nullptr);

bool SelectSmallAxisRoute(const SortKthTileInfo& info, SmallAxisRoutePlan& plan);
bool SelectNonLastSmallAxisRoute(const SortKthTileInfo& info, SmallAxisRoutePlan& plan);
bool SelectSortNonLastSmallAxisRoute(const SortKthTileInfo& info, SmallAxisRoutePlan& plan);

// =============================================================================
// Radix sort
// =============================================================================
struct RadixClearParams {
    uint32_t keyParams0 = 0;
    uint32_t keyParams1 = 0;
    uint32_t keyParams2 = 0;
    uint32_t keyParams3 = 0;
    uint32_t keyParams4 = 0;
    uint32_t keyParams5 = 0;
};

bool QuerySortTmpSizeRadix(ge::DataType dataType, uint32_t sortAxisNum, uint32_t& tmpUbSize);

uint32_t ComputeRadixRemainUb(uint32_t usableUb, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor);

void AdjustRadixTmpUb(uint32_t usableUb, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor, uint32_t blockUbSize,
                      uint32_t& tmpUbSize);
bool ComputeRadixTileDataForAllCore(int64_t axisLen, uint32_t maxCoreNum, uint32_t usableUb, uint32_t ubExtra,
                                    uint32_t tileFactor, uint32_t blockUbSize, uint32_t lastDimTileNum,
                                    uint32_t& tileData, uint32_t& tmpUbSize);
bool NeedAdjustRadixTileData(int64_t axisLen, int64_t unsortedDim, uint32_t maxCoreNum, uint32_t usableUb,
                             uint32_t ubExtra, uint32_t tileFactor, uint32_t blockUbSize, uint32_t& tileData,
                             uint32_t lastDimTileNum, uint32_t& tmpUbSize, bool& adjusted);
bool ComputeRadixTileData(int64_t axisLen, int64_t unsortedDim, uint32_t dtypeSize, uint32_t indexSize,
                          uint32_t maxCoreNum, uint32_t usableUb, uint32_t blockUbSize, uint32_t& tileData,
                          uint32_t& tmpUbSize);
bool FillRadixKernelParams(uint32_t dtypeSize, uint32_t indexSize, uint32_t coreNumNeed, uint32_t lastDimTileNum,
                           uint32_t unsortedDimParallel, uint32_t blockUbSize, uint32_t tmpUbSize,
                           RadixClearParams& out);
bool ComputeRadixSortWorkspace(int64_t axisLen, uint32_t dtypeSize, uint32_t indexSize, uint32_t lastDimTileNum,
                               uint32_t numTileDataSize, uint32_t unsortedDimParallel, uint32_t keyParams0,
                               uint32_t keyParams1, uint32_t keyParams2, uint32_t keyParams3, uint32_t keyParams4,
                               uint32_t blockUbSize, uint64_t& workspaceSize);
bool ComputeRadixOneCoreUbSizes(int64_t lastAxis, uint32_t dtypeSize, uint32_t indexElemSize, uint32_t blockUbSize,
                                uint32_t& xUbSize, uint32_t& idxUbSize);
bool FillRadixMoreCoreInfo(SortKthTileInfo& info);

// =============================================================================
// Merge sort — common
// =============================================================================
constexpr uint32_t MERGE_SORT_LIST_NUM = 4;
constexpr uint32_t MERGE_SORT_DATA_BYTES = 8;
constexpr int64_t MERGE_SORT_WORKSPACE_PARAM = 5;
constexpr uint32_t MULTI_CORE_MERGE_SORT_MAX_AXIS = 32768;
constexpr uint32_t MERGE_INTRA_CORE_SORT_ALIGN = 32; // Sort/Extract API alignment requirement (elements)
// Beyond 4 merge rounds (>256 blocks), radix sort has little performance disadvantage.
constexpr uint32_t MERGE_INTRA_CORE_MAX_BLOCKS = 256;

bool IsMergeSortSupported(ge::DataType dataType, int64_t axisLen);

struct MergeSortPlan {
    uint32_t alignNum = 0;
    uint32_t oneCoreRowNum = 0;
    uint32_t sortLoopTimes = 0;
    uint32_t coreNumNeed = 0;
};

bool ComputeMergeSortPlan(int64_t axisLen, int64_t unsortedDim, uint32_t blockUbSize, uint32_t tileDataNum,
                          uint32_t maxCoreNum, MergeSortPlan& plan);
bool FillMergeSortInfo(SortKthTileInfo& info, uint32_t indexDtypeSize, uint32_t concatTmpSize);

// =============================================================================
// Merge sort — multi-core (more-core)
// =============================================================================
struct MergeMoreCorePlan {
    uint32_t lastDimTileNum = 0;
    uint32_t lastDimNeedCore = 0;
    uint32_t numTileDataSize = 0;
    uint32_t unsortedDimParallel = 0;
    uint32_t sortLoopTimes = 1;
    uint32_t coreNumNeed = 0;
    uint32_t keyParams0 = 0;
};

bool IsMergeMoreCoreSupported(ge::DataType dataType, int64_t axisLen, int64_t unsortedDim, uint32_t maxCoreNum);
bool ComputeMergeMoreCorePlan(int64_t axisLen, int64_t unsortedDim, uint32_t ubSize, uint32_t mergeBytesPerElem,
                              MergeMoreCorePlan& plan);
bool FillMergeMoreCoreInfo(SortKthTileInfo& info, uint32_t mergeBytesPerElem);

// =============================================================================
// Merge sort — intra-core
// =============================================================================
uint32_t ComputeMergeIntraCoreBlockSortSize(uint32_t ubSize);
uint32_t ComputeMergeIntraCoreExtractChunkSize(uint32_t ubSize);

struct MergeIntraCorePlan {
    uint32_t batchPerCore = 0;
    uint32_t actualCoreNum = 0;
    uint32_t blockSortSize = 0;
    uint32_t extractChunkSize = 0;
    uint32_t blocksPerRow = 0;
    uint32_t alignNum = 0;
};

bool IsMergeIntraCoreSupported(ge::DataType dataType, int64_t axisLen, int64_t unsortedDim, uint32_t maxCoreNum,
                               uint32_t ubSize);
bool ComputeMergeIntraCorePlan(int64_t axisLen, int64_t unsortedDim, uint32_t ubSize, uint32_t maxCoreNum,
                               MergeIntraCorePlan& plan);
bool FillMergeIntraCoreInfo(SortKthTileInfo& info);

// =============================================================================
// Two-stage sort
// =============================================================================
uint32_t MaxTwoStageU16SafeBatch(uint32_t axisLen);
bool ComputeTwoStageSortTmpUb(ge::DataType dataType, uint32_t axisLen, uint32_t totalElems, uint32_t blockUbSize,
                              uint32_t& tmpUbSize);
uint64_t EstimateTwoStageUbBytes(const SortKthTileInfo& info, uint32_t totalElems, uint32_t sortTmpUb);
bool PrepareTwoStageBatchCandidate(const SortKthTileInfo& info, uint32_t candidate, uint32_t& totalElems,
                                   uint32_t& tmpUbSize, bool& useRankInverse, uint64_t& totalBytes);

struct TwoStageBatchPlan {
    uint32_t batchSize = 0;
    uint32_t batchNum = 0;
    uint32_t blockDim = 0;
    uint32_t tmpUbSize = 0;
};

bool SearchTwoStageBatchPlan(uint32_t maxBatch, std::function<bool(uint32_t, TwoStageBatchPlan&)> tryCandidate,
                             TwoStageBatchPlan& result);

// =============================================================================
// Tiling data conversion & top-level tiling computation
// =============================================================================
template <typename TilingData>
void PlanToTilingData(const SortKthTileInfo& info, TilingData* tiling)
{
    tiling->numTileDataSize = info.numTileDataSize;
    tiling->unsortedDimParallel = info.unsortedDimParallel;
    tiling->lastDimTileNum = info.lastDimTileNum;
    tiling->sortLoopTimes = info.sortLoopTimes;
    tiling->lastDimNeedCore = info.lastDimNeedCore;
    tiling->keyParams0 = info.keyParams0;
    tiling->keyParams1 = info.keyParams1;
    tiling->keyParams2 = info.keyParams2;
    tiling->keyParams3 = info.keyParams3;
    tiling->keyParams4 = info.keyParams4;
    tiling->keyParams5 = info.keyParams5;
    tiling->tmpUbSize = info.tmpUbSize;
    tiling->lastAxisNum = info.lastAxis;
    tiling->unsortedDimNum = info.unsortedDim;
    tiling->outerSize = info.outerSize;
    tiling->innerSize = info.innerSize;
    tiling->innerLoopNum = info.innerLoopNum;
    tiling->innerChunk = info.innerChunk;
    tiling->inputRowBytes = info.inputRowBytes;
    tiling->valueAxisBytes = info.valueAxisBytes;
    tiling->indexAxisBytes = info.indexAxisBytes;
}

bool ComputeMergeSortTiling(gert::TilingContext* context, SortKthTileInfo& info, uint32_t indexDtypeSize);
bool ComputeMergeMoreCoreTiling(gert::TilingContext* context, SortKthTileInfo& info, uint32_t mergeBytesPerElem);
bool ComputeMergeIntraCoreTiling(gert::TilingContext* context, SortKthTileInfo& info);

} // namespace optiling

#endif // KTH_VALUE_TILING_COMMON_H
