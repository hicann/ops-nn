/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kth_value_tiling_common.h"

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "exe_graph/runtime/tiling_context.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "utils/extern_math_util.h"

namespace optiling {

// =============================================================================
// Safe arithmetic utilities
// =============================================================================
bool CeilAlignUint32(uint64_t rawSize, uint32_t alignSize, uint32_t& alignedSize)
{
    uint64_t result = Ops::Base::CeilAlign(rawSize, static_cast<uint64_t>(alignSize));
    if (result > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    alignedSize = static_cast<uint32_t>(result);
    return true;
}

bool CeilDivUint32(uint64_t value, uint64_t divisor, uint32_t& result)
{
    uint64_t quotient = Ops::Base::CeilDiv(value, divisor);
    if (quotient > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    result = static_cast<uint32_t>(quotient);
    return true;
}

uint64_t ComputeUbAfterSimtReserve(uint32_t ubSize)
{
    uint64_t usableUb = static_cast<uint64_t>(ubSize);
    if (usableUb <= static_cast<uint64_t>(SIMT_UB)) {
        return 0;
    }
    usableUb -= static_cast<uint64_t>(SIMT_UB);
    return usableUb;
}

// =============================================================================
// General utility functions
// =============================================================================

// Return the index-th preferred innerChunk candidate for a given dataType.
// Candidates are powers-of-2 in decreasing order (largest first). Larger dtypes
// get smaller max chunks because each inner position consumes more UB.
// index=0 → largest candidate, increasing index → smaller candidates.
// Returns 0 when index is out of range or dataType is unsupported.
uint32_t GetPreferredInnerChunk(ge::DataType dataType, uint32_t index)
{
    constexpr uint32_t kMaxChunkCandidates = 6;
    constexpr uint32_t kOneByteChunkGroup = 3;
    // Each row: dtype group's chunk candidates (decreasing powers of 2, padded with 0).
    static constexpr uint32_t kChunkCandidates[][kMaxChunkCandidates] = {
        {4, 2, 1, 0, 0, 0},
        {8, 4, 2, 1, 0, 0},
        {16, 8, 4, 2, 1, 0},
        {32, 16, 8, 4, 2, 1},
    };
    static constexpr uint32_t kChunkValidCount[] = {3, 4, 5, 6};
    uint32_t group = 0;
    if (dataType == ge::DT_INT64 || dataType == ge::DT_UINT64) {
        group = 0; // 8-byte types
    } else if (dataType == ge::DT_FLOAT || dataType == ge::DT_INT32 || dataType == ge::DT_UINT32) {
        group = 1; // 4-byte types
    } else if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16 || dataType == ge::DT_INT16 ||
               dataType == ge::DT_UINT16) {
        group = 2; // 2-byte types
    } else if (dataType == ge::DT_INT8 || dataType == ge::DT_UINT8) {
        group = kOneByteChunkGroup;
    } else {
        return 0;
    }
    return index < kChunkValidCount[group] ? kChunkCandidates[group][index] : 0;
}

// =============================================================================
// Small-axis routing
// =============================================================================
uint32_t LookupMinSegs(const uint32_t (*tiers)[2], uint32_t axisNum)
{
    constexpr uint32_t kMaxTiers = 4;
    for (uint32_t i = 0; i < kMaxTiers && tiers[i][0] != 0; ++i) {
        if (axisNum <= tiers[i][0])
            return tiers[i][1];
    }
    return UINT32_MAX;
}

const SmallAxisRule* FindSmallAxisRule(ge::DataType dataType)
{
    for (size_t i = 0; i < kNumSmallAxisRules; ++i) {
        if (kSmallAxisRules[i].dtype == dataType) {
            return &kSmallAxisRules[i];
        }
    }
    return nullptr;
}

bool UseTwoStageRankInverse(uint32_t axisLen) { return axisLen <= TWO_STAGE_RANK_INVERSE_MAX_N; }

static bool ComputeBf16InsertionBytesPerSeg(uint32_t axisLen, uint64_t idxBytes, uint32_t blockUbSize,
                                            uint64_t& bytesPerSeg)
{
    uint64_t castRawBytes = 0U;
    if (ge::MulOverflow(axisLen, sizeof(int16_t), castRawBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeInsertionBytesPerSeg", "axisLen", std::to_string(axisLen).c_str(),
                                              "The value of axisLen must not cause cast raw byte size overflow.");
        return false;
    }
    uint64_t castBytes = Ops::Base::CeilAlign<uint64_t>(castRawBytes, blockUbSize);
    if (castBytes == 0) {
        return false;
    }
    uint64_t castRowElems = castBytes / sizeof(int16_t);
    uint64_t valueBytes = 0U;
    if (ge::MulOverflow(castRowElems, sizeof(float), valueBytes) ||
        ge::AddOverflow(valueBytes, idxBytes, bytesPerSeg) || ge::AddOverflow(bytesPerSeg, castBytes, bytesPerSeg)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "ComputeInsertionBytesPerSeg", "bytesPerSeg",
            (std::to_string(castRowElems) + ", " + std::to_string(idxBytes) + ", " + std::to_string(castBytes)).c_str(),
            "The value of bf16 bytesPerSeg must not overflow.");
        return false;
    }
    return true;
}

uint32_t ComputeInsertionBytesPerSeg(ge::DataType dataType, uint32_t axisLen, uint32_t dtypeSize,
                                     uint32_t indexDtypeSize, uint32_t blockUbSize)
{
    uint64_t valueRawBytes = 0U;
    uint64_t idxRawBytes = 0U;
    if (ge::MulOverflow(axisLen, dtypeSize, valueRawBytes) || ge::MulOverflow(axisLen, indexDtypeSize, idxRawBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeInsertionBytesPerSeg", "axisLen", std::to_string(axisLen).c_str(),
                                              "The value of axisLen must not cause raw byte size overflow.");
        return 0;
    }
    uint64_t valueBytes = Ops::Base::CeilAlign<uint64_t>(valueRawBytes, blockUbSize);
    uint64_t idxBytes = Ops::Base::CeilAlign<uint64_t>(idxRawBytes, blockUbSize);
    if (valueBytes == 0 || idxBytes == 0) {
        return 0;
    }
    uint64_t bytesPerSeg = 0U;
    if (ge::AddOverflow(valueBytes, idxBytes, bytesPerSeg)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeInsertionBytesPerSeg", "bytesPerSeg",
                                              (std::to_string(valueBytes) + ", " + std::to_string(idxBytes)).c_str(),
                                              "The value of valueBytes plus idxBytes must not overflow.");
        return 0;
    }
    if (dataType == ge::DT_BF16 && !ComputeBf16InsertionBytesPerSeg(axisLen, idxBytes, blockUbSize, bytesPerSeg)) {
        return 0;
    }
    if (bytesPerSeg > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeInsertionBytesPerSeg", "bytesPerSeg",
                                              std::to_string(bytesPerSeg).c_str(),
                                              "The value of bytesPerSeg must be less than or equal to uint32 max.");
        return 0;
    }
    return static_cast<uint32_t>(bytesPerSeg);
}

// =============================================================================
// Non-last axis sort
// =============================================================================
bool UseNonLastMergeSort(ge::DataType dataType, uint32_t axisLen)
{
    return dataType == ge::DT_FLOAT ||
           ((dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) && axisLen <= MERGE_SORT_MAX_AXIS_FP16);
}

ge::DataType GetNonLastSortDtype(ge::DataType dataType, bool useMergeSort)
{
    return useMergeSort && dataType == ge::DT_BF16 ? ge::DT_FLOAT : dataType;
}

uint32_t GetNonLastSortDtypeSize(uint32_t dtypeSize, bool useMergeSort, ge::DataType dataType)
{
    return useMergeSort && dataType == ge::DT_BF16 ? static_cast<uint32_t>(sizeof(float)) : dtypeSize;
}

uint32_t GetNonLastSortCount(ge::DataType dataType, uint32_t axisLen)
{
    if (UseNonLastMergeSort(dataType, axisLen)) {
        return Ops::Base::CeilAlign<uint32_t>(axisLen, MERGE_INTRA_CORE_SORT_ALIGN);
    }
    return axisLen;
}

void ComputeAxisDimProducts(const gert::Shape& shape, int64_t axis, SortKthTileInfo& info)
{
    int64_t rank = shape.GetDimNum();
    if (axis < 0 || axis >= rank) {
        return;
    }
    int64_t outerSize = 1;
    int64_t innerSize = 1;
    for (int64_t i = 0; i < rank; ++i) {
        int64_t dimSize = shape.GetDim(i);
        if (i < axis) {
            outerSize *= dimSize;
        } else if (i > axis) {
            innerSize *= dimSize;
        }
    }
    info.outerSize = outerSize;
    info.innerSize = innerSize;
    info.lastAxis = shape.GetDim(axis);
    info.unsortedDim = outerSize * innerSize;
}

bool ComputeNonLastBatchNum(int64_t outerSize, int64_t innerSize, uint32_t innerChunk, uint32_t& batchNum)
{
    if (outerSize <= 0 || innerSize <= 0 || innerChunk == 0U) {
        return false;
    }
    uint32_t innerLoop = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(innerSize), static_cast<uint64_t>(innerChunk), innerLoop)) {
        return false;
    }
    uint64_t batchNum64 = static_cast<uint64_t>(outerSize) * static_cast<uint64_t>(innerLoop);
    if (batchNum64 == 0U || batchNum64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    batchNum = static_cast<uint32_t>(batchNum64);
    return true;
}

bool IsMergeSortSupported(ge::DataType dataType, int64_t axisLen)
{
    return (((dataType == ge::DT_FLOAT16) || (dataType == ge::DT_BF16)) &&
            (axisLen <= static_cast<int64_t>(MERGE_SORT_MAX_AXIS_FP16))) ||
           ((dataType == ge::DT_FLOAT) && (axisLen <= static_cast<int64_t>(MERGE_SORT_MAX_AXIS_FP32)));
}

bool GetNonLastSortTmpSize(ge::DataType dataType, uint32_t sortCount, bool useMergeSort, bool isDescend,
                           uint32_t& tmpUbSize)
{
    std::vector<int64_t> shapeVec = {static_cast<int64_t>(sortCount)};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = useMergeSort ? AscendC::SortType::MERGE_SORT : AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    // Non-last-axis kernels use Sort's implicit-index overload for both radix
    // and merge paths. sourceOrder is consumed only after floating radix Sort.
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, GetNonLastSortDtype(dataType, useMergeSort), ge::DT_UINT32, true, config,
                                  maxValue, minValue);
    tmpUbSize = maxValue;
    return maxValue > 0;
}

bool SearchNonLastSmallAxisPlan(
    const SortKthTileInfo& info, uint64_t usableUb,
    std::function<bool(SortKthTileInfo&, uint32_t, uint64_t&, NonLastSmallAxisCandidate&)> estimateUb,
    NonLastSmallAxisCandidate& best, SortKthTileInfo* selectedInfo)
{
    constexpr uint32_t kMaxChunkCandidates = 6;
    constexpr uint32_t kWideFp32Sort32Chunk = 32U;
    // Sort and KthValue share this layout. For FP32 Sort32, prepend 32/16 to the usual 8..1 candidates;
    // estimateUb below still validates each candidate's aligned UB footprint and rejects one that does not fit.
    bool useWideFp32Sort32 = info.dataType == ge::DT_FLOAT && info.lastAxis <= SORT32_SMALL_AXIS_THRESHOLD;
    for (uint32_t i = 0; i < kMaxChunkCandidates; ++i) {
        uint32_t chunk = useWideFp32Sort32 ? (kWideFp32Sort32Chunk >> i) : GetPreferredInnerChunk(info.dataType, i);
        if (chunk == 0U) {
            break;
        }
        chunk = static_cast<uint32_t>(std::min<uint64_t>(chunk, static_cast<uint64_t>(info.innerSize)));
        if (chunk == 0U) {
            return false;
        }
        uint64_t innerLoopNum64 = (static_cast<uint64_t>(info.innerSize) + chunk - 1U) / chunk;
        if (innerLoopNum64 == 0U || innerLoopNum64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            continue;
        }
        NonLastSmallAxisCandidate cur;
        cur.innerChunk = chunk;
        cur.innerLoopNum = static_cast<uint32_t>(innerLoopNum64);
        cur.tileCount = static_cast<uint64_t>(info.outerSize) * innerLoopNum64;
        if (cur.tileCount > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            continue;
        }
        SortKthTileInfo candidateInfo = info;
        if (!estimateUb(candidateInfo, chunk, cur.peakUb, cur) || cur.peakUb > usableUb) {
            continue;
        }
        cur.activeCore = static_cast<uint32_t>(
            std::min<uint64_t>(static_cast<uint64_t>(info.maxCoreNum), cur.tileCount));
        bool betterCoreUse = cur.activeCore > best.activeCore;
        bool sameCoreUseLargerChunk = cur.activeCore == best.activeCore && cur.innerChunk > best.innerChunk;
        if (betterCoreUse || sameCoreUseLargerChunk) {
            best = cur;
            if (selectedInfo != nullptr) {
                *selectedInfo = candidateInfo;
            }
        }
    }
    return best.innerChunk != 0U && best.tileCount != 0U && best.activeCore != 0U;
}

// =============================================================================
// Radix sort
// =============================================================================
uint32_t ComputeRadixRemainUb(uint32_t usableUb, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor)
{
    uint64_t usedUb = static_cast<uint64_t>(ubExtra) + static_cast<uint64_t>(tileFactor) * tileData;
    if (usedUb >= static_cast<uint64_t>(usableUb)) {
        return 0;
    }
    return usableUb - static_cast<uint32_t>(usedUb);
}

void AdjustRadixTmpUb(uint32_t usableUb, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor, uint32_t blockUbSize,
                      uint32_t& tmpUbSize)
{
    if (blockUbSize == 0U) {
        return;
    }
    uint32_t remainUb = ComputeRadixRemainUb(usableUb, tileData, ubExtra, tileFactor);
    remainUb = remainUb > tmpUbSize ? remainUb - tmpUbSize : 0U;
    remainUb = remainUb > blockUbSize ? remainUb - blockUbSize : 0U;
    tmpUbSize += (remainUb / blockUbSize) * blockUbSize;
}

bool ComputeRadixTileDataForAllCore(int64_t axisLen, uint32_t maxCoreNum, uint32_t usableUb, uint32_t ubExtra,
                                    uint32_t tileFactor, uint32_t blockUbSize, uint32_t lastDimTileNum,
                                    uint32_t& tileData, uint32_t& tmpUbSize)
{
    if (axisLen <= 0 || maxCoreNum == 0U || lastDimTileNum == 0U) {
        return false;
    }
    uint32_t allCoreQuotient = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(lastDimTileNum), static_cast<uint64_t>(maxCoreNum), allCoreQuotient)) {
        return false;
    }
    uint64_t allCore = static_cast<uint64_t>(allCoreQuotient) * static_cast<uint64_t>(maxCoreNum);
    uint32_t newTileData = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(axisLen), allCore, newTileData) ||
        !CeilAlignUint32(newTileData, BIN_NUM, tileData)) {
        return false;
    }
    tileData = std::max(tileData, SMALL_TILE_DATA_NUM);
    if (!QuerySortTmpSizeRadix(ge::DT_UINT8, tileData, tmpUbSize)) {
        return false;
    }
    AdjustRadixTmpUb(usableUb, tileData, ubExtra, tileFactor, blockUbSize, tmpUbSize);
    return true;
}

bool QuerySortTmpSizeRadix(ge::DataType dataType, uint32_t sortAxisNum, uint32_t& tmpUbSize)
{
    std::vector<int64_t> shapeVec = {static_cast<int64_t>(sortAxisNum)};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    tmpUbSize = maxValue;
    return maxValue > 0;
}

static bool AdjustSingleRowSingleTile(int64_t axisLen, uint32_t maxCoreNum, uint32_t usableUb, uint32_t ubExtra,
                                      uint32_t tileFactor, uint32_t blockUbSize, uint32_t& tileData,
                                      uint32_t& tmpUbSize, bool& adjusted)
{
    uint32_t newTileData = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(axisLen), static_cast<uint64_t>(maxCoreNum), newTileData) ||
        !CeilAlignUint32(newTileData, BIN_NUM, newTileData)) {
        return false;
    }
    tileData = std::max(newTileData, SMALL_TILE_DATA_NUM);
    if (!QuerySortTmpSizeRadix(ge::DT_UINT8, tileData, tmpUbSize)) {
        return false;
    }
    AdjustRadixTmpUb(usableUb, tileData, ubExtra, tileFactor, blockUbSize, tmpUbSize);
    adjusted = true;
    return true;
}

static bool AdjustBSharedSingleHTile(int64_t axisLen, int64_t unsortedDim, uint32_t maxCoreNum, uint32_t usableUb,
                                     uint32_t ubExtra, uint32_t tileFactor, uint32_t blockUbSize, uint32_t& tileData,
                                     uint32_t& tmpUbSize, bool& adjusted)
{
    uint32_t hCore = maxCoreNum / static_cast<uint32_t>(unsortedDim);
    if (hCore == 0U) {
        return false;
    }
    uint64_t hTileData64 = static_cast<uint64_t>(axisLen) / hCore;
    if (hTileData64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    if (!CeilAlignUint32(hTileData64, BIN_NUM, tileData) || !QuerySortTmpSizeRadix(ge::DT_UINT8, tileData, tmpUbSize)) {
        return false;
    }
    AdjustRadixTmpUb(usableUb, tileData, ubExtra, tileFactor, blockUbSize, tmpUbSize);
    adjusted = true;
    return true;
}

static bool AdjustMultiTileHWithBSharing(int64_t axisLen, int64_t unsortedDim, uint32_t maxCoreNum, uint32_t usableUb,
                                         uint32_t ubExtra, uint32_t tileFactor, uint32_t blockUbSize,
                                         uint32_t& tileData, uint32_t lastDimTileNum, uint32_t& tmpUbSize,
                                         bool& adjusted)
{
    uint64_t newTileData64 = static_cast<uint64_t>(axisLen) / static_cast<uint64_t>(lastDimTileNum);
    if (!CeilAlignUint32(newTileData64, BIN_NUM, tileData)) {
        return false;
    }
    uint32_t adjustedLastDimTileNum = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(axisLen), static_cast<uint64_t>(tileData), adjustedLastDimTileNum)) {
        return false;
    }
    uint32_t bCore = adjustedLastDimTileNum == 0U ? maxCoreNum : maxCoreNum / adjustedLastDimTileNum;
    if (adjustedLastDimTileNum < maxCoreNum && unsortedDim < static_cast<int64_t>(maxCoreNum) &&
        unsortedDim < static_cast<int64_t>(bCore)) {
        bCore = static_cast<uint32_t>(unsortedDim);
        uint32_t hCore = maxCoreNum / bCore;
        uint32_t tileDataNew = 0;
        if (hCore == 0U || !CeilDivUint32(static_cast<uint64_t>(axisLen), static_cast<uint64_t>(hCore), tileDataNew) ||
            !CeilAlignUint32(tileDataNew, BIN_NUM, tileData)) {
            return false;
        }
    }
    if (bCore == 1U && adjustedLastDimTileNum < maxCoreNum) {
        if (!ComputeRadixTileDataForAllCore(axisLen, maxCoreNum, usableUb, ubExtra, tileFactor, blockUbSize,
                                            adjustedLastDimTileNum, tileData, tmpUbSize)) {
            return false;
        }
        adjusted = true;
        return true;
    }
    if (!QuerySortTmpSizeRadix(ge::DT_UINT8, tileData, tmpUbSize)) {
        return false;
    }
    AdjustRadixTmpUb(usableUb, tileData, ubExtra, tileFactor, blockUbSize, tmpUbSize);
    adjusted = true;
    return true;
}

bool NeedAdjustRadixTileData(int64_t axisLen, int64_t unsortedDim, uint32_t maxCoreNum, uint32_t usableUb,
                             uint32_t ubExtra, uint32_t tileFactor, uint32_t blockUbSize, uint32_t& tileData,
                             uint32_t lastDimTileNum, uint32_t& tmpUbSize, bool& adjusted)
{
    adjusted = false;
    if (axisLen <= 0 || maxCoreNum == 0U) {
        return false;
    }
    if (unsortedDim == static_cast<int64_t>(1) && lastDimTileNum == 1U) {
        return AdjustSingleRowSingleTile(axisLen, maxCoreNum, usableUb, ubExtra, tileFactor, blockUbSize, tileData,
                                         tmpUbSize, adjusted);
    }
    if (unsortedDim == static_cast<int64_t>(1) || lastDimTileNum >= maxCoreNum) {
        if (!ComputeRadixTileDataForAllCore(axisLen, maxCoreNum, usableUb, ubExtra, tileFactor, blockUbSize,
                                            lastDimTileNum, tileData, tmpUbSize)) {
            return false;
        }
        adjusted = true;
        return true;
    }
    if (unsortedDim > static_cast<int64_t>(1) && unsortedDim < static_cast<int64_t>(maxCoreNum) &&
        lastDimTileNum == 1U) {
        return AdjustBSharedSingleHTile(axisLen, unsortedDim, maxCoreNum, usableUb, ubExtra, tileFactor, blockUbSize,
                                        tileData, tmpUbSize, adjusted);
    }
    if (unsortedDim > static_cast<int64_t>(1) && lastDimTileNum > 1U) {
        return AdjustMultiTileHWithBSharing(axisLen, unsortedDim, maxCoreNum, usableUb, ubExtra, tileFactor,
                                            blockUbSize, tileData, lastDimTileNum, tmpUbSize, adjusted);
    }
    return true;
}

bool ComputeRadixTileData(int64_t axisLen, int64_t unsortedDim, uint32_t dtypeSize, uint32_t indexSize,
                          uint32_t maxCoreNum, uint32_t usableUb, uint32_t blockUbSize, uint32_t& tileData,
                          uint32_t& tmpUbSize)
{
    if (maxCoreNum == 0U) {
        return false;
    }
    uint32_t ubExtra = BIN_NUM * (indexSize + static_cast<uint32_t>(sizeof(uint16_t)) +
                                  static_cast<uint32_t>(sizeof(uint16_t)) + indexSize + indexSize);
    uint32_t tileFactor = dtypeSize + indexSize + static_cast<uint32_t>(sizeof(uint32_t)) +
                          static_cast<uint32_t>(sizeof(uint8_t)) + static_cast<uint32_t>(sizeof(uint8_t));
    if (usableUb <= ubExtra || tileFactor == 0U) {
        return false;
    }
    tileData = ((usableUb - ubExtra) / tileFactor) / BIN_NUM * BIN_NUM;
    if (tileData == 0U) {
        return false;
    }
    uint32_t remainUb = ComputeRadixRemainUb(usableUb, tileData, ubExtra, tileFactor);
    if (!QuerySortTmpSizeRadix(ge::DT_UINT8, tileData, tmpUbSize)) {
        return false;
    }
    while (tmpUbSize > remainUb) {
        if (tileData <= BIN_NUM) {
            return false;
        }
        tileData -= BIN_NUM;
        remainUb = ComputeRadixRemainUb(usableUb, tileData, ubExtra, tileFactor);
        if (!QuerySortTmpSizeRadix(ge::DT_UINT8, tileData, tmpUbSize)) {
            return false;
        }
    }
    uint32_t lastDimTileNum = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(axisLen), static_cast<uint64_t>(tileData), lastDimTileNum)) {
        return false;
    }
    bool smallTile = axisLen <= static_cast<int64_t>(SMALL_TILE_DATA_NUM) && lastDimTileNum == 1U;
    bool adjusted = false;
    if ((lastDimTileNum % maxCoreNum == 0U) || smallTile) {
        AdjustRadixTmpUb(usableUb, tileData, ubExtra, tileFactor, blockUbSize, tmpUbSize);
    } else if (!NeedAdjustRadixTileData(axisLen, unsortedDim, maxCoreNum, usableUb, ubExtra, tileFactor, blockUbSize,
                                        tileData, lastDimTileNum, tmpUbSize, adjusted)) {
        return false;
    } else if (!adjusted) {
        AdjustRadixTmpUb(usableUb, tileData, ubExtra, tileFactor, blockUbSize, tmpUbSize);
    }
    return true;
}

bool FillRadixKernelParams(uint32_t dtypeSize, uint32_t indexSize, uint32_t coreNumNeed, uint32_t lastDimTileNum,
                           uint32_t unsortedDimParallel, uint32_t blockUbSize, uint32_t tmpUbSize,
                           RadixClearParams& out)
{
    if (indexSize == 0U) {
        return false;
    }
    uint32_t ubSizeNum = tmpUbSize / indexSize;
    if (ubSizeNum == 0U) {
        return false;
    }
    uint64_t allNumGlobalHist64 = static_cast<uint64_t>(BIN_NUM) * lastDimTileNum * dtypeSize * unsortedDimParallel;
    uint64_t allNumExcusiveBin64 = static_cast<uint64_t>(BIN_NUM) * dtypeSize * unsortedDimParallel;
    if (allNumGlobalHist64 > std::numeric_limits<uint32_t>::max() ||
        allNumExcusiveBin64 > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    uint32_t allNumGlobalHist = static_cast<uint32_t>(allNumGlobalHist64);
    uint32_t allNumExcusiveBin = static_cast<uint32_t>(allNumExcusiveBin64);
    uint32_t oneCoreSize = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(allNumGlobalHist), static_cast<uint64_t>(coreNumNeed)));
    out.keyParams5 = std::max(oneCoreSize, blockUbSize);
    out.keyParams0 = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(allNumGlobalHist), static_cast<uint64_t>(out.keyParams5)));
    out.keyParams3 = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(out.keyParams5), static_cast<uint64_t>(ubSizeNum)));
    out.keyParams2 = out.keyParams5 > ubSizeNum ? ubSizeNum : out.keyParams5;
    uint32_t oneCoreSize1 = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(allNumExcusiveBin), static_cast<uint64_t>(coreNumNeed)));
    out.keyParams4 = std::max(oneCoreSize1, blockUbSize);
    out.keyParams1 = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(allNumExcusiveBin), static_cast<uint64_t>(out.keyParams4)));
    return true;
}

bool ComputeRadixSortWorkspace(int64_t axisLen, uint32_t dtypeSize, uint32_t indexSize, uint32_t lastDimTileNum,
                               uint32_t numTileDataSize, uint32_t unsortedDimParallel, uint32_t keyParams0,
                               uint32_t keyParams1, uint32_t keyParams2, uint32_t keyParams3, uint32_t keyParams4,
                               uint32_t blockUbSize, uint64_t& workspaceSize)
{
    uint64_t indexSize64 = static_cast<uint64_t>(indexSize);
    uint64_t blockUbSize64 = static_cast<uint64_t>(blockUbSize);
    uint64_t unsortedDimParallel64 = static_cast<uint64_t>(unsortedDimParallel);
    uint64_t axisLen64 = static_cast<uint64_t>(axisLen);

    uint64_t excusiveBins = Ops::Base::CeilAlign(static_cast<uint64_t>(keyParams1) * keyParams4 * indexSize64,
                                                 blockUbSize64);
    uint64_t globalHist = Ops::Base::CeilAlign(
        static_cast<uint64_t>(keyParams3) * keyParams2 * keyParams0 * indexSize64, blockUbSize64);
    uint64_t sortedIdx = Ops::Base::CeilAlign(axisLen64 * unsortedDimParallel64 * indexSize64, blockUbSize64);
    uint64_t histTile = static_cast<uint64_t>(lastDimTileNum) * BIN_NUM * unsortedDimParallel64 * sizeof(uint16_t) * 2U;
    uint64_t xB8 = Ops::Base::CeilAlign(static_cast<uint64_t>(lastDimTileNum) * numTileDataSize * unsortedDimParallel64,
                                        blockUbSize64);
    uint64_t sortedValue = Ops::Base::CeilAlign(axisLen64 * unsortedDimParallel64 * static_cast<uint64_t>(dtypeSize),
                                                blockUbSize64);

    workspaceSize = excusiveBins + globalHist + sortedIdx + histTile + xB8 + sortedValue;
    return true;
}

static bool FillRadixKernelResources(SortKthTileInfo& info, uint32_t indexSize, uint32_t lastDimTileNum,
                                     uint32_t tmpUbSize)
{
    RadixClearParams clearParams;
    if (!FillRadixKernelParams(info.dtypeSize, indexSize, info.coreNumNeed, lastDimTileNum, info.unsortedDimParallel,
                               info.blockUbSize, tmpUbSize, clearParams)) {
        return false;
    }
    info.keyParams0 = clearParams.keyParams0;
    info.keyParams1 = clearParams.keyParams1;
    info.keyParams2 = clearParams.keyParams2;
    info.keyParams3 = clearParams.keyParams3;
    info.keyParams4 = clearParams.keyParams4;
    info.keyParams5 = clearParams.keyParams5;
    uint64_t sortWorkspaceSize = 0;
    if (!ComputeRadixSortWorkspace(info.lastAxis, info.dtypeSize, indexSize, lastDimTileNum, info.numTileDataSize,
                                   info.unsortedDimParallel, info.keyParams0, info.keyParams1, info.keyParams2,
                                   info.keyParams3, info.keyParams4, info.blockUbSize, sortWorkspaceSize)) {
        return false;
    }
    info.workspaceSize = static_cast<size_t>(sortWorkspaceSize + WORK_SPACE_SIZE);
    return true;
}

bool FillRadixMoreCoreInfo(SortKthTileInfo& info)
{
    uint32_t usableUb = info.ubSize > SIMT_UB ? info.ubSize - SIMT_UB : 0;
    uint32_t indexSize = info.isInt32 != 0 ? static_cast<uint32_t>(sizeof(int32_t)) :
                                             static_cast<uint32_t>(sizeof(int64_t));
    uint32_t tileData = 0;
    uint32_t tmpUbSize = 0;
    if (!ComputeRadixTileData(info.lastAxis, info.unsortedDim, info.dtypeSize, indexSize, info.maxCoreNum, usableUb,
                              info.blockUbSize, tileData, tmpUbSize)) {
        return false;
    }
    info.tmpUbSize = tmpUbSize;
    uint32_t lastDimTileNum = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(info.lastAxis), static_cast<uint64_t>(tileData), lastDimTileNum)) {
        return false;
    }
    if (info.maxCoreNum <= lastDimTileNum) {
        info.unsortedDimParallel = 1;
    } else {
        info.unsortedDimParallel = lastDimTileNum == 0 ? info.maxCoreNum : info.maxCoreNum / lastDimTileNum;
        if (info.unsortedDim < static_cast<int64_t>(info.unsortedDimParallel)) {
            info.unsortedDimParallel = static_cast<uint32_t>(info.unsortedDim);
        }
    }
    info.numTileDataSize = tileData;
    uint64_t sortLoopTimes64 = (static_cast<uint64_t>(info.unsortedDim) + info.unsortedDimParallel - 1U) /
                               info.unsortedDimParallel;
    if (sortLoopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    info.sortLoopTimes = static_cast<uint32_t>(sortLoopTimes64);
    info.lastDimNeedCore = std::min(info.maxCoreNum, lastDimTileNum);
    info.coreNumNeed = info.unsortedDimParallel * info.lastDimNeedCore;
    info.lastDimTileNum = lastDimTileNum;
    return FillRadixKernelResources(info, indexSize, lastDimTileNum, tmpUbSize);
}

bool ComputeRadixOneCoreUbSizes(int64_t lastAxis, uint32_t dtypeSize, uint32_t indexElemSize, uint32_t blockUbSize,
                                uint32_t& xUbSize, uint32_t& idxUbSize)
{
    uint64_t xBytes = static_cast<uint64_t>(lastAxis) * dtypeSize;
    uint64_t idxBytes = static_cast<uint64_t>(lastAxis) * indexElemSize;
    return CeilAlignUint32(xBytes, blockUbSize, xUbSize) && CeilAlignUint32(idxBytes, blockUbSize, idxUbSize);
}

// =============================================================================
// Merge sort — common
// =============================================================================
bool ComputeMergeSortPlan(int64_t axisLen, int64_t unsortedDim, uint32_t blockUbSize, uint32_t tileDataNum,
                          uint32_t maxCoreNum, MergeSortPlan& plan)
{
    uint64_t alignNum64 = Ops::Base::CeilAlign(static_cast<uint64_t>(axisLen), static_cast<uint64_t>(blockUbSize));
    if (alignNum64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) || alignNum64 == 0U) {
        return false;
    }
    plan.alignNum = static_cast<uint32_t>(alignNum64);
    uint32_t oneCoreRowNumMax = (tileDataNum / 2U) / plan.alignNum;
    oneCoreRowNumMax = oneCoreRowNumMax == 0U ? 1U : oneCoreRowNumMax;
    if (unsortedDim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) || unsortedDim <= 0) {
        return false;
    }
    uint32_t bUnsorted = static_cast<uint32_t>(unsortedDim);
    plan.coreNumNeed = std::min(bUnsorted, maxCoreNum);
    uint32_t batchPerCore = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(bUnsorted), static_cast<int64_t>(plan.coreNumNeed)));
    uint32_t oneCoreRowNum = batchPerCore <= oneCoreRowNumMax ? batchPerCore : oneCoreRowNumMax;
    uint64_t totalRowsPerRound = static_cast<uint64_t>(plan.coreNumNeed) * oneCoreRowNum;
    if (totalRowsPerRound == 0U) {
        return false;
    }
    uint64_t sortLoopTimes64 = (static_cast<uint64_t>(bUnsorted) + totalRowsPerRound - 1U) / totalRowsPerRound;
    if (sortLoopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    plan.oneCoreRowNum = oneCoreRowNum;
    plan.sortLoopTimes = static_cast<uint32_t>(sortLoopTimes64);
    return true;
}

bool FillMergeSortInfo(SortKthTileInfo& info, uint32_t indexDtypeSize, uint32_t concatTmpSize)
{
    constexpr uint32_t TILE_DATA_NUM = 4096;
    MergeSortPlan plan;
    if (!ComputeMergeSortPlan(info.lastAxis, info.unsortedDim, info.blockUbSize, TILE_DATA_NUM, info.maxCoreNum,
                              plan)) {
        return false;
    }
    info.sortLoopTimes = plan.sortLoopTimes;
    info.lastDimTileNum = 1;
    info.unsortedDimParallel = plan.coreNumNeed;
    info.lastDimNeedCore = 1;
    info.numTileDataSize = static_cast<uint32_t>(info.lastAxis);
    info.coreNumNeed = plan.coreNumNeed;
    info.keyParams0 = plan.oneCoreRowNum;
    info.keyParams1 = plan.alignNum * plan.oneCoreRowNum * info.dtypeSize;
    info.keyParams2 = plan.alignNum * plan.oneCoreRowNum * indexDtypeSize;
    info.keyParams3 = plan.alignNum;
    info.keyParams4 = info.lastAxis > ONE_CORE_DATA_SIZE ? 1 : DOUBLE_BUFFER_NUM;
    info.tmpUbSize = std::max(concatTmpSize, info.blockUbSize);
    info.workspaceSize = WORK_SPACE_SIZE;
    return true;
}

bool ComputeMergeSortTiling(gert::TilingContext* context, SortKthTileInfo& info, uint32_t indexDtypeSize)
{
    constexpr uint32_t TILE_DATA_NUM = 4096;
    MergeSortPlan plan;
    if (!ComputeMergeSortPlan(info.lastAxis, info.unsortedDim, info.blockUbSize, TILE_DATA_NUM, info.maxCoreNum,
                              plan)) {
        return false;
    }
    auto platformInfo = context->GetPlatformInfo();
    auto plat = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t maxTypeSize = (info.dataType == ge::DT_BF16) ? static_cast<uint32_t>(sizeof(float)) : info.dtypeSize;
    uint32_t concatTmpSize = AscendC::GetConcatTmpSize(plat, plan.alignNum, maxTypeSize);
    if (!FillMergeSortInfo(info, indexDtypeSize, concatTmpSize)) {
        return false;
    }
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = info.workspaceSize;
    return true;
}

// =============================================================================
// Merge sort — multi-core (more-core)
// =============================================================================
bool IsMergeMoreCoreSupported(ge::DataType dataType, int64_t axisLen, int64_t unsortedDim, uint32_t maxCoreNum)
{
    if (dataType != ge::DT_FLOAT || axisLen <= static_cast<int64_t>(MERGE_SORT_MAX_AXIS_FP32) ||
        axisLen > static_cast<int64_t>(MULTI_CORE_MERGE_SORT_MAX_AXIS)) {
        return false;
    }
    uint64_t hCoreNum = (static_cast<uint64_t>(axisLen) + ONE_CORE_DATA_SIZE - 1U) / ONE_CORE_DATA_SIZE;
    return hCoreNum > 0 && static_cast<uint64_t>(unsortedDim) * hCoreNum <= maxCoreNum;
}

bool ComputeMergeMoreCorePlan(int64_t axisLen, int64_t unsortedDim, uint32_t ubSize, uint32_t mergeBytesPerElem,
                              MergeMoreCorePlan& plan)
{
    if (axisLen <= 0 || unsortedDim <= 0 || mergeBytesPerElem == 0U) {
        return false;
    }
    uint32_t hCoreNum;
    if (!CeilDivUint32(static_cast<uint64_t>(axisLen), ONE_CORE_DATA_SIZE, hCoreNum)) {
        return false;
    }
    uint64_t coreNumNeed64 = static_cast<uint64_t>(unsortedDim) * hCoreNum;
    if (static_cast<uint64_t>(axisLen) > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        static_cast<uint64_t>(unsortedDim) > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        coreNumNeed64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    uint64_t numTileDataSize64 = static_cast<uint64_t>(axisLen) / hCoreNum;
    if (numTileDataSize64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    plan.lastDimTileNum = static_cast<uint32_t>(static_cast<uint64_t>(axisLen));
    plan.lastDimNeedCore = hCoreNum;
    plan.numTileDataSize = static_cast<uint32_t>(numTileDataSize64);
    plan.unsortedDimParallel = static_cast<uint32_t>(static_cast<uint64_t>(unsortedDim));
    plan.sortLoopTimes = 1;
    plan.coreNumNeed = static_cast<uint32_t>(coreNumNeed64);
    // keyParams0 is max elements handled per UB round by the merge more-core pipeline.
    plan.keyParams0 = ubSize / mergeBytesPerElem;
    return true;
}

bool FillMergeMoreCoreInfo(SortKthTileInfo& info, uint32_t mergeBytesPerElem)
{
    MergeMoreCorePlan plan;
    if (!ComputeMergeMoreCorePlan(info.lastAxis, info.unsortedDim, info.ubSize, mergeBytesPerElem, plan)) {
        return false;
    }
    info.lastDimTileNum = plan.lastDimTileNum;
    info.lastDimNeedCore = plan.lastDimNeedCore;
    info.numTileDataSize = plan.numTileDataSize;
    info.unsortedDimParallel = plan.unsortedDimParallel;
    info.sortLoopTimes = plan.sortLoopTimes;
    info.coreNumNeed = plan.coreNumNeed;
    info.keyParams0 = plan.keyParams0;
    uint64_t wsBytes = static_cast<uint64_t>(MERGE_SORT_WORKSPACE_PARAM) * info.lastAxis * info.unsortedDim *
                       sizeof(int32_t);
    info.workspaceSize = static_cast<size_t>(wsBytes + WORK_SPACE_SIZE);
    return true;
}

bool ComputeMergeMoreCoreTiling(gert::TilingContext* context, SortKthTileInfo& info, uint32_t mergeBytesPerElem)
{
    if (!FillMergeMoreCoreInfo(info, mergeBytesPerElem)) {
        return false;
    }
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = info.workspaceSize;
    context->SetScheduleMode(1);
    return true;
}

// =============================================================================
// Merge sort — intra-core
// =============================================================================
uint32_t ComputeMergeIntraCoreBlockSortSize(uint32_t ubSize)
{
    constexpr uint32_t PHASE2_BYTES_PER_ELEM = MERGE_SORT_LIST_NUM * 2 * SORT_STRUCT_BYTES;
    uint32_t blockSortSize = ubSize / PHASE2_BYTES_PER_ELEM;
    return (blockSortSize / MERGE_INTRA_CORE_SORT_ALIGN) * MERGE_INTRA_CORE_SORT_ALIGN;
}

uint32_t ComputeMergeIntraCoreExtractChunkSize(uint32_t ubSize)
{
    constexpr uint32_t PHASE3_BYTES_PER_ELEM = (SORT_STRUCT_BYTES + sizeof(float) + sizeof(int32_t) + sizeof(int64_t)) *
                                               2;
    uint32_t extractChunkSize = ubSize / PHASE3_BYTES_PER_ELEM;
    return (extractChunkSize / MERGE_INTRA_CORE_SORT_ALIGN) * MERGE_INTRA_CORE_SORT_ALIGN;
}

bool IsMergeIntraCoreSupported(ge::DataType dataType, int64_t axisLen, int64_t unsortedDim, uint32_t maxCoreNum,
                               uint32_t ubSize)
{
    if (dataType != ge::DT_FLOAT || axisLen <= static_cast<int64_t>(MERGE_SORT_MAX_AXIS_FP32) ||
        unsortedDim < static_cast<int64_t>(maxCoreNum / DOUBLE_BUFFER_NUM)) {
        return false;
    }
    uint64_t maxBatch = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * static_cast<uint64_t>(maxCoreNum);
    if (unsortedDim <= 0 || static_cast<uint64_t>(unsortedDim) > maxBatch) {
        return false;
    }
    uint32_t blockSortSize = ComputeMergeIntraCoreBlockSortSize(ubSize);
    uint32_t extractChunkSize = ComputeMergeIntraCoreExtractChunkSize(ubSize);
    if (blockSortSize == 0U || extractChunkSize == 0U) {
        return false;
    }
    return axisLen <= static_cast<int64_t>(blockSortSize) * MERGE_INTRA_CORE_MAX_BLOCKS;
}

bool ComputeMergeIntraCorePlan(int64_t axisLen, int64_t unsortedDim, uint32_t ubSize, uint32_t maxCoreNum,
                               MergeIntraCorePlan& plan)
{
    if (axisLen <= 0 || unsortedDim <= 0 || maxCoreNum == 0U) {
        return false;
    }
    uint64_t batchPerCore64 = (static_cast<uint64_t>(unsortedDim) + maxCoreNum - 1U) / maxCoreNum;
    if (batchPerCore64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    plan.batchPerCore = static_cast<uint32_t>(batchPerCore64);
    uint64_t actualCoreNum64 = (static_cast<uint64_t>(unsortedDim) + plan.batchPerCore - 1U) / plan.batchPerCore;
    if (actualCoreNum64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    plan.actualCoreNum = static_cast<uint32_t>(actualCoreNum64);
    plan.blockSortSize = ComputeMergeIntraCoreBlockSortSize(ubSize);
    if (plan.blockSortSize == 0U) {
        return false;
    }
    plan.extractChunkSize = ComputeMergeIntraCoreExtractChunkSize(ubSize);
    if (plan.extractChunkSize == 0U) {
        return false;
    }
    uint64_t blocksPerRow64 = (static_cast<uint64_t>(axisLen) + plan.blockSortSize - 1U) / plan.blockSortSize;
    if (blocksPerRow64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    plan.blocksPerRow = static_cast<uint32_t>(blocksPerRow64);
    uint64_t alignNum64 = static_cast<uint64_t>(plan.blocksPerRow) * plan.blockSortSize;
    if (alignNum64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    plan.alignNum = static_cast<uint32_t>(alignNum64);
    return true;
}

bool FillMergeIntraCoreInfo(SortKthTileInfo& info)
{
    MergeIntraCorePlan plan;
    if (!ComputeMergeIntraCorePlan(info.lastAxis, info.unsortedDim, info.ubSize, info.maxCoreNum, plan)) {
        return false;
    }
    info.keyParams0 = plan.batchPerCore;
    info.coreNumNeed = plan.actualCoreNum;
    info.unsortedDimParallel = plan.actualCoreNum;
    info.numTileDataSize = plan.blockSortSize;
    info.keyParams4 = plan.extractChunkSize;
    info.keyParams5 = plan.blockSortSize > 0U ?
                          static_cast<uint32_t>(std::numeric_limits<int32_t>::max() / plan.blockSortSize) :
                          static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
    info.lastDimTileNum = plan.blocksPerRow;
    info.keyParams3 = plan.alignNum;
    info.lastDimNeedCore = plan.actualCoreNum;
    size_t cachePerCore = static_cast<size_t>(plan.alignNum) * SORT_STRUCT_BYTES * 2;
    if (info.blockUbSize > 0U) {
        cachePerCore = Ops::Base::CeilAlign<size_t>(cachePerCore, static_cast<size_t>(info.blockUbSize));
    }
    info.workspaceSize = cachePerCore * plan.actualCoreNum + WORK_SPACE_SIZE;
    return true;
}

bool ComputeMergeIntraCoreTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    if (!FillMergeIntraCoreInfo(info)) {
        return false;
    }
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = info.workspaceSize;
    return true;
}

// =============================================================================
// Two-stage sort
// =============================================================================
bool SearchTwoStageBatchPlan(uint32_t maxBatch, std::function<bool(uint32_t, TwoStageBatchPlan&)> tryCandidate,
                             TwoStageBatchPlan& result)
{
    if (maxBatch == 0U) {
        return false;
    }
    TwoStageBatchPlan maxPlan;
    TwoStageBatchPlan bestPlan;
    bool hasMaxPlan = false;
    uint32_t maxPlanBatchesPerCore = 0;
    uint64_t bestIdleSlots = std::numeric_limits<uint64_t>::max();
    for (uint32_t candidate = maxBatch; candidate >= 1U; --candidate) {
        TwoStageBatchPlan candidatePlan;
        if (!tryCandidate(candidate, candidatePlan)) {
            continue;
        }
        if (!hasMaxPlan) {
            maxPlan = candidatePlan;
            hasMaxPlan = true;
            maxPlanBatchesPerCore = Ops::Base::CeilDiv(maxPlan.batchNum, maxPlan.blockDim);
            bestPlan = maxPlan;
            bestIdleSlots = static_cast<uint64_t>(maxPlanBatchesPerCore) * maxPlan.blockDim - maxPlan.batchNum;
            continue;
        }
        uint32_t batchesPerCore = Ops::Base::CeilDiv(candidatePlan.batchNum, candidatePlan.blockDim);
        if (batchesPerCore != maxPlanBatchesPerCore) {
            break;
        }
        uint64_t idleSlots = static_cast<uint64_t>(batchesPerCore) * candidatePlan.blockDim - candidatePlan.batchNum;
        if (idleSlots < bestIdleSlots) {
            bestPlan = candidatePlan;
            bestIdleSlots = idleSlots;
        }
    }
    if (!hasMaxPlan) {
        return false;
    }
    result = bestPlan;
    return true;
}

// Sort-only non-last policy: prefer more active cores, then fewer batches per core, then fewer idle slots.
// Last-axis Sort and KthValue keep SearchTwoStageBatchPlan so their established batching remains unchanged.
static bool SearchTwoStageBatchPlanFillCores(uint32_t maxBatch, uint32_t targetBlockDim,
                                             std::function<bool(uint32_t, TwoStageBatchPlan&)> tryCandidate,
                                             TwoStageBatchPlan& result)
{
    if (maxBatch == 0U || targetBlockDim == 0U) {
        return false;
    }
    TwoStageBatchPlan bestPlan;
    bool hasBestPlan = false;
    uint32_t bestActiveCore = 0U;
    uint32_t bestBatchesPerCore = std::numeric_limits<uint32_t>::max();
    uint64_t bestIdleSlots = std::numeric_limits<uint64_t>::max();
    for (uint32_t candidate = maxBatch; candidate >= 1U; --candidate) {
        TwoStageBatchPlan candidatePlan;
        if (!tryCandidate(candidate, candidatePlan) || candidatePlan.batchNum == 0U || candidatePlan.blockDim == 0U) {
            continue;
        }
        uint32_t batchesPerCore = Ops::Base::CeilDiv(candidatePlan.batchNum, candidatePlan.blockDim);
        if (hasBestPlan && bestActiveCore == targetBlockDim && candidatePlan.blockDim == targetBlockDim &&
            batchesPerCore > bestBatchesPerCore) {
            // Decreasing the batch size can only increase the remaining batch count.
            break;
        }
        uint64_t idleSlots = static_cast<uint64_t>(batchesPerCore) * candidatePlan.blockDim - candidatePlan.batchNum;
        bool useCandidate = !hasBestPlan || candidatePlan.blockDim > bestActiveCore ||
                            (candidatePlan.blockDim == bestActiveCore && batchesPerCore < bestBatchesPerCore) ||
                            (candidatePlan.blockDim == bestActiveCore && batchesPerCore == bestBatchesPerCore &&
                             idleSlots < bestIdleSlots);
        if (useCandidate) {
            bestPlan = candidatePlan;
            hasBestPlan = true;
            bestActiveCore = candidatePlan.blockDim;
            bestBatchesPerCore = batchesPerCore;
            bestIdleSlots = idleSlots;
        }
    }
    if (!hasBestPlan) {
        return false;
    }
    result = bestPlan;
    return true;
}

uint32_t MaxTwoStageU16SafeBatch(uint32_t axisLen)
{
    if (axisLen == 0U || axisLen > static_cast<uint32_t>(std::numeric_limits<uint16_t>::max())) {
        return 0U;
    }
    uint32_t maxBatch = static_cast<uint32_t>(
        std::sqrt(static_cast<double>(std::numeric_limits<uint16_t>::max()) / axisLen));
    while (static_cast<uint64_t>(maxBatch) * maxBatch * axisLen >
           static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())) {
        --maxBatch;
    }
    while (static_cast<uint64_t>(maxBatch + 1U) * (maxBatch + 1U) * axisLen <=
           static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())) {
        ++maxBatch;
    }
    return maxBatch;
}

bool ComputeTwoStageSortTmpUb(ge::DataType dataType, uint32_t axisLen, uint32_t totalElems, uint32_t blockUbSize,
                              uint32_t& tmpUbSize)
{
    tmpUbSize = 0;
    QuerySortTmpSizeRadix(dataType, totalElems, tmpUbSize);
    uint32_t aligned = 0;
    if (!CeilAlignUint32(tmpUbSize, blockUbSize, aligned)) {
        return false;
    }
    tmpUbSize = aligned;
    bool useRankInverse = UseTwoStageRankInverse(axisLen);
    if (!useRankInverse) {
        uint32_t stage2TmpUbSize = 0;
        QuerySortTmpSizeRadix(ge::DT_UINT16, totalElems, stage2TmpUbSize);
        uint32_t stage2Aligned = 0;
        if (!CeilAlignUint32(stage2TmpUbSize, blockUbSize, stage2Aligned)) {
            return false;
        }
        tmpUbSize = std::max(tmpUbSize, stage2Aligned);
    }
    return true;
}

uint64_t EstimateTwoStageUbBytes(const SortKthTileInfo& info, uint32_t totalElems, uint32_t sortTmpUb)
{
    uint64_t valueRawBytes = 0U;
    uint64_t idxRawBytes = 0U;
    uint64_t finalIdxRawBytes = 0U;
    uint32_t finalIdxElemBytes = info.finalIdxUbElemBytes == 0U ? info.y2DtypeSize : info.finalIdxUbElemBytes;
    if (ge::MulOverflow(totalElems, info.dtypeSize, valueRawBytes) ||
        ge::MulOverflow(totalElems, sizeof(uint32_t), idxRawBytes) ||
        ge::MulOverflow(totalElems, finalIdxElemBytes, finalIdxRawBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("EstimateTwoStageUbBytes", "totalElems",
                                              std::to_string(totalElems).c_str(),
                                              "The value of totalElems must not cause raw byte size overflow.");
        return std::numeric_limits<uint64_t>::max();
    }
    uint64_t valueBytes = Ops::Base::CeilAlign<uint64_t>(valueRawBytes, info.blockUbSize);
    uint64_t idxBytes = Ops::Base::CeilAlign<uint64_t>(idxRawBytes, info.blockUbSize);
    uint64_t finalIdxBytes = Ops::Base::CeilAlign<uint64_t>(finalIdxRawBytes, info.blockUbSize);
    if (valueBytes == 0U || idxBytes == 0U || finalIdxBytes == 0U) {
        return std::numeric_limits<uint64_t>::max();
    }
    uint32_t idxBufferCount = UseTwoStageRankInverse(static_cast<uint32_t>(info.lastAxis)) ? 2U : 3U;
    uint64_t totalBytes = 0U;
    uint64_t idxTotalBytes = 0U;
    if (ge::MulOverflow(valueBytes, 2U, totalBytes) || ge::MulOverflow(idxBytes, idxBufferCount, idxTotalBytes) ||
        ge::AddOverflow(totalBytes, idxTotalBytes, totalBytes) ||
        ge::AddOverflow(totalBytes, finalIdxBytes, totalBytes) || ge::AddOverflow(totalBytes, sortTmpUb, totalBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "EstimateTwoStageUbBytes", "totalBytes",
            (std::to_string(valueBytes) + ", " + std::to_string(idxBytes) + ", " + std::to_string(idxBufferCount) +
             ", " + std::to_string(finalIdxBytes) + ", " + std::to_string(sortTmpUb))
                .c_str(),
            "The value of totalBytes must not overflow.");
        return std::numeric_limits<uint64_t>::max();
    }
    return totalBytes;
}

bool PrepareTwoStageBatchCandidate(const SortKthTileInfo& info, uint32_t candidate, uint32_t& totalElems,
                                   uint32_t& tmpUbSize, bool& useRankInverse, uint64_t& totalBytes)
{
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    uint64_t totalElems64 = static_cast<uint64_t>(candidate) * axisLen;
    if (totalElems64 > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    totalElems = static_cast<uint32_t>(totalElems64);
    tmpUbSize = 0;
    if (!ComputeTwoStageSortTmpUb(info.dataType, axisLen, totalElems, info.blockUbSize, tmpUbSize)) {
        return false;
    }
    useRankInverse = UseTwoStageRankInverse(axisLen);
    totalBytes = EstimateTwoStageUbBytes(info, totalElems, tmpUbSize);
    return true;
}

static bool ComputeSmallAxisInsertionBatchParams(const SortKthTileInfo& info, uint32_t axisLen, uint32_t& bytesPerSeg,
                                                 uint32_t& usableUb, uint32_t& maxBatchByUb)
{
    if (info.ubSize <= SIMT_UB) {
        return false;
    }
    bytesPerSeg = ComputeInsertionBytesPerSeg(info.dataType, axisLen, info.dtypeSize, info.y2DtypeSize,
                                              info.blockUbSize);
    if (bytesPerSeg == 0U) {
        return false;
    }
    usableUb = info.ubSize - SIMT_UB;
    maxBatchByUb = usableUb / bytesPerSeg;
    return true;
}

template <typename ComputeBatchNumFn>
static bool EstimateSmallAxisInsertionBatching(const SortKthTileInfo& info, uint32_t batchSizeCap,
                                               ComputeBatchNumFn computeBatchNum, SmallAxisRoutePlan& plan)
{
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    uint32_t bytesPerSeg = 0;
    uint32_t usableUb = 0;
    uint32_t maxBatchByUb = 0;
    if (!ComputeSmallAxisInsertionBatchParams(info, axisLen, bytesPerSeg, usableUb, maxBatchByUb) ||
        maxBatchByUb == 0U) {
        return false;
    }
    uint32_t batchSize = std::min({batchSizeCap, maxBatchByUb, SMALL_AXIS_MAX_DATACOPY_BLOCK_COUNT});
    if (batchSize == 0U) {
        return false;
    }
    plan.batchSize = batchSize;
    if (!computeBatchNum(batchSize, plan.batchNum)) {
        return false;
    }
    plan.blockDim = std::min(info.maxCoreNum, plan.batchNum);
    return plan.batchNum > 0U && plan.blockDim > 0U;
}

template <typename ComputeBatchNumFn>
static bool TrySmallAxisTwoStageBatchCandidate(const SortKthTileInfo& info, uint32_t candidate,
                                               ComputeBatchNumFn computeBatchNum, SmallAxisRoutePlan& plan)
{
    // Reject candidates unsupported by the batch mapping before querying Sort temporary UB.
    plan.batchSize = candidate;
    if (!computeBatchNum(candidate, plan.batchNum)) {
        return false;
    }
    uint32_t totalElems = 0;
    uint32_t tmpUbSize = 0;
    bool useRankInverse = false;
    uint64_t totalBytes = 0;
    if (!PrepareTwoStageBatchCandidate(info, candidate, totalElems, tmpUbSize, useRankInverse, totalBytes)) {
        return false;
    }
    if (totalBytes + SIMT_UB > info.ubSize) {
        return false;
    }
    plan.blockDim = std::min(info.maxCoreNum, plan.batchNum);
    plan.tmpUbSize = tmpUbSize;
    plan.useRankInverse = useRankInverse;
    return plan.batchNum > 0U && plan.blockDim > 0U;
}

// Estimate the optimal batch size for two-stage sort on small-axis inputs.
//
// Two-stage sort processes batchSize × axisLen elements per batch in two phases:
//   Stage 1: sort each row by value  → produces per-row sorted indices
//   Stage 2: gather by a derived key → produces the final output ordering
//
// UB budget per element (minBytesPerElem):
//   - value buffer + alias buffer:          dtypeSize × 2
//   - index buffers:                        sizeof(uint32_t) × idxBufferCount
//     · rank-inverse path (axisLen ≤ threshold): 2 (rankInverse + finalIdx)
//     · standard path:                           3 (srcIndex + gatheredIdx + finalIdx)
//   - alias index buffer:                   sizeof(uint32_t) × 1
//
// For the standard (non-rank-inverse) path, stage-2 uses uint16_t keys, so
// batch² × axisLen must not exceed UINT16_MAX to avoid key overflow.
// MaxTwoStageU16SafeBatch() computes the largest batch satisfying this constraint.
//
// After establishing the UB-derived upper bound, last-axis and KthValue routes
// retain the original batching policy. Sort non-last routes may instead prefer
// activating all available cores before minimising per-core loops and idle slots.
template <typename ComputeBatchNumFn>
static bool EstimateSmallAxisTwoStageBatching(const SortKthTileInfo& info, uint32_t batchSizeCap,
                                              ComputeBatchNumFn computeBatchNum, bool preferFullCoreBatching,
                                              SmallAxisRoutePlan& plan)
{
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    if (info.ubSize <= SIMT_UB || axisLen == 0U || batchSizeCap == 0U) {
        return false;
    }
    bool useRankInverse = UseTwoStageRankInverse(axisLen);
    // rank-inverse path needs 2 index buffers; standard path needs 3 (see comment above)
    uint32_t idxBufferCount = useRankInverse ? 2U : 3U;
    // Keep the legacy coarse lower bound for existing routes; sch11 supplies its explicit compact width.
    uint32_t finalIdxElemBytes = info.finalIdxUbElemBytes == 0U ? static_cast<uint32_t>(sizeof(uint32_t)) :
                                                                  info.finalIdxUbElemBytes;
    uint64_t minBytesPerElem = static_cast<uint64_t>(info.dtypeSize) * 2U +
                               static_cast<uint64_t>(sizeof(uint32_t)) * idxBufferCount + finalIdxElemBytes;
    uint64_t maxElemsByUb = (info.ubSize - SIMT_UB) / minBytesPerElem;
    uint64_t maxBatchByUb = maxElemsByUb / axisLen;
    uint32_t maxBatch = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(batchSizeCap), maxBatchByUb));
    if (!useRankInverse) {
        // cap batch so that stage-2 uint16_t keys don't overflow
        maxBatch = std::min(maxBatch, MaxTwoStageU16SafeBatch(axisLen));
    }
    TwoStageBatchPlan result;
    auto tryCandidate = [&info, &computeBatchNum](uint32_t candidate, TwoStageBatchPlan& p) -> bool {
        SmallAxisRoutePlan candidatePlan;
        if (!TrySmallAxisTwoStageBatchCandidate(info, candidate, computeBatchNum, candidatePlan)) {
            return false;
        }
        p.batchSize = candidatePlan.batchSize;
        p.batchNum = candidatePlan.batchNum;
        p.blockDim = candidatePlan.blockDim;
        p.tmpUbSize = candidatePlan.tmpUbSize;
        return true;
    };
    bool foundPlan = preferFullCoreBatching ?
                         SearchTwoStageBatchPlanFillCores(maxBatch, info.maxCoreNum, tryCandidate, result) :
                         SearchTwoStageBatchPlan(maxBatch, tryCandidate, result);
    if (!foundPlan) {
        return false;
    }
    plan.batchSize = result.batchSize;
    plan.batchNum = result.batchNum;
    plan.blockDim = result.blockDim;
    plan.tmpUbSize = result.tmpUbSize;
    plan.useRankInverse = useRankInverse;
    return true;
}

// =============================================================================
// Small-axis route selection
// =============================================================================
static bool SelectSmallAxisRouteImpl(const SortKthTileInfo& info, uint32_t batchSizeCap,
                                     std::function<bool(uint32_t, uint32_t&)> computeBatchNum,
                                     bool preferFullCoreBatching, SmallAxisRoutePlan& plan)
{
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    if (axisLen <= 1U) {
        return false;
    }
    const SmallAxisRule* rule = FindSmallAxisRule(info.dataType);
    if (rule == nullptr) {
        return false;
    }
    uint32_t fullCoreSegs = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(info.unsortedDim), static_cast<uint64_t>(info.maxCoreNum), fullCoreSegs)) {
        return false;
    }
    SmallAxisRoutePlan twoStagePlan;
    if (rule->twoStageMaxN > 0U && axisLen <= rule->twoStageMaxN && axisLen <= SMALL_AXIS_THRESHOLD &&
        fullCoreSegs >= LookupMinSegs(rule->twoStageTiers, axisLen) &&
        EstimateSmallAxisTwoStageBatching(info, batchSizeCap, computeBatchNum, preferFullCoreBatching, twoStagePlan)) {
        plan = twoStagePlan;
        plan.kind = SmallAxisRouteKind::TWO_STAGE;
        return true;
    }
    if (axisLen > rule->insertionMaxN || fullCoreSegs < LookupMinSegs(rule->insertionTiers, axisLen)) {
        return false;
    }
    SmallAxisRoutePlan insertionPlan;
    if (!EstimateSmallAxisInsertionBatching(info, batchSizeCap, computeBatchNum, insertionPlan)) {
        return false;
    }
    plan = insertionPlan;
    plan.kind = SmallAxisRouteKind::INSERTION;
    return true;
}

bool SelectSmallAxisRoute(const SortKthTileInfo& info, SmallAxisRoutePlan& plan)
{
    uint32_t fullCoreSegs = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(info.unsortedDim), static_cast<uint64_t>(info.maxCoreNum), fullCoreSegs)) {
        return false;
    }
    auto computeBatchNum = [&info](uint32_t batchSize, uint32_t& batchNum) -> bool {
        return CeilDivUint32(static_cast<uint64_t>(info.unsortedDim), batchSize, batchNum);
    };
    return SelectSmallAxisRouteImpl(info, fullCoreSegs, computeBatchNum, false, plan);
}

static bool SelectNonLastSmallAxisRouteImpl(const SortKthTileInfo& info, SmallAxisRoutePlan& plan,
                                            bool enableSortNonLastOptimizations)
{
    uint32_t fullCoreSegs = 0U;
    uint32_t innerSize = 0U;
    if (enableSortNonLastOptimizations && info.outerSize > 1 && info.innerSize > 0 &&
        static_cast<uint64_t>(info.innerSize) <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
        CeilDivUint32(static_cast<uint64_t>(info.unsortedDim), static_cast<uint64_t>(info.maxCoreNum), fullCoreSegs)) {
        innerSize = static_cast<uint32_t>(info.innerSize);
        // Group only when the target per-core work can hold at least two complete outer slices. Rounding the
        // batch cap to an innerSize multiple preserves batchSize == outerSlicesPerBatch * innerSize, allowing the
        // kernel to recover each segment's (outer, inner) coordinates without a partial outer slice.
        // Keep rows smaller than one UB block on the established non-grouped path. The grouped 3D NDDMA
        // transpose loses lane data for b64 innerSize == 2 when several complete outer slices share one batch.
        // Element-based NDDMA strides still support rows that span at least one block without being block-aligned.
        const uint64_t innerRowBytes = static_cast<uint64_t>(innerSize) * info.dtypeSize;
        const bool canGroupMultipleOuterSlices = innerSize <= fullCoreSegs / 2U;
        const bool innerRowSpansUbBlock = info.blockUbSize != 0U && innerRowBytes >= info.blockUbSize;
        if (canGroupMultipleOuterSlices && innerRowSpansUbBlock) {
            uint64_t groupedBatchCap64 = Ops::Base::CeilDiv(static_cast<uint64_t>(fullCoreSegs),
                                                            static_cast<uint64_t>(innerSize)) *
                                         innerSize;
            uint32_t groupedBatchCap = static_cast<uint32_t>(
                std::min<uint64_t>(groupedBatchCap64, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
            auto computeGroupedBatchNum = [&info, innerSize](uint32_t batchSize, uint32_t& batchNum) -> bool {
                if (batchSize < innerSize * 2U || batchSize % innerSize != 0U) {
                    return false;
                }
                return CeilDivUint32(static_cast<uint64_t>(info.outerSize), batchSize / innerSize, batchNum);
            };
            SmallAxisRoutePlan groupedPlan;
            SortKthTileInfo groupedInfo = info;
            // sch11 stores row-local indices as uint32_t in a dedicated UB buffer and converts only at GM writeback.
            groupedInfo.finalIdxUbElemBytes = sizeof(uint32_t);
            if (SelectSmallAxisRouteImpl(groupedInfo, groupedBatchCap, computeGroupedBatchNum, false, groupedPlan) &&
                groupedPlan.kind == SmallAxisRouteKind::TWO_STAGE) {
                groupedPlan.groupOuterSlices = true;
                plan = groupedPlan;
                return true;
            }
        }
    }

    uint32_t batchSizeCap = static_cast<uint32_t>(
        std::min<int64_t>(info.innerSize, static_cast<int64_t>(std::numeric_limits<uint32_t>::max())));
    auto computeBatchNum = [&info](uint32_t batchSize, uint32_t& batchNum) -> bool {
        return ComputeNonLastBatchNum(info.outerSize, info.innerSize, batchSize, batchNum);
    };
    return SelectSmallAxisRouteImpl(info, batchSizeCap, computeBatchNum, enableSortNonLastOptimizations, plan);
}

bool SelectNonLastSmallAxisRoute(const SortKthTileInfo& info, SmallAxisRoutePlan& plan)
{
    return SelectNonLastSmallAxisRouteImpl(info, plan, false);
}

bool SelectSortNonLastSmallAxisRoute(const SortKthTileInfo& info, SmallAxisRoutePlan& plan)
{
    return SelectNonLastSmallAxisRouteImpl(info, plan, true);
}

} // namespace optiling
