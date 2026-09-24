/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file unique_sort_tiling.cpp
 * \brief sort ac tiling impl
 */
#include "unique_sort_tiling.h"

#include <string>
#include <vector>

#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "util/platform_util.h"

#include "../../../kth_value/op_host/arch35/kth_value_tiling_common.h"
#include "../../op_kernel/arch35/sort/unique_sort_tiling_data.h"
#include "../../op_kernel/arch35/unique_with_counts_and_sorting_tiling_key.h"

namespace unique_with_counts_and_sorting_sort {
using namespace optiling;
void SetSortTmpSize(ge::DataType dataType, uint32_t tileData, bool isDescend, SortKthTileInfo& sortTileInfo)
{
    int64_t realLen = std::min(sortTileInfo.lastAxis, static_cast<int64_t>(tileData));
    std::vector<int64_t> shapeVec = {realLen};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    // The one-core kernel uses Sort's implicit-index overload. sourceOrder is
    // consumed only after Sort to restore signed-zero signs by output index.
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGI("RadixSortTiling", "api of sort shape is %ld, maxUb is %u", realLen, maxValue);
    sortTileInfo.tmpUbSize = maxValue;
    return;
}

bool IsRadixSortOneCore(SortKthTileInfo& sortTileInfo)
{
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        return false;
    }
    uint32_t xUbSize = 0;
    uint32_t y2UbSize = 0;
    if (!ComputeRadixOneCoreUbSizes(sortTileInfo.lastAxis, sortTileInfo.dtypeSize,
                                    static_cast<uint32_t>(sizeof(int32_t)), sortTileInfo.blockUbSize, xUbSize,
                                    y2UbSize)) {
        return false;
    }

    // Sort API writes uint32 indices first. For int64 output, reserve the other half for Cast result.
    const uint32_t halfNum = y2UbSize / static_cast<uint32_t>(sizeof(int32_t));
    if (sortTileInfo.y2DtypeSize == static_cast<uint32_t>(sizeof(int64_t))) {
        y2UbSize = y2UbSize * static_cast<uint32_t>(sizeof(int64_t) / sizeof(int32_t));
    }
    sortTileInfo.keyParams0 = xUbSize;
    sortTileInfo.keyParams1 = y2UbSize;
    sortTileInfo.keyParams2 = halfNum;
    sortTileInfo.keyParams3 = 1;
    int64_t sourceOrderBytes = NeedsSignedZeroSourceOrder(sortTileInfo.dataType) ?
                                   static_cast<int64_t>(halfNum) * sizeof(uint32_t) :
                                   0;
    // keyParams3 records whether the queues can use double buffer after reserving Sort tmp UB.
    int64_t oneBufferQueSize = static_cast<int64_t>(xUbSize) * 2 + static_cast<int64_t>(y2UbSize);
    int64_t remainUb = static_cast<int64_t>(sortTileInfo.ubSize) - oneBufferQueSize - sourceOrderBytes;
    if (remainUb <= static_cast<int64_t>(0)) {
        return false;
    }
    remainUb = (remainUb / static_cast<int64_t>(sortTileInfo.blockUbSize)) *
               static_cast<int64_t>(sortTileInfo.blockUbSize);
    SetSortTmpSize(sortTileInfo.dataType, static_cast<uint32_t>(sortTileInfo.lastAxis), sortTileInfo.isDescend,
                   sortTileInfo);
    int64_t tmpUb = static_cast<int64_t>(sortTileInfo.tmpUbSize);
    OP_LOGI("RadixSortTiling", "remainUb is %ld, tmpUb is %ld", remainUb, tmpUb);
    if (tmpUb > remainUb) {
        return false;
    }

    int64_t doubleBufferRemainUb = static_cast<int64_t>(sortTileInfo.ubSize) - oneBufferQueSize * DOUBLE_BUFFER_NUM -
                                   sourceOrderBytes;
    doubleBufferRemainUb = (doubleBufferRemainUb / static_cast<int64_t>(sortTileInfo.blockUbSize)) *
                           static_cast<int64_t>(sortTileInfo.blockUbSize);
    if (tmpUb <= doubleBufferRemainUb) {
        sortTileInfo.keyParams3 = DOUBLE_BUFFER_NUM;
    }
    OP_LOGI("RadixSortTiling", "radix one-core bufferNum is %u", sortTileInfo.keyParams3);
    return true;
}

// =============================================================================
// Individual strategy Set functions
// =============================================================================
bool IsAxisOneCopy(const SortKthTileInfo& sortTileInfo) { return sortTileInfo.lastAxis == static_cast<int64_t>(1); }

ge::graphStatus SetAxisOneCopyTiling(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    // double buffer
    uint64_t bytesPerElem = static_cast<uint64_t>(2) * (static_cast<uint64_t>(sortTileInfo.dtypeSize) +
                                                        static_cast<uint64_t>(sortTileInfo.y2DtypeSize));
    if (bytesPerElem == 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                              Ops::Base::ToString(sortTileInfo.dataType).c_str(),
                                              "The dtype size of x must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    // This copy-only template has no SIMT scratch. Align the element count so both double-buffered
    // queues remain block-aligned after InitBuffer rounds each allocation.
    uint64_t copyElemsPerLoop64 = sortTileInfo.ubSize / bytesPerElem;
    copyElemsPerLoop64 = copyElemsPerLoop64 / sortTileInfo.blockUbSize * sortTileInfo.blockUbSize;
    if (copyElemsPerLoop64 == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "copyElemsPerLoop",
                                              std::to_string(copyElemsPerLoop64).c_str(),
                                              "The value of copyElemsPerLoop must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    if (copyElemsPerLoop64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "copyElemsPerLoop", std::to_string(copyElemsPerLoop64).c_str(),
            "The value of copyElemsPerLoop must be less than or equal to uint32 max.");
        return ge::GRAPH_FAILED;
    }
    uint32_t copyElemsPerLoop = static_cast<uint32_t>(copyElemsPerLoop64);
    uint64_t totalElems = static_cast<uint64_t>(sortTileInfo.unsortedDim) *
                          static_cast<uint64_t>(sortTileInfo.lastAxis);
    uint64_t loopTimes64 = (totalElems + copyElemsPerLoop64 - 1) / copyElemsPerLoop64;
    if (loopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "loopTimes", std::to_string(loopTimes64).c_str(),
                                              "The value of loopTimes must be less than or equal to uint32 max.");
        return ge::GRAPH_FAILED;
    }
    uint32_t loopTimes = static_cast<uint32_t>(loopTimes64);

    uint32_t coreNumNeed = std::min(sortTileInfo.maxCoreNum, loopTimes);
    sortTileInfo.numTileDataSize = copyElemsPerLoop;
    sortTileInfo.keyParams0 = copyElemsPerLoop;
    sortTileInfo.keyParams1 = loopTimes;
    sortTileInfo.coreNumNeed = coreNumNeed;
    sortTileInfo.unsortedDimParallel = coreNumNeed;
    sortTileInfo.lastDimTileNum = 1;
    sortTileInfo.lastDimNeedCore = 1;
    sortTileInfo.sortLoopTimes = Ops::Base::CeilDiv(static_cast<int64_t>(loopTimes), static_cast<int64_t>(coreNumNeed));
    sortTileInfo.tmpUbSize = 0;

    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    OP_LOGI("AxisOneCopyTiling", "totalElems %lu, copyElemsPerLoop %u, loopTimes %u, coreNumNeed %u", totalElems,
            sortTileInfo.keyParams0, sortTileInfo.keyParams1, coreNumNeed);
    return ge::GRAPH_SUCCESS;
}

void FillSmallAxisBatched(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, const SmallAxisRoutePlan& plan)
{
    sortTileInfo.ubSize = sortTileInfo.ubSize - SIMT_UB; // reserve 32KB for SIMT kernel scratch
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.lastAxis);
    sortTileInfo.keyParams0 = plan.batchSize;                // rows per batch
    sortTileInfo.keyParams1 = plan.batchNum;                 // total batches
    sortTileInfo.keyParams2 = plan.useRankInverse ? 1U : 0U; // enable rank-inverse second pass
    // Unique always sorts a flattened, contiguous input.
    sortTileInfo.keyParams3 = 0U;
    sortTileInfo.keyParams4 = 0U;
    sortTileInfo.coreNumNeed = plan.blockDim;
    sortTileInfo.tmpUbSize = plan.tmpUbSize;
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
}

// Budget one element in each merge window, then derive the number of elements
// each window can process at once via ComputeMergeMoreCoreTiling.
ge::graphStatus SetMergeMoreCoreTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    constexpr uint32_t MERGE_SORT_RECORD_BUFFERS = 2; // input and output sort-record buffers
    uint32_t byteNum = MERGE_SORT_LIST_NUM * MERGE_SORT_DATA_BYTES * MERGE_SORT_RECORD_BUFFERS;
    byteNum += MERGE_SORT_LIST_NUM * static_cast<uint32_t>(sizeof(uint32_t)); // int32 index
    if (info.y2DtypeSize == sizeof(int64_t)) {
        byteNum += MERGE_SORT_LIST_NUM * static_cast<uint32_t>(sizeof(int64_t)); // int64 index extra
    }
    byteNum += MERGE_SORT_LIST_NUM * info.dtypeSize;
    OP_CHECK_IF(!ComputeMergeMoreCoreTiling(context, info, byteNum),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeMergeMoreCoreTiling", "false",
                                                      "The value of ComputeMergeMoreCoreTiling must be true."),
                return ge::GRAPH_FAILED);
    OP_LOGI("[mergeSort]", "maxDealingNum: %u", info.keyParams0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetRadixOneCoreTiling(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    sortTileInfo.lastDimNeedCore = static_cast<uint32_t>(1);
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.lastAxis);
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(1);
    uint64_t sortLoopTimes64 = Ops::Base::CeilDiv(sortTileInfo.unsortedDim,
                                                  static_cast<int64_t>(sortTileInfo.maxCoreNum));
    if (sortLoopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sortLoopTimes",
                                              std::to_string(sortLoopTimes64).c_str(),
                                              "The value of sortLoopTimes must be less than or equal to uint32 max.");
        return ge::GRAPH_FAILED;
    }
    sortTileInfo.sortLoopTimes = static_cast<uint32_t>(sortLoopTimes64);
    if (sortTileInfo.sortLoopTimes > static_cast<uint32_t>(1)) {
        sortTileInfo.coreNumNeed = sortTileInfo.maxCoreNum;
    } else {
        uint32_t core = static_cast<uint32_t>(sortTileInfo.unsortedDim) % sortTileInfo.maxCoreNum;
        sortTileInfo.coreNumNeed = core == uint32_t(0) ? sortTileInfo.maxCoreNum : core;
    }
    sortTileInfo.unsortedDimParallel = sortTileInfo.coreNumNeed;
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetRadixMoreCoreTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    OP_CHECK_IF(!FillRadixMoreCoreInfo(info),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "FillRadixMoreCoreInfo", "false",
                                                      "The value of FillRadixMoreCoreInfo must be true."),
                return ge::GRAPH_FAILED);
    info.ubSize = info.ubSize - SIMT_UB;
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = info.workspaceSize;
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// Fill and print functions
// =============================================================================
void FillTilingDataSort(SortKthTileInfo& info, UniqueSortRegBaseTilingData* sortTilingData)
{
    PlanToTilingData(info, sortTilingData);
    sortTilingData->outputIndexRowBytes = info.outputIndexRowBytes;
    return;
}

void PrintTilingDataSort(gert::TilingContext* context, const SortKthTileInfo& sortTileInfo)
{
    OP_LOGI(context->GetNodeName(),
            "realCoreNum %u, numTileDataSize %u, unsortedDimParallel %u, "
            "lastDimTileNum %u, sortLoopTimes %u, lastDimNeedCore %u, keyParams0 %u, keyParams1 %u "
            "keyParams2 %u, keyParams3 %u, keyParams4 %u, keyParams5 %u, tmpUbSize %u, "
            "lastAxisNum %ld, unsortedDimNum %ld, outerSize %ld, innerSize %ld, innerChunk %u ",
            sortTileInfo.coreNumNeed, sortTileInfo.numTileDataSize, sortTileInfo.unsortedDimParallel,
            sortTileInfo.lastDimTileNum, sortTileInfo.sortLoopTimes, sortTileInfo.lastDimNeedCore,
            sortTileInfo.keyParams0, sortTileInfo.keyParams1, sortTileInfo.keyParams2, sortTileInfo.keyParams3,
            sortTileInfo.keyParams4, sortTileInfo.keyParams5, sortTileInfo.tmpUbSize, sortTileInfo.lastAxis,
            sortTileInfo.unsortedDim, sortTileInfo.outerSize, sortTileInfo.innerSize, sortTileInfo.innerChunk);
    return;
}

ge::graphStatus SetMergeSortTiling(gert::TilingContext* context, SortKthTileInfo& info, bool useUbCapacity = false)
{
    // A rejected candidate falls through to another schedule; it is not a tiling error.
    if (!ComputeMergeSortTiling(context, info, info.y2DtypeSize, useUbCapacity)) {
        return ge::GRAPH_FAILED;
    }
    info.keyParams5 = 0U;
    // This is a measured batching policy, not a dtype/UB capability check. It must match
    // the FP16/int64-only SortBatchedRows instantiation; other types keep row-local proposals.
    if (info.dataType == ge::DT_FLOAT16 && info.y2DtypeSize == sizeof(int64_t) &&
        info.keyParams0 >= SORT_BATCH_MERGE_MIN_ROWS) {
        uint64_t batchElements = static_cast<uint64_t>(info.keyParams0) * info.keyParams3;
        constexpr uint32_t proposalBytes = sizeof(float) + sizeof(uint32_t);
        uint64_t queueBytesPerElement = static_cast<uint64_t>(info.keyParams4) *
                                        (2U * info.dtypeSize + info.y2DtypeSize);
        // Batched merge keeps a B32 index sequence and two 8-byte proposal arrays
        // for every row, in addition to the existing double-buffered I/O queues.
        constexpr uint32_t maxSortElements = std::numeric_limits<uint8_t>::max() * SORT32_SMALL_AXIS_THRESHOLD;
        if (batchElements <= maxSortElements) {
            uint64_t requiredUb = batchElements * (queueBytesPerElement + sizeof(uint32_t) + 2U * proposalBytes);
            if (requiredUb <= info.ubSize) {
                info.keyParams5 = 1U;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetMergeIntraCoreTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    OP_CHECK_IF(!ComputeMergeIntraCoreTiling(context, info),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeMergeIntraCoreTiling", "false",
                                                      "The value of ComputeMergeIntraCoreTiling must be true."),
                return ge::GRAPH_FAILED);
    if (info.lastDimTileNum == SORT_RESIDENT_MERGE_MAX_BLOCKS) {
        constexpr uint32_t proposalBytes = sizeof(float) + sizeof(uint32_t);
        constexpr uint32_t scratchBytes = sizeof(float) + sizeof(uint32_t) + proposalBytes;
        constexpr uint32_t residentBytes = 2U * SORT_RESIDENT_MERGE_MIN_BLOCKS * proposalBytes + scratchBytes;
        constexpr uint32_t maxSortElements = std::numeric_limits<uint8_t>::max() * SORT32_SMALL_AXIS_THRESHOLD;
        uint64_t blockSize = Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(static_cast<uint64_t>(info.lastAxis),
                               static_cast<uint64_t>(SORT_RESIDENT_MERGE_MIN_BLOCKS)),
            static_cast<uint64_t>(SORT32_SMALL_AXIS_THRESHOLD));
        if (blockSize > 0 && blockSize <= maxSortElements && blockSize * residentBytes <= info.ubSize) {
            // Resident two-list merging permits larger initial blocks; the old workspace remains an upper bound.
            info.numTileDataSize = static_cast<uint32_t>(blockSize);
            info.lastDimTileNum = SORT_RESIDENT_MERGE_MIN_BLOCKS;
            info.keyParams3 = info.numTileDataSize * SORT_RESIDENT_MERGE_MIN_BLOCKS;
            info.keyParams5 = static_cast<uint32_t>(INT32_MAX / blockSize);
        }
    }
    OP_LOGI("MergeIntraCoreTiling",
            "B %ld, N %ld, batchPerCore %u, actualCoreNum %u, blockSortSize %u, extractChunkSize %u, "
            "blocksPerRow %u, alignNum %u, ubSize %u",
            info.unsortedDim, info.lastAxis, info.keyParams0, info.coreNumNeed, info.numTileDataSize, info.keyParams4,
            info.lastDimTileNum, info.keyParams3, info.ubSize);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// Try functions
// =============================================================================
bool TrySmallAxis(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    if (sortTileInfo.lastAxis > static_cast<int64_t>(SMALL_AXIS_THRESHOLD)) {
        return false;
    }
    if (IsAxisOneCopy(sortTileInfo)) {
        if (SetAxisOneCopyTiling(context, sortTileInfo) != ge::GRAPH_SUCCESS) {
            return false;
        }
        schId = UNIQUE_SORT_AXIS_ONE_COPY;
        return true;
    }
    SmallAxisRoutePlan smallAxisRoutePlan;
    if (!SelectSmallAxisRoute(sortTileInfo, smallAxisRoutePlan)) {
        return false;
    }
    if (smallAxisRoutePlan.kind == SmallAxisRouteKind::TWO_STAGE) {
        schId = UNIQUE_SORT_SMALL_AXIS_TWO_STAGE;
    } else if (smallAxisRoutePlan.kind == SmallAxisRouteKind::INSERTION) {
        schId = UNIQUE_SORT_SMALL_AXIS_INSERTION;
    } else {
        return false;
    }
    FillSmallAxisBatched(context, sortTileInfo, smallAxisRoutePlan);
    return true;
}

static bool PreferSortMergeMoreCore(const SortKthTileInfo& info)
{
    uint32_t dataSize = 0U;
    if (!IsMergeMoreCoreProfitable(info.dataType, info.lastAxis, info.unsortedDim, info.maxCoreNum) ||
        !SelectMergeMoreCoreDataSize(info.lastAxis, info.unsortedDim, info.maxCoreNum, dataSize)) {
        return false;
    }
    // Native blocks have no local serial merge; keep this path when all rows fit.
    uint64_t coresPerRow = Ops::Base::CeilDiv(static_cast<uint64_t>(info.lastAxis), static_cast<uint64_t>(dataSize));
    if (static_cast<uint64_t>(info.unsortedDim) <= info.maxCoreNum / coresPerRow) {
        return true;
    }
    // Sync blocks must fit all rows in one round. Prefer radix only when its
    // single-round plan gives each row more cores than the sync-merge plan.
    uint32_t syncBlockSize = 0U;
    if (!SelectMergeSyncMergeBlockSize(info.lastAxis, info.unsortedDim, info.maxCoreNum, syncBlockSize)) {
        return false;
    }
    SortKthTileInfo radix = info;
    uint64_t syncCoresPerRow = Ops::Base::CeilDiv(static_cast<uint64_t>(info.lastAxis),
                                                  static_cast<uint64_t>(syncBlockSize));
    return !FillRadixMoreCoreInfo(radix) || radix.sortLoopTimes > 1U || radix.lastDimNeedCore <= syncCoresPerRow;
}

bool TryMerge(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    // Performance thresholds, independent of kernel buffer capacity.
    constexpr int64_t maxFp16MergeAxis = 2048;
    constexpr int64_t maxBf16MergeAxis = 1536;
    bool useB16Merge = (sortTileInfo.dataType == ge::DT_FLOAT16 && sortTileInfo.lastAxis <= maxFp16MergeAxis) ||
                       (sortTileInfo.dataType == ge::DT_BF16 && sortTileInfo.lastAxis <= maxBf16MergeAxis);
    if (useB16Merge || IsMergeSortSupported(sortTileInfo.dataType, sortTileInfo.lastAxis)) {
        SortKthTileInfo candidate = sortTileInfo;
        if (SetMergeSortTiling(context, candidate) == ge::GRAPH_SUCCESS) {
            sortTileInfo = candidate;
            schId = (sortTileInfo.lastAxis <= SORT32_SMALL_AXIS_THRESHOLD) ? UNIQUE_SORT_MERGE_32_SMALL_AXIS :
                                                                             UNIQUE_SORT_MERGE_ONE_CORE;
            return true;
        }
    }

    if (PreferSortMergeMoreCore(sortTileInfo)) {
        SortKthTileInfo candidate = sortTileInfo;
        if (SetMergeMoreCoreTiling(context, candidate) == ge::GRAPH_SUCCESS) {
            sortTileInfo = candidate;
            const bool useDirectSchedule = sortTileInfo.sortLoopTimes == 1U && sortTileInfo.keyParams1 == 0U;
            schId = useDirectSchedule ? UNIQUE_SORT_MERGE_DIRECT : UNIQUE_SORT_MERGE_MORE_CORE;
            // Sort's Merge Path (sch13) requires at least two independent rows.
            // Flattened Unique has one row, so that policy is deliberately not registered.
            return true;
        }
    }

    // Preserve the large-FP32 policy and more-core priority, but keep every schId 0
    // candidate here. Fitting in UB is necessary; it does not replace the throughput policy.
    if (IsMergeIntraCoreSupported(sortTileInfo.dataType, sortTileInfo.lastAxis, sortTileInfo.unsortedDim,
                                  sortTileInfo.maxCoreNum, sortTileInfo.ubSize)) {
        SortKthTileInfo candidate = sortTileInfo;
        if (SetMergeSortTiling(context, candidate, true) == ge::GRAPH_SUCCESS) {
            sortTileInfo = candidate;
            schId = UNIQUE_SORT_MERGE_ONE_CORE;
            return true;
        }
    }
    return false;
}

bool TryRadixOneCore(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    SortKthTileInfo candidate = sortTileInfo;
    if (!IsRadixSortOneCore(candidate) || SetRadixOneCoreTiling(context, candidate) != ge::GRAPH_SUCCESS) {
        return false;
    }
    sortTileInfo = candidate;
    schId = UNIQUE_SORT_RADIX_ONE_CORE;
    return true;
}

bool TryMergeIntraCore(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    if (!IsMergeIntraCoreSupported(sortTileInfo.dataType, sortTileInfo.lastAxis, sortTileInfo.unsortedDim,
                                   sortTileInfo.maxCoreNum, sortTileInfo.ubSize)) {
        return false;
    }
    SortKthTileInfo candidate = sortTileInfo;
    if (SetMergeIntraCoreTiling(context, candidate) != ge::GRAPH_SUCCESS) {
        return false;
    }
    sortTileInfo = candidate;
    schId = UNIQUE_SORT_MERGE_INTRA_CORE;
    return true;
}

// =============================================================================
// Route selection
// =============================================================================
ge::graphStatus SelectSortSchedule(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    if (TrySmallAxis(context, sortTileInfo, schId)) {
        return ge::GRAPH_SUCCESS;
    }
    if (TryMerge(context, sortTileInfo, schId) || TryMergeIntraCore(context, sortTileInfo, schId) ||
        TryRadixOneCore(context, sortTileInfo, schId)) {
        return ge::GRAPH_SUCCESS;
    }

    schId = UNIQUE_SORT_RADIX_MORE_CORE;
    OP_CHECK_IF(SetRadixMoreCoreTiling(context, sortTileInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SetRadixMoreCoreTiling", "GRAPH_FAILED",
                                                      "The value of SetRadixMoreCoreTiling must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SelectExplicitUniqueSchedule(gert::TilingContext* context, SortKthTileInfo& info,
                                                    int requestedSchedule, uint64_t& schedule)
{
    schedule = requestedSchedule;
    switch (requestedSchedule) {
        case UNIQUE_SORT_MERGE_ONE_CORE:
        case UNIQUE_SORT_MERGE_32_SMALL_AXIS:
            return SetMergeSortTiling(context, info);
        case UNIQUE_SORT_RADIX_ONE_CORE:
            return TryRadixOneCore(context, info, schedule) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
        case UNIQUE_SORT_RADIX_MORE_CORE:
            return SetRadixMoreCoreTiling(context, info);
        case UNIQUE_SORT_MERGE_MORE_CORE:
            return SetMergeMoreCoreTiling(context, info);
        case UNIQUE_SORT_MERGE_DIRECT:
            if (SetMergeMoreCoreTiling(context, info) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            return info.sortLoopTimes == 1U && info.keyParams1 == 0U ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
        case UNIQUE_SORT_MERGE_INTRA_CORE:
            return SetMergeIntraCoreTiling(context, info);
        case UNIQUE_SORT_SMALL_AXIS_INSERTION:
        case UNIQUE_SORT_SMALL_AXIS_TWO_STAGE: {
            SmallAxisRoutePlan plan;
            if (!PlanExplicitSmallAxis(info, requestedSchedule == UNIQUE_SORT_SMALL_AXIS_TWO_STAGE, plan)) {
                return ge::GRAPH_FAILED;
            }
            FillSmallAxisBatched(context, info, plan);
            return ge::GRAPH_SUCCESS;
        }
        case UNIQUE_SORT_AXIS_ONE_COPY:
            return info.lastAxis == 1 ? SetAxisOneCopyTiling(context, info) : ge::GRAPH_FAILED;
        default:
            return ge::GRAPH_FAILED; // Non-last layouts do not describe a flattened global Unique.
    }
}

ge::graphStatus PlanUniqueSort(gert::TilingContext* context, int64_t count, ge::DataType dtype, uint32_t ub,
                               uint32_t cores, UniqueSortPlan& result, int requestedSchedule)
{
    if (context == nullptr || count <= 0 || cores == 0 || ub <= SIMT_UB || Ops::Base::GetUbBlockSize(context) == 0) {
        return ge::GRAPH_FAILED;
    }
    SortKthTileInfo info;
    info.lastAxis = count;
    info.unsortedDim = 1;
    info.outerSize = 1;
    info.innerSize = 1;
    info.rank = 1;
    info.sortAxis = 0;
    info.dataType = dtype;
    info.ubSize = ub;
    info.blockUbSize = Ops::Base::GetUbBlockSize(context);
    info.maxCoreNum = cores;
    info.isInt32 = 1;
    info.isDescend = false;
    info.y2DtypeSize = sizeof(int32_t);
    if (!ge::TypeUtils::GetDataTypeLength(dtype, info.dtypeSize)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t schedule = UNIQUE_SORT_MERGE_ONE_CORE;
    // Explicit selection is a C++ planning helper for isolated template tests, not an op attribute.
    const ge::graphStatus status = requestedSchedule < 0 ?
                                       SelectSortSchedule(context, info, schedule) :
                                       SelectExplicitUniqueSchedule(context, info, requestedSchedule, schedule);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    if (schedule == UNIQUE_SORT_RADIX_MORE_CORE) {
        // Generic radix skips original-index GM buffers and borrows y for values ping-pong.
        // Large-input values-only tiling is planned separately by UniqueWithCountsAndSorting.
        // SetRadixMoreCoreTiling already populated the plan and deducted SIMT_UB.
        // Replanning here would reserve SIMT_UB twice and repeat the Sort workspace queries.
        const uint64_t values = Ops::Base::CeilAlign(uint64_t(count) * info.dtypeSize, uint64_t(info.blockUbSize));
        const uint64_t indices = Ops::Base::CeilAlign(uint64_t(count) * info.unsortedDimParallel * sizeof(int32_t),
                                                      uint64_t(info.blockUbSize));
        if (info.workspaceSize < WORK_SPACE_SIZE + values + indices) {
            return ge::GRAPH_FAILED;
        }
        info.workspaceSize -= values + indices;
    }
    result.schedule = schedule;
    result.cores = info.coreNumNeed;
    result.usableUb = info.ubSize;
    result.workspaceBytes = info.workspaceSize > WORK_SPACE_SIZE ? info.workspaceSize - WORK_SPACE_SIZE : 0;
    result.needsIndices = schedule != UNIQUE_SORT_RADIX_ONE_CORE && schedule != UNIQUE_SORT_RADIX_MORE_CORE;
    PlanToTilingData(info, &result.tiling);
    result.tiling.outputIndexRowBytes = info.outputIndexRowBytes;
    return ge::GRAPH_SUCCESS;
}
} // namespace unique_with_counts_and_sorting_sort
