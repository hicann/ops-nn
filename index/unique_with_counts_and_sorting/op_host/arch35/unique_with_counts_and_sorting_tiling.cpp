/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <limits>
#include <vector>

#include "securec.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "util/platform_util.h"

#include "../../../kth_value/op_host/arch35/kth_value_tiling_common.h"
#include "../../op_kernel/arch35/unique_with_counts_and_sorting_tiling_data.h"
#include "../../op_kernel/arch35/unique_with_counts_and_sorting_tiling_key.h"
#include "unique_sort_tiling.h"

namespace optiling {
namespace {
constexpr uint32_t BINS = 256;
constexpr uint32_t SIMT_RESERVE = 32768;
// Per-bin queues: global histogram, block flags and histogram flags (u32/i64),
// plus exclusive sums and block histograms (u16).
constexpr uint32_t BIN_COUNTER_BUFFERS = 3;
constexpr uint32_t BIN_U16_BUFFERS = 2;
constexpr int64_t MIN_ELEMENTS = 1;
constexpr int64_t COMPACT_RADIX_MIN_ELEMENTS = 65536;
constexpr int64_t UC_ONE_CORE_MAX_ELEMENTS = 8192;
// UC writes three 9-element shape records for values and the two private placeholder outputs.
constexpr int64_t SHAPE_STORAGE_ELEMENTS = 27;
constexpr uint32_t CORE_COUNT_STRIDE = 128;
// Keep shared radix metadata aligned when the value count has a partial DMA block.
// Adjacent tiles update histogram/lookback flags independently.
constexpr uint32_t WORKSPACE_REGION_ALIGNMENT = 512;
constexpr uint32_t INDEX_QUEUE_ALIGNMENT_SCALE = 2;
uint64_t DivUp(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return value / divisor + (value % divisor != 0);
}
uint64_t Align(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        return 0;
    }
    return DivUp(value, alignment) * alignment;
}
uint32_t SortTemporary(uint32_t elements)
{
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maximum = 0;
    uint32_t minimum = 0;
    AscendC::GetSortMaxMinTmpSize(ge::Shape(std::vector<int64_t>{elements}), ge::DT_UINT8, ge::DT_UINT32, false, config,
                                  maximum, minimum);
    return maximum;
}
bool PlanUniqueConsecutive(UniqueWithCountsAndSortingTilingData& data, int64_t count, uint32_t cores,
                           uint32_t elementBytes, uint32_t block, uint32_t usable)
{
    if (count <= 0 || cores == 0 || elementBytes == 0 || block == 0) {
        return false;
    }
    constexpr uint32_t UC_VALUE_BUFFERS = 2;
    constexpr uint32_t UC_COUNT_BUFFERS = 2;
    auto& unique = data.unique;
    unique.totalSize = count;
    unique.useCoreNums = cores;
    if (count <= UC_ONE_CORE_MAX_ELEMENTS) {
        const uint64_t singleBytes = Align(uint64_t(count) * elementBytes, block) * UC_VALUE_BUFFERS +
                                     Align(uint64_t(count) * sizeof(int64_t), block) +
                                     Align(uint64_t(count) * sizeof(int32_t), block) +
                                     Align(SHAPE_STORAGE_ELEMENTS * sizeof(uint64_t), block);
        data.uniqueSingleCore = singleBytes <= usable;
    }
    unique.tileLengthPerCore = DivUp(count, cores);
    unique.tileLengthTailCore = count - (cores - 1) * unique.tileLengthPerCore;
    unique.collectingCntBufSize = Align(cores * sizeof(int64_t), block);
    unique.offsetCntBufSize = unique.collectingCntBufSize;
    unique.prevIdxBufSize = block;
    unique.shapeBufSize = Align(SHAPE_STORAGE_ELEMENTS * sizeof(uint64_t), block);
    const uint64_t fixed = unique.collectingCntBufSize + unique.offsetCntBufSize + unique.prevIdxBufSize +
                           unique.shapeBufSize;
    if (fixed >= usable || unique.tileLengthTailCore <= 0) {
        return false;
    }
    const uint64_t tile = (usable - fixed) / (elementBytes * UC_VALUE_BUFFERS + sizeof(int64_t) * UC_COUNT_BUFFERS);
    unique.valueQueueSize = tile * elementBytes / block * block;
    unique.countQueueSize = tile * sizeof(int64_t) / block * block;
    unique.idxQueueSize = tile * sizeof(int64_t) / (block * INDEX_QUEUE_ALIGNMENT_SCALE) *
                          (block * INDEX_QUEUE_ALIGNMENT_SCALE);
    unique.adjUbTileLength = std::min(uint64_t(unique.valueQueueSize) / elementBytes,
                                      uint64_t(unique.countQueueSize) / sizeof(int64_t));
    return unique.adjUbTileLength > 1;
}
ge::graphStatus TilingGenericUnique(gert::TilingContext* context, int64_t count, ge::DataType dtype, uint32_t ub,
                                    uint32_t cores, uint32_t block, uint64_t systemWorkspace)
{
    unique_with_counts_and_sorting_sort::UniqueSortPlan plan;
    if (unique_with_counts_and_sorting_sort::PlanUniqueSort(context, count, dtype, ub, cores, plan) !=
            ge::GRAPH_SUCCESS ||
        plan.cores == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t elementBytes = 0;
    if (!ge::TypeUtils::GetDataTypeLength(dtype, elementBytes) || elementBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    UniqueWithCountsAndSortingTilingData data{};
    data.sort = plan.tiling;
    data.valuesBytes = Align(uint64_t(count) * elementBytes, WORKSPACE_REGION_ALIGNMENT);
    // The final output is dead until UC. Use its capacity for temporary indices if it is wide enough.
    data.indicesBytes = plan.needsIndices && elementBytes < sizeof(int32_t) ?
                            Align(uint64_t(count) * sizeof(int32_t), WORKSPACE_REGION_ALIGNMENT) :
                            0;
    cores = plan.cores;
    // Sort's selected template owns the scratch policy. UC is SIMD-only and reuses the UB after Reset.
    const uint32_t usable = plan.usableUb;
    if (!PlanUniqueConsecutive(data, count, cores, elementBytes, block, usable)) {
        return ge::GRAPH_FAILED;
    }
    auto ws = context->GetWorkspaceSizes(1);
    auto raw = context->GetRawTilingData();
    if (ws == nullptr || raw == nullptr || raw->GetData() == nullptr || raw->GetCapacity() < sizeof(data)) {
        return ge::GRAPH_FAILED;
    }
    ws[0] = systemWorkspace + data.valuesBytes +
            std::max(data.indicesBytes + plan.workspaceBytes, uint64_t(cores) * CORE_COUNT_STRIDE);
    if (memcpy_s(raw->GetData(), raw->GetCapacity(), &data, sizeof(data)) != EOK) {
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(sizeof(data));
    context->SetBlockDim(cores);
    context->SetLocalMemorySize(usable);
    // Match the fused kernel's cross-core Sort-to-UC boundary, including SIMD-only Sort templates.
    context->SetScheduleMode(1);
    context->SetTilingKey(GET_TPL_TILING_KEY(plan.schedule, false));
    return ge::GRAPH_SUCCESS;
}
struct ValuesOnlyRequest {
    int64_t count = 0;
    ge::DataType dtype = ge::DT_UNDEFINED;
};

bool ValidateValuesOnlyRequest(const gert::TilingContext* context, ValuesOnlyRequest& request)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetInputDesc(0) == nullptr) {
        return false;
    }
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    constexpr size_t RETURN_INVERSE_ATTR = 0;
    constexpr size_t RETURN_COUNTS_ATTR = 1;
    constexpr size_t OUT_IDX_ATTR = 3;
    const auto* inverse = attrs->GetAttrPointer<bool>(RETURN_INVERSE_ATTR);
    const auto* counts = attrs->GetAttrPointer<bool>(RETURN_COUNTS_ATTR);
    const auto* indexType = attrs->GetAttrPointer<int64_t>(OUT_IDX_ATTR);
    if (inverse == nullptr || counts == nullptr || indexType == nullptr || *inverse || *counts ||
        *indexType != ge::DT_INT64) {
        return false;
    }
    const auto inputType = context->GetInputDesc(0)->GetDataType();
    if (inputType == ge::DT_DOUBLE || context->GetOutputDesc(0) == nullptr ||
        context->GetOutputDesc(0)->GetDataType() != inputType) {
        return false;
    }
    constexpr size_t FIRST_INDEX_OUTPUT = 1;
    constexpr size_t OUTPUT_COUNT = 3;
    for (size_t output = FIRST_INDEX_OUTPUT; output < OUTPUT_COUNT; ++output) {
        if (context->GetOutputDesc(output) == nullptr ||
            context->GetOutputDesc(output)->GetDataType() != ge::DT_INT64) {
            return false;
        }
    }
    request.count = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    request.dtype = inputType == ge::DT_BOOL ? ge::DT_UINT8 : inputType;
    return request.count >= MIN_ELEMENTS;
}

struct CompactTilePlan {
    uint32_t tile = 0;
    uint32_t tiles = 0;
    uint32_t temporary = 0;
};

bool ComputeCompactTilePlan(int64_t count, uint32_t elementBytes, uint32_t counterBytes, uint32_t usable,
                            uint32_t cores, uint32_t block, CompactTilePlan& plan)
{
    const uint32_t ubExtra = BINS * (counterBytes * BIN_COUNTER_BUFFERS + sizeof(uint16_t) * BIN_U16_BUFFERS);
    if (count <= 0 || elementBytes == 0 || cores == 0 || block == 0 || usable <= ubExtra ||
        (counterBytes != sizeof(uint32_t) && counterBytes != sizeof(int64_t))) {
        return false;
    }
    // Input values, output indices, two byte queues (outValue/inputB8).
    // The second u32 tile queue conservatively reserves inQueueIndex even in
    // values-only instances, which compile that queue out.
    constexpr uint32_t TILE_U32_BUFFERS = 2;
    constexpr uint32_t TILE_U8_BUFFERS = 2;
    const uint32_t tileFactor = elementBytes + sizeof(uint32_t) * TILE_U32_BUFFERS + sizeof(uint8_t) * TILE_U8_BUFFERS;
    uint32_t tile = (usable - ubExtra) / tileFactor / BINS * BINS;
    uint32_t temporary = 0;
    while (tile != 0) {
        temporary = SortTemporary(tile);
        if (temporary != 0 && std::max(temporary, tile * elementBytes) <= usable - ubExtra - tile * tileFactor) {
            break;
        }
        tile -= BINS;
    }
    if (tile == 0) {
        return false;
    }
    uint64_t tiles = DivUp(count, tile);
    // Tile IDs stay uint32 in the shared radix kernel. Leave room for its final core-strided increment.
    const uint64_t maxTiles = std::numeric_limits<uint32_t>::max() - uint64_t(cores);
    if (tiles > maxTiles) {
        return false;
    }
    if (tiles % cores != 0) {
        const uint64_t activeSlots = DivUp(tiles, cores) * cores;
        tile = std::max<uint64_t>(BINS, Align(DivUp(count, activeSlots), BINS));
        temporary = SortTemporary(tile);
    }
    if (tile * tileFactor + ubExtra >= usable) {
        return false;
    }
    const uint32_t remain = usable - ubExtra - tile * tileFactor;
    if (temporary == 0 || temporary > remain) {
        return false;
    }
    uint32_t padding = remain - temporary;
    padding = padding > block ? padding - block : 0;
    temporary += padding / block * block;
    if (temporary < tile * elementBytes) {
        return false;
    }
    plan.tile = tile;
    tiles = DivUp(count, tile);
    if (tiles > maxTiles) {
        return false;
    }
    plan.tiles = static_cast<uint32_t>(tiles);
    plan.temporary = temporary;
    return true;
}

ge::graphStatus TilingUniqueWithCountsAndSorting(gert::TilingContext* context)
{
    ValuesOnlyRequest request;
    if (!ValidateValuesOnlyRequest(context, request)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t count = request.count;
    const auto dtype = request.dtype;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    uint32_t cores = platform.GetCoreNumAiv();
    const uint32_t block = Ops::Base::GetUbBlockSize(context);
    if (cores == 0 || block == 0 || ub <= SIMT_RESERVE || ub > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usable = ub - SIMT_RESERVE;
    if (count < COMPACT_RADIX_MIN_ELEMENTS) {
        return TilingGenericUnique(context, count, dtype, ub, cores, block, platform.GetLibApiWorkSpaceSize());
    }
    uint32_t elementBytes = 0;
    if (!ge::TypeUtils::GetDataTypeLength(dtype, elementBytes) || elementBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    CompactTilePlan tilePlan;
    // Match Sort/KthValue: packed lookback state leaves 30 count bits in the uint32 fast path.
    // Larger inputs use the existing int64 protocol, including 64-bit histogram and scatter positions.
    const bool useInt64Counters = !IsRadixUint32CounterRange(count);
    const uint32_t counterBytes = useInt64Counters ? sizeof(int64_t) : sizeof(uint32_t);
    if (!ComputeCompactTilePlan(count, elementBytes, counterBytes, usable, cores, block, tilePlan)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tile = tilePlan.tile;
    const uint32_t tiles = tilePlan.tiles;
    const uint32_t temporary = tilePlan.temporary;
    cores = std::min(cores, tiles);
    UniqueWithCountsAndSortingTilingData data{};
    auto& sort = data.sort;
    sort.numTileDataSize = tile;
    sort.unsortedDimParallel = 1;
    sort.lastDimTileNum = tiles;
    sort.sortLoopTimes = 1;
    sort.lastDimNeedCore = cores;
    sort.lastAxisNum = count;
    sort.unsortedDimNum = 1;
    sort.tmpUbSize = temporary;
    const uint64_t allHist = uint64_t{BINS} * tiles * elementBytes;
    const uint64_t clearElementsPerCore = std::max<uint64_t>(DivUp(allHist, cores), block);
    if (clearElementsPerCore > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t allBins = BINS * elementBytes;
    sort.keyParams5 = static_cast<uint32_t>(clearElementsPerCore);
    sort.keyParams0 = DivUp(allHist, sort.keyParams5);
    sort.keyParams3 = DivUp(sort.keyParams5, temporary / counterBytes);
    sort.keyParams2 = std::min<uint32_t>(sort.keyParams5, temporary / counterBytes);
    sort.keyParams4 = std::max<uint64_t>(DivUp(allBins, cores), block);
    sort.keyParams1 = DivUp(allBins, sort.keyParams4);
    const uint64_t binsBytes = Align(uint64_t(sort.keyParams1) * sort.keyParams4 * counterBytes, block);
    const uint64_t histBytes = Align(uint64_t(sort.keyParams3) * sort.keyParams2 * sort.keyParams0 * counterBytes,
                                     block);
    data.valuesBytes = Align(uint64_t(count) * elementBytes, WORKSPACE_REGION_ALIGNMENT);
    // UC chooses its own single/multi-core family after Sort has synchronized all producers.
    if (!PlanUniqueConsecutive(data, count, cores, elementBytes, block, usable)) {
        return ge::GRAPH_FAILED;
    }
    auto ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr || context->GetRawTilingData() == nullptr || context->GetRawTilingData()->GetData() == nullptr ||
        context->GetRawTilingData()->GetCapacity() < sizeof(data)) {
        return ge::GRAPH_FAILED;
    }
    // Cache the radix byte and per-tile histogram/prefix between the two passes.
    // Recomputing these in the scatter pass saves workspace but adds vector work on every radix round.
    constexpr uint32_t HISTOGRAM_AND_PREFIX_BUFFERS = 2;
    const uint64_t tileMetadataBytes = uint64_t(tiles) * BINS * sizeof(uint16_t) * HISTOGRAM_AND_PREFIX_BUFFERS;
    const uint64_t radixBytes = Align(uint64_t(tiles) * tile, block);
    ws[0] = platform.GetLibApiWorkSpaceSize() + data.valuesBytes +
            std::max<uint64_t>(binsBytes + histBytes + tileMetadataBytes + radixBytes, cores * CORE_COUNT_STRIDE);
    auto raw = context->GetRawTilingData();
    if (memcpy_s(raw->GetData(), raw->GetCapacity(), &data, sizeof(data)) != EOK) {
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(sizeof(data));
    context->SetBlockDim(cores);
    // SIMT validates UB addresses against this runtime limit, in addition to the Host allocation budget.
    context->SetLocalMemorySize(usable);
    context->SetScheduleMode(1);
    // Compact and generic plans use the same values-only radix kernel; only their tiling data differs.
    context->SetTilingKey(GET_TPL_TILING_KEY(UNIQUE_SORT_RADIX_MORE_CORE, useInt64Counters));
    return ge::GRAPH_SUCCESS;
}
} // namespace
struct UniqueWithCountsAndSortingCompileInfo {};
static ge::graphStatus ParseUniqueWithCountsAndSorting(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(UniqueWithCountsAndSorting)
    .Tiling(TilingUniqueWithCountsAndSorting)
    .TilingParse<UniqueWithCountsAndSortingCompileInfo>(ParseUniqueWithCountsAndSorting);
} // namespace optiling
