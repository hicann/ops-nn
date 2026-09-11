/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kth_value_tiling_arch35.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/kth_value_tiling_data.h"
#include "../../op_kernel/arch35/kth_value_tiling_key.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "kth_value_tiling_common.h"

namespace optiling {
constexpr uint32_t DEFAULT_OUTPUT_ROWS = 1024;
constexpr uint32_t MEDIAN_RADIX_COUNT_STORAGE_WORDS = 8;
constexpr int64_t LOWER_MEDIAN_DIVISOR = 2;

static ge::graphStatus CheckKthValueDtypes(gert::TilingContext* context, ge::DataType dataType, uint32_t& dtypeSize)
{
    if (!ge::TypeUtils::GetDataTypeLength(dataType, dtypeSize)) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(dataType).c_str(),
                                  "INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT, FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }
    auto valuesDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, valuesDesc);
    auto indicesDesc = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    OP_CHECK_IF(valuesDesc->GetDataType() != dataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "x, values",
                    (Ops::Base::ToString(dataType) + ", " + Ops::Base::ToString(valuesDesc->GetDataType())).c_str(),
                    "The dtypes of x and values must be the same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesDesc->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "indices",
                                          Ops::Base::ToString(indicesDesc->GetDataType()).c_str(), "INT64"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateKthValueShapes(gert::TilingContext* context, const gert::Shape*& xStorageShape)
{
    auto xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto valuesShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, valuesShapePtr);
    auto indicesShapePtr = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShapePtr);
    OP_CHECK_IF(
        xShape->GetStorageShape().GetShapeSize() == 0 || valuesShapePtr->GetStorageShape().GetShapeSize() == 0 ||
            indicesShapePtr->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x, values, indices",
                                              (std::to_string(xShape->GetStorageShape().GetShapeSize()) + ", " +
                                               std::to_string(valuesShapePtr->GetStorageShape().GetShapeSize()) + ", " +
                                               std::to_string(indicesShapePtr->GetStorageShape().GetShapeSize()))
                                                  .c_str(),
                                              "The values of shape sizes of x, values, and indices must be positive."),
        return ge::GRAPH_FAILED);
    xStorageShape = &xShape->GetStorageShape();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseKthValueShapeInfo(gert::TilingContext* context, const int64_t* kAttr,
                                              const int64_t* dimAttr, SortKthTileInfo& info)
{
    const gert::Shape* xStorageShape = nullptr;
    if (ValidateKthValueShapes(context, xStorageShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    info.rank = xStorageShape->GetDimNum();
    if (info.rank <= 0) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", (std::to_string(info.rank) + "D").c_str(),
                                                 "The shape dim of x must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    int64_t originSortAxis = (dimAttr == nullptr) ? -1 : *dimAttr;
    info.sortAxis = originSortAxis < 0 ? originSortAxis + info.rank : originSortAxis;
    if (info.sortAxis < 0 || info.sortAxis >= info.rank) {
        std::string dimValue = std::to_string(originSortAxis);
        std::string dimRange = "[" + std::to_string(-info.rank) + ", " + std::to_string(info.rank - 1) + "]";
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "dim", dimValue.c_str(), dimRange.c_str());
        return ge::GRAPH_FAILED;
    }
    info.lastAxis = xStorageShape->GetDim(info.sortAxis);
    if (info.lastAxis <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(info.lastAxis).c_str(),
                                              "The value of sort axis of x must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    if (*kAttr < 1 || *kAttr > info.lastAxis) {
        std::string kValue = std::to_string(*kAttr);
        std::string kRange = "[1, " + std::to_string(info.lastAxis) + "]";
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "k", kValue.c_str(), kRange.c_str());
        return ge::GRAPH_FAILED;
    }
    ComputeAxisDimProducts(*xStorageShape, info.sortAxis, info);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// UB computation and base tiling init
// =============================================================================
static ge::graphStatus ComputeKthValueUbInfo(gert::TilingContext* context,
                                             const platform_ascendc::PlatformAscendC& ascendcPlatform,
                                             SortKthTileInfo& info, bool& oneCoreUbValid)
{
    uint64_t ubSize64 = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize64);
    OP_CHECK_IF(
        (ubSize64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ubSize64).c_str(),
                                              "The value of ubSize must be less than or equal to uint32 max."),
        return ge::GRAPH_FAILED);
    info.ubSize = static_cast<uint32_t>(ubSize64);
    info.isInt32 = static_cast<uint32_t>(IsRadixUint32CounterRange(info.lastAxis));
    oneCoreUbValid = ComputeRadixOneCoreUbSizes(info.lastAxis, info.dtypeSize, static_cast<uint32_t>(sizeof(uint32_t)),
                                                info.blockUbSize, info.xUbSize, info.idxUbSize) &&
                     (info.isInt32 != 0U);
    info.outputRowsPerLoop = static_cast<uint32_t>(std::min<int64_t>(DEFAULT_OUTPUT_ROWS, info.unsortedDim));
    uint32_t compactValueSize = Ops::Base::CeilAlign(info.outputRowsPerLoop * info.dtypeSize, info.blockUbSize);
    uint32_t compactIndexSize = Ops::Base::CeilAlign(info.outputRowsPerLoop * static_cast<uint32_t>(sizeof(int64_t)),
                                                     info.blockUbSize);
    if (oneCoreUbValid) {
        // One pipeline slot owns input, sorted value/index and compact value/index buffers.
        // SetRadixOneCoreTiling doubles this complete footprint when bufferNum is 2.
        info.oneBufferQueSize = static_cast<uint64_t>(info.xUbSize) * 2 + info.idxUbSize + compactValueSize +
                                compactIndexSize;
        QuerySortTmpSizeRadix(info.dataType, static_cast<uint32_t>(info.lastAxis), info.oneCoreTmpUbSize);
    } else {
        info.oneCoreTmpUbSize = std::numeric_limits<uint32_t>::max();
    }
    return ge::GRAPH_SUCCESS;
}

static void InitKthValueBaseTiling(KthValueTilingData* tilingData, const SortKthTileInfo& info, bool oneCoreUbValid,
                                   int64_t kthIndex)
{
    PlanToTilingData(info, tilingData);
    tilingData->numTileDataSize = oneCoreUbValid ? static_cast<uint32_t>(info.lastAxis) : 0U;
    tilingData->lastDimTileNum = 1;
    tilingData->lastDimNeedCore = 1;
    tilingData->keyParams0 = info.xUbSize;
    tilingData->keyParams1 = info.idxUbSize;
    tilingData->keyParams2 = 0;
    tilingData->keyParams3 = 1;
    tilingData->keyParams4 = info.outputRowsPerLoop;
    tilingData->keyParams5 = 0;
    tilingData->kthIndex = kthIndex;
    tilingData->outerSize = info.unsortedDim;
    tilingData->innerSize = 1;
    tilingData->innerLoopNum = 0;
    tilingData->innerChunk = 0;
    tilingData->inputRowBytes = 0;
    tilingData->valueAxisBytes = 0;
    tilingData->indexAxisBytes = 0;
}

// =============================================================================
// Non-last small axis helpers
// =============================================================================
static bool CheckNonLastSmallAxisInput(int64_t axisLen, int64_t outerSize, int64_t innerSize, uint32_t& axisLen32)
{
    if (axisLen < NON_LAST_SMALL_AXIS_MIN_AXIS_LEN || axisLen > NON_LAST_SMALL_AXIS_THRESHOLD || outerSize <= 0 ||
        innerSize <= 0) {
        return false;
    }
    axisLen32 = static_cast<uint32_t>(axisLen);
    return true;
}

static bool TryComputeKthNonLastSmallAxisLayout(const SortKthTileInfo& info, uint32_t innerChunk, uint32_t sortCount,
                                                bool useMergeSort, uint32_t& inputRowBytes, uint32_t& valueAxisBytes,
                                                uint32_t& indexAxisBytes)
{
    uint32_t sortDtypeSize = GetNonLastSortDtypeSize(info.dtypeSize, useMergeSort, info.dataType);
    if (!CeilAlignUint32(static_cast<uint64_t>(innerChunk) * info.dtypeSize, info.blockUbSize, inputRowBytes) ||
        !CeilAlignUint32(static_cast<uint64_t>(sortCount) * sortDtypeSize, info.blockUbSize, valueAxisBytes) ||
        !CeilAlignUint32(static_cast<uint64_t>(sortCount) * sizeof(uint32_t), info.blockUbSize, indexAxisBytes)) {
        return false;
    }
    if (useMergeSort) {
        uint32_t sortStructBytes = 0;
        if (!CeilAlignUint32(static_cast<uint64_t>(sortCount) * SORT_STRUCT_BYTES, info.blockUbSize, sortStructBytes)) {
            return false;
        }
        valueAxisBytes = std::max(valueAxisBytes, sortStructBytes);
    }
    return true;
}

static bool TryComputeKthNonLastSmallAxisCastBytes(const SortKthTileInfo& info, uint32_t innerChunk, uint32_t sortCount,
                                                   bool useMergeSort, uint32_t sortDtypeSize, uint64_t& inputCastBytes,
                                                   uint64_t& compactCastBytes)
{
    inputCastBytes = 0U;
    compactCastBytes = 0U;
    if (!useMergeSort || info.dataType != ge::DT_BF16) {
        return true;
    }
    uint32_t inputValueAxisBytes = 0U;
    if (!CeilAlignUint32(static_cast<uint64_t>(sortCount) * info.dtypeSize, info.blockUbSize, inputValueAxisBytes)) {
        return false;
    }
    inputCastBytes = static_cast<uint64_t>(innerChunk) * inputValueAxisBytes;
    compactCastBytes = Ops::Base::CeilAlign<uint64_t>(static_cast<uint64_t>(innerChunk) * sortDtypeSize,
                                                      info.blockUbSize);
    return compactCastBytes != 0U;
}

static bool ComputeKthNonLastSmallAxisPeakUb(const SortKthTileInfo& info, uint32_t innerChunk, uint32_t sortCount,
                                             bool useMergeSort, uint64_t& peakUb, NonLastSmallAxisCandidate& plan)
{
    if (innerChunk == 0U) {
        return false;
    }
    uint32_t sortDtypeSize = GetNonLastSortDtypeSize(info.dtypeSize, useMergeSort, info.dataType);
    if (sortDtypeSize == 0U || info.dtypeSize == 0U) {
        return false;
    }
    if (!TryComputeKthNonLastSmallAxisLayout(info, innerChunk, sortCount, useMergeSort, plan.inputRowBytes,
                                             plan.valueAxisBytes, plan.indexAxisBytes)) {
        return false;
    }
    uint64_t inputRowElems = static_cast<uint64_t>(plan.inputRowBytes) / info.dtypeSize;
    uint64_t valueAxisElems = static_cast<uint64_t>(plan.valueAxisBytes) / sortDtypeSize;
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    if (info.dtypeSize <= sizeof(uint16_t) &&
        ((static_cast<uint64_t>(axisLen) - 1U) * inputRowElems > std::numeric_limits<uint16_t>::max() ||
         static_cast<uint64_t>(innerChunk - 1U) * valueAxisElems > std::numeric_limits<uint16_t>::max())) {
        return false;
    }

    uint64_t inputCastBytes = 0;
    uint64_t compactCastBytes = 0;
    if (!TryComputeKthNonLastSmallAxisCastBytes(info, innerChunk, sortCount, useMergeSort, sortDtypeSize,
                                                inputCastBytes, compactCastBytes)) {
        return false;
    }
    uint64_t compactValueBytes = Ops::Base::CeilAlign<uint64_t>(static_cast<uint64_t>(innerChunk) * info.dtypeSize,
                                                                info.blockUbSize);
    uint64_t compactIndexBytes = Ops::Base::CeilAlign<uint64_t>(static_cast<uint64_t>(innerChunk) * sizeof(int64_t),
                                                                info.blockUbSize);
    if (compactValueBytes == 0U || compactIndexBytes == 0U) {
        return false;
    }
    uint64_t inputTileBytes = static_cast<uint64_t>(axisLen) * plan.inputRowBytes;
    if (!useMergeSort && NeedsSignedZeroSourceOrder(info.dataType)) {
        inputTileBytes += plan.indexAxisBytes;
    }
    peakUb = inputTileBytes + inputCastBytes + static_cast<uint64_t>(innerChunk) * plan.valueAxisBytes * 2U +
             static_cast<uint64_t>(innerChunk) * plan.indexAxisBytes + compactValueBytes + compactCastBytes +
             compactIndexBytes + static_cast<uint64_t>(info.tmpUbSize);
    return true;
}

// =============================================================================
// Individual strategy Set functions
// =============================================================================
static ge::graphStatus SetRadixOneCoreTiling(gert::TilingContext* context, const SortKthTileInfo& info,
                                             KthValueTilingData* tilingData)
{
    uint64_t sourceOrderBytes = NeedsSignedZeroSourceOrder(info.dataType) ? info.idxUbSize : 0U;
    OP_CHECK_IF(
        (info.oneBufferQueSize + sourceOrderBytes >= info.ubSize),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(info.ubSize).c_str(),
                                              "The value of ubSize must be greater than oneBufferQueSize."),
        return ge::GRAPH_FAILED);
    tilingData->numTileDataSize = static_cast<uint32_t>(info.lastAxis);
    tilingData->lastDimTileNum = 1;
    tilingData->lastDimNeedCore = 1;
    tilingData->keyParams0 = info.xUbSize;
    tilingData->keyParams1 = info.idxUbSize;
    tilingData->keyParams2 = 0;
    tilingData->keyParams3 = 1;
    tilingData->keyParams4 = info.outputRowsPerLoop;
    tilingData->keyParams5 = 0;
    OP_CHECK_IF(!QuerySortTmpSizeRadix(info.dataType, static_cast<uint32_t>(info.lastAxis), tilingData->tmpUbSize),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "QuerySortTmpSizeRadix", "false",
                                                      "The value of QuerySortTmpSizeRadix must be true."),
                return ge::GRAPH_FAILED);
    uint64_t remainUb = (info.ubSize - info.oneBufferQueSize - sourceOrderBytes) / info.blockUbSize * info.blockUbSize;
    OP_CHECK_IF((static_cast<uint64_t>(tilingData->tmpUbSize) > remainUb),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tmpUbSize",
                                                      std::to_string(tilingData->tmpUbSize).c_str(),
                                                      "The value of tmpUbSize must be less than or equal to remainUb."),
                return ge::GRAPH_FAILED);
    uint64_t doubleBufferUsedUb = info.oneBufferQueSize * DOUBLE_BUFFER_NUM + sourceOrderBytes;
    uint64_t doubleBufferRemainUb = info.ubSize > doubleBufferUsedUb ?
                                        (info.ubSize - doubleBufferUsedUb) / info.blockUbSize * info.blockUbSize :
                                        0;
    if (static_cast<uint64_t>(tilingData->tmpUbSize) <= doubleBufferRemainUb) {
        tilingData->keyParams3 = DOUBLE_BUFFER_NUM;
    }
    return ge::GRAPH_SUCCESS;
}

static bool ComputeKthValueRadixMoreCoreWorkspace(int64_t axisLen, uint32_t dtypeSize, uint32_t indexSize,
                                                  uint32_t unsortedDimParallel, uint32_t blockUbSize,
                                                  uint64_t sortWorkspaceSize, uint64_t& workspaceSize)
{
    uint64_t axisLen64 = static_cast<uint64_t>(axisLen);
    uint64_t unsortedDimParallel64 = static_cast<uint64_t>(unsortedDimParallel);
    uint64_t blockUbSize64 = static_cast<uint64_t>(blockUbSize);

    uint64_t valueWorkspace = Ops::Base::CeilAlign(axisLen64 * unsortedDimParallel64 * static_cast<uint64_t>(dtypeSize),
                                                   blockUbSize64);
    uint64_t indexWorkspace = Ops::Base::CeilAlign(axisLen64 * unsortedDimParallel64 * static_cast<uint64_t>(indexSize),
                                                   blockUbSize64);

    workspaceSize = valueWorkspace + indexWorkspace + sortWorkspaceSize;
    return true;
}

static ge::graphStatus SetRadixMoreCoreTiling(gert::TilingContext* context, SortKthTileInfo& info, uint32_t& blockDim)
{
    OP_CHECK_IF(!FillRadixMoreCoreInfo(info),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "FillRadixMoreCoreInfo", "false",
                                                      "The value of FillRadixMoreCoreInfo must be true."),
                return ge::GRAPH_FAILED);
    blockDim = info.coreNumNeed;
    uint32_t indexSize = info.isInt32 != 0 ? static_cast<uint32_t>(sizeof(int32_t)) :
                                             static_cast<uint32_t>(sizeof(int64_t));
    uint64_t totalWorkspace = 0;
    OP_CHECK_IF(
        !ComputeKthValueRadixMoreCoreWorkspace(info.lastAxis, info.dtypeSize, indexSize, info.unsortedDimParallel,
                                               info.blockUbSize, static_cast<uint64_t>(info.workspaceSize),
                                               totalWorkspace),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeKthValueRadixMoreCoreWorkspace", "false",
                                              "The value of ComputeKthValueRadixMoreCoreWorkspace must be true."),
        return ge::GRAPH_FAILED);
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    userWorkspaceSize[0] = static_cast<size_t>(totalWorkspace);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetKthValueMergeSortTiling(gert::TilingContext* context, SortKthTileInfo& info,
                                                  uint32_t& blockDim, uint64_t& schId)
{
    OP_CHECK_IF(!ComputeMergeSortTiling(context, info, static_cast<uint32_t>(sizeof(uint32_t))),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeMergeSortTiling", "false",
                                                      "The value of ComputeMergeSortTiling must be true."),
                return ge::GRAPH_FAILED);
    blockDim = info.coreNumNeed;
    schId = info.lastAxis <= SORT32_SMALL_AXIS_THRESHOLD ? KTH_VALUE_SCHID_SORT32_SMALL_AXIS :
                                                           KTH_VALUE_SCHID_MERGE_SORT;
    OP_LOGI("KthValueMergeSortTiling", "axis=%ld, unsortedDim=%ld, coreNumNeed=%u, schId=%lu", info.lastAxis,
            info.unsortedDim, info.coreNumNeed, schId);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetMergeMoreCoreTiling(gert::TilingContext* context, SortKthTileInfo& info, uint32_t& blockDim)
{
    constexpr uint32_t mergeBytesPerElem = MERGE_SORT_LIST_NUM * MERGE_SORT_DATA_BYTES * 2 +
                                           MERGE_SORT_LIST_NUM * sizeof(uint32_t) +
                                           MERGE_SORT_LIST_NUM * sizeof(int64_t) + MERGE_SORT_LIST_NUM * sizeof(float);
    OP_CHECK_IF(!ComputeMergeMoreCoreTiling(context, info, mergeBytesPerElem),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeMergeMoreCoreTiling", "false",
                                                      "The value of ComputeMergeMoreCoreTiling must be true."),
                return ge::GRAPH_FAILED);
    blockDim = info.coreNumNeed;
    OP_LOGI("KthValueMergeMoreCoreTiling", "maxDealingNum: %u", info.keyParams0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetMergeIntraCoreTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    OP_CHECK_IF(!ComputeMergeIntraCoreTiling(context, info),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeMergeIntraCoreTiling", "false",
                                                      "The value of ComputeMergeIntraCoreTiling must be true."),
                return ge::GRAPH_FAILED);
    OP_LOGI("KthValueMergeIntraCoreTiling",
            "B %ld, N %ld, batchPerCore %u, actualCoreNum %u, blockSortSize %u, extractChunkSize %u, "
            "blocksPerRow %u, alignNum %u, ubSize %u",
            info.unsortedDim, info.lastAxis, info.keyParams0, info.coreNumNeed, info.numTileDataSize, info.keyParams4,
            info.lastDimTileNum, info.keyParams3, info.ubSize);
    return ge::GRAPH_SUCCESS;
}

static bool FillNonLastSmallAxisTiling(gert::TilingContext* context, const SortKthTileInfo& info,
                                       const NonLastSmallAxisCandidate& plan, uint32_t axisLen, uint32_t sortCount,
                                       uint32_t tmpUbSize, bool useMergeSort, KthValueTilingData* tilingData,
                                       uint32_t& blockDim, uint64_t& schId)
{
    uint32_t inputValueAxisBytes = 0U;
    if (useMergeSort && info.dataType == ge::DT_BF16 &&
        !CeilAlignUint32(static_cast<uint64_t>(sortCount) * info.dtypeSize, info.blockUbSize, inputValueAxisBytes)) {
        return false;
    }
    tilingData->lastAxisNum = info.lastAxis;
    tilingData->unsortedDimNum = info.outerSize * info.innerSize;
    tilingData->outerSize = info.outerSize;
    tilingData->innerSize = info.innerSize;
    tilingData->innerLoopNum = plan.innerLoopNum;
    tilingData->innerChunk = plan.innerChunk;
    tilingData->inputRowBytes = plan.inputRowBytes;
    tilingData->valueAxisBytes = plan.valueAxisBytes;
    tilingData->indexAxisBytes = plan.indexAxisBytes;
    tilingData->numTileDataSize = axisLen;
    tilingData->tmpUbSize = tmpUbSize;
    tilingData->keyParams0 = sortCount;
    tilingData->keyParams1 = inputValueAxisBytes;
    blockDim = plan.activeCore;
    schId = useMergeSort ? KTH_VALUE_SCHID_NON_LAST_SMALL_AXIS : KTH_VALUE_SCHID_NON_LAST_SMALL_AXIS_RADIX;
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    userWorkspaceSize[0] = WORK_SPACE_SIZE;
    return true;
}

static bool SetNonLastSmallAxisTiling(gert::TilingContext* context, const SortKthTileInfo& constInfo,
                                      KthValueTilingData* tilingData, uint32_t& blockDim, uint64_t& schId)
{
    uint32_t axisLen32 = 0;
    if (!CheckNonLastSmallAxisInput(constInfo.lastAxis, constInfo.outerSize, constInfo.innerSize, axisLen32)) {
        return false;
    }
    uint64_t usableUb = ComputeUbAfterSimtReserve(constInfo.ubSize);
    bool useMergeSort = UseNonLastMergeSort(constInfo.dataType, axisLen32);
    uint32_t sortCount = GetNonLastSortCount(constInfo.dataType, axisLen32);
    uint32_t tmpUbSize = 0;
    bool gotTmpSize = useMergeSort ? GetNonLastSortTmpSize(constInfo.dataType, sortCount, true, false, tmpUbSize) :
                                     QuerySortTmpSizeRadix(constInfo.dataType, sortCount, tmpUbSize);
    if (!gotTmpSize) {
        return false;
    }
    SortKthTileInfo info = constInfo;
    info.tmpUbSize = tmpUbSize;
    NonLastSmallAxisCandidate best;
    auto estimateUb = [sortCount, useMergeSort](SortKthTileInfo& candidateInfo, uint32_t innerChunk, uint64_t& peakUb,
                                                NonLastSmallAxisCandidate& candidate) -> bool {
        return ComputeKthNonLastSmallAxisPeakUb(candidateInfo, innerChunk, sortCount, useMergeSort, peakUb, candidate);
    };
    if (!SearchNonLastSmallAxisPlan(info, usableUb, estimateUb, best)) {
        OP_LOGI(context->GetNodeName(), "kth_value non-last small-axis no valid inner chunk.");
        return false;
    }
    return FillNonLastSmallAxisTiling(context, constInfo, best, axisLen32, sortCount, tmpUbSize, useMergeSort,
                                      tilingData, blockDim, schId);
}

// =============================================================================
// Fill functions
// =============================================================================
static void FillSmallAxisTiling(KthValueTilingData* tilingData, const SmallAxisRoutePlan& plan, uint32_t axisLen,
                                uint32_t& blockDim)
{
    tilingData->numTileDataSize = axisLen;
    tilingData->keyParams0 = plan.batchSize;
    tilingData->keyParams1 = plan.batchNum;
    tilingData->keyParams2 = plan.useRankInverse ? 1U : 0U;
    tilingData->keyParams3 = 0;
    tilingData->keyParams4 = 0;
    tilingData->keyParams5 = 0;
    tilingData->tmpUbSize = plan.tmpUbSize;
    tilingData->unsortedDimParallel = plan.blockDim;
    tilingData->sortLoopTimes = plan.batchNum;
    tilingData->lastDimTileNum = 1;
    tilingData->lastDimNeedCore = 1;
    blockDim = plan.blockDim;
}

static bool FillNonLastSmallAxisTiling(KthValueTilingData* tilingData, const SmallAxisRoutePlan& plan,
                                       const SortKthTileInfo& info, uint32_t& blockDim)
{
    uint32_t innerChunk = static_cast<uint32_t>(std::min<int64_t>(plan.batchSize, info.innerSize));
    if (innerChunk == 0U) {
        return false;
    }
    uint64_t innerLoop = Ops::Base::CeilDiv(static_cast<uint64_t>(info.innerSize), static_cast<uint64_t>(innerChunk));
    uint64_t batchNum = static_cast<uint64_t>(info.outerSize) * innerLoop;
    uint32_t inputRowBytes = 0;
    if (batchNum == 0U || batchNum > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        !CeilAlignUint32(static_cast<uint64_t>(innerChunk) * info.dtypeSize, info.blockUbSize, inputRowBytes)) {
        return false;
    }
    tilingData->numTileDataSize = static_cast<uint32_t>(info.lastAxis);
    tilingData->keyParams0 = innerChunk;
    tilingData->keyParams1 = static_cast<uint32_t>(batchNum);
    tilingData->keyParams2 = plan.useRankInverse ? 1U : 0U;
    tilingData->keyParams3 = 0;
    tilingData->keyParams4 = 0;
    tilingData->keyParams5 = 0;
    tilingData->tmpUbSize = plan.tmpUbSize;
    tilingData->unsortedDimParallel = std::min(info.maxCoreNum, static_cast<uint32_t>(batchNum));
    tilingData->sortLoopTimes = static_cast<uint32_t>(batchNum);
    tilingData->lastDimTileNum = 1;
    tilingData->lastDimNeedCore = 1;
    tilingData->outerSize = info.outerSize;
    tilingData->innerSize = info.innerSize;
    tilingData->innerLoopNum = static_cast<uint32_t>(innerLoop);
    tilingData->innerChunk = innerChunk;
    tilingData->inputRowBytes = inputRowBytes;
    blockDim = tilingData->unsortedDimParallel;
    return blockDim > 0U;
}

// =============================================================================
// Axis-one-copy tiling
// =============================================================================
static ge::graphStatus SetAxisOneCopyTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    uint64_t bytesPerElem = static_cast<uint64_t>(2) *
                            (static_cast<uint64_t>(info.dtypeSize) + static_cast<uint64_t>(sizeof(int64_t)));
    if (bytesPerElem == 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(info.dataType).c_str(),
                                              "The dtype size of x must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    uint64_t copyElemsPerLoop64 = static_cast<uint64_t>(info.ubSize) / bytesPerElem;
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
    uint64_t totalElems = static_cast<uint64_t>(info.unsortedDim) * static_cast<uint64_t>(info.lastAxis);
    uint64_t loopTimes64 = (totalElems + copyElemsPerLoop64 - 1) / copyElemsPerLoop64;
    if (loopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "loopTimes", std::to_string(loopTimes64).c_str(),
                                              "The value of loopTimes must be less than or equal to uint32 max.");
        return ge::GRAPH_FAILED;
    }
    uint32_t loopTimes = static_cast<uint32_t>(loopTimes64);

    uint32_t coreNumNeed = std::min(info.maxCoreNum, loopTimes);
    info.numTileDataSize = copyElemsPerLoop;
    info.keyParams0 = copyElemsPerLoop;
    info.keyParams1 = loopTimes;
    info.coreNumNeed = coreNumNeed;
    info.unsortedDimParallel = coreNumNeed;
    info.lastDimTileNum = 1;
    info.lastDimNeedCore = 1;
    info.sortLoopTimes = Ops::Base::CeilDiv(static_cast<int64_t>(loopTimes), static_cast<int64_t>(coreNumNeed));
    info.tmpUbSize = 0;

    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    userWorkSpaceSize[0] = WORK_SPACE_SIZE;
    OP_LOGI("AxisOneCopyTiling", "totalElems %lu, copyElemsPerLoop %u, loopTimes %u, coreNumNeed %u", totalElems,
            info.keyParams0, info.keyParams1, coreNumNeed);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// Try functions
// =============================================================================
static bool TryRadixOneCore(gert::TilingContext* context, const SortKthTileInfo& info, KthValueTilingData* tilingData,
                            uint64_t& schId)
{
    uint64_t sourceOrderBytes = NeedsSignedZeroSourceOrder(info.dataType) ? info.idxUbSize : 0U;
    if (info.oneBufferQueSize + sourceOrderBytes >= info.ubSize) {
        return false;
    }
    uint64_t remainUb = (info.ubSize - info.oneBufferQueSize - sourceOrderBytes) / info.blockUbSize * info.blockUbSize;
    if (static_cast<uint64_t>(info.oneCoreTmpUbSize) > remainUb) {
        return false;
    }
    KthValueTilingData candidate = *tilingData;
    if (SetRadixOneCoreTiling(context, info, &candidate) != ge::GRAPH_SUCCESS) {
        return false;
    }
    *tilingData = candidate;
    schId = KTH_VALUE_SCHID_RADIX_ONE_CORE;
    return true;
}

static bool IsRadixSelectProfitable(const SortKthTileInfo& info, int64_t kthIndex)
{
    if (kthIndex < 0 || info.lastAxis <= 0 || info.unsortedDim <= 0) {
        return false;
    }
    uint64_t axisLen = static_cast<uint64_t>(info.lastAxis);
    uint64_t kth = static_cast<uint64_t>(kthIndex);
    if (kth >= axisLen) {
        return false;
    }

    constexpr uint64_t radixSelectMinAxis = 65536UL;
    constexpr uint64_t radixSelectHugeAxis = 1000000UL;
    constexpr uint64_t radixSelectInt64SmallAxis = 4096UL;
    constexpr int64_t radixSelectInt32InteriorMinRows = 16;
    constexpr int64_t radixSelectInt64InteriorMinRows = 8;
    constexpr int64_t radixSelectInt64SmallAxisMinRows = 32;
    constexpr int64_t radixSelectFp16InteriorMinRows = 64;
    bool isNearHead = kth <= 1U;
    bool isTail = (kth + 1U == axisLen);
    bool isBeforeTail = (kth + 2U == axisLen);
    bool isInterior = !isNearHead && !isTail && !isBeforeTail;
    bool isHugeAxis = axisLen >= radixSelectHugeAxis;
    bool isLargeAxisInterior = axisLen >= radixSelectMinAxis && isInterior;
    bool isInt64SmallAxisInterior = axisLen >= radixSelectInt64SmallAxis && isInterior;
    bool hasEnoughRowsForInt32Interior = info.unsortedDim >= radixSelectInt32InteriorMinRows;
    bool hasEnoughRowsForInt64Interior = info.unsortedDim >= radixSelectInt64InteriorMinRows;
    bool hasEnoughRowsForInt64SmallAxis = info.unsortedDim >= radixSelectInt64SmallAxisMinRows;
    bool hasEnoughRowsForFp16Interior = info.unsortedDim >= radixSelectFp16InteriorMinRows;

    // RadixSelect pays fixed workspace/reduce cost. Huge axes and near-head k amortize it directly; middle-k cases
    // need enough independent rows, otherwise the radix-more-core fallback is faster.
    switch (info.dataType) {
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return isHugeAxis || (axisLen >= radixSelectMinAxis && isNearHead) ||
                   (isLargeAxisInterior && hasEnoughRowsForInt32Interior);
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return isHugeAxis || (isLargeAxisInterior && hasEnoughRowsForInt64Interior) ||
                   (isInt64SmallAxisInterior && hasEnoughRowsForInt64SmallAxis);
        case ge::DT_FLOAT:
            return isHugeAxis || (axisLen >= radixSelectMinAxis && isNearHead);
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return isInterior && (isHugeAxis || (isLargeAxisInterior && hasEnoughRowsForFp16Interior));
        default:
            return false;
    }
}

struct RadixSelectPlan {
    uint64_t tileElems = 0;
    uint64_t tileCount = 0;
    uint32_t rowsParallel = 0;
    uint32_t coresPerRow = 1;
    uint32_t blockDim = 0;
    uint64_t workspace = 0;
};

static uint64_t GetRadixSelectFixedBytes()
{
    constexpr uint64_t radixBuckets = 256UL;
    constexpr uint64_t radixSelectFindThreads = 128UL;
    constexpr uint64_t radixSelectResultWords = 8UL;
    constexpr uint64_t radixSelectActiveIndexCap = 4096UL;
    constexpr uint64_t radixSelectReserveAlign = 1024UL;
    constexpr uint64_t histogramBytes = radixBuckets * sizeof(uint64_t);
    constexpr uint64_t reservedRawBytes = radixBuckets * sizeof(uint16_t) + radixSelectFindThreads * sizeof(uint32_t) +
                                          radixSelectResultWords * sizeof(uint64_t) + 2UL * 32UL +
                                          radixBuckets * sizeof(uint64_t) +
                                          radixSelectActiveIndexCap * sizeof(uint32_t);
    return histogramBytes +
           ((reservedRawBytes + radixSelectReserveAlign - 1UL) / radixSelectReserveAlign) * radixSelectReserveAlign;
}

static bool ComputeRadixSelectTileElems(const SortKthTileInfo& info, uint64_t& tileElems)
{
    constexpr uint64_t radixSelectMaxTileElems = 32768UL;
    constexpr uint64_t radixSelectMinAlignElems = 256UL;
    uint64_t usableUb = ComputeUbAfterSimtReserve(info.ubSize);
    uint64_t bytesPerElem = static_cast<uint64_t>(info.dtypeSize) * 2UL;
    uint64_t fixedBytes = GetRadixSelectFixedBytes();
    if (bytesPerElem == 0U || usableUb <= fixedBytes) {
        return false;
    }
    tileElems = std::min<uint64_t>((usableUb - fixedBytes) / bytesPerElem, radixSelectMaxTileElems);
    uint64_t alignElems = std::max<uint64_t>(radixSelectMinAlignElems, info.blockUbSize / info.dtypeSize);
    tileElems = tileElems / alignElems * alignElems;
    tileElems = std::min<uint64_t>(tileElems, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
    return tileElems != 0U;
}

static bool ComputeRadixSelectPlan(const SortKthTileInfo& info, RadixSelectPlan& plan)
{
    if (!ComputeRadixSelectTileElems(info, plan.tileElems)) {
        return false;
    }
    plan.rowsParallel = static_cast<uint32_t>(
        std::min<int64_t>(static_cast<int64_t>(info.maxCoreNum), info.unsortedDim));
    if (plan.rowsParallel == 0U) {
        return false;
    }
    plan.tileCount = Ops::Base::CeilDiv(static_cast<uint64_t>(info.lastAxis), plan.tileElems);
    uint32_t maxCoresPerRow = std::max<uint32_t>(1U, info.maxCoreNum / plan.rowsParallel);
    if (plan.tileCount > 1UL) {
        plan.coresPerRow = static_cast<uint32_t>(std::min<uint64_t>(maxCoresPerRow, plan.tileCount));
    }
    uint64_t blockDim = static_cast<uint64_t>(plan.rowsParallel) * plan.coresPerRow;
    if (blockDim > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    plan.blockDim = static_cast<uint32_t>(blockDim);
    return true;
}

static bool ComputeRadixSelectWorkspace(const SortKthTileInfo& info, RadixSelectPlan& plan)
{
    constexpr uint64_t radixBuckets = 256UL;
    constexpr uint64_t radixSelectResultWords = 8UL;
    uint64_t histogramWorkspace = static_cast<uint64_t>(plan.blockDim) * radixBuckets * sizeof(uint64_t);
    uint64_t groupStateWorkspace = static_cast<uint64_t>(plan.rowsParallel) * radixSelectResultWords * sizeof(uint64_t);
    if (histogramWorkspace > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - groupStateWorkspace) {
        return false;
    }
    uint64_t workspaceRaw = histogramWorkspace + groupStateWorkspace;
    plan.workspace = Ops::Base::CeilAlign(workspaceRaw, static_cast<uint64_t>(info.blockUbSize));
    return plan.workspace >= workspaceRaw &&
           plan.workspace <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - WORK_SPACE_SIZE;
}

static bool TryRadixSelect(gert::TilingContext* context, SortKthTileInfo& info, KthValueTilingData* tilingData,
                           uint32_t& blockDim, uint64_t& schId)
{
    if (!IsRadixSelectProfitable(info, tilingData->kthIndex) || info.isNonLastAxis ||
        info.lastAxis <= static_cast<int64_t>(SMALL_AXIS_THRESHOLD) || info.maxCoreNum == 0U) {
        return false;
    }
    RadixSelectPlan plan;
    if (!ComputeRadixSelectPlan(info, plan) || !ComputeRadixSelectWorkspace(info, plan)) {
        return false;
    }
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    if (userWorkspaceSize == nullptr) {
        return false;
    }

    SortKthTileInfo candidate = info;
    candidate.numTileDataSize = static_cast<uint32_t>(plan.tileElems);
    candidate.unsortedDimParallel = plan.rowsParallel;
    candidate.lastDimTileNum = static_cast<uint32_t>(
        std::min<uint64_t>(plan.tileCount, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
    candidate.lastDimNeedCore = plan.coresPerRow;
    uint64_t rowLoops = Ops::Base::CeilDiv(static_cast<uint64_t>(info.unsortedDim),
                                           static_cast<uint64_t>(plan.rowsParallel));
    candidate.sortLoopTimes = static_cast<uint32_t>(
        std::min<uint64_t>(rowLoops, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
    candidate.tmpUbSize = 0U;
    info = candidate;
    blockDim = plan.blockDim;
    PlanToTilingData(info, tilingData);
    schId = KTH_VALUE_SCHID_RADIX_SELECT;
    userWorkspaceSize[0] = static_cast<size_t>(WORK_SPACE_SIZE + plan.workspace);
    context->SetScheduleMode(1);
    return true;
}

static bool TrySmallAxisShortRankSelect(const SortKthTileInfo& info, KthValueTilingData* tilingData, uint32_t& blockDim,
                                        uint64_t& schId)
{
    constexpr int64_t shortRankSelectMaxAxis = 32;
    constexpr uint64_t shortRankSelectMaxRank = 8;
    if (info.isNonLastAxis || info.lastAxis <= 1 || info.lastAxis > shortRankSelectMaxAxis || info.unsortedDim <= 0 ||
        info.maxCoreNum == 0U || tilingData->kthIndex < 0 || tilingData->kthIndex >= info.lastAxis) {
        return false;
    }
    if (info.dataType != ge::DT_INT64 && info.dataType != ge::DT_UINT64) {
        return false;
    }

    uint64_t kthIndex = static_cast<uint64_t>(tilingData->kthIndex);
    uint64_t axisLen = static_cast<uint64_t>(info.lastAxis);
    uint64_t shortRank = std::min(kthIndex + 1U, axisLen - kthIndex);
    if (shortRank == 0U || shortRank > shortRankSelectMaxRank ||
        static_cast<uint64_t>(info.unsortedDim) < static_cast<uint64_t>(info.maxCoreNum)) {
        return false;
    }

    SmallAxisRoutePlan plan;
    if (!SelectSmallAxisRoute(info, plan) || plan.kind != SmallAxisRouteKind::TWO_STAGE) {
        return false;
    }
    FillSmallAxisTiling(tilingData, plan, static_cast<uint32_t>(info.lastAxis), blockDim);
    tilingData->keyParams2 = static_cast<uint32_t>(shortRank);
    tilingData->tmpUbSize = 0;
    schId = KTH_VALUE_SCHID_SMALL_AXIS_SHORT_RANK_SELECT;
    return true;
}

static bool TrySmallAxis(gert::TilingContext* context, SortKthTileInfo& info, KthValueTilingData* tilingData,
                         uint32_t& blockDim, uint64_t& schId)
{
    if (info.lastAxis > static_cast<int64_t>(SMALL_AXIS_THRESHOLD)) {
        return false;
    }
    if (info.lastAxis == 1) {
        if (SetAxisOneCopyTiling(context, info) != ge::GRAPH_SUCCESS) {
            return false;
        }
        PlanToTilingData(info, tilingData);
        blockDim = info.coreNumNeed;
        schId = KTH_VALUE_SCHID_AXIS_ONE_COPY;
        return true;
    }
    SmallAxisRoutePlan plan;
    bool selected = info.isNonLastAxis ? SelectNonLastSmallAxisRoute(info, plan) : SelectSmallAxisRoute(info, plan);
    if (!selected) {
        return false;
    }
    if (info.isNonLastAxis) {
        if (!FillNonLastSmallAxisTiling(tilingData, plan, info, blockDim)) {
            return false;
        }
    } else {
        FillSmallAxisTiling(tilingData, plan, static_cast<uint32_t>(info.lastAxis), blockDim);
    }
    schId = plan.kind == SmallAxisRouteKind::TWO_STAGE ? KTH_VALUE_SCHID_SMALL_AXIS_TWO_STAGE :
                                                         KTH_VALUE_SCHID_SMALL_AXIS_INSERTION;
    return true;
}

static bool TryMerge(gert::TilingContext* context, SortKthTileInfo& info, KthValueTilingData* tilingData,
                     uint32_t& blockDim, uint64_t& schId)
{
    if (IsMergeSortSupported(info.dataType, info.lastAxis)) {
        SortKthTileInfo candidate = info;
        if (SetKthValueMergeSortTiling(context, candidate, blockDim, schId) == ge::GRAPH_SUCCESS) {
            info = candidate;
            PlanToTilingData(info, tilingData);
            return true;
        }
    }
    if (IsMergeMoreCoreSupported(info.dataType, info.lastAxis, info.unsortedDim, info.maxCoreNum)) {
        SortKthTileInfo candidate = info;
        if (SetMergeMoreCoreTiling(context, candidate, blockDim) == ge::GRAPH_SUCCESS) {
            info = candidate;
            PlanToTilingData(info, tilingData);
            schId = KTH_VALUE_SCHID_MERGE_MORE_CORE;
            return true;
        }
    }
    return false;
}

static bool TryMergeIntraCore(gert::TilingContext* context, SortKthTileInfo& info, KthValueTilingData* tilingData,
                              uint32_t& blockDim, uint64_t& schId)
{
    if (!IsMergeIntraCoreSupported(info.dataType, info.lastAxis, info.unsortedDim, info.maxCoreNum, info.ubSize)) {
        return false;
    }
    SortKthTileInfo candidate = info;
    if (SetMergeIntraCoreTiling(context, candidate) != ge::GRAPH_SUCCESS) {
        return false;
    }
    info = candidate;
    blockDim = info.coreNumNeed;
    PlanToTilingData(info, tilingData);
    schId = KTH_VALUE_SCHID_MERGE_INTRA_CORE;
    return true;
}

static bool TryNonLastSmallAxis(gert::TilingContext* context, const SortKthTileInfo& info,
                                KthValueTilingData* tilingData, uint32_t& blockDim, uint64_t& schId)
{
    if (!info.isNonLastAxis) {
        return false;
    }
    return SetNonLastSmallAxisTiling(context, info, tilingData, blockDim, schId);
}

// =============================================================================
// Route selection and finalization
// =============================================================================
static ge::graphStatus SelectKthValueRoute(gert::TilingContext* context, SortKthTileInfo& info,
                                           KthValueTilingData* tilingData, uint32_t& blockDim, uint64_t& schId)
{
    if (TrySmallAxisShortRankSelect(info, tilingData, blockDim, schId)) {
        return ge::GRAPH_SUCCESS;
    }
    if (TrySmallAxis(context, info, tilingData, blockDim, schId)) {
        return ge::GRAPH_SUCCESS;
    }
    if (TryNonLastSmallAxis(context, info, tilingData, blockDim, schId)) {
        return ge::GRAPH_SUCCESS;
    }
    if (info.isNonLastAxis) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "sortAxis", std::to_string(info.sortAxis).c_str(),
            "The value of sortAxis must be the last axis or meet no-transpose schedule constraints.");
        return ge::GRAPH_FAILED;
    }
    if (TryMerge(context, info, tilingData, blockDim, schId) || TryRadixOneCore(context, info, tilingData, schId) ||
        TryRadixSelect(context, info, tilingData, blockDim, schId) ||
        TryMergeIntraCore(context, info, tilingData, blockDim, schId)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((SetRadixMoreCoreTiling(context, info, blockDim) != ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SetRadixMoreCoreTiling", "GRAPH_FAILED",
                                                      "The value of SetRadixMoreCoreTiling must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    PlanToTilingData(info, tilingData);
    schId = KTH_VALUE_SCHID_RADIX_MORE_CORE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FinalizeKthValueRoute(gert::TilingContext* context,
                                             const platform_ascendc::PlatformAscendC& ascendcPlatform,
                                             const SortKthTileInfo& info, KthValueTilingData* tilingData,
                                             uint64_t schId, uint32_t& blockDim)
{
    if (schId == KTH_VALUE_SCHID_RADIX_ONE_CORE) {
        blockDim = static_cast<uint32_t>(std::min<int64_t>(ascendcPlatform.GetCoreNumAiv(), info.unsortedDim));
        if (blockDim == 0U) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockDim", std::to_string(blockDim).c_str(),
                                                  "The value of blockDim must be greater than 0.");
            return ge::GRAPH_FAILED;
        }
        uint64_t maxRowsPerCore = Ops::Base::CeilDiv(static_cast<uint64_t>(info.unsortedDim),
                                                     static_cast<uint64_t>(blockDim));
        uint64_t sortLoopTimes = Ops::Base::CeilDiv(maxRowsPerCore, static_cast<uint64_t>(info.outputRowsPerLoop));
        OP_CHECK_IF((sortLoopTimes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sortLoopTimes",
                                                          std::to_string(sortLoopTimes).c_str(),
                                                          "The value of sortLoopTimes must be less than or equal to "
                                                          "uint32 max."),
                    return ge::GRAPH_FAILED);
        tilingData->unsortedDimParallel = blockDim;
        tilingData->sortLoopTimes = static_cast<uint32_t>(sortLoopTimes);
    }
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    if (schId != KTH_VALUE_SCHID_MERGE_MORE_CORE && schId != KTH_VALUE_SCHID_MERGE_INTRA_CORE &&
        schId != KTH_VALUE_SCHID_NON_LAST_SMALL_AXIS && schId != KTH_VALUE_SCHID_NON_LAST_SMALL_AXIS_RADIX &&
        schId != KTH_VALUE_SCHID_RADIX_MORE_CORE && schId != KTH_VALUE_SCHID_RADIX_SELECT) {
        userWorkspaceSize[0] = WORK_SPACE_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

static void SetKthValueTilingContext(gert::TilingContext* context, uint64_t schId, const SortKthTileInfo& info,
                                     uint32_t blockDim)
{
    uint64_t tilingKeyIsInt32 = schId == KTH_VALUE_SCHID_RADIX_MORE_CORE ? info.isInt32 : 1U;
    context->SetTilingKey(GET_TPL_TILING_KEY(schId, tilingKeyIsInt32));
    context->SetBlockDim(blockDim);
    if (schId == KTH_VALUE_SCHID_SMALL_AXIS_INSERTION || schId == KTH_VALUE_SCHID_SMALL_AXIS_TWO_STAGE ||
        schId == KTH_VALUE_SCHID_SMALL_AXIS_SHORT_RANK_SELECT || schId == KTH_VALUE_SCHID_RADIX_MORE_CORE ||
        schId == KTH_VALUE_SCHID_RADIX_SELECT || schId == KTH_VALUE_SCHID_NON_LAST_SMALL_AXIS ||
        schId == KTH_VALUE_SCHID_NON_LAST_SMALL_AXIS_RADIX) {
        context->SetLocalMemorySize(info.ubSize - SIMT_UB);
    } else {
        context->SetLocalMemorySize(info.ubSize);
    }
}

// Extends user workspace for the merge multi-core median schedule: block-alignment
// padding for the algorithm region, plus a per-core uint32 count buffer for aggregation.
static ge::graphStatus AddMergeMedianWorkspace(gert::TilingContext* context, const SortKthTileInfo& info,
                                               uint32_t blockDim)
{
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    OP_CHECK_IF(info.workspaceSize < WORK_SPACE_SIZE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "workspace", "invalid",
                                                      "Merge workspace size must include the system workspace."),
                return ge::GRAPH_FAILED);
    uint64_t rawAlgorithmBytes = static_cast<uint64_t>(info.workspaceSize - WORK_SPACE_SIZE);
    OP_CHECK_IF(
        info.blockUbSize == 0U || rawAlgorithmBytes > std::numeric_limits<uint64_t>::max() - (info.blockUbSize - 1U),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "workspace", "overflow",
                                              "Median workspace alignment must not overflow."),
        return ge::GRAPH_FAILED);
    uint64_t alignedAlgorithmBytes = Ops::Base::CeilAlign(rawAlgorithmBytes, static_cast<uint64_t>(info.blockUbSize));
    uint64_t countBytes = Ops::Base::CeilAlign(static_cast<uint64_t>(blockDim) * sizeof(uint32_t),
                                               static_cast<uint64_t>(info.blockUbSize));
    uint64_t extraBytes = alignedAlgorithmBytes - rawAlgorithmBytes + countBytes;
    OP_CHECK_IF(userWorkspaceSize[0] > std::numeric_limits<size_t>::max() - extraBytes,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "workspace", "overflow",
                                                      "Median count workspace size must not overflow."),
                return ge::GRAPH_FAILED);
    userWorkspaceSize[0] += static_cast<size_t>(extraBytes);
    return ge::GRAPH_SUCCESS;
}

// Extends user workspace for the radix multi-core median schedule with a per-core
// uint32 count buffer; the radix workspace is already block-aligned, so no padding is added.
static ge::graphStatus AddRadixMedianWorkspace(gert::TilingContext* context, const SortKthTileInfo& info,
                                               uint32_t blockDim)
{
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    uint64_t countBytes = Ops::Base::CeilAlign(
        static_cast<uint64_t>(blockDim) * MEDIAN_RADIX_COUNT_STORAGE_WORDS * sizeof(uint32_t),
        static_cast<uint64_t>(info.blockUbSize));
    OP_CHECK_IF(userWorkspaceSize[0] > std::numeric_limits<size_t>::max() - countBytes,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "workspace", "overflow",
                                                      "Median count workspace size must not overflow."),
                return ge::GRAPH_FAILED);
    userWorkspaceSize[0] += static_cast<size_t>(countBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AddMedianWorkspace(gert::TilingContext* context, const SortKthTileInfo& info,
                                          const KthValueTilingData& tilingData, uint64_t schId, uint32_t blockDim,
                                          bool isMedianOp)
{
    if (!isMedianOp || tilingData.medianMode == MEDIAN_MODE_STATIC) {
        return ge::GRAPH_SUCCESS;
    }
    if (schId == KTH_VALUE_SCHID_MERGE_MORE_CORE) {
        return AddMergeMedianWorkspace(context, info, blockDim);
    }
    if (schId == KTH_VALUE_SCHID_RADIX_MORE_CORE) {
        return AddRadixMedianWorkspace(context, info, blockDim);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SelectAndFinalizeKthValueRoute(gert::TilingContext* context,
                                                      const platform_ascendc::PlatformAscendC& ascendcPlatform,
                                                      SortKthTileInfo& info, KthValueTilingData* tilingData,
                                                      bool isMedianOp)
{
    KthValueTilingData candidateTilingData = *tilingData;
    uint64_t schId = KTH_VALUE_SCHID_RADIX_MORE_CORE;
    uint32_t blockDim = 1;
    OP_CHECK_IF((SelectKthValueRoute(context, info, &candidateTilingData, blockDim, schId) != ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SelectKthValueRoute", "GRAPH_FAILED",
                                                      "The value of SelectKthValueRoute must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((FinalizeKthValueRoute(context, ascendcPlatform, info, &candidateTilingData, schId, blockDim) !=
                 ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "FinalizeKthValueRoute", "GRAPH_FAILED",
                                                      "The value of FinalizeKthValueRoute must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        AddMedianWorkspace(context, info, candidateTilingData, schId, blockDim, isMedianOp) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "AddMedianWorkspace", "GRAPH_FAILED",
                                              "The value of AddMedianWorkspace must be GRAPH_SUCCESS."),
        return ge::GRAPH_FAILED);
    *tilingData = candidateTilingData;
    OP_LOGI(context->GetNodeName(),
            "KthValueTiling: schId=%lu, blockDim=%u, lastAxis=%ld, unsortedDim=%ld, "
            "isNonLastAxis=%d, dtypeSize=%u",
            schId, blockDim, info.lastAxis, info.unsortedDim, static_cast<int>(info.isNonLastAxis), info.dtypeSize);
    SetKthValueTilingContext(context, schId, info, blockDim);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// Main entry
// =============================================================================
static ge::graphStatus TilingKthLike(gert::TilingContext* context, int64_t k, int64_t dim, uint32_t medianMode,
                                     bool isMedianOp)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto tilingData = context->GetTilingData<KthValueTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    OP_CHECK_IF((memset_s(tilingData, sizeof(KthValueTilingData), 0, sizeof(KthValueTilingData)) != EOK),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "memset_s", "not EOK",
                                                      "The value of memset_s must be EOK."),
                return ge::GRAPH_FAILED);
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    uint32_t dtypeSize = 0;
    OP_CHECK_IF((CheckKthValueDtypes(context, dataType, dtypeSize) != ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "CheckKthValueDtypes", "GRAPH_FAILED",
                                                      "The value of CheckKthValueDtypes must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    SortKthTileInfo info;
    info.dataType = dataType;
    info.dtypeSize = dtypeSize;
    info.y2DtypeSize = static_cast<uint32_t>(sizeof(uint32_t));
    info.blockUbSize = Ops::Base::GetUbBlockSize(context);
    info.maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(info.maxCoreNum == 0U || info.blockUbSize == 0U || info.dtypeSize == 0U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform resources", "zero",
                                                      "Core count, UB block size and dtype size must be positive."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((ParseKthValueShapeInfo(context, &k, &dim, info) != ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ParseKthValueShapeInfo", "GRAPH_FAILED",
                                                      "The value of ParseKthValueShapeInfo must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    info.isNonLastAxis = (info.sortAxis != info.rank - 1);
    bool oneCoreUbValid = false;
    OP_CHECK_IF((ComputeKthValueUbInfo(context, ascendcPlatform, info, oneCoreUbValid) != ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ComputeKthValueUbInfo", "GRAPH_FAILED",
                                                      "The value of ComputeKthValueUbInfo must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    InitKthValueBaseTiling(tilingData, info, oneCoreUbValid, k - 1);
    tilingData->medianMode = medianMode;
    return SelectAndFinalizeKthValueRoute(context, ascendcPlatform, info, tilingData, isMedianOp);
}

static ge::graphStatus Tiling4KthValue(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* kAttr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, kAttr);
    const int64_t* dimAttr = attrs->GetAttrPointer<int64_t>(1);
    int64_t dim = dimAttr == nullptr ? -1 : *dimAttr;
    return TilingKthLike(context, *kAttr, dim, MEDIAN_MODE_STATIC, false);
}

static ge::graphStatus TilingMedianLike(gert::TilingContext* context, bool ignoreNan)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dimAttr = attrs->GetAttrPointer<int64_t>(0);
    int64_t dim = dimAttr == nullptr ? -1 : *dimAttr;
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();
    bool isFloating = inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_FLOAT || inputDtype == ge::DT_BF16;
    const gert::Shape& shape = inputShape->GetStorageShape();
    int64_t rank = shape.GetDimNum();
    int64_t normDim = dim < 0 ? dim + rank : dim;
    OP_CHECK_IF(
        normDim < 0 || normDim >= rank,
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "dim", std::to_string(dim).c_str(), "a valid dimension"),
        return ge::GRAPH_FAILED);
    int64_t axisLen = shape.GetDim(normDim);
    OP_CHECK_IF(axisLen <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axisLen",
                                                      std::to_string(axisLen).c_str(), "axisLen must be positive."),
                return ge::GRAPH_FAILED);
    // Use the lower median for an even axis length. TilingKthLike expects one-based k,
    // so add one to the zero-based lower-median index (axisLen - 1) / 2.
    int64_t medianK = (axisLen - 1) / LOWER_MEDIAN_DIVISOR + 1;
    // Integer inputs contain no NaN. Keep them on KthValue's original static-k
    // path for both Median and NanMedian, without a scan or extra UB usage.
    uint32_t medianMode = isFloating ? (ignoreNan ? MEDIAN_MODE_IGNORE_NAN : MEDIAN_MODE_PROPAGATE_NAN) :
                                       MEDIAN_MODE_STATIC;
    return TilingKthLike(context, medianK, dim, medianMode, true);
}

ge::graphStatus Tiling4Median(gert::TilingContext* context) { return TilingMedianLike(context, false); }

ge::graphStatus Tiling4NanMedian(gert::TilingContext* context) { return TilingMedianLike(context, true); }

static ge::graphStatus TilingPrepare4KthValue(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<KthValueCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum",
                                                      std::to_string(compileInfo->coreNum).c_str(),
                                                      "The value of coreNum must be greater than 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4Median(gert::TilingParseContext* context) { return TilingPrepare4KthValue(context); }

ge::graphStatus TilingPrepare4NanMedian(gert::TilingParseContext* context) { return TilingPrepare4KthValue(context); }

IMPL_OP_OPTILING(KthValue).Tiling(Tiling4KthValue).TilingParse<KthValueCompileInfo>(TilingPrepare4KthValue);
} // namespace optiling
