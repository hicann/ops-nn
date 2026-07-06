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
 * \file prelu_tiling.cpp
 * \brief Prelu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/prelu_tiling_data.h"
#include "../op_kernel/prelu_tiling_key.h"

#include <algorithm>
#include <limits>

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t CORE_ALIGN_SIZE = 512U;
constexpr int64_t MAX_AIV_CORE_NUM = 40;
constexpr uint64_t MIN_PARALLEL_TILE_NUM = 2U;
constexpr uint64_t UB_RESERVED_SIZE = 1024U;
constexpr uint64_t SMALL_L_WEIGHT_REUSE_MAX_INNER_SIZE = 16U;
constexpr uint64_t SPLIT_C_WEIGHT_REUSE_MAX_INNER_SIZE = 64U;
constexpr uint64_t LARGE_L_SPLIT_C_WEIGHT_REUSE_MAX_INNER_SIZE = 1024U;
constexpr uint64_t LARGE_L_SPLIT_C_WEIGHT_REUSE_MAX_CHANNEL_SIZE = 1024U;
constexpr uint64_t LARGE_C_WEIGHT_REUSE_MIN_CHANNEL_SIZE = 32U;
constexpr uint64_t MIN_SPLIT_C_WEIGHT_REUSE_CORE_NUM = 10U;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    coreNum = std::min(coreNum, MAX_AIV_CORE_NUM);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = WS_SYS_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAndDtypeInfo(gert::TilingContext* context, int64_t& totalNum, int64_t& weightSize,
                                            int64_t& weightMode, int64_t& channelSize, int64_t& innerSize,
                                            int64_t& rowNum, ge::DataType& dataType, uint32_t& typeLength)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputWeight = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputWeight);
    auto outputY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputY);

    auto xShape = inputX->GetStorageShape();
    auto wShape = inputWeight->GetStorageShape();
    auto yShape = outputY->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != yShape.GetDimNum(), OP_LOGE(context, "Prelu: x/y rank mismatch"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        OP_CHECK_IF(xShape.GetDim(i) != yShape.GetDim(i), OP_LOGE(context, "Prelu: x/y shape mismatch"),
                    return ge::GRAPH_FAILED);
    }

    auto xDesc = context->GetInputDesc(0);
    auto weightDesc = context->GetInputDesc(1);
    auto yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    dataType = xDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16,
                OP_LOGE(context, "Prelu: unsupported dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightDesc->GetDataType() != dataType || yDesc->GetDataType() != dataType,
                OP_LOGE(context, "Prelu: x, weight and y must have the same dtype"), return ge::GRAPH_FAILED);

    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "Prelu: dtype length is 0"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(wShape.GetDimNum() != 1,
                OP_LOGE(context, "Prelu: weight must be 1-D, got dim num %zu", wShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    weightSize = wShape.GetDim(0);
    OP_CHECK_IF(weightSize <= 0, OP_LOGE(context, "Prelu: weight size must be positive"), return ge::GRAPH_FAILED);

    weightMode = 0;
    channelSize = 1;
    innerSize = 1;
    rowNum = 0;
    if (weightSize != 1) {
        OP_CHECK_IF(xShape.GetDimNum() < 2, OP_LOGE(context, "Prelu: channel weight requires x rank >= 2"),
                    return ge::GRAPH_FAILED);
        int64_t batchSize = xShape.GetDim(0);
        OP_CHECK_IF(batchSize <= 0, OP_LOGE(context, "Prelu: N must be positive, got %ld", batchSize),
                    return ge::GRAPH_FAILED);
        channelSize = xShape.GetDim(1);
        OP_CHECK_IF(channelSize <= 0, OP_LOGE(context, "Prelu: channel size must be positive, got %ld", channelSize),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            weightSize != channelSize,
            OP_LOGE(context, "Prelu: weight size must be 1 or match channel size, weight size=%ld, channel size=%ld",
                    weightSize, channelSize),
            return ge::GRAPH_FAILED);

        innerSize = 1;
        for (size_t i = 2; i < xShape.GetDimNum(); ++i) {
            int64_t dimValue = xShape.GetDim(i);
            OP_CHECK_IF(dimValue <= 0, OP_LOGE(context, "Prelu: x dim %zu must be positive, got %ld", i, dimValue),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(innerSize > std::numeric_limits<int64_t>::max() / dimValue,
                        OP_LOGE(context, "Prelu: L exceeds int64 range"), return ge::GRAPH_FAILED);
            innerSize *= dimValue;
        }
        OP_CHECK_IF(static_cast<uint64_t>(batchSize) >
                        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / static_cast<uint64_t>(channelSize),
                    OP_LOGE(context, "Prelu: rowNum exceeds int64 range"), return ge::GRAPH_FAILED);
        rowNum = batchSize * channelSize;
        weightMode = 1;
    }

    totalNum = inputX->GetOriginShape().GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetBufferBytesPerElement(ge::DataType dataType)
{
    if (dataType == ge::DT_FLOAT) {
        return 24U;
    }
    if (dataType == ge::DT_FLOAT16) {
        return 12U;
    }
    return 20U;
}

static uint64_t GetNcWeightReuseBufferBytesPerElement(ge::DataType dataType)
{
    if (dataType == ge::DT_FLOAT) {
        return 28U;
    }
    if (dataType == ge::DT_FLOAT16) {
        return 14U;
    }
    return 24U;
}

static uint64_t GetNcResidentWeightBufferBytesPerElement(ge::DataType dataType)
{
    if (dataType == ge::DT_FLOAT) {
        return 28U;
    }
    if (dataType == ge::DT_FLOAT16) {
        return 14U;
    }
    return 24U;
}

static uint64_t GetNcWeightCacheBytesPerElement(ge::DataType dataType, uint32_t typeLength)
{
    uint64_t bytes = typeLength;
    if (dataType == ge::DT_BF16) {
        bytes += sizeof(float);
    }
    return bytes;
}

static uint64_t GetWeightVecBytesPerElement(ge::DataType dataType, uint32_t typeLength)
{
    if (dataType == ge::DT_BF16) {
        return sizeof(float);
    }
    return typeLength;
}

static uint64_t CeilDiv(uint64_t value, uint64_t factor) { return (value + factor - 1U) / factor; }

static uint64_t AlignUp(uint64_t value, uint64_t align) { return CeilDiv(value, align) * align; }

static ge::graphStatus CalcTiling(gert::TilingContext* context, uint64_t ubSize, int64_t coreNum, int64_t totalNum,
                                  ge::DataType dataType, uint32_t typeLength, int64_t weightMode, int64_t channelSize,
                                  int64_t innerSize, int64_t rowNum, PreluTilingData* tiling, uint32_t& usedCoreNum,
                                  bool& useSplitLParallel, bool& useNcWeightReuse, bool& useNcSplitCWeightReuse,
                                  bool& useNcResidentWeightReuse)
{
    useSplitLParallel = false;
    useNcWeightReuse = false;
    useNcSplitCWeightReuse = false;
    useNcResidentWeightReuse = false;
    uint64_t bufferBytesPerElement = GetBufferBytesPerElement(dataType);
    uint64_t usableUbSize = (ubSize > UB_RESERVED_SIZE) ? (ubSize - UB_RESERVED_SIZE) : ubSize;
    uint64_t blockElementNum = BLOCK_SIZE / typeLength;
    uint64_t maxTileElements = usableUbSize / bufferBytesPerElement;
    OP_CHECK_IF(maxTileElements < blockElementNum, OP_LOGE(context, "Prelu: UB is too small for one aligned tile"),
                return ge::GRAPH_FAILED);

    uint64_t ubFactor = (maxTileElements / blockElementNum) * blockElementNum;

    tiling->totalLength = totalNum;
    tiling->tileLength = static_cast<int64_t>(ubFactor);
    tiling->channelSize = channelSize;
    tiling->innerSize = innerSize;
    tiling->innerSizeAligned = innerSize;
    tiling->baseRows = 0;
    tiling->extraRows = 0;
    tiling->tilesPerRow = 0;
    tiling->baseTasks = 0;
    tiling->extraTasks = 0;
    uint64_t coreLimit = static_cast<uint64_t>(coreNum);
    if (weightMode == 0) {
        uint64_t coreAlignElementNum = CORE_ALIGN_SIZE / typeLength;
        uint64_t totalCoreElements = CeilDiv(static_cast<uint64_t>(totalNum), coreLimit);
        uint64_t blockFactor = (CeilDiv(totalCoreElements, coreAlignElementNum)) * coreAlignElementNum;
        if (blockFactor == 0) {
            blockFactor = coreAlignElementNum;
        }
        uint64_t finalCoreNum = static_cast<uint64_t>(totalNum) == 0 ?
                                    1U :
                                    CeilDiv(static_cast<uint64_t>(totalNum), blockFactor);
        finalCoreNum = std::min(coreLimit, finalCoreNum);

        uint64_t tailLength = 0;
        uint64_t formerNum = 0;
        if (totalNum > 0) {
            formerNum = finalCoreNum > 0 ? finalCoreNum - 1U : 0U;
            tailLength = static_cast<uint64_t>(totalNum) - formerNum * blockFactor;
        }

        tiling->usedCoreNum = static_cast<int64_t>(finalCoreNum);
        tiling->formerNum = static_cast<int64_t>(formerNum);
        tiling->formerLength = static_cast<int64_t>(blockFactor);
        tiling->tailLength = static_cast<int64_t>(tailLength);
        usedCoreNum = static_cast<uint32_t>(finalCoreNum);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(innerSize <= 0, OP_LOGE(context, "Prelu: L must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(innerSize) > std::numeric_limits<uint64_t>::max() - blockElementNum + 1U,
                OP_LOGE(context, "Prelu: L is too large to align"), return ge::GRAPH_FAILED);
    uint64_t innerSizeAligned = CeilDiv(static_cast<uint64_t>(innerSize), blockElementNum) * blockElementNum;
    OP_CHECK_IF(innerSizeAligned > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context, "Prelu: aligned L exceeds int64 range"), return ge::GRAPH_FAILED);
    uint64_t rowNumU64 = static_cast<uint64_t>(rowNum);
    tiling->formerNum = 0;
    tiling->formerLength = 0;
    tiling->tailLength = 0;
    tiling->innerSizeAligned = static_cast<int64_t>(innerSizeAligned);

    uint64_t batchSize = rowNumU64 / static_cast<uint64_t>(channelSize);
    bool preferWeightReuseByRow = innerSize == 1 ||
                                  (static_cast<uint64_t>(innerSize) <= SPLIT_C_WEIGHT_REUSE_MAX_INNER_SIZE &&
                                   static_cast<uint64_t>(channelSize) >= LARGE_C_WEIGHT_REUSE_MIN_CHANNEL_SIZE) ||
                                  (static_cast<uint64_t>(innerSize) <= LARGE_L_SPLIT_C_WEIGHT_REUSE_MAX_INNER_SIZE &&
                                   static_cast<uint64_t>(channelSize) <=
                                       LARGE_L_SPLIT_C_WEIGHT_REUSE_MAX_CHANNEL_SIZE &&
                                   static_cast<uint64_t>(channelSize) >= LARGE_C_WEIGHT_REUSE_MIN_CHANNEL_SIZE);
    if (preferWeightReuseByRow && channelSize > 1 && batchSize > 0) {
        OP_CHECK_IF(static_cast<uint64_t>(channelSize) >
                        std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(innerSize),
                    OP_LOGE(context, "Prelu: C*L exceeds uint64 range"), return ge::GRAPH_FAILED);
        uint64_t rowElements = static_cast<uint64_t>(channelSize) * static_cast<uint64_t>(innerSize);
        uint64_t alignedRowElements = AlignUp(rowElements, blockElementNum);
        uint64_t alignedChannelSize = AlignUp(static_cast<uint64_t>(channelSize), blockElementNum);
        uint64_t weightCacheBytesPerElement = GetNcWeightCacheBytesPerElement(dataType, typeLength);
        bool weightCacheSizeValid = alignedChannelSize <=
                                    std::numeric_limits<uint64_t>::max() / weightCacheBytesPerElement;
        uint64_t weightCacheBytes = weightCacheSizeValid ? alignedChannelSize * weightCacheBytesPerElement : 0U;
        uint64_t ncResidentWeightCacheBytes = dataType == ge::DT_BF16 ? weightCacheBytes : 0U;
        if (weightCacheSizeValid && weightCacheBytes < usableUbSize) {
            uint64_t ncBufferBytesPerElement = innerSize == 1 ? GetNcResidentWeightBufferBytesPerElement(dataType) :
                                                                GetNcWeightReuseBufferBytesPerElement(dataType);
            uint64_t ncWeightCacheBytes = innerSize == 1 ? ncResidentWeightCacheBytes : weightCacheBytes;
            uint64_t ncMaxTileElements = ((usableUbSize - ncWeightCacheBytes) / ncBufferBytesPerElement /
                                          alignedRowElements) *
                                         alignedRowElements;
            if (ncMaxTileElements >= alignedRowElements) {
                uint64_t finalCoreNum = std::min(coreLimit, batchSize);
                tiling->tileLength = static_cast<int64_t>(ncMaxTileElements);
                tiling->innerSizeAligned = static_cast<int64_t>(alignedRowElements);
                tiling->usedCoreNum = static_cast<int64_t>(finalCoreNum);
                tiling->baseRows = static_cast<int64_t>(batchSize / finalCoreNum);
                tiling->extraRows = static_cast<int64_t>(batchSize % finalCoreNum);
                usedCoreNum = static_cast<uint32_t>(finalCoreNum);
                bool isResidentWeightReuse = innerSize == 1;
                useNcResidentWeightReuse = isResidentWeightReuse;
                useNcWeightReuse = !isResidentWeightReuse;
                return ge::GRAPH_SUCCESS;
            }
        }

        if (innerSize == 1) {
            uint64_t splitCBytesPerElement = weightCacheBytesPerElement +
                                             GetNcWeightReuseBufferBytesPerElement(dataType);
            uint64_t maxSplitCElements = usableUbSize / splitCBytesPerElement;
            uint64_t splitCTileLength = (maxSplitCElements / blockElementNum) * blockElementNum;
            if (splitCTileLength >= blockElementNum) {
                uint64_t cTileNum = CeilDiv(static_cast<uint64_t>(channelSize), splitCTileLength);
                OP_CHECK_IF(cTileNum > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                            OP_LOGE(context, "Prelu: split-C tile count exceeds int64 range"), return ge::GRAPH_FAILED);
                OP_CHECK_IF(batchSize > std::numeric_limits<uint64_t>::max() / cTileNum,
                            OP_LOGE(context, "Prelu: split-C total task count exceeds uint64 range"),
                            return ge::GRAPH_FAILED);
                uint64_t totalTaskNum = batchSize * cTileNum;
                OP_CHECK_IF(totalTaskNum > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                            OP_LOGE(context, "Prelu: split-C total task count exceeds int64 range"),
                            return ge::GRAPH_FAILED);
                uint64_t finalCoreNum = std::min(coreLimit, totalTaskNum);
                tiling->tileLength = static_cast<int64_t>(splitCTileLength);
                tiling->innerSizeAligned = static_cast<int64_t>(splitCTileLength);
                tiling->usedCoreNum = static_cast<int64_t>(finalCoreNum);
                tiling->tilesPerRow = static_cast<int64_t>(cTileNum);
                tiling->baseTasks = static_cast<int64_t>(totalTaskNum / finalCoreNum);
                tiling->extraTasks = static_cast<int64_t>(totalTaskNum % finalCoreNum);
                usedCoreNum = static_cast<uint32_t>(finalCoreNum);
                useNcSplitCWeightReuse = true;
                return ge::GRAPH_SUCCESS;
            }
        } else {
            uint64_t splitCBytesPerChannel = weightCacheBytesPerElement +
                                             GetNcWeightReuseBufferBytesPerElement(dataType) *
                                                 static_cast<uint64_t>(innerSize);
            uint64_t maxSplitCChannels = usableUbSize / splitCBytesPerChannel;
            uint64_t splitCTileChannels = (maxSplitCChannels / blockElementNum) * blockElementNum;
            if (splitCTileChannels >= blockElementNum) {
                uint64_t alignedSplitCElements = AlignUp(splitCTileChannels * static_cast<uint64_t>(innerSize),
                                                         blockElementNum);
                uint64_t cTileNum = CeilDiv(static_cast<uint64_t>(channelSize), splitCTileChannels);
                OP_CHECK_IF(cTileNum > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                            OP_LOGE(context, "Prelu: small-L split-C tile count exceeds int64 range"),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(batchSize > std::numeric_limits<uint64_t>::max() / cTileNum,
                            OP_LOGE(context, "Prelu: small-L split-C total task count exceeds uint64 range"),
                            return ge::GRAPH_FAILED);
                uint64_t totalTaskNum = batchSize * cTileNum;
                OP_CHECK_IF(totalTaskNum > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                            OP_LOGE(context, "Prelu: small-L split-C total task count exceeds int64 range"),
                            return ge::GRAPH_FAILED);
                uint64_t finalCoreNum = std::min(coreLimit, totalTaskNum);
                if (finalCoreNum >= std::min(coreLimit, MIN_SPLIT_C_WEIGHT_REUSE_CORE_NUM)) {
                    tiling->tileLength = static_cast<int64_t>(splitCTileChannels);
                    tiling->innerSizeAligned = static_cast<int64_t>(alignedSplitCElements);
                    tiling->usedCoreNum = static_cast<int64_t>(finalCoreNum);
                    tiling->tilesPerRow = static_cast<int64_t>(cTileNum);
                    tiling->baseTasks = static_cast<int64_t>(totalTaskNum / finalCoreNum);
                    tiling->extraTasks = static_cast<int64_t>(totalTaskNum % finalCoreNum);
                    usedCoreNum = static_cast<uint32_t>(finalCoreNum);
                    useNcSplitCWeightReuse = true;
                    return ge::GRAPH_SUCCESS;
                }
            }
        }
    }

    uint64_t minParallelTileLength = CORE_ALIGN_SIZE / typeLength;
    uint64_t minParallelL = minParallelTileLength * MIN_PARALLEL_TILE_NUM;
    if (rowNumU64 * 2U <= coreLimit && static_cast<uint64_t>(innerSize) >= minParallelL) {
        uint64_t targetTilesPerRow = CeilDiv(coreLimit, rowNumU64);
        uint64_t parallelTileLength = AlignUp(CeilDiv(static_cast<uint64_t>(innerSize), targetTilesPerRow),
                                              minParallelTileLength);
        parallelTileLength = std::max(parallelTileLength, minParallelTileLength);
        parallelTileLength = std::min(parallelTileLength, ubFactor);

        uint64_t tilesPerRow = CeilDiv(static_cast<uint64_t>(innerSize), parallelTileLength);
        OP_CHECK_IF(tilesPerRow > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    OP_LOGE(context, "Prelu: tilesPerRow exceeds int64 range"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(rowNumU64 > std::numeric_limits<uint64_t>::max() / tilesPerRow,
                    OP_LOGE(context, "Prelu: totalTaskNum exceeds uint64 range"), return ge::GRAPH_FAILED);
        uint64_t totalTaskNum = rowNumU64 * tilesPerRow;
        OP_CHECK_IF(totalTaskNum > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    OP_LOGE(context, "Prelu: totalTaskNum exceeds int64 range"), return ge::GRAPH_FAILED);

        uint64_t finalCoreNum = std::min(coreLimit, totalTaskNum);
        uint64_t minBenefitCoreNum = std::min(coreLimit, rowNumU64 * 2U);
        if (finalCoreNum >= minBenefitCoreNum) {
            tiling->tileLength = static_cast<int64_t>(parallelTileLength);
            tiling->tilesPerRow = static_cast<int64_t>(tilesPerRow);
            tiling->usedCoreNum = static_cast<int64_t>(finalCoreNum);
            tiling->baseTasks = static_cast<int64_t>(totalTaskNum / finalCoreNum);
            tiling->extraTasks = static_cast<int64_t>(totalTaskNum % finalCoreNum);
            usedCoreNum = static_cast<uint32_t>(finalCoreNum);
            useSplitLParallel = true;
            return ge::GRAPH_SUCCESS;
        }
    }

    uint64_t finalCoreNum = rowNumU64 == 0 ? 1U : std::min(coreLimit, rowNumU64);
    if (innerSizeAligned > ubFactor) {
        uint64_t tilesPerRow = CeilDiv(static_cast<uint64_t>(innerSize), ubFactor);
        OP_CHECK_IF(tilesPerRow > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    OP_LOGE(context, "Prelu: tilesPerRow exceeds int64 range"), return ge::GRAPH_FAILED);
        tiling->tilesPerRow = static_cast<int64_t>(tilesPerRow);
    }
    tiling->usedCoreNum = static_cast<int64_t>(finalCoreNum);
    tiling->baseRows = static_cast<int64_t>(rowNumU64 / finalCoreNum);
    tiling->extraRows = static_cast<int64_t>(rowNumU64 % finalCoreNum);
    usedCoreNum = static_cast<uint32_t>(finalCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PreluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    int64_t weightSize = 1;
    int64_t weightMode = 0;
    int64_t channelSize = 1;
    int64_t innerSize = 1;
    int64_t rowNum = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    uint32_t typeLength = 0;
    OP_CHECK_IF(GetShapeAndDtypeInfo(context, totalNum, weightSize, weightMode, channelSize, innerSize, rowNum,
                                     dataType, typeLength) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAndDtypeInfo error"), return ge::GRAPH_FAILED);

    PreluTilingData* tiling = context->GetTilingData<PreluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    uint32_t usedCoreNum = 1;
    bool useSplitLParallel = false;
    bool useNcWeightReuse = false;
    bool useNcSplitCWeightReuse = false;
    bool useNcResidentWeightReuse = false;
    OP_CHECK_IF(CalcTiling(context, ubSize, coreNum, totalNum, dataType, typeLength, weightMode, channelSize, innerSize,
                           rowNum, tiling, usedCoreNum, useSplitLParallel, useNcWeightReuse, useNcSplitCWeightReuse,
                           useNcResidentWeightReuse) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "CalcTiling error"), return ge::GRAPH_FAILED);

    context->SetBlockDim(usedCoreNum);

    uint64_t tilingKey = GET_TPL_TILING_KEY(PRELU_TPL_SCALAR_MODE);
    if (weightMode == 1) {
        if (useNcResidentWeightReuse) {
            tilingKey = GET_TPL_TILING_KEY(PRELU_TPL_CHANNEL_NC_RESIDENT_WEIGHT_MODE);
        } else if (useNcWeightReuse) {
            tilingKey = GET_TPL_TILING_KEY(PRELU_TPL_CHANNEL_NC_WEIGHT_REUSE_MODE);
        } else if (useNcSplitCWeightReuse) {
            tilingKey = GET_TPL_TILING_KEY(PRELU_TPL_CHANNEL_NC_SPLIT_C_WEIGHT_REUSE_MODE);
        } else if (useSplitLParallel) {
            tilingKey = GET_TPL_TILING_KEY(PRELU_TPL_CHANNEL_SPLIT_L_PARALLEL_MODE);
        } else if (tiling->innerSizeAligned <= tiling->tileLength) {
            tilingKey = GET_TPL_TILING_KEY(PRELU_TPL_CHANNEL_FULL_L_MODE);
        } else {
            tilingKey = GET_TPL_TILING_KEY(PRELU_TPL_CHANNEL_SPLIT_L_MODE);
        }
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPrelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct PreluCompileInfo {};

IMPL_OP_OPTILING(Prelu).Tiling(PreluTilingFunc).TilingParse<PreluCompileInfo>(TilingParseForPrelu);

} // namespace optiling
