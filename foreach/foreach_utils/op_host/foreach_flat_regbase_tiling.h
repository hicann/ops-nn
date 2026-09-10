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
 * \file foreach_flat_regbase_tiling.h
 * \brief Validation-free flat tiling builder for homogeneous RegBase foreach flows.
 */
#ifndef FOREACH_FLAT_REGBASE_TILING_H
#define FOREACH_FLAT_REGBASE_TILING_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "../op_kernel/arch35/foreach_flat_tiling_data.h"

namespace optiling {
namespace ForeachFlatRegbaseTiling {

constexpr uint32_t DMA_ALIGN_BYTES = 32U;
constexpr uint32_t BUFFER_DEPTH = 2U;
constexpr int64_t MIN_CORE_WORK_BYTES = 4096;
constexpr uint64_t WORKSPACE_BYTES = 32U;

inline uint32_t Gcd(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0U) {
        uint32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

inline uint32_t GetStorageBytes(ge::DataType dtype)
{
    if (dtype == ge::DT_DOUBLE || dtype == ge::DT_INT64 || dtype == ge::DT_UINT64) {
        return sizeof(uint64_t);
    }
    if (dtype == ge::DT_FLOAT || dtype == ge::DT_INT32 || dtype == ge::DT_UINT32) {
        return sizeof(uint32_t);
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_INT16 || dtype == ge::DT_UINT16) {
        return sizeof(uint16_t);
    }
    if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8 || dtype == ge::DT_BOOL) {
        return sizeof(uint8_t);
    }
    return 0U;
}

inline void LocateFlatOffset(int64_t flatOffset, uint16_t tensorCount, const int64_t* tensorDataCountList,
                             uint16_t& tensorIndex, int64_t& tensorOffset)
{
    for (uint16_t index = 0; index < tensorCount; ++index) {
        if (flatOffset < tensorDataCountList[index]) {
            tensorIndex = index;
            tensorOffset = flatOffset;
            return;
        }
        flatOffset -= tensorDataCountList[index];
    }
    tensorIndex = tensorCount - 1;
    tensorOffset = tensorDataCountList[tensorIndex] - 1;
}

template <uint32_t INPUT_FLOW_COUNT, uint32_t EXTRA_FLOAT_BUFFER_COUNT = 0U, bool USE_OUTPUT_DTYPE = false>
ge::graphStatus Build(gert::TilingContext* context)
{
    static_assert(INPUT_FLOW_COUNT > 0U, "a flat foreach tiler needs at least one input flow");

    auto platformInfo = context->GetPlatformInfo();
    auto computeNodeInfo = context->GetComputeNodeInfo();
    if (platformInfo == nullptr || computeNodeInfo == nullptr) {
        OP_LOGE(context, "platform or compute-node metadata is null");
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t platformCoreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (platformCoreNum == 0U || ubSize == 0U) {
        OP_LOGE(context, "invalid platform resources: coreNum=%lu, ubSize=%lu", platformCoreNum, ubSize);
        return ge::GRAPH_FAILED;
    }

    uint16_t tensorCount = static_cast<uint16_t>(computeNodeInfo->GetInputInstanceInfo(0)->GetInstanceNum());
    ge::DataType dataType = context->GetDynamicInputDesc(0, 0)->GetDataType();
    uint32_t storageBytes = GetStorageBytes(dataType);
    if (storageBytes == 0U) {
        OP_LOGE(context, "validator admitted a dtype unsupported by the flat transport");
        return ge::GRAPH_FAILED;
    }

    ForeachFlatTilingData* tilingData = context->GetTilingData<ForeachFlatTilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context, "tiling data storage is null");
        return ge::GRAPH_FAILED;
    }
    uint32_t outputStorageBytes = storageBytes;
    if constexpr (USE_OUTPUT_DTYPE) {
        auto outputDesc = context->GetOutputDesc(0);
        if (outputDesc == nullptr) {
            OP_LOGE(context, "output descriptor is null");
            return ge::GRAPH_FAILED;
        }
        outputStorageBytes = GetStorageBytes(outputDesc->GetDataType());
        if (outputStorageBytes == 0U) {
            OP_LOGE(context, "validator admitted an output dtype unsupported by the flat transport");
            return ge::GRAPH_FAILED;
        }
    }
    *tilingData = ForeachFlatTilingData{};

    int64_t totalDataCount = 0;
    for (uint16_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex) {
        int64_t dataCount = context->GetDynamicInputShape(0, tensorIndex)->GetStorageShape().GetShapeSize();
        tilingData->tensorDataCountList[tensorIndex] = dataCount;
        totalDataCount += dataCount;
    }

    uint64_t bytesPerElement = static_cast<uint64_t>(BUFFER_DEPTH) *
                               (static_cast<uint64_t>(INPUT_FLOW_COUNT) * storageBytes + outputStorageBytes);
    if (storageBytes == sizeof(uint16_t)) {
        bytesPerElement += static_cast<uint64_t>(EXTRA_FLOAT_BUFFER_COUNT) * sizeof(float);
    }
    uint64_t rawTileElements = ubSize / bytesPerElement;
    uint32_t minStorageBytes = std::min(storageBytes, outputStorageBytes);
    uint32_t alignElements = DMA_ALIGN_BYTES / Gcd(DMA_ALIGN_BYTES, minStorageBytes);
    uint64_t tileElements = rawTileElements / alignElements * alignElements;
    if (tileElements == 0U || tileElements > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context, "UB is insufficient or tile size exceeds the tiling schema");
        return ge::GRAPH_FAILED;
    }
    tilingData->tileElems = static_cast<uint32_t>(tileElements);

    int64_t elementsPerBlock = MIN_CORE_WORK_BYTES / std::max(storageBytes, outputStorageBytes);
    int64_t coreCount = 1;
    if (totalDataCount > 0) {
        coreCount = std::min<int64_t>(
            static_cast<int64_t>(std::min<uint64_t>(platformCoreNum, FOREACH_FLAT_MAX_CORE_COUNT)),
            CeilDiv(totalDataCount, elementsPerBlock));
    }

    if (totalDataCount == 0) {
        tilingData->tensorEndOffsetList[0] = -1;
    } else {
        int64_t totalBlocks = CeilDiv(totalDataCount, elementsPerBlock);
        int64_t baseBlocks = totalBlocks / coreCount;
        int64_t remainderBlocks = totalBlocks % coreCount;
        int64_t flatCursor = 0;
        for (int64_t coreIndex = 0; coreIndex < coreCount; ++coreIndex) {
            int64_t targetCount = (baseBlocks + static_cast<int64_t>(coreIndex < remainderBlocks)) * elementsPerBlock;
            int64_t currentCount = std::min(targetCount, totalDataCount - flatCursor);
            LocateFlatOffset(flatCursor, tensorCount, tilingData->tensorDataCountList,
                             tilingData->tensorStartList[coreIndex], tilingData->tensorStartOffsetList[coreIndex]);
            LocateFlatOffset(flatCursor + currentCount - 1, tensorCount, tilingData->tensorDataCountList,
                             tilingData->tensorEndList[coreIndex], tilingData->tensorEndOffsetList[coreIndex]);
            flatCursor += currentCount;
        }
    }

    context->SetBlockDim(static_cast<uint32_t>(coreCount));
    if (context->SetLocalMemorySize(static_cast<uint32_t>(ubSize)) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "failed to set local memory size");
        return ge::GRAPH_FAILED;
    }

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        OP_LOGE(context, "workspace size storage is null");
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = WORKSPACE_BYTES;
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

} // namespace ForeachFlatRegbaseTiling
} // namespace optiling

#endif // FOREACH_FLAT_REGBASE_TILING_H
