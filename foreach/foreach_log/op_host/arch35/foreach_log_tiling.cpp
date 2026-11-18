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
 * \file foreach_log_tiling.cpp
 * \brief Operator-owned validation and dual-split tiling for ForeachLog
 *        (arch35 / Ascend950). Lists whose average tensor is small (avg <=
 *        TENSOR_BALANCED_AVG) split cores by whole consecutive tensors so each
 *        core drives at most a couple of per-tensor DMA pipelines — per-tensor
 *        pipeline overhead dominates when a core's range covers many tiny
 *        tensors; larger tensors keep the shared element-balanced split.
 */

#include <algorithm>
#include <limits>
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "../../../foreach_utils/op_host/foreach_flat_regbase_validator.h"
#include "../op_kernel/arch35/foreach_log_tiling_data.h"

namespace optiling {

struct ForeachLogCompileInfo {};

static bool IsForeachLogDtype(ge::DataType dtype) { return ForeachFlatRegbaseValidation::IsFloatFamily(dtype); }

static int64_t CeilDivInt64(int64_t value, int64_t divisor)
{
    return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

static void LocateFlatOffset(int64_t flatOffset, int32_t tensorCount, const int64_t* tensorDataCountList,
                             uint16_t& tensorIndex, int64_t& tensorOffset)
{
    for (int32_t index = 0; index < tensorCount; ++index) {
        if (flatOffset < tensorDataCountList[index]) {
            tensorIndex = static_cast<uint16_t>(index);
            tensorOffset = flatOffset;
            return;
        }
        flatOffset -= tensorDataCountList[index];
    }
    tensorIndex = static_cast<uint16_t>(tensorCount - 1);
    tensorOffset = tensorDataCountList[tensorIndex] - 1;
}

// Lists whose average tensor is no larger than this many elements split cores
// by whole tensors; per-core per-tensor pipeline overhead dominates otherwise.
constexpr int64_t TENSOR_BALANCED_AVG = 32768;
constexpr uint32_t BUFFER_DEPTH = 2U;
constexpr uint32_t DMA_ALIGN_BYTES = 32U;
constexpr int64_t MIN_CORE_WORK_BYTES = 4096;
constexpr uint64_t WORKSPACE_BYTES = 32U;

// An empty tensor list is legal (consistent with the aclnn entry): tiling succeeds and
// the kernel runs as a single-core no-op. The shared flat runner skips a range whose
// endOffset == -1 without resolving any tensor pointer, so no GM access happens.
static ge::graphStatus ForeachLogEmptyListNoOp(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context, "platform metadata is null");
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0U) {
        OP_LOGE(context, "invalid platform resources: ubSize=%lu", ubSize);
        return ge::GRAPH_FAILED;
    }
    ForeachLogTilingData* tilingData = context->GetTilingData<ForeachLogTilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context, "tiling data storage is null");
        return ge::GRAPH_FAILED;
    }
    *tilingData = ForeachLogTilingData{};
    tilingData->tileElems = 1U; // keep the mover's InitBuffer positive on the no-op core
    tilingData->tensorEndOffsetList[0] = -1;
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        OP_LOGE(context, "workspace size storage is null");
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = WORKSPACE_BYTES;
    if (context->SetLocalMemorySize(static_cast<uint32_t>(ubSize)) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "failed to set local memory size");
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(1);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ForeachLogTilingFunc(gert::TilingContext* context)
{
    // Empty-list short-circuit must precede ValidateHomogeneous, which rejects a list
    // of size 0 and would turn the legal aclnn no-op entry into a tiling failure.
    auto computeNodeInfoEarly = context->GetComputeNodeInfo();
    if (computeNodeInfoEarly == nullptr) {
        OP_LOGE(context, "compute-node metadata is null");
        return ge::GRAPH_FAILED;
    }
    auto instanceInfoEarly = computeNodeInfoEarly->GetInputInstanceInfo(0);
    if (instanceInfoEarly == nullptr) {
        OP_LOGE(context, "input tensor-list metadata is null");
        return ge::GRAPH_FAILED;
    }
    const int32_t earlyTensorCount = static_cast<int32_t>(instanceInfoEarly->GetInstanceNum());
    if (earlyTensorCount == 0) {
        return ForeachLogEmptyListNoOp(context);
    }

    if (ForeachFlatRegbaseValidation::ValidateHomogeneous<1>(context, IsForeachLogDtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

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

    const int32_t tensorCount = static_cast<int32_t>(computeNodeInfo->GetInputInstanceInfo(0)->GetInstanceNum());
    ForeachLogTilingData* tilingData = context->GetTilingData<ForeachLogTilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context, "tiling data storage is null");
        return ge::GRAPH_FAILED;
    }
    *tilingData = ForeachLogTilingData{};

    int64_t totalDataCount = 0;
    int64_t maxDataCount = 0;
    for (int32_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex) {
        int64_t dataCount = context->GetDynamicInputShape(0, tensorIndex)->GetStorageShape().GetShapeSize();
        // Same contract as the shared flat tiler: reject negative counts and int64
        // accumulation overflow before they can corrupt the per-core range math below.
        OP_CHECK_IF(dataCount < 0 || totalDataCount > std::numeric_limits<int64_t>::max() - dataCount,
                    OP_LOGE(context, "invalid or overflowing element count at tensor %d", tensorIndex),
                    return ge::GRAPH_FAILED);
        // The in-tensor offsets and per-core range fields below are int32 to
        // keep the tiling struct under the ~4KB launch budget; a tensor whose
        // element count cannot be represented there would overflow the fields
        // the kernel consumes as GM cursors.
        OP_CHECK_IF(dataCount > std::numeric_limits<int32_t>::max(),
                    OP_LOGE(context, "tensor %d has %lld elements, exceeding the int32 offset limit", tensorIndex,
                            static_cast<long long>(dataCount)),
                    return ge::GRAPH_FAILED);
        tilingData->tensorDataCountList[tensorIndex] = dataCount;
        totalDataCount += dataCount;
        maxDataCount = std::max(maxDataCount, dataCount);
    }
    tilingData->tensorCount = tensorCount;

    ge::DataType dataType = context->GetDynamicInputDesc(0, 0)->GetDataType();
    uint32_t storageBytes = (dataType == ge::DT_FLOAT) ? sizeof(float) : sizeof(uint16_t);
    // One double-buffered input queue plus one double-buffered output queue.
    uint64_t bytesPerElement = static_cast<uint64_t>(BUFFER_DEPTH) * static_cast<uint64_t>(storageBytes) * 2U;
    uint64_t rawTileElements = ubSize / bytesPerElement;
    uint64_t alignElements = DMA_ALIGN_BYTES / storageBytes;
    uint64_t tileElements = rawTileElements / alignElements * alignElements;
    if (tileElements == 0U || tileElements > 0xFFFFFFFFU) {
        OP_LOGE(context, "UB is insufficient for the hybrid transport tile");
        return ge::GRAPH_FAILED;
    }
    tilingData->tileElems = static_cast<uint32_t>(tileElements);

    const int64_t limitedCoreNum = static_cast<int64_t>(
        std::min<uint64_t>(platformCoreNum, FOREACH_LOG_MAX_CORE_COUNT));
    // A purely average-based gate misfires on skewed lists: 255 tiny tensors plus one
    // huge tensor still average below the threshold, and the tensor-balanced branch
    // would pin the whole huge tensor to a single core while the rest idle. Require the
    // largest tensor to fit the per-core budget as well; otherwise the element-balanced
    // split below spreads every tensor across cores.
    if (totalDataCount / tensorCount <= TENSOR_BALANCED_AVG && maxDataCount <= TENSOR_BALANCED_AVG) {
        // Tensor-balanced split: whole consecutive tensors per core.
        const int64_t coreCount = std::min<int64_t>(limitedCoreNum, tensorCount);
        const int64_t baseTensors = tensorCount / coreCount;
        const int64_t remainderTensors = tensorCount % coreCount;
        int32_t tensorCursor = 0;
        for (int64_t coreIndex = 0; coreIndex < coreCount; ++coreIndex) {
            const int64_t share = baseTensors + static_cast<int64_t>(coreIndex < remainderTensors);
            const int32_t startTensor = tensorCursor;
            const int32_t endTensor = tensorCursor + static_cast<int32_t>(share) - 1;
            tilingData->tensorStartList[coreIndex] = static_cast<uint16_t>(startTensor);
            tilingData->tensorEndList[coreIndex] = static_cast<uint16_t>(endTensor);
            tilingData->tensorStartOffsetList[coreIndex] = 0;
            tilingData->tensorEndOffsetList[coreIndex] = static_cast<int32_t>(
                tilingData->tensorDataCountList[endTensor] - 1);
            tensorCursor = endTensor + 1;
        }
        context->SetBlockDim(static_cast<uint32_t>(coreCount));
    } else {
        // Element-balanced split, same as the shared flat tiler.
        const int64_t elementsPerBlock = MIN_CORE_WORK_BYTES / static_cast<int64_t>(storageBytes);
        const int64_t coreCount = std::max<int64_t>(
            1, std::min<int64_t>(limitedCoreNum, CeilDivInt64(totalDataCount, elementsPerBlock)));
        const int64_t totalBlocks = CeilDivInt64(totalDataCount, elementsPerBlock);
        const int64_t baseBlocks = totalBlocks / coreCount;
        const int64_t remainderBlocks = totalBlocks % coreCount;
        int64_t flatCursor = 0;
        for (int64_t coreIndex = 0; coreIndex < coreCount; ++coreIndex) {
            const int64_t blockCount = baseBlocks + static_cast<int64_t>(coreIndex < remainderBlocks);
            // Overflow-safe target, kept in the shared tiler's guarded formulation so this
            // copy cannot drift from foreach_flat_regbase_tiling.h: when the aligned
            // target would exceed the remaining count (possible for totals close to
            // INT64_MAX), the remaining count is the minimum. Today the per-tensor
            // int32 limit and tensorCount <= 256 keep the product far below INT64_MAX,
            // but the guard makes that coupling a safety net instead of a precondition.
            int64_t currentCount = totalDataCount - flatCursor;
            if (blockCount <= currentCount / elementsPerBlock) {
                currentCount = blockCount * elementsPerBlock;
            }
            uint16_t startTensor = 0;
            int64_t startOffset = 0;
            uint16_t endTensor = 0;
            int64_t endOffset = -1;
            LocateFlatOffset(flatCursor, tensorCount, tilingData->tensorDataCountList, startTensor, startOffset);
            LocateFlatOffset(flatCursor + currentCount - 1, tensorCount, tilingData->tensorDataCountList, endTensor,
                             endOffset);
            OP_CHECK_IF(startOffset > std::numeric_limits<int32_t>::max() ||
                            endOffset > std::numeric_limits<int32_t>::max() ||
                            currentCount > std::numeric_limits<int32_t>::max(),
                        OP_LOGE(context, "core %lld flat range exceeds the int32 offset limit",
                                static_cast<long long>(coreIndex)),
                        return ge::GRAPH_FAILED);
            tilingData->tensorStartList[coreIndex] = startTensor;
            tilingData->tensorEndList[coreIndex] = endTensor;
            tilingData->tensorStartOffsetList[coreIndex] = static_cast<int32_t>(startOffset);
            tilingData->tensorEndOffsetList[coreIndex] = static_cast<int32_t>(endOffset);
            flatCursor += currentCount;
        }
        context->SetBlockDim(static_cast<uint32_t>(coreCount));
    }

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

static ge::graphStatus TilingParseForForeachLog([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachLog).Tiling(ForeachLogTilingFunc).TilingParse<ForeachLogCompileInfo>(TilingParseForForeachLog);

} // namespace optiling
