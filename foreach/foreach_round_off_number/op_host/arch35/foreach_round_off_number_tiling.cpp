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
 * \file foreach_round_off_number_tiling.cpp
 * \brief Tiling implementation for foreach_round_off_number operator.
 *        Computes core split and tensor element counts for tensor list.
 */

#include <vector>
#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "foreach_round_off_number_tiling.h"

using namespace AscendC;

namespace optiling {

constexpr int64_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t SINGLE_CORE_MIN_ELEMENTS = 1024;
constexpr int64_t INPUT_IDX = 0;

// Get tiling key from dtype
static uint64_t GetTilingKeyByDtype(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return 0;
        case ge::DT_FLOAT16:
            return 1;
        case ge::DT_BF16:
            return 2;
        default:
            return 0;
    }
}

static ge::graphStatus GetPlatformInfoFallback(gert::TilingContext* context, int64_t& coreNum, int64_t& ubSize)
{
    // Try compileInfo first (populated by TilingParse on real device)
    auto compileInfo = reinterpret_cast<const ForeachRoundOffNumberCompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr && compileInfo->coreNum > 0 && compileInfo->ubSize > 0) {
        coreNum = compileInfo->coreNum;
        ubSize = compileInfo->ubSize;
        return ge::GRAPH_SUCCESS;
    }
    // Fallback: get platform info directly from TilingContext (works in simulation)
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ub = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        ubSize = static_cast<int64_t>(ub);
        if (coreNum > 0 && ubSize > 0) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

static ge::graphStatus ForeachRoundOffNumberTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info (with fallback for simulation)
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    OP_CHECK_IF(GetPlatformInfoFallback(context, coreNum, ubSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Failed to get platform info"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((ubSize <= DCACHE_SIZE),
        OP_LOGE(context, "ubSize %ld <= DCACHE_SIZE %ld", ubSize, DCACHE_SIZE),
        return ge::GRAPH_FAILED);
    ubSize = ubSize - DCACHE_SIZE;

    // 2. Get tensor count from dynamic input
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfoPtr);
    auto idxInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxInstanceInfoPtr);
    uint64_t tensorNum = idxInstanceInfoPtr->GetInstanceNum();

    OP_CHECK_IF((static_cast<int32_t>(tensorNum) > MAX_TENSOR_NUM_FOREACH_ROUND_HOST),
        OP_LOGE(context, "tensorNum %lu exceeds MAX_TENSOR_NUM %d",
                tensorNum, MAX_TENSOR_NUM_FOREACH_ROUND_HOST),
        return ge::GRAPH_FAILED);

    // 3. Compute element counts and total elements
    int64_t totalElements = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    ForeachRoundOffNumberTilingData* tilingData = context->GetTilingData<ForeachRoundOffNumberTilingData>();

    for (uint64_t i = 0; i < tensorNum; i++) {
        auto idxTensorShapePtr = context->GetDynamicInputShape(INPUT_IDX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, idxTensorShapePtr);
        auto idxTensorShape = idxTensorShapePtr->GetStorageShape();
        int64_t numel = idxTensorShape.GetShapeSize();
        tilingData->tensorElements[i] = numel;
        totalElements += numel;

        if (i == 0) {
            auto idxTensorDtypePtr = context->GetDynamicInputDesc(INPUT_IDX, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, idxTensorDtypePtr);
            dataType = idxTensorDtypePtr->GetDataType();
        }
    }

    // 4. Compute core split
    int64_t needCoreNum = coreNum;
    if (totalElements > 0) {
        needCoreNum = std::min(coreNum,
            (totalElements + SINGLE_CORE_MIN_ELEMENTS - 1) / SINGLE_CORE_MIN_ELEMENTS);
    }
    needCoreNum = std::max(needCoreNum, static_cast<int64_t>(1));

    // 5. Fill TilingData
    tilingData->needCoreNum = static_cast<int32_t>(needCoreNum);
    tilingData->tensorCount = static_cast<int32_t>(tensorNum);
    tilingData->totalElements = totalElements;

    // 6. Set context
    context->SetBlockDim(needCoreNum);
    context->SetTilingKey(GetTilingKeyByDtype(dataType));

    // 7. Set local memory size
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "SetLocalMemorySize ubSize=%ld failed", ubSize),
        return ge::GRAPH_FAILED);

    // 8. Set workspace (not used)
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

// TilingParse callback
static ge::graphStatus TilingParseForForeachRoundOffNumber(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ForeachRoundOffNumberCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
        OP_LOGE(context, "Failed to get core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
        OP_LOGE(context, "Failed to get ub size."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Tiling registration
IMPL_OP_OPTILING(ForeachRoundOffNumber)
    .Tiling(ForeachRoundOffNumberTilingFunc)
    .TilingParse<ForeachRoundOffNumberCompileInfo>(TilingParseForForeachRoundOffNumber);

} // namespace optiling
