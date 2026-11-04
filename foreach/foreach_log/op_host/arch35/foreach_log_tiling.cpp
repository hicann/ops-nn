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
 * \brief tiling implementation for foreach_log (arch35 / Ascend950)
 */

#include <limits>
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/foreach_log_tiling_data.h"
#include "../../op_kernel/arch35/foreach_log_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int32_t INPUT_IDX_0 = 0;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t MIN_PER_CORE = 1024;
constexpr int32_t MAX_TENSOR_NUM = 256;
constexpr size_t MAX_SUPPORT_DIMS_NUMS = 8;

struct ForeachLogCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillTensorListInfo(gert::TilingContext* context, ForeachLogTilingData* tiling,
                                          uint64_t tensorNum, ge::DataType& dataType, int64_t& totalElements)
{
    // An empty tensor list is legal (consistent with the aclnn entry): keep the default dtype
    // for the tiling key and a zero element total.
    if (tensorNum == 0) {
        return ge::GRAPH_SUCCESS;
    }
    for (uint64_t i = 0; i < tensorNum; i++) {
        auto memberDesc = context->GetDynamicInputDesc(INPUT_IDX_0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, memberDesc);
        // The tiling key is chosen from a single dtype, so every member must share it; a mixed
        // list would otherwise be computed with the first member's dtype.
        if (i == 0) {
            dataType = memberDesc->GetDataType();
        } else {
            OP_CHECK_IF(memberDesc->GetDataType() != dataType,
                        OP_LOGE(context, "the dtypes of all tensors in the tensor list must be the same"),
                        return ge::GRAPH_FAILED);
        }
        auto shapePtr = context->GetDynamicInputShape(INPUT_IDX_0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
        auto storageShape = shapePtr->GetStorageShape();
        // The arch22 common tiling rejects tensors with more than 8 dims; keep arch35
        // consistent so a rank that the aclnn entry rejects cannot pass in graph mode.
        OP_CHECK_IF(storageShape.GetDimNum() > MAX_SUPPORT_DIMS_NUMS,
                    OP_LOGE(context, "tensor %lu has %zu dims, which exceeds the limit %zu", i,
                            storageShape.GetDimNum(), MAX_SUPPORT_DIMS_NUMS),
                    return ge::GRAPH_FAILED);
        int64_t numel = storageShape.GetShapeSize();
        // A negative size means an unresolved/unknown dimension (e.g. -1/-2) and cannot be
        // converted into a concrete element count.
        OP_CHECK_IF(numel < 0, OP_LOGE(context, "tensor %lu has unknown or negative shape size", i),
                    return ge::GRAPH_FAILED);
        // Checked accumulation: totalElements feeds CeilDiv/BlockDim below, so an int64
        // overflow would silently corrupt the core split.
        OP_CHECK_IF(numel > std::numeric_limits<int64_t>::max() - totalElements,
                    OP_LOGE(context, "total elements overflow at tensor %lu", i), return ge::GRAPH_FAILED);
        tiling->tensorElements[i] = numel;
        totalElements += numel;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ForeachLogTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Get tensor list info
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfoPtr);
    auto idxInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(INPUT_IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxInstanceInfoPtr);
    uint64_t tensorNum = idxInstanceInfoPtr->GetInstanceNum();
    // The tiling data arrays are sized to MAX_TENSOR_NUM; accepting more members would
    // silently drop the tail, so reject the oversized list instead.
    OP_CHECK_IF(tensorNum > MAX_TENSOR_NUM,
                OP_LOGE(context, "tensorNum[%lu] should be less than or equal to %u", tensorNum, MAX_TENSOR_NUM),
                return ge::GRAPH_FAILED);

    // 3. Fill tiling data
    ForeachLogTilingData* tiling = context->GetTilingData<ForeachLogTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ForeachLogTilingData), 0, sizeof(ForeachLogTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 4. Collect per-member dtype and element counts. An empty tensor list is legal
    //    (consistent with the aclnn entry), so the scan is skipped and the default dtype kept.
    ge::DataType dataType = ge::DT_FLOAT;
    tiling->tensorCount = static_cast<int32_t>(tensorNum);
    int64_t totalElements = 0;
    OP_CHECK_IF(FillTensorListInfo(context, tiling, tensorNum, dataType, totalElements) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "fill tensor list info failed"), return ge::GRAPH_FAILED);
    tiling->totalElements = totalElements;

    // 5. Calculate core split
    if (totalElements == 0) {
        tiling->needCoreNum = 1;
        context->SetBlockDim(1);
    } else {
        int64_t needCoreNum = std::min(coreNum, Ops::Base::CeilDiv(totalElements, MIN_PER_CORE));
        // A non-positive core count would make SetBlockDim invalid.
        OP_CHECK_IF(needCoreNum <= 0, OP_LOGE(context, "needCoreNum is invalid"), return ge::GRAPH_FAILED);
        tiling->needCoreNum = static_cast<int32_t>(needCoreNum);
        context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
    }

    // 6. Set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    // 7. Set local memory size. DCACHE_SIZE is reserved for the cache, so UB must be
    //    strictly larger: at equality the result is 0, and below it the unsigned
    //    subtraction wraps around before the uint32 narrowing.
    OP_CHECK_IF(ubSize <= DCACHE_SIZE, OP_LOGE(context, "ubSize[%lu] should be greater than %u", ubSize, DCACHE_SIZE),
                return ge::GRAPH_FAILED);
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE));

    // 8. Set tiling key by dtype
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_LOG_TPL_SCH_MODE_FLOAT);
    } else if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_LOG_TPL_SCH_MODE_FLOAT16);
    } else if (dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_LOG_TPL_SCH_MODE_BF16);
    } else {
        OP_LOGE(context, "unsupported dtype for foreach_log");
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForForeachLog([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachLog).Tiling(ForeachLogTilingFunc).TilingParse<ForeachLogCompileInfo>(TilingParseForForeachLog);

} // namespace optiling
