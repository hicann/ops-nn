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
 * \file foreach_cosh_tiling.cpp
 * \brief tiling implementation for foreach_cosh
 */

#include <limits>
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/foreach_cosh_tiling_data.h"
#include "../../op_kernel/arch35/foreach_cosh_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int32_t INPUT_IDX_0 = 0;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t ALIGN_SIZE = 32;
constexpr int32_t MAX_SUPPORT_RANK = 8;

struct ForeachCoshCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= static_cast<uint64_t>(DCACHE_SIZE),
                OP_LOGE(context, "ubSize[%lu] should be greater than DCACHE_SIZE[%u]", ubSize, DCACHE_SIZE),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ForeachCoshTilingFunc(gert::TilingContext* context)
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

    // 3. Fill tiling data
    ForeachCoshTilingData* tiling = context->GetTilingData<ForeachCoshTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ForeachCoshTilingData), 0, sizeof(ForeachCoshTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Reject oversized lists against the tiling-data capacity constant so the bound and
    // the prefix-array size cannot drift apart if one of them changes.
    OP_CHECK_IF(tensorNum > static_cast<uint64_t>(FOREACH_COSH_MAX_TENSOR_NUM),
                OP_LOGE(context, "tensorNum[%lu] should be less than or equal to %u", tensorNum,
                        static_cast<uint32_t>(FOREACH_COSH_MAX_TENSOR_NUM)),
                return ge::GRAPH_FAILED);

    // 4. Determine dtype from element 0. An empty TensorList has no element 0 and is
    //    treated as a no-op; default to FLOAT so the tiling key stays valid.
    ge::DataType dataType = ge::DT_FLOAT;
    if (tensorNum > 0) {
        auto inputDesc = context->GetDynamicInputDesc(INPUT_IDX_0, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
        dataType = inputDesc->GetDataType();
    }

    tiling->tensorCount = static_cast<int32_t>(tensorNum);
    int64_t totalElements = 0;

    // 5. Validate a single dtype across the list and accumulate prefix starts so the
    //    kernel can map any flattened index back to its tensor with a binary search.
    for (uint64_t i = 0; i < tensorNum; i++) {
        // The tiling key and SIMT kernel interpret the whole list with element 0's type.
        if (i > 0) {
            auto descPtr = context->GetDynamicInputDesc(INPUT_IDX_0, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, descPtr);
            OP_CHECK_IF(descPtr->GetDataType() != dataType,
                        OP_LOGE(context,
                                "foreach_cosh requires a single dtype across the TensorList; element %lu is %d but "
                                "element 0 is %d",
                                i, static_cast<int>(descPtr->GetDataType()), static_cast<int>(dataType)),
                        return ge::GRAPH_FAILED);
        }
        auto shapePtr = context->GetDynamicInputShape(INPUT_IDX_0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
        auto storageShape = shapePtr->GetStorageShape();
        OP_CHECK_IF(storageShape.GetDimNum() > static_cast<size_t>(MAX_SUPPORT_RANK),
                    OP_LOGE(context, "rank[%zu] of tensor %lu exceeds the supported maximum %d",
                            storageShape.GetDimNum(), i, MAX_SUPPORT_RANK),
                    return ge::GRAPH_FAILED);
        int64_t numel = storageShape.GetShapeSize();
        OP_CHECK_IF(numel < 0 || totalElements > std::numeric_limits<int64_t>::max() - numel,
                    OP_LOGE(context, "invalid shape size[%ld] for tensor %lu of foreach_cosh", numel, i),
                    return ge::GRAPH_FAILED);
        tiling->tensorStarts[i] = totalElements;
        totalElements += numel;
    }
    tiling->tensorStarts[tensorNum] = totalElements;
    tiling->totalElements = totalElements;

    // 6. Calculate core split over the flattened element range. The kernel gives each
    //    core a contiguous chunk, so an empty list is a single no-op core.
    if (totalElements == 0) {
        tiling->needCoreNum = 1;
        tiling->perCoreElements = 0;
        tiling->lastCoreElements = 0;
        context->SetBlockDim(1);
    } else {
        int64_t perCoreElements = Ops::Base::CeilDiv(totalElements, coreNum);
        // Align only when the chunk is large enough: for small tensor lists (e.g. tens of
        // 0-dim scalars) rounding up to 32 would squash needCoreNum to 1 and serialise the
        // per-tensor VF launches on a single core (measured ~0.7us per tensor). Small lists
        // keep the unaligned chunk so needCoreNum stays close to min(coreNum, totalElements).
        if (perCoreElements >= ALIGN_SIZE) {
            perCoreElements = Ops::Base::CeilAlign(perCoreElements, ALIGN_SIZE);
        }
        OP_CHECK_IF(perCoreElements <= 0, OP_LOGE(context, "perCoreElements[%ld] should be positive", perCoreElements),
                    return ge::GRAPH_FAILED);
        int64_t needCoreNum = Ops::Base::CeilDiv(totalElements, perCoreElements);
        OP_CHECK_IF(
            needCoreNum > std::numeric_limits<int32_t>::max(),
            OP_LOGE(context, "needCoreNum[%ld] exceeds the INT32_MAX limit of foreach_cosh tiling data", needCoreNum),
            return ge::GRAPH_FAILED);
        tiling->perCoreElements = perCoreElements;
        tiling->needCoreNum = static_cast<int32_t>(needCoreNum);
        tiling->lastCoreElements = totalElements - perCoreElements * (needCoreNum - 1);
        context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
    }

    // 7. Set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    // 8. Set local memory size
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE));

    // 9. Set tiling key by dtype (FLOAT for the empty-list no-op)
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_COSH_TPL_SCH_MODE_FLOAT);
    } else if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_COSH_TPL_SCH_MODE_FLOAT16);
    } else if (dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_COSH_TPL_SCH_MODE_BF16);
    } else {
        OP_LOGE(context, "unsupported dtype for foreach_cosh: %d", dataType);
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForForeachCosh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachCosh)
    .Tiling(ForeachCoshTilingFunc)
    .TilingParse<ForeachCoshCompileInfo>(TilingParseForForeachCosh);

} // namespace optiling
