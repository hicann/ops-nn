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
 * \file foreach_asin_tiling.cpp
 * \brief tiling implementation for foreach_asin
 */

#include <vector>
#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "foreach/foreach_asin/op_kernel/arch35/foreach_asin_tiling_data.h"
#include "foreach/foreach_asin/op_kernel/arch35/foreach_asin_tiling_key.h"

using namespace AscendC;

namespace optiling {

constexpr int64_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16777216; // 16M
constexpr int64_t MIN_PER_CORE_ELEMENTS = 1024;
constexpr int32_t ALIGN_SIZE = 32;

struct ForeachAsinCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

/**
 * @brief Tiling 主函数：计算切分参数并设置 tiling data
 */
static ge::graphStatus ForeachAsinTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    auto compileInfo = reinterpret_cast<const ForeachAsinCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    int64_t maxCoreNum = compileInfo->coreNum;
    int64_t ubSize = compileInfo->ubSize;
    OP_CHECK_IF((ubSize <= DCACHE_SIZE),
        OP_LOGE(context, "ub size less than DCache Size"), return ge::GRAPH_FAILED);
    int64_t ubSizeActual = ubSize - DCACHE_SIZE;

    // 2. 获取输入 tensor 列表信息
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfoPtr);
    auto instanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, instanceInfoPtr);
    uint64_t tensorNum = instanceInfoPtr->GetInstanceNum();

    // 3. 初始化 tiling data
    ForeachAsinTilingData* tilingData = context->GetTilingData<ForeachAsinTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->tensorNum = static_cast<int32_t>(tensorNum);
    tilingData->tensorCumulativeOffset[0] = 0;
    tilingData->totalElements = 0;

    // 4. 计算每个 tensor 的元素数和累计偏移量
    for (int32_t i = 0; i < tilingData->tensorNum; i++) {
        auto tensorShapePtr = context->GetDynamicInputShape(0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, tensorShapePtr);
        auto tensorShape = tensorShapePtr->GetStorageShape();
        int64_t numel = tensorShape.GetShapeSize();
        tilingData->totalElements += numel;
        tilingData->tensorCumulativeOffset[i + 1] = tilingData->totalElements;
    }

    // 5. 计算核切分参数
    if (tilingData->totalElements == 0) {
        tilingData->perCoreElements = MIN_PER_CORE_ELEMENTS;
        tilingData->needCoreNum = 1;
    } else {
        tilingData->perCoreElements = std::max(MIN_PER_CORE_ELEMENTS,
            (tilingData->totalElements + maxCoreNum - 1) / maxCoreNum);
        // 对齐到 32
        tilingData->perCoreElements = ((tilingData->perCoreElements + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        tilingData->needCoreNum = static_cast<int32_t>(
            (tilingData->totalElements + tilingData->perCoreElements - 1) / tilingData->perCoreElements);
    }

    // 6. 根据数据类型确定 tiling key
    auto inputDesc = context->GetDynamicInputDesc(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dtype = inputDesc->GetDataType();

    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ASIN_TPL_DTYPE_FP32);
    } else if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ASIN_TPL_DTYPE_FP16);
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ASIN_TPL_DTYPE_BF16);
    } else {
        OP_LOGE(context, "Unsupported dtype: %d", static_cast<int>(dtype));
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);

    // 7. 设置核数和 workspace
    context->SetBlockDim(tilingData->needCoreNum);
    auto workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = ASCENDC_TOOLS_WORKSPACE;

    // 8. 设置本地内存大小
    auto res = context->SetLocalMemorySize(ubSizeActual);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "SetLocalMemorySize ubSize = %ld failed.", ubSizeActual),
        return ge::GRAPH_FAILED);

    // 9. 打印 tiling 信息
    OP_LOGI(context, "ForeachAsin tiling: tensorNum=%d, totalElements=%ld, "
        "perCoreElements=%ld, needCoreNum=%d, tilingKey=%lu",
        tilingData->tensorNum, tilingData->totalElements,
        tilingData->perCoreElements, tilingData->needCoreNum, tilingKey);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief TilingParse 回调：获取平台信息
 */
static ge::graphStatus TilingParseForForeachAsin(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingParseForForeachAsin entering.");
    auto compileInfo = context->GetCompiledInfo<ForeachAsinCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
        OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
        OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachAsin)
    .Tiling(ForeachAsinTilingFunc)
    .TilingParse<ForeachAsinCompileInfo>(TilingParseForForeachAsin);

} // namespace optiling
