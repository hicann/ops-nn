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
 * \file optimized_transducer_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "optimized_transducer/op_kernel/optimized_transducer_tiling_data.h"
#include "optimized_transducer/op_kernel/optimized_transducer_tiling_key.h"

namespace optiling {

struct OptimizedTransducerCompileInfo {};

static ge::graphStatus TilingParseForOptimizedTransducer([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, size_t usrSize)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus OptimizedTransducerTilingFunc(gert::TilingContext* context)
{
    OptimizedTransducerTilingData *tiling = context->GetTilingData<OptimizedTransducerTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(OptimizedTransducerTilingData), 0, sizeof(OptimizedTransducerTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    const int64_t *blank = context->GetAttrs()->GetInt(0);
    const float *clamp = context->GetAttrs()->GetFloat(1);
    const bool *fused_log_softmax = context->GetAttrs()->GetBool(2);
    const bool *requires_grad = context->GetAttrs()->GetBool(3);
    const gert::StorageShape* logitsShape = context->GetInputShape(0);
    uint64_t batch_size = logitsShape->GetStorageShape().GetDim(0); // 批次数量
    uint64_t maxT = logitsShape->GetStorageShape().GetDim(1); // 输入序列长度的最大值
    uint64_t maxU = logitsShape->GetStorageShape().GetDim(2); // 目标序列长度的最大值 + 1
    uint64_t V = logitsShape->GetStorageShape().GetDim(3); // 输入类别数量
    coreNum = (batch_size < coreNum) ? batch_size : coreNum;
    uint64_t bigCoreNum = batch_size % coreNum;
    uint64_t smallCoreProcessNum = batch_size / coreNum;
    uint64_t bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
    tiling->maxT = maxT;
    tiling->maxU = maxU;
    tiling->V = V;
    tiling->batch_size = batch_size;
    tiling->bigCoreNum = bigCoreNum;
    tiling->bigCoreProcessNum = bigCoreProcessNum;
    tiling->smallCoreProcessNum = smallCoreProcessNum;
    tiling->blank = *blank;
    tiling->clamp = *clamp;
    tiling->fused_log_softmax = *fused_log_softmax;
    tiling->requires_grad = *requires_grad;
    OP_CHECK_IF(
        GetWorkspaceSize(context, batch_size * (maxT * maxU * 4 + 63) / 64 * 64 * 5) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    uint64_t tilingKey = 0;
    auto dataType = context->GetInputTensor(0)->GetDataType();
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(TRANSDUCER_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    }
    else if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TRANSDUCER_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    }
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(OptimizedTransducer).Tiling(OptimizedTransducerTilingFunc).TilingParse<OptimizedTransducerCompileInfo>(TilingParseForOptimizedTransducer);

} // namespace optiling

