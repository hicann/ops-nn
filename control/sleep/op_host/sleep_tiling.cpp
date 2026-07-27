/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sleep_tiling.cpp
 * \brief Sleep operator tiling — A5 SIMT cycle pass-through
 *
 * cycles 从 input tensor（INT64）读取，def.cpp 中通过 .ValueDepend(REQUIRED)
 * 声明值依赖，确保 tiling 阶段 GetData 可用。
 * cycles 上限由 AICore 看门狗超时决定，不在代码中校验，详见 aclnnSleep.md。
 */

#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "control/sleep/op_kernel/sleep_tiling_data.h"
#include "control/sleep/op_kernel/sleep_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

struct SleepCompileInfo {};

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SleepTilingFunc(gert::TilingContext* context)
{
    auto cyclesTensor = context->GetRequiredInputTensor(0);
    OP_CHECK_IF(cyclesTensor == nullptr, OP_LOGE(context, "GetRequiredInputTensor cycles failed"),
                return ge::GRAPH_FAILED);

    const int64_t* cyclesPtr = cyclesTensor->GetData<int64_t>();
    OP_CHECK_IF(cyclesPtr == nullptr, OP_LOGE(context, "GetData cycles failed"), return ge::GRAPH_FAILED);
    int64_t cycles = *cyclesPtr;

    OP_CHECK_IF(cycles <= 0, OP_LOGE(context, "cycles must be positive, got: %ld", cycles), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    SleepTilingData* tiling = context->GetTilingData<SleepTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    OP_CHECK_IF(memset_s(tiling, sizeof(SleepTilingData), 0, sizeof(SleepTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();

    OP_CHECK_IF(socVersion != platform_ascendc::SocVersion::ASCEND950, OP_LOGE(context, "unsupported soc version"),
                return ge::GRAPH_FAILED);

    tiling->cycles = cycles;

    context->SetBlockDim(1);

    uint64_t tilingKey = GET_TPL_TILING_KEY(SLEEP_TPL_SCH_MODE);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSleep([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Sleep)
    .TilingInputsDataDependency({0})
    .Tiling(SleepTilingFunc)
    .TilingParse<SleepCompileInfo>(TilingParseForSleep);
} // namespace optiling
