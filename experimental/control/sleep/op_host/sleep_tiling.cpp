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
 * \file sleep_tiling.cpp
 * \brief Sleep operator tiling — converts user cycles to SYS_CNT ticks.
 *
 * sleep(N) ⇒ wall time = N / 1.65e9 seconds.
 * Kernel busy-spins on GetSystemCycle(); each loop iteration costs multiple
 * SYS_CNT ticks so the effective counter rate is lower than the register rate.
 */

#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "experimental/control/sleep/op_kernel/sleep_tiling_data.h"
#include "experimental/control/sleep/op_kernel/sleep_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

// SCALE = 50e6 / 1.65e9 ≈ 1/33
// Physical meaning: each do-while loop iteration consumes ~33 SYS_CNT ticks.
// This value was measured on Ascend 910B3 (ascend910_93) platform by:
//   1. Running the kernel with a known number of user_cycles
//   2. Measuring actual elapsed SYS_CNT ticks via GetSystemCycle()
//   3. Deriving the ratio: elapsed_ticks / user_cycles ≈ 1/33
// The loop overhead includes: GetSystemCycle() call, comparison, branch.
// For cross-chip adaptation, this value may need re-measurement on different
// hardware platforms due to varying instruction timing characteristics.
constexpr long double SCALE = 50000000.0L / 1650000000.0L;

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
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* cyclesPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(cyclesPtr == nullptr, OP_LOGE(context, "GetAttr cycles failed"), return ge::GRAPH_FAILED);
    int64_t cycles = *cyclesPtr;

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    SleepTilingData* tiling = context->GetTilingData<SleepTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    OP_CHECK_IF(memset_s(tiling, sizeof(SleepTilingData), 0, sizeof(SleepTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    int64_t kernelCycles = 0;
    if (cycles > 0) {
        kernelCycles = static_cast<int64_t>(static_cast<long double>(cycles) * SCALE);
        if (kernelCycles < 1) {
            kernelCycles = 1;
        }
    }
    tiling->cycles = kernelCycles;

    context->SetBlockDim(1);

    uint64_t tilingKey = GET_TPL_TILING_KEY(SLEEP_TPL_SCH_MODE);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSleep([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Sleep).Tiling(SleepTilingFunc).TilingParse<SleepCompileInfo>(TilingParseForSleep);
} // namespace optiling
