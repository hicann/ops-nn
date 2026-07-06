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
 * \file swiglu_group_quant_grad_tiling.cpp
 * \brief SwiGLU Group Dynamic Quant Backward tiling implementation
 */

#include "register/op_def_registry.h"
#include "swiglu_group_quant_grad_tiling_utils.h"

namespace optiling {

constexpr uint32_t BATCH_MODE = 1;

static ge::graphStatus Tiling4SwigluGroupQuantGrad(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4SwigluGroupQuantGrad start.");
    context->SetScheduleMode(BATCH_MODE);

    SwigluGroupQuantGradCompileInfo compileInfo;
    SwigluGroupQuantGradTilingData tilingData;
    if (GetCompileInfo(context, compileInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOpAllParams(context, compileInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetBasicTilingData(context, compileInfo, tilingData);
    CalculateTilingParams(context, compileInfo, tilingData);
    if (SetTilingDataToContext(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalTokens = tilingData.get_totalTokens();
    uint32_t coreNumAll = tilingData.get_coreNumAll();
    uint32_t usedCoreNum = (totalTokens < coreNumAll) ? totalTokens : coreNumAll;

    context->SetBlockDim(usedCoreNum);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaces[0] = sysWorkspaceSize + usedCoreNum * BLOCK_SIZE;

    OP_LOGD(context, "Tiling4SwigluGroupQuantGrad end. usedCoreNum=%u, totalTokens=%u, tileH=%u, tileTokens=%u",
            usedCoreNum, totalTokens, tilingData.get_tileH(), tilingData.get_tileTokens());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4SwigluGroupQuantGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4SwigluGroupQuantGrad start and end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwigluGroupQuantGrad)
    .Tiling(Tiling4SwigluGroupQuantGrad)
    .TilingParse<CoreCompileInfo>(TilingPrepare4SwigluGroupQuantGrad);

} // namespace optiling