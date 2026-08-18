/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_grad_simt_tiling.cpp
 * \brief SwigluGroupGrad SIMT tiling implementation (arch35, Ascend950)
 */

#include <algorithm>
#include "swiglu_group_grad_simt_tiling.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

constexpr int64_t WS_SYS_SIZE = 16 * 1024 * 1024;

// Keep these two boundaries identical to the dedicated branch in
// swiglu_group_grad_simt.h and to the RegBase IsCapable() rejection.
constexpr int64_t SIMT_ULTRAWIDE_MAX_ROWS_HOST = 2;
constexpr int64_t SIMT_ULTRAWIDE_MIN_H_HOST = 256 * 1024;
constexpr int64_t SIMT_MAX_THREAD_NUM_HOST = 1024;

ge::graphStatus SwigluGroupGradSimtTiling::GetPlatformInfo()
{
    return GetSwigluGroupGradPlatformInfo(context_, ubSize_, coreNumAll_);
}

ge::graphStatus SwigluGroupGradSimtTiling::GetShapeAttrsInfo()
{
    return GetSwigluGroupGradShapeAttrsInfo(context_, inputData);
}

bool SwigluGroupGradSimtTiling::IsCapable() { return true; }

ge::graphStatus SwigluGroupGradSimtTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus SwigluGroupGradSimtTiling::GetWorkspaceSize()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupGradSimtTiling::DoOpTiling()
{
    simtTiling = context_->GetTilingData<SwigluGroupGradSimtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, simtTiling);
    OP_CHECK_IF(
        memset_s(simtTiling, sizeof(SwigluGroupGradSimtTilingData), 0, sizeof(SwigluGroupGradSimtTilingData)) != EOK,
        OP_LOGE(context_->GetNodeName(), "memset simt tiling data error"), return ge::GRAPH_FAILED);

    if (inputData.totalRows == 0 || inputData.H == 0) {
        simtTiling->totalRows = 0;
        simtTiling->hiddenSize = 0;
        simtTiling->groupIndexG = 0;
        simtTiling->clampLimit = 0.0f;
        simtTiling->clampLimitRecp = 0.0f;
        schMode_ = TPL_SIMT_KERNEL;
        context_->SetBlockDim(1);
        OP_LOGI(
            context_->GetNodeName(),
            "Simt simtTiling: totalRows=%ld, H=%ld, clampLimit=%f, clampLimitRecp=%f, groupIndexG=%ld, schMode_=%ld",
            simtTiling->totalRows, simtTiling->hiddenSize, simtTiling->clampLimit, simtTiling->clampLimitRecp,
            simtTiling->groupIndexG, schMode_);
        return ge::GRAPH_SUCCESS;
    }

    simtTiling->totalRows = inputData.totalRows;
    simtTiling->hiddenSize = inputData.H;
    simtTiling->groupIndexG = inputData.groupIndexG;
    simtTiling->clampLimit = inputData.clampLimit;
    simtTiling->clampLimitRecp = (inputData.clampLimit > 0.0f) ? (1.0f / inputData.clampLimit) : 0.0f;
    schMode_ = TPL_SIMT_KERNEL;
    OP_LOGI(context_->GetNodeName(),
            "Simt simtTiling: totalRows=%ld, H=%ld, clampLimit=%f, clampLimitRecp=%f, groupIndexG=%ld, schMode_=%ld",
            simtTiling->totalRows, simtTiling->hiddenSize, simtTiling->clampLimit, simtTiling->clampLimitRecp,
            simtTiling->groupIndexG, schMode_);

    return ge::GRAPH_SUCCESS;
}

uint64_t SwigluGroupGradSimtTiling::GetTilingKey() const
{
    uint64_t key = GET_TPL_TILING_KEY(static_cast<uint32_t>(schMode_), static_cast<uint32_t>(inputData.hasClamp),
                                      static_cast<uint32_t>(inputData.isWeight),
                                      static_cast<uint32_t>(inputData.isYOrigin),
                                      static_cast<uint32_t>(inputData.isGroupIndex));

    OP_LOGI(context_->GetNodeName(),
            "Simt GetTilingKey: key=%lu, schMode=%ld, hasClamp=%ld, isWeight=%ld, isYOrigin=%ld, isGroupIndex=%ld", key,
            schMode_, inputData.hasClamp, inputData.isWeight, inputData.isYOrigin, inputData.isGroupIndex);
    return key;
}

ge::graphStatus SwigluGroupGradSimtTiling::PostTiling()
{
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);

    if (inputData.totalRows == 0 || inputData.H == 0) {
        context_->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // The original SIMT kernel owns work by row, so all established generic
    // shapes retain the exact original min(totalRows, coreNumAll_) launch.
    int64_t simtUsedCoreNum = std::min(inputData.totalRows, coreNumAll_);

    // The ultra-wide kernel instead distributes gradX over the global
    // (block, thread) grid. For T<=2, row-based launch would incorrectly use
    // only one or two AIV cores and serialize millions of hidden elements.
    // Estimate blocks from element work exactly like other SIMT tilings:
    // ceil(totalElements / threadsPerBlock), capped by the platform core count.
    // This narrow branch has at least 262144 hidden elements per row, so the
    // target million-element cases naturally use every available AIV core.
    bool useUltraWideSimt = inputData.totalRows <= SIMT_ULTRAWIDE_MAX_ROWS_HOST &&
                            inputData.H >= SIMT_ULTRAWIDE_MIN_H_HOST && inputData.isWeight == 1 &&
                            inputData.isYOrigin == 1;
    if (useUltraWideSimt) {
        int64_t totalElementCount = inputData.totalRows * inputData.H;
        int64_t threads = std::min(totalElementCount, SIMT_MAX_THREAD_NUM_HOST);
        int64_t blockNumByWork = totalElementCount / threads;
        if (totalElementCount % threads != 0) {
            ++blockNumByWork;
        }
        simtUsedCoreNum = std::min(blockNumByWork, coreNumAll_);
    }

    if (simtUsedCoreNum <= 0) {
        simtUsedCoreNum = 1;
    }
    context_->SetBlockDim(simtUsedCoreNum);

    OP_LOGI(context_->GetNodeName(),
            "Simt PostTiling: totalRows=%ld, H=%ld, ultraWide=%d, usedCoreNum=%ld, coreNumAll=%ld", inputData.totalRows,
            inputData.H, static_cast<int>(useUltraWideSimt), simtUsedCoreNum, coreNumAll_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SwigluGroupGrad, SwigluGroupGradSimtTiling, 100);

} // namespace optiling
