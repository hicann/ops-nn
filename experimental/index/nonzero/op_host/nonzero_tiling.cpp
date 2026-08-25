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
 * \file nonzero_tiling.cpp
 * \brief NonZero tiling — computes rowsPerCore and workspace size
 */
#include <set>

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/nonzero_tiling_data.h"
#include "../op_kernel/nonzero_tiling_key.h"

namespace optiling {

constexpr int64_t WS_PER_CORE = 8; // 8 int32s per core workspace

// Supported input data types (must match the tiling key DATATYPE_DECL list).
static const std::set<ge::DataType> SUPPORTED_DTYPES = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32};

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, uint32_t usedCoreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // System workspace: the AscendC lib API reserves a fixed, platform-defined
    // workspace for kernel execution (same pattern as the sibling index ops).
    // It is required by the framework, not sized by the op, so keep it as a
    // separate term instead of folding it into the user workspace.
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // User workspace: one 8-int32 count header per launched core. The kernel is
    // launched with usedCoreNum cores (single core in the framework build), so
    // size by usedCoreNum, NOT the platform core count (GetCoreNum()).
    size_t usrSize = usedCoreNum * WS_PER_CORE * sizeof(int32_t);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// Select the per-dtype template via the tiling key (must be called on every return path).
static void SetTilingKey(gert::TilingContext* context, ge::DataType dataType)
{
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
}

static ge::graphStatus NonzeroTilingFunc(gert::TilingContext* context)
{
    NonzeroTilingData* tiling = context->GetTilingData<NonzeroTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(NonzeroTilingData), 0, sizeof(NonzeroTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // ---------- 0. Input dtype (drives tiling key dispatch) ----------
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(SUPPORTED_DTYPES.count(dataType) == 0, OP_LOGE(context, "unsupported input dtype"),
                return ge::GRAPH_FAILED);

    // ---------- 1. Platform ----------
    uint64_t ubSize;
    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // ---------- 2. Input shape → 2D ----------
    auto shapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
    auto xShape = shapePtr->GetStorageShape();
    int64_t ndim = static_cast<int64_t>(xShape.GetDimNum());
    int64_t totalRows = static_cast<int64_t>(xShape.GetDim(0));
    int64_t cols = 1;
    for (int64_t d = 1; d < ndim; d++) {
        cols *= static_cast<int64_t>(xShape.GetDim(d));
    }

    // ---------- 3. Core assignment ----------
    // Single core: the aclnn contract requires the output buffer y to be packed
    // contiguously from offset 0 ([num_nonzero, 2]). Multi-core would write each
    // core's pairs at its own region, leaving holes the framework host cannot
    // gather (no second kernel / host gather in the aclnn executor), and a
    // cross-core barrier (SyncAll) deadlocks on this platform even at full
    // blockDim (probe-verified). So all rows are scanned by one core, which
    // writes pairs directly into y at packed offsets.
    uint32_t usedCoreNum = 1;
    int64_t rowsPerCore = (totalRows + usedCoreNum - 1) / usedCoreNum;

    // ---------- 4. Write tiling ----------
    tiling->totalRows = totalRows;
    tiling->cols = cols;
    tiling->rowsPerCore = rowsPerCore;
    tiling->rowStride = cols;       // contiguous after reshape
    tiling->wsStride = WS_PER_CORE; // count header slot; block 0 uses slot 0

    // ---------- 5. Tiling key / Workspace / blockDim ----------
    // Set the per-dtype tiling key so the framework dispatches the matching kernel template.
    SetTilingKey(context, dataType);
    OP_CHECK_IF(GetWorkspaceSize(context, usedCoreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);
    context->SetBlockDim(usedCoreNum);

    return ge::GRAPH_SUCCESS;
}

struct NonzeroCompileInfo {};

static ge::graphStatus TilingParseForNonzero([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Nonzero).Tiling(NonzeroTilingFunc).TilingParse<NonzeroCompileInfo>(TilingParseForNonzero);
} // namespace optiling
