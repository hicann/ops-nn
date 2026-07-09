/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#include <cstring>
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/multilabel_margin_loss_tiling_data.h"
#include "../op_kernel/multilabel_margin_loss_tiling_key.h"

namespace optiling {

struct MultilabelMarginLossCompileInfo {};

constexpr uint32_t BATCH_MODE = 1;
constexpr int32_t REDUCTION_INVALID = -1;

// The reduction attr is a String ("none"/"mean"/"sum"), matching the aclnn launcher and every other
// loss op in this repo. Map it to the int code (0/1/2) the kernel expects. (Previously declared/read
// as Int, which silently mis-decoded the string bytes and made the kernel always take the sum/scalar
// path -> none & mean ST cases failed.) Returns REDUCTION_INVALID(-1) for an unknown value so the
// tiling entry can reject it defensively (aclnn already validates, but tiling is an independent entry).
static int32_t ParseReductionAttr(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return REDUCTION_INVALID;
    }
    const char* reductionStr = attrs->GetAttrPointer<char>(0);
    if (reductionStr == nullptr) {
        return REDUCTION_INVALID;
    }
    if (strcmp(reductionStr, "none") == 0) {
        return 0;
    }
    if (strcmp(reductionStr, "mean") == 0) {
        return 1;
    }
    if (strcmp(reductionStr, "sum") == 0) {
        return 2;
    }
    return REDUCTION_INVALID;
}

static ge::graphStatus MultilabelMarginLossTilingFunc(gert::TilingContext* context)
{
    MultilabelMarginLossTilingData tiling;

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    uint32_t N = 1;
    uint32_t C = 1;
    int dimNum = x1_shape->GetStorageShape().GetDimNum();
    if (dimNum >= 2) {
        N = static_cast<uint32_t>(x1_shape->GetStorageShape().GetDim(0));
        C = static_cast<uint32_t>(x1_shape->GetStorageShape().GetDim(1));
    } else if (dimNum == 1) {
        N = 1;
        C = static_cast<uint32_t>(x1_shape->GetStorageShape().GetDim(0));
    }

    int32_t reduction = ParseReductionAttr(context);
    if (reduction == REDUCTION_INVALID) {
        OP_LOGE(context->GetNodeName(), "The reduction attribute must be 'none', 'mean', or 'sum'.");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }

    // Empty batch (N == 0): keep N == 0 so the kernel processes zero rows; use a single
    // block for a valid grid. Do NOT fake N = 1, which would read an unallocated phantom row.
    uint32_t usedCoreNum = (N == 0) ? 1u : ((N < coreNum) ? N : coreNum);
    context->SetBlockDim(usedCoreNum);
    // The kernel uses SyncAll for the cross-core workspace reduction; batch mode makes all cores
    // start together, otherwise SyncAll can deadlock probabilistically.
    context->SetScheduleMode(BATCH_MODE);

    uint32_t basePerCore = N / usedCoreNum;
    uint32_t pivot = N % usedCoreNum;

    tiling.N = N;
    tiling.C = C;
    tiling.basePerCore = basePerCore;
    tiling.pivot = pivot;
    tiling.usedCoreNum = usedCoreNum;
    tiling.reduction = reduction;

    auto* tilingData = context->GetTilingData<MultilabelMarginLossTilingData>();
    *tilingData = tiling;

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // Float accumulation workspace: N slots for reduction=none (per-row loss), else 1 scalar.
    // Rounded to 32B. The kernel atomic-adds row losses here, then core 0 casts+writes y.
    uint32_t wsElems = (reduction == 0) ? ((N == 0u) ? 1u : N) : 1u;
    size_t accBytes = ((static_cast<size_t>(wsElems) * sizeof(float) + 31u) / 32u) * 32u;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = accBytes + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMultilabelMarginLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MultilabelMarginLoss)
    .Tiling(MultilabelMarginLossTilingFunc)
    .TilingParse<MultilabelMarginLossCompileInfo>(TilingParseForMultilabelMarginLoss);
} // namespace optiling
