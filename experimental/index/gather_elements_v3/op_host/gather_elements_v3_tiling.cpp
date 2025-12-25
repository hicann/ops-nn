/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
/*!
 * \file gather_elements_v3_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/gather_elements_v3_tiling_data.h"
#include "../op_kernel/gather_elements_v3_tiling_key.h"

namespace optiling {
constexpr uint32_t WS_SYS_SIZE = 0;
struct GatherElementsV3CompileInfo {};

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1); 
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus GatherElementsV3TilingFunc(gert::TilingContext* context)
{
    GatherElementsV3TilingData* tiling = context->GetTilingData<GatherElementsV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(GatherElementsV3TilingData), 0, sizeof(GatherElementsV3TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),return ge::GRAPH_FAILED);

    // ---------- 1. 平台信息 ----------
    uint64_t ubSize;
    uint32_t coreNum;
    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = plat.GetCoreNum();

    // ---------- 2. 输入 shape ----------
    auto xShape = context->GetInputShape(0)->GetStorageShape();
    auto idxShape = context->GetInputShape(1)->GetStorageShape();

    // ---------- 3. 获取 dim ----------
    uint64_t dim = 0;
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    dim = *(attrs->GetAttrPointer<uint64_t>(0));
    OP_CHECK_IF(dim >= xShape.GetDimNum(), OP_LOGE(context, "dim out of range"), return ge::GRAPH_FAILED);

    // ---------- 4. 获取 dtype 长度并计算对齐 ----------
    uint32_t typeLength = 0;
    auto dt = context->GetInputDesc(0)->GetDataType();
    ge::TypeUtils::GetDataTypeLength(dt, typeLength);

    uint32_t alignX = 32 / typeLength;
    constexpr uint32_t alignIdx = 8;
    uint32_t alignNum = (alignX > alignIdx) ? alignX : alignIdx;
    if (alignNum == 0) alignNum = 1;

    // ---------- 5. 维度 flatten ----------
    uint32_t xPre = 1, xPost = 1, xGather = xShape.GetDim(dim);
    uint32_t idxPre = 1, idxPost = 1, idxGather = idxShape.GetDim(dim);

    for (uint32_t i = 0; i < dim; ++i) {
        xPre *= xShape.GetDim(i);
        idxPre *= idxShape.GetDim(i);
    }
    for (uint32_t i = dim + 1; i < xShape.GetDimNum(); ++i) {
        xPost *= xShape.GetDim(i);
        idxPost *= idxShape.GetDim(i);
    }

    // ---------- 6. usedCores ----------
    uint64_t totalRows = (uint64_t)idxPre * (uint64_t)idxGather;
    uint32_t usedCore = (uint32_t)coreNum;
    if (totalRows < (uint64_t)coreNum && totalRows > 0) {
        usedCore = (uint32_t)totalRows;
    }
    if (usedCore == 0) {
        OP_LOGE(context, "coreNum is 0");
        return ge::GRAPH_FAILED; 
    }

    // ---------- 7. tileSize ----------
    constexpr uint64_t RESERVED_UB = 1024;
    uint64_t availUbPerBuffer = (ubSize > RESERVED_UB ? ubSize - RESERVED_UB : ubSize / 2) / 2;

    uint32_t bytesPerElement = sizeof(int32_t) + typeLength;
    uint64_t maxTileElements = (bytesPerElement == 0) ? 0 : (availUbPerBuffer / bytesPerElement);

    if (alignNum > 1) {
        maxTileElements &= ~(uint64_t)(alignNum - 1);
    }
    if (maxTileElements == 0) maxTileElements = alignNum;

    uint32_t tileSize = (uint32_t)maxTileElements;
    if (tileSize > idxPost) tileSize = idxPost;

    // ---------- 8. 写入 tiling ----------
    tiling->xPreDim     = xPre;
    tiling->xGatherDim  = xGather;
    tiling->xPostDim    = xPost;
    tiling->idxPreDim   = idxPre;
    tiling->idxGatherDim= idxGather;
    tiling->idxPostDim  = idxPost;
    tiling->tileSize    = tileSize;
    tiling->usedCores   = usedCore;

    // ---------- 9. workspace / blockDim ----------
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(tiling->usedCores);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGatherElementsV3([[maybe_unused]] gert::TilingParseContext* context)
{   
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GatherElementsV3).Tiling(GatherElementsV3TilingFunc).TilingParse<GatherElementsV3CompileInfo>(TilingParseForGatherElementsV3);
} 
