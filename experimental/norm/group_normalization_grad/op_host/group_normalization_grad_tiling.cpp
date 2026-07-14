/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_normalization_grad_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/group_normalization_grad_tiling_data.h"
#include "../op_kernel/group_normalization_grad_tiling_key.h"

namespace optiling {

using namespace ge;

constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t BUFFER_NUM = 2U;

// Buffer configuration matching kernel:
// 4 queues (x, dy, gamma, dx) x BUFFER_NUM each
// 9 temp float TBufs (xFloat, dyFloat, gammaFloat, xhat, tmp0-3, reduce)
// 1 scalar TBuf (8 x float)
constexpr uint32_t NUM_QUEUES = 4U;
constexpr uint32_t NUM_TEMP_FLOAT_BUFS = 9U;
constexpr uint32_t SCALAR_BYTES = 32U; // 8 * sizeof(float)

struct GroupNormalizationGradCompileInfo {};

static ge::graphStatus TilingParseForGroupNormalizationGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetPlatformInfo() == nullptr, OP_LOGE(context, "GetPlatformInfo is nullptr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GroupNormalizationGradTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Get shape info
    OP_CHECK_IF(context->GetInputShape(0) == nullptr, OP_LOGE(context, "GetInputShape is nullptr"),
                return ge::GRAPH_FAILED);
    const gert::Shape& inputShape = context->GetInputShape(0)->GetStorageShape();
    const auto dimNum = inputShape.GetDimNum();
    if (dimNum < 3 || dimNum > 8) {
        OP_LOGE(context, "dimNum must be in [3, 8]");
        return ge::GRAPH_FAILED;
    }

    uint64_t groupCount = static_cast<uint64_t>(inputShape.GetDim(0)) * static_cast<uint64_t>(inputShape.GetDim(1));
    uint64_t groupElemNum = 1;
    uint32_t dimIndex = 2;
    while (dimIndex < dimNum) {
        groupElemNum *= static_cast<uint64_t>(inputShape.GetDim(dimIndex));
        ++dimIndex;
    }

    uint32_t inputBytes = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), inputBytes);
    if (groupElemNum == 0 || inputBytes == 0) {
        OP_LOGE(context, "groupElemNum or inputBytes is 0");
        return ge::GRAPH_FAILED;
    }

    // 3. Get workspace size
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 4. Compute tiling parameters based on UB capacity
    // UB_total = NUM_QUEUES * BUFFER_NUM * alignedTDN * inputBytes
    //          + NUM_TEMP_FLOAT_BUFS * alignedTDN * sizeof(float)
    //          + SCALAR_BYTES
    uint32_t bytesPerAlignedElem = NUM_QUEUES * BUFFER_NUM * inputBytes + NUM_TEMP_FLOAT_BUFS * sizeof(float);
    uint32_t usableUb = (ubSize > SCALAR_BYTES) ? static_cast<uint32_t>(ubSize - SCALAR_BYTES) : 1;

    // alignedTDN must be 32-byte aligned: alignedTDN * inputBytes must be multiple of 32
    uint32_t alignElem = (BLOCK_SIZE + inputBytes - 1) / inputBytes;
    uint32_t maxAlignedTDN = usableUb / bytesPerAlignedElem;
    maxAlignedTDN = (maxAlignedTDN / alignElem) * alignElem;
    maxAlignedTDN = maxAlignedTDN == 0 ? alignElem : maxAlignedTDN;

    uint64_t tileDataNum = (maxAlignedTDN > groupElemNum) ? groupElemNum : maxAlignedTDN;
    uint64_t alignedTileDataNum = ((tileDataNum * inputBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / inputBytes;
    alignedTileDataNum = alignedTileDataNum == 0 ? alignElem : alignedTileDataNum;

    // 5. Compute core distribution (group-based)
    coreNum = (coreNum < static_cast<int64_t>(groupCount)) ? coreNum : static_cast<int64_t>(groupCount);
    coreNum = (coreNum >= 1) ? coreNum : 1;
    uint64_t smallCoreGroupNum = groupCount / static_cast<uint64_t>(coreNum);
    uint64_t tailBlockNum = groupCount % static_cast<uint64_t>(coreNum);
    uint64_t bigCoreGroupNum = smallCoreGroupNum + 1;
    uint64_t segmentNum = groupElemNum / tileDataNum;
    uint64_t finalTileNum = ((groupElemNum % tileDataNum) == 0) ? segmentNum : (segmentNum + 1);
    uint64_t tailDataNum = groupElemNum - segmentNum * tileDataNum;
    tailDataNum = tailDataNum == 0 ? tileDataNum : tailDataNum;

    // 6. Set tiling data
    GroupNormalizationGradTilingData* tiling = context->GetTilingData<GroupNormalizationGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(GroupNormalizationGradTilingData), 0, sizeof(GroupNormalizationGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->groupElemNum = groupElemNum;
    tiling->groupCount = groupCount;
    tiling->smallCoreGroupNum = smallCoreGroupNum;
    tiling->bigCoreGroupNum = bigCoreGroupNum;
    tiling->finalGroupTileNum = finalTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->alignedTileDataNum = alignedTileDataNum;
    tiling->tailDataNum = tailDataNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->groupElemNumFloat = static_cast<float>(groupElemNum);
    tiling->groupElemNumReciprocal = 1.0f / static_cast<float>(groupElemNum);

    context->SetBlockDim(coreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(GROUP_NORMALIZATION_GRAD_SCH_MODE_0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupNormalizationGrad)
    .Tiling(GroupNormalizationGradTilingFunc)
    .TilingParse<GroupNormalizationGradCompileInfo>(TilingParseForGroupNormalizationGrad);
} // namespace optiling
