/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "group_norm/op_kernel/group_norm_tiling_data.h"
#include "group_norm/op_kernel/group_norm_tiling_key.h"

namespace optiling {
namespace {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GAMMA_INDEX = 1;
constexpr uint32_t BETA_INDEX = 2;
constexpr uint32_t ATTR_NUM_GROUPS = 0;
constexpr uint32_t ATTR_EPS = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MIN_TILE_BYTES = 8 * 1024;

struct GroupNormCompileInfo {};

static uint32_t AlignUp(uint32_t value, uint32_t align)
{
    if (align == 0) {
        return value;
    }
    return (value + align - 1) / align * align;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = platform.GetCoreNum();
    OP_CHECK_IF(ubSize == 0 || coreNum <= 0, OP_LOGE(context, "invalid platform info"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReadShapeAndAttrs(gert::TilingContext* context, GroupNormTilingData& tiling)
{
    auto xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto storageShape = xShape->GetStorageShape();
    OP_CHECK_IF(storageShape.GetDimNum() < 2, OP_LOGE(context, "GroupNorm input rank must be >= 2"),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numGroupsAttr = attrs->GetInt(ATTR_NUM_GROUPS);
    OP_CHECK_NULL_WITH_CONTEXT(context, numGroupsAttr);
    OP_CHECK_IF(*numGroupsAttr <= 0, OP_LOGE(context, "num_groups should be greater than 0"), return ge::GRAPH_FAILED);

    const float* epsAttr = attrs->GetFloat(ATTR_EPS);
    const uint32_t n = static_cast<uint32_t>(storageShape.GetDim(0));
    const uint32_t c = static_cast<uint32_t>(storageShape.GetDim(1));
    uint32_t hxw = 1;
    for (size_t i = 2; i < storageShape.GetDimNum(); ++i) {
        hxw *= static_cast<uint32_t>(storageShape.GetDim(i));
    }

    const uint32_t numGroups = static_cast<uint32_t>(*numGroupsAttr);
    OP_CHECK_IF(c == 0 || n == 0 || hxw == 0, OP_LOGE(context, "GroupNorm does not support zero dim here"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(c % numGroups != 0, OP_LOGE(context, "C should be divisible by num_groups"), return ge::GRAPH_FAILED);

    tiling.n = n;
    tiling.c = c;
    tiling.hxw = hxw;
    tiling.numGroups = numGroups;
    tiling.channelsPerGroup = c / numGroups;
    tiling.elementsPerGroup = tiling.channelsPerGroup * hxw;
    tiling.invElementsPerGroup = 1.0f / static_cast<float>(tiling.elementsPerGroup);
    tiling.groupNum = n * numGroups;
    tiling.hasGamma = context->GetOptionalInputTensor(GAMMA_INDEX) == nullptr ? 0U : 1U;
    tiling.hasBeta = context->GetOptionalInputTensor(BETA_INDEX) == nullptr ? 0U : 1U;
    tiling.eps = epsAttr == nullptr ? 0.00001f : *epsAttr;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillSchedule(gert::TilingContext* context, uint64_t ubSize, int64_t coreNum,
                                    GroupNormTilingData& tiling)
{
    const uint32_t maxCore = static_cast<uint32_t>(coreNum);
    const uint32_t blockNum = tiling.groupNum < maxCore ? tiling.groupNum : maxCore;
    OP_CHECK_IF(blockNum == 0, OP_LOGE(context, "blockNum is 0"), return ge::GRAPH_FAILED);

    tiling.groupPerCore = tiling.groupNum / blockNum;
    tiling.groupTailCore = tiling.groupNum % blockNum;

    uint32_t tileLength = static_cast<uint32_t>((ubSize / BUFFER_NUM) / sizeof(float) / 8);
    tileLength = tileLength < MIN_TILE_BYTES / sizeof(float) ? MIN_TILE_BYTES / sizeof(float) : tileLength;
    tileLength = AlignUp(tileLength, BLOCK_SIZE / sizeof(float));
    tiling.tileLength = tiling.elementsPerGroup == 1 ?
                            tileLength :
                            (tileLength > tiling.elementsPerGroup ? tiling.elementsPerGroup : tileLength);

    const uint32_t tilingKey = tiling.elementsPerGroup <= tiling.tileLength ? GROUP_NORM_SCH_SMALL_GROUP :
                                                                              GROUP_NORM_SCH_GENERAL;
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockNum);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus GroupNormTilingFunc(gert::TilingContext* context)
{
    GroupNormTilingData* tiling = context->GetTilingData<GroupNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(GroupNormTilingData), 0, sizeof(GroupNormTilingData)) != EOK,
                OP_LOGE(context, "set tiling data failed"), return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "GetPlatformInfo failed");
        return ge::GRAPH_FAILED;
    }
    if (ReadShapeAndAttrs(context, *tiling) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "ReadShapeAndAttrs failed");
        return ge::GRAPH_FAILED;
    }
    if (FillSchedule(context, ubSize, coreNum, *tiling) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "FillSchedule failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGroupNorm([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupNorm).Tiling(GroupNormTilingFunc).TilingParse<GroupNormCompileInfo>(TilingParseForGroupNorm);
} // namespace optiling
