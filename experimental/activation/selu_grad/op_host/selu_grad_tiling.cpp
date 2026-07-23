/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file selu_grad_tiling.cpp
 * \brief SeluGrad 算子 Tiling 实现
 */

#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/selu_grad_tiling_data.h"
#include "../op_kernel/selu_grad_tiling_key.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint64_t RESERVED_UB_BYTES = 8U * 1024U;
// 小工作集展开到最多 8 个核；更大工作集恢复 32KB/核，避免多核固定开销反超。
constexpr uint64_t SMALL_BYTES_PER_CORE = 8U * 1024U;
constexpr uint64_t LARGE_BYTES_PER_CORE = 32U * 1024U;
constexpr int64_t SMALL_CORE_LIMIT = 8;
constexpr int64_t GM_ALIGN_BYTES = 512;

struct SeluGradCompileInfo {};

static ge::graphStatus TilingParseForSeluGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static int64_t GetRawTypeSize(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 1;
        default:
            return 0;
    }
}

static int64_t GetComputeTypeSize(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_BF16:
        case ge::DT_INT32:
            return 4;
        default:
            return 0;
    }
}

static int64_t GetUbBytesPerElement(ge::DataType dataType)
{
    const int64_t rawSize = GetRawTypeSize(dataType);
    const int64_t computeSize = GetComputeTypeSize(dataType);
    if (rawSize == 0 || computeSize == 0) {
        return 0;
    }

    // 3 条双缓冲 GM 队列 + 比较掩码 + 系数临时区。
    const bool directCompute = dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT;
    const int64_t computeBufferNum = directCompute ? 1 : 4;
    return 6 * rawSize + computeBufferNum * computeSize + 1;
}

static uint32_t GetScheduleMode(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return SELUGRAD_TPL_SCH_MODE_FP16;
        case ge::DT_FLOAT:
            return SELUGRAD_TPL_SCH_MODE_FP32;
        case ge::DT_BF16:
            return SELUGRAD_TPL_SCH_MODE_BF16;
        case ge::DT_INT32:
            return SELUGRAD_TPL_SCH_MODE_INT32;
        case ge::DT_INT8:
            return SELUGRAD_TPL_SCH_MODE_INT8;
        case ge::DT_UINT8:
            return SELUGRAD_TPL_SCH_MODE_UINT8;
        default:
            return UINT32_MAX;
    }
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SeluGradTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetInputDesc(0) == nullptr || context->GetInputDesc(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    SeluGradTilingData* tiling = context->GetTilingData<SeluGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const ge::DataType dataType = context->GetInputDesc(0)->GetDataType();
    if (dataType != context->GetInputDesc(1)->GetDataType()) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rawTypeSize = GetRawTypeSize(dataType);
    const int64_t computeTypeSize = GetComputeTypeSize(dataType);
    const int64_t ubBytesPerElement = GetUbBytesPerElement(dataType);
    const uint32_t scheduleMode = GetScheduleMode(dataType);
    if (rawTypeSize == 0 || computeTypeSize == 0 || ubBytesPerElement == 0 || scheduleMode == UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& gradientsShape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape& outputsShape = context->GetInputShape(1)->GetStorageShape();
    if (gradientsShape.GetDimNum() != outputsShape.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < gradientsShape.GetDimNum(); ++i) {
        if (gradientsShape.GetDim(i) != outputsShape.GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t totalNum = gradientsShape.GetShapeSize();
    if (totalNum < 0) {
        return ge::GRAPH_FAILED;
    }
    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(scheduleMode));
        return ge::GRAPH_SUCCESS;
    }

    const int64_t alignNum = std::max<int64_t>(1, GM_ALIGN_BYTES / rawTypeSize);
    const uint64_t totalBytes = static_cast<uint64_t>(totalNum) * static_cast<uint64_t>(rawTypeSize);
    int64_t usedCoreNum = std::max<int64_t>(1, CeilDiv(totalBytes, SMALL_BYTES_PER_CORE));
    if (usedCoreNum > SMALL_CORE_LIMIT) {
        usedCoreNum = std::max<int64_t>(1, CeilDiv(totalBytes, LARGE_BYTES_PER_CORE));
    }
    usedCoreNum = std::min<int64_t>(usedCoreNum, coreNum);

    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);
    usedCoreNum = std::max<int64_t>(1, CeilDiv(totalNum, blockFactor));

    const uint64_t availableUb = ubSize > RESERVED_UB_BYTES ? ubSize - RESERVED_UB_BYTES : ubSize;
    int64_t maxUbFactor = FloorAlign(static_cast<int64_t>(availableUb / static_cast<uint64_t>(ubBytesPerElement)),
                                     alignNum);
    if (maxUbFactor < alignNum) {
        maxUbFactor = alignNum;
    }
    const int64_t ubFactor = std::max<int64_t>(alignNum, std::min<int64_t>(blockFactor, maxUbFactor));

    if (blockFactor > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        ubFactor > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalNum = static_cast<uint64_t>(totalNum);
    tiling->blockFactor = static_cast<uint32_t>(blockFactor);
    tiling->ubFactor = static_cast<uint32_t>(ubFactor);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(GET_TPL_TILING_KEY(scheduleMode));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SeluGrad).Tiling(SeluGradTilingFunc).TilingParse<SeluGradCompileInfo>(TilingParseForSeluGrad);

} // namespace optiling
