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

/**
 * \file hard_swish_tiling.cpp
 * \brief HardSwish Tiling 实现 (arch35 - Ascend950)
 *
 * 负责：
 *   1. 获取平台信息（UB 大小、AI Core 数量）
 *   2. 获取输入 shape 和 dtype
 *   3. 计算多核切分参数（blockFactor）和 UB 切分参数（ubFactor）
 *   4. 根据 dtype 选择 TilingKey 分支
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/hard_swish_tiling_data.h"
#include "../../op_kernel/arch35/hard_swish_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

// BUFFER_NUM for fp32: inputX(x2 double buffer) + tmpBuf(x2) + outputY(x2) = 6
constexpr int64_t BUFFER_NUM_FP32 = 6;
// BUFFER_NUM for fp16: inputFp16(x2) + inputFp32(x2) + tmpFp32(x2) + outputFp16(x2) = 8
// (fp16 now promotes to fp32 for intermediate computation, same as bf16)
constexpr int64_t BUFFER_NUM_FP16 = 8;
// BUFFER_NUM for bf16: inputBf16(x2) + inputFp32(x2) + tmpFp32(x2) + outputBf16(x2) = 8
constexpr int64_t BUFFER_NUM_BF16 = 8;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    totalNum = inputShapeX.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "HardSwish: unsupported dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HardSwishTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get shape and dtype
    int64_t totalNum;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. Fill TilingData
    HardSwishTilingData* tiling = context->GetTilingData<HardSwishTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(HardSwishTilingData), 0, sizeof(HardSwishTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    // Multi-core split
    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // UB split based on dtype
    int64_t ubCanUse = static_cast<int64_t>(ubSize);
    int64_t ubBlockSize = GetUbBlockSize(context);
    uint64_t tilingKey = 0;

    if (dataType == ge::DT_FLOAT) {
        int64_t typeSize = static_cast<int64_t>(sizeof(float));  // 4
        tiling->ubFactor = FloorAlign(FloorDiv(ubCanUse / typeSize, BUFFER_NUM_FP32), ubBlockSize);
        tilingKey = GET_TPL_TILING_KEY(HARDSWISH_TPL_SCH_MODE_FP32);
    } else if (dataType == ge::DT_FLOAT16) {
        int64_t typeSize = static_cast<int64_t>(sizeof(float));  // compute in fp32 = 4
        tiling->ubFactor = FloorAlign(FloorDiv(ubCanUse / typeSize, BUFFER_NUM_FP16), ubBlockSize);
        tilingKey = GET_TPL_TILING_KEY(HARDSWISH_TPL_SCH_MODE_FP16);
    } else if (dataType == ge::DT_BF16) {
        int64_t typeSize = static_cast<int64_t>(sizeof(float));  // compute in fp32 = 4
        tiling->ubFactor = FloorAlign(FloorDiv(ubCanUse / typeSize, BUFFER_NUM_BF16), ubBlockSize);
        tilingKey = GET_TPL_TILING_KEY(HARDSWISH_TPL_SCH_MODE_BF16);
    } else {
        OP_LOGE(context, "HardSwish: unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    if (tiling->ubFactor <= 0) {
        OP_LOGE(context, "HardSwish: ubFactor is %ld after FloorAlign, ubSize=%lu dtype=%d",
                tiling->ubFactor, ubSize, static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHardSwish([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct HardSwishCompileInfo {};

IMPL_OP_OPTILING(HardSwish)
    .Tiling(HardSwishTilingFunc)
    .TilingParse<HardSwishCompileInfo>(TilingParseForHardSwish);

} // namespace optiling
