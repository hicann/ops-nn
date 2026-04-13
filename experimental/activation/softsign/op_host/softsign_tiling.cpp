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

/*!
 * \file softsign_tiling.cpp
 * \brief Softsign tiling implementation (arch35)
 *
 * Softsign(x) = x / (1 + |x|)
 *
 * UB buffer layout:
 *   float: inputQueue(BUFFER_NUM) + tmpBuf(1) + outputQueue(BUFFER_NUM)
 *     double buffer: 2 + 1 + 2 = 5 buffers * typeSize(4) = 20 bytes/element
 *     single buffer: 1 + 1 + 1 = 3 buffers * typeSize(4) = 12 bytes/element
 *
 *   half / bf16 (Cast path, shared layout):
 *     inputQueue(BUFFER_NUM) + outputQueue(BUFFER_NUM) [2-byte TIn]
 *     + castInBuf(1) + tmpBuf(1) + calcOutBuf(1) [4-byte float]
 *     double buffer: 2*2*2 + 4*3 = 8 + 12 = 20 bytes/element
 *     single buffer: 2*2*1 + 4*3 = 4 + 12 = 16 bytes/element
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/softsign_tiling_data.h"
#include "../op_kernel/softsign_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

// Double buffer threshold: enable double buffer when totalNum > this value
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
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

static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context, ge::DataType dataType)
{
    auto* tiling = context->GetTilingData<SoftsignTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SoftsignTilingData), 0, sizeof(SoftsignTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    context->SetBlockDim(1);
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    uint64_t useDoubleBuffer = 0;
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static int64_t CalcBytesPerElement(ge::DataType dataType, uint64_t useDoubleBuffer)
{
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        // fp16/bf16 Cast path: 2 TIn buffers (in/out) + 3 float buffers (castIn/tmp/calcOut)
        int64_t bufferNum = useDoubleBuffer ? 2 : 1;
        return static_cast<int64_t>(sizeof(uint16_t)) * 2 * bufferNum +
               static_cast<int64_t>(sizeof(float)) * 3;
    }
    // fp32: direct compute path (in + tmp + out) with in/out having BUFFER_NUM copies
    int64_t bufferCount = useDoubleBuffer ? 5 : 3;
    return static_cast<int64_t>(sizeof(float)) * bufferCount;
}

static ge::graphStatus SoftsignTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto storageShape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = storageShape.GetShapeSize();

    if (totalNum == 0) {
        return HandleEmptyTensor(context, dataType);
    }

    auto* tiling = context->GetTilingData<SoftsignTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SoftsignTilingData), 0, sizeof(SoftsignTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t ubBlockSize = GetUbBlockSize(context);
    int64_t bytesPerElement = CalcBytesPerElement(dataType, useDoubleBuffer);
    tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize), bytesPerElement), ubBlockSize);
    context->SetBlockDim(usedCoreNum);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSoftsign([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SoftsignCompileInfo {};

IMPL_OP_OPTILING(Softsign).Tiling(SoftsignTilingFunc).TilingParse<SoftsignCompileInfo>(TilingParseForSoftsign);

} // namespace optiling
