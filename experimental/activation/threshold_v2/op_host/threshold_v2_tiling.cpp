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
 * \file threshold_v2_tiling.cpp
 * \brief ThresholdV2 Host Tiling
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/threshold_v2_tiling_data.h"
#include "../op_kernel/threshold_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static uint64_t GetDtypeSelfFromDtype(ge::DataType dt) {
    return static_cast<uint64_t>(dt);
}

static int64_t GetTypeSizeFromDtype(ge::DataType dt) {
    switch (static_cast<int>(dt)) {
        case 0:  return 4;  // DT_FLOAT
        case 1:  return 2;  // DT_FLOAT16
        case 27: return 2;  // DT_BFLOAT16
        default: return 4;
    }
}

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
    OP_CHECK_IF(static_cast<int64_t>(ubSize) <= 0, OP_LOGE(context, "ubSize is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ThresholdV2TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto shape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalIdx = shape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();
    int dtypeInt = static_cast<int>(inputDtype);
    bool isSupportedDtype = (dtypeInt == 0 || dtypeInt == 1 || dtypeInt == 27);
    OP_CHECK_IF(!isSupportedDtype,
        OP_LOGE(context, "ThresholdV2: unsupported dtype"), return ge::GRAPH_FAILED);
    int64_t typeSize = GetTypeSizeFromDtype(inputDtype);
    OP_CHECK_IF(typeSize <= 0,
        OP_LOGE(context, "ThresholdV2: invalid typeSize"), return ge::GRAPH_FAILED);
    uint64_t dtypeSelf = GetDtypeSelfFromDtype(inputDtype);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    ThresholdV2TilingData* tiling = context->GetTilingData<ThresholdV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ThresholdV2TilingData), 0, sizeof(ThresholdV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    int64_t ubBlockSize = GetUbBlockSize(context);

    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = useDoubleBuffer ? 4 : 2;
    int64_t maskReserve = 4096;
    int64_t availUb = static_cast<int64_t>(ubSize) - maskReserve;
    OP_CHECK_IF(availUb <= 0,
        OP_LOGE(context, "ThresholdV2: insufficient UB memory"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(typeSize == 0,
        OP_LOGE(context, "ThresholdV2: typeSize is 0, division by zero"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(bufferNum == 0,
        OP_LOGE(context, "ThresholdV2: bufferNum is 0, division by zero"), return ge::GRAPH_FAILED);
    int64_t vectorAlign = 256 / typeSize;
    int64_t ubFactorRaw = FloorDiv((availUb / typeSize), bufferNum);
    tiling->ubFactor = (ubFactorRaw / vectorAlign) * vectorAlign;

    const auto attrs = context->GetAttrs();
    tiling->thresholdVal = *attrs->GetAttrPointer<float>(0);
    tiling->valueVal = *attrs->GetAttrPointer<float>(1);

    context->SetBlockDim(usedCoreNum);

    ASCENDC_TPL_SEL_PARAM(context, useDoubleBuffer, dtypeSelf);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForThresholdV2(
    [[maybe_unused]] gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }
struct ThresholdV2CompileInfo {};

IMPL_OP_OPTILING(ThresholdV2)
    .Tiling(ThresholdV2TilingFunc)
    .TilingParse<ThresholdV2CompileInfo>(TilingParseForThresholdV2);

} // namespace optiling
