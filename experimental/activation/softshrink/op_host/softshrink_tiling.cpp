/** 
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. 
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License"). 
 * Please refer to the License for details. You may not use this file except in compliance with the License. 
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License. 
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/softshrink_tiling_data.h"
#include "../op_kernel/softshrink_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context,
    uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"),
        return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context,
    int64_t& totalIdx, ge::DataType& dataType, float& lambd)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto shapeX = EnsureNotScalar(inputX->GetStorageShape());

    totalIdx = shapeX.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "Softshrink: unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    const float* lambdPtr = context->GetAttrs()->GetAttrPointer<float>(0);
    lambd = (lambdPtr != nullptr) ? *lambdPtr : 0.5f;
    OP_CHECK_IF(lambd < 0.0f,
        OP_LOGE(context, "Softshrink: lambd must be >= 0, got %f", lambd),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SoftshrinkTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),return ge::GRAPH_FAILED);

    int64_t totalIdx;
    ge::DataType dataType;
    float lambd;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalIdx, dataType, lambd) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),return ge::GRAPH_FAILED);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    SoftshrinkTilingData* tiling =context->GetTilingData<SoftshrinkTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SoftshrinkTilingData), 0,
        sizeof(SoftshrinkTilingData)) != EOK,OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (totalIdx == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        tiling->lambd = lambd;
        context->SetBlockDim(1);
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, 0ULL);
        return ge::GRAPH_SUCCESS;
    }

    int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
    tiling->totalNum = totalIdx;tiling->lambd = lambd;

    constexpr int64_t MIN_PER_CORE_ELEMS = 1024;
    int64_t maxCores = std::max(1L, totalIdx / MIN_PER_CORE_ELEMS);
    int64_t effectiveCoreNum = std::min(coreNum, maxCores);

    tiling->blockFactor = CeilDiv(totalIdx, effectiveCoreNum);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    uint64_t useDoubleBuffer = (tiling->blockFactor > MIN_SPLIT_THRESHOLD) ? 1 : 0;

    int64_t bufferNum;
    if (dataType == ge::DT_FLOAT) {
        bufferNum = useDoubleBuffer ? 6 : 4;
    } else {
        bufferNum = useDoubleBuffer ? 9 : 7;
    }

    constexpr int64_t VECTOR_ALIGN_ELEM = 256 / static_cast<int64_t>(sizeof(float));
    int64_t ubBlockSize = GetUbBlockSize(context);
    int64_t alignUnit = std::max(ubBlockSize, VECTOR_ALIGN_ELEM);
    tiling->ubFactor = FloorAlign(FloorDiv((static_cast<int64_t>(ubSize) / typeSize), bufferNum),alignUnit);

    context->SetBlockDim(usedCoreNum);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSoftshrink(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SoftshrinkCompileInfo {};

IMPL_OP_OPTILING(Softshrink)
    .Tiling(SoftshrinkTilingFunc)
    .TilingParse<SoftshrinkCompileInfo>(TilingParseForSoftshrink);

} // namespace optiling
