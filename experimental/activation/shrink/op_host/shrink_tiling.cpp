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
 * @file shrink_tiling.cpp
 * @brief Shrink Tiling implementation (arch35)
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/shrink_tiling_data.h"
#include "../op_kernel/shrink_tiling_key.h"

#include <cmath>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Double buffer switching threshold.
// 1024 elements is the minimum threshold; fp16 1024 elems = 2KB, fp32 1024 elems = 4KB.
// The actual optimal threshold should be determined by performance profiling;
// a higher threshold (e.g., 4096~8192) may yield better throughput.
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

// 0-dim scalar tensors are treated as 1-dim with 1 element for Tiling.
// The inferShape preserves the original 0-dim shape, so output shape matches input.
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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context,
                                         int64_t& totalIdx,
                                         ge::DataType& dataType,
                                         float& lambd,
                                         float& bias)
{
    auto input = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    auto inputShape = EnsureNotScalar(input->GetStorageShape());

    auto output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    auto outShape = EnsureNotScalar(output->GetStorageShape());

    OP_CHECK_IF(
        inputShape.GetShapeSize() != outShape.GetShapeSize(),
        OP_LOGE(context, "Shrink: shape size mismatch: input=%ld, out=%ld",
                inputShape.GetShapeSize(), outShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalIdx = inputShape.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "Shrink: invalid dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* lambdPtr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, lambdPtr);
    lambd = *lambdPtr;

    const float* biasPtr = attrs->GetAttrPointer<float>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasPtr);
    bias = *biasPtr;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static int64_t CalcUbFactor(int64_t totalIdx, ge::DataType dataType, uint64_t ubSize,
                            int64_t ubBlockSize, uint64_t useDoubleBuffer)
{
    int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
    int64_t computeAlignment = 256 / typeSize;

    // Reserve for TPipe, queue management, Select mode 2 internal buffer, system overhead.
    // Ascend950 may need more headroom than Ascend910B.
    constexpr int64_t UB_RESERVED_BYTES = 48 * 1024;
    int64_t availableUb = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
    if (availableUb <= 0) {
        availableUb = static_cast<int64_t>(ubSize) / 2;
    }

    // Buffer layout: 2 TQue (input+output) × BUFFER_NUM + 1 TBuf (mid) × 1
    int64_t numTQueBuffers = useDoubleBuffer ? 2 : 1;
    int64_t bufferBytes = (2 * numTQueBuffers + 1) * typeSize;

    int64_t rawUbFactor = FloorDiv(availableUb, bufferBytes);
    int64_t alignFactor = (ubBlockSize > computeAlignment) ? ubBlockSize : computeAlignment;
    int64_t ubFactorAligned = FloorAlign(rawUbFactor, alignFactor);

    if (totalIdx < ubFactorAligned) {
        ubFactorAligned = CeilDiv(totalIdx, alignFactor) * alignFactor;
    }

    constexpr int64_t MAX_COPY_BYTES = 65535;
    int64_t maxElementsByType = MAX_COPY_BYTES / typeSize;
    if (ubFactorAligned > maxElementsByType) {
        ubFactorAligned = FloorAlign(maxElementsByType, alignFactor);
    }
    return ubFactorAligned;
}

static ge::graphStatus ShrinkTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t totalIdx;
    ge::DataType dataType;
    float lambd;
    float bias;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType, lambd, bias) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ShrinkTilingData* tiling = context->GetTilingData<ShrinkTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ShrinkTilingData), 0, sizeof(ShrinkTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilDiv(totalIdx, coreNum);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);
    tiling->lambd = (lambd < 0.0f) ? 0.0f : lambd;
    tiling->bias = bias;

    int64_t ubBlockSize = GetUbBlockSize(context);
    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    tiling->ubFactor = CalcUbFactor(totalIdx, dataType, ubSize, ubBlockSize, useDoubleBuffer);

    context->SetBlockDim(usedCoreNum);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForShrink([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ShrinkCompileInfo {};

IMPL_OP_OPTILING(Shrink)
    .Tiling(ShrinkTilingFunc)
    .TilingParse<ShrinkCompileInfo>(TilingParseForShrink);

} // namespace optiling
