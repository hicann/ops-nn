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
 * \file fused_bias_leaky_relu_tiling.cpp
 * \brief FusedBiasLeakyRelu Tiling implementation (arch35)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/fused_bias_leaky_relu_tiling_data.h"
#include "../../op_kernel/arch35/fused_bias_leaky_relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::CeilAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Number of tensors in UB: inputX, inputBias, outputY
constexpr int64_t TENSOR_NUM = 3;

constexpr int64_t BLOCK_ALIGN = 32;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

// Get platform information: ubSize, coreNum
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

// Get shape and attribute information
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context,
                                          int64_t& totalNum,
                                          ge::DataType& dataType,
                                          float& negativeSlope,
                                          float& scale)
{
    // Get input shape
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    auto inputBias = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputBias);
    auto inputShapeBias = EnsureNotScalar(inputBias->GetStorageShape());

    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto outShapeY = EnsureNotScalar(outY->GetStorageShape());

    // Shape validation: ensure x and bias shapes match
    OP_CHECK_IF(
        inputShapeX.GetShapeSize() != inputShapeBias.GetShapeSize() ||
            inputShapeX.GetShapeSize() != outShapeY.GetShapeSize(),
        OP_LOGE(context, "FusedBiasLeakyRelu: shape size mismatch: x=%ld, bias=%ld, y=%ld",
                inputShapeX.GetShapeSize(), inputShapeBias.GetShapeSize(), outShapeY.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalNum = inputShapeX.GetShapeSize();

    // Dtype validation
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "FusedBiasLeakyRelu: unsupported dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }

    // Get attributes (ACLNN passes double; read as double, then convert to float)
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* pNegativeSlope = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, pNegativeSlope);
    negativeSlope = *pNegativeSlope;
    const float* pScale = attrs->GetFloat(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, pScale);
    scale = *pScale;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// Tiling dispatch entry
static ge::graphStatus FusedBiasLeakyReluTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get shape and attribute info
    int64_t totalNum;
    ge::DataType dataType;
    float negativeSlope;
    float scale;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, dataType, negativeSlope, scale) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. Compute tiling parameters
    FusedBiasLeakyReluTilingData* tiling = context->GetTilingData<FusedBiasLeakyReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(FusedBiasLeakyReluTilingData), 0, sizeof(FusedBiasLeakyReluTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Compute type-dependent parameters
    int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
    int64_t ubBlockSize = BLOCK_ALIGN / typeSize;  // float32: 8 elements, float16: 16 elements

    // Multi-core split: divide total elements evenly across cores
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    // Double buffer decision: enable when at least 2 UB loops per core
    int64_t loopCountIfDouble = CeilDiv(blockFactor, static_cast<int64_t>(ubSize) / typeSize / (TENSOR_NUM * 2));
    uint64_t useDoubleBuffer = (loopCountIfDouble >= 2) ? 1 : 0;

    // UB split
    int64_t bufferNum = TENSOR_NUM * (useDoubleBuffer ? 2 : 1);  // 6 or 3
    int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, bufferNum), ubBlockSize);

    // 5. Fill TilingData (order matches struct layout)
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->negativeSlope = negativeSlope;
    tiling->scale = scale;

    context->SetBlockDim(usedCoreNum);

    // 6. Set TilingKey
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFusedBiasLeakyRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct FusedBiasLeakyReluCompileInfo {};

IMPL_OP_OPTILING(FusedBiasLeakyRelu)
    .Tiling(FusedBiasLeakyReluTilingFunc)
    .TilingParse<FusedBiasLeakyReluCompileInfo>(TilingParseForFusedBiasLeakyRelu);

} // namespace optiling
