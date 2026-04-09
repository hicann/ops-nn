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
 * \file hard_sigmoid_grad_tiling.cpp
 * \brief HardSigmoidGrad Tiling implementation (arch35)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/hard_sigmoid_grad_tiling_data.h"
#include "../../op_kernel/arch35/hard_sigmoid_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Number of UB buffers needed:
// 2 inputs (gradOutput, self) + 3 temp buffers (tmp1, tmp2, result) = 5
// With double buffering: 5 * 2 = 10 (for inputs and output queues)
// Single buffer: 2 input + 1 output (queued) + 2 temp (from pipe) = 5
// Double buffer: 2*2 input + 1*2 output (queued) + 2 temp (from pipe) = 8
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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalLength,
                                          ge::DataType& dataType, float& alpha, float& beta)
{
    auto inputGrad = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGrad);
    auto gradShape = EnsureNotScalar(inputGrad->GetStorageShape());

    auto inputSelf = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto selfShape = EnsureNotScalar(inputSelf->GetStorageShape());

    OP_CHECK_IF(
        gradShape.GetShapeSize() != selfShape.GetShapeSize(),
        OP_LOGE(context, "HardSigmoidGrad: input shape mismatch: grad=%ld, self=%ld",
                gradShape.GetShapeSize(), selfShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalLength = gradShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    // Get alpha and beta from attrs (with defaults)
    const auto* alphaAttr = context->GetAttrs()->GetFloat(0);
    const auto* betaAttr = context->GetAttrs()->GetFloat(1);
    alpha = (alphaAttr != nullptr) ? static_cast<float>(*alphaAttr) : 0.16666666f;
    beta = (betaAttr != nullptr) ? static_cast<float>(*betaAttr) : 0.5f;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HardSigmoidGradTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get shape and attr info
    int64_t totalLength;
    ge::DataType dataType;
    float alpha, beta;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalLength, dataType, alpha, beta) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // 3. Get workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. Handle empty tensor (0 elements)
    if (totalLength == 0) {
        HardSigmoidGradTilingData* tiling = context->GetTilingData<HardSigmoidGradTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        OP_CHECK_IF(
            memset_s(tiling, sizeof(HardSigmoidGradTilingData), 0, sizeof(HardSigmoidGradTilingData)) != EOK,
            OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
        context->SetBlockDim(1);
        uint32_t schMode = HARD_SIGMOID_GRAD_MODE_FLOAT32;
        ASCENDC_TPL_SEL_PARAM(context, schMode);
        return ge::GRAPH_SUCCESS;
    }

    // 5. Compute tiling parameters
    HardSigmoidGradTilingData* tiling = context->GetTilingData<HardSigmoidGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(HardSigmoidGradTilingData), 0, sizeof(HardSigmoidGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Multi-core split
    tiling->blockFactor = CeilDiv(totalLength, coreNum);
    int64_t usedCoreNum = CeilDiv(totalLength, tiling->blockFactor);

    int64_t ubBlockSize = GetUbBlockSize(context);

    // UB split depends on dtype:
    // fp32/fp16: 8 buffer slots (2*2 in queues + 1*2 out queue + 1 tmp + 1 mask ~= 8)
    // bf16: 7 fp32-sized buffer slots
    //   - 3 queued bf16 buffers (2 in + 1 out, double-buffered) count as ~3 fp32 slots
    //     (bf16 is half the size, but double-buffered = 6 bf16 buffers ~= 3 fp32 buffers)
    //   - 3 fp32 temp buffers (gradFp32, selfFp32, tmp) = 3 fp32 slots
    //   - 1 mask buffer (~negligible, approximate as 1 fp32 slot)
    //   Total ~= 7 fp32-sized buffer slots
    int64_t bufferNum;
    int64_t typeSize;

    if (dataType == ge::DT_BF16) {
        // bf16 mode: compute in fp32, buffer size calculated by fp32
        bufferNum = 7;
        typeSize = 4; // fp32 element size for UB calculation
    } else if (dataType == ge::DT_FLOAT16) {
        bufferNum = 8;
        typeSize = 2;
    } else {
        // fp32
        bufferNum = 8;
        typeSize = 4;
    }

    tiling->ubFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize) / typeSize, bufferNum),
        ubBlockSize);

    tiling->totalLength = totalLength;
    tiling->alpha = alpha;
    tiling->beta = beta;

    context->SetBlockDim(usedCoreNum);

    // 6. Set TilingKey
    uint32_t schMode;
    if (dataType == ge::DT_BF16) {
        schMode = HARD_SIGMOID_GRAD_MODE_BFLOAT16;
    } else if (dataType == ge::DT_FLOAT16) {
        schMode = HARD_SIGMOID_GRAD_MODE_FLOAT16;
    } else {
        schMode = HARD_SIGMOID_GRAD_MODE_FLOAT32;
    }
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHardSigmoidGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct HardSigmoidGradCompileInfo {};

IMPL_OP_OPTILING(HardSigmoidGrad)
    .Tiling(HardSigmoidGradTilingFunc)
    .TilingParse<HardSigmoidGradCompileInfo>(TilingParseForHardSigmoidGrad);

} // namespace optiling
