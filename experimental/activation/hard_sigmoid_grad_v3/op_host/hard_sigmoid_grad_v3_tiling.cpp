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
 * \file hard_sigmoid_grad_v3_tiling.cpp
 * \brief HardSigmoidGradV3 tiling implementation (arch32, Ascend910B)
 *
 * Tiling strategy:
 *   - Multi-core: split total elements evenly across AI Cores
 *   - UB split: 5 tensors (2 input + 2 mask + 1 output) x buffer count
 *   - Double buffer when totalNum > 1024
 *
 * TilingKey parameters (via ASCENDC_TPL_SEL_PARAM):
 *   - dTypeX: data type enum value
 *   - useDoubleBuffer: 0 or 1
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/hard_sigmoid_grad_v3_tiling_data.h"
#include "../op_kernel/hard_sigmoid_grad_v3_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;
constexpr int64_t FP32_BUFFER_COEFFICIENT = 25;
constexpr int64_t FP16_VECTOR_BUFFER_COEFFICIENT = 13;
constexpr int64_t COMPARE_ALIGN_BYTES = 256;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static inline int64_t AlignDown(int64_t value, int64_t align)
{
    if (align <= 0) {
        return value;
    }
    return FloorDiv(value, align) * align;
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
    auto inputGradOutput = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGradOutput);
    auto gradOutputShape = EnsureNotScalar(inputGradOutput->GetStorageShape());

    auto inputSelf = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto selfShape = EnsureNotScalar(inputSelf->GetStorageShape());

    // Shape validation: grad_output and self must have same shape
    OP_CHECK_IF(
        gradOutputShape.GetShapeSize() != selfShape.GetShapeSize(),
        OP_LOGE(context, "HardSigmoidGradV3: grad_output and self shape size mismatch: grad=%ld, self=%ld",
                gradOutputShape.GetShapeSize(), selfShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalNum = gradOutputShape.GetShapeSize();

    // dtype validation
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "HardSigmoidGradV3: unsupported dtype");
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

static ge::graphStatus InitBasicTilingInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum, int64_t& totalNum, ge::DataType& dataType)
{
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetEmptyInputTiling(
    gert::TilingContext* context, HardSigmoidGradV3TilingData* tiling, ge::DataType dataType)
{
    tiling->blockFactor = 0;
    tiling->ubFactor = 0;
    context->SetBlockDim(1);
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    uint64_t useDoubleBuffer = 0;
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeBlockTiling(
    gert::TilingContext* context,
    HardSigmoidGradV3TilingData* tiling,
    uint64_t ubSize,
    int64_t coreNum,
    int64_t totalNum,
    ge::DataType dataType)
{
    int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "typeSize is invalid"), return ge::GRAPH_FAILED);

    int64_t cacheLineElements = CACHE_LINE_BYTE_LENGTH / typeSize;
    OP_CHECK_IF(cacheLineElements <= 0, OP_LOGE(context, "cacheLineElements is invalid"), return ge::GRAPH_FAILED);

    int64_t totalLengthCore = CeilDiv(totalNum, coreNum);
    int64_t totalLengthCoreAlign = CeilDiv(totalLengthCore, cacheLineElements) * cacheLineElements;
    OP_CHECK_IF(totalLengthCoreAlign <= 0, OP_LOGE(context, "totalLengthCoreAlign is invalid"),
                return ge::GRAPH_FAILED);

    int64_t usedCoreNum = CeilDiv(totalNum, totalLengthCoreAlign);
    tiling->blockFactor = totalLengthCoreAlign;

    int64_t bufferCoefficient = FP32_BUFFER_COEFFICIENT;
    if (dataType == ge::DT_FLOAT16) {
        bufferCoefficient = FP16_VECTOR_BUFFER_COEFFICIENT;
    }
    OP_CHECK_IF(bufferCoefficient <= 0, OP_LOGE(context, "bufferCoefficient is invalid"), return ge::GRAPH_FAILED);

    int64_t maxTileElements = static_cast<int64_t>(ubSize) / bufferCoefficient;
    int64_t alignElements = 32 / typeSize;
    int64_t compareAlignElements = 32 / typeSize;
    if (dataType == ge::DT_FLOAT16) {
        compareAlignElements = COMPARE_ALIGN_BYTES / typeSize;
    } else {
        compareAlignElements = COMPARE_ALIGN_BYTES / static_cast<int64_t>(sizeof(float));
    }
    if (compareAlignElements > alignElements) {
        alignElements = compareAlignElements;
    }

    int64_t tileLength = AlignDown(maxTileElements, alignElements);
    if (tileLength == 0) {
        tileLength = alignElements;
    }
    int64_t blockTileLimit = AlignDown(totalLengthCoreAlign, alignElements);
    if (blockTileLimit == 0) {
        blockTileLimit = alignElements;
    }
    if (blockTileLimit < tileLength) {
        tileLength = blockTileLimit;
    }
    tiling->ubFactor = tileLength;

    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingKey(gert::TilingContext* context, ge::DataType dataType)
{
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    uint64_t useDoubleBuffer = 1;
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HardSigmoidGradV3TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    OP_CHECK_IF(
        InitBasicTilingInfo(context, ubSize, coreNum, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "InitBasicTilingInfo error"),
        return ge::GRAPH_FAILED);

    HardSigmoidGradV3TilingData* tiling = context->GetTilingData<HardSigmoidGradV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(HardSigmoidGradV3TilingData), 0, sizeof(HardSigmoidGradV3TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    if (totalNum == 0) {
        return SetEmptyInputTiling(context, tiling, dataType);
    }

    OP_CHECK_IF(
        ComputeBlockTiling(context, tiling, ubSize, coreNum, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ComputeBlockTiling error"),
        return ge::GRAPH_FAILED);

    // The BF16 full-tile kernel shares the same double-buffer UB model as FP32:
    // 3 queue buffers + 3 FP32 temp buffers + 1 compare mask = 25 bytes/element.
    // Disabling double buffer here deadlocks the pipelined Process() path when
    // fullTileNum > 1 because CopyIn(next) runs before Compute(curr) frees the
    // only queue slot.
    OP_CHECK_IF(
        SetTilingKey(context, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetTilingKey error"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHardSigmoidGradV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct HardSigmoidGradV3CompileInfo {};

IMPL_OP_OPTILING(HardSigmoidGradV3)
    .Tiling(HardSigmoidGradV3TilingFunc)
    .TilingParse<HardSigmoidGradV3CompileInfo>(TilingParseForHardSigmoidGradV3);

} // namespace optiling
