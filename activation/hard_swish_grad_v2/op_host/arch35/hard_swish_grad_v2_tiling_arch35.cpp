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

/*!
 * \file hard_swish_grad_v2_tiling_arch35.cpp
 * \brief HardSwishGradV2 tiling implementation for arch35 (Ascend950)
 *
 * Computes multi-core split (blockFactor) and UB split (ubFactor) parameters.
 * Selects single/double buffer mode based on data volume threshold.
 * For fp16/bf16, accounts for extra fp32 intermediate buffers.
 */

#include "hard_swish_grad_v2_tiling_arch35.h"
#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/hard_swish_grad_v2_tiling_data.h"
#include "../../op_kernel/arch35/hard_swish_grad_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t SIZE_5 = 5;
constexpr int64_t SIZE_8 = 8;
constexpr int64_t SIZE_11 = 11;
constexpr size_t MAX_DIM_NUM = 8;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
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

// grad_output/self/out 三者 shape 必须相同,且维度数不超过 8;输出 shapeGradOutput 供后续计算 totalIdx。
static ge::graphStatus CheckShapeInfo(gert::TilingContext* context, gert::Shape& shapeGradOutput)
{
    auto inputGradOutput = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGradOutput);
    shapeGradOutput = EnsureNotScalar(inputGradOutput->GetStorageShape());

    auto inputSelf = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto shapeSelf = EnsureNotScalar(inputSelf->GetStorageShape());
    OP_CHECK_IF(shapeGradOutput != shapeSelf,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "grad_output and self",
                    (Ops::Base::ToString(shapeGradOutput) + " and " + Ops::Base::ToString(shapeSelf)).c_str(),
                    "The shape of grad_output must be the same as the shape of self"),
                return ge::GRAPH_FAILED);

    auto outputOut = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputOut);
    auto shapeOut = EnsureNotScalar(outputOut->GetStorageShape());
    OP_CHECK_IF(shapeOut != shapeGradOutput,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "grad_output and out",
                    (Ops::Base::ToString(shapeGradOutput) + " and " + Ops::Base::ToString(shapeOut)).c_str(),
                    "The shape of out must be the same as the shape of grad_output"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(shapeGradOutput.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "grad_output",
                                                         std::to_string(shapeGradOutput.GetDimNum()).c_str(),
                                                         "The dim num of grad_output must be less than or equal to 8"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// grad_output dtype 须为 float32/float16/bfloat16,且 self/out 与 grad_output 一致。
static ge::graphStatus CheckDtypeInfo(gert::TilingContext* context, ge::DataType& dataType)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "grad_output", Ops::Base::ToString(dataType).c_str(),
                                  "float32, float16, bfloat16");
        return ge::GRAPH_FAILED;
    }

    auto inputDescSelf = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescSelf);
    OP_CHECK_IF(inputDescSelf->GetDataType() != dataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "grad_output, self",
                    (Ops::Base::ToString(dataType) + ", " + Ops::Base::ToString(inputDescSelf->GetDataType())).c_str(),
                    "The dtype of self must be the same as the dtype of grad_output"),
                return ge::GRAPH_FAILED);

    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(outputDesc->GetDataType() != dataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "grad_output, out",
                    (Ops::Base::ToString(dataType) + ", " + Ops::Base::ToString(outputDesc->GetDataType())).c_str(),
                    "The dtype of out must be the same as the dtype of grad_output"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    gert::Shape shapeGradOutput;
    OP_CHECK_IF(CheckShapeInfo(context, shapeGradOutput) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckShapeInfo error"),
                return ge::GRAPH_FAILED);
    totalIdx = shapeGradOutput.GetShapeSize();
    OP_CHECK_IF(CheckDtypeInfo(context, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckDtypeInfo error"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HardSwishGradV2TilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Get shape and dtype
    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. Set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    // 4. Compute tiling parameters
    HardSwishGradV2Arch35TilingData* tiling = context->GetTilingData<HardSwishGradV2Arch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(HardSwishGradV2Arch35TilingData), 0, sizeof(HardSwishGradV2Arch35TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (totalIdx == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, 0ULL);
        return ge::GRAPH_SUCCESS;
    }

    int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;

    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilDiv(totalIdx, coreNum);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    uint64_t useDoubleBuffer = totalIdx > MIN_SPLIT_THRESHOLD;
    int64_t bufferNum;
    if (dataType == ge::DT_FLOAT) {
        bufferNum = (useDoubleBuffer != 0) ? SIZE_8 : SIZE_5;
    } else {
        bufferNum = (useDoubleBuffer != 0) ? SIZE_11 : SIZE_8;
    }

    constexpr int64_t VECTOR_ALIGN_ELEM = 256 / static_cast<int64_t>(sizeof(float));
    int64_t ubBlockSize = GetUbBlockSize(context);
    int64_t alignUnit = std::max(ubBlockSize, VECTOR_ALIGN_ELEM);
    tiling->ubFactor = FloorAlign(FloorDiv((static_cast<int64_t>(ubSize) / typeSize), bufferNum), alignUnit);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHardSwishGradV2(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<HardSwishGradV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HardSwishGradV2)
    .Tiling(HardSwishGradV2TilingFunc)
    .TilingParse<HardSwishGradV2CompileInfo>(TilingParseForHardSwishGradV2);

} // namespace optiling
