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
 * \file apply_adadelta_tiling.cpp
 * \brief ApplyAdadelta tiling implementation (arch35)
 *
 * TilingKey selection:
 *   TilingKey_0: D_T_X=C_DT_FLOAT,   BUFFER_MODE=0  (FP32, single buffer)
 *   TilingKey_1: D_T_X=C_DT_FLOAT,   BUFFER_MODE=1  (FP32, double buffer)
 *   TilingKey_2: D_T_X=C_DT_FLOAT16, BUFFER_MODE=0  (FP16, single buffer)
 *   TilingKey_3: D_T_X=C_DT_FLOAT16, BUFFER_MODE=1  (FP16, double buffer)
 *
 * Scalar parameters (lr, rho, epsilon) are passed via TilingData.
 * The ACLNN layer reads aclScalar values and writes them into attrs
 * named "lr", "rho", "epsilon" for the Tiling function to extract.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/apply_adadelta_tiling_data.h"
#include "../op_kernel/apply_adadelta_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// UB tensor slot count per TilingKey (see DESIGN.md 3.4.7/3.5.6/3.6.6/3.7.6)
//
// Per-element UB byte cost (in fp32-equivalent slots; 1 slot == 4 B == sizeof(float)):
//   FP32 single buffer: 7 TQue (fp32, 4B) x 1 + 2 TBuf (fp32, 4B)
//                       = (7 * 4 + 2 * 4) B/elem = 36 B/elem = 9 slots
//   FP32 double buffer: 7 TQue (fp32, 4B) x 2 + 2 TBuf (fp32, 4B)
//                       = (7 * 4 * 2 + 2 * 4) B/elem = 64 B/elem = 16 slots
//   FP16 single buffer: 7 TQue (half, 2B) x 1 + 6 TBuf (fp32, 4B)
//                       = (7 * 2 + 6 * 4) B/elem = 38 B/elem
//                       => ceil(38 / 4) = 10 slots
//   FP16 double buffer: 7 TQue (half, 2B) x 2 + 6 TBuf (fp32, 4B)
//                       = (7 * 2 * 2 + 6 * 4) B/elem = 52 B/elem = 13 slots
//
// Note: BUFFER_MODE only affects TQue (which may be double-buffered via
// BUFFER_NUM=2); TBuf is NEVER double-buffered regardless of BUFFER_MODE.
// The "+2 TBuf" / "+6 fp32 TBuf" terms above stay constant across both
// single-buffer and double-buffer paths.
static int64_t GetTensorSlots(ge::DataType dataType, uint64_t bufferMode)
{
    if (dataType == ge::DT_FLOAT) {
        // FP32: see header comment for derivation
        return bufferMode ? 16 : 9;
    } else {
        // FP16: see header comment for derivation
        return bufferMode ? 13 : 10;
    }
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum must be > 0, got %ld", coreNum), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Attr order matches op_def: lr(0), rho(1), epsilon(2).
// epsilon == 0 would trigger Div-by-zero in sqrt(accum+eps) denominator.
static ge::graphStatus ParseScalarAttrs(gert::TilingContext* context, ApplyAdadeltaTilingData* tiling)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* lrPtr = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, lrPtr);
    const float* rhoPtr = attrs->GetFloat(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, rhoPtr);
    const float* epsPtr = attrs->GetFloat(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsPtr);
    OP_CHECK_IF(
        *epsPtr <= 0.0f,
        OP_LOGE(context, "epsilon must be > 0, got %f", *epsPtr),
        return ge::GRAPH_FAILED);
    tiling->lr = *lrPtr;
    tiling->rho = *rhoPtr;
    tiling->oneMinusRho = 1.0f - *rhoPtr;
    tiling->epsilon = *epsPtr;
    return ge::GRAPH_SUCCESS;
}

static int64_t ComputeSplit(ApplyAdadeltaTilingData* tiling, ge::DataType dataType,
                            uint64_t ubSize, int64_t coreNum, int64_t ubBlockSize, uint64_t& bufferMode)
{
    tiling->blockFactor = CeilAlign(CeilDiv(tiling->totalNum, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(tiling->totalNum, tiling->blockFactor);
    bufferMode = (tiling->totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t slots = GetTensorSlots(dataType, bufferMode);
    constexpr int64_t elemBytes = 4;  // Internal computation in fp32
    tiling->ubFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize) / elemBytes, slots), ubBlockSize);
    return usedCoreNum;
}

static ge::graphStatus GetShapeAndDataType(gert::TilingContext* context,
                                           int64_t& totalNum, ge::DataType& dataType)
{
    auto varShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    totalNum = varShape->GetStorageShape().GetShapeSize();
    auto varDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    dataType = varDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // User workspace is 0 (operator needs no extra workspace), system workspace
    // is obtained from AscendC platform interface to stay consistent with framework
    // expectations across SoC versions.
    ws[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context, uint32_t dTypeX)
{
    context->SetBlockDim(1);
    uint32_t bufMode = 0;
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, bufMode);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InitTilingData(gert::TilingContext* context,
                                      int64_t totalNum, ApplyAdadeltaTilingData*& tiling)
{
    tiling = context->GetTilingData<ApplyAdadeltaTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(*tiling), 0, sizeof(*tiling)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalNum;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyAdadeltaTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(
        GetShapeAndDataType(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAndDataType error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetWorkspace(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetWorkspace error"), return ge::GRAPH_FAILED);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    if (totalNum == 0) {
        return HandleEmptyTensor(context, dTypeX);
    }

    ApplyAdadeltaTilingData* tiling = nullptr;
    OP_CHECK_IF(
        InitTilingData(context, totalNum, tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "InitTilingData error"), return ge::GRAPH_FAILED);

    uint64_t bufferMode = 0;
    int64_t usedCoreNum = ComputeSplit(tiling, dataType, ubSize, coreNum, GetUbBlockSize(context), bufferMode);

    OP_CHECK_IF(
        ParseScalarAttrs(context, tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ParseScalarAttrs error"), return ge::GRAPH_FAILED);

    context->SetBlockDim(usedCoreNum);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, static_cast<uint32_t>(bufferMode));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForApplyAdadelta([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyAdadeltaCompileInfo {};

IMPL_OP_OPTILING(ApplyAdadelta)
    .Tiling(ApplyAdadeltaTilingFunc)
    .TilingParse<ApplyAdadeltaCompileInfo>(TilingParseForApplyAdadelta);

} // namespace optiling
