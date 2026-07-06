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
 * \file deep_norm_tiling_arch35.cpp
 * \brief DeepNorm tiling implementation for arch35 (Ascend950).
 *
 * Splits the leading dims (rows N) across AI cores; the whole reduce axis D is
 * full-loaded per row. Computes the power-of-two fold point used by the regbase
 * reduction helpers.
 */

#include "deep_norm_tiling_arch35.h"
#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/deep_norm_tiling_data.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t VL_FP32 = 256U / sizeof(float); // fp32 vector length (matches kernel)
constexpr int64_t UB_RESERVED_SIZE = 1024;         // covers mean/rstd/sum block bufs + binaryAddBuf + slack
constexpr int64_t UB_QUEUE_COUNT = 5;              // x/gx/gamma/beta/y dtype-sized queues
constexpr int64_t ATTR_ALPHA_INDEX = 0;
constexpr int64_t ATTR_EPSILON_INDEX = 1;
constexpr int32_t INPUT_X_INDEX = 0;
constexpr int32_t INPUT_GAMMA_INDEX = 3;

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

// Parses x/gamma shapes into the reduce-axis length numCol and the leading-dim product numRow.
static ge::graphStatus GetShapeInfo(gert::TilingContext* context, int64_t& numCol, int64_t& numRow)
{
    auto xShapePtr = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto gammaShapePtr = context->GetInputShape(INPUT_GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShapePtr);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    const gert::Shape& gammaShape = gammaShapePtr->GetStorageShape();

    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    OP_CHECK_IF(xDimNum <= gammaDimNum, OP_LOGE(context, "x dim num must be greater than gamma dim num"),
                return ge::GRAPH_FAILED);

    numCol = gammaShape.GetShapeSize();
    OP_CHECK_IF(numCol <= 0, OP_LOGE(context, "numCol must be positive"), return ge::GRAPH_FAILED);
    numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; ++i) {
        numRow *= xShape.GetDim(i);
    }
    OP_CHECK_IF(numRow <= 0, OP_LOGE(context, "numRow must be positive"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Reads the alpha/epsilon attributes, falling back to the operator defaults when absent.
static ge::graphStatus GetAttrInfo(gert::TilingContext* context, float& alpha, float& eps)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* alphaPtr = attrs->GetFloat(ATTR_ALPHA_INDEX);
    const float* epsPtr = attrs->GetFloat(ATTR_EPSILON_INDEX);
    alpha = (alphaPtr == nullptr) ? 0.3f : *alphaPtr;
    eps = (epsPtr == nullptr) ? 1e-6f : *epsPtr;
    return ge::GRAPH_SUCCESS;
}

// Computes the core split, aligned reduce length and power-of-two fold point, with uint32 range guard.
static ge::graphStatus CalcTilingParams(gert::TilingContext* context, int64_t numCol, int64_t numRow, int64_t coreNum,
                                        int64_t& rowPerCore, int64_t& usedCoreNum, int64_t& numColAlign,
                                        int64_t& powerSplit)
{
    rowPerCore = CeilDiv(numRow, coreNum);
    OP_CHECK_IF(rowPerCore <= 0, OP_LOGE(context, "rowPerCore must be positive"), return ge::GRAPH_FAILED);
    usedCoreNum = CeilDiv(numRow, rowPerCore);

    numColAlign = CeilAlign(numCol, static_cast<int64_t>(VL_FP32));
    powerSplit = VL_FP32;
    if (numCol > static_cast<int64_t>(VL_FP32)) {
        while (powerSplit < numCol) {
            powerSplit *= 2;
        }
        powerSplit /= 2;
    }

    // Range guard: tiling-data fields are uint32; reject shapes that would silently truncate.
    OP_CHECK_IF(numRow > static_cast<int64_t>(UINT32_MAX) || numColAlign > static_cast<int64_t>(UINT32_MAX),
                OP_LOGE(context, "numRow/numColAlign exceeds uint32 range"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// UB capacity guard: the whole reduce axis D is full-loaded per row. Per core the kernel
// InitBuffers 5 dtype-sized queues (x/gx/gamma/beta/y) plus an fp32 hBuf of numColAlign, so a
// large D would overflow UB at kernel InitBuffer (template add_layer_norm has the same guard).
static ge::graphStatus CheckUbCapacity(gert::TilingContext* context, int64_t numCol, int64_t numColAlign,
                                       uint64_t ubSize)
{
    auto xDescPtr = context->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDescPtr);
    int64_t dtSize = GetSizeByDataType(xDescPtr->GetDataType());
    OP_CHECK_IF(dtSize <= 0, OP_LOGE(context, "invalid x dtype size"), return ge::GRAPH_FAILED);
    int64_t ubRequired = numColAlign * (UB_QUEUE_COUNT * dtSize + static_cast<int64_t>(sizeof(float))) +
                         UB_RESERVED_SIZE;
    OP_CHECK_IF(ubRequired > static_cast<int64_t>(ubSize),
                OP_LOGE(context, "numCol %ld needs UB %ld over capacity %lu", numCol, ubRequired, ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Sets workspace size and writes the computed values into the tiling data / block dim / tiling key.
static ge::graphStatus SetTilingData(gert::TilingContext* context, int64_t usedCoreNum, int64_t numCol, int64_t numRow,
                                     int64_t rowPerCore, int64_t numColAlign, int64_t powerSplit, float eps,
                                     float alpha)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    DeepNormTilingData* tiling = context->GetTilingData<DeepNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(DeepNormTilingData), 0, sizeof(DeepNormTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->numCore = static_cast<uint32_t>(usedCoreNum);
    tiling->numCol = static_cast<uint32_t>(numCol);
    tiling->numRow = static_cast<uint32_t>(numRow);
    tiling->rowPerCore = static_cast<uint32_t>(rowPerCore);
    tiling->numColAlign = static_cast<uint32_t>(numColAlign);
    tiling->powerSplit = static_cast<uint32_t>(powerSplit);
    tiling->eps = eps;
    tiling->alpha = alpha;
    tiling->avgFactor = 1.0f / static_cast<float>(numCol);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DeepNormTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t numCol = 0;
    int64_t numRow = 0;
    if (GetShapeInfo(context, numCol, numRow) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    float alpha = 0.0f;
    float eps = 0.0f;
    if (GetAttrInfo(context, alpha, eps) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t rowPerCore = 0;
    int64_t usedCoreNum = 0;
    int64_t numColAlign = 0;
    int64_t powerSplit = 0;
    if (CalcTilingParams(context, numCol, numRow, coreNum, rowPerCore, usedCoreNum, numColAlign, powerSplit) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckUbCapacity(context, numCol, numColAlign, ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (SetTilingData(context, usedCoreNum, numCol, numRow, rowPerCore, numColAlign, powerSplit, eps, alpha) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDeepNorm(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<DeepNormCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DeepNorm).Tiling(DeepNormTilingFunc).TilingParse<DeepNormCompileInfo>(TilingParseForDeepNorm);

} // namespace optiling
