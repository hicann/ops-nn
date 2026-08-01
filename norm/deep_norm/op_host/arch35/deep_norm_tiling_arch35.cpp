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
 * Splits the leading dims (rows N) across AI cores. Rows that fit in UB use the
 * full-load kernel; larger reduce axes use a bounded partial-load tile.
 */

#include "deep_norm_tiling_arch35.h"
#include <algorithm>
#include <string>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/deep_norm_tiling_data.h"
#include "../deep_norm_shape_check.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t VL_FP32 = 256U / sizeof(float); // fp32 vector length (matches kernel)
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BLK_B32 = BLOCK_SIZE / sizeof(float);
constexpr uint32_t MIN_REDUCE_TMP_ELEMS = 2 * VL_FP32;
constexpr uint32_t MAX_PARTIAL_TILE_LENGTH = 4096;
// The legacy arch22 kernel switches from its intermediate Extra path to the
// true Common path above these limits. Arch35 can keep that intermediate range
// in the tighter regbase full-load layout, then use bounded three-pass partial-load beyond it.
constexpr uint32_t FP32_FULL_LOAD_LIMIT = 8192;
constexpr uint32_t FP16_BF16_FULL_LOAD_LIMIT = 15360;
constexpr uint64_t PARTIAL_RESERVED_SIZE = 2048;
constexpr uint64_t UB_QUEUE_COUNT = 5; // x/gx/gamma/beta/y dtype-sized queues
constexpr int64_t ATTR_ALPHA_INDEX = 0;
constexpr int64_t ATTR_EPSILON_INDEX = 1;
constexpr int32_t INPUT_X_INDEX = 0;
constexpr int32_t INPUT_GAMMA_INDEX = 3;
constexpr int32_t OUTPUT_MEAN_INDEX = 0;
constexpr int32_t OUTPUT_RSTD_INDEX = 1;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "coreNum is 0"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "ubSize is 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// The shared legacy-compatible checker validates output ranks and leading dims but
// does not require the reduced trailing dims to be one. Enforce that contract
// only in arch35 so this fix does not alter the legacy tiling path.
static ge::graphStatus CheckReduceOutputTail(gert::TilingContext* context)
{
    auto xShapePtr = context->GetInputShape(INPUT_X_INDEX);
    auto gammaShapePtr = context->GetInputShape(INPUT_GAMMA_INDEX);
    auto meanShapePtr = context->GetOutputShape(OUTPUT_MEAN_INDEX);
    auto rstdShapePtr = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstdShapePtr);

    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    const gert::Shape& gammaShape = gammaShapePtr->GetStorageShape();
    const gert::Shape& meanShape = meanShapePtr->GetStorageShape();
    const gert::Shape& rstdShape = rstdShapePtr->GetStorageShape();
    size_t reduceStart = xShape.GetDimNum() - gammaShape.GetDimNum();
    for (size_t i = reduceStart; i < xShape.GetDimNum(); ++i) {
        OP_CHECK_IF(meanShape.GetDim(i) != 1 || rstdShape.GetDim(i) != 1,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "mean/rstd", "invalid",
                                                           "reduced trailing dimensions must be 1"),
                    return ge::GRAPH_FAILED);
    }
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
    // Note: x.dim > gamma.dim is already enforced by CheckDeepNormShapeDim (called at tiling entry).

    numCol = 1;
    for (size_t i = 0; i < gammaDimNum; ++i) {
        int64_t dim = gammaShape.GetDim(i);
        OP_CHECK_IF(dim <= 0 || numCol > static_cast<int64_t>(UINT32_MAX) / dim,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                          "numCol must be positive and fit uint32"),
                    return ge::GRAPH_FAILED);
        numCol *= dim;
    }
    numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; ++i) {
        int64_t dim = xShape.GetDim(i);
        OP_CHECK_IF(dim <= 0 || numRow > static_cast<int64_t>(UINT32_MAX) / dim,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                          "numRow must be positive and fit uint32"),
                    return ge::GRAPH_FAILED);
        numRow *= dim;
    }
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
    OP_CHECK_IF(rowPerCore <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "rowPerCore must be positive"),
                return ge::GRAPH_FAILED);
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
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "numRow/numColAlign exceeds uint32 range"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDtypeSize(gert::TilingContext* context, uint64_t& dtypeSize)
{
    auto xDescPtr = context->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDescPtr);
    int64_t dtSize = GetSizeByDataType(xDescPtr->GetDataType());
    OP_CHECK_IF(
        dtSize <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "invalid x dtype size"),
        return ge::GRAPH_FAILED);
    dtypeSize = static_cast<uint64_t>(dtSize);
    return ge::GRAPH_SUCCESS;
}

// Mirrors the key-0 kernel allocation exactly. The reduce scratch grows with D,
// so a fixed reserve underestimates UB near the full-load boundary.
static uint64_t CalcFullLoadUbRequired(uint64_t numColAlign, uint64_t dtypeSize)
{
    uint64_t foldLoops = CeilDiv(numColAlign, static_cast<uint64_t>(VL_FP32));
    uint64_t reduceTmpElems = CeilAlign(foldLoops, static_cast<uint64_t>(BLK_B32));
    reduceTmpElems = std::max(reduceTmpElems, static_cast<uint64_t>(MIN_REDUCE_TMP_ELEMS));
    uint64_t tensorBytes = numColAlign * (UB_QUEUE_COUNT * dtypeSize + sizeof(float));
    uint64_t scalarBytes = 3 * BLOCK_SIZE; // mean/rstd queues and sum buffer
    return tensorBytes + reduceTmpElems * sizeof(float) + scalarBytes;
}

static ge::graphStatus CalcPartialTileLength(gert::TilingContext* context, uint64_t ubSize, uint64_t dtypeSize,
                                             uint32_t& tileLength)
{
    OP_CHECK_IF(ubSize <= PARTIAL_RESERVED_SIZE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB", std::to_string(ubSize).c_str(),
                                                      "UB cannot hold one partial-load tile"),
                return ge::GRAPH_FAILED);
    uint64_t bytesPerElement = UB_QUEUE_COUNT * dtypeSize + sizeof(float);
    uint64_t maxElements = (ubSize - PARTIAL_RESERVED_SIZE) / bytesPerElement;
    maxElements = std::min(maxElements, static_cast<uint64_t>(MAX_PARTIAL_TILE_LENGTH));
    maxElements = maxElements / VL_FP32 * VL_FP32;
    OP_CHECK_IF(maxElements < VL_FP32,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB", std::to_string(ubSize).c_str(),
                                                      "UB cannot hold one regbase partial-load tile"),
                return ge::GRAPH_FAILED);
    tileLength = static_cast<uint32_t>(maxElements);
    return ge::GRAPH_SUCCESS;
}

// Sets workspace size and writes the computed values into the tiling data / block dim / tiling key.
static ge::graphStatus SetTilingData(gert::TilingContext* context, int64_t usedCoreNum, int64_t numCol, int64_t numRow,
                                     int64_t rowPerCore, int64_t numColAlign, int64_t powerSplit, float eps,
                                     float alpha, uint32_t tileLength, uint64_t tilingKey)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    DeepNormTilingData* tiling = context->GetTilingData<DeepNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(DeepNormTilingData), 0, sizeof(DeepNormTilingData)) != EOK,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "set tiling data error"),
        return ge::GRAPH_FAILED);

    tiling->numCore = static_cast<uint32_t>(usedCoreNum);
    tiling->numCol = static_cast<uint32_t>(numCol);
    tiling->numRow = static_cast<uint32_t>(numRow);
    tiling->rowPerCore = static_cast<uint32_t>(rowPerCore);
    tiling->numColAlign = static_cast<uint32_t>(numColAlign);
    tiling->powerSplit = static_cast<uint32_t>(powerSplit);
    tiling->eps = eps;
    tiling->alpha = alpha;
    tiling->avgFactor = 1.0f / static_cast<float>(numCol);
    tiling->tileLength = tileLength;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DeepNormTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(CheckDeepNormShapeDim(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "Input shape dim invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDeepNormShapeValue(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "Input shape value invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckReduceOutputTail(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "Output reduce shape invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckDeepNormDtype(context) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "Input dtype invalid."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

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

    uint64_t dtypeSize = 0;
    if (GetDtypeSize(context, dtypeSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t fullLoadLimit = dtypeSize == sizeof(float) ? FP32_FULL_LOAD_LIMIT : FP16_BF16_FULL_LOAD_LIMIT;
    bool useFullLoad = static_cast<uint64_t>(numCol) <= fullLoadLimit &&
                       CalcFullLoadUbRequired(static_cast<uint64_t>(numColAlign), dtypeSize) <= ubSize;
    uint32_t tileLength = 0;
    if (!useFullLoad && CalcPartialTileLength(context, ubSize, dtypeSize, tileLength) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tilingKey = useFullLoad ? 0U : 1U;

    if (SetTilingData(context, usedCoreNum, numCol, numRow, rowPerCore, numColAlign, powerSplit, eps, alpha, tileLength,
                      tilingKey) != ge::GRAPH_SUCCESS) {
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
