/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file average_pooling_grad_tiling.cpp
 * \brief AveragePoolingGrad tiling for generic avg_pool2d backward semantics.
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/platform_util.h"
#include "../op_kernel/average_pooling_grad_tiling_data.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_ELEMENTS_PER_CORE = 1024;
constexpr uint32_t TILING_KEY_SCALAR = 0U;
constexpr uint32_t TILING_KEY_TILED = 1U;
constexpr uint32_t TILING_KEY_NO_OVERLAP = 2U;
constexpr int64_t DEFAULT_TILE_H = 8;
constexpr int64_t DEFAULT_TILE_W = 64;
constexpr int64_t ALIGN_ELEMENTS = 8;
constexpr size_t ORIG_SHAPE_IDX = 0;
constexpr size_t GRAD_OUTPUT_IDX = 1;
constexpr size_t OUTPUT_IDX = 0;
constexpr size_t NCHW_DIMS = 4;
constexpr size_t ATTR_KERNEL_H = 0;
constexpr size_t ATTR_KERNEL_W = 1;
constexpr size_t ATTR_STRIDE_H = 2;
constexpr size_t ATTR_STRIDE_W = 3;
constexpr size_t ATTR_PAD_TOP = 4;
constexpr size_t ATTR_PAD_BOTTOM = 5;
constexpr size_t ATTR_PAD_LEFT = 6;
constexpr size_t ATTR_PAD_RIGHT = 7;
constexpr size_t ATTR_CEIL_MODE = 8;
constexpr size_t ATTR_COUNT_INCLUDE_PAD = 9;
constexpr size_t ATTR_DIVISOR_OVERRIDE = 10;

struct AveragePoolingGradCompileInfo {
};

static int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

static int64_t PoolingOutShape(int64_t inputSize, int64_t kernel, int64_t padBefore, int64_t padAfter,
    int64_t stride, bool ceilMode)
{
    if (stride <= 0) {
        return 0;
    }
    int64_t numerator = inputSize + padBefore + padAfter - kernel;
    int64_t out = ceilMode ? CeilDiv(numerator, stride) + 1 : numerator / stride + 1;
    if (ceilMode && (out - 1) * stride >= inputSize + padBefore) {
        --out;
    }
    return out;
}

static uint32_t AlignUpElements(int64_t elements)
{
    return static_cast<uint32_t>(CeilDiv(elements, ALIGN_ELEMENTS) * ALIGN_ELEMENTS);
}

static ge::graphStatus AveragePoolingGradTilingFunc(gert::TilingContext* context)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto getIntAttr = [attrs](size_t index, int64_t defaultValue) -> int64_t {
        const int64_t* value = attrs->GetAttrPointer<int64_t>(index);
        return value == nullptr ? defaultValue : *value;
    };
    auto getBoolAttr = [attrs](size_t index, bool defaultValue) -> bool {
        const bool* value = attrs->GetAttrPointer<bool>(index);
        return value == nullptr ? defaultValue : *value;
    };

    const int64_t kernelH = getIntAttr(ATTR_KERNEL_H, 2);
    const int64_t kernelW = getIntAttr(ATTR_KERNEL_W, 2);
    const int64_t strideH = getIntAttr(ATTR_STRIDE_H, 1);
    const int64_t strideW = getIntAttr(ATTR_STRIDE_W, 1);
    const int64_t padTop = getIntAttr(ATTR_PAD_TOP, 0);
    const int64_t padBottom = getIntAttr(ATTR_PAD_BOTTOM, 0);
    const int64_t padLeft = getIntAttr(ATTR_PAD_LEFT, 0);
    const int64_t padRight = getIntAttr(ATTR_PAD_RIGHT, 0);
    const bool ceilMode = getBoolAttr(ATTR_CEIL_MODE, false);
    const bool countIncludePad = getBoolAttr(ATTR_COUNT_INCLUDE_PAD, true);
    const int64_t divisorOverride = getIntAttr(ATTR_DIVISOR_OVERRIDE, 0);

    OP_CHECK_IF(kernelH <= 0 || kernelW <= 0 || strideH <= 0 || strideW <= 0,
        OP_LOGE(context, "kernel and stride must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0 || divisorOverride < 0,
        OP_LOGE(context, "pads and divisor_override must be non-negative"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(padTop * 2 > kernelH || padBottom * 2 > kernelH || padLeft * 2 > kernelW || padRight * 2 > kernelW,
        OP_LOGE(context, "padding should be no more than half of kernel size"), return ge::GRAPH_FAILED);

    const gert::Tensor* shapeTensor = context->GetInputTensor(ORIG_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeTensor);
    const int64_t* shapeValue = shapeTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeValue);
    OP_CHECK_IF(shapeTensor->GetOriginShape().GetShapeSize() != NCHW_DIMS,
        OP_LOGE(context, "orig_input_shape must have 4 elements"), return ge::GRAPH_FAILED);

    const int64_t n = shapeValue[0];
    const int64_t c = shapeValue[1];
    const int64_t hIn = shapeValue[2];
    const int64_t wIn = shapeValue[3];
    OP_CHECK_IF(n <= 0 || c <= 0 || hIn <= 0 || wIn <= 0,
        OP_LOGE(context, "invalid orig_input_shape"), return ge::GRAPH_FAILED);

    const auto* gradShapeStorage = context->GetInputShape(GRAD_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShapeStorage);
    const gert::Shape& gradShape = gradShapeStorage->GetStorageShape();
    OP_CHECK_IF(gradShape.GetDimNum() != NCHW_DIMS,
        OP_LOGE(context, "grad_output must be 4D"), return ge::GRAPH_FAILED);
    const int64_t hOut = gradShape.GetDim(2);
    const int64_t wOut = gradShape.GetDim(3);
    OP_CHECK_IF(gradShape.GetDim(0) != n || gradShape.GetDim(1) != c || hOut <= 0 || wOut <= 0,
        OP_LOGE(context, "grad_output shape mismatch"), return ge::GRAPH_FAILED);

    const int64_t expectedH = PoolingOutShape(hIn, kernelH, padTop, padBottom, strideH, ceilMode);
    const int64_t expectedW = PoolingOutShape(wIn, kernelW, padLeft, padRight, strideW, ceilMode);
    OP_CHECK_IF(expectedH != hOut || expectedW != wOut,
        OP_LOGE(context, "grad_output shape mismatch, expect [%ld,%ld], got [%ld,%ld]", expectedH, expectedW, hOut, wOut),
        return ge::GRAPH_FAILED);

    const auto* outShapeStorage = context->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapeStorage);
    const gert::Shape& outShape = outShapeStorage->GetStorageShape();
    OP_CHECK_IF(outShape.GetDimNum() != NCHW_DIMS || outShape.GetDim(0) != n || outShape.GetDim(1) != c ||
        outShape.GetDim(2) != hIn || outShape.GetDim(3) != wIn,
        OP_LOGE(context, "grad_input shape mismatch"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(GRAD_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    OP_CHECK_IF(supportedDtype.count(inputDesc->GetDataType()) == 0,
        OP_LOGE(context, "unsupported dtype"), return ge::GRAPH_FAILED);

    const int64_t total = n * c * hIn * wIn;
    OP_CHECK_IF(total <= 0 || total > static_cast<int64_t>(UINT32_MAX),
        OP_LOGE(context, "total elements out of range"), return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = ascendcPlatform.GetCoreNum();
    }
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    uint32_t tileH = 1;
    uint32_t tileW = 1;
    uint32_t tileTaskNum = 0;
    uint32_t gradOutCacheDataNum = 0;
    const uint32_t requestedCoreNum = static_cast<uint32_t>(std::max<int64_t>(1, CeilDiv(total, MIN_ELEMENTS_PER_CORE)));
    uint32_t usedCoreNum = 1;
    uint32_t normalCoreProcessNum = 0;
    uint32_t tailCoreProcessNum = 0;
    uint32_t tilingKey = TILING_KEY_SCALAR;

    tileH = static_cast<uint32_t>(std::min<int64_t>(hIn, DEFAULT_TILE_H));
    tileW = static_cast<uint32_t>(std::min<int64_t>(wIn, DEFAULT_TILE_W));
    const int64_t hTileNum = CeilDiv(hIn, tileH);
    const int64_t wTileNum = CeilDiv(wIn, tileW);
    const int64_t taskNum = n * c * hTileNum * wTileNum;
    OP_CHECK_IF(taskNum <= 0 || taskNum > static_cast<int64_t>(UINT32_MAX),
        OP_LOGE(context, "tile task num out of range"), return ge::GRAPH_FAILED);
    tileTaskNum = static_cast<uint32_t>(taskNum);
    const uint32_t tileCoreNum = std::max(1U, std::min(coreNum, tileTaskNum));
    usedCoreNum = std::min(coreNum, std::max(requestedCoreNum, tileCoreNum));
    OP_CHECK_IF(usedCoreNum == 0, OP_LOGE(context, "usedCoreNum is 0"), return ge::GRAPH_FAILED);
    normalCoreProcessNum = static_cast<uint32_t>((total + usedCoreNum - 1) / usedCoreNum);
    tailCoreProcessNum = static_cast<uint32_t>(total - static_cast<int64_t>(normalCoreProcessNum) * (usedCoreNum - 1));

    const int64_t maxCacheH = std::min<int64_t>(hOut, CeilDiv(tileH + kernelH - 1, strideH) + 1);
    const int64_t maxCacheW = std::min<int64_t>(wOut, CeilDiv(tileW + kernelW - 1, strideW) + 1);
    gradOutCacheDataNum = static_cast<uint32_t>(maxCacheH) * AlignUpElements(maxCacheW);
    OP_CHECK_IF(gradOutCacheDataNum == 0, OP_LOGE(context, "invalid grad output cache"), return ge::GRAPH_FAILED);
    const bool noOverlap = padTop == 0 && padBottom == 0 && padLeft == 0 && padRight == 0 &&
        strideH >= kernelH && strideW >= kernelW;
    tilingKey = noOverlap ? TILING_KEY_NO_OVERLAP : TILING_KEY_TILED;

    AveragePoolingGradTilingData* tiling = context->GetTilingData<AveragePoolingGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(AveragePoolingGradTilingData), 0, sizeof(AveragePoolingGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->n = static_cast<uint32_t>(n);
    tiling->c = static_cast<uint32_t>(c);
    tiling->hIn = static_cast<uint32_t>(hIn);
    tiling->wIn = static_cast<uint32_t>(wIn);
    tiling->hOut = static_cast<uint32_t>(hOut);
    tiling->wOut = static_cast<uint32_t>(wOut);
    tiling->kernelH = static_cast<uint32_t>(kernelH);
    tiling->kernelW = static_cast<uint32_t>(kernelW);
    tiling->strideH = static_cast<uint32_t>(strideH);
    tiling->strideW = static_cast<uint32_t>(strideW);
    tiling->padTop = static_cast<uint32_t>(padTop);
    tiling->padLeft = static_cast<uint32_t>(padLeft);
    tiling->countIncludePad = countIncludePad ? 1U : 0U;
    tiling->divisorOverride = static_cast<int32_t>(divisorOverride);
    const int64_t divisor = divisorOverride != 0 ? divisorOverride : kernelH * kernelW;
    tiling->invDivisor = 1.0f / static_cast<float>(divisor);
    tiling->totalElements = static_cast<uint32_t>(total);
    tiling->normalCoreProcessNum = normalCoreProcessNum;
    tiling->tailCoreProcessNum = tailCoreProcessNum;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tileH = tileH;
    tiling->tileW = tileW;
    tiling->tileTaskNum = tileTaskNum;
    tiling->gradOutCacheDataNum = gradOutCacheDataNum;

    context->SetBlockDim(static_cast<int32_t>(usedCoreNum));
    context->SetTilingKey(tilingKey);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAveragePoolingGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AveragePoolingGrad).Tiling(AveragePoolingGradTilingFunc).TilingParse<AveragePoolingGradCompileInfo>(TilingParseForAveragePoolingGrad);
} // namespace optiling
