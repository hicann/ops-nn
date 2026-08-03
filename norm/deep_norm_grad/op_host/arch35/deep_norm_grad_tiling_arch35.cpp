/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file deep_norm_grad_tiling_arch35.cpp
 *  \brief DeepNormGrad regbase tiling for DAV_3510.
 */

#include "deep_norm_grad_tiling_arch35.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>

#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/deep_norm_grad_tiling_data.h"

namespace optiling {
namespace {

constexpr int32_t INPUT_DY = 0;
constexpr int32_t INPUT_X = 1;
constexpr int32_t INPUT_GX = 2;
constexpr int32_t INPUT_GAMMA = 3;
constexpr int32_t INPUT_MEAN = 4;
constexpr int32_t INPUT_RSTD = 5;
constexpr int32_t OUTPUT_DX = 0;
constexpr int32_t OUTPUT_DGX = 1;
constexpr int32_t OUTPUT_DBETA = 2;
constexpr int32_t OUTPUT_DGAMMA = 3;
constexpr int64_t ATTR_ALPHA = 0;
constexpr size_t MIN_X_RANK = 2;
constexpr size_t MAX_X_RANK = 8;
constexpr size_t MIN_GAMMA_RANK = 1;
constexpr size_t MAX_GAMMA_RANK = 7;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t VL_FP32 = 256 / sizeof(float);
constexpr uint32_t MAX_TILE_LENGTH = 4096;
constexpr uint64_t UB_RESERVED_BYTES = 2048;
constexpr uint64_t DTYPE_QUEUE_COUNT = 6;
constexpr uint64_t FP32_BUFFER_COUNT = 4;
constexpr uint64_t MAX_ELEMENT_BYTES = sizeof(float);
constexpr size_t WORKSPACE_SIZE = 0;
constexpr uint8_t BATCH_MODE = 1;

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

bool SameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "AIV core number is zero"),
                return ge::GRAPH_FAILED);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= UB_RESERVED_BYTES,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB", std::to_string(ubSize).c_str(),
                                                      "UB size should exceed reserved bytes"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTensorPointers(gert::TilingContext* context)
{
    for (int32_t i = INPUT_DY; i <= INPUT_RSTD; ++i) {
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(i));
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(i));
    }
    for (int32_t i = OUTPUT_DX; i <= OUTPUT_DGAMMA; ++i) {
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputShape(i));
        OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(i));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypes(gert::TilingContext* context)
{
    ge::DataType dataType = context->GetInputDesc(INPUT_DY)->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "dy dtype is unsupported"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputDesc(INPUT_X)->GetDataType() != dataType ||
                    context->GetInputDesc(INPUT_GX)->GetDataType() != dataType ||
                    context->GetInputDesc(INPUT_GAMMA)->GetDataType() != dataType ||
                    context->GetOutputDesc(OUTPUT_DX)->GetDataType() != dataType ||
                    context->GetOutputDesc(OUTPUT_DGX)->GetDataType() != dataType,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "dy/x/gx/gamma/dx/dgx dtypes must match"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputDesc(INPUT_MEAN)->GetDataType() != ge::DT_FLOAT ||
                    context->GetInputDesc(INPUT_RSTD)->GetDataType() != ge::DT_FLOAT ||
                    context->GetOutputDesc(OUTPUT_DBETA)->GetDataType() != ge::DT_FLOAT ||
                    context->GetOutputDesc(OUTPUT_DGAMMA)->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "mean/rstd/dbeta/dgamma must be float32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MulShapeDim(gert::TilingContext* context, uint64_t& product, int64_t dim)
{
    OP_CHECK_IF(dim <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "all shape dimensions must be positive"),
                return ge::GRAPH_FAILED);
    uint64_t value = static_cast<uint64_t>(dim);
    OP_CHECK_IF(product > UINT64_MAX / value,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "shape product overflows uint64"),
                return ge::GRAPH_FAILED);
    product *= value;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapes(gert::TilingContext* context, uint64_t& numRows, uint64_t& numCols)
{
    const gert::Shape& dy = context->GetInputShape(INPUT_DY)->GetStorageShape();
    const gert::Shape& x = context->GetInputShape(INPUT_X)->GetStorageShape();
    const gert::Shape& gx = context->GetInputShape(INPUT_GX)->GetStorageShape();
    const gert::Shape& gamma = context->GetInputShape(INPUT_GAMMA)->GetStorageShape();
    const gert::Shape& mean = context->GetInputShape(INPUT_MEAN)->GetStorageShape();
    const gert::Shape& rstd = context->GetInputShape(INPUT_RSTD)->GetStorageShape();
    const gert::Shape& dx = context->GetOutputShape(OUTPUT_DX)->GetStorageShape();
    const gert::Shape& dgx = context->GetOutputShape(OUTPUT_DGX)->GetStorageShape();
    const gert::Shape& dbeta = context->GetOutputShape(OUTPUT_DBETA)->GetStorageShape();
    const gert::Shape& dgamma = context->GetOutputShape(OUTPUT_DGAMMA)->GetStorageShape();

    size_t xRank = dy.GetDimNum();
    size_t gammaRank = gamma.GetDimNum();
    OP_CHECK_IF(xRank < MIN_X_RANK || xRank > MAX_X_RANK || gammaRank < MIN_GAMMA_RANK || gammaRank > MAX_GAMMA_RANK ||
                    xRank <= gammaRank,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "rank is outside the DeepNormGrad support range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SameShape(dy, x) || !SameShape(dy, gx) || !SameShape(dy, dx) || !SameShape(dy, dgx),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "dy/x/gx/dx/dgx shapes must match"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(mean.GetDimNum() != xRank || rstd.GetDimNum() != xRank,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "mean/rstd ranks must match dy rank"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SameShape(gamma, dbeta) || !SameShape(gamma, dgamma),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "dbeta/dgamma shapes must match gamma"),
                return ge::GRAPH_FAILED);

    for (size_t i = 0; i < xRank; ++i) {
        OP_CHECK_IF(dy.GetDim(i) == 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "dy", "0",
                                                              "Input dy shape can not be 0."),
                    return ge::GRAPH_FAILED);
    }

    size_t leadingRank = xRank - gammaRank;
    numRows = 1;
    for (size_t i = 0; i < leadingRank; ++i) {
        OP_CHECK_IF(MulShapeDim(context, numRows, dy.GetDim(i)) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(mean.GetDim(i) != dy.GetDim(i) || rstd.GetDim(i) != dy.GetDim(i),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                          "mean/rstd leading dimensions must match dy"),
                    return ge::GRAPH_FAILED);
    }

    numCols = 1;
    for (size_t i = 0; i < gammaRank; ++i) {
        int64_t gammaDim = gamma.GetDim(i);
        OP_CHECK_IF(MulShapeDim(context, numCols, gammaDim) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(dy.GetDim(leadingRank + i) != gammaDim,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                          "gamma shape must match the trailing dy dimensions"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(numRows > UINT64_MAX / numCols,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "flattened tensor element count overflows uint64"),
                return ge::GRAPH_FAILED);
    uint64_t totalElements = numRows * numCols;
    OP_CHECK_IF(totalElements > UINT64_MAX / MAX_ELEMENT_BYTES,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "flattened tensor byte span overflows uint64"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcTileLength(gert::TilingContext* context, uint64_t ubSize, uint32_t dtypeSize, uint32_t& tileLength)
{
    uint64_t bytesPerElement = DTYPE_QUEUE_COUNT * dtypeSize + FP32_BUFFER_COUNT * sizeof(float);
    uint64_t maxElements = (ubSize - UB_RESERVED_BYTES) / bytesPerElement;
    maxElements = std::min<uint64_t>(maxElements, MAX_TILE_LENGTH);
    maxElements = maxElements / VL_FP32 * VL_FP32;
    OP_CHECK_IF(maxElements < VL_FP32,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "UB cannot hold one regbase tile"),
                return ge::GRAPH_FAILED);
    tileLength = static_cast<uint32_t>(maxElements);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetTilingData(gert::TilingContext* context, uint64_t numRows, uint64_t numCols, uint64_t coreNum,
                              uint64_t ubSize, float alpha)
{
    int64_t dtypeSizeSigned = GetSizeByDataType(context->GetInputDesc(INPUT_DY)->GetDataType());
    OP_CHECK_IF(dtypeSizeSigned <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "invalid input dtype size"),
                return ge::GRAPH_FAILED);
    uint32_t dtypeSize = static_cast<uint32_t>(dtypeSizeSigned);
    uint32_t tileLength = 0;
    OP_CHECK_IF(CalcTileLength(context, ubSize, dtypeSize, tileLength) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    uint64_t rowsPerCore = CeilDiv(numRows, coreNum);
    uint64_t backwardBlockDim = CeilDiv(numRows, rowsPerCore);
    uint64_t blockElements = BLOCK_SIZE / dtypeSize;
    uint64_t colsPerCore = CeilAlign(CeilDiv(numCols, coreNum), blockElements);
    uint64_t gammaBetaBlockDim = CeilDiv(numCols, colsPerCore);
    uint64_t blockDim = std::max(backwardBlockDim, gammaBetaBlockDim);
    OP_CHECK_IF(blockDim > UINT32_MAX,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "block dim exceeds uint32"),
                return ge::GRAPH_FAILED);

    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WORKSPACE_SIZE;

    auto tiling = context->GetTilingData<DeepNormGradTilingDataArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(DeepNormGradTilingDataArch35), 0, sizeof(DeepNormGradTilingDataArch35)) != EOK,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "failed to clear tiling data"),
                return ge::GRAPH_FAILED);
    tiling->numRows = numRows;
    tiling->numCols = numCols;
    tiling->rowsPerCore = rowsPerCore;
    tiling->colsPerCore = colsPerCore;
    tiling->backwardBlockDim = static_cast<uint32_t>(backwardBlockDim);
    tiling->gammaBetaBlockDim = static_cast<uint32_t>(gammaBetaBlockDim);
    tiling->tileLength = tileLength;
    tiling->tileLengthAlign = tileLength;
    tiling->alpha = alpha;
    tiling->invCols = static_cast<float>(1.0 / static_cast<double>(numCols));

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus DeepNormGradTilingArch35(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetScheduleMode(BATCH_MODE);
    OP_CHECK_IF(CheckTensorPointers(context) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtypes(context) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    uint64_t numRows = 0;
    uint64_t numCols = 0;
    OP_CHECK_IF(CheckShapes(context, numRows, numCols) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    uint64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* alphaPtr = attrs->GetFloat(ATTR_ALPHA);
    float alpha = alphaPtr == nullptr ? 0.3f : *alphaPtr;
    return SetTilingData(context, numRows, numCols, coreNum, ubSize, alpha);
}

ge::graphStatus DeepNormGradTilingParseArch35(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<DeepNormGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DeepNormGrad)
    .Tiling(DeepNormGradTilingArch35)
    .TilingParse<DeepNormGradCompileInfo>(DeepNormGradTilingParseArch35);

} // namespace optiling
