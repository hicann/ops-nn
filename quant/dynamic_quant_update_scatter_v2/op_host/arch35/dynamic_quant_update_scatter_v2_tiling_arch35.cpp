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
 * \file dynamic_quant_update_scatter_v2_tiling_arch35.cpp
 * \brief DynamicQuantUpdateScatterV2 RegBase tiling for Ascend 950.
 */

#include "dynamic_quant_update_scatter_v2_tiling_arch35.h"

#include <algorithm>
#include <limits>
#include <securec.h>
#include <string>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {
struct DynamicQuantUpdateScatterV2CompileInfo {
    int32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
};

namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t INDICES_INDEX = 1;
constexpr size_t VAR_INDEX = 2;
constexpr size_t VAR_SCALE_INDEX = 3;
constexpr size_t VAR_OFFSET_INDEX = 4;
constexpr size_t VAR_OUT_INDEX = 0;
constexpr size_t VAR_SCALE_OUT_INDEX = 1;
constexpr size_t VAR_OFFSET_OUT_INDEX = 2;
constexpr int64_t VECTOR_LEN = 64;
constexpr int64_t WORKSPACE_BYTES = 16 * 1024 * 1024;
constexpr uint64_t TILING_KEY_REGBASE = 0;
constexpr size_t DIM_TWO = 2;
constexpr size_t INPUT_X_DIM_NUM = 3;

bool SafeMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckShapePrefix(const gert::Shape& fullShape, const gert::Shape& prefixShape)
{
    if (prefixShape.GetDimNum() > fullShape.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < prefixShape.GetDimNum(); ++i) {
        if (fullShape.GetDim(i) != prefixShape.GetDim(i)) {
            return false;
        }
    }
    return true;
}

ge::graphStatus TilingPrepareForDynamicQuantUpdateScatterV2Regbase(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "platformInfo is null"),
        return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<DynamicQuantUpdateScatterV2CompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "compileInfo is null"),
        return ge::GRAPH_FAILED);

    platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->vectorCoreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(compileInfo->vectorCoreNum <= 0 || compileInfo->ubSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform", "coreNum/ubSize",
                                                      "invalid platform coreNum or ubSize"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtype(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    auto indicesDesc = context->GetInputDesc(INDICES_INDEX);
    auto varDesc = context->GetInputDesc(VAR_INDEX);
    auto scaleDesc = context->GetInputDesc(VAR_SCALE_INDEX);
    auto offsetDesc = context->GetInputDesc(VAR_OFFSET_INDEX);
    auto varOutDesc = context->GetOutputDesc(VAR_OUT_INDEX);
    auto scaleOutDesc = context->GetOutputDesc(VAR_SCALE_OUT_INDEX);
    auto offsetOutDesc = context->GetOutputDesc(VAR_OFFSET_OUT_INDEX);
    OP_CHECK_IF(xDesc == nullptr || indicesDesc == nullptr || varDesc == nullptr || scaleDesc == nullptr ||
                    offsetDesc == nullptr || varOutDesc == nullptr || scaleOutDesc == nullptr ||
                    offsetOutDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "input or output desc is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDesc->GetDataType() != ge::DT_FLOAT16 && xDesc->GetDataType() != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x dtype", "invalid",
                                                      "x dtype should be fp16 or bf16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "indices dtype", "invalid",
                                                      "indices dtype should be int32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(varDesc->GetDataType() != ge::DT_INT4 || varOutDesc->GetDataType() != ge::DT_INT4,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "var dtype", "invalid",
                                                      "var dtype should be int4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleDesc->GetDataType() != ge::DT_FLOAT || offsetDesc->GetDataType() != ge::DT_FLOAT ||
                    scaleOutDesc->GetDataType() != ge::DT_FLOAT || offsetOutDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale/offset dtype", "invalid",
                                                      "scale and offset dtype should be fp32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus Tiling4DynamicQuantUpdateScatterV2Regbase(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "platformInfo is null"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t vectorCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(
        vectorCoreNum <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "vectorCoreNum <= 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckDtype(context) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "dtype check failed"),
        return ge::GRAPH_FAILED);

    // x shape: (..., H) ; rowLen = H ; rowNum = product of leading dims
    auto xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "x shape is null"),
        return ge::GRAPH_FAILED);
    const auto& xStorage = xShape->GetStorageShape();
    size_t xDimNum = xStorage.GetDimNum();
    OP_CHECK_IF(
        xDimNum < DIM_TWO || xDimNum > INPUT_X_DIM_NUM,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "x rank must be 2 or 3"),
        return ge::GRAPH_FAILED);
    int64_t rowLen = xStorage.GetDim(xDimNum - 1);
    OP_CHECK_IF(rowLen <= 0 || (rowLen % DIM_TWO != 0),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(rowLen).c_str(),
                                                         "last dim must be positive and even for int4"),
                return ge::GRAPH_FAILED);
    int64_t rowNum = 1;
    for (size_t i = 0; i + 1 < xDimNum; ++i) {
        OP_CHECK_IF(!SafeMul(rowNum, xStorage.GetDim(i), rowNum),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x shape", "invalid",
                                                          "x row num is negative or overflow"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(rowNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "x row num must be positive"),
                return ge::GRAPH_FAILED);
    auto indicesShape = context->GetInputShape(INDICES_INDEX);
    OP_CHECK_IF(
        indicesShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "indices shape is null"),
        return ge::GRAPH_FAILED);
    const auto& indicesStorage = indicesShape->GetStorageShape();
    OP_CHECK_IF(indicesStorage.GetDimNum() != 1 || indicesStorage.GetDim(0) != rowNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "indices", "invalid shape",
                                                      "indices must be 1D and its length should match x rows"),
                return ge::GRAPH_FAILED);

    auto varInShape = context->GetInputShape(VAR_INDEX);
    auto scaleShape = context->GetInputShape(VAR_SCALE_INDEX);
    auto offsetShape = context->GetInputShape(VAR_OFFSET_INDEX);
    OP_CHECK_IF(varInShape == nullptr || scaleShape == nullptr || offsetShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "var/scale/offset shape is null"),
                return ge::GRAPH_FAILED);
    const auto& varInStorage = varInShape->GetStorageShape();
    const auto& scaleStorage = scaleShape->GetStorageShape();
    const auto& offsetStorage = offsetShape->GetStorageShape();
    if (xDimNum == INPUT_X_DIM_NUM) {
        OP_CHECK_IF(!CheckShapePrefix(varInStorage, scaleStorage),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var_scale", "invalid shape",
                                                          "var_scale shape should match var prefix"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!CheckShapePrefix(varInStorage, offsetStorage),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var_offset", "invalid shape",
                                                          "var_offset shape should match var prefix"),
                    return ge::GRAPH_FAILED);
    }

    // var out shape: rank2 collapsed (B, H) or rank3 scatter (B, S, H)
    auto varShape = context->GetOutputShape(VAR_OUT_INDEX);
    OP_CHECK_IF(
        varShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "var out shape is null"),
        return ge::GRAPH_FAILED);
    const auto& varStorage = varShape->GetStorageShape();
    OP_CHECK_IF(varStorage.GetDimNum() != xDimNum,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "var rank should match x rank"),
                return ge::GRAPH_FAILED);
    int64_t batchSize = varStorage.GetDim(0);
    int64_t dstSeqLen = (varInStorage.GetDimNum() >= DIM_TWO) ? varInStorage.GetDim(1) : 1;
    OP_CHECK_IF(batchSize <= 0 || dstSeqLen <= 0 || varStorage.GetDim(varStorage.GetDimNum() - 1) != rowLen,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var", "invalid shape",
                                                      "var shape should be positive and match x last dim"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleStorage.GetDimNum() >= DIM_TWO && scaleStorage.GetDim(1) != dstSeqLen,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var_scale", "invalid shape",
                                                      "var_scale seq dim should match var seq dim"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(offsetStorage.GetDimNum() >= DIM_TWO && offsetStorage.GetDim(1) != dstSeqLen,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var_offset", "invalid shape",
                                                      "var_offset seq dim should match var seq dim"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rowNum != batchSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x/var", "batch mismatch",
                                                      "x rows should match var batch"),
                return ge::GRAPH_FAILED);

    // Scalar correctness path: use one AIV block to avoid missing rows on
    // release execution and to keep in-place byte writes race-free.
    int64_t coreNum = 1;
    int64_t rowPerHeadCore = rowNum;
    int64_t rowPerTailCore = rowNum;
    int64_t alignRowLen = Ops::Base::CeilAlign(rowLen, VECTOR_LEN);

    auto rawTiling = context->GetTilingData<DynamicQuantUpdateScatterV2RegbaseTilingData>();
    OP_CHECK_IF(
        rawTiling == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "GetTilingData null"),
        return ge::GRAPH_FAILED);
    (void)memset_s(rawTiling, sizeof(DynamicQuantUpdateScatterV2RegbaseTilingData), 0,
                   sizeof(DynamicQuantUpdateScatterV2RegbaseTilingData));
    rawTiling->coreNum = coreNum;
    rawTiling->rowLen = rowLen;
    rawTiling->rowPerHeadCore = rowPerHeadCore;
    rawTiling->rowPerTailCore = rowPerTailCore;
    rawTiling->batchSize = batchSize;
    rawTiling->dstSeqLen = dstSeqLen;
    rawTiling->alignRowLen = alignRowLen;
    rawTiling->outAlignLen = alignRowLen;
    int64_t varElemLen = 1;
    for (size_t i = 0; i < varInStorage.GetDimNum(); ++i) {
        OP_CHECK_IF(!SafeMul(varElemLen, varInStorage.GetDim(i), varElemLen),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "var shape", "invalid",
                                                          "var element num is negative or overflow"),
                    return ge::GRAPH_FAILED);
    }
    int64_t scaleElemLen = 1;
    for (size_t i = 0; i < scaleStorage.GetDimNum(); ++i) {
        OP_CHECK_IF(!SafeMul(scaleElemLen, scaleStorage.GetDim(i), scaleElemLen),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "var_scale shape", "invalid",
                                                          "var_scale element num is negative or overflow"),
                    return ge::GRAPH_FAILED);
    }
    int64_t offsetElemLen = 1;
    for (size_t i = 0; i < offsetStorage.GetDimNum(); ++i) {
        OP_CHECK_IF(!SafeMul(offsetElemLen, offsetStorage.GetDim(i), offsetElemLen),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "var_offset shape", "invalid",
                                                          "var_offset element num is negative or overflow"),
                    return ge::GRAPH_FAILED);
    }
    rawTiling->varByteLen = Ops::Base::CeilDiv(varElemLen, static_cast<int64_t>(DIM_TWO));
    rawTiling->scaleLen = scaleElemLen;
    rawTiling->offsetLen = offsetElemLen;

    auto workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(
        workspaces == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "workspaces null"),
        return ge::GRAPH_FAILED);
    workspaces[0] = WORKSPACE_BYTES;
    context->SetBlockDim(coreNum);
    context->SetTilingKey(TILING_KEY_REGBASE);
    OP_LOGD(context->GetNodeName(),
            "V2 RegBase tiling: cores=%ld rowLen=%ld rowPerHeadCore=%ld rowPerTailCore=%ld B=%ld S=%ld", coreNum,
            rowLen, rowPerHeadCore, rowPerTailCore, batchSize, dstSeqLen);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicQuantUpdateScatterV2)
    .Tiling(Tiling4DynamicQuantUpdateScatterV2Regbase)
    .TilingParse<DynamicQuantUpdateScatterV2CompileInfo>(TilingPrepareForDynamicQuantUpdateScatterV2Regbase);
} // namespace optiling
