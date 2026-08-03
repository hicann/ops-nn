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
constexpr int64_t UB_RESERVED_BYTES = 16 * 1024;
constexpr int64_t PARAM_BUFFER_BYTES = 64;
constexpr int64_t X_TYPE_BYTES = 2;
constexpr int64_t WORKSPACE_BYTES = 16 * 1024 * 1024;
constexpr uint64_t TILING_KEY_REGBASE = 0;
constexpr size_t DIM_TWO = 2;
constexpr size_t INPUT_X_DIM_NUM = 3;
constexpr size_t INPUT_VAR_DIM_NUM = 4;
constexpr size_t INPUT_PARAM_DIM_NUM = 2;
constexpr size_t OUTPUT_VAR_DIM_NUM = 3;
constexpr size_t OUTPUT_PARAM_DIM_NUM = 3;

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

bool SafeAdd(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
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
    const int64_t vectorCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(vectorCoreNum <= 0 || ubSize <= static_cast<uint64_t>(UB_RESERVED_BYTES),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform", "coreNum/ubSize",
                                                      "invalid platform coreNum or ubSize"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckDtype(context) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "dtype check failed"),
        return ge::GRAPH_FAILED);

    // The fused pattern and the A2 implementation use x=(B,1,H).
    auto xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "x shape is null"),
        return ge::GRAPH_FAILED);
    const auto& xStorage = xShape->GetStorageShape();
    OP_CHECK_IF(xStorage.GetDimNum() != INPUT_X_DIM_NUM,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x rank", "invalid", "x rank must be 3"),
                return ge::GRAPH_FAILED);
    const int64_t batchSize = xStorage.GetDim(0);
    const int64_t rowLen = xStorage.GetDim(2);
    OP_CHECK_IF(batchSize <= 0 || xStorage.GetDim(1) != 1 || rowLen <= 0 || (rowLen % DIM_TWO != 0),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(rowLen).c_str(),
                                                         "x must be (B,1,H), with positive B and positive even H"),
                return ge::GRAPH_FAILED);

    auto indicesShape = context->GetInputShape(INDICES_INDEX);
    OP_CHECK_IF(
        indicesShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "indices shape is null"),
        return ge::GRAPH_FAILED);
    const auto& indicesStorage = indicesShape->GetStorageShape();
    OP_CHECK_IF(indicesStorage.GetDimNum() != 1 || indicesStorage.GetDim(0) != batchSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "indices", "invalid shape",
                                                      "indices must be 1D and its length must equal B"),
                return ge::GRAPH_FAILED);

    auto varInShape = context->GetInputShape(VAR_INDEX);
    auto scaleShape = context->GetInputShape(VAR_SCALE_INDEX);
    auto offsetShape = context->GetInputShape(VAR_OFFSET_INDEX);
    auto varOutShape = context->GetOutputShape(VAR_OUT_INDEX);
    auto scaleOutShape = context->GetOutputShape(VAR_SCALE_OUT_INDEX);
    auto offsetOutShape = context->GetOutputShape(VAR_OFFSET_OUT_INDEX);
    OP_CHECK_IF(varInShape == nullptr || scaleShape == nullptr || offsetShape == nullptr || varOutShape == nullptr ||
                    scaleOutShape == nullptr || offsetOutShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid",
                                                      "input or output shape is null"),
                return ge::GRAPH_FAILED);
    const auto& varInStorage = varInShape->GetStorageShape();
    const auto& scaleStorage = scaleShape->GetStorageShape();
    const auto& offsetStorage = offsetShape->GetStorageShape();
    const auto& varOutStorage = varOutShape->GetStorageShape();
    const auto& scaleOutStorage = scaleOutShape->GetStorageShape();
    const auto& offsetOutStorage = offsetOutShape->GetStorageShape();
    OP_CHECK_IF(
        varInStorage.GetDimNum() != INPUT_VAR_DIM_NUM || scaleStorage.GetDimNum() != INPUT_PARAM_DIM_NUM ||
            offsetStorage.GetDimNum() != INPUT_PARAM_DIM_NUM || varOutStorage.GetDimNum() != OUTPUT_VAR_DIM_NUM ||
            scaleOutStorage.GetDimNum() != OUTPUT_PARAM_DIM_NUM || offsetOutStorage.GetDimNum() != OUTPUT_PARAM_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var/scale/offset", "invalid rank",
                                              "expected var input rank 4, parameter input rank 2, and output rank 3"),
        return ge::GRAPH_FAILED);

    const int64_t dstSeqLen = varInStorage.GetDim(1);
    OP_CHECK_IF(dstSeqLen <= 0 || varInStorage.GetDim(0) != batchSize || varInStorage.GetDim(2) != 1 ||
                    varInStorage.GetDim(3) != rowLen || scaleStorage.GetDim(0) != batchSize ||
                    scaleStorage.GetDim(1) != dstSeqLen || offsetStorage.GetDim(0) != batchSize ||
                    offsetStorage.GetDim(1) != dstSeqLen || varOutStorage.GetDim(0) != batchSize ||
                    varOutStorage.GetDim(1) != dstSeqLen || varOutStorage.GetDim(2) != rowLen ||
                    scaleOutStorage.GetDim(0) != 1 || scaleOutStorage.GetDim(1) != batchSize ||
                    scaleOutStorage.GetDim(2) != dstSeqLen || offsetOutStorage.GetDim(0) != 1 ||
                    offsetOutStorage.GetDim(1) != batchSize || offsetOutStorage.GetDim(2) != dstSeqLen,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "input/output", "shape mismatch",
                                                      "expected var=(B,S,1,H), params=(B,S), outputs=(B,S,H)/(1,B,S)"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        rowLen > std::numeric_limits<int64_t>::max() - (VECTOR_LEN - 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "H", "overflow", "H cannot be aligned safely"),
        return ge::GRAPH_FAILED);
    const int64_t alignRowLen = Ops::Base::CeilAlign(rowLen, VECTOR_LEN);
    int64_t xBufferBytes = 0;
    int64_t totalBufferBytes = 0;
    OP_CHECK_IF(!SafeMul(alignRowLen, X_TYPE_BYTES, xBufferBytes) ||
                    !SafeAdd(xBufferBytes, alignRowLen / static_cast<int64_t>(DIM_TWO), totalBufferBytes) ||
                    !SafeAdd(totalBufferBytes, PARAM_BUFFER_BYTES, totalBufferBytes) ||
                    static_cast<uint64_t>(totalBufferBytes) > ubSize - static_cast<uint64_t>(UB_RESERVED_BYTES),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "H/UB", "out of range",
                                                      "aligned input, packed output, and parameter buffers exceed UB"),
                return ge::GRAPH_FAILED);

    const int64_t rowPerHeadCore = Ops::Base::CeilDiv(batchSize, vectorCoreNum);
    const int64_t coreNum = Ops::Base::CeilDiv(batchSize, rowPerHeadCore);
    const int64_t rowPerTailCore = batchSize - rowPerHeadCore * (coreNum - 1);

    int64_t scaleElemLen = 0;
    int64_t varElemLen = 0;
    OP_CHECK_IF(!SafeMul(batchSize, dstSeqLen, scaleElemLen) || !SafeMul(scaleElemLen, rowLen, varElemLen),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "var shape", "overflow",
                                                      "B*S*H exceeds int64 range"),
                return ge::GRAPH_FAILED);

    auto rawTiling = context->GetTilingData<DynamicQuantUpdateScatterV2RegbaseTilingData>();
    OP_CHECK_IF(
        rawTiling == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameter", "invalid", "GetTilingData null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(memset_s(rawTiling, sizeof(DynamicQuantUpdateScatterV2RegbaseTilingData), 0,
                         sizeof(DynamicQuantUpdateScatterV2RegbaseTilingData)) != EOK,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tiling", "memset failed",
                                                      "failed to initialize tiling data"),
                return ge::GRAPH_FAILED);
    rawTiling->coreNum = coreNum;
    rawTiling->rowLen = rowLen;
    rawTiling->rowPerHeadCore = rowPerHeadCore;
    rawTiling->rowPerTailCore = rowPerTailCore;
    rawTiling->batchSize = batchSize;
    rawTiling->dstSeqLen = dstSeqLen;
    rawTiling->alignRowLen = alignRowLen;
    rawTiling->outAlignLen = alignRowLen / static_cast<int64_t>(DIM_TWO);
    rawTiling->varByteLen = varElemLen / static_cast<int64_t>(DIM_TWO);
    rawTiling->scaleLen = scaleElemLen;
    rawTiling->offsetLen = scaleElemLen;

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
