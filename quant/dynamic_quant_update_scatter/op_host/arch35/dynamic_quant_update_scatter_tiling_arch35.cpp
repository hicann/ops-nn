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
 * \file dynamic_quant_update_scatter_tiling_arch35.cpp
 * \brief DynamicQuantUpdateScatter RegBase tiling for Ascend 950.
 */

#include "dynamic_quant_update_scatter_tiling_arch35.h"

#include <algorithm>
#include <limits>
#include <string>

#include <securec.h>
#include "error_util.h"
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {
struct DynamicQuantUpdateScatterCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};

namespace {
using Ops::NN::OpTiling::EnsureNotScalar;

constexpr size_t INDEX_VAR = 0;
constexpr size_t INDEX_VAR_SCALE = 1;
constexpr size_t INDEX_INDICES = 2;
constexpr size_t INDEX_UPDATES = 3;
constexpr size_t INDEX_SMOOTH_SCALES = 4;
constexpr size_t ATTR_REDUCE = 0;
constexpr size_t ATTR_AXIS = 1;

constexpr int64_t INDICES_RANK_ONE = 1;
constexpr int64_t INDICES_RANK_TWO = 2;
constexpr int64_t BYTES_PER_BLOCK = 32;
constexpr int64_t VECTOR_LENGTH_FP32 = 64;
constexpr int64_t UB_RESERVED_BYTES = 16 * 1024;
constexpr int64_t WORKSPACE_BYTES = 32 * 1024 * 1024;
constexpr uint64_t TILING_KEY_REGBASE_NO_SMOOTH = 0;
constexpr uint64_t TILING_KEY_REGBASE_WITH_SMOOTH = 1;

struct Shape4D {
    int64_t batch = 0;
    int64_t head = 0;
    int64_t axis = 0;
    int64_t tail = 0;
};

struct TilingInputs {
    gert::Shape varShape;
    gert::Shape varScaleShape;
    gert::Shape indicesShape;
    gert::Shape updatesShape;
    gert::Shape smoothScalesShape;
    Shape4D var4d;
    Shape4D updates4d;
    int64_t varElements = 0;
    int64_t varScaleElements = 0;
    int64_t indicesElements = 0;
    int64_t updatesElements = 0;
    int64_t smoothScalesElements = 0;
    int64_t originalLastDim = 0;
    int64_t indicesRank = 0;
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t varDtypeSize = 0;
    int64_t updateDtypeSize = 0;
    bool hasSmoothScales = false;
};

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

bool ShapeSize(const gert::Shape& shape, int64_t& size)
{
    size = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0 || !SafeMul(size, dim, size)) {
            return false;
        }
    }
    return true;
}

bool MergeTo4D(const gert::Shape& shape, int64_t axis, Shape4D& merged)
{
    const int64_t rank = static_cast<int64_t>(shape.GetDimNum());
    if (rank < 3 || axis <= 0 || axis >= rank - 1) {
        return false;
    }

    merged.batch = shape.GetDim(0);
    merged.head = 1;
    merged.axis = shape.GetDim(static_cast<size_t>(axis));
    merged.tail = 1;
    if (merged.batch <= 0 || merged.axis <= 0) {
        return false;
    }

    for (int64_t i = 1; i < axis; ++i) {
        if (!SafeMul(merged.head, shape.GetDim(static_cast<size_t>(i)), merged.head)) {
            return false;
        }
    }
    for (int64_t i = axis + 1; i < rank; ++i) {
        if (!SafeMul(merged.tail, shape.GetDim(static_cast<size_t>(i)), merged.tail)) {
            return false;
        }
    }
    return merged.head > 0 && merged.tail > 0;
}

ge::graphStatus GetPlatformInfo(const gert::TilingContext* context, TilingInputs& inputs)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    inputs.coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    inputs.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(inputs.coreNum <= 0 || inputs.ubSize <= UB_RESERVED_BYTES,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform", "coreNum/ubSize",
                                                      "invalid platform coreNum or ubSize"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAndCheckAttributes(const gert::TilingContext* context, int64_t rank, int64_t& axis)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* reduce = attrs->GetAttrPointer<char>(ATTR_REDUCE);
    OP_CHECK_NULL_WITH_CONTEXT(context, reduce);
    const std::string reduceValue(reduce);
    OP_CHECK_IF(reduceValue != "update" && reduceValue != "none" && !reduceValue.empty(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "reduce", reduceValue.c_str(),
                                                      "reduce value is not supported"),
                return ge::GRAPH_FAILED);

    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisPtr);
    axis = *axisPtr < 0 ? rank + *axisPtr : *axisPtr;
    OP_CHECK_IF(axis <= 0 || axis >= rank - 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis", std::to_string(*axisPtr).c_str(),
                                                      "axis must be an inner dimension"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetShapesAndDtypes(const gert::TilingContext* context, TilingInputs& inputs, int64_t& axis)
{
    auto varShape = context->GetInputShape(INDEX_VAR);
    auto varScaleShape = context->GetInputShape(INDEX_VAR_SCALE);
    auto indicesShape = context->GetInputShape(INDEX_INDICES);
    auto updatesShape = context->GetInputShape(INDEX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, varScaleShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesShape);

    inputs.varShape = EnsureNotScalar(varShape->GetOriginShape());
    inputs.varScaleShape = EnsureNotScalar(varScaleShape->GetOriginShape());
    inputs.indicesShape = EnsureNotScalar(indicesShape->GetOriginShape());
    inputs.updatesShape = EnsureNotScalar(updatesShape->GetOriginShape());
    const int64_t rank = static_cast<int64_t>(inputs.varShape.GetDimNum());
    OP_CHECK_IF(rank != static_cast<int64_t>(inputs.updatesShape.GetDimNum()),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var/updates", "different rank",
                                                      "var and updates must have the same rank"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetAndCheckAttributes(context, rank, axis) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "attributes", "attribute validation failed"),
        return ge::GRAPH_FAILED);

    inputs.indicesRank = static_cast<int64_t>(inputs.indicesShape.GetDimNum());
    OP_CHECK_IF(inputs.indicesRank != INDICES_RANK_ONE && inputs.indicesRank != INDICES_RANK_TWO,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "indices", "rank != 1 or 2",
                                                      "indices rank must be 1 or 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputs.indicesRank == INDICES_RANK_TWO && inputs.indicesShape.GetDim(1) != INDICES_RANK_TWO,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "indices", "last dim != 2",
                                                      "the last dimension of rank-2 indices must be 2"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!ShapeSize(inputs.varShape, inputs.varElements) ||
                    !ShapeSize(inputs.varScaleShape, inputs.varScaleElements) ||
                    !ShapeSize(inputs.indicesShape, inputs.indicesElements) ||
                    !ShapeSize(inputs.updatesShape, inputs.updatesElements),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "input", "non-positive or overflow",
                                                      "input shapes must be positive and must not overflow int64"),
                return ge::GRAPH_FAILED);

    inputs.originalLastDim = inputs.updatesShape.GetDim(inputs.updatesShape.GetDimNum() - 1);
    auto smoothShape = context->GetOptionalInputShape(INDEX_SMOOTH_SCALES);
    if (smoothShape != nullptr) {
        inputs.smoothScalesShape = EnsureNotScalar(smoothShape->GetOriginShape());
        OP_CHECK_IF(!ShapeSize(inputs.smoothScalesShape, inputs.smoothScalesElements),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "smooth_scales", "invalid shape",
                                                          "smooth_scales shape is invalid"),
                    return ge::GRAPH_FAILED);
        inputs.hasSmoothScales = true;
    }

    auto varDesc = context->GetInputDesc(INDEX_VAR);
    auto varScaleDesc = context->GetInputDesc(INDEX_VAR_SCALE);
    auto indicesDesc = context->GetInputDesc(INDEX_INDICES);
    auto updatesDesc = context->GetInputDesc(INDEX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, varScaleDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesDesc);
    const ge::DataType varDtype = varDesc->GetDataType();
    const ge::DataType varScaleDtype = varScaleDesc->GetDataType();
    const ge::DataType indicesDtype = indicesDesc->GetDataType();
    const ge::DataType updatesDtype = updatesDesc->GetDataType();
    OP_CHECK_IF(varDtype != ge::DT_INT8 || varScaleDtype != ge::DT_FLOAT ||
                    (indicesDtype != ge::DT_INT32 && indicesDtype != ge::DT_INT64) ||
                    (updatesDtype != ge::DT_FLOAT16 && updatesDtype != ge::DT_BF16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "input", "unsupported dtype combination",
                                                      "input dtype combination is not supported"),
                return ge::GRAPH_FAILED);
    inputs.varDtypeSize = ge::GetSizeByDataType(varDtype);
    inputs.updateDtypeSize = ge::GetSizeByDataType(updatesDtype);
    OP_CHECK_IF(inputs.varDtypeSize == 0 || inputs.updateDtypeSize == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "input", "dtype size is zero",
                                                      "input dtype size must not be zero"),
                return ge::GRAPH_FAILED);

    if (inputs.hasSmoothScales) {
        auto smoothDesc = context->GetOptionalInputDesc(INDEX_SMOOTH_SCALES);
        OP_CHECK_NULL_WITH_CONTEXT(context, smoothDesc);
        OP_CHECK_IF(smoothDesc->GetDataType() != updatesDtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "smooth_scales", "dtype mismatch",
                                                          "smooth_scales dtype must match updates dtype"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeRelations(const gert::TilingContext* context, TilingInputs& inputs, int64_t axis)
{
    const size_t rank = inputs.varShape.GetDimNum();
    OP_CHECK_IF(inputs.varScaleShape.GetDimNum() != rank,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var_scale", "different rank than var",
                                                      "var_scale must have the same rank as var"),
                return ge::GRAPH_FAILED);
    for (size_t dim = 0; dim + 1 < rank; ++dim) {
        OP_CHECK_IF(
            inputs.varScaleShape.GetDim(dim) != inputs.varShape.GetDim(dim),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var_scale", "mismatch before last dim",
                                                  "var_scale must match var before the last dimension"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(inputs.varScaleShape.GetDim(rank - 1) != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var_scale", "last dim != 1",
                                                      "the last dimension of var_scale must be 1"),
                return ge::GRAPH_FAILED);

    for (size_t dim = 1; dim < rank; ++dim) {
        if (static_cast<int64_t>(dim) == axis) {
            continue;
        }
        OP_CHECK_IF(inputs.updatesShape.GetDim(dim) != inputs.varShape.GetDim(dim),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "updates", "non-scatter dim mismatch",
                                                          "updates must match var on every non-scatter dimension"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        !MergeTo4D(inputs.varShape, axis, inputs.var4d) || !MergeTo4D(inputs.updatesShape, axis, inputs.updates4d),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "shape", "merge failed",
                                              "failed to merge input shapes around axis"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputs.updates4d.batch != inputs.indicesShape.GetDim(0) || inputs.updates4d.batch > inputs.var4d.batch,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "updates/indices", "batch mismatch",
                                                      "updates batch must match indices and must not exceed var batch"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputs.updates4d.head != inputs.var4d.head || inputs.updates4d.tail != inputs.var4d.tail,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "updates", "non-scatter dims mismatch",
                                                      "updates non-scatter dimensions must match var"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputs.updates4d.axis > inputs.var4d.axis,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "updates", "scatter-axis exceeds var",
                                              "updates scatter-axis size must not exceed var scatter-axis size"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputs.originalLastDim <= 0 || inputs.updates4d.tail % inputs.originalLastDim != 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "merged tail", "not divisible",
                                                      "merged tail must be divisible by the original last dimension"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputs.var4d.tail % BYTES_PER_BLOCK != 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "merged tail", "not 32B aligned",
                                                      "the merged int8 tail must be 32-byte aligned"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputs.varElements % inputs.originalLastDim != 0 ||
                    inputs.varScaleElements != inputs.varElements / inputs.originalLastDim,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "var_scale", "elements mismatch",
                                                      "var_scale shape must contain one scale per quantized row"),
                return ge::GRAPH_FAILED);
    if (inputs.hasSmoothScales) {
        OP_CHECK_IF(inputs.smoothScalesShape.GetDimNum() != 1 || inputs.smoothScalesElements != inputs.originalLastDim,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "smooth_scales", "invalid rank or size",
                        "smooth_scales must be rank 1 with one value per original last dimension"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillTilingData(gert::TilingContext* context, const TilingInputs& inputs)
{
    const int64_t quantReptNum = inputs.updates4d.tail / inputs.originalLastDim;
    const int64_t totalSegments = inputs.updatesElements / inputs.originalLastDim;
    OP_CHECK_IF(quantReptNum <= 0 || totalSegments <= 0 || totalSegments % quantReptNum != 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "quantized segment", "invalid layout",
                                                      "invalid quantized segment layout"),
                return ge::GRAPH_FAILED);
    const int64_t totalGroups = totalSegments / quantReptNum;

    const int64_t eachCoreGroups = Ops::Base::CeilDiv(totalGroups, inputs.coreNum);
    const int64_t usedCoreNum = Ops::Base::CeilDiv(totalGroups, eachCoreGroups);
    const int64_t lastCoreGroups = totalGroups - eachCoreGroups * (usedCoreNum - 1);

    const int64_t ubBytesPerElement = inputs.updateDtypeSize + static_cast<int64_t>(sizeof(int8_t)) +
                                      (inputs.hasSmoothScales ? inputs.updateDtypeSize : 0);
    int64_t tileElements = (inputs.ubSize - UB_RESERVED_BYTES) / ubBytesPerElement;
    tileElements = tileElements / VECTOR_LENGTH_FP32 * VECTOR_LENGTH_FP32;
    if (tileElements == 0) {
        tileElements = std::min(inputs.originalLastDim, VECTOR_LENGTH_FP32);
    }
    tileElements = std::min(tileElements, inputs.originalLastDim);
    tileElements = Ops::Base::CeilAlign(tileElements, static_cast<int64_t>(VECTOR_LENGTH_FP32));
    OP_CHECK_IF(tileElements <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tileElements", "<=0",
                                                      "UB cannot hold one vector tile"),
                return ge::GRAPH_FAILED);

    auto tiling = context->GetTilingData<DynamicQuantUpdateScatterRegbaseTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(*tiling), 0, sizeof(*tiling)) != EOK,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "tiling", "failed to initialize tiling data"),
        return ge::GRAPH_FAILED);
    tiling->coreNum = usedCoreNum;
    tiling->eachCoreBsNum = eachCoreGroups;
    tiling->lastCoreBsNum = lastCoreGroups;
    tiling->updateAxisShape = inputs.updates4d.axis;
    tiling->srcBsStride = inputs.updates4d.axis * inputs.updates4d.tail;
    tiling->dstBsStride = inputs.var4d.axis * inputs.var4d.tail;
    tiling->indexElements = inputs.indicesElements;
    tiling->numHead = inputs.var4d.head;
    tiling->sizePerHead = inputs.var4d.tail;
    tiling->dataAxisShape = inputs.var4d.axis;
    tiling->numOneBlock = BYTES_PER_BLOCK / inputs.varDtypeSize;
    tiling->innerLoopEle = tileElements;
    tiling->indicesShapeRank = inputs.indicesRank;
    tiling->srcFirBsStride = inputs.updates4d.head * inputs.updates4d.axis * inputs.updates4d.tail;
    tiling->dstFirSecBsStride = inputs.var4d.head * inputs.var4d.axis * inputs.var4d.tail;
    tiling->updateDim0 = inputs.updates4d.batch;
    tiling->updateDim1 = inputs.updates4d.head;
    tiling->varElements = inputs.varElements;
    tiling->varScalesElements = inputs.varScaleElements;
    tiling->updatesElements = inputs.updatesElements;
    tiling->quantReptNum = quantReptNum;
    tiling->varOrigLastDimSize = inputs.originalLastDim;
    tiling->sizeSrcPerHead = inputs.updates4d.tail;
    tiling->innerLoopTimes = Ops::Base::CeilDiv(inputs.originalLastDim, tileElements);
    tiling->innerLoopTail = inputs.originalLastDim % tileElements;

    auto workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_BYTES;
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(inputs.hasSmoothScales ? TILING_KEY_REGBASE_WITH_SMOOTH : TILING_KEY_REGBASE_NO_SMOOTH);
    OP_LOGD(context->GetNodeName(),
            "RegBase tiling: cores=%ld groups/core=%ld last=%ld groups=%ld segments=%ld row=%ld tile=%ld "
            "B/H/U/Q=%ld/%ld/%ld/%ld",
            usedCoreNum, eachCoreGroups, lastCoreGroups, totalGroups, totalSegments, inputs.originalLastDim,
            tileElements, inputs.updates4d.batch, inputs.updates4d.head, inputs.updates4d.axis, quantReptNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDynamicQuantUpdateScatterRegbase(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto compileInfo = context->GetCompiledInfo<DynamicQuantUpdateScatterCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    compileInfo->isRegbase = true;
    OP_CHECK_IF(compileInfo->coreNum <= 0 || compileInfo->ubSize <= UB_RESERVED_BYTES,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform", "coreNum/ubSize",
                                                      "invalid platform coreNum or ubSize"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus Tiling4DynamicQuantUpdateScatterRegbase(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    TilingInputs inputs;
    int64_t axis = 0;
    OP_CHECK_IF(GetPlatformInfo(context, inputs) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform",
                                                         "failed to get AscendC platform information"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapesAndDtypes(context, inputs, axis) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "input validation failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShapeRelations(context, inputs, axis) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "shape", "shape relation validation failed"),
        return ge::GRAPH_FAILED);
    return FillTilingData(context, inputs);
}

IMPL_OP_OPTILING(DynamicQuantUpdateScatter)
    .Tiling(Tiling4DynamicQuantUpdateScatterRegbase)
    .TilingParse<DynamicQuantUpdateScatterCompileInfo>(TilingPrepareForDynamicQuantUpdateScatterRegbase);
} // namespace optiling
