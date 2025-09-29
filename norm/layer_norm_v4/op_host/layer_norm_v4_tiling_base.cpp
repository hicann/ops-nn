/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layer_norm_v4_tiling_base.cpp
 * \brief
 */

#include "layer_norm_v4_tiling.h"
#include "exe_graph/runtime/shape.h"

namespace optiling {
namespace {
constexpr size_t V3_INPUT_IDX_X = 0;
constexpr size_t V3_INPUT_IDX_GAMMA = 1;
constexpr size_t V3_INPUT_IDX_BETA = 2;
constexpr float DEFAULT_EPSILON = 1e-5;

const gert::Shape g_vec_1_shape = {1};

const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

struct LayerNormV3OpInfo {
    LayerNormV4CompileInfo regbaseCompileInfo;
};

bool isIndexValid(const gert::Shape& xShape, int64_t beginAxis)
{
    int64_t dimNum = static_cast<int64_t>(xShape.GetDimNum());
    return (beginAxis >= 0 && beginAxis < dimNum) || (beginAxis < 0 && -beginAxis <= dimNum);
}

bool isFloatDtype(ge::DataType dtype)
{
    static const std::unordered_set<ge::DataType> floatDtypes = {
        ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_BF16};
    return floatDtypes.find(dtype) != floatDtypes.end();
}

ge::graphStatus GetV3PlatformInfo(gert::TilingContext* context, ParamsLayerNomrV4& commonParams)
{
    auto v3CompileInfo = context->GetCompileInfo<LayerNormV3OpInfo>();
    OP_CHECK_IF(
        v3CompileInfo == nullptr, OP_LOGE(context->GetNodeName(), "layer_norm_v3 compile info is null"),
        return ge::GRAPH_FAILED);
    const LayerNormV4CompileInfo* compileInfo =
        static_cast<const LayerNormV4CompileInfo*>(&v3CompileInfo->regbaseCompileInfo);
    return GetCommonPlatformInfo(context, compileInfo, commonParams);
}

ge::graphStatus InputShapeAndAxisCheck(
    gert::TilingContext* context, const gert::Shape& xShape, const gert::Shape& gammaShape,
    const gert::Shape& betaShape, int64_t& beginNormAxis, int64_t& beginParamsAxis)
{
    OP_CHECK_IF(
        xShape.GetDimNum() < gammaShape.GetDimNum(),
        OP_LOGE(
            context->GetNodeName(), "gamma dim num must be less than x dim num, x dim num: %u, gamma dim num: %u",
            static_cast<uint32_t>(xShape.GetDimNum()), static_cast<uint32_t>(gammaShape.GetDimNum())),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        gammaShape != betaShape, OP_LOGE(context->GetNodeName(), "gamma shape must be equal to beta shape."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !isIndexValid(xShape, beginNormAxis), OP_LOGE(context->GetNodeName(), "begin_norm_axis is invalid."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !isIndexValid(xShape, beginParamsAxis), OP_LOGE(context->GetNodeName(), "begin_params_axis is invalid."),
        return ge::GRAPH_FAILED);

    beginNormAxis = beginNormAxis < 0 ? beginNormAxis + static_cast<int64_t>(xShape.GetDimNum()) : beginNormAxis;

    beginParamsAxis =
        beginParamsAxis < 0 ? beginParamsAxis + static_cast<int64_t>(xShape.GetDimNum()) : beginParamsAxis;

    OP_CHECK_IF(
        beginNormAxis != beginParamsAxis,
        OP_LOGE(
            context->GetNodeName(), "begin_norm_axis: %ld must be same as begin_params_axis: %ld.", beginNormAxis,
            beginParamsAxis),
        return ge::GRAPH_FAILED);

    for (size_t index = 0; index < gammaShape.GetDimNum(); index++) {
        int64_t reduceAxis = index + beginNormAxis;
        OP_CHECK_IF(
            !isIndexValid(xShape, reduceAxis), OP_LOGE(context->GetNodeName(), "begin_norm_axis is invalid."),
            return ge::GRAPH_FAILED);
        int64_t inputDim = xShape.GetDim(reduceAxis);
        int64_t normDim = gammaShape.GetDim(index);
        OP_CHECK_IF(
            normDim != inputDim,
            OP_LOGE(
                context->GetNodeName(),
                "expected gamma index [%zu] shape [%ld] be equal to x index [%ld] shape [%ld], but failed.", index,
                normDim, reduceAxis, inputDim),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InputDtypeCheck(
    gert::TilingContext* context, ge::DataType xDtype, ge::DataType gammaDtype, ge::DataType betaDtype)
{
    OP_CHECK_IF(
        !isFloatDtype(xDtype), OP_LOGE(context->GetNodeName(), "x dtype must be in float32, float16, bfloat16."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        gammaDtype != betaDtype, OP_LOGE(context->GetNodeName(), "gamma dtype must be the same as beta dtype."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (gammaDtype != xDtype) && (gammaDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(context->GetNodeName(), "when gamma dtype is not same as x dtype, gamma dtype must be float32."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetV3ShapeAttrsInfo(gert::TilingContext* context, ParamsLayerNomrV4& commonParams)
{
    auto xDesc = context->GetInputDesc(V3_INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto gammaDesc = context->GetInputDesc(V3_INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDesc);
    auto betaDesc = context->GetInputDesc(V3_INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);

    ge::DataType xDtype = xDesc->GetDataType();
    ge::DataType gammaDtype = gammaDesc->GetDataType();
    ge::DataType betaDtype = betaDesc->GetDataType();

    OP_CHECK_IF(
        InputDtypeCheck(context, xDtype, gammaDtype, betaDtype) == ge::GRAPH_FAILED,
        OP_LOGE(context->GetNodeName(), "input dtype check failed."), return ge::GRAPH_FAILED);

    commonParams.tensorDtype = xDtype;
    commonParams.paramDtype = gammaDtype;
    commonParams.gammaNullPtr = 0;
    commonParams.betaNullPtr = 0;
    commonParams.dtypeKey = GetDTypeKey(commonParams.tensorDtype, commonParams.paramDtype);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* beginNormAxisPtr = attrs->GetInt(V3_INPUT_IDX_X);
    int64_t beginNormAxis = (beginNormAxisPtr == nullptr) ? 0 : *beginNormAxisPtr;
    const int64_t* beginParamsAxisPtr = attrs->GetInt(V3_INPUT_IDX_GAMMA);
    int64_t beginParamsAxis = (beginParamsAxisPtr == nullptr) ? 0 : *beginParamsAxisPtr;
    const float* epsilonPtr = attrs->GetFloat(V3_INPUT_IDX_BETA);
    commonParams.eps = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;

    const gert::Shape& xShape = EnsureNotScalar(context->GetInputShape(V3_INPUT_IDX_X)->GetStorageShape());
    const gert::Shape& gammaShape = EnsureNotScalar(context->GetInputShape(V3_INPUT_IDX_GAMMA)->GetStorageShape());
    const gert::Shape& betaShape = EnsureNotScalar(context->GetInputShape(V3_INPUT_IDX_BETA)->GetStorageShape());

    OP_CHECK_IF(
        InputShapeAndAxisCheck(context, xShape, gammaShape, betaShape, beginNormAxis, beginParamsAxis) ==
            ge::GRAPH_FAILED,
        OP_LOGE(context->GetNodeName(), "input shape or normlize axis check failed."), return ge::GRAPH_FAILED);

    // fuse axis
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        if (static_cast<int64_t>(i) < beginNormAxis) {
            colSize *= xShape.GetDim(i);
        } else {
            rowSize *= xShape.GetDim(i);
        }
    }
    return GetCommonShapeAttrsInfo(context, colSize, rowSize, commonParams);
}
} // namespace

constexpr size_t K_INPUT_IDX_X = 0;
constexpr size_t K_INPUT_IDX_NORM_SHAPE = 1;
constexpr size_t K_INPUT_IDX_GAMMA = 2;
constexpr size_t K_INPUT_IDX_BETA = 3;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t SIZE_OF_FLOAT = 4;
constexpr uint64_t SIZE_OF_HALF = 2;
constexpr uint64_t BASE_WSP_SIZE = 32;

static const std::unordered_map<ge::DataType, uint64_t> DTYPE_SIZE_MAP{
    {ge::DataType::DT_FLOAT, 4}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 2}};

int64_t GetDTypeKey(ge::DataType tensorDtype, ge::DataType paramDtype)
{
    constexpr static int64_t LN_TENSOR_KEY_WEIGHT = 10;

    auto GetKeyForDType = [](ge::DataType dtype) -> int64_t {
        switch (dtype) {
            case ge::DT_FLOAT:
                return 0;
            case ge::DT_FLOAT16:
                return 1;
            case ge::DT_BF16:
                return 2;
            default:
                return -1;
        }
    };

    int64_t tensorKey = GetKeyForDType(tensorDtype);
    int64_t paramKey = GetKeyForDType(paramDtype);

    return tensorKey * LN_TENSOR_KEY_WEIGHT + paramKey;
}

bool LayerNormV4TilingBase::IsCapable()
{
    return true;
}

ge::graphStatus LayerNormV4TilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormV4TilingBase::GetTilingKey() const
{
    return 0;
}

ge::graphStatus GetCommonPlatformInfo(
    gert::TilingContext* context, const LayerNormV4CompileInfo* compileInfo, ParamsLayerNomrV4& commonParams)
{
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context, "compile info is null"), return ge::GRAPH_FAILED);
    commonParams.coreNum = compileInfo->coreNum;
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
    commonParams.blockSize = compileInfo->blockSize;
    commonParams.isAscend310P = compileInfo->isAscend310P;
    commonParams.isRegBase = compileInfo->isRegBase;
    commonParams.vlFp32 = compileInfo->vectorLength / sizeof(float);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::GetPlatformInfo()
{
    if (commonParams.isV3) {
        return GetV3PlatformInfo(context_, commonParams);
    }
    const LayerNormV4CompileInfo* compileInfo = context_->GetCompileInfo<LayerNormV4CompileInfo>();
    return GetCommonPlatformInfo(context_, compileInfo, commonParams);
}

ge::graphStatus GetCommonShapeAttrsInfo(
    gert::TilingContext* context, uint64_t colSize, uint64_t rowSize, ParamsLayerNomrV4& commonParams)
{
    OP_CHECK_IF(
        commonParams.colSize <= 0,
        OP_LOGE(context, "colSize must be greater than 0, colSize: %u", static_cast<uint32_t>(commonParams.colSize)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        commonParams.rowSize <= 0,
        OP_LOGE(context, "rowSize must be greater than 0, rowSize: %u", static_cast<uint32_t>(commonParams.rowSize)),
        return ge::GRAPH_FAILED);

    commonParams.colSize = colSize;
    commonParams.rowSize = rowSize;
    commonParams.coefficient = static_cast<float>(1.0) / static_cast<float>(commonParams.rowSize);
    uint64_t alignment = 16;
    if (DTYPE_SIZE_MAP.find(commonParams.tensorDtype) != DTYPE_SIZE_MAP.end()) {
        alignment = BLOCK_SIZE / DTYPE_SIZE_MAP.at(commonParams.tensorDtype);
    } else {
        OP_LOGE(context, "x dtype must be in float32, float16, bfloat16.");
        return ge::GRAPH_FAILED;
    }
    commonParams.rowAlign = (commonParams.rowSize + alignment - 1) / alignment * alignment;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::GetShapeAttrsInfo()
{
    std::string opType(context_->GetNodeType());
    commonParams.isV3 = opType == "LayerNormV3";
    if (commonParams.isV3) {
        return GetV3ShapeAttrsInfo(context_, commonParams);
    }
    commonParams.tensorDtype = context_->GetInputDesc(K_INPUT_IDX_X)->GetDataType();
    commonParams.paramDtype = ge::DT_FLOAT;
    auto gammaDesc = context_->GetOptionalInputDesc(K_INPUT_IDX_GAMMA);
    auto betaDesc = context_->GetOptionalInputDesc(K_INPUT_IDX_BETA);
    if (gammaDesc != nullptr) {
        commonParams.paramDtype = gammaDesc->GetDataType();
    } else if (betaDesc != nullptr) {
        commonParams.paramDtype = betaDesc->GetDataType();
    }
    commonParams.gammaNullPtr = (gammaDesc == nullptr ? 1 : 0);
    commonParams.betaNullPtr = (betaDesc == nullptr ? 1 : 0);
    commonParams.dtypeKey = GetDTypeKey(commonParams.tensorDtype, commonParams.paramDtype);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    commonParams.eps = *(attrs->GetFloat(0));

    uint64_t normalizedShapeLen;
    const gert::Shape xShape = context_->GetInputShape(K_INPUT_IDX_X)->GetStorageShape();
    const gert::Shape normalizedShape = context_->GetInputShape(K_INPUT_IDX_NORM_SHAPE)->GetStorageShape();
    OP_CHECK_IF(
        normalizedShape.GetDimNum() > 1,
        OP_LOGE(
            context_->GetNodeName(), "normalizedShape dim num must be 1, dim num: %u",
            static_cast<uint32_t>(normalizedShape.GetDimNum())),
        return ge::GRAPH_FAILED);
    normalizedShapeLen = normalizedShape.IsScalar() ? 1 : normalizedShape.GetDim(0);
    OP_CHECK_IF(
        static_cast<uint64_t>(xShape.GetDimNum()) < normalizedShapeLen,
        OP_LOGE(
            context_->GetNodeName(),
            "normalizedShape dim num must be less than xShape dim num, xShape dim num: %u, "
            "normalizedShape dim num: %u",
            static_cast<uint32_t>(xShape.GetDimNum()), static_cast<uint32_t>(normalizedShapeLen)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        static_cast<int64_t>(normalizedShapeLen) < 0,
        OP_LOGE(
            context_->GetNodeName(), "normalizedShapeLen must be greater than 0, normalizedShapeLen: %u",
            static_cast<uint32_t>(normalizedShapeLen)),
        return ge::GRAPH_FAILED);

    // fuse axis
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        if (i < xShape.GetDimNum() - normalizedShapeLen) {
            colSize *= xShape.GetDim(i);
        } else {
            rowSize *= xShape.GetDim(i);
        }
    }
    return GetCommonShapeAttrsInfo(context_, colSize, rowSize, commonParams);
}

ge::graphStatus LayerNormV4TilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = BASE_WSP_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
