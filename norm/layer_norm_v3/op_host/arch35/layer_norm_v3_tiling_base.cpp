/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layer_norm_v3_tiling_base.cpp
 * \brief
 */

#include "layer_norm_v3_tiling.h"
#include "layer_norm_v3_tiling_arch35.h"
#include "layer_norm.h"

namespace optiling {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t INPUT_IDX_GAMMA = 1;
constexpr size_t INPUT_IDX_BETA = 2;
constexpr float DEFAULT_EPSILON_V3 = 1e-5;
constexpr uint64_t BASE_WSP_SIZE = 32;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr float DEFAULT_EPSILON_V1 = 1e-7;
const gert::Shape g_vec_1_shape = {1};

static const std::unordered_map<ge::DataType, uint64_t> LN_DTYPE_SIZE_MAP{
    {ge::DataType::DT_FLOAT, 4}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 2}};

bool LayerNormV3TilingBase::isIndexValid(const gert::Shape& xShape, int64_t beginAxis)
{
    int64_t dimNum = static_cast<int64_t>(xShape.GetDimNum());
    return (beginAxis >= 0 && beginAxis < dimNum) || (beginAxis < 0 && -beginAxis <= dimNum);
}

int64_t LayerNormV3TilingBase::GetDTypeKey(ge::DataType tensorDtype, ge::DataType paramDtype)
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

bool LayerNormV3TilingBase::isFloatDtype(ge::DataType dtype)
{
    static const std::unordered_set<ge::DataType> floatDtypes = {
        ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_BF16};
    return floatDtypes.find(dtype) != floatDtypes.end();
}

ge::graphStatus LayerNormV3TilingBase::InputShapeAndAxisCheck(
    const gert::Shape& xShape, const gert::Shape& gammaShape, const gert::Shape& betaShape, int64_t& beginNormAxis,
    int64_t& beginParamsAxis)
{
    OP_CHECK_IF(
        xShape.GetDimNum() < gammaShape.GetDimNum(),
        OP_LOGE(
            context_->GetNodeName(), "gamma dim num must be less than x dim num, x dim num: %u, gamma dim num: %u",
            static_cast<uint32_t>(xShape.GetDimNum()), static_cast<uint32_t>(gammaShape.GetDimNum())),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        gammaShape != betaShape, OP_LOGE(context_->GetNodeName(), "gamma shape must be equal to beta shape."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !isIndexValid(xShape, beginNormAxis), OP_LOGE(context_->GetNodeName(), "begin_norm_axis is invalid."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !isIndexValid(xShape, beginParamsAxis), OP_LOGE(context_->GetNodeName(), "begin_params_axis is invalid."),
        return ge::GRAPH_FAILED);

    beginNormAxis = beginNormAxis < 0 ? beginNormAxis + static_cast<int64_t>(xShape.GetDimNum()) : beginNormAxis;

    beginParamsAxis =
        beginParamsAxis < 0 ? beginParamsAxis + static_cast<int64_t>(xShape.GetDimNum()) : beginParamsAxis;

    OP_CHECK_IF(
        beginNormAxis != beginParamsAxis,
        OP_LOGE(
            context_->GetNodeName(), "begin_norm_axis: %ld must be same as begin_params_axis: %ld.", beginNormAxis,
            beginParamsAxis),
        return ge::GRAPH_FAILED);

    for (size_t index = 0; index < gammaShape.GetDimNum(); index++) {
        int64_t reduceAxis = index + beginNormAxis;
        OP_CHECK_IF(
            !isIndexValid(xShape, reduceAxis), OP_LOGE(context_->GetNodeName(), "begin_norm_axis is invalid."),
            return ge::GRAPH_FAILED);
        int64_t inputDim = xShape.GetDim(reduceAxis);
        int64_t normDim = gammaShape.GetDim(index);
        OP_CHECK_IF(
            normDim != inputDim,
            OP_LOGE(
                context_->GetNodeName(),
                "expected gamma index [%zu] shape [%ld] be equal to x index [%ld] shape [%ld], but failed.", index,
                normDim, reduceAxis, inputDim),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::InputDtypeCheck(
    ge::DataType xDtype, ge::DataType gammaDtype, ge::DataType betaDtype)
{
    OP_CHECK_IF(
        !isFloatDtype(xDtype), OP_LOGE(context_->GetNodeName(), "x dtype must be in float32, float16, bfloat16."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        gammaDtype != betaDtype, OP_LOGE(context_->GetNodeName(), "gamma dtype must be the same as beta dtype."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (gammaDtype != xDtype) && (gammaDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(context_->GetNodeName(), "when gamma dtype is not same as x dtype, gamma dtype must be float32."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus LayerNormV3TilingBase::GetShapeAttrsInfo()
{
    auto xDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto gammaDesc = context_->GetInputDesc(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    auto betaDesc = context_->GetInputDesc(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, betaDesc);

    ge::DataType xDtype = xDesc->GetDataType();
    ge::DataType gammaDtype = gammaDesc->GetDataType();
    ge::DataType betaDtype = betaDesc->GetDataType();

    OP_CHECK_IF(
        InputDtypeCheck(xDtype, gammaDtype, betaDtype) == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "input dtype check failed."), return ge::GRAPH_FAILED);

    commonParams.tensorDtype = xDtype;
    commonParams.paramDtype = gammaDtype;
    commonParams.gammaNullPtr = 0;
    commonParams.betaNullPtr = 0;
    commonParams.dtypeKey = GetDTypeKey(commonParams.tensorDtype, commonParams.paramDtype);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* beginNormAxisPtr = attrs->GetInt(INPUT_IDX_X);
    int64_t beginNormAxis = (beginNormAxisPtr == nullptr) ? 0 : *beginNormAxisPtr;
    const int64_t* beginParamsAxisPtr = attrs->GetInt(INPUT_IDX_GAMMA);
    int64_t beginParamsAxis = (beginParamsAxisPtr == nullptr) ? 0 : *beginParamsAxisPtr;
    const float* epsilonPtr = attrs->GetFloat(INPUT_IDX_BETA);

    std::string opType(context_->GetNodeType());
    commonParams.isV1 = opType == "LayerNorm";
    commonParams.eps =
        (epsilonPtr == nullptr) ? (commonParams.isV1 ? DEFAULT_EPSILON_V1 : DEFAULT_EPSILON_V3) : *epsilonPtr;

    const gert::Shape& xShape = EnsureNotScalar(context_->GetInputShape(INPUT_IDX_X)->GetStorageShape());
    const gert::Shape& gammaShape = EnsureNotScalar(context_->GetInputShape(INPUT_IDX_GAMMA)->GetStorageShape());
    const gert::Shape& betaShape = EnsureNotScalar(context_->GetInputShape(INPUT_IDX_BETA)->GetStorageShape());

    OP_CHECK_IF(
        InputShapeAndAxisCheck(xShape, gammaShape, betaShape, beginNormAxis, beginParamsAxis) == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "input shape or normlize axis check failed."), return ge::GRAPH_FAILED);

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

    OP_CHECK_IF(
        colSize <= 0,
        OP_LOGE(context_->GetNodeName(), "colSize must be greater than 0, colSize: %u", static_cast<uint32_t>(colSize)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        rowSize <= 0,
        OP_LOGE(context_->GetNodeName(), "rowSize must be greater than 0, rowSize: %u", static_cast<uint32_t>(rowSize)),
        return ge::GRAPH_FAILED);

    commonParams.colSize = colSize;
    commonParams.rowSize = rowSize;
    commonParams.coefficient = static_cast<float>(1.0) / static_cast<float>(commonParams.rowSize);
    uint64_t alignment = 16;
    if (LN_DTYPE_SIZE_MAP.find(commonParams.tensorDtype) != LN_DTYPE_SIZE_MAP.end()) {
        alignment = BLOCK_SIZE / LN_DTYPE_SIZE_MAP.at(commonParams.tensorDtype);
    } else {
        OP_LOGE(context_->GetNodeName(), "x dtype must be in float32, float16, bfloat16.");
        return ge::GRAPH_FAILED;
    }
    commonParams.rowAlign = (commonParams.rowSize + alignment - 1) / alignment * alignment;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::GetCommonPlatformInfo(const LayerNormV3CompileInfo* compileInfo)
{
    OP_CHECK_IF(
        compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "ascendc compile info is null"),
        return ge::GRAPH_FAILED);
    commonParams.coreNum = compileInfo->coreNum;
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
    commonParams.blockSize = compileInfo->blockSize;
    commonParams.isAscend310P = compileInfo->isAscend310P;
    commonParams.isRegBase = compileInfo->isRegBase;
    commonParams.vlFp32 = compileInfo->vectorLength / sizeof(float);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::GetPlatformInfo()
{
    if (commonParams.isV1) {
        auto v1CompileInfo = reinterpret_cast<const LayerNormOpInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            v1CompileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
            return ge::GRAPH_FAILED);
        const LayerNormV3CompileInfo* compileInfo =
            reinterpret_cast<const LayerNormV3CompileInfo*>(&v1CompileInfo->regbaseCompileInfo);
        return GetCommonPlatformInfo(compileInfo);
    }
    auto v3CompileInfo = reinterpret_cast<const LayerNormV3OpInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(
        v3CompileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"), return ge::GRAPH_FAILED);
    const LayerNormV3CompileInfo* compileInfo =
        reinterpret_cast<const LayerNormV3CompileInfo*>(&v3CompileInfo->regbaseCompileInfo);
    return GetCommonPlatformInfo(compileInfo);
}

bool LayerNormV3TilingBase::IsCapable()
{
    return true;
}

ge::graphStatus LayerNormV3TilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = BASE_WSP_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormV3TilingBase::GetTilingKey() const
{
    return 0;
}

} // namespace optiling
