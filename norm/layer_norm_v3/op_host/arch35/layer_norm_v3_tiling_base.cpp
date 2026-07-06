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
#include "norm/layer_norm/op_host/arch35/layer_norm_tiling_arch35.h"

using namespace Ops::Base;

namespace optiling {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t INPUT_IDX_GAMMA = 1;
constexpr size_t INPUT_IDX_BETA = 2;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_MEAN = 1;
constexpr size_t OUTPUT_IDX_RSTD = 2;
constexpr float DEFAULT_EPSILON_V3 = 1e-5;
constexpr uint64_t BASE_WSP_SIZE = 32;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr float DEFAULT_EPSILON_V1 = 1e-7;
const gert::Shape g_vec_1_shape = {1};

static const std::unordered_map<ge::DataType, uint64_t> LN_DTYPE_SIZE_MAP{
    {ge::DataType::DT_FLOAT, 4}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 2}};

bool LayerNormV3TilingBase::isIndexValid(const gert::Shape& xShape, int64_t beginAxis) const
{
    int64_t dimNum = static_cast<int64_t>(xShape.GetDimNum());
    return (beginAxis >= 0 && beginAxis < dimNum) || (beginAxis < 0 && -beginAxis <= dimNum);
}

int64_t LayerNormV3TilingBase::GetDTypeKey(ge::DataType tensorDtype, ge::DataType paramDtype) const
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

bool LayerNormV3TilingBase::isFloatDtype(ge::DataType dtype) const
{
    static const std::unordered_set<ge::DataType> floatDtypes = {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT,
                                                                 ge::DataType::DT_BF16};
    return floatDtypes.find(dtype) != floatDtypes.end();
}

ge::graphStatus LayerNormV3TilingBase::InputShapeAndAxisCheck(const gert::Shape& xShape, const gert::Shape& gammaShape,
                                                              const gert::Shape& betaShape, int64_t& beginNormAxis,
                                                              int64_t& beginParamsAxis)
{
    if (xShape.GetDimNum() < gammaShape.GetDimNum()) {
        std::string dimsMsg = std::to_string(gammaShape.GetDimNum()) + " and " + std::to_string(xShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "gamma and x", dimsMsg.c_str(),
            "The shape dim of input gamma must be less than or equal to that of input x");
        return ge::GRAPH_FAILED;
    }

    if (gammaShape != betaShape) {
        std::string shapeMsg = ToString(betaShape) + " and " + ToString(gammaShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "beta and gamma", shapeMsg.c_str(),
                                               "The shapes of input beta and input gamma must be the same");
        return ge::GRAPH_FAILED;
    }

    if (!isIndexValid(xShape, beginNormAxis)) {
        std::string reasonMsg = "The value of attribute begin_norm_axis depends on the number of shape axes of input x";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "begin_norm_axis",
                                              std::to_string(beginNormAxis).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (!isIndexValid(xShape, beginParamsAxis)) {
        std::string
            reasonMsg = "The value of attribute begin_params_axis depends on the number of shape axes of input x";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "begin_params_axis",
                                              std::to_string(beginParamsAxis).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    beginNormAxis = beginNormAxis < 0 ? beginNormAxis + static_cast<int64_t>(xShape.GetDimNum()) : beginNormAxis;

    beginParamsAxis = beginParamsAxis < 0 ? beginParamsAxis + static_cast<int64_t>(xShape.GetDimNum()) :
                                            beginParamsAxis;

    for (size_t index = 0; index < gammaShape.GetDimNum(); index++) {
        int64_t paramsAxis = index + beginParamsAxis;
        if (!isIndexValid(xShape, paramsAxis)) {
            std::string
                reasonMsg = "The value of attribute begin_params_axis depends on the number of shape axes of input x";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "begin_params_axis",
                                                  std::to_string(beginParamsAxis).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        int64_t inputDim = xShape.GetDim(paramsAxis);
        int64_t gammaDim = gammaShape.GetDim(index);
        if (gammaDim != inputDim) {
            std::string shapeMsg = ToString(gammaShape) + " and " + ToString(xShape);
            std::string reasonMsg = "The shape of input gamma must be the same as the shape consisting of " +
                                    std::to_string(gammaShape.GetDimNum()) +
                                    " axes starting from the begin_params_axis of input x, "
                                    "where begin_params_axis refers to the attribute begin_params_axis";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "gamma and x", shapeMsg.c_str(),
                                                   reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::InputDtypeCheck(ge::DataType xDtype, ge::DataType gammaDtype,
                                                       ge::DataType betaDtype)
{
    OP_CHECK_IF(
        !isFloatDtype(xDtype),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", ToString(xDtype).c_str(), "FLOAT, FLOAT16 or BF16"),
        return ge::GRAPH_FAILED);

    if (gammaDtype != betaDtype) {
        std::string dtypeMsg = ToString(gammaDtype) + " and " + ToString(betaDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "gamma and beta", dtypeMsg.c_str(),
                                               "The dtypes of input gamma and input beta must be the same");
        return ge::GRAPH_FAILED;
    }

    if ((gammaDtype != xDtype) && (gammaDtype != ge::DataType::DT_FLOAT)) {
        std::string dtypeMsg = ToString(gammaDtype);
        std::string reasonMsg = "The dtype of input gamma must be FLOAT or the same as the dtype {" + ToString(xDtype) +
                                "} of input x";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "gamma", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus LayerNormV3TilingBase::OutputShapeCheck(const gert::Shape& xShape, int64_t beginNormAxis)
{
    beginNormAxis = beginNormAxis < 0 ? beginNormAxis + static_cast<int64_t>(xShape.GetDimNum()) : beginNormAxis;
    auto yShapePtr = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    const gert::Shape& yShape = EnsureNotScalar(yShapePtr->GetStorageShape());
    if (yShape != xShape) {
        std::string shapeMsg = ToString(yShape) + " and " + ToString(xShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "y and x", shapeMsg.c_str(),
                                               "The shapes of output y and input x must be the same");
        return ge::GRAPH_FAILED;
    }

    gert::Shape expectedShape;
    expectedShape.SetDimNum(xShape.GetDimNum());
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        expectedShape.SetDim(i, static_cast<int64_t>(i) < beginNormAxis ? xShape.GetDim(i) : 1);
    }

    auto meanShapePtr = context_->GetOutputShape(OUTPUT_IDX_MEAN);
    if (meanShapePtr != nullptr && EnsureNotScalar(meanShapePtr->GetStorageShape()) != expectedShape) {
        std::string shapeMsg = ToString(EnsureNotScalar(meanShapePtr->GetStorageShape())) + " and " + ToString(xShape);
        std::string reasonMsg = "The shape dims of output mean and input x must be the same, "
                                "and the first " +
                                std::to_string(beginNormAxis) +
                                " axes of output mean must be the equal to"
                                " the first " +
                                std::to_string(beginNormAxis) +
                                " axes of input x, "
                                "and each axis from " +
                                std::to_string(beginNormAxis) +
                                "th axis to the last of output mean must be 1, "
                                "where " +
                                std::to_string(beginNormAxis) + " depends on the value of attribute begin_norm_axis";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "mean and x", shapeMsg.c_str(),
                                               reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto rstdShapePtr = context_->GetOutputShape(OUTPUT_IDX_RSTD);
    if (rstdShapePtr != nullptr && EnsureNotScalar(rstdShapePtr->GetStorageShape()) != expectedShape) {
        std::string shapeMsg = ToString(EnsureNotScalar(rstdShapePtr->GetStorageShape())) + " and " + ToString(xShape);
        std::string reasonMsg = "The shape dims of output rstd and input x must be the same, "
                                "and the first " +
                                std::to_string(beginNormAxis) +
                                " axes of output rstd must be the equal to"
                                " the first " +
                                std::to_string(beginNormAxis) +
                                " axes of input x, "
                                "and each axis from " +
                                std::to_string(beginNormAxis) +
                                "th axis to the last of output rstd must be 1, "
                                "where " +
                                std::to_string(beginNormAxis) + " depends on the value of attribute begin_norm_axis";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "rstd and x", shapeMsg.c_str(),
                                               reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::OutputDtypeCheck(ge::DataType xDtype, ge::DataType gammaDtype,
                                                        ge::DataType yDtype, ge::DataType meanDtype,
                                                        ge::DataType rstdDtype)
{
    if (yDtype != xDtype) {
        std::string dtypeMsg = ToString(yDtype) + " and " + ToString(xDtype);
        std::string reasonMsg = "The dtype of output y must be the same as the dtype of input x";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "y and x", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (meanDtype != gammaDtype) {
        std::string dtypeMsg = ToString(meanDtype) + " and " + ToString(gammaDtype);
        std::string reasonMsg = "The dtype of output mean must be the same as the dtype of input gamma and input beta";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "mean and gamma", dtypeMsg.c_str(),
                                               reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (rstdDtype != gammaDtype) {
        std::string dtypeMsg = ToString(rstdDtype) + " and " + ToString(gammaDtype);
        std::string reasonMsg = "The dtype of output rstd must be the same as the dtype of input gamma and input beta";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "rstd and gamma", dtypeMsg.c_str(),
                                               reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::GetShapeAttrsInfo()
{
    auto xDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto gammaDesc = context_->GetInputDesc(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    auto betaDesc = context_->GetInputDesc(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, betaDesc);

    auto yDesc = context_->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto meanDesc = context_->GetOutputDesc(OUTPUT_IDX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanDesc);
    auto rstdDesc = context_->GetOutputDesc(OUTPUT_IDX_RSTD);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rstdDesc);

    ge::DataType xDtype = xDesc->GetDataType();
    ge::DataType gammaDtype = gammaDesc->GetDataType();
    ge::DataType betaDtype = betaDesc->GetDataType();
    ge::DataType yDtype = yDesc->GetDataType();
    ge::DataType meanDtype = meanDesc->GetDataType();
    ge::DataType rstdDtype = rstdDesc->GetDataType();

    OP_CHECK_IF(InputDtypeCheck(xDtype, gammaDtype, betaDtype) == ge::GRAPH_FAILED,
                OP_LOGE(context_->GetNodeName(), "input dtype check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(OutputDtypeCheck(xDtype, gammaDtype, yDtype, meanDtype, rstdDtype) == ge::GRAPH_FAILED,
                OP_LOGE(context_->GetNodeName(), "output dtype check failed."), return ge::GRAPH_FAILED);

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
    commonParams.eps = (epsilonPtr == nullptr) ? (commonParams.isV1 ? DEFAULT_EPSILON_V1 : DEFAULT_EPSILON_V3) :
                                                 *epsilonPtr;

    const gert::Shape& xShape = EnsureNotScalar(context_->GetInputShape(INPUT_IDX_X)->GetStorageShape());
    const gert::Shape& gammaShape = EnsureNotScalar(context_->GetInputShape(INPUT_IDX_GAMMA)->GetStorageShape());
    const gert::Shape& betaShape = EnsureNotScalar(context_->GetInputShape(INPUT_IDX_BETA)->GetStorageShape());

    OP_CHECK_IF(
        InputShapeAndAxisCheck(xShape, gammaShape, betaShape, beginNormAxis, beginParamsAxis) == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "input shape or normlize axis check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(OutputShapeCheck(xShape, beginNormAxis) == ge::GRAPH_FAILED,
                OP_LOGE(context_->GetNodeName(), "output shape check failed."), return ge::GRAPH_FAILED);

    int64_t leftBound = std::min(beginNormAxis, beginParamsAxis);
    int64_t rightBound = std::max(beginNormAxis, beginParamsAxis);

    // fuse axis
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    uint64_t normToParamsSize = 1;
    uint64_t paramsToNormSize = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        if (static_cast<int64_t>(i) < beginNormAxis) {
            colSize *= xShape.GetDim(i);
        } else {
            rowSize *= xShape.GetDim(i);
        }

        if (leftBound <= static_cast<int64_t>(i) && static_cast<int64_t>(i) < rightBound) {
            if (beginNormAxis < beginParamsAxis) {
                normToParamsSize *= xShape.GetDim(i);
            } else {
                paramsToNormSize *= xShape.GetDim(i);
            }
        }
    }
    commonParams.normToParamsSize = normToParamsSize;
    commonParams.paramsToNormSize = paramsToNormSize;

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
    uint64_t gammaBetaAlignment = 16;
    if (LN_DTYPE_SIZE_MAP.find(commonParams.tensorDtype) != LN_DTYPE_SIZE_MAP.end()) {
        alignment = BLOCK_SIZE / LN_DTYPE_SIZE_MAP.at(commonParams.tensorDtype);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", ToString(commonParams.tensorDtype).c_str(),
                                  "FLOAT, FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }
    if (LN_DTYPE_SIZE_MAP.find(commonParams.paramDtype) != LN_DTYPE_SIZE_MAP.end()) {
        gammaBetaAlignment = BLOCK_SIZE / LN_DTYPE_SIZE_MAP.at(commonParams.paramDtype);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gamma and beta", ToString(commonParams.paramDtype).c_str(),
                                  "FLOAT, FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }
    commonParams.rowAlign = (commonParams.rowSize + alignment - 1) / alignment * alignment;
    commonParams.gammaBetaRAlign = (commonParams.rowSize + gammaBetaAlignment - 1) / gammaBetaAlignment *
                                   gammaBetaAlignment;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::GetCommonPlatformInfo(const LayerNormV3CompileInfo* compileInfo)
{
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "ascendc compile info is null"),
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
        auto v1CompileInfo = context_->GetCompileInfo<LayerNormOpInfo>();
        OP_CHECK_IF(v1CompileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        return GetCommonPlatformInfo(&v1CompileInfo->regbaseCompileInfo);
    }
    auto v3CompileInfo = context_->GetCompileInfo<LayerNormV3OpInfo>();
    OP_CHECK_IF(v3CompileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                return ge::GRAPH_FAILED);
    return GetCommonPlatformInfo(&v3CompileInfo->regbaseCompileInfo);
}

bool LayerNormV3TilingBase::IsCapable() { return true; }

ge::graphStatus LayerNormV3TilingBase::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LayerNormV3TilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LayerNormV3TilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = BASE_WSP_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3TilingBase::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t LayerNormV3TilingBase::GetTilingKey() const { return 0; }

} // namespace optiling
