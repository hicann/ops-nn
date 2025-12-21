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
 * \file add_layer_norm_tiling_arch35.cpp
 * \brief
 */

#include "add_layer_norm_tiling.h"

namespace optiling {
constexpr int64_t MIN_DATANUM_PER_CORE = 1024;
constexpr int64_t UB_RESERVED_SIZE = 256;
constexpr uint32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t INPUT_NUM = 4;
constexpr int64_t OUTPUT_NUM = 2;
constexpr int64_t TOTAL_OUTPUT_NUM = 4;
constexpr int64_t BUFFER_NUM = 4;
constexpr int64_t ATTR_INDEX_0 = 0;
constexpr int64_t ATTR_INDEX_1 = 1;
constexpr int64_t INDEX_0 = 0;
constexpr int64_t INDEX_1 = 1;
constexpr int64_t INDEX_2 = 2;
constexpr int64_t INDEX_3 = 3;
constexpr int64_t INDEX_4 = 4;
constexpr int64_t TWO = 2;
constexpr int64_t THREE = 3;
constexpr uint32_t TILING_910D_PREFIX = 8000;
constexpr size_t SHAPE_MAX_DIM_NUM = 8;
// full-load: 000, welford: 100
constexpr uint32_t TILING_WELFORD = 100;
// no bias: 0, bias elewise: 1, bias brc: 2
constexpr uint32_t TILING_BIAS_ELEWISE = 1;
constexpr uint32_t TILING_BIAS_BRC = 2;
const std::string OP_NAME = "AddLayerNorm";
constexpr float DEFAULT_EPSILON = 1e-5;
static const gert::Shape g_vec_1_shape = {1};

ge::graphStatus AddLayerNormRegbaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        blockSize_ = Ops::Base::GetUbBlockSize(context_);
        vecRegSize_ = Ops::Base::GetVRegSize(context_);
    } else {
        auto compileInfo = reinterpret_cast<const AddLayerNormCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
            return ge::GRAPH_FAILED);
        ubSize_ = compileInfo->ubSize_;
        aivCoreNum_ = compileInfo->aivCoreNum_;
        blockSize_ = compileInfo->blockSize_;
        vecRegSize_ = compileInfo->vecRegSize_;
    }
    vlFp32_ = vecRegSize_ / sizeof(float);
    return ge::GRAPH_SUCCESS;
}

const gert::Shape& AddLayerNormRegbaseTiling::EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckDimNum(gert::Shape& shape) const
{
    OP_CHECK_IF(
        shape.GetDimNum() > SHAPE_MAX_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "Dim num should be no greater than 8."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckShapeAllPositive(gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) <= 0,
            OP_LOGE(
                context_->GetNodeName(), "Dim %lu of input should be positive, but actual %ld.", i, shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1)
{
    OP_CHECK_IF(
        shape0.GetDimNum() != shape1.GetDimNum(),
        OP_LOGE(
            context_->GetNodeName(), "DimNum of shapes are not equal: %zu vs %zu", shape0.GetDimNum(),
            shape1.GetDimNum()),
        return ge::GRAPH_FAILED);

    for (size_t i = 0; i < shape0.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape0.GetDim(i) != shape1.GetDim(i),
            OP_LOGE(
                context_->GetNodeName(), "Dim %lu of shapes are not equal: %ld vs %ld", i, shape0.GetDim(i),
                shape1.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CalcRowsAndCols(gert::Shape& shapeX, gert::Shape& shapeGamma)
{
    rows_ = 1;
    cols_ = 1;
    if (shapeX.GetDimNum() < shapeGamma.GetDimNum()) {
        OP_LOGE(
            context_->GetNodeName(),
            "Dim num of x1 and x2 [%zu] should be no less than dim num of gamma and beta [%zu].", shapeX.GetDimNum(),
            shapeGamma.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    size_t shapeDiff = shapeX.GetDimNum() - shapeGamma.GetDimNum();
    for (size_t i = 0; i < shapeX.GetDimNum(); i++) {
        if (i < shapeDiff) {
            rows_ *= shapeX.GetDim(i);
        } else {
            if (shapeX.GetDim(i) != shapeGamma.GetDim(i - shapeDiff)) {
                OP_LOGE(
                    context_->GetNodeName(),
                    "The %zu dim of Gamma and beta should be equal to %zu dim of x1 and x2, but got %ld and %ld.",
                    i - shapeDiff, i, shapeGamma.GetDim(i - shapeDiff), shapeX.GetDim(i));
                return ge::GRAPH_FAILED;
            }
            cols_ *= shapeX.GetDim(i);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::BiasShapeProcess(
    gert::Shape& shapeX, gert::Shape& shapeGamma, gert::Shape& shapeBias)
{
    if (CheckShapeAllPositive(shapeBias) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "bias shape contains zero.");
        return ge::GRAPH_FAILED;
    }
    if (CheckDimNum(shapeBias) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "check dim num of bias failed.");
        return ge::GRAPH_FAILED;
    }
    if (shapeBias.GetDimNum() < shapeGamma.GetDimNum()) {
        OP_LOGE(
            context_->GetNodeName(), "Dim num of bias [%zu] should be no less than dim num of gamma and beta [%zu].",
            shapeBias.GetDimNum(), shapeGamma.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (shapeBias.GetDimNum() > shapeX.GetDimNum()) {
        OP_LOGE(
            context_->GetNodeName(), "Dim num of bias [%zu] should be no greater than dim num of x1 and x2 [%zu].",
            shapeBias.GetDimNum(), shapeX.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    size_t biasGammaShapeDiff = shapeBias.GetDimNum() - shapeGamma.GetDimNum();
    for (size_t i = 0; i < shapeGamma.GetDimNum(); i++) {
        if (shapeGamma.GetDim(i) != shapeBias.GetDim(i + biasGammaShapeDiff)) {
            OP_LOGE(
                context_->GetNodeName(),
                "The %zu dim of bias should be equal to %zu dim of gamma and beta, but got %ld and %ld.",
                i + biasGammaShapeDiff, i, shapeBias.GetDim(i + biasGammaShapeDiff), shapeGamma.GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
    int64_t biasSize = 1;
    for (size_t i = 0; i < shapeBias.GetDimNum(); i++) {
        biasSize *= shapeBias.GetDim(i);
    }
    // shape bias == shape x
    if (biasSize == rows_ * cols_ && shapeX.GetDimNum() == shapeBias.GetDimNum()) {
        for (size_t i = 0; i < shapeBias.GetDimNum(); i++) {
            if (shapeBias.GetDim(i) != shapeX.GetDim(i)) {
                OP_LOGE(
                    context_->GetNodeName(),
                    "The %zu dim of bias should be equal to %zu dim of x1 and x2, but got %ld and %ld.", i, i,
                    shapeBias.GetDim(i), shapeX.GetDim(i));
                return ge::GRAPH_FAILED;
            }
        }
    }
    if (biasSize == rows_ * cols_) {
        biasType_ = BIAS::BIAS_ELEWISE;
    } else if (biasSize == cols_) {
        biasType_ = BIAS::BIAS_BRC;
    } else {
        OP_LOGE(
            context_->GetNodeName(),
            "The shape size of bias %ld should be equal to shape size of x1 and x2 %ld, or shape size of gamma %ld.",
            biasSize, rows_ * cols_, cols_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckInputsShape()
{
    // check input0
    auto inputShape = context_->GetInputShape(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape0 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape0) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "x1 shape contains zero.");
        return ge::GRAPH_FAILED;
    }

    // check input1
    inputShape = context_->GetInputShape(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape1 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "x2 shape contains zero.");
        return ge::GRAPH_FAILED;
    }

    // check shapes of input0 and input1 are equal
    if (CheckShapesEqual(storageShape0, storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Shapes of x1 and x2 are not equal.");
        return ge::GRAPH_FAILED;
    }

    if (CheckDimNum(storageShape0) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "check dim num of x1 and x2 failed.");
        return ge::GRAPH_FAILED;
    }

    // check input2
    inputShape = context_->GetInputShape(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape2 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape2) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "gamma shape contains zero.");
        return ge::GRAPH_FAILED;
    }

    // check input3
    inputShape = context_->GetInputShape(INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape3 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape3) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "beta shape contains zero.");
        return ge::GRAPH_FAILED;
    }

    // check shapes of input2 and input3 are equal
    if (CheckShapesEqual(storageShape2, storageShape3) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Shapes of gamma and beta are not equal.");
        return ge::GRAPH_FAILED;
    }

    if (CalcRowsAndCols(storageShape0, storageShape2) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Check shapes of gamma and beta failed.");
        return ge::GRAPH_FAILED;
    }

    // check optional input
    inputShape = context_->GetOptionalInputShape(INDEX_4);
    if (inputShape == nullptr) {
        biasType_ = BIAS::BIAS_NONE;
    } else {
        auto biasShape = EnsureNotScalar(inputShape->GetStorageShape());
        if (BiasShapeProcess(storageShape0, storageShape2, biasShape) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "bias shape is invalid.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::MeanRstdShapeProcess(
    gert::Shape& shapeX, gert::Shape& shapeGamma, gert::Shape& shapeMeanRstd) const
{
    if (shapeX.GetDimNum() != shapeMeanRstd.GetDimNum()) {
        OP_LOGE(
            context_->GetNodeName(), "Dim num of mean and rstd [%zu] should be equal to dim num of x1 and x2 [%zu].",
            shapeMeanRstd.GetDimNum(), shapeX.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < shapeX.GetDimNum(); i++) {
        if (i < shapeX.GetDimNum() - shapeGamma.GetDimNum()) {
            if (shapeX.GetDim(i) != shapeMeanRstd.GetDim(i)) {
                OP_LOGE(
                    context_->GetNodeName(),
                    "The %zu dim of mean and rstd should be equal to %zu dim of x1 and x2, but got %ld and %ld.", i, i,
                    shapeMeanRstd.GetDim(i), shapeX.GetDim(i));
                return ge::GRAPH_FAILED;
            }
        } else {
            if (shapeMeanRstd.GetDim(i) != 1) {
                OP_LOGE(
                    context_->GetNodeName(), "The %zu dim of mean and rstd should be equal to 1, but got %ld.", i,
                    shapeMeanRstd.GetDim(i));
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckOutputsShape()
{
    // has checked nullptr
    auto x1Shape = EnsureNotScalar(context_->GetInputShape(INDEX_0)->GetStorageShape());
    auto gammaShape = EnsureNotScalar(context_->GetInputShape(INDEX_2)->GetStorageShape());
    // check output0
    auto outputShape = context_->GetOutputShape(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto yShape = EnsureNotScalar(outputShape->GetStorageShape());
    if (CheckShapesEqual(x1Shape, yShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Shapes of x1 and y are not equal.");
        return ge::GRAPH_FAILED;
    }
    // check output1 and output2
    outputShape = context_->GetOutputShape(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto meanShape = EnsureNotScalar(outputShape->GetStorageShape());
    outputShape = context_->GetOutputShape(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto rstdShape = EnsureNotScalar(outputShape->GetStorageShape());
    if (CheckShapesEqual(meanShape, rstdShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Shapes of mean and rstd are not equal.");
        return ge::GRAPH_FAILED;
    }
    if (MeanRstdShapeProcess(x1Shape, gammaShape, meanShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Shapes of mean check failed.");
        return ge::GRAPH_FAILED;
    }
    if (MeanRstdShapeProcess(x1Shape, gammaShape, rstdShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Shapes of rstd check failed.");
        return ge::GRAPH_FAILED;
    }
    // check output3
    outputShape = context_->GetOutputShape(INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto xShape = EnsureNotScalar(outputShape->GetStorageShape());
    if (CheckShapesEqual(x1Shape, xShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Shapes of x1 and x are not equal.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckInputsDtype()
{
    int inputNum = (biasType_ == BIAS::BIAS_NONE) ? INPUT_NUM : INPUT_NUM + 1;
    for (int i = 0; i < inputNum; i++) {
        auto inputDesc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
        // check dtype
        auto dtype = inputDesc->GetDataType();
        if (dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(
                context_->GetNodeName(), "Input %d only supports float16, bfloat16, float32.", i);
            return ge::GRAPH_FAILED;
        }
    }

    // check is mix dtype
    for (int i = 0; i < inputNum; i++) {
        auto inputDesc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
        auto dtype = inputDesc->GetDataType();
        if (i == 0) {
            dataType_ = dtype;
        } else if (dtype != dataType_) {
            dataType_ = ge::DataType::DT_FLOAT;
            isMix_ = true;
            break;
        }
        isMix_ = false;
    }

    // check supported dtype
    using SupportedDtype = std::tuple<ge::DataType, ge::DataType, ge::DataType, ge::DataType, ge::DataType>;
    std::vector<SupportedDtype> supportedDtypes = {
        {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT,
         ge::DataType::DT_FLOAT},
        {ge::DataType::DT_FLOAT, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT,
         ge::DataType::DT_FLOAT},
        {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT,
         ge::DataType::DT_FLOAT},
        {ge::DataType::DT_BF16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT,
         ge::DataType::DT_FLOAT},
        {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT,
         ge::DataType::DT_FLOAT16},
        {ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT,
         ge::DataType::DT_BF16},
        {ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16, ge::DataType::DT_BF16,
         ge::DataType::DT_BF16},
        {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16,
         ge::DataType::DT_FLOAT16},
        {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT,
         ge::DataType::DT_FLOAT}};

    constexpr int64_t TUPLE_INDEX_0 = 0;
    constexpr int64_t TUPLE_INDEX_1 = 1;
    constexpr int64_t TUPLE_INDEX_2 = 2;
    constexpr int64_t TUPLE_INDEX_3 = 3;
    constexpr int64_t TUPLE_INDEX_4 = 4;

    auto isSupported = [&](const SupportedDtype& dtypeTuple) {
        if (biasType_ == BIAS::BIAS_NONE) {
            for (const auto& supported : supportedDtypes) {
                if (std::make_tuple(std::get<TUPLE_INDEX_0>(dtypeTuple), std::get<TUPLE_INDEX_1>(dtypeTuple),
                    std::get<TUPLE_INDEX_2>(dtypeTuple), std::get<TUPLE_INDEX_3>(dtypeTuple)) ==
                    std::make_tuple(std::get<TUPLE_INDEX_0>(supported), std::get<TUPLE_INDEX_1>(supported),
                    std::get<TUPLE_INDEX_2>(supported), std::get<TUPLE_INDEX_3>(supported))) {
                    return true;
                }
            }
        } else {
            for (const auto& supported : supportedDtypes) {
                if (dtypeTuple == supported) {
                    return true;
                }
            }
        }
        return false;
    };

    SupportedDtype inputDtypes;
    for (int i = 0; i < inputNum; i++) {
        auto inputDesc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
        switch (i) {
            case INDEX_0:
                std::get<TUPLE_INDEX_0>(inputDtypes) = inputDesc->GetDataType();
                break;
            case INDEX_1:
                std::get<TUPLE_INDEX_1>(inputDtypes) = inputDesc->GetDataType();
                break;
            case INDEX_2:
                std::get<TUPLE_INDEX_2>(inputDtypes) = inputDesc->GetDataType();
                break;
            case INDEX_3:
                std::get<TUPLE_INDEX_3>(inputDtypes) = inputDesc->GetDataType();
                break;
            case INDEX_4:
                std::get<TUPLE_INDEX_4>(inputDtypes) = inputDesc->GetDataType();
                break;
            default:
                break;
        }
    }

    if (!isSupported(inputDtypes)) {
        if (biasType_ == BIAS::BIAS_NONE) {
            OP_LOGE(
                context_->GetNodeName(), "Input dtypes are not supported: %s, %s, %s, %s.",
                Ops::Base::ToString(std::get<TUPLE_INDEX_0>(inputDtypes)).c_str(),
                Ops::Base::ToString(std::get<TUPLE_INDEX_1>(inputDtypes)).c_str(),
                Ops::Base::ToString(std::get<TUPLE_INDEX_2>(inputDtypes)).c_str(),
                Ops::Base::ToString(std::get<TUPLE_INDEX_3>(inputDtypes)).c_str());
        } else {
            OP_LOGE(
                context_->GetNodeName(), "Input dtypes are not supported: %s, %s, %s, %s, %s.",
                Ops::Base::ToString(std::get<TUPLE_INDEX_0>(inputDtypes)).c_str(),
                Ops::Base::ToString(std::get<TUPLE_INDEX_1>(inputDtypes)).c_str(),
                Ops::Base::ToString(std::get<TUPLE_INDEX_2>(inputDtypes)).c_str(),
                Ops::Base::ToString(std::get<TUPLE_INDEX_3>(inputDtypes)).c_str(),
                Ops::Base::ToString(std::get<TUPLE_INDEX_4>(inputDtypes)).c_str());
        }
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckOutputsDtype() const
{
    auto inputDesc = context_->GetInputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto x1Dtype = inputDesc->GetDataType();
    inputDesc = context_->GetInputDesc(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto x2Dtype = inputDesc->GetDataType();

    auto outputDesc = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto yDtype = outputDesc->GetDataType();
    outputDesc = context_->GetOutputDesc(INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto xDtype = outputDesc->GetDataType();

    if (x1Dtype == x2Dtype) {
        if (yDtype != x1Dtype || xDtype != x1Dtype) {
            OP_LOGE(
                context_->GetNodeName(),
                "Dtype of y[%s] and x[%s] should be equal to dtype of x1 and x2[%s], when x1 and x2 has same dtype.",
                ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(x1Dtype).c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (yDtype != ge::DT_FLOAT || xDtype != ge::DT_FLOAT) {
            OP_LOGE(
                context_->GetNodeName(),
                "Dtype of y[%s] and x[%s] should be equal to float32, when x1 and x2 has different dtype.",
                ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(xDtype).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    outputDesc = context_->GetOutputDesc(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto meanDtype = outputDesc->GetDataType();
    outputDesc = context_->GetOutputDesc(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto rstdDtype = outputDesc->GetDataType();
    if (meanDtype != ge::DT_FLOAT || rstdDtype != ge::DT_FLOAT) {
        OP_LOGE(
            context_->GetNodeName(), "Dtype of mean[%s] and rstd[%s] should be equal to float32.",
            ge::TypeUtils::DataTypeToSerialString(meanDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(rstdDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE(OP_NAME, "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const float* epsilonPtr = attrs->GetFloat(ATTR_INDEX_0);
    epsilon_ = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;
    const bool* additionalOutputPtr = attrs->GetBool(ATTR_INDEX_1);
    needOutputX_ = (additionalOutputPtr == nullptr) ? false : *additionalOutputPtr;

    // check inputs shape
    if (CheckInputsShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Inputs shape invalid.");
        return ge::GRAPH_FAILED;
    }
    // check inputs dtype
    if (CheckInputsDtype() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Inputs dtype invalid.");
        return ge::GRAPH_FAILED;
    }
    // check outputs shape
    if (CheckOutputsShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Outputs shape invalid.");
        return ge::GRAPH_FAILED;
    }
    // check outputs dtype
    if (CheckOutputsDtype() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Outputs dtype invalid.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool AddLayerNormRegbaseTiling::IsCapable()
{
    return true;
}

uint64_t AddLayerNormRegbaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus AddLayerNormRegbaseTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::GetWorkspaceSize()
{
    workspaceSize_ = MIN_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

void AddLayerNormRegbaseTiling::CalcUsedCoreNum()
{
    int64_t dataNum = rows_ * cols_;

    if (dataNum <= MIN_DATANUM_PER_CORE) {
        usedCoreNum_ = 1;
        rowsPerCore_ = rows_;
        rowsPerTailCore_ = rows_;
        tailCoreStartIndex_ = usedCoreNum_;
    } else if (dataNum < MIN_DATANUM_PER_CORE * aivCoreNum_) {
        rowsPerCore_ = 1;
        while (rowsPerCore_ * cols_ < MIN_DATANUM_PER_CORE && rowsPerCore_ < rows_) {
            rowsPerCore_++;
        }
        usedCoreNum_ = rows_ / rowsPerCore_;
        tailCoreStartIndex_ = usedCoreNum_;
        rowsPerTailCore_ = 0;
        if (rows_ % rowsPerCore_ != 0) {
            usedCoreNum_++;
            rowsPerTailCore_ = rows_ % rowsPerCore_;
        }
    } else {
        usedCoreNum_ = (rows_ <= aivCoreNum_) ? rows_ : aivCoreNum_;
        rowsPerCore_ = (rows_ + usedCoreNum_ - 1) / usedCoreNum_;
        rowsPerTailCore_ = rowsPerCore_ - 1;
        tailCoreStartIndex_ = rows_ - rowsPerTailCore_ * usedCoreNum_;
    }
}

int64_t AddLayerNormRegbaseTiling::GetSizeOfBlockAlign(int64_t nonAlignSize)
{
    if (isMix_) {
        const int64_t mixAlignSize = blockSize_ * TWO;
        return (nonAlignSize + mixAlignSize - 1) / mixAlignSize * mixAlignSize;
    }
    return (nonAlignSize + blockSize_ - 1) / blockSize_ * blockSize_;
}

ge::graphStatus AddLayerNormRegbaseTiling::CalcUbBufferSize()
{
    auto dataTypeSize = GetSizeByDataType(dataType_);
    int64_t bufferNum = 1;

    // x1, x2, bias(optional), beta, gamma
    int64_t inputNum = (biasType_ != BIAS::BIAS_NONE) ? INPUT_NUM + 1 : INPUT_NUM;
    int64_t x1UbSizeForOneRow = GetSizeOfBlockAlign(cols_ * dataTypeSize) * bufferNum;
    int64_t x2UbSizeForOneRow = GetSizeOfBlockAlign(cols_ * dataTypeSize) * bufferNum;
    int64_t betaUbSize = GetSizeOfBlockAlign(cols_ * dataTypeSize);
    int64_t gammaUbSize = GetSizeOfBlockAlign(cols_ * dataTypeSize);
    int64_t biasUbSize = (biasType_ == BIAS::BIAS_BRC) ? GetSizeOfBlockAlign(cols_ * dataTypeSize) :
                                                         GetSizeOfBlockAlign(cols_ * dataTypeSize) * bufferNum;
    int64_t inputUbSizeForOneRow = x1UbSizeForOneRow + x2UbSizeForOneRow + betaUbSize + gammaUbSize;
    if (biasType_ != BIAS::BIAS_NONE) {
        inputUbSizeForOneRow += biasUbSize;
    }

    // x, y
    int64_t outputUbSizeForOneRow = GetSizeOfBlockAlign(cols_ * dataTypeSize) * bufferNum * TWO;

    // x32Ub
    int64_t x32UbSizeForOneRow = GetSizeOfBlockAlign(cols_ * sizeof(float)) * bufferNum;

    // mean, rstd
    int64_t meanUbSizeForOneRow = blockSize_ * bufferNum;
    int64_t rstdUbSizeForOneRow = meanUbSizeForOneRow;

    // 二分累加buffer
    binaryAddNum_ = vlFp32_;
    if (cols_ > vlFp32_) {
        while (binaryAddNum_ < cols_) {
            binaryAddNum_ *= TWO;
        }
        binaryAddNum_ /= TWO;
    }

    int64_t binaryAddUbSize = GetSizeOfBlockAlign(binaryAddNum_ / vlFp32_ * sizeof(float));
    if (static_cast<int64_t>(ubSize_) >= inputUbSizeForOneRow + outputUbSizeForOneRow + meanUbSizeForOneRow +
                                             rstdUbSizeForOneRow + x32UbSizeForOneRow + binaryAddUbSize +
                                             UB_RESERVED_SIZE) {
        // full-load
        if (biasType_ == BIAS::BIAS_ELEWISE) {
            rowsPerLoop_ = (ubSize_ - UB_RESERVED_SIZE - binaryAddUbSize - betaUbSize - gammaUbSize) /
                           (x1UbSizeForOneRow + x2UbSizeForOneRow + biasUbSize + outputUbSizeForOneRow +
                            x32UbSizeForOneRow + meanUbSizeForOneRow + rstdUbSizeForOneRow);
        } else if (biasType_ == BIAS::BIAS_BRC) {
            rowsPerLoop_ = (ubSize_ - UB_RESERVED_SIZE - binaryAddUbSize - betaUbSize - gammaUbSize - biasUbSize) /
                           (x1UbSizeForOneRow + x2UbSizeForOneRow + outputUbSizeForOneRow + x32UbSizeForOneRow +
                            meanUbSizeForOneRow + rstdUbSizeForOneRow);
        } else {
            rowsPerLoop_ = (ubSize_ - UB_RESERVED_SIZE - binaryAddUbSize - gammaUbSize - biasUbSize) /
                           (x1UbSizeForOneRow + x2UbSizeForOneRow + outputUbSizeForOneRow + x32UbSizeForOneRow +
                            meanUbSizeForOneRow + rstdUbSizeForOneRow);
        }
        if (rowsPerLoop_ > rowsPerCore_) {
            rowsPerLoop_ = rowsPerCore_;
        }
        colsPerLoop_ = cols_;
        isWelford_ = false;
    } else {
        // welford
        colsPerLoop_ =
            (ubSize_ - UB_RESERVED_SIZE - meanUbSizeForOneRow - rstdUbSizeForOneRow) * vlFp32_ /
            (((inputNum + OUTPUT_NUM) * dataTypeSize * bufferNum + BUFFER_NUM * sizeof(float)) * vlFp32_ + 1);
        if (dataTypeSize > 0 && blockSize_ > 0) {
            colsPerLoop_ = colsPerLoop_ * dataTypeSize / blockSize_ * blockSize_ / dataTypeSize;
            colsPerLoop_ = colsPerLoop_ / (blockSize_ / TWO) * (blockSize_ / TWO);
        } else {
            OP_LOGE(
                context_->GetNodeName(), "dataTypeSize(%d) or blockSize(%u) is zero", dataTypeSize, blockSize_);
            return ge::GRAPH_FAILED;
        }
        rowsPerLoop_ = 1;
        isWelford_ = true;
        binaryAddNum_ = vlFp32_;
        while (binaryAddNum_ < colsPerLoop_) {
            binaryAddNum_ *= TWO;
        }
        binaryAddNum_ /= TWO;
    }
    colsLoopCount_ = (cols_ + colsPerLoop_ - 1) / colsPerLoop_;
    colsTail_ = (cols_ % colsPerLoop_ == 0) ? colsPerLoop_ : (cols_ % colsPerLoop_);

    int64_t vcaddNum = binaryAddNum_ / vlFp32_;
    if (vcaddNum <= vlFp32_) {
        binaryAddK_ = 0;
        binaryAddLastNum_ = vcaddNum;
    } else {
        binaryAddK_ = 0;
        int64_t curBinaryAddNum = 1;
        while (curBinaryAddNum < vcaddNum / vlFp32_) {
            binaryAddK_++;
            curBinaryAddNum *= TWO;
        }
        binaryAddLastNum_ = vlFp32_;
    }
    OP_LOGW(
        "ComputeBinaryAddVars", "binaryAddNum:%ld, binaryAddK:%ld, binaryAddLastNum:%ld", binaryAddNum_, binaryAddK_,
        binaryAddLastNum_);

    return ge::GRAPH_SUCCESS;
}

void AddLayerNormRegbaseTiling::LogTilingResult()
{
    OP_LOGD(OP_NAME, "eps: %f", epsilon_);
    OP_LOGD(OP_NAME, "rows: %ld, cols: %ld", rows_, cols_);
    OP_LOGD(
        OP_NAME, "ubSize: %ld, blockSize: %d, vecRegSize: %d, vlFp32: %d, aivCoreNum: %d", ubSize_, blockSize_,
        vecRegSize_, vlFp32_, aivCoreNum_);
    OP_LOGD(
        OP_NAME, "usedCoreNum: %d, tailCoreNum: %d, tailCoreStartIndex: %d", usedCoreNum_, tailCoreNum_,
        tailCoreStartIndex_);
    OP_LOGD(
        OP_NAME, "rowsPerCore: %ld, rowsPerTailCore: %ld, rowsPerLoop: %ld", rowsPerCore_, rowsPerTailCore_,
        rowsPerLoop_);
    OP_LOGD(OP_NAME, "colsPerLoop: %ld, colsLoopCount: %ld, colsTail: %ld", colsPerLoop_, colsLoopCount_, colsTail_);
    OP_LOGD(
        OP_NAME, "binaryAddNum: %ld, binaryAddK: %ld, binaryAddLastNum: %ld", binaryAddNum_, binaryAddK_,
        binaryAddLastNum_);
    OP_LOGD(OP_NAME, "tilingKey: %d", tilingKey_);
}

ge::graphStatus AddLayerNormRegbaseTiling::DoOpTiling()
{
    ge::graphStatus result = ge::GRAPH_SUCCESS;
    // split cores
    CalcUsedCoreNum();

    // choose template and calc ub buffer size
    result = CalcUbBufferSize();

    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_vlFp32(vlFp32_);
    tilingData_.set_tailCoreStartIndex(tailCoreStartIndex_);
    tilingData_.set_rowsPerCore(rowsPerCore_);
    tilingData_.set_rowsPerTailCore(rowsPerTailCore_);
    tilingData_.set_rowsPerLoop(rowsPerLoop_);
    tilingData_.set_cols(cols_);
    tilingData_.set_colsPerLoop(colsPerLoop_);
    tilingData_.set_colsLoopCount(colsLoopCount_);
    tilingData_.set_colsTail(colsTail_);
    tilingData_.set_binaryAddNum(binaryAddNum_);
    tilingData_.set_binaryAddK(binaryAddK_);
    tilingData_.set_binaryAddLastNum(binaryAddLastNum_);
    tilingData_.set_eps(epsilon_);
    tilingData_.set_outputX(needOutputX_);

    tilingKey_ = TILING_910D_PREFIX;
    if (isWelford_) {
        tilingKey_ += TILING_WELFORD;
    }
    if (biasType_ == BIAS::BIAS_ELEWISE) {
        tilingKey_ += TILING_BIAS_ELEWISE;
    } else if (biasType_ == BIAS::BIAS_BRC) {
        tilingKey_ += TILING_BIAS_BRC;
    }

    LogTilingResult();
    return result;
}
} // namespace optiling