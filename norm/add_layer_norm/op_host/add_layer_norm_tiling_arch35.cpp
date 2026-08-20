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
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t SINGLE_BUFFER_NUM = 1;
constexpr int64_t ATTR_INDEX_0 = 0;
constexpr int64_t ATTR_INDEX_1 = 1;
constexpr int64_t INDEX_0 = 0;
constexpr int64_t INDEX_1 = 1;
constexpr int64_t INDEX_2 = 2;
constexpr int64_t INDEX_3 = 3;
constexpr int64_t INDEX_4 = 4;
constexpr int64_t TWO = 2;
constexpr int64_t THREE = 3;
constexpr int64_t FOUR = 4;
constexpr uint32_t TILING_950_PREFIX = 8000;
constexpr size_t SHAPE_MAX_DIM_NUM = 8;
// full-load: 000, welford: 100
constexpr uint32_t TILING_WELFORD = 100;
// full-load 不开 DoubleBuffer 时十位置 1
constexpr uint32_t TILING_FULL_LOAD_NO_DB = 10;
// no bias: 0, bias elewise: 1, bias brc: 2
constexpr uint32_t TILING_BIAS_ELEWISE = 1;
constexpr uint32_t TILING_BIAS_BRC = 2;
// 空张量(reduce empty, R==0 && A>0)兜底 tilingKey。独立key。
constexpr uint32_t TILING_REDUCE_EMPTY = 8200;
// 空张量核间切分阈值:每核活量不满 32KB 不单独开核
constexpr int64_t SINGLE_CORE_MIN_BYTES = 32 * 1024;
const std::string OP_NAME = "AddLayerNorm";
const std::string INPLACE_OP_NAME = "InplaceAddLayerNorm";
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
        OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
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
    OP_CHECK_IF(shape.GetDimNum() > SHAPE_MAX_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "Dim num should be no greater than 8."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckShapeAllPositive(gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) <= 0,
                    OP_LOGE(context_->GetNodeName(), "Dim %lu of input should be positive, but actual %ld.", i,
                            shape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1)
{
    OP_CHECK_IF(shape0.GetDimNum() != shape1.GetDimNum(),
                OP_LOGE(context_->GetNodeName(), "DimNum of shapes are not equal: %zu vs %zu", shape0.GetDimNum(),
                        shape1.GetDimNum()),
                return ge::GRAPH_FAILED);

    for (size_t i = 0; i < shape0.GetDimNum(); i++) {
        OP_CHECK_IF(shape0.GetDim(i) != shape1.GetDim(i),
                    OP_LOGE(context_->GetNodeName(), "Dim %lu of shapes are not equal: %ld vs %ld", i, shape0.GetDim(i),
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
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "x1/x2 and gamma/beta",
            (std::to_string(shapeX.GetDimNum()) + " and " + std::to_string(shapeGamma.GetDimNum())).c_str(),
            "The shape dims of x1 and x2 should be no less than the shape dims of gamma and beta");
        return ge::GRAPH_FAILED;
    }
    size_t shapeDiff = shapeX.GetDimNum() - shapeGamma.GetDimNum();
    for (size_t i = 0; i < shapeX.GetDimNum(); i++) {
        if (i < shapeDiff) {
            rows_ *= shapeX.GetDim(i);
        } else {
            if (shapeX.GetDim(i) != shapeGamma.GetDim(i - shapeDiff)) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "gamma/beta and x1/x2",
                    (Ops::Base::ToString(shapeGamma) + " and " + Ops::Base::ToString(shapeX)).c_str(),
                    ("The " + std::to_string(i - shapeDiff) + " dim of gamma and beta should be equal to the " +
                     std::to_string(i) + " dim of x1 and x2")
                        .c_str());
                return ge::GRAPH_FAILED;
            }
            cols_ *= shapeX.GetDim(i);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::BiasShapeProcess(gert::Shape& shapeX, gert::Shape& shapeGamma,
                                                            gert::Shape& shapeBias)
{
    if (CheckShapeAllPositive(shapeBias) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "bias", Ops::Base::ToString(shapeBias).c_str(),
                                              "bias cannot be an empty tensor");
        return ge::GRAPH_FAILED;
    }
    if (CheckDimNum(shapeBias) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "bias", std::to_string(shapeBias.GetDimNum()).c_str(),
                                     "less than or equal to 8");
        return ge::GRAPH_FAILED;
    }
    if (shapeBias.GetDimNum() < shapeGamma.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "bias and gamma/beta",
            (std::to_string(shapeBias.GetDimNum()) + " and " + std::to_string(shapeGamma.GetDimNum())).c_str(),
            "The shape dim of bias should be no less than the shape dims of gamma and beta");
        return ge::GRAPH_FAILED;
    }
    if (shapeBias.GetDimNum() > shapeX.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "bias and x1/x2",
            (std::to_string(shapeBias.GetDimNum()) + " and " + std::to_string(shapeX.GetDimNum())).c_str(),
            "The shape dim of bias should be no greater than the shape dims of x1 and x2");
        return ge::GRAPH_FAILED;
    }
    size_t biasGammaShapeDiff = shapeBias.GetDimNum() - shapeGamma.GetDimNum();
    for (size_t i = 0; i < shapeGamma.GetDimNum(); i++) {
        if (shapeGamma.GetDim(i) != shapeBias.GetDim(i + biasGammaShapeDiff)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "bias and gamma",
                (Ops::Base::ToString(shapeBias) + " and " + Ops::Base::ToString(shapeGamma)).c_str(),
                ("The " + std::to_string(i + biasGammaShapeDiff) + " dim of bias should be equal to the " +
                 std::to_string(i) + " dim of gamma")
                    .c_str());
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
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "bias and x1",
                    (Ops::Base::ToString(shapeBias) + " and " + Ops::Base::ToString(shapeX)).c_str(),
                    ("The " + std::to_string(i) + " dim of bias should be equal to the " + std::to_string(i) +
                     " dim of x1")
                        .c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    if (biasSize == rows_ * cols_) {
        biasType_ = BIAS::BIAS_ELEWISE;
    } else if (biasSize == cols_) {
        biasType_ = BIAS::BIAS_BRC;
    } else {
        OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "bias", std::to_string(biasSize).c_str(),
                                      "equal to shape size of x1 and x2 or shape size of gamma");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckInputsShape()
{
    // 取全部输入 shape
    auto inputShape0 = context_->GetInputShape(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape0);
    auto storageShape0 = EnsureNotScalar(inputShape0->GetStorageShape());
    auto inputShape1 = context_->GetInputShape(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape1);
    auto storageShape1 = EnsureNotScalar(inputShape1->GetStorageShape());
    auto inputShape2 = context_->GetInputShape(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape2);
    auto storageShape2 = EnsureNotScalar(inputShape2->GetStorageShape());
    auto inputShape3 = context_->GetInputShape(INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape3);
    auto storageShape3 = EnsureNotScalar(inputShape3->GetStorageShape());

    if (CheckShapesEqual(storageShape0, storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "x1 and x2",
            (Ops::Base::ToString(storageShape0) + " and " + Ops::Base::ToString(storageShape1)).c_str(),
            "The shapes of x1 and x2 should be the same");
        return ge::GRAPH_FAILED;
    }
    if (CheckDimNum(storageShape0) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x1", std::to_string(storageShape0.GetDimNum()).c_str(),
                                     "less than or equal to 8");
        return ge::GRAPH_FAILED;
    }
    if (CheckShapesEqual(storageShape2, storageShape3) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "gamma and beta",
            (Ops::Base::ToString(storageShape2) + " and " + Ops::Base::ToString(storageShape3)).c_str(),
            "The shapes of gamma and beta should be the same");
        return ge::GRAPH_FAILED;
    }
    if (CalcRowsAndCols(storageShape0, storageShape2) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Check shapes of gamma and beta failed.");
        return ge::GRAPH_FAILED;
    }
    // 空张量(reduce empty)放行：仅当归一化维 R==0 且外维 A>0。
    if (cols_ == 0 && rows_ > 0) {
        isReduceEmpty_ = true;
        biasType_ = BIAS::BIAS_NONE;
        return ge::GRAPH_SUCCESS;
    }
    if (CheckShapeAllPositive(storageShape0) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x1", Ops::Base::ToString(storageShape0).c_str(),
                                              "x1 cannot be an empty tensor");
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeAllPositive(storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x2", Ops::Base::ToString(storageShape1).c_str(),
                                              "x2 cannot be an empty tensor");
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeAllPositive(storageShape2) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "gamma",
                                              Ops::Base::ToString(storageShape2).c_str(),
                                              "gamma cannot be an empty tensor");
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeAllPositive(storageShape3) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "beta",
                                              Ops::Base::ToString(storageShape3).c_str(),
                                              "beta cannot be an empty tensor");
        return ge::GRAPH_FAILED;
    }
    auto biasShapeP = context_->GetOptionalInputShape(INDEX_4);
    if (biasShapeP == nullptr) {
        biasType_ = BIAS::BIAS_NONE;
    } else {
        auto biasShape = EnsureNotScalar(biasShapeP->GetStorageShape());
        if (BiasShapeProcess(storageShape0, storageShape2, biasShape) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "bias shape is invalid.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::MeanRstdShapeProcess(gert::Shape& shapeX, gert::Shape& shapeGamma,
                                                                gert::Shape& shapeMeanRstd) const
{
    if (shapeX.GetDimNum() != shapeMeanRstd.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "mean/rstd and x1",
            (std::to_string(shapeMeanRstd.GetDimNum()) + " and " + std::to_string(shapeX.GetDimNum())).c_str(),
            "The shape dims of mean and rstd should be equal to the shape dim of x1");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < shapeX.GetDimNum(); i++) {
        if (i < shapeX.GetDimNum() - shapeGamma.GetDimNum()) {
            if (shapeX.GetDim(i) != shapeMeanRstd.GetDim(i)) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "mean/rstd and x1",
                    (Ops::Base::ToString(shapeMeanRstd) + " and " + Ops::Base::ToString(shapeX)).c_str(),
                    ("The " + std::to_string(i) + " dim of mean and rstd should be equal to the " + std::to_string(i) +
                     " dim of x1")
                        .c_str());
                return ge::GRAPH_FAILED;
            }
        } else {
            if (shapeMeanRstd.GetDim(i) != 1) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "mean/rstd", Ops::Base::ToString(shapeMeanRstd).c_str(),
                    ("The " + std::to_string(i) + " dim of mean and rstd should be equal to 1").c_str());
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
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "x1 and y",
            (Ops::Base::ToString(x1Shape) + " and " + Ops::Base::ToString(yShape)).c_str(),
            "The shapes of x1 and y should be the same");
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
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "mean and rstd",
            (Ops::Base::ToString(meanShape) + " and " + Ops::Base::ToString(rstdShape)).c_str(),
            "The shapes of mean and rstd should be the same");
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
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "x1 and x",
            (Ops::Base::ToString(x1Shape) + " and " + Ops::Base::ToString(xShape)).c_str(),
            "The shapes of x1 and x should be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckInputsDtype()
{
    if (std::string(context_->GetNodeType()) == INPLACE_OP_NAME) {
        return CheckInplaceInputsDtype();
    }
    static const char* kInputNames[] = {"x1", "x2", "gamma", "beta", "bias"};
    int inputNum = (biasType_ == BIAS::BIAS_NONE) ? INPUT_NUM : INPUT_NUM + 1;
    for (int i = 0; i < inputNum; i++) {
        auto inputDesc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
        // check dtype
        auto dtype = inputDesc->GetDataType();
        if (dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), kInputNames[i], Ops::Base::ToString(dtype).c_str(),
                                      "float16, bfloat16 or float32.");
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
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1, x2, gamma and beta",
                                                   (Ops::Base::ToString(std::get<TUPLE_INDEX_0>(inputDtypes)) + ", " +
                                                    Ops::Base::ToString(std::get<TUPLE_INDEX_1>(inputDtypes)) + ", " +
                                                    Ops::Base::ToString(std::get<TUPLE_INDEX_2>(inputDtypes)) +
                                                    " and " + Ops::Base::ToString(std::get<TUPLE_INDEX_3>(inputDtypes)))
                                                       .c_str(),
                                                   "Input dtypes are not supported");
        } else {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1, x2, gamma, beta and bias",
                                                   (Ops::Base::ToString(std::get<TUPLE_INDEX_0>(inputDtypes)) + ", " +
                                                    Ops::Base::ToString(std::get<TUPLE_INDEX_1>(inputDtypes)) + ", " +
                                                    Ops::Base::ToString(std::get<TUPLE_INDEX_2>(inputDtypes)) + ", " +
                                                    Ops::Base::ToString(std::get<TUPLE_INDEX_3>(inputDtypes)) +
                                                    " and " + Ops::Base::ToString(std::get<TUPLE_INDEX_4>(inputDtypes)))
                                                       .c_str(),
                                                   "Input dtypes are not supported");
        }
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLayerNormRegbaseTiling::CheckInplaceInputsDtype()
{
    auto x1Desc = context_->GetInputDesc(INDEX_0);
    auto x2Desc = context_->GetInputDesc(INDEX_1);
    auto gammaDesc = context_->GetInputDesc(INDEX_2);
    auto betaDesc = context_->GetInputDesc(INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, betaDesc);

    const auto baseDtype = x1Desc->GetDataType();
    const auto x2Dtype = x2Desc->GetDataType();
    const auto gammaDtype = gammaDesc->GetDataType();
    const auto betaDtype = betaDesc->GetDataType();
    if (baseDtype != ge::DT_FLOAT16 && baseDtype != ge::DT_BF16 && baseDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", Ops::Base::ToString(baseDtype).c_str(),
                                  "float16, bfloat16 or float32.");
        return ge::GRAPH_FAILED;
    }
    if (x2Dtype != baseDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1 and x2",
            (Ops::Base::ToString(baseDtype) + " and " + Ops::Base::ToString(x2Dtype)).c_str(),
            "The dtypes of x1 and x2 should be the same for InplaceAddLayerNorm");
        return ge::GRAPH_FAILED;
    }
    if (gammaDtype != betaDtype || (gammaDtype != baseDtype && gammaDtype != ge::DT_FLOAT)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1, gamma and beta",
            (Ops::Base::ToString(baseDtype) + ", " + Ops::Base::ToString(gammaDtype) + " and " +
             Ops::Base::ToString(betaDtype))
                .c_str(),
            "The dtypes of gamma and beta should be the same, and should match x1 or be float32");
        return ge::GRAPH_FAILED;
    }

    if (biasType_ != BIAS::BIAS_NONE) {
        auto biasDesc = context_->GetInputDesc(INDEX_4);
        OP_CHECK_NULL_WITH_CONTEXT(context_, biasDesc);
        if (biasDesc->GetDataType() != baseDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "x1 and bias",
                (Ops::Base::ToString(baseDtype) + " and " + Ops::Base::ToString(biasDesc->GetDataType())).c_str(),
                "The dtype of bias should match x1 for InplaceAddLayerNorm");
            return ge::GRAPH_FAILED;
        }
    }

    dataType_ = (gammaDtype == baseDtype) ? baseDtype : ge::DT_FLOAT;
    isMix_ = gammaDtype != baseDtype;
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
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "y, x and x1",
                (Ops::Base::ToString(yDtype) + ", " + Ops::Base::ToString(xDtype) + " and " +
                 Ops::Base::ToString(x1Dtype))
                    .c_str(),
                "The dtypes of y and x should be equal to the dtypes of x1 and x2 when x1 and x2 has same dtype");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (yDtype != ge::DT_FLOAT || xDtype != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "y and x",
                (Ops::Base::ToString(yDtype) + " and " + Ops::Base::ToString(xDtype)).c_str(),
                "The dtypes of y and x should be float32 when x1 and x2 has different dtype");
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
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "mean and rstd",
            (Ops::Base::ToString(meanDtype) + " and " + Ops::Base::ToString(rstdDtype)).c_str(),
            "The dtypes of mean and rstd should be float32");
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

bool AddLayerNormRegbaseTiling::IsCapable() { return true; }

uint64_t AddLayerNormRegbaseTiling::GetTilingKey() const { return tilingKey_; }

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

ge::graphStatus AddLayerNormRegbaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

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

    // x1, x2, bias(optional), beta, gamma
    int64_t inputNum = (biasType_ != BIAS::BIAS_NONE) ? INPUT_NUM + 1 : INPUT_NUM;
    int64_t colsUbSizeAlign = GetSizeOfBlockAlign(cols_ * dataTypeSize);
    // x32Ub
    int64_t x32UbSizeForOneRow = GetSizeOfBlockAlign(cols_ * sizeof(float));
    int64_t meanUbSizeForOneRow = blockSize_ * DOUBLE_BUFFER_NUM;
    int64_t rstdUbSizeForOneRow = meanUbSizeForOneRow;
    // 二分累加buffer
    binaryAddNum_ = vlFp32_;
    if (cols_ > vlFp32_) {
        while (binaryAddNum_ < cols_) {
            binaryAddNum_ *= TWO;
        }
        binaryAddNum_ /= TWO;
    }
    int64_t binaryAddUbSize = 0;
    if (cols_ > TWO * vlFp32_) {
        binaryAddUbSize = GetSizeOfBlockAlign(binaryAddNum_ / vlFp32_ * sizeof(float));
    }
    int64_t colsAlign = (cols_ + vlFp32_ - 1) / vlFp32_ * vlFp32_;
    int64_t fullLoadColsCap = static_cast<int64_t>(TWO) * vlFp32_ * vlFp32_ * static_cast<int64_t>(TWO);
    int64_t fixedUbSize = colsUbSizeAlign * TWO + ((biasType_ == BIAS::BIAS_BRC) ? colsUbSizeAlign : 0);
    int64_t rowScaledUbSize = colsUbSizeAlign * FOUR + blockSize_ * TWO +
                              ((biasType_ == BIAS::BIAS_ELEWISE) ? colsUbSizeAlign : 0);
    int64_t rowConstUbSize = x32UbSizeForOneRow + binaryAddUbSize;
    // full-load开 DB -> full-load不开 DB -> welford
    int64_t bufferNum = DOUBLE_BUFFER_NUM;
    while (bufferNum >= SINGLE_BUFFER_NUM && static_cast<int64_t>(ubSize_) < fixedUbSize + rowScaledUbSize * bufferNum +
                                                                                 rowConstUbSize + UB_RESERVED_SIZE) {
        bufferNum--;
    }
    if (bufferNum >= SINGLE_BUFFER_NUM && colsAlign <= fullLoadColsCap) {
        // full-load
        rowsPerLoop_ = (ubSize_ - UB_RESERVED_SIZE - fixedUbSize) / (rowScaledUbSize * bufferNum + rowConstUbSize);
        rowsPerLoop_ = (rowsPerLoop_ > rowsPerCore_) ? rowsPerCore_ : rowsPerLoop_;
        colsPerLoop_ = cols_;
        isWelford_ = false;
        tilingKey_ += (bufferNum == SINGLE_BUFFER_NUM) ? TILING_FULL_LOAD_NO_DB : 0;
    } else {
        // welford
        colsPerLoop_ = (ubSize_ - UB_RESERVED_SIZE - meanUbSizeForOneRow - rstdUbSizeForOneRow) * vlFp32_ /
                       (((inputNum + OUTPUT_NUM) * dataTypeSize * DOUBLE_BUFFER_NUM + BUFFER_NUM * sizeof(float)) *
                            vlFp32_ +
                        1);
        if (dataTypeSize > 0 && blockSize_ > 0) {
            colsPerLoop_ = colsPerLoop_ * dataTypeSize / blockSize_ * blockSize_ / dataTypeSize;
            colsPerLoop_ = colsPerLoop_ / (blockSize_ / TWO) * (blockSize_ / TWO);
        } else {
            OP_LOGE(context_->GetNodeName(), "dataTypeSize(%d) or blockSize(%ld) is zero", dataTypeSize, blockSize_);
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
    OP_LOGW("ComputeBinaryAddVars", "binaryAddNum:%ld, binaryAddK:%ld, binaryAddLastNum:%ld", binaryAddNum_,
            binaryAddK_, binaryAddLastNum_);

    return ge::GRAPH_SUCCESS;
}

void AddLayerNormRegbaseTiling::LogTilingResult()
{
    OP_LOGD(OP_NAME, "eps: %f", epsilon_);
    OP_LOGD(OP_NAME, "rows: %ld, cols: %ld", rows_, cols_);
    OP_LOGD(OP_NAME, "ubSize: %ld, blockSize: %d, vecRegSize: %d, vlFp32: %d, aivCoreNum: %d", ubSize_, blockSize_,
            vecRegSize_, vlFp32_, aivCoreNum_);
    OP_LOGD(OP_NAME, "usedCoreNum: %d, tailCoreNum: %d, tailCoreStartIndex: %d", usedCoreNum_, tailCoreNum_,
            tailCoreStartIndex_);
    OP_LOGD(OP_NAME, "rowsPerCore: %ld, rowsPerTailCore: %ld, rowsPerLoop: %ld", rowsPerCore_, rowsPerTailCore_,
            rowsPerLoop_);
    OP_LOGD(OP_NAME, "colsPerLoop: %ld, colsLoopCount: %ld, colsTail: %ld", colsPerLoop_, colsLoopCount_, colsTail_);
    OP_LOGD(OP_NAME, "binaryAddNum: %ld, binaryAddK: %ld, binaryAddLastNum: %ld", binaryAddNum_, binaryAddK_,
            binaryAddLastNum_);
    OP_LOGD(OP_NAME, "tilingKey: %d", tilingKey_);
}

void AddLayerNormRegbaseTiling::SetTilingData()
{
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
}

ge::graphStatus AddLayerNormRegbaseTiling::DoOpTiling()
{
    ge::graphStatus result = ge::GRAPH_SUCCESS;

    if (isReduceEmpty_) {
        int64_t elemsPerBlock = blockSize_ / static_cast<int64_t>(sizeof(float));
        // 核间:每核活量不满 32KB 不单独开,封顶物理核数
        int64_t blockNum = (rows_ * static_cast<int64_t>(sizeof(float)) + SINGLE_CORE_MIN_BYTES - 1) /
                           SINGLE_CORE_MIN_BYTES;
        if (blockNum > aivCoreNum_) {
            blockNum = aivCoreNum_;
        }
        usedCoreNum_ = blockNum;
        rowsPerCore_ = (rows_ + blockNum - 1) / blockNum;
        rowsPerTailCore_ = rows_ - (blockNum - 1) * rowsPerCore_;
        tailCoreStartIndex_ = blockNum - 1;
        int64_t perLoopMax = static_cast<int64_t>(ubSize_) / static_cast<int64_t>(sizeof(float));
        int64_t chunk = (perLoopMax < rowsPerCore_) ? perLoopMax : rowsPerCore_;
        rowsPerLoop_ = chunk / elemsPerBlock * elemsPerBlock;
        if (rowsPerLoop_ < 1) {
            rowsPerLoop_ = rowsPerCore_;
        }
        colsPerLoop_ = 0;
        colsLoopCount_ = 0;
        colsTail_ = 0;
        binaryAddNum_ = 0;
        binaryAddK_ = 0;
        binaryAddLastNum_ = 0;
        SetTilingData();
        tilingKey_ = TILING_REDUCE_EMPTY;
        return ge::GRAPH_SUCCESS;
    }

    CalcUsedCoreNum();

    tilingKey_ = TILING_950_PREFIX;
    result = CalcUbBufferSize();
    if (result != ge::GRAPH_SUCCESS) {
        return result;
    }
    SetTilingData();

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
