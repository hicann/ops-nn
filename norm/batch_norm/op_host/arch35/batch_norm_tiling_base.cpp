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
 * \file batch_norm_tiling_base.cpp
 * \brief
 */
#include "batch_norm_tiling.h"

using namespace ge;

namespace optiling
{
constexpr float FLT_EPSILON = 1e-6f;
static const char* inputParamNames[] = {"x", "scale", "offset", "mean", "variance"};
static const char* outputParamNames[] = {
        "y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2", "reserve_space_3"};
static bool Isclose(float a, float b, float rel_tol = FLT_EPSILON, float abs_tol = 0.0f)
{
    if (a == b) {
        return true;
    }

    float diff = std::fabs(b - a);
    auto ret = (((diff <= std::fabs(rel_tol * b)) || (diff <= std::fabs(rel_tol * a))) || (diff <= abs_tol));
    return ret;
}

string BNTensorDesc2String(const gert::StorageShape* shape, const gert::CompileTimeTensorDesc* tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()) << "),";
    oss << "(shape:" << Shape2String(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << Shape2String(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToSerialString(
               static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToSerialString(tensor->GetOriginFormat()) << ") ";

    return oss.str();
}

string BNDebugTilingContext(gert::TilingContext* context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << BNTensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << BNTensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

ge::graphStatus BatchNormTilingBase::GetAttrsAndCheckValid()
{
    if (context_ == nullptr) {
        OP_LOGE("BatchNormTilingBase", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(opName, "TilingContext: %s", optiling::BNDebugTilingContext(context_).c_str());

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const bool* isTrainingPtr = attrs->GetBool(INDEX_IS_TRAINING);
    isTraining_ = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;

    const float* epsilonPtr = attrs->GetFloat(0);
    epsilon_ = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;

    const float* exponentialAvgFactorPtr = attrs->GetFloat(INDEX_EXPONENTIAL_AVG_FACTOR);
    exponentialAvgFactor_ =
        (exponentialAvgFactorPtr == nullptr) ? DEFAULT_EXPONENTIAL_AVG_FACTOR : *exponentialAvgFactorPtr;

    OP_LOGD(context_->GetNodeName(),
            "GetAttrsAndCheckValid success, isTraining: %d, eps: %f, exponential_avg_factor: %f.", isTraining_,
            epsilon_, exponentialAvgFactor_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormTilingBase::GetXYShapesAndCheckValid()
{
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = Ops::NN::OpTiling::EnsureNotScalar(xShape->GetStorageShape());
    int64_t xShapeSize = xStorageShape.GetDimNum();

    auto yShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = Ops::NN::OpTiling::EnsureNotScalar(yShape->GetStorageShape());
    int64_t yShapeSize = yStorageShape.GetDimNum();

    auto xDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xFormat_ = xDesc->GetFormat().GetStorageFormat();

    if (xFormat_ != FORMAT_NCHW && xFormat_ != FORMAT_NHWC && xFormat_ != FORMAT_NCDHW && xFormat_ != FORMAT_NDHWC &&
        xFormat_ != FORMAT_ND) {
        std::string formatStr = Ops::Base::ToString(xFormat_);
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", formatStr.c_str(), "NCHW, NHWC, NCDHW, NDHWC or ND");
        return ge::GRAPH_FAILED;
    }

    bool validShapeSize = (xShapeSize == yShapeSize) &&
                          ((xFormat_ == FORMAT_NHWC && xShapeSize == DIM_NUM_4) ||
                           ((xFormat_ == FORMAT_NCDHW || xFormat_ == FORMAT_NDHWC) && xShapeSize == DIM_NUM_5) ||
                           (xFormat_ == FORMAT_ND && xShapeSize >= DIM_NUM_2) ||
                           (xFormat_ == FORMAT_NCHW && xShapeSize >= DIM_NUM_2));
    if (!validShapeSize) {
        std::string sizeMsg = std::to_string(xShapeSize) + " and " + std::to_string(yShapeSize);
        std::string reason =
            "x and y shape size should be the same. NHWC requires 4, NCDHW and NDHWC require 5, NCHW and ND require at "
            "least 2.";
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "x and y", sizeMsg.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    for (int64_t i = 0; i < xShapeSize; i++) {
        if (xStorageShape.GetDim(i) != yStorageShape.GetDim(i) || xStorageShape.GetDim(i) <= 0) {
            std::string shapesMsg = Ops::Base::ToString(xStorageShape) + " and " + Ops::Base::ToString(yStorageShape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y",
                shapesMsg.c_str(), "x and y should have the same shape and each dimension must be greater than 0");
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGD(context_->GetNodeName(), "GetXYShapesAndCheckValid success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormTilingBase::CheckSmallShapesValid(int64_t aDimLen)
{
    
    // input scale, offset
    for (int64_t i = CONST_ONE; i < CONST_THREE; i++) {
        auto shape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        auto storageShape = Ops::NN::OpTiling::EnsureNotScalar(shape->GetStorageShape());
        OP_CHECK_IF(
            storageShape.GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), inputParamNames[i],
                std::to_string(storageShape.GetDimNum()).c_str(), "1"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            storageShape.GetDim(DIM_0) != aDimLen,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputParamNames[i],
                Ops::Base::ToString(storageShape).c_str(),
                (std::string(inputParamNames[i]) + " dimension 0 should be " + std::to_string(aDimLen)).c_str()),
            return ge::GRAPH_FAILED);
    }

    // input mean, var
    for (int64_t i = CONST_THREE; i < CONST_FIVE; i++) {
        auto desc = context_->GetOptionalInputDesc(i);
        OP_CHECK_IF(
            desc == nullptr && !isTraining_,
            OP_LOGE(context_->GetNodeName(), "Input %ld is required in inference mode.", i),
            return ge::GRAPH_FAILED);
        if (!desc) {
            // 和71保持兼容，training模式下，没有输入mean或者var时exponentialAvgFactor置成"1.0"
            OP_LOGD(context_->GetNodeName(), "Input %ld is None in training mode, reset exponentialAvgFactor to 1.0.",
                    i);
            exponentialAvgFactor_ = DEFAULT_EXPONENTIAL_AVG_FACTOR;
            continue;
        }

        auto shape = context_->GetOptionalInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        auto storageShape = Ops::NN::OpTiling::EnsureNotScalar(shape->GetStorageShape());
        OP_CHECK_IF(
            storageShape.GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), inputParamNames[i],
                std::to_string(storageShape.GetDimNum()).c_str(), "1"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            storageShape.GetDim(DIM_0) != aDimLen,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputParamNames[i],
                Ops::Base::ToString(storageShape).c_str(),
                (std::string(inputParamNames[i]) + " dimension 0 should be " + std::to_string(aDimLen)).c_str()),
            return ge::GRAPH_FAILED);
    }

    // output batchMean, batchVar, reserve_space_1, reserve_space_2
    for (int64_t i = CONST_ONE; i < CONST_FIVE; i++) {
        auto shape = context_->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        auto storageShape = Ops::NN::OpTiling::EnsureNotScalar(shape->GetStorageShape());
        OP_CHECK_IF(
            storageShape.GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), outputParamNames[i],
                std::to_string(storageShape.GetDimNum()).c_str(), "1"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            storageShape.GetDim(DIM_0) != aDimLen,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), outputParamNames[i],
                Ops::Base::ToString(storageShape).c_str(),
                (std::string(outputParamNames[i]) + " dimension 0 should be " + std::to_string(aDimLen)).c_str()),
            return ge::GRAPH_FAILED);
    }

    // reserve_space_3
    auto reserveSpace3Shape = context_->GetOutputShape(CONST_FIVE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, reserveSpace3Shape);
    auto reserveSpace3StorageShape = Ops::NN::OpTiling::EnsureNotScalar(reserveSpace3Shape->GetStorageShape());
    OP_CHECK_IF(
        reserveSpace3StorageShape.GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "reserve_space_3",
            std::to_string(reserveSpace3StorageShape.GetDimNum()).c_str(), "1"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        reserveSpace3StorageShape.GetDim(DIM_0) != CONST_ONE,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "reserve_space_3",
            Ops::Base::ToString(reserveSpace3StorageShape).c_str(), "reserve_space_3 dimension 0 should be 1"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context_->GetNodeName(), "CheckSmallShapesValid  success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormTilingBase::GetDtypesAndCheckValid()
{
    // x,y
    auto xDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = xDesc->GetDataType();
    OP_CHECK_IF(std::find(DTYPE_LIST.begin(), DTYPE_LIST.end(), xDtype_) == DTYPE_LIST.end(),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
                    Ops::Base::ToString(xDtype_).c_str(), "float16, bfloat16 or float"),
                return ge::GRAPH_FAILED);

    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDataType = yDesc->GetDataType();
    if (xDtype_ != yDataType) {
        std::string dtypeMsg = Ops::Base::ToString(yDataType) + " and " +  Ops::Base::ToString(xDtype_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "y and x",
            dtypeMsg.c_str(), "y and x should have the same dtype");
        return ge::GRAPH_FAILED;
    }

    // scale, offset
    for (int64_t i = CONST_ONE; i < CONST_THREE; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        ge::DataType subDataType = desc->GetDataType();
        OP_CHECK_IF(subDataType != ge::DataType::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), inputParamNames[i],
                        Ops::Base::ToString(subDataType).c_str(), "float"),
                    return ge::GRAPH_FAILED);
    }

    // mean, var
    for (int64_t i = CONST_THREE; i < CONST_FIVE; i++) {
        auto desc = context_->GetOptionalInputDesc(i);
        if (desc == nullptr) {
            continue;
        }
        ge::DataType subDataType = desc->GetDataType();
        OP_CHECK_IF(subDataType != ge::DataType::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), inputParamNames[i],
                        Ops::Base::ToString(subDataType).c_str(), "float"),
                    return ge::GRAPH_FAILED);
    }

    // reserve_space_1,2,3必选输出(竞品tf可选输出)
    for (int64_t i = CONST_ONE; i < CONST_SIX; i++) {
        auto desc = context_->GetOutputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        ge::DataType subDataType = desc->GetDataType();
        OP_CHECK_IF(subDataType != ge::DataType::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), outputParamNames[i],
                        Ops::Base::ToString(subDataType).c_str(), "float"),
                    return ge::GRAPH_FAILED);
    }

    OP_LOGD(context_->GetNodeName(), "GetDtypesAndCheckValid success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormTilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const BatchNormCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    blockSize_ = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32_ = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT32_BYTES;
    vlFp16_ = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT16_BYTES;

    opName = context_->GetNodeName();
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.blockDim = compileInfo->coreNum;
        aicoreParams_.ubSize = compileInfo->ubSize;
    }

    OP_LOGI(opName, "GetPlatformInfo success. BlockSize: %ld, ubSize: %lu, blockDim: %ld, vlFp32: %ld", blockSize_,
            aicoreParams_.ubSize, aicoreParams_.blockDim, vlFp32_);

    return ge::GRAPH_SUCCESS;
}

// infer
ge::graphStatus BatchNormTilingInferBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(GetAttrsAndCheckValid() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Get attrs failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(isTraining_, OP_LOGI(context_, "This node only support infer."), return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(GetXYShapesAndCheckValid() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = Ops::NN::OpTiling::EnsureNotScalar(xShape->GetStorageShape());
    // 获取输入shape
    if (xFormat_ == FORMAT_NHWC) {
        fusedB0Len_ = xStorageShape.GetDim(DIM_0) * xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2);
        fusedALen_ = xStorageShape.GetDim(DIM_3);
        fusedB1Len_ = CONST_ONE;
    } else if (xFormat_ == FORMAT_NDHWC) {
        fusedB0Len_ = xStorageShape.GetDim(DIM_0) * xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2) *
                      xStorageShape.GetDim(DIM_3);
        fusedALen_ = xStorageShape.GetDim(DIM_4);
        fusedB1Len_ = CONST_ONE;
    } else if (xFormat_ == FORMAT_NCHW) {
        fusedB0Len_ = xStorageShape.GetDim(DIM_0);
        fusedALen_ = xStorageShape.GetDim(DIM_1);
        fusedB1Len_ = xStorageShape.GetShapeSize() / fusedB0Len_ / fusedALen_;
    } else if (xFormat_ == FORMAT_NCDHW) {
        fusedB0Len_ = xStorageShape.GetDim(DIM_0);
        fusedALen_ = xStorageShape.GetDim(DIM_1);
        fusedB1Len_ = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3) * xStorageShape.GetDim(DIM_4);
    } else if (xFormat_ == FORMAT_ND) {
        fusedB0Len_ = xStorageShape.GetDim(DIM_0);
        fusedALen_ = xStorageShape.GetDim(DIM_1);
        fusedB1Len_ = xStorageShape.GetShapeSize() / fusedB0Len_ / fusedALen_;
    }
    OP_CHECK_IF(CheckSmallShapesValid(fusedALen_) != ge::GRAPH_SUCCESS,
                OP_LOGE(opName, "CheckSmallShapesValid failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetDtypesAndCheckValid() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "GetDtypesAndCheckValid failed"),
                return ge::GRAPH_FAILED);

    OP_LOGD(opName, "GetShapeAttrsInfo success. Fused shape is (%ld, %ld, %ld)", fusedB0Len_, fusedALen_, fusedB1Len_);

    return ge::GRAPH_SUCCESS;
}

// training
ge::graphStatus BatchNormRegbaseTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(GetAttrsAndCheckValid() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Get attrs failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!isTraining_, OP_LOGI(context_, "This node only support training."),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(GetXYShapesAndCheckValid() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = Ops::NN::OpTiling::EnsureNotScalar(xShape->GetStorageShape());

    if (xFormat_ == FORMAT_NCHW) {
        r1_ = xStorageShape.GetDim(DIM_0);
        a_ = xStorageShape.GetDim(DIM_1);
        r0_ = xStorageShape.GetShapeSize() / r1_ / a_;
    } else if (xFormat_ == FORMAT_NCDHW) {
        r1_ = xStorageShape.GetDim(DIM_0);
        a_ = xStorageShape.GetDim(DIM_1);
        r0_ = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3) * xStorageShape.GetDim(DIM_4);
    } else if (xFormat_ == FORMAT_NHWC) {
        r1_ = xStorageShape.GetDim(DIM_0) * xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2);
        a_ = xStorageShape.GetDim(DIM_3);
        r0_ = CONST_ONE;
    } else if (xFormat_ == FORMAT_NDHWC) {
        r1_ = xStorageShape.GetDim(DIM_0) * xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2) *
              xStorageShape.GetDim(DIM_3);
        a_ = xStorageShape.GetDim(DIM_4);
        r0_ = CONST_ONE;
    } else if (xFormat_ == FORMAT_ND) {
        r1_ = xStorageShape.GetDim(DIM_0);
        a_ = xStorageShape.GetDim(DIM_1);
        r0_ = xStorageShape.GetShapeSize() / r1_ / a_;
    }
    OP_CHECK_IF(CheckSmallShapesValid(a_) != ge::GRAPH_SUCCESS, OP_LOGE(opName, "CheckSmallShapesValid failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetDtypesAndCheckValid() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "GetDtypesAndCheckValid failed"),
                return ge::GRAPH_FAILED);

    useRunningMeanVar_ = Isclose(exponentialAvgFactor_, DEFAULT_EXPONENTIAL_AVG_FACTOR) ? 0 : CONST_ONE;

    OP_LOGD(opName, "GetShapeAttrsInfo success. Fused shape is (%ld, %ld, %ld), useRunningMeanVar: %d.", r1_, a_, r0_,
            useRunningMeanVar_);

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling