/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_tiling_infer_base.cpp
 * \brief
 */

#include "batch_norm_grad_tiling.h"
#include "batch_norm_grad_tiling_infer_base.h"

using namespace ge;
namespace optiling
{
constexpr int64_t VL_DOUBLE = 2;

static std::vector<std::array<ge::DataType, INPUT_NUM>> validInferInputDtype = {
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}
};


void BatchNormGradInferBase::Reset()
{
    usedCoreNums_ = 0;
    blockSize_ = 0;
    vlFp32_ = 0;
    vlFp16_ = 0;

    bytesPerDy_ = 0;
    bytesPerWeight_ = 0;
    bytesPerRunningVar_ = 0;

    aTileBase_ = 0;
    r1Dim = 0;
    r0Dim = 0;
    aDim = 0;

    dyDimNum_ = 0;
    epsilon_ = DEFAULT_EPSILON;
    enableDx = false;
    enableDgamma = false;
    enableDbeta = false;
}

void BatchNormGradInferBase::CalcBasicInfo()
{
    if (dyDtype_ == ge::DT_FLOAT) {
        bytesPerDy_ = FLOAT32_BYTES;
        aTileBase_ = vlFp32_;
    } else {
        bytesPerDy_ = FLOAT16_BYTES;
        aTileBase_ = vlFp16_;
    }

    if (weightDtype_ == ge::DT_FLOAT) {
        bytesPerWeight_ = FLOAT32_BYTES;
    } else {
        bytesPerWeight_ = FLOAT16_BYTES;
    }

    if (runningVarDtype_ == ge::DT_FLOAT) {
        bytesPerRunningVar_ = FLOAT32_BYTES;
    } else {
        bytesPerRunningVar_ = FLOAT16_BYTES;
    }

    OP_LOGD(context_, "aTileBase_: %ld, bytesPerDy_: %ld, bytesPerWeight_: %ld,bytesPerRunningVar_: %ld.", aTileBase_,
            bytesPerDy_, bytesPerWeight_, bytesPerRunningVar_);
}

ge::graphStatus BatchNormGradInferBase::GetPlatformInfo()
{
    auto compileInfo = static_cast<const BatchNormGradCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    blockSize_ = compileInfo->blockSize;
    vlFp32_ = compileInfo->vlFp32;
    vlFp16_ = vlFp32_ * VL_DOUBLE;

    opName_ = context_->GetNodeName();
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.numBlocks = compileInfo->coreNum;
        aicoreParams_.ubSize = compileInfo->ubSize;
    }

    return ge::GRAPH_SUCCESS;
}

const gert::Shape& BatchNormGradInferBase::EnsureNotScalar(const gert::Shape& in_shape)
{
    static const gert::Shape g_vec_1_shape = {1};
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus BatchNormGradInferBase::CheckBigShapesFormatValid()
{
    auto dyDesc = context_->GetInputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    auto dyFormat = dyDesc->GetFormat().GetStorageFormat();
    auto dyShape = context_->GetInputShape(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyShape);
    auto dyStorageShape = EnsureNotScalar(dyShape->GetStorageShape());
    int64_t dyDimNum = dyStorageShape.GetDimNum();

    auto xDesc = context_->GetInputDesc(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto xFormat = xDesc->GetFormat().GetStorageFormat();
    auto xShape = context_->GetInputShape(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = EnsureNotScalar(xShape->GetStorageShape());
    int64_t xDimNum = xStorageShape.GetDimNum();

    auto dxDesc = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
    auto dxFormat = dxDesc->GetFormat().GetStorageFormat();
    auto dxShape = context_->GetOutputShape(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxShape);
    auto dxStorageShape = EnsureNotScalar(dxShape->GetStorageShape());
    int64_t dxDimNum = dxStorageShape.GetDimNum();

    // 校验dim相等
    OP_CHECK_IF(
        dyDimNum != xDimNum || dyDimNum != dxDimNum,
        OP_LOGE(context_,
            "Input Dy dim size [%ld], x dim size [%ld] and output dx dim size [%ld] should be same", dyDimNum, xDimNum,
            dxDimNum),
        return ge::GRAPH_FAILED);

    // 校验format
    OP_CHECK_IF(
        dyFormat != xFormat || dyFormat != dxFormat,
        OP_LOGE(context_,
            "Input y_backprop format [%s], x format [%s] and output x_backprop format [%s] should be same",
            ge::TypeUtils::FormatToSerialString(dyFormat).c_str(), ge::TypeUtils::FormatToSerialString(xFormat).c_str(),
            ge::TypeUtils::FormatToSerialString(dxFormat).c_str()),
        return ge::GRAPH_FAILED);

    if (dyFormat == FORMAT_NCHW || dyFormat == FORMAT_NHWC) {
        OP_CHECK_IF(
            dyDimNum != DIM_NUM_4,
            OP_LOGE(context_, "Dims should be 4 with format [%s].", ge::TypeUtils::FormatToSerialString(dyFormat).c_str()),
            return ge::GRAPH_FAILED);
    } else if (dyFormat == FORMAT_NCDHW || dyFormat == FORMAT_NDHWC) {
        OP_CHECK_IF(
            dyDimNum != DIM_NUM_5,
            OP_LOGE(context_, "Dims should be 5 with format [%s].", ge::TypeUtils::FormatToSerialString(dyFormat).c_str()),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context_, "Not supported format [%s].", ge::TypeUtils::FormatToSerialString(dyFormat).c_str());
        return ge::GRAPH_FAILED;
    }

    // 校验shape的dims相同
    for (int64_t i = 0; i < dyDimNum; i++) {
        OP_CHECK_IF(
            xStorageShape.GetDim(i) != dyStorageShape.GetDim(i) || dxStorageShape.GetDim(i) != dyStorageShape.GetDim(i),
            OP_LOGE(context_,
                "Input y_backprop dim[%ld]: %ld, x dim[%ld]: %ld and output x_backprop dim[%ld]: %ld should be same.",
                i, dyStorageShape.GetDim(i), i, xStorageShape.GetDim(i), i, dxStorageShape.GetDim(i)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            dyStorageShape.GetDim(i) <= 0, OP_LOGE(context_, "Not support dim[%ld]: %ld.", i, dyStorageShape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::CheckSmallShapesValid()
{
    auto weightDesc = context_->GetInputDesc(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    weightDtype_ = weightDesc->GetDataType();
    auto dyDesc = context_->GetInputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    dyDtype_ = dyDesc->GetDataType();

    for (int i = INDEX_2; i <= INDEX_4; i++) {
        auto shape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        auto storageShape = shape->GetStorageShape();
        OP_CHECK_IF(
            storageShape.GetDimNum() != 1, OP_LOGE(context_, "Dims of input %d should be one.", i), return ge::GRAPH_FAILED);

        int64_t a = storageShape.GetDim(INDEX_0);
        OP_CHECK_IF(
            a != aDim, OP_LOGE(context_, "Shape of input %d should be %ld, actual %ld.", i, aDim, a), return ge::GRAPH_FAILED);
    }

    for (int i = INDEX_1; i <= INDEX_2; i++) {
        auto shape = context_->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        auto storageShape = shape->GetStorageShape();

        OP_CHECK_IF(
            storageShape.GetDimNum() != 1, OP_LOGE(context_, "Dims of output %d should be one.", i), return ge::GRAPH_FAILED);
        int64_t a = storageShape.GetDim(INDEX_0);
        OP_CHECK_IF(
            a != aDim, OP_LOGE(context_, "Shape of output %d should be %ld, actual %ld.", i, aDim, a), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::CheckDtypeValid()
{
    bool invalid = true;
    for (auto& dtypeList : validInferInputDtype) {
        invalid = false;
        for (int i = 0; i < INPUT_NUM; i++) {
            auto inputDesc = (i == INPUT_NUM - 1) ? context_->GetOptionalInputDesc(i) : context_->GetInputDesc(i);
            if (i == INPUT_NUM - 1 && inputDesc == nullptr) {
                continue;
            }
            OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
            auto dtype = inputDesc->GetDataType();
            if (dtype != dtypeList[i]) {
                invalid = true;
                break;
            }
        }
        if (!invalid) {
            break;
        }
    }
    OP_CHECK_IF(invalid == true, OP_LOGE(context_, "dtype of inputs are invalid, please check."), return ge::GRAPH_FAILED);

    auto weightDesc = context_->GetInputDesc(PARAM_INPUT_WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    weightDtype_ = weightDesc->GetDataType();

    auto runningVarDesc = context_->GetInputDesc(PARAM_INPUT_RUNNINGVAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningVarDesc);
    runningVarDtype_ = runningVarDesc->GetDataType();

    auto dxDesc = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
    ge::DataType dxDtype = dxDesc->GetDataType();
    auto dweightDesc = context_->GetOutputDesc(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dweightDesc);
    ge::DataType dweightDtype = dweightDesc->GetDataType();
    auto dbiasDesc = context_->GetOutputDesc(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dbiasDesc);
    ge::DataType dbiasDtype = dbiasDesc->GetDataType();
    OP_CHECK_IF(
        dxDtype != dyDtype_,
        OP_LOGE(context_, 
            "dtype of x_backprop should be same as y_backprop, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dxDtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dweightDtype != weightDtype_,
        OP_LOGE(context_, 
            "dtype of scale_backprop should be same as scale, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dweightDtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dbiasDtype != weightDtype_,
        OP_LOGE(context_, 
            "dtype of offset_backprop should be same as scale, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dbiasDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("BatchNormGradInferBase", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const bool* isTrainingPtr = attrs->GetBool(PARAM_ATTRS_ISTRAING_INDEX);
    bool isTraining = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;
    OP_CHECK_IF(isTraining == true, OP_LOGD(context_, "This node only support inference."), return ge::GRAPH_PARAM_INVALID);

    const float* epsilonPtr = attrs->GetFloat(PARAM_ATTRS_EPSILON_INDEX);
    epsilon_ = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;

    auto outputMask = attrs->GetAttrPointer<gert::ContinuousVector>(PARAM_ATTRS_OUTPUT_MASK);
    if (outputMask == nullptr) {
        enableDx = true;
        enableDgamma = false;
        enableDbeta = false;
    } else {
        OP_CHECK_IF(outputMask->GetSize() != 3, OP_LOGE(context_, "output_mask's size shuold be three"),
                    return ge::GRAPH_PARAM_INVALID);
        auto outputMaskData = static_cast<const bool*>(outputMask->GetData());
        enableDx = static_cast<bool>(outputMaskData[0]);
        enableDgamma = static_cast<bool>(outputMaskData[1]);
        enableDbeta = static_cast<bool>(outputMaskData[2]);
    }

    // 大shape shape和format检查
    OP_CHECK_IF(
        CheckBigShapesFormatValid() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check shape or format error."),
        return ge::GRAPH_FAILED);

    auto ret = GetDyInfo();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetShapeAttrsInfo failed."), return ret);

    // 小shape 校验
    OP_CHECK_IF(
        CheckSmallShapesValid() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Input small shapes are invalid."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckDtypeValid() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Input dtypes are invalid."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::GetDyInfo()
{
    auto dyDesc = context_->GetInputDesc(PARAM_INPUT_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    dyDtype_ = dyDesc->GetDataType();
    auto dyShape = context_->GetInputShape(PARAM_INPUT_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyShape);
    auto dyStorageShape = dyShape->GetStorageShape();
    dyDimNum_ = dyStorageShape.GetDimNum();

    dyFormat_ = dyDesc->GetFormat().GetStorageFormat();
    if (dyFormat_ == FORMAT_NHWC) {
        OP_CHECK_IF(dyDimNum_ != DIM_NUM_4, OP_LOGE(context_, "Dims should be 4 with NHWC format."), return ge::GRAPH_FAILED);
        aDim = dyStorageShape.GetDim(DIM_3);
        r1Dim = dyStorageShape.GetDim(DIM_0) * dyStorageShape.GetDim(DIM_1) * dyStorageShape.GetDim(DIM_2);
        r0Dim = 1;
    } else if (dyFormat_ == FORMAT_NDHWC) {
        OP_CHECK_IF(dyDimNum_ != DIM_NUM_5, OP_LOGE(context_, "Dims should be 5 with NDHWC format."), return ge::GRAPH_FAILED);
        aDim = dyStorageShape.GetDim(DIM_4);
        r1Dim = dyStorageShape.GetDim(DIM_0) * dyStorageShape.GetDim(DIM_1) * dyStorageShape.GetDim(DIM_2) *
                      dyStorageShape.GetDim(DIM_3);
        r0Dim = 1;
    } else if (dyFormat_ == FORMAT_NCHW) {
        OP_CHECK_IF(dyDimNum_ != DIM_NUM_4, OP_LOGE(context_, "Dims should be 4 with NCHW format."), return ge::GRAPH_FAILED);
        r1Dim = dyStorageShape.GetDim(DIM_0);
        aDim = dyStorageShape.GetDim(DIM_1);
        r0Dim = dyStorageShape.GetDim(DIM_2) * dyStorageShape.GetDim(DIM_3);
    } else if (dyFormat_ == FORMAT_NCDHW) {
        OP_CHECK_IF(dyDimNum_ != DIM_NUM_5, OP_LOGE(context_, "Dims should be 5 with NCDHW format."), return ge::GRAPH_FAILED);
        r1Dim = dyStorageShape.GetDim(DIM_0);
        aDim = dyStorageShape.GetDim(DIM_1);
        r0Dim = dyStorageShape.GetDim(DIM_2) * dyStorageShape.GetDim(DIM_3) * dyStorageShape.GetDim(DIM_4);
    } else {
        OP_LOGE(context_, "BatchNormGrad infer only support format NCHW/NCDHW/NHWC/HCDHW.");
        return ge::GRAPH_PARAM_INVALID;
    }
    // 特殊场景处理, R1AR0, A==1时转（1,1,R1*R0)
    if (aDim == 1) {
        r0Dim = r0Dim * r1Dim;
        r1Dim = 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
