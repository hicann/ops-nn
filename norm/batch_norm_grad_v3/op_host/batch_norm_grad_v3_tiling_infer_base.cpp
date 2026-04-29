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
 * \file batch_norm_grad_v3_tiling_infer_base.cpp
 * \brief
 */

#include "batch_norm_grad_v3_tiling.h"
#include "op_api/op_util.h"
#include "batch_norm_grad_v3_tiling_infer_base.h"
#include "log/log.h"

using namespace ge;
namespace optiling {
constexpr int64_t VL_DOUBLE = 2;

void BatchNormGradV3InferBase::Reset()
{
    usedCoreNums_ = 0;
    blockSize_ = 0;
    vlFp32_ = 0;
    vlFp16_ = 0;

    bytesPerDy_ = 0;
    bytesPerWeight_ = 0;
    bytesPerRunningVar_ = 0;

    aTileBase_ = 0;
    fusedB0Len_ = 0;
    fusedB1Len_ = 0;
    fusedALen_ = 0;

    dyDimNum_ = 0;
    weightDimNum_ = 0;
    runningVarDimNum_ = 0;
    dxDimNum_ = 0;

    weightDimLen_ = 0;
    runningVarDimLen_ = 0;

    epsilon_ = DEFAULT_EPSILON_VAL;
}

void BatchNormGradV3InferBase::CalcBasicInfo()
{
    if (dyDtype_ == ge::DT_FLOAT) {
        bytesPerDy_ = FLOAT32_BYTES;
        aTileBase_ = vlFp32_;
    } else {
        bytesPerDy_ = FLOAT16_BYTES;
        aTileBase_ = vlFp16_;
    }

    if (runningVarDtype_ == ge::DT_FLOAT) {
        bytesPerRunningVar_ = FLOAT32_BYTES;
    } else {
        bytesPerRunningVar_ = FLOAT16_BYTES;
    }

    if (weightDtype_ == ge::DT_FLOAT) {
        bytesPerWeight_ = FLOAT32_BYTES;
    } else {
        bytesPerWeight_ = FLOAT16_BYTES;
    }

    OP_LOGD(
        context_->GetNodeName(), "aTileBase_: %ld, bytesPerDy_: %ld, bytesPerWeight_: %ld,bytesPerRunningVar_: %ld.",
        aTileBase_, bytesPerDy_, bytesPerWeight_, bytesPerRunningVar_);
}

ge::graphStatus BatchNormGradV3InferBase::GetPlatformInfo()
{
    auto compileInfo = static_cast<const BatchNormGradV3CompileInfo*>(context_->GetCompileInfo());
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

ge::graphStatus BatchNormGradV3InferBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("BatchNormGradV3InferBase", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const bool* isTrainingPtr = attrs->GetBool(0);
    bool isTraining = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;
    OP_TILING_CHECK(
        isTraining == true, OP_LOGI(context_->GetNodeName(), "This node only support inference."),
        return ge::GRAPH_PARAM_INVALID);

    const float* epsilonPtr = attrs->GetFloat(PARAM_ATTRS_EPSILON_INDEX);
    epsilon_ = (epsilonPtr == nullptr) ? DEFAULT_EPSILON_VAL : *epsilonPtr;

    auto ret = GetDyInfo();
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS, OP_LOGW(context_->GetNodeName(), "GetShapeAttrsInfo failed."), return ret);

    OP_TILING_CHECK(
        GetWeightRunningVarDxInfo() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Inputs are invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        CheckInputValid() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Inputs are invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3InferBase::GetDyInfo()
{
    auto dyDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    dyDtype_ = dyDesc->GetDataType();
    auto dyShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyShape);
    auto dyStorageShape = dyShape->GetStorageShape();
    dyDimNum_ = dyStorageShape.GetDimNum();

    dyFormat_ = dyDesc->GetFormat().GetStorageFormat();
    if (dyFormat_ == FORMAT_NHWC) {
        OP_CHECK_IF(
            dyDimNum_ != DIM_NUM_4,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "dy",
                std::to_string(dyDimNum_).c_str(), "dy should be 4D with NHWC format"),
            return ge::GRAPH_FAILED);
        fusedALen_ = dyStorageShape.GetDim(DIM_3);
        fusedB0Len_ = dyStorageShape.GetDim(DIM_0) * dyStorageShape.GetDim(DIM_1) * dyStorageShape.GetDim(DIM_2);
        fusedB1Len_ = 1;
    } else if (dyFormat_ == FORMAT_NDHWC) {
        OP_CHECK_IF(
            dyDimNum_ != DIM_NUM_5,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "dy",
                std::to_string(dyDimNum_).c_str(), "dy should be 5D with NDHWC format"),
            return ge::GRAPH_FAILED);
        fusedALen_ = dyStorageShape.GetDim(DIM_4);
        fusedB0Len_ = dyStorageShape.GetDim(DIM_0) * dyStorageShape.GetDim(DIM_1) * dyStorageShape.GetDim(DIM_2) *
                      dyStorageShape.GetDim(DIM_3);
        fusedB1Len_ = 1;
    } else if (dyFormat_ == FORMAT_NCHW) {
        OP_CHECK_IF(dyDimNum_ != DIM_NUM_4,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "dy",
                std::to_string(dyDimNum_).c_str(), "dy should be 4D with NCHW format"),
            return ge::GRAPH_FAILED);
        fusedB0Len_ = dyStorageShape.GetDim(DIM_0);
        fusedALen_ = dyStorageShape.GetDim(DIM_1);
        fusedB1Len_ = dyStorageShape.GetDim(DIM_2) * dyStorageShape.GetDim(DIM_3);
    } else if (dyFormat_ == FORMAT_NCDHW) {
        OP_CHECK_IF(dyDimNum_ != DIM_NUM_5,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "dy",
                std::to_string(dyDimNum_).c_str(), "dy should be 5D with NCDHW format"),
            return ge::GRAPH_FAILED);
        fusedB0Len_ = dyStorageShape.GetDim(DIM_0);
        fusedALen_ = dyStorageShape.GetDim(DIM_1);
        fusedB1Len_ = dyStorageShape.GetDim(DIM_2) * dyStorageShape.GetDim(DIM_3) * dyStorageShape.GetDim(DIM_4);
    } else {
        OP_LOGI(context_->GetNodeName(), "batch norm grad infer only support format NCHW/NCDHW/NHWC/HCDHW.");
        return ge::GRAPH_PARAM_INVALID;
    }

    // 特殊场景处理, B0AB1, A==1时转（1,1,B0*B1)
    if (fusedALen_ == 1) {
        fusedB1Len_ = fusedB1Len_ * fusedB0Len_;
        fusedB0Len_ = 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3InferBase::GetWeightRunningVarDxInfo()
{
    auto weightTensorDesc = context_->GetInputDesc(PARAM_INPUT_WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightTensorDesc);
    weightDtype_ = weightTensorDesc->GetDataType();
    auto weightTensorShape = context_->GetInputShape(PARAM_INPUT_WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightTensorShape);
    auto weightTensorStorageShape = weightTensorShape->GetStorageShape();
    weightDimNum_ = weightTensorStorageShape.GetDimNum();
    weightDimLen_ = weightTensorStorageShape.GetDim(0);

    auto runningVarTensorDesc = context_->GetInputDesc(PARAM_INPUT_RUNNINGVAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningVarTensorDesc);
    runningVarDtype_ = runningVarTensorDesc->GetDataType();
    auto runningVarTensorShape = context_->GetInputShape(PARAM_INPUT_RUNNINGVAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningVarTensorShape);
    auto runningVarTensorStorageShape = runningVarTensorShape->GetStorageShape();
    runningVarDimNum_ = runningVarTensorStorageShape.GetDimNum();
    runningVarDimLen_ = runningVarTensorStorageShape.GetDim(0);

    auto dxTensorDesc = context_->GetOutputDesc(PARAM_OUTPUT_DX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxTensorDesc);
    dxDtype_ = dxTensorDesc->GetDataType();
    auto dxTensorShape = context_->GetOutputShape(PARAM_OUTPUT_DX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxTensorShape);
    auto dxTensorStorageShape = dxTensorShape->GetStorageShape();
    dxDimNum_ = dxTensorStorageShape.GetDimNum();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3InferBase::CheckInputValid()
{
    OP_CHECK_IF(
        dyDtype_ != dxDtype_,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "dy and dx",
            (Ops::Base::ToString(dyDtype_) + " and " + Ops::Base::ToString(dxDtype_)).c_str(),
            "dy and dx should have the same dtype"),
        return ge::GRAPH_FAILED);

    bool inputDtypeValid = false;
    for (uint32_t i = 0; i < validInputDtypes.size(); i++) {
        if (dyDtype_ == validInputDtypes[i][DTYPE_DX_OFFSET] &&
            weightDtype_ == validInputDtypes[i][DTYPE_WEIGHT_OFFSET] &&
            runningVarDtype_ == validInputDtypes[i][DTYPE_RUNNINGVAR_OFFSET]) {
            inputDtypeValid = true;
            break;
        }
    }

    OP_CHECK_IF(
        !inputDtypeValid,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "dy, weight and running_var",
            (Ops::Base::ToString(dyDtype_) + ", " + Ops::Base::ToString(weightDtype_) + " and " +
             Ops::Base::ToString(runningVarDtype_)).c_str(),
            "The dtypes of dy, weight and running_var are not within the supported (all float, all bf16, all fp16, "
            "(fp16, float, float), (bf16, float, float), (fp16, fp16, float) or (bf16, bf16, float)) combination "
            "range"),
        return ge::GRAPH_FAILED);

    bool inputShapeValid = weightDimNum_ == runningVarDimNum_ && dyDimNum_ == dxDimNum_ && weightDimNum_ == 1 &&
                      weightDimLen_ == runningVarDimLen_ && weightDimLen_ == fusedALen_;

    OP_CHECK_IF(
        !inputShapeValid,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "dy, weight, running_var and dx",
            (std::to_string(dyDimNum_) + ", " + std::to_string(weightDimNum_) + ", " +
             std::to_string(runningVarDimNum_) + ", " + std::to_string(dxDimNum_)).c_str(),
            "weight and running_var should be 1D, and dy and dx should have the same dim num; the first dim of weight "
            "and running_var should be equal, and they also should be equal to the second dim of dy"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3InferBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3InferBase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
