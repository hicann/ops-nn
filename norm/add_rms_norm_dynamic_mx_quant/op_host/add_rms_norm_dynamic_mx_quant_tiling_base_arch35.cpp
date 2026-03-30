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
 * \file add_rms_norm_dynamic_mx_quant_tiling_base_arch35.cpp
 * \brief
 */
#include "add_rms_norm_dynamic_mx_quant_tiling.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

namespace optiling {
using namespace NormCheck;

MxRoundMode AddRmsNormDynamicMxQuantRegbaseTilingBase::ParseRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return MxRoundMode::RINT;
    } else if (roundMode == "round") {
        return MxRoundMode::ROUND;
    } else if (roundMode == "floor") {
        return MxRoundMode::FLOOR;
    }
    return MxRoundMode::UNDEFINED;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::CheckShapeNull()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckShapeNull.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);

    OP_CHECK_IF((nullptr == x1Shape) || (nullptr == x2Shape) || (nullptr == gammaShape),
        OP_LOGE(context_->GetNodeName(), "Input shape is nullptr, please check."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::CheckOptionalInput()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckOptionalInput.");
    const gert::StorageShape* betaShape = context_->GetOptionalInputShape(BETA_INDEX);
    betaFlag_ = 0;
    if (betaShape != nullptr) {
        int64_t betaShapeSize = betaShape->GetOriginShape().GetShapeSize();
        if (betaShapeSize > 0) {
            betaFlag_ = 1;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::CheckInputShapeDim()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckInputShapeDim.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);

    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (x1DimNum > MAX_DIM_CNT) || (x2DimNum > MAX_DIM_CNT) || (x1DimNum < 1) || (x2DimNum < 1),
        OP_LOGE(context_->GetNodeName(), "Input x1/x2 dim should be greater than 0,"
            "and not bigger than %u.", MAX_DIM_CNT), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckDimBiggerZero(x1Shape, x1DimNum, nodeName, "x1"),
        OP_LOGE(context_->GetNodeName(), "Input x1 shape is invalid, please check."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckDimBiggerZero(x2Shape, x2DimNum, nodeName, "x2"),
        OP_LOGE(context_->GetNodeName(), "Input x2 shape is invalid, please check."),return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::CheckInputShapeValue()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckInputShapeValue.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaShape);

    // x1 and x2 shapes must be equal
    if (!NormCheck::CheckShapeSame(x1Shape, x2Shape, nodeName, "x1", "x2")) {
        OP_LOGE(context_->GetNodeName(), "Input x1 shape is not same with x2 shape.");
        return ge::GRAPH_FAILED;
    }

    // gamma dim num shoule be 1
    if (1 != gammaShape->GetStorageShape().GetDimNum()) {
        OP_LOGE(context_->GetNodeName(), "The shape dim of gamma/beta only support 1, please check.");
        return ge::GRAPH_FAILED;
    }

    // gamma should match last dim of x
    if (!NormCheck::CheckShapeBC(x1Shape, gammaShape, nodeName, "x1", "gamma", true)) {
        OP_LOGE(context_->GetNodeName(), "Input gamma shape value is not valid with x shape value.");
        return ge::GRAPH_FAILED;
    }

    // If beta exists, it should match gamma shape
    if (betaFlag_) {
        const gert::StorageShape* betaShape = context_->GetOptionalInputShape(BETA_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, betaShape);
        if (!NormCheck::CheckShapeSame(gammaShape, betaShape, nodeName, "gamma", "beta")) {
            OP_LOGE(context_->GetNodeName(), "Input beta shape is not same with gamma shape.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::CheckInputDtype()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckInputDtype.");
    std::set<ge::DataType> supportedXDtypes = {ge::DT_FLOAT16, ge::DT_BF16};
    std::set<ge::DataType> supportedGammaDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputTensor(X1_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputTensor(X2_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputTensor(GAMMA_INDEX));
    ge::DataType x1Dtype = context_->GetInputTensor(X1_INDEX)->GetDataType();
    ge::DataType x2Dtype = context_->GetInputTensor(X2_INDEX)->GetDataType();
    ge::DataType gammaDtype = context_->GetInputTensor(GAMMA_INDEX)->GetDataType();

    // x1 and x2 must have same dtype
    if (x1Dtype != x2Dtype) {
        OP_LOGE(context_->GetNodeName(), "Input x1/x2 dtype should be equal.");
        return ge::GRAPH_FAILED;
    }
    // x dtype must be FP16 or BF16
    if (supportedXDtypes.count(x1Dtype) == 0) {
        OP_LOGE(context_->GetNodeName(), "Input x1/x2 dtype should be float16 or bfloat16.");
        return ge::GRAPH_FAILED;
    }
    // gamma must be FP16/BF16/FP32
    if (supportedGammaDtypes.count(gammaDtype) == 0) {
        OP_LOGE(context_->GetNodeName(), "Input gamma dtype should be float16, bfloat16 or float32.");
        return ge::GRAPH_FAILED;
    }
    if (betaFlag_) {
        ge::DataType betaDtype = context_->GetInputTensor(BETA_INDEX)->GetDataType();
        if (gammaDtype != betaDtype) {
            OP_LOGE(context_->GetNodeName(), "Input gamma/beta dtype should be equal.");
            return ge::GRAPH_FAILED;
        }
    }

    xDtype_ = x1Dtype;
    gammaDtype_ = gammaDtype;
    xDtypeSize_ = ge::GetSizeByDataType(x1Dtype);
    gammaDtypeSize_ = ge::GetSizeByDataType(gammaDtype);
    gammaIsFp32_ = (gammaDtype == ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::CheckOutputDtype()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckOutputDtype.");
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_INDEX));
    ge::DataType yDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();

    if (Y_SUPPORT_DTYPE_SET.count(yDtype) == 0) {
        OP_LOGE(context_->GetNodeName(), "Output y dtype should be FP4_E2M1/E1M2 or FP8_E4M3FN/E5M2.");
        return ge::GRAPH_FAILED;
    }

    yDtype_ = yDtype;

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(X_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(MXSCALE_INDEX));
    ge::DataType outputXDtype = context_->GetOutputDesc(X_INDEX)->GetDataType();
    ge::DataType mxScaleDtype = context_->GetOutputDesc(MXSCALE_INDEX)->GetDataType();
    if (outputXDtype != xDtype_){
        OP_LOGE(context_->GetNodeName(), "Output X dtype should be equal to input X1/X2 dtype.");
        return ge::GRAPH_FAILED;
    }
    if (mxScaleDtype != ge::DT_FLOAT8_E8M0){
        OP_LOGE(context_->GetNodeName(), "Output mxScale dtype should be FLOAT8_E8M0.");
        return ge::GRAPH_FAILED;
    }

    // output_rstd ATTR校验：设置rstd_flag
    rstdFlag_ = 0;
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const bool* outputRstdPtr = attrs->GetAttrPointer<bool>(OUTPUT_RSTD_ATTR_INDEX);
    if (outputRstdPtr != nullptr) {
        bool outputRstd = *outputRstdPtr;
        if (outputRstd == true) {
            rstdFlag_ = 1;
        }
    }

    if (rstdFlag_) {
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(RSTD_INDEX));
        ge::DataType rstdDtype = context_->GetOutputDesc(RSTD_INDEX)->GetDataType();
        if (rstdDtype != ge::DT_FLOAT){
            OP_LOGE(context_->GetNodeName(), "Output rstd dtype should be FLOAT32.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::CheckMxQuantParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckMxQuantParams.");
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    // 1. round_mode校验：必须为 rint/floor/round 之一
    const char* roundModeStr = attrs->GetAttrPointer<char>(ROUND_MODE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, roundModeStr);
    std::string rmStr(roundModeStr);
    MxRoundMode rm = ParseRoundMode(rmStr);

    OP_CHECK_IF(
        (rm == MxRoundMode::UNDEFINED),
        OP_LOGE(context_->GetNodeName(),
                "invalid round_mode:%s; round_mode should be one of {rint, floor, round}", roundModeStr),
        return ge::GRAPH_FAILED);

    // FP8输出类型仅支持rint
    OP_CHECK_IF(
        (Y_SUPPORT_DTYPE_FP8_SET.count(yDtype_) != 0 && rm != MxRoundMode::RINT),
        OP_LOGE(context_->GetNodeName(),
                "When output y's data type is FLOAT8_E4M3FN/FLOAT8_E5M2, round_mode:[%s] only support rint, "
                "please check.",
                roundModeStr),
        return ge::GRAPH_FAILED);

    // 2. dst_type校验：必须与y的dtype对应
    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(DST_TYPE_ATTR_INDEX);
    if (dstTypePtr != nullptr) {
        int64_t dstType = *dstTypePtr;
        if ((yDtype_ == ge::DT_FLOAT4_E2M1 && dstType != DST_TYPE_E2M1) ||
            (yDtype_ == ge::DT_FLOAT4_E1M2 && dstType != DST_TYPE_E1M2) ||
            (yDtype_ == ge::DT_FLOAT8_E4M3FN && dstType != DST_TYPE_E4M3FN) ||
            (yDtype_ == ge::DT_FLOAT8_E5M2 && dstType != DST_TYPE_E5M2)) {
            OP_LOGE(context_->GetNodeName(),
                    "y's data type and dst_type is not corresponded. "
                    "FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 correspond to dst_type: 40/41/36/35.");
            return ge::GRAPH_FAILED;
        }
    }

    // 3. quant_alg校验：必须为0或1，FP4仅支持0
    const int64_t* quantAlgPtr = attrs->GetAttrPointer<int64_t>(QUANT_ALG_ATTR_INDEX);
    if (quantAlgPtr != nullptr) {
        int64_t quantAlg = *quantAlgPtr;
        if (quantAlg < 0 || quantAlg > 1) {
            OP_LOGE(context_->GetNodeName(), "The quant_alg[%ld] should be 0 or 1.", quantAlg);
            return ge::GRAPH_FAILED;
        }
        if (quantAlg == 1 && Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0) {
            OP_LOGE(context_->GetNodeName(),
                    "When y's data type is FLOAT4_E2M1/FLOAT4_E1M2, quant_alg must be set to 0.");
            return ge::GRAPH_FAILED;
        }
    }

    // 4. FP4输出时，x的最后一维必须为偶数
    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0) {
        const gert::Shape x1Shape = context_->GetInputShape(X1_INDEX)->GetStorageShape();
        size_t lastDim = x1Shape.GetDimNum() - 1;
        if (x1Shape.GetDim(lastDim) % NUM_TWO != 0) {
            OP_LOGE(context_->GetNodeName(),
                    "When output y's data type is FLOAT4_E2M1/FLOAT4_E1M2, "
                    "the last axis of x should be even, but got %ld.",
                    x1Shape.GetDim(lastDim));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::CheckOutputShapeValue()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckOutputShapeValue.");
    OP_CHECK_IF(CheckMxScaleRstdShape() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "MxScale or Rstd shape invalid."), return ge::GRAPH_FAILED);
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* outputXShape = context_->GetOutputShape(X_INDEX);
    const gert::StorageShape* outputYShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputXShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYShape);

    // outputX and inputX1 shapes must be equal
    if (!NormCheck::CheckShapeSame(x1Shape, outputXShape, nodeName, "inputX1", "outputX")) {
        OP_LOGE(context_->GetNodeName(), "Output X shape is not same with inputX1 shape.");
        return ge::GRAPH_FAILED;
    }

    // outputY and inputX1 shapes must be equal
    if (!NormCheck::CheckShapeSame(x1Shape, outputYShape, nodeName, "inputX1", "outputY")) {
        OP_LOGE(context_->GetNodeName(), "Output Y shape is not same with inputX1 shape.");
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::CheckMxScaleRstdShape()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckMxScaleRstdShape.");
    const gert::StorageShape* x1ShapePtr = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* mxscaleShapePtr = context_->GetOutputShape(MXSCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, mxscaleShapePtr);

    const gert::Shape x1Shape = x1ShapePtr->GetStorageShape();
    const gert::Shape mxscaleShape = mxscaleShapePtr->GetStorageShape();
    const gert::Shape gammaShape = context_->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    size_t xRank = x1Shape.GetDimNum();
    size_t mxscaleRank = mxscaleShape.GetDimNum();
    size_t gammaRank = gammaShape.GetDimNum();

    // mxscale rank必须等于xRank + 1
    if (mxscaleRank != xRank + 1) {
        OP_LOGE(context_->GetNodeName(), "Output mxscale rank [%zu] should be equal to xRank + 1 [%zu].", 
            mxscaleRank, xRank + 1);
        return ge::GRAPH_FAILED;
    }

    // A维度的轴必须一致
    for (size_t i = 0; i < xRank - gammaRank; i++) {
        if (mxscaleShape.GetDim(i) != x1Shape.GetDim(i)) {
            OP_LOGE(context_->GetNodeName(),
                    "mxscale shape dim[%zu]=%ld should match x1 shape dim[%zu]=%ld.",
                    i, mxscaleShape.GetDim(i), i, x1Shape.GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }

    // MxScale输出的 (-2轴, -1轴) shape value: (CeilDiv(CeilDiv(R, 32), 2), 2)
    uint64_t expectedLastDim = Ops::Base::CeilDiv(
        Ops::Base::CeilDiv(numCol_, static_cast<uint64_t>(MX_BLOCK_SIZE_32)), static_cast<uint64_t>(NUM_TWO));
    if (mxscaleShape.GetDim(mxscaleRank - NUM_TWO) != static_cast<int64_t>(expectedLastDim)) {
        OP_LOGE(context_->GetNodeName(), "mxscale -2last dim[%ld] should be CeilDiv(CeilDiv(R=%ld, 32), 2) = %ld.",
                mxscaleShape.GetDim(mxscaleRank - NUM_TWO), static_cast<int64_t>(numCol_), static_cast<int64_t>(expectedLastDim));
        return ge::GRAPH_FAILED;
    }
    if (mxscaleShape.GetDim(mxscaleRank - 1) != static_cast<int64_t>(NUM_TWO)) {
        OP_LOGE(context_->GetNodeName(), "mxscale last dim[%ld] should be 2.", mxscaleShape.GetDim(mxscaleRank - 1));
        return ge::GRAPH_FAILED;
    }

    if (rstdFlag_) {
        const gert::StorageShape* rstdShapePtr = context_->GetOutputShape(RSTD_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, rstdShapePtr);
        const gert::Shape rstdShape = rstdShapePtr->GetStorageShape();
        OP_CHECK_IF(
            (rstdShape.GetDimNum() != xRank),
            OP_LOGE(context_->GetNodeName(), "Invalid rstd shape dim num (must same with input x1)."),
                return ge::GRAPH_FAILED);
        // A维度的轴必须一致，其他维度为1
        for (size_t i = 0; i < xRank - gammaRank; i++) {
            if (i >= xRank - gammaRank) {
                if (rstdShape.GetDim(i) != 1) {
                    OP_LOGE(context_->GetNodeName(), "rstd shape dim[%zu]=%ld should be 1.", i, rstdShape.GetDim(i));
                    return ge::GRAPH_FAILED;
                }
            } else {
                if (rstdShape.GetDim(i) != x1Shape.GetDim(i)) {
                    OP_LOGE(context_->GetNodeName(),
                            "rstd shape dim[%zu]=%ld should match x1 shape dim[%zu]=%ld.",
                            i, rstdShape.GetDim(i), i, x1Shape.GetDim(i));
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::SetInputParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter SetInputParams.");
    // Compute numRow (A) and numCol (R) from shapes
    const gert::Shape x1Shape = context_->GetInputShape(X1_INDEX)->GetStorageShape();
    const gert::Shape gammaShape = context_->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    size_t x1DimNum = x1Shape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();

    uint64_t numRow = 1;
    uint64_t numCol = 1;
    for (size_t i = 0; i < x1DimNum - gammaDimNum; i++) {
        numRow *= x1Shape.GetDim(i);
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        numCol *= gammaShape.GetDim(i);
    }
    numRow_ = numRow;
    numCol_ = numCol;

    OP_CHECK_IF((numRow_ == 0 || numCol_ == 0), // 任意维度的0校验掉
        OP_LOGE(context_->GetNodeName(), "Input shape not support zero dim value in any axis, please check."),
        return ge::GRAPH_FAILED);

    // R-axis alignment
    numColAlign_ = Ops::Base::CeilAlign(numCol_, COL_ALIGN_NUM);

    // Parse attributes
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const float* epsilonPtr = attrs->GetFloat(EPS_ATTR_INDEX);
    if (epsilonPtr != nullptr) {
        epsilon_ = *epsilonPtr;
    }

    const char* roundModeStr = attrs->GetAttrPointer<char>(ROUND_MODE_ATTR_INDEX);
    if (roundModeStr != nullptr) {
        roundMode_ = static_cast<uint64_t>(ParseRoundMode(std::string(roundModeStr)));
    }

    const uint64_t* quantAlgPtr = attrs->GetAttrPointer<uint64_t>(QUANT_ALG_ATTR_INDEX);
    if (quantAlgPtr != nullptr) {
        scaleAlg_ = *quantAlgPtr;
    }

    avgFactor_ = (numCol == 0) ? 0.0f : 1.0f / static_cast<float>(numCol);

    // MX quant derived params
    mxBlockSize_ = MX_BLOCK_SIZE_32;
    blockNumInColAxis_ = Ops::Base::CeilDiv(numColAlign_, mxBlockSize_);
    // mxscale output size per row: CeilAlign(CeilDiv(R, 32), 2) for even-pad
    mxScaleSize_ = Ops::Base::CeilAlign(blockNumInColAxis_, static_cast<uint64_t>(NUM_TWO));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::GetShapeAttrsInfo()
{
    // Get vector/block sizes
    OP_LOGD(context_->GetNodeName(), "Enter GetShapeAttrsInfo.");
    uint64_t vecLength = Ops::Base::GetVRegSize(context_);
    OP_CHECK_IF((vecLength <= 0),OP_LOGE(context_, "Get vector Length failed, vector Length: %u",
        static_cast<uint32_t>(vecLength)), return ge::GRAPH_FAILED);
    vecLengthFP32_ = vecLength / FP32_SIZE;
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF((ubBlockSize_ <= 0), OP_LOGE(context_, "Get block Size failed, block size: %u",
        static_cast<uint32_t>(ubBlockSize_)), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeNull() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Required input is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOptionalInput() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Optional input check failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeDim() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Input shape dim invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeValue() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Input shape value invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Input dtype invalid."), return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(CheckOutputDtype() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Output dtype invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckMxQuantParams() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "MX quant params invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetInputParams() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "SetInputParams failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputShapeValue() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Output shape invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    auto compileInfoPtr = reinterpret_cast<const AddRmsNormDynamicMxQuantCompileInfo*>(context_->GetCompileInfo());
    if (compileInfoPtr != nullptr) {
        totalCoreNum_ = compileInfoPtr->totalCoreNum;
        maxUbSize_ = compileInfoPtr->totalUbSize;
    } else {
        OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        maxUbSize_ = ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantRegbaseTilingBase::GetWorkspaceSize()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
