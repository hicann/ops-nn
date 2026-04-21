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
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "x1/x2",
            (std::to_string(x1DimNum) + "/" + std::to_string(x2DimNum)).c_str(), "1D to 7D"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckDimBiggerZero(x1Shape, x1DimNum, nodeName, "x1"),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x1", "", "all dims should be greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckDimBiggerZero(x2Shape, x2DimNum, nodeName, "x2"),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x2", "", "all dims should be greater than 0"),
        return ge::GRAPH_FAILED);
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
    OP_CHECK_IF(
        !NormCheck::CheckShapeSame(x1Shape, x2Shape, nodeName, "x1", "x2"),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x2",
            Ops::Base::ToString(x2Shape->GetStorageShape()).c_str(), "same as x1"),
        return ge::GRAPH_FAILED);

    // gamma dim num should be 1
    OP_CHECK_IF(
        1 != gammaShape->GetStorageShape().GetDimNum(),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "gamma",
            std::to_string(gammaShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
        return ge::GRAPH_FAILED);

    // gamma should match last dim of x
    OP_CHECK_IF(
        !NormCheck::CheckShapeBC(x1Shape, gammaShape, nodeName, "x1", "gamma", true),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "gamma",
            Ops::Base::ToString(gammaShape->GetStorageShape()).c_str(),
            "should match last dim of x1"),
        return ge::GRAPH_FAILED);

    // If beta exists, it should match gamma shape
    if (betaFlag_) {
        const gert::StorageShape* betaShape = context_->GetOptionalInputShape(BETA_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, betaShape);
        OP_CHECK_IF(
            !NormCheck::CheckShapeSame(gammaShape, betaShape, nodeName, "gamma", "beta"),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "beta",
                Ops::Base::ToString(betaShape->GetStorageShape()).c_str(), "same as gamma"),
            return ge::GRAPH_FAILED);
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
    std::string x2ReasonStr = "same as x1 (" + Ops::Base::ToString(static_cast<ge::DataType>(x1Dtype)) + ")";
    OP_CHECK_IF(
        x1Dtype != x2Dtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "x2", Ops::Base::ToString(static_cast<ge::DataType>(x2Dtype)).c_str(),
            x2ReasonStr.c_str()),
        return ge::GRAPH_FAILED);
    // x dtype must be FP16 or BF16
    OP_CHECK_IF(
        supportedXDtypes.count(x1Dtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", Ops::Base::ToString(static_cast<ge::DataType>(x1Dtype)).c_str(),
            "float16 or bfloat16"),
        return ge::GRAPH_FAILED);
    // gamma must be FP16/BF16/FP32
    OP_CHECK_IF(
        supportedGammaDtypes.count(gammaDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "gamma", Ops::Base::ToString(static_cast<ge::DataType>(gammaDtype)).c_str(),
            "float16, bfloat16 or float32"),
        return ge::GRAPH_FAILED);
    if (betaFlag_) {
        ge::DataType betaDtype = context_->GetInputTensor(BETA_INDEX)->GetDataType();
        std::string betaReasonStr = "same as gamma (" + Ops::Base::ToString(static_cast<ge::DataType>(gammaDtype)) + ")";
        OP_CHECK_IF(
            gammaDtype != betaDtype,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "beta", Ops::Base::ToString(static_cast<ge::DataType>(betaDtype)).c_str(),
                betaReasonStr.c_str()),
            return ge::GRAPH_FAILED);
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

    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "y", Ops::Base::ToString(static_cast<ge::DataType>(yDtype)).c_str(),
            "float4_e2m1, float4_e1m2, float8_e4m3fn or float8_e5m2"),
        return ge::GRAPH_FAILED);

    yDtype_ = yDtype;

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(X_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(MXSCALE_INDEX));
    ge::DataType outputXDtype = context_->GetOutputDesc(X_INDEX)->GetDataType();
    ge::DataType mxScaleDtype = context_->GetOutputDesc(MXSCALE_INDEX)->GetDataType();
    OP_CHECK_IF(
        outputXDtype != xDtype_,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "output_x", Ops::Base::ToString(static_cast<ge::DataType>(outputXDtype)).c_str(),
            "same as input x1/x2"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        mxScaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "mxscale", Ops::Base::ToString(static_cast<ge::DataType>(mxScaleDtype)).c_str(),
            "float8_e8m0"),
        return ge::GRAPH_FAILED);

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
        OP_CHECK_IF(
            rstdDtype != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(
                context_->GetNodeName(), "rstd", Ops::Base::ToString(static_cast<ge::DataType>(rstdDtype)).c_str(),
                "float"),
            return ge::GRAPH_FAILED);
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
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "round_mode", roundModeStr, "rint, round or floor"),
        return ge::GRAPH_FAILED);

    // FP8输出类型仅支持rint
    OP_CHECK_IF(
        (Y_SUPPORT_DTYPE_FP8_SET.count(yDtype_) != 0 && rm != MxRoundMode::RINT),
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "round_mode", roundModeStr, "rint"),
        return ge::GRAPH_FAILED);

    // 2. dst_type校验：必须与y的dtype对应
    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(DST_TYPE_ATTR_INDEX);
    if (dstTypePtr != nullptr) {
        int64_t dstType = *dstTypePtr;
        OP_CHECK_IF(
            (yDtype_ == ge::DT_FLOAT4_E2M1 && dstType != DST_TYPE_E2M1) ||
                (yDtype_ == ge::DT_FLOAT4_E1M2 && dstType != DST_TYPE_E1M2) ||
                (yDtype_ == ge::DT_FLOAT8_E4M3FN && dstType != DST_TYPE_E4M3FN) ||
                (yDtype_ == ge::DT_FLOAT8_E5M2 && dstType != DST_TYPE_E5M2),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), "dst_type", std::to_string(dstType).c_str(),
                "40/41/36/35 for float4_e2m1/float4_e1m2/float8_e4m3fn/float8_e5m2"),
            return ge::GRAPH_FAILED);
    }

    // 3. quant_alg校验：必须为0或1，FP4仅支持0
    const int64_t* quantAlgPtr = attrs->GetAttrPointer<int64_t>(QUANT_ALG_ATTR_INDEX);
    if (quantAlgPtr != nullptr) {
        int64_t quantAlg = *quantAlgPtr;
        OP_CHECK_IF(
            quantAlg < 0 || quantAlg > 1,
            OP_LOGE_WITH_INVALID_ATTR(
                context_->GetNodeName(), "quant_alg", std::to_string(quantAlg).c_str(), "0 or 1"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            quantAlg == 1 && Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0,
            OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "quant_alg", std::to_string(quantAlg).c_str(), "0"),
            return ge::GRAPH_FAILED);
    }

    // 4. FP4输出时，x的最后一维必须为偶数
    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0) {
        const gert::Shape x1Shape = context_->GetInputShape(X1_INDEX)->GetStorageShape();
        size_t lastDim = x1Shape.GetDimNum() - 1;
        OP_CHECK_IF(
            x1Shape.GetDim(lastDim) % NUM_TWO != 0,
            OP_LOGE_FOR_INVALID_SHAPESIZE(
                context_->GetNodeName(), "x1", std::to_string(x1Shape.GetDim(lastDim)).c_str(),
                "even number when y dtype is fp4"),
            return ge::GRAPH_FAILED);
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
    OP_CHECK_IF(
        !NormCheck::CheckShapeSame(x1Shape, outputXShape, nodeName, "inputX1", "outputX"),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "output_x",
            Ops::Base::ToString(outputXShape->GetStorageShape()).c_str(), "same as input x1"),
        return ge::GRAPH_FAILED);

    // outputY and inputX1 shapes must be equal
    OP_CHECK_IF(
        !NormCheck::CheckShapeSame(x1Shape, outputYShape, nodeName, "inputX1", "outputY"),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "output_y",
            Ops::Base::ToString(outputYShape->GetStorageShape()).c_str(), "same as input x1"),
        return ge::GRAPH_FAILED);
    
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
    OP_CHECK_IF(
        mxscaleRank != xRank + 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "mxscale", std::to_string(mxscaleRank).c_str(),
            std::to_string(xRank + 1).c_str()),
        return ge::GRAPH_FAILED);

    // A维度的轴必须一致
    for (size_t i = 0; i < xRank - gammaRank; i++) {
        OP_CHECK_IF(
            mxscaleShape.GetDim(i) != x1Shape.GetDim(i),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "mxscale",
                Ops::Base::ToString(mxscaleShape).c_str(),
                "batch dims should match x1"),
            return ge::GRAPH_FAILED);
    }

    // MxScale输出的 (-2轴, -1轴) shape value: (CeilDiv(CeilDiv(R, 32), 2), 2)
    uint64_t expectedLastDim = Ops::Base::CeilDiv(
        Ops::Base::CeilDiv(numCol_, static_cast<uint64_t>(MX_BLOCK_SIZE_32)), static_cast<uint64_t>(NUM_TWO));
    std::string mxscaleReasonStr = "mxscale -2nd dim should be CeilDiv(CeilDiv(R=" +
        std::to_string(numCol_) + ", 32), 2) = " + std::to_string(expectedLastDim);
    OP_CHECK_IF(
        mxscaleShape.GetDim(mxscaleRank - NUM_TWO) != static_cast<int64_t>(expectedLastDim),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "mxscale",
            Ops::Base::ToString(mxscaleShape).c_str(), mxscaleReasonStr.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        mxscaleShape.GetDim(mxscaleRank - 1) != static_cast<int64_t>(NUM_TWO),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "mxscale",
            Ops::Base::ToString(mxscaleShape).c_str(), "last dim should be 2"),
        return ge::GRAPH_FAILED);

    if (rstdFlag_) {
        const gert::StorageShape* rstdShapePtr = context_->GetOutputShape(RSTD_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, rstdShapePtr);
        const gert::Shape rstdShape = rstdShapePtr->GetStorageShape();
        OP_CHECK_IF(
            (rstdShape.GetDimNum() != xRank),
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                context_->GetNodeName(), "rstd",
                std::to_string(rstdShape.GetDimNum()).c_str(), std::to_string(xRank).c_str()),
            return ge::GRAPH_FAILED);
        // A维度的轴必须一致，其他维度为1
        for (size_t i = 0; i < xRank; i++) {
            if (i >= xRank - gammaRank) {
                OP_CHECK_IF(
                    rstdShape.GetDim(i) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "rstd",
                        Ops::Base::ToString(rstdShape).c_str(),
                        "norm dims should be 1"),
                    return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(
                    rstdShape.GetDim(i) != x1Shape.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "rstd",
                        Ops::Base::ToString(rstdShape).c_str(),
                        "batch dims should match x1"),
                    return ge::GRAPH_FAILED);
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

    const int64_t* quantAlgPtr = attrs->GetAttrPointer<int64_t>(QUANT_ALG_ATTR_INDEX);
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
