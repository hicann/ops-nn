/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file add_rms_norm_hi_f4_quant_tiling_base.cpp
 * \brief
 */
#include "../add_rms_norm_hi_f4_quant_tiling.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"
#include "../../../../../common/inc/op_host/op_const_def_nn.h"

namespace optiling {
using namespace NormCheck;

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::CheckShapeNull()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckShapeNull.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);

    OP_CHECK_IF((nullptr == x1Shape) || (nullptr == x2Shape) || (nullptr == gammaShape),
                OP_LOGE(context_->GetNodeName(), "Input shape is nullptr, please check."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::CheckOptionalInput()
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

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::CheckInputShapeDim()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckInputShapeDim.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);

    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((x1DimNum > MAX_DIM_CNT) || (x2DimNum > MAX_DIM_CNT) || (x1DimNum < 1) || (x2DimNum < 1),
                OP_LOGE(context_->GetNodeName(),
                        "Input x1/x2 dim should be greater than 0,"
                        "and not bigger than %u.",
                        MAX_DIM_CNT),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckDimBiggerZero(x1Shape, x1DimNum, nodeName, "x1"),
                OP_LOGE(context_->GetNodeName(), "Input x1 shape is invalid, please check."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckDimBiggerZero(x2Shape, x2DimNum, nodeName, "x2"),
                OP_LOGE(context_->GetNodeName(), "Input x2 shape is invalid, please check."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::CheckInputShapeValue()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckInputShapeValue.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaShape);

    if (!NormCheck::CheckShapeSame(x1Shape, x2Shape, nodeName, "x1", "x2")) {
        OP_LOGE(context_->GetNodeName(), "Input x1 shape is not same with x2 shape.");
        return ge::GRAPH_FAILED;
    }

    if (1 != gammaShape->GetStorageShape().GetDimNum()) {
        OP_LOGE(context_->GetNodeName(), "The shape dim of gamma/beta only support 1, please check.");
        return ge::GRAPH_FAILED;
    }

    if (!NormCheck::CheckShapeBC(x1Shape, gammaShape, nodeName, "x1", "gamma", true)) {
        OP_LOGE(context_->GetNodeName(), "Input gamma shape value is not valid with x shape value.");
        return ge::GRAPH_FAILED;
    }

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

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::CheckInputDtype()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckInputDtype.");
    std::set<ge::DataType> supportedXDtypes = {Ops::NN::DT_FLOAT16, Ops::NN::DT_BF16};
    std::set<ge::DataType> supportedGammaDtypes = {Ops::NN::DT_FLOAT16, Ops::NN::DT_BF16, Ops::NN::DT_FLOAT};

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputTensor(X1_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputTensor(X2_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputTensor(GAMMA_INDEX));
    ge::DataType x1Dtype = context_->GetInputTensor(X1_INDEX)->GetDataType();
    ge::DataType x2Dtype = context_->GetInputTensor(X2_INDEX)->GetDataType();
    ge::DataType gammaDtype = context_->GetInputTensor(GAMMA_INDEX)->GetDataType();

    if (x1Dtype != x2Dtype) {
        OP_LOGE(context_->GetNodeName(), "Input x1/x2 dtype should be equal.");
        return ge::GRAPH_FAILED;
    }
    if (supportedXDtypes.count(x1Dtype) == 0) {
        OP_LOGE(context_->GetNodeName(), "Input x1/x2 dtype should be float16 or bfloat16.");
        return ge::GRAPH_FAILED;
    }
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
    gammaIsFp32_ = (gammaDtype == Ops::NN::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::CheckOutputDtype()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckOutputDtype.");
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_INDEX));
    ge::DataType yDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();

    if (yDtype != Ops::NN::DT_HIFLOAT4) {
        OP_LOGE(context_->GetNodeName(), "Output y dtype should be HIFLOAT4.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(X_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(SCALE_INDEX));
    ge::DataType outputXDtype = context_->GetOutputDesc(X_INDEX)->GetDataType();
    ge::DataType scaleDtype = context_->GetOutputDesc(SCALE_INDEX)->GetDataType();
    if (outputXDtype != xDtype_) {
        OP_LOGE(context_->GetNodeName(), "Output X dtype should be equal to input X1/X2 dtype.");
        return ge::GRAPH_FAILED;
    }
    if (scaleDtype != Ops::NN::DT_HIFLOAT4_SCALE) {
        OP_LOGE(context_->GetNodeName(), "Output scale dtype should be HIFLOAT4_SCALE.");
        return ge::GRAPH_FAILED;
    }

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
        if (rstdDtype != Ops::NN::DT_FLOAT) {
            OP_LOGE(context_->GetNodeName(), "Output rstd dtype should be FLOAT32.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::CheckOutputShapeValue()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckOutputShapeValue.");

    const gert::StorageShape* x1ShapePtr = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* scaleShapePtr = context_->GetOutputShape(SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);

    const gert::Shape x1Shape = x1ShapePtr->GetStorageShape();
    const gert::Shape scaleShape = scaleShapePtr->GetStorageShape();
    const gert::Shape gammaShape = context_->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    size_t xRank = x1Shape.GetDimNum();
    size_t scaleRank = scaleShape.GetDimNum();
    size_t gammaRank = gammaShape.GetDimNum();

    if (scaleRank != xRank + 1) {
        OP_LOGE(context_->GetNodeName(), "Output scale rank [%zu] should be xRank+1 [%zu].", scaleRank, xRank + 1);
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < xRank - gammaRank; i++) {
        if (scaleShape.GetDim(i) != x1Shape.GetDim(i)) {
            OP_LOGE(context_->GetNodeName(), "scale shape dim[%zu]=%ld should match x1 shape dim[%zu]=%ld.", i,
                    scaleShape.GetDim(i), i, x1Shape.GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }

    uint64_t expectedLastDim = Ops::Base::CeilDiv(numCol_, static_cast<uint64_t>(HIF4_BLOCK_SIZE));
    if (scaleShape.GetDim(scaleRank - 2) != static_cast<int64_t>(expectedLastDim)) {
        OP_LOGE(context_->GetNodeName(), "scale dim[%zu]=%ld should be CeilDiv(R=%ld, 64) = %ld.", scaleRank - 2,
                scaleShape.GetDim(scaleRank - 2), static_cast<int64_t>(numCol_), static_cast<int64_t>(expectedLastDim));
        return ge::GRAPH_FAILED;
    }

    if (scaleShape.GetDim(scaleRank - 1) != 1) {
        OP_LOGE(context_->GetNodeName(), "scale last dim should be 1, got %ld.", scaleShape.GetDim(scaleRank - 1));
        return ge::GRAPH_FAILED;
    }

    if (rstdFlag_) {
        const gert::StorageShape* rstdShapePtr = context_->GetOutputShape(RSTD_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, rstdShapePtr);
        const gert::Shape rstdShape = rstdShapePtr->GetStorageShape();
        OP_CHECK_IF((rstdShape.GetDimNum() != xRank),
                    OP_LOGE(context_->GetNodeName(), "Invalid rstd shape dim num (must same with input x1)."),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < xRank; i++) {
            if (i >= xRank - gammaRank) {
                if (rstdShape.GetDim(i) != 1) {
                    OP_LOGE(context_->GetNodeName(), "rstd shape dim[%zu]=%ld should be 1.", i, rstdShape.GetDim(i));
                    return ge::GRAPH_FAILED;
                }
            } else {
                if (rstdShape.GetDim(i) != x1Shape.GetDim(i)) {
                    OP_LOGE(context_->GetNodeName(), "rstd shape dim[%zu]=%ld should match x1 shape dim[%zu]=%ld.", i,
                            rstdShape.GetDim(i), i, x1Shape.GetDim(i));
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }

    const gert::StorageShape* outputXShape = context_->GetOutputShape(X_INDEX);
    const gert::StorageShape* outputYShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputXShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYShape);

    if (!NormCheck::CheckShapeSame(x1ShapePtr, outputXShape, nodeName, "inputX1", "outputX")) {
        OP_LOGE(context_->GetNodeName(), "Output X shape is not same with inputX1 shape.");
        return ge::GRAPH_FAILED;
    }

    if (!NormCheck::CheckShapeSame(x1ShapePtr, outputYShape, nodeName, "inputX1", "outputY")) {
        OP_LOGE(context_->GetNodeName(), "Output Y shape is not same with inputX1 shape.");
        return ge::GRAPH_FAILED;
    }

    size_t lastDim = x1Shape.GetDimNum() - 1;
    if (x1Shape.GetDim(lastDim) % NUM_TWO != 0) {
        OP_LOGE(context_->GetNodeName(),
                "Output y data type is FLOAT4_E1M2, "
                "the last axis of x should be even, but got %ld.",
                x1Shape.GetDim(lastDim));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::SetInputParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter SetInputParams.");
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

    numColAlign_ = Ops::Base::CeilAlign(numCol_, COL_ALIGN_NUM);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const float* epsilonPtr = attrs->GetFloat(EPS_ATTR_INDEX);
    if (epsilonPtr != nullptr) {
        epsilon_ = *epsilonPtr;
    }

    scaleAlg_ = 0;
    const int64_t* scaleAlgPtr = attrs->GetInt(SCALE_ALG_ATTR_INDEX);
    if (scaleAlgPtr != nullptr) {
        scaleAlg_ = *scaleAlgPtr;
    }
    if (scaleAlg_ != 0 && scaleAlg_ != 1) {
        OP_LOGE(context_->GetNodeName(), "scale_alg should be 0 or 1, but got %ld.", scaleAlg_);
        return ge::GRAPH_FAILED;
    }

    avgFactor_ = (numCol == 0) ? 0.0f : 1.0f / static_cast<float>(numCol);

    mxBlockSize_ = HIF4_BLOCK_SIZE;
    blockNumInColAxis_ = Ops::Base::CeilDiv(numCol_, mxBlockSize_);
    mxScaleSize_ = blockNumInColAxis_;

    if (numRow_ == 0 || numCol_ == 0) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        (numCol_ % HIF4_BLOCK_SIZE != 0),
        OP_LOGE(context_->GetNodeName(),
                "HiF4 quantization requires the last axis size to be a multiple of 64, but got %lu, please check.",
                numCol_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter GetShapeAttrsInfo.");
    uint64_t vecLength = Ops::Base::GetVRegSize(context_);
    OP_CHECK_IF((vecLength <= 0),
                OP_LOGE(context_, "Get vector Length failed, vector Length: %u", static_cast<uint32_t>(vecLength)),
                return ge::GRAPH_FAILED);
    vecLengthFP32_ = vecLength / FP32_SIZE;
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF((ubBlockSize_ <= 0),
                OP_LOGE(context_, "Get block Size failed, block size: %u", static_cast<uint32_t>(ubBlockSize_)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeNull() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Required input is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOptionalInput() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Optional input check failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeDim() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Input shape dim invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeValue() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Input shape value invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Input dtype invalid."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckOutputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Output dtype invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetInputParams() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetInputParams failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputShapeValue() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Output shape invalid."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    auto compileInfoPtr = reinterpret_cast<const AddRmsNormHiF4QuantCompileInfo*>(context_->GetCompileInfo());
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

ge::graphStatus AddRmsNormHiF4QuantRegbaseTilingBase::GetWorkspaceSize()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
