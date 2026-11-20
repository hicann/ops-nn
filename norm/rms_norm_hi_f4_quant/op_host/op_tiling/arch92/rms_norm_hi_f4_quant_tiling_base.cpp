/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file rms_norm_hi_f4_quant_tiling_base.cpp
 * \brief
 */
#include "../rms_norm_hi_f4_quant_tiling.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"
#include "norm/norm_common/op_host/norm_error_report.h"
#include "../../../../../common/inc/op_host/op_const_def_nn.h"

namespace optiling {
using namespace NormCheck;

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::CheckShapeNull()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckShapeNull.");
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);

    OP_CHECK_IF(
        (nullptr == xShape) || (nullptr == gammaShape),
        Ops::NN::ErrorReport::TilingFailure(
            context_->GetNodeName(), std::string("CheckShapeNull: missing input shape metadata:") +
                                         (xShape == nullptr ? " x" : "") + (gammaShape == nullptr ? " gamma" : "")),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::CheckOptionalInput()
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

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::CheckInputShapeDim()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckInputShapeDim.");
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (xDimNum > MAX_DIM_CNT) || (xDimNum < 1),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                                                 ("Rank must be in [1, " + std::to_string(MAX_DIM_CNT) + "]").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckDimBiggerZero(xShape, xDimNum, nodeName, "x"),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Input x shape is invalid, please check."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::CheckInputShapeValue()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckInputShapeValue.");
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    OP_CHECK_IF(xShape == nullptr,
                Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing tiling context object: xShape"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        gammaShape == nullptr,
        Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing tiling context object: gammaShape"),
        return ge::GRAPH_FAILED);

    if (1 != gammaShape->GetStorageShape().GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "gamma",
                                                 std::to_string(gammaShape->GetStorageShape().GetDimNum()).c_str(),
                                                 "Rank must be 1");
        return ge::GRAPH_FAILED;
    }

    if (!NormCheck::CheckShapeBC(xShape, gammaShape, nodeName, "x", "gamma", true)) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Input gamma shape value is not valid with x shape value.");
        return ge::GRAPH_FAILED;
    }

    if (betaFlag_) {
        const gert::StorageShape* betaShape = context_->GetOptionalInputShape(BETA_INDEX);
        OP_CHECK_IF(
            betaShape == nullptr,
            Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing tiling context object: betaShape"),
            return ge::GRAPH_FAILED);
        if (!NormCheck::CheckShapeSame(gammaShape, betaShape, nodeName, "gamma", "beta")) {
            OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Input beta shape is not same with gamma shape.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::CheckInputDtype()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckInputDtype.");
    std::set<ge::DataType> supportedXDtypes = {Ops::NN::DT_FLOAT16, Ops::NN::DT_BF16};
    std::set<ge::DataType> supportedGammaDtypes = {Ops::NN::DT_FLOAT16, Ops::NN::DT_BF16, Ops::NN::DT_FLOAT};

    OP_CHECK_IF(context_->GetInputTensor(X_INDEX) == nullptr,
                Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(),
                                                    "Missing tiling context object: context_->GetInputTensor(X_INDEX)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputTensor(GAMMA_INDEX) == nullptr,
                Ops::NN::ErrorReport::TilingFailure(
                    context_->GetNodeName(), "Missing tiling context object: context_->GetInputTensor(GAMMA_INDEX)"),
                return ge::GRAPH_FAILED);
    ge::DataType xDtype = context_->GetInputTensor(X_INDEX)->GetDataType();
    ge::DataType gammaDtype = context_->GetInputTensor(GAMMA_INDEX)->GetDataType();

    if (supportedXDtypes.count(xDtype) == 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                                              std::to_string(static_cast<int>(xDtype)).c_str(),
                                              "Supported dtypes are float16 and bfloat16");
        return ge::GRAPH_FAILED;
    }
    if (supportedGammaDtypes.count(gammaDtype) == 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "gamma",
                                              std::to_string(static_cast<int>(gammaDtype)).c_str(),
                                              "Supported dtypes are float16, bfloat16 and float32");
        return ge::GRAPH_FAILED;
    }
    if (betaFlag_) {
        ge::DataType betaDtype = context_->GetInputTensor(BETA_INDEX)->GetDataType();
        if (gammaDtype != betaDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "gamma, beta",
                (std::to_string(static_cast<int>(gammaDtype)) + ", " + std::to_string(static_cast<int>(betaDtype)))
                    .c_str(),
                "Input gamma and beta dtypes must be equal");
            return ge::GRAPH_FAILED;
        }
    }

    xDtype_ = xDtype;
    gammaDtype_ = gammaDtype;
    xDtypeSize_ = ge::GetSizeByDataType(xDtype);
    gammaDtypeSize_ = ge::GetSizeByDataType(gammaDtype);
    gammaIsFp32_ = (gammaDtype == Ops::NN::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::CheckOutputDtype()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckOutputDtype.");
    OP_CHECK_IF(context_->GetOutputDesc(Y_INDEX) == nullptr,
                Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(),
                                                    "Missing tiling context object: context_->GetOutputDesc(Y_INDEX)"),
                return ge::GRAPH_FAILED);
    ge::DataType yDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();
    if (yDtype != Ops::NN::DT_HIFLOAT4) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y",
                                              std::to_string(static_cast<int>(yDtype)).c_str(),
                                              "Output y dtype must be HIFLOAT4");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(context_->GetOutputDesc(SCALE_INDEX) == nullptr,
                Ops::NN::ErrorReport::TilingFailure(
                    context_->GetNodeName(), "Missing tiling context object: context_->GetOutputDesc(SCALE_INDEX)"),
                return ge::GRAPH_FAILED);
    ge::DataType scaleDtype = context_->GetOutputDesc(SCALE_INDEX)->GetDataType();
    if (scaleDtype != Ops::NN::DT_HIFLOAT4_SCALE) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "scale",
                                              std::to_string(static_cast<int>(scaleDtype)).c_str(),
                                              "Output scale dtype must be HIFLOAT4_SCALE");
        return ge::GRAPH_FAILED;
    }

    rstdFlag_ = 0;
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing tiling context object: attrs"),
                return ge::GRAPH_FAILED);
    const bool* outputRstdPtr = attrs->GetAttrPointer<bool>(OUTPUT_RSTD_ATTR_INDEX);
    if (outputRstdPtr != nullptr) {
        bool outputRstd = *outputRstdPtr;
        if (outputRstd == true) {
            rstdFlag_ = 1;
        }
    }

    if (rstdFlag_) {
        OP_CHECK_IF(context_->GetOutputDesc(RSTD_INDEX) == nullptr,
                    Ops::NN::ErrorReport::TilingFailure(
                        context_->GetNodeName(), "Missing tiling context object: context_->GetOutputDesc(RSTD_INDEX)"),
                    return ge::GRAPH_FAILED);
        ge::DataType rstdDtype = context_->GetOutputDesc(RSTD_INDEX)->GetDataType();
        if (rstdDtype != Ops::NN::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "rstd",
                                                  std::to_string(static_cast<int>(rstdDtype)).c_str(),
                                                  "Output rstd dtype must be float32");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::CheckOutputShapeValue()
{
    OP_LOGD(context_->GetNodeName(), "Enter CheckOutputShapeValue.");

    const gert::StorageShape* xShapePtr = context_->GetInputShape(X_INDEX);
    const gert::StorageShape* scaleShapePtr = context_->GetOutputShape(SCALE_INDEX);
    OP_CHECK_IF(
        scaleShapePtr == nullptr,
        Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing tiling context object: scaleShapePtr"),
        return ge::GRAPH_FAILED);

    const gert::Shape xShape = xShapePtr->GetStorageShape();
    const gert::Shape scaleShape = scaleShapePtr->GetStorageShape();
    const gert::Shape gammaShape = context_->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    size_t xRank = xShape.GetDimNum();
    size_t scaleRank = scaleShape.GetDimNum();
    size_t gammaRank = gammaShape.GetDimNum();

    if (scaleRank != xRank + 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "scale, x",
                                                  (std::to_string(scaleRank) + ", " + std::to_string(xRank)).c_str(),
                                                  "Output scale rank must equal input rank plus 1");
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < xRank - gammaRank; i++) {
        if (scaleShape.GetDim(i) != xShape.GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "scale", Ops::Base::ToString(scaleShape).c_str(),
                ("Dimension " + std::to_string(i) + " must equal x dimension " + std::to_string(xShape.GetDim(i)))
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }

    uint64_t expectedLastDim = Ops::Base::CeilDiv(numCol_, static_cast<uint64_t>(HIF4_BLOCK_SIZE));
    if (scaleShape.GetDim(scaleRank - NUM_TWO) != static_cast<int64_t>(expectedLastDim)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "scale", Ops::Base::ToString(scaleShape).c_str(),
            ("Penultimate dimension must equal ceil(input last dimension / 64) = " + std::to_string(expectedLastDim))
                .c_str());
        return ge::GRAPH_FAILED;
    }

    if (scaleShape.GetDim(scaleRank - 1) != 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "scale", Ops::Base::ToString(scaleShape).c_str(),
                                              "Last dimension must be 1");
        return ge::GRAPH_FAILED;
    }

    if (rstdFlag_) {
        const gert::StorageShape* rstdShapePtr = context_->GetOutputShape(RSTD_INDEX);
        OP_CHECK_IF(
            rstdShapePtr == nullptr,
            Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing tiling context object: rstdShapePtr"),
            return ge::GRAPH_FAILED);
        const gert::Shape rstdShape = rstdShapePtr->GetStorageShape();
        OP_CHECK_IF((rstdShape.GetDimNum() != xRank),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context_->GetNodeName(), "rstd, x",
                        (std::to_string(rstdShape.GetDimNum()) + ", " + std::to_string(xRank)).c_str(),
                        "Output rstd rank must equal input rank"),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < xRank; i++) {
            if (i >= xRank - gammaRank) {
                if (rstdShape.GetDim(i) != 1) {
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "rstd",
                                                          Ops::Base::ToString(rstdShape).c_str(),
                                                          ("Dimension " + std::to_string(i) + " must be 1").c_str());
                    return ge::GRAPH_FAILED;
                }
            } else {
                if (rstdShape.GetDim(i) != xShape.GetDim(i)) {
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "rstd", Ops::Base::ToString(rstdShape).c_str(),
                        ("Dimension " + std::to_string(i) + " must equal x dimension " +
                         std::to_string(xShape.GetDim(i)))
                            .c_str());
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }

    const gert::StorageShape* outputYShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(
        outputYShape == nullptr,
        Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing tiling context object: outputYShape"),
        return ge::GRAPH_FAILED);

    if (!NormCheck::CheckShapeSame(xShapePtr, outputYShape, nodeName, "inputX", "outputY")) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Output Y shape is not same with inputX shape.");
        return ge::GRAPH_FAILED;
    }

    size_t lastDim = xShape.GetDimNum() - 1;
    if (xShape.GetDim(lastDim) % NUM_TWO != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                              "Last input dimension must be even for packed HiF4 output");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::SetInputParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter SetInputParams.");
    const gert::Shape xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    const gert::Shape gammaShape = context_->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();

    uint64_t numRow = 1;
    uint64_t numCol = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        numCol *= gammaShape.GetDim(i);
    }
    numRow_ = numRow;
    numCol_ = numCol;

    numColAlign_ = Ops::Base::CeilAlign(numCol_, COL_ALIGN_NUM);

    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing tiling context object: attrs"),
                return ge::GRAPH_FAILED);

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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "scale_alg", std::to_string(scaleAlg_).c_str(),
                                              "Supported values are 0 and 1");
        return ge::GRAPH_FAILED;
    }

    avgFactor_ = (numCol == 0) ? 0.0f : 1.0f / static_cast<float>(numCol);

    mxBlockSize_ = HIF4_BLOCK_SIZE;
    blockNumInColAxis_ = Ops::Base::CeilDiv(numCol_, mxBlockSize_);
    mxScaleSize_ = blockNumInColAxis_;

    if (numRow_ == 0 || numCol_ == 0) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF((numCol_ % HIF4_BLOCK_SIZE != 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                                      "Last dimension must be a multiple of 64 for HiF4 quantization"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter GetShapeAttrsInfo.");
    OP_CHECK_IF(
        context_->GetPlatformInfo() == nullptr,
        Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing platform metadata for vector resources"),
        return ge::GRAPH_FAILED);
    uint64_t vecLength = Ops::Base::GetVRegSize(context_);
    OP_CHECK_IF((vecLength <= 0),
                Ops::NN::ErrorReport::TilingFailure(
                    context_->GetNodeName(),
                    "GetShapeAttrsInfo: vecLength must be positive, actual=" + std::to_string(vecLength)),
                return ge::GRAPH_FAILED);
    vecLengthFP32_ = vecLength / FP32_SIZE;
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF((ubBlockSize_ <= 0),
                Ops::NN::ErrorReport::TilingFailure(
                    context_->GetNodeName(),
                    "GetShapeAttrsInfo: ubBlockSize_ must be positive, actual=" + std::to_string(ubBlockSize_)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeNull() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Required input is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOptionalInput() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Optional input check failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeDim() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Input shape dim invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeValue() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Input shape value invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Input dtype invalid."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckOutputDtype() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Output dtype invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "SetInputParams failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputShapeValue() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Output shape invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    auto compileInfoPtr = static_cast<const RmsNormHiF4QuantCompileInfo*>(context_->GetCompileInfo());
    if (compileInfoPtr != nullptr) {
        totalCoreNum_ = compileInfoPtr->totalCoreNum;
        maxUbSize_ = compileInfoPtr->totalUbSize;
    } else {
        OP_CHECK_IF(
            platformInfo == nullptr,
            Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing tiling context object: platformInfo"),
            return ge::GRAPH_FAILED);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        maxUbSize_ = ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormHiF4QuantRegbaseTilingBase::GetWorkspaceSize()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfoPtr == nullptr,
        Ops::NN::ErrorReport::TilingFailure(context_->GetNodeName(), "Missing tiling context object: platformInfoPtr"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
