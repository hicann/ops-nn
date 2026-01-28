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
 * \file rms_norm_grad_regbase_tiling.cpp
 * \brief RmsNormGrad regbase tiling file
 */

#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "rms_norm_grad_tiling.h"

namespace optiling {
constexpr int64_t MIN_DATANUM_PER_CORE = 1024;
constexpr int64_t UB_RESERVED_SIZE = 256;
constexpr uint32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t INPUT_NUM = 4;
constexpr int64_t OUTPUT_NUM = 2;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t ATTR_INDEX_0 = 0;
constexpr int64_t ATTR_INDEX_1 = 1;
constexpr int64_t INPUT_INDEX_0 = 0;
constexpr int64_t INPUT_INDEX_1 = 1;
constexpr int64_t INPUT_INDEX_2 = 2;
constexpr int64_t INPUT_INDEX_3 = 3;
constexpr int64_t INPUT_INDEX_4 = 4;
constexpr int64_t TWO = 2;
constexpr int64_t THREE = 3;
constexpr int64_t FLOATBYTESIZE = 4;
constexpr int64_t DX_UB_FACTOR = 6144;
constexpr uint32_t REG_BASE_KEY = 7000;
constexpr uint32_t DX_SPLIT_KEY = 10;
constexpr uint32_t DG_SPLIT_KEY = 1;
const std::string OP_NAME = "RmsNormGrad";

ge::graphStatus RmsNormGradRegbaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo(); // 判断条件修改
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        blockSize_ = Ops::Base::GetUbBlockSize(context_);
        vecRegSize_ = Ops::Base::GetVRegSize(context_);
    }
    vlFp32_ = vecRegSize_ / sizeof(float);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradRegbaseTiling::CheckShapeAllPositive(gert::Shape& shape)
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

ge::graphStatus RmsNormGradRegbaseTiling::CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1)
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

void RmsNormGradRegbaseTiling::CalcRowsAndCols(gert::Shape& xShape, gert::Shape& gammaShape)
{
    rows_ = 1;
    cols_ = 1;
    for (size_t i = 0; i < xShape.GetDimNum() - gammaShape.GetDimNum(); i++) {
        rows_ *= xShape.GetDim(i);
    }
    for (size_t i = 0; i < gammaShape.GetDimNum(); i++) {
        cols_ *= gammaShape.GetDim(i);
    }
}

ge::graphStatus RmsNormGradRegbaseTiling::CheckInputsShape()
{
    // check dy
    auto inputShape = context_->GetInputShape(INPUT_INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape0 = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(storageShape0) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "dy shape contains zero.");
        return ge::GRAPH_FAILED;
    }

    // check x
    inputShape = context_->GetInputShape(INPUT_INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape1 = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "x shape contains zero.");
        return ge::GRAPH_FAILED;
    }

    // check shapes of input0 and input1 are equal
    if (CheckShapesEqual(storageShape0, storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Shapes of x1 and x2 are not equal.");
        return ge::GRAPH_FAILED;
    }

    // check rstd
    inputShape = context_->GetInputShape(INPUT_INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape2 = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(storageShape2) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "gamma shape contains zero.");
        return ge::GRAPH_FAILED;
    }

    // check gamma
    inputShape = context_->GetInputShape(INPUT_INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape3 = inputShape->GetStorageShape();
    if (CheckShapeAllPositive(storageShape3) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "beta shape contains zero.");
        return ge::GRAPH_FAILED;
    }

    CalcRowsAndCols(storageShape0, storageShape3);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradRegbaseTiling::CheckInputsDtypeAndFormat()
{
    for (int i = 0; i < INPUT_NUM; i++) {
        auto inputDesc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
        // check format
        auto format = inputDesc->GetFormat().GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGE(context_->GetNodeName(), "Input %d only supports ND format.", i);
            return ge::GRAPH_FAILED;
        }
    }

    auto dyDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    dyDtype_ = dyDesc->GetDataType();
    OP_CHECK_IF(
        (dyDtype_ != ge::DataType::DT_FLOAT16 && dyDtype_ != ge::DataType::DT_BF16 &&
         dyDtype_ != ge::DataType::DT_FLOAT),
        OP_LOGE(
            context_->GetNodeName(), "Input dyDtype [%s] not support, only supports float16, bfloat16, float32.",
            ge::TypeUtils::DataTypeToSerialString(dyDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto xDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto xDtype = xDesc->GetDataType();
    OP_CHECK_IF(
        (dyDtype_ != xDtype),
        OP_LOGE(
            context_->GetNodeName(), "Input xDtype [%s] does not match dyDtype [%s].",
            ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(dyDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto rstdDesc = context_->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rstdDesc);
    auto rstdDtype = rstdDesc->GetDataType();
    OP_CHECK_IF(
        (rstdDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(
            context_->GetNodeName(), "Input rstdDtype [%s] not support, only supports float32.",
            ge::TypeUtils::DataTypeToSerialString(rstdDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto gammaDesc = context_->GetInputDesc(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    auto gammaDtype = gammaDesc->GetDataType();
    OP_CHECK_IF(
        (gammaDtype != ge::DataType::DT_FLOAT && gammaDtype != dyDtype_),
        OP_LOGE(
            context_->GetNodeName(), "Input gammaDtype [%s] does not match dyDtype [%s] or match float32.",
            ge::TypeUtils::DataTypeToSerialString(gammaDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(dyDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto dxDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
    auto dxDtype = dxDesc->GetDataType();
    OP_CHECK_IF(
        (dyDtype_ != dxDtype),
        OP_LOGE(
            context_->GetNodeName(), "Input dxDtype [%s] does not match dyDtype [%s].",
            ge::TypeUtils::DataTypeToSerialString(dxDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(dyDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto dgammaDesc = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dgammaDesc);
    auto dgammaDtype = dgammaDesc->GetDataType();
    OP_CHECK_IF(
        (dgammaDtype != ge::DataType::DT_FLOAT),
        OP_LOGE(
            context_->GetNodeName(), "Input dgammaDtype [%s] not support, only supports float32.",
            ge::TypeUtils::DataTypeToSerialString(dgammaDtype).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradRegbaseTiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE(OP_NAME, "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    // check inputs shape
    if (CheckInputsShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Inputs shape invalid.");
        return ge::GRAPH_FAILED;
    }
    // check inputs dtype and format
    if (CheckInputsDtypeAndFormat() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Inputs dtype/format invalid.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool RmsNormGradRegbaseTiling::IsCapable()
{
    return true;
}

uint64_t RmsNormGradRegbaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus RmsNormGradRegbaseTiling::PostTiling()
{
    context_->SetBlockDim(aivCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradRegbaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradRegbaseTiling::GetWorkspaceSize()
{
    workspaceSize_ = MIN_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradRegbaseTiling::CalcTilingDataDx()
{
    blockFactorDx_ = Ops::Base::CeilDiv(rows_, static_cast<int64_t>(aivCoreNum_));
    usedCoreNumDx_ = Ops::Base::CeilDiv(rows_, blockFactorDx_);
    if (cols_ > DX_UB_FACTOR) {
        tilingKey_ += DX_SPLIT_KEY;
        bodyPart_ = 1; // only for splitD
        int32_t powerofTwoValueDx = 0;
        while (bodyPart_ < cols_ / TWO) {
            powerofTwoValueDx += 1;
            bodyPart_ = std::pow(TWO, powerofTwoValueDx);
        }
    }

    return ge::GRAPH_SUCCESS;
}

void RmsNormGradRegbaseTiling::CalcUsedCoreNumGamma()
{
    cols_sets_ = (cols_ + vlFp32_ - 1) / vlFp32_;
    if (cols_sets_ <= aivCoreNum_) {
        // 单核计算单colset
        isMultiColset_ = false;
        usedCoreNumDG_ = cols_sets_;
        colsPerCoreDG_ = vlFp32_;
        colsPerTailCoreDG_ = cols_ - (cols_sets_ - 1) * vlFp32_;
        tailCoreNumDG_ = 1;
        colsLastCoreDG_ = colsPerTailCoreDG_;
    } else {
        // 单核计算多个colset
        isMultiColset_ = true;
        usedCoreNumDG_ = aivCoreNum_;
        colsPerCoreDG_ = (cols_sets_ / usedCoreNumDG_) * vlFp32_;
        if (colsPerCoreDG_ * usedCoreNumDG_ > cols_) {
            colsPerCoreDG_ = colsPerCoreDG_ - vlFp32_;
        }
        colsPerTailCoreDG_ = colsPerCoreDG_ + vlFp32_;
        tailCoreNumDG_ = cols_sets_ - (colsPerCoreDG_ / vlFp32_ * usedCoreNumDG_);
        colsLastCoreDG_ = cols_ - (cols_sets_ - 1) * vlFp32_;
    }
}

int64_t RmsNormGradRegbaseTiling::GetSizeOfBlockAlign(int64_t nonAlignSize)
{
    return (nonAlignSize + blockSize_ - 1) / blockSize_ * blockSize_;
}

int32_t RmsNormGradRegbaseTiling::NearestLowerPowerOfTwo(int32_t temp)
{
    int64_t power = 0;
    int32_t powerofTwoValueDG = 0;
    while (power <= temp) {
        powerofTwoValueDG += 1;
        power = std::pow(TWO, powerofTwoValueDG);
    }
    return powerofTwoValueDG - 1;
}

ge::graphStatus RmsNormGradRegbaseTiling::CalcTilingDataDgamma()
{
    // dy, x, gamma
    int64_t xUbSizeForOneColSet = GetSizeOfBlockAlign(rows_ * FLOATBYTESIZE) * vlFp32_;
    int64_t dyUbSizeForOneColSet = GetSizeOfBlockAlign(rows_ * FLOATBYTESIZE) * vlFp32_;
    int64_t rstdUbSize = GetSizeOfBlockAlign(rows_ * FLOATBYTESIZE);
    int64_t inputUbSizeForOneColSet = xUbSizeForOneColSet + dyUbSizeForOneColSet + rstdUbSize;

    // dgamma
    int64_t outputUbSizeForOneColSet = GetSizeOfBlockAlign((rows_ + 1) * FLOATBYTESIZE) * vlFp32_;
    colsPerUBDG_ = vlFp32_;
    if (static_cast<int64_t>(ubSize_) >= TWO * (inputUbSizeForOneColSet + outputUbSizeForOneColSet)) {
        // full-load
        isFullLoadDG_ = true;
        rowsPerUBDG_ = rows_;
    } else {
        // UB二分累加
        isFullLoadDG_ = false;
        tilingKey_ += DG_SPLIT_KEY;
        // 反算一次UB能放下多少数据rows
        // 一行数据的buffer为(dy + x + dgamma + dgamma1) *64 + (binaryAddCache + rstd)
        maxRowsNumDG_ = ubSize_ / BUFFER_NUM / ((vlFp32_ * 4 + TWO) * FLOATBYTESIZE);
        rowsPerUBDG_ = std::pow(TWO, NearestLowerPowerOfTwo(maxRowsNumDG_));
        OP_CHECK_IF(
            maxRowsNumDG_ <= 0, OP_LOGE(context_->GetNodeName(), "The maxRowsNumDG_ size is neg: %d.", maxRowsNumDG_),
            return ge::GRAPH_FAILED);
        totalBlockCountDG_ = (rows_ + rowsPerUBDG_ - 1) / rowsPerUBDG_;
        mainBlockCountDG_ = totalBlockCountDG_;
        tailBlockCountwithPadDG_ = 0;
        if (rows_ % rowsPerUBDG_) {
            mainBlockCountDG_ = totalBlockCountDG_ - 1;
            tailBlockCountwithPadDG_ = 1;
        }
        binaryAddKDG_ = NearestLowerPowerOfTwo(mainBlockCountDG_);
        powerOfTwoBlockCountDG_ = std::pow(TWO, binaryAddKDG_);
        tailBlockCountWithoutPadDG_ = mainBlockCountDG_ - powerOfTwoBlockCountDG_;
    }
    int32_t n = NearestLowerPowerOfTwo(rows_);
    rowsTailDG_ = rows_ - std::pow(TWO, n);
    return ge::GRAPH_SUCCESS;
}

void RmsNormGradRegbaseTiling::LogTilingResult()
{
    OP_LOGI(OP_NAME, "rows: %ld, cols_: %ld", rows_, cols_);
}

ge::graphStatus RmsNormGradRegbaseTiling::DoOpTiling()
{
    ge::graphStatus result = ge::GRAPH_SUCCESS;
    tilingKey_ = REG_BASE_KEY;
    // split cores
    CalcUsedCoreNumGamma();

    // choose template and calc ub buffer size
    result = CalcTilingDataDgamma();
    OP_CHECK_IF(
        result != ge::GRAPH_SUCCESS, OP_LOGE(OP_NAME, "Regbase template do op tiling dgamma failed."), return result);

    result = CalcTilingDataDx();
    OP_CHECK_IF(
        result != ge::GRAPH_SUCCESS, OP_LOGE(OP_NAME, "Regbase template do op tiling dx failed."), return result);

    tilingData_.dxTilingData.set_cols(cols_);
    tilingData_.dxTilingData.set_rows(rows_);
    tilingData_.dxTilingData.set_blockFactorDx(blockFactorDx_);
    tilingData_.dxTilingData.set_bodyPart(bodyPart_);
    tilingData_.dxTilingData.set_usedCoreNumDx(usedCoreNumDx_);

    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_usedCoreNumDG(usedCoreNumDG_);
    tilingData_.set_colsPerCoreDG(colsPerCoreDG_);
    tilingData_.set_rowsPerUBDG(rowsPerUBDG_);
    tilingData_.set_vlFp32(vlFp32_);
    tilingData_.set_colsPerTailCoreDG(colsPerTailCoreDG_);
    tilingData_.set_isFullLoad(isFullLoadDG_);
    tilingData_.set_isMultiColset(isMultiColset_);
    tilingData_.set_colsLastCoreDG(colsLastCoreDG_);
    tilingData_.set_tailCoreNumDG(tailCoreNumDG_);
    tilingData_.set_powerofTwoValueDG(powerofTwoValueDG_);
    tilingData_.set_rowsTailDG(rowsTailDG_);

    tilingData_.set_totalBlockCountDG(totalBlockCountDG_);
    tilingData_.set_mainBlockCountDG(mainBlockCountDG_);
    tilingData_.set_tailBlockCountwithPadDG(tailBlockCountwithPadDG_);
    tilingData_.set_powerOfTwoBlockCountDG(powerOfTwoBlockCountDG_);
    tilingData_.set_tailBlockCountWithoutPadDG(tailBlockCountWithoutPadDG_);
    tilingData_.set_binaryAddKDG(binaryAddKDG_);

    LogTilingResult();
    return result;
}

REGISTER_TILING_TEMPLATE("RmsNormGrad", RmsNormGradRegbaseTiling, 200);
} // namespace optiling