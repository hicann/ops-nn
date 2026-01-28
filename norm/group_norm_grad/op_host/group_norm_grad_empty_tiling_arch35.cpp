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
 * \file group_norm_grad_empty_tiling_arch35.cpp
 * \brief
 */
#include "group_norm_grad_empty_tiling_arch35.h"
#include "group_norm_grad_tiling.h"

namespace optiling {
constexpr uint32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;
static constexpr uint16_t INPUT_IDX_DY = 0;
static constexpr uint16_t INPUT_IDX_MEAN = 1;
static constexpr uint16_t INPUT_IDX_RSTD = 2;
static constexpr uint16_t INPUT_IDX_X = 3;
static constexpr uint16_t INPUT_IDX_GAMMA = 4;
static constexpr uint16_t OUTPUT_IDX_DX = 0;
static constexpr uint16_t OUTPUT_IDX_DGAMMA = 1;
static constexpr uint16_t OUTPUT_IDX_DBETA = 2;
static constexpr uint16_t DIM0 = 0;
static constexpr uint16_t DIM1 = 1;
static constexpr uint16_t MIN_X_DIM = 2;
constexpr int64_t UPPER_CARRYING_LIMIT = 8000;
constexpr int64_t MAX_CORE_COLS = 8000;
constexpr uint32_t EMPTY_TENSOR_KEY = 500;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t BYTES_OF_FLOAT = 4;
constexpr uint32_t BUFFER_NUM = 2;
const gert::Shape g_vec_1_shape = {1};

enum class GNGDtypeKey : int
{
    FLOAT_FLOAT = 1,
    HALF_HALF = 2,
    BF16_BF16 = 3,
    HALF_FLOAT = 4,
    BF16_FLOAT = 5
};

static const std::unordered_map<ge::DataType, uint32_t> DATA_TYPE_TO_INT{
    {ge::DataType::DT_FLOAT, 1}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 3}};

ge::graphStatus GroupNormGradEmptyTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo(); // 判断条件修改
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
    }
    return ge::GRAPH_SUCCESS;
}

const gert::Shape& GroupNormGradEmptyTiling::EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

ge::graphStatus GroupNormGradEmptyTiling::GetShapeAttrsInfo()
{
    auto dyDesc = context_->GetInputDesc(INPUT_IDX_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    tTypeStr_ = dyDesc->GetDataType();
    auto dyShape = context_->GetInputShape(INPUT_IDX_DY)->GetStorageShape();
    OP_TILING_CHECK(
        InputCheck(dyShape) == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "Input check failed."),
        return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t* gValue = attrs->GetAttrPointer<int64_t>(0);
    OP_TILING_CHECK(
        (*gValue <= 0), OP_LOGE(context_->GetNodeName(), "numGroups must be greater than 0(numGroups > 0)."),
        return ge::GRAPH_FAILED);
    G_ = static_cast<int64_t>(*gValue);
    N_ = dyShape.GetDim(DIM0);
    C_ = dyShape.GetDim(DIM1);

    CPerG_ = C_ / G_;

    OP_TILING_CHECK(
        ParamsCheck() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "Params check failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool GroupNormGradEmptyTiling::IsCapable()
{
    return true;
}

ge::graphStatus GroupNormGradEmptyTiling::InputCheck(gert::Shape& dyShape)
{
    auto xDesc = context_->GetInputDesc(INPUT_IDX_X);
    auto dxDesc = context_->GetOutputDesc(OUTPUT_IDX_DX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
    auto xShape = context_->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    auto dxShape = context_->GetOutputShape(OUTPUT_IDX_DX)->GetStorageShape();
    ge::DataType xDtypeStr = xDesc->GetDataType();
    ge::DataType dxDtypeStr = dxDesc->GetDataType();
    OP_TILING_CHECK(
        (tTypeStr_ != dxDtypeStr || dxDtypeStr != xDtypeStr),
        OP_LOGE(context_->GetNodeName(), "x, dy, dx data type must be same."), return ge::GRAPH_FAILED);
    auto iter = DATA_TYPE_TO_INT.find(tTypeStr_);
    if (iter == DATA_TYPE_TO_INT.end()) {
        OP_LOGE(
            context_->GetNodeName(), "inputdtype [%d] must be in float32:[%d], float16:[%d], bfloat16:[%d]", tTypeStr_,
            ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16);
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        (dyShape != dxShape || dxShape != xShape), OP_LOGE(context_->GetNodeName(), "x, dy, dx shape must be same."),
        return ge::GRAPH_FAILED);

    auto dimNum = dyShape.GetDimNum();
    OP_TILING_CHECK(
        (dimNum < MIN_X_DIM), OP_LOGE(context_->GetNodeName(), "x, dy, dx shape dim must large than 2."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void GroupNormGradEmptyTiling::CalcRowsAndCols(gert::Shape& gammaShape)
{
    rows_ = 0;
    cols_ = 1;
    for (size_t i = 0; i < gammaShape.GetDimNum(); i++) {
        cols_ *= gammaShape.GetDim(i);
    }
}

ge::graphStatus GroupNormGradEmptyTiling::CheckInputAndOutput()
{
    // C / G must not be zero, and C must be an integer multiple of G
    // while also not exceeding the operator's current carrying capacity.
    if (C_ != 0) {
        if (CPerG_ == 0 || C_ % G_ != 0 || CPerG_ > UPPER_CARRYING_LIMIT) {
            OP_LOGE(context_->GetNodeName(), "Group_num is invalid");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (CPerG_ > UPPER_CARRYING_LIMIT) {
            OP_LOGE(context_->GetNodeName(), "Group_num is invalid");
            return ge::GRAPH_FAILED;
        }
    }
    auto gammaDesc = context_->GetInputDesc(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    auto gammaShape = context_->GetInputShape(INPUT_IDX_GAMMA)->GetStorageShape();
    int64_t gammaSize = 1;
    for (uint32_t dimIdx = 0; dimIdx < gammaShape.GetDimNum(); dimIdx++) {
        gammaSize *= gammaShape.GetDim(dimIdx);
    }
    OP_TILING_CHECK(
        gammaShape.GetDimNum() != 1 || gammaSize != this->C_,
        OP_LOGE(
            context_->GetNodeName(), "the shape of gamma must be the same as Channel, current gammaSize is :%ld.",
            gammaSize),
        return ge::GRAPH_FAILED);
    CalcRowsAndCols(gammaShape);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradEmptyTiling::CheckShapeAndType()
{
    auto gammaDesc = context_->GetInputDesc(INPUT_IDX_GAMMA);
    auto meanDesc = context_->GetInputDesc(INPUT_IDX_MEAN);
    auto rstdDesc = context_->GetInputDesc(INPUT_IDX_RSTD);
    auto dGammaDesc = context_->GetOutputDesc(OUTPUT_IDX_DGAMMA);
    auto dBetaDesc = context_->GetOutputDesc(OUTPUT_IDX_DBETA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rstdDesc);
    auto meanShape = EnsureNotScalar(context_->GetInputShape(INPUT_IDX_MEAN)->GetStorageShape());
    auto rstdShape = EnsureNotScalar(context_->GetInputShape(INPUT_IDX_RSTD)->GetStorageShape());
    uTypeStr_ = gammaDesc->GetDataType();
    ge::DataType meanDtypeStr = meanDesc->GetDataType();
    ge::DataType rstdDtypeStr = rstdDesc->GetDataType();
    ge::DataType dGammaDtypeStr = dGammaDesc->GetDataType();
    ge::DataType dBetaDtypeStr = dBetaDesc->GetDataType();
    int64_t meanSize = 1;
    for (uint32_t dimIdx = 0; dimIdx < meanShape.GetDimNum(); dimIdx++) {
        meanSize *= meanShape.GetDim(dimIdx);
    }
    int64_t rstdSize = 1;
    for (uint32_t dimIdx = 0; dimIdx < rstdShape.GetDimNum(); dimIdx++) {
        rstdSize *= rstdShape.GetDim(dimIdx);
    }
    auto iter = DATA_TYPE_TO_INT.find(uTypeStr_);
    if (iter == DATA_TYPE_TO_INT.end()) {
        OP_LOGE(
            context_->GetNodeName(), "inputdtype [%d] must be in float32:[%d], float16:[%d], bfloat16:[%d]", tTypeStr_,
            ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16);
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        (meanDtypeStr != rstdDtypeStr || rstdDtypeStr != dGammaDtypeStr || dGammaDtypeStr != dBetaDtypeStr ||
         dBetaDtypeStr != uTypeStr_),
        OP_LOGE(context_->GetNodeName(), "rstd, mean, gamma, dbeta, dgamma data type must be same."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (tTypeStr_ == ge::DT_FLOAT && (uTypeStr_ == ge::DT_FLOAT16 || uTypeStr_ == ge::DT_BF16)),
        OP_LOGE(
            context_->GetNodeName(), "dy, x, dx is float, mean, rstd gamma, dgamma, dbeta only support float type."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        meanSize != rstdSize || meanSize != this->N_ * this->G_,
        OP_LOGE(
            context_->GetNodeName(), "the shape of mean [%ld], rstd [%ld] must equal N * G, current N * G is :%ld.",
            meanSize, rstdSize, this->N_ * this->G_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradEmptyTiling::ParamsCheck()
{
    auto ret = CheckInputAndOutput();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckShapeAndType();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupNormGradEmptyTiling::NearestLowerPowerOfTwo(int32_t tmp)
{
    int64_t power = 0;
    uint64_t powerOfTwoValue = 0;
    while (power <= tmp) {
        powerOfTwoValue += 1;
        power = std::pow(CONST_TWO, powerOfTwoValue);
    }
    return powerOfTwoValue - 1;
}

ge::graphStatus GroupNormGradEmptyTiling::CalcuTilingData()
{
    if (ubSize_ >= BUFFER_NUM * (colsPerCoreDG_ * BYTES_OF_FLOAT)) {
        // full-load
        coreUbBlockCount_ = 0;
        tailUbCols_ = colsPerCoreDG_;
        lastCoreBlockCount_ = 0;
        lastCoreTailUbCols_ = colsLastCoreDG_;
    } else {
        // UB最多存放多少列
        int64_t maxRowsNumDG_ = ubSize_ / (BUFFER_NUM * BYTES_OF_FLOAT);
        colsPerUBDG_ = std::pow(CONST_TWO, NearestLowerPowerOfTwo(maxRowsNumDG_));
        OP_CHECK_IF(
            maxRowsNumDG_ <= 0, OP_LOGE(context_->GetNodeName(), "The maxRowsNumDG_ size is neg: %ld.", maxRowsNumDG_),
            return ge::GRAPH_FAILED);
        if (colsPerUBDG_ == 0) {
            OP_LOGE(context_->GetNodeName(), "colsPerUBDG_ is zero, cannot perform division.");
            return ge::GRAPH_FAILED;
        }
        coreUbBlockCount_ = (colsPerCoreDG_ + colsPerUBDG_ - 1) / colsPerUBDG_ - 1;
        tailUbCols_ = colsPerCoreDG_ - colsPerUBDG_ * coreUbBlockCount_;
        if (colsPerUBDG_ > colsLastCoreDG_) {
            lastCoreBlockCount_ = 0;
            lastCoreTailUbCols_ = colsLastCoreDG_;
        } else {
            lastCoreBlockCount_ = (colsLastCoreDG_ + colsPerUBDG_ - 1) / colsPerUBDG_ - 1;
            lastCoreTailUbCols_ = colsLastCoreDG_ - colsPerUBDG_ * lastCoreBlockCount_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void GroupNormGradEmptyTiling::CalcUsedCoreNumGamma()
{
    if (cols_ <= MAX_CORE_COLS) {
        // 单核计算单colset
        usedCoreNumDG_ = 1;
        colsPerCoreDG_ = cols_;
        colsLastCoreDG_ = cols_;
        coreUbBlockCount_ = 0;
        tailUbCols_ = colsPerCoreDG_;
        colsPerUBDG_ = colsPerCoreDG_;
        lastCoreBlockCount_ = 0;
        lastCoreTailUbCols_ = colsLastCoreDG_;
    } else {
        // 多核计算
        usedCoreNumDG_ = aivCoreNum_;
        colsPerCoreDG_ = Ops::Base::CeilDiv(cols_, usedCoreNumDG_);
        colsLastCoreDG_ = cols_ - colsPerCoreDG_ * (usedCoreNumDG_ - 1);
        ge::graphStatus result = CalcuTilingData();
        if (result == ge::GRAPH_FAILED) {
            OP_LOGE(context_->GetNodeName(), "CalcuTilingData failed.");
        }
    }
}

ge::graphStatus GroupNormGradEmptyTiling::DoOpTiling()
{
    modeKey_ = EMPTY_TENSOR_KEY;

    CalcUsedCoreNumGamma();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradEmptyTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupNormGradEmptyTiling::GetTilingKey() const
{
    return modeKey_;
}

ge::graphStatus GroupNormGradEmptyTiling::GetWorkspaceSize()
{
    workSpaceSize_ = MIN_WORKSPACE_SIZE;
    tilingData.set_workspaceSize(workSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradEmptyTiling::PostTiling()
{
    SetTilingData();
    context_->SetBlockDim(aivCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workSpaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void GroupNormGradEmptyTiling::SetTilingData()
{
    OP_LOGD(opName, "Tiling start.");
    tilingData.set_colsPerUBDG(colsPerUBDG_);
    tilingData.set_tailUbCols(tailUbCols_);
    tilingData.set_coreUbBlockCount(coreUbBlockCount_);
    tilingData.set_lastCoreBlockCount(lastCoreBlockCount_);
    tilingData.set_lastCoreTailUbCols(lastCoreTailUbCols_);
    tilingData.set_ubSize(ubSize_);
    tilingData.set_cols(cols_);
    tilingData.set_usedCoreNumDG(usedCoreNumDG_);
    tilingData.set_colsPerCoreDG(colsPerCoreDG_);
    tilingData.set_colsLastCoreDG(colsLastCoreDG_);
    PrintTilingData();
}

void GroupNormGradEmptyTiling::PrintTilingData() const
{
    OP_LOGD(opName, "colsPerUBDG_:            %ld.", colsPerUBDG_);
    OP_LOGD(opName, "tailUbCols_:             %ld.", tailUbCols_);
    OP_LOGD(opName, "coreUbBlockCount_:       %ld.", coreUbBlockCount_);
    OP_LOGD(opName, "lastCoreBlockCount_:     %ld.", lastCoreBlockCount_);
    OP_LOGD(opName, "lastCoreTailUbCols_:     %ld.", lastCoreTailUbCols_);
    OP_LOGD(opName, "ubSize_:                 %ld.", ubSize_);
    OP_LOGD(opName, "cols_:                   %ld.", cols_);
    OP_LOGD(opName, "usedCoreNumDG_:          %ld.", usedCoreNumDG_);
    OP_LOGD(opName, "colsPerCoreDG_:          %ld.", colsPerCoreDG_);
    OP_LOGD(opName, "colsLastCoreDG_:         %ld.", colsLastCoreDG_);
}
} // namespace optiling
