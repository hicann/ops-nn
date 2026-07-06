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
 * \file swiglu_group_quant_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include "swiglu_group_quant_tiling.h"

using namespace ge;
namespace optiling {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y)
{
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y) { return CeilDiv(x, y) * y; }

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t D_LIMIT = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t FP8_BYTES = 1;
constexpr int64_t B16_BYTES = 2;
constexpr int64_t B32_BYTES = 4;
constexpr int64_t FP8_ALIGN_NUM = BLOCK_SIZE / FP8_BYTES;
constexpr int64_t B16_ALIGN_NUM = BLOCK_SIZE / B16_BYTES;
constexpr int64_t B32_ALIGN_NUM = BLOCK_SIZE / B32_BYTES;
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t PER_MX_FP16 = 32;
constexpr int64_t FP4_PACK_NUM = 2;
constexpr int64_t BLOCK_QUANT = 0;
constexpr int64_t MX_QUANT = 1;
constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr size_t ATTR_INDEX_QUANT_MODE = 1;
constexpr size_t ATTR_INDEX_BLOCK_SIZE = 2;
constexpr size_t ATTR_INDEX_ROUND_SCALE = 3;
constexpr size_t ATTR_INDEX_CLAMP_LIMIT = 4;
constexpr size_t ATTR_INDEX_OUTPUT_ORIGIN = 6;
constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_WEIGHT = 1;
constexpr size_t INPUT_INDEX_GROUP_INDEX = 2;
constexpr size_t OUTPUT_INDEX_Y = 0;
constexpr size_t OUTPUT_INDEX_Y_SCALE = 1;
constexpr size_t OUTPUT_INDEX_Y_ORIGIN = 2;
constexpr size_t CACHE_LINE_SIZE = 128;
constexpr float DEFAULT_CLAMP_LIMIT = -1.0f;
constexpr int64_t BLOCK_QUANT_TILING_KEY = 1000;
constexpr int64_t BLOCK_QUANT_YORIGIN_TILING_KEY = 1100;
constexpr int64_t MX_QUANT_TILING_KEY = 2000;
constexpr int64_t MX_QUANT_YORIGIN_TILING_KEY = 2100;
constexpr int64_t MXFP4_QUANT_TILING_KEY = 3000;

int64_t ShapeElementNum(const gert::Shape& shape)
{
    int64_t elementNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        elementNum *= shape.GetDim(i);
    }
    return elementNum;
}
} // namespace

ge::graphStatus SwigluGroupQuantTiling::GetPlatformInfoCommon(gert::TilingContext* context, uint64_t& coreNum,
                                                              uint64_t& ubSize)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context->GetCompileInfo<SwigluGroupQuantCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize = ubSizePlatForm;
    }
    OP_CHECK_IF((coreNum == 0 || ubSize == 0),
                OP_LOGE(context->GetNodeName(), "GetPlatformInfo failed, coreNum=%lu, ubSize=%lu.", coreNum, ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetPlatformInfo()
{
    auto ret = GetPlatformInfoCommon(context_, coreNum_, ubSize_);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    if (context_->GetPlatformInfo() != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetClampLimitAttr(const gert::RuntimeAttrs* attrs)
{
    auto clampLimitAttr = attrs->GetAttrPointer<float>(ATTR_INDEX_CLAMP_LIMIT);
    if (clampLimitAttr != nullptr) {
        // DEFAULT_CLAMP_LIMIT means user did not pass clamp_limit.
        if (*clampLimitAttr != DEFAULT_CLAMP_LIMIT) {
            OP_CHECK_IF(!(*clampLimitAttr > 0.0f),
                        OP_LOGE(context_->GetNodeName(), "attr clamp_limit should be greater than 0.0, got %f.",
                                *clampLimitAttr),
                        return ge::GRAPH_FAILED);
            clampLimit_ = *clampLimitAttr;
            hasClampLimit_ = 1;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto dstTypeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DST_TYPE);
    dstType_ = dstTypeAttr == nullptr ? ge::DT_FLOAT8_E4M3FN : static_cast<ge::DataType>(*dstTypeAttr);
    OP_CHECK_IF((dstType_ != ge::DT_FLOAT8_E4M3FN && dstType_ != ge::DT_FLOAT8_E5M2 && dstType_ != ge::DT_FLOAT4_E2M1 &&
                 dstType_ != ge::DT_FLOAT4_E1M2),
                OP_LOGE(context_->GetNodeName(),
                        "attr dst_type only support (FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2), got %d.",
                        static_cast<int>(dstType_)),
                return ge::GRAPH_FAILED);
    isMxFp4Quant_ = dstType_ == ge::DT_FLOAT4_E2M1 || dstType_ == ge::DT_FLOAT4_E1M2;

    auto quantModeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    quantMode_ = quantModeAttr == nullptr ? BLOCK_QUANT : *quantModeAttr;
    OP_CHECK_IF((quantMode_ != BLOCK_QUANT && quantMode_ != MX_QUANT),
                OP_LOGE(context_->GetNodeName(), "attr quant_mode only support 0(block_quant) or 1(mx_quant), got %ld.",
                        quantMode_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (isMxFp4Quant_ && quantMode_ != MX_QUANT),
        OP_LOGE(context_->GetNodeName(),
                "attr quant_mode must be 1(mx_quant) when dst_type is FLOAT4_E2M1/FLOAT4_E1M2, got %ld.", quantMode_),
        return ge::GRAPH_FAILED);

    auto blockSizeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_BLOCK_SIZE);
    int64_t blockSize = blockSizeAttr == nullptr ? 0 : *blockSizeAttr;
    int64_t expectedBlockSize = quantMode_ == MX_QUANT ? PER_MX_FP16 : PER_BLOCK_FP16;
    OP_CHECK_IF((blockSize != 0 && blockSize != expectedBlockSize),
                OP_LOGE(context_->GetNodeName(), "attr block_size should be 0 or %ld when quant_mode is %ld, got %ld.",
                        expectedBlockSize, quantMode_, blockSize),
                return ge::GRAPH_FAILED);

    splitFactor_ = expectedBlockSize;

    auto roundScaleAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_ROUND_SCALE);
    if (roundScaleAttr != nullptr) {
        roundScale_ = (*roundScaleAttr) ? 1 : 0;
    }
    OP_CHECK_IF((quantMode_ == MX_QUANT && roundScale_ != 1),
                OP_LOGE(context_->GetNodeName(), "attr round_scale should be true when quant_mode is 1."),
                return ge::GRAPH_FAILED);

    auto outputOriginAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_OUTPUT_ORIGIN);
    if (outputOriginAttr != nullptr) {
        outputOrigin_ = (*outputOriginAttr) ? 1 : 0;
    }

    if (GetClampLimitAttr(attrs) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CheckWeightInfo()
{
    auto weightDesc = context_->GetOptionalInputDesc(INPUT_INDEX_WEIGHT);
    if (weightDesc != nullptr) {
        auto weightDtype = weightDesc->GetDataType();
        OP_CHECK_IF((weightDtype != ge::DT_FLOAT),
                    OP_LOGE(context_->GetNodeName(), "input weight dtype should be FLOAT, got %d.",
                            static_cast<int>(weightDtype)),
                    return ge::GRAPH_FAILED);
        auto weightShape = context_->GetOptionalInputShape(INPUT_INDEX_WEIGHT);
        if (weightShape != nullptr) {
            auto weightStorageShape = weightShape->GetStorageShape();
            auto weightElementNum = ShapeElementNum(weightStorageShape);
            OP_CHECK_IF((weightElementNum != bs_),
                        OP_LOGE(context_->GetNodeName(),
                                "input weight element num should be equal to input x outer dim product, got %ld, "
                                "expected %ld.",
                                weightElementNum, bs_),
                        return ge::GRAPH_FAILED);
            hasWeight_ = true;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CheckGroupIndexInfo()
{
    auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_INDEX_GROUP_INDEX);
    if (groupIndexDesc != nullptr) {
        auto groupIndexDtype = groupIndexDesc->GetDataType();
        OP_CHECK_IF((groupIndexDtype != ge::DT_INT64),
                    OP_LOGE(context_->GetNodeName(), "input group_index dtype should be INT64, got %d.",
                            static_cast<int>(groupIndexDtype)),
                    return ge::GRAPH_FAILED);
        auto groupIndexShape = context_->GetOptionalInputShape(INPUT_INDEX_GROUP_INDEX);
        if (groupIndexShape != nullptr) {
            auto groupIndexStorageShape = groupIndexShape->GetStorageShape();
            g_ = 1;
            for (size_t i = 0; i < groupIndexStorageShape.GetDimNum(); i++) {
                g_ = g_ * groupIndexStorageShape.GetDim(i);
            }
            hasGroupIndex_ = true;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CheckOutputInfo(ge::DataType xDtype)
{
    auto yDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF(
        (yDtype != dstType_),
        OP_LOGE(context_->GetNodeName(), "output y dtype should be same as dst_type, got y dtype %d, dst_type %d.",
                static_cast<int>(yDtype), static_cast<int>(dstType_)),
        return ge::GRAPH_FAILED);

    auto yScaleDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yScaleDesc);
    auto yScaleDtype = yScaleDesc->GetDataType();
    auto expectedYScaleDtype = quantMode_ == MX_QUANT ? ge::DT_FLOAT8_E8M0 : ge::DT_FLOAT;
    OP_CHECK_IF((yScaleDtype != expectedYScaleDtype),
                OP_LOGE(context_->GetNodeName(), "output y_scale dtype should be %d when quant_mode is %ld, got %d.",
                        static_cast<int>(expectedYScaleDtype), quantMode_, static_cast<int>(yScaleDtype)),
                return ge::GRAPH_FAILED);

    auto yOriginDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y_ORIGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOriginDesc);
    auto yOriginDtype = yOriginDesc->GetDataType();
    OP_CHECK_IF((yOriginDtype != xDtype),
                OP_LOGE(context_->GetNodeName(),
                        "output y_origin dtype should be same as input x, got y_origin dtype %d, x dtype %d.",
                        static_cast<int>(yOriginDtype), static_cast<int>(xDtype)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetShapeAttrsInfoInner()
{
    // (b, s, hc_mix)
    auto shapeX = context_->GetInputShape(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);

    auto xStorageShape = shapeX->GetStorageShape();
    auto xDesc = context_->GetInputDesc(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto xDtype = xDesc->GetDataType();
    OP_CHECK_IF((xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16),
                OP_LOGE(context_->GetNodeName(), "input x dtype only support FLOAT16 or BFLOAT16, got %d.",
                        static_cast<int>(xDtype)),
                return ge::GRAPH_FAILED);
    auto xDimNum = xStorageShape.GetDimNum();
    OP_CHECK_IF((xDimNum == 0), OP_LOGE(context_->GetNodeName(), "input x dim num should be greater than 0."),
                return ge::GRAPH_FAILED);
    bs_ = 1;
    for (size_t i = 0; i < xDimNum - 1; i++) {
        bs_ = bs_ * xStorageShape.GetDim(i);
    }
    d_ = xStorageShape.GetDim(xDimNum - 1);
    OP_CHECK_IF((d_ < D_LIMIT || d_ % D_LIMIT != 0),
                OP_LOGE(context_->GetNodeName(),
                        "input x last dim should be greater than or equal to %ld and divisible by %ld, got %ld.",
                        D_LIMIT, D_LIMIT, d_),
                return ge::GRAPH_FAILED);

    if (CheckWeightInfo() == ge::GRAPH_FAILED || CheckGroupIndexInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    // Get Attrs
    if (GetAttr() == ge::GRAPH_FAILED) {
        OP_LOGE(context_->GetNodeName(), "Get attr failed.");
        return ge::GRAPH_FAILED;
    }

    if (CheckOutputInfo(xDtype) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    splitD_ = d_ / 2;
    scaleCol_ = CeilDiv(splitD_, splitFactor_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcGroupIndexTiling()
{
    if (!hasGroupIndex_) {
        return ge::GRAPH_SUCCESS;
    }

    gFactor_ = g_;
    int64_t groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / sizeof(int64_t)) * DOUBLE_BUFFER * sizeof(int64_t);
    int64_t groupIndexSumSize = BLOCK_SIZE;
    if (groupIndexSize + groupIndexSumSize <= static_cast<int64_t>(ubSize_)) {
        gLoop_ = 1;
        tailGFactor_ = gFactor_;
        return ge::GRAPH_SUCCESS;
    }

    int64_t base = 2;
    int64_t maxBase = std::max(g_, base);
    while (base <= maxBase) {
        gFactor_ = CeilDiv(g_, base);
        groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / sizeof(int64_t)) * DOUBLE_BUFFER * sizeof(int64_t);
        if (groupIndexSize + groupIndexSumSize < static_cast<int64_t>(ubSize_)) {
            break;
        }
        base++;
    }
    gFactor_ = std::max(gFactor_, static_cast<int64_t>(1));
    if (gFactor_ > static_cast<int64_t>(CACHE_LINE_SIZE / sizeof(int64_t))) {
        gFactor_ = DownAlign(gFactor_, CACHE_LINE_SIZE / sizeof(int64_t));
        gFactor_ = std::max(gFactor_, static_cast<int64_t>(1));
    }
    gLoop_ = CeilDiv(g_, gFactor_);
    tailGFactor_ = g_ % gFactor_ == 0 ? gFactor_ : g_ % gFactor_;
    return ge::GRAPH_SUCCESS;
}

void SwigluGroupQuantTiling::InitCoreTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;
}

void SwigluGroupQuantTiling::SetFullDTiling()
{
    dLoop_ = 1;
    dFactor_ = splitD_;
    tailDFactor_ = dFactor_;
}

void SwigluGroupQuantTiling::SetSplitDTiling()
{
    scaleCol_ = CeilDiv(splitD_, splitFactor_);
    dLoop_ = CeilDiv(splitD_, dFactor_);
    tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
}

void SwigluGroupQuantTiling::SetRowLoopTiling()
{
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;
}

int64_t SwigluGroupQuantTiling::AddWeightSize(int64_t totalSize, int64_t rowFactor) const
{
    return hasWeight_ ? totalSize + RoundUp(rowFactor, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER : totalSize;
}

int64_t SwigluGroupQuantTiling::AddYOriginSize(int64_t totalSize, int64_t rowFactor, int64_t dFactor) const
{
    int64_t yOriginSize = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    return outputOrigin_ ? totalSize + yOriginSize : totalSize;
}

int64_t SwigluGroupQuantTiling::CalcMxQuantTotalSize(int64_t rowFactor, int64_t dFactor) const
{
    int64_t tryScaleCol = CeilDiv(dFactor, splitFactor_);
    int64_t x0Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t x1Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t ySize = rowFactor * RoundUp(dFactor, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t scaleSize = RoundUp(rowFactor * tryScaleCol, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t yFp32Size = rowFactor * RoundUp(dFactor, B32_ALIGN_NUM) * B32_BYTES;
    int64_t invScaleSize = rowFactor * RoundUp(CeilDiv(dFactor, B32_ALIGN_NUM), B32_ALIGN_NUM) * B32_BYTES;
    int64_t totalSize = x0Size + x1Size + ySize + scaleSize + yFp32Size + invScaleSize;
    totalSize = AddWeightSize(totalSize, rowFactor);
    return AddYOriginSize(totalSize, rowFactor, dFactor);
}

int64_t SwigluGroupQuantTiling::CalcBlockQuantTotalSize(int64_t rowFactor, int64_t dFactor) const
{
    int64_t tryScaleCol = CeilDiv(dFactor, splitFactor_);
    int64_t x0Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t x1Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t ySize = rowFactor * RoundUp(dFactor, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t scaleSize = RoundUp(rowFactor * tryScaleCol, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER;
    int64_t totalSize = x0Size + x1Size + ySize + scaleSize;
    totalSize = AddWeightSize(totalSize, rowFactor);
    return AddYOriginSize(totalSize, rowFactor, dFactor);
}

int64_t SwigluGroupQuantTiling::CalcMxFp4QuantTotalSize(int64_t rowFactor, int64_t dFactor) const
{
    int64_t tryScaleCol = CeilDiv(dFactor, splitFactor_);
    int64_t x0Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t x1Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t swigluSize = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES;
    int64_t maxExpSize = rowFactor * RoundUp(tryScaleCol, B16_ALIGN_NUM) * B16_BYTES;
    int64_t invScaleSize = rowFactor * RoundUp(tryScaleCol, B16_ALIGN_NUM) * B16_BYTES;
    int64_t ySize = rowFactor * RoundUp(CeilDiv(dFactor, FP4_PACK_NUM), FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t scaleSize = rowFactor * RoundUp(tryScaleCol, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t totalSize = x0Size + x1Size + swigluSize + maxExpSize + invScaleSize + ySize + scaleSize;
    return AddWeightSize(totalSize, rowFactor);
}

void SwigluGroupQuantTiling::CalcDAndRowFactorTiling(int64_t rowOnceLoop, int64_t dStep, TotalSizeFunc calcTotalSize)
{
    rowFactor_ = rowOnceLoop;
    if ((this->*calcTotalSize)(rowOnceLoop, splitD_) <= static_cast<int64_t>(ubSize_)) {
        SetFullDTiling();
    } else {
        int64_t base = 1;
        int64_t maxBase = CeilDiv(splitD_, dStep);
        dFactor_ = dStep;
        while (base <= maxBase) {
            int64_t tryDFactor = base * dStep;
            if (tryDFactor > splitD_) {
                tryDFactor = splitD_;
            }
            if ((this->*calcTotalSize)(rowOnceLoop, tryDFactor) > static_cast<int64_t>(ubSize_)) {
                break;
            }
            dFactor_ = tryDFactor;
            if (tryDFactor == splitD_) {
                break;
            }
            base++;
        }
        SetSplitDTiling();
    }

    if (dFactor_ == splitD_) {
        int64_t lo = 1;
        int64_t hi = rowOfFormerBlock_;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo + 1) / 2;
            if ((this->*calcTotalSize)(mid, dFactor_) <= static_cast<int64_t>(ubSize_)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        rowFactor_ = lo;
    }

    SetRowLoopTiling();
}

ge::graphStatus SwigluGroupQuantTiling::CalcMxQuantOpTiling()
{
    InitCoreTiling();
    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t dChunkAlign = REPEAT_SIZE / B16_BYTES;
    CalcDAndRowFactorTiling(rowOnceLoop, dChunkAlign, &SwigluGroupQuantTiling::CalcMxQuantTotalSize);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcBlockQuantOpTiling()
{
    InitCoreTiling();
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, static_cast<int64_t>(1));
    CalcDAndRowFactorTiling(rowOnceLoop, splitFactor_, &SwigluGroupQuantTiling::CalcBlockQuantTotalSize);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcMxFp4QuantOpTiling()
{
    InitCoreTiling();
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, static_cast<int64_t>(1));
    CalcDAndRowFactorTiling(rowOnceLoop, splitFactor_, &SwigluGroupQuantTiling::CalcMxFp4QuantTotalSize);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void SwigluGroupQuantTiling::SetTilingData()
{
    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_splitD(splitD_);
    tilingData_.set_scaleCol(scaleCol_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_roundScale(roundScale_);
    tilingData_.set_outputOrigin(outputOrigin_);
    tilingData_.set_clampLimit(clampLimit_);
    tilingData_.set_g(g_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_gLoop(gLoop_);
    tilingData_.set_gFactor(gFactor_);
    tilingData_.set_tailGFactor(tailGFactor_);
    tilingData_.set_coreNum(coreNum_);
    tilingData_.set_hasClampLimit(hasClampLimit_);
}

ge::graphStatus SwigluGroupQuantTiling::CalcOpTiling()
{
    ge::graphStatus status;
    status = CalcGroupIndexTiling();
    if (status == ge::GRAPH_FAILED) {
        return status;
    }
    if (quantMode_ == BLOCK_QUANT) {
        status = CalcBlockQuantOpTiling();
    } else if (quantMode_ == MX_QUANT) {
        if (isMxFp4Quant_) {
            status = CalcMxFp4QuantOpTiling();
        } else {
            status = CalcMxQuantOpTiling();
        }
    }
    return status;
}

void SwigluGroupQuantTiling::SetTilingKey()
{
    if (quantMode_ == BLOCK_QUANT) {
        tilingKey_ = outputOrigin_ ? BLOCK_QUANT_YORIGIN_TILING_KEY : BLOCK_QUANT_TILING_KEY;
        context_->SetTilingKey(tilingKey_);
        return;
    }
    if (quantMode_ == MX_QUANT) {
        if (isMxFp4Quant_) {
            tilingKey_ = MXFP4_QUANT_TILING_KEY;
        } else if (outputOrigin_) {
            tilingKey_ = MX_QUANT_YORIGIN_TILING_KEY;
        } else {
            tilingKey_ = MX_QUANT_TILING_KEY;
        }
        context_->SetTilingKey(tilingKey_);
        return;
    }
    context_->SetTilingKey(tilingKey_);
}

ge::graphStatus SwigluGroupQuantTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetTilingKey();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::PostTiling()
{
    if (hasGroupIndex_) {
        context_->SetBlockDim(coreNum_);
    } else {
        context_->SetBlockDim(usedCoreNums_);
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
