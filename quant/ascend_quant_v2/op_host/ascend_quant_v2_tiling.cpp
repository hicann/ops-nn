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
 * \file ascend_quant_v2_tiling.cc
 * \brief
 */

#include "ascend_quant_v2_tiling.h"
#include "ascend_quant_v2_regbase_tiling.h"
#include "tiling_base/tiling_util.h"

namespace optiling {
using namespace Ops::NN::OpTiling;
namespace ascendquantv2 {
constexpr size_t g_XInputIndex = 0;
constexpr size_t g_ScaleIndex = 1;
constexpr size_t g_OffsetIndex = 2;
constexpr size_t g_AttrSqrtMode = 0;
constexpr size_t g_AttrRoundMode = 1;
constexpr size_t g_AttrDstType = 2;
constexpr size_t g_AttrAxis = 3;
constexpr int64_t g_SyncWorkSpaceSize = static_cast<int64_t>(16 * 1024 * 1024);
constexpr int64_t g_CacheLineV220 = 512;
constexpr int64_t g_BaseLen = 128;
constexpr int64_t g_BlockSize = 32;
constexpr int64_t g_HalfBase = 2;
constexpr int64_t nz_format = 29;
constexpr int64_t dim_2 = 2;
constexpr int64_t length = 16;
constexpr int32_t g_AxisMax = 2;
constexpr size_t g_FirstShapeDim = 0;
constexpr size_t g_SecondShapeDim = 1;
constexpr size_t g_ThirdShapeDim = 2;
static constexpr int64_t INT4_NUMS_IN_INT8_SPACE = 2;
static const gert::Shape g_vec_1_shape = {1};

ge::graphStatus AscendQuantV2::DoAscendQuantV2Tiling()
{
    OP_CHECK_IF(
        (GetCompileInfo() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "DoAscendQuantV2Tiling GetCompileInfo Failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (GetOpParam() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "DoAscendQuantV2Tiling GetOpParam Failed."), return ge::GRAPH_FAILED);

    CalcTiling();
    CalcTilingKey();
    WriteTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendQuantV2::DoAscendQuantV2NZTiling()
{
    OP_CHECK_IF(
        (GetCompileInfo() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "DoAscendQuantV2NZTiling GetCompileInfo Failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (GetOpParam() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "DoAscendQuantV2NZTiling GetOpParam Failed."), return ge::GRAPH_FAILED);

    CalcTilingNz();
    CalcTilingKey();
    WriteTilingDataNz();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendQuantV2::GetCompileInfo()
{
    auto compileInfo = context_->GetCompileInfo<AscendQuantV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->vectorCoreNum;
    ubSize_ = compileInfo->ubSize;
    isAscend910B_ = compileInfo->isAscend910B;
    OP_CHECK_IF(
        (coreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE(
            context_->GetNodeName(), "AscendQuantV2 GetCompileInfo Failed, coreNum:%u, ubSize:%lu.", coreNum_, ubSize_),
        return ge::GRAPH_FAILED);

    if (compileInfo->isAscend910B) {
        cacheLine_ = g_CacheLineV220;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendQuantV2::CheckInputValid(const gert::Shape& input1, const gert::Shape& input2, int32_t axis) const
{
    size_t input1DimNum = input1.GetDimNum();
    size_t input2DimNum = input2.GetDimNum();
    if (input1DimNum != input2DimNum && static_cast<int32_t>(input2DimNum) != 1) {
        OP_LOGE(context_->GetNodeName(), "scale/offset's number of dimensions is invalid, should be same as x or 1");
        return ge::GRAPH_FAILED;
    }
    size_t input1Axis =
        (axis >= 0 ? static_cast<size_t>(axis) : static_cast<size_t>(input1DimNum) + static_cast<size_t>(axis));
    size_t input2Axis = (input2DimNum == static_cast<size_t>(1) ? 0 : static_cast<size_t>(input1Axis));
    int64_t input1Dim = input1.GetDim(input1Axis);
    int64_t input2Dim = input2.GetDim(input2Axis);
    if (input1Dim != input2Dim && input2Dim != 1) {
        OP_LOGE(
            context_->GetNodeName(),
            "scale/offset dim(%zu)'s value(%ld) is invalid, should be same as x dim(%zu)'s value(%ld) or 1", input2Axis,
            input2Dim, input1Axis, input1Dim);
        return ge::GRAPH_FAILED;
    }

    auto input2Size = input2.GetShapeSize();
    if (input2Size != input2Dim) {
        OP_LOGE(
            context_->GetNodeName(),
            "scale/offset should be 1-dimensional, otherwise all dimensions except the dimension specified by axis "
            "should be 1.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

RoundMode AscendQuantV2::GetRoundMode(std::string& roundMode)
{
    if (roundMode == "round") {
        return RoundMode::MODE_ROUND;
    } else if (roundMode == "floor") {
        return RoundMode::MODE_FLOOR;
    } else if (roundMode == "ceil") {
        return RoundMode::MODE_CEIL;
    } else if (roundMode == "trunc") {
        return RoundMode::MODE_TRUNC;
    }
    return RoundMode::MODE_UNDEFINED;
}

ge::graphStatus AscendQuantV2::CheckSupport310P()
{
    if (dstType_ != ge::DT_INT8) {
        OP_LOGE(context_->GetNodeName(), "dstType only support int8 in 310P");
        return ge::GRAPH_FAILED;
    }
    axis_ = -1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendQuantV2::CheckShapeEqual(const gert::Shape& shape1, const gert::Shape& shape2) const
{
    size_t x1DimNum = shape1.GetDimNum();
    size_t x2DimNum = shape2.GetDimNum();
    OP_CHECK_IF(
        x1DimNum != x2DimNum, OP_LOGE(context_->GetNodeName(), "scale shape and offset shape must be same."),
        return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < x1DimNum; i++) {
        OP_CHECK_IF(
            shape1.GetDim(i) != shape2.GetDim(i),
            OP_LOGE(context_->GetNodeName(), "scale shape and offset shape must be same."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendQuantV2::CheckAttrs(const gert::Shape& xShape)
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const auto* sqrtMode = attrs->GetAttrPointer<bool>(g_AttrSqrtMode);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sqrtMode);
    sqrtMode_ = static_cast<int16_t>(*sqrtMode);
    const char* roundMode = attrs->GetAttrPointer<char>(g_AttrRoundMode);
    OP_CHECK_NULL_WITH_CONTEXT(context_, roundMode);
    std::string roundModeStr = roundMode;
    roundMode_ = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode_ == RoundMode::MODE_UNDEFINED),
        OP_LOGE(
            context_->GetNodeName(), "invalid round mode:%s, value should be in [round/trunc/ceil/floor].", roundMode),
        return ge::GRAPH_FAILED);
    // get dstType
    const int32_t* dstType = attrs->GetAttrPointer<int32_t>(g_AttrDstType);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstType);
    dstType_ = *dstType;
    // get axis
    const int32_t* axis = attrs->GetAttrPointer<int32_t>(g_AttrAxis);
    OP_CHECK_NULL_WITH_CONTEXT(context_, axis);
    axis_ = *axis;
    // get x dim num
    int32_t xDimNum = static_cast<int32_t>(xShape.GetDimNum());
    // check 310P support dstType
    if (!isAscend910B_) {
        OP_CHECK_IF(
            CheckSupport310P() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "check 310P support failed."),
            return ge::GRAPH_FAILED);
    }
    // check dstType and output dtype, must be same
    if (dstType_ != ge::DT_INT8 && dstType_ != ge::DT_INT4) {
        OP_LOGE(context_->GetNodeName(), "dst type:%d is invalid, type should be int8 or int4", dstType_);
        return ge::GRAPH_FAILED;
    }
    if (dstType_ != yDtype_) {
        OP_LOGE(context_->GetNodeName(), "dst type:%d not equal output dtype:%d", dstType_, yDtype_);
        return ge::GRAPH_FAILED;
    }
    if (dstType_ == ge::DT_INT4 && (xShape.GetDim(xDimNum - 1) % INT4_NUMS_IN_INT8_SPACE)) {
        OP_LOGE(context_->GetNodeName(), "if dst_type is DT_INT4, x last dim must be divisible by 2");
        return ge::GRAPH_FAILED;
    }
    // check axis and shape, support last two dim
    if (axis_ >= xDimNum || axis_ < -xDimNum) {
        OP_LOGE(context_->GetNodeName(), "invalid axis:%d, axis can't exceed the dimension of x", axis_);
        return ge::GRAPH_FAILED;
    }
    if (xDimNum > g_AxisMax) {
        if (axis_ < -g_AxisMax || (axis_ >= 0 && axis_ < xDimNum - g_AxisMax)) {
            OP_LOGE(context_->GetNodeName(), "invalid axis:%d, axis must be one of last two dimensions", axis_);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void AscendQuantV2::MergeInputShape(const gert::Shape& input)
{
    int64_t shape0 = 1;
    int64_t shape1 = input.GetDim(input.GetDimNum() - 1);
    int64_t shape2 = 1;
    if (isPerTensor_) {
        // pertensor merge [1, x0*x1...*xn, 1]
        for (size_t idx = 0; idx < input.GetDimNum() - 1; ++idx) {
            shape1 = shape1 * input.GetDim(idx);
        }
    } else if (isPerHead_) {
        // perhead merge [x0*x1...*x(n-2), x(n-1), xn]
        for (size_t idx = 0; idx < input.GetDimNum() - g_AxisMax; ++idx) {
            shape0 = shape0 * input.GetDim(idx);
        }
        shape1 = input.GetDim(input.GetDimNum() - g_AxisMax);
        shape2 = input.GetDim(input.GetDimNum() - 1);
    } else {
        // perchannel merge [x0*x1*...*x(n-1), xn, 1]
        for (size_t idx = 0; idx < input.GetDimNum() - 1; ++idx) {
            shape0 = shape0 * input.GetDim(idx);
        }
    }
    // last dim is 1, perhead is perchannel
    if (isPerHead_ && shape2 == 1) {
        isPerHead_ = false;
    }
    // merge shape to 3 dim
    xInputShape_.SetDimNum(3);
    xInputShape_.SetDim(g_FirstShapeDim, shape0);
    xInputShape_.SetDim(g_SecondShapeDim, shape1);
    xInputShape_.SetDim(g_ThirdShapeDim, shape2);
    OP_LOGI(context_->GetNodeName(), "merge shape0:%ld, shape1:%ld, shape2:%ld", shape0, shape1, shape2);
}

void AscendQuantV2::MergeInputShapeNz(const gert::Shape& input) {
    int64_t shape0 = input.GetDim(0);
    int64_t shape1 = input.GetDim(1);
    int64_t shape2 = input.GetDim(dim_2);

    xInputShape_.SetDim(g_FirstShapeDim, shape0);
    xInputShape_.SetDim(g_SecondShapeDim, shape1);
    xInputShape_.SetDim(g_ThirdShapeDim, shape2);
    OP_LOGI(context_->GetNodeName(), "merge shape0:%ld, shape1:%ld, shape2:%ld", shape0, shape1, shape2);
}

const gert::Shape& AscendQuantV2::EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

ge::graphStatus AscendQuantV2::GetOpParam()
{
    auto xInput = context_->GetInputShape(g_XInputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    auto scaleInput = context_->GetInputShape(g_ScaleIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleInput);
    auto offsetInput = context_->GetOptionalInputShape(g_OffsetIndex);
    if (offsetInput == nullptr) {
        hasOffset_ = false;
    }
    auto xInputDesc = context_->GetInputDesc(g_XInputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    xDtype_ = xInputDesc->GetDataType();
    auto yInputDesc = context_->GetOutputDesc(0);
    auto xFormat = xInputDesc->GetFormat().GetStorageFormat();
    OP_CHECK_NULL_WITH_CONTEXT(context_, yInputDesc);
    yDtype_ = yInputDesc->GetDataType();

    const gert::Shape& xInputShape = EnsureNotScalar(xInput->GetStorageShape());
    const gert::Shape& scaleInputShape = EnsureNotScalar(scaleInput->GetStorageShape());
    OP_CHECK_IF(
        (CheckAttrs(xInputShape) != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "op attrs is invalid."),
        return ge::GRAPH_FAILED);
    // check the shape of the scale is valid
    OP_CHECK_IF(
        (CheckInputValid(xInputShape, scaleInputShape, axis_) != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "x and scale is invalid."), return ge::GRAPH_FAILED);
    // if offset is not null, check the shape of the offset
    if (hasOffset_) {
        const gert::Shape& offsetInputShape = EnsureNotScalar(offsetInput->GetStorageShape());
        // check scale and offset is same
        OP_CHECK_IF(
            (CheckShapeEqual(scaleInputShape, offsetInputShape) != ge::GRAPH_SUCCESS),
            OP_LOGE(context_->GetNodeName(), "scale and offset is invalid."), return ge::GRAPH_FAILED);
        // check the shape of the offset is valid
        OP_CHECK_IF(
            (CheckInputValid(xInputShape, offsetInputShape, axis_) != ge::GRAPH_SUCCESS),
            OP_LOGE(context_->GetNodeName(), "x and offset is invalid."), return ge::GRAPH_FAILED);
    }
    // check excute mode
    int32_t xDimNum = static_cast<int32_t>(xInputShape.GetDimNum());
    if (scaleInputShape.GetShapeSize() == 1) {
        isPerTensor_ = true;
    } else if (axis_ == -g_AxisMax || (axis_ == xDimNum - g_AxisMax && xDimNum > 1)) {
        // only 910B support perhead
        isPerHead_ = isAscend910B_;
    }

    if(xFormat != nz_format){
        MergeInputShape(xInputShape);
    }else {
        MergeInputShapeNz(xInputShape);
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t AscendQuantV2::GetCoreNum(int64_t factor, int64_t coreNum) const
{
    int64_t elePerCore = Ops::Base::CeilDiv(factor, static_cast<int64_t>(coreNum));
    uint32_t actCore = static_cast<uint32_t>(Ops::Base::CeilDiv(factor, elePerCore));
    return actCore;
}

uint32_t AscendQuantV2::GetCoreNumDoubleCut(int64_t shape0, int64_t shape1, int64_t coreNum) const
{
    if (shape0 == 0) {
        return 0;
    }
    int64_t yCoreNum = coreNum / shape0;
    if (yCoreNum <= 0) {
        return static_cast<uint32_t>(yCoreNum);
    }
    uint32_t actCoreNum = GetCoreNum(shape1, yCoreNum);
    return static_cast<uint32_t>(shape0 * static_cast<int64_t>(actCoreNum));
}

void AscendQuantV2::CalcBlockFactor(int64_t size)
{
    blockFactor_ = Ops::Base::CeilDiv(size, static_cast<int64_t>(actCoreNum_));
    if (blockAxis_ == 0) {
        blockFactor_ = blockFactor_ * blockBind_;
        blockTailFactor_ = xInputShape_.GetDim(blockAxis_) - blockFactor_ * (actCoreNum_ - 1);
    } else {
        int64_t shape = xInputShape_.GetDim(blockAxis_);
        int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
        if (dtypeSize != 0) {
            blockFactor_ = blockFactor_ * cacheLine_ / dtypeSize;
        } else {
            blockFactor_ = 0;
        }
        blockTailFactor_ =
            static_cast<int64_t>(shape) -
            static_cast<int64_t>(blockFactor_) * (static_cast<int64_t>(actCoreNum_) - static_cast<int64_t>(1));
    }
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
}

void AscendQuantV2::CalcBlockFactorDoubleCut(int64_t shape0, int64_t size)
{
    if (shape0 == 0) {
        return;
    }
    blockUnion_ = static_cast<int64_t>(actCoreNum_) / shape0;
    blockFactor_ = Ops::Base::CeilDiv(size, static_cast<int64_t>(blockUnion_));
    int64_t shape = xInputShape_.GetDim(1);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    blockFactor_ = dtypeSize != 0 ? blockFactor_ * cacheLine_ / dtypeSize : blockFactor_;
    blockTailFactor_ = shape - blockFactor_ * (blockUnion_ - 1);
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
}

int64_t AscendQuantV2::CalcMaxBaseLen(int64_t ubSize) const
{
    // set n == 1 to calc max base
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    int64_t yDtypeSize = ge::GetSizeByDataType(ge::DT_INT8);
    int64_t xCastDtypeSize = 0;
    if (xDtype_ == ge::DT_BF16 || xDtype_ == ge::DT_FLOAT16) {
        // bfloat16/float16 -> float32, cast size set to 4
        xCastDtypeSize = 4;
    }
    int64_t baseInput = hasOffset_ ? 3 : 2; // hasoffset means 3 input, else means 2 input
    if (isPerTensor_) {
        baseInput = 1;
    }
    int64_t totalBytes = xDtypeSize * baseInput + yDtypeSize + xCastDtypeSize * baseInput;
    return totalBytes == 0 ? g_BaseLen : ubSize / totalBytes;
}

int64_t AscendQuantV2::CalcMaxN(int64_t ubSize, int64_t base) const
{
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    int64_t yDtypeSize = ge::GetSizeByDataType(ge::DT_INT8);
    int64_t xCastDtypeSize = 0;
    if (xDtype_ == ge::DT_BF16 || xDtype_ == ge::DT_FLOAT16) {
        // bfloat16/float16 -> float32, cast size set to 4
        xCastDtypeSize = 4;
    }
    // except scale and offset input, input x cast base number one time
    int64_t baseInput = hasOffset_ ? 2 : 1; // except x, hasoffset means 2 input, else means 1 input
    if (isPerTensor_) {
        baseInput = 0;
    }
    int64_t leftXBytes =
        (ubSize - base * xDtypeSize * baseInput - base * xCastDtypeSize * baseInput - base * xCastDtypeSize);
    if (leftXBytes <= 0) {
        // n set to 1
        return 1;
    }
    // n related btypes
    int64_t totalNBytes = xDtypeSize + yDtypeSize;
    if (base == 0) {
        return 0;
    }
    return leftXBytes / totalNBytes / base;
}

void AscendQuantV2::CalcUBFactor(int64_t cacheLineNum)
{
    // ub can split to three input: x_dtype_size * n * base, x_dtype_size * base, x_dtype_size * base
    // and one output: y_dtype_size * n * base
    // if bfloat16 and float32, always need cast memory size for all the input
    int64_t availableUb = static_cast<int64_t>(ubSize_) - static_cast<int64_t>(reserveUb_);
    int64_t maxBase = CalcMaxBaseLen(availableUb);
    maxBase = Ops::Base::FloorAlign(maxBase, cacheLineNum);
    // block cut axis 0, means all dim 1 is continous, else each core handle blockFactor
    int64_t blockBase = blockAxis_ == 0 ? xInputShape_.GetDim(1) : blockFactor_;
    blockBase = Ops::Base::CeilAlign(blockBase, cacheLineNum);
    if (blockBase <= maxBase / g_HalfBase) {
        // need calc max n with real base
        int64_t maxN = CalcMaxN(availableUb, blockBase);
        int64_t blockInnerSize = blockAxis_ == 0 ? blockFactor_ : xInputShape_.GetDim(0);
        ubAxis_ = 0;
        baseN_ = std::min(maxN, blockInnerSize);
        baseLen_ = Ops::Base::CeilAlign(blockBase, cacheLineNum);
    } else {
        ubAxis_ = 1;
        baseN_ = 1;
        baseLen_ = std::min(blockBase, maxBase);
    }
}

void AscendQuantV2::CalcUBFactorDoubleCut(int64_t cacheLineNum)
{
    int64_t availableUb = static_cast<int64_t>(ubSize_) - static_cast<int64_t>(reserveUb_);
    int64_t maxBase = CalcMaxBaseLen(availableUb);
    maxBase = Ops::Base::FloorAlign(maxBase, cacheLineNum);
    int64_t blockBase = Ops::Base::CeilAlign(blockFactor_, cacheLineNum);
    ubAxis_ = 1;
    baseN_ = 1;
    baseLen_ = std::min(blockBase, maxBase);
}

int64_t AscendQuantV2::GetNewShape0(int64_t shape0, int64_t shape1)
{
    // 310p copy out alignment
    if (shape1 >= g_BlockSize) {
        return shape0;
    }
    if (isAscend910B_) {
        return shape0;
    }
    blockBind_ = Ops::Base::CeilDiv(g_BlockSize, shape1);
    return std::max(Ops::Base::CeilDiv(shape0, blockBind_), static_cast<int64_t>(1));
}

void AscendQuantV2::CalcTiling()
{
    if (isPerHead_) {
        CalcPerHeadTiling();
        return;
    }
    int64_t shape0 = xInputShape_.GetDim(0);
    int64_t shape1 = xInputShape_.GetDim(1);
    shape0 = GetNewShape0(shape0, shape1);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF(dtypeSize == 0, OP_LOGE(context_->GetNodeName(), "dtypeSize should not be zero."), return);
    if (cacheLine_ == 0 || dtypeSize == 0) {
        return;
    }
    int64_t cacheLineNum = Ops::Base::CeilDiv(shape1, cacheLine_ / dtypeSize);
    uint32_t actCoreNum0 = GetCoreNum(shape0, static_cast<int64_t>(coreNum_));
    uint32_t actCoreNum1 = GetCoreNum(cacheLineNum, static_cast<int64_t>(coreNum_));
    actCoreNum_ = actCoreNum0 >= actCoreNum1 ? actCoreNum0 : actCoreNum1;
    if (actCoreNum_ < coreNum_ && shape0 > 1 && shape1 > g_BlockSize) {
        // not full use core, try cut shape0 and shae1
        uint32_t actCoreNum2 = GetCoreNumDoubleCut(shape0, cacheLineNum, static_cast<int64_t>(coreNum_));
        if (actCoreNum2 > actCoreNum_) {
            actCoreNum_ = actCoreNum2;
            useDoubleCut = true;
        }
    }

    if (useDoubleCut) {
        blockAxis_ = 11; // 11 means double cut
        CalcBlockFactorDoubleCut(shape0, cacheLineNum);
        CalcUBFactorDoubleCut(cacheLine_ / dtypeSize);
    } else {
        blockAxis_ = actCoreNum0 >= actCoreNum1 ? 0 : 1;
        int64_t size = actCoreNum0 >= actCoreNum1 ? shape0 : cacheLineNum;
        CalcBlockFactor(size);
        CalcUBFactor(cacheLine_ / dtypeSize);
    }
}

void AscendQuantV2::CalcTilingNz()
{
    E_ = xInputShape_.GetDim(0);
    K_ = xInputShape_.GetDim(1);
    N_ = xInputShape_.GetDim(dim_2);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF(dtypeSize == 0, OP_LOGE(context_->GetNodeName(), "dtypeSize should not be zero."), return);
    needCoreNum_ = K_ / length;
    if(needCoreNum_ >= static_cast<int64_t>(coreNum_)){
        needCoreNum_ = static_cast<int64_t>(coreNum_);
    }
}

void AscendQuantV2::CalcPerHeadTiling()
{
    int64_t shape0 = xInputShape_.GetDim(g_FirstShapeDim);
    int64_t shape1 = xInputShape_.GetDim(g_SecondShapeDim);
    int64_t shape2 = xInputShape_.GetDim(g_ThirdShapeDim);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF(dtypeSize == 0, OP_LOGE(context_->GetNodeName(), "dtypeSize should not be zero."), return);
    if (cacheLine_ == 0 || dtypeSize == 0) {
        return;
    }
    int64_t cacheLineNum = Ops::Base::CeilDiv(shape2, cacheLine_ / dtypeSize);
    // split core in S, N, D
    // eg. [20, x, x] 40
    uint32_t actCoreNum0 = GetCoreNum(shape0, static_cast<int64_t>(coreNum_));
    uint32_t actCoreNum1 = GetCoreNumDoubleCut(shape0, shape1, static_cast<int64_t>(coreNum_));
    uint32_t actCoreNum2 = GetCoreNumDoubleCut(shape0 * shape1, cacheLineNum, static_cast<int64_t>(coreNum_));

    blockAxis_ = 0;
    actCoreNum_ = actCoreNum0;
    if (actCoreNum1 > actCoreNum_) {
        blockAxis_ = 1;
        actCoreNum_ = actCoreNum1;
    }
    if (actCoreNum2 > actCoreNum_ && shape2 > g_BlockSize) {
        blockAxis_ = static_cast<int>(g_ThirdShapeDim);
        actCoreNum_ = actCoreNum2;
    }

    CalcPerHeadBlockFactor();
    CalcPerHeadUBFactor(cacheLine_ / dtypeSize);
}

void AscendQuantV2::CalcPerHeadBlockFactor()
{
    int64_t shape0 = xInputShape_.GetDim(g_FirstShapeDim);
    int64_t shape1 = xInputShape_.GetDim(g_SecondShapeDim);
    int64_t shape2 = xInputShape_.GetDim(g_ThirdShapeDim);
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    if (blockAxis_ == 0) {
        blockFactor_ = Ops::Base::CeilDiv(shape0, static_cast<int64_t>(actCoreNum_));
        blockTailFactor_ = shape0 - blockFactor_ * (static_cast<int64_t>(actCoreNum_) - static_cast<int64_t>(1));
    } else if (blockAxis_ == 1) {
        if (shape0 == 0) {
            return;
        }
        blockUnion_ = static_cast<int64_t>(actCoreNum_) / shape0;
        blockFactor_ = Ops::Base::CeilDiv(shape1, blockUnion_);
        blockTailFactor_ = shape1 - blockFactor_ * (static_cast<int64_t>(blockUnion_) - static_cast<int64_t>(1));
    } else {
        if (cacheLine_ == 0 || dtypeSize == 0) {
            return;
        }
        int64_t cacheLineNum = Ops::Base::CeilDiv(shape2, cacheLine_ / dtypeSize);
        blockUnion_ =
            (actCoreNum_ != 0U && shape0 != static_cast<int64_t>(0) && shape1 != static_cast<int64_t>(0)) ?
                static_cast<int64_t>(actCoreNum_) / static_cast<int64_t>(shape0) / static_cast<int64_t>(shape1) :
                static_cast<int64_t>(1);
        blockFactor_ = Ops::Base::CeilDiv(cacheLineNum, blockUnion_) * cacheLine_ / dtypeSize;
        blockTailFactor_ = shape2 - blockFactor_ * (blockUnion_ - 1);
    }
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
}

void AscendQuantV2::CalcPerHeadUBFactor(int64_t cacheLineNum)
{
    int64_t shape1 = xInputShape_.GetDim(g_SecondShapeDim);
    int64_t shape2 = xInputShape_.GetDim(g_ThirdShapeDim);
    shape2 = Ops::Base::CeilAlign(shape2, cacheLineNum);

    int64_t availableUb = static_cast<int64_t>(ubSize_) - static_cast<int64_t>(reserveUb_);
    int64_t maxBase = CalcMaxBaseLen(availableUb);
    maxBase = Ops::Base::FloorAlign(maxBase, cacheLineNum);
    int64_t blockBase = Ops::Base::CeilAlign(blockFactor_, cacheLineNum);

    // 圈复杂度重构
    if (blockAxis_ == 0) {
        if (shape1 * shape2 <= maxBase) {
            ubAxis_ = 0;
            baseN_ = shape1;
            baseLen_ = shape2;
        } else if (shape2 <= maxBase) {
            ubAxis_ = 1;
            if (shape2 == 0) {
                return;
            }
            baseN_ = maxBase / shape2; // must less than shape1
            baseLen_ = shape2;
        } else {
            ubAxis_ = static_cast<decltype(ubAxis_)>(g_ThirdShapeDim);
            baseN_ = 1;
            baseLen_ = maxBase;
        }
    } else if (blockAxis_ == 1) {
        if (shape2 <= maxBase) {
            ubAxis_ = 1;
            if (shape2 == 0) {
                return;
            }
            baseN_ = std::min(blockFactor_, maxBase / shape2);
            baseLen_ = shape2;
        } else {
            ubAxis_ = static_cast<decltype(ubAxis_)>(g_ThirdShapeDim);
            baseN_ = 1;
            baseLen_ = maxBase;
        }
    } else {
        ubAxis_ = static_cast<decltype(ubAxis_)>(g_ThirdShapeDim);
        baseN_ = 1;
        baseLen_ = std::min(blockBase, maxBase);
    }
}

void AscendQuantV2::CalcTilingKey()
{
    tilingKey_ = static_cast<uint64_t>(QuantKey::KEY_PER_CHANNEL);
    auto xInputDesc = context_->GetInputDesc(g_XInputIndex);
    auto xFormat = xInputDesc->GetFormat().GetStorageFormat();
    if(xFormat != nz_format){
        if (isPerTensor_) {
        tilingKey_ = static_cast<uint64_t>(QuantKey::KEY_PER_TENSOR);
        }
        if (isPerHead_) {
            tilingKey_ = static_cast<uint64_t>(QuantKey::KEY_PER_HEAD);
        }
    }else{
        tilingKey_ = static_cast<uint64_t>(QuantKey::KEY_NZ);
    }
}

void AscendQuantV2::WriteTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum:%d, tilingKey:%lu", coreNum_, tilingKey_);
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(tilingKey_);

    OP_LOGI(
        context_->GetNodeName(), "hasOffset:%d, sqrtMode:%d, roundMode:%d, dstType:%d", hasOffset_, sqrtMode_,
        static_cast<int16_t>(roundMode_), dstType_);
    tilingData_.set_hasOffset(static_cast<int16_t>(hasOffset_));
    tilingData_.set_sqrtMode(static_cast<int16_t>(sqrtMode_));
    tilingData_.set_roundMode(static_cast<int16_t>(roundMode_));

    OP_LOGI(
        context_->GetNodeName(),
        "actCoreNum:%d, blockAxis:%d, blockUnion:%ld, blockFactor:%ld, blockTailFactor:%ld,"
        "ubAxis:%d, baseN:%ld, baseLen:%ld",
        actCoreNum_, blockAxis_, blockUnion_, blockFactor_, blockTailFactor_, ubAxis_, baseN_, baseLen_);
    tilingData_.set_numCore(actCoreNum_);
    tilingData_.set_blockAxis(blockAxis_);
    tilingData_.set_ubAxis(ubAxis_);
    tilingData_.set_blockUnion(blockUnion_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockTailFactor(blockTailFactor_);
    tilingData_.set_baseN(baseN_);
    tilingData_.set_baseLen(baseLen_);

    tilingData_.set_dim0(xInputShape_.GetDim(g_FirstShapeDim));
    tilingData_.set_dim1(xInputShape_.GetDim(g_SecondShapeDim));
    tilingData_.set_dim2(xInputShape_.GetDim(g_ThirdShapeDim));
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(g_SyncWorkSpaceSize);
}

void AscendQuantV2::WriteTilingDataNz()
{
    OP_LOGD(context_->GetNodeName(), "coreNum:%d, tilingKey:%lu", coreNum_, tilingKey_);
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(tilingKey_);

    OP_LOGI(context_->GetNodeName(), "actCoreNum:%d, blockAxis:%d, blockUnion:%ld, blockFactor:%ld, blockTailFactor:%ld,"
            "ubAxis:%d, baseN:%ld, baseLen:%ld",
            actCoreNum_, blockAxis_, blockUnion_, blockFactor_, blockTailFactor_, ubAxis_, baseN_, baseLen_);
    tilingData_.set_E(E_);
    tilingData_.set_K(K_);
    tilingData_.set_N(N_);
    tilingData_.set_needCoreNum(needCoreNum_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(g_SyncWorkSpaceSize);
}
} // namespace ascendquantv2

static ge::graphStatus TilingPrepare4AscendQuantV2(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<AscendQuantV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    OP_CHECK_IF(
        (compileInfo->vectorCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(
            context->GetNodeName(), "AscendQuantV2 GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%lu.",
            compileInfo->vectorCoreNum, compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "GetCoreNum:%d, ubSize:%lu", compileInfo->vectorCoreNum, compileInfo->ubSize);

    auto npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfo->isAscend910B = ((npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3003) || (npuArch == NpuArch::DAV_3113));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AscendQuantV2(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    bool isRegBase = IsRegbaseSocVersion(context);
    if (isRegBase) {
        ascendquantv2regbase::AscendQuantV2Regbase tiling(context);
        ge::graphStatus status = tiling.DoAscendQuantV2Tiling();
        return status;
    }
    auto xInputDesc = context->GetInputDesc(0);
    auto xFormat = xInputDesc->GetFormat().GetStorageFormat();
    ge::graphStatus status;
    ascendquantv2::AscendQuantV2 tiling(context);
    if(xFormat == optiling::ascendquantv2::nz_format){
        status = tiling.DoAscendQuantV2NZTiling();
    }
    else{
        status = tiling.DoAscendQuantV2Tiling();
    }
    return status;
}

IMPL_OP_OPTILING(AscendQuantV2)
    .Tiling(Tiling4AscendQuantV2)
    .TilingParse<AscendQuantV2CompileInfo>(TilingPrepare4AscendQuantV2);
} // namespace optiling