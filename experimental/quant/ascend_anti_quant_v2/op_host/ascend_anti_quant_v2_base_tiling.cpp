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
 * \file ascend_anti_quant_v2_base_tiling.cpp
 * \brief
 */
#include <numeric>
#include "ascend_anti_quant_v2_base_tiling.h"
#include "../op_kernel/arch22/ascend_anti_quant_v2_struct.h"

using namespace ge;
using namespace AscendC;
using namespace AscendAntiQuantV2Op;
using Ops::Base::GetUbBlockSize;

namespace optiling {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_SCALE_INDEX = 1;
constexpr size_t INPUT_OFFSET_INDEX = 2;
constexpr size_t ATTR_DST_TYPE_INDEX = 0;
constexpr size_t ATTR_SQRT_MODE_INDEX = 1;
constexpr size_t SYNC_WORKSPACE_SIZE = 16777216;
constexpr int64_t GM_BLOCK_SIZE_910B = 512;
constexpr int64_t g_BaseLen = 128;
constexpr int64_t g_HalfBase = 2;
constexpr int32_t g_AxisMax = 2;
constexpr size_t g_FirstShapeDim = 0;
constexpr size_t g_SecondShapeDim = 1;
constexpr size_t g_ThirdShapeDim = 2;
constexpr int64_t FP32_BYTE_SIZE = 4;
constexpr int64_t FP16_BYTE_SIZE = 2;
static constexpr int64_t INT4_NUMS_IN_INT8_SPACE = 2;
constexpr int64_t DEFAULT_BASE_LEN = 512;
constexpr int64_t LAST_DIM_NUM = 128;
constexpr int64_t BUFF_NUM = 2;
static const gert::Shape g_vec_1_shape = {1};
static const gert::Shape g_vec_8_shape = {8};

constexpr int64_t rowNdDmaTh = 1024;
constexpr int64_t colTh = 7680;

const gert::Shape& AscendAntiQuantV2Base::EnsureXAndScaleNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

const gert::Shape& AscendAntiQuantV2Base::EnsureYNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_8_shape;
    }
    return inShape;
}

ge::graphStatus AscendAntiQuantV2Base::DoAscendAntiQuantV2Tiling()
{
    OP_CHECK_IF((GetCompileInfo() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "DoAscendAntiQuantV2Tiling GetCompileInfo Failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((GetOpParam() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "DoAscendAntiQuantV2Tiling GetOpParam Failed."),
                return ge::GRAPH_FAILED);

    CalcTiling();
    CalcTilingKey();
    return WriteTilingData();
}

ge::graphStatus AscendAntiQuantV2Base::GetCompileInfo()
{
    auto compileInfo = context_->GetCompileInfo<AscendAntiQuantV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->vectorCoreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF((coreNum_ <= 0 || ubSize_ <= 0),
                OP_LOGE(context_->GetNodeName(), "AscendAntiQuantV2 GetCompileInfo Failed, coreNum:%ld, ubSize:%lu.",
                        coreNum_, ubSize_),
                return ge::GRAPH_FAILED);
    gmBlockSize_ = GM_BLOCK_SIZE_910B;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendAntiQuantV2Base::CheckInputDtype()
{
    auto xInputDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    xDtype_ = xInputDesc->GetDataType();

    auto scaleInputDesc = context_->GetInputDesc(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleInputDesc);
    scaleDtype_ = scaleInputDesc->GetDataType();

    auto yInputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yInputDesc);
    yDtype_ = yInputDesc->GetDataType();

    if (hasOffset_) {
        auto offsetInputDesc = context_->GetInputDesc(INPUT_OFFSET_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, offsetInputDesc);
        auto offsetDtype = offsetInputDesc->GetDataType();

        OP_CHECK_IF(scaleDtype_ != offsetDtype,
                    OP_LOGE(context_->GetNodeName(), "input scale dtype %s is not same as input offset dtype %s",
                            Ops::Base::ToString(scaleDtype_).c_str(), Ops::Base::ToString(offsetDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    if (xDtype_ == ge::DT_HIFLOAT8 || xDtype_ == ge::DT_FLOAT8_E5M2 || xDtype_ == ge::DT_FLOAT8_E4M3FN) {
        OP_CHECK_IF(scaleDtype_ != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "when x is of type %s, the scale must be type FP32",
                            Ops::Base::ToString(xDtype_).c_str()),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendAntiQuantV2Base::CheckInputValid(const gert::Shape& input1, const gert::Shape& input2,
                                                       const std::string& param) const
{
    size_t input1DimNum = input1.GetDimNum();
    size_t input2DimNum = input2.GetDimNum();
    if (static_cast<int32_t>(input1DimNum) != static_cast<int32_t>(input2DimNum) &&
        static_cast<int32_t>(input2DimNum) != 1) {
        OP_LOGE(context_->GetNodeName(), "the dimension(%zu) of %s is invalid, should be same as x(%zu) or 1",
                input2DimNum, param.c_str(), input1DimNum);
        return ge::GRAPH_FAILED;
    }

    if (input1DimNum == static_cast<size_t>(1) && input2.GetDim(0) != 1 && input2.GetDim(0) != input1.GetDim(0)) {
        OP_LOGE(context_->GetNodeName(), "when the dim of x is 1, the shape of %s must be either 1 or the same as x",
                param.c_str());
        return ge::GRAPH_FAILED;
    }

    if (input1DimNum > 1) {
        if (input2DimNum == static_cast<size_t>(1) && input2.GetDim(0) != input1.GetDim(input1DimNum - 1) &&
            input2.GetDim(0) != input1.GetDim(input1DimNum - 2) && input2.GetDim(0) != 1) {
            OP_LOGE(context_->GetNodeName(),
                    "when the dim of %s is 1, its shape must be either 1, or match the -1 or -2 dim of x",
                    param.c_str());
            return ge::GRAPH_FAILED;
        }
        if (input2DimNum > static_cast<size_t>(1)) {
            size_t input2Axis = (input2.GetDim(input2DimNum - 2) != 1 ? input2DimNum - 2 : input2DimNum - 1);
            auto input2Size = input2.GetShapeSize();
            if (input2Size != input2.GetDim(input2Axis)) {
                OP_LOGE(context_->GetNodeName(), "the %s can have at most one dimension that is not 1.", param.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendAntiQuantV2Base::CheckOutputValid(const gert::Shape& input, const gert::Shape& output) const
{
    size_t inputDimNum = input.GetDimNum();
    size_t outputDimNum = output.GetDimNum();
    OP_CHECK_IF(inputDimNum != outputDimNum, OP_LOGE(context_->GetNodeName(), "x dim and y dim must be same."),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputDimNum; i++) {
        OP_CHECK_IF(input.GetDim(i) != output.GetDim(i),
                    OP_LOGE(context_->GetNodeName(), "x shape and y shape must be same."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendAntiQuantV2Base::CheckShapeEqual(const gert::Shape& shape1, const gert::Shape& shape2) const
{
    size_t x1DimNum = shape1.GetDimNum();
    size_t x2DimNum = shape2.GetDimNum();
    OP_CHECK_IF(x1DimNum != x2DimNum, OP_LOGE(context_->GetNodeName(), "scale shape and offset shape must be same."),
                return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < x1DimNum; i++) {
        OP_CHECK_IF(shape1.GetDim(i) != shape2.GetDim(i),
                    OP_LOGE(context_->GetNodeName(), "scale shape and offset shape must be same."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendAntiQuantV2Base::CheckAttrs(const gert::Shape& xShape)
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // get dstType
    const int32_t* dstType = attrs->GetAttrPointer<int32_t>(ATTR_DST_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstType);
    dstType_ = *dstType;
    // get sqrtMode
    const auto* sqrtMode = attrs->GetAttrPointer<bool>(ATTR_SQRT_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sqrtMode);
    sqrtMode_ = static_cast<int16_t>(*sqrtMode);

    int32_t xDimNum = static_cast<int32_t>(xShape.GetDimNum());

    // check dstType and output dtype, must be same
    if (dstType_ != ge::DT_FLOAT16 && dstType_ != ge::DT_BF16) {
        OP_LOGE(context_->GetNodeName(), "dst type:%s is invalid",
                Ops::Base::ToString(static_cast<ge::DataType>(dstType_)).c_str());
        return ge::GRAPH_FAILED;
    }
    if (dstType_ != yDtype_) {
        OP_LOGE(context_->GetNodeName(), "dst type:%s not equal output y dtype:%s",
                Ops::Base::ToString(static_cast<ge::DataType>(dstType_)).c_str(), Ops::Base::ToString(yDtype_).c_str());
        return ge::GRAPH_FAILED;
    }
    if (xDtype_ == ge::DT_INT4 &&
        (xShape.GetDim(xDimNum - 1) % INT4_NUMS_IN_INT8_SPACE)) { // 如何类型是DT_INT4，最后一维长度必须是2的倍数
        OP_LOGE(context_->GetNodeName(), "if xDtype represents DT_INT4, x last dim:%ld must be divisible by 2",
                xShape.GetDim(xDimNum - 1));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void AscendAntiQuantV2Base::MergeInputShape(const gert::Shape& input)
{
    int64_t shape0 = 1;
    int64_t shape1 = static_cast<int64_t>(input.GetDim(input.GetDimNum() - 1));
    int64_t shape2 = 1;
    if (mode_ == static_cast<uint32_t>(TPL_PER_TENSOR)) {
        // pertensor merge [1, x0*x1...*xn, 1]
        for (size_t idx = 0; idx < static_cast<size_t>(input.GetDimNum() - 1); ++idx) {
            shape1 = shape1 * static_cast<int64_t>(input.GetDim(idx));
        }
    } else {
        // perchannel merge [x0*x1*...*x(n-1), xn, 1]
        for (size_t idx = 0; idx < static_cast<size_t>(input.GetDimNum() - 1); ++idx) {
            shape0 = shape0 * input.GetDim(idx);
        }
    }
    // merge shape to 3 dim
    xInputShape_.SetDimNum(3);
    xInputShape_.SetDim(g_FirstShapeDim, shape0);
    xInputShape_.SetDim(g_SecondShapeDim, shape1);
    xInputShape_.SetDim(g_ThirdShapeDim, shape2);
    OP_LOGI(context_->GetNodeName(), "merge shape0:%ld, shape1:%ld, shape2:%ld", shape0, shape1, shape2);
}

ge::graphStatus AscendAntiQuantV2Base::GetOpParam()
{
    auto xInput = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    auto scaleInput = context_->GetInputShape(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleInput);
    auto offsetInput = context_->GetOptionalInputShape(INPUT_OFFSET_INDEX);
    if (offsetInput == nullptr) {
        hasOffset_ = false;
    }
    auto yOut = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOut);

    // 不支持空tensor
    size_t xSizeNum = xInput->GetStorageShape().GetShapeSize();
    if (xSizeNum == 0ULL) {
        OP_LOGE(context_->GetNodeName(), "ascend_anti_quant_v2 does not support empty tensor.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((CheckInputDtype() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "check input dtype failed."),
                return ge::GRAPH_FAILED);

    const gert::Shape& xInputShape = EnsureXAndScaleNotScalar(xInput->GetStorageShape());
    const gert::Shape& scaleInputShape = EnsureXAndScaleNotScalar(scaleInput->GetStorageShape());
    const gert::Shape& yInputShape = EnsureYNotScalar(yOut->GetStorageShape());
    OP_CHECK_IF((CheckAttrs(xInputShape) != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "op attrs is invalid."), return ge::GRAPH_FAILED);

    // check the shape of the y is valid
    OP_CHECK_IF((CheckOutputValid(xInputShape, yInputShape) != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "x and y is invalid."), return ge::GRAPH_FAILED);

    // check the shape of the scale is valid
    OP_CHECK_IF((CheckInputValid(xInputShape, scaleInputShape, "scale") != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "x and scale is invalid."), return ge::GRAPH_FAILED);
    // if offset is not null, check the shape of the offset
    if (hasOffset_) {
        const gert::Shape& offsetInputShape = EnsureXAndScaleNotScalar(offsetInput->GetStorageShape());
        // check scale and offset is same
        OP_CHECK_IF((CheckShapeEqual(scaleInputShape, offsetInputShape) != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "scale and offset is invalid."), return ge::GRAPH_FAILED);
        // check the shape of the offset is valid
        OP_CHECK_IF((CheckInputValid(xInputShape, offsetInputShape, "offset") != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "x and offset is invalid."), return ge::GRAPH_FAILED);
    }
    // check excute mode
    int32_t xDimNum = static_cast<int32_t>(xInputShape.GetDimNum());
    int32_t scaleDimNum = static_cast<int32_t>(scaleInputShape.GetDimNum());
    if (scaleInputShape.GetShapeSize() == 1) {
        mode_ = static_cast<decltype(mode_)>(TPL_PER_TENSOR);
    } else {
        mode_ = static_cast<decltype(mode_)>(TPL_PER_CHANNEL);
    }

    MergeInputShape(xInputShape);
    return ge::GRAPH_SUCCESS;
}

uint32_t AscendAntiQuantV2Base::GetCoreNum(int64_t factor, int64_t coreNum) const
{
    int64_t elePerCore = Ops::Base::CeilDiv(factor, static_cast<int64_t>(coreNum));
    uint32_t actCore = static_cast<uint32_t>(Ops::Base::CeilDiv(factor, elePerCore));
    return actCore;
}

uint32_t AscendAntiQuantV2Base::GetCoreNumWithMinFactor(int64_t factor, int64_t coreNum, int64_t minFactor) const
{
    OP_CHECK_IF(factor <= 0 || coreNum <= 0,
                OP_LOGE(context_->GetNodeName(), "GetCoreNumWithMinFactor invalid param, factor=%ld, coreNum=%ld",
                        factor, coreNum),
                return 0);

    OP_CHECK_IF(minFactor <= 0,
                OP_LOGE(context_->GetNodeName(), "GetCoreNumWithMinFactor invalid minFactor=%ld", minFactor), return 0);

    int64_t elePerCore = std::max(minFactor, Ops::Base::CeilDiv(factor, coreNum));

    int64_t actCore = Ops::Base::CeilDiv(factor, elePerCore);

    return static_cast<uint32_t>(actCore);
}

int64_t AscendAntiQuantV2Base::CalcMaxBaseLen(int64_t ubSize) const
{
    // set n == 1 to calc max base
    int64_t xDtypeSize = ge::GetSizeByDataType(ge::DT_INT8);
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype_);
    int64_t scaleDtypeSize = ge::GetSizeByDataType(scaleDtype_);
    int64_t baseInput = hasOffset_ ? 2 : 1; // hasoffset means 2 input, else means 1 input
    if (mode_ == static_cast<int64_t>(TPL_PER_TENSOR)) {
        baseInput = 0;
    }
    int64_t tmpInput = (scaleDtypeSize == 4) ? 0 : 1;
    int64_t totalBytes = (xDtypeSize + yDtypeSize) * BUFF_NUM + scaleDtypeSize * baseInput + sizeof(float) +
                         sizeof(int16_t) + sizeof(float) * baseInput * tmpInput;
    return totalBytes == 0 ? DEFAULT_BASE_LEN : ubSize / totalBytes;
}

int64_t AscendAntiQuantV2Base::CalcMaxN(int64_t ubSize, int64_t base) const
{
    int64_t xDtypeSize = ge::GetSizeByDataType(ge::DT_INT8);
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype_);
    int64_t scaleDtypeSize = ge::GetSizeByDataType(scaleDtype_);
    int64_t leftXBytes = ubSize - base * scaleDtypeSize * BUFF_NUM;
    if (hasOffset_) {
        int64_t offsetDtypeSize = ge::GetSizeByDataType(scaleDtype_);
        leftXBytes -= base * offsetDtypeSize * BUFF_NUM;
    }
    // 此处按各种模式中需要ub最大容量进行计算。
    // 2 * base * sizeof(float) → 对应 nddma 中 bf16 时的 tmpBuf(当scale 和 offset 为bfloat16时）
    // 3 * base * sizeof(float) → 对应 nddma 中的 fp32Buf（三个 float 数组：floatX、floats_b、floato_b）
    // base * sizeof(int16_t) → 对应 nddma 中的 fp16Buf（一个 half 数组，用于 int8/int4 到 half 的中间转换）。
    leftXBytes -= 2 * base * sizeof(float);
    leftXBytes -= 3 * base * sizeof(float);
    leftXBytes -= base * sizeof(int16_t);
    if (leftXBytes <= 0) {
        return 1;
    }
    int64_t totalNBytes = xDtypeSize + yDtypeSize;
    if (totalNBytes == 0 || base == 0 || BUFF_NUM == 0) {
        return 0;
    }
    return leftXBytes / totalNBytes / base / BUFF_NUM / 2;
}

void AscendAntiQuantV2Base::CalcPerTensorBlockFactor(int64_t size)
{
    // 以一个cache为基本单位，计算block块的宽度
    blockFactor_ = Ops::Base::CeilDiv(size, actCoreNum_);
    int64_t shape = xInputShape_.GetDim(blockAxis_);
    int64_t dtypeSize = ge::GetSizeByDataType(ge::DT_INT8);
    if (dtypeSize == 0) {
        return;
    }
    blockFactor_ = blockFactor_ * gmBlockSize_ / dtypeSize;
    blockTailFactor_ = shape - blockFactor_ * (actCoreNum_ - 1);
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
}

void AscendAntiQuantV2Base::CalcPerChannelBlockFactor(int64_t size)
{
    blockFactor_ = Ops::Base::CeilDiv(size, actCoreNum_);
    if (blockAxis_ == 0) {
        int64_t shape1 = xInputShape_.GetDim(1); // 尾轴大小
        int64_t dtypeSize = ge::GetSizeByDataType(ge::DT_INT8);
        if (dtypeSize == 0) {
            blockFactor_ = 0;
            blockTailFactor_ = 0;
            return;
        }
        // 计算每个cache能存放的int8元素数量
        int64_t numPerCache = 64 / dtypeSize;
        // 计算每个UB能存放的int8元素数量
        int64_t availableUb = static_cast<int64_t>(ubSize_) - reserveUb_;
        int64_t maxBase = CalcMaxBaseLen(availableUb);
        maxBase = Ops::Base::FloorAlign(maxBase, numPerCache);
        // 将每行元素数对齐到numPerCache
        int64_t alignedShape1 = Ops::Base::CeilAlign(shape1, numPerCache);
        // 计算每个核心一次能处理多少行（基于UB容量）
        int64_t rowsPerCore = 0;
        if (alignedShape1 > 0) {
            rowsPerCore = maxBase / alignedShape1;
            if (rowsPerCore == 0)
                rowsPerCore = 1; // 至少处理一行
        } else {
            rowsPerCore = 1;
        }
        // 计算基于UB容量的最优核心数
        int64_t optimalCoreNum = Ops::Base::CeilDiv(size, rowsPerCore);
        // 核心数不能超过系统总核心数
        optimalCoreNum = std::min(optimalCoreNum, coreNum_);
        // 核心数不能为0
        if (optimalCoreNum == 0)
            optimalCoreNum = 1;
        // 更新实际核心数
        actCoreNum_ = optimalCoreNum;
        // 重新计算块因子
        blockFactor_ = Ops::Base::CeilDiv(size, actCoreNum_);
        blockTailFactor_ = size - blockFactor_ * (actCoreNum_ - 1);
    } else {
        // 切分尾轴的情况，以一个cache为基本单位
        int64_t shape = xInputShape_.GetDim(blockAxis_);
        int64_t dtypeSize = ge::GetSizeByDataType(ge::DT_INT8);
        if (dtypeSize != 0) {
            blockFactor_ = blockFactor_ * gmBlockSize_ / dtypeSize;
        } else {
            blockFactor_ = 0;
        }
        blockTailFactor_ = shape - blockFactor_ * (actCoreNum_ - 1);
    }
    blockTailFactor_ = blockTailFactor_ == 0 ? blockFactor_ : blockTailFactor_;
}

void AscendAntiQuantV2Base::CalcPerTensorUBFactor(int64_t numPerCache)
{
    int64_t availableUb = static_cast<int64_t>(ubSize_) - reserveUb_ - 1024; // scale,offset
    int64_t maxBase = CalcMaxBaseLen(availableUb);                           // 一个UB能算的数
    maxBase = Ops::Base::FloorAlign(maxBase, numPerCache);                   // 用cacheLine对齐
    int64_t blockBase = blockFactor_;                                        // block块的宽度
    blockBase = Ops::Base::CeilAlign(blockBase, numPerCache);
    baseN_ = 1;
    baseLen_ = std::min(blockBase, maxBase);
}

void AscendAntiQuantV2Base::CalcPerChannelUBFactor(int64_t numPerCache)
{
    int64_t availableUb = static_cast<int64_t>(ubSize_) - reserveUb_;
    int64_t maxBase = CalcMaxBaseLen(availableUb);         // 一个UB能算的数
    maxBase = Ops::Base::FloorAlign(maxBase, numPerCache); // 用cacheLine对齐
    // block cut axis 0, means all dim 1 is continous, else each core handle blockFactor
    int64_t blockBase = blockAxis_ == 0 ? xInputShape_.GetDim(1) : blockFactor_; // block的宽度，n方向
    blockBase = Ops::Base::CeilAlign(blockBase, numPerCache);                    // 用cacheLine对齐
    // 至少能放下2行时走第一分支
    if (blockBase <= maxBase / 2) {
        // need calc max n with real base
        int64_t maxN = CalcMaxN(availableUb, blockBase); // 一个UB能处理几行
        int64_t blockNSize = blockAxis_ == 0 ? blockFactor_ : xInputShape_.GetDim(0);
        baseN_ = std::min(maxN, blockNSize);                     // UB块的行数
        baseLen_ = Ops::Base::CeilAlign(blockBase, numPerCache); // UB块的宽度
    } else {
        baseN_ = 1;
        baseLen_ = std::min(blockBase, maxBase);
    }
}

void AscendAntiQuantV2Base::CalcTiling()
{
    if (mode_ == static_cast<int64_t>(TPL_PER_TENSOR)) {
        // per tensor模式，所有轴合一
        int64_t shape = xInputShape_.GetDim(1);
        int64_t dtypeSize = ge::GetSizeByDataType(ge::DT_INT8);
        if (gmBlockSize_ == 0 || dtypeSize == 0) {
            return;
        }
        int64_t gmBlockNum = Ops::Base::CeilDiv(shape, gmBlockSize_ / dtypeSize);
        int64_t minFactor = CalcMaxBaseLen(ubSize_ - reserveUb_ - 1024) / (gmBlockSize_ / dtypeSize);
        minFactor = minFactor / 2 * 2;
        int64_t actCoreNum = static_cast<int64_t>(
            GetCoreNumWithMinFactor(static_cast<int64_t>(gmBlockNum), static_cast<int64_t>(coreNum_), minFactor));
        blockAxis_ = 1;
        actCoreNum_ = actCoreNum;
        int64_t size = gmBlockNum;
        CalcPerTensorBlockFactor(size);
        CalcPerTensorUBFactor(gmBlockSize_ / dtypeSize);
    } else if (mode_ == static_cast<int64_t>(TPL_PER_CHANNEL)) {
        int64_t shape0 = xInputShape_.GetDim(0);
        int64_t shape1 = xInputShape_.GetDim(1);
        int64_t dtypeSize = ge::GetSizeByDataType(ge::DT_INT8);
        if (gmBlockSize_ == 0 || dtypeSize == 0) {
            return;
        }

        if (shape1 <= rowNdDmaTh && hasOffset_ == true) {
            blockAxis_ = 0;
            mode_ = static_cast<decltype(mode_)>(TPL_PER_CHANNEL_NDDMA);
            if (shape1 == 0) {
                return;
            }
            baseN_ = 4096 / shape1;
            lcmN_ = std::lcm(shape1, 64ULL) / shape1;

            actCoreNum_ = Ops::Base::CeilDiv(shape0, baseN_);
            actCoreNum_ = std::min(actCoreNum_, coreNum_);

            // 以一个cache为基本单位，计算block块的宽度
            blockFactor_ = Ops::Base::CeilDiv(shape0, actCoreNum_);
            actCoreNum_ = Ops::Base::CeilDiv(shape0, blockFactor_);
            if (actCoreNum_ == 0) {
                actCoreNum_ = 1;
            }
            blockTailFactor_ = shape0 - blockFactor_ * (actCoreNum_ - 1);

            baseN_ = std::min(baseN_, blockFactor_);
            if (baseN_ == 1) {
                mode_ = static_cast<decltype(mode_)>(TPL_PER_CHANNEL);
                blockAxis_ = 0;
                actCoreNum_ = static_cast<int64_t>(GetCoreNum(shape0, coreNum_));
                CalcPerChannelBlockFactor(shape0);
                CalcPerChannelUBFactor(repeatSize_);
            } else {
                baseLen_ = Ops::Base::CeilAlign(shape1, repeatSize_);
                baseLenEx_ = Ops::Base::CeilAlign(baseN_ * shape1, repeatSize_);
            }
        } else {
            blockAxis_ = 0;
            actCoreNum_ = static_cast<int64_t>(GetCoreNum(shape0, coreNum_));
            CalcPerChannelBlockFactor(shape0);
            CalcPerChannelUBFactor(repeatSize_);
        }
    }
}

void AscendAntiQuantV2Base::CalcTilingKey()
{
    uint32_t zeroPointType = hasOffset_ ? static_cast<uint32_t>(TPL_HAS_OFFSET) : static_cast<uint32_t>(TPL_NO_OFFSET);
    uint32_t sqrtModeKey = static_cast<uint32_t>(sqrtMode_);
    tilingKey_ = GET_TPL_TILING_KEY(mode_, zeroPointType, sqrtModeKey);
}

ge::graphStatus AscendAntiQuantV2Base::WriteTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum:%ld, tilingKey:%lu", actCoreNum_, tilingKey_);
    context_->SetBlockDim(actCoreNum_);
    context_->SetTilingKey(tilingKey_);

    OP_LOGD(context_->GetNodeName(), "hasOffset:%d, sqrtMode:%d, dstType:%d", hasOffset_, sqrtMode_, dstType_);

    OP_LOGD(context_->GetNodeName(),
            "actCoreNum:%ld, blockAxis:%d, blockFactor:%ld, blockTailFactor:%ld, baseN:%ld, baseLen:%ld", actCoreNum_,
            blockAxis_, blockFactor_, blockTailFactor_, baseN_, baseLen_);

    int64_t shape0 = xInputShape_.GetDim(g_FirstShapeDim);
    int64_t shape1 = xInputShape_.GetDim(g_SecondShapeDim);
    int64_t shape2 = xInputShape_.GetDim(g_ThirdShapeDim);

    tilingData.numCore = actCoreNum_;
    tilingData.blockAxis = blockAxis_;
    tilingData.blockUnion = blockUnion_;
    tilingData.dim0 = shape0;
    tilingData.dim1 = shape1;
    tilingData.dim2 = shape2;
    tilingData.blockFactor = blockFactor_;
    tilingData.blockTailFactor = blockTailFactor_;
    tilingData.baseN = baseN_;
    tilingData.lcmN = lcmN_;
    tilingData.baseLen = baseLen_;
    tilingData.baseLenEx = baseLenEx_;
    tilingData.hasOffset = hasOffset_;
    tilingData.sqrtMode = sqrtMode_;

    uint64_t tilingDataSize = sizeof(tilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData), tilingDataSize);
    if (ret != EOK) {
        OP_LOGD(context_->GetNodeName(), "memcpy_s failde, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYNC_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForAscendAntiQuantV2(gert::TilingContext* context)
{
    OP_LOGD("AscendAntiQuantV2Tiling", "Enter TilingForAscendAntiQuantV2Tiling");

    OP_CHECK_IF(context == nullptr, OP_LOGE("AscendAntiQuantV2Tiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);

    AscendAntiQuantV2Base AscendAntiQuantV2Tiling(context);
    return AscendAntiQuantV2Tiling.DoAscendAntiQuantV2Tiling();
}

static ge::graphStatus TilingPrepareForAscendAntiQuantV2(gert::TilingParseContext* context)
{
    OP_LOGD("AscendAntiQuantV2Tiling", "Enter TilingPrepareForAscendAntiQuantV2Tiling");

    OP_CHECK_IF(context == nullptr, OP_LOGE("AscendAntiQuantV2Tiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompiledInfo<AscendAntiQuantV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    OP_CHECK_IF(
        (compileInfo->vectorCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "AscendAntiQuantV2 GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%lu.",
                compileInfo->vectorCoreNum, compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "GetCoreNum:%d, ubSize:%lu", compileInfo->vectorCoreNum, compileInfo->ubSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AscendAntiQuantV2)
    .Tiling(TilingForAscendAntiQuantV2)
    .TilingParse<AscendAntiQuantV2CompileInfo>(TilingPrepareForAscendAntiQuantV2);

} // namespace optiling
