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
 * \file quant_matmul_activation_quant_helper.cpp
 * \brief Implementation of QuantMatmulActivationQuantHelper template methods.
 */
#include "quant_matmul_activation_quant_helper.h"

#include <string>
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_mx_basic_api_tiling.h"

namespace optiling {
using namespace QuantMatmulActivationQuantTilingConstant;
using Ops::NN::FormatString;

template <typename BaseT>
const char* QuantMatmulActivationQuantHelper<BaseT>::GetDefaultOpName() const
{
    return "QuantMatmulActivationQuant";
}

template <typename BaseT>
ge::graphStatus QuantMatmulActivationQuantHelper<BaseT>::GetShapeAttrsInfo()
{
    this->tilingDataSize_ = sizeof(QMMAQ::QuantMatmulActivationQuantTilingData);
    return QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::CalcBasicBlock()
{
    QuantBaseBlockCalculator calculator(this->inputParams_, this->compileInfo_, GetBatchCoreCnt());
    if (!calculator.Compute(BaseBlockMode::DEFAULT)) {
        return false;
    }
    const BaseBlockRes& baseBlockRes = calculator.GetOutput();
    this->adaptiveWin_.baseM = baseBlockRes.baseM;
    this->adaptiveWin_.baseN = baseBlockRes.baseN;
    this->adaptiveWin_.baseK = baseBlockRes.baseK;
    this->adaptiveWin_.useTailWinLogic = baseBlockRes.useTailWinLogic;
    return true;
}

template <typename BaseT>
uint64_t QuantMatmulActivationQuantHelper<BaseT>::GetBaseNAlignSize(uint64_t innerAlignSize) const
{
    return this->inputParams_.transB ? MX_BASEN_ALIGN : GetShapeWithDataType(innerAlignSize, this->inputParams_.bDtype);
}

template <typename BaseT>
void QuantMatmulActivationQuantHelper<BaseT>::CalculateCurrentPerf(uint64_t mergeLen, uint64_t nTail,
                                                                   uint64_t& newTailMain, uint64_t& curPerf)
{
    const uint64_t totalWindows = ops::CeilDiv(this->adaptiveWin_.nBlockCnt * this->adaptiveWin_.mBlockCnt,
                                               this->aicoreParams_.aicNum);
    newTailMain = ops::CeilAlign(ops::CeilDiv((mergeLen * this->adaptiveWin_.baseN + nTail), mergeLen + 1UL),
                                 MX_BASEN_ALIGN);
    uint64_t newTailLast = mergeLen * (this->adaptiveWin_.baseN - newTailMain) + nTail;
    uint64_t newMainRound = 0UL;
    uint64_t newTailRound = 0UL;

    if (mergeLen < this->adaptiveWin_.nBlockCnt - 1UL) {
        newMainRound = ops::CeilDiv((this->adaptiveWin_.nBlockCnt - 1UL - mergeLen) * this->adaptiveWin_.mBlockCnt +
                                        (mergeLen + 1UL) * this->adaptiveWin_.mBlockCnt % this->aicoreParams_.aicNum,
                                    this->aicoreParams_.aicNum);
    }
    if (mergeLen > 0UL) {
        newTailRound = std::min(ops::CeilDiv(mergeLen * this->adaptiveWin_.mBlockCnt +
                                                 this->adaptiveWin_.mBlockCnt % this->aicoreParams_.aicNum,
                                             this->aicoreParams_.aicNum),
                                totalWindows - newMainRound);
    }
    curPerf = newMainRound * this->adaptiveWin_.baseN + newTailRound * newTailMain +
              (totalWindows - newMainRound - newTailRound) * newTailLast;
}

template <typename BaseT>
void QuantMatmulActivationQuantHelper<BaseT>::GetOuterNAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain)
{
    uint64_t baseN = this->adaptiveWin_.baseN;
    uint64_t nTail = this->inputParams_.nSize % baseN;
    uint64_t blockCnt = this->adaptiveWin_.nBlockCnt * this->adaptiveWin_.mBlockCnt;
    uint64_t totalWindows = ops::CeilDiv(blockCnt, this->aicoreParams_.aicNum);

    uint64_t mainWindows = ops::CeilDiv((this->adaptiveWin_.nBlockCnt - 1UL) * this->adaptiveWin_.mBlockCnt +
                                            this->adaptiveWin_.mBlockCnt % this->aicoreParams_.aicNum,
                                        this->aicoreParams_.aicNum);

    if (blockCnt <= this->aicoreParams_.aicNum ||
        (this->adaptiveWin_.mBlockCnt % this->aicoreParams_.aicNum == 0UL &&
         (this->adaptiveWin_.nBlockCnt % qmmv3_tiling_const::WINDOW_LEN == 0UL ||
          qmmv3_tiling_const::WINDOW_LEN % this->adaptiveWin_.nBlockCnt == 0UL))) {
        mainWindows = totalWindows;
    }
    uint64_t tailWindows = totalWindows - mainWindows;
    uint64_t perfRes = mainWindows * baseN + tailWindows * nTail;

    uint64_t baseTailCntMax = std::min((baseN - nTail) / MX_BASEN_ALIGN, this->adaptiveWin_.nBlockCnt);
    for (uint64_t mergeLen = 1UL; mergeLen < baseTailCntMax; ++mergeLen) {
        uint64_t newTailMain = 0UL;
        uint64_t curPerf = 0UL;
        CalculateCurrentPerf(mergeLen, nTail, newTailMain, curPerf);
        if (curPerf < perfRes) {
            perfRes = curPerf;
            tailMain = newTailMain;
            baseTailSplitCnt = mergeLen + 1UL;
        }
    }
}

template <typename BaseT>
void QuantMatmulActivationQuantHelper<BaseT>::CalcTailBasicBlockAfullLoad()
{
    this->adaptiveWin_.mTailTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t nTileAlignValid = 1UL;
    if (this->adaptiveWin_.tailWinBlockCnt != 0UL) {
        while (this->CalUsedCoreNum(this->adaptiveWin_.mTailTile, (nTile + 1UL)) <= this->aicoreParams_.aicNum &&
               ops::CeilDiv(this->adaptiveWin_.baseN, nTile + 1UL) >= MX_BASEN_ALIGN) {
            nTile += 1UL;
            if (this->IsInValidWeighNzTailSplit(nTile, true)) {
                continue;
            }
            if (IsAligned32(ops::CeilDiv(this->adaptiveWin_.baseN, nTile))) {
                nTileAlignValid = nTile;
            }
        }
    }
    this->adaptiveWin_.nTailTile = nTileAlignValid;
}

template <typename BaseT>
uint64_t QuantMatmulActivationQuantHelper<BaseT>::GetTailSplitState(bool isPreSplitM, bool isPreSplit, uint64_t split,
                                                                    uint64_t splitSize) const
{
    auto baseState = BaseT::GetTailSplitState(isPreSplitM, isPreSplit, split, splitSize);
    if (baseState == 0UL) {
        return 0UL;
    }
    uint64_t nTile;
    if (isPreSplitM) {
        nTile = isPreSplit ? 1UL : split;
    } else {
        nTile = isPreSplit ? split : 1UL;
    }
    if (!IsAligned32(ops::CeilDiv(this->adaptiveWin_.baseN, nTile))) {
        return 0UL;
    }
    return baseState;
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::IsAligned32(uint64_t value) const
{
    return value != 0 && value % 32 == 0;
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::InitMatmulSize(const gert::Shape& x1Shape, const gert::Shape& x2Shape)
{
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    // x1维度数量大于等于2
    if (x1ShapeLen < X1_MINIMUM_DIMENSION_LENGTH) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(this->inputParams_.opName, "x1",
                                                  FormatString("%zuD", x1ShapeLen).c_str(),
                                                  "the shape dim of x1 must be greater than or equal to 2");
        return false;
    }

    // x2维度数量大于等于2
    if (x2ShapeLen < X2_MINIMUM_DIMENSION_LENGTH) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(this->inputParams_.opName, "x2",
                                                  FormatString("%zuD", x2ShapeLen).c_str(),
                                                  "the shape dim of x2 must be greater than or equal to 2");
        return false;
    }

    // 根据x1和x2的originalShape解析M/K/N
    // 维度提取，x1 shape: [x1Outer, x1Inner]
    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);

    // M/K/N赋值
    this->inputParams_.mSize = static_cast<uint64_t>(this->inputParams_.transA ? x1Inner : x1Outer);
    this->inputParams_.kSize = static_cast<uint64_t>(this->inputParams_.transA ? x1Outer : x1Inner);
    this->inputParams_.nSize = static_cast<uint64_t>(this->inputParams_.transB ? x2Outer : x2Inner);

    OP_LOGD(this->inputParams_.opName, "mSize: %lu, kSize: %lu, nSize: %lu", this->inputParams_.mSize,
            this->inputParams_.kSize, this->inputParams_.nSize);

    return true;
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::ValidateQuantParams(const gert::Shape& x1Shape,
                                                                  const gert::Shape& x1ScaleShape,
                                                                  const gert::Shape& scaleShape)
{
    if (this->inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0) {
        return CheckParamsForMxQuant(x1Shape, x1ScaleShape, scaleShape);
    } else {
        return false;
    }
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::AnalyzeAttrs()
{
    // 属性分析
    auto attrs = this->context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "attrs", "null", "attrs can not be null"),
        return false);
    OP_CHECK_IF(attrs->GetAttrNum() < ATTR_INDEX_NUMBERS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    this->inputParams_.opName, "attrs num", std::to_string(attrs->GetAttrNum()).c_str(),
                    FormatString("the num of attrs must be greater than or equal to %u", ATTR_INDEX_NUMBERS).c_str()),
                return false);
    const bool* transposeXPtr = attrs->template GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X1);
    OP_CHECK_IF(transposeXPtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "transposeX1", "null",
                                                      "transposeX1 can not be null"),
                return false);
    const bool* transposeWeightPtr = attrs->template GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X2);
    OP_CHECK_IF(transposeWeightPtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "transposeX2", "null",
                                                      "transposeX2 can not be null"),
                return false);
    const int64_t* groupSizePtr = attrs->template GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_SIZE);
    OP_CHECK_IF(groupSizePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "groupSize", "null",
                                                      "groupSize can not be null"),
                return false);
    OP_CHECK_IF(
        *groupSizePtr < 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "groupSize",
                                              std::to_string(*groupSizePtr).c_str(), "groupSize can not be negative"),
        return false);
    this->inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    this->inputParams_.transA = *transposeXPtr;
    this->inputParams_.transB = *transposeWeightPtr;

    // mx场景，[groupSizeM, groupSizeN, groupSizeK]的取值组合仅分别支持[1, 1, 32]
    if (this->inputParams_.groupSize != 0ULL) {
        this->inputParams_.groupSizeK = this->inputParams_.groupSize & GROUP_MKN_BIT_SIZE;
        this->inputParams_.groupSizeN = (this->inputParams_.groupSize >> 16U) & GROUP_MKN_BIT_SIZE;
        this->inputParams_.groupSizeM = (this->inputParams_.groupSize >> 32U) & GROUP_MKN_BIT_SIZE;
        OP_CHECK_IF(this->inputParams_.groupSizeM != 1UL || this->inputParams_.groupSizeN != 1UL ||
                        this->inputParams_.groupSizeK != 32UL,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "groupSize",
                                                          std::to_string(*groupSizePtr).c_str(),
                                                          "mx quant only supports groupSize [M=1, N=1, K=32]"),
                    return false);
    }

    const char* activationTypePtr = attrs->template GetAttrPointer<char>(ATTR_INDEX_ACTIVATION_TYPE);
    OP_CHECK_IF(activationTypePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "activationType", "null",
                                                      "activationType can not be null"),
                return false);
    std::string activationType(activationTypePtr);
    OP_CHECK_IF(!(activationType == "gelu_tanh" || activationType == "gelu_erf"),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "activationType", activationType,
                                                      "activationType optional values are gelu_tanh/gelu_erf"),
                return false);
    if (activationType == "gelu_erf") {
        activationType_ = QMMAQ::GeluAlg::ERF;
    } else {
        activationType_ = QMMAQ::GeluAlg::TANH;
    }

    const char* quantModePtr = attrs->template GetAttrPointer<char>(ATTR_INDEX_QUANT_MODE);
    OP_CHECK_IF(quantModePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "quantMode", "null",
                                                      "quantMode can not be null"),
                return false);
    std::string quantMode(quantModePtr);
    OP_CHECK_IF(quantMode != "mx",
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "quantMode", quantMode,
                                                      "quantMode must be mx"),
                return false);

    const char* roundModePtr = attrs->template GetAttrPointer<char>(ATTR_INDEX_ROUND_MODE);
    OP_CHECK_IF(roundModePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "roundMode", "null",
                                                      "roundMode can not be null"),
                return false);
    std::string roundMode(roundModePtr);
    OP_CHECK_IF(!(roundMode == "rint" || roundMode == "floor" || roundMode == "round"),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "roundMode", roundMode,
                                                      "roundMode optional values are rint/floor/round, it's enabled "
                                                      "when dynamic mx quant, fp8 only support rint, fp4 support all"),
                return false);
    if (roundMode == "round") {
        roundMode_ = QMMAQ::MX_QUANT_ROUND_MODE::ROUND;
    } else if (roundMode == "floor") {
        roundMode_ = QMMAQ::MX_QUANT_ROUND_MODE::FLOOR;
    } else {
        roundMode_ = QMMAQ::MX_QUANT_ROUND_MODE::RINT;
    }

    const int64_t* scaleAlg = attrs->template GetAttrPointer<int64_t>(ATTR_INDEX_SCALE_ALG);
    OP_CHECK_IF(scaleAlg == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "scaleAlg", "null",
                                                      "scaleAlg can not be null"),
                return false);
    OP_CHECK_IF(!(static_cast<QMMAQ::QuantAlg>(*scaleAlg) == QMMAQ::QuantAlg::OCP ||
                  static_cast<QMMAQ::QuantAlg>(*scaleAlg) == QMMAQ::QuantAlg::BLAS ||
                  static_cast<QMMAQ::QuantAlg>(*scaleAlg) == QMMAQ::QuantAlg::DYN_DTYPE_RANGE),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "scaleAlg", std::to_string(*scaleAlg),
                                                      "scaleAlg optional values are 0/1/2"),
                return false);
    scaleAlg_ = static_cast<QMMAQ::QuantAlg>(*scaleAlg);

    const float* dstTypeMax = attrs->template GetAttrPointer<float>(ATTR_INDEX_DST_TYPE_MAX);
    OP_CHECK_IF(dstTypeMax == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "dstTypeMax", "null",
                                                      "dstTypeMax can not be null"),
                return false);
    dstTypeMax_ = *dstTypeMax;
    return true;
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::AnalyzeDtype()
{
    auto xDesc = this->context_->GetInputDesc(X1_INDEX);
    OP_CHECK_IF(xDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "x1", "null", "x1 can not be null"),
                return false);
    this->inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = this->context_->GetInputDesc(X2_INDEX);
    OP_CHECK_IF(wDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "x2", "null", "x2 can not be null"),
                return false);
    this->inputParams_.bDtype = wDesc->GetDataType();

    auto scaleDesc = this->context_->GetOptionalInputDesc(X2_SCALE_INDEX);
    this->inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : this->inputParams_.scaleDtype;

    auto pertokenScaleDesc = this->context_->GetOptionalInputDesc(X1_SCALE_INDEX);
    this->inputParams_.perTokenScaleDtype = pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() :
                                                                           this->inputParams_.perTokenScaleDtype;

    // 输出y
    auto outDesc = this->context_->GetOutputDesc(Y_OUTPUT_INDEX);
    OP_CHECK_IF(
        outDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "output", "null", "output can not be null"),
        return false);

    this->inputParams_.cDtype = ge::DT_FLOAT;

    // 输出yScale
    auto outScaleDesc = this->context_->GetOutputDesc(Y_SCALE_OUTPUT_INDEX);
    OP_CHECK_IF(outScaleDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "outputScale", "null",
                                                      "outputScale can not be null"),
                return false);
    return CheckDtype();
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::IsFp8Dtype(const ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2;
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::IsFp4Dtype(const ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT4_E2M1;
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::IsMxQuant() const
{
    return ((IsFp8Dtype(this->inputParams_.aDtype) && IsFp8Dtype(this->inputParams_.bDtype)) ||
            (IsFp4Dtype(this->inputParams_.aDtype) && IsFp4Dtype(this->inputParams_.bDtype))) &&
           this->inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 &&
           this->inputParams_.perTokenScaleDtype == ge::DT_FLOAT8_E8M0 && this->inputParams_.isMxPerGroup;
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::CheckDtype() const
{
    auto outDesc = this->context_->GetOutputDesc(Y_OUTPUT_INDEX);
    auto outScaleDesc = this->context_->GetOutputDesc(Y_SCALE_OUTPUT_INDEX);
    auto yDtype = outDesc->GetDataType();
    auto yScaleDtype = outScaleDesc->GetDataType();

    bool isFp8 = IsFp8Dtype(this->inputParams_.aDtype) && IsFp8Dtype(this->inputParams_.bDtype);
    bool isFp4 = IsFp4Dtype(this->inputParams_.aDtype) && IsFp4Dtype(this->inputParams_.bDtype);

    // 校验y的Dtype
    if (isFp8) {
        OP_CHECK_IF(yDtype != ge::DT_FLOAT8_E4M3FN && yDtype != ge::DT_FLOAT8_E5M2,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(this->inputParams_.opName, "y",
                                                          ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                                                          "the dtype of output y must be FLOAT8_E4M3FN/FLOAT8_E5M2 "
                                                          "when x1 and x2 are FLOAT8"),
                    return false);
    } else if (isFp4) {
        OP_CHECK_IF(yDtype != ge::DT_FLOAT4_E2M1,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(this->inputParams_.opName, "y",
                                                          ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                                                          "the dtype of output y must be FLOAT4_E2M1 "
                                                          "when x1 and x2 are FLOAT4_E2M1"),
                    return false);
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            this->inputParams_.opName, "x1, x2",
            FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(this->inputParams_.aDtype).c_str(),
                         ge::TypeUtils::DataTypeToSerialString(this->inputParams_.bDtype).c_str())
                .c_str(),
            "the dtypes of x1 and x2 must both be FLOAT8_E4M3FN/FLOAT8_E5M2 or both be FLOAT4_E2M1");
        return false;
    }
    // 校验yScale的Dtype
    OP_CHECK_IF(yScaleDtype != ge::DT_FLOAT8_E8M0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(this->inputParams_.opName, "yScale",
                                                      ge::TypeUtils::DataTypeToSerialString(yScaleDtype).c_str(),
                                                      "the dtype of output yScale must be FLOAT8_E8M0"),
                return false);
    OP_CHECK_IF(
        this->inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 ||
            this->inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            this->inputParams_.opName, "x1Scale, x2Scale",
            FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(this->inputParams_.perTokenScaleDtype).c_str(),
                         ge::TypeUtils::DataTypeToSerialString(this->inputParams_.scaleDtype).c_str())
                .c_str(),
            "when the quantization mode is mx, the dtype of x1Scale and x2Scale must be FLOAT8_E8M0"),
        return false);

    auto attrs = this->context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "attrs", "null", "attrs can not be null"),
        return false);
    const char* roundModePtr = attrs->template GetAttrPointer<char>(ATTR_INDEX_ROUND_MODE);
    OP_CHECK_IF(isFp8 && roundMode_ != QMMAQ::MX_QUANT_ROUND_MODE::RINT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "roundMode",
                                                      roundModePtr != nullptr ? roundModePtr : "null",
                                                      "roundMode must be rint when dtype is fp8 in mx quant"),
                return false);
    OP_CHECK_IF(scaleAlg_ == QMMAQ::QuantAlg::BLAS && !isFp8,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "scaleAlg", "1",
                                                      "The y dtype must be fp8 in mx quant when scaleAlg is 1"),
                return false);
    return true;
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::CheckShapeValid(const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    auto x1ShapeLength = x1Shape.GetDimNum();
    auto x2ShapeLength = x2Shape.GetDimNum();
    OP_CHECK_IF(x1ShapeLength < X1_MINIMUM_DIMENSION_LENGTH || x1ShapeLength > X1_MAXIMUM_DIMENSION_LENGTH ||
                    x2ShapeLength < X2_MINIMUM_DIMENSION_LENGTH || x2ShapeLength > X2_MAXIMUM_DIMENSION_LENGTH,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    this->inputParams_.opName, "x1", FormatString("%zuD, %zuD", x1ShapeLength, x2ShapeLength).c_str(),
                    "the shape dim of x1 must be in the range of 2 to 6, and the shape dim of x2 must be in the range "
                    "of 4 to 8"),
                return false);

    // K 维取最后两维中的对应位置，以跳过 batch 前缀（与 InitMatmulSize 保持一致）
    auto x2KDimValue = static_cast<uint64_t>(this->inputParams_.transB ?
                                                 x2Shape.GetDim(x2ShapeLength - LAST_FIRST_DIM_INDEX) :
                                                 x2Shape.GetDim(x2ShapeLength - LAST_SECOND_DIM_INDEX));
    auto x1KDimValue = static_cast<uint64_t>(this->inputParams_.transA ?
                                                 x1Shape.GetDim(x1ShapeLength - LAST_SECOND_DIM_INDEX) :
                                                 x1Shape.GetDim(x1ShapeLength - LAST_FIRST_DIM_INDEX));
    OP_CHECK_IF(x1KDimValue != x2KDimValue,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(this->inputParams_.opName, "x1K, x2K",
                                                       FormatString("%lu, %lu", x1KDimValue, x2KDimValue).c_str(),
                                                       "the K dimension of x1 must be equal to the K dimension of x2"),
                return false);
    return true;
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::CheckParamsForMxQuant(const gert::Shape& x1Shape,
                                                                    const gert::Shape& x1ScaleShape,
                                                                    const gert::Shape& x2ScaleShape) const
{
    // scale 由固定 3 维（M/K/2 或 K/N/2）加上与 x1 一致的 batch 维度组成
    auto x1DimNum = x1Shape.GetDimNum();
    size_t x1BatchDimNum = (x1DimNum > LAST_SECOND_DIM_INDEX) ? (x1DimNum - LAST_SECOND_DIM_INDEX) : 0;
    size_t expectedScaleDimNum = static_cast<size_t>(MX_X1_SCALE_DIM) + x1BatchDimNum;

    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != expectedScaleDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    this->inputParams_.opName, "x1Scale", FormatString("%zuD", x1ScaleDimNum).c_str(),
                    FormatString("when the quant mode is mx, the shape dim of x1Scale must be %zu "
                                 "(batch dim of x1 %zu + fixed dim %u)",
                                 expectedScaleDimNum, x1BatchDimNum, MX_X1_SCALE_DIM)
                        .c_str()),
                return false);
    OP_CHECK_IF(x2ScaleDimNum != expectedScaleDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    this->inputParams_.opName, "x2Scale", FormatString("%zuD", x2ScaleDimNum).c_str(),
                    FormatString("when the quant mode is mx, the shape dim of x2Scale must be %zu "
                                 "(batch dim of x1 %zu + fixed dim %u)",
                                 expectedScaleDimNum, x1BatchDimNum, MX_X2_SCALE_DIM)
                        .c_str()),
                return false);

    // batch 维度值逐一校验，与 x1 保持一致
    for (size_t i = 0; i < x1BatchDimNum; ++i) {
        auto x1BatchDim = static_cast<int64_t>(x1Shape.GetDim(i));
        auto x1ScaleBatchDim = static_cast<int64_t>(x1ScaleShape.GetDim(i));
        auto x2ScaleBatchDim = static_cast<int64_t>(x2ScaleShape.GetDim(i));
        OP_CHECK_IF(x1ScaleBatchDim != x1BatchDim,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        this->inputParams_.opName, "dimIndex, x1Batch, x1ScaleBatch",
                        FormatString("%zu, %ld, %ld", i, x1BatchDim, x1ScaleBatchDim).c_str(),
                        "when the quant mode is mx, the batch dimension of x1Scale must be equal to that of x1"),
                    return false);
        OP_CHECK_IF(x2ScaleBatchDim != x1BatchDim,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        this->inputParams_.opName, "dimIndex, x1Batch, x2ScaleBatch",
                        FormatString("%zu, %ld, %ld", i, x1BatchDim, x2ScaleBatchDim).c_str(),
                        "when the quant mode is mx, the batch dimension of x2Scale must be equal to that of x1"),
                    return false);
    }

    // M/K/N 与 lastDim 用倒数位置取，以跳过 batch 前缀
    // x1Scale 固定尾部 3 维: [mOrK, kOrM, 2]; x2Scale 固定尾部 3 维: [kOrN, nOrK, 2]
    auto x1ScaleMDim = static_cast<uint64_t>(this->inputParams_.transA ? x1ScaleShape.GetDim(x1ScaleDimNum - 2) :
                                                                         x1ScaleShape.GetDim(x1ScaleDimNum - 3));
    auto x1ScaleKDim = static_cast<uint64_t>(this->inputParams_.transA ? x1ScaleShape.GetDim(x1ScaleDimNum - 3) :
                                                                         x1ScaleShape.GetDim(x1ScaleDimNum - 2));
    auto x2ScaleNDim = static_cast<uint64_t>(this->inputParams_.transB ? x2ScaleShape.GetDim(x2ScaleDimNum - 3) :
                                                                         x2ScaleShape.GetDim(x2ScaleDimNum - 2));
    auto x2ScaleKDim = static_cast<uint64_t>(this->inputParams_.transB ? x2ScaleShape.GetDim(x2ScaleDimNum - 2) :
                                                                         x2ScaleShape.GetDim(x2ScaleDimNum - 3));
    auto x1ScaleLastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - 1));
    auto x2ScaleLastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - 1));
    auto expectedKDimValue = ops::CeilDiv(this->inputParams_.kSize, MXFP_BASEK_FACTOR);
    OP_CHECK_IF(x2ScaleKDim != expectedKDimValue || x2ScaleNDim != this->inputParams_.nSize ||
                    x2ScaleLastDim != MXFP_MULTI_BASE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    this->inputParams_.opName, "x2Scale",
                    FormatString("[%lu, %lu, %lu]", x2ScaleKDim, x2ScaleNDim, x2ScaleLastDim).c_str(),
                    FormatString("when the quant mode is mx, the shape of x2Scale must be [%lu, %lu, 2]",
                                 expectedKDimValue, this->inputParams_.nSize)
                        .c_str()),
                return false);
    OP_CHECK_IF(x1ScaleMDim != this->inputParams_.mSize || x1ScaleKDim != expectedKDimValue ||
                    x1ScaleLastDim != MXFP_MULTI_BASE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    this->inputParams_.opName, "x1Scale",
                    FormatString("[%lu, %lu, %lu]", x1ScaleKDim, x1ScaleMDim, x1ScaleLastDim).c_str(),
                    FormatString("when the quant mode is mx, the shape of x1Scale must be [%lu, %lu, 2]",
                                 expectedKDimValue, this->inputParams_.mSize)
                        .c_str()),
                return false);
    return true;
}

template <typename BaseT>
bool QuantMatmulActivationQuantHelper<BaseT>::AnalyzeInputs()
{
    // 获取输入张量形状
    const gert::StorageShape* x1StorageShape = this->context_->GetInputShape(X1_INDEX);
    OP_CHECK_IF(x1StorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "x1", "null", "x1 can not be null"),
                return false);
    const gert::StorageShape* x2StorageShape = this->context_->GetInputShape(X2_INDEX);
    OP_CHECK_IF(x2StorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "x2", "null", "x2 can not be null"),
                return false);
    auto x1Shape = x1StorageShape->GetOriginShape();
    auto x2Shape = x2StorageShape->GetOriginShape();

    // x1_scale对应pertoken_scale; x2_scale对应scale
    const gert::StorageShape* scaleStorageShape = this->context_->GetOptionalInputShape(X2_SCALE_INDEX);
    OP_CHECK_IF(
        scaleStorageShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "x2Scale", "null", "x2Scale can not be null"),
        return false);
    auto scaleShape = scaleStorageShape->GetOriginShape();
    const gert::StorageShape* pertokenShape = this->context_->GetOptionalInputShape(X1_SCALE_INDEX);
    OP_CHECK_IF(
        pertokenShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "x1Scale", "null", "x1Scale can not be null"),
        return false);
    auto& x1ScaleShape = pertokenShape->GetStorageShape();
    this->inputParams_.isPertoken = false;

    const gert::StorageShape* biasShapePtr = this->context_->GetOptionalInputShape(BIAS_INDEX);
    this->inputParams_.hasBias = (biasShapePtr != nullptr);
    this->inputParams_.batchBias = this->inputParams_.hasBias ?
                                       QuantBatchMatmulV3TilingBase::GetBatchSize(biasShapePtr->GetStorageShape()) :
                                       1;

    // 调用 InitMatmulSize初始化矩阵shape
    if (!InitMatmulSize(x1Shape, x2Shape)) {
        return false;
    }
    // 分析分组信息
    if (!this->AnalyzeGroupInfo(scaleShape, pertokenShape)) {
        return false;
    }

    // 分析batch信息
    this->inputParams_.batchA = this->GetBatchSize(x1Shape);
    this->inputParams_.batchB = this->GetBatchSize(x2Shape);
    this->AnalyzeBatchInfo(x1StorageShape->GetOriginShape(), x2StorageShape->GetOriginShape());
    OP_TILING_CHECK(!this->InferOutBatchDim(x1Shape, x2Shape),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        this->inputParams_.opName, "x1DimNum, x2DimNum",
                        FormatString("%zu, %zu", x1Shape.GetDimNum(), x2Shape.GetDimNum()).c_str(),
                        "the batch dimensions of x1 and x2 must be broadcastable"),
                    return false);

    // 验证量化参数
    if (!this->SetQuantMode(scaleShape, pertokenShape) || !ValidateQuantParams(x1Shape, x1ScaleShape, scaleShape) ||
        !CheckShapeValid(x1Shape, x2Shape)) {
        return false;
    }
    OP_LOGD(this->inputParams_.opName, "batchA: %lu, batchB: %lu, batchC: %lu, isPerTensor: %s, isPertoken: %s",
            this->inputParams_.batchA, this->inputParams_.batchB, this->inputParams_.batchC,
            this->inputParams_.isPerTensor ? "true" : "false", this->inputParams_.isPertoken ? "true" : "false");
    return true;
}

template <typename BaseT>
void QuantMatmulActivationQuantHelper<BaseT>::SetQuantParams(QMMAQ::QuantMatmulActivationQuantTilingData& tilingData)
{
    tilingData.activationType = activationType_;
    tilingData.scaleAlg = scaleAlg_;
    tilingData.roundMode = roundMode_;
    tilingData.dstTypeMax = dstTypeMax_;
}

template <typename BaseT>
uint64_t QuantMatmulActivationQuantHelper<BaseT>::GetBatchCoreCnt() const
{
    return this->inputParams_.batchC;
}

// 清空并重置Tiling数据结构，准备新的tiling计算
template <typename BaseT>
void QuantMatmulActivationQuantHelper<BaseT>::ResetActivationQuantTilingData(
    QMMAQ::QuantMatmulActivationQuantTilingData& tilingData)
{
    if (!this->isTilingOut_) {
        tilingData = QMMAQ::QuantMatmulActivationQuantTilingData();
        OP_TILING_CHECK(
            memset_s(this->context_->GetRawTilingData()->GetData(), this->context_->GetRawTilingData()->GetCapacity(),
                     0, this->context_->GetRawTilingData()->GetCapacity()) != EOK,
            CUBE_INNER_ERR_REPORT(this->inputParams_.opName, "Fail to clear tiling data"), return);
    }
}

// 复制QBMM TilingData
template <typename BaseT>
void QuantMatmulActivationQuantHelper<BaseT>::CopyV3BasicApiTilingData(
    const DequantBmm::QuantBatchMatmulV3BasicAPITilingData& src, DequantBmm::QuantBatchMatmulV3BasicAPITilingData& dst)
{
    dst = src;
}

// 显式实例化：QuantMatmulActivationQuant 唯一实例化点
template class QuantMatmulActivationQuantHelper<AdaptiveSlidingWindowMXBasicAPITiling>;

} // namespace optiling
