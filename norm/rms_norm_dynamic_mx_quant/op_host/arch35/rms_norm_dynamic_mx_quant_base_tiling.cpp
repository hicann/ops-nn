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
 * \file rms_norm_dynamic_mx_quant_base_tiling.cpp
 * \brief RmsNormDynamicMxQuant base tiling implementation
 */

#include "rms_norm_dynamic_mx_quant_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
using namespace platform_ascendc;

namespace optiling {

// ============== RoundMode ==============
RoundModeList RmsNormDynamicMxQuantTilingBase::GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    }
    if (roundMode == "round") {
        return RoundModeList::MODE_ROUND;
    }
    if (roundMode == "floor") {
        return RoundModeList::MODE_FLOOR;
    }
    return RoundModeList::MODE_UNDEFINED;
}

// ============== Platform Info ==============
ge::graphStatus RmsNormDynamicMxQuantTilingBase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfo = context_->GetCompileInfo<RmsNormDynamicMxQuantCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        totalCoreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize_ = static_cast<int64_t>(ubSize);
        workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    }

    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    vlFp32_ = Ops::Base::GetVRegSize(context_) / sizeof(float);

    ubBlockFp32Num_ = ubBlockSize_ / FP32_BYTES;
    ubBlockB16Num_ = ubBlockSize_ / FP16_BYTES;

    OP_CHECK_IF(
        totalCoreNum_ <= 0, OP_LOGE(context_->GetNodeName(), "Failed to get core num."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ubSize_ <= 0, OP_LOGE(context_->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);

    OP_LOGD(context_->GetNodeName(), "GetPlatformInfo: totalCoreNum: %ld, ubSize: %ld", totalCoreNum_, ubSize_);

    return ge::GRAPH_SUCCESS;
}

// ============== Dtype Check ==============
ge::graphStatus RmsNormDynamicMxQuantTilingBase::CheckDtype()
{
    auto inputXPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    xDtype_ = inputXPtr->GetDataType();
    OP_CHECK_IF(
        X_SUPPORT_DTYPE_SET.count(xDtype_) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x", Ops::Base::ToString(static_cast<ge::DataType>(xDtype_)).c_str(),
            "float16 or bfloat16"),
        return ge::GRAPH_FAILED);

    auto inputGammaPtr = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputGammaPtr);
    gammaDtype_ = inputGammaPtr->GetDataType();
    std::string gammaReasonStr = "same as x (" + Ops::Base::ToString(static_cast<ge::DataType>(xDtype_)) + ") or float";
    OP_CHECK_IF(
        gammaDtype_ != xDtype_ && gammaDtype_ != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "gamma", Ops::Base::ToString(static_cast<ge::DataType>(gammaDtype_)).c_str(),
            gammaReasonStr.c_str()),
        return ge::GRAPH_FAILED);

    gammaDtypeSize_ = gammaDtype_ == ge::DT_FLOAT ? FP32_BYTES : FP16_BYTES;

    auto inputBetaPtr = context_->GetOptionalInputDesc(2);
    if (inputBetaPtr != nullptr) {
        auto betaDtype = inputBetaPtr->GetDataType();
        std::string betaReasonStr = "same as gamma (" + Ops::Base::ToString(static_cast<ge::DataType>(gammaDtype_)) + ")";
        OP_CHECK_IF(
            betaDtype != gammaDtype_,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "beta", Ops::Base::ToString(static_cast<ge::DataType>(betaDtype)).c_str(),
                betaReasonStr.c_str()),
            return ge::GRAPH_FAILED);
    }

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    yDtype_ = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype_) == 0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "y", Ops::Base::ToString(static_cast<ge::DataType>(yDtype_)).c_str(),
            "float4_e2m1, float4_e1m2, float8_e4m3fn and float8_e5m2"),
        return ge::GRAPH_FAILED);

    int checkDstType = static_cast<int>(dstType_);
    OP_CHECK_IF(
        (yDtype_ == ge::DT_FLOAT4_E2M1 && checkDstType != 40) ||
            (yDtype_ == ge::DT_FLOAT4_E1M2 && checkDstType != 41) ||
            (yDtype_ == ge::DT_FLOAT8_E4M3FN && checkDstType != 36) ||
            (yDtype_ == ge::DT_FLOAT8_E5M2 && checkDstType != 35),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "dst_type", std::to_string(checkDstType).c_str(),
            "40/41/36/35 for float4_e2m1/float4_e1m2/float8_e4m3fn/float8_e5m2"),
        return ge::GRAPH_FAILED);

    // 暂不支持fp4的cublass方案
    OP_CHECK_IF(
        scaleAlg_ == 1 && Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "scale_alg", std::to_string(scaleAlg_).c_str(), "0"),
        return ge::GRAPH_FAILED);

    // y输出为fp8时只支持rint
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_FP8_SET.count(yDtype_) != 0 && roundMode_ != static_cast<int64_t>(RoundModeList::MODE_RINT),
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "round_mode", std::to_string(roundMode_).c_str(), "rint"),
        return ge::GRAPH_FAILED);

    auto outputMxScalePtr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputMxScalePtr);
    auto mxscaleDtype = outputMxScalePtr->GetDataType();
    OP_CHECK_IF(
        mxscaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "mxscale", Ops::Base::ToString(static_cast<ge::DataType>(mxscaleDtype)).c_str(),
            "float8_e8m0"),
        return ge::GRAPH_FAILED);

    if (hasOutputRstd_) {
        auto outputRstdPtr = context_->GetOutputDesc(2);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputRstdPtr);
        auto rstdDtype = outputRstdPtr->GetDataType();
        OP_CHECK_IF(
            rstdDtype != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(
                context_->GetNodeName(), "rstd", Ops::Base::ToString(static_cast<ge::DataType>(rstdDtype)).c_str(),
                "float"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// ============== Get Attr ==============
ge::graphStatus RmsNormDynamicMxQuantTilingBase::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const float* epsilonPtr = attrs->GetAttrPointer<float>(0);
    epsilon_ = (epsilonPtr != nullptr) ? *epsilonPtr : EPSILON_DEFAULT;

    const int64_t* scaleAlgPtr = attrs->GetAttrPointer<int64_t>(1);
    scaleAlg_ = (scaleAlgPtr != nullptr) ? static_cast<int64_t>(*scaleAlgPtr) : SCALE_ALG_DEFAULT;
    OP_CHECK_IF(
        scaleAlg_ != 0 && scaleAlg_ != 1,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "scale_alg", std::to_string(scaleAlg_).c_str(), "0 or 1"),
        return ge::GRAPH_FAILED);

    const char* roundModePtr = attrs->GetAttrPointer<char>(2);
    if (roundModePtr != nullptr) {
        std::string roundModeStr(roundModePtr);
        RoundModeList roundMode = GetRoundMode(roundModeStr);
        OP_CHECK_IF(
            roundMode == RoundModeList::MODE_UNDEFINED,
            OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "round_mode", roundModePtr, "rint, round or floor"),
            return ge::GRAPH_FAILED);

        roundMode_ = static_cast<int64_t>(roundMode);
    } else {
        roundMode_ = ROUND_MODE_DEFAULT;
    }

    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(3);
    dstType_ = (dstTypePtr != nullptr) ? static_cast<int64_t>(*dstTypePtr) : DST_TYPE_DEFAULT;

    const bool* outputRstdPtr = attrs->GetAttrPointer<bool>(4);
    hasOutputRstd_ = (outputRstdPtr != nullptr) ? static_cast<int64_t>(*outputRstdPtr) : 0;

    auto inputBetaPtr = context_->GetOptionalInputDesc(2);
    hasInputBeta_ = (inputBetaPtr != nullptr) ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

// ============== Shape Check ==============
ge::graphStatus RmsNormDynamicMxQuantTilingBase::CheckShape()
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    int64_t xShapeDimNum = xShape.GetDimNum();

    OP_CHECK_IF(
        xShapeDimNum < 1 || xShapeDimNum > MAX_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "x", std::to_string(xShapeDimNum).c_str(), "1D to 7D"),
        return ge::GRAPH_FAILED);

    int64_t xLastDim = xShape.GetDim(xShapeDimNum - 1);

    numM_ = 1;
    for (int64_t i = 0; i < xShapeDimNum - 1; i++) {
        numM_ *= xShape.GetDim(i);
    }
    numN_ = xLastDim;
    avgFactor_ = float(1.0) / float(numN_);

    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0) {
        OP_CHECK_IF(
            xLastDim % 2 != 0,
            OP_LOGE_FOR_INVALID_SHAPESIZE(
                context_->GetNodeName(), "x", std::to_string(xLastDim).c_str(), "even number when y dtype is fp4"),
            return ge::GRAPH_FAILED);
    }

    auto gammaShapePtr = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaShapePtr);
    auto gammaShape = gammaShapePtr->GetStorageShape();
    OP_CHECK_IF(
        gammaShape.GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "gamma", std::to_string(gammaShape.GetDimNum()).c_str(), "1D"),
        return ge::GRAPH_FAILED);

    int64_t gammaDim = gammaShape.GetDim(0);
    std::string gammaReasonStr = "gamma dim should be " + std::to_string(xLastDim) + ", same as x's last dim";
    OP_CHECK_IF(
        gammaDim != xLastDim,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "gamma", std::to_string(gammaDim).c_str(), gammaReasonStr.c_str()),
        return ge::GRAPH_FAILED);

    auto betaShapePtr = context_->GetOptionalInputShape(2);
    if (betaShapePtr != nullptr) {
        auto betaShape = betaShapePtr->GetStorageShape();
        OP_CHECK_IF(
            betaShape.GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                context_->GetNodeName(), "beta", std::to_string(betaShape.GetDimNum()).c_str(), "1D"),
            return ge::GRAPH_FAILED);
        int64_t betaDim = betaShape.GetDim(0);
        std::string betaReasonStr = "beta dim should be " + std::to_string(xLastDim) + ", same as x's last dim";
        OP_CHECK_IF(
            betaDim != xLastDim,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "beta", std::to_string(betaDim).c_str(), betaReasonStr.c_str()),
            return ge::GRAPH_FAILED);
    }

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    std::string yReasonStr = "y shape should be " + Ops::Base::ToString(xShape) + ", same as x's shape";
    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "y", Ops::Base::ToString(yShape).c_str(), yReasonStr.c_str()),
        return ge::GRAPH_FAILED);

    if (hasOutputRstd_) {
        auto rstdShapePtr = context_->GetOutputShape(2);
        auto rstdShape = rstdShapePtr->GetStorageShape();
        int64_t rstdDimNum = rstdShape.GetDimNum();
        OP_CHECK_IF(
            rstdDimNum != xShapeDimNum,
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                context_->GetNodeName(), "rstd", std::to_string(rstdDimNum).c_str(),
                std::to_string(xShapeDimNum).c_str()),
            return ge::GRAPH_FAILED);

        auto expectedRstdShape = xShape;
        expectedRstdShape.SetDim(xShapeDimNum - 1, 1);
        std::string rstdReasonStr = "rstd shape should be " + Ops::Base::ToString(expectedRstdShape) + ", same dims as x except last dim is 1";
        for (int64_t i = 0; i < xShapeDimNum - 1; i++) {
            OP_CHECK_IF(
                rstdShape.GetDim(i) != xShape.GetDim(i),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "rstd", Ops::Base::ToString(rstdShape).c_str(),
                    rstdReasonStr.c_str()),
                return ge::GRAPH_FAILED);
        }

        OP_CHECK_IF(
            rstdShape.GetDim(xShapeDimNum - 1) != 1,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "rstd", Ops::Base::ToString(rstdShape).c_str(),
                rstdReasonStr.c_str()),
            return ge::GRAPH_FAILED);
    }

    auto mxscaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, mxscaleShapePtr);
    auto mxscaleShape = mxscaleShapePtr->GetStorageShape();
    int64_t mxscaleDimNum = mxscaleShape.GetDimNum();
    OP_CHECK_IF(
        mxscaleDimNum != xShapeDimNum + 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "mxscale", std::to_string(mxscaleDimNum).c_str(),
            std::to_string(xShapeDimNum + 1).c_str()),
        return ge::GRAPH_FAILED);

    auto newScaleShape = xShape;
    newScaleShape.SetDim(xShapeDimNum - 1, Ops::Base::CeilDiv(Ops::Base::CeilDiv(xLastDim, MX_BLOCK_SIZE), CONST_TWO));
    newScaleShape.AppendDim(CONST_TWO);

    std::string mxscaleReasonStr = "mxscale shape should be " + Ops::Base::ToString(newScaleShape);
    OP_CHECK_IF(
        newScaleShape != mxscaleShape,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "mxscale", Ops::Base::ToString(mxscaleShape).c_str(), mxscaleReasonStr.c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============== Is Optimize Condition ==============
bool RmsNormDynamicMxQuantTilingBase::IsOptimizeCondition() const
{
    if (xDtype_ != ge::DT_FLOAT16) {
        return false;
    }
    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) == 0) {
        return false;
    }
    if (scaleAlg_ != 0) {
        return false;
    }
    if (roundMode_ != static_cast<int64_t>(RoundModeList::MODE_RINT) &&
        roundMode_ != static_cast<int64_t>(RoundModeList::MODE_ROUND)) {
        return false;
    }
    return true;
}

// ============== Helper Functions ==============
int64_t RmsNormDynamicMxQuantTilingBase::FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - __builtin_clzl(num);
        return (CONST_ONE << pow);
    }
}

// ============== GetShapeAttrsInfo ==============
ge::graphStatus RmsNormDynamicMxQuantTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        context_ == nullptr, OP_LOGE("RmsNormDynamicMxQuantTilingBase", "context is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE("RmsNormDynamicMxQuantTiling", "GetAttr Failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE("RmsNormDynamicMxQuantTiling", "CheckDtype Failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE("RmsNormDynamicMxQuantTiling", "CheckShape Failed."),
        return ge::GRAPH_FAILED);
    OP_LOGD(
        context_->GetNodeName(), "numM: %ld, numN: %ld, epsilon: %f, dstType: %ld, quantAlg: %ld, roundMode: %ld",
        numM_, numN_, epsilon_, dstType_, scaleAlg_, roundMode_);

    return ge::GRAPH_SUCCESS;
}

// ============== Tiling ==============
ge::graphStatus TilingForRmsNormDynamicMxQuant(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("RmsNormDynamicMxQuant", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "TilingForRmsNormDynamicMxQuant enter");

    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

// ============== TilingPrepare ==============
ge::graphStatus TilingPrepareForRmsNormDynamicMxQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForRmsNormDynamicMxQuant enter.");

    auto compileInfo = context->GetCompiledInfo<RmsNormDynamicMxQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);

    OP_LOGD(
        context->GetNodeName(), "TilingPrepareForRmsNormDynamicMxQuant: coreNum: %ld, ubSize: %ld",
        compileInfo->coreNum, compileInfo->ubSize);

    return ge::GRAPH_SUCCESS;
}

// ============== Register ==============
IMPL_OP_OPTILING(RmsNormDynamicMxQuant)
    .Tiling(TilingForRmsNormDynamicMxQuant)
    .TilingParse<RmsNormDynamicMxQuantCompileInfo>(TilingPrepareForRmsNormDynamicMxQuant);

} // namespace optiling
