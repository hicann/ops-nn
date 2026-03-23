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
 * \file rms_norm_dynamic_mx_quant_base_tiling.cpp
 * \brief RmsNormDynamicMxQuant base tiling implementation
 */

#include "rms_norm_dynamic_mx_quant_tiling_arch35.h"
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
    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    vecRegSize_ = Ops::Base::GetVRegSize(context_);
    vlFp32_ = vecRegSize_ / sizeof(float);

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
    OP_CHECK_IF(X_SUPPORT_DTYPE_SET.count(xDtype_) == 0, 
        OP_LOGE(context_->GetNodeName(),
            "Input x's data type[%s] only support FLOAT16 and BFLOAT16 currently, please check.",
            Ops::Base::ToString(static_cast<ge::DataType>(xDtype_)).c_str()),
        return ge::GRAPH_FAILED);

    auto inputGammaPtr = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputGammaPtr);
    gammaDtype_ = inputGammaPtr->GetDataType();
    OP_CHECK_IF(gammaDtype_ != xDtype_ && gammaDtype_ != ge::DT_FLOAT,
        OP_LOGE(
            context_->GetNodeName(), "Input gamma's data type[%s] should be same as x[%s] or FLOAT, please check.",
            Ops::Base::ToString(static_cast<ge::DataType>(gammaDtype_)).c_str(),
            Ops::Base::ToString(static_cast<ge::DataType>(xDtype_)).c_str()),
        return ge::GRAPH_FAILED);

    gammaDtypeSize_ = gammaDtype_ == ge::DT_FLOAT ?  sizeof(float) : sizeof(float) / NUM_TWO;

    auto inputBetaPtr = context_->GetOptionalInputDesc(2);
    if (inputBetaPtr != nullptr) {
        auto betaDtype = inputBetaPtr->GetDataType();
        OP_CHECK_IF(betaDtype != gammaDtype_,
            OP_LOGE(
                context_->GetNodeName(), "Input beta's data type[%s] should be same as gamma[%s], please check.",
                Ops::Base::ToString(static_cast<ge::DataType>(betaDtype)).c_str(),
                Ops::Base::ToString(static_cast<ge::DataType>(gammaDtype_)).c_str()),
            return ge::GRAPH_FAILED);
    }

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    yDtype_ = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype_) == 0,
        OP_LOGE(context_->GetNodeName(),
            "Output y's data type[%s] only support FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 currently, please "
            "check.",
            Ops::Base::ToString(static_cast<ge::DataType>(yDtype_)).c_str()),
        return ge::GRAPH_FAILED);

    int checkDstType = static_cast<int>(dstType_);
    OP_CHECK_IF(
        (yDtype_ == ge::DT_FLOAT4_E2M1 && checkDstType != 40) ||(yDtype_ == ge::DT_FLOAT4_E1M2 && checkDstType != 41) ||
        (yDtype_ == ge::DT_FLOAT8_E4M3FN && checkDstType != 36) ||(yDtype_ == ge::DT_FLOAT8_E5M2 && checkDstType != 35),
        OP_LOGE(
            context_->GetNodeName(),
            "y's data type[%s] and dst_type[%d] is not corresponded, y's data type: "
            "FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2 correspond to dst_type: 40/41/36/35, please check.",
            Ops::Base::ToString(static_cast<ge::DataType>(yDtype_)).c_str(), checkDstType),
        return ge::GRAPH_FAILED);

    // 暂不支持fp4的cublass方案
    OP_CHECK_IF(
        scaleAlg_ == 1 && Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0,
        OP_LOGE(context_->GetNodeName(), "When y's data type is FLOAT4_E2M1/FLOAT4_E1M2, scale_alg must be set to 0."),
        return ge::GRAPH_FAILED);

    // y输出为fp8时只支持rint
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_FP8_SET.count(yDtype_) != 0 && roundMode_ != static_cast<int64_t>(RoundModeList::MODE_RINT),
        OP_LOGE(
            context_->GetNodeName(),
            "When output y's data type is FLOAT8_E4M3FN/FLOAT8_E5M2, round_mode only support rint, please check."),
        return ge::GRAPH_FAILED);

    auto outputMxScalePtr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputMxScalePtr);
    auto mxscaleDtype = outputMxScalePtr->GetDataType();
    OP_CHECK_IF(
        mxscaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE(
            context_->GetNodeName(), "Output mxscale's data type[%s] only support FLOAT8_E8M0 currently, please check.",
            Ops::Base::ToString(static_cast<ge::DataType>(mxscaleDtype)).c_str()),
        return ge::GRAPH_FAILED);

    if (hasOutputRstd_) {
        auto outputRstdPtr = context_->GetOutputDesc(2);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputRstdPtr);
        auto rstdDtype = outputRstdPtr->GetDataType();
        OP_CHECK_IF(
            rstdDtype != ge::DT_FLOAT,
            OP_LOGE(
                context_->GetNodeName(), "Output rstd's data type[%s] only support FLOAT currently, please check.",
                Ops::Base::ToString(static_cast<ge::DataType>(rstdDtype)).c_str()),
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
    scaleAlg_ = (scaleAlgPtr != nullptr) ? static_cast<int64_t>(*scaleAlgPtr) : QUANT_ALG_DEFAULT;
    OP_CHECK_IF(
        scaleAlg_ != 0 && scaleAlg_ != 1,
        OP_LOGE(context_->GetNodeName(), "The scale_alg[%ld] should be 0 or 1, please check.", scaleAlg_),
        return ge::GRAPH_FAILED);

    const char* roundModePtr = attrs->GetAttrPointer<char>(2);
    if (roundModePtr != nullptr) {
        std::string roundModeStr(roundModePtr);
        RoundModeList roundMode = GetRoundMode(roundModeStr);
        OP_CHECK_IF(
            roundMode == RoundModeList::MODE_UNDEFINED,
            OP_LOGE(
                context_->GetNodeName(), "invalid round_mode:%s; round_mode should be one of {rint, round, floor}",
                roundModePtr),
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
        OP_LOGE(context_->GetNodeName(), "Input x rank[%zu] should be in [1, 7].", xShapeDimNum),
        return ge::GRAPH_FAILED);

    int64_t xLastDim = xShape.GetDim(xShapeDimNum - 1);
    numM_ = 1;
    for (size_t i = 0; i < xShapeDimNum - 1; i++) {
        numM_ *= xShape.GetDim(i);
    }
    numN_ = xLastDim;
    avgFactor_ = float(1.0) / float(numN_);
    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0) {
        OP_CHECK_IF(
            xLastDim % 2 != 0,
            OP_LOGE(
                context_->GetNodeName(),
                "When output y's data type is FLOAT4_E2M1/FLOAT4_E1M2, the last axis should be even, please check."),
            return ge::GRAPH_FAILED);
    }

    auto gammaShapePtr = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaShapePtr);
    auto gammaShape = gammaShapePtr->GetStorageShape();
    OP_CHECK_IF(
        gammaShape.GetDimNum() != 1,
        OP_LOGE(context_->GetNodeName(), "Input gamma rank[%zu] should be 1.", gammaShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    int64_t gammaDim = gammaShape.GetDim(0);
    OP_CHECK_IF(
        gammaDim != xLastDim,
        OP_LOGE(
            context_->GetNodeName(), "Input gamma's last dim[%ld] should be equal to x's last dim[%ld].", gammaDim,
            xLastDim),
        return ge::GRAPH_FAILED);

    auto betaShapePtr = context_->GetOptionalInputShape(2);
    if (betaShapePtr != nullptr) {
        auto betaShape = betaShapePtr->GetStorageShape();
        OP_CHECK_IF(
            betaShape.GetDimNum() != 1,
            OP_LOGE(context_->GetNodeName(), "Input beta rank[%zu] should be 1.", betaShape.GetDimNum()),
            return ge::GRAPH_FAILED);
        int64_t betaDim = betaShape.GetDim(0);
        OP_CHECK_IF(
            betaDim != xLastDim,
            OP_LOGE(
                context_->GetNodeName(), "Input beta's last dim[%ld] should be equal to x's last dim[%ld].", betaDim,
                xLastDim),
            return ge::GRAPH_FAILED);
    }

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE(
            context_->GetNodeName(), "Output y shape[%s] must be same with input x shape[%s].",
            Shape2String(yShape).c_str(), Shape2String(xShape).c_str()),
        return ge::GRAPH_FAILED);

    if (hasOutputRstd_) {
        auto rstdShapePtr = context_->GetOutputShape(2);
        auto rstdShape = rstdShapePtr->GetStorageShape();
        OP_CHECK_IF(
            rstdShape.GetDimNum() != xShapeDimNum && rstdShape.GetDimNum() != xShapeDimNum - 1 &&
                rstdShape.GetDimNum() != 1,
            OP_LOGE(
                context_->GetNodeName(), "Output rstd rank[%zu] should be 1 or %zu, or %zu.", rstdShape.GetDimNum(),
                xShapeDimNum - 1, xShapeDimNum),
            return ge::GRAPH_FAILED);
        if (rstdShape.GetDimNum() == xShapeDimNum - 1 || rstdShape.GetDimNum() == xShapeDimNum) {
            for (size_t i = 0; i < xShapeDimNum - 1; i++) {
                OP_CHECK_IF(
                    rstdShape.GetDim(i) != xShape.GetDim(i),
                    OP_LOGE(
                        context_->GetNodeName(), "Output rstd dim[%zu]=%ld should be equal to x dim[%zu]=%ld.", i,
                        rstdShape.GetDim(i), i, xShape.GetDim(i)),
                    return ge::GRAPH_FAILED);
            }
            if (rstdShape.GetDimNum() == xShapeDimNum) {
                OP_CHECK_IF(
                    rstdShape.GetDim(xShapeDimNum - 1) != 1,
                    OP_LOGE(
                        context_->GetNodeName(), "Output rstd dim[%zu]=%ld should be equal to 1.", xShapeDimNum - 1,
                        rstdShape.GetDim(xShapeDimNum - 1)),
                    return ge::GRAPH_FAILED);
            }
        } else {
            OP_CHECK_IF(
                rstdShape.GetDim(0) != numM_,
                OP_LOGE(
                    context_->GetNodeName(), "Output rstd dim[0]=%ld should be equal to numM=%ld.", rstdShape.GetDim(0),
                    numM_),
                return ge::GRAPH_FAILED);
        }
    }

    auto mxscaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, mxscaleShapePtr);
    auto mxscaleShape = mxscaleShapePtr->GetStorageShape();
    OP_CHECK_IF(
        mxscaleShape.GetDimNum() != xShapeDimNum + 1,
        OP_LOGE(
            context_->GetNodeName(), "Output mxscale rank[%zu] should be equal to x rank[%zu] + 1.",
            mxscaleShape.GetDimNum(), xShapeDimNum),
        return ge::GRAPH_FAILED);

    auto newScaleShape = xShape;
    newScaleShape.SetDim(xShapeDimNum - 1, Ops::Base::CeilDiv(Ops::Base::CeilDiv(xLastDim, MX_BLOCK_SIZE), NUM_TWO));
    newScaleShape.AppendDim(NUM_TWO);

    OP_CHECK_IF(
        newScaleShape != mxscaleShape,
        OP_LOGE(
            context_->GetNodeName(), "The shape of output mxscale %s is incorrect, correct shape is %s, please check.",
            Shape2String(mxscaleShape).c_str(), Shape2String(newScaleShape).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============== Is Optimize Condition ==============
bool RmsNormDynamicMxQuantTilingBase::IsOptimizeCondition()
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

// ============== GetShapeAttrsInfo ==============
ge::graphStatus RmsNormDynamicMxQuantTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        context_ == nullptr, OP_LOGE("RmsNormDynamicMxQuantTilingBase", "context is nullptr."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

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
