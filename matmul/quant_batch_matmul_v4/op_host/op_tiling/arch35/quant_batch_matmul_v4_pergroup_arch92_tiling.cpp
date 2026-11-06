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
 * \file quant_batch_matmul_v4_pergroup_arch92_tiling.cpp
 * \brief
 */

#include "tiling/platform/platform_ascendc.h"
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"

#include "quant_batch_matmul_v4_pergroup_arch92_tiling.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v4_tiling_key.h"
#include "common/inc/op_host/op_const_def_nn.h"

namespace optiling {
using namespace platform_ascendc;
using Ops::NN::FormatString;

constexpr uint64_t BASE_M = 128UL;
constexpr uint64_t BASE_K = 128UL;
constexpr uint64_t BASE_N = 128UL;
constexpr uint64_t NK_ALIGN = 64;
constexpr uint64_t MX_FP8_FP4_HIF4_NK_ALIGN = 8;
constexpr double L2_HBM_BW_RATIO = 1.8;
constexpr uint8_t TILING_KEY_LEFT_SHIFT = 10;
constexpr uint64_t DOUBLE_BUFFER = 2UL;
constexpr uint64_t MAX_BASE_K_SIZE = 1024UL;
constexpr uint64_t FP8FP4_GROUP_SIZE_32 = 32UL;
constexpr uint64_t UB_SCALE_MASK_LEN = 4UL;
constexpr uint64_t STEP_KA = 4;
constexpr uint64_t STEP_KB = 4;
constexpr uint64_t DEPTH_A1 = STEP_KA * 2;
constexpr uint64_t DEPTH_B1 = STEP_KB * 2;
constexpr uint64_t DB_L0C = 1;
static const std::vector<uint64_t> S8S4_GROUP_SIZE_SUPPORT_LIST = {64, 128};

bool QuantBatchMatmulV4PerGroupTiling::IsCapable()
{
    // aDtype 必须是 INT8 或 FLOAT8_E4M3FN 之一
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_INT8 && inputParams_.aDtype != ge::DT_FLOAT8_E4M3FN,
                    OP_LOGI(inputParams_.opName, "Input dtype of a should be int8 or float8_e4m3fn, actual dtype is %s",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
                    return false);

    // 当 aDtype 为 INT8 时，bDtype 必须是 INT4
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype != ge::DT_INT4,
        OP_LOGI(inputParams_.opName, "When aDtype is int8, input dtype of weight should be int4, actual dtype is %s",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
        return false);

    // 当 aDtype 为 FLOAT8_E4M3FN 时，bDtype 必须是 FLOAT4_E2M1
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN && inputParams_.bDtype != ge::DT_FLOAT4_E2M1,
        OP_LOGI(inputParams_.opName,
                "When aDtype is float8_e4m3fn, input dtype of weight should be float4_e2m1, actual dtype is %s",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
        return false);

    OP_TILING_CHECK(!isS8S4_ && !isFp8Fp4_, OP_LOGI(inputParams_.opName, "Unsupported scenario: S8S4 or Fp8Fp4"),
                    return false);

    if (isS8S4_) {
        OP_TILING_CHECK(inputParams_.x1ScaleDtype != ge::DT_FLOAT,
                        OP_LOGI(inputParams_.opName, "Input dtype of x1Scale should be float32, actual dtype is %s",
                                ge::TypeUtils::DataTypeToSerialString(inputParams_.x1ScaleDtype).c_str()),
                        return false);

        OP_TILING_CHECK(
            inputParams_.x2ScaleDtype != ge::DT_UINT64 && inputParams_.x2ScaleDtype != ge::DT_FLOAT,
            OP_LOGI(inputParams_.opName,
                    "In the scenario where the dtype of input A is int8 and the dtype of weight is int4, the dtype of "
                    "x2Scale should be uint64 or float32, but the actual type is %s.",
                    ge::TypeUtils::DataTypeToSerialString(inputParams_.x2ScaleDtype).c_str()),
            return false);

        OP_TILING_CHECK(
            inputParams_.cDtype != ge::DT_FLOAT16 && inputParams_.cDtype != ge::DT_BF16,
            OP_LOGI(inputParams_.opName,
                    "In the scenario where the dtype of input A is int8 and the dtype of weight is int4, the dtype of "
                    "output should be float16 or bfloat16,actual dtype is %s ",
                    ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
            return false);

        OP_TILING_CHECK(!IsGroupSizeSupported(inputParams_.groupSize, S8S4_GROUP_SIZE_SUPPORT_LIST),
                        OP_LOGI(inputParams_.opName,
                                "Group size only support 64 or 128 in the A8W4 scenario, actual group size is [%lu]",
                                inputParams_.groupSize),
                        return false);

        OP_TILING_CHECK(
            inputParams_.x1ScaleShapeOuter != inputParams_.mSize,
            OP_LOGI(inputParams_.opName, "x1Scale outer dim should equal to mSize %zu, but x1Scale outer dim is %zu",
                    inputParams_.mSize, inputParams_.x1ScaleShapeOuter),
            return false);

        OP_TILING_CHECK(
            inputParams_.x1ScaleShapeInner != 1,
            OP_LOGI(inputParams_.opName, "x1Scale inner dim should equal to 1, but x1Scale inner dim is %zu",
                    inputParams_.x1ScaleShapeInner),
            return false);
    }

    if (isFp8Fp4_) {
        OP_TILING_CHECK(
            inputParams_.x2ScaleDtype != ge::DT_BF16,
            OP_LOGI(
                inputParams_.opName,
                "In the scenario where the dtype of input A is float8 and the dtype of weight is float4, the dtype of "
                "x2Scale should be bfloat16, but the actual type is %s.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.x2ScaleDtype).c_str()),
            return false);

        OP_TILING_CHECK(
            inputParams_.cDtype != ge::DT_BF16,
            OP_LOGI(
                inputParams_.opName,
                "In the scenario where the dtype of input A is float8 and the dtype of weight is float4 the dtype of "
                "output should be bfloat16, actual dtype is %s ",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
            return false);

        OP_TILING_CHECK(inputParams_.groupSize != FP8FP4_GROUP_SIZE_32,
                        OP_LOGI(inputParams_.opName,
                                "Group size only support 32 in the Fp8Fp4 scenario, actual group size is [%lu]",
                                inputParams_.groupSize),
                        return false);
    }

    OP_TILING_CHECK(inputParams_.kSize % inputParams_.groupSize != 0,
                    OP_LOGI(inputParams_.opName, "kSize should be divisible by groupSizeK %zu, but kSize is %zu",
                            inputParams_.groupSize, inputParams_.kSize),
                    return false);

    OP_TILING_CHECK(
        inputParams_.transB != false,
        OP_LOGI(inputParams_.opName, "Unsupported transposeX2 is true, only transposeX2 = false is supported."),
        return false;);

    OP_TILING_CHECK(
        inputParams_.x2ScaleShapeOuter * inputParams_.groupSize != inputParams_.kSize,
        OP_LOGI(inputParams_.opName,
                "x2Scale outer dim should equal to kSize %zu / groupSizeK %zu, but x2Scale outer dim is %zu",
                inputParams_.kSize, inputParams_.groupSize, inputParams_.x2ScaleShapeOuter),
        return false);

    OP_TILING_CHECK(
        inputParams_.x2ScaleShapeInner != inputParams_.nSize,
        OP_LOGI(inputParams_.opName, "x2Scale inner dim should equal to nSize %zu, but x2Scale inner dim is %zu",
                inputParams_.nSize, inputParams_.x2ScaleShapeInner),
        return false);

    return true;
}

ge::graphStatus QuantBatchMatmulV4PerGroupTiling::GetPlatformInfo()
{
    auto compileInfoPtr = static_cast<const QuantBatchMatmulV4CompileInfo*>(context_->GetCompileInfo());
    OP_TILING_CHECK(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    compileInfo_ = *compileInfoPtr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PerGroupTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OP_LOGD(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());

    OP_TILING_CHECK(AnalyzeContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "invalid context"),
                    return ge::GRAPH_FAILED);

    inputParams_.aDtype = context_->GetInputDesc(X1_INDEX)->GetDataType();

    OP_TILING_CHECK(!AnalyzeAttrs() || !AnalyzeInputs(), OP_LOGE(inputParams_.opName, "fail to analyze context info"),
                    return ge::GRAPH_FAILED);

    OP_LOGD(inputParams_.opName, "Input params: M K N [%ld, %ld, %ld], transposeA[%s], transposeB[%s]",
            inputParams_.mSize, inputParams_.kSize, inputParams_.nSize, inputParams_.transA ? "true" : "false",
            inputParams_.transB ? "true" : "false");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PerGroupTiling::AnalyzeContext()
{
    auto x1Shape = context_->GetInputShape(X1_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);

    auto x1Desc = context_->GetInputDesc(X1_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    inputParams_.aDtype = x1Desc->GetDataType();

    auto x2Shape = context_->GetInputShape(X2_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);

    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    inputParams_.bDtype = x2Desc->GetDataType();

    isS8S4_ = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT4;
    isFp8Fp4_ = inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN && inputParams_.bDtype == ge::DT_FLOAT4_E2M1;

    auto x1ScaleShape = context_->GetOptionalInputShape(X1_SCALE_INDEX);
    // S8S4数据流x1scale不为空
    if (isS8S4_) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, x1ScaleShape);
    }

    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_INDEX);
    if (isS8S4_) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, x1ScaleDesc);
        inputParams_.x1ScaleDtype = x1ScaleDesc->GetDataType();
    }

    if (isFp8Fp4_) {
        bool isMxQuant = (x1ScaleDesc != nullptr && x1ScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0);
        if (!isMxQuant) {
            OP_TILING_CHECK(x1ScaleDesc != nullptr,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                inputParams_.opName, "x1Scale", "non-null",
                                "when the quant mode is not MxFp8Fp4, the value of x1Scale must be null"),
                            return ge::GRAPH_FAILED);
        } else {
            isFp8Fp4_ = false;
        }
    }

    auto x2ScaleShape = context_->GetOptionalInputShape(X2_SCALE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2ScaleShape);

    auto x2SclaeDesc = context_->GetOptionalInputDesc(X2_SCALE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2SclaeDesc);
    inputParams_.x2ScaleDtype = x2SclaeDesc->GetDataType();
    isVectorAntiquant_ = (inputParams_.x2ScaleDtype == ge::DT_FLOAT);

    auto outputShape = context_->GetOutputShape(Y_OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);

    auto outputDesc = context_->GetOutputDesc(Y_OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    inputParams_.cDtype = outputDesc->GetDataType();

    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(inputParams_.opName, "function context_.GetAttrs() failed"),
                    return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());

    return ge::GRAPH_SUCCESS;
}

bool QuantBatchMatmulV4PerGroupTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();

    const int64_t* groupSizePtr = attrs->GetAttrPointer<int64_t>(GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE(inputParams_.opName, "groupSize can not be null"), return false);

    inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    inputParams_.antiQuantType = QuantType::PER_GROUP;
    if (isFp8Fp4_) {
        inputParams_.antiQuantType = QuantType::T_G;
    }

    auto transposeX1 = attrs->GetAttrPointer<bool>(TRANSPOSE_X1_INDEX);
    OP_TILING_CHECK(transposeX1 == nullptr, OP_LOGE(inputParams_.opName, "transposeX1 can not be null"), return false);
    inputParams_.transA = *transposeX1;
    OP_TILING_CHECK(inputParams_.transA != false,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "transposeX1", "true",
                                                          "the value of transposeX1 must be false"),
                    return false);

    auto transposeX2 = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_INDEX);
    OP_TILING_CHECK(transposeX2 == nullptr, OP_LOGE(inputParams_.opName, "transposeX2 can not be null"), return false);
    inputParams_.transB = *transposeX2;

    if (isFp8Fp4_) {
        OP_TILING_CHECK(inputParams_.transB != false,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "transposeX2", "true",
                            "when the dtype of x1 is FLOAT8_E4M3FN and the dtype of x2 is FLOAT4_E2M1, "
                            "the value of transposeX2 must be false"),
                        return false);
    }

    return true;
}

bool QuantBatchMatmulV4PerGroupTiling::AnalyzeInputs()
{
    auto x1Shape = context_->GetInputShape(X1_INDEX)->GetStorageShape();
    auto x2Shape = context_->GetInputShape(X2_INDEX)->GetStorageShape();

    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    OP_TILING_CHECK(x1ShapeLen != VALID_INPUT_DIM_NUM || x2ShapeLen != VALID_WEIGHT_NZ_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        inputParams_.opName, "x1, x2", FormatString("%zuD, %zuD", x1ShapeLen, x2ShapeLen).c_str(),
                        "the shape dim of x1 must be 2D and the shape dim of x2 must be 4D"),
                    return false);
    auto x1Inner = x1Shape.GetDim(0);
    auto x1Outer = x1Shape.GetDim(1);
    inputParams_.mSize = x1Inner;

    auto x1ScaleDesc = context_->GetOptionalInputDesc(X1_SCALE_INDEX);
    bool isMxFp8MxFp4OrHif4 = inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN &&
                              (inputParams_.bDtype == ge::DT_FLOAT4_E2M1 ||
                               inputParams_.bDtype == Ops::NN::DT_HIFLOAT4) &&
                              x1ScaleDesc != nullptr && x1ScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0;

    if (!inputParams_.transB) {
        auto x2N1 = x2Shape.GetDim(0);
        auto x2K1 = x2Shape.GetDim(1);
        auto x2K0 = x2Shape.GetDim(2);
        auto x2N0 = x2Shape.GetDim(3);
        OP_TILING_CHECK(x1Outer != x2K1 * x2K0,
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(inputParams_.opName, "x1, x2",
                                                               FormatString("%lu, %lu", x1Outer, x2K1 * x2K0).c_str(),
                                                               "the k dims of x1 and x2 must be equal"),
                        return false);
        inputParams_.kSize = x2K1 * x2K0;
        inputParams_.nSize = x2N1 * x2N0;
    } else {
        auto x2K1 = x2Shape.GetDim(0);
        auto x2N1 = x2Shape.GetDim(1);
        auto x2N0 = x2Shape.GetDim(2);
        auto x2K0 = x2Shape.GetDim(3);

        if (isMxFp8MxFp4OrHif4) {
            auto x2OriginShape = context_->GetInputShape(X2_INDEX)->GetOriginShape();
            inputParams_.kSize = x1Outer;
            inputParams_.nSize = x2OriginShape.GetDim(0);
        } else {
            OP_TILING_CHECK(x1Outer != x2K1 * x2K0,
                            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                                inputParams_.opName, "x1, x2", FormatString("%lu, %lu", x1Outer, x2K1 * x2K0).c_str(),
                                "the k dims of x1 and x2 must be equal"),
                            return false);
            inputParams_.kSize = x2K1 * x2K0;
            inputParams_.nSize = x2N1 * x2N0;
        }
    }
    uint64_t nkAlign = isMxFp8MxFp4OrHif4 ? MX_FP8_FP4_HIF4_NK_ALIGN : NK_ALIGN;
    OP_TILING_CHECK(inputParams_.kSize % nkAlign != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "kSize", std::to_string(inputParams_.kSize).c_str(),
                        ("the value of kSize must be aligned to " + std::to_string(nkAlign)).c_str()),
                    return false);

    OP_TILING_CHECK(inputParams_.nSize % nkAlign != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "nSize", std::to_string(inputParams_.nSize).c_str(),
                        ("the value of nSize must be aligned to " + std::to_string(nkAlign)).c_str()),
                    return false);

    auto aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X1_INDEX)->GetStorageFormat()));
    OP_TILING_CHECK(aFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "x1", Ops::Base::ToString(aFormat), "ND"),
                    return false);
    auto bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    inputParams_.bFormat = bFormat;
    OP_TILING_CHECK(bFormat != ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "x2", Ops::Base::ToString(bFormat), "FRACTAL_NZ"),
                    return false);
    return AnalyzeScaleInputs();
}

bool QuantBatchMatmulV4PerGroupTiling::AnalyzeYScaleInput()
{
    auto yScaleDesc = context_->GetOptionalInputDesc(Y_SCALE_INDEX);
    // check yScale dtype
    OP_TILING_CHECK(yScaleDesc == nullptr,
                    OP_LOGE(inputParams_.opName, "yScaleDesc can not be null when the quant mode is Fp8Fp4 per-group"),
                    return false);
    OP_TILING_CHECK(yScaleDesc->GetDataType() != ge::DT_UINT64,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opName, "yScale",
                                              ge::TypeUtils::DataTypeToSerialString(yScaleDesc->GetDataType()).c_str(),
                                              "DT_UINT64"),
                    return false);

    // check yScale shape
    auto yScaleShape = context_->GetOptionalInputShape(Y_SCALE_INDEX);
    OP_TILING_CHECK(yScaleShape == nullptr,
                    OP_LOGE(inputParams_.opName, "yScaleShape can not be null when the quant mode is Fp8Fp4 per-group"),
                    return false);

    OP_TILING_CHECK(yScaleShape->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        inputParams_.opName, "yScale", "0",
                        "when the quant mode is Fp8Fp4 per-group, the shape size of yScale must be greater than 0"),
                    return false);

    std::string yScaleShapeStr = Ops::Base::ToString(yScaleShape->GetStorageShape());
    OP_TILING_CHECK(yScaleShape->GetStorageShape().GetDimNum() > VALID_INPUT_DIM_NUM ||
                        (yScaleShape->GetStorageShape().GetDimNum() == VALID_INPUT_DIM_NUM &&
                         yScaleShape->GetStorageShape().GetDim(0) != 1),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opName, "yScale",
                        FormatString("%s", yScaleShapeStr.substr(1, yScaleShapeStr.size() - 2).c_str()).c_str(),
                        "the shape of yScale must be [1, n] or [n]"),
                    return false);

    return true;
}

bool QuantBatchMatmulV4PerGroupTiling::AnalyzeScaleInputs()
{
    if (isS8S4_) {
        auto x1ScaleShape = context_->GetOptionalInputShape(X1_SCALE_INDEX)->GetStorageShape();
        auto x1ScaleShapeLen = x1ScaleShape.GetDimNum();
        OP_TILING_CHECK(
            x1ScaleShapeLen != VALID_X1_SCALE_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opName, "x1Scale", std::to_string(x1ScaleShapeLen).c_str(), "2D"),
            return false);
        auto x1ScaleShapeInner = x1ScaleShape.GetDim(1);
        auto x1ScaleShapeOuter = x1ScaleShape.GetDim(0);
        inputParams_.x1ScaleShapeInner = x1ScaleShapeInner;
        inputParams_.x1ScaleShapeOuter = x1ScaleShapeOuter;
    } else if (isFp8Fp4_) {
        OP_TILING_CHECK(!AnalyzeYScaleInput(), OP_LOGE(inputParams_.opName, "fail to analyze y scale input"),
                        return false);
    }

    auto x2ScaleShape = context_->GetOptionalInputShape(X2_SCALE_INDEX)->GetStorageShape();
    auto x2ScaleShapeLen = x2ScaleShape.GetDimNum();
    if (isS8S4_ || isFp8Fp4_) {
        OP_TILING_CHECK(
            x2ScaleShapeLen != VALID_X2_SCALE_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opName, "x2Scale", std::to_string(x2ScaleShapeLen).c_str(), "2D"),
            return false);
    }
    auto x2ScaleShapeInner = x2ScaleShape.GetDim(1);
    auto x2ScaleShapeOuter = x2ScaleShape.GetDim(0);
    inputParams_.x2ScaleShapeInner = x2ScaleShapeInner;
    inputParams_.x2ScaleShapeOuter = x2ScaleShapeOuter;

    return true;
}

bool QuantBatchMatmulV4PerGroupTiling::IsGroupSizeSupported(uint64_t groupSize,
                                                            const std::vector<uint64_t>& groupSizeSupportList) const
{
    return (std::find(groupSizeSupportList.begin(), groupSizeSupportList.end(), groupSize) !=
            groupSizeSupportList.end());
}

void SetMatmulTiling(qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams& tilingData,
                     const QuantBatchMatmulPerGroupInfo& inputParams)
{
    tilingData.matmulTiling.M = inputParams.mSize;
    tilingData.matmulTiling.Ka = inputParams.kSize;
    tilingData.matmulTiling.N = inputParams.nSize;
    tilingData.matmulTiling.Kb = inputParams.kSize;

    tilingData.matmulTiling.stepM = 1;
    tilingData.matmulTiling.stepN = 1;
    // stepKa固定为4
    tilingData.matmulTiling.stepKa = STEP_KA;
    // stepKb固定为4
    tilingData.matmulTiling.stepKb = STEP_KB;
    // 8: stepKa * 2
    tilingData.matmulTiling.depthA1 = DEPTH_A1;
    // 8: stepKb * 2
    tilingData.matmulTiling.depthB1 = DEPTH_B1;

    tilingData.matmulTiling.baseM = std::min(BASE_M,
                                             ops::CeilAlign(inputParams.mSize, static_cast<uint64_t>(BLOCK_CUBE)));
    tilingData.matmulTiling.baseN = std::min(BASE_M,
                                             ops::CeilAlign(inputParams.nSize, static_cast<uint64_t>(BLOCK_CUBE)));
    tilingData.matmulTiling.dbL0A = DB_BUFFER;
    tilingData.matmulTiling.dbL0B = DB_BUFFER;
    tilingData.matmulTiling.dbL0C = DB_L0C;

    tilingData.matmulTiling.singleCoreM = BASE_M;
    tilingData.matmulTiling.singleCoreK = tilingData.matmulTiling.baseK;
    tilingData.matmulTiling.singleCoreN = BASE_N;

    tilingData.matmulTiling.isBias = 0;
    tilingData.matmulTiling.shareL1Size = 0;
    tilingData.matmulTiling.shareL0CSize = 0;
}

void SetMatmulTiling(qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams& tilingData,
                     const PollingBasicBlockParam& basicBlockParams)
{
    tilingData.matmulTiling.M = basicBlockParams.mSize;
    tilingData.matmulTiling.Ka = basicBlockParams.kSize;
    tilingData.matmulTiling.N = basicBlockParams.nSize;
    tilingData.matmulTiling.Kb = basicBlockParams.kSize;

    tilingData.matmulTiling.stepM = 1;
    tilingData.matmulTiling.stepN = 1;
    // stepKa固定为4
    tilingData.matmulTiling.stepKa = 4;
    // stepKb固定为4
    tilingData.matmulTiling.stepKb = 4;
    // 8: stepKa * 2
    tilingData.matmulTiling.depthA1 = 8;
    // 8: stepKb * 2
    tilingData.matmulTiling.depthB1 = 8;
    tilingData.matmulTiling.baseM = basicBlockParams.baseM;
    tilingData.matmulTiling.baseN = basicBlockParams.baseN;
    tilingData.matmulTiling.baseK = basicBlockParams.baseK;

    tilingData.matmulTiling.dbL0A = DB_BUFFER;
    tilingData.matmulTiling.dbL0B = DB_BUFFER;
    tilingData.matmulTiling.dbL0C = DB_BUFFER;

    tilingData.matmulTiling.singleCoreM = std::min(basicBlockParams.singleM, basicBlockParams.mSize);
    tilingData.matmulTiling.singleCoreK = basicBlockParams.kSize;
    tilingData.matmulTiling.singleCoreN = std::min(basicBlockParams.singleN, basicBlockParams.nSize);

    tilingData.matmulTiling.isBias = 0;
    tilingData.matmulTiling.shareL1Size = 0;
    tilingData.matmulTiling.shareL0CSize = 0;
    tilingData.matmulTiling.usedCoreNum = basicBlockParams.mDim * basicBlockParams.nDim;
}

uint64_t QuantBatchMatmulV4PerGroupTiling::GetFp8Fp4BaseK() const
{
    uint64_t baseK = inputParams_.groupSize;
    uint64_t maxBaseK = std::min(inputParams_.kSize, MAX_BASE_K_SIZE);
    for (uint64_t tempBaseK = baseK; tempBaseK <= maxBaseK; tempBaseK += inputParams_.groupSize) {
        uint64_t aL1Size = tilingData_->matmulTiling.baseM * tempBaseK * tilingData_->matmulTiling.stepKa *
                           DOUBLE_BUFFER;
        // B在L1上的dtype和A一致，都是FP8
        uint64_t bL1Size = tilingData_->matmulTiling.baseN * tempBaseK * tilingData_->matmulTiling.stepKb *
                           DOUBLE_BUFFER;
        uint64_t yScaleSize = tilingData_->matmulTiling.baseN * sizeof(uint64_t) * DOUBLE_BUFFER;
        if (aL1Size + bL1Size + yScaleSize > compileInfo_.l1Size) {
            break;
        }

        if (tilingData_->matmulTiling.baseM * tempBaseK * DOUBLE_BUFFER > compileInfo_.l0aSize) {
            break;
        }

        if (tilingData_->matmulTiling.baseN * tempBaseK > compileInfo_.l0bSize) {
            break;
        }

        uint64_t weightUbInSize = tilingData_->matmulTiling.baseN * tempBaseK * tilingData_->matmulTiling.stepKb;
        uint64_t weightUbOutSize = tilingData_->matmulTiling.baseN * tempBaseK * tilingData_->matmulTiling.stepKb *
                                   DOUBLE_BUFFER;

        // scale类型为b16
        uint64_t weightScaleUbSize = tilingData_->matmulTiling.baseN * tempBaseK * tilingData_->matmulTiling.stepKb /
                                     inputParams_.groupSize * DOUBLE_BUFFER * sizeof(int16_t);
        uint64_t ubScaleMaskSize = UB_SCALE_MASK_LEN * sizeof(uint64_t);
        if ((weightUbInSize + weightUbOutSize + weightScaleUbSize + ubScaleMaskSize) > compileInfo_.ubSize) {
            break;
        }
        baseK = tempBaseK;
    }
    return baseK;
}

ge::graphStatus QuantBatchMatmulV4PerGroupTiling::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(inputParams_.opName, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);
    if (isVectorAntiquant_) {
        pollingTiling_.SetShape(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupSize);
        pollingTiling_.SetDtypeBits(GetDtypeBits(inputParams_.aDtype), GetDtypeBits(inputParams_.bDtype),
                                    GetDtypeBits(inputParams_.x1ScaleDtype), GetDtypeBits(inputParams_.x2ScaleDtype));
        PlatformInfo platformInfos = {static_cast<uint64_t>(compileInfo_.aicNum),
                                      compileInfo_.ubSize,
                                      compileInfo_.l1Size,
                                      compileInfo_.l0aSize,
                                      compileInfo_.l0bSize,
                                      compileInfo_.l0cSize,
                                      L2_HBM_BW_RATIO};
        pollingTiling_.SetPlatformInfo(platformInfos);
        pollingTiling_.SetLimitParams(BASE_N_ALIGN_SIZE, MAX_BASE_M_BLOCK_SIZE, MAX_BASE_N_BLOCK_SIZE,
                                      MAX_BASE_MN_SIZE);
        OP_TILING_CHECK(pollingTiling_.ComputeTiling() == ge::GRAPH_FAILED,
                        OP_LOGE(inputParams_.opName, "get tiling failed"), return ge::GRAPH_FAILED);
        const PollingBasicBlockParam& tilingRes = pollingTiling_.GetTilingResult();
        tilingData_->cubeBlockDimM = tilingRes.mDim;
        tilingData_->cubeBlockDimN = tilingRes.nDim;
        tilingData_->scaleFactor = tilingRes.scaleFactor;
        tilingData_->enablePreAlloc = tilingRes.enablePreAlloc;

        SetMatmulTiling(*tilingData_, tilingRes);
    } else {
        uint64_t mCount = ops::CeilDiv(inputParams_.mSize, BASE_M);
        uint64_t nCount = ops::CeilDiv(inputParams_.nSize, BASE_N);

        tilingData_->cubeBlockDimM = 1;
        tilingData_->cubeBlockDimN = std::min(mCount * nCount, static_cast<uint64_t>(compileInfo_.aicNum));

        SetMatmulTiling(*tilingData_, inputParams_);
        if (isS8S4_) {
            tilingData_->matmulTiling.baseK = inputParams_.groupSize;
        } else if (isFp8Fp4_) {
            tilingData_->groupSize = inputParams_.groupSize;
            tilingData_->matmulTiling.baseK = GetFp8Fp4BaseK();
        }
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t QuantBatchMatmulV4PerGroupTiling::GetTilingKey() const
{
    uint64_t trans = (static_cast<uint64_t>(inputParams_.transA) << 1) | static_cast<uint64_t>(inputParams_.transB);
    KernelTemplateType templateType = isVectorAntiquant_ ? KernelTemplateType::VECTOR_ANTIQUANT :
                                                           KernelTemplateType::BASIS;

    return GET_TPL_TILING_KEY(trans, static_cast<uint64_t>(inputParams_.antiQuantType),
                              static_cast<uint64_t>(inputParams_.hasAntiQuantOffset),
                              static_cast<uint64_t>(inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ),
                              static_cast<uint64_t>(templateType));
}

ge::graphStatus QuantBatchMatmulV4PerGroupTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspaces is null"),
                    return ge::GRAPH_FAILED);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PerGroupTiling::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingDataManager_ = std::make_unique<qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams>();
        OP_TILING_CHECK(tilingDataManager_ == nullptr, OP_LOGE(inputParams_.opName, "failed to instantiate tilingData"),
                        return ge::GRAPH_FAILED);
        tilingData_ = tilingDataManager_.get();
    }
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(inputParams_.opName, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
                    OP_LOGE(inputParams_.opName, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulV4PerGroupTiling::PostTiling()
{
    OP_LOGD(inputParams_.opName, "final tiling data size: %zu", tilingDataSize_);
    OP_TILING_CHECK(tilingDataSize_ % sizeof(uint64_t) != 0,
                    OP_LOGE(inputParams_.opName, "tiling data size[%zu] is not aligned to 8", tilingDataSize_),
                    return ge::GRAPH_FAILED);
    context_->SetBlockDim(tilingData_->cubeBlockDimM * tilingData_->cubeBlockDimN);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<const void*>(tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
