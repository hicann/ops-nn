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
 * \file quant_batch_matmul_v3_checker_arch_9201.cpp
 * \brief
 */

#include "common/inc/op_host/op_const_def_nn.h"
#include "common/op_host/op_tiling/tiling_type_mm.h"
#include "error_util.h"
#include "log/log.h"
#include "quant_batch_matmul_v3_checker_arch_9201.h"

namespace {
constexpr uint64_t HIF4_GROUP_SIZE = 64UL;
constexpr size_t SCALE_THREE_DIM = 3;
} // namespace

namespace optiling {

const std::vector<ge::DataType>& QuantBatchMatmulV3CheckerArch9201::GetLegalInputDtypes() const
{
    static const std::vector<ge::DataType> legalInputDtypes = {ge::DT_INT8,        ge::DT_FLOAT8_E4M3FN,
                                                               ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8,
                                                               ge::DT_FLOAT4_E2M1, Ops::NN::DT_HIFLOAT4};
    return legalInputDtypes;
}

const std::vector<ge::DataType>& QuantBatchMatmulV3CheckerArch9201::GetLegalOutputDtypes() const
{
    static const std::vector<ge::DataType> legalOutputDtypes = {
        ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_FLOAT, ge::DT_INT32};
    return legalOutputDtypes;
}

const char* QuantBatchMatmulV3CheckerArch9201::GetLegalInputDtypesDesc() const
{
    return "INT8/FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1/HIFLOAT4";
}

const char* QuantBatchMatmulV3CheckerArch9201::GetLegalOutputDtypesDesc() const
{
    return "INT8/FLOAT16/BFLOAT16/FLOAT8_E4M3FN/HIFLOAT8/FLOAT/INT32";
}

bool QuantBatchMatmulV3CheckerArch9201::IsFp4Input() const { return inputParams_.aDtype == ge::DT_FLOAT4_E2M1; }

bool QuantBatchMatmulV3CheckerArch9201::IsHif4Input() const { return inputParams_.aDtype == Ops::NN::DT_HIFLOAT4; }

bool QuantBatchMatmulV3CheckerArch9201::CheckExtraABDtypePairs() const
{
    OP_TILING_CHECK(
        LogicXOR(inputParams_.aDtype == Ops::NN::DT_HIFLOAT4, inputParams_.bDtype == Ops::NN::DT_HIFLOAT4),
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "When one input dtype is HIFLOAT4, then the other input dtype must be HIFLOAT4, actual x1 is %s, x2 is %s.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
        return false);
    if (inputParams_.aDtype == Ops::NN::DT_HIFLOAT4 || inputParams_.bDtype == Ops::NN::DT_HIFLOAT4) {
        OP_TILING_CHECK(
            context_->GetOptionalInputDesc(GetPertokenIdx()) == nullptr ||
                context_->GetOptionalInputShape(GetPertokenIdx()) == nullptr,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "When input is HIFLOAT4, pertokenScale should not be null."),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3CheckerArch9201::CheckHif4ScaleDtypeWithPertoken() const
{
    OP_TILING_CHECK(
        IsHif4Input() && (inputParams_.scaleDtype != Ops::NN::DT_HIFLOAT4_SCALE ||
                          inputParams_.perTokenScaleDtype != Ops::NN::DT_HIFLOAT4_SCALE),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "When input is HIFLOAT4 with pertokenScale, scale and pertokenScale must be "
                              "HIFLOAT4_SCALE, actual scale is %s, pertokenScale is %s.",
                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                              ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
        return false);
    return true;
}

bool QuantBatchMatmulV3CheckerArch9201::CheckExtraOutputDtype() const
{
    const bool isFp4 = IsFp4Input();
    const bool isFp8 = inputParams_.aDtype == ge::DT_HIFLOAT8 || inputParams_.aDtype == ge::DT_FLOAT8_E5M2 ||
                       inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E8M0;
    OP_TILING_CHECK((isFp4 || isFp8) && (inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_HIFLOAT8),
                    CUBE_INNER_ERR_REPORT(
                        inputParams_.opName,
                        "When input dtype is FLOAT8/FLOAT4, output dtype must not be INT8/HIFLOAT8, actual is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
                    return false);
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_HIFLOAT8 &&
                        (inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT8_E5M2 ||
                         inputParams_.cDtype == ge::DT_FLOAT8_E4M3FN),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "When input dtype is HIFLOAT8, output must not be INT8/FLOAT8, actual is %s.",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
                    return false);
    return true;
}

bool QuantBatchMatmulV3CheckerArch9201::CheckExtraBiasConstraint(const gert::Shape& scaleShape,
                                                                 const gert::StorageShape* pertokenShape) const
{
    OP_TILING_CHECK(
        (inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2 ||
         inputParams_.aDtype == ge::DT_HIFLOAT8) &&
            inputParams_.scaleDtype == ge::DT_FLOAT &&
            static_cast<uint64_t>(scaleShape.GetDim(0)) == inputParams_.nSize && inputParams_.nSize != 1UL &&
            pertokenShape != nullptr && inputParams_.perTokenScaleDtype == ge::DT_FLOAT &&
            pertokenShape->GetStorageShape().GetDim(0) == 1L && inputParams_.mSize != 1UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "When input dtype is FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 with per-channel FLOAT scale (dim 0 equals n) "
            "and per-tensor FLOAT pertokenScale (dim 0 equals 1), bias is not supported."),
        return false);
    return true;
}

bool QuantBatchMatmulV3CheckerArch9201::CheckExtraShapeValid(const gert::Shape& scaleShape,
                                                             const gert::StorageShape* pertokenShape,
                                                             const std::vector<int64_t>& dimValueOfMKN) const
{
    return CheckInputValidInHif4PerGroupMode(scaleShape, pertokenShape, dimValueOfMKN);
}

bool QuantBatchMatmulV3CheckerArch9201::Hif4PertokenScaleShapeCheck(const gert::StorageShape* pertokenShape) const
{
    auto& pertoken = pertokenShape->GetStorageShape();
    auto pertokenShapeLen = pertoken.GetDimNum();
    OP_TILING_CHECK(pertokenShapeLen != SCALE_THREE_DIM,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "In hif4 quantification, pertokenScale's dimension must be 3, but it is %zu.",
                                          pertokenShapeLen),
                    return false);
    auto mDimIdx = inputParams_.transA ? 1 : 0;
    auto kDimIdx = inputParams_.transA ? 0 : 1;
    OP_TILING_CHECK(static_cast<uint64_t>(pertoken.GetDim(pertokenShapeLen - 1)) != 1,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "In hif4 quantification, pertokenScale's last dimension \
should be equal to 1, but actual is [%ld].",
                                          pertoken.GetDim(pertokenShapeLen - 1)),
                    return false);
    bool mDimHasOne = (pertoken.GetDim(kDimIdx) == 1 && static_cast<uint64_t>(pertoken.GetDim(mDimIdx)) ==
                                                            ops::CeilDiv(inputParams_.kSize, HIF4_GROUP_SIZE)) ||
                      (pertoken.GetDim(mDimIdx) == 1 && static_cast<uint64_t>(pertoken.GetDim(kDimIdx)) ==
                                                            ops::CeilDiv(inputParams_.kSize, HIF4_GROUP_SIZE));
    bool kDimHasOne = (pertoken.GetDim(kDimIdx) == 1 &&
                       static_cast<uint64_t>(pertoken.GetDim(mDimIdx)) == inputParams_.mSize) ||
                      (pertoken.GetDim(mDimIdx) == 1 &&
                       static_cast<uint64_t>(pertoken.GetDim(kDimIdx)) == inputParams_.mSize);
    if (!mDimHasOne && !kDimHasOne) {
        OP_TILING_CHECK(
            static_cast<uint64_t>(pertoken.GetDim(kDimIdx)) != ops::CeilDiv(inputParams_.kSize, HIF4_GROUP_SIZE),
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "In hif4 quantification, when x1's transposition is %s, pertokenScale's dimension %d shape \
should be equal to x1 k dimension size[%lu] ceil div 64, but actual is [%ld].",
                inputParams_.transA ? "true" : "false", kDimIdx, inputParams_.kSize, pertoken.GetDim(kDimIdx)),
            return false);
        OP_TILING_CHECK(
            static_cast<uint64_t>(pertoken.GetDim(mDimIdx)) != inputParams_.mSize,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "In hif4 quantification, when x1's transposition is %s, pertokenScale's dimension %d shape \
should be equal to x1 m dimension size[%lu], but actual is [%ld].",
                inputParams_.transA ? "true" : "false", mDimIdx, inputParams_.mSize, pertoken.GetDim(mDimIdx)),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3CheckerArch9201::Hif4ScaleShapeCheck(const gert::Shape& scaleShape) const
{
    auto scaleShapeLen = scaleShape.GetDimNum();
    OP_TILING_CHECK(
        scaleShapeLen != SCALE_THREE_DIM,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "In hif4 quantification, scale's dimension must be 3, but it is %zu.", scaleShapeLen),
        return false);
    auto kDimIdx = inputParams_.transB ? 1 : 0;
    auto nDimIdx = inputParams_.transB ? 0 : 1;
    OP_TILING_CHECK(static_cast<uint64_t>(scaleShape.GetDim(scaleShapeLen - 1)) != 1,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "In hif4 quantification, scale's dimension 2 shape should be equal to 1, but \
actual is [%ld].",
                                          scaleShape.GetDim(scaleShapeLen - 1)),
                    return false);
    bool nDimHasOne = (scaleShape.GetDim(kDimIdx) == 1 && static_cast<uint64_t>(scaleShape.GetDim(nDimIdx)) ==
                                                              ops::CeilDiv(inputParams_.kSize, HIF4_GROUP_SIZE)) ||
                      (scaleShape.GetDim(nDimIdx) == 1 && static_cast<uint64_t>(scaleShape.GetDim(kDimIdx)) ==
                                                              ops::CeilDiv(inputParams_.kSize, HIF4_GROUP_SIZE));
    bool kDimHasOne = (scaleShape.GetDim(kDimIdx) == 1 &&
                       static_cast<uint64_t>(scaleShape.GetDim(nDimIdx)) == inputParams_.nSize) ||
                      (scaleShape.GetDim(nDimIdx) == 1 &&
                       static_cast<uint64_t>(scaleShape.GetDim(kDimIdx)) == inputParams_.nSize);
    if (!nDimHasOne && !kDimHasOne) {
        OP_TILING_CHECK(
            static_cast<uint64_t>(scaleShape.GetDim(kDimIdx)) != ops::CeilDiv(inputParams_.kSize, HIF4_GROUP_SIZE),
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "In hif4 quantification, when x2's transposition is %s, scale's dimension %d shape should \
be equal to x2 k dimension size[%lu] ceil div 64, but actual is [%ld].",
                inputParams_.transB ? "true" : "false", kDimIdx, inputParams_.kSize, scaleShape.GetDim(kDimIdx)),
            return false);
        OP_TILING_CHECK(
            static_cast<uint64_t>(scaleShape.GetDim(nDimIdx)) != inputParams_.nSize,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName, "In hif4 quantification, when x2's transposition is %s, scale's dimension %d \
shape should be equal to x2 n dimension size[%lu], but actual is [%ld].",
                inputParams_.transB ? "true" : "false", nDimIdx, inputParams_.nSize, scaleShape.GetDim(nDimIdx)),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3CheckerArch9201::CheckInputValidInHif4PerGroupMode(
    const gert::Shape& scaleShape, const gert::StorageShape* pertokenShape,
    const std::vector<int64_t>& dimValueOfMKN) const
{
    if (!inputParams_.isHif4PerGroup) {
        return true;
    }
    OP_TILING_CHECK(inputParams_.hasBias,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "bias", "not null",
                                                          "when x1 and x2 are HIFLOAT4, bias must be null"),
                    return false);
    OP_TILING_CHECK(
        inputParams_.groupSizeM != 1ULL || inputParams_.groupSizeN != 1ULL || inputParams_.groupSizeK != 64ULL,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "When dtype of both x1 and x2 is hifloat4,\
input or inferred [groupSizeM, groupSizeN, groupSizeK] should be [1, 1, 64], \
but now [groupSizeM, groupSizeN, groupSizeK] is [%lu, %lu, %lu], \
groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.",
                              inputParams_.groupSizeM, inputParams_.groupSizeN, inputParams_.groupSizeK),
        return false);
    OP_TILING_CHECK(pertokenShape == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The pertokenScale is required for HIFLOAT4."),
                    return false);
    OP_TILING_CHECK(!Hif4PertokenScaleShapeCheck(pertokenShape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The perTokenScale shape check failed."), return false);
    OP_TILING_CHECK(!Hif4ScaleShapeCheck(scaleShape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The scale shape check failed."), return false);
    // hif4 场景下不支持 k 轴为 2
    OP_TILING_CHECK(inputParams_.kSize == 2,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "When the dtype of input is HIFLOAT4, the K dimension must not be 2. Actual "
                                          "x1 inner axis shape is [%ld], x2 inner axis shape is [%ld].",
                                          dimValueOfMKN[0], dimValueOfMKN[2]),
                    return false);
    OP_TILING_CHECK(
        (dimValueOfMKN[2] % 2 != 0 || dimValueOfMKN[0] % 2 != 0),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "When the dtype of input is HIFLOAT4, the inner axis of x1 and x2 must be even numbers. "
                              "Actual x1 inner axis shape is [%ld], x2 inner axis shape is [%ld]",
                              dimValueOfMKN[0], dimValueOfMKN[2]),
        return false);
    return true;
}

} // namespace optiling
