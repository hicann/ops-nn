/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_checker_for_mmads8s4.cpp
 * \brief
 */

#include "quant_batch_matmul_v3_checker_for_mmads8s4.h"

#include "common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"

namespace {
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
}  // namespace

namespace optiling {

using Ops::NN::FormatString;

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckDtypesInRange() const
{
    static const std::vector<ge::DataType> legalInputX1Dtypes = {ge::DT_INT8, ge::DT_INT4};
    static const std::vector<ge::DataType> legalInputX2Dtypes = {ge::DT_INT8, ge::DT_INT4, ge::DT_INT32};
    static const std::vector<ge::DataType> legalOutputDtypes = {ge::DT_INT8, ge::DT_FLOAT16};
    // x1 supports INT8 and INT4.
    OP_TILING_CHECK(std::find(legalInputX1Dtypes.begin(), legalInputX1Dtypes.end(), inputParams_.aDtype) ==
                        legalInputX1Dtypes.end(),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x1",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                        "the dtype of x1 must be INT8/INT4"), return false);
    // x2 supports INT8, INT4, and INT32.
    OP_TILING_CHECK(std::find(legalInputX2Dtypes.begin(), legalInputX2Dtypes.end(), inputParams_.bDtype) ==
                        legalInputX2Dtypes.end(),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x2",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                        "the dtype of x2 must be INT8/INT4/INT32"), return false);
    // Output supports INT8 and FLOAT16.
    OP_TILING_CHECK(
        std::find(legalOutputDtypes.begin(), legalOutputDtypes.end(), inputParams_.cDtype) == legalOutputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "output", ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
            "the dtype of output must be INT8/FLOAT16"), return false);
    // offset supports FLOAT only.
    auto offsetDesc = context_->GetOptionalInputDesc(GetOffsetIdx());
    OP_TILING_CHECK((offsetDesc && context_->GetOptionalInputShape(GetOffsetIdx()) != nullptr) &&
                        offsetDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "offset",
                        ge::TypeUtils::DataTypeToSerialString(offsetDesc->GetDataType()).c_str(),
                        "the dtype of offset must be FLOAT"), return false);
    // scale supports UINT64 and INT64 only.
    OP_TILING_CHECK(
        !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "scale", ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
            "the dtype of scale must be UINT64/INT64"), return false);
    // bias supports INT32 only.
    OP_TILING_CHECK((context_->GetOptionalInputDesc(GetBiasIdx()) != nullptr &&
                     context_->GetOptionalInputShape(GetBiasIdx()) != nullptr) &&
                        inputParams_.biasDtype != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                        "the dtype of bias must be INT32"), return false);
    // pertokenScale is not supported in the mmad S8S4 path.
    OP_TILING_CHECK((context_->GetOptionalInputDesc(GetPertokenIdx()) != nullptr &&
                     context_->GetOptionalInputShape(GetPertokenIdx()) != nullptr),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "pertokenScale", "not null", "pertokenScale must be null"),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckABDtypes() const
{
    // INT8 x1 accepts INT8, INT4, or INT32 x2.
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT8 &&
                        !(inputParams_.bDtype == ge::DT_INT8 || inputParams_.bDtype == ge::DT_INT4 ||
                          inputParams_.bDtype == ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x2",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                        "when the dtype of x1 is INT8, the dtype of x2 must be INT8/INT4/INT32"),
                    return false);

    // INT4 x1 accepts INT4 or INT32 x2.
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT4 &&
            !(inputParams_.bDtype == ge::DT_INT4 || inputParams_.bDtype == ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x2", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
            "when the dtype of x1 is INT4, the dtype of x2 must be INT4/INT32"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckDtype() const
{
    if (!CheckDtypesInRange()) {
        return false;
    }
    if (!CheckABDtypes()) {
        return false;
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckBias(const gert::StorageShape* biasShape) const
{
    // Bias must be 1D in this path.
    if (biasShape != nullptr) {
        auto biasDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            biasDimNum != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                inputParams_.opName, "bias", FormatString("%zuD", biasDimNum).c_str(),
                "the shape dim of bias must be 1"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckOffset(const gert::StorageShape *offsetShape) const
{
    if (offsetShape != nullptr) {
        // offset is only allowed when the output dtype is INT8.
        OP_TILING_CHECK(inputParams_.cDtype != ge::DT_INT8,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "offset", "not null",
                            "when the dtype of output is not INT8, offset must be null"),
                        return false);
        // offset must be 1D.
        OP_TILING_CHECK(
            offsetShape->GetStorageShape().GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                inputParams_.opName, "offset",
                FormatString("%zuD", offsetShape->GetStorageShape().GetDimNum()).c_str(),
                "the shape dim of offset must be 1"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckShapeInRangeForOptionalInputs(const gert::StorageShape *biasShape,
                                                                            const gert::StorageShape *offsetShape) const
{
    return CheckBias(biasShape) && CheckOffset(offsetShape);
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckShapeInBoundary(const gert::Shape &shape, uint32_t shapeIdx) const
{
    int64_t mul = 1;
    int64_t mulBound = 1;
    const char *dimName = shapeIdx == GetX1Idx() ? "x1" : "x2";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t curDim = shape.GetDim(i);

        // Each x1/x2 dimension must stay within INT32 range.
        OP_TILING_CHECK(
            curDim <= 0 || curDim > static_cast<int64_t>(INT32_MAX),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams_.opName, FormatString("%s dimension %zu", dimName, i).c_str(),
                std::to_string(curDim).c_str(),
                FormatString("the dimension value of %s must be within the range of [1,%d]",
                             dimName, INT32_MAX).c_str()),
            return false);
        // The total element count must stay within INT64 range.
        mulBound = curDim * mul;
        OP_TILING_CHECK(mulBound / curDim != mul,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opName, dimName, Shape2String(shape).c_str(),
                            "the product of shape dimensions must be within the range of INT64_MAX"),
                        return false);
        mul = mulBound;
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::ExtraInputCheck() const
{
    // INT4 x1 does not support transA.
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT4 && inputParams_.transA,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams_.opName, "trans_a", inputParams_.transA ? "true" : "false",
            "when the dtype of x1 is INT4, trans_a must be false"),
        return false);

    // x1 must use ND format.
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    OP_TILING_CHECK(x1Format != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(
                        inputParams_.opName, "x1", ge::TypeUtils::FormatToSerialString(x1Format).c_str(), "ND"),
                    return false);
    // INT4 x1 requires x2 to use FRACTAL_NZ.
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT4 && x2Format != ge::Format::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                        inputParams_.opName, "x2", ge::TypeUtils::FormatToSerialString(x2Format).c_str(),
                        "when the dtype of x1 is INT4, the format of x2 must be FRACTAL_NZ"),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckDimValue(const gert::Shape &scaleShape,
                                                       const gert::StorageShape *biasShape,
                                                       const gert::StorageShape *offsetShape,
                                                       const std::vector<int64_t> &dimValueOfMKN) const
{
    // dimValueOfMKN stores x1 inner, x1 outer, x2 inner, and x2 outer.
    auto x1Inner = dimValueOfMKN[0];
    auto x2Inner = dimValueOfMKN[2];
    auto x2Outer = dimValueOfMKN[3];
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? x2Inner : x2Outer);
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "x1 K, x2 K",
                        FormatString("%lu, %lu", inputParams_.kSize, kBSize).c_str(),
                        "the K dimension of x1 must be equal to the K dimension of x2"),
                    return false);
    // Bias shape must match N.
    OP_TILING_CHECK(
        biasShape != nullptr && static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams_.nSize,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "biasDim, n",
            FormatString("%ld, %lu", biasShape->GetStorageShape().GetDim(0), inputParams_.nSize).c_str(),
            "dimension 0 of bias must be equal to n"),
        return false);
    // offset shape must be 1 or N.
    OP_TILING_CHECK(
        offsetShape != nullptr &&
            !(offsetShape->GetStorageShape().GetDim(0) == 1 ||
              static_cast<uint64_t>(offsetShape->GetStorageShape().GetDim(0)) == inputParams_.nSize),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "n, offsetDim",
            FormatString("%lu, %ld", inputParams_.nSize, offsetShape->GetStorageShape().GetDim(0)).c_str(),
            "dimension 0 of offset must be 1 or n"),
        return false);
    // scale must be 1D.
    OP_TILING_CHECK(scaleShape.GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            inputParams_.opName, "scale", FormatString("%zuD", scaleShape.GetDimNum()).c_str(),
            "the shape dim of scale must be 1"),
        return false);
    // INT8 x1 supports PerTensor and PerChannel.
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT8 && !(inputParams_.isPerTensor || inputParams_.isPerChannel),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x1", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            "when the dtype of x1 is INT8, the quantization mode must be PerTensor/PerChannel"),
        return false);
    // INT4 x1 supports PerChannel only.
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT4 && !inputParams_.isPerChannel,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x1", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                    "when the dtype of x1 is INT4, the quantization mode must be PerChannel"),
                    return false);
    // INT4 x1 requires an even inner axis.
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT4 && (x1Inner < 0 || x1Inner % 2 != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams_.opName, "x1 last axis", std::to_string(x1Inner).c_str(),
            "when the dtype of x1 is INT4, the last axis of x1 must be a positive even number"),
        return false);
    // INT4 x2 requires an even inner axis.
    OP_TILING_CHECK(
        inputParams_.bDtype == ge::DT_INT4 && (x2Inner < 0 || x2Inner % 2 != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams_.opName, "x2 last axis", std::to_string(x2Inner).c_str(),
            "when the dtype of x2 is INT4, the last axis of x2 must be a positive even number"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckShape(
    const std::vector<gert::Shape*>& mandtoryShape, const gert::StorageShape* biasShape,
    const gert::StorageShape* /* pertokenShape */, const std::vector<int64_t>& dimValueOfMKN) const
{
    auto& x1Shape = *mandtoryShape[0];
    auto& x2Shape = *mandtoryShape[1];
    auto& scaleShape = *mandtoryShape[2];
    auto offsetShape = context_->GetOptionalInputShape(GetOffsetIdx());
    if (!CheckShapeInRangeForOptionalInputs(biasShape, offsetShape) ||
        !CheckDimValue(scaleShape, biasShape, offsetShape, dimValueOfMKN) ||
        !CheckShapeInBoundary(x1Shape, GetX1Idx()) || !CheckShapeInBoundary(x2Shape, GetX2Idx())) {
        return false;
    }
    if (!ExtraInputCheck()) {
        return false;
    }
    return true;
}
}  // namespace optiling
