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
 * \file quant_batch_matmul_v4_checker_for_mmads8s4.cc
 * \brief
 */
#include "quant_batch_matmul_v4_checker_for_mmads8s4.h"

#include <map>

#include "common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "log/log.h"

namespace {

// S4toS8 LUT, int4可表达-8~7, 共16个可能的index
constexpr uint64_t INT4_INDEX_SIZE = 16UL;
// S2toS4 LUT, int2可表达-2~1, 共4个可能的index
constexpr uint64_t INT2_INDEX_SIZE = 4UL;
// S1toS4 LUT, uint1可表达0~1, 共2个可能的index
constexpr uint64_t UINT1_INDEX_SIZE = 2UL;

constexpr uint64_t INT4_BIT_LENGTH = 4UL;
constexpr uint64_t INT8_BIT_LENGTH = 8UL;
constexpr uint64_t LUT_ALIGN_BIT_LENGTH = 64UL;

// 1Byte数据量对应INT1/INT2/INT4个数
constexpr uint32_t INT4_NUMS_IN_BYTE = 2;
constexpr uint32_t INT2_NUMS_IN_BYTE = 4;
constexpr uint32_t UINT1_NUMS_IN_BYTE = 8;

const std::map<ge::DataType, uint32_t> DTYPE_NUMS_IN_BYTE_MAP = {
    {ge::DT_INT4, INT4_NUMS_IN_BYTE}, {ge::DT_INT2, INT2_NUMS_IN_BYTE}, {ge::DT_UINT1, UINT1_NUMS_IN_BYTE}};

const std::map<ge::DataType, uint64_t> DTYPE_INDEX_SIZE_MAP = {
    {ge::DT_INT4, INT4_INDEX_SIZE}, {ge::DT_INT2, INT2_INDEX_SIZE}, {ge::DT_UINT1, UINT1_INDEX_SIZE}};

const std::map<ge::DataType, uint64_t> DTYPE_BIT_LENGTH_MAP = {
    {ge::DT_INT4, INT4_BIT_LENGTH},
    {ge::DT_INT8, INT8_BIT_LENGTH},
};

}  // namespace

namespace optiling {

bool QuantBatchMatmulV4Checker4MmadS8S4::CheckDtypesInRange() const
{
    static const std::vector<ge::DataType> legalInputX1Dtypes = {ge::DT_INT8};
    static const std::vector<ge::DataType> legalInputX2Dtypes = {ge::DT_UINT1, ge::DT_INT2, ge::DT_INT4};
    static const std::vector<ge::DataType> legalOutputDtypes = {ge::DT_INT8, ge::DT_FLOAT16};
    if (inputParams_.isLut) {
        // isLut为true的条件是x2Table存在且平台支持lut_type为mte2_qtable
        // LUT场景，仅支持x1 INT8，x2 UINT1/INT2/INT4
        OP_TILING_CHECK(std::find(legalInputX1Dtypes.begin(), legalInputX1Dtypes.end(), inputParams_.aDtype) ==
                            legalInputX1Dtypes.end(),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "The x1 dtype must be INT8, actual is %s.",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
                        return false);
        // x2可取 UINT1/INT2/INT4
        OP_TILING_CHECK(
            std::find(legalInputX2Dtypes.begin(), legalInputX2Dtypes.end(), inputParams_.bDtype) ==
                legalInputX2Dtypes.end(),
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "The x2 dtype must be UINT1/INT2/INT4, actual is %s.",
                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
            return false);
    }
    // output可取INT8, FLOAT16
    OP_TILING_CHECK(
        std::find(legalOutputDtypes.begin(), legalOutputDtypes.end(), inputParams_.cDtype) == legalOutputDtypes.end(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Output dtype must be INT8/FLOAT16, actual is %s.",
                              ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
        return false);
    // x1Offset不存在
    OP_TILING_CHECK((context_->GetOptionalInputDesc(X1_OFFSET_INDEX_V4) != nullptr &&
                     context_->GetOptionalInputShape(X1_OFFSET_INDEX_V4) != nullptr),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "X1Offset should be null."), return false);
    // x2ffset可取FLOAT
    auto offsetDesc = context_->GetOptionalInputDesc(GetOffsetIdx());
    OP_TILING_CHECK((offsetDesc && context_->GetOptionalInputShape(GetOffsetIdx()) != nullptr) &&
                        offsetDesc->GetDataType() != ge::DT_FLOAT,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2Offset dtype should be FLOAT, actual dtype is %s.",
                                          ge::TypeUtils::DataTypeToSerialString(offsetDesc->GetDataType()).c_str()),
                    return false);
    // yOffset不存在
    OP_TILING_CHECK((context_->GetOptionalInputDesc(Y_OFFSET_INDEX_V4) != nullptr &&
                     context_->GetOptionalInputShape(Y_OFFSET_INDEX_V4) != nullptr),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "YOffset should be null."), return false);
    // x2Scale可取UINT64, INT64
    OP_TILING_CHECK(
        context_->GetOptionalInputDesc(GetScaleIdx()) != nullptr &&
            !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2Scale dtype should be UINT64/INT64, actual dtype is %s.",
                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
        return false);
    // bias可取INT32
    OP_TILING_CHECK((context_->GetOptionalInputDesc(GetBiasIdx()) != nullptr &&
                     context_->GetOptionalInputShape(GetBiasIdx()) != nullptr) &&
                        inputParams_.biasDtype != ge::DT_INT32,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Bias dtype should be INT32, actual dtype is %s.",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),
                    return false);
    // x1Scale不存在
    OP_TILING_CHECK((context_->GetOptionalInputDesc(GetPertokenIdx()) != nullptr &&
                     context_->GetOptionalInputShape(GetPertokenIdx()) != nullptr),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "X1Scale should be null."), return false);
    // yScale不存在
    OP_TILING_CHECK((context_->GetOptionalInputDesc(Y_SCALE_INDEX_V4) != nullptr &&
                     context_->GetOptionalInputShape(Y_SCALE_INDEX_V4) != nullptr),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "YScale should be null."), return false);
    if (inputParams_.isLut) {
        // LUT场景，x2 UINT1/INT2对应x2Table INT4, x2 INT4对应x2Table INT8
        OP_TILING_CHECK((inputParams_.bDtype == ge::DT_INT2 || inputParams_.bDtype == ge::DT_UINT1) &&
                            inputParams_.x2TableDtype != ge::DT_INT4,
                        CUBE_INNER_ERR_REPORT(
                            inputParams_.opName,
                            "In LUT scenario, when x2 dtype is UINT1/INT2, x2Table dtype should be INT4, actual is %s",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.x2TableDtype).c_str()),
                        return false);
        OP_TILING_CHECK(
            inputParams_.bDtype == ge::DT_INT4 && inputParams_.x2TableDtype != ge::DT_INT8,
            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                  "In LUT scenario, when x2 dtype is INT4, x2Table dtype should be INT8, actual is %s",
                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.x2TableDtype).c_str()),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV4Checker4MmadS8S4::CheckABDtypes() const
{
    return true;
}

/*
    This function calculates the ratio between the size of the input x2Table and the actual number of LUT (Look-Up
   Table) entries used. A single LUT contains 'singleLutSize' elements from the x2Table, with the hardware constraint
   that each LUT must be 64-bit aligned, Factor Calculation: singleLutSize = ceilDiv(idxSize(bDtype) *
   bitLength(x2TableDtype), 64) / bitLength(x2TableDtype)
*/
bool QuantBatchMatmulV4Checker4MmadS8S4::CalcSingleLutSize(const ge::DataType bDtype, const ge::DataType x2TableDtype,
                                                           uint64_t &singleLutSize) const
{
    auto dtypeBitLengthIterator = DTYPE_BIT_LENGTH_MAP.find(x2TableDtype);
    OP_TILING_CHECK(dtypeBitLengthIterator == DTYPE_BIT_LENGTH_MAP.end(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "can't find key[%s] in DTYPE_BIT_LENGTH_MAP",
                                          ge::TypeUtils::DataTypeToSerialString(x2TableDtype).c_str()),
                    return false);
    uint64_t bitLength = dtypeBitLengthIterator->second;

    auto dtypeIdxSizeIterator = DTYPE_INDEX_SIZE_MAP.find(bDtype);
    OP_TILING_CHECK(dtypeIdxSizeIterator == DTYPE_INDEX_SIZE_MAP.end(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "can't find key[%s] in DTYPE_INDEX_SIZE_MAP",
                                          ge::TypeUtils::DataTypeToSerialString(bDtype).c_str()),
                    return false);
    uint64_t idxSize = dtypeIdxSizeIterator->second;

    singleLutSize = ops::CeilAlign(idxSize * bitLength, LUT_ALIGN_BIT_LENGTH) / bitLength;
    return true;
}

/*
    LUT Count in k and n Directions: lutK = ceilDiv(k, groupSizeK), lutN = ceilDiv(n, groupSizeN)
    Input x2Table Shape: (lutN, lutK * singleLutSize)
    +------------+--------+--------------+---------------+
    |  LUT type  | bDtype | x2TableDtype | singleLutSize |
    +------------+--------+--------------+---------------+
    | S4toS8 LUT | int4   | int8         |     16        |
    | S2toS4 LUT | int2   | int4         |     16        |
    | S1toS4 LUT | int1   | int4         |     16        |
    +------------+--------+--------------+---------------+
*/
bool QuantBatchMatmulV4Checker4MmadS8S4::CheckX2TableShape() const
{
    uint64_t singleLutSize = 0;
    OP_TILING_CHECK(!CalcSingleLutSize(inputParams_.bDtype, inputParams_.x2TableDtype, singleLutSize),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to calculate single LUT size"), return false);
    OP_TILING_CHECK(
        ops::CeilDiv(inputParams_.kSize, inputParams_.groupSizeK) * singleLutSize != inputParams_.x2TableKSize,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x2TableKSize should be %zu, but it is %zu",
                              ops::CeilDiv(inputParams_.kSize, inputParams_.groupSizeK) * singleLutSize,
                              inputParams_.x2TableKSize),
        return false);
    OP_TILING_CHECK(
        ops::CeilDiv(inputParams_.nSize, inputParams_.groupSizeN) != inputParams_.x2TableNSize,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x2TableNSize should be %zu, but it is %zu",
                              ops::CeilDiv(inputParams_.nSize, inputParams_.groupSizeN), inputParams_.x2TableNSize),
        return false);
    return true;
}

bool QuantBatchMatmulV4Checker4MmadS8S4::CheckDimValue(const gert::StorageShape *scaleShape,
                                                       const gert::StorageShape *biasShape,
                                                       const gert::StorageShape *offsetShape,
                                                       const std::vector<int64_t> &dimValueOfMKN) const
{
    // x1,x2 shapeK必须相等
    auto x2Inner = dimValueOfMKN[2];  // using index 2 to get x2Inner
    auto x2Outer = dimValueOfMKN[3];  // using index 3 to get x2Outer
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? x2Inner : x2Outer);
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "The size of k dimension of x1[%lu] is not equal to \
                                           the size of k dimension of x2[%lu]",
                                          inputParams_.kSize, kBSize),
                    return false);
    // bias shape必须等于shapeN
    OP_TILING_CHECK(
        biasShape != nullptr && static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams_.nSize,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "Input bias dimension shape should equal n, but it is %ld while n is %lu.",
                              biasShape->GetStorageShape().GetDim(0), inputParams_.nSize),
        return false);
    // offset shape必须是1或shapeN
    OP_TILING_CHECK(
        offsetShape != nullptr &&
            !(offsetShape->GetStorageShape().GetDim(0) == 1 ||
              static_cast<uint64_t>(offsetShape->GetStorageShape().GetDim(0)) == inputParams_.nSize),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2Offset dimension value must be 1 or n[%lu], but it is %ld.",
                              inputParams_.nSize, offsetShape->GetStorageShape().GetDim(0)),
        return false);
    // scale维数必须存在
    OP_TILING_CHECK(scaleShape == nullptr, CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2Scale does not exist"),
                    return false);
    // scale维数必须是1维
    OP_TILING_CHECK(scaleShape->GetStorageShape().GetDimNum() != 1,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2Scale's dimension must be 1, actually is : %zu",
                                          scaleShape->GetStorageShape().GetDimNum()),
                    return false);
    // 当x1为INT8时，支持perchannel量化模式
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT8 && !inputParams_.isPerChannel,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "When x1 dtype is INT8, the only supported quant mode is perchannel"),
        return false);
    // LUT场景x2 UIN1/INT2/INT4尾轴shape分别需要关于8/4/2对齐
    if (inputParams_.isLut && (inputParams_.bDtype == ge::DT_INT4 || inputParams_.bDtype == ge::DT_INT2 ||
                               inputParams_.bDtype == ge::DT_UINT1)) {
        auto it = DTYPE_NUMS_IN_BYTE_MAP.find(inputParams_.bDtype);
        OP_TILING_CHECK(it == DTYPE_NUMS_IN_BYTE_MAP.end(),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "DTYPE_NUMS_IN_BYTE_MAP[%s] is not exist",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                        return false);

        OP_TILING_CHECK(x2Inner % DTYPE_NUMS_IN_BYTE_MAP.at(inputParams_.bDtype) != 0,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "the last dim of x2 should be a multiple of %u when x2 dtype is %s",
                                              DTYPE_NUMS_IN_BYTE_MAP.at(inputParams_.bDtype),
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                        return false);
        OP_TILING_CHECK(inputParams_.groupSizeK == 0 || inputParams_.groupSizeN == 0,
                        CUBE_INNER_ERR_REPORT(
                            inputParams_.opName,
                            "groupSizeK or groupSizeN should not be zero when x2 dtype is %s, actual is [%lu, %lu]",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(), inputParams_.groupSizeK,
                            inputParams_.groupSizeN),
                        return false);
        OP_TILING_CHECK(!CheckX2TableShape(),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x2Table shape is invalid"), return false);
    }
    return true;
}

bool QuantBatchMatmulV4Checker4MmadS8S4::CheckShape(
    const std::vector<gert::Shape*>& mandtoryShape, const gert::StorageShape* biasShape,
    const gert::StorageShape* /* pertokenShape */, const std::vector<int64_t>& dimValueOfMKN) const
{
    auto &x1Shape = *mandtoryShape[0];     // using index 0 to get x1Shape
    auto &x2Shape = *mandtoryShape[1];     // using index 1 to get x2Shape
    auto scaleShape = context_->GetOptionalInputShape(GetScaleIdx());
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

bool QuantBatchMatmulV4Checker4MmadS8S4::ExtraInputCheck() const
{
    if (inputParams_.isLut) {
        // LUT场景，x1必须为ND, x2必须为NZ
        auto x1Desc = context_->GetInputDesc(GetX1Idx());
        auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
        auto x2Desc = context_->GetInputDesc(GetX2Idx());
        auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
        OP_TILING_CHECK(
            x1Format != ge::Format::FORMAT_ND || x2Format != ge::Format::FORMAT_FRACTAL_NZ,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "In LUT scenario, input x1 format should be ND, x2 format should be FRACTAL_NZ, actual [%s, %s].",
                ge::TypeUtils::FormatToSerialString(x1Format).c_str(),
                ge::TypeUtils::FormatToSerialString(x2Format).c_str()),
            return false);

        // LUT场景，tranA/transB为false
        OP_TILING_CHECK(
            inputParams_.transA || inputParams_.transB,
            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                  "In LUT scenario, trans_a and trans_b should be false, actual [%s, %s]",
                                  inputParams_.transA ? "true" : "false", inputParams_.transB ? "true" : "false"),
            return false);

        // LUT场景，不支持batch
        OP_TILING_CHECK(
            !(inputParams_.batchA == 1 && inputParams_.batchB == 1),
            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                  "In LUT scenario,  x1 batch and x2 batch should be 1/NULL, actual [%lu, %lu]",
                                  inputParams_.batchA, inputParams_.batchB),
            return false);
    }

    return true;
}

bool QuantBatchMatmulV4Checker4MmadS8S4::CheckOffset(const gert::StorageShape* offsetShape) const
{
    if (offsetShape != nullptr) {
        // 当outDtype不为INT8时，x2Offset不存在
        OP_TILING_CHECK(
            inputParams_.cDtype != ge::DT_INT8,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "When outputDtype is not INT8, x2Offset must be null"),
            return false);
        // x2Offset维数只能是1维
        OP_TILING_CHECK(
            offsetShape->GetStorageShape().GetDimNum() != 1,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2Offset shape should be 1 dimension, but it is %zu",
                                  offsetShape->GetStorageShape().GetDimNum()),
            return false);
    }
    return true;
}

}  // namespace optiling