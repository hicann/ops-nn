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
        OP_TILING_CHECK(
            std::find(legalInputX1Dtypes.begin(), legalInputX1Dtypes.end(), inputParams_.aDtype) ==
                legalInputX1Dtypes.end(),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x1",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                                  "The dtype of x1 must be INT8"),
            return false);
        // x2可取 UINT1/INT2/INT4
        OP_TILING_CHECK(
            std::find(legalInputX2Dtypes.begin(), legalInputX2Dtypes.end(), inputParams_.bDtype) ==
                legalInputX2Dtypes.end(),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x2",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                                                  "The dtype of x2 must be UINT1, INT2, or INT4"),
            return false);
    }
    // output可取INT8, FLOAT16
    OP_TILING_CHECK(
        std::find(legalOutputDtypes.begin(), legalOutputDtypes.end(), inputParams_.cDtype) == legalOutputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "y",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                              "The dtype of y must be INT8 or FLOAT16"),
        return false);
    // x1Offset不存在
    OP_TILING_CHECK((context_->GetOptionalInputDesc(X1_OFFSET_INDEX_V4) != nullptr &&
                     context_->GetOptionalInputShape(X1_OFFSET_INDEX_V4) != nullptr),
                    OP_LOGE(inputParams_.opName, "X1Offset should be null."), return false);
    // x2ffset可取FLOAT
    auto offsetDesc = context_->GetOptionalInputDesc(GetOffsetIdx());
    OP_TILING_CHECK(
        (offsetDesc && context_->GetOptionalInputShape(GetOffsetIdx()) != nullptr) &&
            offsetDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x2Offset", ge::TypeUtils::DataTypeToSerialString(offsetDesc->GetDataType()).c_str(),
            "The dtype of x2Offset must be FLOAT"),
        return false);
    // yOffset不存在
    OP_TILING_CHECK((context_->GetOptionalInputDesc(Y_OFFSET_INDEX_V4) != nullptr &&
                     context_->GetOptionalInputShape(Y_OFFSET_INDEX_V4) != nullptr),
                    OP_LOGE(inputParams_.opName, "YOffset should be null."), return false);
    // x2Scale可取UINT64, INT64
    OP_TILING_CHECK(
        context_->GetOptionalInputDesc(GetScaleIdx()) != nullptr &&
            !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x2Scale", ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
            "The dtype of x2Scale must be UINT64 or INT64"),
        return false);
    // bias可取INT32
    OP_TILING_CHECK(
        (context_->GetOptionalInputDesc(GetBiasIdx()) != nullptr &&
         context_->GetOptionalInputShape(GetBiasIdx()) != nullptr) &&
            inputParams_.biasDtype != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "bias", ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
            "The dtype of bias must be INT32"),
        return false);
    // x1Scale不存在
    OP_TILING_CHECK((context_->GetOptionalInputDesc(GetPertokenIdx()) != nullptr &&
                     context_->GetOptionalInputShape(GetPertokenIdx()) != nullptr),
                    OP_LOGE(inputParams_.opName, "X1Scale should be null."), return false);
    // yScale不存在
    OP_TILING_CHECK((context_->GetOptionalInputDesc(Y_SCALE_INDEX_V4) != nullptr &&
                     context_->GetOptionalInputShape(Y_SCALE_INDEX_V4) != nullptr),
                    OP_LOGE(inputParams_.opName, "YScale should be null."), return false);
    if (inputParams_.isLut) {
        // LUT场景，x2 UINT1/INT2对应x2Table INT4, x2 INT4对应x2Table INT8
        OP_TILING_CHECK(
            (inputParams_.bDtype == ge::DT_INT2 || inputParams_.bDtype == ge::DT_UINT1) &&
                inputParams_.x2TableDtype != ge::DT_INT4,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "x2Table",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.x2TableDtype).c_str(),
                "When the dtype of x2 is UINT1 or INT2, the dtype of x2Table must be INT4"),
            return false);
        OP_TILING_CHECK(
            inputParams_.bDtype == ge::DT_INT4 && inputParams_.x2TableDtype != ge::DT_INT8,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "x2Table",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.x2TableDtype).c_str(),
                "When the dtype of x2 is INT4, the dtype of x2Table must be INT8"),
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
    OP_TILING_CHECK(
        dtypeBitLengthIterator == DTYPE_BIT_LENGTH_MAP.end(),
        OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(
            inputParams_.opName, ge::TypeUtils::DataTypeToSerialString(x2TableDtype).c_str(), "x2TableDtype",
            "DTYPE_BIT_LENGTH_MAP", "dtype not found in bit length map"),
        return false);
    uint64_t bitLength = dtypeBitLengthIterator->second;

    auto dtypeIdxSizeIterator = DTYPE_INDEX_SIZE_MAP.find(bDtype);
    OP_TILING_CHECK(
        dtypeIdxSizeIterator == DTYPE_INDEX_SIZE_MAP.end(),
        OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(
            inputParams_.opName, ge::TypeUtils::DataTypeToSerialString(bDtype).c_str(), "bDtype",
            "DTYPE_INDEX_SIZE_MAP", "dtype not found in index size map"),
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
                    OP_LOGE(inputParams_.opName, "failed to calculate single LUT size"), return false);
    OP_TILING_CHECK(
        ops::CeilDiv(inputParams_.kSize, inputParams_.groupSizeK) * singleLutSize != inputParams_.x2TableKSize,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            inputParams_.opName, "x2TableKSize", std::to_string(inputParams_.x2TableKSize).c_str(),
            "The shape size of x2TableKSize must be " +
                std::to_string(ops::CeilDiv(inputParams_.kSize, inputParams_.groupSizeK) * singleLutSize)),
        return false);
    OP_TILING_CHECK(
        ops::CeilDiv(inputParams_.nSize, inputParams_.groupSizeN) != inputParams_.x2TableNSize,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            inputParams_.opName, "x2TableNSize", std::to_string(inputParams_.x2TableNSize).c_str(),
            "The shape size of x2TableNSize must be " +
                std::to_string(ops::CeilDiv(inputParams_.nSize, inputParams_.groupSizeN))),
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
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opName, "x1, x2", "kSize mismatch",
                                                           "The k dimension sizes of x1 and x2 must be equal"),
                    return false);
    // bias shape必须等于shapeN
    OP_TILING_CHECK(
        biasShape != nullptr && static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams_.nSize,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "bias",
                                                 std::to_string(biasShape->GetStorageShape().GetDim(0)).c_str(),
                                                 "The shape dim of bias must be equal to nSize"),
        return false);
    // offset shape必须是1或shapeN
    OP_TILING_CHECK(
        offsetShape != nullptr &&
            !(offsetShape->GetStorageShape().GetDim(0) == 1 ||
              static_cast<uint64_t>(offsetShape->GetStorageShape().GetDim(0)) == inputParams_.nSize),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x2Offset",
                                                 std::to_string(offsetShape->GetStorageShape().GetDim(0)).c_str(),
                                                 "The shape dim of x2Offset must be 1 or nSize"),
        return false);
    // scale维数必须存在
    OP_TILING_CHECK(scaleShape == nullptr, OP_LOGE(inputParams_.opName, "X2Scale does not exist"),
                    return false);
    // scale维数必须是1维
    OP_TILING_CHECK(
        scaleShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x2Scale",
                                                 std::to_string(scaleShape->GetStorageShape().GetDimNum()).c_str(),
                                                 "The shape dim of x2Scale must be 1D"),
        return false);
    // 当x1为INT8时，支持perchannel量化模式
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT8 && !inputParams_.isPerChannel,
        OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(inputParams_.opName, "perchannel", "antiQuantType", "quantMode",
                                               "When the dtype of x1 is INT8, the quant mode must be per_channel"),
        return false);
    // LUT场景x2 UIN1/INT2/INT4尾轴shape分别需要关于8/4/2对齐
    if (inputParams_.isLut && (inputParams_.bDtype == ge::DT_INT4 || inputParams_.bDtype == ge::DT_INT2 ||
                               inputParams_.bDtype == ge::DT_UINT1)) {
        auto it = DTYPE_NUMS_IN_BYTE_MAP.find(inputParams_.bDtype);
        OP_TILING_CHECK(it == DTYPE_NUMS_IN_BYTE_MAP.end(),
                        OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(
                            inputParams_.opName, ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                            "bDtype", "DTYPE_NUMS_IN_BYTE_MAP", "dtype not found in nums map"),
                        return false);

        OP_TILING_CHECK(
            x2Inner % DTYPE_NUMS_IN_BYTE_MAP.at(inputParams_.bDtype) != 0,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opName, "x2", "x2Inner",
                                                   "The last dim of x2 must be a multiple of dtypesPerByte"),
            return false);
        OP_TILING_CHECK(
            inputParams_.groupSizeK == 0 || inputParams_.groupSizeN == 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams_.opName, "groupSizeK/groupSizeN", "[inputParams_.groupSizeK, inputParams_.groupSizeN]",
                "When in LUT mode, the values of groupSizeK and groupSizeN can not be 0"),
            return false);
        OP_TILING_CHECK(!CheckX2TableShape(),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opName, "x2Table", "x2TableShape",
                                                               "shape validation failed"),
                        return false);
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
        OP_TILING_CHECK(x1Format != ge::Format::FORMAT_ND || x2Format != ge::Format::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                            inputParams_.opName, "x1, x2",
                            Ops::Base::ToString(x1Format) + ", " + Ops::Base::ToString(x2Format),
                            "When in LUT mode, the format of x1 must be ND and the format of x2 must be FRACTAL_NZ"),
                        return false);

        // LUT场景，tranA/transB为false
        OP_TILING_CHECK(
            inputParams_.transA || inputParams_.transB,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "transA/transB", "actual_transA, actual_transB",
                                                  "When in LUT mode, the values of transA and transB must be false"),
            return false);

        // LUT场景，不支持batch
        OP_TILING_CHECK(!(inputParams_.batchA == 1 && inputParams_.batchB == 1),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "x1Batch/x2Batch", "actual_batches",
                                                              "When in LUT mode, the batch of x1 and x2 must be 1"),
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
            OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(inputParams_.opName, "x2Offset", "x2Offset", "quantConfig",
                                                   "When the dtype of y is not INT8, x2Offset can not exist"),
            return false);
        // x2Offset维数只能是1维
        OP_TILING_CHECK(
            offsetShape->GetStorageShape().GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x2Offset",
                                                     std::to_string(offsetShape->GetStorageShape().GetDimNum()).c_str(),
                                                     "The shape dim of x2Offset must be 1D"),
            return false);
    }
    return true;
}

}  // namespace optiling