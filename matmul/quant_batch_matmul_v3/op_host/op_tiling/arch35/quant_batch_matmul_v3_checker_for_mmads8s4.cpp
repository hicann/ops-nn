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
 * \file quant_batch_matmul_v3_checker_for_mmads8s4.cpp
 * \brief
 */

#include "quant_batch_matmul_v3_checker_for_mmads8s4.h"

#include "common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "log/log.h"

namespace {
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr int64_t LAST_AXIS_LIMIT = 2097151;  // 2^21-1
}  // namespace

namespace optiling {

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckDtypesInRange() const
{
    static const std::vector<ge::DataType> legalInputX1Dtypes = {ge::DT_INT8, ge::DT_INT4};
    static const std::vector<ge::DataType> legalInputX2Dtypes = {ge::DT_INT8, ge::DT_INT4, ge::DT_INT32};
    static const std::vector<ge::DataType> legalOutputDtypes = {ge::DT_INT8, ge::DT_FLOAT16};
    // x1可取INT8, INT4
    OP_TILING_CHECK(std::find(legalInputX1Dtypes.begin(), legalInputX1Dtypes.end(), inputParams_.aDtype) ==
                        legalInputX1Dtypes.end(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The x1 dtype must be INT8/INT4, actual is %s.",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
                    return false);
    // x2可取INT8, INT4, INT32
    OP_TILING_CHECK(std::find(legalInputX2Dtypes.begin(), legalInputX2Dtypes.end(), inputParams_.bDtype) ==
                        legalInputX2Dtypes.end(),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The x2 dtype must be INT8/INT4/INT32, actual is %s.",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                    return false);
    // output可取INT8, FLOAT16
    OP_TILING_CHECK(
        std::find(legalOutputDtypes.begin(), legalOutputDtypes.end(), inputParams_.cDtype) == legalOutputDtypes.end(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Output dtype must be INT8/FLOAT16, actual is %s.",
                              ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
        return false);
    // offset可取FLOAT
    auto offsetDesc = context_->GetOptionalInputDesc(GetOffsetIdx());
    OP_TILING_CHECK((offsetDesc && context_->GetOptionalInputShape(GetOffsetIdx()) != nullptr) &&
                        offsetDesc->GetDataType() != ge::DT_FLOAT,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The offset dtype should be FLOAT, actual dtype is %s.",
                                          ge::TypeUtils::DataTypeToSerialString(offsetDesc->GetDataType()).c_str()),
                    return false);
    // scale可取UINT64, INT64
    OP_TILING_CHECK(
        !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Scale dtype should be UINT64/INT64, actual dtype is %s.",
                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
        return false);
    // bias可取INT32
    OP_TILING_CHECK((context_->GetOptionalInputDesc(GetBiasIdx()) != nullptr &&
                     context_->GetOptionalInputShape(GetBiasIdx()) != nullptr) &&
                        inputParams_.biasDtype != ge::DT_INT32,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Bias dtype should be INT32, actual dtype is %s.",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),
                    return false);
    // pertokenScale不存在
    OP_TILING_CHECK((context_->GetOptionalInputDesc(GetPertokenIdx()) != nullptr &&
                     context_->GetOptionalInputShape(GetPertokenIdx()) != nullptr),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "PertokenScale should be null."), return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckABDtypes() const
{
    // 当x为INT8, x2必须是INT8, INT4, INT32
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT8 &&
                        !(inputParams_.bDtype == ge::DT_INT8 || inputParams_.bDtype == ge::DT_INT4 ||
                          inputParams_.bDtype == ge::DT_INT32),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "When x1 dtype is INT8, x2 dtype must be INT8/INT4/INT32, actual is %s.",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                    return false);

    // 当x为INT4, x2必须是INT4, INT32
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT4 &&
            !(inputParams_.bDtype == ge::DT_INT4 || inputParams_.bDtype == ge::DT_INT32),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "When x1 dtype is INT4, x2 dtype must be INT4/INT32, actual is %s.",
                              ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
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

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckShapeInRangeForOptionalInputs(const gert::StorageShape *biasShape,
                                                                            const gert::StorageShape *offsetShape) const
{
    // bias维数只能是1维
    if (biasShape != nullptr) {
        auto biasDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(biasDimNum != 1,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "The bias dimension should equal to 1, but it is %zu.", biasDimNum),
                        return false);
    }
    if (offsetShape != nullptr) {
        // 当outDtype不为INT8时，offset不存在
        OP_TILING_CHECK(inputParams_.cDtype != ge::DT_INT8,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "When outputDtype is not INT8, offset must be null"),
                        return false);
        // offset维数只能是1维
        OP_TILING_CHECK(
            offsetShape->GetStorageShape().GetDimNum() != 1,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "The offset shape should be 1 dimension, but it is %zu",
                                  offsetShape->GetStorageShape().GetDimNum()),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckShapeInBoundary(const gert::Shape &shape, uint32_t shapeIdx) const
{
    int64_t mul = 1;
    int64_t mulBound = 1;
    const char *dimName = shapeIdx == GetX1Idx() ? "x1" : "x2";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t curDim = shape.GetDim(i);
        // 尾轴不超过2^21-1
        OP_TILING_CHECK(i == shape.GetDimNum() - LAST_FIRST_DIM_INDEX && curDim > LAST_AXIS_LIMIT,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "The last dimension size of %s should not be \
larger than 2097151 but actual is %ld",
                                              dimName, curDim),
                        return false);

        // x1或x2的shape每一维元素数不超过INT32_MAX
        OP_TILING_CHECK(
            curDim <= 0 || curDim > static_cast<int64_t>(INT32_MAX),
            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                  "Input shape must be within the range of [1,%d], actual %zu dimension of %s is %ld.",
                                  INT32_MAX, i, dimName, curDim),
            return false);
        // x1或x2的shape总元素数不超过INT64_MAX
        mulBound = curDim * mul;
        OP_TILING_CHECK(mulBound / curDim != mul,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "Multiple of %s shape dims should be in boundary of INT64_MAX.", dimName),
                        return false);
        mul = mulBound;
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::ExtraInputCheck() const
{
    // 当x1为INT4, transA必须为false
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT4 && inputParams_.transA,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "When input dtype is INT4, trans_a should be false, actual [%s]",
                              inputParams_.transA ? "true" : "false"),
        return false);

    // x1必须为ND
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    OP_TILING_CHECK(x1Format != ge::Format::FORMAT_ND,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input x1 format should be ND, actual is %s.",
                                          ge::TypeUtils::FormatToSerialString(x1Format).c_str()),
                    return false);
    // 当x1为INT4时，x2必须是NZ
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT4 && x2Format != ge::Format::FORMAT_FRACTAL_NZ,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "When input x1 dtype is INT4, x2 format should be FRACTAL_NZ, actual is %s.",
                                          ge::TypeUtils::FormatToSerialString(x2Format).c_str()),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckDimValue(const gert::Shape &scaleShape,
                                                       const gert::StorageShape *biasShape,
                                                       const gert::StorageShape *offsetShape,
                                                       const std::vector<int64_t> &dimValueOfMKN) const
{
    // x1,x2 shapeK必须相等
    auto x1Inner = dimValueOfMKN[0];  // using index 0 to get x1Inner
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
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "The offset dimension value must be 1 or n[%lu], but it is %ld.",
                              inputParams_.nSize, offsetShape->GetStorageShape().GetDim(0)),
        return false);
    // scale维数必须是1维
    OP_TILING_CHECK(scaleShape.GetDimNum() != 1,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Scale's dimension must be 1, actually is : %zu",
                                          scaleShape.GetDimNum()),
                    return false);
    // 当x1为INT8时，支持pertensor和perchannel量化模式
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT8 && !(inputParams_.isPerTensor || inputParams_.isPerChannel),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "When x1 dtype is INT8, the only supported quant modes are Pertensor/PerChannel"),
        return false);
    // 当x1为INT4时，支持perchannel量化模式
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT4 && !inputParams_.isPerChannel,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "When x1 dtype is INT4, the only supported quant modes are PerChannel"),
                    return false);
    // 当x1为int4时，x1内轴必须是偶数
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT4 && (x1Inner < 0 || x1Inner % 2 != 0),
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "if x1 dtype is int4, last axis of input x1 has to be a positive even number, but actual is [%ld].",
            x1Inner),
        return false);
    // 当x2为int4时，x2内轴必须是偶数
    OP_TILING_CHECK(
        inputParams_.bDtype == ge::DT_INT4 && (x2Inner < 0 || x2Inner % 2 != 0),
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "if x2 dtype is int4, last axis of input x2 has to be a positive even number, but actual is [%ld].",
            x2Inner),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckShape(
    const std::vector<gert::Shape*>& mandtoryShape, const gert::StorageShape* biasShape,
    const gert::StorageShape* /* pertokenShape */, const std::vector<int64_t>& dimValueOfMKN) const
{
    auto &x1Shape = *mandtoryShape[0];     // using index 0 to get x1Shape
    auto &x2Shape = *mandtoryShape[1];     // using index 1 to get x2Shape
    auto &scaleShape = *mandtoryShape[2];  // using index 2 to get scaleShape
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