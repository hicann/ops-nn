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
 * \file quant_batch_matmul_v3_checker.cpp
 * \brief
 */

#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "quant_batch_matmul_v3_checker.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include "error_util.h"

namespace {
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t SCALE_THREE_DIM = 3;
constexpr size_t MX_SCALE_MIN_DIM_WITH_BATCH = 3;
constexpr size_t BIAS_THREE_DIM = 3;
constexpr size_t X1_INNER_IDX = 0;
constexpr size_t X1_OUTER_IDX = 1;
constexpr size_t X2_INNER_IDX = 2;
constexpr size_t X2_OUTER_IDX = 3;

inline constexpr bool IsLowFloatInputType(ge::DataType dtype)
{
    return dtype == ge::DT_HIFLOAT8 || dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2 ||
           dtype == ge::DT_FLOAT4_E2M1;
}

inline constexpr bool IsFp8E4M3Pair(ge::DataType aDtype, ge::DataType bDtype)
{
    return aDtype == ge::DT_FLOAT8_E4M3FN && bDtype == ge::DT_FLOAT8_E4M3FN;
}

inline constexpr bool IsHifloat8Pair(ge::DataType aDtype, ge::DataType bDtype)
{
    return aDtype == ge::DT_HIFLOAT8 && bDtype == ge::DT_HIFLOAT8;
}

inline constexpr bool IsFp4Pair(ge::DataType aDtype, ge::DataType bDtype)
{
    return aDtype == ge::DT_FLOAT4_E2M1 && bDtype == ge::DT_FLOAT4_E2M1;
}

const std::vector<ge::DataType> AB_DTYPE_LIST = {ge::DT_INT4, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2,
                                                 ge::DT_FLOAT8_E4M3FN};
} // namespace

namespace optiling {

using Ops::NN::FormatString;

bool QuantBatchMatmulV3Checker::LogicXOR(bool cond1, bool cond2) const
{
    uint64_t result = static_cast<uint64_t>(cond1) ^ static_cast<uint64_t>(cond2);
    return static_cast<bool>(result);
}

bool QuantBatchMatmulV3Checker::CheckABDtypesSame() const
{
    OP_TILING_CHECK(LogicXOR(inputParams_.aDtype == ge::DT_INT4, inputParams_.bDtype == ge::DT_INT4),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opName, "x1, x2",
                        FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                     ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str())
                            .c_str(),
                        "when the dtype of one input is INT4, the dtype of the other input must be INT4"),
                    return false);
    OP_TILING_CHECK(LogicXOR(inputParams_.aDtype == ge::DT_INT8, inputParams_.bDtype == ge::DT_INT8),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opName, "x1, x2",
                        FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                     ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str())
                            .c_str(),
                        "when the dtype of one input is INT8, the dtype of the other input must be INT8"),
                    return false);
    OP_TILING_CHECK(LogicXOR(inputParams_.aDtype == ge::DT_HIFLOAT8, inputParams_.bDtype == ge::DT_HIFLOAT8),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opName, "x1, x2",
                        FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                     ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str())
                            .c_str(),
                        "when the dtype of one input is HIFLOAT8, the dtype of the other input must be HIFLOAT8"),
                    return false);
    OP_TILING_CHECK(LogicXOR(inputParams_.aDtype == ge::DT_FLOAT4_E2M1, inputParams_.bDtype == ge::DT_FLOAT4_E2M1),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opName, "x1, x2",
                        FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                     ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str())
                            .c_str(),
                        "when the dtype of one input is FLOAT4, the dtype of the other input must be FLOAT4"),
                    return false);
    OP_TILING_CHECK(
        LogicXOR((inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2),
                 (inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.bDtype == ge::DT_FLOAT8_E5M2)),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "x1, x2",
            FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                         ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str())
                .c_str(),
            "when the dtype of one input is FLOAT8, the dtype of the other input must be FLOAT8"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckABDtypes() const
{
    if (!CheckABDtypesSame()) {
        return false;
    }
    if (context_->GetOptionalInputDesc(GetPertokenIdx()) == nullptr ||
        context_->GetOptionalInputShape(GetPertokenIdx()) == nullptr) {
        OP_TILING_CHECK(
            std::find(AB_DTYPE_LIST.begin(), AB_DTYPE_LIST.end(), inputParams_.aDtype) == AB_DTYPE_LIST.end() ||
                std::find(AB_DTYPE_LIST.begin(), AB_DTYPE_LIST.end(), inputParams_.bDtype) == AB_DTYPE_LIST.end(),
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                inputParams_.opName, "x1, x2",
                FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                             ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str())
                    .c_str(),
                "the dtypes of x1 and x2 must be INT4/INT8/FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker::CheckScaleDtypeWithPertoken() const
{
    bool isFp4 = inputParams_.aDtype == ge::DT_FLOAT4_E2M1;
    bool isFp8 = inputParams_.aDtype == ge::DT_FLOAT8_E5M2 || inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN;
    OP_TILING_CHECK(
        inputParams_.aDtype != ge::DT_INT4 && inputParams_.aDtype != ge::DT_INT8 &&
            inputParams_.perTokenScaleDtype != inputParams_.scaleDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "pertokenScale, scale",
            FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                         ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str())
                .c_str(),
            "when the dtype of input is not INT4/INT8, the dtypes of pertokenScale and scale must be the same"),
        return false);
    OP_TILING_CHECK(
        isFp4 && inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "scale", ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
            "when the dtype of input is FLOAT4 and pertokenScale exists, the dtype of scale must be FLOAT8_E8M0"),
        return false);
    OP_TILING_CHECK(
        isFp8 && (inputParams_.scaleDtype != ge::DT_FLOAT && inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                                              "when the dtype of input is FLOAT8 and pertokenScale exists, the dtype "
                                              "of scale must be FLOAT or FLOAT8_E8M0"),
        return false);
    OP_TILING_CHECK(!(isFp4 || isFp8) && inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "pertokenScale",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                        "when the dtype of input is INT8/HIFLOAT8, the dtype of pertokenScale must be FLOAT"),
                    return false);

    OP_TILING_CHECK(
        (inputParams_.aDtype == ge::DT_INT4 || inputParams_.aDtype == ge::DT_INT8) &&
            inputParams_.cDtype == ge::DT_FLOAT16 && inputParams_.scaleDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                                              "when the dtype of input is INT4/INT8, the dtype of output is FLOAT16, "
                                              "and pertokenScale exists, the dtype of scale must be FLOAT"),
        return false);
    OP_TILING_CHECK(
        (inputParams_.aDtype == ge::DT_INT4 || inputParams_.aDtype == ge::DT_INT8) &&
            inputParams_.cDtype == ge::DT_BF16 &&
            !(inputParams_.scaleDtype == ge::DT_FLOAT || inputParams_.scaleDtype == ge::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                                              "when the dtype of input is INT4/INT8, the dtype of output is BFLOAT16, "
                                              "and pertokenScale exists, the dtype of scale must be FLOAT/BFLOAT16"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckScalesDtype() const
{
    bool isFp4 = inputParams_.aDtype == ge::DT_FLOAT4_E2M1;
    if (context_->GetOptionalInputDesc(GetPertokenIdx()) != nullptr &&
        context_->GetOptionalInputShape(GetPertokenIdx()) != nullptr) {
        OP_TILING_CHECK(!CheckScaleDtypeWithPertoken(),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Failed to check scale dtype with pertokenScale."),
                        return false);
    } else {
        OP_TILING_CHECK(
            isFp4,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "pertokenScale", "null",
                                                  "when the dtype of input is FLOAT4, pertokenScale can not be null"),
            return false);
        OP_TILING_CHECK(
            inputParams_.aDtype != ge::DT_INT8 &&
                !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "scale", ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                "when the dtype of input is not INT8 and pertokenScale does not exist, the dtype of scale must be "
                "UINT64/INT64"),
            return false);
        OP_TILING_CHECK(
            inputParams_.aDtype == ge::DT_INT8 &&
                (inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16) &&
                !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "scale", ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                "when the dtype of input is INT8, the dtype of output is INT8 or FLOAT16, and pertokenScale does not "
                "exist, the dtype of scale must be UINT64/INT64"),
            return false);
        OP_TILING_CHECK(
            inputParams_.aDtype == ge::DT_INT8 && inputParams_.cDtype == ge::DT_BF16 &&
                !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_FLOAT ||
                  inputParams_.scaleDtype == ge::DT_BF16 || inputParams_.scaleDtype == ge::DT_INT64),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "scale", ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                "when the dtype of input is INT8, the dtype of output is BFLOAT16, and pertokenScale does not exist, "
                "the dtype of scale must be UINT64/FLOAT/BFLOAT16/INT64"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker::CheckBiasDtype() const
{
    auto biasDesc = context_->GetOptionalInputDesc(GetBiasIdx());
    OP_TILING_CHECK(
        biasDesc != nullptr && inputParams_.biasDtype != ge::DT_FLOAT && inputParams_.aDtype != ge::DT_INT4 &&
            inputParams_.aDtype != ge::DT_INT8,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "bias", ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
            "when the dtype of input is not INT4/INT8, the dtype of bias must be FLOAT"),
        return false);

    OP_TILING_CHECK(
        biasDesc != nullptr && (inputParams_.aDtype == ge::DT_INT4 || inputParams_.aDtype == ge::DT_INT8) &&
            (inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16 ||
             inputParams_.cDtype == ge::DT_BF16) &&
            (inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64) &&
            inputParams_.biasDtype != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "bias", ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
            "when the dtype of input is INT4/INT8, the dtype of output is INT8/FLOAT16/BFLOAT16, and the dtype of "
            "scale is UINT64/INT64, the dtype of bias must be INT32"),
        return false);

    OP_TILING_CHECK(
        biasDesc != nullptr && (inputParams_.aDtype == ge::DT_INT4 || inputParams_.aDtype == ge::DT_INT8) &&
            inputParams_.cDtype == ge::DT_FLOAT16 && inputParams_.scaleDtype == ge::DT_FLOAT &&
            !(inputParams_.biasDtype == ge::DT_INT32 || inputParams_.biasDtype == ge::DT_FLOAT ||
              inputParams_.biasDtype == ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "bias", ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
            "when the dtype of input is INT4/INT8, the dtype of output is FLOAT16, and the dtype of scale is FLOAT, "
            "the dtype of bias must be INT32/FLOAT/FLOAT16"),
        return false);

    OP_TILING_CHECK(
        biasDesc != nullptr && (inputParams_.aDtype == ge::DT_INT4 || inputParams_.aDtype == ge::DT_INT8) &&
            inputParams_.cDtype == ge::DT_BF16 &&
            (inputParams_.scaleDtype == ge::DT_FLOAT || inputParams_.scaleDtype == ge::DT_BF16) &&
            !(inputParams_.biasDtype == ge::DT_INT32 || inputParams_.biasDtype == ge::DT_FLOAT ||
              inputParams_.biasDtype == ge::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "bias", ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
            "when the dtype of input is INT4/INT8, the dtype of output is BFLOAT16, and the dtype of scale is "
            "FLOAT/BFLOAT16, the dtype of bias must be INT32/FLOAT/BFLOAT16"),
        return false);

    return true;
}

bool QuantBatchMatmulV3Checker::CheckOutputDtype() const
{
    OP_TILING_CHECK(
        (inputParams_.aDtype == ge::DT_FLOAT8_E5M2 || inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN ||
         inputParams_.aDtype == ge::DT_FLOAT8_E8M0 || inputParams_.aDtype == ge::DT_FLOAT4_E2M1) &&
            inputParams_.cDtype == ge::DT_INT8,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "output", ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
            "when the dtype of input is FLOAT8/FLOAT4, the dtype of output can not be INT8"),
        return false);
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_HIFLOAT8 && inputParams_.cDtype == ge::DT_INT8,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "output", ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
            "when the dtype of input is HIFLOAT8, the dtype of output can not be INT8"),
        return false);
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT4 && inputParams_.cDtype == ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                              "when the dtype of input is INT4, the dtype of output can not be INT32"),
        return false);
    bool isFp4 = inputParams_.aDtype == ge::DT_FLOAT4_E2M1;
    bool isFp8 = inputParams_.aDtype == ge::DT_HIFLOAT8 || inputParams_.aDtype == ge::DT_FLOAT8_E5M2 ||
                 inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E8M0;
    if (context_->GetOptionalInputDesc(GetPertokenIdx()) != nullptr &&
        context_->GetOptionalInputShape(GetPertokenIdx()) != nullptr) {
        OP_TILING_CHECK(
            (isFp8 || isFp4) && !(inputParams_.cDtype == ge::DT_FLOAT || inputParams_.cDtype == ge::DT_FLOAT16 ||
                                  inputParams_.cDtype == ge::DT_BF16),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                                  "when the dtype of input is FLOAT8/FLOAT4 and pertokenScale exists, "
                                                  "the dtype of output must be FLOAT/FLOAT16/BFLOAT16"),
            return false);
        OP_TILING_CHECK(
            !(isFp8 || isFp4) && !(inputParams_.cDtype == ge::DT_FLOAT16 || inputParams_.cDtype == ge::DT_BF16),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                                  "when the dtype of input is INT4/INT8 and pertokenScale exists, the "
                                                  "dtype of output must be FLOAT16/BFLOAT16"),
            return false);
        OP_TILING_CHECK(
            !(inputParams_.cDtype == ge::DT_FLOAT || inputParams_.cDtype == ge::DT_FLOAT16 ||
              inputParams_.cDtype == ge::DT_BF16),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "output", ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                "when pertokenScale exists, the dtype of output must be FLOAT/FLOAT16/BFLOAT16"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker::CheckDtypesInRange() const
{
    static const std::vector<ge::DataType> legalInputDtypes = {
        ge::DT_INT4, ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8, ge::DT_FLOAT4_E2M1};
    static const std::vector<ge::DataType> legalOutputDtypes = {ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                                                                ge::DT_INT32};
    OP_TILING_CHECK(
        std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.aDtype) == legalInputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x1", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            "the dtype of x1 must be INT4/INT8/FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1"),
        return false);
    OP_TILING_CHECK(
        std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.bDtype) == legalInputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x2", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
            "the dtype of x2 must be INT4/INT8/FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1"),
        return false);
    OP_TILING_CHECK(
        std::find(legalOutputDtypes.begin(), legalOutputDtypes.end(), inputParams_.cDtype) == legalOutputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                              "the dtype of output must be INT8/FLOAT16/BFLOAT16/FLOAT/INT32"),
        return false);
    OP_TILING_CHECK(inputParams_.cDtype == ge::DT_INT32 &&
                        (inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opName, "x1, x2",
                        FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                     ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str())
                            .c_str(),
                        "when the dtype of output is INT32, the dtypes of x1 and x2 must be INT8"),
                    return false);
    auto offsetDesc = context_->GetOptionalInputDesc(GetOffsetIdx());
    OP_TILING_CHECK(
        offsetDesc != nullptr && offsetDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "offset",
                                              ge::TypeUtils::DataTypeToSerialString(offsetDesc->GetDataType()).c_str(),
                                              "the dtype of offset must be FLOAT"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckWeightNzAbDtypes() const
{
    bool isInt8Pair = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8;
    bool isHifloat8Pair = IsHifloat8Pair(inputParams_.aDtype, inputParams_.bDtype);
    bool isFp8E4M3Pair = IsFp8E4M3Pair(inputParams_.aDtype, inputParams_.bDtype);
    bool isFp4Pair = IsFp4Pair(inputParams_.aDtype, inputParams_.bDtype);

    OP_TILING_CHECK(!(isInt8Pair || isHifloat8Pair || isFp8E4M3Pair || isFp4Pair),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opName, "x1, x2",
                        FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                     ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str())
                            .c_str(),
                        "when the format of x2 is FRACTAL_NZ, the dtypes of x1 and x2 must be "
                        "INT8/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1"),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckWeightNzDtype4Hifloat8() const
{
    bool hasPerTokenScale = context_->GetOptionalInputDesc(GetPertokenIdx()) != nullptr &&
                            context_->GetOptionalInputShape(GetPertokenIdx()) != nullptr;
    bool is64BitScale = inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64;
    // 静态 x2 scale 路径：pertokenScale 为 null 且 scale 为 UINT64/INT64
    bool isStaticX2Scale = !hasPerTokenScale && is64BitScale;
    // FP8-Pertile/MIX WeightNz(HIFLOAT8)：scale 与 pertokenScale 均为 FLOAT
    bool isFloatScale = hasPerTokenScale && inputParams_.scaleDtype == ge::DT_FLOAT &&
                        inputParams_.perTokenScaleDtype == ge::DT_FLOAT;

    OP_TILING_CHECK(
        !(isStaticX2Scale || isFloatScale),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "pertokenScale, scale",
            FormatString("%s, %s",
                         hasPerTokenScale ?
                             ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str() :
                             "null",
                         ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str())
                .c_str(),
            "when the format of x2 is FRACTAL_NZ and input dtype is HIFLOAT8, pertokenScale and scale must both "
            "be FLOAT, or pertokenScale must be null and scale must be UINT64/INT64"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckWeightNzDtype4Fp4() const
{
    bool hasPerTokenScale = context_->GetOptionalInputDesc(GetPertokenIdx()) != nullptr &&
                            context_->GetOptionalInputShape(GetPertokenIdx()) != nullptr;
    bool isE8M0ScalePair = inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 &&
                           inputParams_.perTokenScaleDtype == ge::DT_FLOAT8_E8M0;

    OP_TILING_CHECK(
        !(hasPerTokenScale && isE8M0ScalePair),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "pertokenScale, scale",
            FormatString("%s, %s",
                         hasPerTokenScale ?
                             ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str() :
                             "null",
                         ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str())
                .c_str(),
            "when the format of x2 is FRACTAL_NZ and input dtype is FLOAT4_E2M1, pertokenScale and scale must both "
            "be provided and be FLOAT8_E8M0"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckWeightNzDtype4Fp8E4M3() const
{
    bool hasPerTokenScale = context_->GetOptionalInputDesc(GetPertokenIdx()) != nullptr &&
                            context_->GetOptionalInputShape(GetPertokenIdx()) != nullptr;
    bool isE8M0ScalePair = inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 &&
                           inputParams_.perTokenScaleDtype == ge::DT_FLOAT8_E8M0;
    bool is64BitScale = inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64;
    bool isStaticX2Scale = !hasPerTokenScale && is64BitScale;
    bool isMxScale = hasPerTokenScale && isE8M0ScalePair;
    // FP8-Pretile/MIX WeightNz(only for FLOAT8_E4M3FN): scale and pertokenScale are all FLOAT
    bool isFloatScale = hasPerTokenScale && inputParams_.scaleDtype == ge::DT_FLOAT &&
                        inputParams_.perTokenScaleDtype == ge::DT_FLOAT;

    OP_TILING_CHECK(
        !(isMxScale || isStaticX2Scale || isFloatScale),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "pertokenScale, scale",
            FormatString("%s, %s",
                         hasPerTokenScale ?
                             ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str() :
                             "null",
                         ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str())
                .c_str(),
            "when the format of x2 is FRACTAL_NZ and input dtype is FLOAT8_E4M3FN, pertokenScale and scale must both "
            "be FLOAT8_E8M0, or both be FLOAT, or pertokenScale must be null and scale must be UINT64/INT64"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckDtype4WeightNz() const
{
    if (!CheckWeightNzAbDtypes()) {
        return false;
    }

    if (inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8) {
        return true;
    }

    if (IsHifloat8Pair(inputParams_.aDtype, inputParams_.bDtype)) {
        return CheckWeightNzDtype4Hifloat8();
    }

    if (IsFp4Pair(inputParams_.aDtype, inputParams_.bDtype)) {
        return CheckWeightNzDtype4Fp4();
    }

    return CheckWeightNzDtype4Fp8E4M3();
}

bool QuantBatchMatmulV3Checker::CheckDtype() const
{
    if (!CheckDtypesInRange()) {
        return false;
    }
    if (!CheckABDtypes()) {
        return false;
    }
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && !CheckDtype4WeightNz()) {
        return false;
    }
    if (!CheckScalesDtype()) {
        return false;
    }
    if (!CheckBiasDtype()) {
        return false;
    }
    if (!CheckOutputDtype()) {
        return false;
    }
    return true;
}

bool QuantBatchMatmulV3Checker::CheckInputValidInPerblockMode(const gert::Shape& scaleShape,
                                                              const gert::StorageShape* pertokenShape,
                                                              const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    if (!inputParams_.isPerBlock) {
        return true;
    }
    OP_TILING_CHECK(
        inputParams_.hasBias,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "bias", "not null",
                                              "when the quantization mode is G-B or B-B, bias must be null"),
        return false);
    auto scaleShapeLen = scaleShape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto& pertoken = pertokenShape->GetStorageShape();
    auto pertokenShapeLen = pertoken.GetDimNum();
    auto x1ShapeLen = x1Shape.GetDimNum();
    if (!CheckDimValidInPerblockMode(x1ShapeLen, x2ShapeLen, pertokenShapeLen, scaleShapeLen)) {
        return false;
    }
    if (!CheckBatchValidInPerblockMode(scaleShape, pertoken, x1Shape, x2Shape)) {
        return false;
    }

    if (!CheckGroupValidInPerblockMode()) {
        return false;
    }

    if (!CheckShapeValidInPerblockMode(scaleShape, pertoken, x1Shape, x2Shape)) {
        return false;
    }

    return true;
}

bool QuantBatchMatmulV3Checker::CheckGroupValidInPerblockMode() const
{
    OP_TILING_CHECK(inputParams_.groupSizeM != qmmv3_tiling_const::PER_BLOCK_SIZE && inputParams_.groupSizeM != 1UL,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "groupSizeM", std::to_string(inputParams_.groupSizeM).c_str(),
                        "when the quantization mode is G-B or B-B, input or inferred groupSizeM must be 128 or 1, "
                        "groupSizeM = (groupSize >> 32) & 0xFFFF"),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeK != qmmv3_tiling_const::PER_BLOCK_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSizeK",
                                                          std::to_string(inputParams_.groupSizeK).c_str(),
                                                          "when the quantization mode is G-B or B-B, input or inferred "
                                                          "groupSizeK must be 128, groupSizeK = groupSize & 0xFFFF"),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeN != qmmv3_tiling_const::PER_BLOCK_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "groupSizeN", std::to_string(inputParams_.groupSizeN).c_str(),
                        "when the quantization mode is G-B or B-B, input or inferred groupSizeN must be 128, "
                        "groupSizeN = (groupSize >> 16) & 0xFFFF"),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckShapeValidInPerblockMode(const gert::Shape& scaleShape,
                                                              const gert::Shape& pertoken, const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    // [transpose_x2 fix] x2Scale 的最后两维应等于 x2 最后两维各自 ceildiv(128)。transpose_x2=True 时
    // x2Scale 的排布可能相对 x2 互换(取决于上游是否转置 scale)，故这里接受"原始"与"互换"两种匹配。
    auto scaleShapeLen = scaleShape.GetDimNum();
    uint64_t cdN = ops::CeilDiv(static_cast<uint64_t>(x2Shape.GetDim(x2ShapeLen - 2)),
                                qmmv3_tiling_const::PER_BLOCK_SIZE);
    uint64_t cdK = ops::CeilDiv(static_cast<uint64_t>(x2Shape.GetDim(x2ShapeLen - 1)),
                                qmmv3_tiling_const::PER_BLOCK_SIZE);
    uint64_t scOuter = static_cast<uint64_t>(scaleShape.GetDim(scaleShapeLen - 2));
    uint64_t scInner = static_cast<uint64_t>(scaleShape.GetDim(scaleShapeLen - 1));
    bool matchNoTrans = (cdN == scOuter && cdK == scInner);
    bool matchTrans = (cdN == scInner && cdK == scOuter);
    OP_TILING_CHECK(
        !(matchNoTrans || matchTrans),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            inputParams_.opName, "scale, x2",
            FormatString("[%ld, %ld], [%ld, %ld]", scaleShape.GetDim(scaleShapeLen - 2),
                         scaleShape.GetDim(scaleShapeLen - 1), x2Shape.GetDim(x2ShapeLen - 2),
                         x2Shape.GetDim(x2ShapeLen - 1))
                .c_str(),
            "when the quantization mode is G-B or B-B, the last two dimensions of scale must be equal to the last two "
            "dimensions of x2 ceildivided by 128"),
        return false);
    int64_t x1MIndex = inputParams_.transA ? (x1ShapeLen - 1) : (x1ShapeLen - 2);
    int64_t x1KIndex = inputParams_.transA ? (x1ShapeLen - 2) : (x1ShapeLen - 1);
    uint64_t x1M = x1Shape.GetDim(x1MIndex);
    uint64_t scaleX1M = pertoken.GetDim(x1MIndex);
    uint64_t x1K = x1Shape.GetDim(x1KIndex);
    uint64_t scaleX1K = pertoken.GetDim(x1KIndex);
    OP_TILING_CHECK((ops::CeilDiv(x1M, inputParams_.groupSizeM) != scaleX1M),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "groupSizeM, pertokenScaleM, x1M",
                        FormatString("%lu, %lu, %lu", inputParams_.groupSizeM, scaleX1M, x1M).c_str(),
                        "when the quantization mode is G-B or B-B, the m dimension size of pertokenScale must be equal "
                        "to the m dimension size of x1 ceildivided by groupSizeM"),
                    return false);
    OP_TILING_CHECK((ops::CeilDiv(x1K, inputParams_.groupSizeK) != scaleX1K),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "groupSizeK, pertokenScaleK, x1K",
                        FormatString("%lu, %lu, %lu", inputParams_.groupSizeK, scaleX1K, x1K).c_str(),
                        "when the quantization mode is G-B or B-B, the k dimension size of pertokenScale must be equal "
                        "to the k dimension size of x1 ceildivided by groupSizeK"),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckDimValidInPerblockMode(size_t x1ShapeLen, size_t x2ShapeLen,
                                                            size_t pertokenShapeLen, size_t scaleShapeLen) const
{
    OP_TILING_CHECK(
        scaleShapeLen != x2ShapeLen,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            inputParams_.opName, "x2, scale", FormatString("%zuD, %zuD", x2ShapeLen, scaleShapeLen).c_str(),
            "when the quantization mode is G-B or B-B, the dimension of scale must be equal to the dimension of x2"),
        return false);
    OP_TILING_CHECK(
        pertokenShapeLen != x1ShapeLen,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(inputParams_.opName, "x1, pertokenScale",
                                                  FormatString("%zuD, %zuD", x1ShapeLen, pertokenShapeLen).c_str(),
                                                  "when the quantization mode is G-B or B-B, the dimension of "
                                                  "pertokenScale must be equal to the dimension of x1"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckBatchValidInPerblockMode(const gert::Shape& scaleShape,
                                                              const gert::Shape& pertoken, const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto x1ShapeLen = x1Shape.GetDimNum();
    if (x2ShapeLen > DIM_NUM_TWO) {
        for (size_t i = 0; i < x2ShapeLen - DIM_NUM_TWO; ++i) {
            OP_TILING_CHECK(scaleShape.GetDim(i) != x2Shape.GetDim(i),
                            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                                inputParams_.opName, "dimIndex, x2Batch, scaleBatch",
                                FormatString("%zu, %ld, %ld", i, x2Shape.GetDim(i), scaleShape.GetDim(i)).c_str(),
                                "when the quantization mode is G-B or B-B, the batch dimension of scale must be equal "
                                "to the batch dimension of x2"),
                            return false);
        }
    }
    if (x1ShapeLen > DIM_NUM_TWO) {
        for (size_t i = 0; i < x1ShapeLen - DIM_NUM_TWO; ++i) {
            OP_TILING_CHECK(pertoken.GetDim(i) != x1Shape.GetDim(i),
                            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                                inputParams_.opName, "dimIndex, x1Batch, pertokenBatch",
                                FormatString("%zu, %ld, %ld", i, x1Shape.GetDim(i), pertoken.GetDim(i)).c_str(),
                                "when the quantization mode is G-B or B-B, the batch dimension of pertokenScale must "
                                "be equal to the batch dimension of x1"),
                            return false);
        }
    }
    return true;
}

bool QuantBatchMatmulV3Checker::CheckBatchValidInMxPerGroupMode(const gert::Shape& scaleShape,
                                                                const gert::Shape& pertoken, const gert::Shape& x1Shape,
                                                                const gert::Shape& x2Shape) const
{
    auto perGroupX1ShapeLen = x1Shape.GetDimNum();
    auto perGroupX2ShapeLen = x2Shape.GetDimNum();
    auto perGroupX2ScaleBatchLen = scaleShape.GetDimNum() - SCALE_THREE_DIM;
    auto perGroupX1ScaleBatchLen = pertoken.GetDimNum() - SCALE_THREE_DIM;
    auto perGroupX1BatchLen = (perGroupX1ShapeLen > DIM_NUM_TWO) ? (perGroupX1ShapeLen - DIM_NUM_TWO) : 0;
    auto perGroupX2BatchLen = (perGroupX2ShapeLen > DIM_NUM_TWO) ? (perGroupX2ShapeLen - DIM_NUM_TWO) : 0;
    OP_TILING_CHECK(
        perGroupX2ScaleBatchLen != perGroupX2BatchLen,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "scaleBatchDimNum, x2BatchDimNum",
            FormatString("%zu, %zu", perGroupX2ScaleBatchLen, perGroupX2BatchLen).c_str(),
            "when the quantization mode is mx, the batch dimension num of scale must be equal to that of x2"),
        return false);
    OP_TILING_CHECK(
        perGroupX1ScaleBatchLen != perGroupX1BatchLen,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "pertokenBatchDimNum, x1BatchDimNum",
            FormatString("%zu, %zu", perGroupX1ScaleBatchLen, perGroupX1BatchLen).c_str(),
            "when the quantization mode is mx, the batch dimension num of pertokenScale must be equal to that of x1"),
        return false);
    for (size_t i = 0; i < perGroupX2BatchLen; ++i) {
        OP_TILING_CHECK(
            scaleShape.GetDim(i) != x2Shape.GetDim(i),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                inputParams_.opName, "dimIndex, x2Batch, scaleBatch",
                FormatString("%zu, %ld, %ld", i, x2Shape.GetDim(i), scaleShape.GetDim(i)).c_str(),
                "when the quantization mode is mx, the batch dimension of scale must be equal to that of x2"),
            return false);
    }
    for (size_t i = 0; i < perGroupX1BatchLen; ++i) {
        OP_TILING_CHECK(
            pertoken.GetDim(i) != x1Shape.GetDim(i),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                inputParams_.opName, "dimIndex, x1Batch, pertokenBatch",
                FormatString("%zu, %ld, %ld", i, x1Shape.GetDim(i), pertoken.GetDim(i)).c_str(),
                "when the quantization mode is mx, the batch dimension of pertokenScale must be equal to that of x1"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker::MxPertokenScaleShapeCheck(const gert::StorageShape* pertokenShape) const
{
    auto& pertoken = pertokenShape->GetStorageShape();
    auto pertokenShapeLen = pertoken.GetDimNum();
    OP_TILING_CHECK(pertokenShapeLen < MX_SCALE_MIN_DIM_WITH_BATCH,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        inputParams_.opName, "pertokenScale", FormatString("%zuD", pertokenShapeLen).c_str(),
                        FormatString("when the quantization mode is mx, the shape dim of pertokenScale must be at "
                                     "least %zuD",
                                     MX_SCALE_MIN_DIM_WITH_BATCH)
                            .c_str()),
                    return false);
    size_t batchDimNum = pertokenShapeLen - SCALE_THREE_DIM;
    auto mDimIdx = batchDimNum + (inputParams_.transA ? 1 : 0);
    auto kDimIdx = batchDimNum + (inputParams_.transA ? 0 : 1);
    OP_TILING_CHECK(static_cast<uint64_t>(pertoken.GetDim(kDimIdx)) !=
                        ops::CeilDiv(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "x1K, pertokenScaleK",
                        FormatString("%lu, %ld", inputParams_.kSize, pertoken.GetDim(kDimIdx)).c_str(),
                        "when the quantization mode is mx, the k dimension of pertokenScale must be equal to the k "
                        "dimension size of x1 ceildivided by 64"),
                    return false);
    OP_TILING_CHECK(static_cast<uint64_t>(pertoken.GetDim(mDimIdx)) != inputParams_.mSize,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "x1M, pertokenScaleM",
                        FormatString("%lu, %ld", inputParams_.mSize, pertoken.GetDim(mDimIdx)).c_str(),
                        "when the quantization mode is mx, the m dimension of pertokenScale must be equal to the m "
                        "dimension size of x1"),
                    return false);
    OP_TILING_CHECK(
        static_cast<uint64_t>(pertoken.GetDim(pertokenShapeLen - 1)) != qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams_.opName, "pertokenScale last dimension",
            std::to_string(pertoken.GetDim(pertokenShapeLen - 1)).c_str(),
            "when the quantization mode is mx, the last dimension of pertokenScale must be equal to 2"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::MxScaleShapeCheck(const gert::Shape& scaleShape) const
{
    auto scaleShapeLen = scaleShape.GetDimNum();
    OP_TILING_CHECK(scaleShapeLen < MX_SCALE_MIN_DIM_WITH_BATCH,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        inputParams_.opName, "scale", FormatString("%zuD", scaleShapeLen).c_str(),
                        FormatString("when the quantization mode is mx, the shape dim of scale must be at least %zuD",
                                     MX_SCALE_MIN_DIM_WITH_BATCH)
                            .c_str()),
                    return false);
    size_t batchDimNum = scaleShapeLen - SCALE_THREE_DIM;
    auto kDimIdx = batchDimNum + (inputParams_.transB ? 1 : 0);
    auto nDimIdx = batchDimNum + (inputParams_.transB ? 0 : 1);
    bool nDimHasOne = (scaleShape.GetDim(kDimIdx) == 1 &&
                       static_cast<uint64_t>(scaleShape.GetDim(nDimIdx)) ==
                           ops::CeilDiv(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE)) ||
                      (scaleShape.GetDim(nDimIdx) == 1 &&
                       static_cast<uint64_t>(scaleShape.GetDim(kDimIdx)) ==
                           ops::CeilDiv(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE));
    bool kDimHasOne = (scaleShape.GetDim(kDimIdx) == 1 &&
                       static_cast<uint64_t>(scaleShape.GetDim(nDimIdx)) == inputParams_.nSize) ||
                      (scaleShape.GetDim(nDimIdx) == 1 &&
                       static_cast<uint64_t>(scaleShape.GetDim(kDimIdx)) == inputParams_.nSize);
    if (!nDimHasOne && !kDimHasOne) {
        OP_TILING_CHECK(static_cast<uint64_t>(scaleShape.GetDim(kDimIdx)) !=
                            ops::CeilDiv(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE),
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            inputParams_.opName, "x2K, scaleK",
                            FormatString("%lu, %ld", inputParams_.kSize, scaleShape.GetDim(kDimIdx)).c_str(),
                            "when the quantization mode is mx, the k dimension of scale must be equal to the k "
                            "dimension size of x2 ceildivided by 64"),
                        return false);
        OP_TILING_CHECK(static_cast<uint64_t>(scaleShape.GetDim(nDimIdx)) != inputParams_.nSize,
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            inputParams_.opName, "x2N, scaleN",
                            FormatString("%lu, %ld", inputParams_.nSize, scaleShape.GetDim(nDimIdx)).c_str(),
                            "when the quantization mode is mx, the n dimension of scale must be equal to the n "
                            "dimension size of x2"),
                        return false);
    }
    OP_TILING_CHECK(
        static_cast<uint64_t>(scaleShape.GetDim(scaleShapeLen - 1)) != qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams_.opName, "scale dimension 2", std::to_string(scaleShape.GetDim(scaleShapeLen - 1)).c_str(),
            "when the quantization mode is mx, dimension 2 of scale must be equal to 2"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckKAxisGreaterThanTwo() const
{
    if (inputParams_.kSize <= 2UL) { // FLOAT4 input requires K to be greater than 2.
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams_.opName, "k", std::to_string(inputParams_.kSize).c_str(),
            "when the dtype of input is FLOAT4_E2M1, the K dimension must be greater than 2");
        return false;
    }
    return true;
}

bool QuantBatchMatmulV3Checker::CheckInnerAxisIsEven(const std::vector<int64_t>& dimValueOfMKN) const
{
    // Both inner axes must be even for FLOAT4 packing.
    OP_TILING_CHECK(dimValueOfMKN[X2_INNER_IDX] % 2 != 0 || dimValueOfMKN[X1_INNER_IDX] % 2 != 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "x1 inner, x2 inner",
                        FormatString("%ld, %ld", dimValueOfMKN[X1_INNER_IDX], dimValueOfMKN[X2_INNER_IDX]).c_str(),
                        "when the dtype of input is FLOAT4, the inner axis of x1 and x2 must be an even number"),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckMXFP4Constraints(const std::vector<int64_t>& dimValueOfMKN) const
{
    if (!CheckKAxisGreaterThanTwo()) {
        return false;
    }
    if (!CheckInnerAxisIsEven(dimValueOfMKN)) {
        return false;
    }

    return true;
}

bool QuantBatchMatmulV3Checker::CheckInputValidInMxPerGroupMode(const gert::Shape& scaleShape,
                                                                const gert::StorageShape* pertokenShape,
                                                                const gert::Shape& x1Shape, const gert::Shape& x2Shape,
                                                                const std::vector<int64_t>& dimValueOfMKN) const
{
    if (!inputParams_.isMxPerGroup) {
        return true;
    }
    OP_TILING_CHECK(
        inputParams_.groupSizeM != 1UL || inputParams_.groupSizeN != 1UL || inputParams_.groupSizeK != 32UL,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "groupSizeM, groupSizeN, groupSizeK",
            FormatString("%lu, %lu, %lu", inputParams_.groupSizeM, inputParams_.groupSizeN, inputParams_.groupSizeK)
                .c_str(),
            "when the quantization mode is mx, input or inferred [groupSizeM, groupSizeN, groupSizeK] must be [1, 1, "
            "32], groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32"),
        return false);
    if (pertokenShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "pertokenScale", "null",
                                              "when the quantization mode is mx, pertokenScale can not be null");
        return false;
    }
    OP_TILING_CHECK(!MxPertokenScaleShapeCheck(pertokenShape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The perTokenScale shape check failed."), return false);
    OP_TILING_CHECK(!MxScaleShapeCheck(scaleShape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The scale shape check failed."), return false);
    OP_TILING_CHECK(!CheckBatchValidInMxPerGroupMode(scaleShape, pertokenShape->GetStorageShape(), x1Shape, x2Shape),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "The mx batch check failed."), return false);
    bool isFp4Input = inputParams_.aDtype == ge::DT_FLOAT4_E2M1;
    if (isFp4Input) {
        OP_TILING_CHECK(!CheckMXFP4Constraints(dimValueOfMKN),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckMXFP4Constraints failed."), return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker::CheckShapeInRangeForOptionalInputs(const gert::Shape& scaleShape,
                                                                   const gert::StorageShape* biasShape,
                                                                   const gert::StorageShape* pertokenShape,
                                                                   const gert::StorageShape* offsetShape,
                                                                   size_t outDimNum) const
{
    if (biasShape != nullptr) {
        auto biasDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(!(biasDimNum == 1 || biasDimNum == BIAS_THREE_DIM),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "bias",
                                                                 FormatString("%zuD", biasDimNum).c_str(),
                                                                 "the shape dim of bias must be 1 or 3"),
                        return false);
        OP_TILING_CHECK(biasDimNum == BIAS_THREE_DIM && outDimNum != biasDimNum,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            inputParams_.opName, "output", FormatString("%zuD", outDimNum).c_str(),
                            "when the shape dim of bias is 3, the shape dim of output must be 3"),
                        return false);
    }
    if (offsetShape != nullptr) {
        OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT8 && inputParams_.cDtype != ge::DT_INT8,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "offset", "not null",
                            "when the dtype of input is INT8 and the dtype of output is not INT8, offset must be null"),
                        return false);
        OP_TILING_CHECK(
            inputParams_.cDtype == ge::DT_INT8 && offsetShape->GetStorageShape().GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                inputParams_.opName, "offset", FormatString("%zuD", offsetShape->GetStorageShape().GetDimNum()).c_str(),
                "the shape dim of offset must be 1"),
            return false);
        OP_TILING_CHECK(
            inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "offset", "not null",
                                                  "when the dtype of scale is not UINT64/INT64, offset must be null"),
            return false);
    }
    if (pertokenShape != nullptr) {
        auto pertokenDimNum = pertokenShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            !inputParams_.isPerBlock && !inputParams_.isMxPerGroup &&
                (pertokenDimNum != 1 || scaleShape.GetDimNum() != 1),
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                inputParams_.opName, "pertokenScale, scale",
                FormatString("%zuD, %zuD", pertokenDimNum, scaleShape.GetDimNum()).c_str(),
                "when the quantization mode is not G-B, B-B or mx, the shape dim of pertokenScale and scale must be 1"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker::CheckShapeInBoundary(const gert::Shape& shape, uint32_t shapeIdx) const
{
    int64_t mul = 1;
    int64_t mulBound = 1;
    const char* dimName = shapeIdx == GetX1Idx() ? "x1" : "x2";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t curDim = shape.GetDim(i);

        OP_TILING_CHECK(
            curDim <= 0 || curDim > static_cast<int64_t>(INT32_MAX),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams_.opName, FormatString("%s dimension %zu", dimName, i).c_str(),
                std::to_string(curDim).c_str(),
                FormatString("the dimension value of %s must be within the range of [1,%d]", dimName, INT32_MAX)
                    .c_str()),
            return false);

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

bool QuantBatchMatmulV3Checker::BiasShapeCheck(const gert::Shape& biasShape, const gert::Shape& scaleShape,
                                               const gert::StorageShape* pertokenShape) const
{
    auto biasDimNum = biasShape.GetDimNum();
    // Validate the [batch, 1, n] bias layout.
    if (biasDimNum == BIAS_THREE_DIM) {
        auto biasFirstDim = static_cast<uint64_t>(biasShape.GetDim(0));
        auto biasSecondDim = static_cast<uint64_t>(biasShape.GetDim(1));
        auto biasThirdDim = static_cast<uint64_t>(biasShape.GetDim(2));
        OP_TILING_CHECK(
            biasFirstDim != inputParams_.batchC,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(inputParams_.opName, "biasFirstDim, batchC",
                                                   FormatString("%lu, %lu", biasFirstDim, inputParams_.batchC).c_str(),
                                                   "dimension 1 of bias must be equal to batchC"),
            return false);
        OP_TILING_CHECK(biasSecondDim != 1UL,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "biasSecondDim",
                                                              std::to_string(biasSecondDim).c_str(),
                                                              "dimension 2 of bias must be equal to 1"),
                        return false);
        OP_TILING_CHECK(
            biasThirdDim != inputParams_.nSize,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(inputParams_.opName, "biasThirdDim, n",
                                                   FormatString("%lu, %lu", biasThirdDim, inputParams_.nSize).c_str(),
                                                   "dimension 3 of bias must be equal to n"),
            return false);
    }
    if (biasDimNum == 1) {
        OP_TILING_CHECK(static_cast<uint64_t>(biasShape.GetDim(0)) != inputParams_.nSize,
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            inputParams_.opName, "biasDim, n",
                            FormatString("%ld, %lu", biasShape.GetDim(0), inputParams_.nSize).c_str(),
                            "dimension 1 of bias must be equal to n"),
                        return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker::ExtraInputCheck() const
{
    OP_TILING_CHECK(
        IsLowFloatInputType(inputParams_.aDtype) && context_->GetOptionalInputShape(GetOffsetIdx()) != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams_.opName, "offset", "not null",
            "when the dtype of input is HIFLOAT8/FLOAT8_E4M3FN/FLOAT8_E5M2/FLOAT4_E2M1, offset must be null"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::PerTokenDimValueCheck(const gert::Shape& scaleShape,
                                                      const gert::StorageShape* pertokenShape) const
{
    auto& pertoken = pertokenShape->GetStorageShape();
    bool isFp8HiF8 = inputParams_.aDtype == ge::DT_FLOAT8_E5M2 || inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN ||
                     inputParams_.aDtype == ge::DT_HIFLOAT8;
    uint64_t perTokenDim0 = static_cast<uint64_t>(pertoken.GetDim(0));
    OP_TILING_CHECK(
        isFp8HiF8 && !inputParams_.isMxPerGroup && (perTokenDim0 != 1UL && perTokenDim0 != inputParams_.mSize) &&
            pertoken.GetDimNum() == 1 && scaleShape.GetDimNum() == 1 && scaleShape.GetDim(0) == 1,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(inputParams_.opName, "m, pertokenScaleDim0",
                                               FormatString("%lu, %ld", inputParams_.mSize, pertoken.GetDim(0)).c_str(),
                                               "when the quantization mode is not mx and dimension 0 of scale is 1, "
                                               "dimension 0 of pertokenScale must be equal to m or 1"),
        return false);
    OP_TILING_CHECK(
        isFp8HiF8 && !inputParams_.isMxPerGroup && inputParams_.mSize != 1UL && perTokenDim0 == 1UL &&
            pertoken.GetDimNum() == 1 && scaleShape.GetDimNum() == 1 &&
            (scaleShape.GetDim(0) != 1L && static_cast<uint64_t>(scaleShape.GetDim(0)) != inputParams_.nSize),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "n, scaleDim0",
            FormatString("%lu, %ld", inputParams_.nSize, scaleShape.GetDim(0)).c_str(),
            "when the quantization mode is not mx, dimension 0 of pertokenScale is 1, and m is not 1, dimension 0 of "
            "scale must be equal to 1 or n"),
        return false);
    OP_TILING_CHECK(
        pertoken.GetDimNum() == 1 && static_cast<uint64_t>(perTokenDim0) == inputParams_.mSize &&
            scaleShape.GetDimNum() == 1 &&
            (scaleShape.GetDim(0) != 1L && static_cast<uint64_t>(scaleShape.GetDim(0)) != inputParams_.nSize),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "m, n, scaleDim0",
            FormatString("%lu, %lu, %ld", inputParams_.mSize, inputParams_.nSize, scaleShape.GetDim(0)).c_str(),
            "when dimension 0 of pertokenScale is m, dimension 0 of scale must be equal to 1 or n"),
        return false);
    OP_TILING_CHECK(
        perTokenDim0 != inputParams_.mSize && inputParams_.aDtype == ge::DT_INT8,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(inputParams_.opName, "m, pertokenScaleDim0",
                                               FormatString("%lu, %ld", inputParams_.mSize, pertoken.GetDim(0)).c_str(),
                                               "dimension 0 of pertokenScale must be equal to m"),
        return false);
    return true;
}

bool QuantBatchMatmulV3Checker::CheckDimValue(const gert::Shape& scaleShape, const gert::StorageShape* biasShape,
                                              const gert::StorageShape* pertokenShape,
                                              const gert::StorageShape* offsetShape,
                                              const std::vector<int64_t>& dimValueOfMKN) const
{
    auto x1Inner = dimValueOfMKN[X1_INNER_IDX];
    auto x2Inner = dimValueOfMKN[X2_INNER_IDX];
    auto x2Outer = dimValueOfMKN[X2_OUTER_IDX];
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? x2Inner : x2Outer);
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "x1 K, x2 K", FormatString("%lu, %lu", inputParams_.kSize, kBSize).c_str(),
                        "the K dimension of x1 must be equal to the K dimension of x2"),
                    return false);
    if (biasShape != nullptr && !BiasShapeCheck(biasShape->GetStorageShape(), scaleShape, pertokenShape)) {
        return false;
    }
    if (offsetShape != nullptr) {
        OP_TILING_CHECK(
            inputParams_.cDtype == ge::DT_INT8 &&
                !(offsetShape->GetStorageShape().GetDim(0) == 1L ||
                  static_cast<uint64_t>(offsetShape->GetStorageShape().GetDim(0)) == inputParams_.nSize),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                inputParams_.opName, "n, offsetDim",
                FormatString("%lu, %ld", inputParams_.nSize, offsetShape->GetStorageShape().GetDim(0)).c_str(),
                "dimension 0 of offset must be 1 or n"),
            return false);
    }
    auto scaleShapeLen = scaleShape.GetDimNum();
    OP_TILING_CHECK(scaleShapeLen != 1 && (inputParams_.scaleDtype == ge::DT_UINT64 ||
                                           inputParams_.scaleDtype == ge::DT_INT64 || pertokenShape == nullptr),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "scale",
                                                             FormatString("%zuD", scaleShapeLen).c_str(),
                                                             "when the dtype of scale is UINT64/INT64 or pertokenScale "
                                                             "does not exist, the shape dim of scale must be 1"),
                    return false);
    if (pertokenShape != nullptr) {
        OP_TILING_CHECK(!PerTokenDimValueCheck(scaleShape, pertokenShape),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Failed to check perTokenScale shape."),
                        return false);
    }

    if (inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype == ge::DT_INT4) {
        // remainder by 2 to check if it is a even number
        OP_TILING_CHECK(
            x1Inner < 0 || x1Inner % 2 != 0 || x2Inner < 0 || x2Inner % 2 != 0,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                inputParams_.opName, "x1 last axis, x2 last axis", FormatString("%ld, %ld", x1Inner, x2Inner).c_str(),
                "when the dtype of input is INT4, the last axis of x1 and x2 must be a positive even number"),
            return false);
        // 当输入为INT4, transA必须为false
        OP_TILING_CHECK(
            inputParams_.transA,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "transposeX1", "true",
                                                  "when the dtype of input is INT4, transposeX1 must be false"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker::CheckShape(const std::vector<gert::Shape*>& mandatoryShape,
                                           const gert::StorageShape* biasShape, const gert::StorageShape* pertokenShape,
                                           const std::vector<int64_t>& dimValueOfMKN) const
{
    auto& x1Shape = *mandatoryShape[0];
    auto& x2Shape = *mandatoryShape[1];
    auto& scaleShape = *mandatoryShape[2];
    auto offsetShape = context_->GetOptionalInputShape(GetOffsetIdx());
    size_t outDimNum = std::max(x1Shape.GetDimNum(), x2Shape.GetDimNum());
    if (!CheckShapeInRangeForOptionalInputs(scaleShape, biasShape, pertokenShape, offsetShape, outDimNum) ||
        !CheckDimValue(scaleShape, biasShape, pertokenShape, offsetShape, dimValueOfMKN) ||
        !CheckShapeInBoundary(x1Shape, GetX1Idx()) || !CheckShapeInBoundary(x2Shape, GetX2Idx())) {
        return false;
    }
    if ((!CheckInputValidInPerblockMode(scaleShape, pertokenShape, x1Shape, x2Shape) ||
         !CheckInputValidInMxPerGroupMode(scaleShape, pertokenShape, x1Shape, x2Shape, dimValueOfMKN))) {
        return false;
    }
    if (!ExtraInputCheck()) {
        return false;
    }
    return true;
}
} // namespace optiling
