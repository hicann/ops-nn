/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <dlfcn.h>
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/quant_batch_matmul_v3/op_api/quant_matmul_checker.h"
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "matmul/quant_batch_matmul_v3/op_api/quant_matmul_v3.h"
#include "matmul/common/op_host/op_api/quant_matmul_v4.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_quant_matmul_v5.h"
#include "quant_matmul_common_check.h"
#include "log/log.h"

using namespace op;
using Ops::NN::SwapLastTwoDimValue;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::Base::CeilDiv;

namespace {
static const size_t MAX_SCALE_DIM = 2;
static const size_t MIN_SCALE_DIM = 1;
static const size_t MAX_MX_SCALE_DIM = 3;
static const size_t YOFFSET_DIM = 1;
static const size_t SUPPORTED_K_ALIGN_NUM_INT4 = 2;
static const size_t MX_SCALE_MAX_DIM = 3;
static const size_t PENULTIMATE_DIM = 2;
static const size_t MX_SCALE_LAST_DIM_SIZE = 2;  // MicroScaling场景下scale tensor最后一维的大小
static const int64_t SUPPORTED_GROUP_SIZE = 32;
static const int64_t MAX_SHAPE_SIZE_A8W4_INT = 29576;
static const int64_t SUPPORTED_GROUP_SIZE_A8W4_INT = 256;
static const int64_t SUPPORTED_K_ALIGN_NUM = 64;
static const int64_t SUPPORTED_TCG_A8W4_K_ALIGN_NUM = 32;
static const int64_t SUPPORTED_MX_A8W4_K_ALIGN_NUM = 8;
static const int64_t SUPPORTED_N_ALIGN_NUM = 8;
static const uint64_t GROUP_M_OFFSET = 32;
static const uint64_t GROUP_N_OFFSET = 16;
static const uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
static const std::initializer_list<op::DataType> EIGHT_BIT_INT_INPUT_LIST = {op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> FOUR_BIT_INT_INPUT_LIST = {op::DataType::DT_INT4};
static const std::initializer_list<op::DataType> EIGHT_BIT_FLOAT_INPUT_LIST = {op::DataType::DT_FLOAT8_E4M3FN,
                                                                               op::DataType::DT_FLOAT8_E5M2};
static const std::initializer_list<op::DataType> FOUR_BIT_FLOAT_INPUT_LIST = {op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> Y_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> Y_SCALE_SUPPORT_LIST = {op::DataType::DT_UINT64};
static const std::initializer_list<op::DataType> X1_SCALE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> X2_SCALE_SUPPORT_LIST = {op::DataType::DT_UINT64};
static const std::initializer_list<op::DataType> Y_OFFSET_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static constexpr const char* kOpName = "aclnnQuantMatmulV5";

static inline bool isA8W4Float(const aclTensor *x1, const aclTensor *x2) {
    return (std::find(EIGHT_BIT_FLOAT_INPUT_LIST.begin(), EIGHT_BIT_FLOAT_INPUT_LIST.end(), x1->GetDataType()) !=
            EIGHT_BIT_FLOAT_INPUT_LIST.end()) &&
           (std::find(FOUR_BIT_FLOAT_INPUT_LIST.begin(), FOUR_BIT_FLOAT_INPUT_LIST.end(), x2->GetDataType()) !=
           FOUR_BIT_FLOAT_INPUT_LIST.end());
}

static inline bool isA8W4FloatTCG(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale)
{
    return x1->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN && x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1 &&
           x1Scale == nullptr;
}

static inline bool isA8W4FloatMx(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    return x1->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN && x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1 &&
        x1Scale != nullptr && x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 && x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static inline bool isA8W4Int(const aclTensor *x1, const aclTensor *x2) {
    return x1->GetDataType() == op::DataType::DT_INT8 && x2->GetDataType() == op::DataType::DT_INT32;
}

static inline bool isA8W4IntAfterPre(const aclTensor *x1, const aclTensor *x2) {
    return x1->GetDataType() == op::DataType::DT_INT8 && x2->GetDataType() == op::DataType::DT_INT4;
}

static inline bool IsA8W8Int(const aclTensor *x1, const aclTensor *x2) {
    return x1->GetDataType() == op::DataType::DT_INT8 && x2->GetDataType() == op::DataType::DT_INT8;
}

static inline bool IsA4W4Int(const aclTensor *x1, const aclTensor *x2) {
    return x1->GetDataType() == op::DataType::DT_INT4 && x2->GetDataType() == op::DataType::DT_INT4;
}

static inline bool IsFp8Input(const aclTensor *x1, const aclTensor *x2) {
    return (std::find(EIGHT_BIT_FLOAT_INPUT_LIST.begin(), EIGHT_BIT_FLOAT_INPUT_LIST.end(), x1->GetDataType()) !=
            EIGHT_BIT_FLOAT_INPUT_LIST.end()) &&
            (std::find(EIGHT_BIT_FLOAT_INPUT_LIST.begin(), EIGHT_BIT_FLOAT_INPUT_LIST.end(), x2->GetDataType()) !=
            EIGHT_BIT_FLOAT_INPUT_LIST.end());
}

static inline bool IsFp4Input(const aclTensor *x1, const aclTensor *x2) {
    return (std::find(FOUR_BIT_FLOAT_INPUT_LIST.begin(), FOUR_BIT_FLOAT_INPUT_LIST.end(), x1->GetDataType()) !=
            FOUR_BIT_FLOAT_INPUT_LIST.end()) &&
            (std::find(FOUR_BIT_FLOAT_INPUT_LIST.begin(), FOUR_BIT_FLOAT_INPUT_LIST.end(), x2->GetDataType()) !=
            FOUR_BIT_FLOAT_INPUT_LIST.end());
}

static inline bool IsMicroScaling(const aclTensor *x1Scale, const aclTensor *x2Scale) {
    if (x1Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static inline bool IsHif8Input(const aclTensor *x1, const aclTensor *x2) {
    return x1->GetDataType() == op::DataType::DT_HIFLOAT8 && x2->GetDataType() == op::DataType::DT_HIFLOAT8;
}

static inline bool IsPerblock(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                              const aclTensor *x2Scale)
{
    if (x1Scale == nullptr) {
        return false;
    }
    return ((IsHif8Input(x1, x2) || IsFp8Input(x1, x2)) && x1Scale->GetDataType() == op::DataType::DT_FLOAT &&
            x1Scale->GetViewShape().GetDimNum() > 1 && x2Scale->GetViewShape().GetDimNum() > 1 &&
            x2Scale->GetDataType() == op::DataType::DT_FLOAT);
}

static inline bool IsFormatNZ(const aclTensor* tensor) {
    return ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ ||
           ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ_C0_4 ||
           ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ_C0_32;
}

static inline bool IsA8W8Perblock(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                              const aclTensor *x2Scale)
{
    if (x1Scale == nullptr) {
        return false;
    }
    return (IsA8W8Int(x1, x2) && x1Scale->GetDataType() == op::DataType::DT_FLOAT &&
            x2Scale->GetDataType() == op::DataType::DT_FLOAT &&
            x1Scale->GetViewShape().GetDimNum() == x1->GetViewShape().GetDimNum() &&
            x2Scale->GetViewShape().GetDimNum() == x2->GetViewShape().GetDimNum());
}

static inline bool CheckA8W4IntGroupSize(const aclTensor *x2Scale, int64_t groupSize) {
    if (x2Scale->GetViewShape().GetDimNum() == 1) {
        if (groupSize != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A8W4 [int8/4] perchannel only support groupSize equal to 0.");
            return false;
        }
    } else {
        if (groupSize != SUPPORTED_GROUP_SIZE_A8W4_INT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A8W4 [int8/4] pergroup only support groupSize equal to 256.");
            return false;
        }
    }
    return true;
}

static inline bool CheckInputAttrExistence(const TupleAttr &boolsTrans, int64_t groupSize, const aclTensor *x2,
                                           TupleQuant &quantTensors, bool isA8W4INT = false)
{
    bool transposeX1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans);
    bool transposeX2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans);
    if (transposeX1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Only support transposeX1 is false.");
        return false;
    }

    bool isX2Nz = ge::GetPrimaryFormat(x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ;
    if (!transposeX2 && !isA8W4INT && !isX2Nz) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "transposeX2", "false",
            "When the quant mode is A8W4 fp8/4 ND, the value of transposeX2 must be true");
        return false;
    }
    auto &x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    if (isX2Nz) {
        if (x1Scale == nullptr && transposeX2) {
            // A8W4 mode
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "transposeX2", "true",
                "When the quant mode is A8W4 NZ T-CG, the value of transposeX2 can not be true");
            return false;
        } else if (IsMicroScaling(x1Scale, x2Scale) && !transposeX2) {
            // MxA8W4 mode
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "transposeX2", "false",
                "When the quant mode is A8W4 NZ mx, the value of transposeX2 must be true");
            return false;
        }
    }

    if (isA8W4INT) {    // A8W4 INT
        if (!CheckA8W4IntGroupSize(x2Scale, groupSize)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A8W4 [int8/4] groupsize is not supported");
            return false;
        }
    } else {            // A8W4 FLOAT
        uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
        if (groupSizeK != SUPPORTED_GROUP_SIZE) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "groupSizeK", std::to_string(groupSizeK).c_str(),
                "When the quant mode is A8W4 fp8/4, the value of groupSizeK must be 32");
            return false;
        }
    }
    return true;
}

static inline bool CheckInputExistence(TupleInput &inputTensors, TupleQuant &quantTensors,
                                       const aclTensor *out, bool& isA8W4) {
    auto x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x1Offset = std::get<INDEX_X1_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto x2Offset = std::get<INDEX_X2_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto yOffset = std::get<INDEX_Y_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    // 必选参数
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(x2Scale, return false);
    OP_CHECK_NULL(out, return false);
    isA8W4 = isA8W4Float(x1, x2) || isA8W4Int(x1, x2);
    // 不支持参数
    if (!isA8W4 && yScale != nullptr && !yScale->IsEmpty()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "yScale", "not empty/null", "The value of yScale must be empty tensor or nullptr");
        return false;
    }
    if (x1Offset != nullptr && !x1Offset->IsEmpty()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "x1Offset", "not empty/null", "The value of x1Offset must be empty tensor or nullptr");
        return false;
    }
    if (isA8W4 && x2Offset != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "x2Offset", "not empty/null", "The value of x2Offset must be empty tensor or nullptr");
        return false;
    }
    if (yOffset != nullptr && !yOffset->IsEmpty() && !isA8W4Int(x1, x2)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "yOffset", "not empty/null", "The value of yOffset must be empty tensor or nullptr");
        return false;
    }
    return true;
}

static inline bool CheckDimRange(TupleInput &inputTensors, TupleQuant &quantTensors, const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2StorageFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
    size_t x2MaxDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MAX_DIM_NUM_NZ : MAX_DIM_NUM_ND;
    size_t x2MinDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MIN_DIM_NUM_NZ : MIN_DIM_NUM_ND;
    size_t x2DimNum = x2->GetStorageShape().GetDimNum();
    if (x2DimNum < x2MinDimNum || x2DimNum > x2MaxDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                kOpName, "x2", std::to_string(x2DimNum).c_str(), "The shape dim of x2 must be in the valid range");
        return false;
    }
    if (x1->GetViewShape().GetDimNum() < static_cast<size_t>(MIN_DIM_NUM_ND)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            kOpName, "x1", std::to_string(x1->GetViewShape().GetDimNum()).c_str(),
            "2D");
        return false;
    }
    if (out->GetViewShape().GetDimNum() < static_cast<size_t>(MIN_DIM_NUM_ND)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            kOpName, "out", std::to_string(out->GetViewShape().GetDimNum()).c_str(),
            "2D");
        return false;
    }
    if (x1->GetViewShape().GetDimNum() > static_cast<size_t>(MAX_DIM_NUM_ND)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            kOpName, "x1", std::to_string(x1->GetViewShape().GetDimNum()).c_str(),
            "The shape dim of x1 can not be larger than 6D");
        return false;
    }
    if (out->GetViewShape().GetDimNum() > static_cast<size_t>(MAX_DIM_NUM_ND)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            kOpName, "out", std::to_string(out->GetViewShape().GetDimNum()).c_str(),
            "The shape dim of out can not be larger than 6D");
        return false;
    }
    if (x1Scale != nullptr && op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510 &&
         op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201) {
        OP_CHECK_WRONG_DIMENSION(x2Scale, 1, return false);
    }
    OP_LOGD("QuantMatmul check dimension range success.");
    return true;
}

static inline bool CheckScaleDimRangeA8W4(TupleQuant& quantTensors, bool isA8W4INT = false) {
    auto x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    if (isA8W4INT) {
        size_t x2ScaleDimNum = x2Scale->GetViewShape().GetDimNum();
        if (x2ScaleDimNum != MAX_SCALE_DIM && x2ScaleDimNum != MIN_SCALE_DIM) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "In A8W4 [int8/4] mode, the dimension of x2Scale should be 1 or 2. Actual x2Scale dimension: %zu",
                x2ScaleDimNum);
            return false;
        }
        if (x1Scale != nullptr && x1Scale->GetViewShape().GetDimNum() != x2ScaleDimNum) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "In A8W4 [int8/4] mode, the dimension of x1Scale should equal to that of x2Scale. Actual x1Scale "
                "dimension: %zu, x2Scale dimension: %zu.",
                x2ScaleDimNum, x1Scale->GetViewShape().GetDimNum());
            return false;
        }
    } else {
        size_t expectDimNum = IsMicroScaling(x1Scale, x2Scale) ? MAX_MX_SCALE_DIM : MAX_SCALE_DIM;
        if (x1Scale != nullptr && x1Scale->GetViewShape().GetDimNum() != expectDimNum) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                kOpName, "x1Scale", std::to_string(x1Scale->GetViewShape().GetDimNum()).c_str(),
                ("When the quant mode is A8W4, the shape dim of x1Scale must be " + std::to_string(expectDimNum) + "D")
                    .c_str());
            return false;
        }
        if (x2Scale->GetViewShape().GetDimNum() != expectDimNum) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                kOpName, "x2Scale", std::to_string(x2Scale->GetViewShape().GetDimNum()).c_str(),
                ("When the quant mode is A8W4, the shape dim of x2Scale must be " + std::to_string(expectDimNum) + "D")
                    .c_str());
            return false;
        }
    }
    OP_LOGD("QuantMatmul check scale range success");
    return true;
}

static inline bool CheckDimRangeA8W4(TupleInput& inputTensors, TupleQuant& quantTensors, const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto yOffset = std::get<INDEX_Y_OFFSET_IN_QUANT_TUPLE>(quantTensors);

    if (x1->GetViewShape().GetDimNum() != MAX_SCALE_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of x1 should be 2. Actual x1 dimension: %zu.",
                x1->GetViewShape().GetDimNum());
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != MAX_SCALE_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of x2 should be 2. Actual x2 dimension: %zu.",
                x2->GetViewShape().GetDimNum());
        return false;
    }
    if (bias != nullptr && bias->GetViewShape().GetDimNum() != MAX_SCALE_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of bias should be 2. Actual bias dimension: %zu.",
                bias->GetViewShape().GetDimNum());
        return false;
    }
    if (!CheckScaleDimRangeA8W4(quantTensors, isA8W4Int(x1,x2))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x1Scale x2Scale dimension is not support");
        return false;
    }
    if (yScale != nullptr && yScale->GetViewShape().GetDimNum() != MAX_SCALE_DIM) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The dimension of yScale should be 2. Actual yScale dimension: %zu.",
            yScale->GetViewShape().GetDimNum());
        return false;
    }
    if (yOffset != nullptr && yOffset->GetViewShape().GetDimNum() != YOFFSET_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of yOffset should be 1. Actual yOffset dimension: %zu.",
                yOffset->GetViewShape().GetDimNum());
        return false;
    }
    if (out->GetViewShape().GetDimNum() != MAX_SCALE_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of y should be 2. Actual y dimension: %zu.",
                out->GetViewShape().GetDimNum());
        return false;
    }
    OP_LOGD("QuantMatmul check dimension range success.");
    return true;
}

static bool CheckBasicInputDtypes(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    if (!CheckType(x1->GetDataType(), EIGHT_BIT_FLOAT_INPUT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "x1", op::ToString(x1->GetDataType()).GetString(),
                                              std::string("The dtype of x1 must be within the range ") +
                                                  op::ToString(EIGHT_BIT_FLOAT_INPUT_LIST).GetString());
        return false;
    }
    if (!CheckType(x2->GetDataType(), FOUR_BIT_FLOAT_INPUT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "x2", op::ToString(x2->GetDataType()).GetString(),
                                              std::string("The dtype of x2 must be within the range ") +
                                                  op::ToString(FOUR_BIT_FLOAT_INPUT_LIST).GetString());
        return false;
    }
    if (!CheckType(out->GetDataType(), Y_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            kOpName, "out", op::ToString(out->GetDataType()).GetString(),
            std::string("The dtype of out must be within the range ") + op::ToString(Y_SUPPORT_LIST).GetString());
        return false;
    }
    return true;
}

static bool CheckA8W4FloatDtype(const aclTensor* x2Scale, const aclTensor* bias, const aclTensor* yScale)
{
    if (bias != nullptr || yScale == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "bias, yScale", "non-null bias or null yScale",
            "When the quant mode is A8W4, the value of bias must be null and the value of yScale can not be null");
        return false;
    }
    if (!CheckType(x2Scale->GetDataType(), Y_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "x2Scale", op::ToString(x2Scale->GetDataType()).GetString(),
                                              std::string("The dtype of x2Scale must be within the range ") +
                                                  op::ToString(Y_SUPPORT_LIST).GetString());
        return false;
    }
    if (!CheckType(yScale->GetDataType(), Y_SCALE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "yScale", op::ToString(yScale->GetDataType()).GetString(),
                                              std::string("The dtype of yScale must be within the range ") +
                                                  op::ToString(Y_SCALE_SUPPORT_LIST).GetString());
        return false;
    }
    return true;
}

static bool CheckMxA8W4Dtype(const aclTensor* bias, const aclTensor* yScale, const aclTensor* out)
{
    if (bias != nullptr) {
        if (!CheckType(bias->GetDataType(), Y_SUPPORT_LIST)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "bias", op::ToString(bias->GetDataType()).GetString(),
                                                  std::string("The dtype of bias must be within the range ") +
                                                      op::ToString(Y_SUPPORT_LIST).GetString());
            return false;
        }
        if (bias->GetDataType() != out->GetDataType()) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                kOpName, "bias", op::ToString(bias->GetDataType()).GetString(),
                "When the quant mode is MxA8W4, the dtype of bias must be consistent with the dtype of out");
            return false;
        }
    }
    if (yScale != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "yScale", "not null",
                                              "When the quant mode is MxA8W4, the value of yScale must be null");
        return false;
    }
    return true;
}

static bool CheckDtype(const TupleInput &inputTensors, const TupleQuant &quantTensors,
                       const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);

    if (!CheckBasicInputDtypes(x1, x2, out)) {
        return false;
    }

    if (x1Scale == nullptr) {
        return CheckA8W4FloatDtype(x2Scale, bias, yScale);
    } else if (IsMicroScaling(x1Scale, x2Scale)) {
        return CheckMxA8W4Dtype(bias, yScale, out);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "quantMode", "unexpected", "unexpected quant mode");
        return false;
    }
}

static bool CheckDtypeA8W4Int(const TupleInput &inputTensors, const TupleQuant &quantTensors,
                       const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto yOffset = std::get<INDEX_Y_OFFSET_IN_QUANT_TUPLE>(quantTensors);

    if (x1Scale == nullptr || x1Scale->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In A8W4_Int mode, x1Scale does not support empty tensor or nullptr.");
        return false;
    }
    if (x2Scale == nullptr || x2Scale->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2Scale does not support empty tensor or nullptr.");
        return false;
    }
    if (bias != nullptr || yScale != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In A8W4_Int mode, bias and yScale must be null.");
        return false;
    }
    if (yOffset == nullptr || yOffset->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In A8W4_Int mode, yOffset does not support empty tensor or nullptr.");
        return false;
    }
    if (x2Scale->GetDataType() == DataType::DT_INT64) {
        (void)const_cast<aclTensor *>(x2Scale)->SetDataType(op::DataType::DT_UINT64);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(x1, EIGHT_BIT_INT_INPUT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, FOUR_BIT_INT_INPUT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1Scale, X1_SCALE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Scale, X2_SCALE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(yOffset, Y_OFFSET_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, Y_SUPPORT_LIST, return false);

    return true;
}

static inline bool CheckFormat(const TupleInput &inputTensors, const TupleQuant &quantTensors, const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors);
    auto x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto yOffset = std::get<INDEX_Y_OFFSET_IN_QUANT_TUPLE>(quantTensors);

    CHECK_RET(x1->GetStorageFormat() == op::Format::FORMAT_ND, false);
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        CHECK_RET(x2->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    CHECK_RET(x2Scale->GetStorageFormat() == op::Format::FORMAT_ND, false);
    if (yScale != nullptr) {
        CHECK_RET(yScale->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (bias != nullptr) {
        CHECK_RET(bias->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (x1Scale != nullptr) {
        CHECK_RET(x1Scale->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (yOffset != nullptr) {
        CHECK_RET(yOffset->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    CHECK_RET(out->GetStorageFormat() == op::Format::FORMAT_ND, false);
    return true;
}

static inline bool CheckA8W4ScaleShape(const aclTensor *x1Scale, int64_t x1MDim) {
    if (x1Scale->GetViewShape().GetDimNum() == MAX_SCALE_DIM) {
        if (x1Scale->GetViewShape().GetDim(0) != x1MDim || x1Scale->GetViewShape().GetDim(1) != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x1scale shape must be [%ld, 1] for A8W4[int8/4] pergroup, which are [%ld, %ld]",
                    x1MDim, x1Scale->GetViewShape().GetDim(0), x1Scale->GetViewShape().GetDim(1));
            return false;
        }
    } else {
        if (x1Scale->GetViewShape().GetDim(0) != x1MDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x1scale shape must be [%ld] for A8W4[int8/4] perchannel, which are [%ld]",
                    x1MDim, x1Scale->GetViewShape().GetDim(0));
            return false;
        }
    }
    return true;
}

static inline bool CheckScaleX1Shape(const TupleQuant& quantTensors, int64_t x1MDim, int64_t groupDim,
                                     bool isA8W4INT = false) {
    auto x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    if (isA8W4INT) {    // isA8W4INT对x1scale的校验
        if (x1Scale != nullptr) {
            CHECK_RET(CheckA8W4ScaleShape(x1Scale, x1MDim), false);
        }
    } else if (IsMicroScaling(x1Scale, x2Scale)) {
        // MX_SCALE_LAST_DIM_SIZE: x1Scale (m, k / GroupSize / MX_SCALE_LAST_DIM_SIZE, MX_SCALE_LAST_DIM_SIZE)
        if (x1Scale->GetViewShape().GetDim(0) != x1MDim ||
            x1Scale->GetViewShape().GetDim(1) != CeilDiv(groupDim, static_cast<int64_t>(MX_SCALE_LAST_DIM_SIZE)) ||
            x1Scale->GetViewShape().GetDim(2) != MX_SCALE_LAST_DIM_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                kOpName, "x1Scale", op::ToString(x1Scale->GetViewShape()).GetString(),
                ("When the quant mode is MxA8W4, the shape of x1Scale must be [" + std::to_string(x1MDim) + ", " +
                 std::to_string(CeilDiv(groupDim, static_cast<int64_t>(MX_SCALE_LAST_DIM_SIZE))) + ", " +
                 std::to_string(MX_SCALE_LAST_DIM_SIZE) + "]").c_str());
            return false;
        }
    }
    return true;
}

static inline bool CheckScaleX2Shape(const TupleQuant& quantTensors, int64_t x2NDim, int64_t groupDim, bool transposeX2,
                                     bool isA8W4INT = false) {
    auto x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);
    int64_t x2ScaleNDim = transposeX2 ? x2Scale->GetViewShape().GetDim(0) : x2Scale->GetViewShape().GetDim(1);
    int64_t x2ScaleGroupDim = transposeX2 ? x2Scale->GetViewShape().GetDim(1) : x2Scale->GetViewShape().GetDim(0);
    if (isA8W4INT) {    // isA8W4INT对x2scale的校验
        if (x2Scale->GetViewShape().GetDimNum() == MAX_SCALE_DIM) {
            if (x2Scale->GetViewShape().GetDim(0) != groupDim ||
                x2Scale->GetViewShape().GetDim(1) != x2NDim) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x2scale shape must be [%ld, %ld], which are [%ld, %ld]",
                        groupDim, x2NDim,  x2Scale->GetViewShape().GetDim(0), x2Scale->GetViewShape().GetDim(1));
                return false;
            }
        } else {
            if (x2Scale->GetViewShape().GetDim(0) != x2NDim) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x2Scale shape must be [%ld, %ld], which are [%ld, %ld]",
                        groupDim, x2NDim, x2Scale->GetViewShape().GetDim(0), x2Scale->GetViewShape().GetDim(1));
                return false;
            }
        }
    } else if (IsMicroScaling(x1Scale, x2Scale)) {
        // MX_SCALE_LAST_DIM_SIZE:x2Scale Shape: (n, groupDim / MX_SCALE_LAST_DIM_SIZE, MX_SCALE_LAST_DIM_SIZE)
        if (x2ScaleNDim != x2NDim ||
            x2ScaleGroupDim != CeilDiv(groupDim, static_cast<int64_t>(MX_SCALE_LAST_DIM_SIZE)) ||
            x2Scale->GetViewShape().GetDim(2) != MX_SCALE_LAST_DIM_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPE(
                kOpName, "x2Scale", op::ToString(x2Scale->GetViewShape()).GetString(),
                "[" + std::to_string(x2NDim) + ", " +
                    std::to_string(CeilDiv(groupDim, static_cast<int64_t>(MX_SCALE_LAST_DIM_SIZE))) + ", " +
                    std::to_string(MX_SCALE_LAST_DIM_SIZE) + "]");
            return false;
        }
    } else {
        if (x2ScaleNDim != x2NDim || x2ScaleGroupDim != groupDim) {
            OP_LOGE_FOR_INVALID_SHAPE(kOpName, "x2Scale", op::ToString(x2Scale->GetViewShape()).GetString(),
                                      "[" + std::to_string(x2NDim) + ", " + std::to_string(groupDim) + "]");
            return false;
        }
    }
    if (yScale != nullptr) {
        if (yScale->GetViewShape().GetDim(0) != 1 || yScale->GetViewShape().GetDim(1) != x2NDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The yscale shape must be [1, %ld], which are [%ld, %ld]",
                    x2NDim, yScale->GetViewShape().GetDim(0), yScale->GetViewShape().GetDim(1));
            return false;
        }
    }
    return true;
}

static inline bool CheckOutAndOffsetShape(const TupleQuant& quantTensors, int64_t x1MDim, int64_t x2NDim,
                                          const aclTensor* out)
{
    auto bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors);
    auto yOffset = std::get<INDEX_Y_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    if (bias != nullptr) {
        if (bias->GetViewShape().GetDim(0) != 1 || bias->GetViewShape().GetDim(1) != x2NDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The bias shape must be [1, %ld], which are [%ld, %ld]", x2NDim,
                    bias->GetViewShape().GetDim(0), bias->GetViewShape().GetDim(1));
            return false;
        }
    }
    if (yOffset != nullptr && !yOffset->IsEmpty()) {
        if (yOffset->GetViewShape().GetDim(0) != x2NDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The yOffset shape must be [%ld], which is [%ld]", x2NDim,
                    yOffset->GetViewShape().GetDim(0));
            return false;
        }
    }
    if (out->GetViewShape().GetDim(0) != x1MDim || out->GetViewShape().GetDim(1) != x2NDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The out shape must be [%ld, %ld], which are [%ld, %ld]", x1MDim, x2NDim,
                out->GetViewShape().GetDim(0), out->GetViewShape().GetDim(1));
        return false;
    }
    return true;
}

static inline bool CheckA8W4KDim(const TupleInput& inputTensors, const TupleQuant& quantTensors, int64_t kDim)
{
    auto x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    if (isA8W4IntAfterPre(x1, x2)) {
        bool isPerChannel = x2Scale->GetViewShape().GetDimNum() == 1;
        size_t kAlign = isPerChannel ? SUPPORTED_K_ALIGN_NUM_INT4 : SUPPORTED_GROUP_SIZE_A8W4_INT;
        if (kDim % kAlign != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the k dim must be align to %ld, which is %ld", kAlign, kDim);
            return false;
        }
    } else if (isA8W4FloatTCG(x1, x2, x1Scale)) {
        if (kDim % SUPPORTED_TCG_A8W4_K_ALIGN_NUM != 0 || kDim <= SUPPORTED_TCG_A8W4_K_ALIGN_NUM) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "k", std::to_string(kDim).c_str(),
                "The value of k must be aligned to 32 and greater than 32 for T-CG quantization");
            return false;
        }
    } else if (isA8W4FloatMx(x1, x2, x1Scale, x2Scale)) {
        if (kDim % SUPPORTED_MX_A8W4_K_ALIGN_NUM != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "k", std::to_string(kDim).c_str(),
                                                  "The value of k must be aligned to 8 for MX quantization");
            return false;
        }    
    } else {
        if (kDim % SUPPORTED_K_ALIGN_NUM != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the k dim must be align to %ld, which is %ld", SUPPORTED_K_ALIGN_NUM, kDim);
            return false;
        }
    }
    return true;
}

static inline bool CheckKDimAndBasicShape(
    const TupleInput& inputTensors, const TupleQuant& quantTensors, int64_t x1KDim, int64_t x2KDim, int64_t x2NDim)
{
    // CHECK basic shape
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201) {   // A8W4INT A2 A3
        if (x1KDim <= 0 || x1KDim > MAX_SHAPE_SIZE_A8W4_INT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the k-dim must in [1, %ld], which is %ld", MAX_SHAPE_SIZE_A8W4_INT, x1KDim);
            return false;
        }
    } else {    // A8W4FLOAT DAV3510
        if (x1KDim <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "k", std::to_string(x1KDim).c_str(),
                                                  "The value of k must be >= 1");
            return false;
        }
    }
    if (x2NDim <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the n-dim must be at least 1, which is %ld", x2NDim);
        return false;
    }
    if (x1KDim != x2KDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the k dim of x1 and x2 must be same, which are %ld and %ld",
                x1KDim, x2KDim);
        return false;
    }
    // CHECK kDim
    if (!CheckA8W4KDim(inputTensors, quantTensors, x1KDim)) {
        return false;
    }
    return true;
}

static inline bool CheckX1X2Shape(
    const TupleInput& inputTensors, const TupleQuant& quantTensors, int64_t x1KDim, int64_t x2KDim, int64_t x2NDim)
{
    if (!CheckKDimAndBasicShape(inputTensors, quantTensors, x1KDim, x2KDim, x2NDim)) {
        return false;
    }
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    bool isX2Nz = ge::GetPrimaryFormat(x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ;
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510 && isX2Nz) {
        if (x2NDim % SUPPORTED_N_ALIGN_NUM != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "n", std::to_string(x2NDim).c_str(),
                "When the quant mode is A8W4 NZ, the value of n must be aligned to 8");
            return false;
        }
    }
    return true;
}

static inline bool CheckShape(const TupleInput &inputTensors, const TupleQuant &quantTensors,
                              const TupleAttr &boolsTrans, const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    bool isA8W4INT = isA8W4IntAfterPre(x1, x2);
    bool transposeX1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans);
    bool transposeX2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans);

    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x1MDim = transposeX1 ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];
    int64_t x2KDim = transposeX2 ? x2Shape[x2DimNum - 1] : x2Shape[x2DimNum - PENULTIMATE_DIM];
    int64_t x2NDim = transposeX2 ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];
    if (x1MDim <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the m-dim must > 0, which is %ld", x1MDim);
        return false;
    }
    CHECK_RET(CheckX1X2Shape(inputTensors, quantTensors, x1KDim, x2KDim, x2NDim), false);
    int64_t groupDim = isA8W4INT ? (x2KDim + SUPPORTED_GROUP_SIZE_A8W4_INT - 1) / SUPPORTED_GROUP_SIZE_A8W4_INT :
                                    (x2KDim + SUPPORTED_GROUP_SIZE - 1) / SUPPORTED_GROUP_SIZE;
    CHECK_RET(CheckScaleX1Shape(quantTensors, x1MDim, groupDim, isA8W4INT), false);
    CHECK_RET(CheckScaleX2Shape(quantTensors, x2NDim, groupDim, transposeX2, isA8W4INT), false);
    CHECK_RET(CheckOutAndOffsetShape(quantTensors, x1MDim, x2NDim, out), false);
    return true;
}

static inline aclnnStatus CheckParamsA8W4Float(const TupleInput &inputTensors, const TupleQuant &quantTensors,
                                          const TupleAttr &boolsTrans, const aclTensor *out) {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        // 1. 校验dtype是否符合要求
        CHECK_RET(CheckDtype(inputTensors, quantTensors, out), ACLNN_ERR_PARAM_INVALID);
        // 2. 检查format是否符合要求
        CHECK_RET(CheckFormat(inputTensors, quantTensors, out), ACLNN_ERR_PARAM_INVALID);
        // 3. 检查shape是否符合要求
        CHECK_RET(CheckShape(inputTensors, quantTensors, boolsTrans, out), ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "A8W4 only Support NpuArch DAV3510");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckParamsA8W4Int(const TupleInput &inputTensors, const TupleQuant &quantTensors,
                                          const TupleAttr &boolsTrans, const aclTensor *out) {
    auto socVer = GetCurrentPlatformInfo().GetSocVersion();
    if (socVer == SocVersion::ASCEND910B || socVer == SocVersion::ASCEND910_93) {
        // 1. 校验dtype是否符合要求
        CHECK_RET(CheckDtypeA8W4Int(inputTensors, quantTensors, out), ACLNN_ERR_PARAM_INVALID);
        // 2. 检查format是否符合要求
        CHECK_RET(CheckFormat(inputTensors, quantTensors, out), ACLNN_ERR_PARAM_INVALID);
        // 3. 检查shape是否符合要求
        CHECK_RET(CheckShape(inputTensors, quantTensors, boolsTrans, out), ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "A8W4Int only Support socversion A2 or A3");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const QuantMatmulChecker& qmmV3Checker, bool isA8W4I, bool isA8W4F)
{
    if (isA8W4I) {
        return CheckParamsA8W4Int(
            qmmV3Checker.inputTensors_, qmmV3Checker.quantTensors_, qmmV3Checker.boolsTrans_, qmmV3Checker.out_);
    }
    if (isA8W4F) {
        return CheckParamsA8W4Float(
            qmmV3Checker.inputTensors_, qmmV3Checker.quantTensors_, qmmV3Checker.boolsTrans_, qmmV3Checker.out_);
    }
    aclnnStatus ret = qmmV3Checker.CheckParams();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    OP_LOGD("QuantMatmul check params success.");
    return ACLNN_SUCCESS;
}

static aclIntArray* GetPerm(int64_t dim, aclOpExecutor* executor) {
    CHECK_RET(static_cast<size_t>(dim) >= MIN_DIM_NUM_ND, nullptr);
    std::vector<int64_t> valuePerm(dim);
    for (int64_t i = 0; i < dim; i++) {
        valuePerm[i] = i;
    }
    std::swap(valuePerm[dim - 1], valuePerm[dim - PENULTIMATE_DIM]);
    return executor->AllocIntArray(valuePerm.data(), dim);
}

static aclnnStatus TransposeAndTransDataForInputs(const aclTensor *&x1, const aclTensor *&x2, bool& transposeX1,
                                                  bool& transposeX2, aclOpExecutor* executor) {
    if (transposeX1) {
        auto perm = GetPerm(x1->GetViewShape().GetDimNum(), executor);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        x1 = l0op::Transpose(x1, perm, executor);
        CHECK_RET(x1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transposeX1 = !transposeX1;
    }
    if (ge::GetPrimaryFormat(x2->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ) {
        return ACLNN_SUCCESS;
    }
    x2 = SetTensorToNDFormat(x2);
    if (!transposeX2) {
        auto perm = GetPerm(x2->GetViewShape().GetDimNum(), executor);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        x2 = l0op::Transpose(x2, perm, executor);
        CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transposeX2 = !transposeX2;
    }
    x2 = l0op::TransData(x2, Format::FORMAT_FRACTAL_NZ, 0, executor);
    CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static inline bool MxScaleContiguousProcess(const aclTensor *&mxScaleTensor, aclOpExecutor *executor)
{
    if (mxScaleTensor == nullptr || mxScaleTensor->GetViewShape().GetDimNum() < MX_SCALE_MAX_DIM) {
        OP_LOGD("MX scale no need to do contiguous process.");
        return true;
    }
    auto transposeFlag = false;
    int64_t dimNum = mxScaleTensor->GetViewShape().GetDimNum();
    int64_t lastDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 1);
    int64_t lastSecondDim = mxScaleTensor->GetViewShape().GetDim(dimNum - PENULTIMATE_DIM);
    int64_t lastThirdDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 3);    // 3: 倒数第3维
    if (mxScaleTensor->GetViewStrides()[dimNum - 3] == lastDim &&   // 3： 倒数第3维
        mxScaleTensor->GetViewStrides()[dimNum - PENULTIMATE_DIM] == lastDim * lastThirdDim) {
        int64_t tmpNxD = lastDim * lastSecondDim * lastThirdDim;
        transposeFlag = true;
        // 4：batch维度从倒数第4维起
        for (int64_t batchDim = dimNum - 4; batchDim >= 0; batchDim--) {
            if (mxScaleTensor->GetViewStrides()[batchDim] != tmpNxD) {
                transposeFlag = false;
                break;
            }
            tmpNxD *= mxScaleTensor->GetViewShape().GetDim(batchDim);
        }
        if (lastSecondDim == 1 && lastThirdDim == 1) {
            transposeFlag = false;
        }
    }
    if (transposeFlag) {
        op::Shape swapedShape = mxScaleTensor->GetViewShape();
        swapedShape.SetDim(dimNum - PENULTIMATE_DIM, lastThirdDim);
        swapedShape.SetDim(dimNum - 3, lastSecondDim);  // 3： 倒数第3维
        mxScaleTensor = executor->CreateView(mxScaleTensor, swapedShape, mxScaleTensor->GetViewOffset());
    } else {
        mxScaleTensor = l0op::Contiguous(mxScaleTensor, executor);
    }
    CHECK_RET(mxScaleTensor != nullptr, false);
    return true;
}

static aclnnStatus PreMatmulCalcProcess(TupleInput &inputTensors, TupleQuant &quantTensors,
                                        TupleAttr &boolsTrans, const aclTensor *out,
                                        bool& isA8W4, aclOpExecutor *executor) {
    auto &x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto &x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto &x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &x2Offset = std::get<INDEX_X2_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto &yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors);
    int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_QUANT_TUPLE>(quantTensors);
    bool &transposeX1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans);
    bool &transposeX2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans);

    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(CheckInputExistence(inputTensors, quantTensors, out, isA8W4), ACLNN_ERR_PARAM_NULLPTR);

    bool tempTransposeValue = false;
    CHECK_RET(TensorContiguousProcess(x1, transposeX1, executor), ACLNN_ERR_INNER_NULLPTR);
    if (x1Scale != nullptr) {
        if (IsMicroScaling(x1Scale, x2Scale) && (IsFp4Input(x1, x2) || IsFp8Input(x1, x2) || isA8W4Float(x1, x2))) {
            CHECK_RET(MxScaleContiguousProcess(x1Scale, executor), ACLNN_ERR_INNER_NULLPTR);
        } else {
            CHECK_RET(TensorContiguousProcess(x1Scale, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
        }
    }
    if (IsMicroScaling(x1Scale, x2Scale) && (IsFp4Input(x1, x2) || IsFp8Input(x1, x2) || isA8W4Float(x1, x2))) {
        CHECK_RET(MxScaleContiguousProcess(x2Scale, executor), ACLNN_ERR_INNER_NULLPTR);
    } else {
        CHECK_RET(TensorContiguousProcess(x2Scale, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(TensorContiguousProcess(bias, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(x2Offset, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
    if (yScale != nullptr) {
        CHECK_RET(TensorContiguousProcess(yScale, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    auto ret = WeightNZCaseProcess(x2, transposeX2, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (isA8W4) {
        bool isA8W4INT = isA8W4Int(x1, x2);
        CHECK_RET(CheckInputAttrExistence(boolsTrans, groupSize, x2, quantTensors, isA8W4INT), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckDimRangeA8W4(inputTensors, quantTensors, out), ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckDimRange(inputTensors, quantTensors, out), ACLNN_ERR_PARAM_INVALID);
    }
    A4W4CaseProcess(x1, x2, executor);
    return ACLNN_SUCCESS;
}

static aclTensor* ProcessScaleTensor(const aclTensor *scale) {
    auto castedScale = const_cast<aclTensor*>(scale);
    if (castedScale->GetDataType() == op::DataType::DT_INT64) {
        castedScale->SetDataType(op::DataType::DT_UINT64);
    }
    return castedScale;
}

static void A8W4ProcessYScaleTensor(const aclTensor *x1Scale, const aclTensor *yScale) {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        // A8W4场景输入的INT64转为UINT64
        if (x1Scale == nullptr && yScale != nullptr && yScale->GetDataType() == op::DataType::DT_INT64) {
            auto castedYScale = const_cast<aclTensor*>(yScale);
            castedYScale->SetDataType(op::DataType::DT_UINT64);
            yScale = castedYScale;
            OP_LOGD("The conversion from INT64 to UINT64 is completed.");
        }
    }
}

static inline bool validGroupSize(uint64_t groupSizeM, uint64_t groupSizeN) {
    return (groupSizeM == 0 && groupSizeN == 0) || (groupSizeM == 1 && groupSizeN == 1);
}

static inline bool A8W4InferGroupSize(int64_t& groupSize) {
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    if (!validGroupSize(groupSizeM, groupSizeN)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            kOpName, "groupSizeM, groupSizeN",
            (std::to_string(groupSizeM) + ", " + std::to_string(groupSizeN)).c_str(),
            "The values of groupSizeM and groupSizeN must be 0 or 1");
        return false;
    }

    OP_LOGD(
        "A8W4 Infered groupSize: groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.", groupSizeM, groupSizeN,
        groupSizeK);
    groupSize = groupSizeK;
    return true;
}

static aclnnStatus aclnnQuantMatmulGetWorkspaceSizeCommonProcess(TupleInput &inputTensors, TupleQuant &quantTensors,
                                                                 TupleAttr &boolsTrans, const aclTensor *out,
                                                                 aclOpExecutor *executor) {
    auto &x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto &x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto &x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &x1Offset = std::get<INDEX_X1_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto &x2Offset = std::get<INDEX_X2_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto &yOffset = std::get<INDEX_Y_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto &bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors);
    int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_QUANT_TUPLE>(quantTensors);
    int64_t interfaceType = std::get<INDEX_INTERFACE_TYPE_IN_QUANT_TUPLE>(quantTensors);
    bool &transposeX1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans);
    bool &transposeX2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans);
    bool isA8W4 = false;
    bool isA8W4F = isA8W4Float(x1, x2);
    bool isPseudoQuant = isA8W4F || isA8W4Int(x1, x2);
    if (isA8W4F) {
        CHECK_RET(A8W4InferGroupSize(groupSize), ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("Infer groupSize success. groupSize: %ld.", groupSize);
    }
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 && !isPseudoQuant) {
        auto x1DimNum = x1->GetViewShape().GetDimNum();
        auto inputSizeM = transposeX1 ? x1->GetViewShape().GetDim(x1DimNum - 1) :
                                        x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM);
        auto x2DimNum = x2->GetViewShape().GetDimNum();
        auto inputSizeN = transposeX2 ? x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM) :
                                        x2->GetViewShape().GetDim(x2DimNum - 1);
        if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
            if (inputSizeM == 0) {
                OP_LOGD("aclnnV5 nz m=0");
                return ACLNN_SUCCESS;
            }
        } else {
            if (inputSizeM == 0 || inputSizeN == 0) {
                OP_LOGD("aclnnV5 nd m/n=0");
                return ACLNN_SUCCESS;
            }
        }
    }
    auto ret = PreMatmulCalcProcess(inputTensors, quantTensors, boolsTrans, out, isA8W4, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto reformatedX1 = SetTensorToNDFormat(x1);
    const aclTensor *reformatedX2 = x2;
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        ret = TransposeAndTransDataForInputs(reformatedX1, reformatedX2, transposeX1, transposeX2, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        if (x1Scale != nullptr && !x1Scale->IsEmpty()){
            OP_LOGD("Npu_Arch = 2002 pertoken mode need transData x1");
            reformatedX1 = l0op::TransData(reformatedX1, Format::FORMAT_FRACTAL_NZ, 0, executor);
            CHECK_RET(x1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        if (!isA8W4F) {
            reformatedX2 = SetTensorToNDFormat(x2);
        }
    }
    int64_t groupSizeReal = groupSize;
    if (!isA8W4) {
        QuantMatmulChecker qmmV3Checker(inputTensors, quantTensors, boolsTrans, out);
        qmmV3Checker.Init();
        CHECK_RET(qmmV3Checker.InferGroupSize(groupSizeReal), ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("Infer groupSize success. groupSize: %ld.",
                groupSizeReal);
    } else {
        A8W4ProcessYScaleTensor(x1Scale, yScale);
    }
    const aclTensor *reformatedX1Scale = GetNDFormat(x1Scale);
    const aclTensor *reformatedX2Scale = GetNDFormat(x2Scale);
    const aclTensor *reformatedBias = GetNDFormat(bias);
    const aclTensor *reformatedYScale = GetNDFormat(yScale);
    const TupleInput inputTuple = std::tie(reformatedX1, reformatedX2);
    const TupleQuant quantTuple = std::tie(reformatedX1Scale, reformatedX2Scale, reformatedYScale, x1Offset,
                                           x2Offset, yOffset, reformatedBias, groupSizeReal, interfaceType);
    bool isA8W4I = isA8W4IntAfterPre(x1, x2);   // 前处理是将x2的数据类型转成了INT4
    bool isA8W8Perblock = IsA8W8Perblock(x1, x2, x1Scale, x2Scale);
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;

    auto castedScale = ProcessScaleTensor(reformatedX2Scale);
    int64_t dtype = 0;
    TupleTensor inOutTuple = std::tie(reformatedX1, reformatedX2, out);
    GetDtypeAndTranspose(inOutTuple, dtype, transposeX1, transposeX2);

    QuantMatmulChecker qmmV3Checker(inputTuple, quantTuple, boolsTrans, out);
    qmmV3Checker.Init();
    bool isA4W4PergroupNonSymmetric =
        qmmV3Checker.IsA4W4PergroupNonSymmetric(groupSizeK) && CheckFormat(inputTensors, quantTensors, out);
    ret = CheckParams(qmmV3Checker, isA8W4I, isA8W4F);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *matmulRet = nullptr;
    if (isA8W4 || isA8W8Perblock || isA4W4PergroupNonSymmetric) {
        // 调用l0算子QuantBatchMatmulV4进行计算
        matmulRet = l0op::QuantBatchMatmulV4(reformatedX1, reformatedX2, reformatedBias, reformatedX1Scale,
                                             castedScale, reformatedYScale, x1Offset, x2Offset, yOffset,
                                             nullptr, dtype, -1, transposeX1, transposeX2, groupSize, executor);
    } else {
        // 调用l0算子QuantBatchMatmulV3进行计算
        matmulRet = l0op::QuantBatchMatmulV3(reformatedX1, reformatedX2, castedScale, x2Offset, reformatedBias,
                                             reformatedX1Scale, dtype, transposeX1, transposeX2, groupSizeReal, executor);
    }
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P && x1Scale != nullptr && !x1Scale->IsEmpty()) {
        OP_LOGD("Npu_Arch = 2002 pertoken mode need transData out");
        matmulRet = l0op::TransData(matmulRet, Format::FORMAT_ND, 0, executor);
    }

    CHECK_RET(PostMatmulCalcProcess(matmulRet, x1, x2, out, executor) == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                               const aclTensor *x2Scale, const aclTensor *yScale,
                                               const aclTensor *x1Offset, const aclTensor *x2Offset,
                                               const aclTensor *yOffset, const aclTensor *bias, bool transposeX1,
                                               bool transposeX2, int64_t groupSize, aclTensor *out,
                                               uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnQuantMatmulV5,
                   DFX_IN(x1, x2, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, bias, transposeX1, transposeX2,
                          groupSize),
                   DFX_OUT(out));
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_CHECK(
            x2 != nullptr && !IsFormatNZ(x2),
            OP_LOGE_FOR_INVALID_FORMAT(kOpName, "x2", "FORMAT_FRACTAL_NZ", "FORMAT_ND"),
            return ACLNN_ERR_PARAM_INVALID);
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    TupleInput inputTuple = std::tie(x1, x2);
    // 5 represents the aclnnQuantMatmulV5 interface
    const int64_t interfaceType = 5;
    TupleQuant quantTuple = std::tie(x1Scale, x2Scale, yScale, x1Offset,
                                     x2Offset, yOffset, bias, groupSize, interfaceType);
    TupleAttr attrTuple = std::tie(transposeX1, transposeX2);
    auto ret = aclnnQuantMatmulGetWorkspaceSizeCommonProcess(inputTuple, quantTuple, attrTuple, out,
                                                             uniqueExecutor.get());

    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulV5(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnQuantMatmulV5);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
