/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QUANT_MATMUL_V4_COMMON_H
#define QUANT_MATMUL_V4_COMMON_H

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <tuple>

#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "opdev/common_types.h"

namespace quant_matmul_v4 {
// Constants
static constexpr int MX_SCALE_LAST_DIM_INDEX = 2;
static constexpr int MX_SCALE_LAST_DIM = 2;

static constexpr int INDEX_X1_IN_MANDTORY_TUPLE = 0;
static constexpr int INDEX_X2_IN_MANDTORY_TUPLE = 1;
static constexpr int INDEX_SCALE_IN_MANDTORY_TUPLE = 2;
static constexpr int INDEX_OFFSET_IN_OPTIONAL_TUPLE = 0;
static constexpr int INDEX_PERTOKEN_IN_OPTIONAL_TUPLE = 1;
static constexpr int INDEX_BIAS_IN_OPTIONAL_TUPLE = 2;
static constexpr int INDEX_Y_SCALE_IN_OPTIONAL_TUPLE = 3;
static constexpr int INDEX_Y_OFFSET_IN_OPTIONAL_TUPLE = 4;
static constexpr int INDEX_GROUP_SIZE_IN_OPTIONAL_TUPLE = 5;
static constexpr int INDEX_OUT_IN_TUPLE = 2;
static constexpr int INDEX_ISA4W4_IN_BOOL_TUPLE = 2;
static constexpr size_t LAST_SECOND_DIM_INDEX = 2;

static constexpr int MIN_DIM_NUM_ND = 2;
static constexpr int MAX_DIM_NUM_ND = 6;
static constexpr int MIN_DIM_NUM_NZ = 4;
static constexpr int MAX_DIM_NUM_NZ = 8;
static constexpr int PENULTIMATE_DIM = 2;
static constexpr int NZ_K1_INDEX = 3;
static constexpr int NZ_K1_INDEX_TRANS = 4;
static constexpr int NZ_STORAGE_PENULTIMATE_DIM = 16;
static constexpr int NZ_STORAGE_LAST_DIM = 32;
static constexpr int64_t NZ_K0_VALUE_BMM_BLOCK_NUM = 16;
static constexpr int64_t NZ_K0_VALUE_INT32_TRANS = 8;
static constexpr int64_t NZ_K0_VALUE_INT8_TRANS = 32;
static constexpr int64_t NZ_K0_VALUE_INT4_TRANS = 64;
static constexpr int64_t OUTPUT_INFER_FAIL = -1L;
static constexpr int64_t LAST_AXIS_LIMIT = 65535;
static constexpr int X2_FIXED_DIM_NUM_A4W4 = 2;
static constexpr int64_t INT4_NUMS_IN_INT8 = 2;
static constexpr int64_t INT4_NUMS_IN_INT32 = 8;
static constexpr int64_t INNER_SIZE_MULTIPLE = 64;
static constexpr int64_t K_VALUE = 3696;
static constexpr int64_t N_VALUE = 8192;
static constexpr int64_t M_RANGE1_LEFT = 128;
static constexpr int64_t M_RANGE1_RIGHT = 512;
static constexpr int32_t CORE_NUM_20 = 20;
static constexpr int64_t SUPPORTED_GROUP_SIZE = 32;
static constexpr uint64_t B4_PER_B32 = 8UL;
static constexpr int64_t SUPPORTED_TCG_A8W4_K_ALIGN_NUM = 32;
static constexpr int64_t SUPPORTED_MX_A8W4_K_ALIGN_NUM = 8;
static constexpr int64_t SUPPORTED_N_ALIGN_NUM = 8;
static constexpr size_t MAX_DIM_VALUE = 2;
static constexpr size_t MX_SCALE_DIM_VALUE = 3;
static constexpr uint64_t GROUP_M_OFFSET = 32;
static constexpr uint64_t GROUP_N_OFFSET = 16;
static constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
static constexpr size_t MX_SCALE_MAX_DIM = 3;
static constexpr size_t MX_SCALE_DIM_NUM = 3;
static constexpr int64_t MAX_SHAPE_SIZE_A8W4_INT = 29576;
static constexpr int64_t PPMATMUL_PRIORITY_M = 1024;
static constexpr int64_t NO_BATCH_DIM_SUM = 2;

static const std::initializer_list<op::DataType> IN_TYPE_SUPPORT_LIST = {op::DataType::DT_INT4, op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> INT4_TYPE_SUPPORT_LIST = {op::DataType::DT_INT4,
                                                                           op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> SCALE_TYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT64, op::DataType::DT_BF16, op::DataType::DT_INT64, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BIAS_TYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> Y_SCALE_SUPPORT_LIST = {op::DataType::DT_UINT64};

using TupleTensor = std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>;
using TupleOptional = std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                                 const aclTensor*, const int64_t&>;
using TupleInput = std::tuple<const aclTensor*, const aclTensor*>;
using TupleQuant = std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                              const aclTensor*, const aclTensor*, const int64_t&, const int64_t&>;
using TupleAttr = std::tuple<bool, bool>;

static inline bool isA8W4Float(const aclTensor* x1, const aclTensor* x2)
{
    if (x1 == nullptr || x2 == nullptr) {
        return false;
    }
    return x1->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN &&
           (x2->GetDataType() == op::DataType::DT_FLOAT || x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1);
}

static inline bool isA8W4Int(const aclTensor* x1, const aclTensor* x2)
{
    if (x1 == nullptr || x2 == nullptr) {
        return false;
    }
    return x1->GetDataType() == op::DataType::DT_INT8 &&
           (x2->GetDataType() == op::DataType::DT_INT4 || x2->GetDataType() == op::DataType::DT_INT32);
}

static inline bool isMx(const aclTensor* scale)
{
    if (scale == nullptr) {
        return false;
    }
    return scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static inline bool isA8W4Msd(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale,
                             const aclTensor* pertokenScale)
{
    if (x1 == nullptr || x2 == nullptr || scale == nullptr) {
        return false;
    }
    if (x1->GetDataType() != op::DataType::DT_INT8) {
        return false;
    }

    if (std::find(INT4_TYPE_SUPPORT_LIST.begin(), INT4_TYPE_SUPPORT_LIST.end(), x2->GetDataType()) ==
        INT4_TYPE_SUPPORT_LIST.end()) {
        return false;
    }

    if (scale->GetDataType() != op::DataType::DT_UINT64) {
        return false;
    }

    if (pertokenScale == nullptr || pertokenScale->GetDataType() != op::DataType::DT_FLOAT) {
        return false;
    }

    return true;
}

static inline bool IsMicroScaling(const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    if (x1Scale == nullptr) {
        return false;
    }
    if (x2Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static inline bool IsTCG(const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    return x1Scale == nullptr && x2Scale != nullptr &&
           (x2Scale->GetDataType() == op::DataType::DT_BF16 || x2Scale->GetDataType() == op::DataType::DT_FLOAT16);
}

static inline int64_t SelectNzK0Value(op::DataType dataType, const bool isA8W4Float)
{
    switch (dataType) {
        case op::DataType::DT_INT4:
            return NZ_K0_VALUE_INT4_TRANS;
        case op::DataType::DT_FLOAT4_E2M1:
            if (isA8W4Float) {
                return NZ_K0_VALUE_INT8_TRANS;
            } else {
                return NZ_K0_VALUE_INT4_TRANS;
            }
        case op::DataType::DT_INT32:
            return NZ_K0_VALUE_INT32_TRANS;
        default:
            return NZ_K0_VALUE_INT8_TRANS;
    }
}

template <typename T>
static inline bool IsAligned(T num, T factor)
{
    if (factor == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The divisor cannot be zero.");
        return false;
    }
    return num > 0 && num % factor == 0;
}

static inline bool IsFormatNZ(const aclTensor* tensor)
{
    return tensor != nullptr &&
           (ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ ||
            ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ_C0_4 ||
            ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ_C0_32);
}

} // namespace quant_matmul_v4

#endif // QUANT_MATMUL_V4_COMMON_H
