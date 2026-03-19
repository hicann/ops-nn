/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QUANT_BATCH_MATMUL_INPLACE_ADD_UTIL_H
#define OP_API_INC_QUANT_BATCH_MATMUL_INPLACE_ADD_UTIL_H
#include "opdev/common_types.h"

namespace QBMMInplaceAdd {
using namespace op;
struct QuantBatchMatmulInplaceAddParams {
    const aclTensor *x1 = nullptr;
    const aclTensor *x2 = nullptr;
    const aclTensor *x1ScaleOptional = nullptr;
    const aclTensor *x2Scale = nullptr;
    aclTensor *yRef = nullptr;

    bool transposeX1;
    bool transposeX2;
    int64_t groupSize = 0L;
};

static const std::initializer_list<op::DataType> x1_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2};
static const std::initializer_list<op::DataType> x2_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2};
static const std::initializer_list<op::DataType> X1_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> X2_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> YREF_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

constexpr uint32_t MX_X1_DIM = 2U;
constexpr uint32_t MX_X2_DIM = 2U;
constexpr uint32_t MX_X1_SCALE_DIM = 3U;
constexpr uint32_t MX_X2_SCALE_DIM = 3U;
constexpr uint32_t Y_INPUT_DIM = 2U;
constexpr uint32_t Y_OUTPUT_DIM = 2U;
constexpr uint32_t MX_X1_PER_TOKEN_SCALE_DIM = 3U;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2L;
constexpr int64_t SPLIT_SIZE = 64L;

static const int32_t GROUP_M_OFFSET = 32;
static const int32_t GROUP_N_OFFSET = 16;
static const uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
static const int64_t PERGROUP_GROUP_SIZE = 32L;
static const size_t MX_SCALE_MAX_DIM = 3;

} // namespace QBMMInplaceAdd
#endif