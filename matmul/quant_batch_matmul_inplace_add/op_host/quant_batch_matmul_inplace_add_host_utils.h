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
 * \file quant_batch_matmul_inplace_add_utils.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_INPLACE_ADD_UTILS_H
#define QUANT_BATCH_MATMUL_INPLACE_ADD_UTILS_H

#include <map>

namespace QuantBatchMatmulInplaceAddTilingConstant {
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t X2_SCALE_INDEX = 2;
constexpr uint32_t Y_INDEX = 3;
constexpr uint32_t X1_SCALE_INDEX = 4;

constexpr uint32_t Y_OUTPUT_INDEX = 0;

constexpr uint32_t ATTR_INDEX_TRANSPOSE_X1 = 0;
constexpr uint32_t ATTR_INDEX_TRANSPOSE_X2 = 1;
constexpr uint32_t ATTR_INDEX_GROUP_SIZE = 2;

constexpr uint32_t ATTR_INDEX_NUMBERS = 3;
constexpr uint64_t DEFAULT_VALUE_OF_BATCHC = 1;
constexpr uint64_t USE_BASIC_API = 1UL;
constexpr uint32_t MX_X1_SCALE_DIM = 3;
constexpr uint32_t MX_X2_SCALE_DIM = 3;
constexpr uint32_t X1_MINIMUM_DIMENSION_LENGTH = 2;
constexpr uint32_t X2_MINIMUM_DIMENSION_LENGTH = 2;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr uint64_t GROUP_MKN_BIT_SIZE = 0xFFFF;
constexpr uint64_t MXFP_BASEK_FACTOR = 64UL;
constexpr uint64_t MXFP_MULTI_BASE_SIZE =2UL;

} // namespace QuantBatchMatmulInplaceAddTilingConstant
#endif