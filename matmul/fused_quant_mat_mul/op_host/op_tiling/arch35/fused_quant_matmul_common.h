/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_quant_matmul_common.h
 * \brief
 */
#ifndef __OP_HOST_FUSED_QUANT_MATMUL_COMMON_H__
#define __OP_HOST_FUSED_QUANT_MATMUL_COMMON_H__

#include <cstddef>
#include <cstdint>

// RELU当前以来QBMMV3实现, RELU的枚举值需要对齐quant_batch_matmul_v3_base.h中的FusedOpType
enum class FQMMFusedOpType : uint64_t {
    NONE = 0UL,
    RELU = 1UL,
    SWIGLU = 2UL,
};

namespace optiling {
constexpr uint32_t X1_INDEX_FQMM = 0;
constexpr uint32_t X2_INDEX_FQMM = 1;
constexpr uint32_t BIAS_INDEX_FQMM = 2;
constexpr uint32_t X1_SCALE_INDEX_FQMM = 3;
constexpr uint32_t X2_SCALE_INDEX_FQMM = 4;
constexpr uint32_t Y_SCALE_INDEX_FQMM = 5;
constexpr uint32_t X1_OFFSET_INDEX_FQMM = 6;
constexpr uint32_t X2_OFFSET_INDEX_FQMM = 7;
constexpr uint32_t Y_OFFSET_INDEX_FQMM = 8;
constexpr uint32_t X2_TABLE_INDEX_FQMM = 9;
constexpr uint32_t X3_INDEX_FQMM = 10;

constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_BATCH_DIM_INDEX = 3;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr uint32_t BIAS_THREE_DIM = 3;

constexpr uint32_t FUSED_TYPE_ATTR_INDEX_FQMM = 5;
} // namespace optiling
#endif // __OP_HOST_FUSED_QUANT_MATMUL_COMMON_H__