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
 * \file quant_batch_matmul_v4_constant.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V4_CONSTANT_H
#define QUANT_BATCH_MATMUL_V4_CONSTANT_H
namespace QuantBatchMatmulV4 {
enum class QuantType : uint32_t {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4,
};

constexpr int32_t IDX_0 = 0;
constexpr int32_t IDX_1 = 1;
constexpr int32_t IDX_2 = 2;
constexpr int32_t IDX_3 = 3;
constexpr int32_t AIV_NUM = 2;
constexpr int32_t OFFSET_64 = 64;
constexpr int32_t UB_ALIGN_SIZE_FOR_4BITS = 64;
constexpr int32_t OFFSET_8 = 8;
constexpr int32_t OFFSET_FOR_4BITS = 128;
constexpr int32_t GROUP_SIZE_32 = 32;
constexpr int32_t VECTOR_REG_WIDTH_FOR_4BITS = 512;
constexpr uint32_t MASK_16_BITS = 0xFFFFU;
constexpr uint32_t DUP_CONFIG_2 = 0x2;
constexpr uint32_t DUP_CONFIG_MODE_1C = 0x1C;
constexpr uint32_t DUP_CONFIG_4 = 0x4;
constexpr uint32_t DUP_CONFIG_9C = 0x9C;
constexpr uint32_t DUP_FLAG_80 = 0x80;
constexpr int32_t SCALE_COPY_GROUP_SIZE = 2;
constexpr int32_t SCALE_COPY_DEFAULT_STRIDE = 0;
constexpr int32_t SCALE_COPY_DEFAULT_NS_STRIDE = 1;
constexpr int32_t C0_SIZE_B8 = 32;
constexpr uint32_t E2M1_SHIFT_LEFT_SIZE = 0x2;
constexpr uint32_t SHIFT_RIGHT_SIZE = 0x4;
constexpr uint32_t E2M1_AND_MASK = 0x9C;
constexpr uint32_t E1M2_SHIFT_LEFT_SIZE = 0x3;
constexpr uint32_t E1M2_AND_MASK = 0x8E;
constexpr uint32_t PERGROUP_NZ_MASK_REG = 32;
} // namespace QuantBatchMatmulV4
#endif // QUANT_BATCH_MATMUL_V4_CONSTANT_H
