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
namespace AscendC {
    constexpr uint32_t BUFFER_NUM = 2UL;
    constexpr uint32_t VECCORE_NUM = 2UL;
    constexpr uint32_t L1_MAX_SIZE_910B = 512 * 1024UL;
    constexpr uint32_t BASE_N_128 = 128UL;
    constexpr uint32_t BASE_N_256 = 256UL;
    constexpr uint32_t MUL_DATASIZE_READ_PER_ITER = 256UL;
    constexpr uint32_t SIZE_OF_FP32 = 4UL;
    constexpr uint32_t NUM_ELEMENTS_PER_ITER = MUL_DATASIZE_READ_PER_ITER / SIZE_OF_FP32; // 64
    constexpr uint32_t X1_FORMAT = 2UL;
    constexpr uint32_t X2_FORMAT = 2UL;
    constexpr uint32_t ALIGN_UNIT_8 = 8UL;
    constexpr uint32_t ALIGN_UNIT_16 = 16UL;
    constexpr uint32_t ALIGN_UNIT_64 = 64UL;
    constexpr uint32_t FOLD2 = 2UL;
    constexpr uint32_t FOLD3 = 3UL;
    constexpr uint32_t FOLD4 = 4UL;
    constexpr uint32_t DATA_BLOCK_LEN = 32UL;

    constexpr uint16_t C2V_PING_FLAG = 0x4;
    constexpr uint16_t C2V_PONG_FLAG = 0x5;
    constexpr uint16_t V2C_PING_FLAG = 0x6;
    constexpr uint16_t V2C_PONG_FLAG = 0x7;
    constexpr uint16_t CV_SYNC_FLAG = 2;
} // namespace AscendC
#endif // QUANT_BATCH_MATMUL_V4_CONSTANT_H