/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file elu_grad_v2_tiling_data.h
 */
#ifndef ELU_GRAD_V2_TILING_DATA_H
#define ELU_GRAD_V2_TILING_DATA_H

#include <cstdint>

constexpr uint32_t ELU_GRAD_V2_CORE_CHUNK = 1024U;
constexpr uint32_t ELU_GRAD_V2_FLOAT_EXP_COMPUTE_CHUNK = 4096U;
constexpr uint32_t ELU_GRAD_V2_FLOAT_RESULT_COMPUTE_CHUNK = 4096U;
constexpr uint32_t ELU_GRAD_V2_MIXED_EXP_COMPUTE_CHUNK = 4096U;
constexpr uint32_t ELU_GRAD_V2_MIXED_RESULT_COMPUTE_CHUNK = 4096U;
constexpr uint32_t ELU_GRAD_V2_MAX_BUFFER_NUM = 2U;
constexpr uint32_t ELU_GRAD_V2_RESERVED_UB_BYTES = 1024U;
constexpr uint32_t ELU_GRAD_V2_BLOCK_BYTES = 256U;

struct EluGradV2TilingData {
    uint32_t totalLength = 0;
    uint32_t tileDataNum = 0;
    uint32_t coreNum = 0;
    uint32_t bigCoreNum = 0;
    uint32_t bigCoreDataNum = 0;
    uint32_t smallCoreDataNum = 0;
    uint32_t lastCoreDataNum = 0;
    uint32_t bufferOpen = 0U;
    uint32_t computeChunk = ELU_GRAD_V2_FLOAT_EXP_COMPUTE_CHUNK;
    float alpha = 1.0F;
    float scale = 1.0F;
    float inputScale = 1.0F;
    uint8_t isResult = 0U;
    uint8_t reserved0 = 0U;
    uint8_t reserved1 = 0U;
    uint8_t reserved2 = 0U;
};

#endif // ELU_GRAD_V2_TILING_DATA_H
