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
 * \file quant_batch_matmul_v4_tiling_data.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_TILING_DATA_H
#define QUANT_BATCH_MATMUL_V4_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h" // TCubeTiling结构体通过C++语法定义

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace qbmmv4_tiling {
#pragma pack(push, 8)
struct QuantBatchMatmulV4TilingDataParams {
    uint8_t cubeBlockDimN = 0;
    uint8_t cubeBlockDimM = 0;
    uint8_t vecCoreParallel = 0;
    uint8_t reserve1 = 0;
    uint16_t AL1Pingpong = 0;
    uint16_t BL1Pingpong = 0;

    uint64_t kAlign = 0;
    uint64_t nAlign = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint64_t groupSize = 0;
    uint64_t mSize = 0;

    uint64_t nBubSize = 0;
    uint64_t kBubSize = 0;
    uint64_t mAL1Size = 0;
    uint64_t kAL1Size = 0;
    uint64_t nBL1Size = 0;
    uint64_t kBL1Size = 0;
    uint64_t hasX1Scale = 0;
    uint64_t hasX2Scale = 0;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)
} // namespace qbmmv4_tiling
#endif // QUANT_BATCH_MATMUL_V4_TILING_DATA_H
