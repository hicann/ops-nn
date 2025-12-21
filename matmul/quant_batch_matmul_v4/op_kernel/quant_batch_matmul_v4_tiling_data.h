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
 * \file quant_batch_matmul_v4_tiling_data.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_TILING_DATA_H
#define QUANT_BATCH_MATMUL_V4_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#if defined(__CCE_AICORE__)
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_kernel_tiling_data.h"
#else
#include "../../quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_kernel_tiling_data.h"
#endif

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) QuantBatchMatmulV4MsdTilingData{
    uint8_t coreNum;
    uint32_t vBaseM;
    uint32_t ubRestBytes;
    uint32_t parallNum;
    uint32_t ubCalSize;
    uint32_t mSize;
    uint32_t kSize;
    uint32_t nSize;
    uint32_t groupSize;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) QuantBatchMatmulV4PerblockTilingData{
    TCubeTiling matmulTiling;
    L2cacheTileParam tileL2cacheTiling;
    uint32_t groupSizeM;
    uint32_t groupSizeN;
    uint32_t groupSizeK;
    uint32_t ubCalcM;
    uint32_t ubCalcN;
    bool transA;
    bool transB;
};
#pragma pack(pop)

#endif