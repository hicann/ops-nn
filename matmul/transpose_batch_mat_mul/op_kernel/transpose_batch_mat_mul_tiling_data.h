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
 * \file transpose_batch_mat_mul_tiling_data.h
 * \brief
 */

#ifndef TRANSPOSE_BATCH_MAT_MUL_TILING_DATA_H
#define TRANSPOSE_BATCH_MAT_MUL_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#if defined(__CCE_AICORE__)
#include "../batch_mat_mul_v3/batch_mat_mul_v3_tiling_data.h"
#else
#include "../../batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_tiling_data.h"
#endif

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) TBMMTilingData{
    MatmulTilingData matmulTiling;
    MultiBatchInfo multiBatchInfo;
    int32_t batchSplitFactor;
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) PpMatmulTilingData
{
    uint32_t batch;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t m0;
    uint32_t k0;
    uint32_t n0;
    uint32_t mLoop;
    uint32_t kLoop;
    uint32_t nLoop;
    uint32_t coreLoop;
    uint32_t swizzlCount;
    uint32_t tilingKey;
    uint32_t blockDim;
    uint32_t swizzlDirect;
    uint32_t splitk;
    uint32_t enShuffleK;
};
#pragma pack(pop)

#endif