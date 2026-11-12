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

#ifndef TRANSPOSE_QUANT_BATCH_MAT_MUL_W8A8_TILING_DATA_H
#define TRANSPOSE_QUANT_BATCH_MAT_MUL_W8A8_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#if defined(__CCE_AICORE__)
#include "../mat_mul_v3/mat_mul_v3_tiling_data.h"
#include "../batch_mat_mul_v3/batch_mat_mul_v3_tiling_data.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_kernel_tiling_data.h"
#else
#include "../../mat_mul_v3/op_kernel/mat_mul_v3_tiling_data.h"
#include "../../batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_tiling_data.h"
#include "../../quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_kernel_tiling_data.h"
#endif

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) TQBMMW8A8TilingData {
    MatmulTilingData matmulTiling;
    MultiBatchInfo multiBatchInfo;
    int32_t batchSplitFactor;
    QuantBatchMatmulV3TilingData qbmm;
};
#pragma pack(pop)

#endif
