/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TEST_FUSED_QUANT_MAT_MUL_TILING_TILING_H_
#define _TEST_FUSED_QUANT_MAT_MUL_TILING_TILING_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_kernel_tiling_data.h"

#define __aicore__

constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;

inline void InitFusedQuantMatmulTilingData(uint8_t* tiling, QuantBatchMatmulV3TilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(QuantBatchMatmulV3TilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                       \
    QuantBatchMatmulV3TilingData tiling_data;                                                 \
    InitFusedQuantMatmulTilingData(tiling_arg, &tiling_data)
#endif  // _TEST_FUSED_QUANT_MAT_MUL_TILING_TILING_H_
