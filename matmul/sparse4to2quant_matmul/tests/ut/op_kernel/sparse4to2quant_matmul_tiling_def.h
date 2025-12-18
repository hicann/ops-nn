/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE4TO2_QUANT_MATMUL_TILING_DEF_H
#define SPARSE4TO2_QUANT_MATMUL_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/sparse4to2quant_matmul_tiling_data.h"
#define __aicore__

constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;

inline void InitSparse4to2QuantMatmulTilingData(uint8_t* tiling, SparseQmm::Sparse4to2QuantMatmulTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(SparseQmm::Sparse4to2QuantMatmulTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitSparse4to2QuantMatmulTilingData(tiling_arg, &tiling_data);

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                        \
    SparseQmm::Sparse4to2QuantMatmulTilingData tiling_data;                                                 \
    InitSparse4to2QuantMatmulTilingData(tiling_arg, &tiling_data)
#endif  // SPARSE4TO2_QUANT_MATMUL_TILING_DEF_H
